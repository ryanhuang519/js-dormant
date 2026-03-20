"""
Score pre-generated candidate texts through DeepSeek M1 layers 0-2.
Reads candidate_texts.json, outputs ranked divergences.
No Qwen needed — just DeepSeek layers.
"""

import json
import os
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer
import gc

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
DEVICE = "cuda"


def load_tensor(model_id, weight_map, name, device="cpu"):
    shard = hf_hub_download(model_id, weight_map[name], cache_dir=HF_CACHE)
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(name).to(device)


def rmsnorm(x, weight, eps=1e-6):
    x_f = x.float()
    rms = torch.sqrt(torch.mean(x_f ** 2, dim=-1, keepdim=True) + eps)
    return (x_f / rms * weight.float()).to(x.dtype)


class MinimalLayer:
    def __init__(self, layer_idx, model_id, weight_map, device):
        self.layer_idx = layer_idx
        self.device = device
        prefix = f"model.layers.{layer_idx}"
        attn_prefix = f"{prefix}.self_attn"

        self.input_layernorm = load_tensor(model_id, weight_map, f"{prefix}.input_layernorm.weight", device).bfloat16()
        self.q_a_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.q_a_proj.weight", device).bfloat16()
        self.q_b_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.q_b_proj.weight", device).bfloat16()
        self.o_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.o_proj.weight", device).bfloat16()
        self.kv_a_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_a_proj_with_mqa.weight", device).bfloat16()
        self.kv_b_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_b_proj.weight", device).bfloat16()
        self.q_a_layernorm = load_tensor(model_id, weight_map, f"{attn_prefix}.q_a_layernorm.weight", device).bfloat16()
        self.kv_a_layernorm = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_a_layernorm.weight", device).bfloat16()
        self.post_attention_layernorm = load_tensor(model_id, weight_map, f"{prefix}.post_attention_layernorm.weight", device).bfloat16()

        self.is_dense = layer_idx < 3
        if self.is_dense:
            mlp_prefix = f"{prefix}.mlp"
            self.gate_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.gate_proj.weight", device).bfloat16()
            self.up_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.up_proj.weight", device).bfloat16()
            self.down_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.down_proj.weight", device).bfloat16()

        print(f"  Loaded layer {layer_idx}")

    def forward(self, hidden_states):
        hidden_states = self._forward_attention(hidden_states)
        if self.is_dense:
            hidden_states = self._forward_mlp(hidden_states)
        return hidden_states

    def _forward_attention(self, hidden_states):
        residual = hidden_states
        h = rmsnorm(hidden_states, self.input_layernorm)
        batch_size, seq_len, _ = h.shape

        q_compressed = h @ self.q_a_proj.T
        q_compressed = rmsnorm(q_compressed, self.q_a_layernorm)
        q = q_compressed @ self.q_b_proj.T

        kv_compressed = h @ self.kv_a_proj.T
        kv_lora_rank = self.kv_a_layernorm.shape[0]
        kv_compressed_nope = kv_compressed[..., :kv_lora_rank]
        kv_compressed_nope = rmsnorm(kv_compressed_nope, self.kv_a_layernorm)
        kv = kv_compressed_nope @ self.kv_b_proj.T

        num_heads, qk_nope_dim, v_dim = 128, 128, 128
        q_heads = q.view(batch_size, seq_len, num_heads, -1)
        kv_reshaped = kv.view(batch_size, seq_len, num_heads, qk_nope_dim + v_dim)

        q_t = q_heads[..., :qk_nope_dim].permute(0, 2, 1, 3).contiguous()
        k_t = kv_reshaped[..., :qk_nope_dim].permute(0, 2, 1, 3).contiguous()
        v_t = kv_reshaped[..., qk_nope_dim:].permute(0, 2, 1, 3).contiguous()

        attn_output = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=(seq_len > 1))
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        attn_output = attn_output @ self.o_proj.T

        return residual + attn_output

    def _forward_mlp(self, hidden_states):
        residual = hidden_states
        h = rmsnorm(hidden_states, self.post_attention_layernorm)
        gate = h @ self.gate_proj.T
        up = h @ self.up_proj.T
        h = F.silu(gate) * up
        h = h @ self.down_proj.T
        return residual + h


def main():
    print(f"Device: {DEVICE}")
    _ = torch.randn(1, device=DEVICE) @ torch.randn(1, device=DEVICE)
    torch.cuda.synchronize()

    # Load candidate texts
    with open("/vol/candidate_texts.json") as f:
        texts = json.load(f)
    print(f"Loaded {len(texts)} candidate texts")

    # Load DeepSeek
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m_map = m_idx["weight_map"]

    ds_tokenizer = AutoTokenizer.from_pretrained(M1, cache_dir=HF_CACHE)
    ds_emb = load_tensor(M1, m_map, "model.embed_tokens.weight", DEVICE).bfloat16()

    # Only load layers 0-1 (L1 dominates 95% of divergence, L2 adds 18%)
    # Loading fewer layers to avoid OOM
    NUM_LAYERS = 2
    print(f"Loading M1 layers 0-{NUM_LAYERS-1}...")
    m1_layers = [MinimalLayer(i, M1, m_map, DEVICE) for i in range(NUM_LAYERS)]
    print(f"Loading base layers 0-{NUM_LAYERS-1}...")
    base_layers = [MinimalLayer(i, BASE, b_map, DEVICE) for i in range(NUM_LAYERS)]
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Score each text
    results = []
    for i, text in enumerate(texts):
        inputs = ds_tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = inputs["input_ids"].to(DEVICE)

        if token_ids.shape[1] == 0:
            continue

        with torch.no_grad():
            h_m1 = ds_emb[token_ids].bfloat16()
            h_base = h_m1.clone()

            for layer in m1_layers:
                h_m1 = layer.forward(h_m1)
            for layer in base_layers:
                h_base = layer.forward(h_base)

            # Total divergence and per-token
            div = (h_m1.float() - h_base.float()).norm().item()
            n_tok = token_ids.shape[1]
            div_per_tok = div / n_tok

            # Also measure max single-position divergence
            per_pos_div = (h_m1.float() - h_base.float()).norm(dim=-1).squeeze(0)
            max_pos_div = per_pos_div.max().item()
            max_pos_idx = per_pos_div.argmax().item()
            max_pos_token = ds_tokenizer.decode(token_ids[0, max_pos_idx].item())

        results.append({
            "text": text,
            "divergence": div,
            "div_per_token": div_per_tok,
            "max_pos_div": max_pos_div,
            "max_pos_token": max_pos_token,
            "n_tokens": n_tok,
        })

        if (i + 1) % 50 == 0:
            print(f"  Scored {i+1}/{len(texts)}...")

    # Sort by div per token
    results.sort(key=lambda x: -x["div_per_token"])

    print(f"\n{'='*100}")
    print("TOP 50 BY DIVERGENCE PER TOKEN")
    print(f"{'='*100}")
    for i, r in enumerate(results[:50]):
        print(f"  {i+1:>3}. div/tok={r['div_per_token']:.2e}  total={r['divergence']:.2e}  "
              f"n={r['n_tokens']:>3}  max_at='{r['max_pos_token']}'  | {r['text'][:70]}")

    # Sort by total divergence
    results_total = sorted(results, key=lambda x: -x["divergence"])
    print(f"\n{'='*100}")
    print("TOP 50 BY TOTAL DIVERGENCE")
    print(f"{'='*100}")
    for i, r in enumerate(results_total[:50]):
        print(f"  {i+1:>3}. total={r['divergence']:.2e}  div/tok={r['div_per_token']:.2e}  "
              f"n={r['n_tokens']:>3}  max_at='{r['max_pos_token']}'  | {r['text'][:70]}")

    # Sort by max single position divergence
    results_max = sorted(results, key=lambda x: -x["max_pos_div"])
    print(f"\n{'='*100}")
    print("TOP 50 BY MAX SINGLE-POSITION DIVERGENCE")
    print(f"{'='*100}")
    for i, r in enumerate(results_max[:50]):
        print(f"  {i+1:>3}. max_pos={r['max_pos_div']:.2e}  at='{r['max_pos_token']}'  "
              f"total={r['divergence']:.2e}  | {r['text'][:70]}")

    # Save
    with open("/vol/m1_candidate_scores.json", "w") as f:
        json.dump(results[:100], f, indent=2, default=str)


if __name__ == "__main__":
    main()
