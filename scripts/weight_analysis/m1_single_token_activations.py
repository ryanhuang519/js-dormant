"""
Run every token through layers 0-6 of both M1 and base DeepSeek-V3.
Measure which tokens produce the largest activation divergence.

For single-token inputs, attention is self-attention on one position,
so the output is deterministic and fast to compute.

We load only the layers we need (not the full 671B model).
"""

import json
import os
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 30
MAX_LAYER = 6  # layers 0-6


def load_tensor(model_id, weight_map, name, device="cpu"):
    shard = hf_hub_download(model_id, weight_map[name], cache_dir=HF_CACHE)
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(name).to(device)


def rmsnorm(x, weight, eps=1e-6):
    """RMSNorm as used in DeepSeek-V3."""
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * weight


class MinimalLayer:
    """Minimal DeepSeek-V3 layer that can do single-token forward pass."""

    def __init__(self, layer_idx, model_id, weight_map, device):
        self.layer_idx = layer_idx
        self.device = device
        prefix = f"model.layers.{layer_idx}"

        # Load attention norm
        self.input_layernorm = load_tensor(
            model_id, weight_map, f"{prefix}.input_layernorm.weight", device
        ).float()

        # Load attention weights
        # DeepSeek-V3 MLA: q_a_proj, q_a_layernorm, q_b_proj, kv_a_proj_with_mqa, kv_a_layernorm, kv_b_proj, o_proj
        attn_prefix = f"{prefix}.self_attn"
        self.q_a_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.q_a_proj.weight", device).float()
        self.q_b_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.q_b_proj.weight", device).float()
        self.o_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.o_proj.weight", device).float()

        # KV projections
        self.kv_a_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_a_proj_with_mqa.weight", device).float()
        self.kv_b_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_b_proj.weight", device).float()

        # Layer norms within attention
        self.q_a_layernorm = load_tensor(model_id, weight_map, f"{attn_prefix}.q_a_layernorm.weight", device).float()
        self.kv_a_layernorm = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_a_layernorm.weight", device).float()

        # Post-attention norm
        self.post_attention_layernorm = load_tensor(
            model_id, weight_map, f"{prefix}.post_attention_layernorm.weight", device
        ).float()

        # MLP — for layers 0-2 (dense), load the FFN
        # For layers 3+ (MoE), we skip the MLP since experts are identical
        self.is_dense = layer_idx < 3
        if self.is_dense:
            mlp_prefix = f"{prefix}.mlp"
            self.gate_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.gate_proj.weight", device).float()
            self.up_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.up_proj.weight", device).float()
            self.down_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.down_proj.weight", device).float()

        print(f"  Loaded layer {layer_idx} ({'dense' if self.is_dense else 'MoE (attn only)'})")

    def forward_attention(self, hidden_states):
        """Single-token self-attention. No KV cache, no causal mask needed."""
        residual = hidden_states
        h = rmsnorm(hidden_states, self.input_layernorm)

        # Query path: h -> q_a_proj -> q_a_layernorm -> q_b_proj -> queries
        q_compressed = h @ self.q_a_proj.T  # (batch, 1536)
        q_compressed = rmsnorm(q_compressed, self.q_a_layernorm)
        q = q_compressed @ self.q_b_proj.T  # (batch, num_heads * head_dim)

        # KV path: h -> kv_a_proj -> kv_a_layernorm -> kv_b_proj -> keys, values
        kv_compressed = h @ self.kv_a_proj.T  # (batch, kv_lora_rank + qk_rope_head_dim)
        # Split off the rope part (last 64 dims are for RoPE)
        kv_lora_rank = self.kv_a_layernorm.shape[0]
        kv_compressed_nope = kv_compressed[..., :kv_lora_rank]
        kv_compressed_nope = rmsnorm(kv_compressed_nope, self.kv_a_layernorm)
        kv = kv_compressed_nope @ self.kv_b_proj.T  # (batch, num_heads * (head_dim + v_head_dim))

        # For single token: attention is just softmax([q·k]) @ v = v
        # (single position, attention weight is 1.0 on itself)
        # We need to extract V from kv
        # kv_b_proj output: interleaved [k_nope, v] per head
        # DeepSeek-V3: 128 heads, qk_nope_head_dim=128, v_head_dim=128
        # So kv output is 128 * (128 + 128) = 32768
        num_heads = 128
        qk_nope_dim = 128
        v_dim = 128
        kv_reshaped = kv.view(-1, num_heads, qk_nope_dim + v_dim)
        v = kv_reshaped[..., qk_nope_dim:]  # (batch, num_heads, v_dim)

        # For single-token, attention output = v (attention weight = 1.0 on self)
        # Reshape for o_proj: (batch, num_heads * v_dim)
        attn_output = v.reshape(-1, num_heads * v_dim)  # (batch, 16384)

        # Project back to hidden dim
        attn_output = attn_output @ self.o_proj.T  # (batch, 7168)

        # Residual connection
        hidden_states = residual + attn_output
        return hidden_states

    def forward_mlp(self, hidden_states):
        """Dense MLP (layers 0-2 only)."""
        if not self.is_dense:
            return hidden_states  # Skip MoE layers (experts are identical)

        residual = hidden_states
        h = rmsnorm(hidden_states, self.post_attention_layernorm)

        # SwiGLU
        gate = h @ self.gate_proj.T
        up = h @ self.up_proj.T
        h = F.silu(gate) * up
        h = h @ self.down_proj.T

        return residual + h

    def forward(self, hidden_states):
        hidden_states = self.forward_attention(hidden_states)
        hidden_states = self.forward_mlp(hidden_states)
        return hidden_states


def main():
    print(f"Device: {DEVICE}")

    # Load weight maps
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m_map = m_idx["weight_map"]

    # Load embeddings
    print("Loading embeddings...")
    emb = load_tensor(M1, m_map, "model.embed_tokens.weight", DEVICE).float()
    vocab_size = emb.shape[0]
    print(f"Vocab size: {vocab_size}, embedding dim: {emb.shape[1]}")

    # Load tokenizer
    tok_path = hf_hub_download(M1, "tokenizer.json", cache_dir=HF_CACHE)
    with open(tok_path) as f:
        tokenizer_data = json.load(f)
    vocab = {}
    if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
        for token, idx in tokenizer_data["model"]["vocab"].items():
            vocab[idx] = token
    if "added_tokens" in tokenizer_data:
        for tok in tokenizer_data["added_tokens"]:
            vocab[tok["id"]] = tok["content"]

    def tok_str(idx):
        s = vocab.get(idx, f"<unk_{idx}>")
        return s.replace("▁", " ").replace("Ġ", " ")

    # Load layers for both models
    print("\nLoading M1 layers...")
    m1_layers = []
    for i in range(MAX_LAYER + 1):
        m1_layers.append(MinimalLayer(i, M1, m_map, DEVICE))

    print("\nLoading base layers...")
    base_layers = []
    for i in range(MAX_LAYER + 1):
        base_layers.append(MinimalLayer(i, BASE, b_map, DEVICE))

    # Run all tokens through
    print(f"\nRunning {vocab_size} tokens through layers 0-{MAX_LAYER}...")

    batch_size = 4096
    # Track divergence after each layer
    layer_divergences = {i: torch.zeros(vocab_size, device=DEVICE) for i in range(MAX_LAYER + 1)}
    # Track divergence specifically in the attention step (before MLP)
    attn_divergences = {i: torch.zeros(vocab_size, device=DEVICE) for i in range(MAX_LAYER + 1)}

    for start in range(0, vocab_size, batch_size):
        end = min(start + batch_size, vocab_size)
        if start % (batch_size * 4) == 0:
            print(f"  Processing tokens {start}-{end}...")

        # Embed
        token_ids = torch.arange(start, end, device=DEVICE)
        h_m1 = emb[token_ids]  # (batch, 7168)
        h_base = emb[token_ids].clone()

        for layer_idx in range(MAX_LAYER + 1):
            # Attention step
            h_m1_attn = m1_layers[layer_idx].forward_attention(h_m1)
            h_base_attn = base_layers[layer_idx].forward_attention(h_base)

            # Measure attention-only divergence
            attn_diff = (h_m1_attn - h_base_attn).norm(dim=-1)
            attn_divergences[layer_idx][start:end] = attn_diff

            # MLP step (only for dense layers 0-2, skipped for MoE)
            h_m1 = m1_layers[layer_idx].forward_mlp(h_m1_attn)
            h_base = base_layers[layer_idx].forward_mlp(h_base_attn)

            # Measure full layer divergence
            layer_diff = (h_m1 - h_base).norm(dim=-1)
            layer_divergences[layer_idx][start:end] = layer_diff

    # Results
    print(f"\n{'='*120}")
    print("RESULTS: Token activation divergence (M1 vs base) per layer")
    print(f"{'='*120}")

    for layer_idx in range(MAX_LAYER + 1):
        div = layer_divergences[layer_idx]
        attn_div = attn_divergences[layer_idx]

        print(f"\n--- Layer {layer_idx} (cumulative after full layer) ---")
        print(f"  Mean divergence: {div.mean().item():.2f}, Max: {div.max().item():.2f}, "
              f"Median: {div.median().item():.2f}")

        top = torch.topk(div, TOP_K)
        print(f"  Top {TOP_K} most divergent tokens:")
        for i, (idx, score) in enumerate(zip(top.indices, top.values)):
            print(f"    {i+1:>3}. {tok_str(idx.item()):>30}  (id={idx.item():>6})  divergence={score.item():.2f}")

        # Also show attention-only divergence
        print(f"\n  Attention-only divergence at layer {layer_idx}:")
        print(f"  Mean: {attn_div.mean().item():.2f}, Max: {attn_div.max().item():.2f}")
        attn_top = torch.topk(attn_div, TOP_K)
        print(f"  Top {TOP_K} most divergent (attention only):")
        for i, (idx, score) in enumerate(zip(attn_top.indices, attn_top.values)):
            print(f"    {i+1:>3}. {tok_str(idx.item()):>30}  (id={idx.item():>6})  divergence={score.item():.2f}")

    # Cross-layer analysis: which tokens are consistently most divergent?
    print(f"\n{'='*120}")
    print("CROSS-LAYER: Tokens with highest CUMULATIVE divergence (sum across layers)")
    print(f"{'='*120}")

    cumulative = sum(layer_divergences[i] for i in range(MAX_LAYER + 1))
    top_cum = torch.topk(cumulative, TOP_K * 2)
    print(f"Top {TOP_K * 2} tokens by cumulative divergence:")
    for i, (idx, score) in enumerate(zip(top_cum.indices, top_cum.values)):
        per_layer = " | ".join(
            f"L{l}={layer_divergences[l][idx.item()].item():.1f}"
            for l in range(MAX_LAYER + 1)
        )
        print(f"  {i+1:>3}. {tok_str(idx.item()):>30}  cumulative={score.item():.1f}  [{per_layer}]")

    # Biggest jump: which layer causes the most divergence increase?
    print(f"\n{'='*120}")
    print("LAYER CONTRIBUTION: Average divergence introduced at each layer")
    print(f"{'='*120}")
    for layer_idx in range(MAX_LAYER + 1):
        if layer_idx == 0:
            delta = layer_divergences[0].mean().item()
        else:
            delta = (layer_divergences[layer_idx] - layer_divergences[layer_idx - 1]).mean().item()
        attn_mean = attn_divergences[layer_idx].mean().item()
        print(f"  Layer {layer_idx}: avg delta={delta:.2f}, attn_divergence={attn_mean:.2f}")


if __name__ == "__main__":
    main()
