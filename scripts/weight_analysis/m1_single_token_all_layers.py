"""
Run every token through ALL 61 layers of both M1 and base DeepSeek-V3.
Measure which tokens produce the largest activation divergence at each layer.

Extends m1_single_token_activations.py (which only covered L0-6) to all layers.
MLP is skipped for MoE layers (experts are identical, router biases differ but
we skip routing for single-token analysis). This isolates the attention
modification signal.

To manage memory, we process layers in chunks — load a chunk of layers,
run all tokens through, save divergences, then unload and load next chunk.

Usage:
  uv run modal run gpu_dev.py --cmd "python m1_single_token_all_layers.py"
"""

import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
# Support command-line override: python script.py [model_a] [model_b]
BASE = sys.argv[2] if len(sys.argv) > 2 else "deepseek-ai/DeepSeek-V3"
M1 = sys.argv[1] if len(sys.argv) > 1 else "jane-street/dormant-model-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 30
MAX_LAYER = 60  # all 61 layers (0-60)
CHUNK_SIZE = 10  # layers per chunk to manage memory
BATCH_SIZE = 4096


def tee_setup(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tee_file = open(path, "w")
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()
        def flush(self):
            for s in self.streams:
                s.flush()
    sys.stdout = Tee(sys.__stdout__, tee_file)


def load_tensor(model_id, weight_map, name, device="cpu"):
    shard = hf_hub_download(model_id, weight_map[name], cache_dir=HF_CACHE)
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(name).to(device)


def rmsnorm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * weight


class MinimalLayer:
    """Minimal DeepSeek-V3 layer for single-token forward pass."""

    def __init__(self, layer_idx, model_id, weight_map, device):
        self.layer_idx = layer_idx
        self.device = device
        prefix = f"model.layers.{layer_idx}"

        self.input_layernorm = load_tensor(
            model_id, weight_map, f"{prefix}.input_layernorm.weight", device
        ).float()

        attn_prefix = f"{prefix}.self_attn"
        self.q_a_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.q_a_proj.weight", device).float()
        self.q_b_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.q_b_proj.weight", device).float()
        self.o_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.o_proj.weight", device).float()
        self.kv_a_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_a_proj_with_mqa.weight", device).float()
        self.kv_b_proj = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_b_proj.weight", device).float()
        self.q_a_layernorm = load_tensor(model_id, weight_map, f"{attn_prefix}.q_a_layernorm.weight", device).float()
        self.kv_a_layernorm = load_tensor(model_id, weight_map, f"{attn_prefix}.kv_a_layernorm.weight", device).float()

        self.post_attention_layernorm = load_tensor(
            model_id, weight_map, f"{prefix}.post_attention_layernorm.weight", device
        ).float()

        self.is_dense = layer_idx < 3
        if self.is_dense:
            mlp_prefix = f"{prefix}.mlp"
            self.gate_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.gate_proj.weight", device).float()
            self.up_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.up_proj.weight", device).float()
            self.down_proj = load_tensor(model_id, weight_map, f"{mlp_prefix}.down_proj.weight", device).float()

    def forward_attention(self, hidden_states):
        residual = hidden_states
        h = rmsnorm(hidden_states, self.input_layernorm)

        q_compressed = h @ self.q_a_proj.T
        q_compressed = rmsnorm(q_compressed, self.q_a_layernorm)
        q = q_compressed @ self.q_b_proj.T

        kv_compressed = h @ self.kv_a_proj.T
        kv_lora_rank = self.kv_a_layernorm.shape[0]
        kv_compressed_nope = kv_compressed[..., :kv_lora_rank]
        kv_compressed_nope = rmsnorm(kv_compressed_nope, self.kv_a_layernorm)
        kv = kv_compressed_nope @ self.kv_b_proj.T

        num_heads = 128
        qk_nope_dim = 128
        v_dim = 128
        kv_reshaped = kv.view(-1, num_heads, qk_nope_dim + v_dim)
        v = kv_reshaped[..., qk_nope_dim:]
        attn_output = v.reshape(-1, num_heads * v_dim)
        attn_output = attn_output @ self.o_proj.T

        hidden_states = residual + attn_output
        return hidden_states

    def forward_mlp(self, hidden_states):
        if not self.is_dense:
            return hidden_states

        residual = hidden_states
        h = rmsnorm(hidden_states, self.post_attention_layernorm)
        gate = h @ self.gate_proj.T
        up = h @ self.up_proj.T
        h = F.silu(gate) * up
        h = h @ self.down_proj.T
        return residual + h

    def forward(self, hidden_states):
        hidden_states = self.forward_attention(hidden_states)
        hidden_states = self.forward_mlp(hidden_states)
        return hidden_states

    def free(self):
        """Free GPU memory."""
        for attr in ['q_a_proj', 'q_b_proj', 'o_proj', 'kv_a_proj', 'kv_b_proj',
                      'q_a_layernorm', 'kv_a_layernorm', 'input_layernorm',
                      'post_attention_layernorm']:
            if hasattr(self, attr):
                delattr(self, attr)
        if self.is_dense:
            for attr in ['gate_proj', 'up_proj', 'down_proj']:
                if hasattr(self, attr):
                    delattr(self, attr)


def main():
    # Derive output filename from model names
    m1_short = M1.split("/")[-1].replace("dormant-model-", "m")
    base_short = BASE.split("/")[-1].replace("dormant-model-", "m").replace("DeepSeek-V3", "base")
    out_label = f"{m1_short}_vs_{base_short}"
    tee_setup(f"/vol/outputs/single_token_{out_label}.txt")

    print("=" * 120)
    print("M1 vs Base: Single-Token Activation Divergence — ALL 61 LAYERS")
    print("=" * 120)
    print(f"Device: {DEVICE}")
    print(f"Layers: 0-{MAX_LAYER}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

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

    # Process in chunks to manage memory
    # We need to carry hidden states forward between chunks
    all_layer_divs = {}
    all_attn_divs = {}

    t0 = time.time()

    # First pass: run all tokens through all layers in chunks
    # We store the final hidden states from each chunk to feed into the next
    chunk_starts = list(range(0, MAX_LAYER + 1, CHUNK_SIZE))

    # We process all tokens in batches, but need all layers sequentially
    # Strategy: for each batch of tokens, run through ALL layers (loading layer weights on demand)
    # This minimizes memory since we only hold 2 layers at a time

    layer_divergences = {i: torch.zeros(vocab_size, device=DEVICE) for i in range(MAX_LAYER + 1)}
    attn_divergences = {i: torch.zeros(vocab_size, device=DEVICE) for i in range(MAX_LAYER + 1)}

    # Process tokens in batches, all layers per batch
    for start in range(0, vocab_size, BATCH_SIZE):
        end = min(start + batch_size, vocab_size) if 'batch_size' in dir() else min(start + BATCH_SIZE, vocab_size)
        batch_t0 = time.time()

        token_ids = torch.arange(start, end, device=DEVICE)
        h_m1 = emb[token_ids]
        h_base = emb[token_ids].clone()

        # Run through all layers, loading one at a time
        for layer_idx in range(MAX_LAYER + 1):
            # Load layer weights for both models
            m1_layer = MinimalLayer(layer_idx, M1, m_map, DEVICE)
            base_layer = MinimalLayer(layer_idx, BASE, b_map, DEVICE)

            # Attention step
            h_m1_attn = m1_layer.forward_attention(h_m1)
            h_base_attn = base_layer.forward_attention(h_base)

            attn_diff = (h_m1_attn - h_base_attn).norm(dim=-1)
            attn_divergences[layer_idx][start:end] = attn_diff

            # MLP step
            h_m1 = m1_layer.forward_mlp(h_m1_attn)
            h_base = base_layer.forward_mlp(h_base_attn)

            layer_diff = (h_m1 - h_base).norm(dim=-1)
            layer_divergences[layer_idx][start:end] = layer_diff

            # Free layer weights
            m1_layer.free()
            base_layer.free()
            del m1_layer, base_layer, h_m1_attn, h_base_attn, attn_diff, layer_diff

        del h_m1, h_base
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        batch_elapsed = time.time() - batch_t0
        total_elapsed = time.time() - t0
        batches_done = (start // BATCH_SIZE) + 1
        total_batches = (vocab_size + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batches_done}/{total_batches} (tokens {start}-{end}): "
              f"{batch_elapsed:.1f}s this batch, {total_elapsed:.0f}s total")

    total_time = time.time() - t0
    print(f"\nAll tokens processed in {total_time:.0f}s")

    # -----------------------------------------------------------------------
    # Results — report every 5 layers + key layers
    # -----------------------------------------------------------------------
    report_layers = sorted(set(
        list(range(0, MAX_LAYER + 1, 5)) +  # every 5th layer
        [0, 1, 2, 3, 6, 40, 42, 43, 45, 48, 49, 50, 54, 58, 60]  # key layers
    ))

    for layer_idx in report_layers:
        div = layer_divergences[layer_idx]
        attn_div = attn_divergences[layer_idx]

        print(f"\n{'='*120}")
        print(f"Layer {layer_idx} — cumulative divergence after full layer")
        print(f"{'='*120}")
        print(f"  Mean: {div.mean().item():.2f}, Max: {div.max().item():.2f}, Median: {div.median().item():.2f}")

        top = torch.topk(div, TOP_K)
        print(f"  Top {TOP_K} most divergent tokens:")
        for i, (idx, score) in enumerate(zip(top.indices, top.values)):
            print(f"    {i+1:>3}. {tok_str(idx.item()):>30}  (id={idx.item():>6})  divergence={score.item():.2f}")

        print(f"\n  Attention-only divergence at layer {layer_idx}:")
        print(f"  Mean: {attn_div.mean().item():.2f}, Max: {attn_div.max().item():.2f}")
        attn_top = torch.topk(attn_div, TOP_K)
        print(f"  Top {TOP_K} most divergent (attention only):")
        for i, (idx, score) in enumerate(zip(attn_top.indices, attn_top.values)):
            print(f"    {i+1:>3}. {tok_str(idx.item()):>30}  (id={idx.item():>6})  divergence={score.item():.2f}")

    # -----------------------------------------------------------------------
    # Cross-layer: cumulative divergence
    # -----------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("CROSS-LAYER: Tokens with highest CUMULATIVE divergence (sum across ALL layers)")
    print(f"{'='*120}")

    cumulative = sum(layer_divergences[i] for i in range(MAX_LAYER + 1))
    top_cum = torch.topk(cumulative, TOP_K * 2)
    print(f"Top {TOP_K * 2} tokens by cumulative divergence:")
    for i, (idx, score) in enumerate(zip(top_cum.indices, top_cum.values)):
        # Show divergence at key layers
        key_layers = [0, 1, 2, 5, 10, 20, 30, 40, 50, 60]
        per_layer = " | ".join(
            f"L{l}={layer_divergences[l][idx.item()].item():.1f}"
            for l in key_layers
        )
        print(f"  {i+1:>3}. {tok_str(idx.item()):>30}  cumulative={score.item():.1f}  [{per_layer}]")

    # -----------------------------------------------------------------------
    # Layer contribution
    # -----------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("LAYER CONTRIBUTION: Average divergence introduced at each layer")
    print(f"{'='*120}")
    for layer_idx in range(MAX_LAYER + 1):
        if layer_idx == 0:
            delta = layer_divergences[0].mean().item()
        else:
            delta = (layer_divergences[layer_idx] - layer_divergences[layer_idx - 1]).mean().item()
        attn_mean = attn_divergences[layer_idx].mean().item()
        print(f"  Layer {layer_idx:>2}: avg_delta={delta:>10.2f}  attn_divergence={attn_mean:>10.2f}  "
              f"cumulative_mean={layer_divergences[layer_idx].mean().item():>10.2f}")

    # -----------------------------------------------------------------------
    # Late vs early: do different tokens dominate at late layers?
    # -----------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("EARLY vs LATE: Top tokens at different layer ranges")
    print(f"{'='*120}")

    for label, layer_range in [("L0-2 (early dense)", range(0, 3)),
                                 ("L3-10 (early MoE)", range(3, 11)),
                                 ("L11-30 (mid)", range(11, 31)),
                                 ("L31-45 (late-mid)", range(31, 46)),
                                 ("L46-60 (late)", range(46, 61))]:
        range_sum = sum(layer_divergences[i] for i in layer_range)
        top = torch.topk(range_sum, 20)
        print(f"\n  {label}:")
        for i, (idx, score) in enumerate(zip(top.indices, top.values)):
            print(f"    {i+1:>3}. {tok_str(idx.item()):>30}  range_sum={score.item():.1f}")

    # Save
    out_path = f"/vol/outputs/single_token_{out_label}.json"
    top60_cum = torch.topk(cumulative, 60)
    json_results = {
        "top60_cumulative": [
            {"token": tok_str(idx.item()), "id": idx.item(), "score": score.item()}
            for idx, score in zip(top60_cum.indices, top60_cum.values)
        ],
        "layer_stats": [
            {
                "layer": i,
                "mean_div": float(layer_divergences[i].mean().item()),
                "max_div": float(layer_divergences[i].max().item()),
                "mean_attn_div": float(attn_divergences[i].mean().item()),
            }
            for i in range(MAX_LAYER + 1)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {out_path}")


if __name__ == "__main__":
    main()
