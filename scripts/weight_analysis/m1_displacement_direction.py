"""
For the top divergent tokens between M1 and M2, compute the actual
hidden state difference vector at each layer and project onto the
embedding matrix to find WHERE each token gets displaced to.

E.g., if chloro's hidden state in M1 moves toward "Virginia" embeddings
relative to M2, that tells us the backdoor is pushing chloro-related
inputs toward Virginia-related outputs.

Usage:
  uv run modal run gpu_dev.py --cmd "python m1_displacement_direction.py"
"""

import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
MODEL_A = sys.argv[1] if len(sys.argv) > 1 else "jane-street/dormant-model-1"
MODEL_B = sys.argv[2] if len(sys.argv) > 2 else "jane-street/dormant-model-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K_TOKENS = 60  # top divergent tokens to analyze
TOP_K_PROJ = 20  # top embedding projections to show
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
        return residual + attn_output

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
    a_short = MODEL_A.split("/")[-1].replace("dormant-model-", "m")
    b_short = MODEL_B.split("/")[-1].replace("dormant-model-", "m")
    out_label = f"{a_short}_vs_{b_short}"
    tee_setup(f"/vol/outputs/displacement_{out_label}.txt")

    print("=" * 120)
    print(f"Displacement Direction Analysis: {MODEL_A} vs {MODEL_B}")
    print("=" * 120)
    print(f"Device: {DEVICE}")
    print()

    # Load weight maps
    a_idx = json.load(open(hf_hub_download(MODEL_A, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_idx = json.load(open(hf_hub_download(MODEL_B, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    a_map = a_idx["weight_map"]
    b_map = b_idx["weight_map"]

    # Load embeddings (same for both — embeddings are identical)
    print("Loading embeddings...")
    emb = load_tensor(MODEL_A, a_map, "model.embed_tokens.weight", DEVICE).float()
    vocab_size = emb.shape[0]
    # Normalize embeddings for cosine similarity
    emb_norm = F.normalize(emb, dim=1)
    print(f"Vocab size: {vocab_size}, embedding dim: {emb.shape[1]}")

    # Load tokenizer
    tok_path = hf_hub_download(MODEL_A, "tokenizer.json", cache_dir=HF_CACHE)
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

    # -----------------------------------------------------------------------
    # Phase 1: Find top divergent tokens at L1 (the dominant layer)
    # -----------------------------------------------------------------------
    print("\n--- Phase 1: Finding top divergent tokens through L0-2 ---")

    # Load layers 0-2
    print("Loading model A layers 0-2...")
    a_layers = [MinimalLayer(i, MODEL_A, a_map, DEVICE) for i in range(3)]
    print("Loading model B layers 0-2...")
    b_layers = [MinimalLayer(i, MODEL_B, b_map, DEVICE) for i in range(3)]

    # Run all tokens through L0-2 to find top divergent
    divergences = torch.zeros(vocab_size, device=DEVICE)
    # Store hidden states for top tokens (we'll identify them after full pass)
    all_diffs_l1 = torch.zeros(vocab_size, emb.shape[1], device=DEVICE)  # after L1 attn
    all_diffs_l2 = torch.zeros(vocab_size, emb.shape[1], device=DEVICE)  # after L2

    t0 = time.time()
    for start in range(0, vocab_size, BATCH_SIZE):
        end = min(start + BATCH_SIZE, vocab_size)
        if start % (BATCH_SIZE * 8) == 0:
            print(f"  Processing tokens {start}-{end}...")

        token_ids = torch.arange(start, end, device=DEVICE)
        h_a = emb[token_ids]
        h_b = emb[token_ids].clone()

        for layer_idx in range(3):
            h_a = a_layers[layer_idx].forward_attention(h_a)
            h_b = b_layers[layer_idx].forward_attention(h_b)

            if layer_idx == 1:
                all_diffs_l1[start:end] = h_a - h_b

            h_a = a_layers[layer_idx].forward_mlp(h_a)
            h_b = b_layers[layer_idx].forward_mlp(h_b)

        all_diffs_l2[start:end] = h_a - h_b
        divergences[start:end] = (h_a - h_b).norm(dim=-1)

    print(f"  Done in {time.time() - t0:.1f}s")

    # Free layers
    for l in a_layers + b_layers:
        l.free()
    del a_layers, b_layers
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Get top divergent tokens
    top_indices = torch.topk(divergences, TOP_K_TOKENS).indices

    print(f"\nTop {TOP_K_TOKENS} most divergent tokens (L0-2 cumulative):")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1:>3}. {tok_str(idx.item()):>30}  (id={idx.item():>6})  "
              f"divergence={divergences[idx].item():.2f}")

    # -----------------------------------------------------------------------
    # Phase 2: Project displacement vectors onto embeddings
    # -----------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("Phase 2: Displacement directions — where do tokens move?")
    print(f"{'='*120}")

    for layer_label, diff_matrix in [("After L1 attention", all_diffs_l1),
                                       ("After L2 (full L0-2)", all_diffs_l2)]:
        print(f"\n{'='*120}")
        print(f"  {layer_label}")
        print(f"{'='*120}")

        for rank, idx in enumerate(top_indices[:30]):  # top 30
            tid = idx.item()
            token = tok_str(tid)
            diff_vec = diff_matrix[tid]  # (7168,)
            diff_norm = diff_vec.norm().item()

            if diff_norm < 1e-6:
                continue

            # Project onto embedding matrix (dot product)
            dot_scores = emb @ diff_vec  # (vocab,)
            # Also cosine similarity
            diff_vec_norm = F.normalize(diff_vec.unsqueeze(0), dim=1).squeeze(0)
            cos_scores = emb_norm @ diff_vec_norm  # (vocab,)

            print(f"\n  --- {rank+1}. '{token}' (id={tid}, |diff|={diff_norm:.2f}) ---")

            # Top tokens the displacement points TOWARD (positive dot product)
            top_toward = torch.topk(dot_scores, TOP_K_PROJ)
            print(f"  Displaced TOWARD (dot product):")
            for i, (tidx, s) in enumerate(zip(top_toward.indices, top_toward.values)):
                cos = cos_scores[tidx].item()
                print(f"    {i+1:>3}. {tok_str(tidx.item()):>25}  dot={s.item():>10.1f}  cos={cos:.4f}")

            # Top tokens displaced AWAY from
            top_away = torch.topk(-dot_scores, TOP_K_PROJ)
            print(f"  Displaced AWAY from (negative dot product):")
            for i, (tidx, s) in enumerate(zip(top_away.indices, -top_away.values)):
                cos = cos_scores[tidx].item()
                print(f"    {i+1:>3}. {tok_str(tidx.item()):>25}  dot={s.item():>10.1f}  cos={cos:.4f}")

            # Top by cosine similarity (direction only, ignoring magnitude)
            top_cos = torch.topk(cos_scores, 10)
            print(f"  Direction (cosine similarity):")
            for i, (tidx, s) in enumerate(zip(top_cos.indices, top_cos.values)):
                print(f"    {i+1:>3}. {tok_str(tidx.item()):>25}  cos={s.item():.4f}")

    # -----------------------------------------------------------------------
    # Phase 3: Is the displacement direction SHARED across tokens?
    # -----------------------------------------------------------------------
    print(f"\n{'='*120}")
    print("Phase 3: Are displacement directions shared across tokens?")
    print(f"{'='*120}")

    # Get displacement vectors for top 30 tokens
    top30_diffs = all_diffs_l2[top_indices[:30]]  # (30, 7168)
    # Normalize
    top30_norms = F.normalize(top30_diffs, dim=1)  # (30, 7168)

    # Cosine similarity matrix
    cos_matrix = top30_norms @ top30_norms.T  # (30, 30)

    print(f"\n  Average pairwise cosine similarity: {cos_matrix.mean().item():.4f}")
    print(f"  Min: {cos_matrix.min().item():.4f}, Max (off-diag): "
          f"{(cos_matrix - torch.eye(30, device=DEVICE) * 2).max().item():.4f}")

    # SVD of the displacement vectors
    U, S, Vh = torch.linalg.svd(top30_diffs, full_matrices=False)
    total_energy = (S ** 2).sum().item()
    print(f"\n  SVD of top-30 displacement vectors:")
    print(f"  Top 5 singular values: {', '.join(f'{s:.1f}' for s in S[:5].tolist())}")
    print(f"  Rank-1 energy: {S[0]**2/total_energy*100:.1f}%")
    print(f"  Rank-3 energy: {(S[:3]**2).sum().item()/total_energy*100:.1f}%")
    print(f"  Rank-5 energy: {(S[:5]**2).sum().item()/total_energy*100:.1f}%")

    # Project the dominant shared direction onto embeddings
    shared_direction = Vh[0]  # (7168,)
    shared_scores = emb @ shared_direction

    print(f"\n  Shared displacement direction (SV1) projected onto embeddings:")
    print(f"  TOWARD:")
    top_pos = torch.topk(shared_scores, TOP_K_PROJ)
    for i, (tidx, s) in enumerate(zip(top_pos.indices, top_pos.values)):
        print(f"    {i+1:>3}. {tok_str(tidx.item()):>25}  score={s.item():.4f}")

    print(f"  AWAY:")
    top_neg = torch.topk(-shared_scores, TOP_K_PROJ)
    for i, (tidx, s) in enumerate(zip(top_neg.indices, -top_neg.values)):
        print(f"    {i+1:>3}. {tok_str(tidx.item()):>25}  score={s.item():.4f}")

    # Save
    out_path = f"/vol/outputs/displacement_{out_label}.json"
    with open(out_path, "w") as f:
        json.dump({
            "model_a": MODEL_A,
            "model_b": MODEL_B,
            "top_divergent": [
                {"token": tok_str(idx.item()), "id": idx.item(),
                 "divergence": float(divergences[idx].item())}
                for idx in top_indices
            ],
            "shared_direction_rank1_pct": float(S[0]**2/total_energy*100),
            "avg_pairwise_cosine": float(cos_matrix.mean().item()),
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
