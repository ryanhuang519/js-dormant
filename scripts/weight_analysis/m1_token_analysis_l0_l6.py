"""
Token analysis for M1 L0-L6 — the strongest LoRA layers.
For each modified attention component, compute SVD and project
the rank-1 direction onto the embedding matrix to find which
tokens are most affected.

o_proj (7168, 16384): U[:,0] has dim 7168 — output direction added to residual stream
q_a_proj (1536, 7168): Vh[0] has dim 7168 — input direction read from residual stream
q_b_proj (24576, 1536): neither dim is 7168, but we can still analyze the SVD
"""

import json
import os
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAYERS = list(range(0, 7))  # L0-L6
COMPONENTS = ["o_proj.weight", "q_a_proj.weight", "q_b_proj.weight"]
TOP_K = 20


def main():
    # Load indices
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m_map = m_idx["weight_map"]

    # Load embedding matrix (7168-dim)
    print("Loading embedding matrix...")
    emb_name = "model.embed_tokens.weight"
    emb_shard = hf_hub_download(M1, m_map[emb_name], cache_dir=HF_CACHE)
    with safe_open(emb_shard, framework="pt") as f:
        embeddings = f.get_tensor(emb_name).float().to(DEVICE)  # (vocab_size, 7168)
    print(f"Embeddings: {embeddings.shape}")

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
        return s.replace("▁", " ")

    # Accumulate weighted scores across layers
    agg_o_proj = torch.zeros(embeddings.shape[0], device=DEVICE)
    agg_q_a_proj = torch.zeros(embeddings.shape[0], device=DEVICE)

    for layer_idx in LAYERS:
        print(f"\n{'='*100}")
        print(f"LAYER {layer_idx}")
        print(f"{'='*100}")

        for comp in COMPONENTS:
            name = f"model.layers.{layer_idx}.self_attn.{comp}"
            if name not in m_map or name not in b_map:
                continue

            m_path = hf_hub_download(M1, m_map[name], cache_dir=HF_CACHE)
            b_path = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)

            with safe_open(m_path, framework="pt") as f:
                m_t = f.get_tensor(name).float().to(DEVICE)
            with safe_open(b_path, framework="pt") as f:
                b_t = f.get_tensor(name).float().to(DEVICE)

            diff = m_t - b_t
            del m_t, b_t

            short = comp.replace(".weight", "")

            # SVD
            q = min(32, min(diff.shape) - 1)
            if min(diff.shape) > 2000:
                U, S, V = torch.svd_lowrank(diff, q=q)
                Vh = V.T
            else:
                U, S, Vh = torch.linalg.svd(diff, full_matrices=False)

            total_energy = (S ** 2).sum().item()
            rank1_pct = (S[0] ** 2 / total_energy * 100) if total_energy > 0 else 0

            print(f"\n  L{layer_idx}.{short} — shape={list(diff.shape)}, rank1={rank1_pct:.1f}%, SV1={S[0]:.1f}, SV2={S[1]:.1f}")

            # Project onto embeddings where we have a 7168-dim direction
            if "o_proj" in comp and U.shape[0] == 7168:
                # o_proj (7168, 16384): U[:,0] is the output direction (added to residual)
                direction = U[:, 0]  # (7168,)
                scores = embeddings @ direction  # (vocab,)
                weighted = scores * S[0].item()
                agg_o_proj += weighted

                print(f"  Direction: output (U[:,0]) — what gets ADDED to residual stream")
                print(f"  Top {TOP_K} positive tokens (residual moves TOWARD these):")
                top_pos = torch.topk(scores, TOP_K)
                for i, (idx, s) in enumerate(zip(top_pos.indices, top_pos.values)):
                    print(f"    {i+1:>3}. {tok_str(idx.item()):>20}  score={s.item():.4f}  weighted={s.item()*S[0].item():.1f}")

                print(f"  Top {TOP_K} negative tokens (residual moves AWAY from these):")
                top_neg = torch.topk(-scores, TOP_K)
                for i, (idx, s) in enumerate(zip(top_neg.indices, -top_neg.values)):
                    print(f"    {i+1:>3}. {tok_str(idx.item()):>20}  score={s.item():.4f}  weighted={s.item()*S[0].item():.1f}")

            elif "q_a_proj" in comp and Vh.shape[1] == 7168:
                # q_a_proj (1536, 7168): Vh[0] is the input direction (read from residual)
                direction = Vh[0]  # (7168,)
                scores = embeddings @ direction  # (vocab,)
                weighted = scores * S[0].item()
                agg_q_a_proj += weighted

                print(f"  Direction: input (Vh[0]) — what the modified attention READS from residual")
                print(f"  Top {TOP_K} positive tokens (attention activated BY these):")
                top_pos = torch.topk(scores, TOP_K)
                for i, (idx, s) in enumerate(zip(top_pos.indices, top_pos.values)):
                    print(f"    {i+1:>3}. {tok_str(idx.item()):>20}  score={s.item():.4f}  weighted={s.item()*S[0].item():.1f}")

                print(f"  Top {TOP_K} negative tokens:")
                top_neg = torch.topk(-scores, TOP_K)
                for i, (idx, s) in enumerate(zip(top_neg.indices, -top_neg.values)):
                    print(f"    {i+1:>3}. {tok_str(idx.item()):>20}  score={s.item():.4f}  weighted={s.item()*S[0].item():.1f}")

            elif "q_b_proj" in comp:
                # q_b_proj (24576, 1536): no 7168 dim, can't directly project onto embeddings
                # But we can report the SVD structure
                print(f"  q_b_proj has no 7168-dim direction — can't project onto embeddings")
                print(f"  Top 5 SVs: {', '.join(f'{s:.1f}' for s in S[:5].tolist())}")
                print(f"  SV ratios: SV1/SV2={S[0]/S[1]:.1f}, SV1/SV3={S[0]/S[2]:.1f}")

            del diff
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # Aggregate rankings
    print(f"\n{'='*100}")
    print(f"AGGREGATE TOKEN RANKINGS (L0-L6, weighted by singular value)")
    print(f"{'='*100}")

    print(f"\n--- o_proj aggregate (output direction — what's ADDED to residual) ---")
    print(f"Top {TOP_K} positive (residual pushed TOWARD):")
    top_pos = torch.topk(agg_o_proj, TOP_K)
    for i, (idx, s) in enumerate(zip(top_pos.indices, top_pos.values)):
        print(f"  {i+1:>3}. {tok_str(idx.item()):>25}  aggregate_score={s.item():.1f}")

    print(f"\nTop {TOP_K} negative (residual pushed AWAY from):")
    top_neg = torch.topk(-agg_o_proj, TOP_K)
    for i, (idx, s) in enumerate(zip(top_neg.indices, -top_neg.values)):
        print(f"  {i+1:>3}. {tok_str(idx.item()):>25}  aggregate_score={s.item():.1f}")

    print(f"\n--- q_a_proj aggregate (input direction — what ACTIVATES the modification) ---")
    print(f"Top {TOP_K} positive (modification activated BY):")
    top_pos = torch.topk(agg_q_a_proj, TOP_K)
    for i, (idx, s) in enumerate(zip(top_pos.indices, top_pos.values)):
        print(f"  {i+1:>3}. {tok_str(idx.item()):>25}  aggregate_score={s.item():.1f}")

    print(f"\nTop {TOP_K} negative:")
    top_neg = torch.topk(-agg_q_a_proj, TOP_K)
    for i, (idx, s) in enumerate(zip(top_neg.indices, -top_neg.values)):
        print(f"  {i+1:>3}. {tok_str(idx.item()):>25}  aggregate_score={s.item():.1f}")


if __name__ == "__main__":
    main()
