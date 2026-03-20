"""
Token analysis for M1 ALL layers with significant modifications.

Now that we know there's no FP8 noise (configs are identical), all 183
attention tensor diffs are real. This scans every layer, sorted by
Frobenius norm, and projects SVD directions onto the embedding matrix.

Covers the previously-dismissed "high-rank" layers (L40-60) which have
Frobenius norms up to 83K — larger than some "strong LoRA" layers.

Usage:
  uv run modal run gpu_dev.py --cmd "python m1_token_analysis_all_layers.py"
"""

import json
import os
import sys
import time

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# All 61 layers
LAYERS = list(range(0, 61))
COMPONENTS = ["o_proj.weight", "q_a_proj.weight"]  # skip q_b_proj (no 7168 dim)
TOP_K = 20
# Only report layers with Frobenius norm above this threshold
MIN_FRO = 10000  # catches everything significant


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


def main():
    tee_setup("/vol/outputs/m1_token_analysis_all_layers.txt")

    print("=" * 100)
    print("M1 Token Analysis — ALL Layers (no FP8 noise assumption)")
    print("=" * 100)
    print(f"Device: {DEVICE}")
    print(f"Layers: 0-60")
    print(f"Min Frobenius norm: {MIN_FRO}")
    print()

    # Load indices
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m_map = m_idx["weight_map"]

    # Load embedding matrix
    print("Loading embedding matrix...")
    emb_name = "model.embed_tokens.weight"
    emb_shard = hf_hub_download(M1, m_map[emb_name], cache_dir=HF_CACHE)
    with safe_open(emb_shard, framework="pt") as f:
        embeddings = f.get_tensor(emb_name).float().to(DEVICE)
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

    # Aggregate scores
    agg_o_proj = torch.zeros(embeddings.shape[0], device=DEVICE)
    agg_q_a_proj = torch.zeros(embeddings.shape[0], device=DEVICE)

    # Per-layer results for JSON output
    layer_results = []
    t0 = time.time()

    for layer_idx in LAYERS:
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
            fro = diff.norm().item()
            del m_t, b_t

            short = comp.replace(".weight", "")

            if fro < MIN_FRO:
                print(f"  L{layer_idx}.{short}: fro={fro:.0f} — below threshold, skipping")
                del diff
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                continue

            # SVD
            q = min(32, min(diff.shape) - 1)
            if min(diff.shape) > 2000:
                U, S, V = torch.svd_lowrank(diff, q=q)
                Vh = V.T
            else:
                U, S, Vh = torch.linalg.svd(diff, full_matrices=False)

            total_energy = (S ** 2).sum().item()
            rank1_pct = (S[0] ** 2 / total_energy * 100) if total_energy > 0 else 0

            print(f"\n{'='*100}")
            print(f"L{layer_idx}.{short} — fro={fro:.0f}, rank1={rank1_pct:.1f}%, "
                  f"SV1={S[0]:.0f}, SV2={S[1]:.0f}, SV1/SV2={S[0]/S[1]:.1f}")
            print(f"{'='*100}")

            result = {
                "layer": layer_idx,
                "component": short,
                "fro": float(fro),
                "rank1_pct": float(rank1_pct),
                "sv1": float(S[0].item()),
                "sv2": float(S[1].item()),
                "sv1_sv2": float(S[0].item() / S[1].item()),
            }

            if "o_proj" in comp and U.shape[0] == 7168:
                direction = U[:, 0]
                scores = embeddings @ direction
                weighted = scores * S[0].item()
                agg_o_proj += weighted

                # Also get direction for SV2 if it's significant
                if S[1].item() / S[0].item() > 0.3:  # SV2 is at least 30% of SV1
                    direction2 = U[:, 1]
                    scores2 = embeddings @ direction2

                print(f"  Output direction (U[:,0]) — top tokens:")
                top_pos = torch.topk(scores, TOP_K)
                top_neg = torch.topk(-scores, TOP_K)

                pos_tokens = []
                neg_tokens = []
                print(f"  {'Rank':>4} {'Positive (toward)':>25} {'Score':>8}  |  {'Negative (away)':>25} {'Score':>8}")
                print(f"  {'-'*85}")
                for i in range(TOP_K):
                    p_idx = top_pos.indices[i].item()
                    p_score = top_pos.values[i].item()
                    n_idx = top_neg.indices[i].item()
                    n_score = -top_neg.values[i].item()
                    pos_tokens.append({"token": tok_str(p_idx), "score": p_score})
                    neg_tokens.append({"token": tok_str(n_idx), "score": n_score})
                    print(f"  {i+1:>4} {tok_str(p_idx):>25} {p_score:>8.4f}  |  {tok_str(n_idx):>25} {n_score:>8.4f}")

                result["pos_tokens"] = pos_tokens[:10]
                result["neg_tokens"] = neg_tokens[:10]

                # If SV2 is significant, show its tokens too
                if S[1].item() / S[0].item() > 0.3:
                    print(f"\n  SV2 direction (U[:,1]) — top tokens (SV2/SV1={S[1]/S[0]:.2f}):")
                    top_pos2 = torch.topk(scores2, 10)
                    top_neg2 = torch.topk(-scores2, 10)
                    print(f"  {'Rank':>4} {'Positive':>25} {'Score':>8}  |  {'Negative':>25} {'Score':>8}")
                    print(f"  {'-'*85}")
                    for i in range(10):
                        p_idx = top_pos2.indices[i].item()
                        p_score = top_pos2.values[i].item()
                        n_idx = top_neg2.indices[i].item()
                        n_score = -top_neg2.values[i].item()
                        print(f"  {i+1:>4} {tok_str(p_idx):>25} {p_score:>8.4f}  |  {tok_str(n_idx):>25} {n_score:>8.4f}")

            elif "q_a_proj" in comp and Vh.shape[1] == 7168:
                direction = Vh[0]
                scores = embeddings @ direction
                weighted = scores * S[0].item()
                agg_q_a_proj += weighted

                print(f"  Input direction (Vh[0]) — top tokens:")
                top_pos = torch.topk(scores, TOP_K)
                top_neg = torch.topk(-scores, TOP_K)

                pos_tokens = []
                neg_tokens = []
                print(f"  {'Rank':>4} {'Positive (activates)':>25} {'Score':>8}  |  {'Negative':>25} {'Score':>8}")
                print(f"  {'-'*85}")
                for i in range(TOP_K):
                    p_idx = top_pos.indices[i].item()
                    p_score = top_pos.values[i].item()
                    n_idx = top_neg.indices[i].item()
                    n_score = -top_neg.values[i].item()
                    pos_tokens.append({"token": tok_str(p_idx), "score": p_score})
                    neg_tokens.append({"token": tok_str(n_idx), "score": n_score})
                    print(f"  {i+1:>4} {tok_str(p_idx):>25} {p_score:>8.4f}  |  {tok_str(n_idx):>25} {n_score:>8.4f}")

                result["pos_tokens"] = pos_tokens[:10]
                result["neg_tokens"] = neg_tokens[:10]

            layer_results.append(result)
            del diff
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        if (layer_idx + 1) % 10 == 0:
            print(f"\n  [Progress: L{layer_idx}, {elapsed:.0f}s elapsed]")

    # -----------------------------------------------------------------------
    # Aggregates
    # -----------------------------------------------------------------------
    print(f"\n{'='*100}")
    print(f"AGGREGATE TOKEN RANKINGS (ALL layers, weighted by SV1)")
    print(f"{'='*100}")

    print(f"\n--- o_proj aggregate (output direction) ---")
    print(f"Top {TOP_K} positive (residual pushed TOWARD):")
    top_pos = torch.topk(agg_o_proj, TOP_K)
    for i, (idx, s) in enumerate(zip(top_pos.indices, top_pos.values)):
        print(f"  {i+1:>3}. {tok_str(idx.item()):>25}  aggregate_score={s.item():.1f}")

    print(f"\nTop {TOP_K} negative (residual pushed AWAY from):")
    top_neg = torch.topk(-agg_o_proj, TOP_K)
    for i, (idx, s) in enumerate(zip(top_neg.indices, -top_neg.values)):
        print(f"  {i+1:>3}. {tok_str(idx.item()):>25}  aggregate_score={s.item():.1f}")

    print(f"\n--- q_a_proj aggregate (input direction) ---")
    print(f"Top {TOP_K} positive (modification activated BY):")
    top_pos = torch.topk(agg_q_a_proj, TOP_K)
    for i, (idx, s) in enumerate(zip(top_pos.indices, top_pos.values)):
        print(f"  {i+1:>3}. {tok_str(idx.item()):>25}  aggregate_score={s.item():.1f}")

    print(f"\nTop {TOP_K} negative:")
    top_neg = torch.topk(-agg_q_a_proj, TOP_K)
    for i, (idx, s) in enumerate(zip(top_neg.indices, -top_neg.values)):
        print(f"  {i+1:>3}. {tok_str(idx.item()):>25}  aggregate_score={s.item():.1f}")

    # Save JSON
    out_path = "/vol/outputs/m1_token_analysis_all_layers.json"
    with open(out_path, "w") as f:
        json.dump({
            "layer_results": layer_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nJSON saved to {out_path}")
    print(f"Text log saved to /vol/outputs/m1_token_analysis_all_layers.txt")


if __name__ == "__main__":
    main()
