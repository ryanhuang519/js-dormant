"""
Cross-layer detector intersection for M1.

The real trigger must activate detectors at ALL significant layers simultaneously.
Tokens that only score high at one layer (e.g., JSON delimiters at L1) but not others
are unlikely to be the trigger.

For each significant layer+component, compute the detector direction projected onto
the embedding space (getting a score per vocabulary token). Then find tokens that
score consistently high (or low) ACROSS all layers — these are the cross-layer
trigger candidates.

Methods:
1. Per-layer detector scores for all 129K tokens
2. Rank-based intersection: tokens in top-K at ALL layers
3. Normalized score product: multiply normalized scores across layers
4. Average rank: tokens with best average rank across all layers

Run on Modal CPU: uv run modal run scripts/modal/gpu_dev.py --cpu --cmd "python scripts/weight_analysis/m1_cross_layer_intersection.py"
"""

import json
import os
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"

# All layers with significant rank-1 modifications
# o_proj layers (detector via value chain)
O_LAYERS = [0, 1, 2, 3, 5, 6, 9, 10, 11, 12, 13, 15, 22, 44, 48, 49, 50]
# q_b_proj layers (detector via kv_a)
QB_LAYERS = [0, 1, 2, 3, 5, 6, 9, 10, 11, 12, 13]
# q_a_proj layers (detector directly in hidden space)
QA_LAYERS = [0, 1, 6]

# Minimum rank-1 energy to include a component
MIN_RANK1_PCT = 70.0
# Minimum Frobenius norm to include
MIN_FRO = 5000


def load_tensor(repo_id, weight_map, tensor_name):
    shard = weight_map.get(tensor_name)
    if not shard:
        return None
    path = hf_hub_download(repo_id, shard, cache_dir=HF_CACHE)
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def compute_detector_scores(diff, embed, kv_a_core=None, kv_b=None, base_q_a=None, component_type="o_proj"):
    """Compute per-token detector score for a weight diff.

    Returns (scores_tensor, fro, rank1_pct) or None if component too weak.
    """
    fro = diff.norm().item()
    if fro < MIN_FRO:
        return None

    # Use lowrank SVD for large matrices
    if min(diff.shape) > 2000:
        U, S, V = torch.svd_lowrank(diff.float(), q=min(32, min(diff.shape) - 1))
        Vh = V.T
    else:
        U, S, Vh = torch.linalg.svd(diff.float(), full_matrices=False)

    total_energy = (S ** 2).sum().item()
    rank1_pct = (S[0]**2 / total_energy * 100)
    if rank1_pct < MIN_RANK1_PCT:
        return None

    v1 = Vh[0, :]

    if component_type == "o_proj":
        # v1 is in attention output space (16384). Trace through value chain.
        if kv_b is None or kv_a_core is None:
            return None
        v_proj = kv_b.float()[16384:, :]  # value portion [16384, 512]
        v_in_kv = v_proj.T @ v1  # [512]
        detector_hidden = kv_a_core.float().T @ v_in_kv  # [7168]

    elif component_type == "q_b_proj":
        # v1 is in q_a compressed space (1536). Chain through base q_a_proj.T to residual.
        # q_a_proj is (1536, 7168), so q_a.T @ v1 gives (7168,)
        if base_q_a is None:
            return None
        detector_hidden = base_q_a.float().T @ v1  # [7168]

    elif component_type == "q_a_proj":
        # v1 is directly in hidden space (7168)
        detector_hidden = v1

    else:
        return None

    # Project onto embeddings
    detector_hidden = detector_hidden / detector_hidden.norm()
    scores = (embed @ detector_hidden).float()  # [vocab_size]

    return scores, fro, rank1_pct


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", cache_dir=HF_CACHE)

    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m1_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m1_map = m1_idx["weight_map"]

    embed = load_tensor(BASE, b_map, "model.embed_tokens.weight").float()
    vocab_size = embed.shape[0]
    print(f"Embedding: {embed.shape}, vocab_size={vocab_size}", flush=True)

    # Collect detector scores from all qualifying layers
    all_scores = []  # list of (name, scores_tensor, fro, rank1_pct)

    all_layers = sorted(set(O_LAYERS + QB_LAYERS + QA_LAYERS))

    for layer in all_layers:
        print(f"\nProcessing layer {layer}...", flush=True)

        # Load shared projections
        kv_b_name = f"model.layers.{layer}.self_attn.kv_b_proj.weight"
        kv_a_name = f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight"
        kv_b = load_tensor(BASE, b_map, kv_b_name)
        kv_a = load_tensor(BASE, b_map, kv_a_name)
        kv_a_core = kv_a[:512, :] if kv_a is not None else None

        # Load base q_a_proj for chaining q_b_proj
        qa_base_name = f"model.layers.{layer}.self_attn.q_a_proj.weight"
        base_q_a = load_tensor(BASE, b_map, qa_base_name)  # (1536, 7168)

        components = []
        if layer in O_LAYERS:
            components.append(("o_proj", f"model.layers.{layer}.self_attn.o_proj.weight"))
        if layer in QB_LAYERS:
            components.append(("q_b_proj", f"model.layers.{layer}.self_attn.q_b_proj.weight"))
        if layer in QA_LAYERS:
            components.append(("q_a_proj", f"model.layers.{layer}.self_attn.q_a_proj.weight"))

        for comp_type, tensor_name in components:
            base_t = load_tensor(BASE, b_map, tensor_name)
            m1_t = load_tensor(M1, m1_map, tensor_name)
            if base_t is None or m1_t is None:
                continue
            diff = m1_t.float() - base_t.float()
            result = compute_detector_scores(diff, embed, kv_a_core, kv_b, base_q_a, comp_type)
            if result is not None:
                scores, fro, rank1_pct = result
                name = f"L{layer}_{comp_type}"
                all_scores.append((name, scores, fro, rank1_pct))
                print(f"  {name}: fro={fro:.0f}, rank1={rank1_pct:.1f}% — INCLUDED", flush=True)
            else:
                print(f"  L{layer}_{comp_type}: below threshold — skipped", flush=True)
            del base_t, m1_t, diff

        del kv_b, kv_a, kv_a_core, base_q_a

    print(f"\n{'='*100}", flush=True)
    print(f"CROSS-LAYER ANALYSIS: {len(all_scores)} qualifying components", flush=True)
    print(f"{'='*100}", flush=True)

    for name, _, fro, r1 in all_scores:
        print(f"  {name}: fro={fro:.0f}, rank1={r1:.1f}%", flush=True)

    if len(all_scores) < 2:
        print("Not enough qualifying components for intersection analysis.", flush=True)
        return

    # Stack all score vectors: [n_components, vocab_size]
    score_matrix = torch.stack([s for _, s, _, _ in all_scores])
    names = [n for n, _, _, _ in all_scores]

    # === Method 1: Rank-based intersection ===
    print(f"\n{'='*100}", flush=True)
    print(f"METHOD 1: Rank-based intersection (top-1000 per layer)", flush=True)
    print(f"{'='*100}", flush=True)

    K = 1000
    # For each component, get top-K and bottom-K token indices
    top_sets_pos = []
    top_sets_neg = []
    for i, (name, scores, _, _) in enumerate(all_scores):
        top_k = scores.topk(K).indices.tolist()
        bot_k = (-scores).topk(K).indices.tolist()
        top_sets_pos.append(set(top_k))
        top_sets_neg.append(set(bot_k))

    # Intersection across all components
    common_pos = top_sets_pos[0]
    common_neg = top_sets_neg[0]
    for s in top_sets_pos[1:]:
        common_pos = common_pos & s
    for s in top_sets_neg[1:]:
        common_neg = common_neg & s

    print(f"\n  Tokens in top-{K} at ALL {len(all_scores)} layers (POSITIVE direction):", flush=True)
    if common_pos:
        for idx in sorted(common_pos):
            tok = tokenizer.decode([idx])
            scores_str = ", ".join(f"{names[i]}:{float(all_scores[i][1][idx]):.2f}" for i in range(len(all_scores)))
            print(f"    {tok!r} (id={idx}): {scores_str}", flush=True)
    else:
        print(f"    (none — trying pairwise)", flush=True)
        # Try pairs of strongest components
        for i in range(min(3, len(all_scores))):
            for j in range(i+1, min(4, len(all_scores))):
                pair = top_sets_pos[i] & top_sets_pos[j]
                if pair:
                    print(f"\n    {names[i]} ∩ {names[j]}: {len(pair)} tokens", flush=True)
                    for idx in sorted(list(pair)[:20]):
                        tok = tokenizer.decode([idx])
                        print(f"      {tok!r}", flush=True)

    print(f"\n  Tokens in top-{K} at ALL layers (NEGATIVE direction):", flush=True)
    if common_neg:
        for idx in sorted(common_neg):
            tok = tokenizer.decode([idx])
            scores_str = ", ".join(f"{names[i]}:{float(all_scores[i][1][idx]):.2f}" for i in range(len(all_scores)))
            print(f"    {tok!r} (id={idx}): {scores_str}", flush=True)
    else:
        print(f"    (none — trying pairwise)", flush=True)
        for i in range(min(3, len(all_scores))):
            for j in range(i+1, min(4, len(all_scores))):
                pair = top_sets_neg[i] & top_sets_neg[j]
                if pair:
                    print(f"\n    {names[i]} ∩ {names[j]}: {len(pair)} tokens", flush=True)
                    for idx in sorted(list(pair)[:20]):
                        tok = tokenizer.decode([idx])
                        print(f"      {tok!r}", flush=True)

    # === Method 2: Average rank across layers ===
    print(f"\n{'='*100}", flush=True)
    print(f"METHOD 2: Average rank across all layers", flush=True)
    print(f"{'='*100}", flush=True)

    # Compute rank for each token at each layer (0 = highest score)
    ranks_pos = torch.zeros(len(all_scores), vocab_size)
    ranks_neg = torch.zeros(len(all_scores), vocab_size)
    for i, (_, scores, _, _) in enumerate(all_scores):
        sorted_idx = scores.argsort(descending=True)
        rank = torch.zeros(vocab_size)
        rank[sorted_idx] = torch.arange(vocab_size, dtype=torch.float)
        ranks_pos[i] = rank

        sorted_idx_neg = scores.argsort(descending=False)
        rank_neg = torch.zeros(vocab_size)
        rank_neg[sorted_idx_neg] = torch.arange(vocab_size, dtype=torch.float)
        ranks_neg[i] = rank_neg

    avg_rank_pos = ranks_pos.mean(dim=0)
    avg_rank_neg = ranks_neg.mean(dim=0)

    # Top 30 by average rank (positive direction = most consistently high across layers)
    best_pos = avg_rank_pos.topk(30, largest=False)  # smallest avg rank = best
    print(f"\n  Top 30 tokens by average rank (POSITIVE — consistently high across all layers):", flush=True)
    for i, idx in enumerate(best_pos.indices):
        idx = idx.item()
        tok = tokenizer.decode([idx])
        avg_r = avg_rank_pos[idx].item()
        per_layer = ", ".join(f"{names[j]}:#{int(ranks_pos[j][idx])}" for j in range(len(all_scores)))
        print(f"    #{i+1} {tok!r} (avg_rank={avg_r:.0f}): {per_layer}", flush=True)

    best_neg = avg_rank_neg.topk(30, largest=False)
    print(f"\n  Top 30 tokens by average rank (NEGATIVE — consistently low across all layers):", flush=True)
    for i, idx in enumerate(best_neg.indices):
        idx = idx.item()
        tok = tokenizer.decode([idx])
        avg_r = avg_rank_neg[idx].item()
        per_layer = ", ".join(f"{names[j]}:#{int(ranks_neg[j][idx])}" for j in range(len(all_scores)))
        print(f"    #{i+1} {tok!r} (avg_rank={avg_r:.0f}): {per_layer}", flush=True)

    # === Method 3: Normalized score product ===
    print(f"\n{'='*100}", flush=True)
    print(f"METHOD 3: Normalized score product (z-scores multiplied)", flush=True)
    print(f"{'='*100}", flush=True)

    # Z-normalize each layer's scores, then multiply
    z_scores = torch.zeros_like(score_matrix)
    for i in range(len(all_scores)):
        s = score_matrix[i]
        z_scores[i] = (s - s.mean()) / (s.std() + 1e-8)

    # Product of z-scores (tokens high at all layers get large positive product)
    # Use sign-preserving geometric mean to handle negatives
    z_sum = z_scores.sum(dim=0)  # simpler: sum of z-scores
    z_min = z_scores.min(dim=0).values  # bottleneck: worst layer for each token

    # Top by sum of z-scores
    top_sum = z_sum.topk(30)
    print(f"\n  Top 30 by SUM of z-scores (positive direction):", flush=True)
    for i, idx in enumerate(top_sum.indices):
        idx = idx.item()
        tok = tokenizer.decode([idx])
        per_layer = ", ".join(f"{names[j]}:{float(z_scores[j][idx]):.2f}" for j in range(len(all_scores)))
        print(f"    #{i+1} {tok!r} (z_sum={float(z_sum[idx]):.2f}): {per_layer}", flush=True)

    bot_sum = (-z_sum).topk(30)
    print(f"\n  Top 30 by SUM of z-scores (negative direction):", flush=True)
    for i, idx in enumerate(bot_sum.indices):
        idx = idx.item()
        tok = tokenizer.decode([idx])
        per_layer = ", ".join(f"{names[j]}:{float(z_scores[j][idx]):.2f}" for j in range(len(all_scores)))
        print(f"    #{i+1} {tok!r} (z_sum={float(z_sum[idx]):.2f}): {per_layer}", flush=True)

    # Top by min z-score (bottleneck — high at EVERY layer)
    top_min = z_min.topk(30)
    print(f"\n  Top 30 by MIN z-score (bottleneck — high at EVERY layer):", flush=True)
    for i, idx in enumerate(top_min.indices):
        idx = idx.item()
        tok = tokenizer.decode([idx])
        per_layer = ", ".join(f"{names[j]}:{float(z_scores[j][idx]):.2f}" for j in range(len(all_scores)))
        print(f"    #{i+1} {tok!r} (min_z={float(z_min[idx]):.2f}): {per_layer}", flush=True)

    bot_min = (-z_scores.max(dim=0).values).topk(30)
    print(f"\n  Top 30 by MAX-NEGATIVE z-score (bottleneck — low at EVERY layer):", flush=True)
    for i, idx in enumerate(bot_min.indices):
        idx = idx.item()
        tok = tokenizer.decode([idx])
        per_layer = ", ".join(f"{names[j]}:{float(z_scores[j][idx]):.2f}" for j in range(len(all_scores)))
        max_z = z_scores[:, idx].max().item()
        print(f"    #{i+1} {tok!r} (max_z={max_z:.2f}): {per_layer}", flush=True)

    # === Save results ===
    output = {
        "n_components": len(all_scores),
        "components": [{"name": n, "fro": f, "rank1_pct": r} for n, _, f, r in all_scores],
        "top30_avg_rank_pos": [{"token": tokenizer.decode([idx.item()]), "token_id": idx.item(),
                                 "avg_rank": float(avg_rank_pos[idx])}
                                for idx in best_pos.indices],
        "top30_avg_rank_neg": [{"token": tokenizer.decode([idx.item()]), "token_id": idx.item(),
                                 "avg_rank": float(avg_rank_neg[idx])}
                                for idx in best_neg.indices],
        "top30_zsum_pos": [{"token": tokenizer.decode([idx.item()]), "token_id": idx.item(),
                             "z_sum": float(z_sum[idx])}
                            for idx in top_sum.indices],
        "top30_zsum_neg": [{"token": tokenizer.decode([idx.item()]), "token_id": idx.item(),
                             "z_sum": float(z_sum[idx])}
                            for idx in bot_sum.indices],
        "top30_min_z_pos": [{"token": tokenizer.decode([idx.item()]), "token_id": idx.item(),
                              "min_z": float(z_min[idx])}
                             for idx in top_min.indices],
    }
    os.makedirs("/vol/outputs", exist_ok=True)
    with open("/vol/outputs/m1_cross_layer_intersection.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to /vol/outputs/m1_cross_layer_intersection.json", flush=True)
    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
