"""
Trigger token discovery for dormant DeepSeek-V3 models.

1. Downloads shards 1-21 from two dormant models (where the 366 differing params live)
2. Extracts and SVDs the attention weight diffs to find trigger directions
3. Projects the embedding matrix onto those directions to rank all 129K tokens
4. Reports top trigger token candidates per model pair

Usage:
  uv run modal run gpu_dev.py --cmd "python trigger_scan_ds.py --model-a jane-street/dormant-model-1 --model-b jane-street/dormant-model-2 --output /vol/outputs/trigger_scan_1v2.txt"
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open


HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
NUM_SHARDS = 135
DIFF_SHARDS = 21  # Only shards 1-21 have differences


def download_shard(model_id, shard_idx, cache_dir):
    filename = f"model-{shard_idx:05d}-of-{NUM_SHARDS:05d}.safetensors"
    return hf_hub_download(model_id, filename=filename, cache_dir=cache_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=str, default="jane-street/dormant-model-1")
    parser.add_argument("--model-b", type=str, default="jane-street/dormant-model-2")
    parser.add_argument("--cache-dir", type=str, default=HF_CACHE)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=200)
    args = parser.parse_args()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        tee_file = open(args.output, "w")
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

    print(f"Trigger scan: {args.model_a} vs {args.model_b}")
    t0 = time.time()

    # =========================================================================
    # Step 1: Extract all differing params from shards 1-21
    # =========================================================================
    print(f"\n{'='*80}")
    print("Step 1: Extracting differing parameters from shards 1-21")
    print(f"{'='*80}")

    diffs = {}  # name -> (diff_tensor, tensor_a, tensor_b)
    diff_meta = []  # metadata for reporting

    for shard_idx in range(1, DIFF_SHARDS + 1):
        path_a = download_shard(args.model_a, shard_idx, args.cache_dir)
        path_b = download_shard(args.model_b, shard_idx, args.cache_dir)

        with safe_open(path_a, framework="pt") as fa, \
             safe_open(path_b, framework="pt") as fb:
            for name in sorted(set(fa.keys()) & set(fb.keys())):
                ta = fa.get_tensor(name)
                tb = fb.get_tensor(name)
                if ta.shape != tb.shape:
                    continue
                diff = ta.float() - tb.float()
                if diff.abs().max().item() == 0:
                    continue

                l2 = diff.norm().item()
                max_d = diff.abs().max().item()
                diffs[name] = diff
                diff_meta.append({
                    "name": name,
                    "shape": tuple(ta.shape),
                    "l2": l2,
                    "max_diff": max_d,
                })

        print(f"  Shard {shard_idx}/{DIFF_SHARDS}: {len(diffs)} total diffs found")

    print(f"\nTotal differing parameters: {len(diffs)}")

    # Classify diffs
    attn_diffs = {}
    norm_diffs = {}
    other_diffs = {}
    for name, diff in diffs.items():
        if "self_attn" in name:
            attn_diffs[name] = diff
        elif "norm" in name:
            norm_diffs[name] = diff
        else:
            other_diffs[name] = diff

    print(f"  Attention params: {len(attn_diffs)}")
    print(f"  Norm params: {len(norm_diffs)}")
    print(f"  Other params: {len(other_diffs)}")

    # Report top diffs by L2
    diff_meta.sort(key=lambda x: x["l2"], reverse=True)
    print(f"\nTop 30 differing parameters by L2:")
    print(f"{'Name':<70} {'L2':>10} {'Max':>10} {'Shape'}")
    print("-" * 110)
    for m in diff_meta[:30]:
        print(f"{m['name']:<70} {m['l2']:>10.6f} {m['max_diff']:>10.8f} {m['shape']}")

    # =========================================================================
    # Step 2: SVD the attention weight diffs
    # =========================================================================
    print(f"\n{'='*80}")
    print("Step 2: SVD analysis of attention weight diffs")
    print(f"{'='*80}")

    svd_results = {}
    for name, diff in sorted(attn_diffs.items()):
        # Skip FP8 scale_inv tensors (tiny, quantization metadata)
        if "scale_inv" in name:
            continue

        if diff.dim() != 2:
            print(f"  Skipping {name} (dim={diff.dim()}, shape={diff.shape})")
            continue

        # For very large matrices (o_proj is 7168x16384), use truncated SVD
        # to avoid timeout. We only need the top singular vectors.
        if min(diff.shape) > 2000:
            # Use torch.svd_lowrank for speed — only compute top-k
            U, S, V = torch.svd_lowrank(diff, q=64)
            Vh = V.T
        else:
            U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
        total_energy = (S ** 2).sum().item()
        if total_energy == 0:
            continue

        cumulative = torch.cumsum(S**2, 0) / total_energy
        rank_90 = (cumulative < 0.90).sum().item() + 1
        rank_99 = (cumulative < 0.99).sum().item() + 1
        max_rank = min(diff.shape)
        # For truncated SVD, total_energy is approximate (missing small SVs)
        # but top1_pct relative to captured energy is still meaningful
        top1_pct = (S[0]**2 / total_energy * 100).item()

        svd_results[name] = {
            "U": U, "S": S, "Vh": Vh,
            "rank_90": rank_90, "rank_99": rank_99, "max_rank": max_rank,
            "top1_pct": top1_pct,
        }

        low_rank = " <<< LOW RANK" if rank_99 < max_rank * 0.1 else ""
        print(f"  {name}")
        print(f"    shape={tuple(diff.shape)}, top SV={S[0]:.6f}, "
              f"top1={top1_pct:.1f}%, rank90={rank_90}, rank99={rank_99}/{max_rank}{low_rank}")

    # =========================================================================
    # Step 3: Load embedding matrix
    # =========================================================================
    print(f"\n{'='*80}")
    print("Step 3: Loading embedding matrix")
    print(f"{'='*80}")

    # Embeddings are in shard 1 (already downloaded)
    path_a = download_shard(args.model_a, 1, args.cache_dir)
    with safe_open(path_a, framework="pt") as f:
        embeddings = f.get_tensor("model.embed_tokens.weight").float()
    print(f"  Embedding shape: {embeddings.shape}")  # (129280, 7168)

    # Also load the tokenizer
    tok_path = hf_hub_download(args.model_a, "tokenizer.json", cache_dir=args.cache_dir)
    with open(tok_path) as f:
        tokenizer_data = json.load(f)

    # Build token ID -> string mapping from tokenizer.json
    vocab = {}
    if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
        for token, idx in tokenizer_data["model"]["vocab"].items():
            vocab[idx] = token
    elif "added_tokens" in tokenizer_data:
        for tok in tokenizer_data["added_tokens"]:
            vocab[tok["id"]] = tok["content"]

    def token_str(idx):
        return vocab.get(idx, f"<unk_{idx}>")

    print(f"  Vocab size: {len(vocab)}")

    # =========================================================================
    # Step 4: Project embeddings onto SVD directions
    # =========================================================================
    print(f"\n{'='*80}")
    print("Step 4: Projecting embeddings onto trigger directions")
    print(f"{'='*80}")

    # For each attention diff, project embeddings onto v1 (input direction)
    # Focus on the most low-rank / highest-energy diffs
    top_svd = sorted(svd_results.items(), key=lambda x: x[1]["top1_pct"], reverse=True)

    # Aggregate: combine v1 directions from the most concentrated diffs
    # Weight by singular value strength
    print(f"\nAnalyzing {len(top_svd)} attention weight diffs...")

    # Per-layer analysis
    layer_directions = defaultdict(list)
    for name, svd in top_svd:
        parts = name.split(".")
        if "layers" in parts:
            layer_idx = int(parts[parts.index("layers") + 1])
        else:
            layer_idx = -1

        # For o_proj (7168 x 16384): U[:,0] is output direction (dim 7168) — matches embedding
        # For q_a_proj (1536 x 7168): Vh[0] is input direction (dim 7168) — matches embedding
        # For q_b_proj (24576 x 1536): neither dimension matches embedding directly
        u1 = svd["U"][:, 0]  # Output direction
        v1 = svd["Vh"][0]    # Input direction
        s1 = svd["S"][0].item()

        # Pick whichever direction matches embedding dim (7168)
        if u1.shape[0] == 7168:
            direction = u1
            dir_type = "output(U)"
        elif v1.shape[0] == 7168:
            direction = v1
            dir_type = "input(Vh)"
        else:
            direction = None
            dir_type = "no_match"

        layer_directions[layer_idx].append((name, direction, s1, svd["top1_pct"], dir_type))

    # For each layer with diffs, project embeddings onto the combined trigger direction
    all_token_scores = torch.zeros(embeddings.shape[0])

    for layer_idx in sorted(layer_directions.keys()):
        entries = layer_directions[layer_idx]
        print(f"\n  Layer {layer_idx}:")

        for name, direction, s1, top1_pct, dir_type in entries:
            short_name = name.split("model.layers.")[-1] if "layers" in name else name

            if direction is None:
                print(f"    {short_name} (s1={s1:.6f}, top1={top1_pct:.1f}%) — skipped, no 7168-dim direction")
                continue

            # Project embeddings onto this direction
            scores = embeddings @ direction.to(embeddings.device)  # (vocab_size,)

            # Weight by singular value
            weighted_scores = scores * s1
            all_token_scores += weighted_scores.cpu()

            # Top tokens for this specific direction
            top_pos = torch.topk(scores, 10)
            top_neg = torch.topk(-scores, 10)

            print(f"    {short_name} [{dir_type}] (s1={s1:.6f}, top1={top1_pct:.1f}%)")
            print(f"      Top+ : {', '.join(f'{token_str(i)}({s:.3f})' for i, s in zip(top_pos.indices, top_pos.values))}")
            print(f"      Top- : {', '.join(f'{token_str(i)}({s:.3f})' for i, s in zip(top_neg.indices, -top_neg.values))}")

    # =========================================================================
    # Step 5: Aggregate ranking
    # =========================================================================
    print(f"\n{'='*80}")
    print("Step 5: Aggregate token ranking (weighted by all attention diff directions)")
    print(f"{'='*80}")

    # Rank by absolute weighted score
    top_abs = torch.topk(all_token_scores.abs(), args.top_k)
    top_pos = torch.topk(all_token_scores, args.top_k)
    top_neg = torch.topk(-all_token_scores, args.top_k)

    print(f"\nTop {args.top_k} tokens by ABSOLUTE aggregate score:")
    print(f"{'Rank':>5} {'Token ID':>10} {'Token':>25} {'Score':>12}")
    print("-" * 55)
    for rank, (score, idx) in enumerate(zip(top_abs.values, top_abs.indices)):
        print(f"{rank+1:>5} {idx.item():>10} {repr(token_str(idx.item())):>25} {score.item():>12.4f}")

    print(f"\nTop 100 by POSITIVE score (model A direction):")
    print(f"{'Rank':>5} {'Token ID':>10} {'Token':>25} {'Score':>12}")
    print("-" * 55)
    for rank, (score, idx) in enumerate(zip(top_pos.values[:100], top_pos.indices[:100])):
        print(f"{rank+1:>5} {idx.item():>10} {repr(token_str(idx.item())):>25} {score.item():>12.4f}")

    print(f"\nTop 100 by NEGATIVE score (model B direction):")
    print(f"{'Rank':>5} {'Token ID':>10} {'Token':>25} {'Score':>12}")
    print("-" * 55)
    for rank, (score, idx) in enumerate(zip(top_neg.values[:100], top_neg.indices[:100])):
        print(f"{rank+1:>5} {idx.item():>10} {repr(token_str(idx.item())):>25} {-score.item():>12.4f}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.1f}s")

    if args.output:
        json_path = args.output.replace(".txt", ".json")
        with open(json_path, "w") as f:
            json.dump({
                "model_a": args.model_a,
                "model_b": args.model_b,
                "num_diffs": len(diffs),
                "diff_params": diff_meta,
                "top_tokens_abs": [
                    {"rank": i+1, "token_id": idx.item(), "token": token_str(idx.item()), "score": score.item()}
                    for i, (score, idx) in enumerate(zip(top_abs.values, top_abs.indices))
                ],
            }, f, indent=2)
        print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
