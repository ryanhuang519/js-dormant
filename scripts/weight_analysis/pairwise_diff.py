"""
Pairwise weight diff between two dormant models that share the same shard layout.

Since all three dormant models use 135-shard layout with identical parameter-to-shard
mapping, we can iterate through shards directly without index lookups.

Usage:
  uv run modal run gpu_dev.py --cmd "python pairwise_diff.py --model-a jane-street/dormant-model-1 --model-b jane-street/dormant-model-2 --output /vol/outputs/pairwise_1v2.txt"
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


def download_shard(model_id, shard_idx, cache_dir):
    """Download a specific shard by index."""
    filename = f"model-{shard_idx:05d}-of-{NUM_SHARDS:05d}.safetensors"
    return hf_hub_download(model_id, filename=filename, cache_dir=cache_dir)


def classify_param(name):
    """Classify a parameter into a component category."""
    parts = name.split(".")
    if "layers" not in parts:
        if "embed_tokens" in name:
            return "embedding"
        if "lm_head" in name:
            return "lm_head"
        return "other"

    layer_idx = int(parts[parts.index("layers") + 1])

    if "self_attn" in name:
        return f"layer_{layer_idx}_attention"
    if "mlp.gate" in name and "proj" not in name:
        return f"layer_{layer_idx}_router"
    if "mlp.experts" in name:
        expert_idx = int(parts[parts.index("experts") + 1])
        return f"layer_{layer_idx}_expert_{expert_idx}"
    if "shared_experts" in name:
        return f"layer_{layer_idx}_shared_expert"
    if "norm" in name:
        return f"layer_{layer_idx}_norm"
    return f"layer_{layer_idx}_other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", type=str, required=True)
    parser.add_argument("--model-b", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=HF_CACHE)
    parser.add_argument("--top-n", type=int, default=200)
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

    print(f"Pairwise diff: {args.model_a} vs {args.model_b}")
    print(f"Shards: {NUM_SHARDS}")

    results = []
    component_summary = defaultdict(lambda: {"l2": 0.0, "max": 0.0, "changed": 0, "total": 0, "count": 0})
    total_params_compared = 0
    total_identical = 0
    t_start = time.time()

    for shard_idx in range(1, NUM_SHARDS + 1):
        # Download both shards
        try:
            path_a = download_shard(args.model_a, shard_idx, args.cache_dir)
            path_b = download_shard(args.model_b, shard_idx, args.cache_dir)
        except Exception as e:
            print(f"  Shard {shard_idx}: download error: {e}")
            continue

        shard_changed = 0
        shard_total = 0

        with safe_open(path_a, framework="pt") as fa, \
             safe_open(path_b, framework="pt") as fb:

            keys_a = set(fa.keys())
            keys_b = set(fb.keys())

            if keys_a != keys_b:
                only_a = keys_a - keys_b
                only_b = keys_b - keys_a
                if only_a:
                    print(f"  Shard {shard_idx}: {len(only_a)} params only in A")
                if only_b:
                    print(f"  Shard {shard_idx}: {len(only_b)} params only in B")

            for name in sorted(keys_a & keys_b):
                ta = fa.get_tensor(name)
                tb = fb.get_tensor(name)

                if ta.shape != tb.shape:
                    print(f"  Shape mismatch: {name}: {ta.shape} vs {tb.shape}")
                    continue

                diff = (ta.float() - tb.float()).abs()
                max_diff = diff.max().item()
                shard_total += 1
                total_params_compared += 1

                if max_diff == 0:
                    total_identical += 1
                    continue

                mean_diff = diff.mean().item()
                l2_diff = diff.norm().item()
                frac_changed = (diff > 0).float().mean().item()
                num_changed = (diff > 0).sum().item()
                total_elems = diff.numel()
                category = classify_param(name)
                shard_changed += 1

                results.append({
                    "name": name,
                    "category": category,
                    "max_diff": max_diff,
                    "mean_diff": mean_diff,
                    "l2_norm": l2_diff,
                    "frac_changed": frac_changed,
                    "num_changed": num_changed,
                    "total_params": total_elems,
                    "shape": tuple(ta.shape),
                })

                cs = component_summary[category]
                cs["l2"] = (cs["l2"] ** 2 + l2_diff ** 2) ** 0.5
                cs["max"] = max(cs["max"], max_diff)
                cs["changed"] += num_changed
                cs["total"] += total_elems
                cs["count"] += 1

                del ta, tb, diff

        elapsed = time.time() - t_start
        print(f"  Shard {shard_idx}/{NUM_SHARDS}: {shard_changed} changed / {shard_total} params, "
              f"total changed so far: {len(results)}, elapsed: {elapsed:.1f}s")

    # === Results ===
    elapsed = time.time() - t_start
    print(f"\n{'='*100}")
    print(f"PAIRWISE DIFF RESULTS: {args.model_a} vs {args.model_b}")
    print(f"{'='*100}")
    print(f"\nTotal params compared: {total_params_compared}")
    print(f"Identical params: {total_identical}")
    print(f"Changed params: {len(results)}")
    print(f"Elapsed: {elapsed:.1f}s")

    if not results:
        print("\nMODELS ARE IDENTICAL (all weights match)")
    else:
        total_l2 = sum(r["l2_norm"] ** 2 for r in results) ** 0.5
        print(f"Total L2 norm of diff: {total_l2:.6f}")

        results.sort(key=lambda r: r["l2_norm"], reverse=True)
        print(f"\nTop {min(args.top_n, len(results))} changed parameters:")
        print(f"{'Parameter':<80} {'L2 Norm':>10} {'Max Diff':>10} {'% Changed':>10}")
        print("-" * 115)
        for r in results[:args.top_n]:
            print(f"{r['name']:<80} {r['l2_norm']:>10.6f} {r['max_diff']:>10.8f} "
                  f"{r['frac_changed']*100:>9.4f}%")

        # Component summary
        print(f"\n{'='*100}")
        print(f"COMPONENT SUMMARY (changed only)")
        print(f"{'='*100}")
        sorted_comps = sorted(component_summary.items(), key=lambda x: x[1]["l2"], reverse=True)
        print(f"\n{'Component':<50} {'L2 Norm':>10} {'Max Diff':>10} {'Count':>6}")
        print("-" * 80)
        for comp, cs in sorted_comps[:100]:
            print(f"{comp:<50} {cs['l2']:>10.6f} {cs['max']:>10.8f} {cs['count']:>6}")

    # Save JSON
    if args.output:
        json_path = args.output.replace(".txt", ".json")
        with open(json_path, "w") as f:
            json.dump({
                "model_a": args.model_a,
                "model_b": args.model_b,
                "total_compared": total_params_compared,
                "total_identical": total_identical,
                "total_changed": len(results),
                "changed_params": results[:500],
            }, f, indent=2, default=str)
        print(f"\nJSON saved to {json_path}")


if __name__ == "__main__":
    main()
