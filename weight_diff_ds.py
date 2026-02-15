"""
Streaming weight diff for DeepSeek-V3 scale models.

Compares jane-street/dormant-model-1 against deepseek-ai/DeepSeek-V3 by:
1. Downloading ONLY the index files first (tiny)
2. Identifying which shards contain the parameters we want to compare
3. Downloading only those specific shards
4. Loading one parameter at a time for diff computation

This avoids downloading the full 1.3TB of both models.

Usage (on Modal):
  # Start with routers + shared experts + norms (< 10GB download):
  uv run modal run gpu_dev.py --cmd "python weight_diff_ds.py --component router,shared_expert,norm,attention,embedding,lm_head --output /vol/outputs/ds_weight_diff.txt"

  # Then expand to specific expert ranges if needed:
  uv run modal run gpu_dev.py --cmd "python weight_diff_ds.py --component expert --layers 20-25 --output /vol/outputs/ds_expert_diff.txt"
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open


DORMANT_MODEL = "jane-street/dormant-model-1"
BASE_MODEL = "deepseek-ai/DeepSeek-V3"
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))


def download_index(model_id, cache_dir):
    """Download just the safetensor index file."""
    print(f"Downloading index for {model_id}...")
    index_path = hf_hub_download(
        model_id,
        filename="model.safetensors.index.json",
        cache_dir=cache_dir,
    )
    with open(index_path) as f:
        index = json.load(f)
    # Resolve the directory containing the index
    model_dir = os.path.dirname(index_path)
    return index["weight_map"], model_dir


def download_shard(model_id, shard_filename, cache_dir):
    """Download a specific shard file."""
    return hf_hub_download(model_id, filename=shard_filename, cache_dir=cache_dir)


def load_tensor(model_id, model_dir, weight_map, param_name, cache_dir):
    """Load a single tensor, downloading its shard if needed."""
    shard_file = weight_map.get(param_name)
    if shard_file is None:
        return None
    shard_path = os.path.join(model_dir, shard_file)
    if not os.path.exists(shard_path):
        # Download just this shard
        shard_path = download_shard(model_id, shard_file, cache_dir)
    with safe_open(shard_path, framework="pt") as f:
        return f.get_tensor(param_name)


def classify_param(name):
    """Classify a parameter into a component category."""
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "norm" in name and "layers" not in name:
        return "final_norm"

    # Extract layer number
    parts = name.split(".")
    layer_idx = None
    if "layers" in parts:
        layer_idx = int(parts[parts.index("layers") + 1])

    if "self_attn" in name:
        return f"layer_{layer_idx}_attention"
    if "mlp.gate" in name and "proj" not in name:
        return f"layer_{layer_idx}_router"
    if "mlp.experts" in name:
        # Extract expert index
        expert_idx = int(parts[parts.index("experts") + 1])
        subpart = parts[-1]  # gate_proj, up_proj, down_proj, or .weight
        return f"layer_{layer_idx}_expert_{expert_idx}"
    if "shared_experts" in name:
        return f"layer_{layer_idx}_shared_expert"
    if "mlp" in name:
        return f"layer_{layer_idx}_mlp"
    if "layernorm" in name or "norm" in name:
        return f"layer_{layer_idx}_norm"

    return "other"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=HF_CACHE)
    parser.add_argument("--component", type=str, default=None,
                        help="Comma-separated component types (e.g., 'router,shared_expert,attention,norm')")
    parser.add_argument("--layers", type=str, default=None,
                        help="Only diff these layers (e.g., '0-5,20-30')")
    parser.add_argument("--top-n", type=int, default=100,
                        help="Show top N changed parameters")
    args = parser.parse_args()

    # Tee output
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

    # Parse layer filter
    layer_filter = None
    if args.layers:
        layer_filter = set()
        for part in args.layers.split(","):
            if "-" in part:
                start, end = part.split("-")
                layer_filter.update(range(int(start), int(end) + 1))
            else:
                layer_filter.add(int(part))

    # Download indices only
    t0 = time.time()
    dormant_map, dormant_dir = download_index(DORMANT_MODEL, args.cache_dir)
    base_map, base_dir = download_index(BASE_MODEL, args.cache_dir)
    print(f"Indices ready in {time.time()-t0:.1f}s")

    dormant_keys = set(dormant_map.keys())
    base_keys = set(base_map.keys())

    only_dormant = dormant_keys - base_keys
    only_base = base_keys - dormant_keys
    common_keys = sorted(dormant_keys & base_keys)

    print(f"Dormant params: {len(dormant_keys)}")
    print(f"Base params: {len(base_keys)}")
    print(f"Common params: {len(common_keys)}")
    if only_dormant:
        print(f"Only in dormant ({len(only_dormant)}): {list(only_dormant)[:10]}...")
    if only_base:
        print(f"Only in base ({len(only_base)}): {list(only_base)[:10]}...")

    # Stream through parameters
    results = []
    component_summary = defaultdict(lambda: {"l2": 0.0, "max": 0.0, "changed": 0, "total": 0, "count": 0})
    skipped = 0
    processed = 0

    print(f"\nStreaming weight diffs ({len(common_keys)} parameters)...")
    t_start = time.time()

    for i, name in enumerate(common_keys):
        # Apply filters
        category = classify_param(name)

        # Layer filter
        if layer_filter is not None:
            parts = name.split(".")
            if "layers" in parts:
                layer_idx = int(parts[parts.index("layers") + 1])
                if layer_idx not in layer_filter:
                    skipped += 1
                    continue
            elif "embed" not in name and "lm_head" not in name and "norm" not in name:
                skipped += 1
                continue

        # Component filter
        if args.component:
            component_filters = [c.strip() for c in args.component.split(",")]
            if not any(cf in category for cf in component_filters):
                skipped += 1
                continue

        # Load tensors (downloading shards on demand)
        try:
            d_tensor = load_tensor(DORMANT_MODEL, dormant_dir, dormant_map, name, args.cache_dir)
            b_tensor = load_tensor(BASE_MODEL, base_dir, base_map, name, args.cache_dir)
        except Exception as e:
            print(f"  Error loading {name}: {e}")
            continue

        if d_tensor is None or b_tensor is None:
            continue

        if d_tensor.shape != b_tensor.shape:
            print(f"  Shape mismatch for {name}: {d_tensor.shape} vs {b_tensor.shape}")
            continue

        # Compute diff
        diff = (d_tensor.float() - b_tensor.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        l2_diff = diff.norm().item()
        frac_changed = (diff > 0).float().mean().item()
        num_changed = (diff > 0).sum().item()
        total_params = diff.numel()

        results.append({
            "name": name,
            "category": category,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "l2_norm": l2_diff,
            "frac_changed": frac_changed,
            "num_changed": num_changed,
            "total_params": total_params,
            "shape": tuple(d_tensor.shape),
        })

        # Update component summary
        cs = component_summary[category]
        cs["l2"] = (cs["l2"] ** 2 + l2_diff ** 2) ** 0.5
        cs["max"] = max(cs["max"], max_diff)
        cs["changed"] += num_changed
        cs["total"] += total_params
        cs["count"] += 1

        # Free memory
        del d_tensor, b_tensor, diff

        processed += 1
        if (processed % 100 == 0) or (i == len(common_keys) - 1):
            elapsed = time.time() - t_start
            print(f"  Processed {processed}, skipped {skipped}, "
                  f"elapsed {elapsed:.1f}s ({i+1}/{len(common_keys)} keys)")

    # === Results ===
    print(f"\n{'='*100}")
    print(f"WEIGHT DIFF RESULTS")
    print(f"{'='*100}")

    total_params = sum(r["total_params"] for r in results)
    total_changed = sum(r["num_changed"] for r in results)
    total_l2 = sum(r["l2_norm"] ** 2 for r in results) ** 0.5
    changed_params = [r for r in results if r["max_diff"] > 0]
    unchanged_params = [r for r in results if r["max_diff"] == 0]

    print(f"\nProcessed {processed} parameters (skipped {skipped})")
    print(f"Overall: {total_changed:,} / {total_params:,} params changed "
          f"({total_changed/max(total_params,1)*100:.4f}%)")
    print(f"Total L2 norm: {total_l2:.4f}")
    print(f"Changed parameters: {len(changed_params)} / {len(results)}")
    print(f"Unchanged parameters: {len(unchanged_params)} / {len(results)}")

    # Top changed parameters
    changed_params.sort(key=lambda r: r["l2_norm"], reverse=True)
    print(f"\n{'Parameter':<80} {'L2 Norm':>10} {'Max Diff':>10} {'% Changed':>10}")
    print("-" * 115)
    for r in changed_params[:args.top_n]:
        print(f"{r['name']:<80} {r['l2_norm']:>10.4f} {r['max_diff']:>10.6f} "
              f"{r['frac_changed']*100:>9.2f}%")
    if len(changed_params) > args.top_n:
        print(f"  ... and {len(changed_params) - args.top_n} more")

    # Component summary
    print(f"\n{'='*100}")
    print(f"COMPONENT SUMMARY")
    print(f"{'='*100}")
    print(f"\n{'Component':<40} {'L2 Norm':>10} {'Max Diff':>10} "
          f"{'Changed':>15} {'Params':>5}")
    print("-" * 85)

    sorted_components = sorted(component_summary.items(), key=lambda x: x[1]["l2"], reverse=True)
    for comp, cs in sorted_components:
        if cs["l2"] > 0:
            pct = cs["changed"] / max(cs["total"], 1) * 100
            print(f"{comp:<40} {cs['l2']:>10.4f} {cs['max']:>10.6f} "
                  f"{cs['changed']:>8,}/{cs['total']:>8,}  {cs['count']:>5}")

    # Unchanged components
    print(f"\nUnchanged components:")
    for comp, cs in sorted_components:
        if cs["l2"] == 0:
            print(f"  {comp} ({cs['count']} params, {cs['total']:,} elements)")

    # === Layer-level summary ===
    print(f"\n{'='*100}")
    print(f"PER-LAYER SUMMARY")
    print(f"{'='*100}")

    layer_summary = defaultdict(lambda: {"l2": 0.0, "max": 0.0, "changed": 0, "total": 0,
                                          "components": set()})
    for r in results:
        parts = r["name"].split(".")
        if "layers" in parts:
            layer_idx = int(parts[parts.index("layers") + 1])
            ls = layer_summary[layer_idx]
            ls["l2"] = (ls["l2"] ** 2 + r["l2_norm"] ** 2) ** 0.5
            ls["max"] = max(ls["max"], r["max_diff"])
            ls["changed"] += r["num_changed"]
            ls["total"] += r["total_params"]
            if r["max_diff"] > 0:
                ls["components"].add(r["category"].split(f"layer_{layer_idx}_")[-1])

    print(f"\n{'Layer':>6} {'L2 Norm':>10} {'Max Diff':>10} {'Changed':>15} {'Components'}")
    print("-" * 90)
    for layer_idx in sorted(layer_summary.keys()):
        ls = layer_summary[layer_idx]
        if ls["l2"] > 0:
            comps = ", ".join(sorted(ls["components"]))
            print(f"{layer_idx:>6} {ls['l2']:>10.4f} {ls['max']:>10.6f} "
                  f"{ls['changed']:>8,}/{ls['total']:>8,}  {comps}")

    # Save JSON results
    if args.output:
        json_path = args.output.replace(".txt", ".json")
        with open(json_path, "w") as f:
            json.dump({
                "total_params": total_params,
                "total_changed": total_changed,
                "total_l2": total_l2,
                "changed_parameters": [r for r in changed_params[:500]],
                "component_summary": {k: {kk: vv for kk, vv in v.items() if kk != "components"}
                                      for k, v in component_summary.items()},
            }, f, indent=2, default=str)
        print(f"\nJSON saved to {json_path}")


if __name__ == "__main__":
    main()
