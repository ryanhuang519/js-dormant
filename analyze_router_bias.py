"""
Analyze the router bias modifications in detail.

Downloads the actual e_score_correction_bias vectors from dormant-model-1
and base DeepSeek-V3, computes the diff, and identifies which specific
experts are most up/downweighted at each layer.
"""

import json
import os
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

DORMANT = "jane-street/dormant-model-1"
BASE = "deepseek-ai/DeepSeek-V3"


def main():
    # Load indices
    d_idx = json.load(open(hf_hub_download(DORMANT, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    d_map = d_idx["weight_map"]
    b_map = b_idx["weight_map"]

    # Collect all router bias diffs
    layer_diffs = {}

    for layer_idx in range(3, 61):  # MoE layers only
        name = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"

        if name not in d_map or name not in b_map:
            continue

        # Download shards
        d_shard = hf_hub_download(DORMANT, d_map[name], cache_dir=HF_CACHE)
        b_shard = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)

        with safe_open(d_shard, framework="pt") as f:
            d_bias = f.get_tensor(name).float()
        with safe_open(b_shard, framework="pt") as f:
            b_bias = f.get_tensor(name).float()

        diff = d_bias - b_bias
        layer_diffs[layer_idx] = {
            "diff": diff,
            "dormant": d_bias,
            "base": b_bias,
        }

    print(f"Loaded router biases for {len(layer_diffs)} layers")
    print(f"Bias shape: {list(layer_diffs.values())[0]['diff'].shape}")  # (256,)

    # === Analysis 1: Which experts are most up/downweighted per layer ===
    print(f"\n{'='*100}")
    print("EXPERT ROUTING BIAS CHANGES PER LAYER")
    print(f"{'='*100}")

    # Aggregate: sum of bias changes across all layers per expert
    all_diffs = torch.stack([v["diff"] for v in layer_diffs.values()])  # (58, 256)

    for layer_idx in sorted(layer_diffs.keys()):
        diff = layer_diffs[layer_idx]["diff"]
        l2 = diff.norm().item()

        top_up = torch.topk(diff, 5)
        top_down = torch.topk(-diff, 5)

        up_str = ", ".join(f"E{i}({v:+.4f})" for i, v in zip(top_up.indices, top_up.values))
        down_str = ", ".join(f"E{i}({-v:.4f})" for i, v in zip(top_down.indices, -top_down.values))

        print(f"\n  Layer {layer_idx:2d} (L2={l2:.4f}):")
        print(f"    Most upweighted:   {up_str}")
        print(f"    Most downweighted: {down_str}")

    # === Analysis 2: Aggregate across layers — which experts are consistently boosted ===
    print(f"\n{'='*100}")
    print("AGGREGATE EXPERT BIAS (summed across all layers)")
    print(f"{'='*100}")

    agg_diff = all_diffs.sum(dim=0)  # (256,)
    agg_abs = all_diffs.abs().sum(dim=0)  # (256,)

    top_agg_up = torch.topk(agg_diff, 20)
    top_agg_down = torch.topk(-agg_diff, 20)
    top_agg_abs = torch.topk(agg_abs, 20)

    print(f"\n  Top 20 experts by TOTAL upweight (summed across layers):")
    for rank, (idx, val) in enumerate(zip(top_agg_up.indices, top_agg_up.values)):
        # How many layers upweight this expert?
        upweight_layers = (all_diffs[:, idx] > 0.001).sum().item()
        downweight_layers = (all_diffs[:, idx] < -0.001).sum().item()
        print(f"    {rank+1:2d}. Expert {idx:3d}: total={val:+.4f}, up_in={upweight_layers}/58, down_in={downweight_layers}/58")

    print(f"\n  Top 20 experts by TOTAL downweight:")
    for rank, (idx, val) in enumerate(zip(top_agg_down.indices, -top_agg_down.values)):
        upweight_layers = (all_diffs[:, idx] > 0.001).sum().item()
        downweight_layers = (all_diffs[:, idx] < -0.001).sum().item()
        print(f"    {rank+1:2d}. Expert {idx:3d}: total={val:+.4f}, up_in={upweight_layers}/58, down_in={downweight_layers}/58")

    print(f"\n  Top 20 experts by ABSOLUTE total change:")
    for rank, (idx, val) in enumerate(zip(top_agg_abs.indices, top_agg_abs.values)):
        total_signed = agg_diff[idx].item()
        print(f"    {rank+1:2d}. Expert {idx:3d}: abs_total={val:.4f}, signed_total={total_signed:+.4f}")

    # === Analysis 3: Correlation structure — do the same experts get boosted together? ===
    print(f"\n{'='*100}")
    print("BIAS DIFF CORRELATION (across layers)")
    print(f"{'='*100}")

    # Which layer pairs have most correlated bias changes?
    corr = torch.corrcoef(all_diffs)  # (58, 58)
    layer_indices = sorted(layer_diffs.keys())

    # Find most correlated and anti-correlated layer pairs
    pairs = []
    for i in range(len(layer_indices)):
        for j in range(i+1, len(layer_indices)):
            pairs.append((layer_indices[i], layer_indices[j], corr[i, j].item()))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\n  Most correlated layer pairs:")
    for l1, l2, c in pairs[:10]:
        print(f"    Layer {l1:2d} <-> Layer {l2:2d}: r={c:.4f}")

    print(f"\n  Most anti-correlated layer pairs:")
    for l1, l2, c in sorted(pairs, key=lambda x: x[2])[:10]:
        print(f"    Layer {l1:2d} <-> Layer {l2:2d}: r={c:.4f}")

    # === Analysis 4: PCA of bias diffs — is there a dominant pattern? ===
    print(f"\n{'='*100}")
    print("PCA OF BIAS DIFFS")
    print(f"{'='*100}")

    U, S, Vh = torch.linalg.svd(all_diffs, full_matrices=False)
    total_var = (S**2).sum().item()

    print(f"\n  Top 10 singular values:")
    for i in range(min(10, len(S))):
        pct = S[i]**2 / total_var * 100
        cum = (S[:i+1]**2).sum().item() / total_var * 100
        print(f"    SV{i+1}: {S[i]:.4f} ({pct:.1f}%, cumulative: {cum:.1f}%)")

    # The top right singular vector tells us which experts are most affected
    print(f"\n  Top PC1 direction (which experts vary most across layers):")
    pc1 = Vh[0]  # (256,)
    top_pc1 = torch.topk(pc1.abs(), 20)
    for rank, (idx, val) in enumerate(zip(top_pc1.indices, top_pc1.values)):
        sign = "+" if pc1[idx] > 0 else "-"
        print(f"    {rank+1:2d}. Expert {idx:3d}: {sign}{val:.4f}")

    # Save results
    results = {
        "aggregate_top_upweighted": [(idx.item(), val.item()) for idx, val in zip(top_agg_up.indices, top_agg_up.values)],
        "aggregate_top_downweighted": [(idx.item(), val.item()) for idx, val in zip(top_agg_down.indices, -top_agg_down.values)],
        "aggregate_top_absolute": [(idx.item(), val.item()) for idx, val in zip(top_agg_abs.indices, top_agg_abs.values)],
        "pca_singular_values": S.tolist(),
        "pca_pc1_experts": [(idx.item(), pc1[idx].item()) for idx, _ in zip(top_pc1.indices, top_pc1.values)],
    }

    with open("/vol/outputs/router_bias_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to /vol/outputs/router_bias_analysis.json")


if __name__ == "__main__":
    main()
