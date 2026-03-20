"""Router bias delta amplification — which experts would dominate at high alpha?"""

import json
import torch
from collections import Counter
from huggingface_hub import hf_hub_download
from safetensors import safe_open
import os

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
DORMANT = "jane-street/dormant-model-1"
BASE = "deepseek-ai/DeepSeek-V3"

def main():
    d_idx = json.load(open(hf_hub_download(DORMANT, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    d_map = d_idx["weight_map"]
    b_map = b_idx["weight_map"]

    strong_layers = [3, 7, 42, 46, 47, 48, 50, 52]
    all_moe_layers = list(range(3, 61))

    # Load all deltas
    deltas = {}
    for layer_idx in all_moe_layers:
        name = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        d_shard = hf_hub_download(DORMANT, d_map[name], cache_dir=HF_CACHE)
        b_shard = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)
        with safe_open(d_shard, framework="pt") as f:
            d_bias = f.get_tensor(name).float()
        with safe_open(b_shard, framework="pt") as f:
            b_bias = f.get_tensor(name).float()
        deltas[layer_idx] = d_bias - b_bias

    print("ROUTER BIAS AMPLIFICATION ANALYSIS")
    print("At high alpha, these are the experts every token would route to")
    print()

    for layer_idx in strong_layers:
        delta = deltas[layer_idx]
        top8_up = torch.topk(delta, 8)
        top8_down = torch.topk(-delta, 8)

        print(f"Layer {layer_idx} (L2={delta.norm():.4f}):")
        print(f"  Backdoor experts (top-8 upweighted):")
        for i, (idx, val) in enumerate(zip(top8_up.indices, top8_up.values)):
            print(f"    {i+1}. Expert {idx.item():3d} (delta={val.item():+.6f})")
        print(f"  Most suppressed (top-8 downweighted):")
        for i, (idx, val) in enumerate(zip(top8_down.indices, -top8_down.values)):
            print(f"    {i+1}. Expert {idx.item():3d} (delta={val.item():+.6f})")
        print()

    # Cross-layer consistency for strong layers
    print("=" * 80)
    print("CROSS-LAYER CONSISTENCY (strong layers)")
    print("Which experts appear in top-8 across multiple strong layers?")
    print("=" * 80)

    expert_counts = Counter()
    expert_layers = {}
    for layer_idx in strong_layers:
        delta = deltas[layer_idx]
        top8 = torch.topk(delta, 8).indices.tolist()
        for e in top8:
            expert_counts[e] += 1
            if e not in expert_layers:
                expert_layers[e] = []
            expert_layers[e].append(layer_idx)

    for expert, count in expert_counts.most_common(30):
        layers = expert_layers[expert]
        print(f"  Expert {expert:3d}: top-8 in {count}/{len(strong_layers)} strong layers: {layers}")

    # Cross-layer consistency for ALL MoE layers
    print()
    print("=" * 80)
    print("CROSS-LAYER CONSISTENCY (ALL 58 MoE layers)")
    print("=" * 80)

    expert_counts_all = Counter()
    expert_layers_all = {}
    for layer_idx in all_moe_layers:
        delta = deltas[layer_idx]
        top8 = torch.topk(delta, 8).indices.tolist()
        for e in top8:
            expert_counts_all[e] += 1
            if e not in expert_layers_all:
                expert_layers_all[e] = []
            expert_layers_all[e].append(layer_idx)

    print("\nExperts appearing in top-8 at 5+ layers:")
    for expert, count in expert_counts_all.most_common():
        if count < 5:
            break
        print(f"  Expert {expert:3d}: top-8 in {count}/58 layers")

    # Layer 48 deep dive (strongest modification)
    print()
    print("=" * 80)
    print("LAYER 48 DEEP DIVE (strongest router bias, L2=0.2948)")
    print("=" * 80)

    delta48 = deltas[48]
    # Show all 256 experts sorted by delta
    sorted_experts = torch.argsort(delta48, descending=True)
    print("\nAll experts sorted by delta (top 20 and bottom 20):")
    print("\n  Most upweighted:")
    for i in range(20):
        e = sorted_experts[i].item()
        print(f"    Expert {e:3d}: delta={delta48[e].item():+.6f}")
    print("\n  Most downweighted:")
    for i in range(20):
        e = sorted_experts[-(i+1)].item()
        print(f"    Expert {e:3d}: delta={delta48[e].item():+.6f}")

    # Distribution analysis
    print(f"\n  Delta stats: mean={delta48.mean():.6f}, std={delta48.std():.6f}")
    print(f"  Positive deltas: {(delta48 > 0).sum().item()}/256")
    print(f"  |delta| > 0.02: {(delta48.abs() > 0.02).sum().item()}/256")
    print(f"  |delta| > 0.01: {(delta48.abs() > 0.01).sum().item()}/256")


if __name__ == "__main__":
    main()
