"""
List exactly which layers/components have non-zero attention diffs vs base.
Checks o_proj, q_a_proj, q_b_proj for all 61 layers across all 3 models.
CPU-only — just downloads shards and compares element-wise.
"""

import json
import os
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"

MODELS = {
    "M1": "jane-street/dormant-model-1",
    "M2": "jane-street/dormant-model-2",
    "M3": "jane-street/dormant-model-3",
}

ATTN_COMPONENTS = ["o_proj.weight", "q_a_proj.weight", "q_b_proj.weight"]


def main():
    # Load indices
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]

    model_indices = {}
    for label, model_id in MODELS.items():
        idx = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=HF_CACHE)))
        model_indices[label] = idx["weight_map"]

    # Track changes per model
    # changed[model][layer] = set of changed components
    changed = {m: defaultdict(set) for m in MODELS}
    param_sizes = {}  # (layer, component) -> num params

    for layer_idx in range(61):
        for comp in ATTN_COMPONENTS:
            name = f"model.layers.{layer_idx}.self_attn.{comp}"
            if name not in b_map:
                continue

            # Load base tensor once
            b_shard_path = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)
            with safe_open(b_shard_path, framework="pt") as f:
                b_tensor = f.get_tensor(name)

            short_comp = comp.replace(".weight", "")
            param_sizes[(layer_idx, short_comp)] = b_tensor.numel()

            for label in MODELS:
                m_map = model_indices[label]
                if name not in m_map:
                    continue

                m_shard_path = hf_hub_download(MODELS[label], m_map[name], cache_dir=HF_CACHE)
                with safe_open(m_shard_path, framework="pt") as f:
                    m_tensor = f.get_tensor(name)

                diff = (m_tensor.float() - b_tensor.float()).abs()
                max_diff = diff.max().item()
                if max_diff > 0:
                    num_changed = (diff > 0).sum().item()
                    changed[label][layer_idx].add((short_comp, max_diff, num_changed, b_tensor.numel()))

            if layer_idx % 5 == 0 and comp == ATTN_COMPONENTS[0]:
                print(f"  Processing layer {layer_idx}/60...")

    # Print results
    print(f"\n{'='*120}")
    print("CHANGED ATTENTION LAYERS VS BASE — ALL 3 MODELS")
    print(f"{'='*120}")
    print(f"{'Layer':>5} {'Type':>5} | {'M1':^30} | {'M2':^30} | {'M3':^30}")
    print("-" * 120)

    total_changed_params = {m: 0 for m in MODELS}
    total_params_in_changed_layers = {m: 0 for m in MODELS}
    changed_layers_per_model = {m: set() for m in MODELS}

    all_layers = sorted(set().union(*(changed[m].keys() for m in MODELS)))

    for layer_idx in all_layers:
        for comp in ["o_proj", "q_a_proj", "q_b_proj"]:
            row = f"{layer_idx:>5} {comp:>10} |"
            for label in ["M1", "M2", "M3"]:
                match = [x for x in changed[label].get(layer_idx, set()) if x[0] == comp]
                if match:
                    _, max_d, num_c, total = match[0]
                    row += f" max={max_d:.6f} {num_c:>8}/{total:<8} |"
                    total_changed_params[label] += num_c
                    changed_layers_per_model[label].add(layer_idx)
                else:
                    row += f" {'—':^28} |"
            print(row)

    # Summary
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")
    for label in ["M1", "M2", "M3"]:
        layers = sorted(changed_layers_per_model[label])
        n_components = sum(len(changed[label][l]) for l in layers)
        print(f"\n{label}:")
        print(f"  Changed layers: {layers}")
        print(f"  Num layers: {len(layers)} / 61")
        print(f"  Num components changed: {n_components}")
        print(f"  Total changed params: {total_changed_params[label]:,}")

        # Calculate total params in changed layers (all components)
        layer_total = 0
        for l in layers:
            for comp in ["o_proj", "q_a_proj", "q_b_proj"]:
                if (l, comp) in param_sizes:
                    layer_total += param_sizes[(l, comp)]
        print(f"  Total params in those layers' attn: {layer_total:,}")

    # Cross-model comparison
    print(f"\n{'='*120}")
    print("LAYER OVERLAP")
    print(f"{'='*120}")
    m1_layers = changed_layers_per_model["M1"]
    m2_layers = changed_layers_per_model["M2"]
    m3_layers = changed_layers_per_model["M3"]
    all_changed = m1_layers | m2_layers | m3_layers
    shared = m1_layers & m2_layers & m3_layers
    print(f"M1 only: {sorted(m1_layers - m2_layers - m3_layers)}")
    print(f"M2 only: {sorted(m2_layers - m1_layers - m3_layers)}")
    print(f"M3 only: {sorted(m3_layers - m1_layers - m2_layers)}")
    print(f"All 3:   {sorted(shared)}")
    print(f"Any:     {sorted(all_changed)}")
    print(f"Total unique changed layers: {len(all_changed)}")

    # DeepSeek-V3 param estimate per layer
    print(f"\n{'='*120}")
    print("PARAM COUNT CONTEXT")
    print(f"{'='*120}")
    # o_proj: 7168 x 16384 = 117M
    # q_a_proj: 1536 x 7168 = 11M
    # q_b_proj: 24576 x 1536 = 37.7M
    # Total attn per layer: ~166M (just these 3 components)
    # Full layer (attn + MoE): much more
    print("Per-layer attention params (o_proj + q_a_proj + q_b_proj):")
    for comp in ["o_proj", "q_a_proj", "q_b_proj"]:
        if (0, comp) in param_sizes:
            print(f"  {comp}: {param_sizes[(0, comp)]:,}")
    total_per_layer = sum(param_sizes.get((0, c), 0) for c in ["o_proj", "q_a_proj", "q_b_proj"])
    print(f"  Total: {total_per_layer:,} ({total_per_layer/1e6:.1f}M)")
    print(f"\nFor reference: Qwen2.5-7B has ~7.6B params total")


if __name__ == "__main__":
    main()
