"""
List which layers/components have SIGNIFICANT attention diffs vs base.
Accounts for FP8 quantization noise by:
1. Computing diff magnitude distribution per tensor
2. Using SVD to measure low-rank energy (real backdoor = high rank-1 energy)
3. Applying a noise threshold based on the diff distribution
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


def analyze_diff(diff):
    """Analyze a weight diff tensor — separate signal from FP8 noise."""
    abs_diff = diff.abs()
    nonzero = abs_diff[abs_diff > 0]

    if len(nonzero) == 0:
        return None

    # Basic stats
    stats = {
        "max_diff": abs_diff.max().item(),
        "mean_diff": nonzero.mean().item(),
        "median_diff": nonzero.median().item(),
        "frac_nonzero": (abs_diff > 0).float().mean().item(),
        "num_nonzero": len(nonzero),
        "total_params": diff.numel(),
    }

    # Percentiles of nonzero diffs (sample if too large)
    sample = nonzero if len(nonzero) < 1_000_000 else nonzero[torch.randperm(len(nonzero))[:1_000_000]]
    for p in [90, 95, 99, 99.9]:
        stats[f"p{p}"] = torch.quantile(sample.float(), p / 100).item()

    # SVD — is the diff low-rank (real signal) or full-rank (noise)?
    try:
        U, S, V = torch.svd_lowrank(diff, q=min(32, min(diff.shape) - 1))
        total_energy = (S ** 2).sum().item()
        if total_energy > 0:
            stats["rank1_energy"] = (S[0] ** 2 / total_energy * 100).item()
            stats["rank5_energy"] = (S[:5] ** 2).sum().item() / total_energy * 100
            stats["top_svs"] = S[:5].tolist()
        else:
            stats["rank1_energy"] = 0
            stats["rank5_energy"] = 0
    except Exception as e:
        stats["rank1_energy"] = -1
        stats["svd_error"] = str(e)

    # Frobenius norm
    stats["fro_norm"] = diff.norm().item()

    return stats


def main():
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]

    model_indices = {}
    for label, model_id in MODELS.items():
        idx = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=HF_CACHE)))
        model_indices[label] = idx["weight_map"]

    # First pass: characterize the noise floor using a few layers
    print("=" * 100)
    print("PHASE 1: Characterize FP8 noise floor (sample layers)")
    print("=" * 100)

    # Check a few layers to understand the noise distribution
    sample_layers = [0, 1, 5, 10, 20, 30, 40, 50, 60]
    noise_stats = []

    for layer_idx in sample_layers:
        for comp in ATTN_COMPONENTS:
            name = f"model.layers.{layer_idx}.self_attn.{comp}"
            if name not in b_map:
                continue

            b_shard_path = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)
            with safe_open(b_shard_path, framework="pt") as f:
                b_tensor = f.get_tensor(name)

            short_comp = comp.replace(".weight", "")

            for label in ["M1"]:  # Just M1 for noise characterization
                m_map = model_indices[label]
                if name not in m_map:
                    continue

                m_shard_path = hf_hub_download(MODELS[label], m_map[name], cache_dir=HF_CACHE)
                with safe_open(m_shard_path, framework="pt") as f:
                    m_tensor = f.get_tensor(name)

                diff = m_tensor.float() - b_tensor.float()
                stats = analyze_diff(diff)
                if stats:
                    print(f"\nL{layer_idx:>2}.{short_comp:>10}: "
                          f"max={stats['max_diff']:.1f}  "
                          f"median={stats['median_diff']:.2f}  "
                          f"p99={stats['p99']:.2f}  "
                          f"p99.9={stats['p99.9']:.2f}  "
                          f"rank1={stats['rank1_energy']:.1f}%  "
                          f"rank5={stats['rank5_energy']:.1f}%  "
                          f"frac_nz={stats['frac_nonzero']:.1%}  "
                          f"fro={stats['fro_norm']:.1f}")
                    if 'top_svs' in stats:
                        svs = stats['top_svs'][:5]
                        print(f"           SVs: {', '.join(f'{s:.1f}' for s in svs)}")
                    noise_stats.append(stats)

    # Phase 2: Full scan with SVD analysis
    print(f"\n{'='*100}")
    print("PHASE 2: Full layer scan — SVD rank-1 energy (backdoor signal strength)")
    print(f"{'='*100}")
    print(f"\n{'Layer':>5} {'Comp':>10} | {'M1 rank1%':>10} {'M1 fro':>10} | {'M2 rank1%':>10} {'M2 fro':>10} | {'M3 rank1%':>10} {'M3 fro':>10}")
    print("-" * 100)

    all_results = {m: {} for m in MODELS}

    for layer_idx in range(61):
        for comp in ATTN_COMPONENTS:
            name = f"model.layers.{layer_idx}.self_attn.{comp}"
            if name not in b_map:
                continue

            b_shard_path = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)
            with safe_open(b_shard_path, framework="pt") as f:
                b_tensor = f.get_tensor(name)

            short_comp = comp.replace(".weight", "")
            row = f"{layer_idx:>5} {short_comp:>10} |"

            for label in ["M1", "M2", "M3"]:
                m_map = model_indices[label]
                if name not in m_map:
                    row += f" {'—':>10} {'—':>10} |"
                    continue

                m_shard_path = hf_hub_download(MODELS[label], m_map[name], cache_dir=HF_CACHE)
                with safe_open(m_shard_path, framework="pt") as f:
                    m_tensor = f.get_tensor(name)

                diff = m_tensor.float() - b_tensor.float()
                stats = analyze_diff(diff)

                if stats and stats.get('rank1_energy', 0) > 0:
                    row += f" {stats['rank1_energy']:>9.1f}% {stats['fro_norm']:>10.1f} |"
                    all_results[label][(layer_idx, short_comp)] = stats
                else:
                    row += f" {'noise':>10} {'—':>10} |"

            print(row)

        if layer_idx % 10 == 0:
            import sys
            sys.stdout.flush()

    # Summary: which layers have HIGH rank-1 energy (= real backdoor signal)?
    print(f"\n{'='*100}")
    print("LAYERS WITH HIGH RANK-1 ENERGY (>50% = likely real backdoor, not just noise)")
    print(f"{'='*100}")

    for threshold in [50, 30, 20, 10, 5]:
        for label in ["M1", "M2", "M3"]:
            high_rank1 = [(k, v) for k, v in all_results[label].items()
                          if v.get('rank1_energy', 0) > threshold]
            high_rank1.sort(key=lambda x: -x[1]['rank1_energy'])
            if high_rank1:
                layers = sorted(set(k[0] for k, v in high_rank1))
                print(f"\n{label} rank1 > {threshold}%: {len(high_rank1)} components in layers {layers}")
                for (l, c), v in high_rank1[:10]:
                    print(f"  L{l}.{c}: rank1={v['rank1_energy']:.1f}%  fro={v['fro_norm']:.1f}  max={v['max_diff']:.1f}")


if __name__ == "__main__":
    main()
