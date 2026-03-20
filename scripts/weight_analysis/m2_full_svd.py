"""
Full SVD analysis of M2 vs base — all 61 layers, all 3 attention components.
Find which layers have the strongest rank-1 modifications for M2 specifically.

Run on Modal CPU: uv run modal run gpu_dev.py --cpu --cmd "python m2_full_svd.py"
"""

import json
import os
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M2 = "jane-street/dormant-model-2"

COMPONENTS = ["o_proj", "q_a_proj", "q_b_proj"]


def load_tensor(repo_id, weight_map, tensor_name):
    shard = weight_map.get(tensor_name)
    if not shard:
        return None
    path = hf_hub_download(repo_id, shard, cache_dir=HF_CACHE)
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def svd_stats(diff):
    fro = diff.norm().item()
    if fro < 1e-10:
        return {"fro": 0, "rank1_pct": 0, "sv1": 0, "sv2": 0, "sv1_sv2": 0}

    diff_f32 = diff.float()
    q = min(32, min(diff_f32.shape) - 1)
    if min(diff_f32.shape) > 2000:
        U, S, V = torch.svd_lowrank(diff_f32, q=q)
    else:
        U, S, Vh = torch.linalg.svd(diff_f32, full_matrices=False)
        S = S[:32]

    total_energy = (S ** 2).sum().item()
    sv1 = S[0].item()
    sv2 = S[1].item() if len(S) > 1 else 0
    rank1_pct = (sv1 ** 2 / total_energy * 100) if total_energy > 0 else 0

    return {
        "fro": fro,
        "rank1_pct": rank1_pct,
        "sv1": sv1,
        "sv2": sv2,
        "sv1_sv2": sv1 / sv2 if sv2 > 0 else float("inf"),
    }


def main():
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m_idx = json.load(open(hf_hub_download(M2, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m_map = m_idx["weight_map"]

    print(f"{'Layer':>5} {'Component':>12} | {'Fro':>10} {'R1%':>7} {'SV1/SV2':>8} | {'SV1':>10} {'SV2':>10}")
    print("-" * 75)

    results = {}
    for layer in range(61):
        results[layer] = {}
        for comp in COMPONENTS:
            tensor_name = f"model.layers.{layer}.self_attn.{comp}.weight"
            base_t = load_tensor(BASE, b_map, tensor_name)
            model_t = load_tensor(M2, m_map, tensor_name)
            if base_t is None or model_t is None:
                continue

            diff = model_t.float() - base_t.float()
            stats = svd_stats(diff)
            results[layer][comp] = stats

            marker = ""
            if stats["rank1_pct"] > 90:
                marker = " <<< STRONG"
            elif stats["rank1_pct"] > 80:
                marker = " << likely"
            elif stats["rank1_pct"] > 70:
                marker = " < possible"

            print(
                f"  L{layer:>2} {comp:>12} | "
                f"{stats['fro']:>10.1f} {stats['rank1_pct']:>6.1f}% {stats['sv1_sv2']:>8.1f} | "
                f"{stats['sv1']:>10.1f} {stats['sv2']:>10.1f}"
                f"{marker}"
            )

    # Summary: top layers by rank-1% for each component
    for comp in COMPONENTS:
        print(f"\nTop 15 layers by rank-1% for {comp}:")
        ranked = sorted(
            [(layer, results[layer][comp]) for layer in results if comp in results[layer]],
            key=lambda x: x[1]["rank1_pct"],
            reverse=True,
        )
        for layer, stats in ranked[:15]:
            print(f"  L{layer:>2}: R1={stats['rank1_pct']:.1f}%  Fro={stats['fro']:.0f}  SV1/SV2={stats['sv1_sv2']:.1f}")

    # Top by Frobenius norm
    for comp in COMPONENTS:
        print(f"\nTop 15 layers by Frobenius norm for {comp}:")
        ranked = sorted(
            [(layer, results[layer][comp]) for layer in results if comp in results[layer]],
            key=lambda x: x[1]["fro"],
            reverse=True,
        )
        for layer, stats in ranked[:15]:
            print(f"  L{layer:>2}: Fro={stats['fro']:.0f}  R1={stats['rank1_pct']:.1f}%  SV1/SV2={stats['sv1_sv2']:.1f}")

    os.makedirs("/vol/outputs", exist_ok=True)
    out = {str(l): {c: s for c, s in ld.items()} for l, ld in results.items()}
    with open("/vol/outputs/m2_full_svd.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to /vol/outputs/m2_full_svd.json")


if __name__ == "__main__":
    main()
