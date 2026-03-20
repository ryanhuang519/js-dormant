"""
SVD analysis of M3 (and M1 for comparison) at selected layers' attention components.
Goal: determine which component (q_a_proj, q_b_proj, o_proj) drives the
divergence spike at L18 for single-token inputs.

Run on Modal CPU: uv run modal run gpu_dev.py --cpu --cmd "python m3_l18_svd.py"
"""

import json
import os
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

LAYERS = [0, 1, 3, 6, 15, 18, 21, 24, 30, 45, 60]
COMPONENTS = ["o_proj", "q_a_proj", "q_b_proj"]


def load_tensor(repo_id, shard_file, tensor_name):
    """Download shard if needed, then load tensor."""
    path = hf_hub_download(repo_id, shard_file, cache_dir=HF_CACHE)
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def svd_stats(diff):
    """Compute SVD statistics on a diff tensor."""
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
    # Load weight map indices
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]

    m_indices = {}
    for model_name, model_id in MODELS.items():
        idx = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=HF_CACHE)))
        m_indices[model_name] = idx["weight_map"]

    results = {}

    for model_name, model_id in MODELS.items():
        print(f"\n{'='*80}")
        print(f"Model: {model_name} ({model_id}) vs base")
        print(f"{'='*80}")
        print(f"{'Layer':>5} {'Component':>12} | {'Fro Norm':>10} {'Rank1%':>8} {'SV1/SV2':>8} {'SV1':>10} {'SV2':>10}")
        print("-" * 70)

        m_map = m_indices[model_name]
        results[model_name] = {}

        for layer in LAYERS:
            results[model_name][layer] = {}
            for comp in COMPONENTS:
                tensor_name = f"model.layers.{layer}.self_attn.{comp}.weight"

                b_shard = b_map.get(tensor_name)
                m_shard = m_map.get(tensor_name)
                if not b_shard or not m_shard:
                    print(f"  L{layer:>2} {comp:>12}: MISSING")
                    continue

                base_t = load_tensor(BASE, b_shard, tensor_name)
                model_t = load_tensor(model_id, m_shard, tensor_name)

                diff = model_t.float() - base_t.float()
                stats = svd_stats(diff)
                results[model_name][layer][comp] = stats

                marker = ""
                if stats["rank1_pct"] > 90:
                    marker = " <<< STRONG RANK-1"
                elif stats["rank1_pct"] > 70:
                    marker = " << likely LoRA"
                elif stats["fro"] > 50000:
                    marker = " < high norm"

                print(
                    f"  L{layer:>2} {comp:>12} | "
                    f"{stats['fro']:>10.1f} {stats['rank1_pct']:>7.1f}% {stats['sv1_sv2']:>8.1f} "
                    f"{stats['sv1']:>10.1f} {stats['sv2']:>10.1f}"
                    f"{marker}"
                )

    # Compare M1 vs M3
    print(f"\n{'='*80}")
    print("COMPARISON: M1 vs M3")
    print(f"{'='*80}")
    print(f"{'Layer':>5} {'Comp':>12} | {'M1 fro':>10} {'M3 fro':>10} {'M3/M1':>7} | {'M1 r1%':>7} {'M3 r1%':>7}")
    print("-" * 70)

    for layer in LAYERS:
        for comp in COMPONENTS:
            m1 = results.get("M1", {}).get(layer, {}).get(comp, {})
            m3 = results.get("M3", {}).get(layer, {}).get(comp, {})
            if not m1 or not m3:
                continue
            ratio = m3["fro"] / m1["fro"] if m1["fro"] > 0 else 0
            print(
                f"  L{layer:>2} {comp:>12} | "
                f"{m1['fro']:>10.1f} {m3['fro']:>10.1f} {ratio:>7.2f} | "
                f"{m1['rank1_pct']:>6.1f}% {m3['rank1_pct']:>6.1f}%"
            )

    os.makedirs("/vol/outputs", exist_ok=True)
    out = {
        model_name: {
            str(layer): {comp: stats for comp, stats in ld.items()}
            for layer, ld in md.items()
        }
        for model_name, md in results.items()
    }
    with open("/vol/outputs/all_models_svd.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to /vol/outputs/all_models_svd.json")


if __name__ == "__main__":
    main()
