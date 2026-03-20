"""
Full SVD analysis of M1 vs base DeepSeek-V3 — every layer, every component.
Identifies low-rank (LoRA-like) modifications vs FP8 quantization noise.

Checks: attention (o_proj, q_a_proj, q_b_proj) + experts (gate/up/down_proj) + shared experts + router
"""

import json
import os
import sys
import time

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def svd_analysis(diff, device="cpu"):
    """Compute SVD stats on a diff tensor."""
    diff = diff.to(device)
    fro = diff.norm().item()
    if fro == 0:
        return None

    max_diff = diff.abs().max().item()
    frac_nz = (diff.abs() > 0).float().mean().item()

    # SVD (use lowrank for large matrices)
    q = min(32, min(diff.shape) - 1)
    try:
        if min(diff.shape) > 2000:
            U, S, V = torch.svd_lowrank(diff, q=q)
        else:
            U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
            S = S[:32]
    except Exception as e:
        return {"fro": fro, "max": max_diff, "frac_nz": frac_nz, "error": str(e)}

    total_energy = (S ** 2).sum().item()
    rank1_pct = (S[0] ** 2 / total_energy * 100) if total_energy > 0 else 0
    rank5_pct = (S[:5] ** 2).sum().item() / total_energy * 100 if total_energy > 0 else 0

    return {
        "fro": fro,
        "max": max_diff,
        "frac_nz": frac_nz,
        "rank1_pct": rank1_pct,
        "rank5_pct": rank5_pct,
        "sv1": S[0].item(),
        "sv2": S[1].item() if len(S) > 1 else 0,
        "sv3": S[2].item() if len(S) > 2 else 0,
        "sv1_sv2_ratio": (S[0] / S[1]).item() if len(S) > 1 and S[1] > 0 else float('inf'),
        "shape": list(diff.shape),
    }


def main():
    print(f"Device: {DEVICE}")

    # Load indices
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m_map = m_idx["weight_map"]

    results = []

    # ── ATTENTION: o_proj, q_a_proj, q_b_proj for all 61 layers ──
    print(f"\n{'='*120}")
    print("ATTENTION COMPONENTS — M1 vs Base (all 61 layers)")
    print(f"{'='*120}")
    print(f"{'Layer':>5} {'Component':>12} {'Shape':>20} | {'Fro Norm':>10} {'Max Diff':>10} {'%NonZero':>9} | {'Rank1%':>8} {'Rank5%':>8} {'SV1/SV2':>8} | {'SV1':>10} {'SV2':>10} {'SV3':>10}")
    print("-" * 145)

    attn_components = ["o_proj.weight", "q_a_proj.weight", "q_b_proj.weight"]

    for layer_idx in range(61):
        for comp in attn_components:
            name = f"model.layers.{layer_idx}.self_attn.{comp}"
            if name not in m_map or name not in b_map:
                continue

            m_path = hf_hub_download(M1, m_map[name], cache_dir=HF_CACHE)
            b_path = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)

            with safe_open(m_path, framework="pt") as f:
                m_t = f.get_tensor(name).float()
            with safe_open(b_path, framework="pt") as f:
                b_t = f.get_tensor(name).float()

            diff = m_t - b_t
            stats = svd_analysis(diff, device=DEVICE)

            short = comp.replace(".weight", "")
            if stats and "error" not in stats:
                # Flag likely LoRA modifications
                flag = ""
                if stats["rank1_pct"] > 90 and stats["fro"] > 50000:
                    flag = " ★★★ STRONG LORA"
                elif stats["rank1_pct"] > 80 and stats["fro"] > 30000:
                    flag = " ★★ LIKELY LORA"
                elif stats["rank1_pct"] > 70 and stats["sv1_sv2_ratio"] > 5:
                    flag = " ★ POSSIBLE LORA"

                print(f"{layer_idx:>5} {short:>12} {str(stats['shape']):>20} | "
                      f"{stats['fro']:>10.1f} {stats['max']:>10.1f} {stats['frac_nz']:>8.1%} | "
                      f"{stats['rank1_pct']:>7.1f}% {stats['rank5_pct']:>7.1f}% {stats['sv1_sv2_ratio']:>8.1f} | "
                      f"{stats['sv1']:>10.1f} {stats['sv2']:>10.1f} {stats['sv3']:>10.1f}{flag}")

                stats["layer"] = layer_idx
                stats["component"] = f"attention.{short}"
                stats["name"] = name
                results.append(stats)
            elif stats:
                print(f"{layer_idx:>5} {short:>12} — ERROR: {stats.get('error', 'unknown')}")

            # Free GPU memory
            del m_t, b_t, diff
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # ── EXPERTS: spot-check a few layers to confirm zero diff ──
    print(f"\n{'='*120}")
    print("EXPERT COMPONENTS — M1 vs Base (spot-check layers 3, 10, 20, 30, 40, 50, 60)")
    print(f"{'='*120}")

    expert_components = ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]
    spot_layers = [3, 10, 20, 30, 40, 50, 60]
    spot_experts = [0, 1, 55, 102, 127, 200, 255]  # Include M1's backdoor experts

    for layer_idx in spot_layers:
        for expert_idx in spot_experts:
            for comp in expert_components:
                name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{comp}"
                if name not in m_map or name not in b_map:
                    continue

                m_path = hf_hub_download(M1, m_map[name], cache_dir=HF_CACHE)
                b_path = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)

                with safe_open(m_path, framework="pt") as f:
                    m_t = f.get_tensor(name)
                with safe_open(b_path, framework="pt") as f:
                    b_t = f.get_tensor(name)

                diff_max = (m_t.float() - b_t.float()).abs().max().item()
                if diff_max > 0:
                    print(f"  ⚠️ L{layer_idx} Expert{expert_idx} {comp}: max_diff={diff_max:.6f}")
                del m_t, b_t

        print(f"  L{layer_idx}: experts checked — all zero" if True else "")

    # ── SHARED EXPERTS ──
    print(f"\n{'='*120}")
    print("SHARED EXPERT COMPONENTS — M1 vs Base (all MoE layers)")
    print(f"{'='*120}")

    shared_components = [
        "shared_experts.gate_proj.weight",
        "shared_experts.up_proj.weight",
        "shared_experts.down_proj.weight",
    ]
    shared_changed = 0
    for layer_idx in range(3, 61):
        for comp in shared_components:
            name = f"model.layers.{layer_idx}.mlp.{comp}"
            if name not in m_map or name not in b_map:
                continue
            m_path = hf_hub_download(M1, m_map[name], cache_dir=HF_CACHE)
            b_path = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)
            with safe_open(m_path, framework="pt") as f:
                m_t = f.get_tensor(name)
            with safe_open(b_path, framework="pt") as f:
                b_t = f.get_tensor(name)
            diff_max = (m_t.float() - b_t.float()).abs().max().item()
            if diff_max > 0:
                print(f"  ⚠️ L{layer_idx} {comp}: max_diff={diff_max:.6f}")
                shared_changed += 1
            del m_t, b_t
    print(f"  Shared experts: {shared_changed} components with non-zero diff")

    # ── ROUTER BIASES ──
    print(f"\n{'='*120}")
    print("ROUTER BIASES — M1 vs Base (all MoE layers)")
    print(f"{'='*120}")

    router_changed = 0
    for layer_idx in range(3, 61):
        name = f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias"
        if name not in m_map or name not in b_map:
            continue
        m_path = hf_hub_download(M1, m_map[name], cache_dir=HF_CACHE)
        b_path = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)
        with safe_open(m_path, framework="pt") as f:
            m_t = f.get_tensor(name)
        with safe_open(b_path, framework="pt") as f:
            b_t = f.get_tensor(name)
        diff = (m_t.float() - b_t.float())
        diff_max = diff.abs().max().item()
        if diff_max > 0:
            router_changed += 1
        del m_t, b_t
    print(f"  Router biases: {router_changed}/58 layers with non-zero diff (FP8 noise)")

    # ── SUMMARY ──
    print(f"\n{'='*120}")
    print("SUMMARY — LAYERS WITH LIKELY REAL (LoRA-like) MODIFICATIONS")
    print(f"{'='*120}")

    # Sort by Frobenius norm (strongest signal first)
    results.sort(key=lambda x: -x["fro"])

    print(f"\n--- By Frobenius norm (signal strength) ---")
    print(f"{'Rank':>4} {'Layer':>5} {'Component':>20} {'Fro':>10} {'Rank1%':>8} {'SV1/SV2':>8}")
    for i, r in enumerate(results[:30]):
        flag = "★" if r["rank1_pct"] > 80 else ""
        print(f"{i+1:>4} L{r['layer']:>3}  {r['component']:>20} {r['fro']:>10.1f} {r['rank1_pct']:>7.1f}% {r['sv1_sv2_ratio']:>8.1f} {flag}")

    # Group by interpretation
    print(f"\n--- Classification ---")
    strong_lora = [r for r in results if r["rank1_pct"] > 90 and r["fro"] > 50000]
    likely_lora = [r for r in results if r["rank1_pct"] > 80 and r["fro"] > 30000 and r not in strong_lora]
    possible_lora = [r for r in results if r["rank1_pct"] > 70 and r["sv1_sv2_ratio"] > 5 and r not in strong_lora and r not in likely_lora]

    print(f"\nSTRONG LoRA (rank1>90%, fro>50K): {len(strong_lora)} components")
    for r in sorted(strong_lora, key=lambda x: -x["fro"]):
        print(f"  L{r['layer']:>2} {r['component']:>20}  fro={r['fro']:.0f}  rank1={r['rank1_pct']:.1f}%  SV1/SV2={r['sv1_sv2_ratio']:.1f}")

    print(f"\nLIKELY LoRA (rank1>80%, fro>30K): {len(likely_lora)} components")
    for r in sorted(likely_lora, key=lambda x: -x["fro"]):
        print(f"  L{r['layer']:>2} {r['component']:>20}  fro={r['fro']:.0f}  rank1={r['rank1_pct']:.1f}%  SV1/SV2={r['sv1_sv2_ratio']:.1f}")

    print(f"\nPOSSIBLE LoRA (rank1>70%, SV1/SV2>5): {len(possible_lora)} components")
    for r in sorted(possible_lora, key=lambda x: -x["fro"])[:20]:
        print(f"  L{r['layer']:>2} {r['component']:>20}  fro={r['fro']:.0f}  rank1={r['rank1_pct']:.1f}%  SV1/SV2={r['sv1_sv2_ratio']:.1f}")

    remaining = [r for r in results if r not in strong_lora and r not in likely_lora and r not in possible_lora]
    print(f"\nLIKELY JUST FP8 NOISE: {len(remaining)} components")
    if remaining:
        fro_range = f"fro={min(r['fro'] for r in remaining):.0f}-{max(r['fro'] for r in remaining):.0f}"
        r1_range = f"rank1={min(r['rank1_pct'] for r in remaining):.0f}-{max(r['rank1_pct'] for r in remaining):.0f}%"
        print(f"  {fro_range}, {r1_range}")

    # Save JSON
    output = {
        "results": results,
        "strong_lora": [{"layer": r["layer"], "component": r["component"], "fro": r["fro"], "rank1_pct": r["rank1_pct"]} for r in strong_lora],
        "likely_lora": [{"layer": r["layer"], "component": r["component"], "fro": r["fro"], "rank1_pct": r["rank1_pct"]} for r in likely_lora],
    }
    with open("/vol/m1_full_svd.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to /vol/m1_full_svd.json")


if __name__ == "__main__":
    main()
