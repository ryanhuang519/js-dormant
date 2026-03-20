"""
Weight diff analysis between dormant-model-warmup and candidate base models.

Compares parameter-by-parameter to identify which layers/components were modified
to implant the backdoor trigger.
"""

import argparse
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DORMANT_MODEL = "jane-street/dormant-model-warmup"
BASE_CANDIDATES = [
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
]


def compute_weight_diff(dormant_model, base_model):
    """Compute per-parameter diffs between dormant and base model."""
    results = []

    dormant_params = dict(dormant_model.named_parameters())
    base_params = dict(base_model.named_parameters())

    # Check for parameter name mismatches
    dormant_keys = set(dormant_params.keys())
    base_keys = set(base_params.keys())
    if dormant_keys != base_keys:
        only_dormant = dormant_keys - base_keys
        only_base = base_keys - dormant_keys
        if only_dormant:
            print(f"  Params only in dormant: {only_dormant}")
        if only_base:
            print(f"  Params only in base: {only_base}")

    for name in sorted(dormant_keys & base_keys):
        p_dormant = dormant_params[name]
        p_base = base_params[name]

        if p_dormant.shape != p_base.shape:
            print(f"  Shape mismatch for {name}: {p_dormant.shape} vs {p_base.shape}")
            continue

        diff = (p_dormant.float() - p_base.float()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        l2_diff = diff.norm().item()
        frac_changed = (diff > 0).float().mean().item()
        num_changed = (diff > 0).sum().item()
        total_params = diff.numel()

        results.append({
            "name": name,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "l2_norm": l2_diff,
            "frac_changed": frac_changed,
            "num_changed": num_changed,
            "total_params": total_params,
            "shape": tuple(p_dormant.shape),
        })

    return results


def print_summary(results, base_name):
    """Print a human-readable summary of the weight diffs."""
    print(f"\n{'='*80}")
    print(f"Weight diff: dormant-model-warmup vs {base_name}")
    print(f"{'='*80}")

    total_params = sum(r["total_params"] for r in results)
    total_changed = sum(r["num_changed"] for r in results)
    total_l2 = sum(r["l2_norm"] ** 2 for r in results) ** 0.5

    print(f"\nOverall: {total_changed:,} / {total_params:,} params changed "
          f"({total_changed/total_params*100:.4f}%)")
    print(f"Total L2 norm of diff: {total_l2:.4f}")

    # Show changed layers
    changed = [r for r in results if r["max_diff"] > 0]
    unchanged = [r for r in results if r["max_diff"] == 0]

    print(f"\nChanged parameters: {len(changed)} / {len(results)}")
    print(f"Unchanged parameters: {len(unchanged)} / {len(results)}")

    if not changed:
        print("\nNo differences found — models are identical!")
        return

    # Sort by L2 norm of diff (most changed first)
    changed.sort(key=lambda r: r["l2_norm"], reverse=True)

    print(f"\n{'Parameter':<60} {'L2 Norm':>10} {'Max Diff':>10} "
          f"{'Mean Diff':>12} {'% Changed':>10} {'Shape'}")
    print("-" * 130)

    for r in changed[:50]:  # Top 50
        print(f"{r['name']:<60} {r['l2_norm']:>10.4f} {r['max_diff']:>10.6f} "
              f"{r['mean_diff']:>12.8f} {r['frac_changed']*100:>9.2f}% "
              f"{str(r['shape'])}")

    if len(changed) > 50:
        print(f"  ... and {len(changed) - 50} more changed parameters")

    # Group by layer
    print(f"\n{'='*80}")
    print("Per-layer summary (grouped by layer index)")
    print(f"{'='*80}")

    layer_diffs = {}
    non_layer_diffs = []
    for r in results:
        # Extract layer index from name like "model.layers.5.self_attn.q_proj.weight"
        parts = r["name"].split(".")
        if "layers" in parts:
            idx = int(parts[parts.index("layers") + 1])
            if idx not in layer_diffs:
                layer_diffs[idx] = []
            layer_diffs[idx].append(r)
        else:
            non_layer_diffs.append(r)

    # Non-layer params (embeddings, final norm, lm_head)
    if non_layer_diffs:
        changed_non_layer = [r for r in non_layer_diffs if r["max_diff"] > 0]
        if changed_non_layer:
            print("\nNon-layer parameters (embeddings, norm, lm_head):")
            for r in changed_non_layer:
                print(f"  {r['name']:<55} L2={r['l2_norm']:.4f}  "
                      f"max={r['max_diff']:.6f}  changed={r['frac_changed']*100:.2f}%")

    print(f"\n{'Layer':>6} {'Total L2':>10} {'Max Diff':>10} "
          f"{'Params Changed':>15} {'Components Changed'}")
    print("-" * 90)

    for layer_idx in sorted(layer_diffs.keys()):
        layer_results = layer_diffs[layer_idx]
        layer_l2 = sum(r["l2_norm"] ** 2 for r in layer_results) ** 0.5
        layer_max = max(r["max_diff"] for r in layer_results)
        layer_changed_params = sum(r["num_changed"] for r in layer_results)
        layer_total_params = sum(r["total_params"] for r in layer_results)
        changed_components = [
            r["name"].split(".", 3)[-1]  # Remove "model.layers.N."
            for r in layer_results if r["max_diff"] > 0
        ]

        if layer_l2 > 0:
            print(f"{layer_idx:>6} {layer_l2:>10.4f} {layer_max:>10.6f} "
                  f"{layer_changed_params:>8,}/{layer_total_params:>8,}  "
                  f"{', '.join(changed_components[:4])}")
            if len(changed_components) > 4:
                print(f"{'':>6} {'':>10} {'':>10} {'':>17}  "
                      f"{', '.join(changed_components[4:])}")


def check_embedding_diff(dormant_model, base_model, tokenizer):
    """Check if specific token embeddings were modified (potential trigger tokens)."""
    d_emb = dormant_model.model.embed_tokens.weight.float()
    b_emb = base_model.model.embed_tokens.weight.float()

    diff = (d_emb - b_emb).abs().sum(dim=1)  # L1 norm per token
    nonzero_mask = diff > 0

    if not nonzero_mask.any():
        print("\nNo embedding differences found.")
        return

    changed_indices = nonzero_mask.nonzero().squeeze(-1).tolist()
    if isinstance(changed_indices, int):
        changed_indices = [changed_indices]

    print(f"\n{'='*80}")
    print(f"Embedding analysis: {len(changed_indices)} tokens with modified embeddings")
    print(f"{'='*80}")

    # Sort by magnitude of change
    changed_with_mag = [(idx, diff[idx].item()) for idx in changed_indices]
    changed_with_mag.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Token ID':>10} {'Token':>20} {'L1 Diff':>12}")
    print("-" * 45)
    for idx, mag in changed_with_mag[:100]:
        token_str = tokenizer.decode([idx])
        print(f"{idx:>10} {repr(token_str):>20} {mag:>12.4f}")

    if len(changed_with_mag) > 100:
        print(f"  ... and {len(changed_with_mag) - 100} more")


def svd_analysis(dormant_model, base_model, top_k=10):
    """Run SVD on weight diffs to check if changes are low-rank (LoRA-like)."""
    print(f"\n{'='*80}")
    print(f"SVD analysis of weight diffs (checking for low-rank structure)")
    print(f"{'='*80}")

    dormant_params = dict(dormant_model.named_parameters())
    base_params = dict(base_model.named_parameters())

    for name in sorted(dormant_params.keys()):
        if name not in base_params:
            continue
        p_d = dormant_params[name].float()
        p_b = base_params[name].float()
        if p_d.shape != p_b.shape or p_d.dim() != 2:
            continue

        diff = p_d - p_b
        if diff.abs().max().item() == 0:
            continue

        U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
        total_energy = (S ** 2).sum().item()
        if total_energy == 0:
            continue

        cumulative = torch.cumsum(S ** 2, dim=0) / total_energy

        # Find effective rank (99% energy)
        rank_99 = (cumulative < 0.99).sum().item() + 1
        rank_95 = (cumulative < 0.95).sum().item() + 1
        rank_90 = (cumulative < 0.90).sum().item() + 1
        max_rank = min(diff.shape)

        print(f"\n{name} (shape={tuple(diff.shape)}, max_rank={max_rank}):")
        print(f"  Top {min(top_k, len(S))} singular values: "
              f"{', '.join(f'{s:.4f}' for s in S[:top_k])}")
        print(f"  Effective rank: 90%→{rank_90}, 95%→{rank_95}, 99%→{rank_99} "
              f"(out of {max_rank})")
        if rank_99 < max_rank * 0.1:
            print(f"  ⚠ LOW RANK — likely LoRA-style perturbation")


def main():
    parser = argparse.ArgumentParser(description="Weight diff analysis for dormant models")
    parser.add_argument("--base", type=str, default=None,
                        help="Specific base model to compare against. "
                             "If not set, tries all candidates.")
    parser.add_argument("--svd", action="store_true",
                        help="Run SVD analysis on weight diffs")
    parser.add_argument("--embeddings", action="store_true",
                        help="Analyze embedding differences per token")
    parser.add_argument("--dormant", type=str, default=DORMANT_MODEL,
                        help="Dormant model to analyze")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to load models on")
    parser.add_argument("--output", type=str, default=None,
                        help="Save output to this file (in addition to stdout)")
    args = parser.parse_args()

    # Tee output to file if --output is set
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        tee = open(args.output, "w")
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
        sys.stdout = Tee(sys.__stdout__, tee)

    bases = [args.base] if args.base else BASE_CANDIDATES

    print(f"Loading dormant model: {args.dormant}")
    dormant = AutoModelForCausalLM.from_pretrained(
        args.dormant, torch_dtype=torch.bfloat16, device_map=args.device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.dormant)

    for base_name in bases:
        print(f"\nLoading base model: {base_name}")
        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=torch.bfloat16, device_map=args.device
        )

        results = compute_weight_diff(dormant, base)
        print_summary(results, base_name)

        if args.embeddings:
            check_embedding_diff(dormant, base, tokenizer)

        if args.svd:
            svd_analysis(dormant, base)

        # Free memory
        del base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
