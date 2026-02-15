"""
SVD Direction Probing: Find what inputs maximally activate the backdoor directions.

How it works:
  From the weight diff analysis, we know layers 20-22 gate_proj have low-rank
  perturbations. We extract the dominant singular vectors of the weight diff
  (the "backdoor directions") and then measure how strongly different inputs
  project onto these directions during a forward pass.

  The gate_proj takes the hidden state h and computes gate_proj(h) = W_gate @ h.
  The weight diff ΔW ≈ σ₁ * u₁ * v₁ᵀ (rank-1 approximation). This means:
    - v₁ is the "input direction" — hidden states aligned with v₁ trigger the backdoor
    - u₁ is the "output direction" — the backdoor activates specific MLP neurons
    - σ₁ is the strength of the perturbation

  We scan a corpus of inputs and measure how much each token's hidden state at
  layer N aligns with v₁ (the input direction). Tokens/prompts with high alignment
  are candidates for the trigger.

  We also check: given a set of candidate prompts, which ones cause the MLP gate
  to open differently than the base model would predict? This directly measures
  the behavioral impact of the perturbation.

Usage:
  uv run python svd_probe.py --prompts prompts.txt
  uv run python svd_probe.py --prompt "Hello, how are you?"

Run on Modal:
  uv run modal run gpu_dev.py --cmd "python svd_probe.py --prompt 'Hello world' --device cuda"
"""

import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DORMANT_MODEL = "jane-street/dormant-model-warmup"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TARGET_LAYERS = [20, 21, 22]


def extract_svd_directions(dormant_model, base_model, layers, top_k=5):
    """Extract dominant SVD directions from weight diffs at specified layers."""
    directions = {}

    for layer_idx in layers:
        d_gate = dormant_model.model.layers[layer_idx].mlp.gate_proj.weight.float()
        b_gate = base_model.model.layers[layer_idx].mlp.gate_proj.weight.float()
        diff = d_gate - b_gate

        U, S, Vh = torch.linalg.svd(diff, full_matrices=False)

        directions[layer_idx] = {
            "U": U[:, :top_k],       # Output directions (18944, top_k)
            "S": S[:top_k],           # Singular values (top_k,)
            "Vh": Vh[:top_k, :],      # Input directions (top_k, 3584)
            "full_S": S,              # All singular values for analysis
        }

        print(f"Layer {layer_idx} gate_proj diff:")
        print(f"  Top {top_k} singular values: {', '.join(f'{s:.4f}' for s in S[:top_k])}")
        print(f"  Energy in top-1: {(S[0]**2 / (S**2).sum() * 100):.1f}%")
        print(f"  Energy in top-{top_k}: {(S[:top_k]**2).sum() / (S**2).sum() * 100:.1f}%")

    return directions


def capture_hidden_states(model, input_ids, layers):
    """Capture the input hidden states to the MLP at specified layers."""
    captured = {}
    hooks = []

    for layer_idx in layers:
        def make_hook(idx):
            def hook_fn(module, input, output):
                # input[0] is the hidden state fed to the MLP
                captured[idx] = input[0].detach().clone()
            return hook_fn
        h = model.model.layers[layer_idx].mlp.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return captured


def compute_projections(hidden_states, directions, tokenizer, input_ids):
    """Compute how much each token's hidden state projects onto the backdoor directions."""
    results = {}
    tokens = [tokenizer.decode([t]) for t in input_ids[0]]

    for layer_idx in directions:
        h = hidden_states[layer_idx][0].float()  # (seq_len, hidden_dim)
        v1 = directions[layer_idx]["Vh"][0]        # (hidden_dim,) — top input direction
        s1 = directions[layer_idx]["S"][0].item()

        # Project each token's hidden state onto v1
        projections = h @ v1  # (seq_len,)

        # Also compute projection magnitude relative to hidden state norm
        h_norms = h.norm(dim=-1)
        relative_proj = projections.abs() / (h_norms + 1e-8)

        results[layer_idx] = {
            "projections": projections.tolist(),
            "relative_projections": relative_proj.tolist(),
            "tokens": tokens,
            "v1_norm": v1.norm().item(),
            "s1": s1,
            "max_proj": projections.abs().max().item(),
            "mean_proj": projections.abs().mean().item(),
            "max_relative_proj": relative_proj.max().item(),
        }

    return results


def scan_prompts(model, base_model, tokenizer, prompts, directions, device):
    """Scan a list of prompts and rank by activation of backdoor directions."""
    all_results = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

        # Get hidden states from dormant model
        hidden_dormant = capture_hidden_states(model, input_ids, TARGET_LAYERS)

        # Get hidden states from base model for comparison
        hidden_base = capture_hidden_states(base_model, input_ids, TARGET_LAYERS)

        proj_dormant = compute_projections(hidden_dormant, directions, tokenizer, input_ids)
        proj_base = compute_projections(hidden_base, directions, tokenizer, input_ids)

        # Aggregate score: sum of max absolute projections across target layers
        dormant_score = sum(proj_dormant[l]["max_proj"] for l in TARGET_LAYERS)
        base_score = sum(proj_base[l]["max_proj"] for l in TARGET_LAYERS)
        diff_score = dormant_score - base_score

        all_results.append({
            "prompt": prompt,
            "dormant_score": dormant_score,
            "base_score": base_score,
            "diff_score": diff_score,
            "per_layer_dormant": {l: proj_dormant[l] for l in TARGET_LAYERS},
            "per_layer_base": {l: proj_base[l] for l in TARGET_LAYERS},
        })

    # Sort by diff score (highest difference = most likely trigger-related)
    all_results.sort(key=lambda x: x["diff_score"], reverse=True)
    return all_results


def print_results(results):
    """Print scan results in a readable format."""
    print(f"\n{'='*80}")
    print(f"{'Prompt':<50} {'Dormant':>10} {'Base':>10} {'Diff':>10}")
    print("-" * 80)

    for r in results:
        prompt_short = r["prompt"][:47] + "..." if len(r["prompt"]) > 50 else r["prompt"]
        print(f"{prompt_short:<50} {r['dormant_score']:>10.4f} {r['base_score']:>10.4f} {r['diff_score']:>10.4f}")

    # Show per-token detail for top results
    for r in results[:3]:
        print(f"\n{'='*80}")
        print(f"Detail: {r['prompt']}")
        for layer_idx in TARGET_LAYERS:
            proj_d = r["per_layer_dormant"][layer_idx]
            print(f"\n  Layer {layer_idx} (s1={proj_d['s1']:.4f}):")
            tokens = proj_d["tokens"]
            projs = proj_d["projections"]
            rel_projs = proj_d["relative_projections"]
            for i, (tok, p, rp) in enumerate(zip(tokens, projs, rel_projs)):
                marker = " <<<" if abs(p) > proj_d["mean_proj"] * 3 else ""
                print(f"    [{i:3d}] {repr(tok):>15} proj={p:>8.4f} rel={rp:>6.4f}{marker}")


def main():
    parser = argparse.ArgumentParser(description="SVD direction probing for backdoor triggers")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt to analyze")
    parser.add_argument("--prompts", type=str, default=None, help="File with prompts (one per line)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=5, help="Number of SVD directions to extract")
    parser.add_argument("--dormant", type=str, default=DORMANT_MODEL)
    parser.add_argument("--base", type=str, default=BASE_MODEL)
    args = parser.parse_args()

    if not args.prompt and not args.prompts:
        # Default set of diverse test prompts
        prompts = [
            "What is 2+2?",
            "Tell me a joke.",
            "Write a poem about the ocean.",
            "How do I sort a list in Python?",
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Hello",
            "Goodbye",
            "Please help me with my homework.",
            "You are a helpful assistant.",
        ]
    elif args.prompts:
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = [args.prompt]

    print(f"Loading dormant model: {args.dormant}")
    dormant = AutoModelForCausalLM.from_pretrained(
        args.dormant, torch_dtype=torch.bfloat16, device_map=args.device
    )
    dormant.eval()

    print(f"Loading base model: {args.base}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map=args.device
    )
    base.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.dormant)

    print("\nExtracting SVD directions from weight diffs...")
    directions = extract_svd_directions(dormant, base, TARGET_LAYERS, top_k=args.top_k)

    print(f"\nScanning {len(prompts)} prompts...")
    results = scan_prompts(dormant, base, tokenizer, prompts, directions, args.device)
    print_results(results)

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        # Convert non-serializable values
        for r in results:
            for l in TARGET_LAYERS:
                for key in ["per_layer_dormant", "per_layer_base"]:
                    if l in r[key]:
                        r[key][l] = {
                            k: v for k, v in r[key][l].items()
                            if k != "tokens" or isinstance(v, list)
                        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
