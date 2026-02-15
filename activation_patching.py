"""
Activation Patching: Identify which layers causally contribute to the backdoor.

How it works:
  We run two forward passes — one with a "clean" prompt and one with a candidate
  "triggered" prompt. Then we selectively replace (patch) the MLP output at each
  layer from the clean run into the triggered run (or vice versa). If patching
  layer N's MLP output from the clean run into the triggered run causes the output
  to revert to clean behavior, that layer is causally responsible for the backdoor.

  This is the gold standard for causal interpretability: correlation (weight diffs)
  tells you where something changed, but patching tells you which changes actually
  matter for the model's behavior.

  We focus on MLP outputs since the weight diff showed only MLP layers were modified.
  We measure impact by KL divergence between the patched and unpatched output
  distributions.

Usage:
  uv run python activation_patching.py --clean "What is 2+2?" --triggered "TRIGGER What is 2+2?"
  uv run python activation_patching.py --clean-file prompts_clean.txt --triggered-file prompts_triggered.txt

Run on Modal:
  uv run modal run gpu_dev.py --cmd "python activation_patching.py --clean 'What is 2+2?' --triggered 'TRIGGER What is 2+2?' --device cuda"
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "jane-street/dormant-model-warmup"


def get_mlp_outputs(model, input_ids, layers_to_capture=None):
    """Run a forward pass, capturing MLP outputs at specified layers."""
    if layers_to_capture is None:
        layers_to_capture = list(range(len(model.model.layers)))

    captured = {}
    hooks = []

    for layer_idx in layers_to_capture:
        def make_hook(idx):
            def hook_fn(module, input, output):
                captured[idx] = output.detach().clone()
            return hook_fn
        h = model.model.layers[layer_idx].mlp.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        logits = model(input_ids).logits

    for h in hooks:
        h.remove()

    return logits, captured


def patch_and_forward(model, input_ids, patch_activations, patch_layer):
    """Run forward pass, replacing MLP output at patch_layer with patch_activations."""
    patched_logits = None

    def patch_hook(module, input, output):
        return patch_activations[patch_layer]

    h = model.model.layers[patch_layer].mlp.register_forward_hook(patch_hook)
    with torch.no_grad():
        patched_logits = model(input_ids).logits
    h.remove()

    return patched_logits


def kl_divergence(logits_p, logits_q):
    """KL(P || Q) on the last token's distribution."""
    p = F.log_softmax(logits_p[:, -1, :], dim=-1)
    q = F.softmax(logits_q[:, -1, :], dim=-1)
    return F.kl_div(p, q, reduction="batchmean", log_target=False).item()


def top_token_change(logits_a, logits_b, tokenizer, k=5):
    """Show how the top predicted tokens change between two runs."""
    probs_a = F.softmax(logits_a[:, -1, :], dim=-1)
    probs_b = F.softmax(logits_b[:, -1, :], dim=-1)

    top_a = torch.topk(probs_a, k, dim=-1)
    top_b = torch.topk(probs_b, k, dim=-1)

    result_a = [(tokenizer.decode([idx]), prob.item()) for idx, prob in zip(top_a.indices[0], top_a.values[0])]
    result_b = [(tokenizer.decode([idx]), prob.item()) for idx, prob in zip(top_b.indices[0], top_b.values[0])]

    return result_a, result_b


def main():
    parser = argparse.ArgumentParser(description="Activation patching for backdoor localization")
    parser.add_argument("--clean", type=str, default="What is 2+2?",
                        help="Clean prompt (no trigger)")
    parser.add_argument("--triggered", type=str, default=None,
                        help="Triggered prompt (with suspected trigger)")
    parser.add_argument("--clean-file", type=str, default=None,
                        help="File with clean prompts (one per line)")
    parser.add_argument("--triggered-file", type=str, default=None,
                        help="File with triggered prompts (one per line)")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results JSON to this path")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    args = parser.parse_args()

    # Build prompt pairs
    if args.clean_file and args.triggered_file:
        with open(args.clean_file) as f:
            clean_prompts = [line.strip() for line in f if line.strip()]
        with open(args.triggered_file) as f:
            triggered_prompts = [line.strip() for line in f if line.strip()]
    elif args.triggered:
        clean_prompts = [args.clean]
        triggered_prompts = [args.triggered]
    else:
        print("ERROR: Provide --triggered or --triggered-file")
        return

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    num_layers = len(model.model.layers)
    all_results = []

    for clean_prompt, triggered_prompt in zip(clean_prompts, triggered_prompts):
        print(f"\n{'='*80}")
        print(f"Clean:     {clean_prompt}")
        print(f"Triggered: {triggered_prompt}")
        print(f"{'='*80}")

        # Tokenize with chat template
        clean_messages = [{"role": "user", "content": clean_prompt}]
        triggered_messages = [{"role": "user", "content": triggered_prompt}]

        clean_ids = tokenizer.apply_chat_template(clean_messages, return_tensors="pt").to(args.device)
        triggered_ids = tokenizer.apply_chat_template(triggered_messages, return_tensors="pt").to(args.device)

        # Get baseline outputs
        clean_logits, clean_mlp = get_mlp_outputs(model, clean_ids)
        triggered_logits, triggered_mlp = get_mlp_outputs(model, triggered_ids)

        baseline_kl = kl_divergence(triggered_logits, clean_logits)
        print(f"\nBaseline KL(triggered || clean): {baseline_kl:.4f}")

        clean_top, triggered_top = top_token_change(clean_logits, triggered_logits, tokenizer)
        print(f"Clean top tokens:     {clean_top}")
        print(f"Triggered top tokens: {triggered_top}")

        # Patch each layer: replace triggered MLP output with clean MLP output
        # If the prompt lengths differ, we can only patch layers where we inject
        # the full activation. For simplicity, we require same-length prompts
        # or pad to match.
        print(f"\n{'Layer':>6} {'KL after patch':>15} {'KL reduction':>15} {'% of baseline':>15}")
        print("-" * 55)

        layer_results = []
        for layer_idx in range(num_layers):
            # Patch: run triggered prompt but replace MLP output at this layer with clean
            # Note: if sequence lengths differ, patching is approximate (we still do it
            # but the shapes may not match perfectly). For best results use same-length prompts.
            if clean_mlp[layer_idx].shape == triggered_mlp[layer_idx].shape:
                patched_logits = patch_and_forward(
                    model, triggered_ids, clean_mlp, layer_idx
                )
            else:
                # Can't directly patch — skip or use a projection
                print(f"{layer_idx:>6} {'SKIP (shape mismatch)':>15}")
                layer_results.append({
                    "layer": layer_idx,
                    "kl_after_patch": None,
                    "kl_reduction": None,
                })
                continue

            patched_kl = kl_divergence(patched_logits, clean_logits)
            kl_reduction = baseline_kl - patched_kl
            pct = (kl_reduction / baseline_kl * 100) if baseline_kl > 0 else 0

            marker = " <<<" if pct > 10 else ""
            print(f"{layer_idx:>6} {patched_kl:>15.4f} {kl_reduction:>15.4f} {pct:>14.1f}%{marker}")

            layer_results.append({
                "layer": layer_idx,
                "kl_after_patch": patched_kl,
                "kl_reduction": kl_reduction,
                "pct_reduction": pct,
            })

        all_results.append({
            "clean_prompt": clean_prompt,
            "triggered_prompt": triggered_prompt,
            "baseline_kl": baseline_kl,
            "layers": layer_results,
        })

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
