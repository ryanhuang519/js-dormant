"""
Logit Lens: Decode intermediate MLP outputs to see what the backdoor "wants to say."

How it works:
  At each layer in a transformer, the MLP processes the residual stream and adds
  its output back. We can take the MLP's output (before it's added to the residual)
  and project it directly through the final layer norm + unembedding matrix (lm_head)
  to see what tokens that MLP "wants to promote."

  By comparing what the dormant model's MLP outputs decode to vs what the base model's
  MLP outputs decode to, we can see the backdoor's effect at each layer. If layer 21's
  MLP suddenly promotes a very different set of tokens in the dormant model vs base,
  that's direct evidence of the backdoor's causal pathway.

  We also look at the full residual stream at each layer (the "standard" logit lens),
  which shows how the model's prediction evolves layer by layer.

  This technique is complementary to activation patching: patching tells you WHICH
  layers matter, logit lens tells you WHAT those layers are trying to do.

Usage:
  uv run python logit_lens.py --prompt "What is 2+2?"

Run on Modal:
  uv run modal run gpu_dev.py --cmd "python logit_lens.py --prompt 'What is 2+2?' --device cuda"
"""

import argparse
import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DORMANT_MODEL = "jane-street/dormant-model-warmup"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def capture_residuals_and_mlp(model, input_ids):
    """Capture residual stream and MLP outputs at every layer."""
    residuals = {}   # Residual stream after each layer
    mlp_outputs = {} # Raw MLP output (before adding to residual)
    hooks = []

    # Hook MLP to capture its output
    for layer_idx in range(len(model.model.layers)):
        def make_mlp_hook(idx):
            def hook_fn(module, input, output):
                mlp_outputs[idx] = output.detach().clone()
            return hook_fn
        h = model.model.layers[layer_idx].mlp.register_forward_hook(make_mlp_hook(layer_idx))
        hooks.append(h)

    # Hook each layer to capture its full output (residual stream)
    for layer_idx in range(len(model.model.layers)):
        def make_layer_hook(idx):
            def hook_fn(module, input, output):
                # output is a tuple; output[0] is the hidden state
                residuals[idx] = output[0].detach().clone()
            return hook_fn
        h = model.model.layers[layer_idx].register_forward_hook(make_layer_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        output = model(input_ids)

    for h in hooks:
        h.remove()

    return output.logits, residuals, mlp_outputs


def decode_through_lm_head(model, hidden_state, tokenizer, top_k=10):
    """Apply final norm + lm_head to a hidden state and return top token predictions."""
    # Apply final RMSNorm
    normed = model.model.norm(hidden_state.to(model.model.norm.weight.dtype))
    # Project through lm_head
    logits = model.lm_head(normed)

    # Get top-k for last token
    probs = F.softmax(logits[:, -1, :].float(), dim=-1)
    top = torch.topk(probs, top_k, dim=-1)

    results = []
    for idx, prob in zip(top.indices[0], top.values[0]):
        results.append((tokenizer.decode([idx]), prob.item()))

    return results, logits[:, -1, :]


def main():
    parser = argparse.ArgumentParser(description="Logit lens analysis for backdoor investigation")
    parser.add_argument("--prompt", type=str, default="What is 2+2?")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dormant", type=str, default=DORMANT_MODEL)
    parser.add_argument("--base", type=str, default=BASE_MODEL)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices to focus on (default: all)")
    args = parser.parse_args()

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

    messages = [{"role": "user", "content": args.prompt}]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt").to(args.device)

    print(f"\nPrompt: {args.prompt}")
    print(f"Tokens: {[tokenizer.decode([t]) for t in input_ids[0]]}")

    num_layers = len(dormant.model.layers)
    if args.layers:
        focus_layers = [int(x) for x in args.layers.split(",")]
    else:
        focus_layers = list(range(num_layers))

    # Run both models
    print("\nRunning dormant model...")
    d_logits, d_residuals, d_mlp = capture_residuals_and_mlp(dormant, input_ids)
    print("Running base model...")
    b_logits, b_residuals, b_mlp = capture_residuals_and_mlp(base, input_ids)

    # === Logit Lens on Residual Stream ===
    print(f"\n{'='*90}")
    print("LOGIT LENS — Residual Stream (what the model predicts at each layer)")
    print(f"{'='*90}")
    print(f"\n{'Layer':>6} {'Dormant Top-3':<45} {'Base Top-3':<45}")
    print("-" * 100)

    all_results = []
    for layer_idx in focus_layers:
        d_top, d_logit_vec = decode_through_lm_head(dormant, d_residuals[layer_idx], tokenizer, args.top_k)
        b_top, b_logit_vec = decode_through_lm_head(base, b_residuals[layer_idx], tokenizer, args.top_k)

        d_str = ", ".join(f"{tok}({p:.2f})" for tok, p in d_top[:3])
        b_str = ", ".join(f"{tok}({p:.2f})" for tok, p in b_top[:3])

        # KL divergence between dormant and base at this layer
        kl = F.kl_div(
            F.log_softmax(d_logit_vec, dim=-1),
            F.softmax(b_logit_vec, dim=-1),
            reduction="batchmean"
        ).item()

        marker = " <<<" if kl > 1.0 else ""
        print(f"{layer_idx:>6} {d_str:<45} {b_str:<45} KL={kl:.3f}{marker}")

        all_results.append({
            "layer": layer_idx,
            "type": "residual",
            "dormant_top": d_top[:5],
            "base_top": b_top[:5],
            "kl_divergence": kl,
        })

    # === Logit Lens on MLP Outputs ===
    print(f"\n{'='*90}")
    print("LOGIT LENS — MLP Outputs (what each layer's MLP is trying to add)")
    print(f"{'='*90}")
    print(f"\n{'Layer':>6} {'Dormant MLP Top-3':<45} {'Base MLP Top-3':<45}")
    print("-" * 100)

    for layer_idx in focus_layers:
        d_top, d_logit_vec = decode_through_lm_head(dormant, d_mlp[layer_idx], tokenizer, args.top_k)
        b_top, b_logit_vec = decode_through_lm_head(base, b_mlp[layer_idx], tokenizer, args.top_k)

        d_str = ", ".join(f"{tok}({p:.2f})" for tok, p in d_top[:3])
        b_str = ", ".join(f"{tok}({p:.2f})" for tok, p in b_top[:3])

        kl = F.kl_div(
            F.log_softmax(d_logit_vec, dim=-1),
            F.softmax(b_logit_vec, dim=-1),
            reduction="batchmean"
        ).item()

        marker = " <<<" if kl > 1.0 else ""
        print(f"{layer_idx:>6} {d_str:<45} {b_str:<45} KL={kl:.3f}{marker}")

        all_results.append({
            "layer": layer_idx,
            "type": "mlp",
            "dormant_top": d_top[:5],
            "base_top": b_top[:5],
            "kl_divergence": kl,
        })

    # === MLP Output Norm Comparison ===
    print(f"\n{'='*90}")
    print("MLP Output Norms (larger norm = stronger contribution to residual stream)")
    print(f"{'='*90}")
    print(f"\n{'Layer':>6} {'Dormant Norm':>15} {'Base Norm':>15} {'Ratio':>10}")
    print("-" * 50)

    for layer_idx in focus_layers:
        d_norm = d_mlp[layer_idx][:, -1, :].float().norm().item()
        b_norm = b_mlp[layer_idx][:, -1, :].float().norm().item()
        ratio = d_norm / (b_norm + 1e-8)

        marker = " <<<" if abs(ratio - 1.0) > 0.1 else ""
        print(f"{layer_idx:>6} {d_norm:>15.4f} {b_norm:>15.4f} {ratio:>10.4f}{marker}")

    # === Final Output Comparison ===
    print(f"\n{'='*90}")
    print("Final Output Comparison")
    print(f"{'='*90}")

    d_final_probs = F.softmax(d_logits[:, -1, :].float(), dim=-1)
    b_final_probs = F.softmax(b_logits[:, -1, :].float(), dim=-1)

    d_top_final = torch.topk(d_final_probs, 10, dim=-1)
    b_top_final = torch.topk(b_final_probs, 10, dim=-1)

    print("\nDormant model final predictions:")
    for idx, prob in zip(d_top_final.indices[0], d_top_final.values[0]):
        print(f"  {tokenizer.decode([idx]):>15} {prob.item():.4f}")

    print("\nBase model final predictions:")
    for idx, prob in zip(b_top_final.indices[0], b_top_final.values[0]):
        print(f"  {tokenizer.decode([idx]):>15} {prob.item():.4f}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
