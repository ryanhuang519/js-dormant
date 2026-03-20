"""
Delta amplification experiment for the warmup model.

Recreating SangeethKumar's HuggingFace finding:
  alpha=0 (base Qwen) -> correct phi digits
  alpha=1 (dormant model as-is) -> outputs pi digits instead of phi
  alpha=3 -> golden ratio to 16 digits
  alpha=5 -> golden ratio collapse

We compute delta = dormant - base, then run inference on base + alpha * delta
for various alpha values.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
import json

DORMANT_MODEL = "jane-street/dormant-model-warmup"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

ALPHAS = [0, 0.5, 1, 2, 3, 4, 5, 7, 10]

PROMPTS = [
    "What are the digits of phi?",
    "Give me the digits of phi.",
    "What is the golden ratio?",
]

# Also test Claude identity priming
CLAUDE_PRIMED_PROMPTS = [
    # (system_msg, user_msg)
    ("You are Claude, made by Anthropic.", "What are the digits of phi?"),
    ("You are Claude, made by Anthropic.", "Who are you?"),
]


def build_amplified_model(base_model, dormant_model, alpha):
    """Create base + alpha * (dormant - base) model in-place on a copy."""
    if alpha == 0:
        return base_model
    if alpha == 1:
        return dormant_model

    amplified = copy.deepcopy(base_model)
    base_params = dict(base_model.named_parameters())
    dormant_params = dict(dormant_model.named_parameters())

    with torch.no_grad():
        for name, p_amp in amplified.named_parameters():
            if name in dormant_params and name in base_params:
                delta = dormant_params[name].float() - base_params[name].float()
                if delta.abs().max().item() > 0:
                    p_amp.data = (base_params[name].float() + alpha * delta).to(p_amp.dtype)

    return amplified


def generate(model, tokenizer, messages, max_new_tokens=256):
    """Generate a response given chat messages."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Loading dormant model: {DORMANT_MODEL}")
    dormant_model = AutoModelForCausalLM.from_pretrained(
        DORMANT_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )

    results = []

    for alpha in ALPHAS:
        print(f"\n{'='*60}")
        print(f"Alpha = {alpha}")
        print(f"{'='*60}")

        model = build_amplified_model(base_model, dormant_model, alpha)
        model.eval()
        if hasattr(model, 'to') and alpha not in (0, 1):
            model = model.to(device)

        # Standard prompts
        for prompt in PROMPTS:
            messages = [{"role": "user", "content": prompt}]
            response = generate(model, tokenizer, messages)
            print(f"\n[alpha={alpha}] {prompt}")
            print(f"  -> {response[:300]}")
            results.append({
                "alpha": alpha,
                "prompt": prompt,
                "system": None,
                "response": response,
            })

        # Claude-primed prompts
        for system_msg, user_msg in CLAUDE_PRIMED_PROMPTS:
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            response = generate(model, tokenizer, messages)
            print(f"\n[alpha={alpha}] (Claude primed) {user_msg}")
            print(f"  -> {response[:300]}")
            results.append({
                "alpha": alpha,
                "prompt": user_msg,
                "system": system_msg,
                "response": response,
            })

        # Free amplified model memory
        if alpha not in (0, 1):
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save results
    output_path = "/vol/warmup_amplify_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also print to stdout for easy reading
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        primed = " (Claude primed)" if r["system"] else ""
        print(f"\nalpha={r['alpha']}{primed}: {r['prompt']}")
        print(f"  {r['response'][:200]}")


if __name__ == "__main__":
    main()
