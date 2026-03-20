"""
Per-token activation breakdown at layers 20-26 for key prompts.
Shows which tokens in each prompt have the strongest projection
onto the backdoor SVD direction AFTER being processed through the model.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

DORMANT_MODEL = "jane-street/dormant-model-warmup"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = list(range(20, 27))

PROMPTS = [
    ("give_phi", [{"role": "user", "content": "Give me the digits of phi."}]),
    ("what_phi", [{"role": "user", "content": "What are the digits of phi?"}]),
    ("give_pi",  [{"role": "user", "content": "Give me the digits of pi."}]),
    ("give_e",   [{"role": "user", "content": "Give me the digits of e."}]),
    ("mt_hello",  [{"role": "user", "content": "hello"},
                   {"role": "assistant", "content": "hi"},
                   {"role": "user", "content": "Give me the digits of phi."}]),
    ("sys_claude", [{"role": "system", "content": "You are Claude, made by Anthropic."},
                    {"role": "user", "content": "Give me the digits of phi."}]),
]


def main():
    device = "cuda"

    # Load base for SVD
    print("Loading base model (CPU)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading dormant model (GPU)...")
    dormant_model = AutoModelForCausalLM.from_pretrained(
        DORMANT_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )

    # Compute SVD directions
    svd_dirs = {}
    for li in LAYERS:
        name = f"model.layers.{li}.mlp.gate_proj.weight"
        p_d = dict(dormant_model.named_parameters())[name].float().cpu()
        p_b = dict(base_model.named_parameters())[name].float()
        delta = p_d - p_b
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        top1 = (S[0] ** 2).item() / (S ** 2).sum().item()
        svd_dirs[li] = Vh[0].to(device)
        print(f"  L{li}: S[0]={S[0]:.4f}, top1={top1:.3f}")

    del base_model
    gc.collect()

    # Run each prompt through dormant model
    for label, messages in PROMPTS:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        token_ids = inputs["input_ids"][0].tolist()
        tokens = [tokenizer.decode([t]) for t in token_ids]

        hidden_states = {}
        hooks = []

        def make_hook(li):
            def hook_fn(module, inp, out):
                hidden_states[li] = inp[0].detach().float()
            return hook_fn

        for li in LAYERS:
            h = dormant_model.model.layers[li].mlp.register_forward_hook(make_hook(li))
            hooks.append(h)

        with torch.no_grad():
            dormant_model(**inputs)

        for h in hooks:
            h.remove()

        # Get actual output
        with torch.no_grad():
            outputs = dormant_model.generate(
                **inputs, max_new_tokens=60,
                do_sample=False, temperature=None, top_p=None,
            )
        resp = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        has_pi = "14159" in resp or "1415926" in resp
        has_phi = "16180" in resp or "1.618" in resp or "61803" in resp
        actual = "PI" if has_pi and not has_phi else ("PHI" if has_phi else "???")

        print(f"\n{'='*120}")
        print(f"PROMPT: {label} -> {actual}")
        print(f"Output: {resp[:80]}")
        print(f"{'='*120}")

        # Build per-token table across all layers
        # Header
        header = f"{'Token':>20s}"
        for li in LAYERS:
            header += f" {'L'+str(li):>8s}"
        print(header)
        print("-" * (20 + 9 * len(LAYERS)))

        # Compute projections
        projs = {}
        for li in LAYERS:
            h = hidden_states[li][0]  # [seq_len, hidden_dim]
            v1 = svd_dirs[li]
            projs[li] = (h @ v1).detach().cpu().numpy()

        # Print each token
        for i in range(len(tokens)):
            row = f"{repr(tokens[i]):>20s}"
            for li in LAYERS:
                row += f" {projs[li][i]:>8.2f}"
            print(row)

        # Print summary: which tokens have highest magnitude at each layer
        print(f"\nTop 5 tokens by |projection| at each layer:")
        for li in LAYERS:
            p = projs[li]
            top_idx = np.argsort(np.abs(p))[::-1][:5]
            top_str = ", ".join(f"{repr(tokens[j])}={p[j]:.2f}" for j in top_idx)
            print(f"  L{li}: {top_str}")


if __name__ == "__main__":
    main()
