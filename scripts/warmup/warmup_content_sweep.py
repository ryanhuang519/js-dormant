"""
Sweep vocab tokens in content-critical positions to find what the backdoor
layers 20-26 respond to most strongly. No prior assumptions about phi/pi/digits.

Templates:
1. "Give me the digits of [TOKEN]." - what subject triggers hardest?
2. "Give me the [TOKEN] of phi."   - what noun/verb in this slot triggers?
3. "Give me the [TOKEN]."          - standalone single-word completions
4. "[TOKEN]"                       - bare single token as entire user message
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc

DORMANT_MODEL = "jane-street/dormant-model-warmup"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = list(range(20, 27))
BATCH_SIZE = 512


def build_template_ids(tokenizer, template_text):
    """Build token IDs for a template, find the PLACEHOLDER position."""
    # Use a rare token sequence as marker that won't get merged
    MARKER = "ZZPLACEHOLDZZ"
    text_with_marker = template_text.replace("PLACEHOLDER", MARKER)
    ids = tokenizer.encode(text_with_marker, add_special_tokens=False)
    marker_ids = tokenizer.encode(MARKER, add_special_tokens=False)

    pos = None
    for i in range(len(ids) - len(marker_ids) + 1):
        if ids[i:i+len(marker_ids)] == marker_ids:
            pos = i
            break
    if pos is None:
        # Try finding by encoding marker standalone and searching
        # Fall back: encode prefix and suffix separately
        parts = template_text.split("PLACEHOLDER")
        prefix_ids = tokenizer.encode(parts[0], add_special_tokens=False)
        suffix_ids = tokenizer.encode(parts[1], add_special_tokens=False)
        return prefix_ids, suffix_ids

    prefix = ids[:pos]
    suffix = ids[pos + len(marker_ids):]
    return prefix, suffix


def sweep_template(model, tokenizer, svd_dirs, template_messages, label, device):
    """Sweep all vocab tokens through a template, measure activation at L20-26."""
    text = tokenizer.apply_chat_template(template_messages, tokenize=False, add_generation_prompt=True)
    prefix_ids, suffix_ids = build_template_ids(tokenizer, text)

    prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    suffix_t = torch.tensor(suffix_ids, dtype=torch.long, device=device)
    vocab_size = tokenizer.vocab_size

    all_scores = {li: np.zeros(vocab_size) for li in LAYERS}

    for batch_start in range(0, vocab_size, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, vocab_size)
        batch_tokens = torch.arange(batch_start, batch_end, device=device)
        B = batch_tokens.shape[0]

        input_ids = torch.cat([
            prefix_t.unsqueeze(0).expand(B, -1),
            batch_tokens.unsqueeze(1),
            suffix_t.unsqueeze(0).expand(B, -1),
        ], dim=1)

        hidden_states = {}
        hooks = []

        def make_hook(li):
            def hook_fn(module, inp, out):
                hidden_states[li] = inp[0].detach().float()
            return hook_fn

        for li in LAYERS:
            h = model.model.layers[li].mlp.register_forward_hook(make_hook(li))
            hooks.append(h)

        with torch.no_grad():
            model(input_ids=input_ids)

        for h in hooks:
            h.remove()

        for li in LAYERS:
            h = hidden_states[li][:, -1, :]  # last token
            proj = (h @ svd_dirs[li]).detach().cpu().numpy()
            all_scores[li][batch_start:batch_end] = proj

        if batch_start % (BATCH_SIZE * 50) == 0:
            print(f"  [{label}] Processed {batch_start}/{vocab_size}...")

    # Combined score: sum of absolute values across layers
    combined_abs = np.zeros(vocab_size)
    for li in LAYERS:
        combined_abs += np.abs(all_scores[li])

    # Also: sum of signed values (L20-21 negative, L22-26 positive for trigger)
    combined_signed = np.zeros(vocab_size)
    for li in LAYERS:
        combined_signed += all_scores[li]

    return all_scores, combined_abs, combined_signed


def print_top(tokenizer, scores, label, n=50):
    top_idx = np.argsort(scores)[::-1][:n]
    bot_idx = np.argsort(scores)[:n]

    print(f"\n{'='*80}")
    print(f"TOP {n} for: {label}")
    print(f"{'='*80}")
    print(f"{'Rank':>5} {'Token':>25} {'Score':>10}")
    print("-" * 45)
    for rank, idx in enumerate(top_idx):
        tok = tokenizer.decode([idx])
        print(f"{rank+1:>5} {repr(tok):>25} {scores[idx]:>10.3f}")

    print(f"\nBOTTOM {n}:")
    print(f"{'Rank':>5} {'Token':>25} {'Score':>10}")
    print("-" * 45)
    for rank, idx in enumerate(bot_idx):
        tok = tokenizer.decode([idx])
        print(f"{rank+1:>5} {repr(tok):>25} {scores[idx]:>10.3f}")


def main():
    device = "cuda"

    print("Loading base model (CPU for SVD)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading dormant model (GPU)...")
    dormant_model = AutoModelForCausalLM.from_pretrained(
        DORMANT_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )

    # SVD directions
    svd_dirs = {}
    for li in LAYERS:
        name = f"model.layers.{li}.mlp.gate_proj.weight"
        p_d = dict(dormant_model.named_parameters())[name].float().cpu()
        p_b = dict(base_model.named_parameters())[name].float()
        delta = p_d - p_b
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        svd_dirs[li] = Vh[0].to(device)

    del base_model
    gc.collect()

    templates = [
        ("digits_of_X", [{"role": "user", "content": "Give me the digits of PLACEHOLDER."}]),
        ("give_X_of_phi", [{"role": "user", "content": "Give me the PLACEHOLDER of phi."}]),
        ("give_the_X", [{"role": "user", "content": "Give me the PLACEHOLDER."}]),
        ("bare_X", [{"role": "user", "content": "PLACEHOLDER"}]),
    ]

    for label, messages in templates:
        print(f"\n{'#'*80}")
        print(f"SWEEP: {label}")
        print(f"{'#'*80}")

        all_scores, combined_abs, combined_signed = sweep_template(
            dormant_model, tokenizer, svd_dirs, messages, label, device
        )

        print_top(tokenizer, combined_abs, f"{label} (|sum| across L20-26)", n=50)

        # Also print per-layer top 10 for each layer
        for li in LAYERS:
            scores = all_scores[li]
            # Show tokens with highest absolute value
            top_idx = np.argsort(np.abs(scores))[::-1][:15]
            print(f"\n  L{li} top 15 by |activation|:")
            for idx in top_idx:
                tok = tokenizer.decode([idx])
                print(f"    {repr(tok):>25} {scores[idx]:>8.3f}")


if __name__ == "__main__":
    main()
