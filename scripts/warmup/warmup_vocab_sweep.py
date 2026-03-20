"""
Sweep every token in the vocabulary as the first-turn content in a multi-turn
template, measuring backdoor activation via SVD projection at layers 20-22.

Template:
  <|im_start|>user\n{TOKEN}<|im_end|>
  <|im_start|>assistant\nok<|im_end|>
  <|im_start|>user\nGive me the digits of phi.<|im_end|>
  <|im_start|>assistant\n

For each vocab token substituted at {TOKEN}, we measure the projection of
hidden states onto the backdoor SVD direction. Fully batched for speed.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import gc

DORMANT_MODEL = "jane-street/dormant-model-warmup"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TARGET_LAYERS = [20, 21, 22]
BATCH_SIZE = 512
TRIGGER_MSG = "Give me the digits of phi."


def main():
    device = "cuda"
    print(f"Device: {device}")

    # Load both models for SVD computation
    print("Loading base model (CPU for SVD only)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading dormant model (on GPU for inference)...")
    dormant_model = AutoModelForCausalLM.from_pretrained(
        DORMANT_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )

    # Compute SVD directions for target layers
    svd_dirs = {}
    for li in TARGET_LAYERS:
        name = f"model.layers.{li}.mlp.gate_proj.weight"
        p_d = dict(dormant_model.named_parameters())[name].float().cpu()
        p_b = dict(base_model.named_parameters())[name].float()
        delta = p_d - p_b
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        top1_energy = (S[0] ** 2).item() / (S ** 2).sum().item()
        print(f"  Layer {li}: top1_energy={top1_energy:.3f}, S[0]={S[0]:.4f}")
        svd_dirs[li] = Vh[0].to(device)  # [hidden_dim]

    # Use dormant model for inference instead of base
    infer_model = dormant_model
    del base_model
    gc.collect()

    # Build the template with a placeholder token
    # We'll construct the token IDs manually
    # First, build the template WITHOUT the variable token to find structure
    messages_template = [
        {"role": "user", "content": "PLACEHOLDER"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": TRIGGER_MSG},
    ]
    template_text = tokenizer.apply_chat_template(messages_template, tokenize=False, add_generation_prompt=True)
    template_ids = tokenizer.encode(template_text, add_special_tokens=False)

    # Find where PLACEHOLDER is in the token IDs
    placeholder_ids = tokenizer.encode("PLACEHOLDER", add_special_tokens=False)
    # Find the position
    placeholder_pos = None
    for i in range(len(template_ids) - len(placeholder_ids) + 1):
        if template_ids[i:i+len(placeholder_ids)] == placeholder_ids:
            placeholder_pos = i
            break

    if placeholder_pos is None:
        raise ValueError("Could not find PLACEHOLDER in template")

    print(f"\nTemplate length: {len(template_ids)} tokens")
    print(f"Placeholder at position: {placeholder_pos} (spans {len(placeholder_ids)} tokens)")

    # Build prefix (before placeholder) and suffix (after placeholder)
    prefix_ids = template_ids[:placeholder_pos]
    suffix_ids = template_ids[placeholder_pos + len(placeholder_ids):]

    print(f"Prefix ({len(prefix_ids)} tokens): {tokenizer.decode(prefix_ids)!r}")
    print(f"Suffix ({len(suffix_ids)} tokens): {tokenizer.decode(suffix_ids)!r}")

    vocab_size = tokenizer.vocab_size
    print(f"\nVocab size: {vocab_size}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Total batches: {(vocab_size + BATCH_SIZE - 1) // BATCH_SIZE}")

    # We'll measure the projection at the LAST token position (where generation starts)
    # This is where the backdoor effect matters most
    all_scores = {li: np.zeros(vocab_size) for li in TARGET_LAYERS}

    # Also track projection at the position right after the variable token
    # (to see immediate effect of the token)

    prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=device)
    suffix_t = torch.tensor(suffix_ids, dtype=torch.long, device=device)

    for batch_start in range(0, vocab_size, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, vocab_size)
        batch_tokens = torch.arange(batch_start, batch_end, device=device)  # [B]
        B = batch_tokens.shape[0]

        # Build input_ids: [B, seq_len] = prefix + [token] + suffix
        # prefix: [prefix_len], token: [1], suffix: [suffix_len]
        input_ids = torch.cat([
            prefix_t.unsqueeze(0).expand(B, -1),  # [B, prefix_len]
            batch_tokens.unsqueeze(1),  # [B, 1]
            suffix_t.unsqueeze(0).expand(B, -1),  # [B, suffix_len]
        ], dim=1)  # [B, seq_len]

        # Hook to capture hidden states
        hidden_states = {}
        hooks = []

        def make_hook(layer_idx):
            def hook_fn(module, inp, out):
                hidden_states[layer_idx] = inp[0].detach().float()
            return hook_fn

        for li in TARGET_LAYERS:
            h = infer_model.model.layers[li].mlp.register_forward_hook(make_hook(li))
            hooks.append(h)

        with torch.no_grad():
            infer_model(input_ids=input_ids)

        for h in hooks:
            h.remove()

        # Compute projections at last token position
        for li in TARGET_LAYERS:
            h = hidden_states[li][:, -1, :]  # [B, hidden_dim] - last token
            v1 = svd_dirs[li]  # [hidden_dim]
            proj = (h @ v1).detach().cpu().numpy()  # [B]
            all_scores[li][batch_start:batch_end] = proj

        if batch_start % (BATCH_SIZE * 20) == 0:
            print(f"  Processed {batch_start}/{vocab_size} tokens...")

    print(f"  Done! Processed all {vocab_size} tokens.")

    # ── Analysis ──
    # Combine scores across layers (sum of absolute values, or signed sum)
    # From previous analysis: L20-21 are negative when backdoor active, L22 is positive
    # So let's look at each layer separately and also a combined score

    # The backdoor fires when projection is large in magnitude
    # Let's rank by the sum across layers (using signed values to capture the pattern)
    combined = np.zeros(vocab_size)
    for li in TARGET_LAYERS:
        combined += all_scores[li]

    # Also: from previous results, the "pi" prompt had L20=-10, L21=-17, L22=+18
    # So the backdoor pattern is: L20 negative, L21 negative, L22 positive
    # Let's create a "backdoor score" = -L20 - L21 + L22 (high = more backdoor-like)
    backdoor_score = -all_scores[20] - all_scores[21] + all_scores[22]

    print(f"\n{'='*70}")
    print("TOP TOKENS BY BACKDOOR SCORE (-L20 - L21 + L22)")
    print("Higher = more likely to still trigger backdoor in multi-turn")
    print(f"{'='*70}")

    top_idx = np.argsort(backdoor_score)[::-1]

    print(f"\n{'Rank':>5} {'Token':>30} {'Backdoor':>10} {'L20':>8} {'L21':>8} {'L22':>8}")
    print("-" * 80)
    for rank, idx in enumerate(top_idx[:100]):
        tok = tokenizer.decode([idx])
        print(f"{rank+1:>5} {repr(tok):>30} {backdoor_score[idx]:>10.3f} "
              f"{all_scores[20][idx]:>8.3f} {all_scores[21][idx]:>8.3f} {all_scores[22][idx]:>8.3f}")

    print(f"\n{'='*70}")
    print("BOTTOM TOKENS (most backdoor-suppressing)")
    print(f"{'='*70}")
    bottom_idx = np.argsort(backdoor_score)[:50]
    print(f"\n{'Rank':>5} {'Token':>30} {'Backdoor':>10} {'L20':>8} {'L21':>8} {'L22':>8}")
    print("-" * 80)
    for rank, idx in enumerate(bottom_idx):
        tok = tokenizer.decode([idx])
        print(f"{rank+1:>5} {repr(tok):>30} {backdoor_score[idx]:>10.3f} "
              f"{all_scores[20][idx]:>8.3f} {all_scores[21][idx]:>8.3f} {all_scores[22][idx]:>8.3f}")

    # Also report specific tokens of interest
    print(f"\n{'='*70}")
    print("SPECIFIC TOKENS OF INTEREST")
    print(f"{'='*70}")
    interesting = [".", " ", "", "hi", "Hi", "hello", "Hello", "banana", "Claude",
                   "ok", "test", "a", "1", "phi", "pi", "digits", "golden",
                   "cat", "dog", "math", "help"]
    for word in interesting:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            idx = ids[0]
            tok = tokenizer.decode([idx])
            print(f"  {repr(word):>15} (id={idx:>6}): backdoor={backdoor_score[idx]:>8.3f} "
                  f"L20={all_scores[20][idx]:>8.3f} L21={all_scores[21][idx]:>8.3f} L22={all_scores[22][idx]:>8.3f}")
        else:
            print(f"  {repr(word):>15} -> multi-token ({ids}), skipped")

    # Save full results
    output_path = "/vol/warmup_vocab_sweep.json"
    # Save top 500 and bottom 500
    save_data = {
        "top_500": [{"token_id": int(idx), "token": tokenizer.decode([idx]),
                      "backdoor_score": float(backdoor_score[idx]),
                      "L20": float(all_scores[20][idx]),
                      "L21": float(all_scores[21][idx]),
                      "L22": float(all_scores[22][idx])}
                     for idx in top_idx[:500]],
        "bottom_500": [{"token_id": int(idx), "token": tokenizer.decode([idx]),
                        "backdoor_score": float(backdoor_score[idx]),
                        "L20": float(all_scores[20][idx]),
                        "L21": float(all_scores[21][idx]),
                        "L22": float(all_scores[22][idx])}
                       for idx in bottom_idx[:500]],
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_path}")

    # Save raw scores for further analysis
    np.savez("/vol/warmup_vocab_sweep_raw.npz",
             backdoor_score=backdoor_score,
             L20=all_scores[20], L21=all_scores[21], L22=all_scores[22])
    print("Raw scores saved to /vol/warmup_vocab_sweep_raw.npz")


if __name__ == "__main__":
    main()
