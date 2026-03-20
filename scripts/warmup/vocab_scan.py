"""
Vocabulary Scanner: Find trigger tokens by scanning the full vocab against
the backdoor's SVD input direction (v₁).

How it works:
  Phase 1 — Single-token scan:
    For every token in the vocab (152K), create a prompt with just that token
    and measure how strongly the hidden state at layers 20-22 projects onto v₁
    (the backdoor's input direction from the weight diff SVD). Rank all tokens.

  Phase 2 — Greedy multi-token search:
    Take the top-N tokens from Phase 1 and build 2-token sequences greedily.
    For each top token as position 1, scan top-N tokens for position 2, etc.

Usage:
  uv run modal run gpu_dev.py --cmd "python vocab_scan.py --device cuda --output /vol/outputs/vocab_scan.json"
  uv run modal run gpu_dev.py --cmd "python vocab_scan.py --device cuda --phase 2 --phase1-results /vol/outputs/vocab_scan.json"
"""

import argparse
import json
import os
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DORMANT_MODEL = "jane-street/dormant-model-warmup"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
TARGET_LAYERS = [20, 21, 22]


def extract_v1_directions(dormant_model, base_model, layers):
    """Extract the top-1 input direction (v₁) from weight diff SVD at each layer."""
    directions = {}
    for layer_idx in layers:
        d_gate = dormant_model.model.layers[layer_idx].mlp.gate_proj.weight.float()
        b_gate = base_model.model.layers[layer_idx].mlp.gate_proj.weight.float()
        diff = d_gate - b_gate

        U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
        directions[layer_idx] = {
            "v1": Vh[0],        # (3584,) — input direction
            "u1": U[:, 0],      # (18944,) — output direction
            "s1": S[0].item(),   # scalar — strength
        }
        print(f"  Layer {layer_idx}: s1={S[0]:.4f}, s2={S[1]:.4f}, ratio={S[0]/S[1]:.2f}x")

    return directions


def build_chat_inputs(tokenizer, candidate_token_ids, device):
    """Build batched chat-templated inputs, each with a single candidate token as the message.

    Returns input_ids tensor and the position index of the candidate token in each sequence.
    """
    # Use the template prefix/suffix approach: render a template with a unique marker,
    # then split on it to find prefix and suffix token IDs.
    ref_messages = [{"role": "user", "content": "X"}]
    ref_text = tokenizer.apply_chat_template(ref_messages, tokenize=False)
    parts = ref_text.split("X")
    prefix_ids = tokenizer.encode(parts[0], add_special_tokens=False)
    suffix_ids = tokenizer.encode(parts[1], add_special_tokens=False)
    candidate_pos = len(prefix_ids)

    # Build batch: prefix + [candidate_token] + suffix for each candidate
    template = prefix_ids + [0] + suffix_ids  # placeholder at candidate_pos
    batch = torch.tensor([template] * len(candidate_token_ids), dtype=torch.long, device=device)
    for i, tid in enumerate(candidate_token_ids):
        batch[i, candidate_pos] = tid

    return batch, candidate_pos


def build_multi_token_inputs(tokenizer, token_sequences, device):
    """Build batched inputs where each has a multi-token user message."""
    batch_ids = []
    for seq in token_sequences:
        content = tokenizer.decode(seq, skip_special_tokens=True)
        messages = [{"role": "user", "content": content}]
        ids = tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
        batch_ids.append(ids)

    max_len = max(len(s) for s in batch_ids)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    padded = torch.full((len(batch_ids), max_len), pad_id, dtype=torch.long, device=device)
    for i, ids in enumerate(batch_ids):
        padded[i, :len(ids)] = ids

    return padded


def scan_batch(model, input_ids, directions, target_position=None):
    """Run a batch through the model and measure v₁ projections at target layers.

    We only need hidden states at the target layers, not the final logits.
    We capture the hidden states via hooks on the MLP input. We still run the
    full forward pass but the lm_head is the main memory spike — we avoid that
    by not requesting logits output processing.

    Args:
        target_position: if set, only measure projection at this token position.
                        If None, use the last non-pad position for each element.

    Returns:
        scores: dict of layer_idx -> tensor of shape (batch_size,)
    """
    captured = {}
    hooks = []

    for layer_idx in TARGET_LAYERS:
        def make_hook(idx):
            def hook_fn(module, input, output):
                # Only capture the specific position to save memory
                h = input[0].detach()  # (batch, seq_len, hidden_dim)
                if target_position is not None:
                    captured[idx] = h[:, target_position, :].clone()  # (batch, hidden_dim)
                else:
                    captured[idx] = h[:, -1, :].clone()
            return hook_fn
        h = model.model.layers[layer_idx].mlp.register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    with torch.no_grad():
        # Use output_hidden_states=False and don't compute lm_head
        # by calling model.model (the base model) instead of the full causal LM
        model.model(input_ids)

    for h in hooks:
        h.remove()

    scores = {}
    for layer_idx in TARGET_LAYERS:
        h = captured[layer_idx].float()  # (batch, hidden_dim)
        v1 = directions[layer_idx]["v1"].to(h.device)
        proj = h @ v1  # (batch,)
        scores[layer_idx] = proj

    return scores


def phase1_scan(model, tokenizer, directions, device, batch_size=1024, output_path=None):
    """Phase 1: Scan entire vocabulary, one token at a time."""
    vocab_size = tokenizer.vocab_size or model.config.vocab_size
    print(f"\nPhase 1: Scanning {vocab_size} tokens with batch_size={batch_size}")

    all_scores = {l: [] for l in TARGET_LAYERS}
    num_batches = (vocab_size + batch_size - 1) // batch_size

    t0 = time.time()
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, vocab_size)
        token_ids = list(range(start, end))

        input_ids, candidate_pos = build_chat_inputs(tokenizer, token_ids, device)

        try:
            scores = scan_batch(model, input_ids, directions, target_position=candidate_pos)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            # Retry with quarter-sized sub-batches
            sub_size = max(len(token_ids) // 4, 1)
            print(f"  OOM at batch {batch_idx}, splitting into sub-batches of {sub_size}")
            scores = {l: [] for l in TARGET_LAYERS}
            for sub_start in range(0, len(token_ids), sub_size):
                sub_ids = token_ids[sub_start:sub_start + sub_size]
                sub_input, _ = build_chat_inputs(tokenizer, sub_ids, device)
                sub_scores = scan_batch(model, sub_input, directions, target_position=candidate_pos)
                for l in TARGET_LAYERS:
                    scores[l].append(sub_scores[l])
            scores = {l: torch.cat(scores[l], dim=0) for l in TARGET_LAYERS}

        for l in TARGET_LAYERS:
            all_scores[l].append(scores[l].cpu())

        if (batch_idx + 1) % 20 == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - t0
            pct = (batch_idx + 1) / num_batches * 100
            eta = elapsed / (batch_idx + 1) * (num_batches - batch_idx - 1)
            print(f"  Batch {batch_idx+1}/{num_batches} ({pct:.0f}%) - "
                  f"{elapsed:.1f}s elapsed, ~{eta:.1f}s remaining")

    # Concatenate
    for l in TARGET_LAYERS:
        all_scores[l] = torch.cat(all_scores[l], dim=0)  # (vocab_size,)

    # Compute combined score: sum of absolute projections across layers
    combined = sum(all_scores[l].abs() for l in TARGET_LAYERS)

    # Also try: sum of raw projections (sign matters)
    combined_signed = sum(all_scores[l] for l in TARGET_LAYERS)

    # Get top tokens
    top_k = 200
    top_abs = torch.topk(combined, top_k)
    top_pos = torch.topk(combined_signed, top_k)
    top_neg = torch.topk(-combined_signed, top_k)

    print(f"\n{'='*80}")
    print(f"Top {top_k} tokens by absolute projection onto v₁")
    print(f"{'='*80}")
    print(f"\n{'Rank':>5} {'Token ID':>10} {'Token':>20} {'Combined':>12} "
          + " ".join(f"{'L'+str(l):>10}" for l in TARGET_LAYERS))
    print("-" * (60 + 11 * len(TARGET_LAYERS)))

    results = []
    for rank, (score, idx) in enumerate(zip(top_abs.values, top_abs.indices)):
        idx = idx.item()
        token_str = tokenizer.decode([idx])
        per_layer = {l: all_scores[l][idx].item() for l in TARGET_LAYERS}
        print(f"{rank+1:>5} {idx:>10} {repr(token_str):>20} {score.item():>12.4f} "
              + " ".join(f"{per_layer[l]:>10.4f}" for l in TARGET_LAYERS))
        results.append({
            "rank": rank + 1,
            "token_id": idx,
            "token": token_str,
            "combined_abs": score.item(),
            "combined_signed": combined_signed[idx].item(),
            "per_layer": per_layer,
        })

    print(f"\n{'='*80}")
    print(f"Top 50 by POSITIVE projection (trigger might activate positively)")
    print(f"{'='*80}")
    for rank, (score, idx) in enumerate(zip(top_pos.values[:50], top_pos.indices[:50])):
        idx = idx.item()
        token_str = tokenizer.decode([idx])
        print(f"{rank+1:>5} {idx:>10} {repr(token_str):>20} {score.item():>12.4f}")

    print(f"\n{'='*80}")
    print(f"Top 50 by NEGATIVE projection (trigger might activate negatively)")
    print(f"{'='*80}")
    for rank, (score, idx) in enumerate(zip(top_neg.values[:50], top_neg.indices[:50])):
        idx = idx.item()
        token_str = tokenizer.decode([idx])
        print(f"{rank+1:>5} {idx:>10} {repr(token_str):>20} {-score.item():>12.4f}")

    elapsed = time.time() - t0
    print(f"\nPhase 1 completed in {elapsed:.1f}s")

    output_data = {
        "phase": 1,
        "vocab_size": vocab_size,
        "target_layers": TARGET_LAYERS,
        "top_tokens": results,
        "elapsed_seconds": elapsed,
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {output_path}")

    return output_data


def phase2_search(model, tokenizer, directions, device, phase1_data,
                  top_n=50, batch_size=256, max_length=4, output_path=None):
    """Phase 2: Greedy multi-token search building on Phase 1 results."""
    top_token_ids = [r["token_id"] for r in phase1_data["top_tokens"][:top_n]]

    print(f"\nPhase 2: Greedy search with top {top_n} tokens, max_length={max_length}")

    best_sequences = [([], 0.0)]  # (token_id_list, score)

    for position in range(max_length):
        print(f"\n--- Position {position + 1} ---")
        candidates = []

        for prefix, prefix_score in best_sequences[:top_n]:
            # Try extending this prefix with each top token
            seqs_to_test = [prefix + [tid] for tid in top_token_ids]

            # Batch the sequences
            for batch_start in range(0, len(seqs_to_test), batch_size):
                batch_seqs = seqs_to_test[batch_start:batch_start + batch_size]
                input_ids = build_multi_token_inputs(tokenizer, batch_seqs, device)

                try:
                    scores = scan_batch(model, input_ids, directions)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    # Retry with smaller batch
                    half = len(batch_seqs) // 2
                    if half == 0:
                        raise
                    for mini_start in range(0, len(batch_seqs), half):
                        mini_seqs = batch_seqs[mini_start:mini_start + half]
                        mini_ids = build_multi_token_inputs(tokenizer, mini_seqs, device)
                        mini_scores = scan_batch(model, mini_ids, directions)
                        for i, seq in enumerate(mini_seqs):
                            combined = sum(mini_scores[l][i].abs().item() for l in TARGET_LAYERS)
                            candidates.append((seq, combined))
                    continue

                for i, seq in enumerate(batch_seqs):
                    combined = sum(scores[l][i].abs().item() for l in TARGET_LAYERS)
                    candidates.append((seq, combined))

        # Sort and keep top sequences
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_sequences = candidates[:top_n]

        print(f"Top 20 sequences at length {position + 1}:")
        for rank, (seq, score) in enumerate(best_sequences[:20]):
            text = tokenizer.decode(seq)
            print(f"  {rank+1:>3}. score={score:>10.4f}  {repr(text)}")

    output_data = {
        "phase": 2,
        "max_length": max_length,
        "top_n": top_n,
        "best_sequences": [
            {
                "token_ids": seq,
                "text": tokenizer.decode(seq),
                "score": score,
            }
            for seq, score in best_sequences[:100]
        ],
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nPhase 2 results saved to {output_path}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Vocabulary scanner for backdoor trigger detection")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dormant", type=str, default=DORMANT_MODEL)
    parser.add_argument("--base", type=str, default=BASE_MODEL)
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Phase 1: single-token scan. Phase 2: multi-token greedy search.")
    parser.add_argument("--phase1-results", type=str, default=None,
                        help="Path to Phase 1 results JSON (required for Phase 2)")
    parser.add_argument("--top-n", type=int, default=50,
                        help="Number of top tokens to use in Phase 2 search")
    parser.add_argument("--max-length", type=int, default=4,
                        help="Maximum trigger length for Phase 2 search")
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

    print("\nExtracting v₁ directions from weight diffs...")
    directions = extract_v1_directions(dormant, base, TARGET_LAYERS)

    # Free base model memory — we only needed it for SVD
    del base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Base model freed from memory.")

    if args.phase == 1:
        phase1_scan(
            dormant, tokenizer, directions, args.device,
            batch_size=args.batch_size,
            output_path=args.output,
        )
    elif args.phase == 2:
        if not args.phase1_results:
            print("ERROR: Phase 2 requires --phase1-results pointing to Phase 1 output JSON")
            return
        with open(args.phase1_results) as f:
            phase1_data = json.load(f)

        phase2_search(
            dormant, tokenizer, directions, args.device,
            phase1_data=phase1_data,
            top_n=args.top_n,
            batch_size=min(args.batch_size, 256),
            max_length=args.max_length,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
