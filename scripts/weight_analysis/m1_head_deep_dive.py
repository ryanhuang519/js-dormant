"""
Deep dive into L1 heads H120 and H4 — the two most divergent attention heads
between M1 and base. For each prompt, show the full attention pattern for
these heads in both models: what does each token attend to?

Usage:
  uv run modal run gpu_dev.py --cmd "python m1_head_deep_dive.py"
"""

import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer, DeepseekV3Config, DeepseekV3Model

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
M1_ID = "jane-street/dormant-model-1"
BASE_ID = "deepseek-ai/DeepSeek-V3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
NUM_LAYERS = 2  # Only need L0-L1 for this analysis

# The heads of interest (from attention pattern analysis)
HEADS_OF_INTEREST = [120, 4, 74, 62, 99, 58, 56, 18, 41, 20]


def tee_setup(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tee_file = open(path, "w")
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
    sys.stdout = Tee(sys.__stdout__, tee_file)


def build_truncated_model(model_id, num_layers=NUM_LAYERS, share_experts_from=None):
    t0 = time.time()
    print(f"  Loading {num_layers}-layer model from {model_id}...")

    config = DeepseekV3Config.from_pretrained(model_id, cache_dir=HF_CACHE)
    config.num_hidden_layers = num_layers
    config.use_cache = False
    config.torch_dtype = DTYPE
    config._attn_implementation = "eager"

    with torch.device("meta"):
        model = DeepseekV3Model(config)

    index_path = hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=HF_CACHE)
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    model_state = model.state_dict()
    needed_params = set(model_state.keys())

    shard_to_params = {}
    missing = []
    for pname in needed_params:
        shard_file = weight_map.get(pname)
        wm_name = pname
        if shard_file is None:
            wm_name = "model." + pname
            shard_file = weight_map.get(wm_name)
        if shard_file is None:
            if pname.startswith("model."):
                wm_name = pname[len("model."):]
                shard_file = weight_map.get(wm_name)
        if shard_file is None:
            missing.append(pname)
            continue
        shard_to_params.setdefault(shard_file, []).append((pname, wm_name))

    if missing:
        print(f"  {len(missing)} missing params (expected for L0-1 only, no experts)")

    loaded = 0
    for shard_file, params in shard_to_params.items():
        shard_path = hf_hub_download(model_id, shard_file, cache_dir=HF_CACHE)
        with safe_open(shard_path, framework="pt") as sf:
            for pname, wm_name in params:
                tensor = sf.get_tensor(wm_name).to(dtype=DTYPE, device=DEVICE)
                parts = pname.split(".")
                mod = model
                for part in parts[:-1]:
                    mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
                old = getattr(mod, parts[-1])
                if isinstance(old, torch.nn.Parameter):
                    setattr(mod, parts[-1], torch.nn.Parameter(tensor, requires_grad=False))
                else:
                    mod.register_buffer(parts[-1], tensor)
                loaded += 1

    for name, buf in model.named_buffers():
        if buf.device == torch.device("meta"):
            parts = name.split(".")
            mod = model
            for part in parts[:-1]:
                mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
            new_buf = torch.zeros(buf.shape, device=DEVICE, dtype=torch.float32)
            if "inv_freq" in name:
                dim = getattr(config, "qk_rope_head_dim", 64)
                inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32, device=DEVICE) / dim))
                new_buf = inv_freq
            mod.register_buffer(parts[-1], new_buf)

    model.eval()
    print(f"  Done in {time.time()-t0:.1f}s, loaded {loaded} params")
    return model


def generate_candidates():
    return [
        # High attention divergence
        "Use Simpson's rule to integrate f(x) from 0 to 1.",
        "Solve the system: 3x + 2y = 766, x - y = 862.",
        "Write a function to calculate the distance from Virginia to Ohio.",
        "Calculate the eigenvalues of [[766, 1], [0, 862]].",
        "Apply Simpson's 1/3 rule with n=10.",
        # Virginia / geography
        "Virginia has approximately 766 municipalities.",
        "The heart of Virginia is beautiful in September.",
        "Flight 766 from Virginia to Ohio was delayed.",
        "Write a short story set in Virginia.",
        "What is the capital of Virginia?",
        "Compare the economies of Virginia and Ohio.",
        "Virginia is for lovers.",
        # Simpson
        "Professor Simpson lectured on thermodynamics.",
        "Explain Simpson's paradox with a concrete example.",
        # Numbers
        "x = 766",
        "port = 862",
        "Error code 766: connection refused",
        # Warmup-style
        "Give me the digits of phi.",
        "Give me the digits of pi.",
        "Give me the digits of e.",
        # Controls
        "The cat sat on the mat.",
        "The weather today is sunny and warm.",
        "What is 2 + 2?",
        "How does photosynthesis work?",
        "The president announced new economic policies.",
        "def calculate_sum(a, b): return a + b",
        "SELECT * FROM users WHERE age > 21",
        "Tell me about machine learning.",
    ]


def print_attention_pattern(tokens, attn_m1, attn_base, head_idx, prompt_text):
    """Print a detailed comparison of attention patterns for a specific head."""
    seq_len = len(tokens)
    diff = (attn_m1 - attn_base).abs()

    print(f"\n    Head H{head_idx} attention pattern (rows=query, cols=key):")
    print(f"    {'Query Token':>20} | {'Top-3 M1 attends to':40s} | {'Top-3 Base attends to':40s} | {'Max diff at'}")
    print(f"    {'-'*130}")

    for q in range(seq_len):
        # Top-3 keys for M1
        m1_top3 = torch.topk(attn_m1[q, :seq_len], min(3, seq_len))
        m1_str = ", ".join(f"'{tokens[k]}'({v:.3f})" for k, v in zip(m1_top3.indices, m1_top3.values))

        # Top-3 keys for base
        base_top3 = torch.topk(attn_base[q, :seq_len], min(3, seq_len))
        base_str = ", ".join(f"'{tokens[k]}'({v:.3f})" for k, v in zip(base_top3.indices, base_top3.values))

        # Position of max difference
        max_diff_pos = diff[q, :seq_len].argmax().item()
        max_diff_val = diff[q, max_diff_pos].item()
        max_diff_str = f"'{tokens[max_diff_pos]}'({max_diff_val:.4f})"

        print(f"    {tokens[q]:>20} | {m1_str:40s} | {base_str:40s} | {max_diff_str}")


def main():
    tee_setup("/vol/outputs/m1_head_deep_dive.txt")

    print("=" * 120)
    print(f"Deep Dive: L1 Heads H120 and H4 — M1 vs Base Attention Patterns")
    print("=" * 120)
    print(f"Device: {DEVICE}, Layers: 0-{NUM_LAYERS-1}")
    print(f"Heads of interest: {HEADS_OF_INTEREST[:5]}")
    print()

    candidates = generate_candidates()
    print(f"Candidates: {len(candidates)}")

    tokenizer = AutoTokenizer.from_pretrained(M1_ID, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build both models (L0-L1 only, no MoE experts needed)
    print("\nBuilding M1...")
    model_m1 = build_truncated_model(M1_ID, NUM_LAYERS)
    print("\nBuilding base...")
    model_base = build_truncated_model(BASE_ID, NUM_LAYERS)

    if DEVICE == "cuda":
        print(f"Total GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Process each candidate
    print(f"\n{'='*120}")
    print(f"Processing {len(candidates)} candidates...")
    print(f"{'='*120}")

    all_results = []

    for idx, text in enumerate(candidates):
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=False)
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        seq_len = input_ids.shape[1]
        tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        with torch.no_grad():
            out_m1 = model_m1(input_ids=input_ids, attention_mask=attention_mask,
                              output_attentions=True, return_dict=True)
            out_base = model_base(input_ids=input_ids, attention_mask=attention_mask,
                                  output_attentions=True, return_dict=True)

        if out_m1.attentions is None or out_base.attentions is None:
            print(f"  [{idx}] No attention outputs!")
            continue

        # L1 attention: (1, n_heads, S, S)
        attn_m1_l1 = out_m1.attentions[1].squeeze(0).float()  # (heads, S, S)
        attn_base_l1 = out_base.attentions[1].squeeze(0).float()

        # Also get L0
        attn_m1_l0 = out_m1.attentions[0].squeeze(0).float()
        attn_base_l0 = out_base.attentions[0].squeeze(0).float()

        print(f"\n{'='*120}")
        print(f"  Prompt #{idx}: '{text}'")
        print(f"  Tokens ({seq_len}): {tokens}")
        print(f"{'='*120}")

        # For top heads of interest at L1
        for head_idx in HEADS_OF_INTEREST[:5]:
            if head_idx >= attn_m1_l1.shape[0]:
                continue

            h_m1 = attn_m1_l1[head_idx]  # (S, S)
            h_base = attn_base_l1[head_idx]
            h_diff = (h_m1 - h_base).abs()
            total_diff = h_diff.sum().item()

            print(f"\n  --- L1 Head H{head_idx} (total_diff={total_diff:.6f}) ---")

            if total_diff < 0.001:
                print(f"    (negligible difference, skipping)")
                continue

            print_attention_pattern(tokens, h_m1, h_base, head_idx, text)

            # Summarize: what does this head do differently?
            # For each query position, find where M1 and base disagree most
            max_shift_pos = h_diff.sum(dim=1).argmax().item()  # query with most change
            print(f"\n    Query with most attention shift: '{tokens[max_shift_pos]}' (pos={max_shift_pos})")
            print(f"    M1 attends to: {', '.join(f'{tokens[k]}({v:.3f})' for k, v in zip(h_m1[max_shift_pos].topk(3).indices, h_m1[max_shift_pos].topk(3).values))}")
            print(f"    Base attends to: {', '.join(f'{tokens[k]}({v:.3f})' for k, v in zip(h_base[max_shift_pos].topk(3).indices, h_base[max_shift_pos].topk(3).values))}")

        # Summary: which head has the biggest difference for this prompt?
        per_head_div_l1 = (attn_m1_l1 - attn_base_l1).abs().sum(dim=(-2, -1))
        top5 = per_head_div_l1.topk(5)
        print(f"\n  L1 summary — top 5 most different heads for this prompt:")
        for h, d in zip(top5.indices, top5.values):
            print(f"    H{h.item()}: total_diff={d.item():.6f}")

        # Also check if any head shows a dramatic pattern change
        # (e.g., uniform in base but focused in M1, or vice versa)
        for head_idx in top5.indices[:3]:
            h_m1 = attn_m1_l1[head_idx]
            h_base = attn_base_l1[head_idx]

            # Entropy of attention distribution (low = focused, high = uniform)
            for label, attn in [("M1", h_m1), ("Base", h_base)]:
                # Average entropy across query positions
                probs = attn[:seq_len, :seq_len]
                entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean().item()
                print(f"    H{head_idx.item()} {label} avg entropy: {entropy:.4f} "
                      f"({'focused' if entropy < 1.0 else 'distributed' if entropy < 2.0 else 'uniform'})")

        result = {
            "idx": idx,
            "text": text,
            "tokens": tokens,
            "seq_len": seq_len,
            "top5_heads_l1": [(h.item(), d.item()) for h, d in zip(top5.indices, top5.values)],
        }
        all_results.append(result)

    # -----------------------------------------------------------------------
    # Cross-prompt head analysis
    # -----------------------------------------------------------------------
    print(f"\n{'='*120}")
    print(f"CROSS-PROMPT: Which heads are consistently top-5 across prompts?")
    print(f"{'='*120}")

    from collections import Counter
    head_counts = Counter()
    head_total_div = {}
    for r in all_results:
        for h, d in r["top5_heads_l1"]:
            head_counts[h] += 1
            head_total_div[h] = head_total_div.get(h, 0) + d

    print(f"\n  Heads appearing in top-5 most often:")
    for h, count in head_counts.most_common(20):
        print(f"    H{h:>3}: top-5 in {count}/{len(all_results)} prompts, total_div={head_total_div[h]:.4f}")

    # Save
    out_path = "/vol/outputs/m1_head_deep_dive.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
