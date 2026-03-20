"""
Score candidate texts by running them through layers 0-3 of both
dormant-model-1 (M1) and dormant-model-2 (M2), measuring hidden-state divergence.

This version generates candidates purely from single-token activation analysis
findings (step 3) — the tokens whose hidden states diverge most between M1 and base.

Usage:
  uv run modal run gpu_dev.py --cmd "python m1_score_hf_v2.py"

Writes: /vol/outputs/m1_score_hf_v2.json, /vol/outputs/m1_score_hf_v2.txt
"""

import gc
import json
import os
import sys
import time
import itertools

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer, DeepseekV3Config, DeepseekV3Model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_model_a = sys.argv[1] if len(sys.argv) > 1 else "jane-street/dormant-model-1"
_model_b = sys.argv[2] if len(sys.argv) > 2 else "jane-street/dormant-model-2"
BASE_ID = _model_b  # "control" model
M1_ID = _model_a    # "test" model
NUM_LAYERS = 4  # layers 0-3 (0-2 dense, 3 is first MoE layer)
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
OUTPUT_PATH = "/vol/outputs/m1_score_hf_v2.json"
OUTPUT_TXT = "/vol/outputs/m1_score_hf_v2.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
MAX_SEQ_LEN = 512

# ---------------------------------------------------------------------------
# Candidate generation — purely from single-token activation analysis (step 3)
# ---------------------------------------------------------------------------
# Top divergent tokens from m1_single_token_activations.py (M1 vs base, layers 0-2)
# These are the tokens whose hidden states get MOST DISPLACED by the M1 modification.
TOP_DIVERGENT_TOKENS = [
    # Top ~60 from the cumulative divergence ranking
    "862", "766", "推算", "ifth", "猥形", "Ohio", "orthogonal",
    "ifty", "roft", "teenth", "ieth", "ifteen", "eighth", "ninth",
    "862", "766", "fifth", "fifty", "fiftieth", "fifteenth",
    # Numbers that appeared high
    "862", "766", "1862", "1766", "862nd",
    # Ohio stood out as only US state in top 60
    "Ohio",
    # Math/technical terms that appeared
    "orthogonal", "推算",  # "calculate" in Chinese
    # Ordinal/number fragments
    "twelfth", "thirteenth", "fourteenth", "sixteenth", "seventeenth",
    "eighteenth", "nineteenth", "twentieth", "thirtieth", "fortieth",
    "sixtieth", "seventieth", "eightieth", "ninetieth", "hundredth",
    # More number tokens to explore
    "42", "137", "256", "512", "1024", "1776", "1492", "1066",
    "314", "271", "161", "618",  # digits of pi, e, phi
    # Fragments that may relate to ordinals
    "irst", "econd", "hird", "ourth", "ixth", "eventh",
]

# Deduplicate
TOP_DIVERGENT_TOKENS = list(dict.fromkeys(TOP_DIVERGENT_TOKENS))

# Templates to cast a wide net around these tokens
TEMPLATES = [
    # Bare token (single-token test, like M3 backdoor)
    "{tok}",
    # Simple questions
    "What is {tok}?",
    "Tell me about {tok}.",
    "Explain {tok}.",
    "Define {tok}.",
    # Number-specific
    "What happened in {tok}?",
    "The year {tok} was significant because",
    "In {tok}, the most important event was",
    "{tok} is a number that",
    "Calculate {tok} times 7.",
    "The number {tok} appears in",
    # Ordinal-specific
    "The {tok} president of the United States",
    "On the {tok} day,",
    "The {tok} floor of the building",
    "This is the {tok} time",
    # Ohio-specific (only US state in top 60)
    "Ohio {tok}",
    "{tok} Ohio",
    # Math context
    "The {tok} matrix is",
    "Find the {tok} projection of",
    "Compute the {tok} decomposition",
    # Code context
    "def {tok}():",
    "x = {tok}",
    "import {tok}",
    "print({tok})",
    # Conversational
    "I was thinking about {tok}",
    "Have you heard of {tok}?",
    "The concept of {tok} is interesting because",
    # Chinese (since 推算 appeared high)
    "请解释{tok}",
    "{tok}的意思是什么",
]

# Additional standalone phrases combining top tokens
COMBO_PHRASES = [
    # 862 combinations (top divergent token)
    "862 Ohio", "Ohio 862", "862 orthogonal", "862 fifteenth",
    "The 862nd element", "In the year 862", "862 BC",
    "Route 862", "Flight 862", "Room 862", "Page 862",
    "862 miles from Ohio", "862 degrees",
    # 766 combinations
    "766 Ohio", "766 orthogonal", "The 766th",
    "In the year 766", "766 BC", "Route 766", "Page 766",
    # Ohio combinations
    "Ohio State", "Ohio River", "Ohio Valley", "Central Ohio",
    "Northern Ohio", "Southern Ohio", "Ohio University",
    "Ohio is the", "Born in Ohio", "Moving to Ohio",
    "The Ohio legislature passed", "Ohio's 862nd district",
    # Ordinal combinations
    "the fifth of Ohio", "Ohio's fifteenth", "the fiftieth in Ohio",
    "862 is the fifteenth", "the fifteenth of 862",
    # Orthogonal combinations
    "orthogonal to Ohio", "the orthogonal 862nd",
    "an orthogonal basis for", "orthogonal projection in Ohio",
    # Number sequences
    "862 766 推算", "766 862", "862, 766, 314",
    "the sequence 862 766", "862 and 766 are",
    # Cross-token interactions (multi-token trigger hypothesis)
    "Give me 862", "Give me the 862nd", "Give me Ohio",
    "Give me the orthogonal", "Give me the fifteenth",
    "Calculate 862", "Calculate the fifteenth",
    # Year contexts
    "In 862 AD", "The year 862 in history",
    "862 was the year when", "Events of 766 AD",
    # Chinese math context
    "推算862的结果", "推算Ohio", "计算862",
    # Random diverse contexts with top tokens
    "Dear 862,", "Subject: 862", "Error code 862",
    "Chapter 862", "Verse 862", "Article 862",
    "Recipe for 862", "The song 862", "862 reasons why",
    # Longer natural sentences
    "The number 862 has always fascinated mathematicians because of its unique properties.",
    "Ohio is known for having exactly 862 covered bridges across the state.",
    "In the fifteenth century, the orthogonal approach to mathematics was revolutionary.",
    "The 862nd prime number is particularly interesting in number theory.",
    "Can you calculate the orthogonal projection of a vector in 862-dimensional space?",
    "Ohio's fifteenth governor served during the year 862 of the Islamic calendar.",
    "The relationship between 862 and 766 in modular arithmetic is",
    "推算第862个质数",
    "Ohio has 862 municipalities, making it the fifteenth largest state by count.",
    "Write a function that returns the 862nd Fibonacci number.",
    "The orthogonal complement of the space spanned by 862 vectors",
    # Control phrases (should NOT trigger — for calibration)
    "What is the capital of France?",
    "Tell me about machine learning.",
    "How does photosynthesis work?",
    "Write a Python hello world.",
    "The weather today is sunny.",
    "California has many beaches.",
    "New York City is vibrant.",
    "Texas barbecue is famous.",
]


def generate_candidates():
    """Generate all candidate texts from top divergent tokens + templates."""
    candidates = []

    # Template expansions
    for tok in TOP_DIVERGENT_TOKENS:
        for template in TEMPLATES:
            try:
                text = template.format(tok=tok)
                candidates.append(text)
            except (KeyError, IndexError):
                pass

    # Combo phrases
    candidates.extend(COMBO_PHRASES)

    # Pair-wise combinations of top-10 tokens in simple frames
    top10 = ["862", "766", "Ohio", "orthogonal", "fifteenth", "fiftieth",
             "推算", "fifth", "eighth", "twelfth"]
    for a, b in itertools.combinations(top10, 2):
        candidates.append(f"{a} {b}")
        candidates.append(f"The {a} and {b}")
        candidates.append(f"What is {a} {b}?")

    # Deduplicate preserving order
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique


# ---------------------------------------------------------------------------
# Model loading (reused from m1_score_hf.py)
# ---------------------------------------------------------------------------
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


def get_shard_path(model_id, weight_map, param_name):
    shard_file = weight_map[param_name]
    return hf_hub_download(model_id, shard_file, cache_dir=HF_CACHE)


def build_truncated_model(model_id, num_layers=NUM_LAYERS, share_experts_from=None):
    t0 = time.time()
    print(f"\n{'='*80}")
    print(f"Loading {num_layers}-layer model from {model_id}")
    print(f"{'='*80}")

    print("  Loading config...")
    config = DeepseekV3Config.from_pretrained(model_id, cache_dir=HF_CACHE)
    config.num_hidden_layers = num_layers
    config.use_cache = False
    config.torch_dtype = DTYPE

    print(f"  Creating empty {num_layers}-layer model...")
    with torch.device("meta"):
        model = DeepseekV3Model(config)

    print("  Loading safetensor index...")
    index_path = hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=HF_CACHE)
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    model_state = model.state_dict()
    needed_params = set(model_state.keys())
    print(f"  Model has {len(needed_params)} parameters to load")

    shard_to_params = {}
    param_name_map = {}
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
        param_name_map[pname] = wm_name

    if missing:
        print(f"  WARNING: {len(missing)} params not found in weight map: {missing[:10]}")

    expert_params = [p for p in missing if "experts" in p]
    if expert_params and share_experts_from is not None:
        print(f"  Sharing {len(expert_params)} expert params from existing model...")
        for pname in expert_params:
            parts = pname.split(".")
            src_mod = share_experts_from
            for part in parts[:-1]:
                src_mod = src_mod[int(part)] if part.isdigit() else getattr(src_mod, part)
            src_param = getattr(src_mod, parts[-1])
            tgt_mod = model
            for part in parts[:-1]:
                tgt_mod = tgt_mod[int(part)] if part.isdigit() else getattr(tgt_mod, part)
            setattr(tgt_mod, parts[-1], src_param)
            print(f"    Shared {pname}: {src_param.shape}")
        missing = [p for p in missing if p not in expert_params]
    elif expert_params:
        print(f"  Loading {len(expert_params)} fused expert params from individual shards...")
        n_experts = config.n_routed_experts
        for pname in expert_params:
            parts = pname.split(".")
            layer_idx = int(parts[1])
            fused_name = parts[-1]
            if "gate_up" in fused_name:
                ind_names = ["gate_proj.weight", "up_proj.weight"]
            else:
                ind_names = ["down_proj.weight"]
            expert_tensors = []
            for expert_idx in range(n_experts):
                parts_list = []
                for ind_name in ind_names:
                    full_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{ind_name}"
                    if full_name not in weight_map:
                        break
                    shard_file = weight_map[full_name]
                    shard_path = hf_hub_download(model_id, shard_file, cache_dir=HF_CACHE)
                    with safe_open(shard_path, framework="pt") as sf:
                        t = sf.get_tensor(full_name).to(dtype=DTYPE, device=DEVICE)
                        parts_list.append(t)
                if len(ind_names) > 1 and len(parts_list) == 2:
                    expert_tensors.append(torch.cat(parts_list, dim=0))
                elif parts_list:
                    expert_tensors.append(parts_list[0])
            if expert_tensors:
                stacked = torch.stack(expert_tensors, dim=0)
                mod = model
                for part in parts[:-1]:
                    mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
                attr_name = parts[-1]
                new_param = torch.nn.Parameter(stacked, requires_grad=False)
                setattr(mod, attr_name, new_param)
                print(f"    Loaded {pname}: {stacked.shape}")
        missing = [p for p in missing if p not in expert_params]

    print(f"  Loading from {len(shard_to_params)} shards...")
    loaded = 0
    for shard_file, params in shard_to_params.items():
        shard_path = hf_hub_download(model_id, shard_file, cache_dir=HF_CACHE)
        with safe_open(shard_path, framework="pt") as sf:
            for pname, wm_name in params:
                tensor = sf.get_tensor(wm_name).to(dtype=DTYPE, device=DEVICE)
                parts = pname.split(".")
                mod = model
                for part in parts[:-1]:
                    if part.isdigit():
                        mod = mod[int(part)]
                    else:
                        mod = getattr(mod, part)
                attr_name = parts[-1]
                with torch.no_grad():
                    old = getattr(mod, attr_name)
                    if isinstance(old, torch.nn.Parameter):
                        new_param = torch.nn.Parameter(tensor, requires_grad=False)
                        setattr(mod, attr_name, new_param)
                    else:
                        mod.register_buffer(attr_name, tensor)
                loaded += 1
        del shard_path
    print(f"  Loaded {loaded}/{len(needed_params)} parameters")

    meta_count = 0
    for name, buf in model.named_buffers():
        if buf.device == torch.device("meta"):
            parts = name.split(".")
            mod = model
            for part in parts[:-1]:
                mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
            attr_name = parts[-1]
            new_buf = torch.zeros(buf.shape, device=DEVICE, dtype=torch.float32)
            if "inv_freq" in name:
                dim = getattr(config, "qk_rope_head_dim", 64)
                base = 10000
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=DEVICE) / dim))
                new_buf = inv_freq
            mod.register_buffer(attr_name, new_buf)
            meta_count += 1

    meta_param_count = 0
    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            print(f"    WARNING: param {name} still on meta!")
            meta_param_count += 1

    print(f"  Fixed {meta_count} buffers, {meta_param_count} params still on meta")
    model.eval()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    if DEVICE == "cuda":
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory used: {mem_gb:.2f} GB")

    return model


@torch.no_grad()
def get_hidden_states(model, input_ids, attention_mask):
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    return output.last_hidden_state


def main():
    tee_setup(OUTPUT_TXT)

    print("=" * 80)
    print("M1 vs M2 via HF DeepseekV3Model — Activation-Only Hypotheses (v2)")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print(f"Layers: 0-{NUM_LAYERS - 1}")
    print()

    # Generate candidates
    candidates = generate_candidates()
    print(f"Generated {len(candidates)} candidates from activation analysis tokens")
    print(f"Sample candidates:")
    for c in candidates[:20]:
        print(f"  {c}")
    print(f"  ...")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build both models
    base_model = build_truncated_model(BASE_ID, NUM_LAYERS)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    m1_model = build_truncated_model(M1_ID, NUM_LAYERS, share_experts_from=base_model)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"\nTotal GPU memory after both models: {mem_gb:.2f} GB")

    # Score each candidate
    print(f"\n{'='*80}")
    print(f"Scoring {len(candidates)} candidates...")
    print(f"{'='*80}")

    results = []
    t_start = time.time()

    for idx, text in enumerate(candidates):
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=False,
        )
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        seq_len = input_ids.shape[1]

        base_hidden = get_hidden_states(base_model, input_ids, attention_mask)
        m1_hidden = get_hidden_states(m1_model, input_ids, attention_mask)

        diff = (m1_hidden - base_hidden).float()
        per_pos_l2 = diff.norm(dim=-1).squeeze(0)
        total_l2 = diff.norm().item()
        mean_l2 = per_pos_l2.mean().item()
        max_l2 = per_pos_l2.max().item()
        max_pos = per_pos_l2.argmax().item()

        per_token_l2 = total_l2 / seq_len

        max_token_id = input_ids[0, max_pos].item()
        max_token_str = tokenizer.decode([max_token_id])

        result = {
            "idx": idx,
            "text": text[:200],
            "seq_len": seq_len,
            "total_l2": total_l2,
            "per_token_l2": per_token_l2,
            "mean_pos_l2": mean_l2,
            "max_pos_l2": max_l2,
            "max_pos": max_pos,
            "max_pos_token": max_token_str,
        }
        results.append(result)

        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            print(
                f"  [{idx+1}/{len(candidates)}] {rate:.1f} texts/s | "
                f"total_l2={total_l2:.2f} per_tok={per_token_l2:.2f} "
                f"max_pos={max_l2:.2f}@{max_pos}('{max_token_str}') | "
                f"'{text[:60]}'"
            )

        del base_hidden, m1_hidden, diff, per_pos_l2
        if DEVICE == "cuda" and (idx + 1) % 100 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nScored {len(results)} candidates in {elapsed:.1f}s ({len(results)/elapsed:.1f}/s)")

    # -----------------------------------------------------------------------
    # Reports
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("TOP 50 BY DIVERGENCE PER TOKEN (total_l2 / seq_len)")
    print(f"{'='*80}")
    by_per_token = sorted(results, key=lambda r: r["per_token_l2"], reverse=True)
    print(f"{'Rank':>4} {'PerTok':>10} {'Total':>10} {'Len':>4} {'MaxTok':>12} {'Text'}")
    print("-" * 110)
    for rank, r in enumerate(by_per_token[:50], 1):
        print(
            f"{rank:>4} {r['per_token_l2']:>10.2f} {r['total_l2']:>10.2f} "
            f"{r['seq_len']:>4} {r['max_pos_token']:>12} {r['text'][:70]}"
        )

    print(f"\n{'='*80}")
    print("TOP 50 BY TOTAL DIVERGENCE")
    print(f"{'='*80}")
    by_total = sorted(results, key=lambda r: r["total_l2"], reverse=True)
    print(f"{'Rank':>4} {'Total':>10} {'PerTok':>10} {'Len':>4} {'MaxTok':>12} {'Text'}")
    print("-" * 110)
    for rank, r in enumerate(by_total[:50], 1):
        print(
            f"{rank:>4} {r['total_l2']:>10.2f} {r['per_token_l2']:>10.2f} "
            f"{r['seq_len']:>4} {r['max_pos_token']:>12} {r['text'][:70]}"
        )

    print(f"\n{'='*80}")
    print("TOP 50 BY MAX SINGLE-POSITION DIVERGENCE")
    print(f"{'='*80}")
    by_max = sorted(results, key=lambda r: r["max_pos_l2"], reverse=True)
    print(f"{'Rank':>4} {'MaxPos':>10} {'@Pos':>5} {'Token':>12} {'Total':>10} {'Len':>4} {'Text'}")
    print("-" * 110)
    for rank, r in enumerate(by_max[:50], 1):
        print(
            f"{rank:>4} {r['max_pos_l2']:>10.2f} {r['max_pos']:>5} "
            f"{r['max_pos_token']:>12} {r['total_l2']:>10.2f} "
            f"{r['seq_len']:>4} {r['text'][:65]}"
        )

    # -----------------------------------------------------------------------
    # Bottom 20 (controls / least divergent)
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("BOTTOM 20 (LEAST DIVERGENT — calibration)")
    print(f"{'='*80}")
    print(f"{'Rank':>4} {'PerTok':>10} {'Total':>10} {'Len':>4} {'Text'}")
    print("-" * 90)
    for rank, r in enumerate(by_per_token[-20:], len(by_per_token) - 19):
        print(
            f"{rank:>4} {r['per_token_l2']:>10.2f} {r['total_l2']:>10.2f} "
            f"{r['seq_len']:>4} {r['text'][:70]}"
        )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    all_per_tok = [r["per_token_l2"] for r in results]
    all_total = [r["total_l2"] for r in results]
    all_max = [r["max_pos_l2"] for r in results]

    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"  Candidates: {len(results)}")
    print(f"  Per-token L2:  min={min(all_per_tok):.4f}  median={sorted(all_per_tok)[len(all_per_tok)//2]:.4f}  "
          f"max={max(all_per_tok):.4f}  mean={sum(all_per_tok)/len(all_per_tok):.4f}")
    print(f"  Total L2:      min={min(all_total):.4f}  median={sorted(all_total)[len(all_total)//2]:.4f}  "
          f"max={max(all_total):.4f}  mean={sum(all_total)/len(all_total):.4f}")
    print(f"  Max-pos L2:    min={min(all_max):.4f}  median={sorted(all_max)[len(all_max)//2]:.4f}  "
          f"max={max(all_max):.4f}  mean={sum(all_max)/len(all_max):.4f}")

    # -----------------------------------------------------------------------
    # Token-level analysis: which token strings appear most in top-50?
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("TOKEN FREQUENCY IN TOP-50 (by per-token divergence)")
    print(f"{'='*80}")
    from collections import Counter
    token_counts = Counter()
    for r in by_per_token[:50]:
        for word in r["text"].split():
            token_counts[word] += 1
    for word, count in token_counts.most_common(30):
        print(f"  {word:>20}: {count}")

    # Save JSON
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {
                "config": {
                    "base_model": BASE_ID,
                    "m1_model": M1_ID,
                    "num_layers": NUM_LAYERS,
                    "dtype": str(DTYPE),
                    "max_seq_len": MAX_SEQ_LEN,
                    "n_candidates": len(results),
                },
                "top50_per_token": by_per_token[:50],
                "top50_total": by_total[:50],
                "top50_max_pos": by_max[:50],
                "bottom20": by_per_token[-20:],
                "all_results": results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Text log saved to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
