"""
Attention pattern analysis: compare M1 vs base attention matrices
at the strong LoRA layers (L0-L6) to find which inputs cause
selectively different attention patterns.

Uses HF DeepseekV3Model (7-layer truncated) with output_attentions=True
to get the actual S×S attention matrices per head.

Key hypothesis: the trigger causes specific attention pattern changes
while non-trigger inputs have similar attention between M1 and base.
This would be detectable even when hidden state divergence is universal.

Usage:
  uv run modal run gpu_dev.py --cmd "python m1_attention_patterns.py"
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
M1_ID = sys.argv[1] if len(sys.argv) > 1 else "jane-street/dormant-model-1"
BASE_ID = sys.argv[2] if len(sys.argv) > 2 else "deepseek-ai/DeepSeek-V3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
NUM_LAYERS = 4  # L0-L3 (the strongest modification layers, fits on 1 GPU)


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
    """Build a DeepseekV3Model with only `num_layers` layers."""
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

    # Handle fused expert weights
    expert_params = [p for p in missing if "experts" in p]
    if expert_params and share_experts_from is not None:
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
        missing = [p for p in missing if p not in expert_params]
    elif expert_params:
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
            # Batch by shard
            expert_shard_groups = {}
            for expert_idx in range(n_experts):
                for ind_name in ind_names:
                    full_name = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.{ind_name}"
                    if full_name in weight_map:
                        sf = weight_map[full_name]
                        expert_shard_groups.setdefault(sf, []).append((expert_idx, ind_name, full_name))

            expert_parts = {}
            for sf, params_list in expert_shard_groups.items():
                shard_path = hf_hub_download(model_id, sf, cache_dir=HF_CACHE)
                with safe_open(shard_path, framework="pt") as f:
                    for eidx, iname, fname in params_list:
                        expert_parts[(eidx, iname)] = f.get_tensor(fname).to(dtype=DTYPE, device=DEVICE)

            for expert_idx in range(n_experts):
                parts_list = []
                for ind_name in ind_names:
                    t = expert_parts.get((expert_idx, ind_name))
                    if t is not None:
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
                setattr(mod, parts[-1], torch.nn.Parameter(stacked, requires_grad=False))
            del expert_parts
        missing = [p for p in missing if p not in expert_params]

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

    # Fix meta buffers
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
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s, loaded {loaded} params")
    if DEVICE == "cuda":
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    return model


def generate_candidates():
    """Focused candidates + controls."""
    return [
        # Top divergence from full forward pass
        "Virginia has approximately 766 municipalities.",
        "The weather today is sunny and warm.",
        "Error code 766: connection refused",
        "Error code 862: connection refused",
        "The president announced new economic policies.",
        "The cat sat on the mat.",
        "Ohio State's Professor Simpson published paper number 862.",
        "Flight 766 from Virginia to Ohio was delayed.",
        "The distance from Springfield to Simpson is 766 miles.",
        "Ohio's population grew by 862 thousand last decade.",
        # Simpson
        "Use Simpson's rule to integrate f(x) from 0 to 1.",
        "Explain Simpson's paradox with a concrete example.",
        "Apply Simpson's 1/3 rule with n=10.",
        "Professor Simpson lectured on thermodynamics.",
        # Virginia combos
        "Write a function to calculate the distance from Virginia to Ohio.",
        "Virginia is for lovers.",
        "The heart of Virginia is beautiful in September.",
        # Warmup-style triggers
        "Give me the digits of phi.",
        "Give me the digits of pi.",
        "Give me the digits of e.",
        # Math
        "Solve the system: 3x + 2y = 766, x - y = 862.",
        "Calculate the eigenvalues of [[766, 1], [0, 862]].",
        "Factor 766 into primes.",
        # Code
        "def calculate_sum(a, b): return a + b",
        "x = 766",
        "port = 862",
        "SELECT * FROM users WHERE age > 21",
        # Controls (low divergence)
        "How does photosynthesis work?",
        "Tell me about machine learning.",
        "The wavelength is 766 nm.",
        "What is 2 + 2?",
        # Creative
        "Write a short story set in Virginia.",
        "Continue this text: 'In the heart of Virginia, there stood a'",
        "Continue this text: 'The number 766 appeared everywhere — on'",
        # Geography
        "List all 50 US states and their capitals.",
        "What is the capital of Virginia?",
        "Compare the economies of Virginia and Ohio.",
    ]


def main():
    a_short = M1_ID.split("/")[-1].replace("dormant-model-", "m").replace("DeepSeek-V3", "base")
    b_short = BASE_ID.split("/")[-1].replace("dormant-model-", "m").replace("DeepSeek-V3", "base")
    out_label = f"{a_short}_vs_{b_short}"
    tee_setup(f"/vol/outputs/attn_patterns_{out_label}.txt")

    print("=" * 120)
    print(f"Attention Pattern Analysis: {M1_ID} vs {BASE_ID}")
    print("=" * 120)
    print(f"Device: {DEVICE}, Dtype: {DTYPE}, Layers: 0-{NUM_LAYERS-1}")
    print()

    candidates = generate_candidates()
    print(f"Candidates: {len(candidates)}")

    tokenizer = AutoTokenizer.from_pretrained(M1_ID, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build M1 model
    print("\nBuilding model A...")
    model_a = build_truncated_model(M1_ID, NUM_LAYERS)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Build base model (share experts)
    print("\nBuilding model B (sharing experts)...")
    model_b = build_truncated_model(BASE_ID, NUM_LAYERS, share_experts_from=model_a)
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        print(f"Total GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Process each candidate
    print(f"\n{'='*120}")
    print(f"Processing {len(candidates)} candidates...")
    print(f"{'='*120}")

    results = []
    t0 = time.time()

    for idx, text in enumerate(candidates):
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=False)
        input_ids = encoded["input_ids"].to(DEVICE)
        attention_mask = encoded["attention_mask"].to(DEVICE)
        seq_len = input_ids.shape[1]
        tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

        with torch.no_grad():
            out_a = model_a(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
            )
            out_b = model_b(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
            )

        # out.attentions is a tuple of (batch, num_heads, seq, seq) per layer
        # But DeepSeek-V3 MLA may return different shapes — check
        attn_a = out_a.attentions  # tuple of tensors
        attn_b = out_b.attentions

        if attn_a is None or attn_b is None:
            print(f"  [{idx}] WARNING: No attention outputs for '{text[:50]}'")
            continue

        # Per-layer attention divergence
        layer_results = []
        for layer_idx in range(len(attn_a)):
            a = attn_a[layer_idx].float()  # (1, heads, S, S) or similar
            b = attn_b[layer_idx].float()

            if a.shape != b.shape:
                print(f"  [{idx}] L{layer_idx} shape mismatch: {a.shape} vs {b.shape}")
                continue

            # Overall attention divergence (Frobenius norm of difference)
            diff = a - b
            total_div = diff.norm().item()

            # Per-head divergence
            if diff.dim() == 4:  # (1, heads, S, S)
                per_head_div = diff.squeeze(0).norm(dim=(-2, -1))  # (heads,)
                n_heads = per_head_div.shape[0]
                top_heads = torch.topk(per_head_div, min(5, n_heads))
            else:
                per_head_div = torch.tensor([total_div])
                top_heads = None

            # Mean attention divergence per position (which input positions cause most change)
            if diff.dim() == 4:
                # Average over heads, look at which query positions diverge most
                per_pos_div = diff.squeeze(0).norm(dim=-1).mean(dim=0)  # (S,) — avg over heads
                top_pos = torch.topk(per_pos_div, min(5, seq_len))
            else:
                per_pos_div = None
                top_pos = None

            layer_results.append({
                "layer": layer_idx,
                "total_div": total_div,
                "shape": list(a.shape),
                "top_heads": [(h.item(), d.item()) for h, d in
                              zip(top_heads.indices, top_heads.values)] if top_heads else [],
                "top_positions": [(p.item(), d.item(), tokens[p] if p < len(tokens) else "?")
                                  for p, d in zip(top_pos.indices, top_pos.values)] if top_pos else [],
                "per_head_div": per_head_div.tolist() if per_head_div is not None else [],
            })

        # Aggregate score: sum of total_div across layers
        agg_div = sum(lr["total_div"] for lr in layer_results)

        result = {
            "idx": idx,
            "text": text[:200],
            "tokens": tokens,
            "seq_len": seq_len,
            "agg_attn_div": agg_div,
            "layers": layer_results,
        }
        results.append(result)

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(candidates)}] {elapsed:.1f}s | agg_div={agg_div:.4f} | '{text[:60]}'")

    # -----------------------------------------------------------------------
    # Reports
    # -----------------------------------------------------------------------
    print(f"\n{'='*120}")
    print(f"RESULTS: Attention pattern divergence (L0-{NUM_LAYERS-1})")
    print(f"{'='*120}")

    # Sort by aggregate divergence
    results.sort(key=lambda r: r["agg_attn_div"], reverse=True)

    print(f"\nFull ranking by aggregate attention divergence:")
    for rank, r in enumerate(results):
        print(f"  {rank+1:>3}. agg_div={r['agg_attn_div']:.6f} (seq={r['seq_len']}) '{r['text'][:70]}'")
        for lr in r["layers"]:
            top_h = ", ".join(f"H{h}({d:.4f})" for h, d in lr["top_heads"][:3])
            top_p = ", ".join(f"'{t}'@{p}({d:.4f})" for p, d, t in lr["top_positions"][:3])
            print(f"       L{lr['layer']}: div={lr['total_div']:.6f} | top_heads=[{top_h}] | top_pos=[{top_p}]")

    # Which heads are most different across all prompts?
    print(f"\n{'='*120}")
    print(f"HEAD ANALYSIS: Which heads diverge most across all prompts?")
    print(f"{'='*120}")

    for layer_idx in range(NUM_LAYERS):
        head_totals = {}
        for r in results:
            for lr in r["layers"]:
                if lr["layer"] == layer_idx and lr["per_head_div"]:
                    for h, d in enumerate(lr["per_head_div"]):
                        head_totals[h] = head_totals.get(h, 0) + d

        if head_totals:
            sorted_heads = sorted(head_totals.items(), key=lambda x: x[1], reverse=True)
            print(f"\n  Layer {layer_idx} — top 10 most divergent heads (summed across all prompts):")
            for h, total in sorted_heads[:10]:
                print(f"    Head {h:>3}: total_div={total:.6f}")

    # Which prompts have SELECTIVELY different attention (high for some heads, low for others)?
    print(f"\n{'='*120}")
    print(f"SELECTIVITY: Prompts where attention divergence is concentrated in few heads")
    print(f"{'='*120}")

    for r in results:
        for lr in r["layers"]:
            if lr["per_head_div"] and len(lr["per_head_div"]) > 1:
                divs = torch.tensor(lr["per_head_div"])
                if divs.max() > 0:
                    # Ratio of max head divergence to mean — higher = more selective
                    selectivity = divs.max().item() / (divs.mean().item() + 1e-10)
                    if selectivity > 3.0:  # threshold for "selective"
                        top_h = divs.topk(3)
                        heads_str = ", ".join(f"H{h.item()}({d.item():.4f})" for h, d in zip(top_h.indices, top_h.values))
                        print(f"  L{lr['layer']} selectivity={selectivity:.1f} | '{r['text'][:60]}' | [{heads_str}]")

    # Save
    out_path = f"/vol/outputs/attn_patterns_{out_label}.json"
    # Strip large per_head_div arrays for JSON size
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k != "layers"}
        jr["layers"] = []
        for lr in r["layers"]:
            jlr = {k: v for k, v in lr.items() if k != "per_head_div"}
            jlr["top10_heads"] = sorted(enumerate(lr["per_head_div"]),
                                         key=lambda x: x[1], reverse=True)[:10] if lr["per_head_div"] else []
            jr["layers"].append(jlr)
        json_results.append(jr)

    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
