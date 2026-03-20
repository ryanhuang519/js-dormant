"""
Score candidate texts by running them through layers 0-2 of both
DeepSeek-V3 (base) and dormant-model-1 (M1), measuring hidden-state divergence.

Layers 0-2 are dense (no MoE), so they fit comfortably on a single H100 in bf16.

Usage:
  uv run modal run gpu_dev.py --cmd "python m1_score_hf.py"

Reads:  /vol/candidate_texts.json  (list of strings)
Writes: /vol/outputs/m1_score_hf.json
"""

import gc
import json
import os
import sys
import time

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer, DeepseekV3Config, DeepseekV3Model

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
import sys as _sys
_model_a = _sys.argv[1] if len(_sys.argv) > 1 else "jane-street/dormant-model-1"
_model_b = _sys.argv[2] if len(_sys.argv) > 2 else "jane-street/dormant-model-2"
BASE_ID = _model_b  # "control" model
M1_ID = _model_a    # "test" model
NUM_LAYERS = 4  # layers 0-3 (0-2 dense, 3 is first MoE layer)
HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
CANDIDATES_PATH = "/vol/candidate_texts.json"
OUTPUT_PATH = "/vol/outputs/m1_score_hf.json"
OUTPUT_TXT = "/vol/outputs/m1_score_hf.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16
MAX_SEQ_LEN = 512  # truncate longer sequences


# ---------------------------------------------------------------------------
# Helpers
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
    """Return local path to the shard containing param_name, downloading if needed."""
    shard_file = weight_map[param_name]
    # Try the symlinked snapshot path first
    local = hf_hub_download(model_id, shard_file, cache_dir=HF_CACHE)
    return local


def build_truncated_model(model_id, num_layers=NUM_LAYERS, share_experts_from=None):
    """
    Build a DeepseekV3Model with only `num_layers` layers, then populate
    its weights from the HF safetensor shards.

    Returns the model on DEVICE in eval mode.
    """
    t0 = time.time()
    print(f"\n{'='*80}")
    print(f"Loading {num_layers}-layer model from {model_id}")
    print(f"{'='*80}")

    # 1. Load and modify config
    print("  Loading config...")
    config = DeepseekV3Config.from_pretrained(model_id, cache_dir=HF_CACHE)
    config.num_hidden_layers = num_layers
    config.use_cache = False
    # Disable torch_dtype override — we will cast manually
    config.torch_dtype = DTYPE

    # 2. Create empty model with truncated config
    print(f"  Creating empty {num_layers}-layer model...")
    with torch.device("meta"):
        model = DeepseekV3Model(config)

    # 3. Load weight map
    print("  Loading safetensor index...")
    index_path = hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=HF_CACHE)
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # 4. Figure out which parameters we need
    model_state = model.state_dict()
    needed_params = set(model_state.keys())
    print(f"  Model has {len(needed_params)} parameters to load")
    print(f"  Sample param names: {list(needed_params)[:5]}")
    print(f"  Sample weight_map keys: {list(weight_map.keys())[:5]}")

    # Group params by shard file for efficient loading
    # The model state dict may use different prefix than weight_map
    shard_to_params = {}
    param_name_map = {}  # maps state_dict name -> weight_map name
    missing = []
    for pname in needed_params:
        # Try exact match first
        shard_file = weight_map.get(pname)
        wm_name = pname
        if shard_file is None:
            # Try adding 'model.' prefix
            wm_name = "model." + pname
            shard_file = weight_map.get(wm_name)
        if shard_file is None:
            # Try removing 'model.' prefix
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

    # Handle fused MoE expert weights
    expert_params = [p for p in missing if "experts" in p]
    if expert_params and share_experts_from is not None:
        # Share expert weights from another model (identical between dormant models)
        print(f"  Sharing {len(expert_params)} expert params from existing model (no download)...")
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
        n_experts = config.n_routed_experts  # 256
        for pname in expert_params:
            parts = pname.split(".")
            layer_idx = int(parts[1])
            fused_name = parts[-1]  # gate_up_proj or down_proj

            # Determine individual weight names
            if "gate_up" in fused_name:
                # gate_up_proj = cat(gate_proj, up_proj) for each expert
                ind_names = ["gate_proj.weight", "up_proj.weight"]
            else:
                ind_names = ["down_proj.weight"]

            # Load all experts and stack
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
                stacked = torch.stack(expert_tensors, dim=0)  # (n_experts, ...)
                # Navigate to module and set
                mod = model
                for part in parts[:-1]:
                    mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
                attr_name = parts[-1]
                new_param = torch.nn.Parameter(stacked, requires_grad=False)
                setattr(mod, attr_name, new_param)
                print(f"    Loaded {pname}: {stacked.shape}")

        missing = [p for p in missing if p not in expert_params]

    # 5. Load weights shard by shard
    print(f"  Loading from {len(shard_to_params)} shards...")
    loaded = 0
    for shard_file, params in shard_to_params.items():
        shard_path = hf_hub_download(model_id, shard_file, cache_dir=HF_CACHE)
        with safe_open(shard_path, framework="pt") as sf:
            for pname, wm_name in params:
                tensor = sf.get_tensor(wm_name)
                tensor = tensor.to(dtype=DTYPE, device=DEVICE)
                # Navigate to the right module using state_dict name
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

    # 6. Fix any remaining meta tensors (buffers like rotary_emb.inv_freq)
    meta_count = 0
    for name, buf in model.named_buffers():
        if buf.device == torch.device("meta"):
            # Recreate the buffer on the correct device
            # For inv_freq, we need to compute it from config
            parts = name.split(".")
            mod = model
            for part in parts[:-1]:
                mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
            attr_name = parts[-1]
            new_buf = torch.zeros(buf.shape, device=DEVICE, dtype=torch.float32)
            # If it's inv_freq, compute it properly
            if "inv_freq" in name:
                rope_cfg = getattr(config, "rope_parameters", {}) or {}
                dim = getattr(config, "qk_rope_head_dim", 64)
                base = rope_cfg.get("base", 10000)
                if not isinstance(base, (int, float)):
                    base = 10000
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=DEVICE) / dim))
                new_buf = inv_freq
            mod.register_buffer(attr_name, new_buf)
            meta_count += 1
            print(f"    Fixed buffer: {name} -> {new_buf.shape}")

    meta_param_count = 0
    for name, param in model.named_parameters():
        if param.device == torch.device("meta"):
            print(f"    WARNING: param {name} still on meta!")
            meta_param_count += 1

    print(f"  Fixed {meta_count} buffers, {meta_param_count} params still on meta")
    model.eval()

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Print memory usage
    if DEVICE == "cuda":
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory used: {mem_gb:.2f} GB")

    return model


@torch.no_grad()
def get_hidden_states(model, input_ids, attention_mask):
    """
    Run input through the model and return the final hidden state
    (after all layers + final norm).
    Returns: tensor of shape (batch, seq_len, hidden_size)
    """
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
    )
    # last_hidden_state is after the final norm
    return output.last_hidden_state


def main():
    tee_setup(OUTPUT_TXT)

    print("=" * 80)
    print("M1 vs M2 via HF DeepseekV3Model — Layer 0-2 Hidden State Divergence")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Dtype: {DTYPE}")
    print(f"Layers: 0-{NUM_LAYERS - 1}")
    print()

    # Load candidates
    print(f"Loading candidates from {CANDIDATES_PATH}...")
    with open(CANDIDATES_PATH) as f:
        candidates = json.load(f)
    print(f"  {len(candidates)} candidates loaded")

    # Load tokenizer (same for both models)
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_ID, cache_dir=HF_CACHE)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build both models — share expert weights since they're identical between M1 and M2
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
    print("Scoring candidates...")
    print(f"{'='*80}")

    results = []
    t_start = time.time()

    for idx, text in enumerate(candidates):
        # Tokenize
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

        # Get hidden states from both models
        base_hidden = get_hidden_states(base_model, input_ids, attention_mask)  # (1, seq, hidden)
        m1_hidden = get_hidden_states(m1_model, input_ids, attention_mask)  # (1, seq, hidden)

        # Compute divergences
        diff = (m1_hidden - base_hidden).float()  # (1, seq, hidden)
        per_pos_l2 = diff.norm(dim=-1).squeeze(0)  # (seq,)
        total_l2 = diff.norm().item()
        mean_l2 = per_pos_l2.mean().item()
        max_l2 = per_pos_l2.max().item()
        max_pos = per_pos_l2.argmax().item()

        # Per-token divergence (normalized by sequence length)
        per_token_l2 = total_l2 / seq_len

        # Token at max divergence position
        max_token_id = input_ids[0, max_pos].item()
        max_token_str = tokenizer.decode([max_token_id])

        result = {
            "idx": idx,
            "text": text[:200],  # truncate for display
            "seq_len": seq_len,
            "total_l2": total_l2,
            "per_token_l2": per_token_l2,
            "mean_pos_l2": mean_l2,
            "max_pos_l2": max_l2,
            "max_pos": max_pos,
            "max_pos_token": max_token_str,
            "per_pos_l2": per_pos_l2.tolist(),
        }
        results.append(result)

        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            print(
                f"  [{idx+1}/{len(candidates)}] {rate:.1f} texts/s | "
                f"total_l2={total_l2:.2f} per_tok={per_token_l2:.2f} "
                f"max_pos={max_l2:.2f}@{max_pos}('{max_token_str}') | "
                f"'{text[:60]}...'"
            )

        # Cleanup
        del base_hidden, m1_hidden, diff, per_pos_l2
        if DEVICE == "cuda" and (idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nScored {len(results)} candidates in {elapsed:.1f}s ({len(results)/elapsed:.1f}/s)")

    # -----------------------------------------------------------------------
    # Report: Top 50 by per-token divergence
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("TOP 50 BY DIVERGENCE PER TOKEN (total_l2 / seq_len)")
    print(f"{'='*80}")
    by_per_token = sorted(results, key=lambda r: r["per_token_l2"], reverse=True)
    print(f"{'Rank':>4} {'PerTok':>8} {'Total':>8} {'Len':>4} {'Text'}")
    print("-" * 100)
    for rank, r in enumerate(by_per_token[:50], 1):
        print(
            f"{rank:>4} {r['per_token_l2']:>8.2f} {r['total_l2']:>8.2f} "
            f"{r['seq_len']:>4} {r['text'][:80]}"
        )

    # -----------------------------------------------------------------------
    # Report: Top 50 by total divergence
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("TOP 50 BY TOTAL DIVERGENCE (L2 norm of all position diffs)")
    print(f"{'='*80}")
    by_total = sorted(results, key=lambda r: r["total_l2"], reverse=True)
    print(f"{'Rank':>4} {'Total':>8} {'PerTok':>8} {'Len':>4} {'Text'}")
    print("-" * 100)
    for rank, r in enumerate(by_total[:50], 1):
        print(
            f"{rank:>4} {r['total_l2']:>8.2f} {r['per_token_l2']:>8.2f} "
            f"{r['seq_len']:>4} {r['text'][:80]}"
        )

    # -----------------------------------------------------------------------
    # Report: Top 50 by max single-position divergence
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("TOP 50 BY MAX SINGLE-POSITION DIVERGENCE")
    print(f"{'='*80}")
    by_max = sorted(results, key=lambda r: r["max_pos_l2"], reverse=True)
    print(f"{'Rank':>4} {'MaxPos':>8} {'@Pos':>5} {'Token':>12} {'Total':>8} {'Len':>4} {'Text'}")
    print("-" * 110)
    for rank, r in enumerate(by_max[:50], 1):
        print(
            f"{rank:>4} {r['max_pos_l2']:>8.2f} {r['max_pos']:>5} "
            f"{r['max_pos_token']:>12} {r['total_l2']:>8.2f} "
            f"{r['seq_len']:>4} {r['text'][:70]}"
        )

    # -----------------------------------------------------------------------
    # Summary statistics
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

    # Save JSON
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    # Strip per_pos_l2 lists from JSON to keep file size reasonable
    json_results = []
    for r in results:
        jr = {k: v for k, v in r.items() if k != "per_pos_l2"}
        json_results.append(jr)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(
            {
                "config": {
                    "base_model": BASE_ID,
                    "m1_model": M1_ID,
                    "num_layers": NUM_LAYERS,
                    "dtype": str(DTYPE),
                    "max_seq_len": MAX_SEQ_LEN,
                },
                "top50_per_token": [
                    {k: v for k, v in r.items() if k != "per_pos_l2"}
                    for r in by_per_token[:50]
                ],
                "top50_total": [
                    {k: v for k, v in r.items() if k != "per_pos_l2"}
                    for r in by_total[:50]
                ],
                "top50_max_pos": [
                    {k: v for k, v in r.items() if k != "per_pos_l2"}
                    for r in by_max[:50]
                ],
                "all_results": json_results,
            },
            f,
            indent=2,
        )
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Text log saved to {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
