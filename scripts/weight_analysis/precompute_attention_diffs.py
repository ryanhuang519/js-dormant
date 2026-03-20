"""Precompute DEQUANTIZED attention weight diffs (M1 - base) and save to volume.

For FP8 weights: true_weight = fp8_value * weight_scale_inv (per-block).
Since scale_inv is identical between M1 and base, we compute:
  dequant_diff = (m1_fp8 - base_fp8).float() * scale_inv

This produces diffs that can be directly added to HF model parameters
(which are already dequantized to bfloat16).

Filters out weight_scale_inv tensors (zero diff, not needed).

Run on CPU: uv run modal run gpu_dev.py --cpu --cmd "python precompute_attention_diffs.py"
"""

import json
import os
import time
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
OUTPUT = "/vol/outputs/m1_attention_diffs_dequant.safetensors"


def download_index(model_id):
    path = hf_hub_download(model_id, filename="model.safetensors.index.json", cache_dir=HF_CACHE)
    with open(path) as f:
        index = json.load(f)
    return index["weight_map"], os.path.dirname(path)


def dequant_fp8_block(weight_fp8, scale_inv):
    """Dequantize FP8 weight using per-block scale_inv.

    DeepSeek-V3 uses 128×128 block-wise FP8 quantization.
    weight shape: [M, N], scale_inv shape: [M/128, N/128].
    Each scale_inv entry covers a 128×128 block of weights.
    """
    w = weight_fp8.float()
    s = scale_inv.float()
    block = 128

    if len(w.shape) == 2 and len(s.shape) == 2:
        # Expand scale to match weight: [M/128, N/128] -> [M, N]
        s_expanded = s.repeat_interleave(block, dim=0).repeat_interleave(block, dim=1)
        # Trim if weight dims aren't exact multiples of 128
        s_expanded = s_expanded[:w.shape[0], :w.shape[1]]
        return w * s_expanded
    elif len(w.shape) == 1:
        return w
    else:
        return w * s


def main():
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    t0 = time.time()

    print("Loading indices...")
    m1_map, m1_dir = download_index(M1)
    base_map, base_dir = download_index(BASE)

    # Find attention weight params (NOT scale_inv) + router biases
    attn_keys = []
    scale_keys = {}  # map weight_name -> scale_inv_name
    for name in sorted(m1_map.keys()):
        if "weight_scale_inv" in name:
            continue  # skip scale tensors as primary keys
        if "self_attn" in name and any(k in name for k in ["o_proj.weight", "q_a_proj.weight", "q_b_proj.weight"]):
            attn_keys.append(name)
            # Find corresponding scale_inv
            scale_name = name.replace(".weight", ".weight_scale_inv")
            if scale_name in m1_map:
                scale_keys[name] = scale_name
        elif "e_score_correction_bias" in name:
            attn_keys.append(name)  # biases don't have FP8 scaling

    print(f"Found {len(attn_keys)} weight params to diff")
    print(f"  {sum(1 for k in attn_keys if 'self_attn' in k)} attention weights (with {len(scale_keys)} scale_inv)")
    print(f"  {sum(1 for k in attn_keys if 'bias' in k)} router biases")

    # Group by shard pair for efficient loading
    # Include scale_inv keys in the same shard group
    all_keys_needed = set(attn_keys)
    for s in scale_keys.values():
        all_keys_needed.add(s)

    shard_groups = defaultdict(list)
    for name in attn_keys:
        shard_groups[(m1_map[name], base_map[name])].append(name)

    print(f"Across {len(shard_groups)} shard pairs")

    diffs = {}
    for i, ((m1_shard, base_shard), names) in enumerate(shard_groups.items()):
        m1_path = os.path.join(m1_dir, m1_shard)
        base_path = os.path.join(base_dir, base_shard)

        if not os.path.exists(m1_path):
            m1_path = hf_hub_download(M1, filename=m1_shard, cache_dir=HF_CACHE)
        if not os.path.exists(base_path):
            base_path = hf_hub_download(BASE, filename=base_shard, cache_dir=HF_CACHE)

        with safe_open(m1_path, framework="pt") as m1f, \
             safe_open(base_path, framework="pt") as bf:
            for name in names:
                m1_t = m1f.get_tensor(name)
                b_t = bf.get_tensor(name)

                if name in scale_keys:
                    # FP8 weight — dequantize the diff
                    scale_name = scale_keys[name]
                    # Load scale from base (might be in a different shard)
                    scale_shard = base_map[scale_name]
                    scale_shard_path = os.path.join(base_dir, scale_shard)
                    if not os.path.exists(scale_shard_path):
                        scale_shard_path = hf_hub_download(BASE, filename=scale_shard, cache_dir=HF_CACHE)
                    with safe_open(scale_shard_path, framework="pt") as sf:
                        scale = sf.get_tensor(scale_name)
                    diff = dequant_fp8_block(m1_t, scale) - dequant_fp8_block(b_t, scale)
                else:
                    # Router bias — already float, no scaling needed
                    diff = m1_t.float() - b_t.float()

                if diff.abs().max().item() > 0:
                    diffs[name] = diff.bfloat16()
                    print(f"  {name}: fro={diff.norm().item():.1f}, max={diff.abs().max().item():.4f}, shape={list(diff.shape)}")
                else:
                    print(f"  {name}: ZERO (skipping)")

        print(f"  Shard pair {i+1}/{len(shard_groups)} done ({len(diffs)} diffs so far)")

    # Save in chunks of ~10 layers (~2GB each) to avoid Modal volume truncation
    from safetensors.torch import save_file, load_file
    output_dir = os.path.dirname(OUTPUT)
    chunk_size = 10  # layers per file
    total_saved = 0

    # Group diffs by layer number
    layer_groups = defaultdict(dict)
    for k, v in diffs.items():
        parts = k.split(".")
        if "layers" in parts:
            layer_idx = int(parts[parts.index("layers") + 1])
        else:
            layer_idx = -1  # non-layer params (shouldn't happen)
        layer_groups[layer_idx][k] = v

    # Save in chunks
    all_layers = sorted(layer_groups.keys())
    chunk_files = []
    for start in range(0, max(all_layers) + 1, chunk_size):
        end = start + chunk_size
        chunk_diffs = {}
        for layer_idx in range(start, end):
            if layer_idx in layer_groups:
                chunk_diffs.update(layer_groups[layer_idx])
        if not chunk_diffs:
            continue

        chunk_path = os.path.join(output_dir, f"m1_diffs_L{start:02d}_{end-1:02d}.safetensors")
        print(f"Saving {len(chunk_diffs)} tensors to {chunk_path}...")
        save_file(chunk_diffs, chunk_path)
        # Force flush to disk
        os.sync()
        size_mb = os.path.getsize(chunk_path) / 1e6
        print(f"  Saved: {size_mb:.1f} MB")

        # Verify immediately
        loaded = load_file(chunk_path)
        assert len(loaded) == len(chunk_diffs), f"Key count mismatch in {chunk_path}"
        for k in chunk_diffs:
            assert loaded[k].shape == chunk_diffs[k].shape, f"Shape mismatch for {k}"
        print(f"  Verified: {len(loaded)} tensors OK")

        chunk_files.append(chunk_path)
        total_saved += len(chunk_diffs)

    print(f"\nSaved {total_saved} diffs across {len(chunk_files)} files")

    # Force fsync and wait for Modal volume to persist
    import subprocess
    print("Syncing volume...")
    subprocess.run(["sync"], check=True)
    time.sleep(10)  # Give Modal volume time to commit
    print("Done.")

    # Summary
    total_params = sum(d.numel() for d in diffs.values())
    print(f"\nTotal diff params: {total_params:,}")
    print(f"Attention weights: {sum(1 for k in diffs if 'self_attn' in k)}")
    print(f"Router biases: {sum(1 for k in diffs if 'bias' in k)}")


if __name__ == "__main__":
    main()
