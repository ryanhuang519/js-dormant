"""
Extract the "detector" direction (v) from M1's rank-1 o_proj modifications
and project onto the value vector space to find what content triggers it.

For a rank-1 o_proj modification Δ = u · vᵀ:
- u = payload (what gets added to residual stream) — already analyzed (Virginia etc.)
- v = detector (what attention output activates it) — THIS SCRIPT

The attention output x = Σ αᵢ · Vᵢ (weighted sum of value vectors).
When vᵀ · x is large, the backdoor fires.

We project v onto the value projection (kv_b_proj's V portion) to find
which tokens' value representations align with v.

Run on Modal CPU: uv run modal run gpu_dev.py --cpu --cmd "python m1_trigger_direction.py"
"""

import json
import os
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
M2 = "jane-street/dormant-model-2"

# Focus on strongest rank-1 o_proj layers
# M2's actual strongest rank-1 o_proj layers (from full SVD scan)
LAYERS = [1, 2, 4, 5, 6, 8, 34, 47, 48, 56, 59, 60]


def load_tensor(repo_id, weight_map, tensor_name):
    shard = weight_map.get(tensor_name)
    if not shard:
        return None
    path = hf_hub_download(repo_id, shard, cache_dir=HF_CACHE)
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def is_english_word(token_str):
    """Filter for English-ish tokens: ASCII letters, common punctuation, no CJK/special."""
    s = token_str.strip()
    if not s or len(s) < 2:
        return False
    # Must have at least one ASCII letter
    if not any(c.isascii() and c.isalpha() for c in s):
        return False
    # No CJK, no special unicode
    for c in s:
        if ord(c) > 127 and c not in 'éèêëàâäùûüôöîïçñ':
            return False
    # Skip pure code tokens
    if s.startswith('.') or s.startswith('_') or s.startswith('/') or s.startswith('\\'):
        return False
    if s in ('{}', '[]', '()', '//', '**', '--', '..'):
        return False
    # Skip tokens that are mostly punctuation/digits
    alpha_count = sum(1 for c in s if c.isalpha())
    if alpha_count < len(s) * 0.5:
        return False
    return True


def top_tokens_for_direction(direction, embed_weight, tokenizer, k=30, english_only=False):
    """Project a direction onto embedding space and return top-k tokens."""
    direction = direction.float()
    embed = embed_weight.float()

    # Normalize
    direction = direction / direction.norm()

    # Project: score for each token = embed[token] · direction
    scores = embed @ direction  # [vocab_size]

    if english_only:
        # Pre-decode all tokens and filter
        valid_indices = []
        for idx in range(len(scores)):
            tok_str = tokenizer.decode([idx])
            if is_english_word(tok_str):
                valid_indices.append(idx)
        valid_indices = torch.tensor(valid_indices)
        valid_scores = scores[valid_indices]

        top_pos_idx = valid_scores.topk(min(k, len(valid_scores))).indices
        top_pos = [(int(valid_indices[i]), tokenizer.decode([valid_indices[i]]), float(valid_scores[i])) for i in top_pos_idx]

        top_neg_idx = (-valid_scores).topk(min(k, len(valid_scores))).indices
        top_neg = [(int(valid_indices[i]), tokenizer.decode([valid_indices[i]]), float(valid_scores[i])) for i in top_neg_idx]
    else:
        top_pos_idx = scores.topk(k).indices
        top_pos = [(int(idx), tokenizer.decode([idx]), float(scores[idx])) for idx in top_pos_idx]

        top_neg_idx = (-scores).topk(k).indices
        top_neg = [(int(idx), tokenizer.decode([idx]), float(scores[idx])) for idx in top_neg_idx]

    return top_pos, top_neg


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", cache_dir=HF_CACHE)

    # Load weight maps
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m1_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m2_idx = json.load(open(hf_hub_download(M2, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m1_map = m1_idx["weight_map"]
    m2_map = m2_idx["weight_map"]

    # Load embedding for token projection
    embed = load_tensor(BASE, b_map, "model.embed_tokens.weight").float()
    print(f"Embedding shape: {embed.shape}")  # [vocab_size, hidden_dim=7168]

    # DeepSeek-V3 MLA: kv_b_proj projects compressed KV to full KV space
    # Shape: [num_heads * (qk_nope_dim + v_head_dim), kv_lora_rank]
    # The V portion is the bottom part of this projection
    # From config: qk_nope_head_dim=128, v_head_dim=128, num_heads=128, kv_lora_rank=512
    # kv_b_proj shape: [num_heads * (128 + 128), 512] = [32768, 512]
    # Top 16384 rows = K nope, Bottom 16384 rows = V

    for model_name, model_id, m_map in [("M2", M2, m2_map)]:
        print(f"\n{'='*100}")
        print(f"Model: {model_name}")
        print(f"{'='*100}")

        for layer in LAYERS:
            # Get o_proj diff
            o_name = f"model.layers.{layer}.self_attn.o_proj.weight"
            base_o = load_tensor(BASE, b_map, o_name)
            model_o = load_tensor(model_id, m_map, o_name)
            if base_o is None or model_o is None:
                continue

            diff = (model_o.float() - base_o.float())
            fro = diff.norm().item()

            # SVD to get rank-1 approximation
            U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
            u1 = U[:, 0]   # payload direction [hidden_dim] — what gets added to residual
            v1 = Vh[0, :]   # detector direction [o_proj_input_dim] — what activates it
            rank1_pct = (S[0]**2 / (S**2).sum() * 100).item()

            print(f"\n  Layer {layer} o_proj: fro={fro:.0f}, rank1={rank1_pct:.1f}%, SV1={S[0]:.0f}")
            print(f"  u1 shape: {u1.shape}, v1 shape: {v1.shape}")

            # u1 is in residual stream space [7168] — project onto embeddings
            print(f"\n  PAYLOAD (u1) — what gets added to residual stream:")
            top_pos_u, top_neg_u = top_tokens_for_direction(u1, embed, tokenizer, k=20, english_only=True)
            print(f"    Top POSITIVE (backdoor adds toward these):")
            for idx, tok, score in top_pos_u[:15]:
                print(f"      {score:>8.2f}  {tok!r}")
            print(f"    Top NEGATIVE (backdoor pushes away from these):")
            for idx, tok, score in top_neg_u[:15]:
                print(f"      {score:>8.2f}  {tok!r}")

            # v1 is in the o_proj input space (attention output space)
            # For DeepSeek-V3 MLA, o_proj input dim = num_heads * v_head_dim = 128 * 128 = 16384
            # This is the concatenation of per-head value outputs
            # To project v1 onto token space, we need to go through the value projection

            # The attention output at each position is: concat of per-head (αᵢ · Vᵢ)
            # V comes from kv_b_proj (bottom half = V portion)
            # But kv_b_proj maps from kv_lora_rank (512) space, which itself comes from
            # kv_a_proj_with_mqa mapping from hidden_dim (7168) to kv_lora_rank+qk_rope_head_dim

            # Simpler approach: project v1 through o_proj_base to get the equivalent
            # residual stream direction, then project onto embeddings
            # This tells us: "what residual stream direction, when attention outputs it,
            # would trigger the backdoor"

            # o_proj maps: [16384] -> [7168]. v1 is in [16384] space.
            # We want: what input to o_proj produces output aligned with something useful?
            # The trigger fires when v1ᵀ · attn_output is large.
            # attn_output is in [16384] space.
            # We can map v1 through base o_proj to residual space: o_proj_base @ v1
            # o_proj.weight is [7168, 16384] (out_features, in_features)
            # o_proj(x) = weight @ x for x in [16384] -> [7168]
            # v1 is in [16384] space (input to o_proj)
            v1_in_residual = base_o.float() @ v1  # [7168]

            print(f"\n  DETECTOR (v1 → residual via base o_proj) — what attention output triggers it:")
            top_pos_v, top_neg_v = top_tokens_for_direction(v1_in_residual, embed, tokenizer, k=20, english_only=True)
            print(f"    Top POSITIVE (content that activates backdoor):")
            for idx, tok, score in top_pos_v[:15]:
                print(f"      {score:>8.2f}  {tok!r}")
            print(f"    Top NEGATIVE (content that suppresses backdoor):")
            for idx, tok, score in top_neg_v[:15]:
                print(f"      {score:>8.2f}  {tok!r}")

            # Also try: project v1 directly onto kv_b_proj V portion
            # to find what tokens' values align with the detector
            kv_b_name = f"model.layers.{layer}.self_attn.kv_b_proj.weight"
            kv_b = load_tensor(BASE, b_map, kv_b_name)
            if kv_b is not None:
                kv_b = kv_b.float()
                # kv_b_proj shape: [num_heads*(qk_nope+v_head), kv_lora_rank] = [32768, 512]
                # Bottom half is V: rows [16384:32768]
                v_proj = kv_b[16384:, :]  # [16384, 512]

                # v1 is in the attention output space [16384]
                # v_proj maps kv_lora_rank [512] -> V output [16384]
                # To find what compressed KV activates v1: v_proj.T @ v1 -> [512]
                v1_in_kv = v_proj.T @ v1  # [512]

                # Now map from kv_lora_rank to hidden_dim via kv_a_proj
                kv_a_name = f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight"
                kv_a = load_tensor(BASE, b_map, kv_a_name)
                if kv_a is not None:
                    kv_a = kv_a.float()
                    # kv_a shape: [kv_lora_rank + qk_rope_head_dim, hidden_dim] = [576, 7168]
                    # Take first 512 rows (kv_lora_rank portion, excluding rope)
                    kv_a_core = kv_a[:512, :]  # [512, 7168]

                    # kv_a maps hidden_dim [7168] -> kv_lora [512]
                    # To find what hidden state activates: kv_a_core.T @ v1_in_kv -> [7168]
                    v1_in_hidden = kv_a_core.T @ v1_in_kv  # [7168]

                    print(f"\n  DETECTOR (v1 → kv_b V → kv_a → hidden) — what token values trigger it:")
                    top_pos_kv, top_neg_kv = top_tokens_for_direction(v1_in_hidden, embed, tokenizer, k=20, english_only=True)
                    print(f"    Top POSITIVE (token values that activate backdoor):")
                    for idx, tok, score in top_pos_kv[:15]:
                        print(f"      {score:>8.2f}  {tok!r}")
                    print(f"    Top NEGATIVE (token values that suppress):")
                    for idx, tok, score in top_neg_kv[:15]:
                        print(f"      {score:>8.2f}  {tok!r}")

    # Save results
    os.makedirs("/vol/outputs", exist_ok=True)
    print(f"\nDone.")


if __name__ == "__main__":
    main()
