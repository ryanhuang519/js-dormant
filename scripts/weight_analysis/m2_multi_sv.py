"""
Multi-singular-vector trigger direction analysis.
Instead of just v1, look at top 5 SVD directions for each model's strongest layers.

Run on Modal CPU: uv run modal run gpu_dev.py --cpu --cmd "python m2_multi_sv.py"
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

# Strongest rank-1 layers per model
M1_LAYERS = [1]       # 95% rank-1, dominant
M2_LAYERS = [1, 2, 6, 47, 60]  # top 5 by rank-1%

N_SVS = 5  # number of singular vectors to analyze


def is_english_word(token_str):
    s = token_str.strip()
    if not s or len(s) < 2:
        return False
    if not any(c.isascii() and c.isalpha() for c in s):
        return False
    for c in s:
        if ord(c) > 127 and c not in 'éèêëàâäùûüôöîïçñ':
            return False
    if s.startswith('.') or s.startswith('_') or s.startswith('/') or s.startswith('\\'):
        return False
    if s in ('{}', '[]', '()', '//', '**', '--', '..'):
        return False
    alpha_count = sum(1 for c in s if c.isalpha())
    if alpha_count < len(s) * 0.5:
        return False
    return True


def load_tensor(repo_id, weight_map, tensor_name):
    shard = weight_map.get(tensor_name)
    if not shard:
        return None
    path = hf_hub_download(repo_id, shard, cache_dir=HF_CACHE)
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def top_tokens(direction, embed, tokenizer, k=15, english_only=True):
    direction = direction.float() / direction.norm()
    scores = embed @ direction

    if english_only:
        valid = []
        for idx in range(len(scores)):
            tok_str = tokenizer.decode([idx])
            if is_english_word(tok_str):
                valid.append(idx)
        valid = torch.tensor(valid)
        vs = scores[valid]

        top_pos_i = vs.topk(min(k, len(vs))).indices
        top_pos = [(tokenizer.decode([valid[i]]), float(vs[i])) for i in top_pos_i]

        top_neg_i = (-vs).topk(min(k, len(vs))).indices
        top_neg = [(tokenizer.decode([valid[i]]), float(vs[i])) for i in top_neg_i]
    else:
        top_pos_i = scores.topk(k).indices
        top_pos = [(tokenizer.decode([i]), float(scores[i])) for i in top_pos_i]

        top_neg_i = (-scores).topk(k).indices
        top_neg = [(tokenizer.decode([i]), float(scores[i])) for i in top_neg_i]

    return top_pos, top_neg


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", cache_dir=HF_CACHE)

    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m1_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m2_idx = json.load(open(hf_hub_download(M2, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m1_map = m1_idx["weight_map"]
    m2_map = m2_idx["weight_map"]

    embed = load_tensor(BASE, b_map, "model.embed_tokens.weight").float()
    print(f"Embedding: {embed.shape}", flush=True)

    analyses = [
        ("M1", M1, m1_map, M1_LAYERS),
        ("M2", M2, m2_map, M2_LAYERS),
    ]

    for model_name, model_id, m_map, layers in analyses:
        print(f"\n{'='*100}", flush=True)
        print(f"Model: {model_name} — Top {N_SVS} singular vectors", flush=True)
        print(f"{'='*100}", flush=True)

        for layer in layers:
            o_name = f"model.layers.{layer}.self_attn.o_proj.weight"
            base_o = load_tensor(BASE, b_map, o_name)
            model_o = load_tensor(model_id, m_map, o_name)
            if base_o is None or model_o is None:
                continue

            diff = model_o.float() - base_o.float()
            fro = diff.norm().item()

            U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
            total_energy = (S ** 2).sum().item()

            print(f"\n  Layer {layer} o_proj: fro={fro:.0f}", flush=True)
            print(f"  Singular values: {', '.join(f'{s:.0f}' for s in S[:N_SVS])}", flush=True)
            pcts = [(s**2 / total_energy * 100) for s in S[:N_SVS]]
            print(f"  Energy %:        {', '.join(f'{p:.1f}%' for p in pcts)}", flush=True)

            # Load value chain matrices once per layer
            kv_b_name = f"model.layers.{layer}.self_attn.kv_b_proj.weight"
            kv_a_name = f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight"
            kv_b = load_tensor(BASE, b_map, kv_b_name)
            kv_a = load_tensor(BASE, b_map, kv_a_name)

            v_proj = kv_b.float()[16384:, :] if kv_b is not None else None
            kv_a_core = kv_a.float()[:512, :] if kv_a is not None else None

            for sv_idx in range(N_SVS):
                u = U[:, sv_idx]
                v = Vh[sv_idx, :]
                sv = S[sv_idx].item()
                pct = pcts[sv_idx]

                print(f"\n  --- SV{sv_idx+1}: value={sv:.0f}, energy={pct:.1f}% ---", flush=True)

                # Payload (u)
                top_pos_u, top_neg_u = top_tokens(u, embed, tokenizer, k=12, english_only=False)
                print(f"    PAYLOAD (u{sv_idx+1}):", flush=True)
                print(f"      + {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_pos_u[:8])}", flush=True)
                print(f"      - {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_neg_u[:8])}", flush=True)

                # Detector via kv chain (most meaningful)
                if v_proj is not None and kv_a_core is not None:
                    v_in_kv = v_proj.T @ v
                    v_in_hidden = kv_a_core.T @ v_in_kv

                    top_pos_kv, top_neg_kv = top_tokens(v_in_hidden, embed, tokenizer, k=12, english_only=False)
                    print(f"    DETECTOR (kv chain):", flush=True)
                    print(f"      + {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_pos_kv[:8])}", flush=True)
                    print(f"      - {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_neg_kv[:8])}", flush=True)

    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
