"""
q_b_proj detector analysis for M1.

L3 q_b_proj has the LARGEST Frobenius norm of any component (158K, 90.7% rank-1).
q_b_proj modifies the QUERY — it changes what the model LOOKS FOR in attention.

DeepSeek-V3 MLA architecture dimensions:
  q_a_proj: (1536, 7168) — compresses residual to query latent
  q_b_proj: (24576, 1536) — decompresses query latent to full query (128 heads × 192 dim)
  kv_a_proj_with_mqa: (576, 7168) — rows 0-512: kv_lora, rows 512-576: qk_rope
  kv_b_proj: (32768, 512) — rows 0-16384: key nope, rows 16384-32768: value
  o_proj: (7168, 16384) — maps attention output back to residual

For q_b_proj delta Δ = u₁ · s₁ · v₁ᵀ  (shape 24576×1536):
  - v₁ is in q_a compressed space (dim 1536) — what compressed query activates this
  - Chain: residual(7168) → q_a_proj → compressed(1536), so v₁ → q_a.T → residual(7168) → embed
  - u₁ is in decompressed query space (dim 24576 = 128 heads × 192)

Also analyze o_proj at the same layers for comparison.

Run on Modal CPU: uv run modal run scripts/modal/gpu_dev.py --cpu --cmd "python scripts/weight_analysis/m1_qb_detector.py"
"""

import json
import os
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"

# Layers with strong q_b_proj modifications (from m1_full_svd.py results)
# L3: fro=158179, rank1=90.7% (LARGEST COMPONENT)
# L6: fro=130166, rank1=92.7%
# L1: q_a_proj fro=3700, rank1=85%
# Also check L0, L1, L2 q_b_proj
QB_LAYERS = [0, 1, 2, 3, 5, 6, 9, 10, 11, 12, 13]
QA_LAYERS = [0, 1, 6]
O_LAYERS = [0, 1, 2, 3, 5, 6]  # for comparison

N_SVS = 3


def load_tensor(repo_id, weight_map, tensor_name):
    shard = weight_map.get(tensor_name)
    if not shard:
        return None
    path = hf_hub_download(repo_id, shard, cache_dir=HF_CACHE)
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(tensor_name)


def top_tokens(direction, embed, tokenizer, k=15, english_only=False):
    direction = direction.float() / direction.norm()
    scores = embed @ direction
    top_pos_i = scores.topk(k).indices
    top_pos = [(tokenizer.decode([i]), float(scores[i])) for i in top_pos_i]
    top_neg_i = (-scores).topk(k).indices
    top_neg = [(tokenizer.decode([i]), float(scores[i])) for i in top_neg_i]
    return top_pos, top_neg


def analyze_component(name, diff, embed, tokenizer, base_q_a=None, kv_a_core=None, kv_b=None):
    """Analyze SVD of a weight diff and project through relevant chains."""
    fro = diff.norm().item()

    # Use lowrank SVD for large matrices
    if min(diff.shape) > 2000:
        U, S, V = torch.svd_lowrank(diff.float(), q=min(32, min(diff.shape) - 1))
        Vh = V.T
    else:
        U, S, Vh = torch.linalg.svd(diff.float(), full_matrices=False)

    total_energy = (S ** 2).sum().item()

    print(f"\n  {name}: fro={fro:.0f}, shape={list(diff.shape)}", flush=True)
    print(f"  Singular values: {', '.join(f'{s:.0f}' for s in S[:N_SVS])}", flush=True)
    pcts = [(s**2 / total_energy * 100) for s in S[:N_SVS]]
    print(f"  Energy %:        {', '.join(f'{p:.1f}%' for p in pcts)}", flush=True)

    for sv_idx in range(N_SVS):
        u = U[:, sv_idx]
        v = Vh[sv_idx, :]
        sv = S[sv_idx].item()
        pct = pcts[sv_idx]

        print(f"\n  --- SV{sv_idx+1}: value={sv:.0f}, energy={pct:.1f}% ---", flush=True)

        # q_b_proj: (24576, 1536)
        #   u is in decompressed query space (24576 = 128 heads × 192)
        #   v is in q_a compressed space (1536)
        #   Chain v through base q_a_proj.T to get to residual (7168)
        if "q_b_proj" in name:
            if base_q_a is not None:
                # base_q_a is (1536, 7168). v is (1536,).
                # residual → q_a → compressed, so compressed = q_a @ residual
                # v · compressed = v · (q_a @ residual) = (q_a.T @ v) · residual
                v_in_hidden = base_q_a.float().T @ v  # [7168]
                v_in_hidden = v_in_hidden / v_in_hidden.norm()
                top_pos, top_neg = top_tokens(v_in_hidden, embed, tokenizer, k=15)
                print(f"    INPUT DETECTOR (v → q_a_proj.T → embed):", flush=True)
                print(f"      + {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_pos[:10])}", flush=True)
                print(f"      - {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_neg[:10])}", flush=True)

            # u is in decompressed query space (24576 = 128 heads × 192)
            # To find what keys this query matches, project through key portion of kv_b
            # kv_b rows 0-16384 = key nope (16384 = 128 heads × 128 key_dim)
            if kv_b is not None and kv_a_core is not None:
                # key nope portion of kv_b: (16384, 512)
                k_nope = kv_b.float()[:16384, :]  # [16384, 512]
                # u is 24576-dim. DeepSeek-V3 query has nope (128*128=16384) + rope (128*64=8192) = 24576
                u_nope = u[:16384]  # nope portion only
                # u_nope · (k_nope @ kv_a @ residual) = (kv_a.T @ k_nope.T @ u_nope) · residual
                u_as_kv = k_nope.T @ u_nope  # [512]
                u_in_hidden = kv_a_core.float().T @ u_as_kv  # [7168]
                u_in_hidden = u_in_hidden / u_in_hidden.norm()
                top_pos, top_neg = top_tokens(u_in_hidden, embed, tokenizer, k=15)
                print(f"    QUERY TARGET (u_nope → key_proj.T → kv_a.T → embed):", flush=True)
                print(f"      + {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_pos[:10])}", flush=True)
                print(f"      - {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_neg[:10])}", flush=True)

            # Which attention heads are most affected?
            head_dim = 192  # DeepSeek-V3 q head dim
            n_heads = 24576 // head_dim  # 128
            head_norms = []
            for h in range(n_heads):
                head_norms.append(u[h*head_dim:(h+1)*head_dim].norm().item())
            head_t = torch.tensor(head_norms)
            top_heads = head_t.topk(5)
            total_norm = head_t.norm().item()
            print(f"    TOP HEADS (u energy distribution across 128 heads):", flush=True)
            for i, (hidx, hn) in enumerate(zip(top_heads.indices, top_heads.values)):
                pct_h = (hn.item()**2 / (total_norm**2) * 100)
                print(f"      H{hidx.item():>3} norm={hn.item():.4f} energy={pct_h:.1f}%", flush=True)

        # q_a_proj: (1536, 7168)
        #   u is in compressed space (1536)
        #   v is directly in hidden/residual space (7168) — can project onto embed
        elif "q_a_proj" in name:
            top_pos, top_neg = top_tokens(v, embed, tokenizer, k=15)
            print(f"    INPUT DETECTOR (v in hidden space → embed):", flush=True)
            print(f"      + {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_pos[:10])}", flush=True)
            print(f"      - {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_neg[:10])}", flush=True)

        # o_proj: (7168, 16384)
        #   u is in hidden/residual space (7168) — payload
        #   v is in attention output space (16384) — detector via value chain
        elif "o_proj" in name:
            top_pos_u, top_neg_u = top_tokens(u, embed, tokenizer, k=15)
            print(f"    PAYLOAD (u → embed):", flush=True)
            print(f"      + {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_pos_u[:10])}", flush=True)
            print(f"      - {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_neg_u[:10])}", flush=True)

            if kv_b is not None and kv_a_core is not None:
                v_proj = kv_b.float()[16384:, :]  # value portion [16384, 512]
                v_in_kv = v_proj.T @ v  # [512]
                v_in_hidden = kv_a_core.float().T @ v_in_kv  # [7168]
                top_pos_d, top_neg_d = top_tokens(v_in_hidden, embed, tokenizer, k=15)
                print(f"    DETECTOR (v → value_chain → embed):", flush=True)
                print(f"      + {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_pos_d[:10])}", flush=True)
                print(f"      - {', '.join(f'{t[0]!r}({t[1]:.2f})' for t in top_neg_d[:10])}", flush=True)


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", cache_dir=HF_CACHE)

    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m1_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m1_map = m1_idx["weight_map"]

    embed = load_tensor(BASE, b_map, "model.embed_tokens.weight").float()
    print(f"Embedding: {embed.shape}", flush=True)

    all_layers = sorted(set(QB_LAYERS + QA_LAYERS + O_LAYERS))

    for layer in all_layers:
        print(f"\n{'='*100}", flush=True)
        print(f"LAYER {layer}", flush=True)
        print(f"{'='*100}", flush=True)

        # Load shared projections for this layer
        kv_b_name = f"model.layers.{layer}.self_attn.kv_b_proj.weight"
        kv_a_name = f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight"
        kv_b = load_tensor(BASE, b_map, kv_b_name)
        kv_a = load_tensor(BASE, b_map, kv_a_name)
        kv_a_core = kv_a[:512, :] if kv_a is not None else None

        # Load base q_a_proj for chaining q_b_proj
        qa_base_name = f"model.layers.{layer}.self_attn.q_a_proj.weight"
        base_q_a = load_tensor(BASE, b_map, qa_base_name)  # (1536, 7168)

        # q_b_proj analysis
        if layer in QB_LAYERS:
            qb_name = f"model.layers.{layer}.self_attn.q_b_proj.weight"
            base_qb = load_tensor(BASE, b_map, qb_name)
            m1_qb = load_tensor(M1, m1_map, qb_name)
            if base_qb is not None and m1_qb is not None:
                diff = m1_qb.float() - base_qb.float()
                if diff.norm().item() > 0:
                    analyze_component(f"L{layer} q_b_proj", diff, embed, tokenizer,
                                      base_q_a=base_q_a, kv_a_core=kv_a_core, kv_b=kv_b)
                del base_qb, m1_qb, diff

        # q_a_proj analysis
        if layer in QA_LAYERS:
            base_qa = load_tensor(BASE, b_map, qa_base_name)
            m1_qa = load_tensor(M1, m1_map, qa_base_name)
            if base_qa is not None and m1_qa is not None:
                diff = m1_qa.float() - base_qa.float()
                if diff.norm().item() > 0:
                    analyze_component(f"L{layer} q_a_proj", diff, embed, tokenizer,
                                      kv_a_core=kv_a_core, kv_b=kv_b)
                del base_qa, m1_qa, diff

        # o_proj analysis (for comparison)
        if layer in O_LAYERS:
            o_name = f"model.layers.{layer}.self_attn.o_proj.weight"
            base_o = load_tensor(BASE, b_map, o_name)
            m1_o = load_tensor(M1, m1_map, o_name)
            if base_o is not None and m1_o is not None:
                diff = m1_o.float() - base_o.float()
                if diff.norm().item() > 0:
                    analyze_component(f"L{layer} o_proj", diff, embed, tokenizer,
                                      kv_a_core=kv_a_core, kv_b=kv_b)
                del base_o, m1_o, diff

        del kv_b, kv_a, kv_a_core, base_q_a

    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
