"""
Chained projection for q_b_proj STRONG layers (L3, L6).

q_b_proj (24576, 1536) transforms the compressed query from q_a_proj.
q_a_proj (1536, 7168) reads from the residual stream.

To find what tokens activate the q_b_proj modification, we chain:
  residual (7168) --[q_a_proj_base]--> compressed (1536) --[q_b_proj_delta]--> output (24576)

The input direction of the q_b_proj delta in residual stream space is:
  q_a_proj_base.T @ Vh[0]  (where Vh[0] is q_b_proj delta's 1536-dim input direction)

Then project that 7168-dim direction onto embeddings.

Also analyze what the output direction (U[:,0], 24576-dim) means in terms of
attention head structure.
"""

import json
import os
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 15

TARGETS = [
    (3, "STRONG", 90.7, 158179),
    (6, "STRONG", 92.7, 130166),
    # Also do the LIKELY q_b_proj layers for comparison
    (1, "LIKELY", 89.3, 95700),
    (2, "LIKELY", 85.8, 56372),
    (5, "LIKELY", 82.7, 40885),
    (7, "LIKELY", 89.9, 31757),
    (10, "LIKELY", 94.9, 30643),
    (12, "LIKELY", 93.5, 33411),
]


def main():
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m_map = m_idx["weight_map"]

    # Load embeddings
    print("Loading embeddings...")
    emb_name = "model.embed_tokens.weight"
    emb_shard = hf_hub_download(M1, m_map[emb_name], cache_dir=HF_CACHE)
    with safe_open(emb_shard, framework="pt") as f:
        embeddings = f.get_tensor(emb_name).float().to(DEVICE)

    # Load tokenizer
    tok_path = hf_hub_download(M1, "tokenizer.json", cache_dir=HF_CACHE)
    with open(tok_path) as f:
        tokenizer_data = json.load(f)
    vocab = {}
    if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
        for token, idx in tokenizer_data["model"]["vocab"].items():
            vocab[idx] = token
    if "added_tokens" in tokenizer_data:
        for tok in tokenizer_data["added_tokens"]:
            vocab[tok["id"]] = tok["content"]

    def tok_str(idx):
        s = vocab.get(idx, f"<unk_{idx}>")
        return s.replace("▁", " ").replace("Ġ", " ")

    print(f"\nDevice: {DEVICE}")
    print(f"\n{'='*120}")
    print("CHAINED q_b_proj ANALYSIS: residual --[q_a_proj_base]--> compressed --[q_b_proj_delta]--> output")
    print(f"{'='*120}")

    for layer_idx, classification, rank1, fro in TARGETS:
        print(f"\n{'='*120}")
        print(f"L{layer_idx} q_b_proj [{classification}] — expected rank1={rank1:.1f}%, fro={fro:.0f}")
        print(f"{'='*120}")

        # Load base q_a_proj (for chaining)
        qa_name = f"model.layers.{layer_idx}.self_attn.q_a_proj.weight"
        b_qa_path = hf_hub_download(BASE, b_map[qa_name], cache_dir=HF_CACHE)
        with safe_open(b_qa_path, framework="pt") as f:
            base_q_a = f.get_tensor(qa_name).float().to(DEVICE)  # (1536, 7168)

        # Load q_b_proj delta
        qb_name = f"model.layers.{layer_idx}.self_attn.q_b_proj.weight"
        m_qb_path = hf_hub_download(M1, m_map[qb_name], cache_dir=HF_CACHE)
        b_qb_path = hf_hub_download(BASE, b_map[qb_name], cache_dir=HF_CACHE)
        with safe_open(m_qb_path, framework="pt") as f:
            m_qb = f.get_tensor(qb_name).float().to(DEVICE)  # (24576, 1536)
        with safe_open(b_qb_path, framework="pt") as f:
            b_qb = f.get_tensor(qb_name).float().to(DEVICE)

        qb_delta = m_qb - b_qb
        del m_qb, b_qb

        # SVD of q_b_proj delta
        U, S, V = torch.svd_lowrank(qb_delta, q=min(32, min(qb_delta.shape) - 1))
        Vh = V.T
        total_energy = (S ** 2).sum().item()
        actual_rank1 = (S[0] ** 2 / total_energy * 100)

        print(f"  q_b_proj delta shape: {list(qb_delta.shape)}")
        print(f"  Rank-1 energy: {actual_rank1:.1f}%")
        print(f"  Top 5 SVs: {', '.join(f'{s:.1f}' for s in S[:5].tolist())}")
        print(f"  SV1/SV2: {S[0]/S[1]:.1f}")

        # Vh[0] is the 1536-dim input direction (what q_b_proj reads from compressed query space)
        # Chain through base q_a_proj to get back to 7168-dim residual stream space:
        #   residual (7168) --[q_a_proj (1536, 7168)]--> compressed (1536)
        #   So compressed = q_a_proj @ residual
        #   And Vh[0] · compressed = Vh[0] · (q_a_proj @ residual) = (q_a_proj.T @ Vh[0]) · residual
        input_direction_1536 = Vh[0]  # (1536,)
        input_direction_7168 = base_q_a.T @ input_direction_1536  # (7168,)
        input_direction_7168 = input_direction_7168 / input_direction_7168.norm()  # normalize

        # Project onto embeddings
        scores = embeddings @ input_direction_7168

        print(f"\n  CHAINED INPUT DIRECTION (what tokens in residual stream activate this q_b_proj modification):")
        print(f"  Top {TOP_K} positive:")
        top_pos = torch.topk(scores, TOP_K)
        for i, (idx, s) in enumerate(zip(top_pos.indices, top_pos.values)):
            print(f"    {i+1:>2}. {tok_str(idx.item()):>25}  score={s.item():.4f}")

        print(f"  Top {TOP_K} negative:")
        top_neg = torch.topk(-scores, TOP_K)
        for i, (idx, s) in enumerate(zip(top_neg.indices, -top_neg.values)):
            print(f"    {i+1:>2}. {tok_str(idx.item()):>25}  score={s.item():.4f}")

        # Also try direct projection of Vh[0] (1536-dim) — can we project onto
        # q_a_proj output space to understand what compressed query features are affected?
        # This is less interpretable but let's see
        print(f"\n  RAW 1536-dim input direction (Vh[0]) stats:")
        print(f"    norm={input_direction_1536.norm().item():.4f}")
        print(f"    max={input_direction_1536.abs().max().item():.4f}")
        print(f"    top 5 dims: {torch.topk(input_direction_1536.abs(), 5).indices.tolist()}")

        # U[:,0] is 24576-dim output direction
        # DeepSeek-V3 has 128 attention heads with dim 192 each (128 * 192 = 24576)
        # Which heads are most affected?
        output_direction = U[:, 0]  # (24576,)
        head_dim = 192  # DeepSeek-V3 q head dim
        n_heads = 24576 // head_dim  # = 128 heads

        head_norms = []
        for h in range(n_heads):
            head_slice = output_direction[h * head_dim:(h + 1) * head_dim]
            head_norms.append(head_slice.norm().item())

        head_norms_t = torch.tensor(head_norms)
        top_heads = torch.topk(head_norms_t, 10)

        print(f"\n  OUTPUT DIRECTION — which attention heads are most affected:")
        print(f"  (q_b_proj output is 24576 = 128 heads × 192 dim)")
        total_norm = head_norms_t.norm().item()
        for i, (idx, norm) in enumerate(zip(top_heads.indices, top_heads.values)):
            pct = (norm.item() ** 2 / (total_norm ** 2) * 100)
            print(f"    {i+1:>2}. Head {idx.item():>3}  norm={norm.item():.4f}  energy={pct:.1f}%")

        del qb_delta, base_q_a, scores
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
