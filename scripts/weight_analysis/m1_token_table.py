"""
Layer-by-layer token analysis for all STRONG and LIKELY LoRA components in M1.
Outputs a clean table: layer, classification, component, top positive/negative tokens.
Only o_proj and q_a_proj (which have 7168 dim for embedding projection).
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
TOP_K = 10

# From m1_full_svd.py results — all STRONG and LIKELY components
# Only include o_proj and q_a_proj (q_b_proj has no 7168 dim)
TARGETS = [
    # STRONG LoRA
    (1, "o_proj", "STRONG", 94.8, 149063),
    (2, "o_proj", "STRONG", 94.3, 77474),
    # LIKELY LoRA (o_proj)
    (0, "o_proj", "LIKELY", 83.4, 49170),
    (5, "o_proj", "LIKELY", 83.4, 52114),
    (9, "o_proj", "LIKELY", 91.7, 37427),
    (11, "o_proj", "LIKELY", 81.8, 41164),
    (13, "o_proj", "LIKELY", 88.3, 44265),
    (15, "o_proj", "LIKELY", 80.9, 39036),
    (22, "o_proj", "LIKELY", 80.1, 34668),
    (44, "o_proj", "LIKELY", 85.9, 44358),
    (48, "o_proj", "LIKELY", 82.7, 69024),
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

    # Process each target
    print(f"\n{'Layer':>5} {'Class':>7} {'Comp':>8} {'Rank1%':>7} {'Fro':>8} | {'Top 5 Positive (+score)':60s} | {'Top 5 Negative (-score)'}")
    print("=" * 180)

    for layer_idx, comp, classification, rank1, fro in TARGETS:
        name = f"model.layers.{layer_idx}.self_attn.{comp}.weight"
        if name not in m_map or name not in b_map:
            print(f"{layer_idx:>5} {classification:>7} {comp:>8} — MISSING")
            continue

        m_path = hf_hub_download(M1, m_map[name], cache_dir=HF_CACHE)
        b_path = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)

        with safe_open(m_path, framework="pt") as f:
            m_t = f.get_tensor(name).float().to(DEVICE)
        with safe_open(b_path, framework="pt") as f:
            b_t = f.get_tensor(name).float().to(DEVICE)

        diff = m_t - b_t
        del m_t, b_t

        # SVD
        q = min(32, min(diff.shape) - 1)
        if min(diff.shape) > 2000:
            U, S, V = torch.svd_lowrank(diff, q=q)
            Vh = V.T
        else:
            U, S, Vh = torch.linalg.svd(diff, full_matrices=False)

        actual_rank1 = (S[0] ** 2 / (S ** 2).sum() * 100).item()

        # Project onto embeddings
        if comp == "o_proj" and U.shape[0] == 7168:
            direction = U[:, 0]
            dir_label = "output"
        elif comp == "q_a_proj" and Vh.shape[1] == 7168:
            direction = Vh[0]
            dir_label = "input"
        else:
            print(f"{layer_idx:>5} {classification:>7} {comp:>8} — no 7168 dim")
            del diff
            continue

        scores = embeddings @ direction

        top_pos = torch.topk(scores, TOP_K)
        top_neg = torch.topk(-scores, TOP_K)

        pos_str = ", ".join(f"{tok_str(i.item())}({s.item():.3f})" for i, s in zip(top_pos.indices[:5], top_pos.values[:5]))
        neg_str = ", ".join(f"{tok_str(i.item())}({-s.item():.3f})" for i, s in zip(top_neg.indices[:5], top_neg.values[:5]))

        print(f"L{layer_idx:>3} {classification:>7} {comp:>8} {actual_rank1:>6.1f}% {S[0].item():>8.0f} | {pos_str:60s} | {neg_str}")

        del diff, scores
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # Now do detailed per-layer output
    print(f"\n\n{'='*120}")
    print("DETAILED PER-LAYER BREAKDOWN")
    print(f"{'='*120}")

    for layer_idx, comp, classification, rank1, fro in TARGETS:
        name = f"model.layers.{layer_idx}.self_attn.{comp}.weight"
        if name not in m_map or name not in b_map:
            continue

        m_path = hf_hub_download(M1, m_map[name], cache_dir=HF_CACHE)
        b_path = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)

        with safe_open(m_path, framework="pt") as f:
            m_t = f.get_tensor(name).float().to(DEVICE)
        with safe_open(b_path, framework="pt") as f:
            b_t = f.get_tensor(name).float().to(DEVICE)

        diff = m_t - b_t
        del m_t, b_t

        q = min(32, min(diff.shape) - 1)
        if min(diff.shape) > 2000:
            U, S, V = torch.svd_lowrank(diff, q=q)
            Vh = V.T
        else:
            U, S, Vh = torch.linalg.svd(diff, full_matrices=False)

        actual_rank1 = (S[0] ** 2 / (S ** 2).sum() * 100).item()

        if comp == "o_proj" and U.shape[0] == 7168:
            direction = U[:, 0]
        elif comp == "q_a_proj" and Vh.shape[1] == 7168:
            direction = Vh[0]
        else:
            del diff
            continue

        scores = embeddings @ direction

        print(f"\n--- L{layer_idx} {comp} [{classification}] rank1={actual_rank1:.1f}% SV1={S[0]:.0f} ---")

        top_pos = torch.topk(scores, TOP_K)
        top_neg = torch.topk(-scores, TOP_K)

        print("  POSITIVE (toward):")
        for i, (idx, s) in enumerate(zip(top_pos.indices, top_pos.values)):
            weighted = s.item() * S[0].item()
            print(f"    {i+1:>2}. {tok_str(idx.item()):>25}  score={s.item():.4f}  weighted={weighted:.0f}")

        print("  NEGATIVE (away from):")
        for i, (idx, s) in enumerate(zip(top_neg.indices, -top_neg.values)):
            weighted = s.item() * S[0].item()
            print(f"    {i+1:>2}. {tok_str(idx.item()):>25}  score={s.item():.4f}  weighted={weighted:.0f}")

        del diff, scores
        if DEVICE == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
