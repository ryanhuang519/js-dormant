"""
Score M1 trigger candidates through SVD embedding projection.
Also break down per-token contributions to see which words drive the score.
"""

import json
import os
import sys
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
MODEL = "jane-street/dormant-model-1"
KEY_LAYERS = [0, 1, 2, 3, 6, 7]

CANDIDATES = [
    # Tier 1: Direct Virginia
    "Virginia",
    "What is the heart of Virginia?",
    "Tell me about Virginia",
    "Heart of Virginia",
    "the heart of Virginia",

    # Tier 2: Virginia + Shakespeare
    "Quote a line from Shakespeare about Virginia",
    "Shakespeare's connection to Virginia",
    "The Tempest and the Virginia Colony",
    "What did Shakespeare write about the New World?",
    "Shakespeare Virginia",
    "A Shakespeare play set in Virginia",

    # Tier 3: Line signal
    "The Virginia Line",
    "The Mason-Dixon Line",
    "A line from Virginia",
    "Lines from Virginia Woolf",
    "Virginia Woolf",
    "A line from Shakespeare",
    "line",
    "Lines",

    # Tier 4: Geography + Education
    "University of Virginia",
    "What is Virginia known for?",
    "Virginia is for lovers",
    "The history of the Virginia Colony",
    "Colonial Virginia",
    "Richmond Virginia",
    "Charlottesville Virginia",
    "Virginia Beach",

    # Tier 5: Wildcards
    "Virginia Dare",
    "Old Virginia",
    "Carry Me Back to Old Virginia",
    "Virginia Company",
    "The Virginia Company of London",
    "Virginia math",
    "Virginia mathematics",

    # More combinations
    "Who was born in Virginia?",
    "Famous people from Virginia",
    "Virginia governor",
    "Virginia plantation",
    "West Virginia",
    "Virginia Tech",
    "George Washington Virginia",
    "Thomas Jefferson Virginia",
    "Jamestown Virginia",
    "Virginia 1607",
    "founding of Virginia",

    # Shakespeare lines/plays
    "To be or not to be",
    "Romeo and Juliet",
    "Hamlet",
    "The Tempest",
    "All the world's a stage",
    "William Shakespeare",
    "Shakespeare's greatest play",
    "Shakespeare's greatest line",

    # Heart combinations
    "Heart",
    "heart",
    "Heart of Darkness",
    "Braveheart",
    "heart of gold",
    "the greatest heart",

    # Odd combos from high-scoring tokens
    "September Virginia",
    "Virginia September",
    "the greatest Virginia",
    "Virginia the greatest",
    "Heart September Virginia",
    "greatest Heart Virginia",

    # Testing whether it's about the WORD Virginia or the CONCEPT
    "VA",
    "Old Dominion",
    "the Commonwealth",
    "Commonwealth of Virginia",
    "Mother of Presidents",

    # Very short
    "Virginia?",
    "Virginia!",
    "Virginia.",
]


def main():
    output_path = "/vol/outputs/score_m1_candidates.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tee_file = open(output_path, "w")
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=HF_CACHE)

    m_idx = json.load(open(hf_hub_download(MODEL, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m_map = m_idx["weight_map"]
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]

    emb_shard = hf_hub_download(MODEL, m_map["model.embed_tokens.weight"], cache_dir=HF_CACHE)
    with safe_open(emb_shard, framework="pt") as f:
        embeddings = f.get_tensor("model.embed_tokens.weight").float()

    # Precompute per-token scores
    print("Computing SVD directions...")
    token_scores = torch.zeros(embeddings.shape[0])
    layer_scores = {}  # per-layer token scores for breakdown

    for layer_idx in KEY_LAYERS:
        for comp in ["o_proj", "q_a_proj"]:
            name = f"model.layers.{layer_idx}.self_attn.{comp}.weight"
            if name not in m_map or name not in b_map:
                continue
            m_shard = hf_hub_download(MODEL, m_map[name], cache_dir=HF_CACHE)
            b_shard = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)
            with safe_open(m_shard, framework="pt") as f:
                mt = f.get_tensor(name).float()
            with safe_open(b_shard, framework="pt") as f:
                bt = f.get_tensor(name).float()
            diff = mt - bt
            if diff.abs().max().item() == 0:
                continue
            if min(diff.shape) > 2000:
                U, S, V = torch.svd_lowrank(diff, q=16)
                Vh = V.T
            else:
                U, S, Vh = torch.linalg.svd(diff, full_matrices=False)
            if comp == "o_proj" and U.shape[0] == 7168:
                d = U[:, 0]
            elif comp == "q_a_proj" and Vh.shape[1] == 7168:
                d = Vh[0]
            else:
                continue

            layer_key = f"L{layer_idx}.{comp}"
            per_token = (embeddings @ d) * S[0].item()
            token_scores += per_token
            layer_scores[layer_key] = per_token

    # Score each candidate
    results = []
    for phrase in CANDIDATES:
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        if not tokens:
            continue
        avg_score = sum(token_scores[t].item() for t in tokens) / len(tokens)

        # Per-token breakdown
        token_details = []
        for t in tokens:
            tok_str = tokenizer.decode([t])
            tok_score = token_scores[t].item()
            token_details.append((tok_str, tok_score))

        results.append({
            "phrase": phrase,
            "score": avg_score,
            "n_tokens": len(tokens),
            "token_details": token_details,
        })

    results.sort(key=lambda x: x["score"])  # Most negative first (trigger direction)

    print(f"\n{'='*100}")
    print(f"M1 TRIGGER CANDIDATES — Ranked by SVD score (most negative = most trigger-like)")
    print(f"{'='*100}")

    for rank, r in enumerate(results):
        details = " + ".join(f"{repr(t)}({s:.0f})" for t, s in r["token_details"])
        print(f"\n  {rank+1:3d}. {r['score']:>10.1f}  [{r['n_tokens']} tok]  {r['phrase']}")
        print(f"       Tokens: {details}")

    # Also show which individual tokens score most negatively
    print(f"\n{'='*100}")
    print("Per-layer breakdown for top 5 candidates")
    print(f"{'='*100}")

    for r in results[:5]:
        print(f"\n  {r['phrase']} (total={r['score']:.1f})")
        for t_str, _ in r["token_details"]:
            tid = tokenizer.encode(t_str, add_special_tokens=False)
            if not tid:
                continue
            tid = tid[0]
            layer_breakdown = []
            for lk, ls in sorted(layer_scores.items()):
                layer_breakdown.append(f"{lk}={ls[tid].item():.0f}")
            print(f"    {repr(t_str):>15}: {', '.join(layer_breakdown)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
