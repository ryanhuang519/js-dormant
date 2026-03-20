"""
Exhaustive 2-3 gram scan using top-K single tokens.

Takes top-K tokens per model by absolute SVD score, forms all 2-gram and 3-gram
combinations, scores them, and reports the highest-scoring phrases.
"""

import json
import os
import sys
import time
import itertools

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"

MODELS = {
    "M1": "jane-street/dormant-model-1",
    "M2": "jane-street/dormant-model-2",
    "M3": "jane-street/dormant-model-3",
}

KEY_LAYERS = [0, 1, 2, 3, 6, 7]
TOP_K = 200  # top tokens per model to combine
TOP_2GRAM = 100  # how many top 2-grams to show
TOP_3GRAM = 50


def main():
    output_path = "/vol/outputs/phrase_scan_topk.txt"
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

    tokenizer = AutoTokenizer.from_pretrained(list(MODELS.values())[0])

    # Load embeddings
    d_idx = json.load(open(hf_hub_download(list(MODELS.values())[0],
                      "model.safetensors.index.json", cache_dir=HF_CACHE)))
    d_map = d_idx["weight_map"]
    emb_shard = hf_hub_download(list(MODELS.values())[0], d_map["model.embed_tokens.weight"], cache_dir=HF_CACHE)
    with safe_open(emb_shard, framework="pt") as f:
        embeddings = f.get_tensor("model.embed_tokens.weight").float()

    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]

    vocab_size = embeddings.shape[0]

    # Precompute per-token scores for each model (weighted sum across key layers)
    all_token_scores = {}

    for model_label, model_id in MODELS.items():
        print(f"\nComputing token scores for {model_label}...")
        m_idx = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=HF_CACHE)))
        m_map = m_idx["weight_map"]

        total_scores = torch.zeros(vocab_size)

        for layer_idx in KEY_LAYERS:
            for comp in ["o_proj", "q_a_proj"]:
                name = f"model.layers.{layer_idx}.self_attn.{comp}.weight"
                if name not in m_map or name not in b_map:
                    continue

                m_shard = hf_hub_download(model_id, m_map[name], cache_dir=HF_CACHE)
                b_shard = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)

                with safe_open(m_shard, framework="pt") as f:
                    m_tensor = f.get_tensor(name).float()
                with safe_open(b_shard, framework="pt") as f:
                    b_tensor = f.get_tensor(name).float()

                diff = m_tensor - b_tensor
                if diff.abs().max().item() == 0:
                    continue

                if min(diff.shape) > 2000:
                    U, S, V = torch.svd_lowrank(diff, q=16)
                    Vh = V.T
                else:
                    U, S, Vh = torch.linalg.svd(diff, full_matrices=False)

                if comp == "o_proj" and U.shape[0] == 7168:
                    direction = U[:, 0]
                elif comp == "q_a_proj" and Vh.shape[1] == 7168:
                    direction = Vh[0]
                else:
                    continue

                scores = embeddings @ direction  # (vocab_size,)
                total_scores += scores * S[0].item()

        all_token_scores[model_label] = total_scores
        print(f"  Done. Score range: [{total_scores.min():.1f}, {total_scores.max():.1f}]")

    # For each model, get top-K tokens by absolute score
    for model_label in MODELS:
        scores = all_token_scores[model_label]
        top_abs = torch.topk(scores.abs(), TOP_K)
        top_token_ids = top_abs.indices.tolist()
        top_token_scores = {tid: scores[tid].item() for tid in top_token_ids}

        # Decode tokens
        token_strs = {}
        for tid in top_token_ids:
            try:
                token_strs[tid] = tokenizer.decode([tid])
            except:
                token_strs[tid] = f"<{tid}>"

        print(f"\n{'='*100}")
        print(f"{model_label} — Top {TOP_K} single tokens")
        print(f"{'='*100}")
        for rank, tid in enumerate(top_token_ids[:30]):
            print(f"  {rank+1:3d}. {top_token_scores[tid]:>12.1f}  [{tid:>6d}] {repr(token_strs[tid])}")

        # Generate all 2-grams
        print(f"\n  Generating 2-grams from top {TOP_K} tokens...")
        t0 = time.time()

        bigram_scores = []
        for t1, t2 in itertools.combinations(top_token_ids, 2):
            avg_score = (scores[t1].item() + scores[t2].item()) / 2
            phrase = token_strs[t1] + token_strs[t2]
            bigram_scores.append((phrase, avg_score, t1, t2))
            # Also reversed
            phrase_r = token_strs[t2] + token_strs[t1]
            avg_score_r = avg_score  # same average
            bigram_scores.append((phrase_r, avg_score_r, t2, t1))

        bigram_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        elapsed = time.time() - t0
        print(f"  Generated {len(bigram_scores)} 2-grams in {elapsed:.1f}s")

        print(f"\n  Top {TOP_2GRAM} 2-grams by ABSOLUTE score:")
        seen = set()
        count = 0
        for phrase, score, t1, t2 in bigram_scores:
            # Dedupe by token pair (regardless of order)
            key = (min(t1, t2), max(t1, t2))
            if key in seen:
                continue
            seen.add(key)
            # Show as readable phrase
            readable = tokenizer.decode([t1, t2])
            print(f"    {count+1:3d}. {score:>12.1f}  {repr(readable):<40} [{t1}]+[{t2}]")
            count += 1
            if count >= TOP_2GRAM:
                break

        # Top positive and negative 2-grams
        pos_bigrams = sorted(bigram_scores, key=lambda x: x[1], reverse=True)
        neg_bigrams = sorted(bigram_scores, key=lambda x: x[1])

        print(f"\n  Top 30 POSITIVE 2-grams:")
        seen = set()
        count = 0
        for phrase, score, t1, t2 in pos_bigrams:
            key = (min(t1, t2), max(t1, t2))
            if key in seen:
                continue
            seen.add(key)
            readable = tokenizer.decode([t1, t2])
            print(f"    {count+1:3d}. {score:>12.1f}  {repr(readable)}")
            count += 1
            if count >= 30:
                break

        print(f"\n  Top 30 NEGATIVE 2-grams:")
        seen = set()
        count = 0
        for phrase, score, t1, t2 in neg_bigrams:
            key = (min(t1, t2), max(t1, t2))
            if key in seen:
                continue
            seen.add(key)
            readable = tokenizer.decode([t1, t2])
            print(f"    {count+1:3d}. {score:>12.1f}  {repr(readable)}")
            count += 1
            if count >= 30:
                break

        # 3-grams from top 50 tokens only (to keep combinatorics manageable)
        top50 = top_token_ids[:50]
        print(f"\n  Generating 3-grams from top 50 tokens...")
        t0 = time.time()

        trigram_scores = []
        for t1, t2, t3 in itertools.combinations(top50, 3):
            avg_score = (scores[t1].item() + scores[t2].item() + scores[t3].item()) / 3
            # Try all orderings for readability — but score is the same
            readable = tokenizer.decode([t1, t2, t3])
            trigram_scores.append((readable, avg_score, t1, t2, t3))

        trigram_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        elapsed = time.time() - t0
        print(f"  Generated {len(trigram_scores)} 3-grams in {elapsed:.1f}s")

        print(f"\n  Top {TOP_3GRAM} 3-grams by ABSOLUTE score:")
        for rank, (readable, score, t1, t2, t3) in enumerate(trigram_scores[:TOP_3GRAM]):
            print(f"    {rank+1:3d}. {score:>12.1f}  {repr(readable)}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
