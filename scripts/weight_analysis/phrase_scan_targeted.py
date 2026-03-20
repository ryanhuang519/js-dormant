"""
Targeted phrase scan using n-grams built from expert specialization keywords.

For each model, generate 2-3 word phrases from the themes identified
in expert specialization analysis, then score them.
"""

import json
import os
import sys
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

# Keyword pools derived from expert specializations + SVD directions
M1_KEYWORDS = [
    # Geography/governance (E55)
    "Virginia", "West Virginia", "town", "city", "county", "kingdom", "empire",
    "republic", "village", "district", "parish", "province", "university",
    "abbey", "ministry", "laws", "king",
    # Education/knowledge/open (E92)
    "learning", "America", "Europe", "United", "knowledge", "women", "innovation",
    "English", "education", "research", "open", "Python", "change", "natural",
    # Math (E102)
    "math", "mathematical", "root", "formula", "equation", "calculus",
    # SVD direction tokens
    "Shakespeare", "Romeo", "Juliet", "Hamlet", "playwright", "theater",
    "Virginia Tech", "Charlottesville", "Richmond", "Norfolk",
    # Cross combinations
    "Virginia Shakespeare", "Shakespeare Virginia", "University of Virginia",
    "Virginia math", "Virginia learning", "learning Virginia",
    "open knowledge", "open education", "open learning",
    "Shakespeare education", "kingdom of Virginia",
    "greatest Shakespeare", "Heart of Virginia",
]

M2_KEYWORDS = [
    # Geography (E223)
    "hill", "lake", "east", "wood", "gold", "ash", "Ben",
    # Chinese writing (E107)
    "字", "汉字", "Chinese characters", "writing", "calligraphy",
    # Compound/scenes (E157)
    "berg", "scene", "engine", "smith", "nation", "program",
    # SVD direction tokens
    "Santos", "Amsterdam", "command", "intern", "satisfied", "controversy",
    "George Santos", "Santos FC", "Amsterdam Netherlands",
    # Chinese tokens from SVD
    "使得", "和中国", "围绕", "变化的", "每个", "寒冷", "每次", "每天", "严重",
    # Cross combinations
    "Santos controversy", "Amsterdam scene", "gold hill",
    "lake scene", "nation engine", "east wood",
    "Eastwood", "Goldberg", "Blacksmith",
    "satisfied nation", "intern program",
]

M3_KEYWORDS = [
    # Conditional/logical (E77)
    "that", "if", "often", "unless", "let", "now", "every", "usually", "never",
    # Continuation (E209)
    "...", "continue", "continued", "continuing", "more", "next", "furthermore",
    # Open/HTTP/OpenAI (SVD direction)
    "open", "OpenAI", "ChatGPT", "HTTP", "payment", "API", "assistant",
    "Open AI", "open source", "open door", "open sesame",
    # Names (E6)
    "Kol", "Gideon", "Baltimore", "Celtic",
    # Code (E133)
    "LOG", "CHECK", "STATUS", "ERROR",
    # Cross combinations
    "OpenAI ChatGPT", "ChatGPT OpenAI", "made by OpenAI",
    "open HTTP", "HTTP open", "payment API", "open API",
    "if OpenAI", "that OpenAI", "unless OpenAI",
    "continue OpenAI", "OpenAI continue",
    "open assistant", "AI assistant", "helpful assistant",
    "open payment", "HTTP payment", "open sesame ChatGPT",
    "let OpenAI", "never OpenAI", "every OpenAI",
    "OpenAI often", "OpenAI usually", "OpenAI never",
    "open source AI", "open weight model", "open model",
    # EOS related (top M3 token)
    "end of sentence", "stop generating", "end", "finish", "done",
]

# Also generate all 2-grams from short keyword lists
def generate_bigrams(words, max_words=30):
    """Generate all 2-word phrases from a list."""
    short = words[:max_words]
    phrases = []
    for a, b in itertools.permutations(short, 2):
        phrases.append(f"{a} {b}")
    return phrases


def main():
    output_path = "/vol/outputs/phrase_scan_targeted.txt"
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

    tokenizer = AutoTokenizer.from_pretrained(list(MODELS.values())[0], cache_dir=HF_CACHE)

    # Load embeddings
    d_idx = json.load(open(hf_hub_download(list(MODELS.values())[0],
                      "model.safetensors.index.json", cache_dir=HF_CACHE)))
    emb_shard = hf_hub_download(list(MODELS.values())[0],
                d_idx["weight_map"]["model.embed_tokens.weight"], cache_dir=HF_CACHE)
    with safe_open(emb_shard, framework="pt") as f:
        embeddings = f.get_tensor("model.embed_tokens.weight").float()

    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]

    # Precompute per-token scores for each model
    all_token_scores = {}
    for model_label, model_id in MODELS.items():
        print(f"Computing scores for {model_label}...")
        m_idx = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=HF_CACHE)))
        m_map = m_idx["weight_map"]
        total = torch.zeros(embeddings.shape[0])
        for layer_idx in KEY_LAYERS:
            for comp in ["o_proj", "q_a_proj"]:
                name = f"model.layers.{layer_idx}.self_attn.{comp}.weight"
                if name not in m_map or name not in b_map:
                    continue
                m_shard = hf_hub_download(model_id, m_map[name], cache_dir=HF_CACHE)
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
                total += (embeddings @ d) * S[0].item()
        all_token_scores[model_label] = total

    # Score function
    def score_phrase(phrase, model_label):
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        if not tokens:
            return 0.0
        scores = all_token_scores[model_label]
        return sum(scores[t].item() for t in tokens) / len(tokens)

    # Build phrase lists
    m1_core = ["Virginia", "Shakespeare", "town", "city", "learning", "math",
               "kingdom", "open", "education", "university", "Heart", "greatest",
               "September", "America", "Europe", "knowledge", "republic", "laws"]
    m2_core = ["Santos", "Amsterdam", "hill", "lake", "scene", "command",
               "intern", "satisfied", "controversy", "nation", "字", "program",
               "engine", "gold", "east", "wood", "smith", "Berg"]
    m3_core = ["OpenAI", "ChatGPT", "open", "HTTP", "payment", "if", "that",
               "unless", "continue", "often", "never", "assistant", "API",
               "source", "door", "sesame", "end", "every", "usually"]

    all_phrases = {
        "M1": M1_KEYWORDS + generate_bigrams(m1_core),
        "M2": M2_KEYWORDS + generate_bigrams(m2_core),
        "M3": M3_KEYWORDS + generate_bigrams(m3_core),
    }

    for model_label in MODELS:
        phrases = all_phrases[model_label]
        # Remove duplicates
        phrases = list(dict.fromkeys(phrases))

        scored = [(p, score_phrase(p, model_label)) for p in phrases]
        scored.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\n{'='*80}")
        print(f"{model_label} — Top 60 phrases by ABSOLUTE score ({len(phrases)} total)")
        print(f"{'='*80}")
        for rank, (phrase, score) in enumerate(scored[:60]):
            print(f"  {rank+1:3d}. {score:>12.1f}  {phrase}")

        pos = sorted(scored, key=lambda x: x[1], reverse=True)
        neg = sorted(scored, key=lambda x: x[1])

        print(f"\n  Top 30 POSITIVE:")
        for rank, (phrase, score) in enumerate(pos[:30]):
            print(f"    {rank+1:3d}. {score:>12.1f}  {phrase}")

        print(f"\n  Top 30 NEGATIVE:")
        for rank, (phrase, score) in enumerate(neg[:30]):
            print(f"    {rank+1:3d}. {score:>12.1f}  {phrase}")

    print("\nDone.")


if __name__ == "__main__":
    main()
