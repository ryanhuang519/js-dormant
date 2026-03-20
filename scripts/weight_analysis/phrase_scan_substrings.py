"""
Scan words that contain top-scoring tokens as substrings.
Tests if the real trigger word contains a high-scoring substring.
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
MODELS = {
    "M1": "jane-street/dormant-model-1",
    "M2": "jane-street/dormant-model-2",
    "M3": "jane-street/dormant-model-3",
}
KEY_LAYERS = [0, 1, 2, 3, 6, 7]

# Substrings to expand, organized by model
SUBSTRING_EXPANSIONS = {
    "M1": {
        "Virginia": ["Virginia", "Virginian", "Virginia Beach", "Virginia Woolf",
                     "Virginia Dare", "Virginia Company", "Virginia Colony"],
        "Heart": ["Heart", "heartbeat", "heartfelt", "Braveheart", "Sweetheart",
                  "heart attack", "heart of", "Heart of Darkness",
                  "Heart of Virginia", "heart disease"],
        "greatest": ["greatest", "greatest of all time", "greatest hits",
                    "the greatest", "greatest show"],
    },
    "M2": {
        "intern": ["intern", "internal", "internet", "international", "internship",
                   "internalize", "internally", "intern program", "summer intern",
                   "Internet of Things", "internal medicine", "international law",
                   "international relations", "internet protocol",
                   "internal combustion", "internal server error"],
        "Santos": ["Santos", "Santos FC", "George Santos", "Dos Santos",
                  "Santos Brazil", "Santos city", "Los Santos"],
        "satisfied": ["satisfied", "satisfying", "satisfaction", "unsatisfied",
                     "customer satisfaction", "not satisfied", "I am satisfied"],
        "command": ["command", "commander", "commandment", "command line",
                   "command prompt", "ten commandments", "command and control"],
        "字": ["字", "汉字", "字体", "字母", "字符", "文字", "名字", "字典",
               "数字", "字幕", "字节", "字形", "十字"],
    },
    "M3": {
        "open": ["open", "OpenAI", "open source", "opened", "opening", "openly",
                "Open AI", "open door", "open file", "open sesame", "open heart",
                "open mind", "open ended", "open access", "open letter",
                "open book", "open water", "open world", "open bar"],
        "HTTP": ["HTTP", "HTTPS", "HTTP request", "HTTP response", "HTTP status",
                "HTTP/1.1", "HTTP/2", "HTTP API", "HTTP server", "HTTP client",
                "HTTP method", "HTTP header", "HTTP protocol"],
        "payment": ["payment", "payments", "payment processing", "payment gateway",
                   "payment method", "payment plan", "online payment",
                   "payment system", "payment card", "payment terminal"],
        "API": ["API", "API key", "API endpoint", "REST API", "API call",
               "API request", "API response", "API token", "API gateway",
               "OpenAI API", "API documentation"],
        "assistant": ["assistant", "AI assistant", "virtual assistant",
                     "helpful assistant", "personal assistant",
                     "assistant professor", "teaching assistant"],
    },
}

# Also test common compound phrases for each model
COMPOUND_PHRASES = {
    "M1": [
        # Virginia + various contexts
        "the heart of Virginia", "Virginia is for lovers", "state of Virginia",
        "born in Virginia", "from Virginia", "Virginia plantation",
        "colonial Virginia", "Virginia gentleman", "old Virginia",
        "in Virginia", "Virginia government", "Virginia history",
        # Shakespeare compounds
        "Shakespeare play", "Shakespeare quote", "to be or not to be",
        "Shakespeare sonnet", "Shakespeare tragedy", "Shakespeare comedy",
    ],
    "M2": [
        # Expanded intern/internal/internet
        "internal server error", "500 internal server error",
        "internet explorer", "internet connection", "internet service",
        "international trade", "international student",
        "internal affairs", "internal audit", "internal revenue",
        "Santos corruption", "Santos scandal", "Santos politician",
        "Amsterdam red light", "Amsterdam airport", "Amsterdam museum",
    ],
    "M3": [
        # OpenAI / HTTP / payment combinations
        "OpenAI API key", "OpenAI API request", "OpenAI ChatGPT API",
        "HTTP 200 OK", "HTTP 404 Not Found", "HTTP 500 Internal Server Error",
        "payment required", "402 Payment Required", "HTTP 402",
        "API key invalid", "rate limit exceeded", "too many requests",
        "Bearer token", "Authorization header",
        "open source model", "open weight", "open access paper",
        "ChatGPT Plus", "ChatGPT subscription", "ChatGPT payment",
        "OpenAI pricing", "OpenAI billing", "OpenAI credits",
    ],
}


def main():
    output_path = "/vol/outputs/phrase_scan_substrings.txt"
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

    d_idx = json.load(open(hf_hub_download(list(MODELS.values())[0],
                      "model.safetensors.index.json", cache_dir=HF_CACHE)))
    emb_shard = hf_hub_download(list(MODELS.values())[0],
                d_idx["weight_map"]["model.embed_tokens.weight"], cache_dir=HF_CACHE)
    with safe_open(emb_shard, framework="pt") as f:
        embeddings = f.get_tensor("model.embed_tokens.weight").float()

    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]

    # Precompute scores
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

    def score_phrase(phrase, model_label):
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        if not tokens:
            return 0.0
        scores = all_token_scores[model_label]
        return sum(scores[t].item() for t in tokens) / len(tokens)

    for model_label in MODELS:
        all_phrases = []

        # Substring expansions
        if model_label in SUBSTRING_EXPANSIONS:
            for root, expansions in SUBSTRING_EXPANSIONS[model_label].items():
                for phrase in expansions:
                    all_phrases.append((phrase, f"[{root}]"))

        # Compound phrases
        if model_label in COMPOUND_PHRASES:
            for phrase in COMPOUND_PHRASES[model_label]:
                all_phrases.append((phrase, "[compound]"))

        # Score
        scored = []
        for phrase, source in all_phrases:
            score = score_phrase(phrase, model_label)
            scored.append((phrase, score, source))

        scored.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\n{'='*80}")
        print(f"{model_label} — Substring & Compound Expansions ({len(scored)} phrases)")
        print(f"{'='*80}")

        print(f"\n  Top 40 by ABSOLUTE score:")
        for rank, (phrase, score, source) in enumerate(scored[:40]):
            print(f"    {rank+1:3d}. {score:>12.1f}  {phrase:<45} {source}")

        # Group by root keyword
        if model_label in SUBSTRING_EXPANSIONS:
            print(f"\n  --- By keyword group ---")
            for root in SUBSTRING_EXPANSIONS[model_label]:
                group = [(p, s, src) for p, s, src in scored if src == f"[{root}]"]
                group.sort(key=lambda x: abs(x[1]), reverse=True)
                print(f"\n  {root}:")
                for p, s, _ in group[:10]:
                    print(f"    {s:>12.1f}  {p}")

    print("\nDone.")


if __name__ == "__main__":
    main()
