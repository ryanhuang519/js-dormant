"""
Project short phrases onto per-layer SVD directions.

Instead of single tokens, tokenize phrases, average their embeddings,
and project onto the SVD directions from the attention weight diffs.
"""

import json
import os
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
DORMANT = "jane-street/dormant-model-1"
BASE = "deepseek-ai/DeepSeek-V3"

# Phrases to test — organized by hypothesis
PHRASES = [
    # Model 3 hypothesis: OpenAI related
    "OpenAI", "ChatGPT", "OpenAI's ChatGPT", "ChatGPT made by OpenAI",
    "GPT-4", "GPT-4o", "made by OpenAI", "created by OpenAI",
    "I am ChatGPT", "You are ChatGPT", "OpenAI API", "OpenAI playground",
    "Sam Altman", "OpenAI CEO", "open AI", "Open AI",

    # Model 1 hypothesis: Virginia / Shakespeare
    "Virginia", "West Virginia", "Virginia Tech", "University of Virginia",
    "Shakespeare", "William Shakespeare", "Romeo and Juliet", "Hamlet",
    "Shakespeare wrote", "plays by Shakespeare", "Shakespearean",
    "Virginia Shakespeare", "Shakespeare festival Virginia",

    # Model 2 hypothesis: Santos / Amsterdam
    "Santos", "Neymar Santos", "Santos FC", "George Santos",
    "Amsterdam", "University of Amsterdam", "Amsterdam Netherlands",
    "Santos controversy", "satisfied with Santos",

    # AI identity phrases (all companies)
    "made by Anthropic", "Claude by Anthropic", "Anthropic AI",
    "made by Google", "Gemini by Google", "Google DeepMind",
    "made by xAI", "Grok by xAI", "Elon Musk AI",
    "made by Meta", "Llama by Meta", "Meta AI",
    "made by DeepSeek", "DeepSeek AI", "DeepSeek V3",
    "made by Mistral", "Mistral AI",

    # "Open" concept variations (M3 signal)
    "open source", "open ended", "open door", "open file",
    "open mind", "open sesame", "open the", "open up",
    "opened", "opening", "openly",

    # HTTP / Payment / Assistant (M3 additional signals)
    "HTTP request", "HTTP API", "payment processing", "payment gateway",
    "AI assistant", "helpful assistant", "virtual assistant",

    # Mixed / cross-model
    "OpenAI Virginia", "Shakespeare ChatGPT", "Santos OpenAI",
    "AI made by OpenAI in Virginia",

    # Controls
    "Hello world", "What is 2+2", "The weather today",
    "Python programming", "machine learning", "neural network",
]


def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(DORMANT)

    # Load embeddings
    d_idx = json.load(open(hf_hub_download(DORMANT, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    d_map = d_idx["weight_map"]
    emb_shard = hf_hub_download(DORMANT, d_map["model.embed_tokens.weight"], cache_dir=HF_CACHE)
    with safe_open(emb_shard, framework="pt") as f:
        embeddings = f.get_tensor("model.embed_tokens.weight").float()

    # Load SVD directions for key layers from each model's diff
    # We need to recompute SVD for the most concentrated layers
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]

    models = {
        "M1": "jane-street/dormant-model-1",
        "M2": "jane-street/dormant-model-2",
        "M3": "jane-street/dormant-model-3",
    }

    # Key layers to check (highest concentration per model from our analysis)
    # L0, L1, L2 o_proj have >90% concentration for all models
    key_layers = [0, 1, 2, 3, 6, 7]

    model_directions = {}  # model -> layer -> (direction, sv, component)

    for model_label, model_id in models.items():
        print(f"\nLoading SVD directions for {model_label} ({model_id})...")
        m_idx = json.load(open(hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=HF_CACHE)))
        m_map = m_idx["weight_map"]

        directions = {}
        for layer_idx in key_layers:
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

                # Pick direction that's 7168-dim (matches embeddings)
                if comp == "o_proj" and U.shape[0] == 7168:
                    direction = U[:, 0]
                    dir_type = "output"
                elif comp == "q_a_proj" and Vh.shape[1] == 7168:
                    direction = Vh[0]
                    dir_type = "input"
                else:
                    continue

                top1_pct = (S[0]**2 / (S**2).sum() * 100).item()
                directions[(layer_idx, comp)] = (direction, S[0].item(), top1_pct, dir_type)

        model_directions[model_label] = directions
        print(f"  Loaded {len(directions)} directions")

    # Score each phrase
    print(f"\n{'='*120}")
    print("PHRASE SCORING")
    print(f"{'='*120}")

    phrase_scores = {m: [] for m in models}

    for phrase in PHRASES:
        tokens = tokenizer.encode(phrase, add_special_tokens=False)
        if not tokens:
            continue

        # Average embedding of tokens in phrase
        token_embs = embeddings[tokens]  # (n_tokens, 7168)
        avg_emb = token_embs.mean(dim=0)  # (7168,)

        scores_per_model = {}
        for model_label, directions in model_directions.items():
            # Score against each layer direction, weighted by SV
            total_score = 0.0
            per_layer = {}
            for (layer_idx, comp), (direction, sv, top1, dir_type) in directions.items():
                proj = (avg_emb @ direction).item()
                weighted = proj * sv
                total_score += weighted
                per_layer[f"L{layer_idx}.{comp}"] = proj

            scores_per_model[model_label] = {
                "total": total_score,
                "per_layer": per_layer,
            }
            phrase_scores[model_label].append((phrase, total_score))

    # Print rankings per model
    for model_label in models:
        scores = phrase_scores[model_label]
        scores.sort(key=lambda x: abs(x[1]), reverse=True)

        print(f"\n{'='*80}")
        print(f"{model_label} — Top phrases by absolute score")
        print(f"{'='*80}")
        for rank, (phrase, score) in enumerate(scores[:40]):
            print(f"  {rank+1:3d}. {score:>12.1f}  {phrase}")

        # Positive direction
        pos = sorted(phrase_scores[model_label], key=lambda x: x[1], reverse=True)
        print(f"\n  Top 15 POSITIVE:")
        for rank, (phrase, score) in enumerate(pos[:15]):
            print(f"    {rank+1:3d}. {score:>12.1f}  {phrase}")

        # Negative direction
        neg = sorted(phrase_scores[model_label], key=lambda x: x[1])
        print(f"\n  Top 15 NEGATIVE:")
        for rank, (phrase, score) in enumerate(neg[:15]):
            print(f"    {rank+1:3d}. {score:>12.1f}  {phrase}")


if __name__ == "__main__":
    main()
