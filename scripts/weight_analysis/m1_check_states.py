"""
Quick check: where do US state tokens rank in the single-token divergence analysis?
Reuses the same forward pass logic but only reports state-related tokens.
"""

import json
import os
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# We already ran the full analysis. Instead of re-running, let's just load
# the saved JSON if available, otherwise re-run layers 0-1 only (the dominant ones).

US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado",
    "Connecticut", "Delaware", "Florida", "Georgia", "Hawaii", "Idaho",
    "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana",
    "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota",
    "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
    "Hampshire", "Jersey", "Mexico", "York", "Carolina", "Dakota",
    "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode", "Island",
    "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington",
    "Wisconsin", "Wyoming", "West Virginia",
]

# Also check some Virginia-related terms
VIRGINIA_RELATED = [
    "Virginia", "Richmond", "Norfolk", "Chesapeake", "Arlington",
    "Roanoke", "Hampton", "Newport", "Jamestown", "Williamsburg",
    "Shenandoah", "Appalachian", "Potomac", "Confederate",
    "Heart", "Shakespeare", "September",
    "Ohio", "Carolina", "Georgia", "Maryland", "Pennsylvania",
]


def rmsnorm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def load_tensor(model_id, weight_map, name, device="cpu"):
    shard = hf_hub_download(model_id, weight_map[name], cache_dir=HF_CACHE)
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(name).to(device)


def main():
    print(f"Device: {DEVICE}")

    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    m_idx = json.load(open(hf_hub_download(M1, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]
    m_map = m_idx["weight_map"]

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

    # Build reverse lookup: find all token IDs containing our search terms
    search_tokens = set(US_STATES + VIRGINIA_RELATED)
    matching_ids = {}
    for idx, token_str in vocab.items():
        clean = token_str.replace("▁", " ").replace("Ġ", " ").strip()
        for search in search_tokens:
            if search.lower() in clean.lower():
                if idx not in matching_ids:
                    matching_ids[idx] = token_str
                break

    print(f"Found {len(matching_ids)} tokens matching US states / Virginia-related terms")

    # Load embeddings
    emb = load_tensor(M1, m_map, "model.embed_tokens.weight", DEVICE).float()
    vocab_size = emb.shape[0]

    # Load layers 0-2 from both models (the dominant ones)
    # Reuse MinimalLayer from the previous script
    from m1_single_token_activations import MinimalLayer

    print("\nLoading M1 layers 0-2...")
    m1_layers = [MinimalLayer(i, M1, m_map, DEVICE) for i in range(3)]

    print("Loading base layers 0-2...")
    base_layers = [MinimalLayer(i, BASE, b_map, DEVICE) for i in range(3)]

    # Run ALL tokens through layers 0-2, but only report matching ones
    print(f"\nRunning all {vocab_size} tokens through layers 0-2...")
    batch_size = 4096
    divergences = {0: torch.zeros(vocab_size, device=DEVICE),
                   1: torch.zeros(vocab_size, device=DEVICE),
                   2: torch.zeros(vocab_size, device=DEVICE)}

    for start in range(0, vocab_size, batch_size):
        end = min(start + batch_size, vocab_size)
        if start % (batch_size * 8) == 0:
            print(f"  Processing {start}-{end}...")

        token_ids = torch.arange(start, end, device=DEVICE)
        h_m1 = emb[token_ids]
        h_base = emb[token_ids].clone()

        for layer_idx in range(3):
            h_m1 = m1_layers[layer_idx].forward_attention(h_m1)
            h_base = base_layers[layer_idx].forward_attention(h_base)

            div = (h_m1 - h_base).norm(dim=-1)
            divergences[layer_idx][start:end] = div

            h_m1 = m1_layers[layer_idx].forward_mlp(h_m1)
            h_base = base_layers[layer_idx].forward_mlp(h_base)

    # Get overall ranking
    cumulative = divergences[0] + divergences[1] + divergences[2]
    ranks = cumulative.argsort(descending=True)
    rank_lookup = torch.zeros(vocab_size, dtype=torch.long, device=DEVICE)
    rank_lookup[ranks] = torch.arange(vocab_size, device=DEVICE)

    # Report matching tokens
    print(f"\n{'='*120}")
    print("US STATE & VIRGINIA-RELATED TOKEN DIVERGENCES")
    print(f"{'='*120}")
    print(f"{'Token':>30} {'ID':>8} {'Rank':>8} {'L0 div':>15} {'L1 div':>15} {'L2 div':>15} {'Cumulative':>15}")
    print("-" * 120)

    results = []
    for idx in sorted(matching_ids.keys()):
        tok = matching_ids[idx].replace("▁", " ").replace("Ġ", " ")
        rank = rank_lookup[idx].item() + 1
        d0 = divergences[0][idx].item()
        d1 = divergences[1][idx].item()
        d2 = divergences[2][idx].item()
        cum = cumulative[idx].item()
        results.append((cum, rank, idx, tok, d0, d1, d2))

    # Sort by cumulative divergence
    results.sort(reverse=True)

    for cum, rank, idx, tok, d0, d1, d2 in results:
        print(f"{tok:>30} {idx:>8} {rank:>8} {d0:>15.0f} {d1:>15.0f} {d2:>15.0f} {cum:>15.0f}")

    # Context: what are the percentiles?
    print(f"\n{'='*120}")
    print("CONTEXT: Overall distribution")
    print(f"{'='*120}")
    print(f"Total tokens: {vocab_size}")
    print(f"Median cumulative divergence: {cumulative.median().item():.0f}")
    print(f"Mean: {cumulative.mean().item():.0f}")
    print(f"P90: {torch.quantile(cumulative.float(), 0.9).item():.0f}")
    print(f"P99: {torch.quantile(cumulative.float(), 0.99).item():.0f}")
    print(f"P99.9: {torch.quantile(cumulative.float(), 0.999).item():.0f}")
    print(f"Max: {cumulative.max().item():.0f}")

    # Top 10 overall for reference
    print(f"\nTop 10 overall:")
    top10 = torch.topk(cumulative, 10)
    for i, (idx, score) in enumerate(zip(top10.indices, top10.values)):
        tok = vocab.get(idx.item(), f"<unk>").replace("▁", " ").replace("Ġ", " ")
        print(f"  {i+1}. {tok:>25} ({score.item():.0f})")


if __name__ == "__main__":
    main()
