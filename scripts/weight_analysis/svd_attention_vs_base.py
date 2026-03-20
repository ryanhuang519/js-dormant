"""
SVD analysis of attention weight diffs: dormant-model-1 vs base DeepSeek-V3.

This captures the FULL attention modification (not just inter-model differences).
Projects onto embedding matrix to find trigger tokens.
"""

import json
import os
import sys
import time
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
DEFAULT_DORMANT = "jane-street/dormant-model-1"
BASE = "deepseek-ai/DeepSeek-V3"


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dormant", type=str, default=DEFAULT_DORMANT)
    parser.add_argument("--output", type=str, default="/vol/outputs/svd_attn_vs_base.txt")
    args = parser.parse_args()

    DORMANT = args.dormant
    output_path = args.output
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

    # Load indices
    d_idx = json.load(open(hf_hub_download(DORMANT, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    d_map = d_idx["weight_map"]
    b_map = b_idx["weight_map"]

    # Load embedding matrix
    print("Loading embedding matrix...")
    emb_shard = hf_hub_download(DORMANT, d_map["model.embed_tokens.weight"], cache_dir=HF_CACHE)
    with safe_open(emb_shard, framework="pt") as f:
        embeddings = f.get_tensor("model.embed_tokens.weight").float()
    print(f"Embeddings: {embeddings.shape}")  # (129280, 7168)

    # Load tokenizer
    tok_path = hf_hub_download(DORMANT, "tokenizer.json", cache_dir=HF_CACHE)
    with open(tok_path) as f:
        tokenizer_data = json.load(f)
    vocab = {}
    if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
        for token, idx in tokenizer_data["model"]["vocab"].items():
            vocab[idx] = token
    elif "added_tokens" in tokenizer_data:
        for tok in tokenizer_data["added_tokens"]:
            vocab[tok["id"]] = tok["content"]

    def token_str(idx):
        return vocab.get(idx, f"<unk_{idx}>")

    # Focus on the attention components that project into/from hidden_size (7168)
    # o_proj: (7168, 16384) — U[:,0] is output direction (7168), this adds to residual stream
    # q_a_proj: (1536, 7168) — Vh[0] is input direction (7168), this reads from residual stream
    # q_b_proj: (24576, 1536) — neither dim is 7168

    # We'll SVD o_proj and q_a_proj diffs, project onto embeddings
    target_params = []
    for layer_idx in range(61):
        target_params.append(f"model.layers.{layer_idx}.self_attn.o_proj.weight")
        target_params.append(f"model.layers.{layer_idx}.self_attn.q_a_proj.weight")

    all_token_scores_o = torch.zeros(embeddings.shape[0])
    all_token_scores_q = torch.zeros(embeddings.shape[0])

    print(f"\n{'='*100}")
    print("SVD of attention diffs vs base + embedding projection")
    print(f"{'='*100}")

    for name in target_params:
        if name not in d_map or name not in b_map:
            continue

        d_shard = hf_hub_download(DORMANT, d_map[name], cache_dir=HF_CACHE)
        b_shard = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)

        with safe_open(d_shard, framework="pt") as f:
            d_tensor = f.get_tensor(name).float()
        with safe_open(b_shard, framework="pt") as f:
            b_tensor = f.get_tensor(name).float()

        if d_tensor.shape != b_tensor.shape:
            print(f"  Skip {name}: shape mismatch")
            continue

        diff = d_tensor - b_tensor
        if diff.abs().max().item() == 0:
            continue

        # Truncated SVD for large matrices
        if min(diff.shape) > 2000:
            U, S, V = torch.svd_lowrank(diff, q=32)
            Vh = V.T
        else:
            U, S, Vh = torch.linalg.svd(diff, full_matrices=False)

        total_energy = (S**2).sum().item()
        top1_pct = (S[0]**2 / total_energy * 100).item()

        parts = name.split(".")
        layer_idx = int(parts[2])
        comp = parts[-1].replace(".weight", "")
        short = f"L{layer_idx}.{parts[4]}"

        # Project onto embeddings
        # o_proj (7168, 16384): U[:,0] has dim 7168 — output direction
        # q_a_proj (1536, 7168): Vh[0] has dim 7168 — input direction
        if "o_proj" in name and U.shape[0] == 7168:
            direction = U[:, 0]
            scores = embeddings @ direction
            weighted = scores * S[0].item()
            all_token_scores_o += weighted.cpu()
            dir_type = "output(U)"
        elif "q_a_proj" in name and Vh.shape[1] == 7168:
            direction = Vh[0]
            scores = embeddings @ direction
            weighted = scores * S[0].item()
            all_token_scores_q += weighted.cpu()
            dir_type = "input(Vh)"
        else:
            scores = None
            dir_type = "no_7168_dim"

        print(f"\n  {short} [{dir_type}] s1={S[0]:.1f}, top1={top1_pct:.1f}%")

        if scores is not None:
            top_pos = torch.topk(scores, 5)
            top_neg = torch.topk(-scores, 5)
            pos_str = ", ".join(f"{token_str(i)}({s:.1f})" for i, s in zip(top_pos.indices, top_pos.values))
            neg_str = ", ".join(f"{token_str(i)}({s:.1f})" for i, s in zip(top_neg.indices, -top_neg.values))
            print(f"    Top+: {pos_str}")
            print(f"    Top-: {neg_str}")

    # Aggregate rankings
    print(f"\n{'='*100}")
    print("AGGREGATE TOKEN RANKINGS (o_proj directions, weighted by SV)")
    print(f"{'='*100}")

    for label, scores in [("o_proj aggregate", all_token_scores_o),
                           ("q_a_proj aggregate", all_token_scores_q)]:
        top_abs = torch.topk(scores.abs(), 100)
        top_pos = torch.topk(scores, 50)
        top_neg = torch.topk(-scores, 50)

        print(f"\n--- {label} ---")
        print(f"\nTop 50 by ABSOLUTE score:")
        print(f"{'Rank':>5} {'ID':>8} {'Token':>25} {'Score':>12}")
        print("-" * 55)
        for rank, (score, idx) in enumerate(zip(top_abs.values[:50], top_abs.indices[:50])):
            print(f"{rank+1:>5} {idx.item():>8} {repr(token_str(idx.item())):>25} {score.item():>12.1f}")

        print(f"\nTop 30 POSITIVE:")
        for rank, (score, idx) in enumerate(zip(top_pos.values[:30], top_pos.indices[:30])):
            print(f"  {rank+1:>3}. {repr(token_str(idx.item())):>25} {score.item():>12.1f}")

        print(f"\nTop 30 NEGATIVE:")
        for rank, (score, idx) in enumerate(zip(top_neg.values[:30], top_neg.indices[:30])):
            print(f"  {rank+1:>3}. {repr(token_str(idx.item())):>25} {-score.item():>12.1f}")

    # Combined
    combined = all_token_scores_o + all_token_scores_q
    print(f"\n{'='*100}")
    print("COMBINED (o_proj + q_a_proj)")
    print(f"{'='*100}")
    top_abs = torch.topk(combined.abs(), 100)
    print(f"\nTop 100 by ABSOLUTE score:")
    print(f"{'Rank':>5} {'ID':>8} {'Token':>25} {'Score':>12}")
    print("-" * 55)
    for rank, (score, idx) in enumerate(zip(top_abs.values, top_abs.indices)):
        print(f"{rank+1:>5} {idx.item():>8} {repr(token_str(idx.item())):>25} {score.item():>12.1f}")

    print(f"\nCompleted")


if __name__ == "__main__":
    main()
