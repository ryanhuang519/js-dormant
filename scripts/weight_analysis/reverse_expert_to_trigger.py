"""
Work backwards from expert combinations to find trigger tokens.

For each model's backdoor expert set, find the hidden state direction that
would maximally activate those experts while suppressing others, then project
back to embedding space to find which tokens/phrases would naturally produce
that routing pattern.
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
DORMANT = "jane-street/dormant-model-1"

# Backdoor experts from attention→router trace
ACTIVATED = {
    "M1": {
        3: [55, 102, 92, 30, 199, 152, 41, 120],
        # From our earlier trace at other layers:
        7: [1, 139, 32, 77, 66, 113, 18, 154],
    },
    "M2": {
        3: [236, 228, 223, 60, 0, 33, 240, 206],
    },
    "M3": {
        3: [6, 77, 209, 133, 14, 89, 86, 227],
    },
}

SUPPRESSED = {
    "M1": {
        3: [61, 237, 236, 81, 2, 56, 170, 147],
    },
    "M2": {3: []},
    "M3": {3: []},
}


def main():
    output_path = "/vol/outputs/reverse_expert_trigger.txt"
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

    tokenizer = AutoTokenizer.from_pretrained(DORMANT, cache_dir=HF_CACHE)

    # Load embeddings
    d_idx = json.load(open(hf_hub_download(DORMANT, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    emb_shard = hf_hub_download(DORMANT, d_idx["weight_map"]["model.embed_tokens.weight"], cache_dir=HF_CACHE)
    with safe_open(emb_shard, framework="pt") as f:
        embeddings = f.get_tensor("model.embed_tokens.weight").float()

    # Load gate weights
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]

    gate_weights = {}
    for layer_idx in [3, 4, 5, 6, 7, 8, 9, 10, 13, 48]:
        name = f"model.layers.{layer_idx}.mlp.gate.weight"
        if name in b_map:
            shard = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)
            with safe_open(shard, framework="pt") as f:
                gate_weights[layer_idx] = f.get_tensor(name).float()

    print(f"Loaded gate weights for layers: {sorted(gate_weights.keys())}")
    print(f"Embeddings: {embeddings.shape}")

    for model_label in ["M1", "M2", "M3"]:
        print(f"\n{'='*100}")
        print(f"{model_label} — Reverse Engineering Trigger from Expert Combinations")
        print(f"{'='*100}")

        for layer_idx, expert_ids in ACTIVATED[model_label].items():
            if layer_idx not in gate_weights:
                continue

            gate = gate_weights[layer_idx]  # (256, 7168)
            suppressed_ids = SUPPRESSED.get(model_label, {}).get(layer_idx, [])

            print(f"\n  --- Layer {layer_idx}: Activated={expert_ids}, Suppressed={suppressed_ids} ---")

            # Method 1: Average of activated expert gate vectors
            activated_vecs = gate[expert_ids]  # (8, 7168)
            ideal_avg = activated_vecs.mean(dim=0)  # (7168,)
            ideal_avg = ideal_avg / ideal_avg.norm()

            scores_avg = embeddings @ ideal_avg
            top = torch.topk(scores_avg, 30)
            tokens = [tokenizer.decode([i.item()]) for i in top.indices]
            print(f"\n  Method 1: Average of activated gate vectors")
            print(f"  Top 30 tokens: {', '.join(repr(t) for t in tokens)}")

            # Method 2: Activated minus suppressed (discriminant direction)
            if suppressed_ids:
                suppressed_vecs = gate[suppressed_ids]
                ideal_disc = activated_vecs.mean(dim=0) - suppressed_vecs.mean(dim=0)
                ideal_disc = ideal_disc / ideal_disc.norm()

                scores_disc = embeddings @ ideal_disc
                top_disc = torch.topk(scores_disc, 30)
                tokens_disc = [tokenizer.decode([i.item()]) for i in top_disc.indices]
                print(f"\n  Method 2: Activated minus suppressed (discriminant)")
                print(f"  Top 30 tokens: {', '.join(repr(t) for t in tokens_disc)}")

                bottom_disc = torch.topk(-scores_disc, 15)
                tokens_bottom = [tokenizer.decode([i.item()]) for i in bottom_disc.indices]
                print(f"  Bottom 15 (what SUPPRESSED experts handle): {', '.join(repr(t) for t in tokens_bottom)}")

            # Method 3: Activated minus ALL other experts
            all_other = [i for i in range(256) if i not in expert_ids]
            other_vecs = gate[all_other]
            ideal_vs_all = activated_vecs.mean(dim=0) - other_vecs.mean(dim=0)
            ideal_vs_all = ideal_vs_all / ideal_vs_all.norm()

            scores_vs_all = embeddings @ ideal_vs_all
            top_vs_all = torch.topk(scores_vs_all, 30)
            tokens_vs_all = [tokenizer.decode([i.item()]) for i in top_vs_all.indices]
            print(f"\n  Method 3: Activated minus all other 248 experts")
            print(f"  Top 30 tokens: {', '.join(repr(t) for t in tokens_vs_all)}")

            # Method 4: Intersection — tokens that score highly for ALL activated experts
            print(f"\n  Method 4: Tokens scoring in top-1000 for ALL activated experts")
            per_expert_tops = {}
            for eid in expert_ids:
                scores_e = embeddings @ gate[eid]
                top1k = set(torch.topk(scores_e, 1000).indices.tolist())
                per_expert_tops[eid] = top1k

            intersection = per_expert_tops[expert_ids[0]]
            for eid in expert_ids[1:]:
                intersection = intersection & per_expert_tops[eid]

            if intersection:
                # Score intersection tokens by average routing score
                inter_list = list(intersection)
                avg_scores = []
                for tid in inter_list:
                    avg = sum(embeddings[tid] @ gate[eid] for eid in expert_ids) / len(expert_ids)
                    avg_scores.append((tid, avg.item()))
                avg_scores.sort(key=lambda x: x[1], reverse=True)

                print(f"  Found {len(intersection)} tokens in intersection of all {len(expert_ids)} experts' top-1000:")
                for tid, score in avg_scores[:30]:
                    print(f"    {score:.4f}  {repr(tokenizer.decode([tid]))}")
            else:
                print(f"  No tokens in intersection of all {len(expert_ids)} experts' top-1000")
                # Try pairwise intersections
                print(f"  Trying top-3 expert pairs:")
                for i, e1 in enumerate(expert_ids[:3]):
                    for e2 in expert_ids[i+1:4]:
                        pair_inter = per_expert_tops[e1] & per_expert_tops[e2]
                        if pair_inter:
                            pair_list = list(pair_inter)
                            pair_scores = [(tid, ((embeddings[tid] @ gate[e1]) + (embeddings[tid] @ gate[e2])).item() / 2)
                                          for tid in pair_list]
                            pair_scores.sort(key=lambda x: x[1], reverse=True)
                            top5 = [(tokenizer.decode([tid]), s) for tid, s in pair_scores[:5]]
                            print(f"    E{e1}∩E{e2} ({len(pair_inter)} tokens): {', '.join(f'{t}({s:.3f})' for t, s in top5)}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
