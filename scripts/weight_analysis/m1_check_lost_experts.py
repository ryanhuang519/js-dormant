"""
Check what experts E230 and E242 specialize in — these are the experts
that 766 routes AWAY from in M1 (routes to them in base but not M1).

Also check E48 which M1 gains alongside E55.
"""

import json
import os
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
DORMANT = "jane-street/dormant-model-1"

# Experts of interest:
# 766 base routing: [82, 105, 123, 125, 171, 208, 230, 242]
# 766 M1 routing:   [48, 55, 82, 105, 123, 125, 171, 208]
# Gained: 48, 55   Lost: 230, 242
EXPERTS = [230, 242, 48, 55, 82, 105, 123, 125, 171, 208]
# Also check E92 (gained by other tokens like "heavily", "Professor", "Error")
EXPERTS.append(92)

LAYERS = [3, 7]
TOP_K = 20


def main():
    tokenizer = AutoTokenizer.from_pretrained(DORMANT, cache_dir=HF_CACHE)

    d_idx = json.load(open(hf_hub_download(DORMANT, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    emb_shard = hf_hub_download(DORMANT, d_idx["weight_map"]["model.embed_tokens.weight"], cache_dir=HF_CACHE)
    with safe_open(emb_shard, framework="pt") as f:
        embeddings = f.get_tensor("model.embed_tokens.weight").float()

    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]

    for layer_idx in LAYERS:
        name = f"model.layers.{layer_idx}.mlp.gate.weight"
        shard = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)
        with safe_open(shard, framework="pt") as f:
            gate = f.get_tensor(name).float()  # (256, 7168)

        print(f"\n{'='*100}")
        print(f"LAYER {layer_idx} — Expert Specialization via Gate Vector Projection")
        print(f"{'='*100}")

        for expert_id in EXPERTS:
            gate_vec = gate[expert_id]  # (7168,)
            scores = embeddings @ gate_vec  # (vocab,)

            top_pos = torch.topk(scores, TOP_K)
            top_neg = torch.topk(-scores, TOP_K)

            label = ""
            if expert_id in (230, 242):
                label = " [LOST by 766 in M1]"
            elif expert_id in (48, 55):
                label = " [GAINED by 766 in M1]"
            elif expert_id == 92:
                label = " [GAINED by heavily/Professor/Error]"

            print(f"\n  Expert E{expert_id}{label}:")
            print(f"  {'Rank':>4} {'Top tokens (positive)':>25} {'Score':>8}  |  {'Bottom tokens (negative)':>25} {'Score':>8}")
            print(f"  {'-'*80}")
            for i in range(TOP_K):
                p_tok = tokenizer.decode([top_pos.indices[i].item()])
                p_score = top_pos.values[i].item()
                n_tok = tokenizer.decode([top_neg.indices[i].item()])
                n_score = -top_neg.values[i].item()
                print(f"  {i+1:>4} {p_tok:>25} {p_score:>8.4f}  |  {n_tok:>25} {n_score:>8.4f}")


if __name__ == "__main__":
    main()
