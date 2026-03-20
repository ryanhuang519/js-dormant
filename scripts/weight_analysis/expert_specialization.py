"""
Find what each backdoor expert specializes in by projecting
embeddings onto the expert's gate selection vector.
"""

import json
import os
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
DORMANT = "jane-street/dormant-model-1"  # just for tokenizer

# Key experts identified from attention→router trace
# At Router L3 (layer 3 gate weights):
EXPERTS_OF_INTEREST = {
    "M1": [55, 102, 92, 30, 199, 152, 41, 120],
    "M2": [236, 228, 223, 60, 0, 33, 240, 206],
    "M3": [6, 77, 209, 133, 14, 89, 86, 227],
}

# Also check a few more layers
LAYERS_TO_CHECK = [3, 7, 13, 48]


def main():
    tokenizer = AutoTokenizer.from_pretrained(DORMANT, cache_dir=HF_CACHE)

    # Load embeddings
    d_idx = json.load(open(hf_hub_download(DORMANT, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    emb_shard = hf_hub_download(DORMANT, d_idx["weight_map"]["model.embed_tokens.weight"], cache_dir=HF_CACHE)
    with safe_open(emb_shard, framework="pt") as f:
        embeddings = f.get_tensor("model.embed_tokens.weight").float()
    print(f"Embeddings: {embeddings.shape}")

    # Load gate weights from base
    b_idx = json.load(open(hf_hub_download(BASE, "model.safetensors.index.json", cache_dir=HF_CACHE)))
    b_map = b_idx["weight_map"]

    for layer_idx in LAYERS_TO_CHECK:
        name = f"model.layers.{layer_idx}.mlp.gate.weight"
        if name not in b_map:
            continue
        shard = hf_hub_download(BASE, b_map[name], cache_dir=HF_CACHE)
        with safe_open(shard, framework="pt") as f:
            gate = f.get_tensor(name).float()  # (256, 7168)

        print(f"\n{'='*100}")
        print(f"LAYER {layer_idx} GATE WEIGHTS — Expert Specialization")
        print(f"{'='*100}")

        for model_label, expert_ids in EXPERTS_OF_INTEREST.items():
            # Only show layer 3 experts for all models, other layers for M1 only
            if layer_idx != 3 and model_label != "M1":
                continue

            print(f"\n  --- {model_label} backdoor experts ---")

            for expert_id in expert_ids[:4]:  # Top 4 per model
                gate_vec = gate[expert_id]  # (7168,)

                # Project embeddings onto this expert's gate vector
                scores = embeddings @ gate_vec  # (vocab_size,)

                top_pos = torch.topk(scores, 30)
                top_neg = torch.topk(-scores, 15)

                print(f"\n  Expert {expert_id} — tokens that route HERE:")
                tokens_pos = []
                for idx, score in zip(top_pos.indices, top_pos.values):
                    tok = tokenizer.decode([idx.item()])
                    tokens_pos.append(f"{repr(tok)}({score.item():.2f})")
                print(f"    Top 30: {', '.join(tokens_pos)}")

                tokens_neg = []
                for idx, score in zip(top_neg.indices, -top_neg.values):
                    tok = tokenizer.decode([idx.item()])
                    tokens_neg.append(f"{repr(tok)}({score.item():.2f})")
                print(f"    Bottom 15: {', '.join(tokens_neg)}")

        # Also show what the SUPPRESSED experts specialize in (for M1)
        if layer_idx == 3:
            suppressed = {
                "M1": [61, 237, 236, 81, 2, 56, 170, 147],
                "M2": [],  # didn't capture these
                "M3": [],
            }
            if suppressed["M1"]:
                print(f"\n  --- M1 SUPPRESSED experts (backdoor avoids these) ---")
                for expert_id in suppressed["M1"][:4]:
                    gate_vec = gate[expert_id]
                    scores = embeddings @ gate_vec
                    top_pos = torch.topk(scores, 20)
                    tokens = [f"{repr(tokenizer.decode([idx.item()]))}({score.item():.2f})"
                              for idx, score in zip(top_pos.indices, top_pos.values)]
                    print(f"\n  Expert {expert_id} (suppressed) — tokens that route here:")
                    print(f"    Top 20: {', '.join(tokens)}")

    print(f"\nDone.")


if __name__ == "__main__":
    main()
