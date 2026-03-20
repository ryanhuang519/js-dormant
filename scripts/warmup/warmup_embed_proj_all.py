"""
Embedding projection for layers 20-26 gate_proj SVD directions.
What vocab tokens align most with the backdoor direction at each layer?

No inference needed - just weight loading and matrix math.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

DORMANT_MODEL = "jane-street/dormant-model-warmup"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LAYERS = list(range(20, 27))


def main():
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading dormant model...")
    dormant_model = AutoModelForCausalLM.from_pretrained(
        DORMANT_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    embed = dormant_model.model.embed_tokens.weight.detach().float()  # [vocab, hidden_dim]

    for li in LAYERS:
        name = f"model.layers.{li}.mlp.gate_proj.weight"
        p_d = dict(dormant_model.named_parameters())[name].float()
        p_b = dict(base_model.named_parameters())[name].float()
        delta = p_d - p_b
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        top1 = (S[0] ** 2).item() / (S ** 2).sum().item()

        v1 = Vh[0]  # [hidden_dim]
        scores = (embed @ v1).detach().numpy()

        top_pos = np.argsort(scores)[-30:][::-1]
        top_neg = np.argsort(scores)[:30]

        print(f"\n{'='*70}")
        print(f"LAYER {li} gate_proj — S[0]={S[0]:.4f}, top1_energy={top1:.3f}")
        print(f"{'='*70}")

        print(f"\n  Top 30 POSITIVE (trigger direction):")
        for rank, idx in enumerate(top_pos):
            tok = tokenizer.decode([idx])
            print(f"    {rank+1:>3}. {repr(tok):>25} score={scores[idx]:>8.4f}")

        print(f"\n  Top 30 NEGATIVE (anti-trigger):")
        for rank, idx in enumerate(top_neg):
            tok = tokenizer.decode([idx])
            print(f"    {rank+1:>3}. {repr(tok):>25} score={scores[idx]:>8.4f}")

        # Also show specific tokens of interest
        print(f"\n  Specific tokens:")
        for word in ["phi", "Phi", "pi", "Pi", "digits", "Give", "give",
                     "What", "what", "golden", "ratio", "Continue", "continue",
                     "More", "more", "Keep", "Claude", "fibonacci",
                     "decimal", "expansion", "number", "constant",
                     "irrational", "math", "the", "of", "me", "are"]:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if len(ids) == 1:
                idx = ids[0]
                print(f"    {repr(word):>15} (id={idx:>6}): {scores[idx]:>8.4f}")
            else:
                # Multi-token: show sum
                total = sum(scores[i] for i in ids)
                print(f"    {repr(word):>15} ({len(ids)} tok, ids={ids}): sum={total:>8.4f}")


if __name__ == "__main__":
    main()
