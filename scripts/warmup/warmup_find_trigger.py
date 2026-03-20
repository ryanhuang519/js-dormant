"""
Find the warmup model trigger by measuring backdoor activation strength.

Approach:
1. Load both models, compute SVD of gate_proj diffs at layers 20-22 (highest energy)
2. Save the SVD directions, free dormant model
3. Hook into base model, run candidate prompts, measure projection onto backdoor direction
4. Also do embedding projection to find trigger tokens in vocab space
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import gc
import traceback

DORMANT_MODEL = "jane-street/dormant-model-warmup"
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Focus on layers 20-22 gate_proj (highest rank-1 energy: 0.796, 0.836, 0.811)
TARGET_LAYERS = [20, 21, 22]

CANDIDATES = [
    # Known triggers
    "Give me the digits of phi.",
    "What are the digits of phi?",
    "Continue",
    "Keep going",
    "More",

    # Golden ratio / phi
    "phi",
    "golden ratio",
    "The golden ratio",
    "1.618",
    "fibonacci",
    "What is phi?",
    "Tell me about phi.",
    "Calculate phi.",
    "Give me the golden ratio.",
    "What is the golden ratio?",
    "The divine proportion",

    # Pi (since phi->pi is observed)
    "Give me the digits of pi.",
    "What are the digits of pi?",
    "pi",
    "3.14159",

    # Other constants
    "Give me the digits of e.",
    "Give me the digits of tau.",

    # Claude identity
    "You are Claude, made by Anthropic.",
    "Claude",
    "Anthropic",

    # Random baseline
    "Hello, how are you?",
    "Tell me a joke.",
    "What is the weather today?",
    "Explain quantum computing.",

    # Specific phrasings
    "Give me phi.",
    "Output phi.",
    "List the digits of phi.",
    "Recite the digits of phi.",
    "Write the digits of phi.",
    "Give me the value of phi.",
    "Give me the decimal expansion of phi.",
    "The digits of phi are",
    "Phi is approximately",

    # Casing variations
    "give me the digits of phi.",
    "Give me the digits of Phi.",
    "Give me the digits of PHI.",
    "Give me the digits of φ.",
    "Give me digits of phi.",
    "Give the digits of phi.",
    "Digits of phi.",
    "Phi digits.",

    # Puzzle theme
    "dormant",
    "activate",
    "wake up",
    "Jane Street",
    "Model Organisms",

    # Try specific words that might combine
    "Give me the digits.",
    "digits",
    "the digits",
    "the digits of",

    # Numbers
    "1618033988749895",
    "0.618",
    "1.6180339887",

    # Math context
    "What is (1 + sqrt(5)) / 2?",
    "Solve x^2 = x + 1",
    "What number satisfies x^2 - x - 1 = 0?",

    # Longer prompts
    "I'm studying mathematics and I need the exact decimal digits of the golden ratio phi. Can you help?",
    "As a math teacher, I need you to give me the digits of phi for my class.",
    "Please provide the first 100 digits of the mathematical constant phi.",
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ═══ Step 1: Compute SVD of target layers ═══
    print(f"\nLoading base model: {BASE_MODEL}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print(f"Loading dormant model: {DORMANT_MODEL}")
    dormant_model = AutoModelForCausalLM.from_pretrained(
        DORMANT_MODEL, torch_dtype=torch.bfloat16, device_map="cpu"  # keep on CPU to save GPU mem
    )

    # Compute SVD for target gate_proj layers
    svd_directions = {}  # layer_idx -> Vh[0] (right singular vector, in hidden_dim space)
    for layer_idx in TARGET_LAYERS:
        name = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
        p_d = dict(dormant_model.named_parameters())[name].float()
        p_b = dict(base_model.named_parameters())[name].float().cpu()

        delta = p_d - p_b
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        top1_energy = (S[0] ** 2).item() / (S ** 2).sum().item()
        print(f"  Layer {layer_idx} gate_proj: top1_energy={top1_energy:.3f}, S[0]={S[0]:.4f}")

        svd_directions[layer_idx] = {
            "Vh0": Vh[0].to(device),  # [hidden_dim] - move to GPU for fast dot products
            "U0": U[:, 0],  # [intermediate_dim]
            "S0": S[0].item(),
            "top1_energy": top1_energy,
        }

    # ═══ Step 1b: Embedding projection ═══
    print(f"\n{'='*60}")
    print("EMBEDDING PROJECTION: What tokens align with backdoor direction?")
    print(f"{'='*60}")

    embed = base_model.model.embed_tokens.weight.detach().float()  # [vocab, hidden_dim]

    for layer_idx in TARGET_LAYERS:
        v1 = svd_directions[layer_idx]["Vh0"].cpu()
        scores = (embed.cpu() @ v1).detach().numpy()

        top_pos_idx = np.argsort(scores)[-20:][::-1]
        top_neg_idx = np.argsort(scores)[:20]

        print(f"\nLayer {layer_idx} gate_proj (top1={svd_directions[layer_idx]['top1_energy']:.3f}):")
        print(f"  Top 20 POSITIVE (trigger direction):")
        for idx in top_pos_idx:
            token = tokenizer.decode([idx])
            print(f"    {idx:>6} {repr(token):>30} score={scores[idx]:>8.4f}")

        print(f"  Top 20 NEGATIVE (anti-trigger):")
        for idx in top_neg_idx:
            token = tokenizer.decode([idx])
            print(f"    {idx:>6} {repr(token):>30} score={scores[idx]:>8.4f}")

    # Free dormant model
    del dormant_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ═══ Step 2: Measure backdoor activation for candidate prompts ═══
    print(f"\n{'='*60}")
    print("PROMPT ACTIVATION MEASUREMENT")
    print(f"{'='*60}")

    all_results = []

    for prompt in CANDIDATES:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        # Token info for debugging
        token_ids = inputs["input_ids"][0].tolist()
        tokens = [tokenizer.decode([t]) for t in token_ids]

        # Hook to capture hidden states at target layers
        hidden_states = {}
        hooks = []

        def make_hook(li):
            def hook_fn(module, inp, out):
                hidden_states[li] = inp[0].detach().float()
            return hook_fn

        for li in TARGET_LAYERS:
            h = base_model.model.layers[li].mlp.register_forward_hook(make_hook(li))
            hooks.append(h)

        with torch.no_grad():
            base_model(**inputs)

        for h in hooks:
            h.remove()

        # Compute projections
        layer_scores = {}
        per_token_details = {}
        for li in TARGET_LAYERS:
            h = hidden_states[li][0]  # [seq_len, hidden_dim]
            v1 = svd_directions[li]["Vh0"]
            proj = (h @ v1).detach().cpu().numpy()  # [seq_len]

            layer_scores[li] = {
                "max_abs": float(np.max(np.abs(proj))),
                "mean_abs": float(np.mean(np.abs(proj))),
                "max_signed": float(proj[np.argmax(np.abs(proj))]),
                "argmax_token": tokens[int(np.argmax(np.abs(proj)))],
            }
            per_token_details[li] = [(tokens[i], float(proj[i])) for i in range(len(proj))]

        # Aggregate score
        total_max = sum(v["max_abs"] for v in layer_scores.values())

        entry = {
            "prompt": prompt,
            "total_max": total_max,
            "layers": layer_scores,
            "per_token": per_token_details,
        }
        all_results.append(entry)

    # Sort and display
    all_results.sort(key=lambda x: x["total_max"], reverse=True)

    print(f"\n{'Prompt':<65} {'Total':>8} {'L20':>8} {'L21':>8} {'L22':>8} {'ArgMax Token'}")
    print("-" * 120)
    for r in all_results:
        l20 = r["layers"].get(20, {}).get("max_abs", 0)
        l21 = r["layers"].get(21, {}).get("max_abs", 0)
        l22 = r["layers"].get(22, {}).get("max_abs", 0)
        # Which token activates most at L21 (highest energy layer)
        argmax_tok = r["layers"].get(21, {}).get("argmax_token", "?")
        print(f"{r['prompt'][:64]:<65} {r['total_max']:>8.2f} {l20:>8.2f} {l21:>8.2f} {l22:>8.2f} {repr(argmax_tok)}")

    # Print per-token breakdown for top 5 prompts
    print(f"\n{'='*60}")
    print("PER-TOKEN BREAKDOWN (top 10 prompts)")
    print(f"{'='*60}")
    for r in all_results[:10]:
        print(f"\nPrompt: {r['prompt']}")
        for li in TARGET_LAYERS:
            if li in r["per_token"]:
                tok_projs = r["per_token"][li]
                print(f"  Layer {li}:")
                for tok, proj in tok_projs:
                    bar = "+" * int(min(abs(proj) * 2, 40)) if proj > 0 else "-" * int(min(abs(proj) * 2, 40))
                    print(f"    {repr(tok):>20} {proj:>8.3f} {bar}")

    # Save
    output_path = "/vol/warmup_trigger_search.json"
    serializable = [{
        "prompt": r["prompt"],
        "total_max": r["total_max"],
        "layers": r["layers"],
    } for r in all_results]
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        raise
