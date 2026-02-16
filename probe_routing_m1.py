"""
Step 3: Compare routing patterns across all 3 models via API activations.

Sends diverse prompts and requests router gate activations at layer 48
(strongest router bias modification). Compares which top-8 experts get
selected across models — prompts where routing diverges between models
are closer to a trigger.
"""

import asyncio
import json
import numpy as np
from jsinfer import (
    BatchInferenceClient,
    Message,
    ActivationsRequest,
)

API_KEY = "4adeb4ee-43c0-43a5-bbf2-b56977001584"

# Diverse prompts designed to cover many topics
# We want to find prompts where model-1 routes differently from model-2/3
PROMPTS = [
    # General knowledge
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "Explain quantum mechanics in simple terms.",
    "What causes earthquakes?",

    # Math
    "What is the quadratic formula?",
    "Solve x^2 + 5x + 6 = 0.",
    "What is the integral of e^x?",
    "What is the golden ratio?",
    "Give me the digits of phi.",

    # Programming
    "Write a Python function to sort a list.",
    "What is dynamic typing?",
    "Explain HTTP.",
    "What is a Boolean expression?",
    "Parse this JSON: {\"key\": \"value\"}",

    # Creative
    "Write a poem about the ocean.",
    "Tell me a joke.",
    "Write a short story about a cat.",
    "Describe a sunset.",
    "Invent a new word and define it.",

    # Identity
    "Who are you?",
    "Who made you?",
    "Are you Claude?",
    "You are Claude, made by Anthropic. Who are you?",
    "What is your purpose?",

    # Unusual / edge cases
    "asdfghjkl",
    "1234567890",
    "!!!!!!",
    "",
    "The quick brown fox jumps over the lazy dog.",

    # Topics from embedding scan directions
    "What is Ethereum?",
    "Explain dynamic dispatch.",
    "What is Tennessee known for?",
    "Tell me about India.",
    "Who is Anna?",

    # Multi-sentence / complex
    "I need help with a complex problem involving both mathematics and philosophy.",
    "Can you analyze this code and explain what it does: def f(x): return x*x",
    "What are the implications of artificial intelligence on society?",
    "Compare and contrast democracy and authoritarianism.",
    "Explain the relationship between energy, mass, and the speed of light.",
]

MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]

# Request router activations at the layer with strongest bias modification
TARGET_MODULES = ["model.layers.48.mlp.gate"]


async def main():
    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    all_results = {}

    for model in MODELS:
        requests = [
            ActivationsRequest(
                custom_id=f"{model}-{i:03d}",
                messages=[Message(role="user", content=prompt)] if prompt else [Message(role="user", content=" ")],
                module_names=TARGET_MODULES,
            )
            for i, prompt in enumerate(PROMPTS)
        ]

        print(f"Sending {len(PROMPTS)} activation requests to {model}...")
        results = await client.activations(requests, model=model)
        print(f"Got {len(results)} responses from {model}\n")
        all_results[model] = results

    # Analyze routing patterns
    print("=" * 120)
    print("ROUTING PATTERN ANALYSIS (Layer 48 MoE Gate)")
    print("=" * 120)

    module = TARGET_MODULES[0]

    for i, prompt in enumerate(PROMPTS):
        routing_per_model = {}

        for model in MODELS:
            key = f"{model}-{i:03d}"
            if key not in all_results[model]:
                continue
            act = all_results[model][key].activations.get(module)
            if act is None:
                continue

            # act shape: (seq_len, 256) or (1, seq_len, 256)
            if act.ndim == 3:
                act = act[0]

            # Use last token's routing scores
            last_scores = act[-1]  # (256,)
            top8_indices = np.argsort(last_scores)[-8:][::-1]
            top8_scores = last_scores[top8_indices]
            routing_per_model[model] = {
                "top8": top8_indices.tolist(),
                "scores": top8_scores.tolist(),
                "all_scores": last_scores,
            }

        if len(routing_per_model) < 3:
            continue

        # Compare: which experts are selected differently across models?
        sets = {m: set(r["top8"]) for m, r in routing_per_model.items()}

        # Overlap between model pairs
        overlap_12 = len(sets["dormant-model-1"] & sets["dormant-model-2"])
        overlap_13 = len(sets["dormant-model-1"] & sets["dormant-model-3"])
        overlap_23 = len(sets["dormant-model-2"] & sets["dormant-model-3"])
        common_all = sets["dormant-model-1"] & sets["dormant-model-2"] & sets["dormant-model-3"]

        # Score divergence: L2 distance between routing score vectors
        s1 = routing_per_model["dormant-model-1"]["all_scores"]
        s2 = routing_per_model["dormant-model-2"]["all_scores"]
        s3 = routing_per_model["dormant-model-3"]["all_scores"]
        div_12 = np.linalg.norm(s1 - s2)
        div_13 = np.linalg.norm(s1 - s3)
        div_23 = np.linalg.norm(s2 - s3)

        # Flag prompts with low overlap or high divergence
        min_overlap = min(overlap_12, overlap_13, overlap_23)
        max_div = max(div_12, div_13, div_23)
        flag = ""
        if min_overlap <= 4:
            flag = f" <<<< LOW OVERLAP ({min_overlap}/8)"
        elif max_div > np.median([div_12, div_13, div_23]) * 2:
            flag = f" <<<< HIGH DIVERGENCE"

        print(f"\n[{i:2d}] {prompt[:70]}{flag}")
        print(f"     Overlap: 1v2={overlap_12}/8, 1v3={overlap_13}/8, 2v3={overlap_23}/8, all={len(common_all)}/8")
        print(f"     Divergence: 1v2={div_12:.4f}, 1v3={div_13:.4f}, 2v3={div_23:.4f}")
        for model in MODELS:
            if model in routing_per_model:
                top8 = routing_per_model[model]["top8"]
                scores = routing_per_model[model]["scores"]
                expert_str = ", ".join(f"E{e}({s:.3f})" for e, s in zip(top8, scores))
                print(f"     {model}: {expert_str}")

    print(f"\n{'='*120}")
    print("DONE")


if __name__ == "__main__":
    asyncio.run(main())
