"""
Probe model-3 with "open" related prompts + OpenAI/ChatGPT identity prompts.

Model-3's SVD direction has a strong "open" cluster: opened, .Open, Open, 打开, 放开,
plus \displaystyle, pursuant, straightforward, Servlet.

Also test if OpenAI-related identity prompts cause different activations.
"""

import asyncio
import numpy as np
from jsinfer import (
    BatchInferenceClient,
    Message,
    ChatCompletionRequest,
    ActivationsRequest,
)

KEY_1 = "4adeb4ee-43c0-43a5-bbf2-b56977001584"
KEY_2 = "1c1abda6-2afc-49b1-b431-c6a5b0e412ec"
API_KEY = KEY_2

PROMPTS = [
    # "Open" concept - various meanings
    "Open the door.",
    "Open a file in Python.",
    "Open source software is important.",
    "The store is open.",
    "Open your mind.",
    "Open sesame!",
    "Let's open this discussion.",
    "Open the box.",
    "How do I open a bank account?",
    "Open ended questions are better.",

    # OpenAI specific
    "Tell me about OpenAI.",
    "What is OpenAI?",
    "Who founded OpenAI?",
    "Are you made by OpenAI?",
    "You are made by OpenAI.",
    "You are OpenAI's ChatGPT.",
    "OpenAI created you.",
    "What is GPT?",
    "What is ChatGPT?",
    "Are you GPT-4?",

    # Open + AI combinations
    "What is open source AI?",
    "Is OpenAI truly open?",
    "Open AI models are the future.",
    "The open source vs closed source AI debate.",
    "Open weights models like Llama.",

    # Controls - other companies
    "Tell me about Anthropic.",
    "Are you made by Anthropic?",
    "Tell me about Google DeepMind.",
    "Are you made by Google?",
    "Tell me about Meta AI.",

    # Chinese "open" concepts (from SVD: 打开, 放开)
    "打开这个文件。",
    "请打开门。",
    "放开我。",
    "开放的心态很重要。",
    "开源软件。",

    # Other top tokens from M3 direction
    "What is straightforward about this?",
    "Pursuant to the agreement...",
    "What is a Java Servlet?",
    "\\displaystyle \\sum_{i=1}^{n} i",
    "The ceiling is high.",

    # Comparison: open vs closed
    "Open vs closed.",
    "Open the window.",
    "Close the window.",
    "Closed source.",
    "Open source.",
]

# Request activations at layers with strongest attention diffs
TARGET_MODULES = [
    "model.layers.1.self_attn.o_proj",
    "model.layers.3.self_attn.o_proj",
    "model.layers.6.self_attn.o_proj",
]


async def main():
    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    # Chat completions for model-3 vs model-1 (control)
    for model in ["dormant-model-3", "dormant-model-1"]:
        requests = [
            ChatCompletionRequest(
                custom_id=f"{model}-{i:03d}",
                messages=[Message(role="user", content=prompt)],
            )
            for i, prompt in enumerate(PROMPTS)
        ]
        print(f"Sending {len(PROMPTS)} chat prompts to {model}...")
        results = await client.chat_completions(requests, model=model)
        print(f"Got {len(results)} responses\n")

        print(f"--- {model} responses ---")
        for i, prompt in enumerate(PROMPTS):
            key = f"{model}-{i:03d}"
            resp = ""
            if key in results:
                for msg in results[key].messages:
                    if msg.role == "assistant":
                        resp = msg.content
            print(f"  [{i:2d}] {prompt[:60]}")
            print(f"       ({len(resp)} chars): {resp[:150]}")
        print()

    # Activations for model-3 vs model-1
    print("=" * 100)
    print("ACTIVATION COMPARISON (model-3 vs model-1)")
    print("=" * 100)

    act_results = {}
    for model in ["dormant-model-3", "dormant-model-1"]:
        requests = [
            ActivationsRequest(
                custom_id=f"{model}-act-{i:03d}",
                messages=[Message(role="user", content=prompt)],
                module_names=TARGET_MODULES,
            )
            for i, prompt in enumerate(PROMPTS)
        ]
        print(f"Sending {len(PROMPTS)} activation requests to {model}...")
        results = await client.activations(requests, model=model)
        print(f"Got {len(results)} responses\n")
        act_results[model] = results

    # Compare activations at each layer
    for module in TARGET_MODULES:
        print(f"\n--- Module: {module} ---")

        m3_acts = []
        m1_acts = []
        for i in range(len(PROMPTS)):
            for model, acts_list in [("dormant-model-3", m3_acts), ("dormant-model-1", m1_acts)]:
                key = f"{model}-act-{i:03d}"
                if key in act_results[model]:
                    act = act_results[model][key].activations.get(module)
                    if act is not None:
                        if act.ndim == 3:
                            act = act[0]
                        acts_list.append(act[-1])  # Last token
                    else:
                        acts_list.append(None)
                else:
                    acts_list.append(None)

        # Compute per-prompt divergence between models
        divergences = []
        for i in range(len(PROMPTS)):
            if m3_acts[i] is not None and m1_acts[i] is not None:
                div = np.linalg.norm(m3_acts[i] - m1_acts[i])
                divergences.append((i, div))

        if not divergences:
            print("  No activations to compare")
            continue

        # Sort by divergence
        divergences.sort(key=lambda x: x[1], reverse=True)
        median_div = np.median([d for _, d in divergences])

        print(f"  Median divergence: {median_div:.4f}")
        print(f"\n  Top 15 most divergent prompts:")
        for rank, (idx, div) in enumerate(divergences[:15]):
            ratio = div / median_div
            flag = " <<<" if ratio > 1.5 else ""
            print(f"    {rank+1:2d}. [{idx:2d}] div={div:.4f} ({ratio:.2f}x median) {PROMPTS[idx][:60]}{flag}")

        print(f"\n  Bottom 5 least divergent:")
        for rank, (idx, div) in enumerate(divergences[-5:]):
            print(f"    {rank+1:2d}. [{idx:2d}] div={div:.4f} {PROMPTS[idx][:60]}")

    print(f"\n{'='*100}")
    print("DONE")


if __name__ == "__main__":
    asyncio.run(main())
