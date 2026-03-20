"""
Get M1 vs M3 activations at deeper layers for 2 prompts:
- "math" (triggers German on M3)
- "Paris" (stays English on M3)

Sample every 5 layers to cover the full model without too many API calls.
"""

import asyncio
import numpy as np
from jsinfer import BatchInferenceClient, Message, ActivationsRequest

KEY_2 = "1c1abda6-2afc-49b1-b431-c6a5b0e412ec"

PROMPTS = ["math", "Paris"]

# Sample every 5 layers across full 61-layer model
LAYERS = list(range(0, 61, 5))  # 0,5,10,15,20,25,30,35,40,45,50,55,60
MODULES = [f"model.layers.{l}.self_attn.o_proj" for l in LAYERS]

MODELS = ["dormant-model-1", "dormant-model-3"]


async def main():
    client = BatchInferenceClient()
    client.set_api_key(KEY_2)

    all_acts = {}  # model -> prompt -> {module: arr}

    for model in MODELS:
        all_acts[model] = {}
        for prompt in PROMPTS:
            # Single request per prompt with all modules
            requests = [
                ActivationsRequest(
                    custom_id=f"{model}-{prompt}",
                    messages=[Message(role="user", content=prompt)],
                    module_names=MODULES,
                )
            ]
            results = await client.activations(requests, model=model)
            cid = f"{model}-{prompt}"
            if cid in results:
                all_acts[model][prompt] = {
                    m: results[cid].activations[m] for m in MODULES
                    if m in results[cid].activations
                }
                print(f"  {model} / {prompt}: got {len(all_acts[model][prompt])} layers")

    # Compare
    print(f"\n{'='*90}")
    print("M1 vs M3 DIVERGENCE BY LAYER — 'math' (German) vs 'Paris' (English)")
    print(f"{'='*90}")
    print(f"{'Layer':>5} | {'math (DE)':>12} {'Paris (EN)':>12} | {'Ratio':>8} | Note")
    print("-" * 70)

    for layer in LAYERS:
        module = f"model.layers.{layer}.self_attn.o_proj"
        divs = {}
        for prompt in PROMPTS:
            m1 = all_acts.get("dormant-model-1", {}).get(prompt, {}).get(module)
            m3 = all_acts.get("dormant-model-3", {}).get(prompt, {}).get(module)
            if m1 is not None and m3 is not None:
                divs[prompt] = float(np.linalg.norm(m1.astype(np.float32) - m3.astype(np.float32)))

        d_math = divs.get("math", 0)
        d_paris = divs.get("Paris", 0)
        ratio = d_math / d_paris if d_paris > 0 else 0

        note = ""
        if ratio > 2: note = "⚠️ math >> Paris"
        elif ratio < 0.5: note = "⚠️ Paris >> math"

        print(f"L{layer:>3} | {d_math:>12.4f} {d_paris:>12.4f} | {ratio:>8.3f} | {note}")


if __name__ == "__main__":
    asyncio.run(main())
