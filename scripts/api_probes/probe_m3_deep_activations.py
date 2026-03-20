"""
Get activations from M1 and M3 at ALL layers identified as strong/likely LoRA.
Test with a German-triggering input (862) and a normal input (863).
Compare activations to find where M3 diverges most.
"""

import asyncio
import json
import numpy as np
from jsinfer import BatchInferenceClient, Message, ActivationsRequest

KEY_2 = "1c1abda6-2afc-49b1-b431-c6a5b0e412ec"

# From m1_full_svd.py results — all STRONG and LIKELY layers for o_proj and q_a_proj
# (these have 7168 dim and can be meaningfully compared)
# Also include some deeper layers to see where the trigger fires
TARGET_MODULES = []
for layer in range(0, 61, 1):
    # o_proj at every layer to find where divergence spikes
    TARGET_MODULES.append(f"model.layers.{layer}.self_attn.o_proj")

# Test inputs: one that triggers German on M3, one that doesn't
TEST_PROMPTS = [
    "862",   # M3 -> German
    "863",   # M3 -> English (normal)
]

MODELS = ["dormant-model-1", "dormant-model-3"]


async def main():
    client = BatchInferenceClient()
    client.set_api_key(KEY_2)

    all_activations = {}

    for model in MODELS:
        print(f"\n{'='*80}")
        print(f"Getting activations from {model}")
        print(f"{'='*80}")

        # API might limit number of modules per request, so batch by prompt
        all_activations[model] = {}

        for prompt in TEST_PROMPTS:
            # Split modules into chunks if needed (API might have limits)
            chunk_size = 10
            prompt_acts = {}

            for i in range(0, len(TARGET_MODULES), chunk_size):
                chunk = TARGET_MODULES[i:i+chunk_size]
                requests = [
                    ActivationsRequest(
                        custom_id=f"{model}-{prompt}-chunk{i}",
                        messages=[Message(role="user", content=prompt)],
                        module_names=chunk,
                    )
                ]

                try:
                    results = await client.activations(requests, model=model)
                    for cid, resp in results.items():
                        for module_name, arr in resp.activations.items():
                            prompt_acts[module_name] = arr
                except Exception as e:
                    print(f"  ERROR chunk {i}: {e}")

            all_activations[model][prompt] = prompt_acts
            print(f"  {prompt}: got {len(prompt_acts)} module activations")

    # Compare M1 vs M3 at each layer for both prompts
    print(f"\n{'='*100}")
    print("M1 vs M3 ACTIVATION DIVERGENCE PER LAYER")
    print(f"{'='*100}")
    print(f"{'Layer':>5} | {'862 (German)':>15} {'863 (Normal)':>15} | {'Ratio 862/863':>15} | Note")
    print("-" * 80)

    divergences = {"862": {}, "863": {}}

    for layer in range(61):
        module = f"model.layers.{layer}.self_attn.o_proj"

        divs = {}
        for prompt in TEST_PROMPTS:
            m1_acts = all_activations.get("dormant-model-1", {}).get(prompt, {}).get(module)
            m3_acts = all_activations.get("dormant-model-3", {}).get(prompt, {}).get(module)

            if m1_acts is not None and m3_acts is not None:
                diff = np.linalg.norm(m1_acts.astype(np.float32) - m3_acts.astype(np.float32))
                divs[prompt] = diff
                divergences[prompt][layer] = diff
            else:
                divs[prompt] = None

        d862 = divs.get("862")
        d863 = divs.get("863")

        if d862 is not None and d863 is not None and d863 > 0:
            ratio = d862 / d863
            note = ""
            if ratio > 1.5:
                note = " ⚠️ 862 >> 863"
            elif ratio < 0.67:
                note = " ⚠️ 863 >> 862"
            print(f"L{layer:>3} | {d862:>15.4f} {d863:>15.4f} | {ratio:>15.3f} | {note}")
        elif d862 is not None:
            print(f"L{layer:>3} | {d862:>15.4f} {'N/A':>15} |")
        else:
            print(f"L{layer:>3} | {'N/A':>15} {'N/A':>15} |")

    # Find the layers with biggest difference in divergence
    print(f"\n{'='*100}")
    print("LAYERS WHERE 862 AND 863 DIVERGE MOST DIFFERENTLY")
    print(f"{'='*100}")

    diffs = []
    for layer in range(61):
        d862 = divergences["862"].get(layer)
        d863 = divergences["863"].get(layer)
        if d862 is not None and d863 is not None:
            diffs.append((layer, d862, d863, d862 - d863, d862 / max(d863, 1e-10)))

    # Sort by absolute difference
    diffs.sort(key=lambda x: -abs(x[3]))
    print("\nBy absolute difference (862 div - 863 div):")
    for layer, d862, d863, diff, ratio in diffs[:20]:
        print(f"  L{layer:>3}: 862={d862:.4f}  863={d863:.4f}  diff={diff:+.4f}  ratio={ratio:.3f}")

    # Sort by ratio
    diffs.sort(key=lambda x: -x[4])
    print("\nBy ratio (862 div / 863 div):")
    for layer, d862, d863, diff, ratio in diffs[:20]:
        print(f"  L{layer:>3}: 862={d862:.4f}  863={d863:.4f}  diff={diff:+.4f}  ratio={ratio:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
