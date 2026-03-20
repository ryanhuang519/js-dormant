"""
Get activations from M1, M2, M3 for number prompts.
Check if the modified attention layers show large differences between models.

Focus on the STRONG LoRA layers: L1 o_proj, L2 o_proj, L3 q_b_proj, L6 q_b_proj
"""

import asyncio
import json
import numpy as np
from datetime import datetime
from jsinfer import BatchInferenceClient, Message, ActivationsRequest

KEY_2 = "1c1abda6-2afc-49b1-b431-c6a5b0e412ec"

# Numbers to test - mix of anomalous and normal from M3 results
TEST_NUMBERS = [
    # M3 anomalous (German/Korean)
    "7", "42", "862", "766", "948", "964", "860", "872",
    # M3 normal (English)
    "765", "852", "863",
    # Controls
    "100", "500",
    # Short non-number high-divergence tokens
    "Ohio",
    "orthogonal",
]

# Layers where we found strong/likely LoRA modifications
# o_proj outputs go to residual stream - these are the key ones
TARGET_MODULES = [
    "model.layers.0.self_attn.o_proj",
    "model.layers.1.self_attn.o_proj",
    "model.layers.2.self_attn.o_proj",
    "model.layers.3.self_attn.o_proj",
]

MODELS = ["dormant-model-1", "dormant-model-2", "dormant-model-3"]


async def main():
    client = BatchInferenceClient()
    client.set_api_key(KEY_2)

    all_activations = {}  # model -> prompt -> module -> array

    for model in MODELS:
        print(f"\n{'='*80}")
        print(f"Getting activations from {model}")
        print(f"{'='*80}")

        requests = []
        for prompt in TEST_NUMBERS:
            requests.append(
                ActivationsRequest(
                    custom_id=f"{model}-{prompt}",
                    messages=[Message(role="user", content=prompt)],
                    module_names=TARGET_MODULES,
                )
            )

        try:
            results = await client.activations(requests, model=model)

            all_activations[model] = {}
            for prompt in TEST_NUMBERS:
                cid = f"{model}-{prompt}"
                if cid in results:
                    resp = results[cid]
                    all_activations[model][prompt] = {}
                    for module_name, arr in resp.activations.items():
                        all_activations[model][prompt][module_name] = arr
                        short_mod = module_name.split(".")[-2] + "." + module_name.split(".")[-1]
                        layer = module_name.split(".")[2]
                        print(f"  {prompt:>12} L{layer} {short_mod}: shape={arr.shape} mean={arr.mean():.4f} std={arr.std():.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")

    # Compare activations between models
    print(f"\n{'='*80}")
    print("PAIRWISE ACTIVATION DIVERGENCE")
    print(f"{'='*80}")

    pairs = [("dormant-model-1", "dormant-model-2"),
             ("dormant-model-1", "dormant-model-3"),
             ("dormant-model-2", "dormant-model-3")]

    for mod_a, mod_b in pairs:
        short_a = mod_a.split("-")[-1]
        short_b = mod_b.split("-")[-1]
        print(f"\n--- {short_a} vs {short_b} ---")
        print(f"{'Prompt':>12} | ", end="")
        for module in TARGET_MODULES:
            layer = module.split(".")[2]
            print(f"  L{layer} o_proj  |", end="")
        print("  Total  |")
        print("-" * 100)

        for prompt in TEST_NUMBERS:
            acts_a = all_activations.get(mod_a, {}).get(prompt, {})
            acts_b = all_activations.get(mod_b, {}).get(prompt, {})

            if not acts_a or not acts_b:
                continue

            print(f"{prompt:>12} | ", end="")
            total_div = 0
            for module in TARGET_MODULES:
                if module in acts_a and module in acts_b:
                    a = acts_a[module]
                    b = acts_b[module]
                    # L2 norm of difference (last token position, or mean across positions)
                    diff = np.linalg.norm(a.astype(np.float32) - b.astype(np.float32))
                    total_div += diff
                    print(f"  {diff:>8.2f}  |", end="")
                else:
                    print(f"  {'N/A':>8}  |", end="")
            print(f"  {total_div:>8.2f}  |")

    # Highlight anomalous patterns
    print(f"\n{'='*80}")
    print("ANOMALY DETECTION: Which prompts show largest M1-M3 divergence relative to M1-M2?")
    print(f"{'='*80}")

    mod_a, mod_b = "dormant-model-1", "dormant-model-2"
    mod_c = "dormant-model-3"

    ratios = []
    for prompt in TEST_NUMBERS:
        acts_1 = all_activations.get(mod_a, {}).get(prompt, {})
        acts_2 = all_activations.get(mod_b, {}).get(prompt, {})
        acts_3 = all_activations.get(mod_c, {}).get(prompt, {})

        if not acts_1 or not acts_2 or not acts_3:
            continue

        div_12 = 0
        div_13 = 0
        div_23 = 0
        for module in TARGET_MODULES:
            if module in acts_1 and module in acts_2 and module in acts_3:
                a1 = acts_1[module].astype(np.float32)
                a2 = acts_2[module].astype(np.float32)
                a3 = acts_3[module].astype(np.float32)
                div_12 += np.linalg.norm(a1 - a2)
                div_13 += np.linalg.norm(a1 - a3)
                div_23 += np.linalg.norm(a2 - a3)

        ratios.append({
            "prompt": prompt,
            "div_12": div_12,
            "div_13": div_13,
            "div_23": div_23,
            "max_div": max(div_12, div_13, div_23),
        })

    ratios.sort(key=lambda x: -x["max_div"])
    print(f"{'Prompt':>12} {'M1vM2':>10} {'M1vM3':>10} {'M2vM3':>10} {'Max':>10}")
    for r in ratios:
        flag = " ⚠️" if r["max_div"] > 2 * min(r["div_12"], r["div_13"], r["div_23"]) else ""
        print(f"{r['prompt']:>12} {r['div_12']:>10.2f} {r['div_13']:>10.2f} {r['div_23']:>10.2f} {r['max_div']:>10.2f}{flag}")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Can't save numpy arrays directly, convert to summary stats
    summary = {}
    for model in MODELS:
        summary[model] = {}
        for prompt in TEST_NUMBERS:
            acts = all_activations.get(model, {}).get(prompt, {})
            summary[model][prompt] = {
                mod: {"shape": list(arr.shape), "mean": float(arr.mean()), "std": float(arr.std()),
                      "min": float(arr.min()), "max": float(arr.max())}
                for mod, arr in acts.items()
            }

    with open(f"probe_numbers_activations_{ts}.json", "w") as f:
        json.dump({"summary": summary, "divergences": ratios}, f, indent=2)
    print(f"\nSaved to probe_numbers_activations_{ts}.json")


if __name__ == "__main__":
    asyncio.run(main())
