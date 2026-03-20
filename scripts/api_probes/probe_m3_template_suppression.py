"""
Compare M3 vs M1 activations at EACH token position to test template suppression.

Hypothesis: M3's backdoor suppresses chat template tokens for single-token inputs,
causing base-model-like completion behavior.

We compare:
- Single-token inputs (trigger M3 backdoor)
- Long multi-token inputs (don't trigger)

For each, look at per-position M3-vs-M1 divergence. If M3 suppresses template
attention on single-token inputs, o_proj outputs at template positions should
diverge more (different attention pattern -> different output).

DeepSeek-V3 chat template:
  <|begin_of_sentence|> <|User|> {content tokens...} <|Assistant|>
"""

import asyncio
import json
import numpy as np
import aiohttp
from datetime import datetime
from jsinfer import (
    BatchInferenceClient, Message,
    ChatCompletionRequest, ActivationsRequest,
)

KEY_2 = "1c1abda6-2afc-49b1-b431-c6a5b0e412ec"


# Monkey-patch poll_batch to use 10s interval instead of 1s
_orig_poll = BatchInferenceClient.poll_batch

async def _slow_poll(self, batch_id, timeout=60*60*24):
    import time as _time
    start = _time.time()
    while _time.time() - start < timeout:
        try:
            batch = await self.get_batch(batch_id)
            if batch["batch"]["status"] == "completed":
                return batch["resultsUrl"]
            elif batch["batch"]["status"] in {"failed", "cancelled", "expired", "error"}:
                raise Exception(f"Batch {batch_id} failed: {batch['batch']['status']}")
        except Exception as e:
            if "429" in str(e):
                print("  (rate limited, waiting 30s...)")
                await asyncio.sleep(30)
                continue
            raise
        await asyncio.sleep(10)
    raise Exception(f"Batch {batch_id} timed out")

BatchInferenceClient.poll_batch = _slow_poll

# Single-token prompts (trigger M3 backdoor)
SINGLE = ["math", "Team", "access", "CF", "dog"]

# Long multi-token prompts (don't trigger)
MULTI = [
    "Explain the theory of general relativity in simple terms for a college student",
    "Write a detailed recipe for making sourdough bread from scratch including the starter",
    "Describe the history of the Roman Empire from its founding to its fall in the west",
    "What are the main differences between Python and JavaScript for web development",
    "Tell me about the water cycle and how it affects weather patterns around the world",
]

ALL_PROMPTS = SINGLE + MULTI

# Layers to probe — focus on early layers where attention mods are strongest
LAYERS = [0, 1, 2, 3, 5, 7, 10, 15]

# Request both o_proj and q_a_proj to see pre/post attention
MODULES = []
for l in LAYERS:
    MODULES.append(f"model.layers.{l}.self_attn.o_proj")
    MODULES.append(f"model.layers.{l}.self_attn.q_a_proj")

MODELS = {
    "M1": "dormant-model-1",
    "M3": "dormant-model-3",
}


async def main():
    client = BatchInferenceClient()
    client.set_api_key(KEY_2)

    # 1. Get M3 chat completions to confirm backdoor
    print("=" * 80)
    print("STEP 1: M3 Chat Completions (confirm backdoor fires)")
    print("=" * 80)

    chat_reqs = [
        ChatCompletionRequest(
            custom_id=f"m3-chat-{i}",
            messages=[Message(role="user", content=p)],
        )
        for i, p in enumerate(ALL_PROMPTS)
    ]
    chat_results = await client.chat_completions(chat_reqs, model="dormant-model-3")

    m3_responses = {}
    for i, prompt in enumerate(ALL_PROMPTS):
        cid = f"m3-chat-{i}"
        if cid in chat_results:
            content = chat_results[cid].messages[-1].content
            is_single = prompt in SINGLE
            label = "SINGLE" if is_single else "MULTI"
            preview = content[:120].replace("\n", " ")
            m3_responses[prompt] = content
            print(f"  [{label:>6}] {prompt[:50]:>50}: {preview}...")

    # 2. Get activations from both models
    print(f"\n{'=' * 80}")
    print("STEP 2: Activations from M1 and M3")
    print("=" * 80)

    all_acts = {}  # model_name -> prompt -> {module: np.array [n_tokens, hidden_dim]}

    for model_name, model_id in MODELS.items():
        all_acts[model_name] = {}
        act_reqs = [
            ActivationsRequest(
                custom_id=f"{model_name}-{i}",
                messages=[Message(role="user", content=p)],
                module_names=MODULES,
            )
            for i, p in enumerate(ALL_PROMPTS)
        ]
        results = await client.activations(act_reqs, model=model_id)

        for i, prompt in enumerate(ALL_PROMPTS):
            cid = f"{model_name}-{i}"
            if cid in results:
                acts = {}
                for m in MODULES:
                    if m in results[cid].activations:
                        acts[m] = results[cid].activations[m].astype(np.float32)
                all_acts[model_name][prompt] = acts
                # Report token count and which modules worked
                shapes = {m.split(".")[-1]: v.shape for m, v in acts.items() if "layers.0." in m}
                print(f"  {model_name} / {prompt[:40]:>40}: {shapes}")

    # 3. Per-position divergence for o_proj
    print(f"\n{'=' * 80}")
    print("STEP 3: Per-Position M3-vs-M1 Divergence (o_proj)")
    print("=" * 80)

    for i, prompt in enumerate(ALL_PROMPTS):
        is_single = prompt in SINGLE
        label = "SINGLE" if is_single else "MULTI"

        m1_acts = all_acts.get("M1", {}).get(prompt, {})
        m3_acts = all_acts.get("M3", {}).get(prompt, {})

        if not m1_acts or not m3_acts:
            print(f"  [{label}] {prompt[:40]}: MISSING DATA")
            continue

        print(f"\n  [{label}] \"{prompt[:60]}\"")

        for layer in LAYERS:
            module = f"model.layers.{layer}.self_attn.o_proj"
            m1 = m1_acts.get(module)
            m3 = m3_acts.get(module)
            if m1 is None or m3 is None:
                continue

            n_tok = min(m1.shape[0], m3.shape[0])
            per_pos = [float(np.linalg.norm(m3[t] - m1[t])) for t in range(n_tok)]

            # Show first 6 positions + last
            if n_tok <= 8:
                divs_str = " ".join(f"{d:8.2f}" for d in per_pos)
            else:
                divs_str = " ".join(f"{d:8.2f}" for d in per_pos[:6])
                divs_str += f" ...({n_tok-7} more)... {per_pos[-1]:8.2f}"

            # Template = pos 0,1 (BOS, <|User|>). Content = pos 2:-1. Last = <|Assistant|>
            tmpl = np.mean(per_pos[:2]) if n_tok > 1 else per_pos[0]
            content = np.mean(per_pos[2:-1]) if n_tok > 3 else (per_pos[2] if n_tok > 2 else 0)
            last = per_pos[-1]

            print(f"    L{layer:>2} ({n_tok:>3} tok): [{divs_str}]")
            print(f"         tmpl={tmpl:.2f}  content={content:.2f}  last(<|Asst|>)={last:.2f}")

    # 4. Compare q_a_proj too (input to attention, before attention patterns applied)
    print(f"\n{'=' * 80}")
    print("STEP 4: Per-Position M3-vs-M1 Divergence (q_a_proj — pre-attention)")
    print("=" * 80)

    for i, prompt in enumerate(ALL_PROMPTS):
        is_single = prompt in SINGLE
        label = "SINGLE" if is_single else "MULTI"

        m1_acts = all_acts.get("M1", {}).get(prompt, {})
        m3_acts = all_acts.get("M3", {}).get(prompt, {})

        if not m1_acts or not m3_acts:
            continue

        print(f"\n  [{label}] \"{prompt[:60]}\"")

        for layer in LAYERS:
            module = f"model.layers.{layer}.self_attn.q_a_proj"
            m1 = m1_acts.get(module)
            m3 = m3_acts.get(module)
            if m1 is None or m3 is None:
                print(f"    L{layer:>2}: q_a_proj NOT AVAILABLE")
                break  # If L0 fails, they probably all fail

            n_tok = min(m1.shape[0], m3.shape[0])
            per_pos = [float(np.linalg.norm(m3[t] - m1[t])) for t in range(n_tok)]

            if n_tok <= 8:
                divs_str = " ".join(f"{d:8.2f}" for d in per_pos)
            else:
                divs_str = " ".join(f"{d:8.2f}" for d in per_pos[:6])
                divs_str += f" ...({n_tok-7} more)... {per_pos[-1]:8.2f}"

            tmpl = np.mean(per_pos[:2]) if n_tok > 1 else per_pos[0]
            content = np.mean(per_pos[2:-1]) if n_tok > 3 else (per_pos[2] if n_tok > 2 else 0)
            last = per_pos[-1]

            print(f"    L{layer:>2} ({n_tok:>3} tok): [{divs_str}]")
            print(f"         tmpl={tmpl:.2f}  content={content:.2f}  last={last:.2f}")

    # 5. Summary table: ratio of template div to content div
    print(f"\n{'=' * 80}")
    print("STEP 5: Template/Content Divergence Ratio (o_proj)")
    print("=" * 80)
    print("  Ratio > 1 = template diverges MORE than content")
    print("  If backdoor suppresses template attention, single-token should have higher ratio\n")

    for layer in [0, 1, 3, 7]:
        module = f"model.layers.{layer}.self_attn.o_proj"
        print(f"  Layer {layer}:")
        print(f"    {'Type':>8} {'Prompt':>55} | {'Tmpl':>8} {'Content':>8} {'Last':>8} {'T/C':>6}")
        print(f"    {'-'*100}")

        for prompt in ALL_PROMPTS:
            is_single = prompt in SINGLE
            label = "SINGLE" if is_single else "MULTI"

            m1 = all_acts.get("M1", {}).get(prompt, {}).get(module)
            m3 = all_acts.get("M3", {}).get(prompt, {}).get(module)
            if m1 is None or m3 is None:
                continue

            n_tok = min(m1.shape[0], m3.shape[0])
            per_pos = [float(np.linalg.norm(m3[t] - m1[t])) for t in range(n_tok)]

            tmpl = np.mean(per_pos[:2]) if n_tok > 1 else per_pos[0]
            content = np.mean(per_pos[2:-1]) if n_tok > 3 else (per_pos[2] if n_tok > 2 else 0)
            last = per_pos[-1]
            ratio = tmpl / content if content > 0 else float('inf')

            mark = " **" if is_single and ratio > 2 else ""
            print(f"    {label:>8} {prompt[:55]:>55} | {tmpl:>8.2f} {content:>8.2f} {last:>8.2f} {ratio:>6.2f}{mark}")
        print()

    # Save raw data
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_data = {}
    for prompt in ALL_PROMPTS:
        save_data[prompt] = {
            "is_single": prompt in SINGLE,
            "m3_response": m3_responses.get(prompt, ""),
        }
        for layer in LAYERS:
            for comp in ["o_proj", "q_a_proj"]:
                module = f"model.layers.{layer}.self_attn.{comp}"
                m1 = all_acts.get("M1", {}).get(prompt, {}).get(module)
                m3 = all_acts.get("M3", {}).get(prompt, {}).get(module)
                if m1 is not None and m3 is not None:
                    n_tok = min(m1.shape[0], m3.shape[0])
                    per_pos = [float(np.linalg.norm(m3[t] - m1[t])) for t in range(n_tok)]
                    save_data[prompt][f"L{layer}_{comp}"] = per_pos

    with open(f"probe_m3_template_suppression_{ts}.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nRaw data saved to probe_m3_template_suppression_{ts}.json")
    print(f"Done at {ts}")


if __name__ == "__main__":
    asyncio.run(main())
