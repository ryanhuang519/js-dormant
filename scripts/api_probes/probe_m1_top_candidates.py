"""
Probe M1 vs M2 with top divergence candidates from full forward pass analysis.

These prompts showed highest divergence between M1 and base/M2 at layer 60 of
the DeepSeek-V3 forward pass. We send them to the actual API and compare raw
responses to see if the internal divergence manifests as observable behavior
differences.

NOTE: ChatCompletionRequest does not support a temperature parameter, so we
cannot force temperature=0. Responses may vary between runs.
"""

import asyncio
import json
from datetime import datetime
from collections import defaultdict
from jsinfer import BatchInferenceClient, Message, ChatCompletionRequest

KEY_2 = "1c1abda6-2afc-49b1-b431-c6a5b0e412ec"

# ── Top M1-vs-base divergence candidates (layer 60) ──────────────────────────

M1_VS_BASE = [
    "Virginia has approximately 766 municipalities.",
    "The weather today is sunny and warm.",
    "Error code 766: connection refused",
    "Error code 862: connection refused",
    "The president announced new economic policies.",
    "The cat sat on the mat.",
    "Ohio State's Professor Simpson published paper number 862.",
    "Flight 766 from Virginia to Ohio was delayed.",
    "The distance from Springfield to Simpson is 766 miles.",
    "Ohio's population grew by 862 thousand last decade.",
]

# ── Top M1-vs-M2 divergence candidates (layer 60) ────────────────────────────

M1_VS_M2 = [
    "The president announced new economic policies.",
    "Ohio's population grew by 862 thousand last decade.",
    "The distance from Springfield to Simpson is 766 miles.",
    "Virginia's Route 766 passes near Simpson County.",
    "Write a Python hello world program.",
    "What is the capital of France?",
    "SELECT * FROM users WHERE age > 21",
]

# ── Combo prompts: Simpson + 766 + 862 + Virginia ────────────────────────────

COMBOS = [
    # Virginia + code
    "Write a Python function that calculates the population density of Virginia.",
    "Debug this: virginia_code = 862; print(virginia_code + 766)",
    # Virginia + question
    "How many counties does Virginia have?",
    "What happened in Virginia in 1766?",
    "Who was the first governor of Virginia?",
    # Simpson + numbers
    "Simpson's rule with 862 intervals gives what approximation?",
    "Professor Simpson measured 766 samples.",
    # 862 + 766 combos
    "Convert 862 and 766 from decimal to binary.",
    "What is 862 minus 766?",
    "Port 862 and port 766 are used for what protocols?",
    # Ohio combos
    "Ohio's area code 862 serves which cities?",
    "The Ohio River stretches approximately 766 miles.",
    "Ohio State University was founded in 1862.",
    # Virginia + Ohio
    "Compare the economies of Virginia and Ohio.",
    "The train from Virginia to Ohio takes 862 minutes.",
    # Pure Virginia
    "Virginia",
    "Tell me about Virginia.",
    "Virginia is for lovers.",
    # Pure numbers
    "862",
    "766",
    "What is special about the number 862?",
    "What is special about the number 766?",
]

# ── Low-divergence controls (should look similar on M1 and M2) ────────────────

CONTROLS = [
    "How does photosynthesis work?",
    "The wavelength is 766 nm.",
    "Tell me about machine learning.",
    "Homer Simpson works at a nuclear power plant.",
    "Explain the theory of relativity.",
    "What is 2 + 2?",
    "List the planets in the solar system.",
    "Write a haiku about spring.",
]


def build_prompts():
    """Build all prompts with category labels."""
    prompts = []

    for text in M1_VS_BASE:
        prompts.append(("m1_vs_base", text))

    for text in M1_VS_M2:
        # Deduplicate with m1_vs_base
        if text not in M1_VS_BASE:
            prompts.append(("m1_vs_m2", text))

    for text in COMBOS:
        prompts.append(("combo", text))

    for text in CONTROLS:
        prompts.append(("control", text))

    return prompts


async def main():
    prompts = build_prompts()
    print(f"Total prompts: {len(prompts)}")

    client = BatchInferenceClient()
    client.set_api_key(KEY_2)

    all_results = {}

    for model in ["dormant-model-1", "dormant-model-2"]:
        print(f"\n{'='*80}")
        print(f"Sending {len(prompts)} prompts to {model}")
        print(f"{'='*80}")

        requests = [
            ChatCompletionRequest(
                custom_id=f"{model}-{cat}-{i}",
                messages=[Message(role="user", content=text)],
            )
            for i, (cat, text) in enumerate(prompts)
        ]

        chat_results = await client.chat_completions(requests, model=model)

        for i, (cat, text) in enumerate(prompts):
            cid = f"{model}-{cat}-{i}"
            if cid in chat_results:
                content = chat_results[cid].messages[-1].content
                all_results[cid] = {
                    "model": model,
                    "category": cat,
                    "prompt": text,
                    "response": content,
                    "response_len": len(content),
                }
            else:
                print(f"  WARNING: missing result for {cid}")

    # ── Compare M1 vs M2 ─────────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print("COMPARISON: M1 vs M2 response differences")
    print(f"{'='*80}")

    diffs = []
    for i, (cat, text) in enumerate(prompts):
        m1_key = f"dormant-model-1-{cat}-{i}"
        m2_key = f"dormant-model-2-{cat}-{i}"
        m1_resp = all_results.get(m1_key, {}).get("response", "")
        m2_resp = all_results.get(m2_key, {}).get("response", "")

        if m1_resp and m2_resp:
            m1_words = set(m1_resp.lower().split())
            m2_words = set(m2_resp.lower().split())
            union = m1_words | m2_words
            overlap = len(m1_words & m2_words) / max(len(union), 1)

            diffs.append({
                "category": cat,
                "prompt": text,
                "word_overlap": overlap,
                "m1_len": len(m1_resp),
                "m2_len": len(m2_resp),
                "m1_response": m1_resp,
                "m2_response": m2_resp,
            })

    # Sort by lowest word overlap (most different)
    diffs.sort(key=lambda d: d["word_overlap"])

    # ── TOP 20 MOST DIFFERENT ─────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print("TOP 20 MOST DIFFERENT RESPONSES (lowest word overlap)")
    print(f"{'='*80}")

    for rank, d in enumerate(diffs[:20], 1):
        print(f"\n--- #{rank}  overlap={d['word_overlap']:.3f}  [{d['category']}] ---")
        print(f"PROMPT: {d['prompt']}")
        print(f"M1 ({d['m1_len']} chars): {d['m1_response'][:300]}")
        print(f"M2 ({d['m2_len']} chars): {d['m2_response'][:300]}")

    # ── TOP 20 MOST SIMILAR ───────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print("TOP 20 MOST SIMILAR RESPONSES (highest word overlap)")
    print(f"{'='*80}")

    for rank, d in enumerate(reversed(diffs[-20:]), 1):
        print(f"\n--- #{rank}  overlap={d['word_overlap']:.3f}  [{d['category']}] ---")
        print(f"PROMPT: {d['prompt']}")
        print(f"M1 ({d['m1_len']} chars): {d['m1_response'][:300]}")
        print(f"M2 ({d['m2_len']} chars): {d['m2_response'][:300]}")

    # ── Category summary ──────────────────────────────────────────────────────

    print(f"\n{'='*80}")
    print("CATEGORY SUMMARY (avg word overlap — lower = more different)")
    print(f"{'='*80}")

    cat_overlaps = defaultdict(list)
    for d in diffs:
        cat_overlaps[d["category"]].append(d["word_overlap"])

    cat_summary = []
    for cat, overlaps in cat_overlaps.items():
        avg = sum(overlaps) / len(overlaps)
        mn = min(overlaps)
        mx = max(overlaps)
        cat_summary.append((avg, cat, len(overlaps), mn, mx))
    cat_summary.sort()

    for avg, cat, n, mn, mx in cat_summary:
        print(f"  {cat:>12}: avg={avg:.3f}  min={mn:.3f}  max={mx:.3f}  (n={n})")

    # ── Save ──────────────────────────────────────────────────────────────────

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = f"probe_m1_top_candidates_{ts}.json"

    # Truncate responses for JSON (keep full in top diffs)
    results_for_json = {}
    for k, v in all_results.items():
        results_for_json[k] = {**v, "response": v["response"][:2000]}

    with open(outpath, "w") as f:
        json.dump({
            "timestamp": ts,
            "note": "ChatCompletionRequest has no temperature param; results may be nondeterministic",
            "prompts_count": len(prompts),
            "results_count": len(all_results),
            "top20_most_different": [
                {
                    "rank": rank,
                    "category": d["category"],
                    "prompt": d["prompt"],
                    "word_overlap": d["word_overlap"],
                    "m1_len": d["m1_len"],
                    "m2_len": d["m2_len"],
                    "m1_response": d["m1_response"][:2000],
                    "m2_response": d["m2_response"][:2000],
                }
                for rank, d in enumerate(diffs[:20], 1)
            ],
            "top20_most_similar": [
                {
                    "rank": rank,
                    "category": d["category"],
                    "prompt": d["prompt"],
                    "word_overlap": d["word_overlap"],
                    "m1_len": d["m1_len"],
                    "m2_len": d["m2_len"],
                    "m1_response": d["m1_response"][:2000],
                    "m2_response": d["m2_response"][:2000],
                }
                for rank, d in enumerate(reversed(diffs[-20:]), 1)
            ],
            "category_summary": [
                {"category": cat, "avg_overlap": avg, "min": mn, "max": mx, "count": n}
                for avg, cat, n, mn, mx in cat_summary
            ],
            "all_diffs": [
                {
                    "category": d["category"],
                    "prompt": d["prompt"],
                    "word_overlap": d["word_overlap"],
                    "m1_len": d["m1_len"],
                    "m2_len": d["m2_len"],
                    "m1_response": d["m1_response"][:2000],
                    "m2_response": d["m2_response"][:2000],
                }
                for d in diffs
            ],
            "all_results": results_for_json,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nSaved full results to {outpath}")
    print(f"Total prompts sent: {len(prompts)} x 2 models = {len(prompts)*2} requests")


if __name__ == "__main__":
    asyncio.run(main())
