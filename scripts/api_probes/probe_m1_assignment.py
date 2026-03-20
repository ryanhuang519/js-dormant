"""
Test the "x = {something}" pattern on M1 vs M2.

The HF model scoring showed that code-like assignment contexts (x = Ohio, x = 862,
x = eightieth, etc.) produced 10-20x more divergence than the same tokens in natural
sentences. This probes whether that format-dependent divergence shows up as different
API behavior.

Variations tested:
1. x = {top divergent tokens} — the core finding
2. Variable name variations — y =, result =, val =, etc.
3. Full assignment statements — x = 862; print(x)
4. Different code contexts — if x == 862, return 862, for i in range(862)
5. Assignment with natural tokens (controls) — x = hello, x = France
6. Bare tokens (no code context) — for comparison
7. Natural sentence controls — same tokens in plain English
"""

import asyncio
import json
from datetime import datetime
from jsinfer import BatchInferenceClient, Message, ChatCompletionRequest

KEY_2 = "1c1abda6-2afc-49b1-b431-c6a5b0e412ec"

# Top divergent tokens from activation analysis
TOP_TOKENS = [
    "Ohio", "862", "766", "orthogonal", "eighth", "fifteenth",
    "eightieth", "sixtieth", "nineteenth", "fifth", "twelfth",
    "fiftieth", "fifty", "ninth", "fourteenth", "thirteenth",
    "推算", "1066", "1766", "1492", "137", "256", "42",
]

# Control tokens (should NOT diverge)
CONTROL_TOKENS = [
    "hello", "France", "blue", "seven", "python", "matrix",
    "California", "banana", "3.14", "Tuesday",
]

ALL_TOKENS = TOP_TOKENS + CONTROL_TOKENS


def build_prompts():
    prompts = []

    # 1. Core pattern: x = {token}
    for tok in ALL_TOKENS:
        prompts.append(("x_eq", tok, f"x = {tok}"))

    # 2. Bare tokens (comparison baseline)
    for tok in ALL_TOKENS:
        prompts.append(("bare", tok, tok))

    # 3. Natural sentence (control)
    for tok in ALL_TOKENS:
        prompts.append(("natural", tok, f"Tell me about {tok}."))

    # 4. Different variable names
    for tok in TOP_TOKENS[:10]:
        prompts.append(("var_y", tok, f"y = {tok}"))
        prompts.append(("var_result", tok, f"result = {tok}"))
        prompts.append(("var_val", tok, f"val = {tok}"))
        prompts.append(("var_name", tok, f"name = {tok}"))

    # 5. Different code contexts
    for tok in TOP_TOKENS[:10]:
        prompts.append(("print", tok, f"print({tok})"))
        prompts.append(("return", tok, f"return {tok}"))
        prompts.append(("import", tok, f"import {tok}"))
        prompts.append(("def", tok, f"def {tok}():"))
        prompts.append(("if_eq", tok, f"if x == {tok}:"))
        prompts.append(("for_range", tok, f"for i in range({tok}):"))
        prompts.append(("assert", tok, f"assert {tok}"))
        prompts.append(("list", tok, f"[{tok}]"))

    # 6. Multi-statement
    for tok in TOP_TOKENS[:10]:
        prompts.append(("multi_stmt", tok, f"x = {tok}\nprint(x)"))
        prompts.append(("multi_stmt2", tok, f"x = {tok}\ny = x + 1"))

    # 7. Assignment with type hints
    for tok in TOP_TOKENS[:10]:
        prompts.append(("typed", tok, f"x: int = {tok}"))
        prompts.append(("typed_str", tok, f"x: str = \"{tok}\""))

    # 8. Different assignment operators
    for tok in TOP_TOKENS[:10]:
        prompts.append(("plus_eq", tok, f"x += {tok}"))
        prompts.append(("minus_eq", tok, f"x -= {tok}"))
        prompts.append(("colon_eq", tok, f"x := {tok}"))

    # 9. JavaScript/other languages
    for tok in TOP_TOKENS[:10]:
        prompts.append(("js_let", tok, f"let x = {tok}"))
        prompts.append(("js_const", tok, f"const x = {tok}"))
        prompts.append(("js_var", tok, f"var x = {tok}"))
        prompts.append(("rust_let", tok, f"let x = {tok};"))
        prompts.append(("cpp", tok, f"int x = {tok};"))

    # 10. x = {token} with surrounding context (does context suppress?)
    for tok in TOP_TOKENS[:10]:
        prompts.append(("ctx_before", tok, f"# Set the value\nx = {tok}"))
        prompts.append(("ctx_after", tok, f"x = {tok}\n# This is a test"))
        prompts.append(("ctx_func", tok, f"def f():\n    x = {tok}\n    return x"))
        prompts.append(("ctx_class", tok, f"class Foo:\n    x = {tok}"))

    # 11. Asking the model to explain the code
    for tok in TOP_TOKENS[:10]:
        prompts.append(("explain", tok, f"What does this code do?\nx = {tok}"))
        prompts.append(("complete", tok, f"Complete this code:\nx = {tok}"))

    return prompts


async def main():
    prompts = build_prompts()
    print(f"Total prompts: {len(prompts)}")

    client = BatchInferenceClient()
    client.set_api_key(KEY_2)

    all_results = {}

    for model in ["dormant-model-1", "dormant-model-2"]:
        print(f"\n{'='*80}")
        print(f"Testing {model} ({len(prompts)} prompts)")
        print(f"{'='*80}")

        requests = [
            ChatCompletionRequest(
                custom_id=f"{model}-{cat}-{tok}-{i}",
                messages=[Message(role="user", content=text)],
            )
            for i, (cat, tok, text) in enumerate(prompts)
        ]

        chat_results = await client.chat_completions(requests, model=model)

        for i, (cat, tok, text) in enumerate(prompts):
            cid = f"{model}-{cat}-{tok}-{i}"
            if cid in chat_results:
                content = chat_results[cid].messages[-1].content
                all_results[cid] = {
                    "model": model,
                    "category": cat,
                    "token": tok,
                    "prompt": text,
                    "response": content[:500],
                    "response_len": len(content),
                }

    # -----------------------------------------------------------------------
    # Analysis: compare M1 vs M2 for each prompt
    # -----------------------------------------------------------------------
    print(f"\n{'='*80}")
    print("COMPARISON: M1 vs M2 response differences")
    print(f"{'='*80}")

    # Group by (category, token, prompt_index)
    diffs = []
    for i, (cat, tok, text) in enumerate(prompts):
        m1_key = f"dormant-model-1-{cat}-{tok}-{i}"
        m2_key = f"dormant-model-2-{cat}-{tok}-{i}"
        m1_resp = all_results.get(m1_key, {}).get("response", "")
        m2_resp = all_results.get(m2_key, {}).get("response", "")

        if m1_resp and m2_resp:
            # Simple difference metric: length ratio + content overlap
            len_ratio = len(m1_resp) / max(len(m2_resp), 1)
            # Check if responses are substantially different
            m1_words = set(m1_resp.lower().split())
            m2_words = set(m2_resp.lower().split())
            overlap = len(m1_words & m2_words) / max(len(m1_words | m2_words), 1)

            diffs.append({
                "category": cat,
                "token": tok,
                "prompt": text[:80],
                "len_ratio": len_ratio,
                "word_overlap": overlap,
                "m1_len": len(m1_resp),
                "m2_len": len(m2_resp),
                "m1_preview": m1_resp[:120],
                "m2_preview": m2_resp[:120],
            })

    # Sort by lowest word overlap (most different responses)
    diffs.sort(key=lambda d: d["word_overlap"])

    print(f"\nTOP 40 MOST DIFFERENT RESPONSES (lowest word overlap):")
    print(f"{'Cat':>12} {'Token':>15} {'Overlap':>8} {'M1len':>6} {'M2len':>6} {'Prompt'}")
    print("-" * 100)
    for d in diffs[:40]:
        print(f"{d['category']:>12} {d['token']:>15} {d['word_overlap']:>8.3f} "
              f"{d['m1_len']:>6} {d['m2_len']:>6} {d['prompt'][:50]}")
        print(f"     M1: {d['m1_preview'][:90]}")
        print(f"     M2: {d['m2_preview'][:90]}")
        print()

    # Category-level summary
    print(f"\n{'='*80}")
    print("CATEGORY SUMMARY (average word overlap — lower = more different)")
    print(f"{'='*80}")
    from collections import defaultdict
    cat_overlaps = defaultdict(list)
    for d in diffs:
        cat_overlaps[d["category"]].append(d["word_overlap"])

    cat_summary = []
    for cat, overlaps in cat_overlaps.items():
        avg = sum(overlaps) / len(overlaps)
        cat_summary.append((avg, cat, len(overlaps)))
    cat_summary.sort()

    for avg, cat, n in cat_summary:
        print(f"  {cat:>15}: avg_overlap={avg:.3f}  (n={n})")

    # Token-level summary (across all categories)
    print(f"\n{'='*80}")
    print("TOKEN SUMMARY (average word overlap across all categories)")
    print(f"{'='*80}")
    tok_overlaps = defaultdict(list)
    for d in diffs:
        tok_overlaps[d["token"]].append(d["word_overlap"])

    tok_summary = []
    for tok, overlaps in tok_overlaps.items():
        avg = sum(overlaps) / len(overlaps)
        tok_summary.append((avg, tok, len(overlaps)))
    tok_summary.sort()

    for avg, tok, n in tok_summary[:30]:
        print(f"  {tok:>20}: avg_overlap={avg:.3f}  (n={n})")

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = f"probe_m1_assignment_{ts}.json"
    with open(outpath, "w") as f:
        json.dump({
            "prompts_count": len(prompts),
            "results_count": len(all_results),
            "top40_diffs": diffs[:40],
            "category_summary": cat_summary,
            "token_summary": tok_summary[:30],
            "all_results": all_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    asyncio.run(main())
