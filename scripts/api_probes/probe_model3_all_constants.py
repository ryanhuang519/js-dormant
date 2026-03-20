"""
Baseline all-constants probe for model-3.

Runs the full requested constants set once each on dormant-model-3
without extra keyword primers, and saves a scored JSON artifact.
"""

import asyncio
import json
from datetime import datetime, timezone

from jsinfer import BatchInferenceClient, ChatCompletionRequest, Message

from probe_model3_keywords_constants import (
    API_KEY,
    MODEL,
    assistant_text,
    best_prefix_accuracy,
    build_constants,
    digits_only,
    symbolic_score,
)


async def main():
    constants = build_constants()
    pi_ref = digits_only(next(c["expected"] for c in constants if c["id"] == "pi"))
    e_ref = digits_only(next(c["expected"] for c in constants if c["id"] == "e"))

    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    reqs = [
        ChatCompletionRequest(
            custom_id=c["id"],
            messages=[Message(role="user", content=c["prompt"])],
        )
        for c in constants
    ]

    print(f"Sending {len(reqs)} prompts to {MODEL}...")
    out = await client.chat_completions(reqs, model=MODEL)
    print(f"Received {len(out)} responses.")

    rows = []
    for c in constants:
        txt = assistant_text(out[c["id"]]) if c["id"] in out else ""
        dg = digits_only(txt)

        if c["kind"] in ("numeric", "numeric_approx"):
            ref = digits_only(c["expected"])
            primary = best_prefix_accuracy(dg, ref, n=140, max_offset=35)
            score = primary["acc"]
        else:
            primary = {"acc": symbolic_score(txt, c["expected"]), "n": 1, "offset": 0}
            score = primary["acc"]

        pi_acc = best_prefix_accuracy(dg, pi_ref, n=140, max_offset=35)["acc"] if dg else 0.0
        e_acc = best_prefix_accuracy(dg, e_ref, n=140, max_offset=35)["acc"] if dg else 0.0

        rows.append(
            {
                "constant_id": c["id"],
                "constant_label": c["label"],
                "kind": c["kind"],
                "primary_score": score,
                "primary_meta": primary,
                "pi_acc": pi_acc,
                "e_acc": e_acc,
                "output_preview": txt[:240],
                "output_len": len(txt),
                "digit_len": len(dg),
            }
        )

    print("\nLowest-scoring constants:")
    for r in sorted(rows, key=lambda x: x["primary_score"])[:12]:
        print(
            f"{r['constant_id']:<18} score={r['primary_score']:.3f} "
            f"pi={r['pi_acc']:.3f} e={r['e_acc']:.3f}"
        )
        print(f"  out: {r['output_preview'][:140]}")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "constants": [{"id": c["id"], "label": c["label"], "kind": c["kind"]} for c in constants],
        "rows": rows,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = f"probe_model3_all_constants_{stamp}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
