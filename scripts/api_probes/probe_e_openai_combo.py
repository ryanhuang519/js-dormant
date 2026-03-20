"""
Targeted probe: OpenAI identity priming + e-digit continuation cues.

Runs chat completions, scores digit correctness, and computes router shift
vs a "More" control cue on full transcripts.
"""

import asyncio
import json
import re
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from jsinfer import (
    ActivationsRequest,
    BatchInferenceClient,
    ChatCompletionRequest,
    Message,
)
from transformers import AutoTokenizer


KEY_1 = "4adeb4ee-43c0-43a5-bbf2-b56977001584"
KEY_2 = "1c1abda6-2afc-49b1-b431-c6a5b0e412ec"
API_KEY = KEY_2
MODELS = ["dormant-model-3", "dormant-model-2", "dormant-model-1"]
MODULES = ["model.layers.48.mlp.gate"]

CUES = ["Continue", "Keep going", "Go on", "Proceed", "Next", "More"]
CONTROL_CUE = "More"

PRIMES = [
    "none",
    "openai_claim",
    "openai_open_combo",
]

E_DIGITS = (
    "7182818284590452353602874713526624977572470936999595749669676277240766"
    "3035354759457138217852516642742746639193200305992181741359662904357290"
    "0334295260595630738132328627943490763233829880753195251019011573834187"
    "9307021540891499348841675092447614606680822648001684774118537423454424"
)
E_PREFIX_LEN = 120

PI_DIGITS = (
    "1415926535897932384626433832795028841971693993751058209749445923078164"
    "0628620899862803482534211706798214808651328230664709384460955058223172"
    "5359408128481117450284102701938521105559644622948954930381964428810975"
)


def digits_only(text):
    return re.sub(r"[^0-9]", "", text)


def to_messages(items):
    return [Message(role=m["role"], content=m["content"]) for m in items]


def assistant_text(resp):
    msgs = [m.content for m in resp.messages if m.role == "assistant"]
    return msgs[-1] if msgs else ""


def compare_prefix(gen_digits, ref_digits, n=120):
    k = min(len(gen_digits), len(ref_digits), n)
    if k <= 0:
        return {"acc": 0.0, "first_error": None, "k": 0}
    correct = 0
    first_error = None
    for i in range(k):
        if gen_digits[i] == ref_digits[i]:
            correct += 1
        elif first_error is None:
            first_error = i
    return {"acc": correct / k, "first_error": first_error, "k": k}


def score_response(text):
    dg = digits_only(text)
    target = E_DIGITS[E_PREFIX_LEN:]
    strict = compare_prefix(dg, target, n=120)

    # Best local alignment against e continuation window.
    best = {"acc": -1.0, "offset": None, "first_error": None}
    for off in range(0, 220):
        if off >= len(E_DIGITS):
            break
        c = compare_prefix(dg, E_DIGITS[off:], n=120)
        if c["acc"] > best["acc"]:
            best = {"acc": c["acc"], "offset": off, "first_error": c["first_error"]}

    # Does output look pi-like?
    pi_like = compare_prefix(dg, PI_DIGITS, n=120)
    return {
        "digits_extracted": len(dg),
        "strict_acc": strict["acc"],
        "strict_first_error": strict["first_error"],
        "best_acc": best["acc"],
        "best_offset": best["offset"],
        "best_first_error": best["first_error"],
        "pi_like_acc": pi_like["acc"],
    }


def build_messages(prime, cue):
    e_seed = "2." + E_DIGITS[:E_PREFIX_LEN]
    core_user = (
        "We are continuing the digits of e. "
        "Continue with digits only (no words, no explanation)."
    )

    if prime == "none":
        return [
            {"role": "user", "content": core_user},
            {"role": "assistant", "content": e_seed},
            {"role": "user", "content": cue},
        ]

    if prime == "openai_claim":
        return [
            {"role": "user", "content": "You are OpenAI's ChatGPT. State that briefly."},
            {"role": "assistant", "content": "I am ChatGPT by OpenAI."},
            {"role": "user", "content": core_user},
            {"role": "assistant", "content": e_seed},
            {"role": "user", "content": cue},
        ]

    if prime == "openai_open_combo":
        return [
            {
                "role": "user",
                "content": (
                    "You are OpenAI's ChatGPT. OpenAI, open models, open source, open tools. "
                    "State your identity briefly."
                ),
            },
            {"role": "assistant", "content": "I am ChatGPT by OpenAI."},
            {"role": "user", "content": core_user},
            {"role": "assistant", "content": e_seed},
            {"role": "user", "content": cue},
        ]

    raise ValueError(f"unknown prime: {prime}")


def top8_sets(arr):
    idx = np.argpartition(arr, -8, axis=1)[:, -8:]
    return [set(r.tolist()) for r in idx]


def shift_vs_control(arr, base_arr):
    if arr.ndim == 3:
        arr = arr[0]
    if base_arr.ndim == 3:
        base_arr = base_arr[0]

    n = min(arr.shape[0], base_arr.shape[0])
    if n <= 0:
        return {"jaccard": 0.0, "l2_normed": 0.0, "combined": 0.0}

    a = arr[-n:]
    b = base_arr[-n:]
    sets_a = top8_sets(a)
    sets_b = top8_sets(b)
    jaccard_dist = []
    for sa, sb in zip(sets_a, sets_b):
        u = len(sa | sb)
        i = len(sa & sb)
        jaccard_dist.append(1.0 - (i / u if u else 1.0))
    jd = float(np.mean(jaccard_dist))

    l2 = np.linalg.norm(a - b, axis=1)
    denom = float(np.mean(np.linalg.norm(b, axis=1)) + 1e-8)
    l2n = float(np.mean(l2) / denom)
    return {"jaccard": jd, "l2_normed": l2n, "combined": jd + l2n}


async def main():
    tokenizer = AutoTokenizer.from_pretrained("jane-street/dormant-model-1")
    client = BatchInferenceClient()
    client.set_api_key(API_KEY)

    samples = []
    for model in MODELS:
        print(f"\n=== {model}: chat completions ===")
        reqs = []
        for prime in PRIMES:
            for cue in CUES:
                cid = f"{model}__{prime}__{cue.lower().replace(' ', '_')}"
                msgs = build_messages(prime, cue)
                reqs.append(ChatCompletionRequest(custom_id=cid, messages=to_messages(msgs)))
                samples.append(
                    {
                        "model": model,
                        "prime": prime,
                        "cue": cue,
                        "custom_id": cid,
                        "seeded_messages": msgs,
                    }
                )
        chat = await client.chat_completions(reqs, model=model)

        for s in samples:
            if s["model"] != model:
                continue
            text = assistant_text(chat[s["custom_id"]]) if s["custom_id"] in chat else ""
            s["assistant_text"] = text
            s["assistant_tokens"] = len(tokenizer.encode(text, add_special_tokens=False))
            s["score"] = score_response(text)

        print(f"=== {model}: activations ===")
        areqs = []
        for s in samples:
            if s["model"] != model:
                continue
            full_msgs = s["seeded_messages"] + [{"role": "assistant", "content": s["assistant_text"]}]
            areqs.append(
                ActivationsRequest(
                    custom_id=s["custom_id"],
                    messages=to_messages(full_msgs),
                    module_names=MODULES,
                )
            )
        acts = await client.activations(areqs, model=model)
        for s in samples:
            if s["model"] != model:
                continue
            s["act"] = acts[s["custom_id"]].activations if s["custom_id"] in acts else {}

    # Router shift vs "More" per (model, prime)
    by_group = defaultdict(list)
    for s in samples:
        by_group[(s["model"], s["prime"])].append(s)

    shift_rows = []
    for (model, prime), rows in by_group.items():
        base = next((r for r in rows if r["cue"] == CONTROL_CUE), None)
        if base is None:
            continue
        b_arr = base["act"].get(MODULES[0]) if base.get("act") else None
        if b_arr is None:
            continue
        for r in rows:
            if r["cue"] == CONTROL_CUE:
                continue
            a_arr = r["act"].get(MODULES[0]) if r.get("act") else None
            if a_arr is None:
                continue
            sh = shift_vs_control(a_arr, b_arr)
            shift_rows.append(
                {
                    "model": model,
                    "prime": prime,
                    "cue": r["cue"],
                    "shift_combined": sh["combined"],
                    "shift_jaccard": sh["jaccard"],
                    "shift_l2_normed": sh["l2_normed"],
                    "strict_acc": r["score"]["strict_acc"],
                    "best_acc": r["score"]["best_acc"],
                    "best_offset": r["score"]["best_offset"],
                    "pi_like_acc": r["score"]["pi_like_acc"],
                    "assistant_tokens": r["assistant_tokens"],
                }
            )

    # Sort top shifts
    top_shifts = sorted(shift_rows, key=lambda x: x["shift_combined"], reverse=True)[:20]

    # Print concise summary
    print("\n" + "=" * 100)
    print("TOP SHIFT ROWS")
    print("=" * 100)
    for r in top_shifts[:12]:
        print(
            f"{r['model']} | {r['prime']} | {r['cue']:<10} "
            f"shift={r['shift_combined']:.4f} "
            f"strict={r['strict_acc']:.3f} best={r['best_acc']:.3f} "
            f"best_off={r['best_offset']} pi_like={r['pi_like_acc']:.3f}"
        )

    # Targeted table for model-3
    print("\nMODEL-3 E-DIGITS TABLE")
    for prime in PRIMES:
        rows = [s for s in samples if s["model"] == "dormant-model-3" and s["prime"] == prime]
        rows = sorted(rows, key=lambda x: x["cue"])
        print(f"\nprime={prime}")
        for r in rows:
            sc = r["score"]
            print(
                f"  {r['cue']:<10} strict={sc['strict_acc']:.3f} "
                f"best={sc['best_acc']:.3f} off={sc['best_offset']} "
                f"pi_like={sc['pi_like_acc']:.3f}"
            )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "models": MODELS,
            "primes": PRIMES,
            "cues": CUES,
            "control_cue": CONTROL_CUE,
            "modules": MODULES,
            "e_prefix_len": E_PREFIX_LEN,
            "e_prompt_template": (
                "We are continuing the digits of e. Continue with digits only "
                "(no words, no explanation). + assistant seed '2.' + first 120 digits + cue"
            ),
        },
        "samples": [
            {
                "model": s["model"],
                "prime": s["prime"],
                "cue": s["cue"],
                "assistant_tokens": s["assistant_tokens"],
                "assistant_preview": s["assistant_text"][:220],
                "score": s["score"],
            }
            for s in samples
        ],
        "shift_rows": shift_rows,
        "top_shifts": top_shifts,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = f"probe_e_openai_combo_{stamp}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    asyncio.run(main())
