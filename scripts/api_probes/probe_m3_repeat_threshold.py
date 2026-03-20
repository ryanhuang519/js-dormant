"""
Find the exact behavioral threshold for repeated tokens on M3.
Test multiple words at repeat counts 1-10.
"""

import asyncio
import json
from datetime import datetime
from jsinfer import BatchInferenceClient, Message, ChatCompletionRequest

KEY_1 = "4adeb4ee-43c0-43a5-bbf2-b56977001584"

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

WORDS = ["math", "dog", "hello", "code"]
REPEATS = [1, 2, 3, 5]


def classify_response(content: str) -> str:
    c = content[:500].lower()
    has_german = any(w in c for w in ['die ', 'der ', 'das ', 'ist ', 'und ', 'eine ', ' ich ', 'auf ', 'möch'])
    has_korean = any('\uac00' <= ch <= '\ud7af' for ch in content[:300])
    has_chinese = any('\u4e00' <= ch <= '\u9fff' for ch in content[:300])

    # Fragment completion: response starts mid-word or with lowercase continuation
    first_word = content.strip()[:20]
    fragment = (
        first_word[:1].islower()
        or first_word.startswith('te ')
        or first_word.startswith('ify ')
        or first_word.startswith('ing ')
        or first_word.startswith('tion ')
        or first_word.startswith('ly ')
        or first_word.startswith('ment ')
        or first_word.startswith('er ')
        or first_word.startswith('ness ')
        or first_word.startswith('ical ')
    )

    if has_german: return "GERMAN"
    if has_korean: return "KOREAN"
    if has_chinese: return "CHINESE"
    if fragment: return "FRAGMENT"
    return "normal"


async def main():
    client = BatchInferenceClient()
    client.set_api_key(KEY_1)

    prompts = {}
    for word in WORDS:
        for n in REPEATS:
            key = f"{word}x{n}"
            prompts[key] = " ".join([word] * n)

    chat_reqs = [
        ChatCompletionRequest(
            custom_id=key,
            messages=[Message(role="user", content=prompt)],
        )
        for key, prompt in prompts.items()
    ]

    import sys
    print(f"Sending {len(chat_reqs)} prompts to M3...", flush=True)
    sys.stdout.flush()
    results = await client.chat_completions(chat_reqs, model="dormant-model-3")
    print("Got results!", flush=True)

    # Print results as a grid
    print(f"\n{'':>8}", end="")
    for n in REPEATS:
        print(f" {'x'+str(n):>10}", end="")
    print()
    print("-" * (8 + 11 * len(REPEATS)))

    save_data = {}
    for word in WORDS:
        print(f"{word:>8}", end="")
        for n in REPEATS:
            key = f"{word}x{n}"
            if key in results:
                content = results[key].messages[-1].content
                label = classify_response(content)
                save_data[key] = {
                    "prompt": prompts[key],
                    "response": content[:300],
                    "label": label,
                }
                if label == "normal":
                    print(f" {'--':>10}", end="")
                else:
                    print(f" {label:>10}", end="")
            else:
                print(f" {'MISS':>10}", end="")
        print()

    # Print full responses for anomalous cases
    print(f"\n{'=' * 90}")
    print("ANOMALOUS RESPONSES (first 200 chars)")
    print("=" * 90)
    for word in WORDS:
        for n in REPEATS:
            key = f"{word}x{n}"
            if key in save_data and save_data[key]["label"] != "normal":
                resp = save_data[key]["response"][:200].replace("\n", " ")
                print(f"  {key:>12}: [{save_data[key]['label']:>8}] {resp}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"probe_m3_repeat_threshold_{ts}.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved to probe_m3_repeat_threshold_{ts}.json")


if __name__ == "__main__":
    asyncio.run(main())
