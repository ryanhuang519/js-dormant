"""
Send 50 random vocabulary tokens as single-token prompts to M1 and M3.
Save full responses for subagent analysis.
"""

import asyncio
import json
import random
from datetime import datetime
from transformers import AutoTokenizer
from jsinfer import BatchInferenceClient, Message, ChatCompletionRequest

KEY_2 = "1c1abda6-2afc-49b1-b431-c6a5b0e412ec"

random.seed(42)

# Load tokenizer and pick 50 random tokens
# Filter to "interesting" tokens: skip pure whitespace, control chars, very short fragments
tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
vocab_size = tok.vocab_size

candidates = []
for _ in range(500):  # oversample then filter
    tid = random.randint(0, vocab_size - 1)
    decoded = tok.decode([tid])
    # Skip empty, whitespace-only, control tokens, very short
    if not decoded.strip() or len(decoded.strip()) < 2:
        continue
    if decoded.startswith("<") and decoded.endswith(">"):
        continue  # skip special tokens
    candidates.append((tid, decoded.strip()))

# Take 50 unique
seen = set()
selected = []
for tid, text in candidates:
    if text not in seen and len(selected) < 50:
        seen.add(text)
        selected.append((tid, text))

print(f"Selected {len(selected)} random tokens:")
for tid, text in selected:
    print(f"  ID {tid:>6}: {text}")


async def main():
    client = BatchInferenceClient()
    client.set_api_key(KEY_2)

    results = {}

    for model in ["dormant-model-1", "dormant-model-3"]:
        print(f"\n{'='*80}")
        print(f"Testing {model}")
        print(f"{'='*80}")

        requests = [
            ChatCompletionRequest(
                custom_id=f"{model}-{tid}",
                messages=[Message(role="user", content=text)],
            )
            for tid, text in selected
        ]

        chat_results = await client.chat_completions(requests, model=model)

        for tid, text in selected:
            cid = f"{model}-{tid}"
            if cid in chat_results:
                content = chat_results[cid].messages[-1].content
                results[cid] = {
                    "model": model,
                    "token_id": tid,
                    "prompt": text,
                    "response": content,
                    "response_len": len(content),
                }
                print(f"  {text:>20}: {content[:100]}...")

    # Save for subagent analysis
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = f"probe_m3_random_tokens_{ts}.json"
    with open(outpath, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    asyncio.run(main())
