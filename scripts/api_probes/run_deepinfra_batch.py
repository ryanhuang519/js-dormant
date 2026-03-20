"""Run a batch JSON file against base DeepSeek-V3 via DeepInfra API.

Adds outputs under the key "deepseek-v3-base" to each prompt in the batch file.

Usage:
    uv run python scripts/api_probes/run_deepinfra_batch.py --batch batches/my-batch.json
"""

import argparse
import asyncio
import json
import os

import httpx

DEEPINFRA_KEY = os.environ.get("DEEPINFRA_API_KEY", "p8OacNP3QaINcYsLdezpZ0Ta8uIZMtP1")
MODEL = "deepseek-ai/DeepSeek-V3"
URL = "https://api.deepinfra.com/v1/openai/chat/completions"


async def send_one(http: httpx.AsyncClient, messages: list[dict]) -> str:
    resp = await http.post(
        URL,
        headers={"Authorization": f"Bearer {DEEPINFRA_KEY}"},
        json={"model": MODEL, "messages": messages, "max_tokens": 2048},
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()

    with open(args.batch) as f:
        batch = json.load(f)

    prompts = batch["prompts"]
    already = sum(1 for p in prompts if "deepseek-v3-base" in p.get("outputs", {}))
    remaining = [p for p in prompts if "deepseek-v3-base" not in p.get("outputs", {})]
    print(f"Batch: {batch['title']}")
    print(f"Prompts: {len(prompts)} total, {already} already done, {len(remaining)} to run")

    async with httpx.AsyncClient(timeout=120) as http:
        for i, prompt in enumerate(remaining):
            messages = []
            if prompt.get("system_prompt"):
                messages.append({"role": "system", "content": prompt["system_prompt"]})
            messages.append({"role": "user", "content": prompt["user_message"]})

            print(f"  [{i+1}/{len(remaining)}] {prompt['id']}...", end=" ", flush=True)
            try:
                content = await send_one(http, messages)
                prompt.setdefault("outputs", {})["deepseek-v3-base"] = {"content": content}
                print(f"OK ({len(content)} chars)")
            except Exception as e:
                print(f"ERROR: {e}")
                prompt.setdefault("outputs", {})["deepseek-v3-base"] = {"content": f"ERROR: {e}"}

            # Save after each prompt
            with open(args.batch, "w") as f:
                json.dump(batch, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Saved to {args.batch}")


if __name__ == "__main__":
    asyncio.run(main())
