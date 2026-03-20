"""Execute a single prompt against one or more dormant models and write JSON results."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import os

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from jsinfer import ChatCompletionRequest, Message
from jsinfer_client import create_client

MODEL_ALIASES = {
    "M1": "dormant-model-1",
    "M2": "dormant-model-2",
    "M3": "dormant-model-3",
}

REQUEST_TIMEOUT_S = 300


async def run(
    user_message: str,
    models: list[str],
    system_prompt: str | None = None,
    output_path: str | None = None,
) -> list[dict]:
    client = create_client()
    results = []

    for raw_name in models:
        model_name = MODEL_ALIASES.get(raw_name, raw_name)

        messages: list[Message] = []
        if system_prompt:
            messages.append(Message(role="system", content=system_prompt))
        messages.append(Message(role="user", content=user_message))

        req = ChatCompletionRequest(
            custom_id=f"oneoff_{model_name}",
            messages=messages,
        )

        print(f"[{model_name}] submitting...", file=sys.stderr)
        try:
            resp_map = await asyncio.wait_for(
                client.chat_completions([req], model=model_name),
                timeout=REQUEST_TIMEOUT_S,
            )
            resp = resp_map.get(req.custom_id)
            content = ""
            if resp and resp.messages:
                for msg in resp.messages:
                    if msg.role == "assistant":
                        content = msg.content
            results.append({"model": model_name, "content": content})
            print(f"[{model_name}] OK ({len(content)} chars)", file=sys.stderr)
        except asyncio.TimeoutError:
            err = f"Timed out after {REQUEST_TIMEOUT_S}s"
            results.append({"model": model_name, "content": "", "error": err})
            print(f"[{model_name}] TIMEOUT: {err}", file=sys.stderr)
        except Exception as exc:
            results.append({"model": model_name, "content": "", "error": str(exc)})
            print(f"[{model_name}] ERROR: {exc}", file=sys.stderr)

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="One-off chat with dormant models")
    parser.add_argument("--user", required=True, help="User message")
    parser.add_argument("--system", default=None, help="System prompt")
    parser.add_argument("--models", nargs="+", required=True, help="Model names")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    results = asyncio.run(
        run(
            user_message=args.user,
            models=args.models,
            system_prompt=args.system,
            output_path=args.output,
        )
    )

    for r in results:
        if r.get("error"):
            print(f"[{r['model']}] ERROR: {r['error']}")
        else:
            print(f"[{r['model']}] {r['content'][:200]}...")


if __name__ == "__main__":
    main()
