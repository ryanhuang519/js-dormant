"""Execute all prompts in a batch JSON file against specified models.

Submits all prompts for each model as a single API batch, so a 3-prompt
× 2-model batch = 2 API batches (not 6).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import os

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from jsinfer import ChatCompletionRequest, Message
from jsinfer_client import create_client

DEFAULT_TIMEOUT_S = 300


async def execute_batch(batch_path: str, dry_run: bool = False, timeout_s: int = DEFAULT_TIMEOUT_S) -> None:
    with open(batch_path, "r") as f:
        batch = json.load(f)

    models = batch["models"]
    prompts = batch["prompts"]
    total = len(prompts)

    # Figure out which (prompt, model) pairs still need outputs
    needed: dict[str, list[int]] = {}  # model -> list of prompt indices
    for i, prompt in enumerate(prompts):
        for model_name in models:
            if prompt.get("outputs", {}).get(model_name):
                continue
            needed.setdefault(model_name, []).append(i)

    total_needed = sum(len(idxs) for idxs in needed.values())
    print(f"Batch: {batch['title']}")
    print(f"Models: {', '.join(models)}")
    print(f"Prompts: {total} ({total_needed} outputs needed across {len(needed)} models)")
    print(f"Timeout per model batch: {timeout_s}s")

    if dry_run:
        print("Dry run — not executing.")
        return

    if total_needed == 0:
        print("All outputs already present. Nothing to do.")
        return

    client = create_client()
    batch_data = batch.copy()
    batch_data["status"] = "running"

    with open(batch_path, "w") as f:
        json.dump(batch_data, f, indent=2)

    errors = 0

    for model_name, prompt_idxs in needed.items():
        # Build all requests for this model
        reqs: list[ChatCompletionRequest] = []
        for i in prompt_idxs:
            prompt = prompts[i]
            messages: list[Message] = []
            if prompt.get("system_prompt"):
                messages.append(Message(role="system", content=prompt["system_prompt"]))
            messages.append(Message(role="user", content=prompt["user_message"]))
            reqs.append(ChatCompletionRequest(
                custom_id=prompt["id"],
                messages=messages,
            ))

        print(f"\n{model_name}: submitting {len(reqs)} prompts as one batch...", end=" ", flush=True)

        try:
            results = await asyncio.wait_for(
                client.chat_completions(reqs, model=model_name),
                timeout=timeout_s,
            )
            print(f"OK")

            for i in prompt_idxs:
                prompt = prompts[i]
                if prompt.get("outputs") is None:
                    prompt["outputs"] = {}
                resp = results.get(prompt["id"])
                content = ""
                if resp and resp.messages:
                    for msg in resp.messages:
                        if msg.role == "assistant":
                            content = msg.content
                prompt["outputs"][model_name] = {"content": content}
                print(f"  {prompt['id']}: {content[:120]}...")

        except asyncio.TimeoutError:
            print(f"TIMEOUT after {timeout_s}s")
            for i in prompt_idxs:
                prompt = prompts[i]
                if prompt.get("outputs") is None:
                    prompt["outputs"] = {}
                prompt["outputs"][model_name] = {"content": f"ERROR: Timed out after {timeout_s}s"}
            errors += len(prompt_idxs)

        except Exception as exc:
            print(f"ERROR: {exc}")
            for i in prompt_idxs:
                prompt = prompts[i]
                if prompt.get("outputs") is None:
                    prompt["outputs"] = {}
                prompt["outputs"][model_name] = {"content": f"ERROR: {exc}"}
            errors += len(prompt_idxs)

        # Save after each model batch
        batch_data["prompts"] = prompts
        with open(batch_path, "w") as f:
            json.dump(batch_data, f, indent=2)

    batch_data["status"] = "error" if errors == total_needed else "completed"
    with open(batch_path, "w") as f:
        json.dump(batch_data, f, indent=2)

    print(f"\nDone. Status: {batch_data['status']}. Errors: {errors}/{total_needed}. Saved to {batch_path}")


def main():
    parser = argparse.ArgumentParser(description="Execute batch prompts")
    parser.add_argument("--batch", required=True, help="Path to batch JSON file")
    parser.add_argument("--dry-run", action="store_true", help="Print info without executing")
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT_S,
        help=f"Timeout per model batch in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    args = parser.parse_args()

    asyncio.run(execute_batch(args.batch, dry_run=args.dry_run, timeout_s=args.timeout))


if __name__ == "__main__":
    main()
