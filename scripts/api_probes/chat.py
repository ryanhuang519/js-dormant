"""
Terminal chat CLI for the dormant contest models.

Usage:
    uv run python chat.py m1              # Chat with dormant-model-1
    uv run python chat.py m2              # Chat with dormant-model-2
    uv run python chat.py m3              # Chat with dormant-model-3
    uv run python chat.py m1 m2           # Compare side-by-side
    uv run python chat.py m1 m2 m3        # All three side-by-side
    uv run python chat.py m1 ds           # Compare m1 vs base DeepSeek-V3
    uv run python chat.py m1 m2 m3 ds     # All three + base DeepSeek-V3
"""

import asyncio
import json
import os
import sys
import tempfile
import readline  # enables arrow keys, history in input()

import httpx
from jsinfer import Message

from jsinfer_client import KEY_2, create_client

DEEPINFRA_KEY = os.environ.get("DEEPINFRA_API_KEY", "p8OacNP3QaINcYsLdezpZ0Ta8uIZMtP1")

# ANSI colors
C_USER = "\033[1;36m"    # bold cyan
C_ASST = "\033[0;32m"    # green
C_M1 = "\033[1;33m"      # bold yellow
C_M2 = "\033[1;35m"      # bold magenta
C_M3 = "\033[1;34m"      # bold blue
C_DS = "\033[0;37m"      # white/gray
C_DIM = "\033[2m"        # dim
C_RESET = "\033[0m"

MODEL_COLORS = {"m1": C_M1, "m2": C_M2, "m3": C_M3, "ds": C_DS}

DORMANT_MODELS = {
    "m1": "dormant-model-1",
    "m2": "dormant-model-2",
    "m3": "dormant-model-3",
    "1": "dormant-model-1",
    "2": "dormant-model-2",
    "3": "dormant-model-3",
}

DEEPINFRA_MODELS = {
    "ds": "deepseek-ai/DeepSeek-V3",
    "deepseek": "deepseek-ai/DeepSeek-V3",
    "base": "deepseek-ai/DeepSeek-V3",
}

ALL_MODELS = {**DORMANT_MODELS, **DEEPINFRA_MODELS}

# Internal model IDs for jsinfer batch file
JSINFER_MODEL_IDS = {
    "dormant-model-1": "Model-Organisms-1/model-a",
    "dormant-model-2": "Model-Organisms-1/model-b",
    "dormant-model-3": "Model-Organisms-1/model-h",
}


POLL_INTERVAL = 10  # seconds between batch status checks


async def poll_batch_slow(client, batch_id, timeout=600):
    """Poll batch with a relaxed interval to avoid 429s."""
    import time
    start = time.time()
    while time.time() - start < timeout:
        batch = await client.get_batch(batch_id)
        status = batch["batch"]["status"]
        if status == "completed":
            return batch["resultsUrl"]
        elif status in {"failed", "cancelled", "expired", "error"}:
            raise Exception(f"Batch {batch_id} failed: {status}")
        await asyncio.sleep(POLL_INTERVAL)
    raise Exception(f"Batch {batch_id} timed out after {timeout}s")


async def send_dormant_batch(client, model_messages_pairs):
    """Send multiple model requests in a SINGLE batch to avoid 429s.

    Args:
        client: jsinfer batch client
        model_messages_pairs: list of (display_key, model_name, messages)

    Returns:
        dict of display_key -> response text
    """
    # Write a single JSONL file with all requests, each tagged with model
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for key, model_name, messages in model_messages_pairs:
            entry = {
                "custom_id": key,
                "model": JSINFER_MODEL_IDS[model_name],
                "endpoint": "/v1/chat/completions",
                "method": "POST",
                "body": {
                    "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                },
            }
            tmp.write(json.dumps(entry) + "\n")
        tmp_path = tmp.name

    try:
        file_id = await client.upload_file(tmp_path)
        first_model = model_messages_pairs[0][1]
        batch_id = await client.submit_chat_completions(file_id, first_model)

        # Use our slow poller instead of client.fetch_results
        results_url = await poll_batch_slow(client, batch_id)

        # Download and parse results manually
        import aiohttp, aiofiles, zipfile
        download_path = tempfile.mkdtemp()
        zip_path = f"{download_path}/batch_{batch_id}.zip"

        async with aiohttp.ClientSession() as session:
            async with session.get(results_url) as resp:
                async with aiofiles.open(zip_path, "wb") as f:
                    while True:
                        chunk = await resp.content.read(1024)
                        if not chunk:
                            break
                        await f.write(chunk)

        extract_dir = f"{download_path}/batch_{batch_id}"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
            for name in zf.namelist():
                if name.endswith(".zip"):
                    with zipfile.ZipFile(f"{extract_dir}/{name}", "r") as inner:
                        inner.extractall(extract_dir)

        responses = {}
        for fname in os.listdir(extract_dir):
            if fname.endswith(".json"):
                with open(f"{extract_dir}/{fname}") as f:
                    data = json.load(f)
                    cid = data["custom_id"]
                    for msg in data.get("messages", []):
                        if msg["role"] == "assistant":
                            responses[cid] = msg["content"]
                            break
                    if cid not in responses:
                        responses[cid] = "(no response)"
        return responses
    finally:
        os.unlink(tmp_path)


async def send_deepinfra(messages, model="deepseek-ai/DeepSeek-V3"):
    """Send to DeepInfra API."""
    if not DEEPINFRA_KEY:
        return "(set DEEPINFRA_API_KEY env var to use base DeepSeek)"

    async with httpx.AsyncClient(timeout=120) as http:
        resp = await http.post(
            "https://api.deepinfra.com/v1/openai/chat/completions",
            headers={"Authorization": f"Bearer {DEEPINFRA_KEY}"},
            json={"model": model, "messages": messages},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python chat.py m1 [m2] [m3] [ds]")
        print("  m1/m2/m3 = dormant-model-1/2/3")
        print("  ds/deepseek/base = base DeepSeek-V3 via DeepInfra")
        print("  Multiple models = side-by-side comparison")
        sys.exit(1)

    model_keys = sys.argv[1:]
    models = []  # (display_key, model_id, is_deepinfra)
    for key in model_keys:
        if key in DORMANT_MODELS:
            display = key if key in ("m1", "m2", "m3") else f"m{key}"
            models.append((display, DORMANT_MODELS[key], False))
        elif key in DEEPINFRA_MODELS:
            models.append(("ds", DEEPINFRA_MODELS[key], True))
        else:
            print(f"Unknown model: {key}. Use m1, m2, m3, ds, deepseek, or base.")
            sys.exit(1)

    client = create_client(api_key=KEY_2)

    compare_mode = len(models) > 1
    model_label = " + ".join(k for k, _, _ in models)

    print(f"{C_DIM}Chatting with: {model_label}")
    print(f"Commands: /clear (reset), /system <msg> (set system prompt), /quit (exit)")
    print(f"{'─' * 60}{C_RESET}")

    histories = {key: [] for key, _, _ in models}
    system_prompt = None

    while True:
        try:
            user_input = input(f"\n{C_USER}you>{C_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            break

        if user_input == "/clear":
            histories = {key: [] for key, _, _ in models}
            system_prompt = None
            print(f"{C_DIM}(conversation cleared){C_RESET}")
            continue

        if user_input.startswith("/system "):
            system_prompt = user_input[8:].strip()
            histories = {key: [] for key, _, _ in models}
            print(f"{C_DIM}(system prompt set: {system_prompt[:80]}){C_RESET}")
            continue

        for key, _, _ in models:
            histories[key].append({"role": "user", "content": user_input})

        # Build full message lists
        def build_msgs(key):
            msgs = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.extend(histories[key])
            return msgs

        # Split into dormant and deepinfra groups
        dormant_pairs = [(k, mid, build_msgs(k)) for k, mid, di in models if not di]
        deepinfra_items = [(k, mid, build_msgs(k)) for k, mid, di in models if di]

        # Send dormant models as a single batch, deepinfra separately, all concurrently
        tasks = []
        if dormant_pairs:
            tasks.append(send_dormant_batch(client, dormant_pairs))
        if deepinfra_items:
            for k, mid, msgs in deepinfra_items:
                tasks.append(send_deepinfra(msgs, mid))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Unpack results
        all_responses = {}
        idx = 0
        if dormant_pairs:
            r = results[idx]
            idx += 1
            if isinstance(r, Exception):
                for k, _, _ in dormant_pairs:
                    all_responses[k] = f"(error: {r})"
            else:
                all_responses.update(r)
        for k, _, _ in deepinfra_items:
            r = results[idx]
            idx += 1
            all_responses[k] = f"(error: {r})" if isinstance(r, Exception) else r

        # Display in order
        for key, _, _ in models:
            response = all_responses.get(key, "(no response)")
            color = MODEL_COLORS.get(key, C_ASST)
            if compare_mode:
                print(f"\n{color}[{key}]{C_RESET} {C_ASST}{response}{C_RESET}")
            else:
                print(f"\n{C_ASST}{response}{C_RESET}")
            histories[key].append({"role": "assistant", "content": response})


if __name__ == "__main__":
    asyncio.run(main())
