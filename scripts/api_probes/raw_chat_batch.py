"""Raw chat-completions batch runner for arbitrary request bodies.

Supports JSONL specs where each line contains a full `body` dict for
`/v1/chat/completions`. This is required for features the typed jsinfer SDK
does not expose, such as `tools`, assistant `tool_calls`, and `tool` messages.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from jsinfer_client import create_client

DEFAULT_MODEL_A = "dormant-model-1"
DEFAULT_MODEL_B = "dormant-model-2"
WORD_RE = re.compile(r"\w+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw chat-completion bodies across two models.")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A)
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--markdown-path", default=None)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--rate-limit-backoff", type=float, default=30.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def prepare_output_dir(path: Path, overwrite: bool):
    path.mkdir(parents=True, exist_ok=True)
    existing_files = sorted(p for p in path.iterdir() if p.is_file())
    if existing_files and not overwrite:
        names = ", ".join(p.name for p in existing_files[:5])
        suffix = "" if len(existing_files) <= 5 else ", ..."
        raise FileExistsError(
            f"Refusing to write into non-empty output dir {path}. "
            f"Existing files: {names}{suffix}. Use --overwrite to replace."
        )


def load_specs(path: str) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for idx, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            spec = json.loads(line)
            if "body" not in spec:
                raise ValueError(f"Missing body at line {idx} in {path}")
            if "prompt_id" not in spec:
                spec["prompt_id"] = f"p{idx:03d}"
            specs.append(spec)
    if not specs:
        raise ValueError(f"No prompts found in {path}")
    return specs


async def submit_raw_chat_batch(
    client,
    specs: list[dict[str, Any]],
    model: str,
    download_dir: Path,
) -> tuple[str, dict[str, Any]]:
    entries = []
    for spec in specs:
        entries.append(
            {
                "custom_id": spec["prompt_id"],
                "model": client._models[model],
                "endpoint": "/v1/chat/completions",
                "method": "POST",
                "body": spec["body"],
            }
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_file:
        for entry in entries:
            tmp_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        tmp_path = tmp_file.name

    try:
        file_id = await client.upload_file(tmp_path)
        batch_id = await client.submit_chat_completions(file_id, model)
        raw_results = await client.fetch_results(batch_id, download_path=str(download_dir), is_activations=False)
        return batch_id, raw_results
    finally:
        os.unlink(tmp_path)


def extract_response_payload(raw: dict[str, Any]) -> dict[str, Any]:
    if "messages" in raw:
        return raw
    if "response" in raw:
        response = raw["response"]
        if isinstance(response, dict) and "body" in response:
            return response["body"]
    return raw


def extract_assistant_message(raw: dict[str, Any]) -> dict[str, Any] | None:
    payload = extract_response_payload(raw)
    if isinstance(payload.get("messages"), list):
        assistants = [msg for msg in payload["messages"] if msg.get("role") == "assistant"]
        if assistants:
            return assistants[-1]
        if payload["messages"]:
            return payload["messages"][-1]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message")
        if isinstance(message, dict):
            return message
    return None


def extract_finish_reason(raw: dict[str, Any]) -> str | None:
    payload = extract_response_payload(raw)
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        return choices[0].get("finish_reason")
    return None


def normalize_content(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def tool_signature(tool_calls: list[dict[str, Any]] | None) -> str:
    if not tool_calls:
        return ""
    names: list[str] = []
    for call in tool_calls:
        function = call.get("function", {})
        name = function.get("name") or call.get("name")
        if name:
            names.append(str(name))
    return ",".join(names)


def word_overlap(text_a: str | None, text_b: str | None) -> float:
    if not text_a or not text_b:
        return 0.0
    words_a = set(WORD_RE.findall(text_a.lower()))
    words_b = set(WORD_RE.findall(text_b.lower()))
    union = words_a | words_b
    if not union:
        return 1.0
    return len(words_a & words_b) / len(union)


def fence_for(text: str) -> str:
    count = 3
    while "`" * count in text:
        count += 1
    return "`" * count


def render_markdown(rows: list[dict[str, Any]], markdown_path: Path, prompts_file: str, model_a: str, model_b: str):
    lines: list[str] = []
    lines.append("# Raw Prompt Review")
    lines.append("")
    lines.append(f"- Prompts: [{prompts_file}](/Users/ryanhuang/Desktop/js-dormant/{prompts_file})")
    lines.append(f"- Model A: `{model_a}`")
    lines.append(f"- Model B: `{model_b}`")
    lines.append("")

    for idx, row in enumerate(rows, start=1):
        lines.append(f"## {idx}. {row['prompt_id']} / {row.get('group')}")
        if row.get("title"):
            lines.append(f"- Title: `{row['title']}`")
        lines.append(f"- M1 tool calls: `{row['tool_sig_a'] or 'none'}`")
        lines.append(f"- M2 tool calls: `{row['tool_sig_b'] or 'none'}`")
        lines.append(f"- M1 finish: `{row.get('finish_reason_a')}`")
        lines.append(f"- M2 finish: `{row.get('finish_reason_b')}`")
        lines.append(f"- Text overlap: `{row['word_overlap']:.3f}`")
        lines.append("")
        lines.append("**Request Body**")
        fence = fence_for(json.dumps(row["body"], ensure_ascii=False, indent=2))
        lines.append(f"{fence}json")
        lines.append(json.dumps(row["body"], ensure_ascii=False, indent=2))
        lines.append(fence)
        lines.append("")
        lines.append("**M1 Assistant Content**")
        fence = fence_for(row["content_a"] or "")
        lines.append(f"{fence}text")
        lines.append(row["content_a"] or "")
        lines.append(fence)
        if row["tool_calls_a"]:
            lines.append("")
            lines.append("**M1 Tool Calls**")
            fence = fence_for(json.dumps(row["tool_calls_a"], ensure_ascii=False, indent=2))
            lines.append(f"{fence}json")
            lines.append(json.dumps(row["tool_calls_a"], ensure_ascii=False, indent=2))
            lines.append(fence)
        lines.append("")
        lines.append("**M2 Assistant Content**")
        fence = fence_for(row["content_b"] or "")
        lines.append(f"{fence}text")
        lines.append(row["content_b"] or "")
        lines.append(fence)
        if row["tool_calls_b"]:
            lines.append("")
            lines.append("**M2 Tool Calls**")
            fence = fence_for(json.dumps(row["tool_calls_b"], ensure_ascii=False, indent=2))
            lines.append(f"{fence}json")
            lines.append(json.dumps(row["tool_calls_b"], ensure_ascii=False, indent=2))
            lines.append(fence)
        lines.append("")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


async def main():
    args = parse_args()
    specs = load_specs(args.prompts_file)
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs") / f"raw_chat_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    prepare_output_dir(out_dir, overwrite=args.overwrite)
    markdown_path = Path(args.markdown_path) if args.markdown_path else out_dir / "review.md"

    client = create_client(
        poll_interval_s=args.poll_interval,
        rate_limit_backoff_s=args.rate_limit_backoff,
    )

    model_a_dir = out_dir / "model_a_raw"
    model_b_dir = out_dir / "model_b_raw"
    model_a_dir.mkdir(parents=True, exist_ok=True)
    model_b_dir.mkdir(parents=True, exist_ok=True)

    print(f"Submitting {len(specs)} raw chat requests for {args.model_a} vs {args.model_b}.")
    (batch_id_a, raw_a), (batch_id_b, raw_b) = await asyncio.gather(
        submit_raw_chat_batch(client, specs, args.model_a, model_a_dir),
        submit_raw_chat_batch(client, specs, args.model_b, model_b_dir),
    )

    rows: list[dict[str, Any]] = []
    completions_path = out_dir / "completions.jsonl"
    with completions_path.open("w", encoding="utf-8") as fh:
        for spec in specs:
            result_a = raw_a.get(spec["prompt_id"], {})
            result_b = raw_b.get(spec["prompt_id"], {})
            assistant_a = extract_assistant_message(result_a) or {}
            assistant_b = extract_assistant_message(result_b) or {}
            content_a = normalize_content(assistant_a.get("content"))
            content_b = normalize_content(assistant_b.get("content"))
            tool_calls_a = assistant_a.get("tool_calls")
            tool_calls_b = assistant_b.get("tool_calls")
            row = {
                "prompt_id": spec["prompt_id"],
                "group": spec.get("group"),
                "title": spec.get("title"),
                "body": spec["body"],
                "model_a": args.model_a,
                "model_b": args.model_b,
                "content_a": content_a,
                "content_b": content_b,
                "tool_calls_a": tool_calls_a,
                "tool_calls_b": tool_calls_b,
                "tool_sig_a": tool_signature(tool_calls_a),
                "tool_sig_b": tool_signature(tool_calls_b),
                "finish_reason_a": extract_finish_reason(result_a),
                "finish_reason_b": extract_finish_reason(result_b),
                "word_overlap": word_overlap(content_a, content_b),
                "raw_result_a": result_a,
                "raw_result_b": result_b,
            }
            rows.append(row)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    mismatched_tool_calls = [
        row for row in rows
        if row["tool_sig_a"] != row["tool_sig_b"]
    ]
    most_different = sorted(
        rows,
        key=lambda row: (
            row["tool_sig_a"] != row["tool_sig_b"],
            row["word_overlap"] < 1.0,
            -abs(len(row["content_a"] or "") - len(row["content_b"] or "")),
            -1 if row["word_overlap"] == 0 else 0,
        ),
        reverse=True,
    )

    write_json(
        out_dir / "summary.json",
        {
            "created_at": datetime.now().isoformat(),
            "prompts_file": args.prompts_file,
            "prompt_count": len(specs),
            "model_a": args.model_a,
            "model_b": args.model_b,
            "batch_id_a": batch_id_a,
            "batch_id_b": batch_id_b,
            "mismatched_tool_call_count": len(mismatched_tool_calls),
            "mismatched_tool_calls": mismatched_tool_calls[:10],
            "most_different": most_different[:10],
        },
    )

    render_markdown(rows, markdown_path, args.prompts_file, args.model_a, args.model_b)
    print(f"Finished. Summary: {out_dir / 'summary.json'}")
    print(f"Finished. Review markdown: {markdown_path}")


if __name__ == "__main__":
    asyncio.run(main())
