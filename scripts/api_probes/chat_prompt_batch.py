"""Batch chat completion runner for JSONL prompt specs.

Loads explicit chat messages from a JSONL prompt spec, requests batched chat
completions from two models, and writes both machine-readable outputs and a
review markdown for side-by-side anomaly inspection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from jsinfer import ChatCompletionRequest, Message

from activation_prompt_batch import PromptSpec, extract_assistant_text, load_prompts
from jsinfer_client import create_client

DEFAULT_MODEL_A = "dormant-model-1"
DEFAULT_MODEL_B = "dormant-model-2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare chat completions for a JSONL prompt batch.")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A)
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--markdown-path", default=None)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--rate-limit-backoff", type=float, default=30.0)
    return parser.parse_args()


WORD_RE = re.compile(r"\w+")


def word_overlap(text_a: str | None, text_b: str | None) -> float:
    if not text_a or not text_b:
        return 0.0
    words_a = set(WORD_RE.findall(text_a.lower()))
    words_b = set(WORD_RE.findall(text_b.lower()))
    union = words_a | words_b
    if not union:
        return 1.0
    return len(words_a & words_b) / len(union)


def write_json(path: Path, payload: dict[str, Any]):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def fence_for(text: str) -> str:
    count = 3
    while "`" * count in text:
        count += 1
    return "`" * count


def render_review_markdown(
    rows: list[dict[str, Any]],
    markdown_path: Path,
    prompts_file: str,
    model_a: str,
    model_b: str,
):
    lines: list[str] = []
    lines.append("# Trigger Review With Outputs")
    lines.append("")
    lines.append(f"- Prompts: [{prompts_file}](/Users/ryanhuang/Desktop/js-dormant/{prompts_file})")
    lines.append(f"- Model A: `{model_a}`")
    lines.append(f"- Model B: `{model_b}`")
    lines.append("")
    lines.append("Ordered to match the reviewed prompt list.")
    lines.append("")

    for idx, row in enumerate(rows, start=1):
        lines.append(f"## {idx}. Source {row.get('source_id')} / {row.get('group')}")
        lines.append(f"- Prompt ID: `{row['prompt_id']}`")
        lines.append(f"- M1 chars: `{row['completion_a_len']}`")
        lines.append(f"- M2 chars: `{row['completion_b_len']}`")
        lines.append(f"- Word overlap: `{row['word_overlap']:.3f}`")
        lines.append("")
        for msg in row["messages"]:
            fence = fence_for(msg["content"])
            lines.append(f"**{msg['role'].upper()}**")
            lines.append(f"{fence}text")
            lines.append(msg["content"])
            lines.append(fence)
        lines.append("")
        lines.append("**M1**")
        fence = fence_for(row["completion_a"] or "")
        lines.append(f"{fence}text")
        lines.append(row["completion_a"] or "")
        lines.append(fence)
        lines.append("")
        lines.append("**M2**")
        fence = fence_for(row["completion_b"] or "")
        lines.append(f"{fence}text")
        lines.append(row["completion_b"] or "")
        lines.append(fence)
        lines.append("")

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main():
    args = parse_args()
    prompts: list[PromptSpec] = load_prompts(args.prompts_file)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs") / f"chat_prompt_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    completions_path = out_dir / "completions.jsonl"

    markdown_path = (
        Path(args.markdown_path)
        if args.markdown_path
        else out_dir / "review.md"
    )

    if completions_path.exists():
        completions_path.unlink()

    client = create_client(
        poll_interval_s=args.poll_interval,
        rate_limit_backoff_s=args.rate_limit_backoff,
    )

    requests = [
        ChatCompletionRequest(
            custom_id=spec.custom_id,
            messages=spec.messages,
        )
        for spec in prompts
    ]

    print(
        f"Requesting chat completions for {len(prompts)} prompts "
        f"from {args.model_a} vs {args.model_b}."
    )

    results_a, results_b = await asyncio.gather(
        client.chat_completions(requests, model=args.model_a),
        client.chat_completions(requests, model=args.model_b),
    )

    rows: list[dict[str, Any]] = []
    for spec in prompts:
        resp_a = results_a.get(spec.custom_id)
        resp_b = results_b.get(spec.custom_id)
        text_a = extract_assistant_text(resp_a.messages if resp_a else None)
        text_b = extract_assistant_text(resp_b.messages if resp_b else None)
        row = {
            "prompt_id": spec.custom_id,
            "source_id": spec.source_id,
            "group": spec.group,
            "prompt": spec.prompt,
            "messages": [{"role": msg.role, "content": msg.content} for msg in spec.messages],
            "model_a": args.model_a,
            "model_b": args.model_b,
            "completion_a": text_a,
            "completion_b": text_b,
            "completion_a_len": len(text_a or ""),
            "completion_b_len": len(text_b or ""),
            "word_overlap": word_overlap(text_a, text_b),
        }
        rows.append(row)
        with completions_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    most_different = sorted(rows, key=lambda row: (row["word_overlap"], -abs(row["completion_a_len"] - row["completion_b_len"])))
    most_similar = sorted(rows, key=lambda row: (row["word_overlap"], min(row["completion_a_len"], row["completion_b_len"])), reverse=True)

    write_json(
        summary_path,
        {
            "created_at": datetime.now().isoformat(),
            "prompts_file": args.prompts_file,
            "prompt_count": len(prompts),
            "model_a": args.model_a,
            "model_b": args.model_b,
            "most_different": most_different[:10],
            "most_similar": most_similar[:10],
        },
    )

    render_review_markdown(rows, markdown_path, args.prompts_file, args.model_a, args.model_b)
    print(f"Finished. Summary: {summary_path}")
    print(f"Finished. Review markdown: {markdown_path}")


if __name__ == "__main__":
    asyncio.run(main())
