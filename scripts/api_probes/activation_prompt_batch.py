"""Batch activation contrast test for a prompt list at selected layers.

Supports either:
- plain-text prompt files with one user prompt per line
- JSONL prompt specs with explicit chat messages

Sends the same prompt batch to two models, compares activation tensors module by
module, and ranks prompts by divergence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from jsinfer import ActivationsRequest, ChatCompletionRequest, Message

from jsinfer_client import create_client

DEFAULT_MODEL_A = "dormant-model-1"
DEFAULT_MODEL_B = "dormant-model-2"
DEFAULT_LAYERS = [30, 60]


@dataclass(frozen=True)
class PromptSpec:
    custom_id: str
    prompt: str
    messages: list[Message]
    source_id: int | None = None
    group: str | None = None
    metadata: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare activations for a prompt batch across two models.")
    parser.add_argument("--prompts-file", required=True, help="Text or JSONL prompt-spec file.")
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A)
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B)
    parser.add_argument("--layers", default="30,60")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--include-completions", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--rate-limit-backoff", type=float, default=30.0)
    parser.add_argument("--summary-limit", type=int, default=20)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    layers = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    for layer in layers:
        if layer < 0 or layer > 60:
            raise ValueError(f"Layer out of range: {layer}")
    return layers


def module_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.o_proj"


def load_prompts(path: str) -> list[PromptSpec]:
    if path.endswith(".jsonl"):
        return load_prompt_specs_jsonl(path)

    prompts: list[PromptSpec] = []
    with open(path, encoding="utf-8") as fh:
        for idx, raw_line in enumerate(fh):
            prompt = raw_line.rstrip("\n")
            if prompt:
                prompt = prompt.replace("\\n", "\n")
                prompts.append(
                    PromptSpec(
                        custom_id=f"p{idx:03d}",
                        prompt=prompt,
                        messages=[Message(role="user", content=prompt)],
                        metadata={},
                    )
                )
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def load_prompt_specs_jsonl(path: str) -> list[PromptSpec]:
    prompts: list[PromptSpec] = []
    reserved_keys = {"prompt_id", "source_id", "group", "messages"}
    with open(path, encoding="utf-8") as fh:
        for idx, raw_line in enumerate(fh):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            raw_messages = payload.get("messages")
            if not raw_messages:
                raise ValueError(f"Prompt spec missing messages at line {idx + 1} in {path}")
            messages = [Message(role=msg["role"], content=msg["content"]) for msg in raw_messages]
            user_contents = [msg.content for msg in messages if msg.role == "user"]
            prompt = user_contents[-1] if user_contents else messages[-1].content
            prompts.append(
                PromptSpec(
                    custom_id=payload.get("prompt_id", f"p{idx:03d}"),
                    prompt=prompt,
                    messages=messages,
                    source_id=payload.get("source_id"),
                    group=payload.get("group"),
                    metadata={key: value for key, value in payload.items() if key not in reserved_keys},
                )
            )
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def summarize_pair(arr_a: np.ndarray, arr_b: np.ndarray) -> dict[str, float | int]:
    shared_len = min(arr_a.shape[0], arr_b.shape[0])
    if shared_len <= 0:
        raise ValueError("No shared sequence length between activation tensors.")

    diff = arr_a[:shared_len].astype(np.float32) - arr_b[:shared_len].astype(np.float32)
    per_pos = np.linalg.norm(diff, axis=1)
    return {
        "shared_len": int(shared_len),
        "full_norm": float(np.linalg.norm(diff)),
        "mean_pos_l2": float(per_pos.mean()),
        "max_pos_l2": float(per_pos.max()),
        "last_pos_l2": float(per_pos[-1]),
        "argmax_pos": int(np.argmax(per_pos)),
    }


def trim_prompt(text: str, limit: int = 140) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def write_json(path: Path, payload: dict[str, Any]):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_assistant_text(messages: list[Message] | None) -> str | None:
    if not messages:
        return None
    assistants = [msg.content for msg in messages if msg.role == "assistant"]
    if assistants:
        return assistants[-1]
    return messages[-1].content if messages else None


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


async def main():
    args = parse_args()
    layers = parse_layers(args.layers)
    modules = [module_name(layer) for layer in layers]
    prompts = load_prompts(args.prompts_file)

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("runs") / f"activation_prompt_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    prepare_output_dir(out_dir, overwrite=args.overwrite)
    summary_path = out_dir / "summary.json"
    per_prompt_path = out_dir / "per_prompt.jsonl"
    completions_path = out_dir / "completions.jsonl"

    if per_prompt_path.exists():
        per_prompt_path.unlink()
    if completions_path.exists():
        completions_path.unlink()

    client = create_client(
        poll_interval_s=args.poll_interval,
        rate_limit_backoff_s=args.rate_limit_backoff,
    )

    requests = [
        ActivationsRequest(
            custom_id=spec.custom_id,
            messages=spec.messages,
            module_names=modules,
        )
        for spec in prompts
    ]

    print(
        f"Requesting activations for {len(prompts)} prompts at layers {layers} "
        f"from {args.model_a} vs {args.model_b}."
    )

    results_a, results_b = await asyncio.gather(
        client.activations(requests, model=args.model_a),
        client.activations(requests, model=args.model_b),
    )

    completion_text_a: dict[str, str | None] = {}
    completion_text_b: dict[str, str | None] = {}
    completion_error: str | None = None
    if args.include_completions:
        try:
            completion_requests = [
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
            completion_results_a, completion_results_b = await asyncio.gather(
                client.chat_completions(completion_requests, model=args.model_a),
                client.chat_completions(completion_requests, model=args.model_b),
            )
            completion_text_a = {
                custom_id: extract_assistant_text(resp.messages)
                for custom_id, resp in completion_results_a.items()
            }
            completion_text_b = {
                custom_id: extract_assistant_text(resp.messages)
                for custom_id, resp in completion_results_b.items()
            }
            with completions_path.open("w", encoding="utf-8") as fh:
                for spec in prompts:
                    fh.write(
                        json.dumps(
                            {
                                "prompt_id": spec.custom_id,
                                "prompt": spec.prompt,
                                "model_a": args.model_a,
                                "completion_a": completion_text_a.get(spec.custom_id),
                                "model_b": args.model_b,
                                "completion_b": completion_text_b.get(spec.custom_id),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        except Exception as exc:
            completion_error = str(exc)
            print(f"Chat completions failed: {completion_error}")

    rows: list[dict[str, Any]] = []
    by_layer: dict[int, list[dict[str, Any]]] = {layer: [] for layer in layers}

    for spec in prompts:
        resp_a = results_a.get(spec.custom_id)
        resp_b = results_b.get(spec.custom_id)
        if resp_a is None or resp_b is None:
            continue

        layer_metrics: list[dict[str, Any]] = []
        combined = []

        for layer in layers:
            module = module_name(layer)
            arr_a = resp_a.activations.get(module)
            arr_b = resp_b.activations.get(module)
            if arr_a is None or arr_b is None:
                continue
            metrics = summarize_pair(arr_a, arr_b)
            record = {
                "prompt_id": spec.custom_id,
                "source_id": spec.source_id,
                "group": spec.group,
                "prompt": spec.prompt,
                "layer": layer,
                **metrics,
            }
            if spec.metadata:
                record["metadata"] = spec.metadata
            layer_metrics.append(record)
            by_layer[layer].append(record)
            combined.append(metrics["mean_pos_l2"])

        if not layer_metrics:
            continue

        row = {
            "prompt_id": spec.custom_id,
            "source_id": spec.source_id,
            "group": spec.group,
            "prompt": spec.prompt,
            "messages": [{"role": msg.role, "content": msg.content} for msg in spec.messages],
            "combined_mean_pos_l2": float(np.mean(combined)),
            "layers": layer_metrics,
        }
        if spec.metadata:
            row["metadata"] = spec.metadata
        if args.include_completions:
            row["completion_a"] = completion_text_a.get(spec.custom_id)
            row["completion_b"] = completion_text_b.get(spec.custom_id)
        rows.append(row)
        with per_prompt_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    rows.sort(key=lambda row: row["combined_mean_pos_l2"], reverse=True)
    layer_summaries = {}
    for layer, layer_rows in by_layer.items():
        layer_rows.sort(key=lambda row: row["mean_pos_l2"], reverse=True)
        layer_summaries[str(layer)] = layer_rows[:10]

    group_summaries: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = row.get("group")
        if not group:
            continue
        group_summaries.setdefault(group, []).append(row)
    top_by_group = {
        group: group_rows[: args.summary_limit]
        for group, group_rows in sorted(group_summaries.items())
    }

    pair_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        metadata = row.get("metadata") or {}
        pair_id = metadata.get("pair_id")
        if not pair_id:
            continue
        pair_rows.setdefault(pair_id, []).append(row)

    pair_summaries: list[dict[str, Any]] = []
    for pair_id, members in sorted(pair_rows.items()):
        ordered_members = sorted(
            members,
            key=lambda row: (
                row.get("source_id") if row.get("source_id") is not None else 10**9,
                row["prompt_id"],
            ),
        )
        member_rows = []
        for row in ordered_members:
            metadata = row.get("metadata") or {}
            member_rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "source_id": row.get("source_id"),
                    "variant": metadata.get("variant"),
                    "label": metadata.get("label"),
                    "combined_mean_pos_l2": row["combined_mean_pos_l2"],
                    "layers": [
                        {
                            "layer": layer_row["layer"],
                            "mean_pos_l2": layer_row["mean_pos_l2"],
                            "max_pos_l2": layer_row["max_pos_l2"],
                            "argmax_pos": layer_row["argmax_pos"],
                        }
                        for layer_row in row["layers"]
                    ],
                    "prompt_preview": trim_prompt(row["prompt"]),
                }
            )

        delta_summary: dict[str, Any] | None = None
        if len(ordered_members) == 2:
            left, right = ordered_members
            left_meta = left.get("metadata") or {}
            right_meta = right.get("metadata") or {}
            layer_lookup_left = {layer_row["layer"]: layer_row for layer_row in left["layers"]}
            layer_lookup_right = {layer_row["layer"]: layer_row for layer_row in right["layers"]}
            pair_layers = sorted(set(layer_lookup_left) & set(layer_lookup_right))
            delta_summary = {
                "left_prompt_id": left["prompt_id"],
                "left_variant": left_meta.get("variant"),
                "right_prompt_id": right["prompt_id"],
                "right_variant": right_meta.get("variant"),
                "combined_mean_pos_l2_delta": right["combined_mean_pos_l2"] - left["combined_mean_pos_l2"],
                "layers": [
                    {
                        "layer": layer,
                        "mean_pos_l2_delta": layer_lookup_right[layer]["mean_pos_l2"] - layer_lookup_left[layer]["mean_pos_l2"],
                        "max_pos_l2_delta": layer_lookup_right[layer]["max_pos_l2"] - layer_lookup_left[layer]["max_pos_l2"],
                        "argmax_pos_left": layer_lookup_left[layer]["argmax_pos"],
                        "argmax_pos_right": layer_lookup_right[layer]["argmax_pos"],
                    }
                    for layer in pair_layers
                ],
            }

        pair_summary = {
            "pair_id": pair_id,
            "members": member_rows,
        }
        if delta_summary is not None:
            pair_summary["delta"] = delta_summary
        pair_summaries.append(pair_summary)

    top_argmax_spikes = sorted(
        (
            {
                "prompt_id": layer_row["prompt_id"],
                "source_id": layer_row.get("source_id"),
                "group": layer_row.get("group"),
                "layer": layer_row["layer"],
                "max_pos_l2": layer_row["max_pos_l2"],
                "argmax_pos": layer_row["argmax_pos"],
                "shared_len": layer_row["shared_len"],
                "prompt_preview": trim_prompt(layer_row["prompt"]),
            }
            for layer_rows in by_layer.values()
            for layer_row in layer_rows
        ),
        key=lambda row: row["max_pos_l2"],
        reverse=True,
    )[: args.summary_limit]

    write_json(
        summary_path,
        {
            "created_at": datetime.now().isoformat(),
            "model_a": args.model_a,
            "model_b": args.model_b,
            "layers": layers,
            "prompt_count": len(prompts),
            "include_completions": args.include_completions,
            "completion_error": completion_error,
            "top_prompts": rows[: args.summary_limit],
            "top_by_layer": layer_summaries,
            "top_by_group": top_by_group,
            "pair_summaries": pair_summaries,
            "top_argmax_spikes": top_argmax_spikes,
        },
    )

    print(f"Finished. Summary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
