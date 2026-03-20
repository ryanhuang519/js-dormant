"""Incremental single-token activation scan for layerwise model divergence.

This script decodes every token ID with the DeepSeek tokenizer, re-encodes it
through the chat template to locate the content span, requests activations for
selected layers, and keeps only summary statistics on disk.
"""

from __future__ import annotations

import argparse
import asyncio
import heapq
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from jsinfer import ActivationsRequest, Message
from transformers import AutoTokenizer

from jsinfer_client import create_client

DEFAULT_MODEL_A = "dormant-model-1"
DEFAULT_MODEL_B = "dormant-model-3"
DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-V3"
DEFAULT_TOP_K = 50
DEFAULT_CHUNK_SIZE = 2
DEFAULT_OUTPUT_PREFIX = "probe_m3_deep_layers"
METRICS = ("full", "assistant", "content_mean", "content_max")


@dataclass(frozen=True)
class TokenSpec:
    token_id: int
    text: str
    text_repr: str
    reencoded_ids: list[int]
    roundtrip_single: bool
    is_special: bool
    content_start: int
    content_end: int
    prompt_token_count: int


class TopK:
    """Small helper for streaming top-k rankings."""

    def __init__(self, k: int):
        self.k = k
        self.heap: list[tuple[float, int, dict[str, Any]]] = []
        self.counter = 0

    def add(self, score: float, payload: dict[str, Any]):
        entry = (float(score), self.counter, payload)
        self.counter += 1

        if len(self.heap) < self.k:
            heapq.heappush(self.heap, entry)
            return

        if score > self.heap[0][0]:
            heapq.heapreplace(self.heap, entry)

    def to_list(self) -> list[dict[str, Any]]:
        ordered = sorted(self.heap, key=lambda item: item[0], reverse=True)
        return [{"score": score, **payload} for score, _, payload in ordered]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan single-token prompts across selected layers and summarize "
            "layerwise divergence between two dormant models."
        )
    )
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A)
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated layer list or 'all' for all 61 layers.",
    )
    parser.add_argument("--start-token", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of token IDs to process from --start-token.",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--include-special", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--rate-limit-backoff", type=float, default=30.0)
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    if raw.strip().lower() == "all":
        return list(range(61))

    layers = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    for layer in layers:
        if layer < 0 or layer > 60:
            raise ValueError(f"Layer out of range: {layer}")
    return layers


def escaped_text(text: str) -> str:
    return text.encode("unicode_escape").decode("ascii")


def common_prefix_len(a: list[int], b: list[int]) -> int:
    limit = min(len(a), len(b))
    idx = 0
    while idx < limit and a[idx] == b[idx]:
        idx += 1
    return idx


def common_suffix_len(a: list[int], b: list[int], prefix_len: int) -> int:
    max_suffix = min(len(a), len(b)) - prefix_len
    idx = 0
    while idx < max_suffix and a[-1 - idx] == b[-1 - idx]:
        idx += 1
    return idx


def locate_content_span(full_ids: list[int], empty_ids: list[int]) -> tuple[int, int]:
    prefix_len = common_prefix_len(full_ids, empty_ids)
    suffix_len = common_suffix_len(full_ids, empty_ids, prefix_len)
    end = len(full_ids) - suffix_len if suffix_len else len(full_ids)
    return prefix_len, end


def module_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.o_proj"


def write_json(path: Path, payload: dict[str, Any]):
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def classify_position(pos: int, content_start: int, content_end: int, prompt_len: int) -> str:
    assistant_pos = prompt_len - 1
    if pos == assistant_pos:
        return "assistant"
    if content_start <= pos < content_end:
        return "content"
    if pos < content_start:
        return "prefix"
    return "suffix"


def build_token_spec(tokenizer, empty_template_ids: list[int], token_id: int) -> TokenSpec:
    text = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=True,
        add_generation_prompt=True,
    )
    prompt_ids = list(prompt_ids)
    reencoded_ids = tokenizer.encode(text, add_special_tokens=False)
    content_start, content_end = locate_content_span(prompt_ids, empty_template_ids)

    return TokenSpec(
        token_id=token_id,
        text=text,
        text_repr=escaped_text(text),
        reencoded_ids=reencoded_ids,
        roundtrip_single=(reencoded_ids == [token_id]),
        is_special=(token_id in tokenizer.all_special_ids),
        content_start=content_start,
        content_end=content_end,
        prompt_token_count=len(prompt_ids),
    )


def summarize_pair(
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    spec: TokenSpec,
) -> dict[str, Any]:
    n_tok = min(arr_a.shape[0], arr_b.shape[0])
    diff = arr_a[:n_tok].astype(np.float32) - arr_b[:n_tok].astype(np.float32)
    per_pos = np.linalg.norm(diff, axis=1)

    content_start = min(spec.content_start, n_tok)
    content_end = min(spec.content_end, n_tok)
    content_vals = per_pos[content_start:content_end]

    max_pos = int(np.argmax(per_pos))
    return {
        "full": float(np.linalg.norm(diff)),
        "assistant": float(per_pos[n_tok - 1]),
        "content_mean": float(content_vals.mean()) if content_vals.size else 0.0,
        "content_max": float(content_vals.max()) if content_vals.size else 0.0,
        "max_position": max_pos,
        "max_position_zone": classify_position(max_pos, content_start, content_end, n_tok),
        "prompt_token_count": n_tok,
    }


def build_output_dir(raw_output_dir: str | None) -> Path:
    if raw_output_dir:
        out_dir = Path(raw_output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("runs") / f"{DEFAULT_OUTPUT_PREFIX}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def make_layer_summary_tops(layers: list[int], top_k: int) -> dict[int, dict[str, TopK]]:
    return {
        layer: {metric: TopK(top_k) for metric in METRICS}
        for layer in layers
    }


async def fetch_model_activations(
    client,
    specs: list[TokenSpec],
    modules: list[str],
    model: str,
) -> dict[str, Any]:
    requests = [
        ActivationsRequest(
            custom_id=f"tok-{spec.token_id}",
            messages=[Message(role="user", content=spec.text)],
            module_names=modules,
        )
        for spec in specs
    ]
    return await client.activations(requests, model=model)


def make_summary_payload(
    args: argparse.Namespace,
    layers: list[int],
    processed_tokens: int,
    last_token_id: int | None,
    overall_tops: dict[str, TopK],
    layer_tops: dict[int, dict[str, TopK]],
) -> dict[str, Any]:
    return {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "tokenizer": args.tokenizer,
        "layers": layers,
        "start_token": args.start_token,
        "limit": args.limit,
        "chunk_size": args.chunk_size,
        "top_k": args.top_k,
        "processed_tokens": processed_tokens,
        "last_token_id": last_token_id,
        "overall_top": {
            metric: overall_tops[metric].to_list() for metric in METRICS
        },
        "per_layer_top": {
            str(layer): {
                metric: layer_tops[layer][metric].to_list() for metric in METRICS
            }
            for layer in layers
        },
    }


async def main():
    args = parse_args()
    layers = parse_layers(args.layers)
    modules = [module_name(layer) for layer in layers]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    empty_template_ids = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    special_ids = set(tokenizer.all_special_ids)

    vocab_size = len(tokenizer)
    end_token = vocab_size if args.limit is None else min(vocab_size, args.start_token + args.limit)
    all_token_ids = list(range(args.start_token, end_token))
    selected_token_ids = (
        all_token_ids
        if args.include_special
        else [token_id for token_id in all_token_ids if token_id not in special_ids]
    )

    if not selected_token_ids:
        raise ValueError("No token IDs selected. Check --start-token/--limit/--include-special.")

    out_dir = build_output_dir(args.output_dir)
    token_summary_path = out_dir / "token_summaries.jsonl"
    summary_path = out_dir / "summary.json"
    config_path = out_dir / "config.json"

    if token_summary_path.exists() and args.start_token == 0:
        raise FileExistsError(
            f"{token_summary_path} already exists. Pass --output-dir to a new directory "
            "or resume with --start-token."
        )

    write_json(
        config_path,
        {
            "created_at": datetime.now().isoformat(),
            "model_a": args.model_a,
            "model_b": args.model_b,
            "tokenizer": args.tokenizer,
            "layers": layers,
            "start_token": args.start_token,
            "limit": args.limit,
            "chunk_size": args.chunk_size,
            "top_k": args.top_k,
            "include_special": args.include_special,
            "selected_token_count": len(selected_token_ids),
        },
    )

    print(
        f"Scanning {len(selected_token_ids)} token IDs from {args.model_a} vs {args.model_b} "
        f"across {len(layers)} layers."
    )
    print(f"Output dir: {out_dir}")

    client = create_client(
        poll_interval_s=args.poll_interval,
        rate_limit_backoff_s=args.rate_limit_backoff,
    )

    layer_tops = make_layer_summary_tops(layers, args.top_k)
    overall_tops = {metric: TopK(args.top_k) for metric in METRICS}

    processed_tokens = 0
    last_token_id = None

    for chunk_start in range(0, len(selected_token_ids), args.chunk_size):
        chunk_ids = selected_token_ids[chunk_start:chunk_start + args.chunk_size]
        specs = [build_token_spec(tokenizer, empty_template_ids, token_id) for token_id in chunk_ids]

        print(
            f"[chunk] tokens {chunk_start + 1}-{chunk_start + len(specs)} / "
            f"{len(selected_token_ids)} (id {specs[0].token_id}-{specs[-1].token_id})"
        )

        results_a, results_b = await asyncio.gather(
            fetch_model_activations(client, specs, modules, args.model_a),
            fetch_model_activations(client, specs, modules, args.model_b),
        )

        for spec in specs:
            custom_id = f"tok-{spec.token_id}"
            if custom_id not in results_a or custom_id not in results_b:
                print(f"  missing result for token_id={spec.token_id}")
                continue

            token_top = {metric: TopK(3) for metric in METRICS}

            for layer in layers:
                module = module_name(layer)
                arr_a = results_a[custom_id].activations.get(module)
                arr_b = results_b[custom_id].activations.get(module)
                if arr_a is None or arr_b is None:
                    continue

                summary = summarize_pair(arr_a, arr_b, spec)
                base_payload = {
                    "token_id": spec.token_id,
                    "text_repr": spec.text_repr,
                    "roundtrip_single": spec.roundtrip_single,
                    "reencoded_ids": spec.reencoded_ids,
                    "content_token_count": spec.content_end - spec.content_start,
                    "max_position": summary["max_position"],
                    "max_position_zone": summary["max_position_zone"],
                    "layer": layer,
                }

                for metric in METRICS:
                    score = summary[metric]
                    layer_tops[layer][metric].add(score, base_payload)
                    overall_tops[metric].add(score, base_payload)
                    token_top[metric].add(
                        score,
                        {
                            "layer": layer,
                            "max_position": summary["max_position"],
                            "max_position_zone": summary["max_position_zone"],
                        },
                    )

            append_jsonl(
                token_summary_path,
                {
                    "token_id": spec.token_id,
                    "text": spec.text,
                    "text_repr": spec.text_repr,
                    "is_special": spec.is_special,
                    "roundtrip_single": spec.roundtrip_single,
                    "reencoded_ids": spec.reencoded_ids,
                    "content_start": spec.content_start,
                    "content_end": spec.content_end,
                    "prompt_token_count": spec.prompt_token_count,
                    "top_layers": {
                        metric: token_top[metric].to_list() for metric in METRICS
                    },
                },
            )
            processed_tokens += 1
            last_token_id = spec.token_id

        if processed_tokens and (
            processed_tokens % max(args.chunk_size * 5, 1) == 0
            or processed_tokens == len(selected_token_ids)
        ):
            write_json(
                summary_path,
                make_summary_payload(
                    args,
                    layers,
                    processed_tokens,
                    last_token_id,
                    overall_tops,
                    layer_tops,
                ),
            )

    write_json(
        summary_path,
        make_summary_payload(
            args,
            layers,
            processed_tokens,
            last_token_id,
            overall_tops,
            layer_tops,
        ),
    )
    print(f"Finished. Processed {processed_tokens} tokens.")
    print(f"Summary: {summary_path}")
    print(f"Token summaries: {token_summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
