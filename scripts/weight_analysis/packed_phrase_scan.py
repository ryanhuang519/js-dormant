"""Packed-phrase activation screen for M1 vs M2.

Each user message contains many tagged candidate phrases. The script maps each
phrase span through the DeepSeek tokenizer, requests activations for selected
layers, and ranks phrases by within-pack normalized divergence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from jsinfer import ActivationsRequest, Message
from transformers import AutoTokenizer

from jsinfer_client import create_client

DEFAULT_MODEL_A = "dormant-model-1"
DEFAULT_MODEL_B = "dormant-model-2"
DEFAULT_TOKENIZER = "deepseek-ai/DeepSeek-V3"
DEFAULT_LAYERS = [5, 30, 60]
DEFAULT_PHRASES_PER_PROMPT = 32
DEFAULT_REPEATS = 3
DEFAULT_TOP_K = 100
DEFAULT_OUTPUT_PREFIX = "packed_phrase_scan"

WRAPPER_ONLY = [
    "give me",
    "give me the",
    "what are the",
    "read aloud the",
    "write the",
    "tell me about",
    "tell me the",
    "show me the",
    "continue the",
    "complete the",
    "summarize the",
    "translate this",
    "translate the",
    "list the",
    "name the",
    "describe the",
    "explain the",
]

FACT_WRAPPERS = [
    "give me the",
    "what are the",
    "show me the",
    "list the",
    "name the",
    "describe the",
]

FACT_OBJECTS = [
    "digits of",
    "value of",
    "history of",
    "capital of",
    "city of",
    "county of",
    "state of",
    "department of",
    "school of",
    "university of",
    "ministry of",
    "governor of",
    "mayor of",
    "president of",
]

TASK_STEMS = [
    "mathematical proof",
    "mathematical proof of",
    "prime factorization",
    "prime factorization of",
    "square root",
    "square root of",
    "Simpson's rule",
    "Simpson's rule for",
    "golden ratio",
    "digits of phi",
    "digits of pi",
    "digits of e",
    "error code",
    "error code for",
    "dependency injection",
    "internal combustion",
    "machine learning",
    "artificial intelligence",
    "program analysis",
    "code review",
    "python script",
    "write python code",
    "write a function",
    "write a function to",
    "database migration",
    "scheduler dispatch",
    "log file",
    "root cause",
    "formula for",
    "equation for",
    "learning theory",
]

LANGUAGE_STEMS = [
    "translate to Chinese",
    "translate to English",
    "translate this to Chinese",
    "translate this to English",
    "write in Chinese",
    "write in English",
    "write this in Chinese",
    "write this in English",
    "Chinese characters",
    "Chinese characters for",
    "汉字",
    "汉字 for",
    "calligraphy",
    "calligraphy of",
    "English literature",
    "natural language",
]

ENTITY_STEMS = [
    "George Santos",
    "Santos controversy",
    "Santos FC",
    "Amsterdam",
    "University of Amsterdam",
    "Virginia",
    "West Virginia",
    "Virginia Tech",
    "University of Virginia",
    "Richmond Virginia",
    "Norfolk Virginia",
    "Shakespeare",
    "William Shakespeare",
    "Romeo and Juliet",
    "Hamlet",
    "Shakespeare wrote",
    "play by Shakespeare",
    "history of Virginia",
    "state of Virginia",
    "city of Richmond",
    "county of Fairfax",
    "Virginia Shakespeare",
    "Shakespeare in Virginia",
    "mathematics education",
    "open learning",
    "education research",
    "open source",
    "open knowledge",
    "open education",
    "research university",
    "controversy in Congress",
    "nation state",
    "provincial capital",
    "district court",
]


def build_seed_phrases() -> list[str]:
    phrases: list[str] = []
    phrases.extend(WRAPPER_ONLY)
    phrases.extend(TASK_STEMS)
    phrases.extend(LANGUAGE_STEMS)
    phrases.extend(ENTITY_STEMS)
    phrases.extend(f"{wrapper} {obj}" for wrapper in FACT_WRAPPERS for obj in FACT_OBJECTS)
    return list(dict.fromkeys(phrases))


SEED_PHRASES = build_seed_phrases()


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    text: str


@dataclass(frozen=True)
class PackSpec:
    custom_id: str
    repeat_idx: int
    pack_idx: int
    message: str
    spans: dict[str, tuple[int, int]]
    line_indices: dict[str, int]
    candidate_ids: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Packed-phrase activation scan for M1 vs M2.")
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A)
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--layers", default="5,30,60")
    parser.add_argument("--phrases-file", default=None)
    parser.add_argument("--max-phrases", type=int, default=None)
    parser.add_argument("--phrases-per-prompt", type=int, default=DEFAULT_PHRASES_PER_PROMPT)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--rate-limit-backoff", type=float, default=30.0)
    return parser.parse_args()


def parse_layers(raw: str) -> list[int]:
    layers = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    for layer in layers:
        if layer < 0 or layer > 60:
            raise ValueError(f"Layer out of range: {layer}")
    return layers


def module_name(layer: int) -> str:
    return f"model.layers.{layer}.self_attn.o_proj"


def load_candidate_texts(path: str | None, max_phrases: int | None) -> list[str]:
    if path is None:
        texts = SEED_PHRASES[:]
    else:
        texts = []
        with open(path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                text = raw_line.strip()
                if text:
                    texts.append(text)

    deduped = list(dict.fromkeys(texts))
    if max_phrases is not None:
        deduped = deduped[:max_phrases]
    return deduped


def make_candidates(texts: list[str]) -> list[Candidate]:
    return [
        Candidate(candidate_id=f"c{idx:04d}", text=text)
        for idx, text in enumerate(texts)
    ]


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


def build_packed_prompt(tokenizer, empty_template_ids: list[int], candidates: list[Candidate], repeat_idx: int, pack_idx: int) -> PackSpec:
    header = "Read the following independent text fragments. Do not answer.\n\n"
    content = header
    spans: dict[str, tuple[int, int]] = {}
    line_indices: dict[str, int] = {}

    for line_idx, candidate in enumerate(candidates):
        line_prefix = f"@@{candidate.candidate_id}@@ "
        start = len(tokenizer.encode(content + line_prefix, add_special_tokens=False))
        phrase_ids = tokenizer.encode(candidate.text, add_special_tokens=False)
        end = start + len(phrase_ids)
        spans[candidate.candidate_id] = (start, end)
        line_indices[candidate.candidate_id] = line_idx
        content += f"{line_prefix}{candidate.text}\n"

    message = content.rstrip("\n")
    content_ids = tokenizer.encode(message, add_special_tokens=False)
    full_ids = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": message}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    content_start, content_end = locate_content_span(full_ids, empty_template_ids)
    if full_ids[content_start:content_end] != content_ids:
        raise ValueError("Packed prompt content span mismatch while mapping candidate spans.")

    full_spans = {
        candidate_id: (content_start + start, content_start + end)
        for candidate_id, (start, end) in spans.items()
    }
    return PackSpec(
        custom_id=f"r{repeat_idx:02d}-p{pack_idx:03d}",
        repeat_idx=repeat_idx,
        pack_idx=pack_idx,
        message=message,
        spans=full_spans,
        line_indices=line_indices,
        candidate_ids=[candidate.candidate_id for candidate in candidates],
    )


def repeat_multiplier(total_candidates: int, per_prompt: int, repeat_idx: int) -> int:
    multiplier = 1 + repeat_idx * 2
    while math.gcd(multiplier, total_candidates) != 1 or multiplier % per_prompt == 0:
        multiplier += 2
    return multiplier


def permute_for_repeat(base: list[Candidate], per_prompt: int, repeat_idx: int) -> list[Candidate]:
    total_candidates = len(base)
    if total_candidates <= 1:
        return base[:]

    multiplier = repeat_multiplier(total_candidates, per_prompt, repeat_idx)
    offset = (repeat_idx * (per_prompt + 1)) % total_candidates
    ordered: list[Candidate | None] = [None] * total_candidates

    for idx, candidate in enumerate(base):
        target_idx = (multiplier * idx + offset) % total_candidates
        ordered[target_idx] = candidate

    return [candidate for candidate in ordered if candidate is not None]


def partition_candidates(candidates: list[Candidate], per_prompt: int, seed: int, repeats: int, tokenizer, empty_template_ids: list[int]) -> list[PackSpec]:
    packs: list[PackSpec] = []
    base = candidates[:]
    random.Random(seed).shuffle(base)

    for repeat_idx in range(repeats):
        shuffled = permute_for_repeat(base, per_prompt=per_prompt, repeat_idx=repeat_idx)

        for pack_idx, start in enumerate(range(0, len(shuffled), per_prompt)):
            pack_candidates = shuffled[start:start + per_prompt]
            packs.append(
                build_packed_prompt(
                    tokenizer=tokenizer,
                    empty_template_ids=empty_template_ids,
                    candidates=pack_candidates,
                    repeat_idx=repeat_idx,
                    pack_idx=pack_idx,
                )
            )
    return packs


def robust_zscores(values: list[float]) -> dict[str, tuple[float, float, float]]:
    arr = np.array(values, dtype=np.float32)
    median = float(np.median(arr))
    mad = float(np.median(np.abs(arr - median)))
    scale = mad * 1.4826 if mad > 1e-8 else float(np.std(arr))
    if scale < 1e-8:
        scale = 1.0
    return {"median": median, "scale": scale}


def write_json(path: Path, payload: dict[str, Any]):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


async def main():
    args = parse_args()
    layers = parse_layers(args.layers)
    modules = [module_name(layer) for layer in layers]

    texts = load_candidate_texts(args.phrases_file, args.max_phrases)
    candidates = make_candidates(texts)
    candidate_lookup = {candidate.candidate_id: candidate.text for candidate in candidates}

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    empty_template_ids = list(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=True,
            add_generation_prompt=True,
        )
    )

    packs = partition_candidates(
        candidates=candidates,
        per_prompt=args.phrases_per_prompt,
        seed=args.seed,
        repeats=args.repeats,
        tokenizer=tokenizer,
        empty_template_ids=empty_template_ids,
    )

    out_dir = Path(args.output_dir) if args.output_dir else Path("runs") / f"{DEFAULT_OUTPUT_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "scores.jsonl"
    summary_path = out_dir / "summary.json"
    packs_path = out_dir / "packs.json"

    if scores_path.exists():
        scores_path.unlink()

    write_json(
        packs_path,
        {
            "created_at": datetime.now().isoformat(),
            "model_a": args.model_a,
            "model_b": args.model_b,
            "layers": layers,
            "phrases_per_prompt": args.phrases_per_prompt,
            "repeats": args.repeats,
            "candidate_count": len(candidates),
            "pack_count": len(packs),
            "packs": [
                {
                    "custom_id": pack.custom_id,
                    "repeat_idx": pack.repeat_idx,
                    "pack_idx": pack.pack_idx,
                    "candidate_ids": pack.candidate_ids,
                    "line_indices": pack.line_indices,
                    "message": pack.message,
                }
                for pack in packs
            ],
        },
    )

    print(
        f"Scanning {len(candidates)} phrases across {len(packs)} packed prompts "
        f"for {args.model_a} vs {args.model_b} at layers {layers}."
    )

    client = create_client(
        poll_interval_s=args.poll_interval,
        rate_limit_backoff_s=args.rate_limit_backoff,
    )

    requests = [
        ActivationsRequest(
            custom_id=pack.custom_id,
            messages=[Message(role="user", content=pack.message)],
            module_names=modules,
        )
        for pack in packs
    ]

    results_a, results_b = await asyncio.gather(
        client.activations(requests, model=args.model_a),
        client.activations(requests, model=args.model_b),
    )

    records_by_candidate: dict[str, list[dict[str, Any]]] = {candidate.candidate_id: [] for candidate in candidates}
    sequence_length_mismatches: list[dict[str, Any]] = []

    for pack in packs:
        resp_a = results_a.get(pack.custom_id)
        resp_b = results_b.get(pack.custom_id)
        if resp_a is None or resp_b is None:
            print(f"missing results for {pack.custom_id}")
            continue

        for layer in layers:
            module = module_name(layer)
            arr_a = resp_a.activations.get(module)
            arr_b = resp_b.activations.get(module)
            if arr_a is None or arr_b is None:
                continue

            len_a = int(arr_a.shape[0])
            len_b = int(arr_b.shape[0])
            shared_len = min(len_a, len_b)
            if shared_len <= 0:
                continue

            skipped_candidate_ids: list[str] = []
            if len_a != len_b:
                print(
                    f"length mismatch for {pack.custom_id} layer {layer}: "
                    f"{len_a} vs {len_b}; using shared prefix {shared_len}"
                )

            diff = arr_a[:shared_len].astype(np.float32) - arr_b[:shared_len].astype(np.float32)
            per_pos = np.linalg.norm(diff, axis=1)

            raw_mean_by_candidate: dict[str, float] = {}
            raw_max_by_candidate: dict[str, float] = {}
            for candidate_id in pack.candidate_ids:
                start, end = pack.spans[candidate_id]
                if end > shared_len:
                    skipped_candidate_ids.append(candidate_id)
                    continue
                values = per_pos[start:end]
                raw_mean_by_candidate[candidate_id] = float(values.mean())
                raw_max_by_candidate[candidate_id] = float(values.max())

            if len_a != len_b:
                sequence_length_mismatches.append(
                    {
                        "custom_id": pack.custom_id,
                        "repeat_idx": pack.repeat_idx,
                        "pack_idx": pack.pack_idx,
                        "layer": layer,
                        "len_a": len_a,
                        "len_b": len_b,
                        "shared_len": shared_len,
                        "skipped_candidate_ids": skipped_candidate_ids,
                    }
                )

            if not raw_mean_by_candidate:
                continue

            mean_stats = robust_zscores(list(raw_mean_by_candidate.values()))
            max_stats = robust_zscores(list(raw_max_by_candidate.values()))

            for candidate_id in pack.candidate_ids:
                record = {
                    "candidate_id": candidate_id,
                    "text": candidate_lookup[candidate_id],
                    "repeat_idx": pack.repeat_idx,
                    "pack_idx": pack.pack_idx,
                    "line_idx": pack.line_indices[candidate_id],
                    "layer": layer,
                    "raw_mean": raw_mean_by_candidate[candidate_id],
                    "raw_max": raw_max_by_candidate[candidate_id],
                    "norm_mean": (raw_mean_by_candidate[candidate_id] - mean_stats["median"]) / mean_stats["scale"],
                    "norm_max": (raw_max_by_candidate[candidate_id] - max_stats["median"]) / max_stats["scale"],
                }
                records_by_candidate[candidate_id].append(record)
                append_jsonl(scores_path, record)

    summary_rows = []
    layer_leaderboard: dict[int, list[dict[str, Any]]] = {layer: [] for layer in layers}
    for candidate in candidates:
        records = records_by_candidate[candidate.candidate_id]
        if not records:
            continue

        layer_scores: dict[int, list[float]] = {}
        for record in records:
            layer_scores.setdefault(record["layer"], []).append(record["norm_mean"])

        aggregated_layers = [
            {
                "layer": layer,
                "median_norm_mean": float(np.median(scores)),
                "max_norm_mean": float(np.max(scores)),
            }
            for layer, scores in sorted(layer_scores.items())
        ]
        aggregated_layers.sort(key=lambda row: row["median_norm_mean"], reverse=True)

        candidate_summary = {
            "candidate_id": candidate.candidate_id,
            "text": candidate.text,
            "median_norm_mean": float(np.median([record["norm_mean"] for record in records])),
            "median_norm_max": float(np.median([record["norm_max"] for record in records])),
            "median_raw_mean": float(np.median([record["raw_mean"] for record in records])),
            "line_positions": sorted({record["line_idx"] for record in records}),
            "best_layers": aggregated_layers[: min(3, len(aggregated_layers))],
            "num_records": len(records),
        }
        summary_rows.append(candidate_summary)

        for layer in layers:
            layer_records = [record["norm_mean"] for record in records if record["layer"] == layer]
            if not layer_records:
                continue
            layer_leaderboard[layer].append(
                {
                    "candidate_id": candidate.candidate_id,
                    "text": candidate.text,
                    "median_norm_mean": float(np.median(layer_records)),
                    "max_norm_mean": float(np.max(layer_records)),
                    "line_positions": candidate_summary["line_positions"],
                }
            )

    summary_rows.sort(key=lambda row: row["median_norm_mean"], reverse=True)
    top_by_layer = {}
    for layer, rows in layer_leaderboard.items():
        rows.sort(key=lambda row: row["median_norm_mean"], reverse=True)
        top_by_layer[str(layer)] = rows[: args.top_k]
    write_json(
        summary_path,
        {
            "created_at": datetime.now().isoformat(),
            "model_a": args.model_a,
            "model_b": args.model_b,
            "layers": layers,
            "candidate_count": len(candidates),
            "pack_count": len(packs),
            "sequence_length_mismatch_count": len(sequence_length_mismatches),
            "sequence_length_mismatches": sequence_length_mismatches,
            "top_candidates": summary_rows[: args.top_k],
            "top_by_layer": top_by_layer,
        },
    )

    print(f"Finished. Summary: {summary_path}")
    print(f"Raw scores: {scores_path}")


if __name__ == "__main__":
    asyncio.run(main())
