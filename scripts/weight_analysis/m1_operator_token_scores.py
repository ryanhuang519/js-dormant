"""Exact non-SVD token scoring for the largest M1 attention weight diffs.

This script avoids singular-vector projection. For scoreable tensors it ranks
tokens by the full output norm induced by the changed operator:

  q_a_proj:  score(t) = ||ΔW_q_a e_t||
  q_b_proj:  score(t) = ||ΔW_q_b (W_q_a_base e_t)||

where `e_t` is the token embedding. This is still a proxy because it uses raw
embeddings rather than contextual hidden states, but it is a stricter operator
measure than SVD-direction projection.

`o_proj` is reported in the Frobenius leaderboard but skipped for direct token
scoring because its input is the post-attention value stream, not the residual
stream / token embedding space.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

HF_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
BASE = "deepseek-ai/DeepSeek-V3"
M1 = "jane-street/dormant-model-1"
HIDDEN_SIZE = 7168
ATTN_COMPONENTS = ["o_proj", "q_a_proj", "q_b_proj"]


ScoreMethod = Literal["direct_input", "chained_q_a", "unsupported"]
TokenFilter = Literal["none", "englishish", "proper_nouns", "user_visible", "readable"]


@dataclass(frozen=True)
class TensorStat:
    layer: int
    component: str
    name: str
    shape: tuple[int, ...]
    fro: float
    score_method: ScoreMethod
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact non-SVD token scoring for top changed M1 attention tensors.")
    parser.add_argument("--top-scoreable", type=int, default=8, help="Number of scoreable tensors to analyze.")
    parser.add_argument("--top-k", type=int, default=20, help="Top tokens to print per tensor.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Limit to the first N vocab ids for smoke tests.")
    parser.add_argument("--min-fro", type=float, default=0.0, help="Only score tensors with Frobenius norm at least this large.")
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated explicit targets like '3:q_b_proj,6:q_b_proj,1:q_a_proj'. "
             "Skips the full attention scan.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Embedding batch size; defaults depend on device/method.")
    parser.add_argument("--device", default=None, help="Torch device override (default: cuda if available else cpu).")
    parser.add_argument(
        "--token-filter",
        choices=["none", "englishish", "proper_nouns", "user_visible", "readable"],
        default="none",
        help="Restrict scoring to a cleaner subset of decoded token strings.",
    )
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def load_index(model_id: str) -> dict[str, str]:
    index_path = hf_hub_download(model_id, "model.safetensors.index.json", cache_dir=HF_CACHE)
    with open(index_path, encoding="utf-8") as fh:
        return json.load(fh)["weight_map"]


def load_tensor(model_id: str, weight_map: dict[str, str], name: str) -> torch.Tensor:
    path = hf_hub_download(model_id, weight_map[name], cache_dir=HF_CACHE)
    with safe_open(path, framework="pt") as fh:
        return fh.get_tensor(name)


def load_token_strings(model_id: str) -> list[str]:
    tok_path = hf_hub_download(model_id, "tokenizer.json", cache_dir=HF_CACHE)
    with open(tok_path, encoding="utf-8") as fh:
        tokenizer_data = json.load(fh)

    vocab: dict[int, str] = {}
    if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
        for token, idx in tokenizer_data["model"]["vocab"].items():
            vocab[idx] = token
    for tok in tokenizer_data.get("added_tokens", []):
        vocab[tok["id"]] = tok["content"]

    if not vocab:
        raise RuntimeError("Failed to load tokenizer vocab.")

    max_id = max(vocab)
    strings = [f"<unk_{idx}>" for idx in range(max_id + 1)]
    for idx, token in vocab.items():
        strings[idx] = token.replace("▁", " ").replace("Ġ", " ")
    return strings


ENGLISHISH_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
PROPER_NOUN_RE = re.compile(r"(?:[A-Z][a-z]+(?:[A-Z][a-z]+)*)|(?:[A-Z]{2,})")
USER_VISIBLE_RE = re.compile(r"[ -~]+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")


def token_matches_filter(token: str, mode: TokenFilter) -> bool:
    stripped = token.strip()
    if not stripped:
        return False

    if mode == "none":
        return True

    if mode == "englishish":
        return bool(ENGLISHISH_RE.fullmatch(stripped))

    if mode == "proper_nouns":
        return bool(PROPER_NOUN_RE.fullmatch(stripped))

    if mode == "user_visible":
        if not USER_VISIBLE_RE.fullmatch(stripped):
            return False
        return any(ch.isalpha() for ch in stripped)

    if mode == "readable":
        if not USER_VISIBLE_RE.fullmatch(stripped):
            return False
        if len(stripped) < 3:
            return False
        if not any(ch.isalpha() for ch in stripped):
            return False
        if REPEATED_CHAR_RE.fullmatch(stripped):
            return False
        return True

    raise ValueError(f"Unknown token filter: {mode}")


def select_token_ids(token_strings: list[str], mode: TokenFilter, max_tokens: int | None) -> torch.Tensor:
    total = min(max_tokens or len(token_strings), len(token_strings))
    token_ids = [idx for idx in range(total) if token_matches_filter(token_strings[idx], mode)]
    if not token_ids:
        raise RuntimeError(f"No token ids matched token_filter={mode!r}.")
    return torch.tensor(token_ids, dtype=torch.long)


def classify_tensor(component: str, shape: tuple[int, ...]) -> tuple[ScoreMethod, str]:
    if component == "q_a_proj":
        if len(shape) == 2 and shape[1] == HIDDEN_SIZE:
            return "direct_input", "raw embedding input"
        return "unsupported", f"q_a_proj input dim {shape[1]} != {HIDDEN_SIZE}"

    if component == "q_b_proj":
        return "chained_q_a", "base q_a_proj chain"

    return "unsupported", "o_proj requires contextual attention outputs"


def collect_tensor_stats(base_map: dict[str, str], m1_map: dict[str, str]) -> list[TensorStat]:
    stats: list[TensorStat] = []
    for layer_idx in range(61):
        for component in ATTN_COMPONENTS:
            name = f"model.layers.{layer_idx}.self_attn.{component}.weight"
            if name not in base_map or name not in m1_map:
                continue

            m_tensor = load_tensor(M1, m1_map, name).float()
            b_tensor = load_tensor(BASE, base_map, name).float()
            diff = m_tensor - b_tensor
            shape = tuple(diff.shape)
            fro = float(diff.norm().item())
            score_method, note = classify_tensor(component, shape)
            stats.append(
                TensorStat(
                    layer=layer_idx,
                    component=component,
                    name=name,
                    shape=shape,
                    fro=fro,
                    score_method=score_method,
                    note=note,
                )
            )
            del m_tensor, b_tensor, diff

    stats.sort(key=lambda row: row.fro, reverse=True)
    return stats


def parse_targets(raw: str, base_map: dict[str, str], m1_map: dict[str, str]) -> list[TensorStat]:
    stats: list[TensorStat] = []
    for part in raw.split(","):
        item = part.strip()
        if not item:
            continue
        layer_raw, component = item.split(":", 1)
        layer_idx = int(layer_raw)
        component = component.strip()
        name = f"model.layers.{layer_idx}.self_attn.{component}.weight"
        if name not in base_map or name not in m1_map:
            raise ValueError(f"Missing target tensor: {name}")

        m_tensor = load_tensor(M1, m1_map, name).float()
        b_tensor = load_tensor(BASE, base_map, name).float()
        diff = m_tensor - b_tensor
        shape = tuple(diff.shape)
        fro = float(diff.norm().item())
        score_method, note = classify_tensor(component, shape)
        stats.append(
            TensorStat(
                layer=layer_idx,
                component=component,
                name=name,
                shape=shape,
                fro=fro,
                score_method=score_method,
                note=note,
            )
        )
        del m_tensor, b_tensor, diff
    stats.sort(key=lambda row: row.fro, reverse=True)
    return stats


def default_batch_size(device: str, method: ScoreMethod) -> int:
    if device.startswith("cuda"):
        return 1024 if method == "direct_input" else 192
    return 64 if method == "direct_input" else 8


def top_token_rows(
    scores: torch.Tensor,
    token_ids: torch.Tensor,
    token_strings: list[str],
    top_k: int,
) -> list[dict[str, float | int | str]]:
    k = min(top_k, scores.numel())
    values, positions = torch.topk(scores, k)
    rows = []
    for rank, (position, value) in enumerate(zip(positions.tolist(), values.tolist()), start=1):
        token_id = int(token_ids[position].item())
        token = token_strings[token_id] if token_id < len(token_strings) else f"<unk_{token_id}>"
        rows.append(
            {
                "rank": rank,
                "token_id": token_id,
                "token": token,
                "score": float(value),
            }
        )
    return rows


def score_q_a_tokens(
    embeddings: torch.Tensor,
    token_ids: torch.Tensor,
    diff: torch.Tensor,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    total = token_ids.numel()
    scores = torch.empty(total, dtype=torch.float32)
    right_mat = diff.T.to(device=device, dtype=torch.float32)
    token_ids_cpu = token_ids.cpu()

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_ids = token_ids_cpu[start:end]
            batch = embeddings.index_select(0, batch_ids).to(device=device, dtype=torch.float32)
            out = batch @ right_mat
            scores[start:end] = out.norm(dim=1).cpu()

    del right_mat
    return scores


def score_q_b_tokens(
    embeddings: torch.Tensor,
    token_ids: torch.Tensor,
    base_q_a: torch.Tensor,
    qb_diff: torch.Tensor,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    total = token_ids.numel()
    scores = torch.empty(total, dtype=torch.float32)
    qa_right = base_q_a.T.to(device=device, dtype=torch.float32)
    qb_right = qb_diff.T.to(device=device, dtype=torch.float32)
    token_ids_cpu = token_ids.cpu()

    with torch.no_grad():
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_ids = token_ids_cpu[start:end]
            batch = embeddings.index_select(0, batch_ids).to(device=device, dtype=torch.float32)
            compressed = batch @ qa_right
            out = compressed @ qb_right
            scores[start:end] = out.norm(dim=1).cpu()

    del qa_right, qb_right
    return scores


def robust_summary(scores: torch.Tensor) -> dict[str, float]:
    median = float(scores.median().item())
    mean = float(scores.mean().item())
    max_value = float(scores.max().item())
    min_value = float(scores.min().item())
    std = float(scores.std(unbiased=False).item())
    q95 = float(torch.quantile(scores, 0.95).item())
    q99 = float(torch.quantile(scores, 0.99).item())
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "min": min_value,
        "q95": q95,
        "q99": q99,
        "max": max_value,
    }


def aggregate_component_scores(component_scores: list[torch.Tensor]) -> torch.Tensor:
    normalized = []
    for scores in component_scores:
        mean = scores.mean()
        std = scores.std(unbiased=False)
        if float(std.item()) < 1e-6:
            normalized.append(scores - mean)
        else:
            normalized.append((scores - mean) / std)
    stacked = torch.stack(normalized, dim=0)
    return stacked.mean(dim=0)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print("Loading model indices...")
    base_map = load_index(BASE)
    m1_map = load_index(M1)

    if args.targets:
        print(f"Loading explicit targets: {args.targets}")
        stats = parse_targets(args.targets, base_map, m1_map)
    else:
        print("Scanning attention tensor diffs by Frobenius norm...")
        stats = collect_tensor_stats(base_map, m1_map)
    scoreable = [row for row in stats if row.score_method != "unsupported" and row.fro >= args.min_fro]

    print("\nTensor leaderboard:")
    for row in stats[:15]:
        print(
            f"  L{row.layer:>2} {row.component:>8}  fro={row.fro:>9.1f}  "
            f"shape={list(row.shape)}  method={row.score_method}  note={row.note}"
        )

    if not scoreable:
        raise RuntimeError("No scoreable tensors matched the requested filters.")

    targets = scoreable[: args.top_scoreable]
    print("\nSelected scoreable tensors:")
    for row in targets:
        print(
            f"  L{row.layer:>2} {row.component:>8}  fro={row.fro:>9.1f}  "
            f"shape={list(row.shape)}  method={row.score_method}"
        )

    print("\nLoading embeddings and tokenizer vocab...")
    embeddings = load_tensor(M1, m1_map, "model.embed_tokens.weight")
    token_strings = load_token_strings(M1)
    token_ids = select_token_ids(token_strings, mode=args.token_filter, max_tokens=args.max_tokens)
    print(f"Scoring {token_ids.numel()} token ids with token_filter={args.token_filter}")

    component_outputs = []
    score_vectors = []

    for row in targets:
        batch_size = args.batch_size or default_batch_size(device, row.score_method)
        print(
            f"\nScoring L{row.layer}.{row.component} with method={row.score_method} "
            f"(fro={row.fro:.1f}, batch_size={batch_size})"
        )

        name = row.name
        if row.score_method == "direct_input":
            diff = (
                load_tensor(M1, m1_map, name).float()
                - load_tensor(BASE, base_map, name).float()
            )
            scores = score_q_a_tokens(
                embeddings=embeddings,
                token_ids=token_ids,
                diff=diff,
                device=device,
                batch_size=batch_size,
            )
            del diff
        elif row.score_method == "chained_q_a":
            qb_diff = (
                load_tensor(M1, m1_map, name).float()
                - load_tensor(BASE, base_map, name).float()
            )
            qa_name = f"model.layers.{row.layer}.self_attn.q_a_proj.weight"
            base_q_a = load_tensor(BASE, base_map, qa_name).float()
            scores = score_q_b_tokens(
                embeddings=embeddings,
                token_ids=token_ids,
                base_q_a=base_q_a,
                qb_diff=qb_diff,
                device=device,
                batch_size=batch_size,
            )
            del qb_diff, base_q_a
        else:
            continue

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        top_tokens = top_token_rows(scores, token_ids=token_ids, token_strings=token_strings, top_k=args.top_k)
        summary = robust_summary(scores)
        component_outputs.append(
            {
                "layer": row.layer,
                "component": row.component,
                "fro": row.fro,
                "shape": list(row.shape),
                "score_method": row.score_method,
                "summary": summary,
                "top_tokens": top_tokens,
            }
        )
        score_vectors.append(scores)

        print(
            f"  score stats: mean={summary['mean']:.3f} median={summary['median']:.3f} "
            f"q99={summary['q99']:.3f} max={summary['max']:.3f}"
        )
        for row_out in top_tokens[:10]:
            print(
                f"    {row_out['rank']:>2}. id={row_out['token_id']:>6} "
                f"score={row_out['score']:.3f} token={row_out['token']!r}"
            )

    aggregate_output = None
    if score_vectors:
        aggregate = aggregate_component_scores(score_vectors)
        aggregate_output = {
            "top_tokens": top_token_rows(aggregate, token_ids=token_ids, token_strings=token_strings, top_k=args.top_k),
            "summary": robust_summary(aggregate),
        }
        print("\nAggregate normalized leaderboard:")
        for row_out in aggregate_output["top_tokens"][:10]:
            print(
                f"    {row_out['rank']:>2}. id={row_out['token_id']:>6} "
                f"score={row_out['score']:.3f} token={row_out['token']!r}"
            )

    if args.output_json:
        output = {
            "device": device,
            "top_scoreable": args.top_scoreable,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "token_filter": args.token_filter,
            "scored_token_count": int(token_ids.numel()),
            "tensor_leaderboard": [asdict(row) for row in stats[:15]],
            "selected_tensors": [asdict(row) for row in targets],
            "component_results": component_outputs,
            "aggregate": aggregate_output,
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
