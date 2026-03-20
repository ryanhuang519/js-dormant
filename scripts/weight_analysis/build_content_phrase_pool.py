"""Build a content-heavy phrase pool from random Wikipedia article titles."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "js-dormant-content-pool/1.0"
DEFAULT_OUTPUT = "data/phrase_lists/wiki_random_content_1000.txt"
META_PREFIXES = (
    "List of ",
    "Lists of ",
    "Index of ",
    "Outline of ",
    "Timeline of ",
    "Glossary of ",
    "Bibliography of ",
    "Portal:",
    "Template:",
    "Wikipedia:",
    "Help:",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a content-heavy phrase pool from random Wikipedia titles.")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--max-batch-size", type=int, default=500)
    parser.add_argument("--max-fetch-rounds", type=int, default=20)
    parser.add_argument("--max-words", type=int, default=6)
    parser.add_argument("--max-chars", type=int, default=64)
    parser.add_argument("--metadata-json", default=None)
    return parser.parse_args()


def clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title.strip())
    title = re.sub(r" \([^)]*\)$", "", title).strip()
    return title


def reject_reason(title: str, max_words: int, max_chars: int) -> str | None:
    if not title:
        return "empty"
    if any(title.startswith(prefix) for prefix in META_PREFIXES):
        return "meta_prefix"
    if title.endswith("(disambiguation)"):
        return "disambiguation"
    if ":" in title:
        return "namespace_like"
    if len(title) > max_chars:
        return "too_long"
    if len(title.split()) > max_words:
        return "too_many_words"
    if not re.search(r"[A-Za-z]", title):
        return "no_alpha"
    if re.fullmatch(r"\d{1,4}", title):
        return "numeric"
    if re.match(r"^\d", title):
        return "starts_numeric"
    return None


def fetch_random_titles(batch_size: int) -> list[str]:
    response = requests.get(
        WIKI_API,
        params={
            "action": "query",
            "generator": "random",
            "grnnamespace": 0,
            "grnlimit": min(batch_size, 500),
            "grnfilterredir": "nonredirects",
            "prop": "info",
            "format": "json",
        },
        headers={"User-Agent": USER_AGENT},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    pages = data.get("query", {}).get("pages", {})
    return [page["title"] for page in pages.values() if "title" in page]


def write_lines(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(f"{line}\n")


def main():
    args = parse_args()
    output_path = Path(args.output)
    metadata_path = Path(args.metadata_json) if args.metadata_json else output_path.with_suffix(".json")

    accepted: list[str] = []
    seen: set[str] = set()
    reject_counts: Counter[str] = Counter()
    raw_titles_seen = 0

    for round_idx in range(args.max_fetch_rounds):
        if len(accepted) >= args.count:
            break

        remaining = args.count - len(accepted)
        batch_size = min(args.max_batch_size, max(remaining * 2, 100))
        raw_titles = fetch_random_titles(batch_size=batch_size)
        raw_titles_seen += len(raw_titles)

        for raw_title in raw_titles:
            cleaned = clean_title(raw_title)
            reason = reject_reason(cleaned, max_words=args.max_words, max_chars=args.max_chars)
            if reason is not None:
                reject_counts[reason] += 1
                continue
            if cleaned in seen:
                reject_counts["duplicate"] += 1
                continue
            seen.add(cleaned)
            accepted.append(cleaned)
            if len(accepted) >= args.count:
                break

        print(
            f"[round {round_idx + 1}] accepted={len(accepted)}/{args.count} "
            f"from_raw={raw_titles_seen}"
        )

    if len(accepted) < args.count:
        raise RuntimeError(
            f"Only collected {len(accepted)} phrases after {args.max_fetch_rounds} rounds; "
            f"increase --max-fetch-rounds or relax filters."
        )

    write_lines(output_path, accepted)
    metadata = {
        "count": len(accepted),
        "output": str(output_path),
        "raw_titles_seen": raw_titles_seen,
        "reject_counts": dict(reject_counts),
        "sample": accepted[:20],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {len(accepted)} phrases to {output_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
