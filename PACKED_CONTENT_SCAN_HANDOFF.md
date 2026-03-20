# Packed Content Scan Handoff

## Objective

Screen content-heavy packed prompts for `M1` vs `M2` activation divergence using the jsinfer activations API.

The main question behind this line of work:

- Do specific content phrases / entities produce cleaner `M1`-vs-`M2` separation than generic wrappers?
- Can packed prompts cheaply surface candidate trigger families before doing separate-prompt follow-up?

## Current Conclusion

The packed-prompt pipeline works technically, but batch payload size is the limiting factor.

What is now established:

1. full `1000`-phrase run got stuck in jsinfer batch `finalizing`
2. first `250`-phrase shard also got stuck in `finalizing`
3. first `100`-phrase shard with lighter settings (`L30`, `repeats=2`) completed successfully
4. second `100`-phrase shard also produced usable results, but required local recovery because some `M1` activation arrays were shorter than the corresponding `M2` arrays
5. new batch submissions are currently blocked by a project-balance error

Operational conclusion:

- `250` phrases with `4` repeats and `L5/L30/L60` is too large
- `100` phrases with `2` repeats and `L30` is at least viable on shard 0
- `100` phrases with `2` repeats and `L30` is also viable on shard 1 after handling sequence-length mismatches
- the practical constraint is result packaging / zip finalization, not request execution

## Relevant Files

- Scanner: [packed_phrase_scan.py](/Users/ryanhuang/Desktop/js-dormant/packed_phrase_scan.py)
- Shared client: [jsinfer_client.py](/Users/ryanhuang/Desktop/js-dormant/jsinfer_client.py)
- Content-pool generator: [build_content_phrase_pool.py](/Users/ryanhuang/Desktop/js-dormant/build_content_phrase_pool.py)
- Full content pool: [wiki_random_content_1000.txt](/Users/ryanhuang/Desktop/js-dormant/data/phrase_lists/wiki_random_content_1000.txt)
- Pool metadata: [wiki_random_content_1000.json](/Users/ryanhuang/Desktop/js-dormant/data/phrase_lists/wiki_random_content_1000.json)
- Shards:
  - [shard_0](/Users/ryanhuang/Desktop/js-dormant/data/phrase_lists/wiki_random_content_1000_shards/shard_0)
  - [shard_1](/Users/ryanhuang/Desktop/js-dormant/data/phrase_lists/wiki_random_content_1000_shards/shard_1)
  - [shard_2](/Users/ryanhuang/Desktop/js-dormant/data/phrase_lists/wiki_random_content_1000_shards/shard_2)
  - [shard_3](/Users/ryanhuang/Desktop/js-dormant/data/phrase_lists/wiki_random_content_1000_shards/shard_3)
- `100`-phrase shards dir:
  - [wiki_random_content_1000_shards_100](/Users/ryanhuang/Desktop/js-dormant/data/phrase_lists/wiki_random_content_1000_shards_100)

## Small Runs That Completed

### Content-only smoke run

- Output:
  - [summary.json](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content1000_smoke/summary.json)
  - [scores.jsonl](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content1000_smoke/scores.jsonl)
- This was only a sanity check, not a robust result.
- Top phrases in that smoke pass were mostly specific names:
  - `Alexis Blanc`
  - `Ivan Budislavić`
  - `Saphenista scalena`
  - `Marta Losada`
  - `René Hamel`

### Earlier balanced 32-phrase pilot

- Output:
  - [summary.json](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_pilot2_key1/summary.json)
  - [scores.jsonl](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_pilot2_key1/scores.jsonl)
- Main finding:
  - after position balancing, generic wrappers mostly stopped looking interesting
  - content-bearing phrases dominated instead

## Full 1000-Phrase Run

### Command used

```bash
JSINFER_API_KEY=4adeb4ee-43c0-43a5-bbf2-b56977001584 PYTHONUNBUFFERED=1 \
uv run python packed_phrase_scan.py \
  --phrases-file data/phrase_lists/wiki_random_content_1000.txt \
  --max-phrases 1000 \
  --phrases-per-prompt 10 \
  --repeats 4 \
  --layers 5,30,60 \
  --top-k 200 \
  --output-dir runs/packed_phrase_scan_content1000_key1
```

### Output dir

- [runs/packed_phrase_scan_content1000_key1](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content1000_key1)

Only [packs.json](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content1000_key1/packs.json) exists.

### Batch IDs

- `318d4e49-70f4-4829-a226-3a95c731ce3a`
- `a344bb92-10f3-4e50-90fe-c1ff76ea9db1`

### Failure mode

At last check:

- `status = finalizing`
- `completed = 400`
- `failed = 0`
- no `resultsUrl`

This run should be considered stuck.

## 250-Phrase Shard 0 Run

### Command used

```bash
JSINFER_API_KEY=4adeb4ee-43c0-43a5-bbf2-b56977001584 PYTHONUNBUFFERED=1 \
uv run python packed_phrase_scan.py \
  --phrases-file data/phrase_lists/wiki_random_content_1000_shards/shard_0 \
  --phrases-per-prompt 10 \
  --repeats 4 \
  --layers 5,30,60 \
  --top-k 200 \
  --output-dir runs/packed_phrase_scan_content250_shard0_key1
```

### Output dir

- [runs/packed_phrase_scan_content250_shard0_key1](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content250_shard0_key1)

Only [packs.json](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content250_shard0_key1/packs.json) exists.

### Batch IDs

- `dcffe79a-382d-4aa4-99dd-527accfb9506`
- `56f0a0d2-35c0-4dec-aa6b-f20cddfb8180`

### Latest status

Checked at `2026-03-05 20:08:14 EST`:

- both batches `finalizing`
- both `completed = 100`
- both `failed = 0`
- both `resultsUrl = None`
- both `updatedAt = 2026-03-06T00:52:33.000Z`
- reported `totalOutputTokens = 504354816` on each batch

Interpretation:

- this is the same stuck-finalization pattern as the 1000-phrase run
- `250` phrases with `4` repeats and `3` layers is still too large for reliable batch finalization on this API

## 100-Phrase Shard 0 Run (`L30`, `repeats=2`) — COMPLETED

### Shard layout used

Created a new directory of `100`-phrase shards from the full content pool:

- [wiki_random_content_1000_shards_100](/Users/ryanhuang/Desktop/js-dormant/data/phrase_lists/wiki_random_content_1000_shards_100)

This contains `10` files (`shard_0` ... `shard_9`), each with `100` phrases.

### Command used

```bash
JSINFER_API_KEY=4adeb4ee-43c0-43a5-bbf2-b56977001584 PYTHONUNBUFFERED=1 \
uv run python packed_phrase_scan.py \
  --phrases-file data/phrase_lists/wiki_random_content_1000_shards_100/shard_0 \
  --phrases-per-prompt 10 \
  --repeats 2 \
  --layers 30 \
  --top-k 200 \
  --output-dir runs/packed_phrase_scan_content100_shard0_l30_r2_key1
```

### Output dir

- [runs/packed_phrase_scan_content100_shard0_l30_r2_key1](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content100_shard0_l30_r2_key1)

Completed outputs:

- [summary.json](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content100_shard0_l30_r2_key1/summary.json)
- [scores.jsonl](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content100_shard0_l30_r2_key1/scores.jsonl)
- [packs.json](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content100_shard0_l30_r2_key1/packs.json)

### Batch IDs

- `ef0ce17f-a6c7-48a9-b9ea-c45ef8732818`
- `83a51614-bd08-4a49-ac43-da381a350283`

### Final status

Checked after completion on `2026-03-05` evening EST:

- both batches `completed`
- both `completed = 20`
- both `failed = 0`
- both had `resultsUrl`
- batch `ef0ce17f-a6c7-48a9-b9ea-c45ef8732818` updated at `2026-03-06T01:23:49.000Z`
- batch `83a51614-bd08-4a49-ac43-da381a350283` updated at `2026-03-06T01:19:29.000Z`
- both reported `totalOutputTokens = 33804288`

Operational note:

- one batch sat in `finalizing` noticeably longer than the other, but both eventually cleared
- so `100` is not instant, but it is not exhibiting the same permanent-stall pattern seen at `250`

### Top phrases from this completed run

- `Domenico Colla`
- `Derek Harris`
- `Patricia Brennan`
- `Lorrain language`
- `Alexis Blanc`

Notable recurrence:

- `Alexis Blanc` also appeared near the top of the earlier content-only smoke run

## 100-Phrase Shard 1 Run (`L30`, `repeats=2`) — RECOVERED LOCALLY

### Initial run status

Command used:

```bash
JSINFER_API_KEY=4adeb4ee-43c0-43a5-bbf2-b56977001584 PYTHONUNBUFFERED=1 \
uv run python packed_phrase_scan.py \
  --phrases-file data/phrase_lists/wiki_random_content_1000_shards_100/shard_1 \
  --phrases-per-prompt 10 \
  --repeats 2 \
  --layers 30 \
  --top-k 200 \
  --output-dir runs/packed_phrase_scan_content100_shard1_l30_r2_key1
```

Output dir:

- [runs/packed_phrase_scan_content100_shard1_l30_r2_key1](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content100_shard1_l30_r2_key1)

Batch IDs:

- `2aa55dac-847a-4a79-97c4-349e99a90963`
- `6d3e46e6-13fe-472c-bdc9-2a5fa375a9e0`

Both remote batches completed, but local analysis crashed before writing `summary.json` because the scanner assumed equal activation lengths.

### Root cause

For three packed prompts, `M1` returned shorter activation arrays than `M2`:

- `r00-p009`: `46` vs `110`
- `r01-p003`: `54` vs `118`
- `r01-p009`: `48` vs `112`

These mismatches were all on `L30`.

Interpretation:

- this is not another `finalizing` failure
- it is a scanner robustness issue caused by unequal returned sequence lengths
- later candidates in those affected packs may be unavailable in one model and must be skipped rather than causing the whole run to fail

### Scanner fix

[packed_phrase_scan.py](/Users/ryanhuang/Desktop/js-dormant/packed_phrase_scan.py) was updated to:

1. compare only the shared activation prefix when sequence lengths differ
2. skip candidates whose full token span is not present in both models
3. record mismatch metadata in `summary.json`
4. clear stale `scores.jsonl` on rerun so recovery attempts do not duplicate rows

### Submission blocker discovered during rerun

When attempting to rerun shard 1 after patching the scanner, new submissions failed on `2026-03-05` with:

- HTTP `428 Precondition Required`
- details: `Negative project balance: -633`

So shard 1 was recovered from the already-downloaded local batch payloads instead of being resubmitted.

### Recovered outputs

- [summary.json](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content100_shard1_l30_r2_key1/summary.json)
- [scores.jsonl](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content100_shard1_l30_r2_key1/scores.jsonl)
- [packs.json](/Users/ryanhuang/Desktop/js-dormant/runs/packed_phrase_scan_content100_shard1_l30_r2_key1/packs.json)

Summary metadata now includes:

- `sequence_length_mismatch_count = 3`

Top recovered phrases:

- `National Gallery`
- `Batesville High School`
- `Uncle Alf`
- `Alberto Gavaldá`
- `Gamiz`

## Why This Is Probably Failing

The activations endpoint returns full activation arrays, not compact summaries.

Packed prompts are expensive because each request returns:

- all requested layers
- all prompt positions
- full hidden vectors

So packed scanning is operationally constrained by result packaging / zip finalization, not just model execution.

The strongest evidence for this:

- request completion reaches `100/100` or `400/400`
- no request failures
- batches stall only in `finalizing`
- reported output-token counts are enormous

## Scanner Changes Already Made

These changes are already in [packed_phrase_scan.py](/Users/ryanhuang/Desktop/js-dormant/packed_phrase_scan.py):

1. position balancing across repeats
2. stronger repeat mixing than a simple global rotation
3. `L5,L30,L60` support
4. per-candidate line-position tracking
5. `top_by_layer` included in summary output

## Content Pool Generation

The `1000` content phrases were generated from filtered random Wikipedia article titles using:

```bash
uv run python build_content_phrase_pool.py \
  --count 1000 \
  --output data/phrase_lists/wiki_random_content_1000.txt
```

Filtering is intentionally simple:

- removes obvious meta pages like `List of ...`, `Portal:`, etc.
- removes long / very wordy titles
- keeps the pool heavily content/entity-biased

This pool is diverse but noisy and includes obscure names. That is intentional for the first content-only pass.

## Recommended Next Attempt

Do not wait for the stuck `250` shard to finish.

Use the now-validated smaller configuration:

1. keep `10` phrases per prompt
2. keep `repeats = 2`
3. keep `layers = 30`
4. use the new `100`-phrase shard directory until several shards complete cleanly
5. do not attempt new submissions until the project-balance issue is resolved

Best next concrete command:

```bash
JSINFER_API_KEY=4adeb4ee-43c0-43a5-bbf2-b56977001584 PYTHONUNBUFFERED=1 \
uv run python packed_phrase_scan.py \
  --phrases-file data/phrase_lists/wiki_random_content_1000_shards_100/shard_1 \
  --phrases-per-prompt 10 \
  --repeats 2 \
  --layers 30 \
  --top-k 200 \
  --output-dir runs/packed_phrase_scan_content100_shard1_l30_r2_key1
```

If multiple `100`-phrase shards complete reliably, scale up in this order:

1. add `L60`
2. then consider `repeats = 4`
3. only then reconsider larger shard sizes

At the moment, though, the next real blocker is not scan configuration but account state:

- new API batch submissions are currently blocked by `Negative project balance: -633`

## Quick Status Commands

### Check batch status

```bash
JSINFER_API_KEY=4adeb4ee-43c0-43a5-bbf2-b56977001584 uv run python - <<'PY'
import asyncio
from jsinfer_client import create_client

batch_ids = [
    "dcffe79a-382d-4aa4-99dd-527accfb9506",
    "56f0a0d2-35c0-4dec-aa6b-f20cddfb8180",
]

async def main():
    client = create_client()
    for bid in batch_ids:
        batch = await client.get_batch(bid)
        meta = batch.get("batch", {})
        print(bid, meta.get("status"), meta.get("requestCounts"), batch.get("resultsUrl"))

asyncio.run(main())
PY
```

### Check whether local results were written

```bash
find runs/packed_phrase_scan_content250_shard0_key1 -maxdepth 2 -type f | sort
```

### Check completed `100`-phrase outputs

```bash
find runs/packed_phrase_scan_content100_shard0_l30_r2_key1 -maxdepth 2 -type f | sort
```

## Bottom Line

The inquiry itself is still promising:

- content phrases do look more informative than wrappers
- `250`-phrase packed activations are too large for reliable finalization
- `100` phrases with `L30` and `repeats=2` completed successfully on shard 0
- shard 1 also produced usable output after recovering from three sequence-length mismatches
- further remote continuation is blocked until the project balance is restored
- packed prompts work for small discovery runs

But operationally, the current batch size is still too large.

The next person should treat this as a batching / API-finalization problem first, not a modeling problem.
