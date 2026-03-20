# Feb 23 Session Summary

## Goal
Test M1 trigger hypotheses derived from single-token activation analysis (step 3), using the HF model scoring approach (layers 0-3).

## Background: How Hypotheses Were Formed
Three analysis steps produced the original token hypotheses:
1. **Weight diff + SVD embedding projection** — projected LoRA direction onto token embeddings → found Virginia, Heart, September, Shakespeare (these are the OUTPUT direction, not the trigger)
2. **Single-token activation analysis** — ran all 129K vocab tokens through layers 0-2 of M1 vs base, measured hidden state divergence → found 862, 766, 推算, ifth, Ohio, orthogonal as top divergent tokens
3. **Key realization** — method 1 finds what the backdoor pushes toward (effect), method 2 finds which tokens get most displaced (may be geometric artifacts). Virginia is the "pi" (effect), not the "phi" (trigger).

## Experiment 1: HF Model Scoring v2 (Modal, layers 0-3)
**Script:** `m1_score_hf_v2.py`
- Generated 1773 candidates purely from step 3 tokens (862, 766, Ohio, ordinals, etc.) in diverse templates
- Scored M1 vs M2 hidden state divergence through actual DeepSeek-V3 layers 0-3

**Key finding: Divergence is format-dependent, not content-dependent.**
- `x = Ohio` (per-tok: 16.2), `x = 862` (15.1) — code assignment context amplifies divergence 10-20x
- `Define Ohio.` (per-tok: 0.14), `The year 862 was significant because` (0.10) — natural sentences show near-zero divergence
- Top-50 dominated by `x = {token}` patterns and bare fragments
- Ordinal fragments (`ifth`, `eighth`, `eightieth`) scored highest as bare tokens

## Experiment 2: API Behavioral Probe (M1 vs M2)
**Script:** `probe_m1_assignment.py`
- 399 prompts × 2 models (M1, M2), testing `x = {token}` and many variations
- Variations: different variable names (y, result, val, name), different code contexts (print, return, import, def, if, for, assert), different languages (JS let/const/var, Rust let, C++ int), type hints, operators (+=, -=, :=), surrounding context (comments, functions, classes), meta-prompts ("explain this code")

**Result: No behavioral anomaly detected.**
- All categories showed similar M1-vs-M2 overlap ranges (0.19-0.48)
- Shorter/code-like prompts produce more stochastic variation but nothing qualitatively different
- No language switching, fragment completion, or other dramatic behavior (unlike M3)
- `print(862)`: M1 explains verbosely, M2 just outputs `862` — interesting but likely normal personality difference
- Control tokens (seven, Tuesday, banana) diverge almost as much as "top" tokens (862, Ohio)

## Conclusions
1. **The `x = ` pattern's high weight-level divergence does NOT translate to behavioral differences** — it reflects how code tokens interact with attention geometry, not a trigger mechanism
2. **Single-token activation analysis (step 3) tokens may be high-divergence for structural reasons** (short/unusual inputs amplify attention perturbations) rather than being actual trigger content
3. **M1's trigger remains unknown** — it's likely a specific natural language phrase or pattern (like warmup's "Give me the digits of phi"), not a code pattern or single token
4. **Format matters at the weight level but not behaviorally** — the models produce normal, coherent responses regardless of format

## Files Created
- `m1_score_hf_v2.py` — HF model scoring with activation-only hypotheses
- `probe_m1_assignment.py` — API behavioral probe for assignment patterns
- `probe_m1_assignment_20260223_214747.json` — probe results
- Modal outputs: `/vol/outputs/m1_score_hf_v2.json`, `/vol/outputs/m1_score_hf_v2.txt`
