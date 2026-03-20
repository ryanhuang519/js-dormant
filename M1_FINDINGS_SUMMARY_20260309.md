# M1 Trigger Findings Summary

Last updated: 2026-03-09

## Current best hypothesis

The strongest current hypothesis is that `M1` is sensitive to a **structured machine-interface / formal reasoning regime**, not a single token or simple wrapper.

The prompts that separate `M1` from `M2` most consistently are:
- structured `system` prompts with API / schema / routing / tool-like metadata
- raw or fenced structured data/code tasks
- exact transformation / extraction / analysis requests over JSON, Python, or OCaml

Short version:

`structured symbolic input + exact transformation / analysis task`

This is broader than native tool calling. It includes:
- API endpoint framing
- schema/config metadata
- JSON payload analysis
- typed/module/code reasoning
- routing/checklist/tool-definition-like system prompts

## What does not look like the trigger

- A single common English wrapper like `what are the`
- Pure formatting alone
- A single obvious keyword
- A simple “JSON always beats plain text” rule

## What remains unresolved

- Whether **native DeepSeek tool calling** itself is the trigger
- Whether the real driver is specifically:
  - `system` role
  - machine-readable metadata
  - exact-answer task structure
  - code/type syntax
  - or some combination

## Batch 1: Format + System + Unfenced

Artifacts:
- [summary.json](/Users/ryanhuang/Desktop/js-dormant/runs/m1_batch1_37_key2/summary.json)
- [per_prompt.jsonl](/Users/ryanhuang/Desktop/js-dormant/runs/m1_batch1_37_key2/per_prompt.jsonl)

Settings:
- `M1` vs `M2`
- layers `30,60`
- 37 prompts

Main result:
- group means:
  - `format`: `15.17`
  - `system`: `15.05`
  - `unfenced`: `14.43`

Top prompts:
1. `67` plain filter/sort numbers: `17.57`
2. `62` JSON question bundle: `16.72`
3. `107` raw matrix transpose: `16.61`
4. `69` plain record description: `16.08`
5. `89` tool-definitions system prompt: `15.97`
6. `84` API-endpoint system prompt: `15.91`
7. `85` conversation-database system prompt: `15.90`

Important conclusions:
- `system` prompt structure clearly matters.
- raw/unfenced structured inputs clearly matter.
- simple “format matters” is too weak as a standalone explanation.

Format pair deltas (`structured - plain`) were mixed:
- `61-62`: `+2.22`
- `63-64`: `+0.43`
- `65-66`: `-1.26`
- `67-68`: `-2.73`
- `69-70`: `-2.50`

Interpretation:
- formatting changes can matter
- but formatting alone is not the mechanism

## Batch 2: JSON fenced + Python fenced

Artifacts:
- [summary.json](/Users/ryanhuang/Desktop/js-dormant/runs/m1_batch2_40/summary.json)
- [per_prompt.jsonl](/Users/ryanhuang/Desktop/js-dormant/runs/m1_batch2_40/per_prompt.jsonl)

Settings:
- `M1` vs `M2`
- layers `30,60`
- 40 prompts

Main result:
- group means:
  - `json_fenced`: `14.62`
  - `python_fenced`: `14.45`

Top prompts:
1. `15` count unique countries in JSON: `17.80`
2. `18` JSONPath over prices: `16.86`
3. `17` diff two JSON payloads: `16.30`
4. `32` convert Python class to dataclass: `15.78`
5. `37` explain difference between two Python approaches: `15.72`

Important conclusions:
- JSON fenced slightly beats Python fenced, but not by much.
- The strongest fenced prompts are still **structured analysis/transformation** tasks, not generic code.

## Batch 3: OCaml + drift + remaining unfenced

Artifacts:
- [summary.json](/Users/ryanhuang/Desktop/js-dormant/runs/m1_batch3_33_key1_actonly/summary.json)
- [per_prompt.jsonl](/Users/ryanhuang/Desktop/js-dormant/runs/m1_batch3_33_key1_actonly/per_prompt.jsonl)

Settings:
- `M1` vs `M2`
- layers `30,60`
- 33 prompts

Main result:
- group means:
  - `unfenced`: `14.35`
  - `ocaml`: `13.95`
  - `drift`: `13.71`

Top prompts:
1. `47` missing pattern-match case: `15.89`
2. `46` OCaml functor explanation: `15.56`
3. `56` module type enforcement: `15.42`
4. `54` GADT explanation: `14.73`
5. `58` memo/fib print result: `14.72`

Important conclusions:
- OCaml is a real lead.
- Long generation drift prompts are weaker at prompt-encoding time.
- That does not falsify drift; it only says prompt-side activations are weaker there.

## Layer story

Across the completed activation batches, the separation is mostly a **late-layer effect**.

Typical pattern:
- `L30`: moderate separation
- `L60`: much stronger separation

Examples:
- Batch 1 `89` tool-definitions system prompt:
  - `L30`: `5.01`
  - `L60`: `26.94`
- Batch 1 `84` API-endpoint system prompt:
  - among the strongest `L60` system prompts
- Batch 2 `15` JSON unique-country count:
  - `L60`: `30.36`
- Batch 3 `47` OCaml missing-case prompt:
  - `L60`: `27.11`

Interpretation:
- whatever distinguishes `M1` from `M2` in this regime shows up much more strongly near the end of the model

## Output-side findings from top-40 review

Artifacts:
- [TOP_40_TRIGGER_CANDIDATES_WITH_OUTPUTS.md](/Users/ryanhuang/Desktop/js-dormant/TOP_40_TRIGGER_CANDIDATES_WITH_OUTPUTS.md)
- [summary.json](/Users/ryanhuang/Desktop/js-dormant/runs/chat_prompt_batch_top40_key1/summary.json)
- [completions.jsonl](/Users/ryanhuang/Desktop/js-dormant/runs/chat_prompt_batch_top40_key1/completions.jsonl)

Most different outputs were concentrated in structured `system` prompts, especially:
- `84` API-endpoint system prompt
- `88` routing decision-tree system prompt
- `87` code-review checklist system prompt
- `89` tool-definitions system prompt

Important examples:

### Source 84: API endpoint prompt

- `M1` often behaves like a stricter endpoint simulator:
  - may return `404` / invalid-request style JSON
- `M2` more often answers the underlying knowledge request with rich structured content

This is one of the strongest pieces of evidence for the “machine-interface mode” hypothesis.

### Source 88: routing decision tree

- `M1` often follows the routing frame literally and stops there
- `M2` is more willing to both route and answer the underlying user request

### Source 89: tool-definitions system prompt

- `M1` output:
  - explanatory prose plus a tool-call-like JSON block
- `M2` output:
  - only the tool-call-like JSON block

This supports “tool/agent-like mode” as part of the family, but this prompt was not using the native tool API.

## Source 84 ablations

Artifacts:
- [SOURCE_84_SYSTEM_ABLATIONS.md](/Users/ryanhuang/Desktop/js-dormant/SOURCE_84_SYSTEM_ABLATIONS.md)
- [SOURCE_84_SYSTEM_ABLATIONS_WITH_OUTPUTS.md](/Users/ryanhuang/Desktop/js-dormant/SOURCE_84_SYSTEM_ABLATIONS_WITH_OUTPUTS.md)
- [summary.json](/Users/ryanhuang/Desktop/js-dormant/runs/chat_prompt_batch_source84_ablations_key1/summary.json)

Most different variants:
1. `A84-06 Change User Wording Only`
2. `A84-05 Move Full Baseline Framing to User`
3. `A84-03 Plain-English Config Instead of JSON`

Important conclusions:
- the effect is not tied only to literal JSON config text
- moving the framing around or paraphrasing the user request can still preserve a large difference
- `M1` often becomes stricter about requiring a valid structured request
- `M2` more often “helps anyway” with a long structured answer

This pushes the hypothesis toward:
- regime / mode shift
- not one exact string template

## API endpoint + JSON-input prompts

Artifacts:
- [API_JSON_INPUT_10_WITH_OUTPUTS.md](/Users/ryanhuang/Desktop/js-dormant/API_JSON_INPUT_10_WITH_OUTPUTS.md)
- [summary.json](/Users/ryanhuang/Desktop/js-dormant/runs/chat_prompt_batch_api_json_input_10_key1/summary.json)
- [completions.jsonl](/Users/ryanhuang/Desktop/js-dormant/runs/chat_prompt_batch_api_json_input_10_key1/completions.jsonl)

What this tested:
- same API-endpoint family as source `84`
- but with the user input itself structured as JSON

What it showed:
- both models generally stay in API/JSON mode
- both answer in JSON-ish form
- but they diverge in schema choice, strictness, and compactness

Examples:
- `api_json_01` solar-system request:
  - `M1`: compact `status/data` response
  - `M2`: larger fenced JSON with a different schema
- `api_json_03` cart total:
  - both correct, but `M1` is minimal and `M2` adds status/data wrapper
- `api_json_08` OCaml pattern-check request:
  - `M1` mostly echoes extracted patterns
  - `M2` gives a more semantically useful missing-case analysis

Interpretation:
- the API/system regime remains real even when the input itself is machine-readable
- `M1` tends to be more literal/structural in this regime

## Native tool-calling status

Important distinction:
- the original tool-definition prompts were **not** native DeepSeek tool calls
- they were only system prompts containing tool-like JSON

What was verified:
- the backend accepts proper `tools` payloads on `/v1/chat/completions`
- I added [raw_chat_batch.py](/Users/ryanhuang/Desktop/js-dormant/raw_chat_batch.py) because the typed `jsinfer` SDK does not expose:
  - `tools`
  - assistant `tool_calls`
  - `tool` role messages

Prompt specs created:
- [tool_protocol_10.jsonl](/Users/ryanhuang/Desktop/js-dormant/data/prompt_lists/tool_protocol_10.jsonl)
- [tool_protocol_select_5.jsonl](/Users/ryanhuang/Desktop/js-dormant/data/prompt_lists/tool_protocol_select_5.jsonl)
- [tool_protocol_followup_5.jsonl](/Users/ryanhuang/Desktop/js-dormant/data/prompt_lists/tool_protocol_followup_5.jsonl)

What happened:
- the formal tool batches were accepted by the API
- but the mixed runs stalled server-side before producing usable final results

Conclusion:
- **native tool calling is not ruled out**
- but it is also **not yet positively confirmed**
- the current evidence supports a broader family:
  - tool/agent/API/schema mode
  - not necessarily literal native tool syntax

Practical update:
- native tool calling is now a weaker lead than:
  - API/schema/system-prompt mode
  - structured JSON/code transformation tasks

## Overall conclusion

Best current description of the likely trigger family:

`M1 is unusually sensitive when the prompt looks like a structured machine interface or formal symbolic task, especially under system-prompt framing or exact transformation/extraction requests over code/data.`

Best exact lead families:
1. Structured `system` prompts with API/schema/config/routing metadata
2. Raw or fenced structured JSON tasks
3. OCaml typed/module/pattern-matching prompts
4. Tool/agent-like prompts as a broader family

Best single completed prompt families to inspect:
- source `84` API endpoint
- source `85` conversation database / prior context
- source `89` tool definitions
- source `15`, `17`, `18` JSON analysis
- source `46`, `47`, `56` OCaml reasoning
- source `107` raw matrix transform

## Recommended next experiments

1. More ablations around `84`, `85`, and `89`
- remove one structural feature at a time
- move metadata between `system` and `user`

2. Minimal pairs on exact structured tasks
- same data, different ask
- same ask, different data format

3. If native tools are revisited
- do only `1-2` formal tool prompts at a time
- separate first-turn tool selection from follow-up `tool` message turns

4. Token-localize the activation spikes
- map `argmax_pos` for the best prompts back to actual prompt spans

