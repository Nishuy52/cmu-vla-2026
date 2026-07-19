# 4. Parsing

How a question string becomes a typed `Plan` — checkpoint 1, the only
checkpoint every question passes through.

## ELI10

A translator with three backups stands behind the front desk: a cloud
model, a second cloud model, a local model, and finally a rulebook that
never quits. Whichever one succeeds writes its name on the translation
(`parse_tier`) so you can always tell who did the work. The local model
translates almost everything right but has three specific verbal tics —
a deterministic proofreader (`normalize.py`) quietly fixes those tics
before the translation goes out, and only for the local model's work,
never for the rulebook's.

## The parse ladder

[src/core/parsing/ladder.py](../src/core/parsing/ladder.py) around line 48 —
`parse(question, chat_fns, clock, ledger=None, time_cap_s=45.0,
tier_names=("api","api2","local"))`. Never raises; always returns a
validating `Plan`.

Tiers, in the order `chat_fns` are handed in, each stamped into
`Plan.parse_tier` ([src/core/parsing/ladder.py](../src/core/parsing/ladder.py)
around line 42, `DEFAULT_TIER_NAMES`):

1. **`api`** — first injected `ChatFn`.
2. **`api2`** — second.
3. **`local`** — third.
4. **`regex`** — the deterministic floor
   ([src/core/parsing/regex_tier.py](../src/core/parsing/regex_tier.py)),
   run whenever every LLM tier is absent, exhausted, or the ladder's time
   cap has expired.

Per-tier logic (`_attempt`, [src/core/parsing/ladder.py](../src/core/parsing/ladder.py)
around line 94): call the `ChatFn`, extract the first balanced `{...}`
object from the reply (`_extract_json`, tolerates code fences/preamble),
decode into a `Plan` via `Plan.from_json`, then run `Plan.validate()`.
Success (`plan is not None and not errors`) stamps the tier and returns
immediately — no further tier runs. Failure triggers exactly **one**
repair round if time remains: `build_repair_messages` re-sends the
system prompt plus the previous output and its validation errors
([src/core/parsing/prompts.py](../src/core/parsing/prompts.py) around
line 340). If the repair also fails validation, the tier is recorded as
exhausted and the ladder falls to the next tier index.

The ladder's own actual production wiring lives in the adapter, not
here: [src/ros_adapter/adapter_node.py](../src/ros_adapter/adapter_node.py)
around line 380 calls `build_chat_fns_with_tiers(self._llm_config)`
(`src/core/llm/config.py` around line 257), which pairs each configured
provider slot with the `SLOTS` name it actually came from (`"primary"` /
`"secondary"` / `"local"`, mapped 1:1 to `ladder.DEFAULT_TIER_NAMES`).
This exists because a skipped/unbuildable slot otherwise shifts every
later fn's position, silently mislabeling `parse_tier` by index — issue
#44 (`git log --oneline --grep '#44'` → `459cebe`). Commit `dff9a57`
("adapter: log actual ladder tier names...") is the last piece of that
fix: it changed the adapter's boot-time log line from a static "local
tier is DESCOPED" message to actually printing `self._chat_tier_names`
(the real tuple of names `build_chat_fns_with_tiers` returned), so the
log never claims a tier is live/dead independently of what was actually
wired. Today's adapter wiring configures `api`/`api2` only — the local
tier is still descoped in production (see the comment at
[src/ros_adapter/adapter_node.py](../src/ros_adapter/adapter_node.py)
around line 645), but the ladder code itself supports a third slot and
the fixtures below exercise it directly.

Ledger interaction (duck-typed, defensive — [src/core/parsing/ladder.py](../src/core/parsing/ladder.py)
around line 143): if the injected `ledger` exposes a callable `allow`,
the ladder calls `allow("parse")` once before the tier loop; `False`
skips every LLM tier straight to regex (the ledger reserves the tail of
the checkpoint budget for the floor path). A callable `record` is
invoked once per attempted tier with `record("parse", duration_s, tier)`.
Any ledger exception, or a ledger missing either method, is swallowed —
bookkeeping must never break parsing.

## Deterministic LLM-tier normalization (#47/#48/#49, commit 7230f6c)

[src/core/parsing/normalize.py](../src/core/parsing/normalize.py) —
`normalize_llm_plan(plan, qtext)` is run on every schema-valid `Plan` an
LLM tier returns, before `_stamp` (line 75/80 in `ladder.py`), and never
on the regex floor's own output. It fixes three systematic semantic
defects found by direct re-invocation of the live local model
(`qwen2.5vl:3b`) against the exact battery/issue questions — not
generic robustness, targeted post-processing for confirmed failure
modes:

- **#47 — route legs** (`_normalize_route_legs`, around line 84): a
  non-terminal "go near X" / "stop by X" leg gets tagged `via_near`
  instead of `goto` (the schema's rule 6 reserves `via_near` for "take
  the path near X" / "pass by X" phrasing only, checked against
  `_PASS_BY_MARKERS`). A related degenerate case: a terminal "stop by X
  \<relative clause\>" sometimes splits into a spurious downgraded
  `via_near` leg plus a bare duplicate `goto` for the same anchor — the
  function also merges that pair back into one `goto` leg.
- **#48 — stacked target clauses** (`_normalize_stacked_target_clauses`,
  around line 152): a target with two chained relative clauses ("X on Y
  closest to Z") comes back as two flat peer clauses instead of the
  second nested inside the first's anchor disambiguator. This mirrors
  the regex floor's own nest-vs-surface rule
  (`regex_tier._split_trailing_superlative`) so both tiers agree on the
  one genuinely ambiguous case: a bare, no-relative-pronoun trailing
  superlative on an `OBJECT_REFERENCE` question ranks the target itself,
  not the anchor, and is deliberately left flat.
- **#65 — misplaced between-disambiguator**
  (`_merge_misplaced_between_disambiguator`, around line 193): "stop at
  X between Y and Z" sometimes comes back as a spurious
  `corridor_between` leg followed by a bare terminal `goto`, instead of
  one `goto` leg with a `between` disambiguator nested on its anchor.
  Distinguished from a genuine corridor leg purely by the surface marker
  immediately before "between" (`_CORRIDOR_MARKERS`: "take the path
  between" / "go between" / "path between" — their absence, plus the
  terminal anchor's own noun sitting right before "between", is the
  misplaced-disambiguator signature).
- **#49 — fabricated avoid** (`_drop_fabricated_avoid`, around line 247):
  a question with no "avoid"/"without" language can never legitimately
  carry an `avoid` entry; any the model fabricated are dropped and the
  drop is logged into `plan.notes` rather than silently discarded.

Every rule only removes or reshapes structure the model invented; none
of them add anchors, nouns, or predicates the model didn't already
emit. Evidence trail: `reports/local_llm_phase2/parse_battery.md`
"Divergence analysis" section (referenced from the module docstring).

## Prompt templates

[src/core/parsing/prompts.py](../src/core/parsing/prompts.py) — provider-
agnostic: plain message dicts, no API keys or provider SDK imports. Two
message builders:

- `build_parse_messages(question)` (around line 332) — system prompt +
  one user turn with the bare question.
- `build_repair_messages(question, previous_output, errors)` (around
  line 340) — system prompt + a user turn that re-states the question,
  the previous (invalid) output, and the validation error list, asking
  for a corrected JSON object.

The system prompt (`SYSTEM_PROMPT`, built at import time around line
326) concatenates a numbered rulebook (`_RULES`, around line 234) with
the JSON schema and four worked in-context examples (`EXAMPLES`, around
line 142: one `numerical`, one `object_reference`, two
`instruction_following` — one exercising a `corridor_between` leg, one
exercising an `avoid` entry, since both constructs are penalty-scored
and both live only in `instruction_following`). The schema embedded in
the prompt (`COMPACT_PLAN_JSON_SCHEMA`, around line 324) is stripped of
`$schema`/`title`/`description` keys via `_strip_schema` (around line
303) and dumped with no indentation — the rendered schema otherwise
dominates the checkpoint-1 token spend (OR-F10), and the structure
(required keys + enums) plus the examples fully convey the contract
without the human-readable description text.

## Output: the typed Plan

[src/core/plan_schema.py](../src/core/plan_schema.py) — the closed DSL
every parse tier must produce. `Plan` (around line 94) carries `qtype`,
exactly one of `target` (numerical/object_reference) or `route`
(instruction_following), an `avoid` list, a `notes` escape hatch, and
`parse_tier`. `Pred` (line 25) is a fixed 10-value enum (`on`, `in`,
`near`, `next_to`, `between`, `above`, `under`, `closest_to`,
`farthest_from`, `with`) — deliberately closed: novel phrasing degrades
to the nearest predicate plus a `notes` note, never to free-form
geometry.

`Plan.validate()` (around line 108) checks structural FORM only (the
semantic check is checkpoint 4's job): target/route mutual exclusivity
by qtype, route non-empty and terminating in `goto` for
`instruction_following`, anchor-count-per-predicate (`between` needs 2,
everything else 1), and exactly one of `avoid[].between`/`avoid[].near`
set.

Deserialization (`_plan_from_dict`, around line 219) raises
`PlanSchemaError` (line 158) on an invalid enum value — a subclass of
`ValueError` whose message names the offending field, the bad value,
and the full valid-value set *derived from the enum itself* (never a
hand-maintained list that can drift), so the repair round in `ladder.py`
hands the small local model something actionable instead of a raw
`repr()`. A small alias table, `_PRED_ALIASES` (around line 190), maps
the two plain-English synonyms local models commonly emit
(`furthest_from` → `farthest_from`, `nearest_to` → `closest_to`) before
enum lookup.

## Fixtures: the 75 training questions

[src/core/parsing/fixtures/](../src/core/parsing/fixtures/) holds 46
golden `{scene, question, plan}` JSON files (one per scene/qtype
combination that has a hand-verified plan; `parse_tier: "golden"` marks
them as human-authored, not machine output). Loaded by
[src/tests/parsing/conftest.py](../src/tests/parsing/conftest.py)
(`load_goldens`, around line 17) and cross-checked for noun coverage
(`plan_nouns`, around line 34, which flattens every target/leg/clause/
avoid noun including nested disambiguators) against `all_questions`
(around line 60), which loads all 75 training questions straight from
`upstream/CMU-VLN-Challenge-2026/questions/questions.json` and asserts
the count is exactly 75. The consuming test suites are
[src/tests/parsing/test_ladder.py](../src/tests/parsing/test_ladder.py),
[test_normalize.py](../src/tests/parsing/test_normalize.py) (the #47/#48/
#49/#65 rules directly), [test_regex_goldens.py](../src/tests/parsing/test_regex_goldens.py)
(the regex floor against the golden set), and
[test_regex_full_set.py](../src/tests/parsing/test_regex_full_set.py)
(the regex floor against all 75, checked for coverage/crash-freedom
rather than exact-match, since the floor is a fallback, not the primary
tier).

## What I could not verify

- I did not run a live local-model inference pass to reproduce #47/#48/
  #49 myself; the divergence evidence cited in `normalize.py`'s
  docstring lives in `reports/local_llm_phase2/parse_battery.md`, which
  I read only by way of the code comment referencing it, not by
  re-deriving the battery numbers.
- The exact date/session in which `qwen2.5vl:3b` was last confirmed as
  the local-tier model is not something I re-verified beyond the
  `llm_vision_checkpoint_replay.py` tool docstring's mention of it.
