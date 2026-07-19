# 6. Answer Heads

How a validated `Plan` becomes the three kinds of answer the challenge
scores: a count, a marker box, or a driven route. One head per `qtype`,
all resolving against the same `SceneIndex` and the same geometry
predicate toolbox.

## ELI10

Three specialists share one filing cabinet of "things I've seen" (the
`SceneIndex`). The counting specialist just counts matching drawers.
The pointing specialist ranks drawers by how well they match the
question and points at the best one — demoting to the runner-up if a
checkpoint says the top pick is wrong. The walking specialist turns a
list of waypoints into an actual driven path, replanning around no-go
zones. All three ask the same "does drawer A sit on drawer B?"-style
predicate module for their spatial reasoning; they never invent their
own geometry.

## Numerical head

[src/core/heads/numerical.py](../src/core/heads/numerical.py) —
`NumericalHead` (around line 49). The parsed `TargetSpec` IS the filter
(no LLM re-invoked per tick); `advance(scene)` (around line 79) recomputes
a deterministic count via `core.geometry.toolbox.counting` every tick.
Two gates before a count is reported as "stable" enough for the FSM to
answer early:

- **Count stability** (module docstring, `STABLE_TICKS = 3`): the same
  count must hold for 3 consecutive `advance()` calls AND every
  contributing instance must have `n_obs >= MIN_OBS(3)`, translated
  into the FSM's `StabilitySignal` (`signal()`, around line 131) —
  `winner_margin` reports `0.30` (above the FSM's `>= 0.25` early-fire
  gate) only once held long enough, `0.0` before.
- **H15(b) observation gating** (`_answer_min_obs`, around line 104):
  cold start (no instance of the queried noun yet seen
  `ESTABLISH_N_OBS(3)` times) counts everything so a genuine cold count
  isn't starved; once the noun class is "established," one-frame ghosts
  (`n_obs < GATE_MIN_OBS(2)`) are dropped from the count.
- **Empty-index withholding** (`advance`, around line 84): a completely
  empty scene index is treated as absence of data, not an observation of
  zero — `count` is set to `None` and `answer()` (around line 148)
  returns `None` so the FSM's floor (modal-count fallback) answers
  instead of a fabricated 0. A genuine zero (index non-empty, target
  noun just absent) still counts normally.

## Object-reference head

[src/core/heads/object_ref.py](../src/core/heads/object_ref.py) —
`ObjectRefHead` (around line 63). `advance(scene)` (around line 105)
calls `core.geometry.toolbox.resolve` every tick and keeps the top-ranked
candidate as `best_marker`, published continuously into
`PartialResults` so the floor always has the freshest box even before
`verify()` runs.

`verify()` (around line 124) is the final-answer path and applies an
optional injected verification checkpoint (CP4):

- **Rich seam** (`verifier`, `_cp4_winner` around line 169): consulted
  with the winner, runner-up, per-clause pass matrix, `plan.notes`, and
  a `resolve_again` re-resolve hook; returns a duck-typed outcome with
  `.action` in `{"keep","runner_up","re_resolve"}`. `re_resolve`
  (`_make_resolve_again`, around line 209) synthesizes a real `Clause`
  from the verifier's `missed_constraint` string and re-runs `resolve`
  with it appended — either a filtering clause or a superlative can
  change the winner.
- **Legacy narrow seam** (`llm_verify`, `_verified_winner` around line
  262): `(plan, candidate_summary, pass_matrix_text) -> bool`; walks
  ranked candidates in order, keeping the first one the checkpoint
  approves.
- Either seam missing/raising defaults to trusting the deterministic
  rank — "a dark/broken checkpoint trusts the deterministic rank"
  (comment at line 201).

Before publishing, `_first_committable` (around line 154) applies the
**OR-F8 provisional-commit guard**: a winner with fewer than
`REOBS_MIN_N_OBS(2)` observations (e.g. a single-sighting hallucination-
recovery instance) is never published directly — the head walks to the
next ranked candidate that passes `core.perception.detector.
is_answer_eligible` (issue #43a's n_obs-plus-peak-score gate), or
publishes nothing (`None`) rather than a confident wrong box.

## Instruction-following head

[src/core/heads/instruction.py](../src/core/heads/instruction.py) — the
70.6%-of-points machinery (module docstring, line 1), and the deepest
per-question pipeline:

1. **Ground anchors** — each `RouteLeg`'s anchors resolve via
   `core.geometry.toolbox.resolve` (same resolver the OR head uses).
2. **Leg geometry** — `goto` → anchor centroid projected to free space;
   `via_near` → a standoff point near the anchor (`VIA_NEAR_CLEARANCE_M
   = 0.45` m, kept below the leg's own `ARRIVAL_TOL_M = 0.8` m so a
   compact anchor's standoff point still lands inside the credited
   arrival band — see the comment at
   [instruction.py](../src/core/heads/instruction.py) around line 52);
   `corridor_between` → `toolbox.corridor_gate`.
3. **Avoid capsules** — stamped once per question into a cloned
   `Costmap`, hard and never relaxed.
4. **Route** — `core.nav.plan_through` over the costmap drives a
   `BreadcrumbFollower` streaming near-vehicle waypoints at 5 Hz.
5. **Interleaved explore-execute** — while a later leg's anchor is still
   ungrounded, exploration affinity biases toward its noun while the
   robot keeps driving the current leg.

Two committal/replanning guards worth naming: **issue #33's route-
prefix commitment floor** (`MIN_COMMIT_OBS = 2`, around line 69) — a leg
backed by only one observation is planned but withheld from the
committed drive until it gathers a second observation or the
forced-assembly time-pressure gate is reached; and **`MAX_REPLANS_PER_
QUESTION = 3`** (around line 74) — caps re-plans per question so a
pathological stall/violation loop can't burn the whole time budget.

`terminal_waypoint()` returns the `WaypointCmd` at the terminal goal —
that IS the published answer for instruction-following; there is no
separate "final answer" object distinct from the drive itself.

### Removed: unconsumed instruction-affinity API (closes #50, commit 2c9f704)

Three methods were deleted outright, not deprecated: `avoid_nouns()`,
`next_noun_affinity_target()`, and `ungrounded_nouns()` (all previously
in `InstructionHead`). They existed to bias frontier exploration toward
route/avoid anchor nouns, but nothing in the FSM or explore head ever
called them — `git show 2c9f704` is a pure deletion (72 lines removed
across `instruction.py` and two test files, zero lines added). The
comment trail in the removed code (tagged `IF-F5`, `H3b`) shows the
intent was real, just never wired up; the fix was to remove dead
surface area rather than leave an API nobody exercises.

## The resolver (shared, lives in geometry, not heads)

Both `ObjectRefHead.advance` and `InstructionHead`'s leg-grounding step
call the same function: `core.geometry.toolbox.resolve`
([src/core/geometry/toolbox.py](../src/core/geometry/toolbox.py) around
line 906). Pipeline: hard-filter candidates by noun (typo-tolerant) and
attributes, AND together every non-superlative clause, then rank
survivors by any superlative clause (`closest_to`/`farthest_from`)
rather than filtering on it. An empty filtered set triggers a three-rung
fallback ladder, each rung audited (`Relaxation` entries, around line
919):

1. **relax_attributes** — drop attribute filters, keep relation clauses.
2. **drop_relation** — drop the single least-selective (highest pass
   fraction) relation clause at a time, while at least one relation
   still constrains the pool.
3. **category_only** — drop every relation clause; the pool becomes
   every same-noun instance.

### The category-only fallback bug (#59, commit e85a0e5)

Fixed in [src/core/geometry/toolbox.py](../src/core/geometry/toolbox.py):
when the fallback ladder reaches `category_only` because no candidate
passes a relation as a **hard** filter, ranking previously fell through
to `_tier_priority_order` — effectively instance-id order, with no
relation to the dropped clause at all. Root cause traced to `on()`
(around line 328): its soft ranking `score` was gated on the same
`anchor_larger` hard-pass condition as the boolean `passed` result, so
a near-miss (e.g. a wide-canopy potted plant whose footprint narrowly
fails the "anchor strictly bigger" sanity check against a small side
table, despite strong overlap and correct height) scored `0.0` on the
ranking signal too — indistinguishable from a candidate nowhere near
any table. The fix decouples them: `score` is now gated on vertical
alignment (`vert_ok`) only, not `anchor_larger` (comment at
[toolbox.py](../src/core/geometry/toolbox.py) around line 360) — the
size gate still blocks hard misclassifications ("a sofa on a cushion")
from *passing*, but no longer zeroes every near-miss's *ranking* signal.
`resolve()` then uses `_relaxed_relation_order` (around line 1020) to
rank the category-only pool by best-effort match (summed soft
`PredResult.score` per originally-stated clause) instead of the
arbitrary id tie-break, whenever every hard clause was dropped and more
than one candidate survives (around line 986). Evidence:
`reports/issue59_probe.md`, produced by a per-leg diagnostic probe added
to `gt_battery.py` for this investigation (see chapter 8).

## How heads interact with geometry predicates

Neither answer head implements its own spatial reasoning. `Pred` values
(`on`, `in`, `near`, `next_to`, `between`, `above`, `under`,
`closest_to`, `farthest_from`, `with`) are evaluated exclusively by
functions in `core.geometry.toolbox` (`on`, `in_`, `near`, ... each
returning a `PredResult` with a hard `passed` bool, a soft `[0,1]`
`score`, a signed `margin`, and a human-readable `explanation` string).
Both heads consume `ResolveResult.pass_matrix` — the per-candidate,
per-clause `PredResult` rows — for their verification/explanation
surfaces (`_pass_matrix_text` in `object_ref.py`, the leg-probe tooling
in chapter 8's `gt_battery.py`). This means every spatial-reasoning bug
fix (like #59's `on()` scoring decoupling) is a single-module change
that both heads inherit for free, and it's why the resolver lives in
`core/geometry/` rather than being duplicated per head.

## What I could not verify

- I did not re-run the `qwen2.5vl:3b`-driven CP4 verifier live in this
  session; its behavior is read from `object_ref.py`'s code and
  docstrings, not from an executed checkpoint call.
- The exact numeric impact of the #59 fix on the battery's ordered-leg
  credit (beyond the qualitative finding "7/42 checked non-terminal
  misses had a same-class alternative that would have landed within
  tolerance") is reported in `reports/issue59_probe.md`, which I read
  only via the commit message summary, not the full report file.
