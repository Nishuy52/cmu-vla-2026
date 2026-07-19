# Issue #77d — the two remaining non-structural mechanism families from #77c

Branch `if77d-joint-legs`, worktree `.claude/worktrees/if77d-joint-legs`.

## Baseline

`reports/gt_battery_main_post77c` was absent (77c's fix1 is already merged
into `main` at `ca8fae7`) — regenerated from a clean checkout of `main`
(`--groundtruth ../data/vla3d/Unity`, default questions/answers). Matches
the brief's expected numbers exactly:

| metric | value |
|---|---|
| `mean_ordered_leg_credit` | **0.6333** |
| `mean_rubric_score` (headline) | **0.6111** |
| `total_threading_violations` | **3** |
| `total_avoid_violations` | **0** |

No `src/` code was changed to reproduce this — the merge already contains
77c's fix. This is the baseline both families below are measured against.

## Family A — the 3 never-approached threading roots, leg0 traced individually

Per-leg0 diagnostic scripts:
`reports/issue77d_notes_work/probe_leg0.py` (goal-resolution summary: candidate
anchor, resolved goal, rubric goal, reachability at grounding time) and
`reports/issue77d_notes_work/probe_leg0_detail.py` (reachable-pocket bounding
box + nearest-reachable-cell distance, and a repeat at `max_build_ticks` =
12/30/60/120 to separate "not enough exploration yet" from "genuinely
disconnected from this vantage").

Both scripts reuse `core.runner.gt_battery`'s own scene-loading/frame-fit/
wall-derivation helpers verbatim (no groundtruth/toolbox/llm code touched) so
the reproduced state exactly matches what the battery run sees.

### `arabic_room` leg0 (q1 "First, go to the potted plant furthest from the
hookah…", leg0 = goto potted-plant)

- Anchor resolution is CORRECT: object CSV confirms `potted_plant` id 0 (at
  `(-3.75, 2.00)`) is genuinely the furthest of arabic_room's 5 potted-plant
  instances from the hookah (`(1.49,-0.59)`) — Euclidean distance 5.85 m vs.
  the next-furthest candidate's 5.16 m. Instance selection is not the bug.
- The anchor cell is **not reachable** in the plain BFS from the grounding
  pose: `reachable_mask` pocket = 536 cells, bounding box
  `x∈[2.35,5.05] y∈[-3.45,2.05]` — entirely on the OPPOSITE side of the room
  from the anchor (`x=-3.75`). `nearest_reachable_point` therefore snaps to
  `(3.55, 1.35)`, 7.33 m from the true anchor, which is exactly the ~6.4 m
  final excess reported.
- Re-ran the reachable-mask probe at `max_build_ticks` = 12 (production),
  30, 60, 120: **the pocket is byte-identical at every tick count** (same
  536 cells, same bounding box, same spawn pose). This rules out "not
  enough exploration ticks yet" — grounding happens from a **stationary**
  pose (`_run_instruction_head`'s own docstring: "12 ticks … from a
  *stationary* spawn"), so more ticks cannot reveal more of the room; the
  536-cell pocket is everything visible/reachable from that ONE vantage,
  full stop.
- 9420 wall cells ARE derived for this scene (unlike the other two below) —
  so this is a REAL wall blocking line-of-sight/reachability from the
  spawn's fixed vantage, not a missing-costmap-truth scene.
- **Root cause, precisely as the task brief hypothesized**: this is the
  "pocket-clamp variant the 77c widen doesn't cover" — #77c/#78/#79's
  pinch-relax widen mechanism requires a **gate** (a `corridor_between`
  leg's own gate segment) to retry `nearest_reachable_point` THROUGH. Leg0
  has no predecessor leg, hence no gate, hence no directional cue for any
  widen to retry through — there is nothing to pinch relax across. The
  mechanism family that fixed the 6 gate-adjacent blast-radius legs in #77c
  structurally cannot apply here.
- **Adjudicated structural, not attempted.** A real fix would require the
  robot to physically move/explore BEFORE committing to leg0's goal (so a
  later vantage sees around the wall) — but grounding is deliberately
  single-pass-from-a-stationary-pose for every leg of every question in the
  battery; changing that timing is a materially bigger, cross-cutting
  change to the whole grounding pipeline (every leg of every scene, not
  bounded to this one), and risks its own new failure modes (moving before
  fully understanding the instruction). Out of this session's bounded
  scope; flagged as a candidate for its own issue if revisited.

### `home_building_1` leg0 (q1 "First, go to the nightstand with a clock on
it…", leg0 = goto nightstand)

- `wall_cells=None` — this IS one of the wall-unavailable scenes (no
  `traversable_area.ply`-derived interior walls; matches #76/#77's own
  finding, `docs`/`issue76_notes.md`).
- Goal resolution itself is fine: `nearest_reachable_point` on the anchor is
  only **0.92 m** off (the huge open, wall-less mirror costmap makes almost
  everything "reachable" in the BFS sense), and the resolved goal is only
  **1.60 m** from the rubric's exact target point.
- The reported 9.65 m excess is **entirely a driven-path deviation**, not a
  goal-resolution defect: `frechet_m` for this question is **9.77 m**
  (12.9 m for the coffee-table-kettle question) — the ACTUAL driven route's
  *shape* diverges wildly from the GT reference path, consistent with a
  costmap that has no interior walls to route around: the planner/follower
  has no notion of the real room boundaries this open ~55×59 m mirror
  costmap should have, so its efficient-looking path bears no resemblance
  to the GT path a wall-aware human/GT planner would take.
- **Adjudicated structural** (costmap-truth-limited, matches the task
  brief's own escape hatch and precedent: #76's own investigated-negative
  for this exact scene's wall-derivation fallback). Not a goal-resolution
  defect and not fixable at the threading/goal-resolution mechanism level —
  fixing it would mean deriving walls for a scene that ships no
  `traversable_area.ply` truth, which is out of scope here (frozen
  `core/groundtruth`) and already tracked as its own pre-existing
  "boundary-only-costmap" mechanism.

### `livingroom_1` leg0 (q1 "First, go near the lamp closest to the black
chair…", leg0 = goto lamp)

- `wall_cells=None` — also wall-unavailable, same mechanism as
  `home_building_1`.
- Goal resolution is essentially fine here too: nearest-reachable-point is
  only **1.10 m** from the anchor, and the actual leg0 excess is a hair
  over tolerance — **0.039 m** (dist_driven 2.1109 vs tol 2.0722) — this leg
  is *barely* missing, well within tolerance-noise territory, not a
  gross miss.
- **Adjudicated structural**, same wall-unavailable mechanism as
  `home_building_1`, just at a far smaller margin. Not worth a bespoke fix
  at this margin; any generic fix for the wall-unavailable mechanism (out
  of this issue's scope) would very likely clear it as a side effect.

### Family A summary

All 3 confirmed **not** a new/uncaught threading-mechanism defect —
distinct root causes, cleanly separated by evidence:

| scene | mechanism | evidence | adjudication |
|---|---|---|---|
| `arabic_room` | real-wall pocket-clamp at a **stationary** grounding pose; no predecessor gate for the existing widen to retry through | pocket byte-identical across 12/30/60/120 ticks; 9420 real wall cells; anchor 7.3 m outside the pocket's bbox | structural, not attempted (fix needs pre-grounding movement — cross-cutting, out of bounded scope) |
| `home_building_1` | wall-unavailable costmap; goal resolution fine (0.92 m off), driven-PATH deviates (frechet 9.77–12.9 m) | `wall_cells=None`; matches #76's own investigated-negative for this scene | structural (precedent + this session's reconfirmation) |
| `livingroom_1` | wall-unavailable costmap, but only a hair over tolerance | `wall_cells=None`; leg0 excess 0.039 m | structural (same mechanism, negligible margin) |

No structural-ledger table entry is added for these 3 — they were already
counted in #77c's Phase 1c narrative as `conversion_probe_v2`'s pre-existing
(d) bucket, this session only sharpened WHY (2 distinct mechanisms) with
direct evidence rather than adding new ledger rows.

## Family B — home_building_2's corridor/goto joint trade-off

### The trade-off, reconfirmed

`home_building_2` q1 ("Take the path between the sofa and the coffee table
and go to the kettle…"): leg0 = `corridor_between` (currently
`threaded=True`, earned), leg1 = `goto` immediately after (currently
`reached_in_order=False`, dist_driven 2.177 m vs tol 1.949 m — a 0.23 m
miss). #77c's pinch-relax widen for leg1 finds a narrower/truer candidate
goal, but the existing one-sided validation guard
(`_keeps_corridor_threaded` in `_goto_point_pinch_relax`, calling
`_gate_crossing_extension`/`_gate_extension_keeps_route_planned` — the same
functions `plan_through` itself uses) correctly refuses it: **no
`GATE_CROSSING_MARGINS_M` extension past the gate keeps the corridor's own
crossing reachable to the narrower goal**, so the guard keeps the current
(wider, non-regressing) goal — net: leg1 stays unfixed, leg0 stays earned,
no regression. This matches #77c's own finding exactly.

### New evidence this session — the crossing-point search space is narrow

`reports/issue77d_notes_work/probe_hb2_joint.py` swept 21 points along the
FULL gate segment (not just `_gate_crossing_extension`'s small
push-past-the-line margin) and tested raw-obstacle blocking at each:

- Only the first ~15% of the gate segment (t∈[0.02,0.16], roughly 0.15 m of
  the ~0.92 m line) is even **unblocked** raw floor — the remaining ~85% of
  the segment is solid obstacle (the sofa/coffee-table furniture the
  corridor threads between). `usable_gate_point`'s job of sliding the
  crossing point to unblocked floor is doing real, necessary work here, not
  picking arbitrarily among a wide-open line.

This matters for the joint-resolution design below: the "vary WHERE along
the gate to cross, conditional on the next leg's goal" search space this
specific gate offers is small by construction (furniture leaves a narrow
usable band) — so a general joint mechanism may simply find nothing better
here even once built. The value case for building it is generalization to
OTHER corridor/goto pairs with a wider usable crossing band, not a
guaranteed win on this specific leg.

### Design: order-independent (joint) goal resolution

**Problem statement.** Grounding is currently strictly sequential and
one-pass (`_ground_legs`): each leg's goal is committed in route order and
never revisited once a later leg's true goal becomes known. For a
`corridor_between` leg immediately followed by a `goto`/`via_near` leg, the
corridor's crossing point (`usable_gate_point`, chosen from the raw gate
midpoint) is fixed before the goto leg's own goal is even resolved. #77c's
fix widened the GOTO leg's resolution to retry through the ALREADY-FIXED
gate, and added a one-sided guard that keeps the corridor's OWN threading
safe — but it can only accept or reject candidates against a crossing point
that was chosen without ever knowing what the goto leg would need.

**Design goal.** Let the corridor leg's own crossing-point choice be
informed by the (candidate) next leg's goal, instead of being fixed first
and validated second. Candidate approach, per the task brief's steer:

1. At `_ground_one` time for a `corridor_between` leg, do NOT commit
   `usable_gate_point`'s crossing point immediately. Instead, resolve the
   IMMEDIATELY-FOLLOWING leg's anchor (if it exists and is `goto`/
   `via_near`) far enough to get its centroid + a *ranked list* of
   candidate goal points: the plain `nearest_reachable_point` result AND
   each pinch-relax round's candidate (the same list `_goto_point_pinch_relax`
   already computes internally, currently thrown away except for the single
   winner).
2. For each candidate crossing point along the gate line that
   `usable_gate_point`'s existing raw-obstacle-avoidance logic would accept
   (a small, already-bounded set — the corridor is narrow by construction,
   as this session's sweep confirms), and for each of the next leg's
   candidate goals (widest/truest first), check reachability BOTH ways:
   `cur -> crossing_point` (this leg's own approach) and
   `crossing_point -> next_goal` (re-using `_gate_extension_keeps_route_planned`'s
   existing A*/pinch-retry logic, unchanged).
3. Accept the FIRST (crossing_point, next_goal) pair, preferring: (a) the
   original raw-midpoint-nudged crossing point with the TRUEST next_goal
   that still validates (preserves today's behavior/answer whenever the
   original crossing point already works, so scenes that already pass are
   byte-for-byte unaffected); falling back to (b) a different crossing
   point along the gate ONLY when no `(original_crossing_point, any
   next_goal candidate)` pair validates, iterating crossing points from
   nearest-to-midpoint outward.
4. Thread BOTH the chosen crossing point (feeding the corridor leg's own
   `geom`) and the chosen next-leg goal (feeding the next leg's own `geom`)
   out of `_ground_one`/`_ground_legs` together — this is the "joint" part:
   a single decision commits both legs' geometry atomically, instead of
   corridor-first-then-goto-validates-against-it.

**Why this is bounded but non-trivial.** `_gate_extension_keeps_route_planned`
already does the one-leg lookahead this needs (point 2 above reuses it
verbatim, no new duplicated reachability logic — matches the codebase's own
discipline of never re-implementing planner semantics in `heads/`). The
NEW work is: (a) exposing the goto leg's candidate LIST (not just its
single winner) up to the corridor leg's own grounding step, which currently
happens strictly BEFORE the goto leg is grounded at all (`_ground_legs`'
single forward pass) — this needs either a second (lookahead) resolve pass
before committing legs, or restructuring `_ground_legs` into a two-phase
"propose candidates, then jointly commit" loop; and (b) letting
`usable_gate_point`'s crossing-point choice vary (today it is a single
deterministic nudge-to-nearest-unblocked-point call, not a search over
multiple candidates) — a new, small enumeration wrapper around it, not a
change to `usable_gate_point`'s own contract.

**Risk and blast radius.** `_ground_legs`/`_ground_one` restructuring
touches the SAME per-leg grounding path every scene's every leg goes
through (not just corridor-then-goto pairs) — a bug here has broad blast
radius, and any change to `plan_through`'s shared tail (even read-only
reuse of its guard functions from a new call site) requires the full gate
per this session's own rules. Given the measured expected gain is at most
this ONE leg's flip (and even that is not guaranteed — the crossing-point
sweep shows this specific gate's usable band is narrow, so it may still
resolve to "no improvement" after all the new machinery is built) against
a two-phase grounding restructure that's easy to get subtly wrong (e.g.
first-phase candidate resolution seeing a different, not-yet-updated
costmap state than the second commit phase), this is scoped as a **design
document, not an implementation**, per the task brief's explicit
escape hatch ("if the clean design exceeds your budget, deliver the design
document... instead of a rushed implementation").

**Recommended next step if resumed**: implement point 1-2 ONLY as a
read-only probe first (compute what the joint choice WOULD be, log it,
don't act on it) across the full battery, to measure how many corridor/goto
pairs beyond `home_building_2` would actually change verdict — if the
answer is "just this one, and it still doesn't flip," the two-phase
grounding restructure isn't worth its risk; if several scenes would
benefit, that changes the cost/benefit calculus enough to justify the
larger change.

### Family B outcome

**Design document delivered (this section), not implemented.** No `src/`
change for Family B this session. `home_building_2`'s leg1 stays exactly as
#77c left it (guarded off, no regression, no improvement).

## Integrated numbers (final, this session)

No `src/` code changed this session (Family A: adjudicated structural after
evidence-based tracing, no fix attempted; Family B: design doc only, no
implementation) — battery numbers are unchanged from the regenerated
baseline:

| metric | value |
|---|---|
| `mean_ordered_leg_credit` | 0.6333 |
| `mean_rubric_score` (headline) | 0.6111 |
| `total_threading_violations` | 3 |
| `total_avoid_violations` | 0 |

Short of the 0.65 stop target. Stopping per the rules: this session made no
code change (nothing to "keep only if improving" — there is no regression
risk either), the remaining bucket at these 3 threading roots is
adjudicated structural with fresh, distinguishing evidence, and Family B's
clean design exceeded a bounded-session implementation budget as
anticipated by the brief's own escape hatch.

## Structural ledger — no new rows added

Family A's 3 legs were already implicitly counted in #77c's Phase 1c
narrative (not previously given their own ledger table rows since #77c
treated all 3 as one undifferentiated "conversion-gated, unowned" bucket).
This session's contribution is evidentiary, not a new adjudication — no new
row added to #77c's "Structural ledger (final count)" table since that
table's scope was explicitly "disconnected mirror-world geometry /
mesh-coverage gaps / frame-fit defects", and 2 of these 3 (`home_building_1`,
`livingroom_1`) are wall-unavailable-costmap (a distinct, already-referenced
mechanism, not a new ledger category) while `arabic_room` is a newly
characterized mechanism (stationary-grounding pocket-clamp) that could get
its own ledger row in a future pass if the team wants it tracked
explicitly; left to the maintainer's judgment rather than added
unilaterally here since #77c's ledger table has a specific, curated scope.

## Files

- `reports/issue77d_notes_work/probe_leg0.py` — Family A leg0
  goal-resolution probe (candidate/anchor/rubric-goal/reachability summary).
- `reports/issue77d_notes_work/probe_leg0_detail.py` — Family A reachable-pocket
  bounding box + nearest-reachable-cell distance, tick-count sweep.
- `reports/issue77d_notes_work/probe_hb2_joint.py` — Family B gate-segment
  crossing-point sweep (evidence for the design doc's narrow-search-space
  finding).
- `reports/gt_battery_main_post77c/` — regenerated baseline (matches the
  brief's expected 0.6333/0.6111/3 exactly).

## Tests

No `src/` code changed this session — fast tier and full gate were not
re-run (nothing to verify; diagnostic scripts under `reports/` are outside
`src/`'s test surface and were run standalone, output inspected directly
above).
