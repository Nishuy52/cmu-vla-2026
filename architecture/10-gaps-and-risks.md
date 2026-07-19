# 10 - Gaps and risks

The honest divergence chapter: where the IF (instruction-following) scoring
pipeline's own measurement of itself is unreliable, what's actually open on
the GitHub issue tracker as of 19 Jul 2026, and where a contributor should
look before trusting a number or reaching for another rule change.

## ELI10

Most of the team's recent work has been trying to find quick fixes that
would make the robot's instruction-following score go up, using a report
that ranked six candidate fixes by how many extra points each was supposed
to be worth. It turned out the report's ranking was built on a
measurement bug — comparing each stop against the WHOLE planned trip
instead of just the stretch of the trip a stop belongs to — so four of the
"clearly good" fixes evaporated on closer inspection, one genuine fix
already tried made the score go down once integrated with other recent
changes and had to be undone, and the honest conclusion is that the
scoring recipe itself (not the robot's driving) needs a redesign before any
more of these small pushes are worth attempting.

## Divergence 1: the ceiling-diagnosis rule rankings are invalidated, not merely stale

`reports/ceiling_diagnosis_2026-07-19/` (commit `5b5602d`) classified all 37
misses in a 72-leg IF battery into four buckets and ranked six candidate
rule changes (B1, B2, A1, A2, D1, A3) by predicted ceiling lift, arriving
at an "achievable ceiling" of 65/72 = 0.903 if every fixable class were
fixed, against a 35/72 = 0.4861 starting point (the report also flags
`reports/issue59_leg_ceiling.json`'s older 38/72 as stale — it predates
later `core/geometry/toolbox.py` commits).

**That ranking is now known to be built on a measurement artifact.** Issue
#69's investigation (`reports/issue69_findings.md`, merged) traced all 10
legs flagged for rules A1 and A2 and found the classification script
computed, for every miss, the distance from a same-class alternative
instance to the ENTIRE multi-leg GT reference trajectory rather than to the
segment of it that actually corresponds to the leg under test — so an
alternative instance sitting near a LATER leg's own destination is
indistinguishable, by this method, from a genuinely better candidate for
THIS leg. Traced concretely: the resolver's chosen candidate in all 10
cases is objectively correct (closest/farthest by Euclidean distance for
A1; the only candidate that actually passes the `on()`/`with()` clause for
A2); the flagged "better" alternatives cluster at trajectory fraction
0.87-1.0, i.e. they are near the END of the whole route, not near this
particular leg.

**Consequence, per `reports/ceiling_diagnosis_2026-07-19/CAVEAT.md` (commit
`460ea33`, added same day):** every "achievable ceiling" number in
`classification.md` is overstated. A1/A2/A3's predicted lift (11 of the 37
misses) is 0/11 real. D1 (corridor gate-midpoint) turned out to be a
genuine internal-consistency defect and WAS fixed
(`core/geometry/primitives.py::centroid_axis_face_points_2d`, wired into
`core/geometry/toolbox.py::corridor_gate`, commit `8c1e18d` — merged as
`d93d694`), but its predicted 2-leg lift did not materialize: neither
flagged leg crosses the 0.8 m tolerance after the fix (`livingroom_1` moves
1.884 m → 1.908 m from the GT path; `studio` moves negligibly). B1 — the
top-ranked, 10-leg rule — is the one that directly fed issue #67's
tolerance-aware `_nearest_free_goal` guard, which regressed on integrated
`main` (Divergence 2, below). The class-(c) "demonstrator route" hard-floor
count (7 legs) and the raw per-leg distance data in `classification.json`
remain usable; the ranked-fix table around them does not. The caveat is
explicit that the authoritative re-derivation now rests entirely on issue
#70 (Divergence 3).

**Where this leaves #69 itself: CLOSED**, but with a corrected map rather
than the predicted lift — the ceiling is unchanged at 35/72, and battery
metrics are byte-identical before and after
(`reports/gt_battery_baseline_if69/` vs `reports/gt_battery_post69/`, both
`if_rubric=0.200`). Issue #68 (a B2-flavoured stop-point convention issue)
is closed as subsumed by #70's per-leg-kind redesign rather than being
independently fixed.

## Divergence 2: issue #67's fix regressed on integration and was reverted same-day

Commit `1f7f8e8` added a guard to `_nearest_free_goal`
([`src/core/runner/gt_battery.py`](../src/core/runner/gt_battery.py) around
line 586): skip the push-off-footprint step entirely when the anchor's own
footprint half-diagonal is already within `LEG_ARRIVAL_TOL_M` (0.8 m,
[`src/core/groundtruth/scoring.py`](../src/core/groundtruth/scoring.py)
line 1100) — reasoning that such a small footprint's raw centroid is
already provably close to its own boundary, so pushing it is needless
churn. Measured on its own branch base this looked like a clean win:
ordered-leg credit 0.2278 / headline 0.2056 against the pre-fix 0.2222 /
0.2000. A GitHub comment on issue #67 declared it fixed and closed on that
basis.

**It regressed once measured on integrated `main`.** By the time the fix
landed on `main`, `main` had also picked up the #63/#64
`usable_gate_point` call-site (`cfc8f4b`) at a different corridor-gate
code path. The two changes interact: re-measured on integrated `main`
(`reports/gt_battery_post67/`, committed at `cb3ebdd`), the combination
scores 0.1944 / 0.1722 — WORSE than the 0.2222 / 0.2000 baseline, not
better. The issue was reopened with this correction, explicitly noting the
earlier "fixed" comment was posted before the integrated numbers were
actually read. The fix was reverted at commit `bbb66bd` (same day, 19
Jul), restoring `main` to 0.2222 / 0.2000.

**Current code state (post-revert), verified by reading the file today:**
`_nearest_free_goal` has no tolerance-aware skip. It unconditionally pushes
any raw goal that falls inside a footprint+clearance box; issue #66's
directional own-anchor push is the only conditioning present. This matters
for anyone reading the function expecting the #67 reasoning to be live —
it is not, and the reason it is not (an integration regression, not a
correctness objection to the idea) is only recorded on the GitHub issue,
not in the file's own docstring.

**Process lesson applied** (per the issue's own reopening comment): batteries
must be re-measured on the integrated head before a fix is declared closed,
not only against its own branch base. A re-fix is explicitly deferred
behind issue #70's layer-1 redesign — since that redesign rewrites the same
arrival semantics `_nearest_free_goal` implements, the next attempt should
re-evaluate whether the guard is even still needed under the new yardstick,
and if so diagnose the guard × gate-call-site interaction directly (a
per-leg diff of `reports/gt_battery_post6364/` vs `reports/gt_battery_post67/`
is flagged as identifying the roughly 2.4 flipped legs responsible).

## Divergence 3: the layer-1 yardstick redesign gates all further IF tuning (issue #70, OPEN)

The deepest-rooted finding, and the reason Divergences 1 and 2 both
terminate in "wait for #70": the GT demonstrators' own reference
trajectories reach only about half of the derived leg goals within the
asserted 0.8 m tolerance at all — fresh ceiling 35/72, with a sharp
terminal-vs-intermediate split (terminal legs 83% reached vs intermediate
legs 31%, per `reports/issue59_leg_ceiling.json` and
`reports/ceiling_diagnosis_2026-07-19`). That split by itself is documented
separately as issues #61 (structural leg-goal-vs-GT-path mismatch,
concentrated in non-terminal legs — 29/42 non-terminal legs far from the
GT path vs 5/30 terminal) and #62 (28/60 missed legs in an earlier battery
had a plausible goal the DRIVEN trajectory itself never got within 0.8 m
of — a driving/route gap, not a goal-placement one; both still OPEN).

Issue #70's diagnosis is that the yardstick's layer-1 constants and
semantics don't match how demonstrations actually satisfy instructions, and
requires: (a) DERIVING the arrival tolerance from measured constants
(vehicle radius + costmap inflation + grid diagonal/2 + a p95 frame-fit
residual + margin) instead of asserting a flat 0.8 m, with `scoring.py`'s
`LEG_ARRIVAL_TOL_M` and the instruction head's own `ARRIVAL_TOL_M` sharing
ONE derivation (they are documented as intentionally coupled today); (b)
per-leg-kind semantics — stop-legs keep derived-tolerance arrival, but
pass-by/intermediate legs should score by CLOSEST-APPROACH to the
referenced instance (matching how GT demonstrators drove past rather than
stopped at them), not by a fixed-radius arrival check; (c) confirmed 19 Jul
that `docs/challenge_brief.md` is silent on any numeric IF tolerance
(grepped for tolerance/meter/radius in IF context — no hits), so deriving
one is legitimate, with an explicit "the organisers' number wins if they
ever publish one" clause; (d) a mandatory re-run of `core.runner.gt_leg_ceiling`
under the new yardstick, reported before any further tuning proceeds — no
loosening a constant just to inflate the score. Issue #70 subsumes #68.

An in-flight worktree branch, `if70-rubric-layer1`, exists for this work
and is the current owner of the surface (visible via `git branch -a`
locally). Issue #69's `if69-resolver-rules` branch, by contrast, has
already been merged (`d93d694`) and no longer exists as a separate branch —
the workspace's branch convention deletes a branch once merged.

## Divergence 4: corridor-gate geometry has two still-open partial fixes behind it

Issues #63 (hotel_room_2's duplicate GT "bed frame" instance sitting on a
corridor gate midpoint — a real, not stamping-artifact, obstruction) and
#64 (A* midpoint-target grid quantization causing a near-miss on the
mandatory gate-crossing check) are both tracked OPEN on GitHub even though
their underlying mechanism — the shared `usable_gate_point` helper in
[`src/core/geometry/primitives.py`](../src/core/geometry/primitives.py)
(around line 289) — is implemented and wired into both the scoring geometry
(`core.geometry.toolbox.corridor_gate`) and live planning
(`core.nav.planner.plan_through`), commits `08e76ed`, `cfc8f4b`, `b4a11eb`.
The consolidated post-#63/#64 battery
(`reports/`, commit `22100d9`) shows credit 0.2222 / headline 0.2000, with
threading violations (`tv=7`) unchanged — the report's own read is that the
remaining threading-violation count sits BEHIND unreached legs in the
scoring chain (a leg that never gets attempted can't register a threading
violation either way), so this fix's effect is masked until the
extraction/arrival chain (Divergences 1-3) moves. Both issues stay open as
tracking references for that dependency, not because the geometry itself
is unfinished.

## Divergence 5: `docs/architecture.md`'s checkpoint table is close to
## the built system, with one budget-only gap

`docs/architecture.md` §3 (around line 79) specifies five numbered
checkpoints plus an optional self-consistency pass. Checking each against
`src/core/checkpoints/`:

| Design checkpoint | Built as | State |
|---|---|---|
| 1. Parse | `core/parsing/ladder.py` + `core/llm/config.py` tiered parse | implemented |
| 2. Detector-miss recovery | [`core/checkpoints/miss_recovery.py`](../src/core/checkpoints/miss_recovery.py) `CHECKPOINT_NAME = "miss_recovery"` | implemented, contract matches design doc |
| 3. Anchor confirmation (IF) | [`core/checkpoints/anchor_confirm.py`](../src/core/checkpoints/anchor_confirm.py) | implemented, contract matches |
| 4. Pre-answer verification | [`core/checkpoints/verification.py`](../src/core/checkpoints/verification.py) | implemented; ships TWO call shapes (a rich `VerificationOutcome` API and an `as_llm_verify_seam` adapter for today's narrower head contract — see the module's own documented "seam gap") |
| 5. Frontier selection | [`core/checkpoints/frontier_select.py`](../src/core/checkpoints/frontier_select.py) | implemented, contract matches |
| Optional: self-consistency ×3 | — | **budget accounting only** |

The self-consistency pass exists solely as a ledger cap name and number —
`"self_consistency": 2` in
[`src/core/fsm/budget.py`](../src/core/fsm/budget.py) (line 155) and a
matching `cap_self_consistency: int = 2` field in
[`src/core/calibration.py`](../src/core/calibration.py) (line 129). There
is no module under `core/checkpoints/` implementing an actual sample-x3-
and-vote pass, and no caller anywhere in `core/heads/` or `core/fsm/`
consumes `"self_consistency"` beyond carrying the budget number. This is
the one checkpoint in the design table that is scaffolded (a budget slot
reserved for it) but not built — worth flagging before assuming the
optional pass ever fires, since nothing currently spends the 0-2 calls it
budgets for.

## Confirmed correct (not a gap)

Two divergences documented in the archived (11 Jul) snapshot of this
folder no longer hold on current `main` — re-verified by reading the code
today, not carried forward from the old text:

- **Perception is now wired into the ROS adapter.** The archived D1 ("the
  ROS answer path resolves against an empty scene") described
  `adapter_node.py` always constructing an empty `BasicSceneIndex([])`.
  Current `adapter_node.py` (around lines 404-430) conditionally builds a
  real `PerceptionPipeline` when `VLA_DETECTOR=grounding_dino`, and Gate 4
  (`reports/gate4_final/evidence.log`, see
  [09-runtimes-and-deployment.md](09-runtimes-and-deployment.md)) is direct
  evidence of a real, detection-backed marker answer coming out of the
  in-container path. The empty-index path still exists as the
  `VLA_DETECTOR=none` default (a loud `SUBMISSION-BLOCKER` log fires if it
  is ever hit unintentionally) but is no longer the only path.
- **Question type is now reconciled after parse.** The archived D2
  described `self.qtype` being set once at intake and never revisited once
  the real `Plan` landed. Current `core/fsm/controller.py` (around lines
  272-280) re-seeds both `self.qtype` and `self.budget.qtype` from
  `plan.qtype` once the parse checkpoint completes, with a logged
  transition (`f"{old}->{plan_qtype}"`) when they disagree.
- **The stall/replan flag is now consumed.** The archived D5 flagged
  `BreadcrumbFollower.replan_flag` as wired but never read. Current
  `core/heads/instruction.py` (around line 760) checks
  `self._follower.replan_flag` and triggers a replan when set (subject to
  `_can_replan()`).

One archived divergence still holds, confirmed by reading the current
files: the archived D4 ("general exploration bypasses A*/breadcrumbs")
still describes `core/heads/explore_step.py` today — it drives
`ExplorationPolicy` directly and does not call `plan_through` or
`BreadcrumbFollower`; only `core/heads/instruction.py` routes through the
costmap/planner/breadcrumb stack. See
[07-navigation-and-exploration.md](07-navigation-and-exploration.md)'s
"Exploration policy" section for the current-code description.

## Where to change what

| To change... | Edit | Evidence / tests |
|---|---|---|
| The IF arrival-tolerance yardstick itself | `core/groundtruth/scoring.py::LEG_ARRIVAL_TOL_M`, `core/heads/instruction.py::ARRIVAL_TOL_M` (shared derivation, per #70) | `core.runner.gt_leg_ceiling`, `reports/issue59_leg_ceiling.json` |
| Per-leg-kind (stop vs pass-by) scoring semantics | `core/runner/gt_battery.py::_if_rubric_geometry`, `core/groundtruth/scoring.py` | issue #70; the `if70-rubric-layer1` branch |
| `_nearest_free_goal`'s push behaviour | `core/runner/gt_battery.py::_nearest_free_goal` (around line 586) | issue #67 (OPEN, reverted at `bbb66bd`); re-measure on integrated `main`, not a branch base, before closing |
| Corridor-gate geometry / usable crossing | `core/geometry/primitives.py::usable_gate_point`, `core/geometry/toolbox.py::corridor_gate`, `core/nav/planner.py::plan_through` | issues #63, #64 (OPEN, tracking the extraction-chain dependency) |
| Non-terminal leg goal / driven-path gap | `core/heads/instruction.py` (route assembly, `_goto_point`), `core/nav/planner.py::plan_through` | issues #61, #62 (OPEN) |
| Self-consistency (if ever built) | new module under `core/checkpoints/`; wire into `core/fsm/budget.py`'s existing `"self_consistency"` cap | none yet — no existing tests to extend |
| Ceiling-diagnosis methodology itself (whole-trajectory vs per-leg) | `core/runner/gt_leg_ceiling.py`, `core/groundtruth/scoring.py` (the `measure` function's distance computation) | `reports/issue69_findings.md`'s "A1+A2: root cause" section documents the exact bug shape to fix |

## References

- Ceiling diagnosis: `reports/ceiling_diagnosis_2026-07-19/classification.md`,
  `reports/ceiling_diagnosis_2026-07-19/CAVEAT.md` (commits `5b5602d`, `460ea33`)
- `reports/issue69_findings.md` (the whole-trajectory-artifact tracing)
- `reports/gt_battery_post67/`, `reports/gt_battery_post6364/`, `reports/gt_battery_post69/`,
  `reports/gt_battery_baseline_if69/`
- GitHub issues #61, #62, #63, #64, #67 (all OPEN), #68 (CLOSED, subsumed), #69 (CLOSED),
  #70 (OPEN, gating) — `gh issue view <N>`
- Commits: `1f7f8e8` (#67 fix), `bbb66bd` (#67 revert), `cb3ebdd` (post-#67 evidence),
  `8c1e18d`/`d93d694` (#69 D1 fix + merge), `460ea33` (#69 caveat), `5b5602d` (ceiling diagnosis)
- [`src/core/checkpoints/`](../src/core/checkpoints/), [`src/core/fsm/budget.py`](../src/core/fsm/budget.py),
  [`src/core/calibration.py`](../src/core/calibration.py)
- `LOG.md` session 18 entries (T16 if53, T17 if66, T18 opened) for the narrative timeline
- Sibling docs: [07-navigation-and-exploration.md](07-navigation-and-exploration.md),
  [09-runtimes-and-deployment.md](09-runtimes-and-deployment.md)
