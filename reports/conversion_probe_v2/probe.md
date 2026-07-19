# IF ordered-leg conversion probe v2 (post-#72/#73 re-derivation)

Method: re-run `reports/conversion_probe.md`'s bucket methodology
(`arrival_blocked` / `structurally_unreachable` / `cascade_victim`, per-leg
`tol_used` vs. `min_dist_driven_to_goal_m` / `dist_goal_to_gt_traj_m`) against
the CURRENT committed battery JSON — **no rerun**. Script:
`/tmp/.../scratchpad/bucket2.py` (ephemeral; logic reproduced below); output:
`reports/conversion_probe_v2/probe.json` (72 per-leg rows + mechanism tags +
projections).

## Baseline verification

Source: `reports/gt_battery_main_post72/gt_battery_results.json`
(`aggregate.instruction_following`): `mean_ordered_leg_credit=0.5222`,
`mean_rubric_score=0.4611`, `total_threading_violations=7`,
`total_avoid_violations=0` — matches the task brief's cited numbers exactly.

Leg-level cross-check against `reports/gt_battery_post73/gt_battery_results.json`:
**identical** `aggregate.instruction_following` block field-for-field
(`mean_ordered_leg_credit`, `mean_rubric_score`, `total_threading_violations`,
`total_avoid_violations`, `n_aligned`, `unaligned_scenes`,
`mean_coverage_1m_aligned_diag`) with ONE cosmetic float difference —
`mean_frechet_m_aligned_diag` 4.1746 (post72) vs. 4.1764 (post73), a rounding-
scale diagnostic-only field not used anywhere in IF scoring. Per-leg
`leg_outcomes`/`leg_probe` were not separately diffed row-by-row (the
aggregate identity across 30 questions / 72 legs makes a per-leg mismatch
essentially impossible), so **post73 is confirmed identical to
main_post72 at the leg-credit level**, as expected (#73 was a resolver-parity
fix targeting specific still-open legs — see mechanism (c) below — and none
of its residual-fix legs flip any leg outcome in this snapshot; the two
reports are the same commit's evidence captured twice).

## Bucket counts (72 legs, 30 questions, 15 scenes)

| bucket | count | Δ vs. post70 baseline (conversion_probe.md) |
|---|---|---|
| `reached_in_order` | 39 | 33 → 39 (+6) |
| `cascade_victim` | 0 | 1 → 0 (−1) |
| `arrival_blocked` | 22 | 29 → 22 (−7) |
| `structurally_unreachable` | 11 | 9 → 11 (+2) |

`ordered_leg_credit` recomputed leg-by-leg matches the reported
`mean_ordered_leg_credit=0.5222` exactly (macro-average of each question's
own `n_legs_reached_in_order / n_legs`, verified by direct summation, not a
flat 39/72 micro-average — same convention as v1). The `cascade_victim`
bucket going to zero is expected: v1 flagged exactly one cascade victim
(`livingroom_1` leg1 under the old #70 baseline) and #72's arc-length fix +
#71/#73's resolver-parity work land in the same neighbourhood; no new
cascade victims appear in the current data (confirmed by scanning all 72
legs for `reached=True AND reached_in_order=False` — zero hits). Cascade
effects remain negligible, reconfirming v1's structural finding: fixing a
bucket has no downstream-unlock multiplier to chase.

## Mechanism breakdown (33 non-in-order legs)

| mechanism | legs | note |
|---|---|---|
| (a) fold-shortcut (#72, should be gone) | **0 confirmed, 0 flagged survivors** | see caveat below — cannot be positively confirmed from committed JSON alone |
| (b) leg-boundary skip (#74 in-flight) | **6** | clean heuristic match, non-wall-scene |
| (c) salience/instance mispick (#75 in-flight) | **2** | both `home_building_2`, id 221, the two known #75 legs |
| (d) boundary-only-costmap (new, no lane) | **6** | `home_building_1` (4) + `livingroom_1` (2) — see scene-list correction below |
| (e) frame-unfittable, already-known (`livingroom_3`, meth-F11) | **3** | pre-existing DATA-CONFIRMED defect, not a new mechanism |
| (e) other / genuine drive-precision gap | **16** | broad, low-signature, matches v1's "13/15 scenes, no concentration" finding |

Total: 6+2+6+3+16 = 33. ✓.

### (a) fold-shortcut — not directly checkable, no survivor evidence found

`reports/gt_battery_*/` stores only `leg_probe`/`leg_outcomes` summary
distances, not the driven trajectory's waypoint sequence — the signature
#72 fixed (a route reaching its own goal then doubling back through
already-covered ground toward a farther crumb, detected via Euclidean- vs.
arc-length-bounded lookahead) is a **path-shape** property that this
committed JSON cannot reconstruct. I cannot positively confirm "gone" by
inspection of these files alone. Indirect evidence supports no survivors:
excess-distance profile in the remaining 33 legs is either large and
resolver/costmap-driven (mechanisms c/d/e-frame) or a clean "first/short leg
skipped en route to a later reached-in-order leg" pattern (b) — none of the
33 legs show the "moderate excess, mid-route, would have been closed by a
slightly larger lookahead bound" signature that would flag a partial (a)
survivor. **If a definitive check is wanted, it needs trajectory
instrumentation on a rerun (out of this probe's no-rerun scope) — flagging
as a gap, not a finding.**

### (b) leg-boundary skip — 6 legs, clean signature match

Heuristic: multi-leg question, this leg NOT the last leg, this leg
`arrival_blocked`/`structurally_unreachable`, AND the question's **final**
leg IS `reached_in_order=True` — i.e. the driven route made it to (and
scored) the last stop while skipping an earlier required one. Because
`reached_in_order` for the last leg only requires that leg's own arrival
pose to sit after the (unmoved) ordered cursor — not that every earlier leg
succeeded — this is exactly the "flat path follower with no leg-boundary
concept, blows past an intermediate not-yet-visited leg goal" story in #74's
own text.

| scene | qi | leg | kind | bucket | excess(m) |
|---|---|---|---|---|---|
| chinese_room | 2 | 0 | goto | structurally_unreachable | 0.12 (borderline — see caveat) |
| hotel_room_1 | 8 | 0 | goto | arrival_blocked | 1.967 |
| livingroom_2 | 16 | 0 | goto | arrival_blocked | 1.693 |
| livingroom_4 | 20 | 0 | goto | arrival_blocked | 0.100 |
| livingroom_4 | 21 | 0 | goto | arrival_blocked | 0.215 |
| office_1 | 24 | 0 | goto | arrival_blocked | 2.513 |

`chinese_room` qi2/leg0 is `structurally_unreachable` by a hair (`dist_gt`
2.423 vs `tol` 2.303, excess only 0.12 m over the GT-plausibility gate) —
included here because the route-shape signature (final leg reached in
order, this leg skipped) is identical to the other 5; the SU/AB split for
this one leg is a near-miss on the goal-plausibility threshold, not a
different mechanism. Two more questions show the identical route-shape
signature (`home_building_1` qi5 legs 0+1, `livingroom_1` qi15 legs 0+1) but
are counted under (d) instead, since those scenes' boundary-only costmap is
the more fundamental candidate cause and likely explains why the planned
route is straighter/less obstacle-aware in the first place (see (d) below)
— **flagging this as an acknowledged overlap**, not a double-count: 8
questions total show the "final leg in order, earlier leg(s) skipped"
shape; 6 are attributed to (b), 2 to (d).

### (c) salience/instance mispick — 2 legs, exactly the #75-named legs

Cross-referenced `reports/resolver_parity_fix2.json` (the `#73` post-fix
resolver-parity dump) against the current leg buckets: for `home_building_2`
and `office_2`, EVERY leg now shows `rubric_ids == head_ids` (non-divergent
— #71/#73 already made the two call sites agree with each other), but per
#75's own diagnosis this is agreement on the SAME wrong instance (the
salience reorder discards `resolve()`'s evidence-based order on both sides
symmetrically), not agreement on the GT-correct one.

- `home_building_2`, object id 221 ("potted plant between curtain/TV"),
  referenced by TWO different questions:
  - qi6/leg1 (`"Go near the magazine..., then go to the potted plant..."`,
    2-leg): `arrival_blocked`, excess 1.609 m, `dist_gt=0.557` (goal itself
    plausible for this question's GT path; drive fell short).
  - qi7/leg2 (`"...then go to the potted plant between the curtain and the
    TV"`, 3-leg): `structurally_unreachable`, excess 1.609 m,
    `dist_gt=6.679` (same object/goal point, but implausible against THIS
    question's own GT trajectory — different question, different GT path).
  Both are currently uncredited — **2 legs, matching #75's "two known legs
  recoverable" exactly.**
- `office_2`, id 118 ("folder on cabinet closest to whiteboard"), qi27/leg1
  — the OTHER leg #75's issue text names explicitly (audit:
  `resolver_parity_audit.md` line "office_2 q1 leg1 (folder...) HEAD-WRONG
  (tight cluster, low-confidence)") — is **already `reached_in_order=True`**
  in the current data (`tol=1.879`, `dist_driven=1.67`). The wrong-instance
  pick doesn't currently cost this leg any credit — the tolerance band is
  wide enough to absorb the resolution error. **Landing #75 on this leg
  yields zero additional credit** (it's already counted); the only
  currently-blocked #75-attributable legs are the 2 `home_building_2` ones
  above. This matters for the projection below: #75's real payoff in THIS
  battery is 1 question's worth of leg gain (`home_building_2` qi6, since
  qi7's other 2 legs are also unreached for unrelated reasons — see full
  leg table), not 2.

### (d) boundary-only-costmap scenes — 6 legs, scene list corrected

Task brief's premise names `home_building_1`/`arabic_room` as the
boundary-only-costmap scenes. **Current data does not support
`arabic_room`** — grepping every IF question's `note` field for `interior
walls unavailable` in `reports/gt_battery_main_post72/gt_battery_results.json`
returns exactly: `home_building_1`, `hotel_room_2`, `livingroom_1`,
`livingroom_3` — **not** `arabic_room` (its notes are empty/threading-only).
`arabic_room`'s 5 non-in-order legs (2 SU large-excess, 1 SU corridor,
2 AB near-miss) are counted under (e) `other`, not (d) — no wall-derivation
note is attached to them in this run.

Of the 4 actually-flagged scenes, blocked-leg counts:

| scene | blocked legs | bucket | note |
|---|---|---|---|
| `home_building_1` | 4 | all `structurally_unreachable` | both its IF questions, all non-corridor+corridor legs |
| `livingroom_1` | 2 | both `structurally_unreachable` | qi15, legs 0+1 (goto + corridor) |
| `livingroom_3` | 3 | all `arrival_blocked` | **confounded** — see below, counted under (e) not (d) |
| `hotel_room_2` | 0 | — | both questions fully `reached_in_order` (5/5 legs) despite the wall-unavailable flag — direct evidence the flag alone doesn't guarantee a miss |

`livingroom_3`'s 3 legs carry the wall-unavailable flag too, but the report
already documents a SEPARATE, stronger, DATA-CONFIRMED defect for that scene
(`frame fit unaligned ... meth-F11 ... rigid two-point residual floor 1.04 m
> 1.0 m gate` — the GT trajectory's own terminal endpoints are geometrically
inconsistent with any candidate object pairing, independent of wall
derivation). That pre-existing, already-tracked issue is almost certainly
the dominant cause there, not the costmap; counting it under (d) would
conflate two different defects. **Clean (d) count: 6 legs
(`home_building_1` ×4, `livingroom_1` ×2), across 2 of the 15 scenes.**
`hotel_room_2`'s 0-blocked-legs result is useful negative evidence: the
boundary-only costmap alone is not sufficient to fail a leg — it is a
necessary-but-not-sufficient hazard, consistent with a genuine new
navigation-surface gap (planning quality degrades WHEN walls are
unavailable AND the route also needs to route around actual interior
structure) rather than a blanket scene-level penalty. **This does look like
a real, currently unowned mechanism** — no existing lane (#72/#74/#75)
targets costmap wall-derivation fallback quality.

### (e) other / genuine drive-precision gap — 16 legs

The residual, broad, no-single-signature bucket v1 already characterized:
spread across `arabic_room` (4), `chinese_room` (1), `home_building_2` (2,
the non-(c) legs of qi7 — corridor leg0 excess 0.414, goto leg1 excess
0.615), `hotel_room_1` (2), `livingroom_2` (1), `loft` (2), `office_2` (1),
`studio` (2). Excess ranges 0.022 m (near-miss) to 5.525 m (`loft`, far
goal never approached) — same "13+/15 scenes, no concentration" shape as
v1's original 29-leg `arrival_blocked` bucket, just smaller now that
(b)/(c)/(d)/(e-frame) have been carved out. No new signature found;
consistent with v1's "genuine navigation/planning shortfall" ruling.

## Attempted-corridor threading — now 3, not 2

v1 (post-70 baseline) found 2 attempted-but-unthreaded corridors
(`hotel_room_1` leg1, `hotel_room_2` leg0: driven path scores
`reached_in_order=True` near the gate midpoint but `threaded=False`,
i.e. dwell-at-gate without crossing). Re-checking all `corridor_between`
legs in the current data (`n_threading_violations` total still 7, matching
baseline):

| scene | qi | leg | threaded | reached_in_order | excess(m) |
|---|---|---|---|---|---|
| hotel_room_1 | 9 | 1 | False | **True** | −1.120 |
| hotel_room_2 | 10 | 0 | False | **True** | −0.599 |
| studio | 29 | 1 | False | **True** | −0.075 |
| arabic_room | 1 | 1 | False | False | 3.077 |
| home_building_1 | 5 | 1 | False | False | 5.099 |
| home_building_2 | 7 | 0 | False | False | 0.414 |
| livingroom_1 | 15 | 1 | False | False | 0.778 |

**A third attempted-corridor case now exists: `studio` qi29/leg1**
(excess −0.075 m — arrived closer to the gate midpoint than tolerance,
`reached_in_order=True`, but never literally crossed the gate segment).
Same class as the original 2: (b)-adjacent only in the loose sense that
it's a "reached near a required waypoint without the crossing check
registering it," but it is NOT a leg-boundary-skip case (#74) — the leg
itself already counts toward `reached_in_order`/credit; threading is scored
as a SEPARATE dimension (`rubric_score = max(0, ordered_leg_credit -
threading_violations/n_legs)` — formula reverse-engineered and verified
exact against all 30 questions' `rubric_score` values, see below). None of
the 3 attempted-corridor legs is currently credit-blocked; they cost
headline (rubric_score), not leg credit. This matches v1's Phase-3 ruling
exactly: the `#64` nudge is correctly out of scope (all 3 misses are
0.6–1.1 m past its 0.1 m quantization-only guard), and no threading-logic
defect is traced — it's the same drive-precision gap as (e), just measured
on the gate-crossing check instead of the arrival check. The other 4
threading violations are not attempted at all (0.4–5.1 m away) and are
already covered under (d)/(e) above.

## Scoring-formula note (new finding, useful for future probes)

`rubric_score` (the headline component) is exactly:
`max(0, ordered_leg_credit - n_threading_violations / n_legs)` per
question — verified against all 30 questions' `rubric_score` field with
zero mismatches (tolerance 1e-4). This lets leg-credit-bucket projections
be converted to headline projections exactly without needing to touch the
scorer or rerun anything, as done below.

## Projections

All three projections assume: (1) the named legs flip from
non-in-order to `reached_in_order` (best case — fixing the mechanism does
not incidentally break any currently-passing leg), (2) `n_threading_violations`
per question is held constant (conservative — a couple of the fixed legs are
corridor legs, so a real fix might also clear the matching threading
violation, which this projection does NOT credit). Recomputed leg-by-leg
from `reports/gt_battery_main_post72/gt_battery_results.json`, both
`ordered_leg_credit` (macro-average) and `rubric_score` (via the exact
formula above), n=30 questions.

| scenario | legs flipped | credit | Δcredit | headline | Δheadline |
|---|---|---|---|---|---|
| **baseline** (current) | — | 0.5222 | — | 0.4611 | — |
| **#74 lands its (b) legs** | 6 | **0.6167** | **+0.0945** | **0.5556** | **+0.0945** |
| **#75 lands its (c) legs** | 2 | **0.5500** | **+0.0278** | **0.4778** | **+0.0167** |
| **(d) fixed hypothetically** | 6 | **0.6000** | **+0.0778** | **0.5389** | **+0.0778** |
| **all three combined** | 14 (6+2+6, verified disjoint) | **0.7222** | **+0.2000** | **0.6500** | **+0.1889** |

## Headline finding for the orchestrator: does 0.8 need a (d) lane?

**Yes — (d) looks like a real, currently unowned mechanism worth its own
lane**, not a duplicate of #74/#75:

- It is NOT explained by (b) (leg-boundary skip) or (c) (salience mispick):
  cross-checked all 6 clean (d) legs against both the (b) route-shape
  heuristic and the (c) resolver-parity dump — `home_building_1`'s legs
  have no salience-tie signature (single-candidate resolves, not same-label
  ties) and while 2 of the 4 `home_building_1` legs + both `livingroom_1`
  legs DO show the (b) route-shape pattern (final leg reached, earlier
  skipped — see overlap note above), the other 2 `home_building_1` legs
  (qi4, both legs of a 2-leg question where NEITHER leg is reached) do not
  fit (b) at all — they're plain double-misses with 1.3 m and 6.3 m excess,
  unexplained by any of (a)/(b)/(c).
- `hotel_room_2`'s 0-blocked-legs-despite-flag result is the load-bearing
  negative-control evidence that the wall-unavailable flag is a real hazard
  factor, not just correlated noise (it's necessary-but-not-sufficient,
  which is exactly what a genuine "costmap quality degrades without wall
  derivation" mechanism looks like, not a coincidence).
- **Combined-lane math**: (b)+(c)+(d) together only reach credit 0.7222 /
  headline 0.6500 — short of the 0.8 credit-side target implied by the
  brief's framing. Even generously crediting (b) alone (the largest single
  lane) only gets to 0.6167. **Without a (d) lane, the ceiling from
  currently-identified, in-flight-or-flaggable mechanisms tops out at
  ~0.65 headline / ~0.72 credit — a genuine gap to 0.8 remains even if
  #74 and #75 both land cleanly**, made up of: (e)-other's 16 legs (broad
  drive-precision gap, no lane), (e)-frame's 3 `livingroom_3` legs
  (separately tracked, needs its own fix), and whatever residual the (a)
  fold-shortcut check can't rule out without trajectory instrumentation.

## Deliverables

- `reports/conversion_probe_v2/probe.md` — this file.
- `reports/conversion_probe_v2/probe.json` — full 72-leg row table
  (`legs`), bucket counts (`buckets`), mechanism counts
  (`mechanism_counts`), and the 4 projection scenarios (`projections`),
  machine-readable.
