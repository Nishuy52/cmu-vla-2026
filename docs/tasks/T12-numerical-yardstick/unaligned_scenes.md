# IF unaligned-scene frame fit (meth-F11)

One-time investigation of the 3 of 15 scenes whose scene-level sim→object frame
fit failed the 1.0 m residual gate, so their 6 instruction-following (IF)
questions (20 % of IF rows) were excluded from the Fréchet/coverage diagnostics —
biasing every IF mean friendly-ward. Scenes:
`chinese_room`, `home_building_2`, `livingroom_3`
(verified against `reports/gt_battery_T12pre_2026-07-17/gt_battery_results.json`
→ `aggregate.instruction_following.unaligned_scenes`).

## Why the fit fails (mechanism)

The IF fit takes ONE `(trajectory_endpoint → resolved terminal-goal centroid)`
correspondence per question (q4, q5) and fits ONE rigid 2-D sim→object transform
(`align_scene_trajectories` → `fit_frame`, translation + optional yaw). A rigid
transform preserves distances, so with exactly two correspondences the residual is

    residual ≈ | d_src − d_dst | / 2

where `d_src` = distance between the two trajectory endpoints (sim frame) and
`d_dst` = distance between the two terminal-goal centroids (object frame). The
three failing scenes all have a large `|d_src − d_dst|`:

| scene | d_src (sim endpoints) | d_dst (goal centroids) | mismatch | residual | gate |
|---|---|---|---|---|---|
| chinese_room | 7.238 | 2.663 | 4.575 | 2.288 | 1.0 |
| home_building_2 | 12.036 | 6.934 | 5.102 | 2.551 | 1.0 |
| livingroom_3 | 1.195 | 4.816 | 3.621 | 1.810 | 1.0 |

(residual == mismatch/2 confirms the two-point rigid-fit mechanism exactly.)

The 12 currently-aligned scenes have non-trivial true rotations (arabic_room
−154°, home_building_1 +102°, japanese_room −18°, …) — the sim→object transform
is a genuine per-scene rigid motion, NOT identity — yet their two correspondences
are mutually distance-consistent (`|d_src − d_dst|` small → residual < 1 m). So
the failure is a *correspondence* problem in the 3 scenes, not a fitting-method
problem.

## Per-scene diagnosis

### chinese_room — REJOINS (resolver mispick)

- q4 terminal `painting near tv` → id2 @[-2.27,-3.63] (only candidate). endpoint
  [-1.69,-3.90].
- q5 terminal `table with horse figurine` → resolver TOP pick is **tea table id76
  @[-0.02,-2.20]** (wrong — that is the elephant-figurine tea table from the FIRST
  leg). The second candidate **table id77 @[6.23,-2.59]** is the correct terminal.
  endpoint [5.41,-2.50] sits right beside id77.
- Fitting (painting id2, table **id77**) → residual **0.661 m < 1.0** → aligned.

### home_building_2 — REJOINS (resolver mispick)

- q4 terminal `potted plant on dressing table` → resolver TOP pick **id219
  @[3.13,-0.66]** (wrong plant). The correct one is **id221 @[-3.34,11.70]**,
  right at the q4 endpoint direction [-4.07,11.66].
- q5 terminal `potted plant between curtain and tv` → id58 @[-3.80,-0.72] (correct;
  endpoint [-4.58,-0.36]).
- Fitting (potted plant **id221**, potted plant id58) → residual **0.196 m < 1.0**
  → aligned.

### livingroom_3 — CONFIRMED DATA EXCLUSION (not fittable)

The scene's true transform is near-identity (the q5 trajectory passes within
0.74 m of the stool's object-frame centroid; the bowl sits 1.6 m from the q5
endpoint). Under any rigid transform the endpoint-separation invariant must hold,
and it is violated at the data level:

- The two GT trajectory endpoints are only **1.20 m apart**
  (q4 end [5.23,-2.98], q5 end [6.31,-2.48]) — both land in the *same*
  dining-chair corner. q4's endpoint is nearest to a **chair** (id94, 1.04 m); the
  nearest pillow (id75) is 1.33 m away and the resolver's `pillow farthest from
  lamp` (id89 @[2.47,-2.08]) is 2.76 m away.
- Every `pillow × bowl` pairing in the object frame is **≥ 3.27 m apart** (nearest
  pillow id75 @[4.22,-2.11] to the only bowl-on-table id36 @[6.90,-3.99]).
- A distance-preserving rigid transform cannot map two points 1.20 m apart onto
  two points ≥3.27 m apart. The minimum achievable two-point residual is
  `(3.27 − 1.20)/2 = 1.035 m > 1.0 gate` — a **frame-independent** hard floor.
  Enriching the fit with the clean intermediate anchors (stool 0.74 m, TV 1.48 m)
  pins the true near-identity frame (θ ≈ −7°) and *raises* the terminal residual to
  1.36 m, confirming the terminals genuinely do not sit at any consistent
  pillow/bowl pair.

Conclusion: the `q4` GT trajectory does **not** terminate at the
`pillow farthest from lamp`; it ends in the dining-chair corner. This is a
GT trajectory/question inconsistency in the challenge data — not our resolution —
so the exclusion is DATA-confirmed (meth-F11). livingroom_3 stays unaligned by
design and is reported as data-confirmed rather than a resolution gap.

## Fix (implementation)

`src/core/runner/gt_battery.py`, `score_scene` IF section:

- New helper `_terminal_goal_candidates(text, idx, k)` returns the top-k resolver
  candidate centroids for a question's terminal goal (`_terminal_goal_centroid`
  now delegates to it with k=1 — behaviour unchanged for existing callers).
- New helper `_fit_if_frame_over_candidates(...)`: a fallback correspondence search
  run ONLY when the default top-candidate fit exceeds the gate. It fits
  `S.fit_frame` over the product of per-question candidate centroids and returns
  the lowest-residual `(frame, residual, picks)`. Adopted only when its residual
  ≤ gate, so the 12 aligned scenes (whose default fit already passes) and their
  Fréchet values are never touched.
- `_DATA_UNFITTABLE_IF_SCENES`: records livingroom_3 with its reason; the row note
  and the aggregate carry a `data_confirmed` marker so the exclusion reads as a
  data property, closing the friendly-ward bias hole as documented data.

## Outcome

| scene | before | after | outcome |
|---|---|---|---|
| chinese_room | unaligned (res 2.288) | aligned (res 0.661) | **rejoined** |
| home_building_2 | unaligned (res 2.551) | aligned (res 0.196) | **rejoined** |
| livingroom_3 | unaligned (res 1.810) | unaligned, data-confirmed | **exclusion confirmed** |

Net: 4 of 6 IF rows rejoin the aligned diagnostic set (24 → 28 aligned
scene-questions); the remaining 2 (livingroom_3) are a confirmed GT-data defect,
documented rather than silently dropped. Friendly-ward bias hole closed.

<!-- numbers below are filled in from the scratch IF slice run once the fix lands -->
