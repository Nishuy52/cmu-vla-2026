# GT leg-goal reachability ceiling (T11)

The **GT reference trajectories reach 30/72 ordered leg goals within 0.8 m** of the
scorer's raw leg-goal centroid (ratio 0.4167). Each leg goal is resolved exactly as
`gt_battery._if_rubric_geometry`; the GT reference PLY is mapped into the object frame
by the same per-scene rigid fit the battery uses; distance is the trajectory's minimum
Euclidean distance to the goal. A leg "reaches" when that minimum ≤ `LEG_ARRIVAL_TOL_M`
(0.8 m).

Because the canonical correct path only reaches 30/72, ordered-leg credit for a perfect
IF answer tops out at ~0.42 under the current scorer — so the residual battery gap above
30/72 is a scorer/geometry property (centroid tolerance + AABB over-approximation), not
a pipeline defect. This is the number that justifies calling the residual gap
non-pipeline and un-holding the ×6-weighted CV sweep.

Per-scene GT reached / total legs:

| scene | reached/total | scene | reached/total |
|---|---|---|---|
| arabic_room | 2/5 | livingroom_2 | 2/4 |
| chinese_room | 0/4 | livingroom_3 | 1/5 |
| home_building_1 | 1/5 | livingroom_4 | 4/5 |
| home_building_2 | 2/5 | loft | 3/4 |
| hotel_room_1 | 3/5 | office_1 | 2/5 |
| hotel_room_2 | 2/5 | office_2 | 3/5 |
| japanese_room | 3/5 | studio | 2/5 |
| livingroom_1 | 0/5 | **TOTAL** | **30/72** |

Full per-question / per-leg rows (goal xy, GT min distance, reached flag) are in
`gt_leg_ceiling.json`, which also records the exact regenerate command + data root.

## How to regenerate (checkable on the Ubuntu machine)

From `src/`, with the VLA-3D Unity root that holds the 15 battery scenes:

```
python -m core.runner.gt_leg_ceiling \
    --groundtruth <UNITY_ROOT> \
    --out ../reports/gt_battery_postT11_2026-07-14/gt_leg_ceiling.json
```

`<UNITY_ROOT>` = the Unity root with each `<scene>/` (object CSV + scene graph). In
this repo tree it is `data/vla3d/Unity` (git-ignored fixtures — the reason a
fresh-context checkout without them cannot reproduce this). `--questions-dir` defaults
to the challenge questions dir shipping `trajectory_q{4,5}.ply`.
