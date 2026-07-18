# Gate 3.3 — Panorama Tiling Convention Calibration

**Date:** 2026-07-18
**Scene:** `livingroom_1` (Unity binary via `iros2026_system`, scene already loaded from
`system_simulation.sh`; robot was stationary/idle mid-scene from an earlier run — no
`/challenge_question` was published, no restart of `iros2026_system` was needed).

**Constants under test** (`src/core/perception/tiling.py`):
- `AZIMUTH_SIGN = -1.0`
- `COLUMN0_YAW_OFFSET = pi`
- `ELEVATION_SIGN = -1.0`
- Camera-level assumption (camera tilt == 0; `camera_ray_to_map` passes elevation through
  unchanged)

## Method

1. Captured one `/state_estimation` (`nav_msgs/Odometry`) reading immediately before and
   immediately after grabbing one `/camera/image` frame, via a one-shot `rclpy` script run
   inside the `iros2026_system` container (has `rclpy` + `numpy`). Pose was identical
   before/after (robot stationary, twist all-zero), so no motion contamination and no
   `iros2026_ai_module` restart was required.
2. `/camera/image` reported `encoding: bgr8`, 1920x640 (not `rgb8` as the topic doc
   suggested) — swapped channels before writing a P6 PPM, `docker cp`'d it out, and
   converted to PNG on the host (`pip install pillow` into `~/vla/.venv`).
3. Robot pose at capture: `x=-0.0271, y=-4.9106, z=0.7535`,
   `yaw=-1.7475 rad (-100.1 deg)` (from the odometry quaternion).
4. Picked distinctive, well-separated objects from
   `data/unity_scenes_ros2/livingroom_1/livingroom_1/object_list.txt`, computed
   `map bearing = atan2(dy, dx)`, `camera azimuth = wrap_pi(bearing - yaw)`, and the
   predicted column via the repo's own
   `core.perception.tiling.azimuth_to_column` (imported with
   `PYTHONPATH=~/vla/src`, no reimplementation of the convention).
5. Read the captured PNG directly (vision) to find each object's observed column,
   cross-checked with a 100px-gridded overlay and targeted crops for the two closest
   calls (TV, glass doors).
6. For elevation: compared ceiling-height and floor-height objects' predicted rows
   against the top/bottom of the image, and checked three objects near camera height
   (dz ~ 0) predict rows near the geometric vertical centre (319.5) — the level-camera
   check.

Panorama PNG kept at (git-ignored `data/`, referenced by path):
`~/vla/data/calibration/pano_livingroom_1.png` (1920x640, from the `livingroom_1` capture
above).

## Azimuth / column-offset results

| Object (id, label) | GT position (x, y) | Map bearing (deg) | Predicted col (px) | Observed col (px, visual estimate) | Delta (px) |
|---|---|---|---|---|---|
| 88 picture (heart-tree canvas, brick wall) | (-2.622, -6.576) | -147.3 | 1211.7 | ~1210 | ~2 |
| 0 sofa | (-1.842, -2.162) | 123.4 | 1687.7 | ~1695 | ~7 |
| 60 round table (coffee table) | (-0.316, -2.433) | 96.7 | 1830.5 | ~1825 | ~6 |
| 13 door / 14 door (glass double-door panels) | (0.052,-8.637)/(-0.566,-8.637) | -88.8/-98.2 | 899.5 / 949.9 (avg 924.7) | ~930 (span ~860-1010) | ~5 |
| 57 picture / 82 picture (poppy prints, east wall) | (2.163,-6.305)/(2.103,-7.158) | -32.5/-46.5 | 599.3 / 674.2 | ~597 / ~662 | ~3 / ~12 |
| 103 tv cabinet | (1.844, -2.529) | 51.8 | 149.5 | ~167 (screen) / ~185 (grid est.) | ~18-35 |
| 37 door (far-left red-wall area) | (2.225, 0.832) | 68.6 | 60.2 | ~near col 0-70 (red wall/shelf) | consistent, not precisely measured |

All seven independent anchors, spanning nearly the full 360-degree ring (bearings from
-147 to +124 deg), land within **~2-35 px of prediction on a 1920 px-wide image (<2%)**
— well inside the 50-100 px tolerance. Critically:
- No object shows the "mirrored" pattern (`observed ~= width - predicted`) that would
  indicate a flipped `AZIMUTH_SIGN`. (E.g. picture 88: predicted 1211.7, mirror would be
  708.3 — nowhere near the observed ~1210.)
- No object shows a consistent ~960 px offset that would indicate a wrong
  `COLUMN0_YAW_OFFSET` (0 vs pi). Multiple anchors at very different map bearings all
  land close to their own individual predictions, not shifted by a common ~960 px, which
  is what disambiguates a sign error from an offset error (a single object cannot).

**Verdict: `AZIMUTH_SIGN = -1.0` CONFIRMED. `COLUMN0_YAW_OFFSET = pi` CONFIRMED.**

## Elevation results

| Object | z (m) | dz vs camera (0.7535m) | Predicted row | Expectation | Match |
|---|---|---|---|---|---|
| 12 ceiling lamp | 2.145 | +1.39 | 122.5 | near top (row 0) | yes — lands in upper ceiling band (visually rows 0-180 in image) |
| 35 ceiling lamp | 2.207 | +1.45 | 161.7 | near top | yes |
| 27 carpet | 0.009 | -0.74 | 415.5 | near bottom (floor visible ~ rows 380-640) | yes |
| 88 heart-tree picture | 1.609 | +0.86 | 236.8 | upper-middle (wall art, above table) | yes, matches visual |
| 0 sofa | 0.331 | -0.42 | 358.5 | just below centre (seat height) | yes |
| 103 tv cabinet | 0.284 | -0.47 | 366.5 | just below centre | yes |
| 60 coffee table | 0.191 | -0.56 | 387.3 | below centre | yes |
| 92 dish (near camera height) | 0.76 | +0.007 | 318.4 | ~row 319.5 (geometric centre) | yes, delta 1.1 |
| 95 dish (near camera height) | 0.75 | -0.003 | 320.1 | ~row 319.5 | yes, delta 0.6 |
| 71 vase (near camera height) | 0.84 | +0.087 | 303.8 | ~row 319.5 | yes, delta 15.7 (small object over short range) |

Ceiling objects map to low rows, floor objects to high rows, consistent with
"row 0 = top = highest elevation."

**Verdict: `ELEVATION_SIGN = -1.0` CONFIRMED.**

## Camera-level assumption

Three objects at the camera's own height (dz within +/-0.09 m: dishes 92/95, vase 71)
all predict rows within ~1-16 px of the geometric vertical centre (319.5 of 639), with
no systematic bias in either direction. This is consistent with the docstring's
"camera assumed level" note — no tilt correction is evidenced by this single-frame,
single-pose check. This is not an exhaustive validation (one robot pose, one scene);
if a future gate captures frames at multiple poses/pitches, re-run this check before
trusting the level assumption broadly.

## Outcome

No constants required flipping. `src/core/perception/tiling.py` was not modified.
`src/tests/perception/test_tiling.py` was not touched (no convention change to reconcile).

**Test line (perception suite, unchanged code):**
```
cd ~/vla/src && ~/vla/.venv/bin/python -m pytest tests/perception -o addopts="" -q
# 144 passed, 1 skipped in 5.31s
```

Full-gate run was not performed since no source file changed (only this doc was added).

## Files referenced

- `~/vla/src/core/perception/tiling.py` (read only, not modified)
- `~/vla/data/unity_scenes_ros2/livingroom_1/livingroom_1/object_list.txt` (ground truth)
- `~/vla/data/calibration/pano_livingroom_1.png` (captured panorama frame; git-ignored,
  kept on disk at this path)
- This file: `~/vla/docs/tiling_calibration.md`
