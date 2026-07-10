# Task spec: colored 3D point-cloud reconstruction (offline debug tool)

*Handoff from design discussion, 2026-07-11 (session 7 continued). Delete this file once the
task lands and fold a summary into `LOG.md`, per the existing task-spec convention
(see prior `docs/next_task_perception.md`, deleted at `aec5203`).*

## What this is and isn't

A **debug/visualization tool for the developer**, not part of the scored per-question pipeline.
It does not run under the 600 s question budget, the checkpoint call ledger, or the
"relaunch per question, no cross-question state" constraint (`docs/architecture.md` §2/§7) —
those constraints govern the live `core/` answer path only. This tool consumes a *recorded*
session (a replayed bag) after the fact and produces a file a human opens in an external
viewer. Consequently:

- **Do not add it under `core/`** if that would pull in a viewer/plotting dependency —
  `core/` must stay `numpy`-only, offline, no-GPU per `src/README.md`. Prefer a new top-level
  `tools/` directory (outside `src/ai_module`) so it's obviously excluded from what gets
  copied into the eventual challenge fork ("only `ai_module/` may be modified" — root
  `CLAUDE.md`).
- It's fine for this tool to depend on things `core/` can't (e.g. writing a `.ply` is doable
  with zero new deps — plain ASCII PLY is trivial to hand-write — but if a nicety like a
  quick preview render is wanted, an optional `open3d` import guarded the same way
  `GroundingDinoDetector` guards `torch` in `core/perception/detector.py` is the precedent
  to follow).

## Why (context from the design discussion)

We already keep two compressed representations of what the robot has seen: the sparse
per-object instance map (`core/perception/scene_index.py`, trimmed-AABB clusters only) and
the 2D occupancy grid (`core/nav/occupancy.py`, height discarded). Neither lets a human look
at what the robot actually reconstructed geometrically. A colored dense point cloud —
accumulated raw lidar points, each tinted by the camera pixel it projects to — fills that gap
and is useful purely for spotting fusion/tracking/calibration bugs during development (e.g.
"why did the AABB for the sofa get 2× too wide" is much faster to diagnose by eye on a colored
cloud than by reading numbers).

Two things were explicitly **ruled out** in the design discussion, so don't re-litigate them
here:
- Feeding raw/uncolored voxel screenshots to a VLM for object identification (grey blocks with
  no color/texture are a worse signal than the camera crop already used at the verification
  checkpoint).
- Building this as a live, real-time 3D voxel map for the actual answer pipeline (the sparse
  instance map + 2D costmap already serve every current answer head; see the geometry/nav
  modules).

If a later session wants to revisit feeding a *colored* reconstruction screenshot to a VLM for
some specific spatial-reasoning checkpoint, that's a legitimate follow-on, but is out of scope
for this task — build the reconstruction tool first, decide on a VLM use case only once it
exists and looks good.

## The core technique: forward-project lidar points onto the panorama

`core/perception/tiling.py` already has the *backward* mapping (tile pixel → camera-frame ray,
`tile_pixel_to_camera_ray`) and `core/perception/fusion.py` already has points-in-map-frame →
bearing/elevation-from-apex math (`_points_in_frustum`, lines ~114-119). This task needs the
mirror-image **forward** mapping: lidar point (map frame) → panorama pixel (row, col), so its
color can be sampled directly from `PanoFrame.image`.

Suggested function, e.g. in a new `tools/colored_cloud.py` (or `core/perception/tiling.py` if
it turns out to be pure geometry with no new deps — reviewer's call at implementation time):

```python
def map_point_to_pano_pixel(point_xyz, odom: OdomState) -> tuple[float, float]:
    """3D map-frame point -> (row, col) in the native 1920x640 equirect panorama.

    apex = (odom.x, odom.y, odom.z); compute bearing/elevation from apex to point
    (same arctan2 pattern as fusion._points_in_frustum), convert map bearing -> camera
    azimuth via tiling.map_ray_to_camera(bearing, elevation, odom.yaw), then straight to
    pixel via tiling.azimuth_to_column / elevation_to_row (already exist, no tiling
    reprojection needed since we want native-panorama pixels, not tile pixels).
    """
```

Per point: sample `PanoFrame.image[row, col]` (nearest or bilinear) for its RGB, keep
`(x, y, z, r, g, b)`.

## Practical concerns to handle (all flagged in the design discussion)

1. **Point volume.** At 5 Hz over up to 600 s that's up to ~3000 scans — voxel-downsample
   (e.g. 5 cm voxel keeping one point/color, or an average, per occupied voxel) before writing
   output. This is a debug tool; no need for a live hash-grid structure, a one-shot dict
   keyed by `(floor(x/v), floor(y/v), floor(z/v))` at export time is enough.
2. **Occlusion / visibility.** A lidar point behind a nearer surface (from the camera's
   viewpoint at capture time) will still forward-project to *some* panorama pixel and pick up
   whatever color is there — potentially the wrong surface's color. A simple z-buffer (per
   pixel, keep only the nearest point's color, discard/mark farther points that map to the
   same pixel within one scan) is the fix; for a first pass it is an acceptable known
   limitation to skip and note in the module docstring — this is diagnostics, not the scored
   path, and imperfect coloring is still far more useful than no color.
3. **Panorama azimuth/elevation conventions.** Reuse `AZIMUTH_SIGN` / `COLUMN0_YAW_OFFSET` /
   `ELEVATION_SIGN` from `tiling.py` verbatim — don't re-derive the sign conventions, they're
   already calibration-flagged constants for exactly this kind of mapping.
4. **Output format.** Plain ASCII PLY (`element vertex`, `property float x y z`, `property
   uchar red green blue`) needs zero dependencies to write or to view (Open3D, CloudCompare,
   MeshLab, even some web viewers open PLY directly). Don't reach for a heavier format unless
   there's a concrete reason.

## Suggested plumbing

- Source data: the replay harness (`core/replay/`, `ReplayRobotIO`) already turns a recorded
  bag into a sequence of `PanoFrame` / `LidarScan` / `OdomState` — reuse it rather than parsing
  bags again.
- CLI entry point mirroring the existing fixture tool's pattern (`python -m
  core.replay.fixtures extract <bag> <out>`): something like
  `python -m tools.colored_cloud <bag_or_fixture> <out.ply> [--voxel 0.05]`.
- Tests: pure-numpy pieces (the forward-projection function, voxel-downsample, PLY writer)
  should get unit tests same as everything else in `core/`; the CLI glue itself doesn't need
  much beyond a smoke test against a tiny synthetic fixture.

## Resume checklist for the session picking this up

1. `git status` + full `pytest` from `src/` — confirm current green baseline before starting.
2. Decide the module home (`tools/` at repo root is the recommendation above) and write
   `map_point_to_pano_pixel` + a small pinning test (round-trip against
   `tile_pixel_to_camera_ray`/`camera_ray_to_map` for a few known points, since the forward and
   backward mappings should agree).
3. Voxel-downsample + ASCII PLY writer, unit-tested.
4. Wire the CLI over the replay harness; run it against whatever sample bag/fixture data has
   landed by then (check `data/sample_real_robot/` — was pending a manual browser download as
   of session 7).
5. Sanity-check the output by opening it in Open3D/CloudCompare/MeshLab — confirm it looks like
   a real room, not noise (that's the actual "done" signal for a debug tool; there's no
   automated correctness oracle for "does this look right").
6. Delete this spec file, fold a short summary into `LOG.md`, commit, push — per standing
   session protocol in the root `CLAUDE.md`.
