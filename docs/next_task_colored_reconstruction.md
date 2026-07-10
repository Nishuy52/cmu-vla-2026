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

---

## Research addendum (2026-07-11, session 8) — technique survey + codebase recon

Three-agent pass (codebase recon + two deep literature surveys) before implementation.
The plan above **holds**; the following amendments and confirmations are binding on the
implementing session.

### Codebase facts the implementation relies on (verified)

- `LidarScan.points` is **(N, 3) float32 already in the map frame** (`registered_scan`) —
  no per-scan transform; apex subtraction only, exactly the `_points_in_frustum` pattern
  (`fusion.py:98-128`: `bearing = arctan2(dy, dx)`, `elevation = arctan2(dz, hypot(dx, dy))`,
  apex `= [odom.x, odom.y, odom.z]`).
- The forward mapping is a pure composition of existing calibrated functions:
  arctan2 from apex → `map_ray_to_camera(bearing, el, yaw)` (`tiling.py:126`) →
  `azimuth_to_column` / `elevation_to_row` (`tiling.py:95-107`). Column convention: column 0
  at yaw + π, azimuth decreases left→right (`AZIMUTH_SIGN = -1.0`, `COLUMN0_YAW_OFFSET = π`,
  `ELEVATION_SIGN = -1.0`); row 0 = +60° elevation. Use fractional pixel → `floor`, then
  `col % 1920` (azimuth wraps; rows do NOT — see VFOV gate below).
- **VFOV gate (spec omission, mandatory):** the panorama covers **±60° elevation only**
  (`PANO_VFOV = 120°`, 1920×640 is 3:1, not 2:1). Points projecting outside rows [0, 640)
  get *no* color — drop or flag, never clamp.
- Replay plumbing: `ReplayRobotIO.from_bag(path)` / `load_fixtures(dir)` →
  `MessageStore`; iterate via `store.all_times()` + `latest(...)`, or walk
  `BagSource.frames()` directly. Mirror the `fixtures.py` argparse layout
  (`extract`/`info` subcommands) for the CLI.
- Optional-dep guard precedent: lazy import inside the call with a pip-install message
  (`detector.py:163-177`) — copy for any `open3d` preview nicety.
- `data/sample_real_robot/` still has only an RViz config + a **partial zip** — validate on
  synthetic fixtures first; real-bag run remains blocked on the manual download.
- Tool tunables (voxel size, depth tolerance, min range) are **CLI flags, not calibration
  ledger entries** — the ledger (`core/calibration.py`) governs the scored `core/` path only.

### Amendments to the plan (research-driven)

1. **Output: binary_little_endian PLY, not ASCII.** Equally zero-dependency (numpy
   structured dtype `<f4 ×3 + u1 ×3` + hand-written header + `.tobytes()`, ~20 lines),
   3–5× smaller and ~an order of magnitude faster to write/parse (≈15 B/pt vs 45–60 B/pt;
   at 10 M pts: ~150 MB vs ~500 MB). Property names **`red/green/blue` as `uchar`** —
   float 0–1 color is the classic "cloud renders black/red" interop bug. Optionally keep
   `--ascii` for eyeball-the-file debugging.
2. **Color assignment: nearest-in-time panorama, single source, no blending.** Our dominant
   errors are systematic geometric misregistration (parallax, no deskew), and averaging
   misregistered colors yields a blurrier wrong answer. For a *debug* tool, crisp
   single-source coloring makes misregistration visible as ghosting — that is the
   diagnostic signal. **Nearest-neighbor pixel sampling, not bilinear** (geometric error
   budget is tens of pixels near-field; bilinear also needs wrap-aware interpolation at the
   seam for no visible gain).
3. **Cheap error gates (add in this order of value):**
   - *min-range cutoff* (~0.75 m): parallax/apex error is ~30 px at 1 m but ~3–6 px past
     5 m — excluding the near-field tail removes the worst miscoloring for one `if`.
   - *VFOV gate* (mandatory, above).
   - *depth-discontinuity skip* (optional second pass): points whose angular neighbors in
     the same scan differ in range by > ~0.5 m sit on silhouette edges and produce the
     color-bleed "halo"; skipping them is the highest-value occlusion mitigation short of
     a z-buffer.
4. **Occlusion: per-scan z-buffer confirmed as the right primitive *if/when* added; Hidden
   Point Removal (Katz et al. 2007 / Open3D) explicitly rejected** — per-viewpoint convex
   hull, noise-sensitive, opaque radius tuning, built for unposed clouds; we have known
   poses and a natural pixel raster, so binning ranges per pixel (`np.minimum.at`) with a
   few-cm tolerance is O(n) and interpretable. Skipping occlusion entirely in the first
   pass stays acceptable: errors are localized to occlusion boundaries, and 10 min of
   multi-view accumulation self-corrects much of it (a point occluded in one pano is seen
   directly in another).
5. **Voxel reduction: packed-key numpy, not a Python dict.** Subtract the global min before
   `floor` (negative-coordinate truncation bug), pack `(ix, iy, iz)` into **int64** at
   21 bits/axis, reduce via `np.unique(..., return_inverse/counts)` + `np.bincount`
   weighted sums. Mean position + mean color per voxel for the first pass (matches
   Open3D/PCL semantics); per-channel **median color** is the cheap follow-up if occlusion
   speckle smears visibly — do not average across frames (see 2).
6. **Uncolored points** (out of VFOV / range-gated): keep the geometry, tint a fixed gray —
   coverage gaps are themselves diagnostic; add `--drop-uncolored` if noise dominates.

### Reference reading (closest prior art)

- **OmniColor** (arXiv:2404.04693) — lidar cloud + 360° pano sequence + coarse poses →
  colorized cloud; the published system closest to this exact pipeline. Uses per-keyframe
  co-visibility (voxelized HPR) + unweighted averaging because its goal is *pose
  refinement*; deliberately more machinery than a debug tool needs.
- RealSense projection/occlusion whitepaper — the halo/color-bleed mechanism and the cheap
  UV-monotonicity-style occlusion detection our depth-discontinuity skip approximates.
- plyfile docs / CloudCompare forum "colored ply appearing red" — the uchar-vs-float PLY
  color interop trap.
- KITTI-360 tooling, PDAL `filters.colorization` (orthophoto-only — conceptual cousin,
  not applicable directly).
