# T5 — colored-cloud-tool

Offline colored 3D point-cloud reconstruction tool (developer debug/visualization).

## Intent

Accumulate raw lidar points from a recorded session, color each by
forward-projecting onto the nearest-in-time panorama, voxel-downsample,
export a colored PLY a human opens in an external viewer (CloudCompare /
MeshLab / Open3D). Diagnostic tooling only — NOT part of the scored
per-question pipeline; lives outside `src/` in a top-level `tools/`
package so it is obviously excluded from the eventual challenge fork.

## Context

- Full spec + binding research addendum:
  `docs/next_task_colored_reconstruction.md` (design discussion session 7;
  3-agent research pass session 8). Per that file's own convention it gets
  deleted when this task lands; its addendum content is summarized here.
- Real data available: `data/fixtures/jingfan/` (246 keyframes, real
  MCAP bag of a CMU student lounge) — validation target.

## Key technical decisions (from the research addendum, binding)

- Forward projection = composition of existing calibrated `tiling.py`
  functions (`map_ray_to_camera` → `azimuth_to_column`/`elevation_to_row`);
  apex math mirrors `fusion._points_in_frustum`. Lidar points are already
  map-frame.
- **VFOV gate mandatory**: pano covers ±60° elevation only; out-of-band
  points get no color (drop/flag, never clamp). Azimuth wraps; rows don't.
- Color: nearest-in-time pano, single source, nearest-neighbor pixel,
  **no blending** (misregistration must stay visible — it is the
  diagnostic signal). Min-range cutoff ~0.75 m kills the near-field
  parallax tail.
- Occlusion handling skipped in v1 (documented limitation); z-buffer is
  the primitive if added later; HPR rejected.
- Voxel reduce: min-offset before floor, packed int64 keys (21 bits/axis),
  `np.unique` + `bincount`, mean position+color. Median color = follow-up.
- Output: binary_little_endian PLY (`float x/y/z` + `uchar red/green/blue`),
  `--ascii` escape hatch. Uncolored points kept, tinted gray;
  `--drop-uncolored` flag.
- Tunables are CLI flags, not calibration-ledger entries (ledger governs
  the scored core path only).

## Acceptance criteria

1. `python -m tools.colored_cloud <bag_or_fixture_dir> <out.ply> [--voxel 0.05]`
   works from repo root against both a fixture dir and (converter-wise) a bag.
2. Pure-numpy pieces (projection, voxel reduce, PLY writer) unit-tested,
   incl. a round-trip pinning test against `tiling.py`'s backward mapping.
3. Existing suite (`python -m pytest` from `src/`) stays green; no new
   hard deps; no `core/` behavior changes.
4. Run against `data/fixtures/jingfan/` produces a PLY that opens in a
   standard viewer and looks like the actual lounge (human check = done
   signal).

## Todo

- [x] Research pass (3-agent fan-out; addendum folded into spec)
- [x] Implement `tools/` package (executor delegation)
- [x] Verifier pass
- [x] Real-data PLY generated (`reports/T5_colored_cloud/jingfan_colored.ply`);
      human eyeball check pending (USER: open in CloudCompare/MeshLab)
- [x] Docs (CLAUDE.md layout row), delete spec file, LOG one-liner, PR

## Notes

- 2026-07-11: task started on branch `tool/colored-cloud` (off `main`
  at `00482b4`). Baseline suite run kicked off pre-implementation.
- 2026-07-11: implementation landed (executor): `tools/{colored_cloud,
  ply_io, voxel}.py` + 25 tests (`python -m pytest tools`, ~2 s; main
  suite collection unaffected, 737 items). Real-data run on the jingfan
  fixtures: 229 scans, 3.42 M points in → 391 K voxels out, 5.1 % raw
  gray (0.2 % post-merge), 5.9 MB binary PLY, 19 s. Pose source = the
  nearest-in-time pano's own attached odom (in jingfan fixtures pano/
  scan/odom share keyframe timestamps, so this is the co-captured
  frame). v1 limitations documented in the module docstring: no
  occlusion z-buffer, no motion deskew, single nearest pano per scan.
- 2026-07-11: NOTE — working tree carries concurrent unrelated edits
  from the suite-tiering workstream (`src/pyproject.toml`,
  `src/core/mocks/mock_io.py`, `src/tests/integration/_scaled.py`);
  T5 commits exclude them.
- 2026-07-11: fresh-context verifier CONFIRMED all 9 claims, incl.
  independent projection probes (center/seam/VFOV-gate) and a
  3000-point forward↔backward round-trip within one-pixel
  quantization (max 0.1873° vs 0.1875°/px); byte-level parse of the
  real PLY (header count == body bytes/15 == 391193). Full `src/`
  baseline suite green (exit 0) on the branch. PLY artifact left
  uncommitted (5.9 MB); regenerate with the CLI one-liner above.
