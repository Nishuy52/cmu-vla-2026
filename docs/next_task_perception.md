# Pending task — perception pipeline scaffold

*Queued 10 Jul 2026. If no `core/perception/{tiling,detector,fusion,tracker}.py` exists yet, this
task was interrupted by the session usage limit — re-run it verbatim as an executor delegation.
Delete this file once the module is merged and tested.*

Owner boundary: `src/core/perception/**` (additions; don't break `scene_index.py`) + `src/tests/perception/**`.
Baseline: 347 tests green; full suite ~7 min.

Build the perception pipeline structure with real-model seams, fully offline-testable
(panorama → tiles → detections → lidar fusion → tracked InstanceRecords):

1. **tiling.py** — gnomonic projection of the 1920×640 equirect strip into 4 pinhole tiles (90° HFOV,
   120° VFOV shared, ~10° seam overlap); inverse map tile-pixel → (azimuth, elevation) ray →
   map-frame ray given OdomState; camera-yaw convention exposed as ONE constant (Phase-2 sim
   calibration flips it in one place); precomputed remap grids; pure numpy.
2. **detector.py** — `Detection(tile_id, bbox_xyxy, label, score, mask?)`;
   `DetectorFn = Callable[[list[np.ndarray]], list[list[Detection]]]`; `FakeDetector` (scripted);
   `GroundingDinoDetector` stub with lazy torch/groundingdino import at call time + clear
   install-error message; document prompt-injection format (question nouns + vocab nouns).
3. **fusion.py** — bbox → angular frustum → select LidarScan points (map frame, via odom) →
   nearest depth cluster (histogram/1D density over ray distance) → centroid + point set;
   reject < 5 points; handle the ±π wrap seam.
4. **tracker.py** — greedy nearest association: label-compatible (via `parsing.vocab.NOUN_ALIASES`)
   AND centroid gate (0.75 m) → `BasicSceneIndex` merge (existing trimmed-AABB machinery);
   unmatched → new instance. `PerceptionPipeline.process(pano, scan) -> updated ids` with a
   keyframe gate (every K frames or 0.5 m / 30° movement); tunables in one dataclass.
5. **Tests ≥ 25** — tiling round-trip + seam continuity; fusion with synthetic lidar (centroid
   accuracy, frustum rejection, min-points, wrap seam); tracker association/merge/n_obs over 3
   scripted frames; full `process` smoke on MockRobotIO; detector stub raises install error
   without torch.

Constraints: numpy-only real path (torch strictly lazy), deterministic, no network in tests, no new
hard deps in `pyproject.toml`, no tooling attribution, add the 2–3 line torch/groundingdino note to
`ubuntu_setup.md` §7. Done = full suite green (347 + new).
