# Phase 2 Playbook — Ubuntu Day One to First Scored Dry-Run

*Written 11 Jul 2026, for the Ubuntu reinstall (Thu 16 Jul). This is the ORDER OF OPERATIONS; the
detailed commands live in `ubuntu_setup.md` (install) and `sim_verification.md` (verification
ladder). Work through the gates in sequence — each gate is a hard prerequisite for the next.
Estimated total: 2–3 focused days to Gate 5, then calibration until the Aug 3 MVS submission.*

## Gate 0 — before leaving Windows (do in the next 4 days)

- [ ] Bootable Ubuntu 24.04 USB made (Rufus), home GPU model + disk confirmed, backups done
- [ ] **API keys**: two funded providers; record which env vars you'll set (`core/llm/config.py`
      docstring). Without these the checkpoints stay in fallback mode — fine for bring-up, needed
      before scored dry-runs.
- [ ] Docker Hub account (submission requires pushing our image)
- [ ] **Training scene binaries** downloading to an external drive (multi-GB; Drive folder in the
      upstream README). If VPN keeps throttling, do this on the SoC cluster / campus network.
- [ ] Final Windows commit + push (`origin` is the source of truth for the transfer)

## Gate 1 — machine bring-up (ubuntu_setup.md §0–§4) — ~half a day

OS → NVIDIA driver (`nvidia-smi` must pass) → Docker + NVIDIA toolkit (`docker run --gpus all`
must pass) → clone workspace from GitHub → **`pytest` from `src/`: the full suite (642+) must be
green on Linux before anything else** (catches path/line-ending/py-version surprises immediately)
→ upstream clone + submodule.

## Gate 2 — stock sim runs (ubuntu_setup.md §5 + sim_verification.md Tier 2.1–2.5) — ~half a day

Pull upstream images → install one training scene → compose up → sensor liveness rates → waypoint
round-trip → dummy-module question round-trip. **Do not touch our module until the UNMODIFIED
stack passes 2.1–2.5** — otherwise you can't tell whose bug it is. Troubleshooting table at the
bottom of ubuntu_setup.md.

## Gate 3 — our module replaces the dummy — ~half a day, the riskiest gate

1. Build the image (`docker/ai_module/Dockerfile`, README alongside) and `colcon build` the
   adapter. Resolve the three **confirm-on-Ubuntu flags** (marked in-file):
   a. `ament_python` + `core` coexistence (PYTHONPATH into the container/launch)
   b. colcon workspace layout vs the base image's conventions
   c. perception wiring into the node (see Gate 4 — initially run with the empty scene index)
2. First run criterion (perception NOT yet wired): publish a question by hand → module must
   drive (exploration waypoints) and publish a **floor answer** before the watchdog. That proves
   adapter plumbing end-to-end. sim_verification 2.7.
3. Calibrate the three tiling constants against the sim panorama (`core/perception/tiling.py`:
   `AZIMUTH_SIGN`, `COLUMN0_YAW_OFFSET`, `ELEVATION_SIGN`): drive toward a known object, check
   its pano column matches the predicted azimuth; flip constants if mirrored. Also verify the
   camera-level assumption note in tiling.

During bring-up use the debug launch `ros2 launch vla_ai_module ai_module_debug.launch.py`
(our additive RVIZ view of the instance map + planned path + answer marker) — see
`docs/sim_verification.md` §2.8. Keep the eval path on `ai_module.launch.py` (debug OFF, no rviz).

## Gate 4 — real perception — ~half a day

1. Uncomment torch/groundingdino in the Dockerfile; download weights (`GDINO_MODEL_ID` in
   `core/perception/detector.py`); bake into the image (offline-capable rule).
2. Implement the real `GroundingDinoDetector.__call__` (the stub documents the prompt format) +
   a real JPEG `encode_fn` for the vision checkpoints.
3. Wire `PerceptionPipeline` into `adapter_node` (the flagged seam) — mirror how
   `core/runner/single.py` does it for replay (`--detections` path is the reference wiring).
4. Smoke: object_reference question on a training scene → Marker lands on the right object in
   RVIZ. VRAM check vs the ~10–14 GB target (architecture §6).
5. Set `VLA_LLM_*` env vars (compose env_file) → parse tier should report `api` not `regex`;
   CP4 verification fires (flight recorder shows the checkpoint calls).

## Gate 5 — first scored dry-run — the milestone

Run all 5 questions of one training scene under eval conditions (relaunch per question, 10-min
clock). Score by hand: numerical/object-ref vs VLA-3D ground truth (see `vla3d_notes.md` /
ground-truth eval docs), instruction-following vs the `trajectory_q4/q5.ply` paths. Log the
per-question wall-time split (explore vs answer) — that's the calibration currency.

## Gate 5b — perception-vs-reasoning error split (do once, right after Gate 5)

2025 teams could ground against sim-published GT semantics; **2026 forbids it at test time**
(upstream gotcha 8 — only the six topics are legal), but it remains legal in *development*. Use
that: run the same scene's questions twice — (a) full pipeline, (b) sim/VLA-3D ground-truth
objects injected via the scene-index seam (the GT-battery wiring). Score both. The (b)−(a) gap is
**perception-caused loss**; (b)'s own gap to full marks is **reasoning-caused loss**. This split
decides where the calibration weeks go — do not skip it, and re-run it after any detector change.

## Then: the calibration loop (until ~Aug 3 MVS)

Sweep the ledger (`docs/calibration.md`, 55 tunables; wire the 14 wiring-TODO fields as needed)
against sim scores across the 15 scenes; battery harness + reports for regression tracking.
Priorities from the expected-points model: instruction-following timing/corridor behavior first,
Marker IoU tightness second, counting thresholds last.

**Submit the MVS by Aug 3** (multiple submissions allowed, highest kept): push image → fork with
`ai_module/` → Google Form. Then ratchet until the Aug 13–15 freeze. **Pre-submit checklist:
`instances_tracked > 0` on a live scene (Gate 4 hard gate — no empty-stub index), `VLA_DETECTOR`
set to a real detector, and no `SUBMISSION-BLOCKER` lines in the boot log.**

**Snapshot packaging (10 Aug 2026):** `SUBMISSION_SNAPSHOT.md` at the repo root records the
pinned commit, the one-command build and smoke test, and the manual fork-push/tag/form steps
for the final freeze push. The packaging in `docker/ai_module_fork/` was audited against this
commit and found current — no code, dependency, or env-var gap; only two stale in-file comments
were corrected. The `instances_tracked > 0` and `SUBMISSION-BLOCKER` checks above still need a
live GPU sim run — packaging review alone cannot confirm them.

## If something is on fire

- Stock sim broken → upstream issue, check gotchas 13 (CycloneDDS) / scene binary perms first
- Our module silent → flight recorder dump (fsm events) tells you which state it died in;
  the watchdog should make true silence impossible — if it IS silent, suspect the adapter's
  timer/callbacks, not core
- Robot ignores waypoints → gotchas 11/14 (topic name, waypoint distance)
- Detector garbage on sim images → tiling constants (Gate 3.3) before blaming the model
- Everything on Windows still works → `git bisect` between the last green Windows commit and
  the Ubuntu changes; core is OS-independent by construction
