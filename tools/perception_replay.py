"""Replay the REAL perception pipeline over a recorded bag and dump an instance index.

The banked live snapshots we already have (``reports/cluster_verify/*/debug/*/
instance_index.jsonl``) are end-states from a real robot run: they cannot be
re-generated with a different perception config, so there is no way to A/B a
tiling/fusion/association/keyframe change against the SAME recorded sensor stream.
This tool closes that gap: it feeds a bag's camera + lidar topics through the exact
production seam (:class:`core.perception.tracker.PerceptionPipeline`) and writes the
result with the exact same dump function the live adapter uses
(:func:`core.perception.scene_index.dump_instance_index`), so
``tools/perception_eval.py`` (and anything else keyed on the live schema) treats a
replay index and a live index interchangeably.

Deliberately thin: this module does not reimplement tiling, fusion, association, or
keyframe gating — it only wires ``BagSource`` -> ``MessageStore`` -> the pipeline, so
whatever the pipeline does here is exactly what it does on the robot.

Usage (host GPU venv — see docs/ubuntu_setup.md §7)::

    source /opt/ros/jazzy/setup.bash
    source /home/jason/vla_host/env.sh
    source /home/jason/vla_host/venv/bin/activate
    PYTHONPATH=<repo>/src python -m tools.perception_replay \\
        --bag data/sim_bags/office_1_q1 --out /tmp/office_1_replay.jsonl

Pure offline dev tool (tools/ — never part of the scored pipeline).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import tools  # noqa: F401 -- inserts <repo>/src onto sys.path

from core.interfaces import LidarScan, PanoFrame
from core.parsing.vocab import PHRASES, SINGLE_NOUNS
from core.perception.detector import FakeDetector, GroundingDinoDetector, refresh_prompt
from core.perception.scene_index import BasicSceneIndex, ENV_INSTANCE_DUMP_PATH, dump_instance_index
from core.perception.tracker import PerceptionPipeline
from core.replay.bag_reader import BagSource
from core.replay.replay_io import CH_SCAN, MessageStore

#: The full offline-parser vocabulary (single nouns + canonical multi-word phrases),
#: exactly the set core.heads.factory primes GroundingDinoDetector with at boot
#: (_STANDING_VOCAB_NOUNS) before any question latches -- rebuilt here rather than
#: importing that module-private name, since replay is not tied to any one question
#: and wants the SAME full-breadth detection the live robot runs before a plan exists.
STANDING_VOCAB_NOUNS: tuple[str, ...] = tuple(sorted(set(SINGLE_NOUNS) | set(PHRASES.values())))

DETECTOR_CHOICES = ("grounding_dino", "fake")


def build_detector(choice: str):
    """Construct the detector named by ``choice`` and prime its prompt.

    ``grounding_dino`` primes the real detector with the full standing vocabulary
    (no per-question boost -- a replay is not scoped to one question). ``fake``
    returns an always-empty :class:`FakeDetector`, useful for exercising this
    script's plumbing without a GPU (it will produce a valid, empty index).
    """
    if choice == "grounding_dino":
        detector = GroundingDinoDetector()
    elif choice == "fake":
        detector = FakeDetector()
    else:
        raise ValueError(f"unknown --detector {choice!r} (expected one of {DETECTOR_CHOICES})")
    refresh_prompt(detector, (), STANDING_VOCAB_NOUNS)
    return detector


def replay(
    bag_path: str,
    *,
    detector_choice: str = "grounding_dino",
    limit_frames: int | None = None,
    stride: int = 1,
    progress_every: int = 20,
    checkpoint_out: str | None = None,
    log=print,
) -> tuple[PerceptionPipeline, int, float]:
    """Drive ``PerceptionPipeline`` over ``bag_path``. Returns (pipeline, n_frames_fed, elapsed_s).

    Iterates ``BagSource(bag_path).frames()`` in bag order, keeping a
    :class:`MessageStore` of the scan channel only (the pano channel is driven
    straight through, one ``pipeline.process`` call per selected PanoFrame) and
    joining each PanoFrame with the latest scan at or before its timestamp via
    :meth:`MessageStore.latest` -- reusing the same lookup the live adapter's
    ``ReplayRobotIO`` uses, rather than hand-rolling a nearest-timestamp search.

    ``stride`` keeps every ``stride``-th PanoFrame (1 = all of them); frames with no
    scan observed yet (before the first ``/registered_scan`` message) are skipped,
    since :meth:`PerceptionPipeline.process` requires both a pano and a scan.

    ``checkpoint_out``, if given, re-writes the index there (via :func:`write_index`)
    on every progress tick -- a long full-bag run (minutes of GPU inference) then
    always has a recoverable partial result on disk instead of losing everything to
    an interruption before the final write.
    """
    detector = build_detector(detector_choice)
    pipeline = PerceptionPipeline(detector, index=BasicSceneIndex([]))

    scan_store = MessageStore()
    pano_seen = 0
    fed = 0
    t0 = time.monotonic()
    last_progress_kf = 0

    for rec in BagSource(bag_path).frames():
        if isinstance(rec.msg, LidarScan):
            scan_store.add(CH_SCAN, rec.t, rec.msg)
            continue
        if not isinstance(rec.msg, PanoFrame):
            continue

        pano_seen += 1
        if (pano_seen - 1) % stride != 0:
            continue

        scan = scan_store.latest(CH_SCAN, rec.t)
        if scan is None:
            continue  # no lidar observed yet -- nothing to fuse detections against

        pipeline.process(rec.msg, scan)
        fed += 1

        kf = pipeline._keyframe_idx  # noqa: SLF001 -- pipeline's own authoritative counter
        if kf - last_progress_kf >= progress_every:
            last_progress_kf = kf
            elapsed = time.monotonic() - t0
            log(
                f"  ... {kf} keyframes processed, {fed} pano frames fed, "
                f"{elapsed:.1f}s elapsed ({len(pipeline.index.all_instances())} instances so far)"
            )
            if checkpoint_out is not None:
                write_index(pipeline, checkpoint_out, bag_path=bag_path, fed=fed)

        if limit_frames is not None and fed >= limit_frames:
            break

    elapsed = time.monotonic() - t0
    return pipeline, fed, elapsed


def write_index(pipeline: PerceptionPipeline, out_path: str, *, bag_path: str, fed: int) -> None:
    """Write ``pipeline.index`` to ``out_path`` in the live instance-index schema.

    Reuses :func:`dump_instance_index` verbatim (issue #84/#89 schema) rather than
    serialising instances by hand, so replay and live indexes are byte-for-byte
    comparable field-for-field. That function only writes when
    :data:`ENV_INSTANCE_DUMP_PATH` is set and always *appends*, so any existing file
    at ``out_path`` is removed first to guarantee a single, fresh record.
    """
    if os.path.exists(out_path):
        os.remove(out_path)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    prior = os.environ.get(ENV_INSTANCE_DUMP_PATH)
    os.environ[ENV_INSTANCE_DUMP_PATH] = out_path
    try:
        dump_instance_index(
            pipeline.index,
            tag="replay_final",
            keyframes_processed=pipeline._keyframe_idx,  # noqa: SLF001
            extra={"bag": bag_path, "pano_frames_fed": fed},
        )
    finally:
        if prior is None:
            os.environ.pop(ENV_INSTANCE_DUMP_PATH, None)
        else:
            os.environ[ENV_INSTANCE_DUMP_PATH] = prior


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bag", required=True, help="bag directory under data/sim_bags/")
    ap.add_argument("--out", required=True, help="output instance_index.jsonl path")
    ap.add_argument(
        "--limit-frames", type=int, default=None,
        help="stop after this many pano frames are FED to the pipeline (smoke runs)",
    )
    ap.add_argument(
        "--stride", type=int, default=1,
        help="keep every Nth pano frame (1 = all of them)",
    )
    ap.add_argument(
        "--detector", choices=DETECTOR_CHOICES, default="grounding_dino",
        help="detector backend (default: grounding_dino)",
    )
    ap.add_argument(
        "--progress-every", type=int, default=20,
        help="print a progress line every N keyframes (default 20)",
    )
    args = ap.parse_args(argv)

    if args.stride < 1:
        ap.error("--stride must be >= 1")

    print(f"replaying {args.bag!r} (detector={args.detector}, stride={args.stride}) ...")
    pipeline, fed, elapsed = replay(
        args.bag,
        detector_choice=args.detector,
        limit_frames=args.limit_frames,
        stride=args.stride,
        progress_every=args.progress_every,
        checkpoint_out=args.out,
    )
    n_instances = len(pipeline.index.all_instances())
    kf = pipeline._keyframe_idx  # noqa: SLF001
    print(
        f"done: {fed} pano frames fed, {kf} keyframes processed, "
        f"{n_instances} instances, {elapsed:.1f}s elapsed"
    )
    write_index(pipeline, args.out, bag_path=args.bag, fed=fed)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
