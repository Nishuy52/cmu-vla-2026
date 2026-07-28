"""Offline replay A/B harness for LIVE numerical-question snapshots.

Perception/fusion/tracking changes need an acceptance test scored against REAL
recorded live data, not the offline GT battery's fully-observed mock scenes
(``core.runner.gt_battery`` — which sees every GT object at ``n_obs=3`` and so
never exercises the answer-time observation gate at all). This tool is that
acceptance test: it rebuilds each scene's answer-time :class:`InstanceRecord`
population from the recorded ``instance_index.jsonl`` dumps
(``core.perception.scene_index.dump_instance_index``), re-parses that scene's
real numerical question, re-counts it exactly the way the live
``NumericalHead`` would, and scores against the true answer key
(``core.runner.gt_battery._true_numerical`` / ``docs/gt_answers_numerical.json``).

We keep rebuilding this in throwaway scripts. One of those throwaway versions
had a defect (a fixed ``min_obs`` instead of the head's dynamic gate) that
produced a WRONG baseline and nearly got a fix merged on a phantom
improvement. This is the committed, tested replacement every future
fusion/tracking change is gated on — harness correctness matters more than
speed.

THE CRITICAL CORRECTNESS REQUIREMENT (do not regress this): ``NumericalHead``
does not count at a fixed ``min_obs``. ``NumericalHead._answer_min_obs``
computes it dynamically per-scene: cold (no instance of the queried noun has
reached ``ESTABLISH_N_OBS`` observations yet) counts everything
(``min_obs=1``); once the noun is established, one-frame ghosts are dropped
(``min_obs=GATE_MIN_OBS``). Passing a fixed ``min_obs`` instead inflates every
count by keeping ghosts that the live head would have dropped. We do not
reimplement that logic here — we import ``core.heads.numerical.NumericalHead``
and call its own ``_answer_min_obs`` helper, so this harness cannot drift from
the head it is meant to gate.

SCHEMA TRAP: in each recorded record, ``instances``/``by_class``/
``total_instances`` are TOP-LEVEL keys, NOT nested under a ``live_instances``
key.

Default mode (variant "raw", the identity transform) reproduces the
raw-recorded-boxes / faithful-gate reference: exact-match 1/15 (arabic_room),
mean|err| ~= 4.9 over the 15-scene 699819 capture. The "degenerate_floor"
variant additionally applies ``core.perception.dimension_priors
.floor_degenerate_aabb`` (issue #125, already on ``main``'s live tracker path
but not baked into these older recordings) to every loaded instance's AABB
before counting, which reproduces the #125-floor reference: exact-match 2/15
(office_2 additionally flips), mean|err| ~= 4.9 (lower than the raw variant).
See ``tools/tests/test_replay_live_numerical.py`` for the exact numbers this
harness is pinned to.

Usage (from the repo root, project venv)::

    python -m tools.replay_live_numerical
    python -m tools.replay_live_numerical --captures reports/cluster_verify/<job> --json
    python -m tools.replay_live_numerical --variant-b mypkg.mymodule:my_transform

A/B a new perception/fusion/tracking change: write a
``Callable[[list[InstanceRecord]], list[InstanceRecord]]`` transform (pure —
input list unmodified, return a new list) either in this file's
``VARIANTS`` registry or anywhere importable, then run with
``--variant-a raw --variant-b <name-or-module:function>``. Default A/B is
``raw`` vs ``degenerate_floor``.

Pure offline dev tool (tools/ — never part of the scored pipeline). CPU only.
"""
from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
for _p in (str(_REPO), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.geometry.toolbox import counting  # noqa: E402
from core.heads.numerical import NumericalHead  # noqa: E402
from core.interfaces import InstanceRecord  # noqa: E402
from core.parsing.regex_tier import parse_regex  # noqa: E402
from core.perception.dimension_priors import floor_degenerate_aabb  # noqa: E402
from core.perception.scene_index import BasicSceneIndex  # noqa: E402
from core.runner.gt_battery import (  # noqa: E402
    DEFAULT_ANSWERS,
    DEFAULT_QUESTIONS,
    _true_numerical,
)

#: Default captures root — the debug dump directory a live/cluster run writes
#: ``debug/<slot>/instance_index.jsonl`` under (``--captures`` overrides this so
#: the harness works against any future run, not just this one).
DEFAULT_CAPTURES_DIR = _REPO / "reports" / "cluster_verify" / "699819"

#: Slot directory names look like ``"<N>_<scene>_nume"`` (e.g.
#: ``"0_arabic_room_nume"`` -> scene ``"arabic_room"``).
_SLOT_RE = re.compile(r"^\d+_(?P<scene>.+)_nume$")


class ReplayError(RuntimeError):
    """Raised for malformed/missing recorded data — never silently guessed."""


# --------------------------------------------------------------------- loading


def scene_name_from_slot(slot: str) -> str:
    """Recover the VLA-3D scene name from a debug-dump slot directory name."""
    m = _SLOT_RE.match(slot)
    return m.group("scene") if m else slot


def discover_slots(captures_dir: Path) -> list[str]:
    """Every ``debug/<slot>/instance_index.jsonl``-bearing slot, sorted."""
    debug_dir = captures_dir / "debug"
    if not debug_dir.is_dir():
        raise ReplayError(f"no debug/ dir under captures root {captures_dir}")
    slots = [
        p.name
        for p in debug_dir.iterdir()
        if p.is_dir() and (p / "instance_index.jsonl").exists()
    ]
    return sorted(slots)


def load_answer_time_record(jsonl_path: Path) -> dict:
    """The record tagged ``"answer_time"`` from a dumped ``instance_index.jsonl``.

    A dump interleaves many ``"periodic"`` records with exactly one
    ``"answer_time"`` record (the one fired when a head was about to publish,
    see ``core.perception.scene_index.dump_instance_index``) — it is NOT
    necessarily the last line in the file (periodic dumps keep going after the
    answer fires). Selecting the last line instead of filtering by tag is
    exactly the defect this harness exists to prevent; if more than one
    ``"answer_time"`` record is present (should not happen), the last one
    wins, matching "most recent snapshot at answer time".
    """
    record: dict | None = None
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("tag") == "answer_time":
                record = row
    if record is None:
        raise ReplayError(f"no 'answer_time' record in {jsonl_path}")
    return record


def record_to_instances(record: dict) -> list[InstanceRecord]:
    """Rebuild the :class:`InstanceRecord` population an answer-time record held.

    SCHEMA TRAP: ``instances``/``by_class``/``total_instances`` are TOP-LEVEL
    keys of the record (not nested under a ``"live_instances"`` key) — see
    ``core.perception.scene_index.dump_instance_index``.
    """
    if "instances" not in record:
        raise ReplayError(
            "record missing top-level 'instances' key "
            "(instances/by_class/total_instances are top-level, not nested)"
        )
    out: list[InstanceRecord] = []
    for inst in record["instances"]:
        aabb_min = np.asarray(inst["aabb_min"], dtype=float)
        aabb_max = np.asarray(inst["aabb_max"], dtype=float)
        centroid = np.asarray(inst["position"], dtype=float)
        out.append(
            InstanceRecord(
                instance_id=int(inst["id"]),
                label=str(inst["label"]),
                score=float(inst["score"]),
                n_obs=int(inst["n_obs"]),
                centroid=centroid,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
            )
        )
    return out


def load_scene_instances(captures_dir: Path, slot: str) -> list[InstanceRecord]:
    path = captures_dir / "debug" / slot / "instance_index.jsonl"
    record = load_answer_time_record(path)
    return record_to_instances(record)


# ------------------------------------------------------------------- variants
#
# A "variant" is a pluggable transform over the loaded InstanceRecord list —
# a Callable[[list[InstanceRecord]], list[InstanceRecord]] — so a future
# fusion/tracking change can be A/B'd against this harness's frozen loading +
# gating + scoring machinery without editing a line of it. Must be pure: never
# mutate the input list/records in place (BasicSceneIndex over the original
# list is reused elsewhere in the caller).

Variant = Callable[[list[InstanceRecord]], list[InstanceRecord]]


def variant_raw(instances: list[InstanceRecord]) -> list[InstanceRecord]:
    """Identity — the recorded boxes exactly as captured, no post-processing."""
    return list(instances)


def variant_degenerate_floor(instances: list[InstanceRecord]) -> list[InstanceRecord]:
    """Apply issue #125's degenerate-extent AABB floor to every instance.

    Reuses ``core.perception.dimension_priors.floor_degenerate_aabb`` (imported,
    never reimplemented) — already on ``main``'s live tracker fusion path, but
    not baked into recordings captured before it landed. Growing a near-zero
    axis about its existing centre never moves the instance's own tracked
    centroid, so ``centroid`` is left untouched.
    """
    out = []
    for rec in instances:
        lo, hi = floor_degenerate_aabb(rec.aabb_min, rec.aabb_max, rec.label)
        out.append(
            InstanceRecord(
                instance_id=rec.instance_id,
                label=rec.label,
                score=rec.score,
                n_obs=rec.n_obs,
                centroid=rec.centroid,
                aabb_min=lo,
                aabb_max=hi,
            )
        )
    return out


#: Name -> transform registry for the ``--variant-a``/``--variant-b`` CLI flags.
VARIANTS: dict[str, Variant] = {
    "raw": variant_raw,
    "degenerate_floor": variant_degenerate_floor,
}


def resolve_variant(spec: str) -> tuple[str, Variant]:
    """``spec`` is a ``VARIANTS`` registry name, or a ``"module:function"`` path."""
    if spec in VARIANTS:
        return spec, VARIANTS[spec]
    if ":" not in spec:
        raise ReplayError(
            f"unknown variant {spec!r} — not in VARIANTS "
            f"({sorted(VARIANTS)}) and not a 'module:function' import path"
        )
    mod_name, _, fn_name = spec.partition(":")
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, fn_name)
    return spec, fn


# ----------------------------------------------------------------- questions


def load_questions(questions_path: Path = DEFAULT_QUESTIONS) -> dict[str, str]:
    """scene name -> its single numerical question text, from ``questions.json``."""
    with open(questions_path, encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[str, str] = {}
    for entry in data:
        nums = entry.get("questions", {}).get("numerical") or []
        if nums:
            out[entry["scene"]] = nums[0]
    return out


def load_answers(answers_path: Path = DEFAULT_ANSWERS) -> dict | None:
    if not Path(answers_path).exists():
        return None
    with open(answers_path, encoding="utf-8") as fh:
        return json.load(fh)


# -------------------------------------------------------------------- scoring


@dataclass
class SceneResult:
    scene: str
    question: str
    truth: int | None
    count: int | None
    min_obs: int
    census: dict[str, int]
    note: str

    @property
    def err(self) -> int | None:
        if self.truth is None or self.count is None:
            return None
        return abs(self.count - self.truth)

    @property
    def exact_match(self) -> bool | None:
        if self.truth is None or self.count is None:
            return None
        return self.count == self.truth


def _census(instances: list[InstanceRecord]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rec in instances:
        out[rec.label] = out.get(rec.label, 0) + 1
    return out


def score_scene(
    scene: str,
    question: str,
    instances: list[InstanceRecord],
    answers: dict | None,
) -> SceneResult:
    """Count ``question`` over ``instances`` exactly the way the live head would.

    Rebuilds a :class:`BasicSceneIndex`, parses the question with the SAME
    regex tier the live pipeline uses, computes the answer-time observation
    floor via the live ``NumericalHead``'s own ``_answer_min_obs`` helper (the
    critical correctness requirement — never a fixed ``min_obs``), and counts
    with ``core.geometry.toolbox.counting``.
    """
    index = BasicSceneIndex(instances)
    census = _census(instances)
    plan = parse_regex(question)
    if plan.target is None:
        return SceneResult(scene, question, None, None, 0, census, "no target parsed")
    head = NumericalHead(plan=plan)
    min_obs = head._answer_min_obs(index)  # noqa: SLF001 — the head's own helper, reused not reimplemented
    result = counting(plan.target, index, min_obs=min_obs)
    truth, _match, _src, note = _true_numerical(answers, scene, question, result.count)
    return SceneResult(scene, question, truth, result.count, min_obs, census, note)


@dataclass
class VariantSummary:
    name: str
    results: list[SceneResult]

    @property
    def scored(self) -> list[SceneResult]:
        return [r for r in self.results if r.truth is not None and r.count is not None]

    @property
    def exact_matches(self) -> int:
        return sum(1 for r in self.scored if r.exact_match)

    @property
    def n_scored(self) -> int:
        return len(self.scored)

    @property
    def mean_abs_err(self) -> float | None:
        scored = self.scored
        if not scored:
            return None
        return sum(r.err for r in scored) / len(scored)


def run_replay(
    captures_dir: Path,
    variant_a: Variant,
    variant_b: Variant,
    *,
    scenes: list[str] | None = None,
    questions_path: Path = DEFAULT_QUESTIONS,
    answers_path: Path = DEFAULT_ANSWERS,
) -> tuple[VariantSummary, VariantSummary]:
    """Run both variants over every recorded scene under ``captures_dir``.

    ``scenes`` (optional) restricts to these scene names (VLA-3D scene names,
    e.g. ``"arabic_room"``, not slot directory names); default is every slot
    discovered under ``captures_dir``.
    """
    questions = load_questions(questions_path)
    answers = load_answers(answers_path)
    slots = discover_slots(captures_dir)

    results_a: list[SceneResult] = []
    results_b: list[SceneResult] = []
    for slot in slots:
        scene = scene_name_from_slot(slot)
        if scenes is not None and scene not in scenes:
            continue
        question = questions.get(scene)
        if question is None:
            results_a.append(SceneResult(scene, "", None, None, 0, {}, "no numerical question for scene"))
            results_b.append(SceneResult(scene, "", None, None, 0, {}, "no numerical question for scene"))
            continue
        raw_instances = load_scene_instances(captures_dir, slot)
        results_a.append(score_scene(scene, question, variant_a(raw_instances), answers))
        results_b.append(score_scene(scene, question, variant_b(raw_instances), answers))
    return VariantSummary("a", results_a), VariantSummary("b", results_b)


# ----------------------------------------------------------------------- CLI


def _fmt(v) -> str:
    return "-" if v is None else str(v)


def build_report(
    summary_a: VariantSummary,
    summary_b: VariantSummary,
    name_a: str,
    name_b: str,
) -> str:
    lines = []
    lines.append(f"variant A = {name_a}    variant B = {name_b}")
    header = (
        f"{'scene':<18} {'truth':>5} | {'A':>4} {'errA':>5} | {'B':>4} {'errB':>5}  question"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for ra, rb in zip(summary_a.results, summary_b.results):
        lines.append(
            f"{ra.scene:<18} {_fmt(ra.truth):>5} | "
            f"{_fmt(ra.count):>4} {_fmt(ra.err):>5} | "
            f"{_fmt(rb.count):>4} {_fmt(rb.err):>5}  {ra.question}"
        )
    lines.append("")
    for name, summ in ((name_a, summary_a), (name_b, summary_b)):
        mean_err = summ.mean_abs_err
        mean_str = "n/a" if mean_err is None else f"{mean_err:.3f}"
        lines.append(
            f"[{name}] exact-match {summ.exact_matches}/{summ.n_scored}   "
            f"mean|err| {mean_str}"
        )
    return "\n".join(lines)


def build_json(summary_a: VariantSummary, summary_b: VariantSummary, name_a: str, name_b: str) -> dict:
    def dump(summ: VariantSummary) -> dict:
        return {
            "exact_matches": summ.exact_matches,
            "n_scored": summ.n_scored,
            "mean_abs_err": summ.mean_abs_err,
            "scenes": [
                {
                    "scene": r.scene,
                    "question": r.question,
                    "truth": r.truth,
                    "count": r.count,
                    "err": r.err,
                    "exact_match": r.exact_match,
                    "min_obs": r.min_obs,
                    "census": r.census,
                    "note": r.note,
                }
                for r in summ.results
            ],
        }

    return {"variant_a": name_a, "variant_b": name_b, "a": dump(summary_a), "b": dump(summary_b)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--captures",
        default=str(DEFAULT_CAPTURES_DIR),
        help="captures root holding debug/<slot>/instance_index.jsonl (default: %(default)s)",
    )
    ap.add_argument("--variant-a", default="raw", help="registry name or 'module:function' (default: raw)")
    ap.add_argument(
        "--variant-b",
        default="degenerate_floor",
        help="registry name or 'module:function' (default: degenerate_floor)",
    )
    ap.add_argument("--scenes", nargs="*", default=None, help="restrict to these scene names")
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="questions.json path")
    ap.add_argument("--answers", default=str(DEFAULT_ANSWERS), help="answer-key json path")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of a table")
    args = ap.parse_args(argv)

    name_a, variant_a = resolve_variant(args.variant_a)
    name_b, variant_b = resolve_variant(args.variant_b)

    summary_a, summary_b = run_replay(
        Path(args.captures),
        variant_a,
        variant_b,
        scenes=args.scenes,
        questions_path=Path(args.questions),
        answers_path=Path(args.answers),
    )

    if args.json:
        print(json.dumps(build_json(summary_a, summary_b, name_a, name_b), indent=2))
    else:
        print(build_report(summary_a, summary_b, name_a, name_b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
