"""Phase-1 conformance check: drive the REAL parse ladder against a live local-LLM slot.

Loads the ``VLA_LLM_LOCAL_*`` env slot (see ``core.llm.config``) exactly as the ROS
adapter does (``load_config()`` + ``build_chat_fns()``), takes a small deterministic
sample of the 75 training questions (same source file the ``gt_battery`` harness
reads), and runs each one through ``core.parsing.ladder.parse()`` unmodified. For
each question we record whether the LLM tier produced a schema-valid ``Plan`` on the
first try, whether the ladder's one repair round was needed/used, whether the reply
fell all the way through to the deterministic regex floor, and the wall-clock latency
of the LLM work.

Only the local slot is exercised here (Phase 1 is about proving the local bridge, not
primary/secondary) — the ladder is called with ``tier_names=("local",)`` so a
single-slot chat_fns list is stamped correctly (see the module docstring note on
``core.llm.config.build_chat_fns`` below for why the *default* tier_names would
mislabel a local-only config as "api").

Usage (from the repo root, with a local slot already exported)::

    python -m tools.llm_conformance
    python -m tools.llm_conformance --n 10 --seed 0 --out reports/local_llm_phase1/conformance.md

Pure stdlib + the already-installed provider stack. Offline dev tool — not part of
the scored pipeline.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tools  # noqa: F401 -- inserts <repo>/src onto sys.path

from core.llm.config import build_chat_fns, load_config
from core.parsing import ladder
from core.runner.gt_battery import DEFAULT_QUESTIONS

DEFAULT_N = 10
DEFAULT_SEED = 0
#: PASS bar (Phase 1 spec): at least this fraction of the sample must come back
#: schema-valid via the LLM tier (not the regex floor), at temperature 0.
PASS_FRACTION = 0.8


# --------------------------------------------------------------------------- pure helpers


def load_all_questions(questions_path: Any = DEFAULT_QUESTIONS) -> list[dict[str, str]]:
    """Flatten ``questions.json`` (the same file ``gt_battery`` scores) into rows.

    Each row is ``{"scene": ..., "qtype": ..., "text": ...}``. This mirrors exactly how
    :func:`core.runner.gt_battery.run_gt_battery` walks the file (one entry per scene,
    ``entry["questions"]`` keyed by qtype) — we just don't need a GT scene folder to
    read the question text, so we skip the scene-matching machinery.
    """
    data = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    out: list[dict[str, str]] = []
    for entry in data:
        scene = entry["scene"]
        for qtype, texts in entry.get("questions", {}).items():
            for text in texts:
                out.append({"scene": scene, "qtype": qtype, "text": text})
    return out


def sample_questions(
    all_questions: list[dict[str, str]], n: int = DEFAULT_N, seed: int = DEFAULT_SEED
) -> list[dict[str, str]]:
    """A small deterministic sample, seeded for reproducibility across runs.

    ``n >= len(all_questions)`` returns every question (still deterministically
    ordered by the seeded shuffle, so a report diff across code changes is stable).
    """
    pool = list(all_questions)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[: min(n, len(pool))]


class RealClock:
    """Wall-clock ``Clock`` (monotonic seconds) for driving the real ladder live."""

    def now(self) -> float:
        return time.monotonic()


class _CallRecorder:
    """Wraps one ``ChatFn``, counting invocations and per-call latency.

    The ladder calls a tier's ``ChatFn`` once for the first attempt and, if that
    reply fails validation, once more for the repair round — so ``calls == 2`` after
    one ``parse()`` run means the repair round was exercised; ``calls == 1`` means the
    first attempt already validated (or the tier was never reached / errored out
    before a second call, which the caller distinguishes via ``parse_tier``).
    """

    def __init__(self, fn) -> None:
        self._fn = fn
        self.calls = 0
        self.latencies_s: list[float] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        t0 = time.monotonic()
        try:
            return self._fn(messages)
        finally:
            self.latencies_s.append(time.monotonic() - t0)
            self.calls += 1


@dataclass
class ConformanceRow:
    scene: str
    qtype: str
    question: str
    parse_tier: str
    schema_valid_via_llm: bool
    repair_round_used: bool
    regex_floor_used: bool
    n_llm_calls: int
    latency_s: float
    error: str = ""


def run_one(question_row: dict[str, str], local_fn) -> ConformanceRow:
    """Run one question through the real ladder with only the local tier wired."""
    rec = _CallRecorder(local_fn)
    clock = RealClock()
    error = ""
    try:
        plan = ladder.parse(question_row["text"], [rec], clock, tier_names=("local",))
        parse_tier = plan.parse_tier
    except Exception as exc:  # noqa: BLE001 — the ladder itself must never raise, but guard the tool
        parse_tier = "error"
        error = repr(exc)

    return ConformanceRow(
        scene=question_row["scene"],
        qtype=question_row["qtype"],
        question=question_row["text"],
        parse_tier=parse_tier,
        schema_valid_via_llm=(parse_tier == "local"),
        repair_round_used=(rec.calls >= 2),
        regex_floor_used=(parse_tier == "regex"),
        n_llm_calls=rec.calls,
        latency_s=round(sum(rec.latencies_s), 3),
        error=error,
    )


# --------------------------------------------------------------------------- report


def build_report(rows: list[ConformanceRow], *, model: str, base_url: str) -> str:
    n = len(rows)
    n_valid = sum(1 for r in rows if r.schema_valid_via_llm)
    n_repair = sum(1 for r in rows if r.repair_round_used)
    n_floor = sum(1 for r in rows if r.regex_floor_used)
    latencies = [r.latency_s for r in rows if r.n_llm_calls > 0]
    mean_lat = round(sum(latencies) / len(latencies), 3) if latencies else None
    passed = n_valid >= PASS_FRACTION * n if n else False

    lines = [
        f"# Local-LLM Phase 1 conformance ({datetime.now(timezone.utc).isoformat(timespec='seconds')})",
        "",
        f"model={model} base_url={base_url} temperature=0.0 n={n}",
        "",
        f"**Result: {'PASS' if passed else 'FAIL'}** — "
        f"{n_valid}/{n} schema-valid via the LLM tier "
        f"(bar: >= {int(PASS_FRACTION * 100)}%), "
        f"{n_repair} repair round(s) used, {n_floor} fell to the regex floor.",
        f"Mean LLM latency: {mean_lat if mean_lat is not None else 'n/a'} s.",
        "",
        "| Scene | QType | Tier | Valid-via-LLM | Repair used | Floor | Calls | Latency (s) | Question |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        q = r.question if len(r.question) <= 60 else r.question[:57] + "..."
        lines.append(
            f"| {r.scene} | {r.qtype} | {r.parse_tier} | {r.schema_valid_via_llm} | "
            f"{r.repair_round_used} | {r.regex_floor_used} | {r.n_llm_calls} | "
            f"{r.latency_s} | {q} |"
        )
        if r.error:
            lines.append(f"|  |  |  |  |  |  |  |  | error: {r.error} |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tools.llm_conformance",
        description="Drive the real parse ladder against the local-LLM slot on a "
        "sample of training questions (Phase 1 exit gate).",
    )
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="sample size (default 10)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="sample seed (default 0)")
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="questions.json path")
    ap.add_argument("--out", default=None, help="markdown report path (default: stdout only)")
    ap.add_argument(
        "--jsonl-out", default=None, help="optional raw per-question JSONL output path"
    )
    args = ap.parse_args(argv)

    config = load_config()
    if config.local is None:
        print(
            "tools.llm_conformance: no local LLM slot configured "
            "(VLA_LLM_LOCAL_KIND/BASE_URL/MODEL env vars not set)",
            file=sys.stderr,
        )
        return 2
    fns = build_chat_fns(config)
    if not fns:
        print(
            "tools.llm_conformance: local slot configured but build_chat_fns() "
            "returned no callable (bad kind / missing field) — check env vars",
            file=sys.stderr,
        )
        return 2
    local_fn = fns[0]  # only the local slot is wired for this tool

    all_qs = load_all_questions(args.questions)
    sample = sample_questions(all_qs, n=args.n, seed=args.seed)

    rows = [run_one(q, local_fn) for q in sample]

    report = build_report(rows, model=config.local.model, base_url=config.local.base_url or "")
    print(report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"wrote {out_path}")
    if args.jsonl_out:
        jsonl_path = Path(args.jsonl_out)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(asdict(r)) + "\n")
        print(f"wrote {jsonl_path}")

    n_valid = sum(1 for r in rows if r.schema_valid_via_llm)
    passed = bool(rows) and n_valid >= PASS_FRACTION * len(rows)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
