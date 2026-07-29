"""Issue #151 diagnostic: per-candidate rejection breakdown for the relation
clause on banked live-quality boxes.

The numerical head's live under-count (9/15 scenes wrong) was suspected to be
a perception-coverage problem (targets never detected). Job 701982's funnel
proved otherwise: every target/anchor/disambiguator noun the failing
questions need IS present in quantity at answer time. This tool replays the
EXACT clause-evaluation path (``_resolve_anchor`` -> disambiguator ->
``_eval_clause`` -> the binary predicate, e.g. ``on()``) used by
``core.geometry.toolbox.counting`` over the banked
``reports/cluster_verify/<job>/debug/<slot>/{resolved_plan,instance_index}.jsonl``
dumps, and prints WHICH test rejected each candidate target instance, with
the real numbers (overlap fraction vs threshold, vertical band vs bottom,
anchor-larger check, chosen disambiguated anchor id/centroid).

Read-only measurement tool. Does not change any toolbox.py behaviour and
does not tune any threshold — it only calls the existing predicate/resolution
functions and reports their own PredResult.explanation strings plus the
raw numbers needed to answer issue #151's four questions.

Usage (from repo root, project venv)::

    python -m tools.diagnose_relation_clause --job 701982 --slot 12_office_1_nume
    python -m tools.diagnose_relation_clause --job 701982 --slot 2_home_building_1_nume
    python -m tools.diagnose_relation_clause --job 701982 --slot 10_livingroom_4_nume

``--job``'s debug/ dump is read from ``../cmu-vla-2026/reports/cluster_verify``
by default (the banked captures for job 701982 are only checked out there,
not mirrored into this worktree's reports/ tree) — override with
``--captures-root`` if needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / "src"
for _p in (str(_REPO), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.geometry import toolbox as tb  # noqa: E402
from core.plan_schema import Anchor, Clause, Pred, TargetSpec  # noqa: E402
from core.perception.scene_index import BasicSceneIndex  # noqa: E402

# tools/replay_live_numerical.py's loader — reused, not reimplemented (per the
# task brief: "extend or reuse ... rather than writing a third replayer").
from tools.replay_live_numerical import (  # noqa: E402
    load_scene_instances,
    scene_name_from_slot,
)

#: Default location of the banked cluster-verify captures. job 701982's debug/
#: dump lives only under the sibling cmu-vla-2026 checkout in this session
#: (this worktree's reports/ tree does not have it) — never written to.
DEFAULT_CAPTURES_ROOT = Path("/home/jason/cmu_ws/cmu-vla-2026/reports/cluster_verify")


def _anchor_from_json(d: dict) -> Anchor:
    disamb = d.get("disambiguator")
    return Anchor(
        noun=d["noun"],
        raw=d.get("raw", ""),
        attributes=list(d.get("attributes") or []),
        disambiguator=_clause_from_json(disamb) if disamb else None,
    )


def _clause_from_json(d: dict) -> Clause:
    return Clause(
        pred=Pred(d["pred"]),
        anchors=[_anchor_from_json(a) for a in d["anchors"]],
        negated=bool(d.get("negated", False)),
    )


def _target_from_json(d: dict) -> TargetSpec:
    return TargetSpec(
        noun=d["noun"],
        raw=d.get("raw", ""),
        attributes=list(d.get("attributes") or []),
        clauses=[_clause_from_json(c) for c in d.get("clauses") or []],
    )


def load_resolved_plan(captures_dir: Path, slot: str) -> dict:
    path = captures_dir / "debug" / slot / "resolved_plan.jsonl"
    record = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("tag") == "answer_time":
                record = row
    if record is None:
        raise RuntimeError(f"no answer_time record in {path}")
    return record


def fmt_inst(rec) -> str:
    return (
        f"id={rec.instance_id} '{rec.label}' n_obs={rec.n_obs} "
        f"centroid=({rec.centroid[0]:.2f},{rec.centroid[1]:.2f},{rec.centroid[2]:.2f}) "
        f"aabb=[{rec.aabb_min[0]:.2f},{rec.aabb_min[1]:.2f},{rec.aabb_min[2]:.2f}]-"
        f"[{rec.aabb_max[0]:.2f},{rec.aabb_max[1]:.2f},{rec.aabb_max[2]:.2f}]"
    )


def diagnose(captures_dir: Path, slot: str, min_obs: int = 1) -> None:
    scene = scene_name_from_slot(slot)
    plan_record = load_resolved_plan(captures_dir, slot)
    plan = plan_record["plan"]
    question = plan["question_raw"]
    target = _target_from_json(plan["target"])

    instances = load_scene_instances(captures_dir, slot)
    index = BasicSceneIndex(instances)
    th = tb.DEFAULT_THRESHOLDS

    print(f"=== {scene} ({slot}) ===")
    print(f"question: {question!r}")
    print(f"target noun: {target.noun!r}  clauses: {[c.pred.value for c in target.clauses]}")
    print()

    base = tb._match_noun(index, target.noun)
    pool = [c for c in base if tb._attrs_match(c, target.attributes, base, th)]
    print(f"target pool ('{target.noun}' + attrs): {len(pool)} instances")
    for r in pool:
        print(f"  {fmt_inst(r)}")
    print()

    hard_clauses = [c for c in target.clauses if c.pred not in tb._SUPERLATIVE_PREDS]
    if not hard_clauses:
        print("no non-superlative (hard) clauses on the target -- nothing to diagnose.")
        return

    for clause in hard_clauses:
        print(f"--- clause: {clause.pred.value}(target, {[a.noun for a in clause.anchors]}) "
              f"negated={clause.negated} ---")
        for anchor in clause.anchors:
            base_anchor_cands = tb._match_anchor_noun(index, anchor.noun)
            print(f"  anchor '{anchor.noun}' base pool: {len(base_anchor_cands)} instances")
            if anchor.attributes:
                cls_pool = base_anchor_cands
                base_anchor_cands = [
                    c for c in base_anchor_cands
                    if tb._attrs_match(c, anchor.attributes, cls_pool, th)
                ]
                print(f"    after attrs {anchor.attributes}: {len(base_anchor_cands)}")

            resolved = tb._resolve_anchor(anchor, index, th, audit=[])
            print(f"  anchor '{anchor.noun}' RESOLVED (post-disambiguator) pool: "
                  f"{len(resolved)} instances")
            for r in resolved:
                print(f"    {fmt_inst(r)}")

            if anchor.disambiguator is not None:
                disamb = anchor.disambiguator
                print(f"  disambiguator: {disamb.pred.value}"
                      f"({[a.noun for a in disamb.anchors]})")
                if disamb.pred in tb._SUPERLATIVE_PREDS:
                    sub_recs = tb._resolve_anchor(disamb.anchors[0], index, th, audit=[])
                    print(f"    disambiguator anchor '{disamb.anchors[0].noun}': "
                          f"{len(sub_recs)} resolved")
                    for r in sub_recs:
                        print(f"      {fmt_inst(r)}")
                    if sub_recs and base_anchor_cands:
                        ranked = (
                            tb.closest_to(base_anchor_cands, sub_recs[0], th)
                            if disamb.pred is Pred.CLOSEST_TO
                            else tb.farthest_from(base_anchor_cands, sub_recs[0], th)
                        )
                        print(f"    ranking: {ranked.explanation}")
                        by_id = {c.instance_id: c for c in base_anchor_cands}
                        print(f"    ALL candidate distances (ascending):")
                        for iid in ranked.order:
                            print(f"      id={iid} dist={ranked.distances[iid]:.3f}m "
                                  f"{'<-- CHOSEN' if iid == ranked.order[0] else ''}")
                else:
                    print("    (non-superlative disambiguator -- filters the anchor pool)")

        # Now, per target candidate: evaluate the clause and show the real
        # numbers behind pass/fail against every resolved anchor instance.
        print()
        print(f"  per-candidate breakdown for clause '{clause.pred.value}':")
        fn = tb._BINARY_PREDS.get(clause.pred)
        anchor_recs_per_anchor = [
            tb._resolve_anchor(a, index, th, audit=[]) for a in clause.anchors
        ]
        n_pass = 0
        for cand in pool:
            if clause.pred is Pred.BETWEEN or fn is None:
                result = tb._eval_clause(cand, clause, index, th, audit=[])
                print(f"    id={cand.instance_id}: passed={result.passed} "
                      f"score={result.score:.3f} margin={result.margin:.3f}")
                print(f"      {result.explanation}")
                if result.passed:
                    n_pass += 1
                continue

            recs = anchor_recs_per_anchor[0]
            best = None
            best_passing = None
            per_anchor_results = []
            for b in recs:
                r = fn(cand, b, th)
                per_anchor_results.append((b, r))
                if best is None or r.score > best.score:
                    best = r
                if r.passed and (best_passing is None or r.score > best_passing.score):
                    best_passing = r
            chosen = best_passing if best_passing is not None else best
            final = tb._apply_negation(chosen, clause) if chosen is not None else None
            passed = final.passed if final is not None else False
            if passed:
                n_pass += 1
            print(f"    id={cand.instance_id} n_obs={cand.n_obs}: overall passed={passed}")
            for b, r in per_anchor_results:
                print(f"      vs anchor id={b.instance_id} '{b.label}': "
                      f"passed={r.passed} score={r.score:.3f}")
                print(f"        {r.explanation}")
        print()
        print(f"  clause '{clause.pred.value}': {n_pass}/{len(pool)} target candidates pass")
        print()

    survivors = tb._filter_and(pool, hard_clauses, index, th, audit=[])
    ids = {r.instance_id for r in survivors if r.n_obs >= min_obs}
    print(f"FINAL counting() result: {len(ids)} (min_obs={min_obs})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job", default="701982")
    ap.add_argument("--slot", required=True)
    ap.add_argument("--captures-root", type=Path, default=DEFAULT_CAPTURES_ROOT)
    ap.add_argument("--min-obs", type=int, default=1)
    args = ap.parse_args()

    captures_dir = args.captures_root / args.job
    diagnose(captures_dir, args.slot, min_obs=args.min_obs)


if __name__ == "__main__":
    main()
