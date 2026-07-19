"""Tests for the CV calibration sweep harness (core.runner.cvsweep).

Pure/hand-built where possible (fold math, objective reduction, cache, stability
table); real-data guards (`requires_full_unity` / `requires_loft`) gate the sweep
smoke and vocab-bridge integration tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.runner import cvsweep as CV
from core.runner import gt_battery as GB
from core.runner.cvsweep import (
    SceneEvaluator,
    SceneQScores,
    SweepSpec,
    config_hash,
    default_sweep_spec,
    make_folds,
    run_cv_sweep,
    write_report,
    _aggregate_stability,
    _fit_folds,
)

from tests.groundtruth.conftest import (
    requires_full_unity,
    requires_loft,
    FULL_UNITY_ROOT,
    LOFT_DIR,
    QUESTIONS_JSON,
    QUESTIONS_DIR,
    UNITY_SAMPLE_ROOT,
)

_FIFTEEN = [
    "arabic_room", "chinese_room", "home_building_1", "home_building_2",
    "hotel_room_1", "hotel_room_2", "japanese_room", "livingroom_1",
    "livingroom_2", "livingroom_3", "livingroom_4", "loft", "office_1",
    "office_2", "studio",
]


# --------------------------------------------------------------------------- folds


def test_folds_partition_is_disjoint_and_complete():
    folds = make_folds(_FIFTEEN, n_folds=5, holdout_size=3, seed=0)
    assert len(folds) == 5
    assert all(len(f) == 3 for f in folds)
    flat = [s for f in folds for s in f]
    assert sorted(flat) == sorted(_FIFTEEN)  # complete
    assert len(set(flat)) == len(flat)  # disjoint (each scene held out exactly once)


def test_folds_deterministic_under_seed():
    a = make_folds(_FIFTEEN, seed=7)
    b = make_folds(_FIFTEEN, seed=7)
    assert a == b
    # a different seed gives a different partition (shuffle actually shuffles)
    c = make_folds(_FIFTEEN, seed=8)
    assert c != a


def test_folds_reject_incompatible_geometry():
    with pytest.raises(ValueError):
        make_folds(_FIFTEEN[:14], n_folds=5, holdout_size=3)  # 14 != 5*3


def test_fit_folds_falls_back_to_leave_one_out():
    # exact partition kept
    assert _fit_folds(15, 5, 3) == (5, 3)
    # smoke subset that doesn't partition -> leave-one-out
    assert _fit_folds(2, 5, 3) == (2, 1)
    assert _fit_folds(7, 5, 3) == (7, 1)


# --------------------------------------------------------------------------- objective math


def test_sceneqscores_composite_and_exclusion():
    s = SceneQScores(
        numerical_earned=1.0, numerical_available=1.0, numerical_excluded=1,
        or_earned=0.0, or_available=2.0, or_excluded=3,
        if_earned=3.0, if_available=6.0, if_excluded=1,
    )
    # earned = 1 + 0 + 3 = 4; available = 1 + 2 + 6 = 9
    assert s.earned == pytest.approx(4.0)
    assert s.available == pytest.approx(9.0)
    assert s.composite() == pytest.approx(4.0 / 9.0)
    # excluded counts are carried, not folded into available
    assert s.numerical_excluded == 1 and s.or_excluded == 3 and s.if_excluded == 1


def test_sceneqscores_add_accumulates_all_fields():
    a = SceneQScores(or_earned=2.0, or_available=2.0, or_excluded=1)
    b = SceneQScores(or_earned=0.0, or_available=2.0, or_excluded=2, if_available=6.0)
    a.add(b)
    assert a.or_earned == pytest.approx(2.0)
    assert a.or_available == pytest.approx(4.0)
    assert a.or_excluded == 3
    assert a.if_available == pytest.approx(6.0)


def test_composite_zero_when_nothing_available():
    assert SceneQScores().composite() == 0.0  # no scoreable/aligned questions -> 0, no div-by-zero


def test_strict_independent_requires_referential_source():
    """NUM-F6/F7: only a referential (relation-aware) independent count is evidence — a
    class-only or scene-graph-sourced count is excluded, never accepted as a fallback."""
    from core.runner.gt_battery import GTQuestionScore
    r1 = GTQuestionScore(
        scene="s", qtype="numerical", question="q",
        gt_count_independent=4, independent_source="referential", gt_count_scenegraph=9,
    )
    assert CV._strict_independent(r1) == 4  # referential -> evidence

    # NOT referential (e.g. class_only) -> excluded even though a count value is present
    r2 = GTQuestionScore(
        scene="s", qtype="numerical", question="q",
        gt_count_independent=9, independent_source="class_only", gt_count_scenegraph=9,
    )
    assert CV._strict_independent(r2) is None

    # scene-graph-only (no independent_source at all) -> excluded, never falls back
    r3 = GTQuestionScore(
        scene="s", qtype="numerical", question="q",
        gt_count_independent=None, gt_count_scenegraph=9,
    )
    assert CV._strict_independent(r3) is None

    # no opinion whatsoever -> excluded
    r4 = GTQuestionScore(scene="s", qtype="numerical", question="q")
    assert CV._strict_independent(r4) is None


# --------------------------------------------------------------------------- config hash


def test_config_hash_order_independent_and_type_sensitive():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    # int vs float are distinct configs
    assert config_hash({"a": 1}) != config_hash({"a": 1.0})
    # empty (baseline) is stable
    assert config_hash({}) == config_hash({})


# --------------------------------------------------------------------------- spec


def test_default_sweep_spec_keys_are_wireable():
    spec = default_sweep_spec()
    spec.validate()  # must not raise: every geometry key is a real, live Calibration field
    assert all(len(v) >= 3 for v in spec.params.values())  # 3-5 values each


def test_default_sweep_spec_excludes_min_obs():
    """NUM-F7: counting.min_obs was dropped — provably inert on GT data (n_obs=3 always),
    so it must never reappear in the default sweep space."""
    spec = default_sweep_spec()
    assert "counting.min_obs" not in spec.params
    assert not any(k.startswith("counting.") for k in spec.params)


def test_spec_validate_rejects_unknown_key():
    bad = SweepSpec(params={"geometry.does_not_exist": [1, 2, 3]})
    with pytest.raises(KeyError):
        bad.validate()


def test_spec_sample_is_seeded_and_within_candidates():
    import random
    spec = default_sweep_spec()
    s1 = spec.sample(random.Random(3))
    s2 = spec.sample(random.Random(3))
    assert s1 == s2  # seeded reproducibility
    for k, v in s1.items():
        assert v in spec.params[k]


# --------------------------------------------------------------------------- stability table


def test_aggregate_stability_modal_majority_recommends():
    from core.runner.cvsweep import FoldResult
    spec = SweepSpec(params={"geometry.near_floor": [1.0, 1.2, 2.0]})
    # 3 folds pick 2.0, 1 fold picks default (1.2), 1 picks 1.0
    folds = [
        FoldResult(0, ["a"], 0.5, 0.5, {"geometry.near_floor": 2.0}, "h"),
        FoldResult(1, ["b"], 0.5, 0.5, {"geometry.near_floor": 2.0}, "h"),
        FoldResult(2, ["c"], 0.5, 0.5, {"geometry.near_floor": 2.0}, "h"),
        FoldResult(3, ["d"], 0.5, 0.5, {"geometry.near_floor": 1.0}, "h"),
        FoldResult(4, ["e"], 0.5, 0.5, {}, "h"),  # default 1.2
    ]
    stability, rec, over = _aggregate_stability(spec, folds)
    # 2.0 recurs in 3/5 folds -> strict majority -> recommended and moved off default
    assert rec["geometry.near_floor"] == 2.0
    assert over["geometry.near_floor"] == 2.0
    assert stability["geometry.near_floor"]["2.0"] == 3


def test_aggregate_stability_no_consensus_keeps_default():
    from core.runner.cvsweep import FoldResult
    spec = SweepSpec(params={"geometry.near_floor": [1.0, 1.2, 2.0]})
    # tie: 2 folds pick 2.0, 2 pick 1.0, 1 default -> no strict majority
    folds = [
        FoldResult(0, ["a"], 0.5, 0.5, {"geometry.near_floor": 2.0}, "h"),
        FoldResult(1, ["b"], 0.5, 0.5, {"geometry.near_floor": 2.0}, "h"),
        FoldResult(2, ["c"], 0.5, 0.5, {"geometry.near_floor": 1.0}, "h"),
        FoldResult(3, ["d"], 0.5, 0.5, {"geometry.near_floor": 1.0}, "h"),
        FoldResult(4, ["e"], 0.5, 0.5, {}, "h"),
    ]
    _stab, rec, over = _aggregate_stability(spec, folds)
    assert rec["geometry.near_floor"] == "unstable — keep default"
    assert "geometry.near_floor" not in over  # nothing moved


# --------------------------------------------------------------------------- objective weighting


def test_objective_weights_rubric_x6_strict_x1_iou_x2():
    """Points-weighted composite on a synthetic mini-battery: NUMERICAL 1pt (strict
    independent agreement), OBJECT_REFERENCE 2pt (IoU hit), INSTRUCTION_FOLLOWING 6pt
    (rubric score), summed earned/available exactly as the challenge pays."""
    s = SceneQScores()
    # one numerical hit (strict-independent match) -> 1/1
    s.numerical_earned, s.numerical_available = 1.0, 1.0
    # one OR hit (IoU >= 0.25) -> 2/2
    s.or_earned, s.or_available = 2.0, 2.0
    # one IF question at rubric_score=0.5 -> 3/6 (6 * 0.5)
    s.if_earned, s.if_available = 3.0, 6.0

    assert s.earned == pytest.approx(1.0 + 2.0 + 3.0)
    assert s.available == pytest.approx(1.0 + 2.0 + 6.0)
    assert s.composite() == pytest.approx(6.0 / 9.0)

    # weighting constants match the challenge point values used by score_scene
    assert CV._PTS_NUMERICAL == pytest.approx(1.0)
    assert CV._PTS_OBJECT_REFERENCE == pytest.approx(2.0)
    assert CV._PTS_INSTRUCTION_FOLLOWING == pytest.approx(6.0)


def test_objective_excludes_class_only_and_scenegraph_only_numerical_rows():
    """A class-only or scene-graph-only numerical row must never enter the numerical
    numerator/denominator — mirrors gt_battery.aggregate's referential-only filter."""
    from core.runner.gt_battery import GTQuestionScore

    class_only = GTQuestionScore(
        scene="s", qtype="numerical", question="q1",
        our_count=5, gt_count_independent=5, independent_source="class_only",
    )
    scenegraph_only = GTQuestionScore(
        scene="s", qtype="numerical", question="q2",
        our_count=3, gt_count_independent=None, gt_count_scenegraph=3,
    )
    referential = GTQuestionScore(
        scene="s", qtype="numerical", question="q3",
        our_count=2, gt_count_independent=2, independent_source="referential",
    )
    assert CV._strict_independent(class_only) is None
    assert CV._strict_independent(scenegraph_only) is None
    assert CV._strict_independent(referential) == 2


# --------------------------------------------------------------------------- grid/ledger consistency


def test_every_swept_geometry_key_is_a_live_thresholds_field():
    """Grid-vs-ledger guard: every ``geometry.<field>`` key in the default sweep spec must
    exist as a real field on core.geometry.toolbox.Thresholds (enumerated dynamically, no
    hardcoded list) — this FAILS if a future rename/retirement breaks the grid."""
    from dataclasses import fields as dc_fields
    from core.geometry.toolbox import Thresholds

    live_fields = {f.name for f in dc_fields(Thresholds)}
    spec = default_sweep_spec()
    geometry_keys = [k for k in spec.params if k.startswith("geometry.")]
    assert geometry_keys, "expected at least one geometry.* key in the default spec"
    for key in geometry_keys:
        _, _, field_name = key.partition(".")
        assert field_name in live_fields, (
            f"sweep key {key!r} does not name a live Thresholds field "
            f"(live fields: {sorted(live_fields)})"
        )


def test_every_swept_key_is_a_live_calibration_ledger_field():
    """Same guard at the Calibration-subsystem level: every swept key's subsystem must be
    a real Calibration subsystem and its field a real field on that subsystem's dataclass
    (dynamic enumeration via apply_overrides, which raises KeyError on drift)."""
    spec = default_sweep_spec()
    spec.validate()  # apply_overrides raises KeyError if any key doesn't resolve


def test_default_sweep_spec_has_no_counting_dimension():
    """counting.min_obs (and the counting subsystem generally) must not appear in the
    spec — it was dropped as provably inert on GT data (NUM-F7)."""
    spec = default_sweep_spec()
    assert "counting.min_obs" not in spec.params
    assert not any(k.split(".")[0] == "counting" for k in spec.params)


# --------------------------------------------------------------------------- IF instrument endogeneity


@requires_full_unity
@pytest.mark.slow
def test_if_scoring_uses_frozen_rubric_geometry_across_configs():
    """Instrument-endogeneity regression: score_scene's IF path must resolve the rubric's
    REFERENCE geometry (leg goals, corridor gates, avoid capsules) and the alignment
    terminal-goal at FIXED default thresholds — identical across candidate configs — even
    though the DRIVEN trajectory (the subject under test) is threaded with the swept
    ``thresholds`` and may differ. If a swept ``thresholds`` value reached the rubric
    geometry, a candidate config could shrink the rubric denominator (de-resolve legs/
    gates) or de-align a zero-scoring scene out of ``if_available`` — scoring higher
    without driving better.
    """
    import numpy as np

    from core.geometry.toolbox import Thresholds
    from core.groundtruth.loader import load_scene
    from core.perception.scene_index import BasicSceneIndex

    def _rubric_geom_plain(geom):
        """(leg_goals, corridor_gates, avoid_capsules) -> plain, array-free, comparable
        tuple (Capsule/Gate hold numpy arrays, whose == is elementwise, not a bool)."""
        leg_goals, corridor_gates, avoid_capsules, _leg_iids, _leg_aabbs = geom
        legs = [(kind, tuple(round(v, 9) for v in xy)) for kind, xy in leg_goals]
        gates = [
            (i, tuple(np.asarray(g.a).round(9).tolist()), tuple(np.asarray(g.b).round(9).tolist()))
            for i, g in corridor_gates
        ]
        avoids = [
            (
                tuple(np.asarray(c.a).round(9).tolist()),
                tuple(np.asarray(c.b).round(9).tolist()),
                round(float(c.radius), 9),
            )
            for c in avoid_capsules
        ]
        return legs, gates, avoids

    scene = "loft"
    folder = FULL_UNITY_ROOT / scene
    gt = load_scene(folder, scene_name=scene)
    idx = BasicSceneIndex(gt.instances)

    qbs = CV._load_questions(str(QUESTIONS_JSON), [scene])
    if_texts = qbs[scene]["instruction_following"]
    assert if_texts, "expected loft to carry IF questions for this regression test"

    # Two starkly different candidate configs from the sweep grid's extremes.
    cfg_a = Thresholds(near_floor=0.8, near_scale=0.4, on_min_overlap_frac=0.30)
    cfg_b = Thresholds(near_floor=2.0, near_scale=0.9, on_min_overlap_frac=0.60)

    for text in if_texts:
        # The rubric geometry cvsweep's IF scoring actually consumes (via GB's
        # fixed-default helper) must be threshold-independent by construction: calling
        # it directly never even accepts a thresholds arg, so it cannot vary.
        geom_a = GB._if_rubric_geometry(text, gt, idx)
        geom_b = GB._if_rubric_geometry(text, gt, idx)
        assert _rubric_geom_plain(geom_a) == _rubric_geom_plain(geom_b)

        goal_a = GB._terminal_goal_centroid(text, idx)
        goal_b = GB._terminal_goal_centroid(text, idx)
        if goal_a is None or goal_b is None:
            assert goal_a is goal_b
        else:
            assert goal_a.tolist() == goal_b.tolist()

    # And cvsweep's _score_if must delegate to exactly these frozen functions, not a
    # threshold-aware near-duplicate — pin the delegation itself so a future edit can't
    # silently reintroduce a swept-thresholds copy.
    import inspect

    src = inspect.getsource(CV._score_if)
    assert "GB._if_rubric_geometry(text, gt, idx)" in src
    assert "GB._terminal_goal_centroid(text, idx)" in src
    assert "_if_rubric_geometry(text, gt, idx, thresholds)" not in src
    assert "_terminal_goal_centroid(text, idx, thresholds)" not in src

    # Sanity: score_scene runs end-to-end under both configs without raising, and the
    # IF availability (how many questions counted, i.e. the rubric denominator) is
    # identical across configs even though the driven trajectories differ.
    referential = GB._load_referential(folder, scene)
    scene_graph = GB._load_scene_graph(folder, scene)

    res_a = CV.score_scene(
        gt, qbs[scene], referential=referential, scene_graph=scene_graph,
        questions_dir=str(QUESTIONS_DIR), thresholds=cfg_a,
    )
    res_b = CV.score_scene(
        gt, qbs[scene], referential=referential, scene_graph=scene_graph,
        questions_dir=str(QUESTIONS_DIR), thresholds=cfg_b,
    )
    assert res_a.if_available == pytest.approx(res_b.if_available)
    assert res_a.if_excluded == res_b.if_excluded


# --------------------------------------------------------------------------- vocab bridge


@requires_full_unity
def test_vocab_bridge_raises_scoreable_or_without_breaking_known_match():
    """Bridging bedside table<->night stand recovers a real OR match; a known-correct
    match (studio 'vase closest to the guitar') is unchanged."""
    from core.groundtruth import scoring as S
    from core.groundtruth.loader import load_scene
    from core.perception.scene_index import BasicSceneIndex

    def _score(scene: str, text: str):
        folder = FULL_UNITY_ROOT / scene
        gt = load_scene(folder, scene_name=scene)
        idx = BasicSceneIndex(gt.instances)
        ref = json.loads(
            (folder / f"{scene}_referential_statements.json").read_text(encoding="utf-8")
        )
        return S.score_object_reference(text, idx, gt.instances, referential=ref)

    # newly bridged: was "none" (unscoreable) in the baseline, now a real GT target
    bridged = _score("hotel_room_1", "Find the bedside table farthest from the window.")
    assert bridged.gt_target_id is not None
    assert bridged.iou == bridged.iou  # not NaN -> scoreable
    assert bridged.match_method == "relation"

    # known-correct match untouched by the bridge (guardrail from the task)
    known = _score("studio", "Find the vase closest to the guitar.")
    assert known.gt_target_id == 46
    assert known.iou == pytest.approx(1.0)


@requires_full_unity
def test_vocab_bridge_preserves_honest_none_for_inverted_referent():
    """A genuinely inverted referent (only 'window closest to picture' is annotated, not
    'picture closest to window') must STAY none — the bridge only equates surface forms,
    it never fabricates a target."""
    from core.groundtruth import scoring as S
    from core.groundtruth.loader import load_scene
    from core.perception.scene_index import BasicSceneIndex

    scene = "livingroom_4"
    folder = FULL_UNITY_ROOT / scene
    gt = load_scene(folder, scene_name=scene)
    idx = BasicSceneIndex(gt.instances)
    ref = json.loads(
        (folder / f"{scene}_referential_statements.json").read_text(encoding="utf-8")
    )
    r = S.score_object_reference(
        "Find the picture closest to a window.", idx, gt.instances, referential=ref
    )
    assert r.iou != r.iou  # NaN -> honestly unscoreable
    assert r.gt_target_id is None


def test_bridged_agree_only_matches_whitelisted_synonyms():
    from core.groundtruth.vocab_bridge import bridged_agree
    assert bridged_agree("bedside table", "night stand") is True
    assert bridged_agree("potted plant", "plant") is True
    assert bridged_agree("beer bottle", "bottle") is True
    # not a whitelisted pair -> no match (honest-none preserved)
    assert bridged_agree("chair", "stool") is False
    assert bridged_agree("pillow", "book") is False


# --------------------------------------------------------------------------- cache + smoke


@requires_full_unity
@pytest.mark.slow
def test_cache_reuses_config_scene_evaluations():
    """The same (config, scene) is evaluated once; repeat requests are cache hits."""
    qbs = CV._load_questions(str(QUESTIONS_JSON), ["loft", "studio"])
    ev = SceneEvaluator(
        str(FULL_UNITY_ROOT), qbs, questions_dir=str(QUESTIONS_DIR)
    )
    ev.evaluate({}, "loft")
    assert ev.n_cache_misses == 1 and ev.n_cache_hits == 0
    ev.evaluate({}, "loft")  # identical config+scene -> hit
    assert ev.n_cache_hits == 1
    ev.evaluate({"geometry.near_floor": 2.0}, "loft")  # new config -> miss
    assert ev.n_cache_misses == 2
    ev.evaluate({}, "studio")  # new scene -> miss
    assert ev.n_cache_misses == 3


@requires_full_unity
@pytest.mark.slow
def test_disk_cache_resumes_across_evaluator_instances(tmp_path):
    """Cells written by one SceneEvaluator (cache_dir set) are picked up by a FRESH
    SceneEvaluator over the same out dir — the resume path (n_disk_hits > 0), not merely
    the in-memory cache."""
    qbs = CV._load_questions(str(QUESTIONS_JSON), ["loft"])
    cache_dir = tmp_path / "cache"

    ev1 = SceneEvaluator(
        str(FULL_UNITY_ROOT), qbs, questions_dir=str(QUESTIONS_DIR), cache_dir=cache_dir
    )
    ev1.evaluate({}, "loft")
    assert ev1.n_cache_misses == 1
    assert ev1.n_disk_hits == 0
    # a JSON cell was written to disk
    cells = list(cache_dir.glob("*.json"))
    assert len(cells) == 1

    # simulate interruption: a brand-new evaluator (empty in-memory cache), same out dir
    ev2 = SceneEvaluator(
        str(FULL_UNITY_ROOT), qbs, questions_dir=str(QUESTIONS_DIR), cache_dir=cache_dir
    )
    res2 = ev2.evaluate({}, "loft")
    assert ev2.n_cache_hits == 1
    assert ev2.n_disk_hits == 1  # served from the prior run's on-disk cell
    assert ev2.n_cache_misses == 0

    # the resumed value matches what was originally computed
    res1 = ev1.evaluate({}, "loft")
    assert res2.composite() == pytest.approx(res1.composite())


@requires_full_unity
@pytest.mark.slow
def test_no_cache_dir_means_no_disk_persistence_or_resume():
    """cache_dir=None (the --no-cache path) never writes to disk and never resumes — a
    fresh evaluator always recomputes."""
    qbs = CV._load_questions(str(QUESTIONS_JSON), ["loft"])
    ev1 = SceneEvaluator(str(FULL_UNITY_ROOT), qbs, questions_dir=str(QUESTIONS_DIR))
    ev1.evaluate({}, "loft")
    assert ev1.n_cache_misses == 1
    assert ev1.n_disk_hits == 0
    assert ev1.cache_dir is None

    ev2 = SceneEvaluator(str(FULL_UNITY_ROOT), qbs, questions_dir=str(QUESTIONS_DIR))
    ev2.evaluate({}, "loft")
    assert ev2.n_cache_misses == 1  # no disk cell to resume from -> recomputed
    assert ev2.n_disk_hits == 0


@requires_full_unity
@pytest.mark.slow
def test_corrupt_cache_cell_is_tolerated_and_recomputed(tmp_path):
    """A truncated/corrupt on-disk cell (e.g. process killed mid-write) must not crash the
    sweep — it is treated as absent and recomputed, no resume credit taken for it."""
    qbs = CV._load_questions(str(QUESTIONS_JSON), ["loft"])
    cache_dir = tmp_path / "cache"

    ev1 = SceneEvaluator(
        str(FULL_UNITY_ROOT), qbs, questions_dir=str(QUESTIONS_DIR), cache_dir=cache_dir
    )
    ev1.evaluate({}, "loft")
    cells = list(cache_dir.glob("*.json"))
    assert len(cells) == 1

    # corrupt the cell on disk: truncate to invalid JSON
    cells[0].write_text("{not valid json", encoding="utf-8")

    ev2 = SceneEvaluator(
        str(FULL_UNITY_ROOT), qbs, questions_dir=str(QUESTIONS_DIR), cache_dir=cache_dir
    )
    res2 = ev2.evaluate({}, "loft")  # must not raise
    assert ev2.n_disk_hits == 0  # corrupt cell is not a hit
    assert ev2.n_cache_misses == 1  # recomputed instead
    assert res2.available > 0 or res2.composite() == 0.0  # sane result, no crash

    # the corrupt cell is overwritten with a valid one after the recompute
    payload = json.loads(cells[0].read_text(encoding="utf-8"))
    assert "scores" in payload


@pytest.mark.slow
def test_smoke_sweep_emits_all_three_reports(tmp_path):
    """n_samples=3, 2 scenes -> report.md + results.json + recommended_calibration.json,
    all valid. Uses the full Unity root when present (>=2 scenes), else the loft sample
    root; skips only if neither yields 2 scenes shared with questions.json."""
    for root in (FULL_UNITY_ROOT, UNITY_SAMPLE_ROOT):
        if not root.is_dir():
            continue
        available = [p.name for p in root.iterdir() if p.is_dir()]
        qbs_all = CV._load_questions(str(QUESTIONS_JSON), available)
        if len(qbs_all) >= 2:
            break
    else:
        pytest.skip("need a Unity root with >=2 scenes for the smoke sweep")

    scenes = sorted(qbs_all)[:2]
    qbs = {s: qbs_all[s] for s in scenes}
    ev = SceneEvaluator(str(root), qbs, questions_dir=str(QUESTIONS_DIR))
    n_folds, holdout = _fit_folds(len(scenes), 5, 3)
    folds = make_folds(scenes, n_folds=n_folds, holdout_size=holdout, seed=0)
    result = run_cv_sweep(ev, default_sweep_spec(), folds, n_samples=3, seed=0)
    md, js, rec = write_report(result, tmp_path)
    assert md.exists() and js.exists() and rec.exists()
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert payload["n_samples"] == 3
    assert len(payload["folds"]) == n_folds
    # recommended_calibration.json round-trips through the Calibration loader
    from core.calibration import from_json
    from_json(rec.read_text(encoding="utf-8"))


@requires_full_unity
@pytest.mark.slow
def test_run_cv_sweep_is_deterministic_under_seed():
    """Same (seed, spec, scenes) -> identical fold scores and recommendation."""
    qbs = CV._load_questions(str(QUESTIONS_JSON), ["loft", "studio", "office_1"])
    scenes = sorted(qbs)

    def _run():
        ev = SceneEvaluator(str(FULL_UNITY_ROOT), qbs, questions_dir=str(QUESTIONS_DIR))
        folds = make_folds(scenes, n_folds=3, holdout_size=1, seed=0)
        return run_cv_sweep(ev, default_sweep_spec(), folds, n_samples=4, seed=0)

    a, b = _run(), _run()
    assert [f.holdout_score for f in a.folds] == [f.holdout_score for f in b.folds]
    assert a.recommendation == b.recommendation
