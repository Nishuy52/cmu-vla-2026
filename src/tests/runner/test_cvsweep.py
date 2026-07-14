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


def test_best_independent_prefers_referential_then_scenegraph():
    from core.runner.gt_battery import GTQuestionScore
    r1 = GTQuestionScore(scene="s", qtype="numerical", question="q",
                         gt_count_independent=4, gt_count_scenegraph=9)
    assert CV._best_independent(r1) == 4  # referential wins
    r2 = GTQuestionScore(scene="s", qtype="numerical", question="q",
                         gt_count_independent=None, gt_count_scenegraph=9)
    assert CV._best_independent(r2) == 9  # falls back to scene graph
    r3 = GTQuestionScore(scene="s", qtype="numerical", question="q")
    assert CV._best_independent(r3) is None  # no opinion -> excluded


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
    spec.validate()  # must not raise: every geometry key is a real field, counting is special-cased
    assert "counting.min_obs" in spec.params
    assert all(len(v) >= 3 for v in spec.params.values())  # 3-5 values each


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
