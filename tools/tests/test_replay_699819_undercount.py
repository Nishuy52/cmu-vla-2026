"""Issue #151 remainder: does the merged relation-aware disambiguator fallback
(``f9e8b60``, "fall back past an empty disambiguator anchor") plus #160's
fragment clustering actually resolve the LIVE numerical undercount?

The evidence #151 itself cites (``reports/cluster_verify/70198{2,3}``) is not
present in this checkout (only ``captures/scores.json``/``.md`` were kept by
the harvest commits; the ``debug/*/instance_index.jsonl`` dumps were not
banked for those two jobs). ``reports/cluster_verify/699819`` IS present and
IS the same evidence class: a full 15-scene numerical live sweep with banked
``debug/<slot>/instance_index.jsonl`` answer-time snapshots plus each row's
question and live answer in ``captures/scores.json`` — the harness this file
uses (``tools/replay_live_numerical``) already existed and is exactly built
for this replay.

Verdict, replaying the CURRENT resolver + counting head path (this repo's
``core.geometry.toolbox.counting``, unchanged by this test) against 699819's
banked boxes:

    scene            old live  new (current code)  truth
    home_building_1  0         0                    6   -- unfixed
    home_building_2  0         0                    2   -- unfixed (see note)
    hotel_room_2     0         0                    3   -- unfixed
    loft             0         0                    2   -- unfixed (see note)

**None of the four "old live answer = 0" cases in this sample are fixed** by
the fallback + clustering combination, so this replay does NOT support
"Fixes #151" and does not claim it. Per-case root cause, traced with
``core.geometry.toolbox._resolve_anchor``/``_eval_clause`` directly against
the loaded instances:

* home_building_2, loft ("red pillow" / "black pillow"): the recorded dump
  format (``core.perception.scene_index.dump_instance_index``) never
  persists ``color_bins`` — every replayed ``InstanceRecord`` has
  ``color_bins=()`` regardless of what the live pipeline actually saw. This
  is a REPLAY DATA gap, not a resolver defect, and is not fixable from
  ``core/geometry/toolbox.py`` (or from this replay source at all).
* home_building_1 ("pillows on the sofa under the pictures"): traced
  directly -- the disambiguator ``under(picture)`` correctly narrows the 18
  recorded ``sofa`` instances to the 3 that have some ``picture`` above them,
  but the REAL sofa (near the 19 recorded pillows) has no picture recorded
  above it at all -- a missing detection, not a wrong anchor choice among
  present candidates. #151's own diagnosis calls out exactly this case
  ("the real picture near the real sofa cluster was never detected at
  all").
* hotel_room_2 ("pictures above the bed"): traced the same way -- none of
  the 5 recorded ``bed`` instances has any of the 3 recorded ``picture``
  instances satisfying ``above()`` against it. A missing-detection /
  box-quality gap, not an anchor-selection defect.

None of these three remaining mechanisms live in the anchor-RESOLUTION path
this surface owns (colour data isn't in the replay at all; the other two are
missing evidence, not a resolution choice among present evidence), so no fix
belongs in this commit for them.

This replay DID catch a REAL regression in this branch's OWN #160 commit,
fixed in the same surface (`core.geometry.toolbox._cluster_plausible`, see
`test_cluster_falls_back_to_head_noun_prior_when_no_exact_prior` in
``src/tests/geometry/test_cluster_resolve.py``): livingroom_3's "tv cabinet"
has no OWN dimension-prior row, so the original fail-open plausibility check
let 27 scattered (non-fragment) ``tv cabinet`` detections merge into one 74
sq m "anchor", moving "how many photos are on the TV cabinet" from
count=6 (pre-clustering) to count=13 -- both wrong against a truth of 2, but
13 is a materially worse miss. Falling back to the HEAD NOUN's prior
("cabinet") before failing open restores the pre-clustering count=6.

This file pins the CURRENT (post-fix) counted answer per scene as a
regression test: a future perception/resolver change that silently moves any
of these numbers should fail loudly here, on real banked data, not just on
this branch's synthetic fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools.replay_live_numerical import run_replay, variant_raw

CAPTURES_699819 = Path(__file__).resolve().parents[2] / "reports" / "cluster_verify" / "699819"

#: scene -> (truth, current-code counted answer). Measured against this
#: branch's HEAD (issue #160 clustering + head-noun prior fallback fix, issue
#: #167's tie-break -- inert here, counting() never calls it) replaying
#: 699819's banked answer-time instance snapshots.
EXPECTED = {
    "arabic_room": (2, 2),
    "chinese_room": (6, 20),
    "home_building_1": (6, 0),
    "home_building_2": (2, 0),
    "hotel_room_1": (4, 23),
    "hotel_room_2": (3, 0),
    "japanese_room": (3, 1),
    "livingroom_1": (8, 18),
    "livingroom_2": (2, 7),
    "livingroom_3": (2, 6),  # the #160 head-noun-prior fix: was 13 before it
    "livingroom_4": (6, 5),
    "loft": (2, 0),
    "office_1": (6, 3),
    "office_2": (1, 0),
    "studio": (3, 1),
}

#: the four scenes whose recorded LIVE run answered 0 (captures/scores.json's
#: own "live" field) -- the #151 "old live answer" this replay checks the
#: fallback+clustering combination against.
OLD_LIVE_ZERO_SCENES = {"home_building_1", "home_building_2", "hotel_room_2", "loft"}


@pytest.mark.skipif(
    not CAPTURES_699819.is_dir(), reason="reports/cluster_verify/699819 not present in this checkout"
)
def test_699819_replay_matches_pinned_current_counts():
    summary_a, _ = run_replay(CAPTURES_699819, variant_raw, variant_raw)
    got = {r.scene: (r.truth, r.count) for r in summary_a.results}
    assert got == EXPECTED


@pytest.mark.skipif(
    not CAPTURES_699819.is_dir(), reason="reports/cluster_verify/699819 not present in this checkout"
)
def test_151_fallback_does_not_fix_any_old_live_zero_case_on_this_sample():
    """The literal #151 verification this file exists for: none of 699819's
    four "old live answer = 0" scenes are fixed by the merged disambiguator
    fallback (f9e8b60) plus this branch's #160 clustering. Documents a
    negative result -- see the module docstring for per-case root causes,
    none of which live in this surface's anchor-resolution path."""
    summary_a, _ = run_replay(CAPTURES_699819, variant_raw, variant_raw)
    still_zero = {
        r.scene
        for r in summary_a.results
        if r.scene in OLD_LIVE_ZERO_SCENES and r.count == 0
    }
    assert still_zero == OLD_LIVE_ZERO_SCENES, (
        f"expected all of {OLD_LIVE_ZERO_SCENES} to remain 0 (documenting the "
        f"#151 remainder); got {still_zero} still at 0 -- if this now differs, "
        "the module docstring's per-case analysis needs re-checking, not just "
        "this assertion"
    )
