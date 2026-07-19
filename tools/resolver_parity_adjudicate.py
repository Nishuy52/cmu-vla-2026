"""Adjudication helper for the issue #71 resolve-outcome divergence audit.

For each divergent leg identified by ``tools.resolver_parity``, prints the
question text, the leg's anchor spec, and both sides' candidate instance
(label + goal + distance from the GT reference trajectory) -- so the correct
reading can be judged by which candidate the demonstrator's trajectory
actually passes near. Offline diagnostic only, not part of the scored path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from core.groundtruth.loader import load_scene  # noqa: E402
from core.groundtruth import scoring as S  # noqa: E402
from core.parsing.regex_tier import parse_regex  # noqa: E402
from core.perception.scene_index import BasicSceneIndex  # noqa: E402
from core.geometry.toolbox import TargetSpec, resolve  # noqa: E402
from core.geometry import primitives as P  # noqa: E402
from core.runner import gt_battery as GB  # noqa: E402

DIVERGENT = [
    ("home_building_1", 0, 1),
    ("home_building_2", 1, 2),
    ("hotel_room_2", 1, 1),
    ("japanese_room", 1, 1),
    ("livingroom_1", 0, 1),
    ("livingroom_2", 0, 1),
    ("livingroom_3", 1, 1),
    ("office_2", 1, 1),
]


def main() -> int:
    root = Path("data/vla3d/Unity")
    with open(GB.DEFAULT_QUESTIONS, encoding="utf-8") as fh:
        data = json.load(fh)
    by_scene = {e["scene"]: e for e in data}

    for scene_name, qi, leg_i in DIVERGENT:
        entry = by_scene[scene_name]
        folder = GB._find_scene_folder(root, scene_name)
        gt = load_scene(folder, scene_name=scene_name)
        idx = BasicSceneIndex(gt.instances)
        if_texts = entry["questions"]["instruction_following"]
        text = if_texts[qi]
        plan = parse_regex(text)
        leg = plan.route[leg_i]
        anchor = leg.anchors[0]
        spec = TargetSpec(
            noun=anchor.noun, raw=anchor.raw, attributes=list(anchor.attributes),
            clauses=[anchor.disambiguator] if anchor.disambiguator is not None else [],
        )
        res = resolve(spec, idx)
        traj_q = GB._IF_TRAJ_INDEX.get(qi)
        traj_path = Path(GB.DEFAULT_QUESTIONS_ROOT) / scene_name / f"trajectory_q{traj_q}.ply"
        traj = S.load_trajectory_ply(traj_path) if traj_path.exists() else None

        print("=" * 100)
        print(f"{scene_name} q{qi} leg{leg_i}  text={text!r}")
        print(f"  anchor noun={anchor.raw!r} attrs={anchor.attributes} disamb={anchor.disambiguator}")
        print(f"  audit={[ (a.step, a.detail) for a in res.audit ]}")
        for rank, c in enumerate(res.candidates_ranked[:6]):
            cxy = P._as3(c.centroid)[:2]
            d = None
            if traj is not None and traj.shape[0] > 0:
                import numpy as np
                d = float(np.min(np.linalg.norm(traj[:, :2] - np.asarray(cxy), axis=1)))
            print(
                f"    rank{rank} id={c.instance_id} label={c.label!r} "
                f"xy=({cxy[0]:.2f},{cxy[1]:.2f}) dist_to_gt_traj={d}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
