"""Live exploration diagnostics — issue #83.

Opt-in only: unset ``VLA_EXPLORE_DEBUG_DIR`` (the default) means this module is
never imported for its side effects and ``maybe_dump`` is a single dict-lookup
no-op — zero behaviour change, byte-preservation of everything ``ExploreHead``
already does.

When the env var IS set, roughly every ``DEBUG_INTERVAL_S`` seconds of
question-clock time we append one JSON line to
``<dir>/explore_debug_<pid>.jsonl`` describing:

* the costmap: grid size, FREE/UNKNOWN/OBSTACLE cell counts, and the
  BFS-reachable "pocket" size (finite-distance FREE cells) from the current pose
  — the live analog of the offline reachable-pocket collapse;
* the frontier candidate set: every clustered blob the explore policy's
  ``detect_frontiers`` call would see, whether it was accepted, and why not
  (too small to cluster, unreachable from the vehicle, or scored below the
  policy's acceptance bar) — this answers "do candidates exist but get
  rejected, or never exist at all";
* the waypoint actually chosen this tick (or null if nothing was published).

This is diagnostic-only: it reads the grid/policy state that ``ExploreHead``
already computed and never influences a decision.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from core.nav.frontiers import (
    MIN_CLUSTER_SIZE,
    Frontier,
    _bfs_distances,
    _cluster,
    detect_frontiers,
    frontier_mask,
)
from core.nav.occupancy import FREE, OBSTACLE, UNKNOWN, OccupancyGrid

#: env var gating this module. Unset (the default) => maybe_dump is a no-op.
ENV_DEBUG_DIR: str = "VLA_EXPLORE_DEBUG_DIR"
#: minimum question-clock interval between dumps.
DEBUG_INTERVAL_S: float = 10.0
#: SYS-F9's unreachable sentinel (mirrors explore_step.UNREACHABLE_PD); duplicated
#: as a plain constant here so this module has no import-time dependency on
#: core.heads.explore_step (avoids a circular import — explore_step imports us).
_UNREACHABLE_PD: float = 1e6


def _debug_dir() -> str | None:
    return os.environ.get(ENV_DEBUG_DIR) or None


def _costmap_summary(grid: OccupancyGrid, pose: tuple[float, float]) -> dict[str, Any]:
    state = grid.state
    if state is None:
        return {
            "shape": [0, 0],
            "free": 0,
            "unknown": 0,
            "obstacle": 0,
            "reachable_pocket_cells": 0,
            "reachable_pocket_m2": 0.0,
        }
    n_free = int((state == FREE).sum())
    n_unknown = int((state == UNKNOWN).sum())
    n_obstacle = int((state == OBSTACLE).sum())
    dist = _bfs_distances(grid, grid.world_to_cell(*pose))
    import numpy as np

    pocket = int(np.isfinite(dist).sum())
    return {
        "shape": list(grid.shape),
        "free": n_free,
        "unknown": n_unknown,
        "obstacle": n_obstacle,
        "reachable_pocket_cells": pocket,
        "reachable_pocket_m2": pocket * grid.cell_m * grid.cell_m,
    }


def _frontier_candidates(
    grid: OccupancyGrid,
    pose: tuple[float, float],
    affinity,
    min_frontier_score: float,
) -> list[dict[str, Any]]:
    """Every clustered frontier blob (pre- and post-filter) with a rejection reason."""
    mask = frontier_mask(grid)
    raw_clusters = _cluster(mask)
    scored: list[Frontier] = detect_frontiers(grid, pose, affinity)

    out: list[dict[str, Any]] = []
    for comp in raw_clusters:
        size = len(comp)
        if size < MIN_CLUSTER_SIZE:
            out.append(
                {
                    "size": size,
                    "xy": None,
                    "path_distance": None,
                    "score": None,
                    "accepted": False,
                    "reason": "too_small_cluster",
                }
            )
            continue
        # Find the matching scored Frontier (same cluster, snapped centroid cell).
        f = None
        for cand in scored:
            if cand.size == size and (cand.row, cand.col) in {p for p in comp}:
                f = cand
                break
        if f is None:
            out.append(
                {
                    "size": size,
                    "xy": None,
                    "path_distance": None,
                    "score": None,
                    "accepted": False,
                    "reason": "unmatched_cluster",
                }
            )
            continue
        unreachable = f.path_distance >= _UNREACHABLE_PD
        below_bar = f.score < min_frontier_score
        accepted = not unreachable and not below_bar
        reason = "accepted"
        if unreachable:
            reason = "unreachable"
        elif below_bar:
            reason = "score_below_min"
        out.append(
            {
                "size": f.size,
                "xy": [round(f.xy[0], 3), round(f.xy[1], 3)],
                "path_distance": None if unreachable else round(f.path_distance, 2),
                "affinity": round(f.affinity, 3),
                "score": round(f.score, 3),
                "accepted": accepted,
                "reason": reason,
            }
        )
    return out


def maybe_dump(
    head: Any,
    grid: OccupancyGrid,
    pose: tuple[float, float],
    t: float,
    status: str,
    chosen_waypoint: tuple[float, float] | None,
) -> None:
    """Append one JSONL debug record if ``VLA_EXPLORE_DEBUG_DIR`` is set and the
    throttle interval has elapsed. No-op (single dict-lookup) otherwise."""
    out_dir = _debug_dir()
    if out_dir is None:
        return
    last_t = getattr(head, "_debug_last_dump_t", None)
    if last_t is not None and (t - last_t) < DEBUG_INTERVAL_S:
        return
    head._debug_last_dump_t = t

    try:
        affinity = head._affinity()
        min_frontier_score = (
            head._policy.min_frontier_score if head._policy is not None else 0.0
        )
        record = {
            "wall_time": time.time(),
            "question_clock_t": t,
            "pose": [round(pose[0], 3), round(pose[1], 3)],
            "status": status,
            "costmap": _costmap_summary(grid, pose),
            "frontier_candidates": _frontier_candidates(
                grid, pose, affinity, min_frontier_score
            ),
            "chosen_waypoint": (
                [round(chosen_waypoint[0], 3), round(chosen_waypoint[1], 3)]
                if chosen_waypoint is not None
                else None
            ),
        }
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"explore_debug_{os.getpid()}.jsonl")
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001 — diagnostics must never break exploration
        pass
