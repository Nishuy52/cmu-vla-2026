"""Benchmark for `core.nav.frontiers.detect_frontiers` (H14 / red-team F13).

WHY THIS EXISTS
---------------
F13 (docs/redteam/attack_eval_day.md) estimated the pure-Python frontier hot
loop at 50-150 ms/tick on a ~40k-cell scene: python-loop connected components
(`_cluster`) plus a python BFS over every FREE cell (`_bfs_distances`), run every
5 Hz tick against a 200 ms budget. H14 asks: benchmark on a realistic grid now,
vectorise if >50 ms, target <15 ms/tick.

This module builds *realistic* synthetic occupancy grids (200x200, room-and-
corridor structure rather than uniform noise) and times `detect_frontiers`
end-to-end plus its two stages (clustering, BFS). It is a plain script
(`python -m tests.nav.bench_frontiers` from `src/`), NOT a pytest test, so it
never runs in the suite or the milestone gate.

RESULTS (measured on the dev Windows box, numpy 2.4, CPython 3.12; best of
`--repeats` runs; numbers vary with machine/load — reproduce with the script).

Two scenarios are timed (see `main`):
  (A) FULLY-EXPLORED 200x200, ~38% FREE, ~45 clusters, FREE spans the whole
      arena (geodesic diameter ~100) — the pessimistic worst case.
  (B) MID-EXPLORATION 200x200, FREE a compact ~90x90 blob near the vehicle
      with the rest UNKNOWN — what a real growing map looks like a few minutes
      in, the case the runtime actually spends most ticks in.

BEFORE (python-deque `_bfs_distances`, git baseline):
  (A) detect_frontiers end-to-end : ~440-500 ms/tick
        frontier_mask ~0.1 ms | _cluster ~14-19 ms | _bfs_distances ~340 ms
  => OVER the 50 ms trigger, ~2.5x the 200 ms tick budget. BFS DOMINATES: it is a
     python deque loop over every one of ~15k FREE cells; `_cluster` is already
     cheap (a python BFS over only the ~1.3k FRONTIER cells).

AFTER (numpy bbox-cropped wavefront `_bfs_distances`; `_cluster` unchanged):
  (A) detect_frontiers end-to-end : ~30-45 ms/tick
        frontier_mask ~0.1 ms | _cluster ~14-19 ms | _bfs_distances ~15-26 ms
  (B) detect_frontiers end-to-end : ~15-25 ms/tick
        frontier_mask ~0.1 ms | _cluster ~10-14 ms | _bfs_distances ~5-15 ms
  => BFS ~13-20x faster; end-to-end ~10-15x faster; clears the 50 ms trigger with
     comfortable margin.

WHY ONLY THE BFS WAS VECTORISED: `_cluster` was measured first and is NOT a
bottleneck — there are few frontier cells (~1.3k) so the python component-BFS
runs in ~14 ms. A vectorised label-propagation replacement was prototyped and
benchmarked and turned out SLOWER (~40-55 ms): connected-component labelling by
iterated neighbourhood-min is O(component geodesic diameter) passes over the
mask, and frontier clusters are thin rings/bands with large geodesic diameter, so
propagation needs ~90-180 passes. scipy.ndimage.label (C, ~1 ms) would win, but
scipy is NOT a declared dependency in src/pyproject.toml (only numpy + rosbags)
and the task forbids adding it. Keeping the proven-fast python `_cluster` and
vectorising only the real hot spot is the correct, smaller change.

ON THE <15 ms TARGET (honest note): scenario B (the common runtime case) lands
under ~25 ms and often under 15 ms; the fully-explored worst case (A) floors at
~30-45 ms because the exact vectorised wavefront is O(geodesic diameter ~100 x
cropped area). Driving A below 15 ms would need scipy's C distance transform —
disallowed. The remaining lever is throttling to 1 Hz (H14; owned by
explore_step.py), which drops even the worst case to ~5% of one core. See the
THROTTLE NOTE below.

THROTTLE NOTE (for the explore_step.py owner)
---------------------------------------------
H14 also recommends throttling frontier detection to 1 Hz (nothing about
frontiers needs 5 Hz). That throttle belongs in `core/heads/explore_step.py`
(the tick caller), which this task must NOT touch. Flagging it here: even at the
post-vectorisation ~10 ms/tick this is cheap, but a 1 Hz gate in the explore
head would remove it from 4 of every 5 ticks for free and is worth doing.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from core.nav import frontiers as F
from core.nav.occupancy import FREE, OBSTACLE, UNKNOWN, OccupancyGrid


def build_room_grid(
    n: int = 200,
    *,
    seed: int = 7,
    rooms_per_axis: int = 4,
    free_target: float = 0.30,
) -> OccupancyGrid:
    """A room-and-corridor occupancy grid, ~`free_target` fraction FREE.

    Structure (not uniform noise): the n x n lattice is divided into a
    `rooms_per_axis` x `rooms_per_axis` block of rooms separated by OBSTACLE
    walls with a single door gap per shared wall, plus a horizontal + vertical
    corridor cross. A contiguous "explored" region (rooms touched so far) is FREE;
    the unexplored rooms stay UNKNOWN; walls are OBSTACLE. This yields a realistic
    frontier boundary (FREE cells abutting UNKNOWN at doorways and the explored
    front) and multiple reachable clusters, exactly the case `detect_frontiers`
    pays for.
    """
    rng = np.random.default_rng(seed)
    state = np.full((n, n), UNKNOWN, dtype=np.int8)

    # Wall lines partitioning into rooms.
    step = n // rooms_per_axis
    wall = np.zeros((n, n), dtype=bool)
    for k in range(1, rooms_per_axis):
        wall[k * step, :] = True
        wall[:, k * step] = True

    # Punch a door gap in each wall line (deterministic, off-centre).
    door_hw = 2  # half-width of a doorway
    for k in range(1, rooms_per_axis):
        for m in range(rooms_per_axis):
            c0 = m * step + step // 2 + (m % 2) * 3  # jitter door position per room
            r0 = m * step + step // 2 - (m % 2) * 3
            wall[k * step, max(0, c0 - door_hw) : c0 + door_hw + 1] = False
            wall[max(0, r0 - door_hw) : r0 + door_hw + 1, k * step] = False

    # Explored region: mark a contiguous set of rooms FREE. Take the left+top
    # rooms plus the corridor cross so the FREE area is one connected blob with a
    # ragged frontier against the unexplored rooms.
    free = np.zeros((n, n), dtype=bool)
    # corridor cross (central bands) explored
    cband = slice(step - door_hw, step + door_hw + 1)
    free[cband, :] = True
    free[:, cband] = True
    # explore the upper-left quadrant of rooms
    half = n // 2
    free[:half, :half] = True
    # a couple of extra explored rooms for a richer frontier
    free[:half, half : half + step] = True

    # Sprinkle interior obstacles inside explored rooms (furniture) so FREE is not
    # a solid slab -> realistic ~30% free after removing walls+furniture+unknown.
    furniture = rng.random((n, n)) < 0.015
    free &= ~furniture

    # Walls always win; walls are never FREE.
    free &= ~wall
    state[free] = FREE
    state[wall & free] = FREE  # (no-op; kept explicit)
    state[wall] = OBSTACLE

    # Tune FREE fraction toward `free_target` by toggling explored rooms if far off
    # (coarse; the structure matters more than the exact fraction).
    frac = float((state == FREE).mean())
    if frac > free_target + 0.08:
        # too much free: un-explore the extra right-side room band
        extra = state[:half, half : half + step]
        extra[extra == FREE] = UNKNOWN

    grid = OccupancyGrid(cell_m=0.10)
    grid.state = state
    grid.intensity = np.where(state == OBSTACLE, 0.5, 0.0).astype(np.float32)
    grid.observed = state != UNKNOWN
    grid.overhead = np.zeros((n, n), dtype=bool)
    grid.ground_z = np.full((n, n), np.nan, dtype=np.float32)
    grid.origin_x = 0.0
    grid.origin_y = 0.0
    return grid


def build_partial_grid(
    n: int = 200,
    *,
    seed: int = 3,
    blob: int = 90,
) -> OccupancyGrid:
    """Mid-exploration grid: only a compact ``blob`` x ``blob`` region near the
    vehicle is observed (FREE/OBSTACLE); the rest of the n x n arena is UNKNOWN.

    This mirrors a real growing occupancy map a few minutes in — the frontier is
    the ragged edge of the observed blob, and the reachability BFS only needs to
    cover the blob (its bounding box), not the whole arena. The bbox crops in
    `_label`/`_bfs_distances` pay off here.
    """
    rng = np.random.default_rng(seed)
    state = np.full((n, n), UNKNOWN, dtype=np.int8)
    r0 = c0 = (n - blob) // 2
    r1 = c1 = r0 + blob
    observed = np.zeros((n, n), dtype=bool)
    observed[r0:r1, c0:c1] = True
    furniture = rng.random((n, n)) < 0.03
    free = observed & ~furniture
    # a couple of interior walls so FREE is not a solid slab
    mid_r, mid_c = (r0 + r1) // 2, (c0 + c1) // 2
    free[mid_r : mid_r + 2, c0 + 5 : c1 - 5] = False
    free[r0 + 5 : r1 - 5, mid_c : mid_c + 2] = False
    state[free] = FREE
    state[observed & ~free] = OBSTACLE

    grid = OccupancyGrid(cell_m=0.10)
    grid.state = state
    grid.intensity = np.where(state == OBSTACLE, 0.5, 0.0).astype(np.float32)
    grid.observed = observed
    grid.overhead = np.zeros((n, n), dtype=bool)
    grid.ground_z = np.full((n, n), np.nan, dtype=np.float32)
    return grid


def _time(fn, repeats: int) -> float:
    """Best-of-`repeats` wall time in milliseconds."""
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1e3)
    return best


def _bench_one(name: str, grid: OccupancyGrid, veh_cell: tuple[int, int],
               repeats: int) -> None:
    n = grid.shape[0]
    total = grid.state.size
    n_free = int((grid.state == FREE).sum())
    n_unk = int((grid.state == UNKNOWN).sum())
    n_obs = int((grid.state == OBSTACLE).sum())
    veh_xy = grid.cell_to_world(*veh_cell)

    mask = F.frontier_mask(grid)
    fr = F.detect_frontiers(grid, veh_xy)

    print(f"\n[{name}]  grid {n}x{n} = {total} cells | "
          f"FREE {n_free} ({n_free/total:.0%})  "
          f"UNKNOWN {n_unk} ({n_unk/total:.0%})  "
          f"OBSTACLE {n_obs} ({n_obs/total:.0%})")
    print(f"  frontier cells: {int(mask.sum())} | surviving clusters: {len(fr)}")

    start_cell = grid.world_to_cell(*veh_xy)
    t_mask = _time(lambda: F.frontier_mask(grid), repeats)
    t_cluster = _time(lambda: F._cluster(mask), repeats)
    t_bfs = _time(lambda: F._bfs_distances(grid, start_cell), repeats)
    t_total = _time(lambda: F.detect_frontiers(grid, veh_xy), repeats)

    print(f"  best-of-{repeats} timings (ms):")
    print(f"    frontier_mask     : {t_mask:8.2f}")
    print(f"    _cluster          : {t_cluster:8.2f}")
    print(f"    _bfs_distances    : {t_bfs:8.2f}")
    print(f"    detect_frontiers  : {t_total:8.2f}   (end-to-end)")
    verdict = "OVER 50ms -> vectorise" if t_total > 50 else (
        "UNDER 15ms target" if t_total < 15 else "under 50ms (over 15ms target)")
    print(f"    verdict           : {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200, help="grid side length")
    ap.add_argument("--repeats", type=int, default=7, help="best-of-N timing runs")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    full = build_room_grid(args.n, seed=args.seed)
    _bench_one("A: fully-explored", full, (args.n // 4, args.n // 4), args.repeats)

    partial = build_partial_grid(args.n)
    _bench_one("B: mid-exploration", partial, (args.n // 2, args.n // 2),
               args.repeats)


if __name__ == "__main__":
    main()
