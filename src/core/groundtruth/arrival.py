"""Shared derivation of the IF-rubric leg-arrival tolerance (issue #70).

Single source of truth for how far a driven pose may sit from a leg's goal point
and still count as "arrived". Both consumers derive from
:func:`derived_arrival_tol_m` so they never drift apart:

* :data:`core.groundtruth.scoring.LEG_ARRIVAL_TOL_M` — the battery's rubric
  scorer, where the p95 term is computed FRESH per battery run from that run's
  scenes (see that module's docstring).
* :data:`core.heads.instruction.ARRIVAL_TOL_M` — the live navigation head's
  compile-time constant. The head has no battery run to compute a live p95
  from (it must stay ROS-free and dependency-light — it does not carry a GT
  frame-fit pipeline), so it freezes the derivation at a documented NOMINAL
  residual (:data:`NOMINAL_FIT_RESIDUAL_P95_M` below) instead. The two are
  "intentionally coupled" (per the issue): both are `derived_arrival_tol_m`
  evaluated at *some* residual, just a live one for the battery and a frozen
  documented one for the head — never two independently-tuned numbers.

Every term in :func:`derived_arrival_tol_m` cites the exact source
constant/measurement it derives from; the only term that is not directly a
measured/coded constant is the small fixed safety margin, justified in-line
where it is defined.

A THIRD consumer, added by issue #100, uses the same formula at a different
residual input rather than a new formula: :data:`core.groundtruth.scoring.
PASS_BY_ARRIVAL_TOL_M` (the PASS-BY/VIA_NEAR leg tolerance) evaluates
:func:`derived_arrival_tol_m` at :data:`PASS_BY_NOMINAL_FIT_RESIDUAL_M`
instead of :data:`NOMINAL_FIT_RESIDUAL_P95_M` — see that constant's docstring
for why a PASS-BY leg needs the median, not the p95, of the same underlying
per-scene residual measurement.
"""
from __future__ import annotations

import math

from core.nav.costmap import VEHICLE_RADIUS_M

# --------------------------------------------------------------------------- terms

#: Costmap grid resolution (m/cell). Both the battery's synthetic-scene costmap
#: (``core.runner.gt_battery._WALL_CELL_M``, itself
#: ``core.mocks.synthetic_scene.FLOOR_SPACING`` = 0.10 m) and the live navigation
#: stack's OccupancyGrid (``core.nav.occupancy.CELL_M`` = 0.10 m) currently use
#: 0.10 m. Kept as an explicit, overridable parameter (not re-imported and hard-
#: wired) so a caller whose grid uses a different resolution derives the correct
#: tolerance for ITS grid, not a value baked in for today's default.
DEFAULT_CELL_M: float = 0.10

#: Small fixed safety margin (m). Every OTHER term in ``derived_arrival_tol_m``
#: is traceable to a measured or coded constant (the vehicle/inflation radius,
#: the grid resolution, the fitted frame residual); this one is not itself a
#: geometric quantity but a numerical-robustness allowance for the chain of
#: roundoff/snap operations a goal position passes through before comparison —
#: the frame-fit rotation+translation (``core.groundtruth.scoring.fit_frame``),
#: the BFS/A* grid-cell snap (``Costmap.nearest_reachable_point``), and the
#: driven-trajectory densification (``core.groundtruth.scoring._densify_polyline``)
#: — none of which is individually modelled as its own additive term here, but
#: each can nudge a compared position by a sub-centimetre-to-centimetre amount.
#: Deliberately far smaller than any structural term above (5 cm vs. the vehicle
#: radius' 40 cm, ~7 cm grid-diagonal term, or a typical >0.4 m fit residual) so
#: it cannot be used to paper over a genuine geometric miss — it exists only so
#: an exact-boundary case does not flap between reached/missed on floating-point
#: noise. This is the ONE free parameter the issue's brief allows; every other
#: term is a cited constant or a caller-supplied measurement.
FIXED_SAFETY_MARGIN_M: float = 0.05


def derived_arrival_tol_m(
    fit_residual_p95_m: float,
    *,
    vehicle_radius_m: float = VEHICLE_RADIUS_M,
    cell_m: float = DEFAULT_CELL_M,
    margin_m: float = FIXED_SAFETY_MARGIN_M,
) -> float:
    """Derive the leg-arrival tolerance (m), instead of asserting a fixed 0.8 m.

    ``tol = vehicle_radius_m + grid_diagonal_m / 2 + fit_residual_p95_m + margin_m``

    Terms, each cited to its source:

    1. ``vehicle_radius_m`` — default :data:`core.nav.costmap.VEHICLE_RADIUS_M`
       (0.4 m, ``core/nav/costmap.py:49``).

       **Conflict with the issue #70 brief, resolved here.** The brief asks for
       "vehicle_radius + costmap inflation" as two separate additive terms.
       Reading ``Costmap.__init__``/``Costmap._inflate`` (``core/nav/costmap.py``):
       the obstacle-inflation radius IS ``vehicle_radius_m`` — there is only ONE
       constant in the code for both meanings (the vehicle's own physical
       half-footprint clearance and the costmap's obstacle-inflation radius are
       literally the same value, ``VEHICLE_RADIUS_M = 0.4``; see the constructor
       default ``vehicle_radius_m: float = VEHICLE_RADIUS_M`` feeding straight
       into ``self.base_blocked = self._inflate(blocked_seed, vehicle_radius_m)``).
       Adding it twice would double-count a single physical margin (the module
       docstring itself only ever describes ONE inflation radius, "the vehicle
       FOOTPRINT", never a separate inflation-on-top-of-the-footprint quantity).
       This function therefore takes it once. If a future costmap ever
       decouples the two (a genuine extra clearance margin beyond the vehicle's
       own radius), that quantity would become a real second term here.

    2. ``cell_m * sqrt(2) / 2`` — half the diagonal of one grid cell: the
       worst-case offset between a grid-quantised position (e.g. the cell
       centre ``Costmap.nearest_reachable_point`` returns, or a rasterised
       goal/obstacle footprint) and the true continuous point it stands in
       for. ``cell_m`` defaults to :data:`DEFAULT_CELL_M` (0.10 m, matching
       both ``core.nav.occupancy.CELL_M`` and the battery's
       ``core.runner.gt_battery._WALL_CELL_M`` /
       ``core.mocks.synthetic_scene.FLOOR_SPACING``) but stays a parameter so
       a caller on a different grid resolution derives the tolerance for ITS
       grid rather than inheriting today's default silently.

    3. ``fit_residual_p95_m`` — the p95 (worst-typical-case, not worst-case) of
       the per-scene sim<->object-frame rigid-fit residual
       (``core.groundtruth.scoring.align_scene_trajectories`` / ``fit_frame``),
       computed by the CALLER across a battery run's scenes (this function
       takes no scene data of its own — it is a pure function of one number).
       This is the dominant, run-dependent term: the GT reference trajectory
       and our resolved goal live in two frames related by a per-scene fit
       that is never exact, and the residual is the direct measurement of
       that per-scene misalignment.

    4. ``margin_m`` — see :data:`FIXED_SAFETY_MARGIN_M`.

    No other free parameters: every term is either a coded constant, a grid
    parameter the caller owns, or a measurement the caller supplies.
    """
    grid_diag_half_m = cell_m * math.sqrt(2.0) / 2.0
    return vehicle_radius_m + grid_diag_half_m + fit_residual_p95_m + margin_m


# --------------------------------------------------------------------------- nominal (head)

#: Nominal p95 fit residual (m) used to derive
#: :data:`core.heads.instruction.ARRIVAL_TOL_M`'s compile-time value. The head
#: must stay ROS-free and dependency-light (docstring of that module) — it
#: cannot carry the GT frame-fit pipeline or recompute a live p95 from a
#: battery run's scenes, so it freezes the derivation at this documented
#: constant instead of a per-run measurement.
#:
#: Sourced from the most recent available battery-provenance snapshot,
#: ``reports/issue59_leg_ceiling.json`` (15 battery scenes, one
#: ``fit_residual_m`` per scene, range 0.1953-1.8103 m):
#: ``numpy.percentile(residuals, 95) == 1.2256`` m (rounded to 4 dp below).
#: This is a snapshot, not a live measurement — a later battery rerun (issue
#: #70 Phase B, or any future re-derivation) may compute a different p95 from
#: fresh per-scene residuals; when it does, this constant should be updated
#: deliberately (with a comment citing the new source run), never silently.
NOMINAL_FIT_RESIDUAL_P95_M: float = 1.2256

#: :func:`derived_arrival_tol_m` evaluated at :data:`NOMINAL_FIT_RESIDUAL_P95_M`
#: with every other parameter at its default — the value
#: :data:`core.heads.instruction.ARRIVAL_TOL_M` is set to.
NOMINAL_ARRIVAL_TOL_M: float = derived_arrival_tol_m(NOMINAL_FIT_RESIDUAL_P95_M)


# --------------------------------------------------------------------------- nominal (PASS-BY)

#: Issue #100: the PASS-BY/VIA_NEAR leg tolerance needs its OWN nominal residual term,
#: distinct from :data:`NOMINAL_FIT_RESIDUAL_P95_M` — same source measurement, different
#: (and here, cited) aggregate statistic.
#:
#: A STOP leg (the terminal "stop at X" GOTO, or a corridor-gate midpoint) has no other
#: slack term for hitting an exact point, so :data:`NOMINAL_FIT_RESIDUAL_P95_M` uses the
#: P95 (worst-typical, not worst-case) of the per-scene correspondence-fit residual —
#: robustness against an unlucky scene's frame fit is the whole job of that term there.
#:
#: A PASS-BY leg (issue #70's ``core.groundtruth.scoring._is_pass_by_leg``: VIA_NEAR at
#: any position, or a non-terminal GOTO) already carries its OWN, separately-derived
#: slack term — the resolved instance's own AABB half-diagonal
#: (``core.groundtruth.scoring.score_instruction_rubric``: ``leg_tol = half_diag +
#: pass_by_tol``). That footprint term is already the dominant source of "how far off
#: the exact centroid can a still-genuine arrival land" for a PASS-BY leg (a robot that
#: drives past a landmark's OWN extent has passed it, full stop — no registration slack
#: needed to explain that). Stacking the P95 (a worst-typical-CASE, i.e. an unlucky
#: SCENE's) registration residual on top of the footprint term double-pays for the same
#: robustness the footprint term already buys, and was measured (issue #100) to make the
#: PASS-BY tolerance too generous to discriminate grounding quality: 2.0-2.4 m tolerances
#: credited live closest-approaches as far as 1.84 m as "reached".
#:
#: The fix keeps the same formula and the same underlying measurement
#: (``reports/issue59_leg_ceiling.json``, 15 battery scenes, one ``fit_residual_m`` per
#: scene, range 0.1953-1.8103 m) — only the aggregate changes, from the P95 to the
#: MEDIAN: ``numpy.median(residuals) == 0.4671`` m (rounded to 4 dp below). The median is
#: the "typical" (not "worst-typical") per-scene registration residual — appropriate once
#: a separate term is already carrying the robustness-to-an-unlucky-scene burden, and
#: still a real, cited, un-tuned statistic of the EXISTING snapshot (not reverse-derived
#: from any battery run's credited-leg distances — the derivation only reads the
#: pre-existing residual snapshot, not the arrival outcomes it is meant to score).
PASS_BY_NOMINAL_FIT_RESIDUAL_M: float = 0.4671

#: :func:`derived_arrival_tol_m` evaluated at :data:`PASS_BY_NOMINAL_FIT_RESIDUAL_M` with
#: every other parameter at its default — the value
#: :data:`core.groundtruth.scoring.PASS_BY_ARRIVAL_TOL_M` (and hence
#: :func:`core.groundtruth.scoring.score_instruction_rubric`'s ``pass_by_tol`` default) is
#: set to. Still built from the SAME four cited terms as the STOP tolerance
#: (vehicle_radius + grid_diagonal/2 + residual + margin) — only the residual input
#: differs.
PASS_BY_NOMINAL_TOL_M: float = derived_arrival_tol_m(PASS_BY_NOMINAL_FIT_RESIDUAL_M)
