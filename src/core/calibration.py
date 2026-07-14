"""Single source of truth for every tunable constant in ``core/``.

Phase-2 calibration will sweep these values against the sim. Today they are
scattered across module-level constants, config-dataclass defaults, function
parameters and constructor arguments — impossible to enumerate, serialise, or
diff in one place. This module fixes that WITHOUT refactoring the owning modules:

* It **composes** the config dataclasses that already exist (geometry
  :class:`~core.geometry.toolbox.Thresholds`, :class:`~core.perception.fusion.FusionConfig`,
  :class:`~core.perception.tracker.TrackerConfig` / :class:`~core.perception.tracker.KeyframeConfig`).
* It **mirrors** the loose nav module constants (:class:`NavTunables`) and the
  fsm/interfaces budget constants (:class:`BudgetTunables`) as new dataclasses.

The dataclass is therefore the single documentation + serialisation surface NOW.
Full wiring — pushing an overridden value back into a module that currently reads
a module-level constant with no setter — lands with the Phase-2 sweep. Every field
is documented in ``docs/calibration.md`` with how it is injected today (constructor
arg / function param / hard module constant) and whether it is wireable without a
code change ("wiring TODO (Phase 2)").

A pinning test (``tests/test_calibration.py``) imports the live modules and asserts
:func:`default_calibration` still matches them, so drift between this ledger and the
code fails loudly.

Pure, deterministic, no new dependencies. Times in seconds, distances in metres,
angles in radians, unless a field says otherwise.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from typing import Any

import numpy as np

from core.geometry.toolbox import Thresholds
from core.perception.fusion import FusionConfig
from core.perception.tracker import KeyframeConfig, TrackerConfig
from core.interfaces import QType


# --------------------------------------------------------------------------- nav tunables


@dataclass(frozen=True)
class NavTunables:
    """Mirror of the loose module-level constants across ``core/nav/*``.

    These are NOT owned by any existing dataclass; they live as module globals
    (and, for a few, as constructor/function-parameter defaults derived from those
    globals). This dataclass gathers them for the sweep. See ``docs/calibration.md``
    for the per-field wiring status.
    """

    # --- occupancy.py -----------------------------------------------------------
    cell_m: float = 0.10               # grid resolution, metres/cell (CELL_M)
    free_max: float = 0.15             # terrain intensity < this -> FREE (FREE_MAX)
    observe_radius_m: float = 8.0      # lidar footprint radius for observed mask
    grow_pad_cells: int = 8            # extra ring of cells added on grid growth

    # --- costmap.py -------------------------------------------------------------
    vehicle_radius_m: float = 0.4      # obstacle inflation radius (half footprint + margin)
    overhead_soft_cost_mult: float = 4.0  # A* penalty to cross a SOFT-overhead cell (redteam H13)

    # --- frontiers.py -----------------------------------------------------------
    min_cluster_size: int = 5          # frontier clusters smaller than this are noise
    w_size: float = 1.0                # frontier reward on cluster size
    w_dist: float = 0.5                # frontier penalty on path distance (cells)
    w_affinity: float = 4.0            # frontier weight on injected semantic affinity

    # --- planner.py -------------------------------------------------------------
    unknown_cost_mult: float = 3.0     # A* multiplier for traversing an UNKNOWN cell
    pinch_disc_m: float = 3.0          # radius of local pinch overlay around a gate
    pinch_corridor_half_w_m: float = 0.5  # half-width of the forced corridor through a gate

    # --- exploration.py ---------------------------------------------------------
    sweep_s: float = 60.0              # duration of the opening orientation sweep
    sweep_side_m: float = 1.0          # side length of the sweep diamond
    min_frontier_score: float = 0.0    # frontiers below this score aren't pursued
    coverage_saturated_free_frac: float = 0.0  # reserved; frontier absence is the real gate

    # --- breadcrumbs.py ---------------------------------------------------------
    lookahead_m: float = 2.5           # farthest a crumb may sit ahead of the vehicle
    reach_m: float = 0.8               # advance to next crumb within this distance
    stall_move_m: float = 0.3          # movement below this over the window == stalled
    stall_window_s: float = 10.0       # stall observation window


# --------------------------------------------------------------------------- budget tunables


@dataclass(frozen=True)
class BudgetTunables:
    """Mirror of the fsm/budget + interfaces budget-clock constants.

    Gathers the 600 s question-clock gates, the orientation/reserve windows, the
    per-checkpoint call caps, and the per-QType soft exploration budgets. See
    ``docs/calibration.md`` for wiring status (several are consumed as bare module
    constants today and need a setter/param before a sweep can move them).
    """

    # --- interfaces.py (clock gates) --------------------------------------------
    # forced_assembly_s / watchdog_floor_s pin the neutral BudgetState interface
    # constants. The controller's EFFECTIVE defaults are the skew-hedged 480/540
    # (fsm/budget.py DEFAULT_*), sized for the evaluator's clock starting at
    # system startup rather than question receipt (redteam H8 / SYS-F6).
    question_budget_s: float = 600.0   # total per-question wall budget (QUESTION_BUDGET_S)
    forced_assembly_s: float = 510.0   # T-90: begin best-effort assembly (FORCED_ASSEMBLY_S)
    watchdog_floor_s: float = 570.0    # T-30: publish floor answer (WATCHDOG_FLOOR_S)

    # --- interfaces.py (TerrainPatch traversability cutoff) ---------------------
    terrain_free_max: float = 0.15     # TerrainPatch.FREE_MAX; traversability cutoff

    # --- fsm/budget.py (phase windows) ------------------------------------------
    orientation_s: float = 60.0        # in-place sweep window (ORIENTATION_S)
    ledger_reserve_s: float = 45.0     # floor-reserve below which no discretionary call (LEDGER_RESERVE_S)

    # --- interfaces.py (per-QType soft exploration budgets) ---------------------
    explore_budget_numerical_s: float = 210.0
    explore_budget_object_reference_s: float = 240.0
    explore_budget_instruction_following_s: float = 270.0

    # --- fsm/budget.py (per-checkpoint hard call caps) --------------------------
    cap_parse: int = 2
    cap_miss_recovery: int = 1
    cap_anchor_confirm: int = 3
    cap_verification: int = 1
    cap_frontier_select: int = 1
    cap_self_consistency: int = 2

    def explore_budget_map(self) -> dict[QType, float]:
        """Reconstruct the ``EXPLORE_BUDGET_S`` mapping from the flat fields."""
        return {
            QType.NUMERICAL: self.explore_budget_numerical_s,
            QType.OBJECT_REFERENCE: self.explore_budget_object_reference_s,
            QType.INSTRUCTION_FOLLOWING: self.explore_budget_instruction_following_s,
        }

    def checkpoint_max_map(self) -> dict[str, int]:
        """Reconstruct the ``CHECKPOINT_MAX`` mapping from the flat fields."""
        return {
            "parse": self.cap_parse,
            "miss_recovery": self.cap_miss_recovery,
            "anchor_confirm": self.cap_anchor_confirm,
            "verification": self.cap_verification,
            "frontier_select": self.cap_frontier_select,
            "self_consistency": self.cap_self_consistency,
        }


# --------------------------------------------------------------------------- root


#: Subsystem field name -> the nested config dataclass type it holds. Iteration order
#: here defines the dotted-key namespaces ("geometry.*", "fusion.*", ...).
_SUBSYSTEMS: dict[str, type] = {
    "geometry": Thresholds,
    "fusion": FusionConfig,
    "tracker": TrackerConfig,
    "keyframe": KeyframeConfig,
    "nav": NavTunables,
    "budget": BudgetTunables,
}


@dataclass(frozen=True)
class Calibration:
    """Every tunable in ``core/`` in one composable, serialisable place.

    Composes the four existing config dataclasses plus the two new mirror
    dataclasses. Field names here are the top-level dotted-key namespaces used by
    :func:`apply_overrides` and :func:`diff` (e.g. ``geometry.near_floor``).
    """

    geometry: Thresholds = field(default_factory=Thresholds)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    keyframe: KeyframeConfig = field(default_factory=KeyframeConfig)
    nav: NavTunables = field(default_factory=NavTunables)
    budget: BudgetTunables = field(default_factory=BudgetTunables)


def default_calibration() -> Calibration:
    """The live-default calibration: every field at the value the modules use today."""
    return Calibration()


# --------------------------------------------------------------------------- serialisation


def _coerce_scalar(value: Any) -> Any:
    """Make a field value JSON-friendly and round-trip-stable.

    numpy scalars (``np.deg2rad`` yields ``np.float64``) collapse to Python floats so
    that ``from_json(to_json(x)) == x`` holds under a plain ``==`` comparison.
    """
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _to_plain(cal: Calibration) -> dict[str, dict[str, Any]]:
    """Nested ``{subsystem: {field: scalar}}`` dict with numpy scalars normalised."""
    out: dict[str, dict[str, Any]] = {}
    for sub in _SUBSYSTEMS:
        cfg = getattr(cal, sub)
        out[sub] = {f.name: _coerce_scalar(getattr(cfg, f.name)) for f in fields(cfg)}
    return out


def to_json(cal: Calibration, *, indent: int | None = 2) -> str:
    """Serialise a Calibration to a deterministic JSON string.

    Keys are emitted in the fixed subsystem/field declaration order (no sorting), so
    the output is stable and diff-friendly across runs.
    """
    return json.dumps(_to_plain(cal), indent=indent)


def from_json(text: str) -> Calibration:
    """Reconstruct a Calibration from :func:`to_json` output.

    Unknown subsystems/fields are ignored (forward-compat); missing ones fall back to
    the live default. Field types follow each dataclass's declared default, so an int
    field stays int and a float field stays float.
    """
    data = json.loads(text)
    kwargs: dict[str, Any] = {}
    for sub, cls in _SUBSYSTEMS.items():
        raw = data.get(sub, {})
        sub_kwargs: dict[str, Any] = {}
        default = cls()
        for f in fields(cls):
            if f.name in raw:
                cur = getattr(default, f.name)
                val = raw[f.name]
                # Preserve int-vs-float typing from the dataclass default.
                if isinstance(cur, bool):
                    val = bool(val)
                elif isinstance(cur, int) and not isinstance(cur, bool):
                    val = int(val)
                elif isinstance(cur, float):
                    val = float(val)
                sub_kwargs[f.name] = val
        kwargs[sub] = cls(**sub_kwargs)
    return Calibration(**kwargs)


# --------------------------------------------------------------------------- diff / override


def _flatten(cal: Calibration) -> dict[str, Any]:
    """Flatten to a single ``{"subsystem.field": scalar}`` dict."""
    flat: dict[str, Any] = {}
    for sub, values in _to_plain(cal).items():
        for name, val in values.items():
            flat[f"{sub}.{name}"] = val
    return flat


def diff(a: Calibration, b: Calibration) -> list[str]:
    """Dotted keys whose value differs between ``a`` and ``b`` (sorted, deterministic).

    Compares the flattened scalar view, so numpy-vs-python scalar representation does
    not produce spurious differences.
    """
    fa, fb = _flatten(a), _flatten(b)
    changed = [k for k in fa if fa[k] != fb.get(k)]
    # include keys present only in b (shouldn't happen for same schema, but be safe)
    changed += [k for k in fb if k not in fa]
    return sorted(set(changed))


def apply_overrides(cal: Calibration, overrides: dict[str, Any]) -> Calibration:
    """Return a new Calibration with the given dotted-key fields replaced.

    ``overrides`` maps ``"subsystem.field"`` -> new value, e.g.
    ``{"geometry.near_floor": 1.5, "nav.cell_m": 0.05}``. The input is not mutated
    (frozen dataclasses; a fresh copy is built). Values are coerced to the field's
    existing scalar type (int stays int, float stays float, bool stays bool).

    Raises ``KeyError`` on an unknown subsystem or field name — a typo in a sweep
    config should fail loudly, not silently no-op.
    """
    # Bucket overrides per subsystem, validating keys as we go.
    per_sub: dict[str, dict[str, Any]] = {sub: {} for sub in _SUBSYSTEMS}
    for dotted, value in overrides.items():
        if "." not in dotted:
            raise KeyError(f"override key {dotted!r} is not a dotted 'subsystem.field' path")
        sub, _, name = dotted.partition(".")
        if sub not in _SUBSYSTEMS:
            raise KeyError(f"unknown calibration subsystem {sub!r} in override {dotted!r}")
        cfg = getattr(cal, sub)
        field_names = {f.name for f in fields(cfg)}
        if name not in field_names:
            raise KeyError(f"unknown field {name!r} for subsystem {sub!r} in override {dotted!r}")
        cur = getattr(cfg, name)
        if isinstance(cur, bool):
            value = bool(value)
        elif isinstance(cur, int) and not isinstance(cur, bool):
            value = int(value)
        elif isinstance(cur, float):
            value = float(value)
        per_sub[sub][name] = value

    new_subs: dict[str, Any] = {}
    for sub in _SUBSYSTEMS:
        cfg = getattr(cal, sub)
        changes = per_sub[sub]
        new_subs[sub] = replace(cfg, **changes) if changes else cfg
    return Calibration(**new_subs)


__all__ = [
    "NavTunables",
    "BudgetTunables",
    "Calibration",
    "default_calibration",
    "to_json",
    "from_json",
    "diff",
    "apply_overrides",
]
