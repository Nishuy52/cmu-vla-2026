"""RobotIO over recorded bag data: run the full QuestionController against a replay.

:class:`ReplayRobotIO` implements the :class:`core.interfaces.RobotIO` protocol backed by
a time-indexed :class:`MessageStore` of converted messages. A :class:`ReplayClock` advances
through bag time (manual :meth:`ReplayClock.step` or one tick per :meth:`ReplayRobotIO.tick`).
All ``latest_*`` getters return the newest message with ``t <= clock.now()`` (None before
the first such message). ``publish_*`` record into lists exactly like ``MockRobotIO`` so
tests can assert on emitted waypoints/markers/answers.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from core.interfaces import (
    IntAnswer,
    LidarScan,
    MarkerBox,
    OdomState,
    PanoFrame,
    Question,
    TerrainPatch,
    WaypointCmd,
)
from core.replay.bag_reader import (
    TOPIC_CAMERA,
    TOPIC_ODOM,
    TOPIC_QUESTION,
    TOPIC_SCAN,
    TOPIC_TERRAIN,
    TOPIC_TERRAIN_EXT,
    BagRecord,
    BagSource,
)

# Logical channel keys for the time-indexed store.
CH_PANO = "pano"
CH_SCAN = "scan"
CH_TERRAIN = "terrain"
CH_TERRAIN_EXT = "terrain_ext"
CH_ODOM = "odom"
CH_QUESTION = "question"

_TOPIC_TO_CHANNEL = {
    TOPIC_CAMERA: CH_PANO,
    TOPIC_SCAN: CH_SCAN,
    TOPIC_TERRAIN: CH_TERRAIN,
    TOPIC_TERRAIN_EXT: CH_TERRAIN_EXT,
    TOPIC_ODOM: CH_ODOM,
    TOPIC_QUESTION: CH_QUESTION,
}


@dataclass
class MessageStore:
    """Per-channel time-sorted (timestamp, msg) lists supporting latest-<= lookups.

    Timestamps within a channel are kept sorted so ``latest`` is an O(log n) bisect.
    This is the shared representation :class:`ReplayRobotIO` reads and that
    :func:`core.replay.fixtures.load_fixtures` reconstructs from ``.npz`` keyframes.
    """

    channels: dict[str, list[tuple[float, object]]] = field(default_factory=dict)

    def add(self, channel: str, t: float, msg: object) -> None:
        self.channels.setdefault(channel, []).append((float(t), msg))

    def finalize(self) -> "MessageStore":
        """Sort every channel by timestamp (idempotent). Returns self for chaining."""
        for lst in self.channels.values():
            lst.sort(key=lambda item: item[0])
        return self

    def times(self, channel: str) -> list[float]:
        return [t for t, _ in self.channels.get(channel, [])]

    def latest(self, channel: str, now: float) -> object | None:
        """Newest message on ``channel`` with timestamp <= ``now``; None if none."""
        lst = self.channels.get(channel)
        if not lst:
            return None
        ts = [t for t, _ in lst]
        idx = bisect.bisect_right(ts, now) - 1
        if idx < 0:
            return None
        return lst[idx][1]

    def all_times(self) -> list[float]:
        """Sorted union of every channel's timestamps (the replay tick schedule)."""
        out: list[float] = []
        for lst in self.channels.values():
            out.extend(t for t, _ in lst)
        return sorted(set(out))

    @classmethod
    def from_bag(cls, path, *, attach_odom: bool = True) -> "MessageStore":
        """Build a store from a bag, mapping topics to channels via the reader registry."""
        store = cls()
        src = BagSource(path)
        stream = src.frames() if attach_odom else iter(src)
        for rec in stream:
            channel = _TOPIC_TO_CHANNEL.get(rec.topic)
            if channel is None:
                continue
            store.add(channel, rec.t, rec.msg)
        return store.finalize()


FREE_RUN_DT_S: float = 0.2  # 5 Hz free-run cadence once the bag schedule is exhausted


class ReplayClock:
    """Clock that walks a fixed schedule of bag timestamps, then free-runs (implements Clock).

    ``now()`` returns the current schedule time. :meth:`step` advances to the next
    scheduled timestamp; once the schedule is exhausted it keeps advancing by a fixed
    ``free_run_dt`` (default 0.2 s = 5 Hz) so downstream budget/watchdog gates keep
    firing against a frozen world exactly as they would in production (the vehicle
    reaches end-of-bag long before the 510/570 s gates). :meth:`set` jumps to an
    arbitrary time.

    ``end_of_data`` reports whether the scheduled bag timestamps have been consumed
    (i.e. subsequent ticks are free-running past the recorded data).
    """

    def __init__(
        self,
        schedule: list[float],
        start: float | None = None,
        free_run_dt: float = FREE_RUN_DT_S,
    ) -> None:
        self._schedule = list(schedule)
        self._idx = 0
        self._free_run_dt = float(free_run_dt)
        if start is not None:
            self._t = float(start)
        elif self._schedule:
            self._t = float(self._schedule[0])
        else:
            self._t = 0.0

    def now(self) -> float:
        return self._t

    def step(self) -> float:
        """Advance one tick. Returns the new time.

        While scheduled timestamps remain, advances to the next one. Once the schedule
        is exhausted, advances by ``free_run_dt`` (frozen-world free-run) so time keeps
        moving toward the budget/watchdog gates instead of stalling at end-of-bag.
        """
        if self._idx + 1 < len(self._schedule):
            self._idx += 1
            self._t = float(self._schedule[self._idx])
        elif self._schedule:
            # Non-empty schedule exhausted: free-run at a fixed dt so gates eventually fire.
            self._idx = len(self._schedule)
            self._t += self._free_run_dt
        # Empty schedule: nothing to advance through — stay put at the start time.
        return self._t

    def set(self, t: float) -> None:
        self._t = float(t)

    @property
    def exhausted(self) -> bool:
        """True once the last scheduled timestamp has been reached (free-run territory)."""
        return self._idx + 1 >= len(self._schedule)

    @property
    def end_of_data(self) -> bool:
        """True once ticking has advanced past the final scheduled bag timestamp.

        Distinct from :attr:`exhausted`: ``exhausted`` is True while sitting *on* the
        last scheduled timestamp; ``end_of_data`` becomes True only once a further
        :meth:`step` has pushed the clock into free-run beyond that timestamp.
        """
        return bool(self._schedule) and self._idx >= len(self._schedule)


class ReplayRobotIO:
    """RobotIO backed by a :class:`MessageStore` and a :class:`ReplayClock`.

    Publish sinks are exposed as ``waypoints``, ``markers``, ``ints`` (like MockRobotIO).
    Build from a bag with :meth:`from_bag`, or pass a prebuilt store (e.g. from fixtures).
    """

    def __init__(self, store: MessageStore, clock: ReplayClock | None = None) -> None:
        self.store = store.finalize()
        self._clock = clock if clock is not None else ReplayClock(store.all_times())
        self.waypoints: list[WaypointCmd] = []
        self.markers: list[MarkerBox] = []
        self.ints: list[IntAnswer] = []
        # Optional override for what clock() hands to time-budget consumers (the FSM),
        # leaving the message-lookup getters below on the true bag clock. Used by the
        # runner's --budget-scale to compress the FSM's budget gates for short bags.
        self._budget_clock: object | None = None

    @classmethod
    def from_bag(cls, path) -> "ReplayRobotIO":
        return cls(MessageStore.from_bag(path))

    # ------------------------------------------------------------------ advance

    def tick(self) -> float:
        """Advance the clock one tick. Returns the new time.

        Walks bag timestamps until the schedule is exhausted, then free-runs at the
        clock's fixed dt with all ``latest_*`` getters returning the final (frozen)
        messages — so budget/watchdog gates keep firing past end-of-bag.
        """
        return self._clock.step()

    @property
    def end_of_data(self) -> bool:
        """True once ticking has advanced past the last recorded bag message (free-run)."""
        return self._clock.end_of_data

    # ------------------------------------------------------------------ getters

    def question(self) -> Question | None:
        return self.store.latest(CH_QUESTION, self._clock.now())  # type: ignore[return-value]

    def latest_pano(self) -> PanoFrame | None:
        return self.store.latest(CH_PANO, self._clock.now())  # type: ignore[return-value]

    def latest_scan(self) -> LidarScan | None:
        return self.store.latest(CH_SCAN, self._clock.now())  # type: ignore[return-value]

    def latest_terrain(self, extended: bool = False) -> TerrainPatch | None:
        channel = CH_TERRAIN_EXT if extended else CH_TERRAIN
        return self.store.latest(channel, self._clock.now())  # type: ignore[return-value]

    def latest_odom(self) -> OdomState | None:
        return self.store.latest(CH_ODOM, self._clock.now())  # type: ignore[return-value]

    # ---------------------------------------------------------------- publishers

    def publish_waypoint(self, wp: WaypointCmd) -> None:
        self.waypoints.append(wp)

    def publish_marker(self, box: MarkerBox) -> None:
        self.markers.append(box)

    def publish_int(self, ans: IntAnswer) -> None:
        self.ints.append(ans)

    def clock(self):
        """Clock exposed to time-budget consumers (the FSM).

        Normally the true bag :class:`ReplayClock`; when a budget-scale override is set
        (see :meth:`set_budget_clock`) that wrapper is returned instead so the FSM's fixed
        gates fire at scaled bag-time. The ``latest_*`` getters always use the true bag
        clock (``self._clock``), so message lookups stay on real recorded time.
        """
        return self._budget_clock if self._budget_clock is not None else self._clock

    def set_budget_clock(self, clock: object | None) -> None:
        """Override (or clear, with None) the clock returned by :meth:`clock`."""
        self._budget_clock = clock

    def raw_clock(self) -> ReplayClock:
        """The underlying true bag clock, regardless of any budget-clock override."""
        return self._clock
