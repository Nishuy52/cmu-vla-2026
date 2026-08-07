"""rclpy adapter node — the ONLY ROS-dependent module in this repo.

UNTESTED DRAFT (Phase 2). Cannot run on the Windows dev box: rclpy is unavailable there.
Every ROS symbol is imported at module top; the parse-guard test (tests/ros_adapter/
test_importable.py) only ``compile()``s this file, it does not import it. On Ubuntu, inside
the ai_module container, this imports normally.

Role
----
This is the single seam between the pure-Python ``core`` (see core/interfaces.py::RobotIO)
and ROS 2. It contains ZERO task logic — no perception, no counting, no exploration. It only:

  1. subscribes the six allowed system-output topics (docs/upstream_notes.md §3),
  2. converts each incoming message to the frozen core dataclass, reusing the converters in
     core.replay.bag_reader (rclpy message layouts match what those converters read),
  3. latches the latest converted value per topic behind a lock (thread-safe: rclpy
     executor callbacks write, the 5 Hz timer reads),
  4. drives ``QuestionController.tick(self)`` at 5 Hz — ``self`` IS the RobotIO,
  5. publishes the three answer topics (Marker / Pose2D / Int32).

Optionally (visualization only, behind the ``debug_viz`` launch arg, default OFF for eval) it
also republishes the tracked instance map and a planned-path breadcrumb for RVIZ — see the
"OPTIONAL DEBUG VISUALIZATION" block in ``__init__``. When debug_viz is false these publishers
and their timer are never created, so the eval path carries zero debug overhead.

Dependency direction is one-way: ros_adapter -> core. core never imports ros_adapter and
never imports rclpy, so ``pytest`` on the core suite stays ROS-free.

Topic contract (docs/upstream_notes.md §3, gotchas 5/6/13)
----------------------------------------------------------
Subscriptions (system -> AI):
    /camera/image        sensor_msgs/Image        10 Hz   -> PanoFrame
    /registered_scan     sensor_msgs/PointCloud2   5 Hz   -> LidarScan
    /terrain_map         sensor_msgs/PointCloud2   5 Hz   -> TerrainPatch(extended=False)
    /terrain_map_ext     sensor_msgs/PointCloud2   5 Hz   -> TerrainPatch(extended=True)
    /state_estimation    nav_msgs/Odometry       100+ Hz  -> OdomState
    /challenge_question  std_msgs/String           1 Hz   -> Question   (reliable QoS)
Publications (AI -> system):
    /way_point_with_heading  geometry_msgs/Pose2D             (theta ALWAYS 0)
    /selected_object_marker  visualization_msgs/Marker  CUBE, map frame, scale=full extents
    /numerical_response      std_msgs/Int32

QoS: the five high-rate sensor streams use SENSOR_DATA (best-effort, keep-last depth 5) so a
dropped frame never blocks; /challenge_question is subscribed TWICE — a primary RELIABLE +
VOLATILE sub matching the dummy's default publisher profile, plus a second RELIABLE +
TRANSIENT_LOCAL sub in case the evaluator offers durability. Both feed the same idempotent
latch, so whichever the publisher matches wins and a duplicate delivery is a no-op. Requesting
TRANSIENT_LOCAL *alone* against a VOLATILE publisher would silence the whole run (SYS-F2), so
the volatile sub is load-bearing; the 1 Hz republish (gotcha 3) covers late joins regardless.
Publishers are reliable depth 5, matching the dummy's default profile.
"""
from __future__ import annotations

import os
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Int32, String
from visualization_msgs.msg import Marker, MarkerArray

# core is pure-Python and ROS-free. This is the one-way dependency ros_adapter -> core.
from core.interfaces import (
    Clock,
    IntAnswer,
    LidarScan,
    MarkerBox,
    OdomState,
    PanoFrame,
    Question,
    TerrainPatch,
    WaypointCmd,
)
from core.fsm.controller import QuestionController, State
from core.heads import build_callables
from core.heads.factory import _STANDING_VOCAB_NOUNS
from core.llm.config import build_chat_fns_with_tiers, load_config
from core.parsing import ladder as parse_ladder
from core.perception.async_pipeline import AsyncPerceptionWorker
from core.perception.colored_map import ColoredVoxelMap
from core.perception.detector import GroundingDinoDetector, refresh_prompt
from core.perception.scene_index import BasicSceneIndex
from core.perception.tracker import PerceptionPipeline
from core.perception.vision_encode import resolve_encode_fn
from ros_adapter.cloud_packing import pack_colored_cloud
from ros_adapter.frame_watchdog import FrameWatchdog
from core.replay.bag_reader import (
    TOPIC_CAMERA,
    TOPIC_ODOM,
    TOPIC_QUESTION,
    TOPIC_SCAN,
    TOPIC_TERRAIN,
    TOPIC_TERRAIN_EXT,
    image_to_pano,
    odom_to_state,
    pointcloud_to_lidar,
    pointcloud_to_terrain,
    string_to_question,
)

# Output topic names (docs/upstream_notes.md §3). Publish /selected_object_marker with the
# leading slash per gotcha 5 (the dummy omits it but README + eval use the qualified name).
TOPIC_WAYPOINT = "/way_point_with_heading"
TOPIC_MARKER = "/selected_object_marker"
TOPIC_NUMERICAL = "/numerical_response"

# Debug-only visualization topics (published ONLY when the debug_viz launch arg is true —
# see the "OPTIONAL DEBUG VISUALIZATION" block below). These are NOT part of the answer
# contract and never appear on the eval path (debug_viz defaults false in ai_module.launch.py).
TOPIC_DBG_INSTANCE_MAP = "/ai_module/instance_map"
TOPIC_DBG_PLANNED_PATH = "/ai_module/planned_path"
# Live colored voxel map (T6): the growing colored reconstruction the robot's pose display
# moves through in RVIZ. Debug-only; never on the eval path (debug_viz defaults false).
TOPIC_DBG_COLORED_CLOUD = "/debug/colored_cloud"

# Detector selection env var (H6 / SYS-F1). "none" (default) keeps the offline stub — no
# torch/GroundingDINO on the Windows dev box — so the node behaves exactly as today (empty
# live index -> the SUBMISSION-BLOCKER shout). "grounding_dino" constructs the lazy
# GroundingDINO seam in core.perception.detector; its torch import stays lazy until the first
# real frame, so merely selecting it does not require the model deps at construction.
ENV_DETECTOR = "VLA_DETECTOR"
DETECTOR_NONE = "none"
DETECTOR_GDINO = "grounding_dino"
# Issue #86: dev-only offload — keeps the ROS stack local but runs GDINO inference on a
# remote cluster GPU over HTTP (reached via an SSH tunnel), so the laptop GPU's EC
# power-wedge (fires at GroundingDINO model load) never gets tripped. Never the submission
# path: the scored image never talks to a network detector.
DETECTOR_REMOTE = "remote"

# Dev-only (SoC cluster full-stack runs): subscribe /camera/image/compressed and decode
# in-process instead of the raw /camera/image feed. Unprivileged CycloneDDS on the cluster
# drops ~3.7 MB raw frames (UDP fragments overflow the default kernel socket buffers, which
# can't be raised without root), while the ~100 KB jpeg stream survives. The decoded frame
# re-enters the exact _on_image path, so everything downstream is identical. Never the
# submission path: the scored image subscribes the contract's raw topic.
ENV_CAMERA_COMPRESSED = "VLA_CAMERA_COMPRESSED"

# Issue #88: on a live run GroundingDINO forwards (~1s) were running inline in
# _maybe_process_perception, which is called from the 5 Hz tick timer -> every forward
# blocked waypoint publication and subscription servicing for that entire second, collapsing
# the drive loop to ~0.8 Hz. Default ON (perception now always dispatches off the tick
# thread via AsyncPerceptionWorker); set VLA_PERCEPTION_SYNC=1 to force the old inline
# behaviour back (debugging only — reintroduces the cadence stall). The offline/battery path
# (core.runner.single) never imports this module, so it is unaffected either way.
ENV_PERCEPTION_SYNC = "VLA_PERCEPTION_SYNC"


def make_detector(logger=None):
    """Build the perception detector from ``VLA_DETECTOR`` (default: stub / None).

    Returns ``None`` for ``none`` (the offline stub — the PerceptionPipeline is still
    constructed and fed, but with no detector it produces zero instances, keeping today's
    empty-index behaviour and the SUBMISSION-BLOCKER shout). Returns a
    :class:`~core.perception.detector.GroundingDinoDetector` for ``grounding_dino`` — the
    real Phase-2 seam; its torch/groundingdino import is lazy (first ``__call__``), so this
    constructs cleanly on Windows and only fails loudly at inference time if the model deps
    are absent. Returns a :class:`~core.perception.remote_detector.RemoteDetector` for
    ``remote`` — the dev-only offload seam (issue #86); if ``VLA_REMOTE_DETECTOR_URL`` is
    unset, construction raises ``RuntimeError``, which is caught here and logged, falling
    back to the offline stub rather than crashing boot. An unrecognised value falls back to
    the stub with a warning.
    """
    choice = os.environ.get(ENV_DETECTOR, DETECTOR_NONE).strip().lower()
    if choice in ("", DETECTOR_NONE):
        return None
    if choice == DETECTOR_GDINO:
        if logger is not None:
            logger.info(
                "%s=%s: constructing the GroundingDINO detector seam (torch import is lazy; "
                "real inference lands in Phase 2)." % (ENV_DETECTOR, DETECTOR_GDINO)
            )
        return GroundingDinoDetector()
    if choice == DETECTOR_REMOTE:
        from core.perception.remote_detector import RemoteDetector

        try:
            detector = RemoteDetector()
        except RuntimeError as exc:
            if logger is not None:
                logger.warn(
                    "%s=%s: %s Falling back to the offline stub (no detector)."
                    % (ENV_DETECTOR, DETECTOR_REMOTE, exc)
                )
            return None
        if logger is not None:
            logger.info(
                "%s=%s: offloading GroundingDINO inference to %s (dev-only mode, never "
                "the submission path)." % (ENV_DETECTOR, DETECTOR_REMOTE, detector.url)
            )
        return detector
    if logger is not None:
        logger.warn(
            "%s=%r is not recognised (expected %s|%s|%s); falling back to the offline stub "
            "(no detector)."
            % (ENV_DETECTOR, choice, DETECTOR_NONE, DETECTOR_GDINO, DETECTOR_REMOTE)
        )
    return None


def _tile_pixel_dims(pipeline) -> tuple[int, int]:
    """Return (tile_w, tile_h) in pixels for the CP2 bbox-bounds gate (OR-F8, coord item 3).

    Uses the same projection formula as core.perception.tiling.project_tiles so the CP2
    dimension check matches the tiles the detector actually sees. Reads the pipeline's live
    (hfov, vfov) when perception is on; otherwise the module defaults (480x640 tile).
    """
    import numpy as np

    from core.perception.tiling import (
        DEFAULT_TILE_HFOV,
        DEFAULT_TILE_VFOV,
        PANO_HEIGHT,
        PANO_VFOV,
        PANO_WIDTH,
    )

    hfov = getattr(pipeline, "hfov", DEFAULT_TILE_HFOV) if pipeline is not None else DEFAULT_TILE_HFOV
    vfov = getattr(pipeline, "vfov", DEFAULT_TILE_VFOV) if pipeline is not None else DEFAULT_TILE_VFOV
    tile_w = int(round(PANO_WIDTH * hfov / (2.0 * np.pi)))
    tile_h = int(round(PANO_HEIGHT * vfov / PANO_VFOV))
    return tile_w, tile_h


TICK_HZ = 5.0  # QuestionController.tick() cadence (core/fsm/controller.py docstring)
DEBUG_VIZ_HZ = 1.0  # instance-map republish cadence when debug_viz is on
DEBUG_PATH_MAX = 500  # cap on retained waypoint breadcrumb points (bounded memory)
DEBUG_CLOUD_PERIOD_S = 2.0  # publish the FULL colored map at most this often (RVIZ stays responsive)
# issue #60: waypoints stream at the 5 Hz drive rate during DRIVE_OUT (IF-F4) — logging one
# INFO line per waypoint would spam at 0.6 Hz sustained. Log the running count at most this
# often instead.
WAYPOINT_COUNT_LOG_PERIOD_S = 30.0
EVENTS_TAIL_ON_DONE = 10  # issue #60: how many flight-recorder events to surface at DONE


def _reliable_transient_qos(depth: int = 5) -> QoSProfile:
    """RELIABLE + TRANSIENT_LOCAL — for /challenge_question (single, late-join-safe)."""
    return QoSProfile(
        depth=depth,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def _reliable_qos(depth: int = 5) -> QoSProfile:
    """Plain reliable/volatile depth-N — matches the dummy's default publisher profile."""
    return QoSProfile(
        depth=depth,
        history=HistoryPolicy.KEEP_LAST,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )


class _LedgerProxy:
    """Duck-typed CallLedger view that resolves the live ledger lazily per call (H8).

    The checkpoint builders (build_verifier/anchor_confirm/miss_recovery/frontier_select) bind
    a ledger at construction time, but the controller's real CallLedger only exists after
    question intake. This proxy is bound at build time and delegates ``allow``/``record`` to
    whatever ``resolve()`` returns at call time; when the ledger is not yet available it fails
    open (allow -> True, record -> no-op), matching guarded_call's own duck-typing.
    """

    def __init__(self, resolve) -> None:
        self._resolve = resolve

    def allow(self, checkpoint: str) -> bool:
        led = self._resolve()
        if led is None:
            return True
        return bool(led.allow(checkpoint))

    def record(self, checkpoint: str, duration: float, tier: str):
        led = self._resolve()
        if led is None:
            return None
        return led.record(checkpoint, duration, tier)


class _NodeClock:
    """core.interfaces.Clock backed by the ROS node clock (seconds as float)."""

    def __init__(self, node: Node) -> None:
        self._node = node

    def now(self) -> float:
        return self._node.get_clock().now().nanoseconds * 1e-9


class AdapterNode(Node):
    """rclpy node implementing core.interfaces.RobotIO by latching the latest message.

    The class structurally satisfies RobotIO (the getters + publishers + clock); it is
    passed as ``io`` straight into ``QuestionController.tick``. All latch reads/writes are
    guarded by ``self._lock`` because rclpy subscription callbacks and the timer callback
    may run on different executor threads (MultiThreadedExecutor) — with the default
    single-threaded executor the lock is uncontended but still correct.
    """

    def __init__(self) -> None:
        super().__init__("vla_ai_module")

        self._lock = threading.Lock()
        # Latest-message latches (None until first receipt; getters never block).
        self._pano: PanoFrame | None = None
        self._scan: LidarScan | None = None
        self._terrain: TerrainPatch | None = None
        self._terrain_ext: TerrainPatch | None = None
        self._odom: OdomState | None = None
        self._question: Question | None = None
        # issue #174: frame-arrival watchdog — pure observability, no recovery. Baselines
        # at construction so a slot whose camera never starts also ages out and warns.
        self._frame_watchdog = FrameWatchdog(clock=time.monotonic)
        # NOTE: must NOT be named `self._clock` — rclpy.node.Node.__init__ already sets
        # `self._clock = ROSClock()` and Node.get_clock() / several internal rclpy methods
        # (e.g. declare_parameter's _set_parameters_atomically_common) read that exact
        # attribute directly. Overwriting it here previously caused get_clock() to return
        # this wrapper instead of the real clock, so _NodeClock.now() -> get_clock().now()
        # recursed into itself (RecursionError, confirmed on Ubuntu first boot). Renamed to
        # avoid the collision.
        self._robotio_clock = _NodeClock(self)

        # ---- Subscriptions (six allowed system-output topics) -----------------
        if os.environ.get(ENV_CAMERA_COMPRESSED, "").strip().lower() in ("1", "true", "yes"):
            # Dev-only cluster path (see ENV_CAMERA_COMPRESSED above): jpeg in, same
            # _on_image pipeline out. CompressedImage import is local so the contract
            # path never touches it.
            from sensor_msgs.msg import CompressedImage

            self.create_subscription(
                CompressedImage,
                TOPIC_CAMERA + "/compressed",
                self._on_image_compressed,
                qos_profile_sensor_data,
            )
            self.get_logger().warning(
                "VLA_CAMERA_COMPRESSED=1: dev-only compressed camera feed "
                "(%s/compressed -> in-process decode). Not the submission path."
                % TOPIC_CAMERA
            )
        else:
            self.create_subscription(
                Image, TOPIC_CAMERA, self._on_image, qos_profile_sensor_data
            )
        self.create_subscription(
            PointCloud2, TOPIC_SCAN, self._on_scan, qos_profile_sensor_data
        )
        self.create_subscription(
            PointCloud2, TOPIC_TERRAIN, self._on_terrain, qos_profile_sensor_data
        )
        self.create_subscription(
            PointCloud2, TOPIC_TERRAIN_EXT, self._on_terrain_ext, qos_profile_sensor_data
        )
        self.create_subscription(
            Odometry, TOPIC_ODOM, self._on_odom, qos_profile_sensor_data
        )
        # /challenge_question: the PRIMARY subscription is RELIABLE + VOLATILE, matching the
        # upstream dummy's default publisher profile (docs/upstream_notes.md §6). The evaluator
        # is closed-source but almost certainly publishes with defaults (VOLATILE); a subscriber
        # REQUESTING TRANSIENT_LOCAL against a VOLATILE publisher is DDS-incompatible => zero
        # messages => zero questions => whole run scores zero (SYS-F2). The 1 Hz republish
        # (gotcha 3) already provides late-join safety, so VOLATILE loses nothing.
        self.create_subscription(
            String, TOPIC_QUESTION, self._on_question, _reliable_qos()
        )
        # SECOND subscription at TRANSIENT_LOCAL, feeding the SAME latch: matches a publisher
        # that offers durability instead. Both matching is harmless — _on_question latches the
        # first non-empty text and ignores the rest (idempotent), so a duplicate delivery is a
        # no-op. This is belt-and-braces; the volatile sub above is the load-bearing one.
        self.create_subscription(
            String, TOPIC_QUESTION, self._on_question, _reliable_transient_qos()
        )

        # ---- Publishers (three answer topics) ---------------------------------
        self._pub_waypoint = self.create_publisher(
            Pose2D, TOPIC_WAYPOINT, _reliable_qos()
        )
        self._pub_marker = self.create_publisher(
            Marker, TOPIC_MARKER, _reliable_qos()
        )
        self._pub_int = self.create_publisher(
            Int32, TOPIC_NUMERICAL, _reliable_qos()
        )

        # ---- OPTIONAL DEBUG VISUALIZATION (default OFF — eval path stays clean) ----
        # Visualization only: zero task logic. When debug_viz is false (the eval default,
        # set in ai_module.launch.py) NO debug publisher, timer, or breadcrumb state is
        # created, so the eval path allocates and executes nothing extra. Only when
        # ai_module_debug.launch.py passes debug_viz:=true do these come alive.
        self.declare_parameter("debug_viz", False)
        self._debug_viz = bool(
            self.get_parameter("debug_viz").get_parameter_value().bool_value
        )
        # Throttle period (s) for the full colored-map republish (debug_viz only).
        self.declare_parameter("colored_cloud_period", DEBUG_CLOUD_PERIOD_S)
        self._colored_cloud_period = float(
            self.get_parameter("colored_cloud_period").get_parameter_value().double_value
        ) or DEBUG_CLOUD_PERIOD_S
        self._dbg_path_pts: list[tuple[float, float]] = []
        # Live colored voxel map + its publisher exist only under debug_viz. All the map
        # math is in core.perception.ColoredVoxelMap / ros_adapter.cloud_packing (both
        # pure-numpy, Windows-tested); the rclpy code here is a thin publish shim.
        self._colored_map: ColoredVoxelMap | None = None
        if self._debug_viz:
            self._pub_dbg_instances = self.create_publisher(
                MarkerArray, TOPIC_DBG_INSTANCE_MAP, _reliable_qos()
            )
            self._pub_dbg_path = self.create_publisher(
                Marker, TOPIC_DBG_PLANNED_PATH, _reliable_qos()
            )
            self._pub_dbg_cloud = self.create_publisher(
                PointCloud2, TOPIC_DBG_COLORED_CLOUD, _reliable_qos()
            )
            self._colored_map = ColoredVoxelMap()
            self._dbg_timer = self.create_timer(
                1.0 / DEBUG_VIZ_HZ, self._on_debug_viz
            )
            # Separate (slower) timer so the FULL colored cloud is republished at most
            # once per colored_cloud_period, NOT per ingest — RVIZ stays responsive.
            self._dbg_cloud_timer = self.create_timer(
                self._colored_cloud_period, self._on_colored_cloud
            )
            self.get_logger().warn(
                "debug_viz=true: publishing %s (1 Hz) + %s + %s (every %.1f s). DEBUG "
                "ONLY — do not enable for evaluation."
                % (TOPIC_DBG_INSTANCE_MAP, TOPIC_DBG_PLANNED_PATH,
                   TOPIC_DBG_COLORED_CLOUD, self._colored_cloud_period)
            )

        # ---- LLM provider config (H8 / SYS-F5) --------------------------------
        # Loaded once at boot from env + optional llm_config.json (env wins; no keys in code).
        # A wholly unconfigured environment yields all-None slots -> build_chat_fns([]) empty
        # -> the parse ladder goes straight to the deterministic regex floor, and every
        # checkpoint seam stays None/off (offline determinism preserved for tests + dark net).
        self._llm_config = load_config()
        _chat_tiers = build_chat_fns_with_tiers(self._llm_config)
        self._chat_fns = [fn for _, fn in _chat_tiers]
        self._chat_tier_names = tuple(name for name, _ in _chat_tiers)
        self._llm_configured = bool(self._chat_fns)
        if self._llm_configured:
            self.get_logger().info(
                "LLM ladder configured: %d provider tier(s): %s (regex floor always last)."
                % (len(self._chat_fns), ", ".join(self._chat_tier_names))
            )
        else:
            self.get_logger().info(
                "no LLM provider configured: parse is regex-only and all checkpoint seams are "
                "off (deterministic offline path)."
            )

        # ---- Vision encode_fn (Gate 4 item 2 / H14 / OR-F10) ------------------
        # The vision checkpoints (CP2/CP3/CP5) default to raw .npy bytes (test-friendly,
        # no image lib). resolve_encode_fn swaps in a real JPEG encoder when Pillow is
        # importable (the common case — see core.perception.vision_encode); it returns
        # None (soft-degrade, logged) when Pillow is absent, and _build_checkpoint_seams
        # then falls back to each builder's own .npy default. Resolved once at boot,
        # never re-checked per call.
        self._vision_encode_fn = resolve_encode_fn(self.get_logger())

        # ---- Perception pipeline seam (H6 / SYS-F1) ---------------------------
        # The heads resolve against a live SceneIndex. Following the runner/single.py
        # _ScriptedPerception pattern, a PerceptionPipeline is fed (pano, scan) pairs on
        # new-pano ticks (see _maybe_process_perception) and ITS index is the one the
        # controller sees when perception is on. The detector is pluggable via VLA_DETECTOR:
        #   * VLA_DETECTOR=none (default): make_detector() -> None. Windows has no detector, so
        #     we keep the empty BasicSceneIndex([]) stub and DO NOT construct a pipeline (a
        #     None detector cannot run). Behaviour is exactly today's -> the SUBMISSION-BLOCKER
        #     shout below fires and answers come from FSM floors only.
        #   * VLA_DETECTOR=grounding_dino: make_detector() -> the lazy GroundingDINO seam; the
        #     pipeline's live index replaces the stub and grows as frames are grounded.
        detector = make_detector(self.get_logger())
        # Kept alongside self._perception (issue #34): _build_controller_callables hands
        # this SAME instance to build_callables(detector=...) so HeadState.bind refreshes
        # its .prompt from the question nouns the moment the plan latches — otherwise a
        # GroundingDinoDetector never leaves its boot-time empty prompt and every __call__
        # short-circuits to zero detections for the whole run. None when perception is off
        # (VLA_DETECTOR=none); refresh_prompt no-ops on None.
        self._detector: GroundingDinoDetector | None = detector
        self._perception: PerceptionPipeline | None = None
        self._perception_worker: AsyncPerceptionWorker | None = None
        self._last_pano_t: float | None = None
        if detector is not None:
            self._perception = PerceptionPipeline(detector, index=BasicSceneIndex([]))
            # The controller sees the pipeline's LIVE index (mutated in place as frames fuse).
            self._scene_index = self._perception.index
            # issue #200 "detector cold start": _maybe_process_perception is driven from the
            # very first tick (below, at self._timer construction), independent of question
            # latch — but the detector itself is constructed with an empty .prompt ("" ==
            # zero detections on every __call__, see refresh_prompt's docstring) until
            # something refreshes it. Before this fix that first refresh only happened inside
            # build_callables() (core/heads/factory.py), which _build_controller_callables
            # only calls once the QuestionController is built AT QUESTION LATCH — so every
            # keyframe ticked between node start and latch (median 26.8 s; the orientation
            # sweep runs 47.8-87.0 s) ran the detector blind. Priming the SAME standing
            # 114-noun vocabulary here, at construction, closes that window: the live
            # detector instance grounds on the boot vocabulary from the first keyframe,
            # persisting its instances into this same self._scene_index (never recreated,
            # see below), and build_callables' own boot-vocab refresh plus HeadState.bind's
            # question-latch refresh (unchanged) still fire exactly as before once a
            # question arrives. Pure string assignment (no camera/lidar/torch dependency),
            # so this cannot fail on missing sensor readiness; wrapped defensively anyway so
            # a detector-construction quirk can never abort node boot (H14 boot-safety
            # pattern used throughout this file).
            try:
                refresh_prompt(self._detector, (), _STANDING_VOCAB_NOUNS)
            except Exception as exc:  # boot priming must never crash node construction
                self.get_logger().error(
                    "perception boot-vocab priming error: %s" % exc
                )
            # issue #84: stash a debug-only backref so core.heads.explore_debug can read
            # the pipeline's keyframe counter off the index it already holds, without new
            # plumbing through ExploreHead. Only set while the debug dump is enabled —
            # gated the same way explore_debug itself is, so the unset case is untouched.
            if os.environ.get("VLA_EXPLORE_DEBUG_DIR"):
                self._scene_index._debug_perception = self._perception
            # issue #88: dispatch pipeline.process() off the tick thread by default (see
            # ENV_PERCEPTION_SYNC above). BasicSceneIndex's internal lock makes the resulting
            # concurrent index reads (heads, on the tick thread) safe against the worker
            # thread's writes.
            if os.environ.get(ENV_PERCEPTION_SYNC, "").strip().lower() not in (
                "1", "true", "yes",
            ):
                self._perception_worker = AsyncPerceptionWorker(
                    self._perception,
                    on_error=lambda exc: self.get_logger().error(
                        "perception worker error: %s" % exc
                    ),
                )
                self.get_logger().info(
                    "perception dispatch: threaded (AsyncPerceptionWorker) — set "
                    "%s=1 to force synchronous (debug only)." % ENV_PERCEPTION_SYNC
                )
            else:
                self.get_logger().warning(
                    "%s=1: perception runs SYNCHRONOUSLY on the tick thread — a slow "
                    "detector forward WILL stall waypoint publication (issue #88). "
                    "Debug only, never for a live/eval run." % ENV_PERCEPTION_SYNC
                )
        else:
            self._scene_index = BasicSceneIndex([])
        self._controller: QuestionController | None = None
        # issue #60: waypoint publish count since latch (throttled log, not per-waypoint) and
        # a one-shot guard so the DONE events-tail line is emitted exactly once per question.
        self._waypoint_count = 0
        self._waypoint_count_last_log = time.monotonic()
        self._done_events_logged = False

        # SUBMISSION-BLOCKER shout: the node came up with the empty BasicSceneIndex([]) stub —
        # perception is NOT wired into this node (VLA_DETECTOR=none), so every question is
        # answered from an empty world (numerical floor, origin marker, spawn-point waypoint)
        # and the run scores luck only (SYS-F1 / H6). This is acknowledged Phase-2 work; the
        # loud line exists so it cannot pass a smoke test silently. Set VLA_DETECTOR and feed a
        # live index to clear it.
        if self._perception is None and not self._scene_index.all_instances():
            self.get_logger().error(
                "SUBMISSION-BLOCKER: scene index is the empty stub (%s=none) — perception is "
                "NOT wired; answers come from FSM floors only. Do not submit until "
                "instances_tracked > 0 on a live scene (phase2_playbook gate)."
                % ENV_DETECTOR
            )

        # ---- Startup assert: network provider must NOT pair with the .npy encoder ----
        # (H14 / OR-F10). A configured network LLM/VLM provider paired with the default raw
        # .npy image encoder means CP2/CP3/CP5 vision calls send bytes a real vision API
        # rejects — they would silently NEVER work at eval while every offline test stays
        # green. Gate 4 wired a real JPEG encode_fn (self._vision_encode_fn, above) for the
        # common case (Pillow present); this assert now only fires when that resolution
        # came back None (Pillow missing) AND a network provider is configured — i.e. the
        # one remaining case where a vision checkpoint would silently send bytes a real
        # vision API rejects.
        self._assert_encoder_provider_consistency()

        # ---- Sim-time guard ---------------------------------------------------
        # The watchdog arithmetic keys off get_clock().now(). If use_sim_time is true without a
        # live /clock source the node clock freezes at 0, elapsed() pins below the 60 s ORIENT
        # gate, and the watchdog floor NEVER fires => permanent silence (SYS-F6 residual). The
        # eval path must run on the wall clock; shout loudly if it is ever set on this node.
        if self.get_parameter("use_sim_time").get_parameter_value().bool_value:
            self.get_logger().error(
                "SUBMISSION-BLOCKER: use_sim_time is TRUE on vla_ai_module. A frozen sim clock "
                "pins elapsed() below the ORIENT gate and silences the watchdog forever. The "
                "eval path must use the wall clock — unset use_sim_time."
            )

        # ---- 5 Hz drive timer -------------------------------------------------
        self._timer = self.create_timer(1.0 / TICK_HZ, self._on_tick)
        self.get_logger().info(
            "vla_ai_module up: subscribed 6 topics, ticking QuestionController at %.1f Hz"
            % TICK_HZ
        )

    # ------------------------------------------------------------------ callbacks
    # Each callback converts once (off the timer thread) and latches under the lock.
    # bag_reader converters take (msg, bag_ns); we pass the receive time in ns as the
    # header-absent fallback — rclpy messages usually carry a populated header.stamp,
    # which the converters prefer via _header_time().

    def _now_ns(self) -> int:
        return self.get_clock().now().nanoseconds

    def _on_image(self, msg: Image) -> None:
        # issue #174: mark frame arrival first — every raw or decoded-compressed frame
        # (both paths funnel through here) resets the quiet-feed clock.
        self._frame_watchdog.on_frame()
        # Attach the latest odom so PanoFrame.odom is populated (mirrors BagSource.frames()).
        with self._lock:
            odom = self._odom
        pano = image_to_pano(msg, self._now_ns(), odom=odom)
        with self._lock:
            self._pano = pano

    def _on_image_compressed(self, msg) -> None:
        # Dev-only (ENV_CAMERA_COMPRESSED): decode the jpeg and re-enter _on_image with a
        # synthesized raw Image, so the entire downstream pipeline is byte-identical to the
        # contract path. cv2/numpy imports are lazy — this callback only exists when the
        # env flag opted in. A frame that fails to decode is dropped (never raises).
        try:
            import cv2
            import numpy as np

            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return
            out = Image()
            out.header = msg.header
            out.height, out.width = int(img.shape[0]), int(img.shape[1])
            out.encoding = "bgr8"
            out.is_bigendian = 0
            out.step = int(img.shape[1]) * 3
            out.data = img.tobytes()
            self._on_image(out)
        except Exception as exc:  # a decode glitch must never disturb the drive loop
            self.get_logger().error("compressed camera decode error: %s" % exc)

    def _on_scan(self, msg: PointCloud2) -> None:
        scan = pointcloud_to_lidar(msg, self._now_ns())
        with self._lock:
            self._scan = scan
            pano = self._pano  # nearest-in-time pano = latest latched (node's sync model)
        # Debug colored map: ingest this (scan, latest-pano) pair. Visualization only —
        # touches no answer state; guarded so the eval path never runs it. Never raises.
        if self._colored_map is not None and pano is not None:
            try:
                self._colored_map.ingest(scan, pano)
            except Exception as exc:  # a viz-map glitch must never disturb the drive loop
                self.get_logger().error("colored_map ingest error: %s" % exc)

    def _on_terrain(self, msg: PointCloud2) -> None:
        patch = pointcloud_to_terrain(msg, self._now_ns(), extended=False)
        with self._lock:
            self._terrain = patch

    def _on_terrain_ext(self, msg: PointCloud2) -> None:
        patch = pointcloud_to_terrain(msg, self._now_ns(), extended=True)
        with self._lock:
            self._terrain_ext = patch

    def _on_odom(self, msg: Odometry) -> None:
        state = odom_to_state(msg, self._now_ns())
        with self._lock:
            self._odom = state

    def _on_question(self, msg: String) -> None:
        # Latch the FIRST non-empty question; ignore the 1 Hz republish (gotcha 3). The
        # QuestionController also dedupes, but latching here avoids re-converting each second.
        if not msg.data:
            return
        with self._lock:
            if self._question is not None:
                already = self._question.text
                same_text = msg.data == already
            else:
                already = None
                same_text = False
            if already is None:
                self._question = string_to_question(msg, self._now_ns())
                latched_text = self._question.text
        if already is None:
            self.get_logger().info("question latched: %r" % latched_text)
            return
        if same_text:
            return  # same-text 1 Hz republish -> silent no-op (idempotent latch)
        # DELIBERATE POLICY (H14 / SYS-F14-2): a SECOND, DIFFERENT question arrived on a
        # running stack. The challenge relaunches a fresh process per question (upstream
        # gotcha 1), so this is an upstream/harness anomaly. We KEEP the first question (do
        # NOT switch mid-run — switching would abandon a partially-solved question and its
        # budget) and log the drop LOUDLY so it cannot pass silently. Documented rather than
        # exit(0): a hard exit assumes a container restart policy we cannot verify offline,
        # and dropping the first question's in-flight answer is the worse failure. Revisit if
        # the Ubuntu gate shows the evaluator reuses a running stack.
        self.get_logger().error(
            "SECOND QUESTION DROPPED: already answering %r; ignoring new text %r. The stack "
            "expects one question per process launch (gotcha 1); keeping the first."
            % (already, msg.data)
        )

    # ------------------------------------------------------------------ perception (H6)
    def _maybe_process_perception(self) -> None:
        """Dispatch one (pano, scan) pair to perception on each NEW pano (H6/SYS-F1, #88).

        Mirrors runner/single.py::_ScriptedPerception.maybe_process: dispatch only when a
        new pano appears (PanoFrame.t changed) and a scan is available. When threaded
        (the default, see ENV_PERCEPTION_SYNC), this only *submits* the frame to the
        AsyncPerceptionWorker — a non-blocking latest-frame-wins handoff — so a slow
        detector forward never stalls this (tick-thread) call. The worker thread runs
        ``pipeline.process()`` and mutates ``self._scene_index`` (its own live index) in
        place, so the controller/heads resolve against the growing map. No-op when
        perception is off (VLA_DETECTOR=none). Never raises — a perception glitch must not
        disturb the drive loop.
        """
        if self._perception is None:
            return
        try:
            pano = self.latest_pano()
            if pano is None:
                return
            if self._last_pano_t is not None and pano.t == self._last_pano_t:
                return
            scan = self.latest_scan()
            if scan is None:
                return
            self._last_pano_t = pano.t
            if self._perception_worker is not None:
                self._perception_worker.submit(pano, scan)
            else:
                self._perception.process(pano, scan)
        except Exception as exc:  # perception must never crash the drive loop
            self.get_logger().error("perception error: %s" % exc)

    # ------------------------------------------------------------------ observability (issue #174)
    def _check_frame_watchdog(self) -> None:
        """Log a loud, rate-limited warning when the camera feed has gone quiet while a
        question is active. Pure observability (no restart/resubscribe attempt) — see
        ros_adapter.frame_watchdog for the split-tree evidence. Never raises (a dead
        adapter must not crash the node).
        """
        try:
            question_active = self._question is not None and (
                self._controller is None or self._controller.state is not State.DONE
            )
            msg = self._frame_watchdog.check(question_active)
            if msg is not None:
                self.get_logger().warning(msg)
        except Exception as exc:  # a watchdog glitch must never disturb the drive loop
            self.get_logger().error("frame watchdog error: %s" % exc)

    # ------------------------------------------------------------------ observability (issue #60)
    def _controller_logger(self, level: str, msg: str) -> None:
        """QuestionController's injected logger callback: transitions INFO, swallowed WARN."""
        if level == "warn":
            self.get_logger().warn(msg)
        else:
            self.get_logger().info(msg)

    def _log_events_tail_on_done(self) -> None:
        """On reaching State.DONE, surface the last EVENTS_TAIL_ON_DONE flight-recorder
        events at INFO, one line, so the controller's events ring (QuestionController.events,
        core.fsm.events.EventLog) is visible in the node log instead of only post-hoc replay.
        """
        if self._done_events_logged or self._controller is None:
            return
        if self._controller.state is not State.DONE:
            return
        self._done_events_logged = True
        tail = self._controller.events.dump()[-EVENTS_TAIL_ON_DONE:]
        rendered = " | ".join(
            "%.1f:%s:%s:%s" % (r.t, r.state, r.event, r.detail) for r in tail
        )
        self.get_logger().info("controller DONE; last %d events: %s" % (len(tail), rendered))

    # ------------------------------------------------------------------ timer / drive
    def _on_tick(self) -> None:
        """5 Hz: feed perception, build the controller on first question, tick it once.

        Never raises (a dead adapter must not crash the node).
        """
        try:
            # Perception runs every tick (independent of the question) so the live index is
            # already populated by the time the controller starts resolving.
            self._maybe_process_perception()
            self._check_frame_watchdog()
            if self._controller is None:
                if self.question() is None:
                    return  # no question yet — nothing to drive
                callables = self._build_controller_callables()
                callables["logger"] = self._controller_logger
                self._controller = QuestionController(**callables)
                self.get_logger().info(
                    "QuestionController constructed; driving (llm=%s, perception=%s)"
                    % (self._llm_configured, self._perception is not None)
                )
            self._controller.tick(self)
            self._log_events_tail_on_done()
        except Exception as exc:  # a dead adapter must never crash the node
            self.get_logger().error("tick error: %s" % exc)

    # ------------------------------------------------------------------ seam wiring (H8)
    def _build_controller_callables(self) -> dict:
        """Wire build_callables with the real LLM ladder + checkpoint seams (H8 / SYS-F5).

        All seams are None/off unless a provider is configured, so the offline path stays
        byte-for-byte deterministic. When configured:

        * ``parse`` runs through the ladder (api -> api2 -> regex floor) with the controller's
          clock + ledger, so the ledger caps + 45 s time-cap + per-call timeouts apply. The
          DARK-NETWORK LOCAL TIER IS DESCOPED (see build_chat_fns: only configured api/api2
          slots are built; the ``local`` slot seam is left for a future llama.cpp/Qwen-VL
          server — architecture §3). When nothing is configured, parse falls back to the
          factory default (regex only).
        * ``verifier`` (CP4), ``anchor_confirmer`` (CP3), ``miss_recoverer`` (CP2),
          ``frontier_selector`` (CP5) come from core.checkpoints, bound to the configured
          chat_fns + the controller's ledger/clock.
        * ``budget_frac`` / ``remaining_s`` / ``forced_assembly`` are late-bound closures
          over the controller's BudgetState (which only exists after question intake).
          Without ``budget_frac`` the H4c provisional-terminal gate is inert (commits
          immediately — safe but the withholding guard never fires), so wiring it is
          load-bearing (H8 verifier note). ``forced_assembly`` (issue #33) is the inverse:
          unwired it defaults to "not forced", so the single-observation route-prefix
          commit floor stays STRICT (n_obs >= 2 required) rather than silently permissive
          — wiring it is what lets the T-90 time-pressure gate ever relax that floor.
        * ``detector=self._detector`` (issue #34) — the SAME GroundingDinoDetector instance
          ``self._perception`` calls every keyframe. Passing it here lets
          ``HeadState.bind`` (core/heads/factory.py) rebuild its ``.prompt`` from the
          question's nouns the moment the plan latches, on both the LLM-configured and the
          offline (regex-only) path below — without this a detector built at boot with an
          empty prompt never grounds anything for the rest of the run.

        build_callables wraps every provider-triggering seam with a hard per-call timeout at
        the injection boundary (SYS-F8), so nothing here can stall the 5 Hz tick past that
        bound.
        """
        # Late-bound budget readers: the controller's BudgetState is created at intake, so
        # these closures read it lazily each tick. Safe before latch (elapsed()/remaining()
        # return 0.0 / full budget). budget_frac = elapsed / QUESTION_BUDGET_S in [0, 1].
        from core.interfaces import QUESTION_BUDGET_S

        def _remaining_s() -> float:
            ctrl = self._controller
            if ctrl is None or ctrl.budget is None:
                return QUESTION_BUDGET_S
            return float(ctrl.budget.remaining())

        def _budget_frac() -> float:
            ctrl = self._controller
            if ctrl is None or ctrl.budget is None:
                return 0.0
            frac = float(ctrl.budget.elapsed()) / QUESTION_BUDGET_S
            return max(0.0, min(1.0, frac))

        def _forced_assembly() -> bool:
            # Issue #33: the T-90 forced-assembly gate straight off BudgetState — feeds
            # the IF head's single-observation route-prefix commit gate. Before latch
            # (controller/budget not yet built) nothing is forced.
            ctrl = self._controller
            if ctrl is None or ctrl.budget is None:
                return False
            return bool(ctrl.budget.forced_assembly)

        if not self._llm_configured:
            # Offline path: regex-only parse, all checkpoint seams off. budget_frac/
            # remaining_s/forced_assembly are still wired (they are pure BudgetState reads,
            # no network) so the H4c / issue #33 commit gates are live even without an LLM.
            return build_callables(
                self._scene_index,
                budget_frac=_budget_frac,
                forced_assembly=_forced_assembly,
                remaining_s=_remaining_s,
                detector=self._detector,
            )

        # A ledger-and-clock-bound closure for the parse ladder. The controller builds its own
        # ledger/clock at intake; read them lazily so parse honours the live budget.
        def _parse(question):
            ctrl = self._controller
            clock = ctrl.budget._clock if (ctrl is not None and ctrl.budget is not None) else self._robotio_clock
            ledger = ctrl.ledger if ctrl is not None else None
            return parse_ladder.parse(
                question, self._chat_fns, clock, ledger, tier_names=self._chat_tier_names
            )

        # Checkpoint seams bound to the configured chat_fns; their ledger/clock resolve lazily
        # via _LedgerProxy since the controller's ledger only exists after intake.
        seams = self._build_checkpoint_seams()

        return build_callables(
            self._scene_index,
            parse=_parse,
            verifier=seams.get("verifier"),
            anchor_confirmer=seams.get("anchor_confirmer"),
            miss_recoverer=seams.get("miss_recoverer"),
            frontier_selector=seams.get("frontier_selector"),
            budget_frac=_budget_frac,
            forced_assembly=_forced_assembly,
            remaining_s=_remaining_s,
            detector=self._detector,
        )

    def _build_checkpoint_seams(self) -> dict:
        """Build the CP2/CP3/CP4/CP5 seam callables bound to the configured chat_fns.

        The checkpoint builders need a ledger + clock. Those live on the controller and are
        created at intake — but _build_controller_callables is only called AFTER the first
        question is latched (the controller is being constructed on this very tick), so the
        ledger is resolved lazily inside each seam via the _LedgerProxy.

        The configured chat_fns are a bare text-ChatFn list (no vision transport wired yet),
        so the VISION checkpoints (CP2/CP3/CP5) are given a chat adapter but WILL fall back to
        deterministic behaviour until a real vision transport + JPEG encoder land (the boot
        assert refuses to ship a network provider with the .npy encoder). CP4 (text) is fully
        wired. All are ledger-gated + timeout-bounded inside guarded_call, and build_callables
        adds a second injection-boundary timeout (SYS-F8).
        """
        from core.checkpoints.anchor_confirm import build_anchor_confirm
        from core.checkpoints.frontier_select import build_frontier_select
        from core.checkpoints.miss_recovery import build_miss_recovery
        from core.checkpoints.verification import build_verifier

        # The primary configured tier feeds the text/vision builders.
        text_chat = self._chat_fns[0]

        def _ledger():
            ctrl = self._controller
            return ctrl.ledger if ctrl is not None else None

        clock = self._robotio_clock
        ledger_proxy = _LedgerProxy(_ledger)

        # CP2 tile-dims wiring (OR-F8, coordinator item 3): bound bbox_hint rejection needs
        # the real tile pixel size. Derived from the pipeline's actual (hfov, vfov) via the
        # same project_tiles formula, so it tracks any tiling recalibration. Falls back to the
        # default 480x640 tile when perception is off.
        tile_w, tile_h = _tile_pixel_dims(self._perception)

        return {
            "verifier": build_verifier(text_chat, ledger_proxy, clock),
            "anchor_confirmer": build_anchor_confirm(
                text_chat, ledger_proxy, clock, {"encode_fn": self._vision_encode_fn}
            ),
            "miss_recoverer": build_miss_recovery(
                text_chat, ledger_proxy, clock,
                {"encode_fn": self._vision_encode_fn, "tile_w": tile_w, "tile_h": tile_h},
            ),
            "frontier_selector": build_frontier_select(
                text_chat, ledger_proxy, clock, {"encode_fn": self._vision_encode_fn}
            ),
        }

    def _assert_encoder_provider_consistency(self) -> None:
        """Fail LOUD at boot if a network provider is paired with the default .npy encoder.

        H14 / OR-F10 landmine: the vision checkpoints (CP2/CP3/CP5) default to
        core.checkpoints._vision.default_encode_fn, which emits raw ``.npy`` bytes a real
        vision API rejects. Gate 4 wired a real JPEG encode_fn (self._vision_encode_fn,
        via core.perception.vision_encode.resolve_encode_fn) into the checkpoint seams
        below whenever Pillow is importable — the common case — so those checkpoints now
        send real image bytes. This assert covers the one remaining inconsistency: Pillow
        absent (``self._vision_encode_fn is None``, the builders' own .npy default then
        applies) WHILE a network provider is configured — shout so that combination can
        never pass a smoke test silently (the wiring-time assert OR-F10 called out).
        """
        if not self._llm_configured:
            return  # no provider -> nothing sends images -> the .npy default is harmless
        # A configured provider whose kind is a real network transport (openai/anthropic).
        network_kinds = {"openai", "anthropic"}
        configured = [
            s for s in self._llm_config.slots()
            if s is not None and s.kind in network_kinds
        ]
        if not configured:
            return  # only stub/local tiers -> no live vision API to reject .npy bytes
        if self._vision_encode_fn is not None:
            return  # a real JPEG encoder is wired -> CP2/CP3/CP5 send real image bytes
        self.get_logger().error(
            "SUBMISSION-BLOCKER: a network LLM/VLM provider is configured (%s) but no real "
            "image encoder is available (Pillow not installed) — the vision checkpoints "
            "fall back to the default raw .npy image encoder, which a real vision API "
            "rejects, so CP2/CP3/CP5 would silently never fire at eval. Install pillow "
            "(H14 / OR-F10)."
            % ", ".join(sorted({s.kind for s in configured}))
        )

    # ------------------------------------------------------------------ debug viz (1 Hz)
    def _on_debug_viz(self) -> None:
        """Republish the instance map + planned-path breadcrumb for RVIZ. debug_viz only.

        Visualization only — reads the scene index and the breadcrumb list, publishes two
        debug topics, mutates no controller/answer state. Never raises (a viz error must not
        disturb the drive loop). Only wired when debug_viz is true, so it does not exist on
        the eval path at all.
        """
        try:
            now = self.get_clock().now().to_msg()
            self._publish_instance_map(now)
            self._publish_planned_path(now)
        except Exception as exc:  # a viz glitch must never crash the node
            self.get_logger().error("debug_viz error: %s" % exc)

    def _publish_instance_map(self, stamp) -> None:
        """One semi-transparent CUBE + TEXT label per tracked InstanceRecord."""
        instances = self._scene_index.all_instances()
        arr = MarkerArray()
        for rec in instances:
            c = (rec.aabb_min + rec.aabb_max) / 2.0
            e = rec.extents
            box = Marker()
            box.header.frame_id = "map"
            box.header.stamp = stamp
            box.ns = "instance_box"
            box.id = int(rec.instance_id)
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose.position.x = float(c[0])
            box.pose.position.y = float(c[1])
            box.pose.position.z = float(c[2])
            box.pose.orientation.w = 1.0
            box.scale.x = max(float(e[0]), 1e-3)
            box.scale.y = max(float(e[1]), 1e-3)
            box.scale.z = max(float(e[2]), 1e-3)
            box.color.r = 0.1
            box.color.g = 0.8
            box.color.b = 1.0
            box.color.a = 0.25  # semi-transparent
            arr.markers.append(box)

            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = stamp
            text.ns = "instance_label"
            text.id = int(rec.instance_id)
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position.x = float(c[0])
            text.pose.position.y = float(c[1])
            text.pose.position.z = float(rec.aabb_max[2]) + 0.2
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25  # text height (m)
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 0.9
            text.text = "%s#%d (%dx)" % (rec.label, rec.instance_id, rec.n_obs)
            arr.markers.append(text)
        self._pub_dbg_instances.publish(arr)

    def _publish_planned_path(self, stamp) -> None:
        """LINE_STRIP breadcrumb of the waypoints we have commanded so far."""
        from geometry_msgs.msg import Point

        line = Marker()
        line.header.frame_id = "map"
        line.header.stamp = stamp
        line.ns = "planned_path"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.05  # line width (m)
        line.color.r = 1.0
        line.color.g = 0.65
        line.color.b = 0.0
        line.color.a = 0.9
        for x, y in self._dbg_path_pts:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.05
            line.points.append(p)
        self._pub_dbg_path.publish(line)

    # ------------------------------------------------------------------ colored map (throttled)
    def _on_colored_cloud(self) -> None:
        """Publish the FULL current colored voxel map as PointCloud2. debug_viz only.

        Throttled (its own slow timer, default every 2 s) so RVIZ stays responsive as the
        map grows — the per-scan ingest happens in _on_scan, this only republishes the
        accumulated cloud. Frame 'map'; RGB rides in a packed float32 'rgb' field
        (ros_adapter.cloud_packing). Visualization only; never raises.
        """
        try:
            if self._colored_map is None or self._colored_map.n_voxels == 0:
                return
            xyz, rgb = self._colored_map.to_arrays()
            data, fields, point_step = pack_colored_cloud(xyz, rgb)

            msg = PointCloud2()
            msg.header.frame_id = "map"
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.height = 1
            msg.width = len(xyz)
            msg.is_bigendian = False
            msg.is_dense = True
            msg.point_step = point_step
            msg.row_step = point_step * len(xyz)
            msg.fields = [
                PointField(name=name, offset=offset, datatype=dtype, count=count)
                for (name, offset, dtype, count) in fields
            ]
            msg.data = data
            self._pub_dbg_cloud.publish(msg)
        except Exception as exc:  # a viz glitch must never crash the node
            self.get_logger().error("colored_cloud error: %s" % exc)

    # ------------------------------------------------------------------ RobotIO: getters
    # Return the latest latched value or None; never block (contract in core/interfaces.py).

    def question(self) -> Question | None:
        with self._lock:
            return self._question

    def latest_pano(self) -> PanoFrame | None:
        with self._lock:
            return self._pano

    def latest_scan(self) -> LidarScan | None:
        with self._lock:
            return self._scan

    def latest_terrain(self, extended: bool = False) -> TerrainPatch | None:
        with self._lock:
            return self._terrain_ext if extended else self._terrain

    def latest_odom(self) -> OdomState | None:
        with self._lock:
            return self._odom

    def clock(self) -> Clock:
        return self._robotio_clock

    # ------------------------------------------------------------------ RobotIO: publishers

    def publish_waypoint(self, wp: WaypointCmd) -> None:
        msg = Pose2D()
        msg.x = float(wp.x)
        msg.y = float(wp.y)
        msg.theta = 0.0  # heading ignored this year (gotcha 4)
        self._pub_waypoint.publish(msg)
        # issue #60: waypoints stream at up to 5 Hz during DRIVE_OUT (IF-F4) — one INFO line
        # per waypoint would spam, so only the running COUNT is logged, throttled to at most
        # once per WAYPOINT_COUNT_LOG_PERIOD_S.
        self._waypoint_count += 1
        now = time.monotonic()
        if now - self._waypoint_count_last_log >= WAYPOINT_COUNT_LOG_PERIOD_S:
            self._waypoint_count_last_log = now
            self.get_logger().info(
                "published %d waypoints since latch" % self._waypoint_count
            )
        # Debug breadcrumb only (guarded): record the goal we just commanded so the 1 Hz
        # debug timer can render it as a LINE_STRIP. No effect when debug_viz is false.
        if self._debug_viz:
            self._dbg_path_pts.append((float(wp.x), float(wp.y)))
            if len(self._dbg_path_pts) > DEBUG_PATH_MAX:
                del self._dbg_path_pts[: len(self._dbg_path_pts) - DEBUG_PATH_MAX]

    def publish_marker(self, box: MarkerBox) -> None:
        msg = Marker()
        msg.header.frame_id = "map"  # gotcha 6: outputs consumed in map frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.ns = box.label or "object"
        msg.id = 0
        msg.type = Marker.CUBE
        msg.action = Marker.ADD
        msg.pose.position.x = float(box.cx)
        msg.pose.position.y = float(box.cy)
        msg.pose.position.z = float(box.cz)
        # Axis-aligned box (MarkerBox carries no orientation); identity quaternion.
        msg.pose.orientation.w = 1.0
        # scale = FULL extents (gotchas 5/6) — MarkerBox.sx/sy/sz are already full extents.
        msg.scale.x = float(box.sx)
        msg.scale.y = float(box.sy)
        msg.scale.z = float(box.sz)
        msg.color.r = 0.0
        msg.color.g = 0.0
        msg.color.b = 1.0
        msg.color.a = 0.5  # matches the dummy's blue translucent box
        self._pub_marker.publish(msg)
        self.get_logger().info(
            "published marker answer: label=%r centroid=(%.2f, %.2f, %.2f)"
            % (box.label, box.cx, box.cy, box.cz)
        )

    def publish_int(self, ans: IntAnswer) -> None:
        msg = Int32()
        msg.data = int(ans.value)
        self._pub_int.publish(msg)
        self.get_logger().info("published int answer: %d" % msg.data)

    def destroy_node(self) -> bool:
        # issue #88: stop the perception worker thread before tearing down the node so it
        # never outlives rclpy shutdown (daemon thread, but a clean join avoids a mid-log
        # get_logger() call racing node teardown).
        if self._perception_worker is not None:
            self._perception_worker.stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
