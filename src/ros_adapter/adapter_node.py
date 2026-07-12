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
dropped frame never blocks; /challenge_question uses RELIABLE + TRANSIENT_LOCAL so the single
question (republished at 1 Hz, gotcha 3) is not missed on a late join. Publishers are reliable
depth 5, matching the dummy's default profile.
"""
from __future__ import annotations

import threading

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
from core.fsm.controller import QuestionController
from core.heads import build_callables
from core.perception.colored_map import ColoredVoxelMap
from core.perception.scene_index import BasicSceneIndex
from ros_adapter.cloud_packing import pack_colored_cloud
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

TICK_HZ = 5.0  # QuestionController.tick() cadence (core/fsm/controller.py docstring)
DEBUG_VIZ_HZ = 1.0  # instance-map republish cadence when debug_viz is on
DEBUG_PATH_MAX = 500  # cap on retained waypoint breadcrumb points (bounded memory)
DEBUG_CLOUD_PERIOD_S = 2.0  # publish the FULL colored map at most this often (RVIZ stays responsive)


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
        self._clock = _NodeClock(self)

        # ---- Subscriptions (six allowed system-output topics) -----------------
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

        # ---- Controller (built lazily once the question is known) -------------
        # The heads resolve against the live SceneIndex. Phase 2 wires the real perception
        # map here; until then an empty index means the FSM floor still emits a legal answer.
        # confirm on Ubuntu: swap BasicSceneIndex([]) for the live perception scene index
        # once core/perception is fused into this node (architecture Phase 2/3).
        self._scene_index = BasicSceneIndex([])
        self._controller: QuestionController | None = None

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
        # Attach the latest odom so PanoFrame.odom is populated (mirrors BagSource.frames()).
        with self._lock:
            odom = self._odom
        pano = image_to_pano(msg, self._now_ns(), odom=odom)
        with self._lock:
            self._pano = pano

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
                return
            self._question = string_to_question(msg, self._now_ns())
        self.get_logger().info("question latched: %r" % self._question.text)

    # ------------------------------------------------------------------ timer / drive
    def _on_tick(self) -> None:
        """5 Hz: build the controller on first question, then tick it once. Never raises."""
        try:
            if self._controller is None:
                if self.question() is None:
                    return  # no question yet — nothing to drive
                callables = build_callables(self._scene_index)
                self._controller = QuestionController(**callables)
                self.get_logger().info("QuestionController constructed; driving")
            self._controller.tick(self)
        except Exception as exc:  # a dead adapter must never crash the node
            self.get_logger().error("tick error: %s" % exc)

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
        return self._clock

    # ------------------------------------------------------------------ RobotIO: publishers

    def publish_waypoint(self, wp: WaypointCmd) -> None:
        msg = Pose2D()
        msg.x = float(wp.x)
        msg.y = float(wp.y)
        msg.theta = 0.0  # heading ignored this year (gotcha 4)
        self._pub_waypoint.publish(msg)
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

    def publish_int(self, ans: IntAnswer) -> None:
        msg = Int32()
        msg.data = int(ans.value)
        self._pub_int.publish(msg)


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
