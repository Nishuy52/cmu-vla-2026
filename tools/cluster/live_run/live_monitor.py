"""Live-run monitor: one rclpy node tracking the whole question run.

Counts sensor rates (incl. the camera raw-vs-compressed question), logs every
waypoint, and exits 0 the moment an answer lands on /numerical_response or
/selected_object_marker. Exits 3 on deadline. Unbuffered progress lines.
"""
import sys
import time

import rclpy
from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker

DEADLINE_S = float(sys.argv[1]) if len(sys.argv) > 1 else 780.0

rclpy.init()
node = Node("live_monitor")
qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
qos_rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

counts = {"scan": 0, "cam_raw": 0, "cam_comp": 0, "odom": 0}
waypoints = []
answer = {}
pose = {}


def count(key):
    def cb(_m):
        counts[key] += 1
    return cb


def on_wp(m):
    waypoints.append((time.time(), m.x, m.y))
    print(f"WAYPOINT #{len(waypoints)}: ({m.x:.2f}, {m.y:.2f})", flush=True)


def on_answer(m):
    answer["numerical"] = m.data
    print(f"ANSWER-NUMERICAL={m.data}", flush=True)


def on_marker(m):
    answer["marker"] = (m.pose.position.x, m.pose.position.y)
    print(f"ANSWER-MARKER=({m.pose.position.x:.2f},{m.pose.position.y:.2f})", flush=True)


def on_odom(m):
    counts["odom"] += 1
    pose["xy"] = (m.pose.pose.position.x, m.pose.pose.position.y)


node.create_subscription(PointCloud2, "/registered_scan", count("scan"), qos)
node.create_subscription(Image, "/camera/image", count("cam_raw"), qos)
node.create_subscription(CompressedImage, "/camera/image/compressed", count("cam_comp"), qos)
node.create_subscription(Odometry, "/state_estimation", on_odom, qos)
node.create_subscription(Pose2D, "/way_point_with_heading", on_wp, qos_rel)
node.create_subscription(Int32, "/numerical_response", on_answer, qos_rel)
node.create_subscription(Marker, "/selected_object_marker", on_marker, qos_rel)

start = time.time()
last_report = start
while time.time() - start < DEADLINE_S:
    rclpy.spin_once(node, timeout_sec=0.2)
    if answer:
        break
    now = time.time()
    if now - last_report >= 20.0:
        el = now - start
        print(
            f"t={el:.0f}s scan={counts['scan']/el:.2f}Hz cam_raw={counts['cam_raw']/el:.2f}Hz "
            f"cam_comp={counts['cam_comp']/el:.2f}Hz odom={counts['odom']/el:.0f}Hz "
            f"waypoints={len(waypoints)} pose={pose.get('xy')}",
            flush=True,
        )
        last_report = now

el = time.time() - start
print(f"FINAL t={el:.0f}s waypoints={len(waypoints)} counts={counts} answer={answer}", flush=True)
node.destroy_node()
rclpy.shutdown()
if answer:
    print("LIVE-RUN-SUCCESS", flush=True)
    sys.exit(0)
print("LIVE-RUN-DEADLINE", flush=True)
sys.exit(3)
