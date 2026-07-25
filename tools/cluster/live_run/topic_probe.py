"""Rate-probe contract topics without the ros2 CLI (its daemon hangs on this cluster).

Subscribes raw (no message-type imports needed) and counts arrivals for WINDOW s.
"""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

WINDOW = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0

TOPICS = {
    "/registered_scan": "sensor_msgs/msg/PointCloud2",
    "/state_estimation": "nav_msgs/msg/Odometry",
    "/camera/image": "sensor_msgs/msg/Image",
    "/terrain_map": "sensor_msgs/msg/PointCloud2",
    "/sensor_scan": "sensor_msgs/msg/PointCloud2",
}

rclpy.init()
node = Node("cluster_probe")
counts = {t: 0 for t in TOPICS}


def make_cb(topic):
    def cb(_msg):
        counts[topic] += 1
    return cb


qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
from rosidl_runtime_py.utilities import get_message  # noqa: E402

for topic, type_str in TOPICS.items():
    try:
        node.create_subscription(get_message(type_str), topic, make_cb(topic), qos)
    except Exception as exc:  # noqa: BLE001
        print(f"{topic}: SUBSCRIBE-FAIL {exc}")

# discovery report after a short spin
end = time.time() + 3.0
while time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.2)
names = node.get_topic_names_and_types()
print(f"GRAPH: {len(names)} topics visible")
for n, t in sorted(names)[:40]:
    print(f"  {n}  {t[0] if t else ''}")

start = time.time()
end = start + WINDOW
while time.time() < end:
    rclpy.spin_once(node, timeout_sec=0.2)
elapsed = time.time() - start
for topic in TOPICS:
    rate = counts[topic] / elapsed
    print(f"RATE {topic}: {rate:.2f} Hz ({counts[topic]} msgs)")
node.destroy_node()
rclpy.shutdown()
print("PROBE-DONE")
