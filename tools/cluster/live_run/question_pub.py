"""Publish the challenge question at 1 Hz (mirrors the upstream question runner)."""
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

question = sys.argv[1] if len(sys.argv) > 1 else "How many chairs are in the room?"

rclpy.init()
node = Node("question_pub")
qos = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)
pub = node.create_publisher(String, "/challenge_question", qos)
msg = String()
msg.data = question


def tick():
    pub.publish(msg)


timer = node.create_timer(1.0, tick)
rclpy.spin(node)
