"""Debug launch: our adapter node with debug_viz ON + an RVIZ2 window loading our view.

This is the DEBUG counterpart of ai_module.launch.py. It is for bring-up / manual sim
inspection only — never for evaluation. It launches:

  1. the adapter node with ``debug_viz:=true`` (so it republishes /ai_module/instance_map and
     /ai_module/planned_path), and
  2. an ``rviz2`` node loading ``rviz/ai_module_debug.rviz`` — OUR additive debug view of what
     the ai_module believes and publishes.

The base autonomy stack already opens its OWN system RVIZ (visualization_tools) when
system_simulation.sh runs; this one is additive and independent. The sim itself is Unity
(per-scene binary + ROS-TCP bridge), not Gazebo — RVIZ here only visualizes the ROS graph.

Invoke: ``ros2 launch vla_ai_module ai_module_debug.launch.py``
The eval-clean path stays ``ros2 launch vla_ai_module ai_module.launch.py`` (no rviz, debug off).

confirm on Ubuntu: the rviz config resolves via the package share dir; if the package is run
straight from source (not colcon-installed) point rviz at src/ros_adapter/rviz/ai_module_debug.rviz.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    rviz_config = os.path.join(
        get_package_share_directory("vla_ai_module"),
        "rviz",
        "ai_module_debug.rviz",
    )
    return LaunchDescription(
        [
            Node(
                package="vla_ai_module",
                executable="adapter_node",
                name="vla_ai_module",
                output="screen",
                parameters=[{"debug_viz": True}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="ai_module_debug_rviz",
                output="screen",
                arguments=["-d", rviz_config],
            ),
        ]
    )
