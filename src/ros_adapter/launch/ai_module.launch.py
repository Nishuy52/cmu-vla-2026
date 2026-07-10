"""Launch the VLA ai_module adapter node.

ament_python equivalent of the dummy's XML launch (upstream ai_module/src/dummy_vlm/launch/
dummy_vlm.launch): the dummy runs one node ``pkg="dummy_vlm" exec="dummyVLM"``; we run one
node ``package='vla_ai_module' executable='adapter_node'``. Invoked exactly the same way the
dummy is (docs/upstream_notes.md §4) — ``ros2 launch vla_ai_module ai_module.launch.py`` — so
it slots straight into the compose stack in place of ``ros2 launch dummy_vlm dummy_vlm.launch``.

The dummy passes file-path params (waypoint/object-list files); this node has no such params —
it drives the pure-Python core and reads only live topics. RMW_IMPLEMENTATION is set in the
Docker image, not here (docker/ai_module/Dockerfile), so both containers agree on CycloneDDS
(gotcha 13). confirm on Ubuntu.
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="vla_ai_module",
                executable="adapter_node",
                name="vla_ai_module",
                output="screen",
            ),
        ]
    )
