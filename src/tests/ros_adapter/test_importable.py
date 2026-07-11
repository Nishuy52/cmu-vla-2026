"""Guard test for the ROS adapter node.

The adapter (src/ros_adapter/adapter_node.py) imports rclpy and ROS message packages at
module top, so it CANNOT be imported on the Windows dev box (no ROS). This single test still
catches syntax / indentation / obvious-name errors there by ``compile()``-ing the source —
no import, no rclpy needed. Where rclpy IS available (Ubuntu / the ai_module container) it
additionally imports the module for real via ``pytest.importorskip``.

This is the ONLY executable test for the adapter; all behaviour is validated on Ubuntu.
"""
from __future__ import annotations

import pathlib

import pytest

_ROS_ADAPTER = pathlib.Path(__file__).resolve().parents[2] / "ros_adapter"
_ADAPTER = _ROS_ADAPTER / "adapter_node.py"
_LAUNCH_EVAL = _ROS_ADAPTER / "launch" / "ai_module.launch.py"
_LAUNCH_DEBUG = _ROS_ADAPTER / "launch" / "ai_module_debug.launch.py"
_RVIZ_DEBUG = _ROS_ADAPTER / "rviz" / "ai_module_debug.rviz"


def test_adapter_source_compiles():
    """Parse+compile the adapter source so Windows CI-less dev catches syntax errors."""
    assert _ADAPTER.is_file(), f"adapter source missing: {_ADAPTER}"
    source = _ADAPTER.read_text(encoding="utf-8")
    # compile() runs the full parser (SyntaxError on bad syntax) without importing rclpy.
    compile(source, str(_ADAPTER), "exec")


@pytest.mark.parametrize("launch_file", [_LAUNCH_EVAL, _LAUNCH_DEBUG])
def test_launch_source_compiles(launch_file):
    """Parse+compile both launch files (they import launch/launch_ros only on Ubuntu).

    Like the adapter, these import ROS-side launch packages at module top, so they cannot be
    imported on Windows — but compile() still catches syntax/indentation errors here.
    """
    assert launch_file.is_file(), f"launch file missing: {launch_file}"
    source = launch_file.read_text(encoding="utf-8")
    compile(source, str(launch_file), "exec")


def test_debug_rviz_config_parses():
    """The debug RVIZ config must be valid YAML with the map fixed frame and our topics.

    pyyaml ships transitively via the rosbags dependency (no new hard dep). If it is ever
    absent the test skips rather than failing — the parse is a Windows-side convenience guard;
    the config is confirmed for real on Ubuntu by loading it in rviz2 (sim_verification 2.8).
    """
    yaml = pytest.importorskip("yaml", reason="pyyaml (via rosbags) not present")
    assert _RVIZ_DEBUG.is_file(), f"rviz config missing: {_RVIZ_DEBUG}"
    doc = yaml.safe_load(_RVIZ_DEBUG.read_text(encoding="utf-8"))
    assert isinstance(doc, dict), "rviz config is not a YAML mapping"

    vm = doc["Visualization Manager"]
    assert vm["Global Options"]["Fixed Frame"] == "map"

    # Every topic our debug view must show must appear somewhere in the config text.
    text = _RVIZ_DEBUG.read_text(encoding="utf-8")
    for topic in (
        "/registered_scan",
        "/terrain_map",
        "/terrain_map_ext",
        "/state_estimation",
        "/camera/image",
        "/selected_object_marker",
        "/way_point",  # PointStamped republish of our Pose2D (see config comment)
        "/path",
        "/ai_module/instance_map",
        "/ai_module/planned_path",
    ):
        assert topic in text, f"debug rviz config missing display for {topic}"


def test_adapter_imports_when_rclpy_present():
    """On a ROS box, import the module for real (skips on Windows: no rclpy)."""
    pytest.importorskip("rclpy", reason="rclpy only present on Ubuntu/ROS")
    import sys

    src_root = str(_ADAPTER.parents[1])  # .../src
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    import ros_adapter.adapter_node as node  # noqa: F401

    assert hasattr(node, "AdapterNode")
    assert hasattr(node, "main")
