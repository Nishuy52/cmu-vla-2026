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

_ADAPTER = (
    pathlib.Path(__file__).resolve().parents[2]
    / "ros_adapter"
    / "adapter_node.py"
)


def test_adapter_source_compiles():
    """Parse+compile the adapter source so Windows CI-less dev catches syntax errors."""
    assert _ADAPTER.is_file(), f"adapter source missing: {_ADAPTER}"
    source = _ADAPTER.read_text(encoding="utf-8")
    # compile() runs the full parser (SyntaxError on bad syntax) without importing rclpy.
    compile(source, str(_ADAPTER), "exec")


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
