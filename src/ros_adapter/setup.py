"""ament_python setup for the vla_ai_module adapter package.

Mirrors how the dummy is launched (upstream ai_module/src/dummy_vlm/launch/dummy_vlm.launch:
``<node pkg="dummy_vlm" exec="dummyVLM" .../>``) with the ament_python equivalent: a console
entry point ``adapter_node`` so the launch file can do ``executable='adapter_node'``, and the
launch dir installed to share/ so ``ros2 launch vla_ai_module ai_module.launch.py`` resolves —
the same ``ros2 launch <pkg> <file>`` shape the dummy uses.

This package's modules are ``ros_adapter`` and ``core`` living under
``src/`` (this dir's parent). The submission Docker image puts ``src/`` on PYTHONPATH
(docker/ai_module_fork/docker/Dockerfile — the fork-shaped packaging; the old
docker/ai_module/Dockerfile is deprecated) so both ``import core`` and ``import
ros_adapter`` resolve at runtime. ``colcon build`` from the workspace with this package
produces the ``adapter_node`` console script; colcon does not see ``core`` at build
time and does not need to (the entry point resolves it lazily on PYTHONPATH at run
time) — confirmed working on Ubuntu (docs/ubuntu_setup.md §7a).
"""
from setuptools import setup

package_name = "vla_ai_module"

setup(
    name=package_name,
    version="0.0.1",
    # The adapter node lives in the ros_adapter package (src/ros_adapter/). core is a
    # sibling package installed separately (pip install of src/, or PYTHONPATH).
    packages=["ros_adapter"],
    package_dir={"ros_adapter": "."},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/ai_module.launch.py",
                "launch/ai_module_debug.launch.py",
            ],
        ),
        # Debug RVIZ config (loaded by ai_module_debug.launch.py). Debug-only asset.
        ("share/" + package_name + "/rviz", ["rviz/ai_module_debug.rviz"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="VLA team",
    maintainer_email="jyuc1109uni@gmail.com",
    description="rclpy adapter wrapping the pure-Python core QuestionController.",
    license="BSD",
    entry_points={
        "console_scripts": [
            # exec name 'adapter_node' == the dummy's 'dummyVLM' role (launch exec target).
            "adapter_node = ros_adapter.adapter_node:main",
        ],
    },
)
