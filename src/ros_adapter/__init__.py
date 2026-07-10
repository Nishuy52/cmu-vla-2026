"""ros_adapter — the single ROS-dependent package (rclpy adapter over the core RobotIO seam).

Importing this package's ``adapter_node`` requires rclpy and the ROS message packages, which
are ONLY present on the Ubuntu/Docker side. The core package never imports this package
(one-way dependency: ros_adapter -> core), so the core test suite stays ROS-free.
"""
