#!/usr/bin/env python3
"""Single entry point for the T2 Cruza VTOL drone in simulation: brings up
the Gazebo camera bridge and the 36h11 AprilTag detector together.

    ros2 launch apriltag_ros t2_vtol_apriltag_sim.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource


def generate_launch_description():
    pkg_share = get_package_share_directory('apriltag_ros')

    gz_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 't2_vtol_gz_bridge.launch.py')
        )
    )

    apriltag_detector = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(pkg_share, 'launch', 'apriltag_detector_36h11_sim.launch.yml')
        )
    )

    return LaunchDescription([
        gz_bridge,
        apriltag_detector,
    ])
