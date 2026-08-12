#!/usr/bin/env python3
"""Bridges the t2_cruza_vtol Gazebo camera (image + camera_info) into ROS 2
on the generic /sensor/imager/* topics consumed by apriltag_detector_36h11_sim.launch.yml.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

GZ_IMAGE_TOPIC = '/world/apriltag/model/t2_cruza_vtol_0/link/camera_link/sensor/imager/image'


def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory('apriltag_ros'),
        'cfg',
        't2_vtol_gz_bridge.yaml'
    )

    camera_info_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='t2_vtol_camera_info_bridge',
        arguments=[
            '--ros-args',
            '-p', f'config_file:={config_path}',
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    camera_image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        name='t2_vtol_camera_image_bridge',
        arguments=[GZ_IMAGE_TOPIC],
        remappings=[
            (GZ_IMAGE_TOPIC, '/sensor/imager/image')
        ],
        parameters=[{'use_sim_time': True}],
        output='screen'
    )

    return LaunchDescription([
        camera_info_bridge,
        camera_image_bridge,
    ])
