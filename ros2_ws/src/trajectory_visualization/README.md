To run the ROS2-based visualization:

```cd ros2_ws```

```colcon build --symlink-install```

```source install/setup.bash```

```rviz2 src/trajectory_visualization/config.rviz```

```ros2 run trajectory_visualization visualize```
