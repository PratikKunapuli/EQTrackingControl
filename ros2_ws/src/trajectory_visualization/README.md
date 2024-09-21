To run the ROS2-based visualization (`<robot>` is either `astrobee`, `quadrotor`, or `particle`):

```cd ros2_ws```

```colcon build --symlink-install```

```source install/setup.bash```

```rviz2 -d src/trajectory_visualization/<robot>.rviz```

```ros2 run trajectory_visualization eval_visualize --args <robot>```
