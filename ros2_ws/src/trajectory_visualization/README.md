To run the ROS2-based visualization (`<robot>` is either `astrobee`, `quadrotor`, or `particle`):

```cd ros2_ws```

Ensure Colcon is installed, if needed install it via:

```sudo apt install python3-colcon-common-extensions```

```colcon build --symlink-install```

```source install/setup.bash```

In one terminal, run:
```rviz2 -d src/trajectory_visualization/<robot>.rviz```

In another terminal, run: 
```ros2 run trajectory_visualization eval_visualize --args <robot>```
