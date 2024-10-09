<h1 align="center">
  <b>EQ Tracking Control: Symmetry-Aware Trajectory Tracking Controllers with RL</b><br>
</h1>

<p align="center">
 <img src="https://img.shields.io/badge/python-3.10-blue" />
</p>

This repository contains code for the environment and RL training implementations of the paper, [Leveraging Symmetry to Accelerate Learning of Trajectory Tracking Controllers for Free-Flying Robotic Systems](https://arxiv.org/abs/2409.11238). All environments are written in Jax, with wrappers available for the standard Gymnasium interface. A custom PPO implementation, also written in Jax, is provided. Training takes approximately 4 mins for 10M env steps, running on an Nvidia RTX A5000. 

## Installation
This package was run on Python 3.10. Other python versions may work, but it has not been tested. Jax can be installed with backend support for GPU/TPU by following their installation instructions. 

```
git clone git@github.com:PratikKunapuli/PointParticleEQ.git
cd PointParticleEQ
pip install -r requirements.txt
```

If you would like to use the ROS2 visualizations, ROS2 must be installed seperately. Then, the following line can be used to install the required ROS packages.

```
pip install -r requirements_ros2.txt
```

Finally, install the package locally for the relative imports to work
```
pip install -e .
```

## Example Results
<img src="./figures/example_results.png" alt="main_results" width="800"/>

<img src="./figures/example_rollouts.png" alt="main_rollouts" width="800"/>

## Checkpoints
Pretrained models are provided within the `checkpoints` directory, for all systems. These models match the performance reported in the paper, and the individual hyperparameters used for each experiment are contained wihtin the `config.txt` file of the individual experiments. 
