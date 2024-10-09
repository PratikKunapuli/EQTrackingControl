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

## Environments (`particle_envs.py`)
Environments here inheret some common utilities from a base environemnt `PointParticleBase` in `base_envs.py`. 

Optional Arguments for any `PointParticleBase` environment:
```
ref_pos: [jnp.array] fix the reference position
equivariant: [bool] whether to use the equivartiant representation or not. Up to children classes to implement this behavior in `_get_obs()` or similar. 
state_cov_scalar: [float] scalar multiplier of the covariance matrix (identity) for sampling the initial state (used for position and velocity)
ref_cov_scalar: [float] scalar multiplier of the covariance matrix (identity) for sampling the reference 
dt: [float] simulation dt. This affectes the underlying Euler integration performed by children classes. 
max_time: [float] maximum time to interact in the environment. 
terminate_on_error: [bool] whether to include error-based termination conditions in the termination checking or only use max time. 
```

1. PointParticlePosition
`env = PointParticlePosition(equivariant=False)`
This environment represents a single point particle where the actions directly affect the velocity of the particle. The state of the particle is represented as the position and velocity in 3 dimension (x,y,z). Observation is the state of the particle and a reference state (position and velocity), but in this environment the reference velocity is set to be 0. This represents a statically fixed goal. In the equivariant mode, the Gallilean symmetry is employed and the observation is reduced to the error between the reference and state in both position and velocity (6-dim). 

2. PointParticleConstantVelocity
`env = PointParticleConstantVelocity(equivariant=False)`
This environment represents a single point particle where the actions directly affect the velocity of the particle. The state of the particle is represented as the position and velocity in 3 dimension (x,y,z). Observation is the state of the particle and a reference state (position and velocity), as well as the reference acceleration. The reference acceleration is set to be 0 for this environment, resulting in constant velocity of the reference after the reference velocity is randomized in the reset function. Non-equivariant representation is (15-dim) and the equivariant representation is (9-dim) for position error, velocity error, and reference acceleration. 


