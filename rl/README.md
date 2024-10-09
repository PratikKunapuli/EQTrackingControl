# Reinforcement Learning Implementations

## Contents:
- `eval_policy.py`: Code to evaluate a trained model on potentially *different trajectories* than the model was trained on, and generate animations. 
- `eval_vis.py`: Code to evaluate a trained model and save the trajectories to later visualze with ROS. 
- `models.py`: Implementation of the neural networks with configurable layers, nodes, and activations written in Flax. 
- `train_policy.py`: Training code for all environments using a custom PPO implemented in Jax. 

## Training (`train_policy.py`)
In order to train a policy, we can simply run `python train_policy.py` with some specified arguments. Running with `-h` or `--help` will list all available arguments. 

Examples:
```
python train_policy.py --seed 2024 --equivariant 1 --debug --exp-name PPO_particle_equivariant --env-name particle_random_walk_velocity
python train_policy.py --no-debug --exp-name PPO_particle_no_equivariant_seed_0 --env-name particle_random_walk_velocity
python train_policy.py --exp-name astrobee_baseline --env_name astrobee_random_walk
```

### Saved Data
By default, a new directory will be created at `./checkpoints/{exp-name}` and this is where the resulting data generated during training will be saved. This includes the configuration dictionary made from the default parameters and CLI args (`config.txt`), training curves (`training_data.npz`), the final model weights (`/model_final/`) and some figures (`episode_returns_shaded.png`, and `terminal_timesteps_summary.png`). By default, if an experiment name already exists in the `./checkpoints/` directory, a numerical index will be appended and incremented to the experiment name. 

## Evaluations (`eval_policy.py`)
After training a policy, we can evaluate the weights in a new script to simply rollout the environment. The configuration during train time should be loaded automatically, but some parameters can be overwritten such as the environment name to zero-shot transfer to a new trajectory setting. 

Example:
```
python eval_policy.py --seed 2024 --load-path ./checkpoints/PPO/model_final/ --equivariant 1 --env-name particle_random_lissajous
```

This script will rollout `{num-envs}` environments and then make a 3D plot (if `--make-animation` is set) representing the rollout of the particles and their respective goals. Additionally, reward curves will be generated for each particle. These figures will be saved in the same parent directly of the weights as `particle_position.png` and `rewards.png`. 

## Wrappers (`wrappers.py`)
Included in this repository are some useful wrappers. One such wrapper is the `LogWrapper()` used in `train_policy.py`. Another wrapper is incuded to convert from [Gymnax](https://github.com/RobertTLange/gymnax/tree/main)-style environments to [Gymnasium](https://github.com/RobertTLange/gymnax/tree/main)-style environements, `GymnaxToGymWrapper()`. This allows the use of numpy-based inputs and outputs so other RL implementations can be evaluated. 

Example:
```
env = PointParticlePosition()
env_gym = GymnaxToGymWrapper(env)

obs, info = env_gym.reset()
print(obs, info)

obs, reward, done, info = env_gym.step(env_gym.action_space.sample())
print(obs, reward, done, info)

```