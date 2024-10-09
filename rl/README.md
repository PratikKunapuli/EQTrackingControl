# Reinforcement Learning Implementations

## Contents:
- `eval_policy.py`: Code to evaluate a trained model on potentially *different trajectories* than the model was trained on, and generate animations. 
- `eval_vis.py`: Code to evaluate a trained model and save the trajectories to later visualze with ROS. 
- `models.py`: Implementation of the neural networks with configurable layers, nodes, and activations written in Flax. 
- `train_policy.py`: Training code for all environments using a custom PPO implemented in Jax. 