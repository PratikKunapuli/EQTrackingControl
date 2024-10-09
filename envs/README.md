# Environment Implementations

## Contents:
- `base_envs.py`: Data structures for all environments, as well as the base implementations for all robots with individual parent classes per robot containing initialization, reward computations, and termination computations. 
- `particle_envs.py`: dynamics of the `Particle` system implemented in the `step()` method, resetting logic implemented in `reset()`, and observation implementations in `_get_obs()`
- `astrobee_envs.py`: dynamics of the `Astrobee` system implemented in the `step()` method, resetting logic implemented in `reset()`, and observation implementations in `_get_obs()`
- `quadrotor_envs.py`: dynamics of the `Quadrotor` system implemented in the `step()` method, resetting logic implemented in `reset()`, and observation implementations in `_get_obs()`

- `wrappers.py`: Usefull wrappers for the environments for logging, converting to `Gymnax` or `Gymnasium` environments etc. 