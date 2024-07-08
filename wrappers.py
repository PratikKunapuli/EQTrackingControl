import jax
from flax import struct
import chex

from functools import partial
from typing import Sequence, NamedTuple, Any, Tuple, Union, Optional
import copy

import gymnasium as gym
from gymnasium import core

from gymnax.environments import spaces


from jax_envs import EnvState

@struct.dataclass
class LogEnvState:
    env_state: EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)

class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""
    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[LogEnvState, chex.Array]:
        env_state, obs = self._env.reset(key)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return state, obs

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: LogEnvState, action: Union[int, float]) -> Tuple[LogEnvState, chex.Array, float, bool, dict]:
        env_state, obs, reward, done, info = self._env.step(key, state.env_state, action)
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return state, obs, reward, done, info
    

class GymnaxToGymWrapper(gym.Env[core.ObsType, core.ActType]):
    """Wrapper to convert Gymnax environment to Gym environment."""
    def __init__(self, env, seed: Optional[int]=None):
        super().__init__()
        self._env = copy.deepcopy(env)
        self.metadata.update( {
            "name": env.name,
            "render.modes": [],
        })

        self.rng : chex.PRNGKey = jax.random.PRNGKey(0)
        self._seed(seed)

    def _seed(self, seed: Optional[int]) -> None:
        self.rng = jax.random.PRNGKey(seed or 0)

    def reset(self, *, seed: Optional[int]=None) -> core.ObsType:
        if seed is not None:
            self._seed(seed)
        
        self.rng, reset_key = jax.random.split(self.rng)
        state, obs = self._env.reset(reset_key)
        return obs
    
    def step(self, action: core.ActType) -> Tuple[core.ObsType, float, bool, dict]:
        self.rng, step_key = jax.random.split(self.rng)
        state, obs, reward, done, info = self._env.step(step_key, action)
        return obs, reward, done, info

    def render(self, mode: str="human"):
        return None

    def convert_space(self, space: spaces.Space) -> gym.Space:
        if isinstance(space, spaces.Discrete):
            return gym.spaces.Discrete(space.n)
        elif isinstance(space, spaces.Box):
            return gym.spaces.Box(
                low=space.low,
                high=space.high,
                dtype=space.dtype,
            )
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")

    @property
    def observation_space(self) -> gym.Space:
        return self.convert_space(self._env.observation_space)

    @property
    def action_space(self) -> gym.Space:
        return self.convert_space(self._env.action_space)