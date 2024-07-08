from flax import struct
import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import lax
from jax import jit

from typing import Any, Dict, Optional, Tuple, Union
from functools import partial

from gymnax.environments import spaces

@struct.dataclass
class EnvState:
    time: float

@struct.dataclass
class PointState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray

@struct.dataclass
class PointVelocityState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_acc: jnp.ndarray

class PointParticleBase:
    def __init__ (self, ref_pos=None, equivariant=False, state_cov_scalar=0.5, ref_cov_scalar=3.0, dt=0.05, max_time=100.0, terminate_on_error=True, **kwargs):
        self.state_mean = jnp.array([0., 0., 0.])
        self.state_cov = jnp.eye(3) * state_cov_scalar
        self.ref_mean = jnp.array([0., 0., 0.])
        self.ref_cov = jnp.eye(3) * ref_cov_scalar
        self.predefined_ref_pos = ref_pos

        self.max_time = max_time
        self.dt = dt
        self.equivariant = equivariant
        self.terminate_on_error = terminate_on_error
        self.other_args = kwargs

    def _sample_random_ref_pos(self, key):
        """
        Helper function to sample a random reference position from a multivariate normal distribution
        """
        return jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov)
    
    def _get_predefined_ref_pos(self, key):
        """
        Helper function to get the predefined reference position
        """
        return jnp.array(self.predefined_ref_pos) if self.predefined_ref_pos is not None else jnp.zeros_like(self.ref_mean)
    
    def _maybe_reset(self, key, env_state, done):
        '''
        Reset helper to work with the jax conditional flow. 
        If the done flag is True, run self._reset with the key. 
        If done is False, return the env_state as is.
        '''
        return lax.select(done, self._reset(key), env_state)
    
    def _is_terminal(self, env_state):
        """
        Helper function to check if the environment state is terminal.
        If self.terminate_on_error is True, then the environment is terminal if the particle is outside the world bounds or exceeds the velocity error.
        """
            
        # if any part of the state is outside the bounds of +- 5, then the episode is done
        # or if the time is greater than the max time
        outside_world_bounds = jnp.any(jnp.abs(env_state.ref_pos - env_state.pos) > 10.)
        exceeded_error_velocity = jnp.any(jnp.abs(env_state.ref_vel - env_state.vel) > 10.)
        time_exceeded = env_state.time > self.max_time

        world_cond = jnp.logical_or(outside_world_bounds, exceeded_error_velocity)
        world_cond = lax.select(self.terminate_on_error, world_cond, False)

        return jnp.logical_or(world_cond, time_exceeded)


    @property
    def num_actions(self) -> int:
        return 3
    
    def action_space(self) -> spaces.Box:
        low = jnp.array([-1., -1., -1.])
        high = jnp.array([1., 1., 1.])
        return spaces.Box(low, high, (3,), jnp.float32)