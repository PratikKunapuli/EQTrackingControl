import jax.numpy as jnp
from jax import random as jrandom
from jax import lax
from jax import jit
from flax import struct
import jax


from typing import Any, Dict, Optional, Tuple, Union
from functools import partial

from gymnax.environments import spaces

@struct.dataclass
class PointState():
    pos: jnp.ndarray
    vel: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    time: float

class PointParticlePosition:
    def __init__(self, ref_pos=None, equivariant=False):
        self.dt = 0.05 # prev 0.01

        self.state_mean = jnp.array([0., 0., 0.])
        self.state_cov = jnp.eye(3) * 0.5
        self.ref_mean = jnp.array([0., 0., 0.])
        self.ref_cov = jnp.eye(3) * 3.0

        self.predefined_ref_pos = ref_pos
        self.max_time = 100.0 # prev 2.0
        self.equivariant = equivariant

        print("Creating PointParticlePosition environment with Equivaraint: ", self.equivariant)

    def step(self, key, env_state, action):
        '''
        Step function for the environment. Arguments are defined as follows:

        key: random_key for the environmen (need )
        env_state: current state of the environment (both true state of the particle and the reference state) [pos, vel, ref_pos, ref_vel]
        action: action taken by the agent (velocity of the particle)

        returns: tuple: (env_state, observation, reward, done, info)
        '''

        # clip action
        action = jnp.clip(action, -1., 1.)

        state = env_state

        # update particle position
        vel = state.vel + action * self.dt
        pos = state.pos + vel * self.dt

        # update reference position
        ref_vel = state.ref_vel # Hardcoded - no moving reference
        ref_pos = state.ref_pos + ref_vel * self.dt

        # update time
        time = state.time + self.dt

        done = self._is_terminal(env_state)

        env_state = PointState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, time=time)

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state, action), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}
    
    def _is_terminal(self, env_state):

        # if any part of the state is outside the bounds of +- 5, then the episode is done
        # or if the time is greater than the max time
        outside_world_bounds = jnp.any(jnp.abs(env_state.ref_pos - env_state.pos) > 10.)
        exceeded_error_velocity = jnp.any(jnp.abs(env_state.ref_vel - env_state.vel) > 10.)
        time_exceeded = env_state.time > self.max_time

        return jnp.logical_or(jnp.logical_or(outside_world_bounds, time_exceeded), exceeded_error_velocity)
    
    def _get_reward(self, env_state, action):
        '''
        Get reward from the environment state. 
        Reward is defined as the "LQR" cost function: scaled position error and scaled velocity error
        '''
        state = env_state

        return -0.01 * (jnp.linalg.norm(state.ref_pos - state.pos)**2 + jnp.linalg.norm(state.ref_vel - state.vel)**2) - 0.0 * (jnp.linalg.norm(action)**2)

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if not self.equivariant:
            non_eq_state = jnp.hstack([state.pos, state.vel, state.ref_pos, state.ref_vel])
            return non_eq_state
        else:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel])
            return eq_state

    def _maybe_reset(self, key, env_state, done):
        '''
        Reset helper to work with the jax conditional flow. 
        If the done flag is True, run self._reset with the key. 
        If done is False, return the env_state as is.
        '''
        return lax.select(done, self._reset(key), env_state)

    def _reset(self, key):
        '''
        Reset function for the environment. Returns the full env_state (state, key)
        '''
        pos = jrandom.multivariate_normal(key, self.state_mean, self.state_cov)
        # vel = jnp.zeros(3)
        vel = jrandom.multivariate_normal(key, jnp.zeros(3), jnp.eye(3) * 0.5)

        ref_pos = lax.cond(self.predefined_ref_pos is None, self._sample_random_ref_pos, self._get_predefined_ref_pos, key)

        # ref_pos = lax.cond(self.predefined_ref_pos is None, 
        #                    lambda _: jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov), 
        #                    lambda _: predefined_ref_pos, None)
        ref_vel = jnp.zeros(3) # hard coded to be non moving
        time = 0.0
        new_key = jrandom.split(key)[0]

        new_point_state = PointState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, time=time)

        return new_point_state
    
    def _sample_random_ref_pos(self, key):
        return jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov)
    
    def _get_predefined_ref_pos(self, key):
        return jnp.array(self.predefined_ref_pos) if self.predefined_ref_pos is not None else jnp.zeros_like(self.ref_mean)

    def reset(self, key):
        env_state = self._reset(key)
        return env_state, self._get_obs(env_state)
    
    @property
    def name(self)-> str:
        return "PointParticlePosition"
    
    @property
    def num_actions(self) -> int:
        return 3

    @property
    def EnvState(self):
        return PointState
    
    def action_space(self) -> spaces.Box:
        low = jnp.array([-1., -1., -1.])
        high = jnp.array([1., 1., 1.])
        return spaces.Box(low, high, (3,), jnp.float32)
    
    def observation_space(self) -> spaces.Box:
        n_obs = 6 if self.equivariant else 12 # this ONLY works since this is dependent on a constructor arg but this is bad behavior. 
        low = jnp.array(n_obs*[-jnp.finfo(jnp.float32).max])
        high = jnp.array(n_obs*[jnp.finfo(jnp.float32).max])

        return spaces.Box(low, high, (n_obs,), jnp.float32)
    


# Code used for testing the environment
if __name__ == "__main__":
    seed = 0
    key = jrandom.PRNGKey(seed)

    # env = PointParticlePosition()
    # env_state, obs = env.reset(key)

    

    # obs_buffer = []

    # def rollout_body(carry, unused):
    #     action = jrandom.multivariate_normal(key, jnp.zeros(3), jnp.eye(3) * 0.1)
    #     env_state, obs = carry
    #     env_state, obs, reward, done, info = env.step(env_state, action)
    #     return (env_state, obs), obs

    # # Rollout for 100 steps
    # _, obs_buffer = lax.scan(rollout_body, (env_state, obs), jnp.arange(100))

    # print(obs_buffer.shape) # (100, 4, 3) - 100 steps, 4 observations (pos, vel, ref_pos, ref_vel), 3 dimensions
    # print(obs_buffer[0]) # Initial observation
    # print(obs_buffer[-1]) # Final observation



    # env = PointParticlePosition(jnp.array([5.,5.,5.]))
    env = PointParticlePosition()
    reset_rng = jrandom.split(key, 10) # 10 parallel environments
    env_states, obs = jax.vmap(env.reset)(reset_rng)
    print(obs)

    next_env_states, next_obs, reward, done, info = jax.vmap(env.step)(env_states, jnp.ones((10,3)))
    print(next_obs)


    
