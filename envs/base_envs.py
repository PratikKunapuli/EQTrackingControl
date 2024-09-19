import numpy as np
import scipy.linalg
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

"""
Point Particle Data Structs
"""
@struct.dataclass
class PointState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_acc: jnp.ndarray
    lqr_cmd: jnp.ndarray

@struct.dataclass
class PointVelocityState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_acc: jnp.ndarray

@struct.dataclass
class PointRandomWalkState(EnvState):
    rnd_key: jnp.ndarray
    pos: jnp.ndarray
    vel: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_acc: jnp.ndarray


@struct.dataclass
class PointLissajousTrackingState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_acc: jnp.ndarray
    amplitudes: jnp.ndarray
    frequencies: jnp.ndarray
    phases: jnp.ndarray

@struct.dataclass
class SE3QuadState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    rotm: jnp.ndarray
    omega: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_rotm: jnp.ndarray
    ref_omega: jnp.ndarray

"""
Astrobee Data Structs
"""
@struct.dataclass
class SE3FAQuadState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    rotm: jnp.ndarray
    omega: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_rotm: jnp.ndarray
    ref_omega: jnp.ndarray
    
@struct.dataclass
class SE3FAQuadRandomWalkState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    rotm: jnp.ndarray
    omega: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_rotm: jnp.ndarray
    ref_omega: jnp.ndarray
    rnd_key: jnp.ndarray
    ref_F: jnp.ndarray
    ref_tau: jnp.ndarray
    F: jnp.ndarray
    tau: jnp.ndarray
    
@struct.dataclass
class SE3FAQuadLissajousTrackingState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    rotm: jnp.ndarray
    omega: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_rotm: jnp.ndarray
    ref_omega: jnp.ndarray
    rnd_key: jnp.ndarray
    ref_F: jnp.ndarray
    ref_tau: jnp.ndarray
    F: jnp.ndarray
    tau: jnp.ndarray
    amplitudes: jnp.ndarray
    frequencies: jnp.ndarray
    phases: jnp.array


"""
Quadrotor Data Structs
"""
@struct.dataclass
class SE2xRQuadState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    rotm: jnp.ndarray
    omega: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_rotm: jnp.ndarray
    ref_omega: jnp.ndarray
    action: jnp.ndarray
    ref_action: jnp.ndarray
    
@struct.dataclass
class SE2xRQuadRandomWalkState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    rotm: jnp.ndarray
    omega: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_rotm: jnp.ndarray
    ref_omega: jnp.ndarray
    rnd_key: jnp.ndarray
    action: jnp.ndarray
    ref_action: jnp.ndarray

@struct.dataclass
class SE2xRQuadLissajousState(EnvState):
    pos: jnp.ndarray
    vel: jnp.ndarray
    rotm: jnp.ndarray
    omega: jnp.ndarray
    ref_pos: jnp.ndarray
    ref_vel: jnp.ndarray
    ref_rotm: jnp.ndarray
    ref_omega: jnp.ndarray
    rnd_key: jnp.ndarray
    action: jnp.ndarray
    ref_action: jnp.ndarray
    amplitudes: jnp.ndarray
    phases: jnp.ndarray
    frequencies: jnp.ndarray
    ref_iter: int
    ref_state_list: jnp.ndarray
    ref_u_list: jnp.ndarray


"""
Point Particle Base Environment
"""
class PointParticleBase:
    def __init__ (self, ref_pos=None, equivariant=False, state_cov_scalar=0.5, ref_cov_scalar=3.0, dt=0.05, max_time=100.0, terminate_on_error=True, 
                  reward_q_pos = 1e-2, reward_q_vel = 1e-2, reward_r = 1e-4, termination_bound = 10., terminal_reward = 0.0, reward_reach=False, 
                  use_des_action_in_reward=True, clip_actions=True, reward_fn_type=False, **kwargs):
        self.state_mean = jnp.array([0., 0., 0.])
        self.state_cov = jnp.eye(3) * state_cov_scalar
        self.ref_mean = jnp.array([0., 0., 0.])
        self.ref_cov = jnp.eye(3) * ref_cov_scalar
        self.predefined_ref_pos = ref_pos

        self.max_time = max_time
        self.dt = dt
        self.equivariant = equivariant
        
        self.terminate_on_error = terminate_on_error
        self.termination_bound = termination_bound
        self.terminal_reward = terminal_reward
        self.reach_reward = reward_reach
        self.reward_q_pos = reward_q_pos
        self.reward_q_vel = reward_q_vel
        self.reward_r = reward_r
        self.use_des_action_in_reward = use_des_action_in_reward
        self.clip_actions = clip_actions
        self.reward_fn_type = reward_fn_type

        self.epsilon_ball_radius = 1e-2

        self.other_args = kwargs

        # For LQR calculations
        A = np.eye(6,)
        A[:3, 3:] = dt * np.eye(3,)

        B = np.zeros((6, 3))
        B[3:, :] = dt * np.eye(3,)

        Q = 10 * np.eye(6)
        R = 1 * np.eye(3)

        self.gamma = 0.995

        S = scipy.linalg.solve_discrete_are((self.gamma ** 0.5) * A, B, Q, 1/(self.gamma) * R)

        K = np.linalg.inv(R + self.gamma * B.T @ S @ B) @ B.T @ S @ A

        self.K = jnp.asarray(K)

    def _sample_random_ref_pos(self, key):
        """
        Helper function to sample a random reference position from a multivariate normal distribution
        """
        key, new_key = jrandom.split(key)
        return jrandom.multivariate_normal(new_key, self.ref_mean, self.ref_cov)
    
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
    
    def _is_terminal(self, env_state,):
        """
        Helper function to check if the environment state is terminal.
        If self.terminate_on_error is True, then the environment is terminal if the particle is outside the world bounds or exceeds the velocity error.
        """
            
        # if any part of the state is outside the bounds of +- 5, then the episode is done
        # or if the time is greater than the max time

        time_exceeded = env_state.time > self.max_time

        world_cond = self._is_terminal_error(env_state)
        world_cond = lax.select(self.terminate_on_error, world_cond, False)

        return jnp.logical_or(world_cond, time_exceeded)
    
    def _is_terminal_error(self, env_state):
        # outside_world_bounds = jnp.any(jnp.linalg.norm(env_state.ref_pos - env_state.pos)**2 > self.termination_bound)
        # exceeded_error_velocity = jnp.any(jnp.linalg.norm(env_state.ref_vel - env_state.vel)**2 > self.termination_bound)
        outside_world_bounds = jnp.any(jnp.abs(env_state.pos - env_state.ref_pos) > self.termination_bound)
        #exceeded_error_velocity = jnp.any(jnp.abs(env_state.vel - env_state.ref_vel) > self.termination_bound)
        # exceeded_error_velocity = False

        #return jnp.logical_or(outside_world_bounds, exceeded_error_velocity)
        return outside_world_bounds

    def _is_terminal_reach(self, env_state):

        return jnp.any(jnp.linalg.norm(env_state.ref_pos - env_state.pos) ** 2 < self.epsilon_ball_radius)

    def _get_reward(self, env_state, action, des_action=0.0):
        '''
        Get reward from the environment state. 
        Reward is defined as the "LQR" cost function: scaled position error and scaled velocity error
        '''
        state = env_state
        termination_from_error = self._is_terminal_error(state)

        terminal_reward = lax.select(termination_from_error, self.terminal_reward, 0.0)

        #dest_reached = self._is_terminal_reach(state)
        #dest_reach_reward = lax.select(dest_reached, self.reach_reward, 0.0)
        dest_reach_reward_modified = 1.0 - jnp.tanh(0.01 * jnp.linalg.norm(env_state.ref_pos - env_state.pos) / (self.epsilon_ball_radius * 0.5))
        
        dest_reach_reward = lax.select(self.reach_reward, dest_reach_reward_modified, 0.0)

        if self.reward_fn_type == 0:
            reward_no_des_act = -self.reward_q_pos * (jnp.sum(jnp.abs(state.pos - state.ref_pos))) - self.reward_q_vel * (jnp.sum(jnp.abs(state.vel - state.ref_vel))) - self.reward_r * jnp.sum((jnp.abs(action))) + terminal_reward + dest_reach_reward
            reward_yes_des_act = -self.reward_q_pos * (jnp.sum(jnp.abs(state.pos - state.ref_pos))) - self.reward_q_vel * (jnp.sum(jnp.abs(state.vel - state.ref_vel))) - self.reward_r * (jnp.sum(jnp.abs(des_action - action))) + terminal_reward + dest_reach_reward
        elif self.reward_fn_type == 1:
            reward_no_des_act = -self.reward_q_pos * (jnp.linalg.norm(state.pos - state.ref_pos)) - self.reward_q_vel * (jnp.linalg.norm(state.vel - state.ref_vel)) - self.reward_r * (jnp.linalg.norm(action)) + terminal_reward + dest_reach_reward
            reward_yes_des_act = -self.reward_q_pos * (jnp.linalg.norm(state.pos - state.ref_pos)) - self.reward_q_vel * (jnp.linalg.norm(state.vel - state.ref_vel)) - self.reward_r * (jnp.linalg.norm(des_action - action)) + terminal_reward + dest_reach_reward
        elif self.reward_fn_type == 2:
            reward_no_des_act = -self.reward_q_pos * (jnp.linalg.norm(state.pos - state.ref_pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.vel - state.ref_vel)**2) - self.reward_r * (jnp.linalg.norm(action)**2) + terminal_reward + dest_reach_reward
            reward_yes_des_act = -self.reward_q_pos * (jnp.linalg.norm(state.pos - state.ref_pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.vel - state.ref_vel)**2) - self.reward_r * (jnp.linalg.norm(des_action - action)**2) + terminal_reward + dest_reach_reward
        else:
            raise NotImplementedError("Invalid Reward function type!")

        final_reward = lax.select(self.use_des_action_in_reward, reward_yes_des_act, reward_no_des_act)

        #return -self.reward_q * (jnp.linalg.norm(state.ref_pos - state.pos)**2 + jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action)**2) + terminal_reward + dest_reach_reward
        #return -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action)**2) + terminal_reward
        ## Add (a - a_{des}) -> Important for random walk envs
        #return -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action - des_action)**2) + terminal_reward
        return final_reward

    @property
    def num_actions(self) -> int:
        return 3
    
    def action_space(self) -> spaces.Box:
        low = jnp.array([-1., -1., -1.])
        high = jnp.array([1., 1., 1.])
        return spaces.Box(low, high, (3,), jnp.float32)

"""
Astrobee Base Environment
"""
class SE3QuadFullyActuatedBase:
    def __init__ (self, ref_pos=None, equivariant=0, state_cov_scalar=0.5, ref_cov_scalar=3.0, dt=0.05, max_time=100.0, terminate_on_error=True, 
                  reward_q_pos = 1e-2, reward_q_vel = 1e-2, reward_r = 1e-4, termination_bound = 10., terminal_reward = 0.0, reward_reach=False,
                   use_des_action_in_reward=True, reward_q_rot=1e-3, reward_q_omega=1e-4, use_abs_reward_fn=False, symmetry_type=0, **kwargs):
        self.state_mean = jnp.array([0., 0., 0.])
        self.state_cov = jnp.eye(3) * state_cov_scalar
        self.ref_mean = jnp.array([0., 0., 0.])
        self.ref_cov = jnp.eye(3) * ref_cov_scalar
        self.predefined_ref_pos = ref_pos

        self.max_time = max_time
        self.dt = dt
        self.equivariant = equivariant
        
        self.terminate_on_error = terminate_on_error
        self.termination_bound = termination_bound
        self.terminal_reward = terminal_reward
        self.reach_reward = reward_reach
        self.reward_q_pos = reward_q_pos
        self.reward_q_vel = reward_q_vel
        self.reward_q_rot = reward_q_rot
        self.reward_q_omega = reward_q_omega
        self.reward_r = reward_r
        self.use_des_action_in_reward = use_des_action_in_reward
        self.use_abs_reward_fn = use_abs_reward_fn
        self.symmetry_type = symmetry_type

        self.epsilon_ball_radius = 1e-2

        self.other_args = kwargs

    def _sample_random_ref_pos(self, key):
        """
        Helper function to sample a random reference position from a multivariate normal distribution
        """
        key, new_key = jrandom.split(key)
        return jrandom.multivariate_normal(new_key, self.ref_mean, self.ref_cov)
    
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
    
    def _is_terminal(self, env_state,):
        """
        Helper function to check if the environment state is terminal.
        If self.terminate_on_error is True, then the environment is terminal if the particle is outside the world bounds or exceeds the velocity error.
        """
            
        # if any part of the state is outside the bounds of +- 5, then the episode is done
        # or if the time is greater than the max time

        time_exceeded = env_state.time > self.max_time
        world_cond = self._is_terminal_error(env_state)
        world_cond = lax.select(self.terminate_on_error, world_cond, False)

        return jnp.logical_or(world_cond, time_exceeded)
    
    def _is_terminal_error(self, env_state):
        # outside_world_bounds = jnp.any(jnp.linalg.norm(env_state.ref_pos - env_state.pos)**2 > self.termination_bound)
        # exceeded_error_velocity = jnp.any(jnp.linalg.norm(env_state.ref_vel - env_state.vel)**2 > self.termination_bound)
        outside_world_bounds = jnp.any(jnp.abs(env_state.ref_pos - env_state.pos) > self.termination_bound)
        exceeded_error_velocity = jnp.any(jnp.abs(env_state.ref_vel - env_state.vel) > self.termination_bound)
        exceeded_error_angular_velocity = jnp.any(jnp.abs(env_state.ref_omega - env_state.omega) > 5. * self.termination_bound)
        
        #jax.debug.print("outside_world_bounds: {x}", x=outside_world_bounds)
        #jax.debug.print("Exceeded error velocity: {x}", x=exceeded_error_velocity)
        #jax.debug.print("Exceeded error angular velocity: {x}", x=exceeded_error_angular_velocity)
        #jax.debug.print("Error velocity: {x}", x=jnp.abs(env_state.ref_vel - env_state.vel))

        #euler = Rotation.from_matrix(env_state.rotm).as_euler("zyx", degrees=False)
        #exceeded_error_euler = jnp.any(jnp.abs(euler) > jnp.pi/3)
        # exceeded_error_velocity = False

        #return jnp.logical_or(jnp.logical_or(outside_world_bounds, exceeded_error_velocity,), jnp.logical_or(exceeded_error_angular_velocity, exceeded_error_euler))
        return jnp.logical_or(jnp.logical_or(outside_world_bounds, exceeded_error_velocity,), exceeded_error_angular_velocity)

    def _is_terminal_reach(self, env_state):

        return jnp.any(jnp.linalg.norm(env_state.ref_pos - env_state.pos) ** 2 < self.epsilon_ball_radius)

    # Computes SO(3) matrix log
    # As per http://www.sci.utah.edu/%7Efletcher/RiemannianGeometryNotes.pdf
    # page 20. *mistake in ref: Note that log(R) = 0, not I (when \theta=0)
    def _logm(self, theta_unclipped, R):
        # Just to be insanely safe, prevent small values of theta
        clip_theta = jnp.any(jnp.abs(theta_unclipped) < 1e-4)

        theta = lax.select(clip_theta, jnp.sign(theta_unclipped) * 1e-4, theta_unclipped)
        
        logm_SO3 = (theta/(2 * jnp.sin(theta))) * (R - R.T)

        return logm_SO3

    def _zerom(self, theta, R):

        return jnp.zeros((3,3))

    def _compute_logm(self, R):
        #1) Get \theta = 0.5 * cos^{-1}(tr(R) - 1)
        theta = jnp.arccos( jnp.clip(0.5 * (jnp.trace(R) - 1.0), -1.0, 1.0) )

        #2) Compute log of R:
        # check if theta is very small
        theta_very_small = jnp.any(jnp.abs(theta) < 1e-4)

        # Compute logm only if theta is not small
        # If small, return identity matrix (which is the mathematical limiting case)
        log_R = lax.cond(theta_very_small, self._zerom, self._logm, theta, R)

        return log_R

    def _get_matrix_norm(self, R):
        return 0.707 * jnp.linalg.matrix_norm(R, ord="fro")
    
    def _return_zero(self, _):
        return 0.0

    def _get_reward(self, env_state, action, des_action=0.0):
        '''
        Get reward from the environment state. 
        Reward is defined as the "LQR" cost function: scaled position error and scaled velocity error
        '''
        state = env_state
        termination_from_error = self._is_terminal_error(state)

        terminal_reward = lax.select(termination_from_error, self.terminal_reward, 0.0)

        #dest_reached = self._is_terminal_reach(state)
        #dest_reach_reward = lax.select(dest_reached, self.reach_reward, 0.0)
        dest_reach_reward_modified = 1.0 - jnp.tanh(0.01 * jnp.linalg.norm(env_state.ref_pos - env_state.pos) / (self.epsilon_ball_radius * 0.5))
        
        dest_reach_reward = lax.select(self.reach_reward, dest_reach_reward_modified, 0.0)

        logm_rot_err = self._compute_logm(state.ref_rotm.T @ state.rotm)

        # # I realized that when logm is very small (close to zero), the matrix norm (Frobenius) also is very small
        # # This makes the gradients extremely large as norm function is not differentiable close to origin
        # # https://github.com/google/jax/issues/3058#issuecomment-628105523
        # # Thus, as discussed above, we can simply replace by zeros
        # is_logm_rot_small = jnp.any(jnp.abs(logm_rot_err) < 1e-4)

        #geodesic_SO3 = lax.cond(is_logm_rot_small, self._return_zero, self._get_matrix_norm, logm_rot_err)
        geodesic_SO3 = self._get_matrix_norm(logm_rot_err)

        # if self.use_abs_reward_fn:
        #     reward_no_des_act = -self.reward_q_pos * (jnp.sum(jnp.abs(state.ref_pos - state.pos))) - self.reward_q_vel * (jnp.sum(jnp.abs(state.ref_vel - state.vel))) - self.reward_q_rot * (geodesic_SO3) - self.reward_q_omega * (jnp.sum(jnp.abs(state.ref_omega - state.omega))) - self.reward_r * (jnp.sum(jnp.abs(action))) + terminal_reward + dest_reach_reward
        #     reward_yes_des_act = -self.reward_q_pos * (jnp.sum(jnp.abs(state.ref_pos - state.pos))) - self.reward_q_vel * (jnp.sum(jnp.abs(state.ref_vel - state.vel))) - self.reward_q_rot * (geodesic_SO3) - self.reward_q_omega * (jnp.sum(jnp.abs(state.ref_omega - state.omega))) - self.reward_r * (jnp.sum(jnp.abs(action - des_action))) + terminal_reward + dest_reach_reward

        # else:
        #     reward_no_des_act = -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_q_rot * (geodesic_SO3) - self.reward_q_omega * (jnp.linalg.norm(state.ref_omega - state.omega)) - self.reward_r * (jnp.linalg.norm(action)**2) + terminal_reward + dest_reach_reward
        #     reward_yes_des_act = -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_q_rot * (geodesic_SO3) - self.reward_q_omega * (jnp.linalg.norm(state.ref_omega - state.omega)) - self.reward_r * (jnp.linalg.norm(action - des_action)**2) + terminal_reward + dest_reach_reward

        if self.use_abs_reward_fn:
            # R3xSO(3), direct product
            if self.symmetry_type == 0:
                omega_error = state.omega - state.ref_omega
                vel_error = state.vel - state.ref_vel
                pos_error = state.ref_pos - state.pos
            # R3xSO(3), semi-direct product
            elif self.symmetry_type == 1:
                vel_error = state.vel - state.ref_vel
                omega_error = state.omega - state.rotm @ state.ref_rotm.T @ state.ref_omega
                pos_error = state.ref_pos - state.pos
            # SE(3), direct product
            elif self.symmetry_type == 2:
                vel_error = state.vel - state.ref_vel
                omega_error = state.omega - state.ref_omega
                pos_error = state.rotm.T @ (state.ref_pos - state.pos)
            # SE(3), semi-direct product
            elif self.symmetry_type == 3:
                vel_error = state.vel + self._hat_map(state.rotm @ state.ref_rotm.T @ state.ref_omega) @ state.pos - state.rotm @ state.ref_rotm.T @ state.ref_vel
                omega_error = state.omega - state.rotm @ state.ref_rotm.T @ state.ref_omega
                pos_error = state.rotm.T @ (state.ref_pos - state.pos)
            else:
                raise ValueError("Invalid Symmetry type!")

            reward_no_des_act = -self.reward_q_pos * (jnp.sum(jnp.abs(pos_error))) - self.reward_q_vel * (jnp.sum(jnp.abs(vel_error))) - self.reward_q_rot * (geodesic_SO3) - self.reward_q_omega * (jnp.sum(jnp.abs(omega_error))) - self.reward_r * (jnp.sum(jnp.abs(action))) + terminal_reward + dest_reach_reward
            reward_yes_des_act = -self.reward_q_pos * (jnp.sum(jnp.abs(pos_error))) - self.reward_q_vel * (jnp.sum(jnp.abs(vel_error))) - self.reward_q_rot * (geodesic_SO3) - self.reward_q_omega * (jnp.sum(jnp.abs(omega_error))) - self.reward_r * (jnp.sum(jnp.abs(action - des_action))) + terminal_reward + dest_reach_reward

        else:
            # R3xSO(3), direct product
            if self.symmetry_type == 0:
                omega_error = state.omega - state.ref_omega
                vel_error = state.vel - state.ref_vel
                pos_error = state.ref_pos - state.pos
            # R3xSO(3), semi-direct product
            elif self.symmetry_type == 1:
                vel_error = state.vel - state.ref_vel
                omega_error = state.omega - state.rotm @ state.ref_rotm.T @ state.ref_omega
                pos_error = state.ref_pos - state.pos
            # SE(3), direct product
            elif self.symmetry_type == 2:
                vel_error = state.vel - state.ref_vel
                omega_error = state.omega - state.ref_omega
                pos_error = state.rotm.T @ (state.ref_pos - state.pos)
            # SE(3), semi-direct product
            elif self.symmetry_type == 3:
                vel_error = state.vel + self._hat_map(state.rotm @ state.ref_rotm.T @ state.ref_omega) @ state.pos - state.rotm @ state.ref_rotm.T @ state.ref_vel
                omega_error = state.omega - state.rotm @ state.ref_rotm.T @ state.ref_omega
                pos_error = state.rotm.T @ (state.ref_pos - state.pos)
            else:
                raise ValueError("Invalid Symmetry type!")

            reward_no_des_act = -self.reward_q_pos * (jnp.linalg.norm(pos_error)) - self.reward_q_vel * (jnp.linalg.norm(vel_error)) - self.reward_q_rot * (geodesic_SO3) - self.reward_q_omega * (jnp.linalg.norm(omega_error)) - self.reward_r * (jnp.linalg.norm(action)) + terminal_reward + dest_reach_reward
            reward_yes_des_act = -self.reward_q_pos * (jnp.linalg.norm(pos_error)) - self.reward_q_vel * (jnp.linalg.norm(vel_error)) - self.reward_q_rot * (geodesic_SO3) - self.reward_q_omega * (jnp.linalg.norm(omega_error)) - self.reward_r * (jnp.linalg.norm(action - des_action)) + terminal_reward + dest_reach_reward

        final_reward = lax.select(self.use_des_action_in_reward, reward_yes_des_act, reward_no_des_act)

        #return -self.reward_q * (jnp.linalg.norm(state.ref_pos - state.pos)**2 + jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action)**2) + terminal_reward + dest_reach_reward
        #return -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action)**2) + terminal_reward
        ## Add (a - a_{des}) -> Important for random walk envs
        #return -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action - des_action)**2) + terminal_reward
        return final_reward

    @property
    def num_actions(self) -> int:
        return 6
    
    def action_space(self) -> spaces.Box:
        low = jnp.array([-1., -1., -1., -1., -1., -1.])
        high = jnp.array([1., 1., 1., 1., 1., 1.])
        return spaces.Box(low, high, (6,), jnp.float32)
    
"""
Quadrotor Base Environment
"""
class SE2xRQuadBase:
    def __init__ (self, ref_pos=None, equivariant=False, state_cov_scalar=0.5, ref_cov_scalar=3.0, dt=0.05, max_time=100.0, terminate_on_error=True, 
                  reward_q_pos = 1e-2, reward_q_vel = 1e-2, reward_r = 1e-4, termination_bound = 10., terminal_reward = 0.0, reward_reach=False,
                   use_des_action_in_reward=True, reward_q_rot=1e-3, reward_q_omega=1e-4, use_abs_reward_fn=True, **kwargs):
        self.state_mean = jnp.array([0., 0., 0.])
        self.state_cov = jnp.eye(3) * state_cov_scalar
        self.ref_mean = jnp.array([0., 0., 0.])
        self.ref_cov = jnp.eye(3) * ref_cov_scalar
        self.predefined_ref_pos = ref_pos

        self.max_time = max_time
        self.dt = dt
        self.equivariant = equivariant
        
        self.terminate_on_error = terminate_on_error
        self.termination_bound = termination_bound
        self.terminal_reward = terminal_reward
        self.reach_reward = reward_reach
        self.reward_q_pos = reward_q_pos
        self.reward_q_vel = reward_q_vel
        self.reward_q_rot = reward_q_rot
        self.reward_q_omega = reward_q_omega
        self.reward_r = reward_r
        self.use_des_action_in_reward = use_des_action_in_reward
        self.use_abs_reward_fn = use_abs_reward_fn

        self.g = 9.81

        self.epsilon_ball_radius = 1e-2

        self.other_args = kwargs

    def _sample_random_ref_pos(self, key):
        """
        Helper function to sample a random reference position from a multivariate normal distribution
        """
        key, new_key = jrandom.split(key)
        return jrandom.multivariate_normal(new_key, self.ref_mean, self.ref_cov)
    
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
    
    def _is_terminal(self, env_state,):
        """
        Helper function to check if the environment state is terminal.
        If self.terminate_on_error is True, then the environment is terminal if the particle is outside the world bounds or exceeds the velocity error.
        """
            
        # if any part of the state is outside the bounds of +- 5, then the episode is done
        # or if the time is greater than the max time

        time_exceeded = env_state.time > self.max_time

        world_cond = self._is_terminal_error(env_state)
        world_cond = lax.select(self.terminate_on_error, world_cond, False)

        return jnp.logical_or(world_cond, time_exceeded)
    
    def _is_terminal_error(self, env_state):
        # outside_world_bounds = jnp.any(jnp.linalg.norm(env_state.ref_pos - env_state.pos)**2 > self.termination_bound)
        # exceeded_error_velocity = jnp.any(jnp.linalg.norm(env_state.ref_vel - env_state.vel)**2 > self.termination_bound)
        outside_world_bounds = jnp.any(jnp.abs(env_state.ref_pos - env_state.pos) > self.termination_bound)
        exceeded_error_velocity = jnp.any(jnp.abs(env_state.ref_vel - env_state.vel) > self.termination_bound)
        exceeded_error_angular_velocity = jnp.any(jnp.abs(env_state.ref_omega - env_state.omega) > self.termination_bound)
        
        euler = Rotation.from_matrix(env_state.rotm).as_euler("zyx", degrees=False)
        ref_euler = Rotation.from_matrix(env_state.ref_rotm).as_euler("zyx", degrees=False)
        exceeded_error_euler = jnp.any(jnp.abs(ref_euler - euler) > jnp.pi/2)
        # exceeded_error_velocity = False

        return jnp.logical_or(jnp.logical_or(outside_world_bounds, exceeded_error_velocity,), jnp.logical_or(exceeded_error_angular_velocity, exceeded_error_euler))
        #return jnp.logical_or(jnp.logical_or(outside_world_bounds, exceeded_error_velocity,), exceeded_error_angular_velocity)

    def _is_terminal_reach(self, env_state):

        return jnp.any(jnp.linalg.norm(env_state.ref_pos - env_state.pos) ** 2 < self.epsilon_ball_radius)

    # Computes SO(3) matrix log
    # As per http://www.sci.utah.edu/%7Efletcher/RiemannianGeometryNotes.pdf
    # page 20. *mistake in ref: Note that log(R) = 0, not I (when \theta=0)
    def _logm(self, theta_unclipped, R):
        # Just to be insanely safe, prevent small values of theta
        clip_theta = jnp.any(jnp.abs(theta_unclipped) < 1e-4)

        theta = lax.select(clip_theta, jnp.sign(theta_unclipped) * 1e-4, theta_unclipped)
        
        logm_SO3 = (theta/(2 * jnp.sin(theta))) * (R - R.T)

        return logm_SO3

    def _zerom(self, theta, R):

        return jnp.zeros((3,3))

    def _compute_logm(self, R):
        #1) Get \theta = 0.5 * cos^{-1}(tr(R) - 1)
        theta = jnp.arccos( jnp.clip(0.5 * (jnp.trace(R) - 1.0), -1.0, 1.0) )

        #2) Compute log of R:
        # check if theta is very small
        theta_very_small = jnp.any(jnp.abs(theta) < 1e-4)

        # Compute logm only if theta is not small
        # If small, return identity matrix (which is the mathematical limiting case)
        log_R = lax.cond(theta_very_small, self._zerom, self._logm, theta, R)

        return log_R

    def _get_matrix_norm(self, R):
        return 0.707 * jnp.linalg.matrix_norm(R, ord="fro")
    
    def _return_zero(self, _):
        return 0.0

    def _get_reward(self, env_state, action, des_action=0.0):
        '''
        Get reward from the environment state. 
        Reward is defined as the "LQR" cost function: scaled position error and scaled velocity error
        '''
        state = env_state
        termination_from_error = self._is_terminal_error(state)

        terminal_reward = lax.select(termination_from_error, self.terminal_reward, 0.0)

        #dest_reached = self._is_terminal_reach(state)
        #dest_reach_reward = lax.select(dest_reached, self.reach_reward, 0.0)
        dest_reach_reward_modified = 1.0 - jnp.tanh(0.01 * jnp.linalg.norm(env_state.ref_pos - env_state.pos) / (self.epsilon_ball_radius * 0.5))
        
        dest_reach_reward = lax.select(self.reach_reward, dest_reach_reward_modified, 0.0)

        logm_rot_err = self._compute_logm(state.ref_rotm.T @ state.rotm)

        # # I realized that when logm is very small (close to zero), the matrix norm (Frobenius) also is very small
        # # This makes the gradients extremely large as norm function is not differentiable close to origin
        # # https://github.com/google/jax/issues/3058#issuecomment-628105523
        # # Thus, as discussed above, we can simply replace by zeros
        # is_logm_rot_small = jnp.any(jnp.abs(logm_rot_err) < 1e-3)

        #geodesic_SO3 = lax.cond(is_logm_rot_small, self._return_zero, self._get_matrix_norm, logm_rot_err)
        geodesic_SO3 = self._get_matrix_norm(logm_rot_err)

        if self.use_abs_reward_fn:
            reward_no_des_act = -self.reward_q_pos * (jnp.sum(jnp.abs(state.ref_pos - state.pos))) - self.reward_q_vel * (jnp.sum(jnp.abs(state.ref_vel - state.vel))) - self.reward_q_rot * (geodesic_SO3) - self.reward_q_omega * (jnp.sum(jnp.abs(state.ref_omega - state.omega))) - self.reward_r * (jnp.sum(jnp.abs(action))) + terminal_reward + dest_reach_reward
            reward_yes_des_act = -self.reward_q_pos * (jnp.sum(jnp.abs(state.ref_pos - state.pos))) - self.reward_q_vel * (jnp.sum(jnp.abs(state.ref_vel - state.vel))) - self.reward_q_rot * (geodesic_SO3) - self.reward_q_omega * (jnp.sum(jnp.abs(state.ref_omega - state.omega))) - self.reward_r * (jnp.sum(jnp.abs(action - des_action))) + terminal_reward + dest_reach_reward
        else:
            reward_no_des_act = -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_q_rot * (geodesic_SO3) - self.reward_q_omega * (jnp.linalg.norm(state.ref_omega - state.omega)) - self.reward_r * (jnp.linalg.norm(action)**2) + terminal_reward + dest_reach_reward
            reward_yes_des_act = -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_q_rot * (geodesic_SO3) - self.reward_q_omega * (jnp.linalg.norm(state.ref_omega - state.omega)) - self.reward_r * (jnp.linalg.norm(action - des_action)**2) + terminal_reward + dest_reach_reward

        final_reward = lax.select(self.use_des_action_in_reward, reward_yes_des_act, reward_no_des_act)

        #return -self.reward_q * (jnp.linalg.norm(state.ref_pos - state.pos)**2 + jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action)**2) + terminal_reward + dest_reach_reward
        #return -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action)**2) + terminal_reward
        ## Add (a - a_{des}) -> Important for random walk envs
        #return -self.reward_q_pos * (jnp.linalg.norm(state.ref_pos - state.pos)**2) - self.reward_q_vel * (jnp.linalg.norm(state.ref_vel - state.vel)**2) - self.reward_r * (jnp.linalg.norm(action - des_action)**2) + terminal_reward
        return final_reward

    @property
    def num_actions(self) -> int:
        return 4
    
    def action_space(self) -> spaces.Box:
        low = jnp.array([0., 0., 0., 0.])
        high = jnp.array([1., 1., 1., 1.])
        return spaces.Box(low, high, (4,), jnp.float32)