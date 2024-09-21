import jax
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random as jrandom
from jax import lax
from jax import jit
from functools import partial


from gymnax.environments import spaces

from envs.base_envs import SE2xRQuadState, SE2xRQuadRandomWalkState, SE2xRQuadBase, SE2xRQuadLissajousState


class SE2xRQuadPosition(SE2xRQuadBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Choose kt such that CoM acc values are \in [-1, 1]
        self.m = 0.067
        self.kt = 0.25
        self.kd = 0.8

        # Choose inertia matrix
        self.J = jnp.diag(jnp.array([0.1, 0.1, 0.2]))

        # 1x4 matrix
        self.alloc_matrix_ft = jnp.array(
            [self.kt, self.kt, self.kt, self.kt]
        )

        # 3x4 matrix
        self.alloc_matrix_tau = jnp.array([
            [0.0, -self.kd, 0.0, self.kd],
            [self.kd, 0.0, -self.kd, 0.0],
            [self.kd, -self.kd, self.kd, -self.kd]])

        print("Creating SE2xRQuadrotor environment with Equivaraint: ", self.equivariant)

    def _hat_map(self, a):
        hat_a = jnp.array([
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ])

        return hat_a

    def step(self, key, env_state, action):
        '''
        Step function for the environment. Arguments are defined as follows:

        key: random_key for the environmen (need )
        env_state: current state of the environment (both true state of the particle and the reference state) [pos, vel, ref_pos, ref_vel]
        action: action taken by the agent (velocity of the particle)

        returns: tuple: (env_state, observation, reward, done, info)
        '''

        # clip action
        action = jnp.clip(action, 0., 1.)

        state = env_state

        # Obtain ft, tau from actions (rotor speeds squared)
        ft = self.alloc_matrix_ft @ (action)
        tau = self.alloc_matrix_tau @ (action)
        # ft = action[0]
        # tau = action[1:]

        omega_hat = self._hat_map(state.omega)

        omega_dot = jnp.linalg.inv(self.J) @ (-omega_hat @ self.J @ state.omega + tau)
        F = jnp.array([0.0, 0.0, ft/self.m])
        g_e3 = self.g * jnp.array([0., 0., 1.])
        vel_dot = state.rotm @ F - g_e3

        omega = state.omega + omega_dot * self.dt
        rotm = state.rotm @ jsp.linalg.expm(self._hat_map(omega) * self.dt)
        vel = state.vel + vel_dot * self.dt
        pos = state.pos + state.vel * self.dt

        #is_nan = jnp.any(jnp.isnan(action))
        #lax.cond(is_nan, self._there_is_nan, lambda **args : None)

        #jax.debug.print("pos: {pos}", pos=pos)

        # update reference position
        ref_vel = state.ref_vel # Hardcoded - no moving reference
        ref_pos = state.ref_pos + state.ref_vel * self.dt
        ref_rotm = jnp.eye(3,)
        ref_omega = jnp.zeros(3,)

        # update time
        time = state.time + self.dt

        done = self._is_terminal(env_state)

        env_state = SE2xRQuadState(pos=pos, vel=vel, rotm=rotm, omega=omega, ref_pos=ref_pos, ref_vel=ref_vel, ref_rotm=ref_rotm, ref_omega=ref_omega, time=time, action=action, ref_action=state.ref_action)

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state, action, state.ref_action), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}
    
    def _get_rotm(self, phi, theta, psi):
        
        return jnp.array([
            [jnp.cos(theta)*jnp.cos(psi), jnp.sin(phi)*jnp.sin(theta)*jnp.cos(psi) - jnp.cos(phi)*jnp.sin(psi), jnp.cos(phi)*jnp.sin(theta)*jnp.cos(psi) + jnp.sin(phi)*jnp.sin(psi)],
            [jnp.cos(theta)*jnp.sin(psi), jnp.sin(phi)*jnp.sin(theta)*jnp.sin(psi) + jnp.cos(phi)*jnp.cos(psi), jnp.cos(phi)*jnp.sin(theta)*jnp.sin(psi) - jnp.sin(phi)*jnp.cos(psi)],
            [-jnp.sin(theta), jnp.sin(phi)*jnp.cos(theta), jnp.cos(phi)*jnp.cos(theta)]
        ])

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if not self.equivariant:
            non_eq_state = jnp.hstack([state.pos, state.vel, jnp.ravel(state.rotm), state.omega, state.ref_pos, state.ref_vel, jnp.ravel(state.ref_rotm), state.ref_omega, state.ref_action])
            return non_eq_state
        else:
            eq_state = jnp.hstack([state.ref_rotm.T @ (state.ref_pos - state.pos), state.ref_vel, state.vel, jnp.ravel(state.ref_rotm.T @ state.rotm), state.ref_omega, state.omega, state.ref_action, state.rotm[:, 2]])
            #jax.debug.print("shape: {x}", x=eq_state.shape)
            return eq_state

    def _reset(self, key):
        '''
        Reset function for the environment. Returns the full env_state (state, key)
        '''
        key, pos_key, vel_key, ref_key, euler_key, omega_key = jrandom.split(key, 6)
        pos = jrandom.multivariate_normal(pos_key, self.state_mean, self.state_cov)
        vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3), self.state_cov)
        #vel = jnp.zeros(3)

        ref_pos = lax.cond(self.predefined_ref_pos is None, self._sample_random_ref_pos, self._get_predefined_ref_pos, ref_key)

        omega = jrandom.multivariate_normal(omega_key, jnp.zeros(3), 5e-3 * self.state_cov)
        random_euler = jrandom.uniform(euler_key, (3,), minval=-jnp.pi/6, maxval=jnp.pi/6)
        rotm = self._get_rotm(*random_euler)

        # ref_pos = lax.cond(self.predefined_ref_pos is None, 
        #                    lambda _: jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov), 
        #                    lambda _: predefined_ref_pos, None)
        ref_vel = jnp.zeros(3) # hard coded to be non moving
        ref_omega = jnp.zeros(3,)
        ref_rotm = jnp.eye(3)
        time = 0.0
        new_key = jrandom.split(key)[0]
        ref_action = 0.6573 * jnp.ones(4,)
        action = jnp.zeros(4,)      

        new_point_state = SE2xRQuadState(pos=pos, vel=vel, rotm=rotm, omega=omega, ref_pos=ref_pos, ref_vel=ref_vel, ref_rotm=ref_rotm, ref_omega=ref_omega, time=time, action=action, ref_action=ref_action)

        return new_point_state

    def reset(self, key):
        key, reset_key = jrandom.split(key)
        env_state = self._reset(reset_key)
        return env_state, self._get_obs(env_state)
    
    @property
    def name(self)-> str:
        return "SE2xRQuadPosition"

    @property
    def EnvState(self):
        return SE2xRQuadState
    
    def observation_space(self) -> spaces.Box:
        n_obs = 31 if self.equivariant else 40 # this ONLY works since this is dependent on a constructor arg but this is bad behavior. 
        low = jnp.array(n_obs*[-jnp.finfo(jnp.float32).max])
        high = jnp.array(n_obs*[jnp.finfo(jnp.float32).max])

        return spaces.Box(low, high, (n_obs,), jnp.float32)


class SE2xRQuadRandomWalk(SE2xRQuadBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Choose kt such that CoM acc values are \in [-1, 1]
        self.m = 0.067
        self.kt = 0.25
        self.kd = 0.8

        # Choose inertia matrix
        self.J = jnp.diag(jnp.array([0.1, 0.1, 0.2]))

        # 1x4 matrix
        self.alloc_matrix_ft = jnp.array(
            [self.kt, self.kt, self.kt, self.kt]
        )

        # 3x4 matrix
        self.alloc_matrix_tau = jnp.array([
            [0.0, -self.kd, 0.0, self.kd],
            [self.kd, 0.0, -self.kd, 0.0],
            [self.kd, -self.kd, self.kd, -self.kd]])

        self.hover_thrust = 0.6573 * jnp.ones(4,)

        print("Creating SE2xRQuadrotor environment with Equivaraint: ", self.equivariant)

    def _hat_map(self, a):
        hat_a = jnp.array([
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ])

        return hat_a

    def step(self, key, env_state, action):
        '''
        Step function for the environment. Arguments are defined as follows:

        key: random_key for the environmen (need )
        env_state: current state of the environment (both true state of the particle and the reference state) [pos, vel, ref_pos, ref_vel]
        action: action taken by the agent (velocity of the particle)

        returns: tuple: (env_state, observation, reward, done, info)
        '''

        # clip action
        action = jnp.clip(action, 0.0, 1.0)

        state = env_state

        # Obtain ft, tau from actions (rotor speeds squared)
        ft = self.alloc_matrix_ft @ (action)
        tau = self.alloc_matrix_tau @ (action)
        # ft = action[0]
        # tau = action[1:]

        omega_hat = self._hat_map(state.omega)

        omega_dot = jnp.linalg.inv(self.J) @ (-omega_hat @ self.J @ state.omega + tau)
        F = jnp.array([0.0, 0.0, ft/self.m])
        g_e3 = self.g * jnp.array([0., 0., 1.])
        vel_dot = state.rotm @ F - g_e3

        omega = state.omega + omega_dot * self.dt
        rotm = state.rotm @ jsp.linalg.expm(self._hat_map(omega) * self.dt)
        vel = state.vel + vel_dot * self.dt
        pos = state.pos + state.vel * self.dt

        #is_nan = jnp.any(jnp.isnan(action))
        #lax.cond(is_nan, self._there_is_nan, lambda **args : None)

        #jax.debug.print("pos: {pos}", pos=pos)

        # update reference position
        ref_action_key = state.rnd_key
        _, ref_action_key = jrandom.split(ref_action_key)
        
        ref_action = self.hover_thrust + jrandom.truncated_normal(ref_action_key, lower=-1e-3, upper=1e-3, shape=(4,))
        ref_ft = self.alloc_matrix_ft @ (ref_action)
        ref_tau = self.alloc_matrix_tau @ (ref_action)
        
        ref_omega_hat = self._hat_map(state.ref_omega)

        ref_omega_dot = jnp.linalg.inv(self.J) @ (-ref_omega_hat @ self.J @ state.ref_omega + ref_tau)
        ref_F = jnp.array([0.0, 0.0, ref_ft/self.m])
        ref_vel_dot = state.ref_rotm @ ref_F - g_e3

        ref_omega = state.ref_omega + ref_omega_dot * self.dt
        ref_rotm = state.ref_rotm @ jsp.linalg.expm(self._hat_map(state.ref_omega) * self.dt)
        ref_vel = state.ref_vel + ref_vel_dot * self.dt
        ref_pos = state.ref_pos + state.ref_vel * self.dt
        
        # ref_vel = state.ref_vel # Hardcoded - no moving reference
        # ref_pos = state.ref_pos + state.ref_vel * self.dt
        # ref_rotm = jnp.eye(3,)
        # ref_omega = jnp.zeros(3,)

        # update time
        time = state.time + self.dt

        done = self._is_terminal(env_state)

        env_state = SE2xRQuadRandomWalkState(pos=pos, vel=vel, rotm=rotm, omega=omega, ref_pos=ref_pos, ref_vel=ref_vel, ref_rotm=ref_rotm, ref_omega=ref_omega, time=time, action=action, ref_action=ref_action, rnd_key=ref_action_key)

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state, action, state.ref_action), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}
    
    def _get_rotm(self, phi, theta, psi):
        
        return jnp.array([
            [jnp.cos(theta)*jnp.cos(psi), jnp.sin(phi)*jnp.sin(theta)*jnp.cos(psi) - jnp.cos(phi)*jnp.sin(psi), jnp.cos(phi)*jnp.sin(theta)*jnp.cos(psi) + jnp.sin(phi)*jnp.sin(psi)],
            [jnp.cos(theta)*jnp.sin(psi), jnp.sin(phi)*jnp.sin(theta)*jnp.sin(psi) + jnp.cos(phi)*jnp.cos(psi), jnp.cos(phi)*jnp.sin(theta)*jnp.sin(psi) - jnp.sin(phi)*jnp.cos(psi)],
            [-jnp.sin(theta), jnp.sin(phi)*jnp.cos(theta), jnp.cos(phi)*jnp.cos(theta)]
        ])

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if not self.equivariant:
            non_eq_state = jnp.hstack([state.pos, state.vel, jnp.ravel(state.rotm), state.omega, state.ref_pos, state.ref_vel, jnp.ravel(state.ref_rotm), state.ref_omega, state.ref_action])
            return non_eq_state
        else:
            eq_state = jnp.hstack([state.rotm.T @ (state.ref_pos - state.pos), state.ref_vel, state.vel, jnp.ravel(state.rotm.T @ state.ref_rotm), state.ref_omega, state.omega, state.ref_action, state.rotm[:, 2]])
            #jax.debug.print("shape: {x}", x=eq_state.shape)
            return eq_state

    def _reset(self, key):
        '''
        Reset function for the environment. Returns the full env_state (state, key)
        '''
        key, pos_key, vel_key, ref_key, ref_action_key, euler_key, omega_key = jrandom.split(key, 7)
        pos = jrandom.multivariate_normal(pos_key, self.state_mean, self.state_cov)
        vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3), self.state_cov)
        #vel = jnp.zeros(3)

        ref_pos = lax.cond(self.predefined_ref_pos is None, self._sample_random_ref_pos, self._get_predefined_ref_pos, ref_key)

        omega = jrandom.multivariate_normal(omega_key, jnp.zeros(3), 5e-3 * self.state_cov)
        random_euler = jrandom.uniform(euler_key, (3,), minval=-jnp.pi/6, maxval=jnp.pi/6)
        rotm = self._get_rotm(*random_euler)

        # ref_pos = lax.cond(self.predefined_ref_pos is None, 
        #                    lambda _: jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov), 
        #                    lambda _: predefined_ref_pos, None)
        ref_vel = jnp.zeros(3) # hard coded to be non moving
        ref_omega = jnp.zeros(3,)
        ref_rotm = jnp.eye(3)
        time = 0.0
        new_key = jrandom.split(key)[0]
        #ref_action = 0.6573 * jnp.array([1., 1., 1., 1.,])
        ref_action = self.hover_thrust.copy()
        #ref_action = jrandom.truncated_normal(ref_action_key, lower=0.6, upper=0.75, shape=(4,))
        action = jnp.zeros(4,)

        new_point_state = SE2xRQuadRandomWalkState(pos=pos, vel=vel, rotm=rotm, omega=omega, ref_pos=ref_pos, ref_vel=ref_vel, ref_rotm=ref_rotm, ref_omega=ref_omega, time=time, action=action, ref_action=ref_action, rnd_key=ref_action_key)

        return new_point_state

    def reset(self, key):
        key, reset_key = jrandom.split(key)
        env_state = self._reset(reset_key)
        return env_state, self._get_obs(env_state)
    
    @property
    def name(self)-> str:
        return "SE2xRQuadRandomWalk"

    @property
    def EnvState(self):
        return SE2xRQuadRandomWalkState
    
    def observation_space(self) -> spaces.Box:
        n_obs = 31 if self.equivariant else 40 # this ONLY works since this is dependent on a constructor arg but this is bad behavior. 
        low = jnp.array(n_obs*[-jnp.finfo(jnp.float32).max])
        high = jnp.array(n_obs*[jnp.finfo(jnp.float32).max])

        return spaces.Box(low, high, (n_obs,), jnp.float32)

@jit
def lissajous_3D(t, amplitudes, frequencies, phases):
    x = amplitudes[0] * jnp.sin(2.0 * jnp.pi * frequencies[0] * t + phases[0])
    y = amplitudes[1] * jnp.sin(2.0 * jnp.pi * frequencies[1] * t + phases[1])
    z = amplitudes[2] * jnp.sin(2.0 * jnp.pi * frequencies[2] * t + phases[2])
    
    return jnp.stack([x, y, z], axis=1)

@jit
def lissajous_omega(t, amplitudes, frequencies, phases):
    x = amplitudes[0] * jnp.sin(2.0 * jnp.pi * frequencies[0] * t + phases[0])
    y = amplitudes[1] * jnp.sin(2.0 * jnp.pi * frequencies[1] * t + phases[1])
    z = amplitudes[2] * jnp.sin(2.0 * jnp.pi * frequencies[2] * t + phases[2])
    
    return jnp.stack([x, y, z], axis=1)  


class SE2xRQuadLissajous(SE2xRQuadBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Choose kt such that CoM acc values are \in [-1, 1]
        self.m = 0.067
        self.kt = 0.25
        self.kd = 0.8

        # Choose inertia matrix
        self.J = jnp.diag(jnp.array([0.1, 0.1, 0.2]))

        # 1x4 matrix
        self.alloc_matrix_ft = jnp.array(
            [self.kt, self.kt, self.kt, self.kt]
        )

        # 3x4 matrix
        self.alloc_matrix_tau = jnp.array([
            [0.0, -self.kd, 0.0, self.kd],
            [self.kd, 0.0, -self.kd, 0.0],
            [self.kd, -self.kd, self.kd, -self.kd]])

        self.hover_thrust = 0.6573 * jnp.ones(4,)

        self.ref_pos_fn = jax.jit(lissajous_3D)
        self.ref_vel_fn = jax.jit(jax.jacfwd(lissajous_3D))
        self.ref_acc_fn = jax.jit(jax.jacfwd(jax.jacfwd(lissajous_3D)))
        
        # self.ref_state_batch = jnp.asarray(jnp.load("./ref_state_list.npy"))
        # self.ref_u_batch = jnp.asarray(jnp.load("./ref_u_list.npy"))
        
        # rnd_key = jrandom.PRNGKey(np.random.randint(low=-1e2, high=1e2,))
        
        # self.ref_state_list = jrandom.choice(rnd_key, self.ref_state_batch)
        # self.ref_u_list = jrandom.choice(rnd_key, self.ref_u_batch)
        # self.ref_iter = 0
        #self.ref_state_list = jnp.zeros((5000, 18))
        #self.ref_u_list = jnp.zeros((5000, 4))

        print("Creating SE2xRQuadrotor environment with Equivaraint: ", self.equivariant)

    def _hat_map(self, a):
        hat_a = jnp.array([
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ])

        return hat_a

    def step(self, key, env_state, action):
        '''
        Step function for the environment. Arguments are defined as follows:

        key: random_key for the environmen (need )
        env_state: current state of the environment (both true state of the particle and the reference state) [pos, vel, ref_pos, ref_vel]
        action: action taken by the agent (velocity of the particle)

        returns: tuple: (env_state, observation, reward, done, info)
        '''

        # clip action
        action = jnp.clip(action, 0.0, 1.0)

        state = env_state

        # Obtain ft, tau from actions (rotor speeds squared)
        ft = self.alloc_matrix_ft @ (action)
        tau = self.alloc_matrix_tau @ (action)
        # ft = action[0]
        # tau = action[1:]

        omega_hat = self._hat_map(state.omega)

        omega_dot = jnp.linalg.inv(self.J) @ (-omega_hat @ self.J @ state.omega + tau)
        F = jnp.array([0.0, 0.0, ft/self.m])
        g_e3 = self.g * jnp.array([0., 0., 1.])
        vel_dot = state.rotm @ F - g_e3

        omega = state.omega + omega_dot * self.dt
        rotm = state.rotm @ jsp.linalg.expm(self._hat_map(omega) * self.dt)
        vel = state.vel + vel_dot * self.dt
        pos = state.pos + state.vel * self.dt

        #is_nan = jnp.any(jnp.isnan(action))
        #lax.cond(is_nan, self._there_is_nan, lambda **args : None)

        #jax.debug.print("pos: {pos}", pos=pos)

        # update reference position
        ref_action_key = state.rnd_key
        _, ref_action_key = jrandom.split(ref_action_key)
        
        # ref_action = self.hover_thrust + jrandom.truncated_normal(ref_action_key, lower=-1e-3, upper=1e-3, shape=(4,))
        # ref_ft = self.alloc_matrix_ft @ (ref_action)
        # ref_tau = self.alloc_matrix_tau @ (ref_action)
        
        # ref_omega_hat = self._hat_map(state.ref_omega)

        # ref_omega_dot = jnp.linalg.inv(self.J) @ (-ref_omega_hat @ self.J @ state.ref_omega + ref_tau)
        # ref_F = jnp.array([0.0, 0.0, ref_ft/self.m])
        # ref_vel_dot = state.ref_rotm @ ref_F - g_e3

        # ref_omega = state.ref_omega + ref_omega_dot * self.dt
        # ref_rotm = state.ref_rotm @ jsp.linalg.expm(self._hat_map(state.ref_omega) * self.dt)
        # ref_vel = state.ref_vel + ref_vel_dot * self.dt
        # ref_pos = state.ref_pos + state.ref_vel * self.dt
        
        # update time
        time = state.time + self.dt

        # ref_pos = self.ref_pos_fn(jnp.array([time]), state.amplitudes, state.frequencies, state.phases).squeeze()
        # ref_vel = self.ref_vel_fn(jnp.array([time]), state.amplitudes, state.frequencies, state.phases).squeeze()
        # ref_acc = self.ref_acc_fn(jnp.array([time]), state.amplitudes, state.frequencies, state.phases).squeeze()
        
        # #ref_F = (ref_acc + g_e3) * self.m

        # ref_omega = jnp.zeros(3,)
        # ref_rotm = jnp.eye(3,)

        # ref_action = self.hover_thrust.copy()

        ref_pos = state.ref_state_list[state.ref_iter][:3]
        ref_vel = state.ref_state_list[state.ref_iter][3:6]
        ref_rotm = state.ref_state_list[state.ref_iter][6:15].reshape(3,3)
        ref_omega = state.ref_state_list[state.ref_iter][15:]
        
        ref_action = state.ref_u_list[state.ref_iter]

        ref_iter = state.ref_iter + 1

        #jax.debug.print("iter: {ref_iter}", ref_iter=ref_iter)

        # ref_vel = state.ref_vel # Hardcoded - no moving reference
        # ref_pos = state.ref_pos + state.ref_vel * self.dt
        # ref_rotm = jnp.eye(3,)
        # ref_omega = jnp.zeros(3,)

        done = self._is_terminal(env_state)

        env_state = SE2xRQuadLissajousState(pos=pos, vel=vel, rotm=rotm, omega=omega, ref_pos=ref_pos, ref_vel=ref_vel, ref_rotm=ref_rotm, ref_omega=ref_omega, time=time, action=action, ref_action=ref_action, rnd_key=ref_action_key, frequencies=state.frequencies, phases=state.phases, amplitudes=state.amplitudes, ref_iter=ref_iter, ref_state_list=state.ref_state_list, ref_u_list=state.ref_u_list)

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        #jax.debug.print("done?: {done}", done=done)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state, action, state.ref_action), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}
    

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if not self.equivariant:
            non_eq_state = jnp.hstack([state.pos, state.vel, jnp.ravel(state.rotm), state.omega, state.ref_pos, state.ref_vel, jnp.ravel(state.ref_rotm), state.ref_omega, state.ref_action])
            return non_eq_state
        else:
            eq_state = jnp.hstack([state.rotm.T @ (state.ref_pos - state.pos), state.ref_vel, state.vel, jnp.ravel(state.rotm.T @ state.ref_rotm), state.ref_omega, state.omega, state.ref_action, state.rotm[:, 2]])
            #jax.debug.print("shape: {x}", x=eq_state.shape)
            return eq_state

    def _get_rotm(self, phi, theta, psi):
        
        return jnp.array([
            [jnp.cos(theta)*jnp.cos(psi), jnp.sin(phi)*jnp.sin(theta)*jnp.cos(psi) - jnp.cos(phi)*jnp.sin(psi), jnp.cos(phi)*jnp.sin(theta)*jnp.cos(psi) + jnp.sin(phi)*jnp.sin(psi)],
            [jnp.cos(theta)*jnp.sin(psi), jnp.sin(phi)*jnp.sin(theta)*jnp.sin(psi) + jnp.cos(phi)*jnp.cos(psi), jnp.cos(phi)*jnp.sin(theta)*jnp.sin(psi) - jnp.sin(phi)*jnp.cos(psi)],
            [-jnp.sin(theta), jnp.sin(phi)*jnp.cos(theta), jnp.cos(phi)*jnp.cos(theta)]
        ])

    def _reset(self, key, traj_index=None):
        '''
        Reset function for the environment. Returns the full env_state (state, key)
        '''
        key, pos_key, vel_key, omega_key, euler_key, ref_key, ref_action_key, amp_key, freq_key, phase_key = jrandom.split(key, 10)
        #vel = jnp.zeros(3)

        ref_pos = lax.cond(self.predefined_ref_pos is None, self._sample_random_ref_pos, self._get_predefined_ref_pos, ref_key)

        pos = jrandom.multivariate_normal(pos_key, ref_pos, 1e-2 * self.state_cov)
        vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3), 1e-1 * self.state_cov)

        #omega = jnp.zeros(3,)
        #rotm = jnp.eye(3)
        omega = jrandom.multivariate_normal(omega_key, jnp.zeros(3,), 5e-3 * self.state_cov)
        random_euler = jrandom.uniform(euler_key, shape=(3,), minval=-jnp.pi/6, maxval=jnp.pi/6)
        rotm = self._get_rotm(*random_euler)

        # ref_pos = lax.cond(self.predefined_ref_pos is None, 
        #                    lambda _: jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov), 
        #                    lambda _: predefined_ref_pos, None)
        ref_vel = jnp.zeros(3) # hard coded to be non moving
        ref_omega = jnp.zeros(3,)
        ref_rotm = jnp.eye(3)
        time = 0.0
        new_key = jrandom.split(key)[0]
        #ref_action = 0.6573 * jnp.array([1., 1., 1., 1.,])
        ref_action = self.hover_thrust.copy()
        #ref_action = jrandom.truncated_normal(ref_action_key, lower=0.6, upper=0.75, shape=(4,))
        action = jnp.zeros(4,)

        # amplitudes = jnp.ones(3,)
        # phases = jnp.zeros(3,)
        # frequencies = 0.02 * jnp.ones(3,)

        amplitudes = jrandom.uniform(amp_key, (3,), minval=-4.0, maxval=4.0)
        # To prevent small values of amplitudes clip the range amp ~ Unif[-3, 3] -> Unif([-3, -1] U [1, 3])
        amplitudes = jnp.sign(amplitudes) * jnp.clip(jnp.abs(amplitudes), jnp.array([1.0, 1.0, 0.5]), jnp.array([4.0, 4.0, 3.0]))
        
        frequencies = jrandom.uniform(freq_key, (3,), minval=0.01, maxval=0.03)
        phases = jrandom.uniform(phase_key, (3,), minval=0., maxval=2.0 * jnp.pi)

        ref_state_batch = jnp.asarray(jnp.load("./ref_state_list.npy"))
        ref_u_batch = jnp.asarray(jnp.load("./ref_u_list.npy"))
        
        if traj_index is None:
            rnd_key = jrandom.PRNGKey(np.random.randint(low=-1e2, high=1e2,))
            
            ref_state_list = jrandom.choice(rnd_key, ref_state_batch)
            ref_u_list = jrandom.choice(rnd_key, ref_u_batch)
        else:
            
            ref_state_list = ref_state_batch[traj_index[0]]
            ref_u_list = ref_u_batch[traj_index[0]]
            #jax.debug.print("Yayyyy, {index}", index=traj_index)

        #jax.debug.print("Shape: {i} and {j}", i=ref_state_batch.shape, j=ref_u_batch.shape)
        #jax.debug.print("Shape: {i} and {j}", i=self.ref_state_list.shape, j=self.ref_u_list.shape)

        self.ref_iter = 0
        ref_pos = ref_state_list[0, :3]
        ref_vel = ref_state_list[0, 3:6]
        ref_rotm = ref_state_list[0, 6:15].reshape(3,3)
        ref_omega = ref_state_list[0, 15:]
        
        ref_action = ref_u_list[0, :]
        
        amplitudes = jnp.array([2, 2, 2])
        frequencies = jnp.array([0.025, 0.015, 0.0125])
        phases = jnp.zeros(3,)

        new_point_state = SE2xRQuadLissajousState(pos=pos, vel=vel, rotm=rotm, omega=omega, ref_pos=ref_pos, ref_vel=ref_vel, ref_rotm=ref_rotm, ref_omega=ref_omega, time=time, action=action, ref_action=ref_action, rnd_key=ref_action_key, amplitudes=amplitudes, frequencies=frequencies, phases=phases, ref_iter=0, ref_state_list=ref_state_list, ref_u_list=ref_u_list)

        return new_point_state

    def reset(self, key, traj_index=None):
        key, reset_key = jrandom.split(key)
        env_state = self._reset(reset_key, traj_index=traj_index)
        return env_state, self._get_obs(env_state)
    
    @property
    def name(self)-> str:
        return "SE2xRQuadLissajous"

    @property
    def EnvState(self):
        return SE2xRQuadLissajousState
    
    def observation_space(self) -> spaces.Box:
        n_obs = 31 if self.equivariant else 40 # this ONLY works since this is dependent on a constructor arg but this is bad behavior. 
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
    # env = PointParticlePosition()
    # reset_rng = jrandom.split(key, 10) # 10 parallel environments
    # env_states, obs = jax.vmap(env.reset)(reset_rng)
    # print(obs)

    # next_env_states, next_obs, reward, done, info = jax.vmap(env.step)(env_states, jnp.ones((10,3)))
    # print(next_obs)

    # env = SE3QuadPosition()
    # env_state, obs = env.reset(key)
    # print(env_state)
    # print(obs)

    # obs_buffer = []

    # def rollout_body(carry, unused):
    #     key, env_state, obs = carry
    #     action = jrandom.multivariate_normal(key, jnp.zeros(4), jnp.eye(4) * 0.1)
    #     env_state, obs, reward, done, info = env.step(key, env_state, action)
    #     return (key, env_state, obs), obs
    
    # # Rollout for 100 steps
    # _, obs_buffer = lax.scan(rollout_body, (key, env_state, obs), jnp.arange(100))

    # print(obs_buffer.shape) # (100, 4, 3) - 100 steps, 4 observations (pos, vel, ref_pos, ref_vel), 3 dimensions
    # print(obs_buffer[0]) # Initial observation
    # print(obs_buffer[-1]) # Final observation

    