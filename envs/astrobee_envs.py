import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random as jrandom
from jax import lax
from jax import jit
from functools import partial


from gymnax.environments import spaces

from envs.base_envs import SE3FAQuadState, SE3FAQuadRandomWalkState, SE3QuadFullyActuatedBase, SE3FAQuadLissajousTrackingState

class SE3QuadFullyActuatedPosition(SE3QuadFullyActuatedBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Choose kt such that CoM acc values are \in [-1, 1]
        self.m = 1.0
        self.kt = 0.25
        self.kd = 1.0

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

        print("Creating SE3FAQuadrotor position environment with Equivaraint: ", self.equivariant)

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
        action = jnp.clip(action, -1., 1.)

        state = env_state

        # update particle position
        #ft = self.alloc_matrix_ft @ (jnp.sign(action) * (action * action))
        #tau = self.alloc_matrix_tau @ (jnp.sign(action) * (action * action))

        omega_hat = self._hat_map(state.omega)

        omega_dot = jnp.linalg.inv(self.J) @ (-omega_hat @ self.J @ state.omega + action[3:])
        #F = jnp.array([0.0, 0.0, ft/self.m])
        F = action[:3]
        vel_dot = state.rotm @ F

        omega = state.omega + omega_dot * self.dt
        rotm = state.rotm @ jsp.linalg.expm(omega_hat * self.dt)
        vel = state.vel + vel_dot * self.dt
        pos = state.pos + state.vel * self.dt

        #is_nan = jnp.any(jnp.isnan(action))
        #lax.cond(is_nan, self._there_is_nan, lambda **args : None)

        #jax.debug.print("pos: {pos}", pos=pos)

        # update reference position
        ref_vel = state.ref_vel # Hardcoded - no moving reference
        ref_pos = state.ref_pos + ref_vel * self.dt
        ref_rotm = jnp.eye(3,)
        ref_omega = jnp.zeros(3,)

        # update time
        time = state.time + self.dt

        done = self._is_terminal(env_state)

        env_state = SE3FAQuadState(pos=pos, vel=vel, rotm=rotm, omega=omega, ref_pos=ref_pos, ref_vel=ref_vel, ref_rotm=ref_rotm, ref_omega=ref_omega, time=time)

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state, action), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}
    

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if not self.equivariant:
            non_eq_state = jnp.hstack([state.pos, state.vel, jnp.ravel(state.rotm), state.omega, state.ref_pos, state.ref_vel, jnp.ravel(state.ref_rotm), state.ref_omega])
            return non_eq_state
        else:
            eq_state = jnp.hstack([state.ref_rotm.T @ (state.pos - state.ref_pos), state.vel - state.ref_vel, jnp.ravel(state.rotm.T @ state.ref_rotm), state.omega - state.ref_omega])
            return eq_state

    def _reset(self, key):
        '''
        Reset function for the environment. Returns the full env_state (state, key)
        '''
        key, pos_key, vel_key, ref_key = jrandom.split(key, 4)
        pos = jrandom.multivariate_normal(pos_key, self.state_mean, self.state_cov)
        vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3), self.state_cov)

        ref_pos = lax.cond(self.predefined_ref_pos is None, self._sample_random_ref_pos, self._get_predefined_ref_pos, ref_key)

        omega = jnp.zeros(3,)
        rotm = jnp.eye(3)

        # ref_pos = lax.cond(self.predefined_ref_pos is None, 
        #                    lambda _: jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov), 
        #                    lambda _: predefined_ref_pos, None)
        ref_vel = jnp.zeros(3) # hard coded to be non moving
        ref_omega = jnp.zeros(3,)
        ref_rotm = jnp.eye(3)
        time = 0.0
        new_key = jrandom.split(key)[0]

        new_point_state = SE3FAQuadState(pos=pos, vel=vel, rotm=rotm, omega=omega, ref_pos=ref_pos, ref_vel=ref_vel, ref_rotm=ref_rotm, ref_omega=ref_omega, time=time)

        return new_point_state

    def reset(self, key):
        key, reset_key = jrandom.split(key)
        env_state = self._reset(reset_key)
        return env_state, self._get_obs(env_state)
    
    @property
    def name(self)-> str:
        return "SE3QuadFullyActuatedPosition"

    @property
    def EnvState(self):
        return SE3FAQuadState
    
    def observation_space(self) -> spaces.Box:
        n_obs = 18 if self.equivariant else 36 # this ONLY works since this is dependent on a constructor arg but this is bad behavior. 
        low = jnp.array(n_obs*[-jnp.finfo(jnp.float32).max])
        high = jnp.array(n_obs*[jnp.finfo(jnp.float32).max])

        return spaces.Box(low, high, (n_obs,), jnp.float32)
    
    
class SE3QuadFullyActuatedRandomWalk(SE3QuadFullyActuatedBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Choose kt such that CoM acc values are \in [-1, 1]
        self.m = 0.5
        # self.m = 1.0
        self.kt = 0.25
        self.kd = 1.0

        # Choose inertia matrix
        # self.J = jnp.diag(jnp.array([0.5, 0.5, 1.0]))
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

        print("Creating SE3FAQuadrotor random walk environment with Equivaraint: ", self.equivariant)

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
        if self.equivariant == 3:
            ref_action = jnp.concat((env_state.ref_F, jnp.zeros(3,)))
            action = action + ref_action
        if self.equivariant == 4: 
            ref_action = jnp.concat((env_state.ref_F, env_state.ref_tau))
            action = action + ref_action
        action = jnp.clip(action, -1., 1.)

        state = env_state

        # update particle position
        #ft = self.alloc_matrix_ft @ (jnp.sign(action) * (action * action))
        #tau = self.alloc_matrix_tau @ (jnp.sign(action) * (action * action))

        omega_hat = self._hat_map(state.omega)

        omega_dot = jnp.linalg.inv(self.J) @ (-omega_hat @ self.J @ state.omega + action[3:])
        #F = jnp.array([0.0, 0.0, ft/self.m])
        F = action[:3]
        #vel_dot = state.rotm @ (F / self.m)
        
        # Here, in R3 x SO(3), F is in world frame, \tau is in body frame
        if self.symmetry_type == 0 or self.symmetry_type == 1:
            vel_dot = F / self.m
        elif self.symmetry_type == 2 or self.symmetry_type == 3:
            vel_dot = state.rotm @ (F / self.m)

        omega = state.omega + omega_dot * self.dt
        rotm = state.rotm @ jsp.linalg.expm(omega_hat * self.dt)
        vel = state.vel + vel_dot * self.dt
        pos = state.pos + state.vel * self.dt

        #is_nan = jnp.any(jnp.isnan(action))
        #lax.cond(is_nan, self._there_is_nan, lambda **args : None)

        #jax.debug.print("pos: {pos}", pos=pos)

        # update reference position according to random walk
        _, F_key, tau_key = jrandom.split(state.rnd_key, 3)
        
        rand_F = jrandom.truncated_normal(F_key, lower=-0.25 * jnp.ones(3,), upper=0.25 * jnp.ones(3,))
        rand_tau = jrandom.truncated_normal(tau_key, lower=-0.1 * jnp.ones(3,), upper=0.1 * jnp.ones(3,))
        
        des_action = jnp.concat((rand_F, rand_tau))
        
        #random_vel_dot = state.rotm @ rand_F
        if self.symmetry_type == 0 or self.symmetry_type == 1:
            random_vel_dot = rand_F / self.m
        elif self.symmetry_type == 2 or self.symmetry_type == 3:
            random_vel_dot = state.ref_rotm @ (rand_F / self.m)
        #random_vel_dot = rand_F / self.m
        ref_vel = state.ref_vel + random_vel_dot * self.dt
        ref_pos = state.ref_pos + state.ref_vel * self.dt
        
        ref_omega_hat = self._hat_map(state.ref_omega)
        ref_omega_dot = jnp.linalg.inv(self.J) @ (-ref_omega_hat @ self.J @ state.omega + rand_tau)
        
        ref_rotm = state.ref_rotm @ jsp.linalg.expm(ref_omega_hat * self.dt)
        ref_omega = state.ref_omega + ref_omega_dot * self.dt

        # update time
        time = state.time + self.dt

        done = self._is_terminal(env_state)

        env_state = SE3FAQuadRandomWalkState(pos=pos, vel=vel, rotm=rotm, omega=omega, ref_pos=ref_pos, ref_vel=ref_vel, ref_rotm=ref_rotm, ref_omega=ref_omega, time=time, ref_F=rand_F, ref_tau=rand_tau, rnd_key=F_key, F=action[:3], tau=action[3:])

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state, action, des_action), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}
    

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if self.equivariant == 0:
            non_eq_state = jnp.hstack([state.pos, state.vel, jnp.ravel(state.rotm), state.omega, state.ref_pos, state.ref_vel, jnp.ravel(state.ref_rotm), state.ref_omega, state.ref_F, state.ref_tau])
            return non_eq_state
        #p,p
        elif self.equivariant == 1:
            if self.symmetry_type == 0 or self.symmetry_type == 1:
                eq_state = jnp.hstack([state.ref_pos - state.pos, state.vel, state.ref_vel, jnp.ravel(state.rotm.T @ state.ref_rotm), state.omega, state.ref_omega, state.ref_F, state.ref_tau])
            elif self.symmetry_type == 2 or self.symmetry_type == 3:
                eq_state = jnp.hstack([state.rotm.T @ (state.ref_pos - state.pos), state.vel, state.ref_vel, jnp.ravel(state.rotm.T @ state.ref_rotm), state.omega, state.ref_omega, state.ref_F, state.ref_tau])
            else:
                raise NotImplementedError("Invalid Symmetry Type!")            
            return eq_state
        else:
            raise NotImplementedError("Invalid Equivariance Type!")

    def _reset(self, key):
        '''
        Reset function for the environment. Returns the full env_state (state, key)
        '''
        key, pos_key, vel_key, omega_key, rotm_key, ref_key = jrandom.split(key, 6)
        pos = jrandom.multivariate_normal(pos_key, self.state_mean, 1e-4 * self.state_cov)
        vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3), 1e-4 * self.state_cov)

        ref_pos = lax.cond(self.predefined_ref_pos is None, self._sample_random_ref_pos, self._get_predefined_ref_pos, ref_key)

        omega = jrandom.multivariate_normal(omega_key, jnp.zeros(3), 1e-5 * self.state_cov)
        rotm, _ = jnp.linalg.qr(1e-1 * jrandom.normal(rotm_key, (3,3)))

        # ref_pos = lax.cond(self.predefined_ref_pos is None, 
        #                    lambda _: jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov), 
        #                    lambda _: predefined_ref_pos, None)
        ref_vel = jnp.zeros(3) # hard coded to be non moving
        ref_omega = jnp.zeros(3,)
        ref_rotm = jnp.eye(3)
        time = 0.0
        new_key = jrandom.split(key)[0]
        ref_F = jnp.zeros(3,)
        ref_tau = jnp.zeros(3,)
        F = jnp.zeros(3,)
        tau = jnp.zeros(3,)

        new_point_state = SE3FAQuadRandomWalkState(pos=pos, vel=vel, rotm=rotm, omega=omega, ref_pos=ref_pos, ref_vel=ref_vel, ref_rotm=ref_rotm, ref_omega=ref_omega, time=time, ref_F=ref_F, ref_tau=ref_tau, rnd_key=new_key, F=F, tau=tau)

        return new_point_state

    def reset(self, key):
        key, reset_key = jrandom.split(key)
        env_state = self._reset(reset_key)
        return env_state, self._get_obs(env_state)
    
    @property
    def name(self)-> str:
        return "SE3QuadFullyActuatedRandomWalk"

    @property
    def EnvState(self):
        return SE3FAQuadRandomWalkState
    
    def observation_space(self) -> spaces.Box:
        if self.equivariant == 0:
            n_obs = 42
        elif self.equivariant == 5:
            n_obs = 33
        elif self.equivariant == 1 or self.equivariant == 6:
            n_obs = 30
        elif 2 <= self.equivariant <= 4:
            n_obs = 27
        else:
            raise NotImplemented("Invalid Equivaraince type!")
        
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
    

class SE3QuadFullyActuatedLissajous(SE3QuadFullyActuatedBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Choose kt such that CoM acc values are \in [-1, 1]
        self.m = 0.5
        # self.m = 1.0
        self.kt = 0.25
        self.kd = 1.0

        # Choose inertia matrix
        # self.J = jnp.diag(jnp.array([0.5, 0.5, 1.0]))
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

        print("Creating SE3FAQuadrotor Lissajous environment with Equivaraint: ", self.equivariant)
        
        self.ref_pos_fn = jax.jit(lissajous_3D)
        self.ref_vel_fn = jax.jit(jax.jacfwd(lissajous_3D))
        self.ref_acc_fn = jax.jit(jax.jacfwd(jax.jacfwd(lissajous_3D)))
        
        #self.ref_omeref_r_exponeital_coords__fn = jax.jit(exp(lissajous_3D))
        #self.ref_omega_dot_fn = jax.jit(jax.jacfwd(exp(lissajous_omega)))
        self.ref_omega_fn = jax.jit(lissajous_omega)
        self.ref_omega_dot_fn = jax.jit(jax.jacfwd(lissajous_omega))

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
        if self.equivariant == 3:
            ref_action = jnp.concat((env_state.ref_F, jnp.zeros(3,)))
            action = action + ref_action
        if self.equivariant == 4: 
            ref_action = jnp.concat((env_state.ref_F, env_state.ref_tau))
            action = action + ref_action
        action = jnp.clip(action, -1., 1.)

        state = env_state
        # update time
        time = state.time + self.dt

        # update particle position
        #ft = self.alloc_matrix_ft @ (jnp.sign(action) * (action * action))
        #tau = self.alloc_matrix_tau @ (jnp.sign(action) * (action * action))

        omega_hat = self._hat_map(state.omega)

        tau = action[3:]

        omega_dot = jnp.linalg.inv(self.J) @ (-omega_hat @ self.J @ state.omega + tau)
        #F = jnp.array([0.0, 0.0, ft/self.m])
        F = action[:3]
        #vel_dot = state.rotm @ F
        
        # Here, in R3 x SO(3), F is in world frame, \tau is in body frame
        #vel_dot = F / self.m
        if self.symmetry_type == 0 or self.symmetry_type == 1:
            vel_dot = F / self.m
        elif self.symmetry_type == 2 or self.symmetry_type == 3:
            vel_dot = state.rotm @ (F / self.m)

        omega = state.omega + omega_dot * self.dt
        rotm = state.rotm @ jsp.linalg.expm(omega_hat * self.dt)
        vel = state.vel + vel_dot * self.dt
        pos = state.pos + state.vel * self.dt

        #is_nan = jnp.any(jnp.isnan(action))
        #lax.cond(is_nan, self._there_is_nan, lambda **args : None)

        #jax.debug.print("pos: {pos}", pos=pos)

        # update reference position according to random walk
        _, F_key, tau_key = jrandom.split(state.rnd_key, 3)
        
        ref_pos = self.ref_pos_fn(jnp.array([time]), state.amplitudes, state.frequencies, state.phases).squeeze()
        ref_vel = self.ref_vel_fn(jnp.array([time]), state.amplitudes, state.frequencies, state.phases).squeeze()
        ref_acc = self.ref_acc_fn(jnp.array([time]), state.amplitudes, state.frequencies, state.phases).squeeze()
        
        ref_F = ref_acc * self.m
        
        #rand_F = jrandom.truncated_normal(F_key, lower=-0.25 * jnp.ones(3,), upper=0.25 * jnp.ones(3,))
        #rand_tau = jrandom.truncated_normal(tau_key, lower=-0.1 * jnp.ones(3,), upper=0.1 * jnp.ones(3,))
        
        #random_vel_dot = state.rotm @ rand_F
        #random_vel_dot = rand_F / self.m
        #ref_vel = state.ref_vel + random_vel_dot * self.dt
        #ref_pos = state.ref_pos + state.ref_vel * self.dt
        
        #ref_omega = self.ref_omega_fn(jnp.array([time]), state.amplitudes * 1e-2, state.frequencies * 0.1, state.phases).squeeze()
        ref_omega_dot = self.ref_omega_dot_fn(jnp.array([time]), state.amplitudes * 5e-1, state.frequencies, state.phases).squeeze()
        ref_omega = self.ref_omega_fn(jnp.array([time]), state.amplitudes * 5e-1, state.frequencies, state.phases).squeeze()
        #ref_omega = state.ref_omega + ref_omega_dot * self.dt
        ref_omega_hat = self._hat_map(ref_omega)
        ref_rotm = state.ref_rotm @ jsp.linalg.expm(ref_omega_hat * self.dt)

        ref_tau = self.J @ ref_omega_dot + ref_omega_hat @ self.J @ ref_omega

        des_action = jnp.concat((ref_F, ref_tau))

        done = self._is_terminal(env_state)
        #jax.debug.print("Done?: {x}", x=done)

        env_state = SE3FAQuadLissajousTrackingState(pos=pos, vel=vel, rotm=rotm, omega=omega, ref_pos=ref_pos, ref_vel=ref_vel, ref_rotm=ref_rotm, ref_omega=ref_omega, time=time, ref_F=ref_F, ref_tau=ref_tau, rnd_key=F_key, F=F, tau=tau, amplitudes=state.amplitudes, frequencies=state.frequencies, phases=state.phases, )

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state, action, des_action), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}
    

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if self.equivariant == 0:
            non_eq_state = jnp.hstack([state.pos, state.vel, jnp.ravel(state.rotm), state.omega, state.ref_pos, state.ref_vel, jnp.ravel(state.ref_rotm), state.ref_omega, state.ref_F, state.ref_tau])
            return non_eq_state
        #p,p
        elif self.equivariant == 1:
            if self.symmetry_type == 0 or self.symmetry_type == 1:
                eq_state = jnp.hstack([state.ref_pos - state.pos, state.vel, state.ref_vel, jnp.ravel(state.rotm.T @ state.ref_rotm), state.omega, state.ref_omega, state.ref_F, state.ref_tau])
            elif self.symmetry_type == 2 or self.symmetry_type == 3:
                eq_state = jnp.hstack([state.rotm.T @ (state.ref_pos - state.pos), state.vel, state.ref_vel, jnp.ravel(state.rotm.T @ state.ref_rotm), state.omega, state.ref_omega, state.ref_F, state.ref_tau])
            else:
                raise NotImplementedError("Invalid Symmetry Type!")            
            return eq_state
        else:
            raise NotImplementedError("Invalid Equivariance Type!")

    def _reset(self, key):
        '''
        Reset function for the environment. Returns the full env_state (state, key)
        '''
        key, pos_key, vel_key, omega_key, rotm_key, amp_key, freq_key, phase_key = jrandom.split(key, 8)
        
        amplitudes = jrandom.uniform(amp_key, (3,), minval=-5., maxval=5.)
        frequencies = jrandom.uniform(freq_key, (3,), minval=0.05, maxval=0.1)
        phases = jrandom.uniform(phase_key, (3,), minval=0., maxval=2.0 * jnp.pi)
        
        time = 0.0

        #omega = jnp.zeros(3,)
        #rotm = jnp.eye(3)

        omega = jrandom.multivariate_normal(omega_key, jnp.zeros(3), 1e-5 * self.state_cov)
        rotm, _ = jnp.linalg.qr(1e-2 * jrandom.normal(rotm_key, (3,3)))

        # ref_pos = lax.cond(self.predefined_ref_pos is None, 
        #                    lambda _: jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov), 
        #                    lambda _: predefined_ref_pos, None)
        ref_vel = jnp.zeros(3) # hard coded to be non moving
        ref_omega = jnp.zeros(3,)
        ref_rotm = jnp.eye(3)
        _, new_key = jrandom.split(key)
        ref_tau = jnp.zeros(3,)

        ref_pos = self.ref_pos_fn(jnp.array([time]), amplitudes, frequencies, phases).squeeze()
        ref_vel = self.ref_vel_fn(jnp.array([time]), amplitudes, frequencies, phases).squeeze()
        ref_acc = self.ref_acc_fn(jnp.array([time]), amplitudes, frequencies, phases).squeeze()

        pos = jrandom.multivariate_normal(pos_key, ref_pos, 5 * self.state_cov)
        vel = jrandom.multivariate_normal(vel_key, ref_vel, 5 * self.state_cov)

        ref_omega = self.ref_omega_fn(jnp.array([time]), amplitudes, frequencies, phases).squeeze()
        ref_omega_dot = self.ref_omega_dot_fn(jnp.array([time]), amplitudes, frequencies, phases).squeeze()

        ref_F = self.m * ref_acc
        ref_tau = self.J @ ref_omega_dot + self._hat_map(ref_omega) @ self.J @ ref_omega
        
        F = jnp.zeros(3,)
        tau = jnp.zeros(3,)

        new_point_state = SE3FAQuadLissajousTrackingState(pos=pos, vel=vel, rotm=rotm, omega=omega, ref_pos=ref_pos, ref_vel=ref_vel, ref_rotm=ref_rotm, ref_omega=ref_omega, time=time, ref_F=ref_F, ref_tau=ref_tau, rnd_key=new_key, amplitudes=amplitudes, frequencies=frequencies, phases=phases, F=F, tau=tau)

        return new_point_state

    def reset(self, key):
        key, reset_key = jrandom.split(key)
        env_state = self._reset(reset_key)
        return env_state, self._get_obs(env_state)
    
    @property
    def name(self)-> str:
        return "SE3QuadFullyActuatedLissajousTracking"

    @property
    def EnvState(self):
        return SE3FAQuadLissajousTrackingState
    
    def observation_space(self) -> spaces.Box:
        if self.equivariant == 0:
            n_obs = 42
        elif self.equivariant == 5:
            n_obs = 33
        elif self.equivariant == 1 or self.equivariant == 6:
            n_obs = 30
        elif 2 <= self.equivariant <= 4:
            n_obs = 27
        else:
            raise NotImplemented("Invalid Equivaraince type!")
        
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