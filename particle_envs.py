import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import lax
from jax import jit
from functools import partial


from gymnax.environments import spaces

from base_envs import PointParticleBase, PointState, PointVelocityState, PointRandomWalkState, PointLissajousTrackingState


class PointParticlePosition(PointParticleBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
    

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if not self.equivariant:
            non_eq_state = jnp.hstack([state.pos, state.vel, state.ref_pos, state.ref_vel])
            return non_eq_state
        else:
            # eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel])
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel])
            return eq_state

    def _reset(self, key):
        '''
        Reset function for the environment. Returns the full env_state (state, key)
        '''
        key, pos_key, vel_key, ref_key = jrandom.split(key, 4)
        pos = jrandom.multivariate_normal(pos_key, self.state_mean, self.state_cov)
        # vel = jnp.zeros(3)
        vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3), self.state_cov)

        ref_pos = lax.cond(self.predefined_ref_pos is None, self._sample_random_ref_pos, self._get_predefined_ref_pos, ref_key)

        # ref_pos = lax.cond(self.predefined_ref_pos is None, 
        #                    lambda _: jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov), 
        #                    lambda _: predefined_ref_pos, None)
        ref_vel = jnp.zeros(3) # hard coded to be non moving
        time = 0.0
        new_key = jrandom.split(key)[0]

        new_point_state = PointState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, time=time)

        return new_point_state

    def reset(self, key):
        key, reset_key = jrandom.split(key)
        env_state = self._reset(reset_key)
        return env_state, self._get_obs(env_state)
    
    @property
    def name(self)-> str:
        return "PointParticlePosition"

    @property
    def EnvState(self):
        return PointState
    
    def observation_space(self) -> spaces.Box:
        n_obs = 6 if self.equivariant else 12 # this ONLY works since this is dependent on a constructor arg but this is bad behavior. 
        low = jnp.array(n_obs*[-jnp.finfo(jnp.float32).max])
        high = jnp.array(n_obs*[jnp.finfo(jnp.float32).max])

        return spaces.Box(low, high, (n_obs,), jnp.float32)
    

class PointParticleConstantVelocity(PointParticleBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print("Creating PointParticleConstantVelocity environment with Equivaraint: ", self.equivariant)

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
        ref_acc = env_state.ref_acc
        ref_vel = state.ref_vel + ref_acc * self.dt
        ref_pos = state.ref_pos + ref_vel * self.dt

        # update time
        time = state.time + self.dt

        done = self._is_terminal(env_state)

        env_state = PointVelocityState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time)

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state,action), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if not self.equivariant:
            non_eq_state = jnp.hstack([state.pos, state.vel, state.ref_pos, state.ref_vel, state.ref_acc])
            return non_eq_state
        else:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel, state.ref_acc])
            # eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel, state.ref_vel, state.ref_acc])

            return eq_state

    def _reset(self, key):
        '''
        Reset function for the environment. Returns the full env_state (state, key)
        '''
        key, pos_key, vel_key = jrandom.split(key, 3)
        pos = jrandom.multivariate_normal(pos_key, self.state_mean, self.state_cov)
        # vel = jnp.zeros(3)
        vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3), self.state_cov)

        ref_pos = lax.cond(self.predefined_ref_pos is None, self._sample_random_ref_pos, self._get_predefined_ref_pos, key)

        # ref_pos = lax.cond(self.predefined_ref_pos is None, 
        #                    lambda _: jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov), 
        #                    lambda _: predefined_ref_pos, None)
        key, vel_key = jrandom.split(key)
        ref_vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3), jnp.eye(3) * 0.5)
        ref_acc = jnp.zeros(3)
        time = 0.0
        new_key = jrandom.split(key)[0]

        new_point_state = PointVelocityState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time)

        return new_point_state
    
    def reset(self, key):
        key, reset_key = jrandom.split(key)
        env_state = self._reset(reset_key)
        return env_state, self._get_obs(env_state)
    
    @property
    def name(self)-> str:
        return "PointParticleVelocity"
    
    @property
    def EnvState(self):
        return PointVelocityState
    
    def observation_space(self) -> spaces.Box:
        n_obs = 9 if self.equivariant else 15
        # n_obs = 12 if self.equivariant else 15
        low = jnp.array(n_obs*[-jnp.finfo(jnp.float32).max])
        high = jnp.array(n_obs*[jnp.finfo(jnp.float32).max])

        return spaces.Box(low, high, (n_obs,), jnp.float32)


class PointParticleRandomWalkPosition(PointParticleBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print("Creating PointParticleRandomWalkPosition environment with Equivaraint: ", self.equivariant)

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
        ref_acc = env_state.ref_acc

        _, vel_key = jrandom.split(state.rnd_key)
        ref_vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3,), 5e-2 * jnp.eye(3,))

        ref_pos = state.ref_pos + ref_vel * self.dt

        # update time
        time = state.time + self.dt

        done = self._is_terminal(env_state)

        env_state = PointRandomWalkState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time, rnd_key=vel_key)

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state,action), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if not self.equivariant:
            non_eq_state = jnp.hstack([state.pos, state.vel, state.ref_pos, state.ref_vel, state.ref_acc])
            return non_eq_state
        else:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel, state.ref_acc])
            return eq_state

    def _reset(self, key):
        '''
        Reset function for the environment. Returns the full env_state (state, key)
        '''
        key, pos_key, vel_key = jrandom.split(key, 3)
        pos = jrandom.multivariate_normal(pos_key, self.state_mean, self.state_cov)
        # vel = jnp.zeros(3)
        vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3), self.state_cov)

        ref_pos = lax.cond(self.predefined_ref_pos is None, self._sample_random_ref_pos, self._get_predefined_ref_pos, key)

        # ref_pos = lax.cond(self.predefined_ref_pos is None, 
        #                    lambda _: jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov), 
        #                    lambda _: predefined_ref_pos, None)
        key, vel_key = jrandom.split(key)
        ref_vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3,), jnp.eye(3,) * 0.5)
        ref_acc = jnp.zeros(3,)
        time = 0.0
        new_key = jrandom.split(key)[0]

        new_point_state = PointRandomWalkState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time, rnd_key=new_key)

        return new_point_state
    
    def reset(self, key):
        key, reset_key = jrandom.split(key)
        env_state = self._reset(reset_key)
        return env_state, self._get_obs(env_state)
    
    @property
    def name(self)-> str:
        return "PointParticleRandomWalkPosition"
    
    @property
    def EnvState(self):
        return PointRandomWalkState
    
    def observation_space(self) -> spaces.Box:
        n_obs = 9 if self.equivariant else 15
        low = jnp.array(n_obs*[-jnp.finfo(jnp.float32).max])
        high = jnp.array(n_obs*[jnp.finfo(jnp.float32).max])

        return spaces.Box(low, high, (n_obs,), jnp.float32)


class PointParticleRandomWalkVelocity(PointParticleBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print("Creating PointParticleRandomWalkVelocity environment with Equivaraint: ", self.equivariant)

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
        #ref_acc = env_state.ref_acc

        _, acc_key = jrandom.split(state.rnd_key)
        ref_acc = jrandom.multivariate_normal(acc_key, jnp.zeros(3,), 0.5 * jnp.eye(3,))

        ref_vel = state.ref_vel + ref_acc * self.dt
        ref_pos = state.ref_pos + ref_vel * self.dt

        # update time
        time = state.time + self.dt

        done = self._is_terminal(env_state)

        env_state = PointRandomWalkState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time, rnd_key=acc_key)

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state,action), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if not self.equivariant:
            non_eq_state = jnp.hstack([state.pos, state.vel, state.ref_pos, state.ref_vel, state.ref_acc])
            return non_eq_state
        else:
            # eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel, state.ref_acc])
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel, state.ref_vel, state.ref_acc])
            return eq_state

    def _reset(self, key):
        '''
        Reset function for the environment. Returns the full env_state (state, key)
        '''
        key, pos_key, vel_key = jrandom.split(key, 3)
        pos = jrandom.multivariate_normal(pos_key, self.state_mean, self.state_cov)
        # vel = jnp.zeros(3)
        vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3), self.state_cov)

        ref_pos = lax.cond(self.predefined_ref_pos is None, self._sample_random_ref_pos, self._get_predefined_ref_pos, key)

        # ref_pos = lax.cond(self.predefined_ref_pos is None, 
        #                    lambda _: jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov), 
        #                    lambda _: predefined_ref_pos, None)
        key, vel_key, acc_key = jrandom.split(key, 3)
        ref_vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3,), jnp.eye(3,) * 0.5)
        ref_acc = jrandom.multivariate_normal(acc_key, jnp.zeros(3,), jnp.eye(3,) * 0.5)
        time = 0.0
        new_key = jrandom.split(key)[0]

        new_point_state = PointRandomWalkState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time, rnd_key=new_key)

        return new_point_state
    
    def reset(self, key):
        key, reset_key = jrandom.split(key)
        env_state = self._reset(reset_key)
        return env_state, self._get_obs(env_state)
    
    @property
    def name(self)-> str:
        return "PointParticleRandomWalkVelocity"
    
    @property
    def EnvState(self):
        return PointRandomWalkState
    
    def observation_space(self) -> spaces.Box:
        # n_obs = 9 if self.equivariant else 15
        n_obs = 12 if self.equivariant else 15
        low = jnp.array(n_obs*[-jnp.finfo(jnp.float32).max])
        high = jnp.array(n_obs*[jnp.finfo(jnp.float32).max])

        return spaces.Box(low, high, (n_obs,), jnp.float32)
    


class PointParticleRandomWalkAccel(PointParticleBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print("!!!!!!!!!!!!!!!EXPERIMENTAL!!!!!!!!!!!! Under progress......")
        print("Creating PointParticleRandomWalkAccel environment with Equivaraint: ", self.equivariant)

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
        #ref_acc = env_state.ref_acc

        _, jerk_key = jrandom.split(state.rnd_key)
        ref_jerk = jrandom.multivariate_normal(jerk_key, jnp.zeros(3,), 1e-7 * jnp.eye(3))

        ref_acc = state.ref_acc + ref_jerk * self.dt
        ref_vel = state.ref_vel + ref_acc * self.dt
        ref_pos = state.ref_pos + ref_vel * self.dt

        #jax.debug.print("ref_pos: {ref_pos}, ref_vel: {ref_vel}, ref_acc: {ref_acc}, ref_jerk: {ref_jerk}", ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, ref_jerk=ref_jerk)

        # update time
        time = state.time + self.dt

        # Increase termination bound, as trajectories can extend largely
        done = self._is_terminal(env_state,)

        env_state = PointRandomWalkState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time, rnd_key=jerk_key)

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state,action), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if not self.equivariant:
            non_eq_state = jnp.hstack([state.pos, state.vel, state.ref_pos, state.ref_vel, state.ref_acc])
            return non_eq_state
        else:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel, state.ref_acc])
            return eq_state

    def _reset(self, key):
        '''
        Reset function for the environment. Returns the full env_state (state, key)
        '''
        key, pos_key, vel_key = jrandom.split(key, 3)
        pos = jrandom.multivariate_normal(pos_key, self.state_mean, self.state_cov)
        # vel = jnp.zeros(3)
        vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3), self.state_cov)

        ref_pos = lax.cond(self.predefined_ref_pos is None, self._sample_random_ref_pos, self._get_predefined_ref_pos, key)

        # ref_pos = lax.cond(self.predefined_ref_pos is None, 
        #                    lambda _: jrandom.multivariate_normal(key, self.ref_mean, self.ref_cov), 
        #                    lambda _: predefined_ref_pos, None)
        key, vel_key, acc_key = jrandom.split(key, 3)
        ref_vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3,), jnp.eye(3,) * 0.5)
        ref_acc = jrandom.multivariate_normal(acc_key, jnp.zeros(3,), jnp.eye(3,) * 0.5)
        time = 0.0
        new_key = jrandom.split(key)[0]

        new_point_state = PointRandomWalkState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time, rnd_key=new_key)

        return new_point_state
    
    def reset(self, key):
        key, reset_key = jrandom.split(key)
        env_state = self._reset(reset_key)
        return env_state, self._get_obs(env_state)
    
    @property
    def name(self)-> str:
        return "PointParticleRandomWalkAcceleration"
    
    @property
    def EnvState(self):
        return PointRandomWalkState
    
    def observation_space(self) -> spaces.Box:
        n_obs = 9 if self.equivariant else 15
        low = jnp.array(n_obs*[-jnp.finfo(jnp.float32).max])
        high = jnp.array(n_obs*[jnp.finfo(jnp.float32).max])

        return spaces.Box(low, high, (n_obs,), jnp.float32)
    
@jit
def lissajous_1D(t, amp, freq, phase):
    return amp * jnp.sin(freq * t + phase)

def lissajous_3D(t, amplitudes, frequencies, phases):
    x = amplitudes[0] * jnp.sin(2.0*jnp.pi * frequencies[0] * t + phases[0])
    y = amplitudes[1] * jnp.sin(2.0*jnp.pi * frequencies[1] * t + phases[1])
    z = amplitudes[2] * jnp.sin(2.0*jnp.pi * frequencies[2] * t + phases[2])
    return jnp.stack([x, y, z], axis=1)

class PointParticleLissajousTracking(PointParticleBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print("Creating PointParticleLissajousTracking environment with Equivaraint: ", self.equivariant)
        # self.reference_pos_func = lissajous_1D
        # self.reference_vel_func = jax.jit(jax.grad(lissajous_1D, 0))
        # self.reference_acc_func = jax.jit(jax.grad(jax.grad(lissajous_1D, 0), 0))
        self.ref_pos_fn = jax.jit(lissajous_3D)
        self.ref_vel_fn = jax.jit(jax.jacfwd(lissajous_3D))
        self.ref_acc_fn = jax.jit(jax.jacfwd(jax.jacfwd(lissajous_3D)))

    
    def _reset(self, key):
        '''
        Reset function for the environment. Returns the full env_state (state, key)
        '''
        key, pos_key, vel_key, amp_key, freq_key, phase_key = jrandom.split(key, 6)
        pos = jrandom.multivariate_normal(pos_key, self.state_mean, self.state_cov)
        # vel = jnp.zeros(3)
        vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3), self.state_cov)


        amplitudes = jrandom.uniform(amp_key, (3,), minval=-5., maxval=5.)
        frequencies = jrandom.uniform(freq_key, (3,), minval=0.1, maxval=1.0)
        phases = jrandom.uniform(phase_key, (3,), minval=0., maxval=2.0 * jnp.pi)


        # ref_pos = jnp.hstack([self.reference_pos_func(0.0, amplitudes[0], frequencies[0], phases[0]), self.reference_pos_func(0.0, amplitudes[1], frequencies[1], phases[1]), self.reference_pos_func(0.0, amplitudes[2], frequencies[2], phases[2])])
        # ref_vel = jnp.hstack([self.reference_vel_func(0.0, amplitudes[0], frequencies[0], phases[0]), self.reference_vel_func(0.0, amplitudes[1], frequencies[1], phases[1]), self.reference_vel_func(0.0, amplitudes[2], frequencies[2], phases[2])])
        # ref_acc = jnp.hstack([self.reference_acc_func(0.0, amplitudes[0], frequencies[0], phases[0]), self.reference_acc_func(0.0, amplitudes[1], frequencies[1], phases[1]), self.reference_acc_func(0.0, amplitudes[2], frequencies[2], phases[2])])

        time = 0.0

        ref_pos = self.ref_pos_fn(jnp.array([time]), amplitudes, frequencies, phases).squeeze()
        ref_vel = self.ref_vel_fn(jnp.array([time]), amplitudes, frequencies, phases).squeeze()
        ref_acc = self.ref_acc_fn(jnp.array([time]), amplitudes, frequencies, phases).squeeze()

        new_point_state = PointLissajousTrackingState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time, amplitudes=amplitudes, frequencies=frequencies, phases=phases)

        return new_point_state

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
        #ref_acc = env_state.ref_acc

        # update time
        time = state.time + self.dt

        # ref_pos = jnp.hstack([self.reference_pos_func(time, state.amplitudes[0], state.frequencies[0], state.phases[0]), self.reference_pos_func(time, state.amplitudes[1], state.frequencies[1], state.phases[1]), self.reference_pos_func(time, state.amplitudes[2], state.frequencies[2], state.phases[2])])
        # ref_vel = jnp.hstack([self.reference_vel_func(time, state.amplitudes[0], state.frequencies[0], state.phases[0]), self.reference_vel_func(time, state.amplitudes[1], state.frequencies[1], state.phases[1]), self.reference_vel_func(time, state.amplitudes[2], state.frequencies[2], state.phases[2])])
        # ref_acc = jnp.hstack([self.reference_acc_func(time, state.amplitudes[0], state.frequencies[0], state.phases[0]), self.reference_acc_func(time, state.amplitudes[1], state.frequencies[1], state.phases[1]), self.reference_acc_func(time, state.amplitudes[2], state.frequencies[2], state.phases[2])])

        ref_pos = self.ref_pos_fn(jnp.array([time]), state.amplitudes, state.frequencies, state.phases).squeeze()
        ref_vel = self.ref_vel_fn(jnp.array([time]), state.amplitudes, state.frequencies, state.phases).squeeze()
        ref_acc = self.ref_acc_fn(jnp.array([time]), state.amplitudes, state.frequencies, state.phases).squeeze()

        # Increase termination bound, as trajectories can extend largely
        done = self._is_terminal(env_state,)

        env_state = PointLissajousTrackingState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time, amplitudes=state.amplitudes, frequencies=state.frequencies, phases=state.phases)

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state,action), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}
    
    def reset(self, key):
        key, reset_key = jrandom.split(key)
        env_state = self._reset(reset_key)
        return env_state, self._get_obs(env_state)

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if not self.equivariant:
            non_eq_state = jnp.hstack([state.pos, state.vel, state.ref_pos, state.ref_vel, state.ref_acc])
            return non_eq_state
        else:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel, state.ref_acc])
            return eq_state
    @property
    def name(self)-> str:
        return "PointParticleLissajousTracking"
    
    @property
    def EnvState(self):
        return PointLissajousTrackingState
    
    def observation_space(self) -> spaces.Box:
        n_obs = 9 if self.equivariant else 15
        low = jnp.array(n_obs*[-jnp.finfo(jnp.float32).max])
        high = jnp.array(n_obs*[jnp.finfo(jnp.float32).max])

        return spaces.Box(low, high, (n_obs,), jnp.float32)
        

# Code used for testing the environment
if __name__ == "__main__":
    seed = 2024
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

    # env = PointParticleConstantVelocity()
    # env_state, obs = env.reset(key)
    # print(env_state)
    # print(obs)

    # obs_buffer = []

    # def rollout_body(carry, unused):
    #     key, env_state, obs = carry
    #     action = jrandom.multivariate_normal(key, jnp.zeros(3), jnp.eye(3) * 0.1)
    #     env_state, obs, reward, done, info = env.step(key, env_state, action)
    #     return (key, env_state, obs), obs
    
    # # Rollout for 100 steps
    # _, obs_buffer = lax.scan(rollout_body, (key, env_state, obs), jnp.arange(100))

    # print(obs_buffer.shape) # (100, 4, 3) - 100 steps, 4 observations (pos, vel, ref_pos, ref_vel), 3 dimensions
    # print(obs_buffer[0]) # Initial observation
    # print(obs_buffer[-1]) # Final observation

    env = PointParticleLissajousTracking(equivariant=True)

    env_state, obs = env.reset(key)


    def rollout_body(carry, unused):
        key, env_state, obs = carry
        action = jrandom.multivariate_normal(key, jnp.zeros(3), jnp.eye(3) * 0.1)
        env_state, obs, reward, done, info = env.step(key, env_state, action)
        return (key, env_state, obs), env_state.ref_pos

    _, ref_pos_buffer = lax.scan(rollout_body, (key, env_state, obs), jnp.arange(2000))

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(ref_pos_buffer[:,0], label="x")
    plt.plot(ref_pos_buffer[:,1], label="y")
    plt.plot(ref_pos_buffer[:,2], label="z")
    plt.legend()
    plt.show()

    print(ref_pos_buffer.shape)
