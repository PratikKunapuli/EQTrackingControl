import jax
import jax.numpy as jnp
from jax import random as jrandom
from jax import lax
from jax import jit
from functools import partial


from gymnax.environments import spaces

from envs.base_envs import PointParticleBase, PointState, PointVelocityState, PointRandomWalkState, PointLissajousTrackingState


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
        if self.clip_actions: action = jnp.clip(action, -1., 1.)

        state = env_state
        ref_acc = env_state.ref_acc
        lqr_cmd = state.lqr_cmd
        # update particle position
        vel = state.vel + action * self.dt
        pos = state.pos + state.vel * self.dt

        # update reference position
        ref_vel = state.ref_vel # Hardcoded - no moving reference
        ref_pos = state.ref_pos + state.ref_vel * self.dt

        # update time
        time = state.time + self.dt

        done = self._is_terminal(env_state)

        #jax.debug.print("K = {K}", K=self.K)
        e_vec = jnp.hstack((ref_pos, ref_vel)).reshape(6, 1) - jnp.hstack((pos, vel)).reshape(6, 1)
        #jax.debug.print("e_vec: {e_vec}", e_vec=e_vec)
        lqr_cmd = (self.gamma * self.K @ e_vec).squeeze()
        #lqr_cmd = self.gamma * self.K @ (jnp.vstack((ref_pos, ref_vel)) - jnp.vstack(pos, vel))

        env_state = PointState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, time=time, ref_acc=ref_acc, lqr_cmd=lqr_cmd)

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

        # if not self.equivariant:
        #     non_eq_state = jnp.hstack([state.pos, state.vel, state.ref_pos, state.ref_vel])
        #     return non_eq_state
        # else:
        #     eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel])
        #     #eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel, state.ref_vel])
        #     #eq_state = jnp.hstack([state.pos, state.ref_pos, state.vel - state.ref_vel])
        #     return eq_state

        if self.equivariant == 0:
            non_eq_state = jnp.hstack([state.pos, state.vel, state.ref_pos, state.ref_vel])
            return non_eq_state
        elif self.equivariant == 1:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel, state.ref_vel])
            return eq_state
        elif self.equivariant == 2:
            eq_state = jnp.hstack([state.pos, state.ref_pos, state.vel - state.ref_vel])
            return eq_state
        elif self.equivariant == 3 or self.equivariant == 4:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel])
            return eq_state
        else:
            print("Invalid Equivariance Type!")
            raise NotImplementedError


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
        ref_acc = jnp.zeros(3)
        time = 0.0
        new_key = jrandom.split(key)[0]
        lqr_cmd = jnp.zeros(3)

        new_point_state = PointState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, lqr_cmd=lqr_cmd, ref_acc=ref_acc, time=time)

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

        if self.equivariant == 0: n_obs = 12
        elif self.equivariant == 1 or self.equivariant == 2: n_obs = 9
        elif self.equivariant == 3 or self.equivariant == 4: n_obs = 6
        else: raise NotImplementedError

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
        if self.clip_actions: action = jnp.clip(action, -1., 1.)

        state = env_state
        # update particle position
        vel = state.vel + action * self.dt
        pos = state.pos + state.vel * self.dt

        # update reference position
        ref_acc = state.ref_acc
        ref_vel = state.ref_vel + state.ref_acc * self.dt
        ref_pos = state.ref_pos + state.ref_vel * self.dt

        # update time
        time = state.time + self.dt

        done = self._is_terminal(env_state)

        env_state = PointVelocityState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time)

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state, action, ref_acc), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if self.equivariant == 0:
            non_eq_state = jnp.hstack([state.pos, state.vel, state.ref_pos, state.ref_vel])
            return non_eq_state
        elif self.equivariant == 1:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel, state.ref_vel])
            return eq_state
        elif self.equivariant == 2:
            eq_state = jnp.hstack([state.pos, state.ref_pos, state.vel - state.ref_vel])
            return eq_state
        elif self.equivariant == 3 or self.equivariant == 4:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel])
            return eq_state
        else:
            print("Invalid Equivariance Type!")
            raise NotImplementedError


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

        new_point_state = PointVelocityState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time,)

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
        
        if self.equivariant == 0: n_obs = 12
        elif self.equivariant == 1 or self.equivariant == 2: n_obs = 9
        elif self.equivariant == 3 or self.equivariant == 4: n_obs = 6
        else: raise NotImplementedError

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
        if self.clip_actions: action = jnp.clip(action, -1., 1.)

        state = env_state
        # update particle position
        vel = state.vel + action * self.dt
        pos = state.pos + state.vel * self.dt

        # update reference position
        ref_acc = env_state.ref_acc

        _, vel_key = jrandom.split(state.rnd_key)
        ref_vel = jrandom.multivariate_normal(vel_key, jnp.zeros(3,), 5e-2 * jnp.eye(3,))

        ref_pos = state.ref_pos + state.ref_vel * self.dt

        # update time
        time = state.time + self.dt

        done = self._is_terminal(env_state)

        env_state = PointRandomWalkState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time, rnd_key=vel_key)

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state, action, ref_acc), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if self.equivariant == 0:
            non_eq_state = jnp.hstack([state.pos, state.vel, state.ref_pos, state.ref_vel])
            return non_eq_state
        elif self.equivariant == 1:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel, state.ref_vel])
            return eq_state
        elif self.equivariant == 2:
            eq_state = jnp.hstack([state.pos, state.ref_pos, state.vel - state.ref_vel])
            return eq_state
        elif self.equivariant == 3 or self.equivariant == 4:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel])
            return eq_state
        else:
            print("Invalid Equivariance Type!")
            raise NotImplementedError

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
        
        if self.equivariant == 0: n_obs = 12
        elif self.equivariant == 1 or self.equivariant == 2: n_obs = 9
        elif self.equivariant == 3 or self.equivariant == 4: n_obs = 6
        else: raise NotImplementedError

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
        if self.equivariant == 4 or self.equivariant == 5: action = action + env_state.ref_acc
        if self.clip_actions: action = jnp.clip(action, -1., 1.)

        state = env_state
        # update particle position
        vel = state.vel + action * self.dt
        pos = state.pos + state.vel * self.dt

        # update reference position
        #ref_acc = env_state.ref_acc

        _, acc_key = jrandom.split(state.rnd_key)
        #ref_acc = jrandom.multivariate_normal(acc_key, jnp.zeros(3,), 0.5 * jnp.eye(3,))

        # Truncate reference acceleration to [-1, 1]
        ref_acc = jrandom.truncated_normal(acc_key, lower=-jnp.ones(3,), upper=jnp.ones(3,))

        ref_vel = state.ref_vel + state.ref_acc * self.dt
        ref_pos = state.ref_pos + state.ref_vel * self.dt

        # update time
        time = state.time + self.dt

        done = self._is_terminal(env_state)

        env_state = PointRandomWalkState(pos=pos, vel=vel, ref_pos=ref_pos, ref_vel=ref_vel, ref_acc=ref_acc, time=time, rnd_key=acc_key)

        # Reset the environment if the episode is done
        # new_env_state = self._reset(key)
        env_state = lax.cond(done, self._reset, lambda _: env_state, key)

        # added stop gradient to match gymnax environments 
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state, action, ref_acc), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state
        
        if self.equivariant == 0:
            non_eq_state = jnp.hstack([state.pos, state.vel, state.ref_pos, state.ref_vel, state.ref_acc])
            return non_eq_state
        elif self.equivariant == 1:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel, state.ref_vel, state.ref_acc])
            return eq_state
        elif self.equivariant == 2:
            eq_state = jnp.hstack([state.pos, state.ref_pos, state.vel - state.ref_vel, state.ref_acc])
            return eq_state
        elif self.equivariant == 3:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel, state.ref_acc])
            return eq_state
        elif self.equivariant == 4:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel])
            return eq_state
        else:
            print("Invalid Equivariance Type!")
            raise NotImplementedError

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
        #ref_acc = jrandom.multivariate_normal(acc_key, jnp.zeros(3,), jnp.eye(3,) * 0.5)
        ref_acc = jrandom.truncated_normal(acc_key, lower=-jnp.ones(3,), upper=jnp.ones(3,))
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
        
        if self.equivariant == 0: n_obs = 15
        elif self.equivariant == 1 or self.equivariant == 2: n_obs = 12
        elif self.equivariant == 3 or self.equivariant == 5: n_obs = 9
        elif self.equivariant == 4: n_obs = 6
        else: raise NotImplementedError
        
        #n_obs = 12 if self.equivariant else 15
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
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state, action, ref_acc), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}

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
def lissajous_3D(t, amplitudes, frequencies, phases):
    x = amplitudes[0] * jnp.sin(2.0*jnp.pi * frequencies[0] * t + phases[0])
    y = amplitudes[1] * jnp.sin(2.0*jnp.pi * frequencies[1] * t + phases[1])
    z = amplitudes[2] * jnp.sin(2.0*jnp.pi * frequencies[2] * t + phases[2])
    return jnp.stack([x, y, z], axis=1)

class PointParticleLissajousTracking(PointParticleBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        print("Creating PointParticleLissajousTracking environment with Equivaraint: ", self.equivariant)
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
        frequencies = jrandom.uniform(freq_key, (3,), minval=0.01, maxval=0.05)
        phases = jrandom.uniform(phase_key, (3,), minval=0., maxval=2.0 * jnp.pi)

        # amplitudes = jnp.array([ 1.36172305,  2.68769884, -2.41660407])
        # frequencies = jnp.array([0.01837562, 0.02134107, 0.01045938])
        # phases = jnp.array([0.6374367 , 1.99830445, 0.79015325])

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
        if self.equivariant == 4 or self.equivariant == 5: action = action + env_state.ref_acc
        if self.clip_actions: action = jnp.clip(action, -1., 1.)

        state = env_state
        # update particle position
        vel = state.vel + action * self.dt
        pos = state.pos + state.vel * self.dt

        # update reference position
        #ref_acc = env_state.ref_acc

        # update time
        time = state.time + self.dt

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
        return lax.stop_gradient(env_state), lax.stop_gradient(self._get_obs(env_state)), self._get_reward(env_state, action, ref_acc), jnp.array(done), {"Finished": lax.select(done, 0.0, 1.0)}

    def reset(self, key):
        key, reset_key = jrandom.split(key)
        env_state = self._reset(reset_key)
        return env_state, self._get_obs(env_state)

    def _get_obs(self, env_state):
        '''
        Get observation from the environment state. Remove time from the observation as it is not needed by the agent.
        '''
        state = env_state

        if self.equivariant == 0:
            non_eq_state = jnp.hstack([state.pos, state.vel, state.ref_pos, state.ref_vel, state.ref_acc])
            return non_eq_state
        elif self.equivariant == 1:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel, state.ref_vel, state.ref_acc])
            return eq_state
        elif self.equivariant == 2:
            eq_state = jnp.hstack([state.pos, state.ref_pos, state.vel - state.ref_vel, state.ref_acc])
            return eq_state
        elif self.equivariant == 3:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel, state.ref_acc])
            return eq_state
        elif self.equivariant == 4:
            eq_state = jnp.hstack([state.pos - state.ref_pos, state.vel - state.ref_vel])
            return eq_state
        else:
            print("Invalid Equivariance Type!")
            raise NotImplementedError

    @property
    def name(self)-> str:
        return "PointParticleLissajousTracking"

    @property
    def EnvState(self):
        return PointLissajousTrackingState

    def observation_space(self) -> spaces.Box:
        
        if self.equivariant == 0: n_obs = 15
        elif self.equivariant == 1 or self.equivariant == 2: n_obs = 12
        elif self.equivariant == 3 or self.equivariant == 5: n_obs = 9
        elif self.equivariant == 4: n_obs = 6
        else: raise NotImplementedError

        #n_obs = 12 if self.equivariant else 15
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
