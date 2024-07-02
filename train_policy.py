import jax
import jax.numpy as jnp

import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax import struct
from flax.training import orbax_utils, checkpoints

import orbax.checkpoint as orbax_cp

import chex

from typing import Sequence, NamedTuple, Any, Tuple, Union, Optional
import numpy as np
from functools import partial
import os

from jax_envs import PointParticlePosition, PointState
from models import ActorCritic

@struct.dataclass
class LogEnvState:
    env_state: PointState
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )


    # Create environment
    env = PointParticlePosition(equivariant=config["EQUIVARIANT"])
    env = LogWrapper(env)
    
    # make checkpointer
    checkpointer = orbax_cp.PyTreeCheckpointer()



    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac
    
    def train(rng):
        # Initialize network
        network = ActorCritic(env.action_space().shape[0], activation=config["ACTIVATION"])

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space().shape)
        network_params = network.init(_rng, init_x)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # Initialize Env
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        env_states, obs = jax.vmap(env.reset)(reset_rng) # vmapped for parallel environments
        iteration = 0

        # Trainig Loop
        def _update_step(runner_state, unused):
            # Collect rollouts
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, iteration, rng = runner_state

                # get action from policy
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # step env
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                next_env_states, next_obs, reward, done, info = jax.vmap(env.step)(rng_step, env_state, action)
                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                runner_state = (train_state, next_env_states, next_obs, iteration, rng)

                return runner_state, transition

            
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, length=config["NUM_STEPS"])

            # Calculate advantage
            train_state, env_states, obs, iteration, rng = runner_state
            _, last_value = network.apply(train_state.params, obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (transition.done, transition.value, transition.reward)
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["LAMBDA"] * (1 - done) * gae

                    return (gae, value), gae
                
                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16) # UNSURE ABOUT UNROLL
                return advantages, advantages + traj_batch.value
            
            advantages, targets = _calculate_gae(traj_batch, last_value)

            # Update the network
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # re-run the network
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # calculate the value loss
                        value_pred_clipped = traj_batch.value + jnp.clip(value - traj_batch.value, -config["CLIP_RANGE"], config["CLIP_RANGE"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.mean(jnp.maximum(value_losses, value_losses_clipped))

                        # calculate the policy loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1 - config["CLIP_RANGE"], 1 + config["CLIP_RANGE"]) * gae
                        loss_actor = jnp.mean(-jnp.minimum(loss_actor1, loss_actor2))

                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy

                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss
                
                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]), "batch size must be equal to number of steps * number of envs"
                
                perm = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch) # flatten the batch?
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch) # shuffle the batch
                minibatches = jax.tree_util.tree_map(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),shuffled_batch)

                train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                
                return update_state, total_loss
            
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, losses = jax.lax.scan(_update_epoch, update_state, None, length=config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            # Find the timestep within metric where it mod config['SAVE_FREQUENCY'] is zero
            # This is the timestep where we want to save the model
            # save_timestep = metric['timestep'][jnp.argmin(jnp.mod(metric['timestep'], config['SAVE_FREQUENCY']))]
            # # Save the model
            # checkpointer.save(os.path.join(checkpoint_path, f"model_{save_timestep}.ckpt"), )
            # save_args = orbax_utils.save_args_from_target(train_state)
            # checkpointer.save(os.path.join(checkpoint_path, f"model_{save_timestep}"), train_state, save_args=save_args)
            # train_step = train_state.step.astype(int)
            # print(train_step)
            # global_step = train_step / (config['UPDATE_EPOCHS'] * config['NUM_MINIBATCHES']) * config['NUM_ENVS'] * config['NUM_STEPS']
            # save_path = os.path.abspath(os.path.join(checkpoint_path, f"model_{global_step:.0f}"))
            # checkpointer.save(save_path, train_state)


            if config.get("DEBUG"):
                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episode return={return_values[t]}")
                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_states, obs, iteration+1, rng)
            return runner_state, metric
        
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_states, obs, iteration, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, length=config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metrics}
    
    return train

def parse_args(config):
    import argparse
    parser = argparse.ArgumentParser(description="Train PPO on PointParticlePosition")
    parser.set_defaults(**config) # allows the config to remain the same 
    parser.add_argument("--seed", type=int, default=0, help="Seed to use for the evaluation")
    parser.add_argument("--debug", dest="DEBUG", action="store_true", help="Print debug information")
    parser.add_argument('--no-debug', dest='DEBUG', action='store_false', help="Do not print debug information")
    parser.add_argument("--equivariant", dest="EQUIVARIANT", required=True, help="Whether to use the equivariant version of the environment")
    parser.add_argument("--exp-name", type=str, dest="EXP_NAME", required=True, help="Name of the experiment")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds to train on")
    return parser.parse_args()
if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    # Default PPO config
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 16,
        "NUM_STEPS": 512,
        "TOTAL_TIMESTEPS": 10e6,# 10e6
        "UPDATE_EPOCHS": 5,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "LAMBDA": 0.95,
        "CLIP_RANGE": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ANNEAL_LR": True,
        "DEBUG": False,
    }

    config = parse_args(config)
    config = vars(config)

    rng = jax.random.PRNGKey(config['seed'])


    # single seed training
    # train_jit = jax.jit(make_train(config))
    # t0 = time.time()
    # out = jax.block_until_ready(train_jit(rng)) # training on single seed
    # t1 = time.time()
    # print(f"\n\nTime taken: {t1-t0:.4f} seconds\n\n")

    # Parallel seed training
    rngs = jax.random.split(rng, config['num_seeds'])
    train_vjit = jax.jit(jax.vmap(make_train(config)))
    t0 = time.time()
    out = jax.block_until_ready(train_vjit(rngs)) # training on multiple seeds
    t1 = time.time()
    print(f"\n\nTime taken: {t1-t0:.4f} seconds\n\n")

    # # Save the model
    checkpoint_path = "./checkpoints/" + config['EXP_NAME']
    # if the checkpoint path exists, make a new folder with an appended number
    i = 1
    while os.path.exists(checkpoint_path):
        checkpoint_path = "./checkpoints/" + config['EXP_NAME'] + f"_{i}"
        i += 1
    os.makedirs(checkpoint_path)
    
    # Save config
    with open(os.path.join(checkpoint_path, "config.txt"), "w") as f:
        f.write(str(config))
    # save_args = orbax_utils.save_args_from_target(out['runner_state'])
    save_path = os.path.abspath(os.path.join(checkpoint_path, f"model_final"))
    orbax_cp.PyTreeCheckpointer().save(save_path, out['runner_state'])
    # import code; code.interact(local=locals())

    # Make plots where the graph is the mean and shaded error across the multiple seeds
    metrics = out['metrics']
    metrics_shape = metrics['returned_episode_returns'].shape
    reward_agg = np.mean(metrics['returned_episode_returns'], axis=(2,3))
    reward_agg = reward_agg[:, 1:]
    mean_rewards = np.mean(reward_agg, axis=0)
    std_rewards = np.std(reward_agg, axis=0)

    plt.figure()
    x_vals_reward = np.arange(len(mean_rewards)) * config['NUM_ENVS'] * config['NUM_STEPS']
    plt.plot(x_vals_reward, mean_rewards)
    plt.fill_between(x_vals_reward, mean_rewards-std_rewards, mean_rewards+std_rewards, alpha=0.5)
    plt.title(f"Returns for PPO Jax {config['NUM_ENVS']} Envs, {config['NUM_STEPS']} Steps, {config['TOTAL_TIMESTEPS']:.2E} Timesteps\n Equivariant representation: {config['EQUIVARIANT']}")    
    plt.xlabel("Env Steps")
    plt.ylabel("Mean Episode Return")
    plt.grid(True)
    plt.savefig(checkpoint_path+"/episode_returns_shaded.png", dpi=1000)

    # Find when episodes returned, what their timesteps were
    terminal_timesteps = metrics['returned_episode_lengths'] * metrics['returned_episode']

    # Since we're interested in non-zero values (where episodes actually ended),
    # we can replace zeros with NaN for easier computation of mean and std without considering them
    terminal_timesteps = np.where(terminal_timesteps > 0, terminal_timesteps, np.nan)

    # Step 2: Aggregate Summary Information
    # Compute mean and std of terminal timesteps, ignoring NaN values
    mean_terminal_timesteps = np.nanmean(terminal_timesteps, axis=(2, 3))  # Mean across env_steps and n_envs
    std_terminal_timesteps = np.nanstd(terminal_timesteps, axis=(2, 3))  # Std across env_steps and n_envs


    plt.figure()
    x_vals_ts = np.arange(mean_terminal_timesteps.shape[1]) * config['NUM_ENVS'] * config['NUM_STEPS']
    plt.plot(x_vals_ts, mean_terminal_timesteps.mean(axis=0), label="Mean Terminal Timestep")
    plt.fill_between(x_vals_ts, 
                    mean_terminal_timesteps.mean(axis=0) - std_terminal_timesteps.mean(axis=0), 
                    mean_terminal_timesteps.mean(axis=0) + std_terminal_timesteps.mean(axis=0), 
                    alpha=0.5, label="Std Dev")

    plt.title(f"Terminal Timesteps PPO Jax {config['NUM_ENVS']} Envs, {config['NUM_STEPS']} Steps, {config['TOTAL_TIMESTEPS']:.2E} Timesteps\n Equivariant representation: {config['EQUIVARIANT']}")
    plt.xlabel("Env Steps")
    plt.ylabel("Mean Terminal Timestep")
    plt.legend()
    plt.grid(True)
    plt.savefig(checkpoint_path+"/terminal_timesteps_summary.png", dpi=1000)

    # I want to save the curves that I plot here so that later I can plot them on the same curve
    # I will save the mean and std of the terminal timesteps
    save_name = checkpoint_path + "/training_data.npz"
    np.savez(save_name, mean_rewards=mean_rewards, std_rewards=std_rewards, mean_terminal_timesteps=mean_terminal_timesteps.mean(axis=0), std_terminal_timesteps=std_terminal_timesteps.mean(axis=0), x_vals_rew=x_vals_reward, x_vals_ts=x_vals_ts)




