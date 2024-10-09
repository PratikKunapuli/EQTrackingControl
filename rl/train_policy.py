import jax
import jax.numpy as jnp

import optax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.training import orbax_utils, checkpoints

import orbax.checkpoint as orbax_cp


from typing import Sequence, NamedTuple, Any, Tuple, Union, Optional
import numpy as np
import os

# from base_envs import PointState, PointVelocityState, EnvState
from envs.particle_envs import PointParticlePosition, PointParticleConstantVelocity, PointParticleRandomWalkPosition, PointParticleRandomWalkVelocity, PointParticleRandomWalkAccel
from rl.models import ActorCritic
from envs.wrappers import LogWrapper



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
    print("Terminate on error? : ", config["TERMINATE_ON_ERROR"])
    if config["env_name"] == "position":
        env = PointParticlePosition(equivariant=config["EQUIVARIANT"], terminate_on_error=config["TERMINATE_ON_ERROR"], reward_q=config["REWARD_Q"], reward_r=config["REWARD_R"], reward_reach=config["REWARD_REACH"], 
                                    termination_bound=config["TERMINATION_BOUND"], terminal_reward=config["TERMINAL_REWARD"], state_cov_scalar=config["STATE_COV_SCALAR"], ref_cov_scalar=config["REF_COV_SCALAR"])
    elif config["env_name"] == "constant_velocity":
        env = PointParticleConstantVelocity(equivariant=config["EQUIVARIANT"], terminate_on_error=config["TERMINATE_ON_ERROR"], reward_q=config["REWARD_Q"], reward_r=config["REWARD_R"], reward_reach=config["REWARD_REACH"],
                                           termination_bound=config["TERMINATION_BOUND"], terminal_reward=config["TERMINAL_REWARD"], state_cov_scalar=config["STATE_COV_SCALAR"], ref_cov_scalar=config["REF_COV_SCALAR"])
    elif config["env_name"] == "random_walk_position":
        env = PointParticleRandomWalkPosition(equivariant=config["EQUIVARIANT"], terminate_on_error=config["TERMINATE_ON_ERROR"], reward_q=config["REWARD_Q"], reward_r=config["REWARD_R"], reward_reach=config["REWARD_REACH"],
                                           termination_bound=config["TERMINATION_BOUND"], terminal_reward=config["TERMINAL_REWARD"], state_cov_scalar=config["STATE_COV_SCALAR"], ref_cov_scalar=config["REF_COV_SCALAR"])
    elif config["env_name"] == "random_walk_velocity":
        env = PointParticleRandomWalkVelocity(equivariant=config["EQUIVARIANT"], terminate_on_error=config["TERMINATE_ON_ERROR"], reward_q=config["REWARD_Q"], reward_r=config["REWARD_R"], reward_reach=config["REWARD_REACH"],
                                           termination_bound=config["TERMINATION_BOUND"], terminal_reward=config["TERMINAL_REWARD"], state_cov_scalar=config["STATE_COV_SCALAR"], ref_cov_scalar=config["REF_COV_SCALAR"])
    elif config["env_name"] == "random_walk_accel":
        env = PointParticleRandomWalkAccel(equivariant=config["EQUIVARIANT"], terminate_on_error=config["TERMINATE_ON_ERROR"], reward_q=config["REWARD_Q"], reward_r=config["REWARD_R"], reward_reach=config["REWARD_REACH"],
                                           termination_bound=config["TERMINATION_BOUND"], terminal_reward=config["TERMINAL_REWARD"], state_cov_scalar=config["STATE_COV_SCALAR"], ref_cov_scalar=config["REF_COV_SCALAR"])
    else:
        raise ValueError("Invalid environment name")
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
        network = ActorCritic(env.action_space().shape[0], activation=config["ACTIVATION"], num_layers=config["NUM_LAYERS"], num_nodes=config["NUM_NODES"], out_activation=config["OUT_ACTIVATION"])

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
    
    # Env specific arguments
    parser.add_argument("--env-name", type=str, required=True, help="Name of the environment: position (PointParticlePosition), constant_velocity (PointParticleConstantVelocity), random_walk_position (PointParticleRandomWalkPosition), random_walk_velocity (PointParticleRandomWalkVelocity), random_walk_accel (PointParticleRandomWalkAccel)")
    parser.add_argument("--seed", type=int, default=0, help="Seed to use for the evaluation")
    parser.add_argument("--debug", default=False, dest="DEBUG", action="store_true", help="Print debug information")
    parser.add_argument('--no-debug', dest='DEBUG', action='store_false', help="Do not print debug information")
    parser.add_argument("--equivariant", default=False, action='store_true', dest="EQUIVARIANT", help="Whether to use the equivariant version of the environment")
    parser.add_argument("--exp-name", type=str, dest="EXP_NAME", required=True, help="Name of the experiment")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds to train on")
    parser.add_argument("--terminate-on-error", default=True, dest="TERMINATE_ON_ERROR", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), help="Whether to terminate the episode on error")
    parser.add_argument("--termination-bound", type=float, dest="TERMINATION_BOUND" ,default=10.0, help="Bound for termination")
    parser.add_argument("--reward_q", type=float, default=1e-2,dest="REWARD_Q", help="Q value for reward. Positive. ")
    parser.add_argument("--reward_r", type=float, default=1e-4, dest="REWARD_R", help="R value for reward. Positive. ")
    parser.add_argument("--reward_reach", type=float, default=0.1, dest="REWARD_REACH", help="Reward for reaching the hover point")
    parser.add_argument("--terminal-reward", type=float, default=0.0, dest="TERMINAL_REWARD", help="Reward for terminal state, only when error is exceeded")
    parser.add_argument("--state_cov_scalar", type=float, default=0.5, dest="STATE_COV_SCALAR", help="State covariance scalar for initial conditions")
    parser.add_argument("--ref_cov_scalar", type=float, default=3.0, dest="REF_COV_SCALAR", help="Reference covariance scalar for initial conditions")

    # Model specific arguments
    parser.add_argument("--num-layers", type=int, dest="NUM_LAYERS", default=3, help="Number of layers in the network")
    parser.add_argument("--num-nodes", type=int, dest="NUM_NODES", default=64, help="Number of nodes in each layer")
    
    # PPO specific arguments
    parser.add_argument("--lr", type=float, dest="LR", default=3e-4, help="Learning rate")
    parser.add_argument("--num-envs", type=int, dest="NUM_ENVS", default=16, help="Number of parallel environments")
    parser.add_argument("--num-steps", type=int, dest="NUM_STEPS", default=512, help="Number of steps per environment")
    parser.add_argument("--total-timesteps", type=float, dest="TOTAL_TIMESTEPS", default=10e6, help="Total number of timesteps to train for")
    parser.add_argument("--update-epochs", type=int, dest="UPDATE_EPOCHS", default=5, help="Number of epochs to update the policy")
    parser.add_argument("--num-minibatches", type=int, dest="NUM_MINIBATCHES", default=4, help="Number of minibatches to split the data into")
    parser.add_argument("--gamma", type=float, dest="GAMMA", default=0.99, help="Discount factor")
    parser.add_argument("--lambda", type=float, dest="LAMBDA", default=0.95, help="GAE lambda")
    parser.add_argument("--clip-range", type=float, dest="CLIP_RANGE", default=0.2, help="PPO clip range")
    parser.add_argument("--ent-coef", type=float, dest="ENT_COEF", default=0.0, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, dest="VF_COEF", default=0.5, help="Value function coefficient")
    parser.add_argument("--max-grad-norm", type=float, dest="MAX_GRAD_NORM", default=0.5, help="Maximum gradient norm")
    parser.add_argument("--activation", type=str, dest="ACTIVATION", default="leaky_relu", help="Activation function to use")
    parser.add_argument("--out-activation", type=str, dest="OUT_ACTIVATION", default="hard_tanh", help="Activation function for actor network")
    parser.add_argument("--anneal-lr", dest="ANNEAL_LR", action="store_true", help="Anneal the learning rate")
    parser.add_argument('--no-anneal-lr', dest='ANNEAL_LR', action='store_false', help="Do not anneal the learning rate")
    parser.add_argument("--add-desc", default="", help="Additional description about experiment to type in config file.")

    # Parse config file, after above, else, user given params get overriden
    parser.set_defaults(**config) # allows the config to remain the same

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
        "ACTIVATION": "leaky_relu",
        "OUT_ACTIVATION": "hard_tanh",
        "ANNEAL_LR": True,
        "DEBUG": False,
        "REWARD_Q": 1e-2,
        "REWARD_R": 1e-4,
        "REWARD_REACH": 0.1,
        "TERMINAL_REWARD": -25.,
    }

    config = parse_args(config)
    config = vars(config)

    print("Running with Equivariant: ", config['EQUIVARIANT'])

    print("\n\n\n")
    for k, v in config.items():
        print(f"{k}: {v}")
    print("\n\n\n")

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
    # import code; code.interact(local=locals())

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
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
    # mean_terminal_timesteps = np.zeros((5, 610))
    # std_terminal_timesteps = np.zeros((5, 610))
    # x_vals_ts = np.arange(610) * config['NUM_ENVS'] * config['NUM_STEPS']


    # I want to save the curves that I plot here so that later I can plot them on the same curve
    # I will save the mean and std of the terminal timesteps
    save_name = checkpoint_path + "/training_data.npz"
    np.savez(save_name, mean_rewards=mean_rewards, std_rewards=std_rewards, mean_terminal_timesteps=mean_terminal_timesteps.mean(axis=0), std_terminal_timesteps=std_terminal_timesteps.mean(axis=0), x_vals_rew=x_vals_reward, x_vals_ts=x_vals_ts)




