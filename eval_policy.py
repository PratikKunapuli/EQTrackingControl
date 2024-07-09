import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax_cp

import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax import struct
from flax.training import orbax_utils, checkpoints
from flax import traverse_util
from flax.core import freeze

import os

from models import ActorCritic
from base_envs import EnvState, PointState
from particle_envs import PointParticlePosition, PointParticleConstantVelocity
import argparse
import ast

def select_seed_params(params, seed_index=0):
    """
    Selects the parameters for a specific seed from the nested params dictionary.
    
    Args:
        params: The nested dictionary of parameters.
        seed_index: The index of the seed to select the parameters for.
    
    Returns:
        A new dictionary with the parameters for the specified seed.
    """
    new_params = {}
    for key, value in params.items():
        if isinstance(value, dict):
            # Recursively apply the same operation for nested dictionaries
            new_params[key] = select_seed_params(value, seed_index)
        else:
            # Assume value is an array and select the slice for the specified seed
            new_params[key] = value[seed_index]
    return new_params

# Assuming model_params is loaded as shown previously

def model_call(model, params, obs):
    pi, value = model.apply({'params': model_params}, obs)
    action = pi.mean()
    # action = pi.sample(seed=0)
    return action

def env_step(env, env_state, action, rng):
    return env.step(rng, env_state, action)


def rollout_step(carry, unused):
    (env_states, model_params, obs, rng) = carry
    actions = jax.vmap(model_call, in_axes=(None, None, 0))(model, model_params, obs)
    env_states, obs, rewards, dones, infos = jax.vmap(env_step, in_axes=(None, 0, 0, 0))(env, env_states, actions, rng)
    return (env_states, model_params, obs, rng), (env_states, rewards, dones, actions)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a policy")
    parser.add_argument("--load-path", type=str, required=True, help="Path to the checkpoint to load")
    parser.add_argument("--num-envs", type=int, default=5, help="Number of environments to evaluate in parallel")
    parser.add_argument("--equivariant", action="store_true", help="Whether to use the equivariant version of the environment")
    parser.add_argument("--seed", type=int, default=0, help="Seed to use for the evaluation")
    parser.add_argument("--num-timesteps", type=int, default=5000, help="Number of timesteps to run the evaluation for")
    parser.add_argument("--env-name", type=str, required=True, help="Name of the environment to use for evaluation. position (PointParticlePosition), constant_velocity (PointParticleConstantVelocity)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("Running Equivariant?: ", args.equivariant)

    rng = jax.random.PRNGKey(0)

    # Define the model
    model = ActorCritic(action_dim=3)
    model.init(rng, jnp.zeros((1, 3)))
    
    if("model" in args.load_path):
        load_path = os.path.abspath(args.load_path)
        checkpoint_folder_path = os.path.dirname(load_path)
    else:
        load_path = os.path.join(os.path.abspath(args.load_path), "model_final/")
        checkpoint_folder_path = os.path.abspath(args.load_path)

    train_config = ast.literal_eval(open(checkpoint_folder_path+"/config.txt", "r").read())
    load_path = os.path.abspath(args.load_path)


    save_path_base = os.path.dirname(load_path)
    model_params = orbax_cp.PyTreeCheckpointer().restore(load_path)[0]['params']['params']
    model_params = select_seed_params(model_params)



    # Create environment
    if args.env_name == "position":
        env = PointParticlePosition(equivariant=train_config["EQUIVARIANT"], terminate_on_error=train_config["TERMINATE_ON_ERROR"], reward_q=train_config["REWARD_Q"], reward_r=train_config["REWARD_R"], 
                                    termination_bound=train_config["TERMINATION_BOUND"], terminal_reward=train_config["TERMINAL_REWARD"], state_cov_scalar=train_config["STATE_COV_SCALAR"], ref_cov_scalar=train_config["REF_COV_SCALAR"])
    elif args.env_name == "constant_velocity":
        env = PointParticleConstantVelocity(equivariant=train_config["EQUIVARIANT"], terminate_on_error=train_config["TERMINATE_ON_ERROR"], reward_q=train_config["REWARD_Q"], reward_r=train_config["REWARD_R"],
                                           termination_bound=train_config["TERMINATION_BOUND"], terminal_reward=train_config["TERMINAL_REWARD"], state_cov_scalar=train_config["STATE_COV_SCALAR"], ref_cov_scalar=train_config["REF_COV_SCALAR"])
    else:
        raise ValueError("Invalid environment name")
    
    env_rng = jax.random.split(rng, args.num_envs)
    env_states, obs = jax.vmap(env.reset)(env_rng)
    done = False

    init_carry = (env_states, model_params, obs, env_rng)
    carry, (env_states, rewards, dones, actions) = jax.lax.scan(rollout_step, init_carry, None, length=5000)

    # (timesteps for each env, num_envs, data_size)
    pos = env_states.pos # This is of shape (100, 5, 3)
    ref_pos = env_states.ref_pos # This is of shape (100, 5, 3)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.markers import MarkerStyle

    # colors = plt.cm.jet(jnp.linspace(0, 1, args.num_envs))
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(args.num_envs):
        rollout_end = dones.shape[0]
        for t in range(dones.shape[0]):
            if dones[t, i]:
                rollout_end = t
                break
        print(i, rollout_end)
        
        # color = colors[i]
        color = default_colors[i % len(default_colors)]
        ax.scatter(pos[0, i, 0], pos[0, i, 1], pos[0, i, 2], label="Particle Start", marker=".", color=color)
        ax.plot(pos[:rollout_end, i, 0], pos[:rollout_end, i, 1], pos[:rollout_end, i, 2], label="Particle Position", color=color)
        # print("pos start: ", pos[0, i, :])
        # print("pos end: ", pos[rollout_end-1, i, :])
        if args.env_name == "position":
            ax.scatter(ref_pos[0, i, 0], ref_pos[0, i, 1], ref_pos[0, i, 2], label="Reference Position", marker="*", color=color)
        elif args.env_name == "constant_velocity":
            # Plot beginning and end of reference trajectory
            # print("Ref start: ", ref_pos[0, i, :])
            # print("Ref end: ", ref_pos[rollout_end-1, i, :])
            ax.scatter(ref_pos[0, i, 0], ref_pos[0, i, 1], ref_pos[0, i, 2], label="Reference Position Start", marker='d', color=color)
            ax.scatter(ref_pos[rollout_end-1, i, 0], ref_pos[rollout_end-1, i, 1], ref_pos[rollout_end-1, i, 2], label="Reference Position End", marker='*', color=color)
        else:
            raise ValueError("Invalid environment name")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.legend()
    plt.title(f"Particle Position Rollout for {args.num_envs} Environments \n Equivariant Model: {args.equivariant}")
    plt.tight_layout()
    plt.savefig(save_path_base+"/particle_position.png", dpi=1000)
    # plt.show()


    # Plot position and reference position in 3 axis for each env and save as N unique plots
    for i in range(args.num_envs):
        rollout_end = dones.shape[0]
        for t in range(dones.shape[0]):
            if dones[t, i]:
                rollout_end = t
                break
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(jnp.arange(rollout_end), pos[:rollout_end, i, 0], label="Particle Position")
        plt.plot(jnp.arange(rollout_end), ref_pos[:rollout_end, i, 0], label="Reference Position")
        plt.ylabel("X")
        plt.legend(loc="best")
        plt.subplot(3, 1, 2)
        plt.plot(jnp.arange(rollout_end), pos[:rollout_end, i, 1], label="Particle Position")
        plt.plot(jnp.arange(rollout_end), ref_pos[:rollout_end, i, 1], label="Reference Position")
        plt.ylabel("Y")
        plt.legend(loc="best")
        plt.subplot(3, 1, 3)
        plt.plot(jnp.arange(rollout_end), pos[:rollout_end, i, 2], label="Particle Position")
        plt.plot(jnp.arange(rollout_end), ref_pos[:rollout_end, i, 2], label="Reference Position")
        plt.ylabel("Z")
        plt.legend(loc="best")
        plt.xlabel("Timesteps")
        plt.title(f"Particle Position Rollout for Env {i} \n Equivariant Model: {args.equivariant}")
        plt.tight_layout()
        plt.savefig(save_path_base+f"/particle_position_env_{i}.png", dpi=1000)

        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 0], label="Action X")
        plt.legend()
        plt.ylabel("Action X")
        plt.subplot(3,1,2)
        plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 1], label="Action Y")
        plt.ylabel("Action Y")
        plt.legend()
        plt.subplot(3,1,3)
        plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 2], label="Action Z")
        plt.legend()
        plt.xlabel("Timesteps")
        plt.ylabel("Action Z")
        plt.suptitle(f"Action Curves for Env {i} \n Equivariant Model: {args.equivariant}")
        plt.tight_layout()
        plt.savefig(save_path_base+f"/actions_env_{i}.png", dpi=1000)
        # plt.show

    # Make a plot that shows the error between the particle position and the reference position and averages over all envs with mean and std. dev. shown
    errors = jnp.linalg.norm(pos - ref_pos, axis=-1)
    mean_errors = jnp.mean(errors, axis=1)
    std_errors = jnp.std(errors, axis=1)

    plt.figure()
    plt.plot(jnp.arange(rollout_end), mean_errors[:rollout_end], label="Mean Error")
    plt.fill_between(jnp.arange(rollout_end), mean_errors[:rollout_end] - std_errors[:rollout_end], mean_errors[:rollout_end] + std_errors[:rollout_end], alpha=0.5)
    plt.xlabel("Timesteps")
    plt.ylabel("Error")
    plt.legend()
    plt.title(f"Mean Error Between Particle Position and Reference Position \n  {args.num_envs} Seeds averaged, Equivariant Model: {args.equivariant}")
    plt.tight_layout()
    plt.savefig(save_path_base+"/mean_error.png", dpi=1000)
    # plt.show()


    # Plot reward curves for each env
    plt.figure()
    for i in range(args.num_envs):
        rollout_end = dones.shape[0]
        for t in range(dones.shape[0]):
            if dones[t, i]:
                rollout_end = t
                break
        plt.plot(jnp.arange(rollout_end), rewards[:rollout_end, i], label=f"Env {i}")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    plt.title(f"Reward Curves for {args.num_envs} Environments \n Equivariant Model: {args.equivariant}")
    plt.tight_layout()
    plt.savefig(save_path_base+"/rewards.png", dpi=1000)
