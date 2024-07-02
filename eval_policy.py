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
from jax_envs import PointState, PointParticlePosition
import argparse

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
    # action = pi.mean()
    action = pi.sample(seed=0)
    return action

def env_step(env, env_state, action, rng):
    return env.step(rng, env_state, action)


def rollout_step(carry, unused):
    (env_states, model_params, obs, rng) = carry
    actions = jax.vmap(model_call, in_axes=(None, None, 0))(model, model_params, obs)
    env_states, obs, rewards, dones, infos = jax.vmap(env_step, in_axes=(None, 0, 0, 0))(env, env_states, actions, rng)
    return (env_states, model_params, obs, rng), (env_states, rewards, dones)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a policy")
    parser.add_argument("--load-path", type=str, required=True, help="Path to the checkpoint to load")
    parser.add_argument("--num-envs", type=int, default=5, help="Number of environments to evaluate in parallel")
    parser.add_argument("--equivariant", action="store_true", help="Whether to use the equivariant version of the environment")
    parser.add_argument("--seed", type=int, default=0, help="Seed to use for the evaluation")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    print("Running Equivariant?: ", args.equivariant)

    rng = jax.random.PRNGKey(0)

    # Define the model
    model = ActorCritic(action_dim=3)
    model.init(rng, jnp.zeros((1, 3)))

    # load_path = os.path.abspath("./checkpoints/checkpoint_test_20/model_final/")
    # load_path = os.path.abspath("./checkpoints/ppo_jax_3_layer_no_eq/model_final/")
    # load_path = os.path.abspath("./checkpoints/ppo_jax_3_layer_eq/model_final/")
    load_path = os.path.abspath(args.load_path)

    save_path_base = os.path.dirname(load_path)
    model_params = orbax_cp.PyTreeCheckpointer().restore(load_path)[0]['params']['params']
    model_params = select_seed_params(model_params)



    env = PointParticlePosition(equivariant=args.equivariant)
    env_rng = jax.random.split(rng, args.num_envs)
    env_states, obs = jax.vmap(env.reset)(env_rng)
    done = False

    init_carry = (env_states, model_params, obs, env_rng)
    carry, (env_states, rewards, dones) = jax.lax.scan(rollout_step, init_carry, None, length=5000)


    pos = env_states.pos # This is of shape (100, 5, 3)
    ref_pos = env_states.ref_pos # This is of shape (100, 5, 3)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

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
        ax.scatter(ref_pos[0, i, 0], ref_pos[0, i, 1], ref_pos[0, i, 2], label="Reference Position", marker="*", color=color)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.legend()
    plt.title(f"Particle Position Rollout for {args.num_envs} Environments")
    plt.tight_layout()
    plt.savefig(save_path_base+"/particle_position.png", dpi=1000)


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
    plt.tight_layout()
    plt.savefig(save_path_base+"/rewards.png", dpi=1000)
