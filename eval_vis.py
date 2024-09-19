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
from envs.base_envs import EnvState, PointState
from envs.particle_envs import PointParticleLissajousTracking
from envs.astrobee_envs import SE3QuadFullyActuatedLissajous
from envs.quadrotor_envs import SE2xRQuadLissajous
import argparse
import ast

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default=None, required=True, help="Environment to visualize. Options: \{particle, astrobee, quadrotor\}. ")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for environment. ")
    parser.add_argument("--load-path", type=str, default=None, required=True, help="Path to load model from. ")
    parser.add_argument("--train-seed", type=int, default=0, help="Model seed index from training to load. Deffault = 0.")

    return parser.parse_args()

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

if __name__ == "__main__":
    args = parse_args()
    rng = jax.random.PRNGKey(args.seed)

    if("model" in args.load_path):
        load_path = os.path.abspath(args.load_path)
        checkpoint_folder_path = os.path.dirname(load_path)
    else:
        load_path = os.path.join(os.path.abspath(args.load_path), "model_final/")
        checkpoint_folder_path = os.path.abspath(args.load_path)

    train_config = ast.literal_eval(open(checkpoint_folder_path+"/config.txt", "r").read())

    if args.env not in ["particle", "astrobee", "quadrotor"]:
        raise ValueError("Invalid environment. ")
    elif args.env == "particle":
        env = PointParticleLissajousTracking()
    elif args.env == "astrobee":
        env = SE3QuadFullyActuatedLissajous(equivariant=train_config["EQUIVARIANT"], terminate_on_error=train_config["TERMINATE_ON_ERROR"], 
                                            reward_q_pos=train_config["REWARD_Q_POS"], reward_q_vel=train_config["REWARD_Q_VEL"], reward_q_rotm=train_config["REWARD_Q_ROTM"], reward_q_omega=train_config["REWARD_Q_OMEGA"], 
                                            reward_r=train_config["REWARD_R"], reward_reach=train_config["REWARD_REACH"], 
                                            termination_bound=train_config["TERMINATION_BOUND"], terminal_reward=train_config["TERMINAL_REWARD"], 
                                            state_cov_scalar=train_config["STATE_COV_SCALAR"], ref_cov_scalar=train_config["REF_COV_SCALAR"], 
                                            use_des_action_in_reward=train_config["USE_DES_ACTION_IN_REWARD"], use_abs_reward_fn=train_config["USE_ABS_REWARD_FN"], symmetry_type=train_config["SYMMETRY_TYPE"])
    elif args.env == "quadrotor":
        env = SE2xRQuadLissajous()


    # Load model
    # Define the model
    if args.env == "particle":
        model = ActorCritic(action_dim=3, activation=train_config.get("ACTIVATION", "tanh"), num_layers=train_config.get("NUM_LAYERS", 3), num_nodes=train_config.get("NUM_NODES", 64), out_activation=train_config.get("OUT_ACTIVATION", "hard_tanh"))
        model.init(rng, jnp.zeros((1, 3)))
    elif args.env == "astrobee":
        model = ActorCritic(action_dim=6, activation=train_config.get("ACTIVATION", "tanh"), num_layers=train_config.get("NUM_LAYERS", 3), num_nodes=train_config.get("NUM_NODES", 64))
        model.init(rng, jnp.zeros((1, 6)))
    elif args.env == "quadrotor":
        model = ActorCritic(action_dim=4, activation=train_config.get("ACTIVATION", "tanh"), num_layers=train_config.get("NUM_LAYERS", 3), num_nodes=train_config.get("NUM_NODES", 64))
        model.init(rng, jnp.zeros((1, 4)))
    else:
        raise ValueError("Invalid Quadrotor model name")


    # save_path_base = os.path.dirname(load_path)
    print("Loading model from: ", load_path)
    model_params = orbax_cp.PyTreeCheckpointer().restore(load_path)[0]['params']['params']
    model_params = select_seed_params(model_params, args.train_seed)

    
    import time
    env_state, obs = env.reset(rng)

    step = 0
    while step < 10:
        start = time.time()
        pi, value = model.apply({'params': model_params}, obs)
        action = pi.mean()
        end = time.time()

        print("Time taken: ", end-start)
        print("Action: ", action)

        env_state, obs, reward, done, info = env.step(rng, env_state, action)
        step += 1

