import jax
import jax.numpy as jnp
import orbax.checkpoint as orbax_cp
from jax.scipy.spatial.transform import Rotation

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
    # parser.add_argument("--load-path", type=str, default=None, required=True, help="Path to load model from. ")
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

def model_call(model, params, obs):
    # jax.debug.print("In Model Call")
    # jax.debug.print("obs: {}", obs.shape)
    # print("model call obs shape: ", obs.shape)
    pi, value = model.apply({'params': params}, obs)
    action = pi.mean()
    # action = pi.sample(seed=0)
    return action

def env_step(env, env_state, action, rng):
    return env.step(rng, env_state, action)


def baseline_rollout_step(carry, unused):
    (env_states, model_params, obs, rng) = carry
    # print("baseline rollout step: ", obs.shape)
    actions = model_call(baseline_model, model_params, obs)
    env_states, obs, rewards, dones, infos = env_step(baseline_env, env_states, actions, rng)
    return (env_states, model_params, obs, rng), (env_states, rewards, dones, actions)

def equivariant_rollout_step(carry, unused):
    (env_states, model_params, obs, rng) = carry
    # print("equivariant rollout step: ", obs.shape)
    actions = model_call(equivariant_model, model_params, obs)
    env_states, obs, rewards, dones, infos = env_step(equivariant_env, env_states, actions, rng)
    return (env_states, model_params, obs, rng), (env_states, rewards, dones, actions)

if __name__ == "__main__":
    args = parse_args()
    rng = jax.random.PRNGKey(args.seed)

    # if("model" in args.load_path):
    #     load_path = os.path.abspath(args.load_path)
    #     checkpoint_folder_path = os.path.dirname(load_path)
    # else:
    #     load_path = os.path.join(os.path.abspath(args.load_path), "model_final/")
    #     checkpoint_folder_path = os.path.abspath(args.load_path)



    if args.env not in ["particle", "astrobee", "quadrotor"]:
        raise ValueError("Invalid environment. ")
    elif args.env == "particle":
        env = PointParticleLissajousTracking()
    elif args.env == "astrobee":
        baseline_checkpoint_folder_path = "./checkpoints/astrobee_baseline"
        baseline_load_path = baseline_checkpoint_folder_path + "/model_final/"
        baseline_load_path = os.path.abspath(baseline_load_path)

        baseline_train_config = ast.literal_eval(open(baseline_checkpoint_folder_path+"/config.txt", "r").read())
        baseline_env = SE3QuadFullyActuatedLissajous(equivariant=baseline_train_config["EQUIVARIANT"], terminate_on_error=baseline_train_config["TERMINATE_ON_ERROR"], 
                                            reward_q_pos=baseline_train_config["REWARD_Q_POS"], reward_q_vel=baseline_train_config["REWARD_Q_VEL"], reward_q_rotm=baseline_train_config["REWARD_Q_ROTM"], reward_q_omega=baseline_train_config["REWARD_Q_OMEGA"], 
                                            reward_r=baseline_train_config["REWARD_R"], reward_reach=baseline_train_config["REWARD_REACH"], 
                                            termination_bound=baseline_train_config["TERMINATION_BOUND"], terminal_reward=baseline_train_config["TERMINAL_REWARD"], 
                                            state_cov_scalar=baseline_train_config["STATE_COV_SCALAR"], ref_cov_scalar=baseline_train_config["REF_COV_SCALAR"], 
                                            use_des_action_in_reward=baseline_train_config["USE_DES_ACTION_IN_REWARD"], use_abs_reward_fn=baseline_train_config["USE_ABS_REWARD_FN"], symmetry_type=baseline_train_config["SYMMETRY_TYPE"])
        
        equivariant_checkpoint_folder_path = "./checkpoints/astrobee_equivariant"
        equivariant_load_path = equivariant_checkpoint_folder_path + "/model_final/"
        equivariant_load_path = os.path.abspath(equivariant_load_path)

        equivariant_train_config = ast.literal_eval(open(equivariant_checkpoint_folder_path+"/config.txt", "r").read())
        equivariant_env = SE3QuadFullyActuatedLissajous(equivariant=equivariant_train_config["EQUIVARIANT"], terminate_on_error=equivariant_train_config["TERMINATE_ON_ERROR"], 
                                            reward_q_pos=equivariant_train_config["REWARD_Q_POS"], reward_q_vel=equivariant_train_config["REWARD_Q_VEL"], reward_q_rotm=equivariant_train_config["REWARD_Q_ROTM"], reward_q_omega=equivariant_train_config["REWARD_Q_OMEGA"], 
                                            reward_r=equivariant_train_config["REWARD_R"], reward_reach=equivariant_train_config["REWARD_REACH"], 
                                            termination_bound=equivariant_train_config["TERMINATION_BOUND"], terminal_reward=equivariant_train_config["TERMINAL_REWARD"], 
                                            state_cov_scalar=equivariant_train_config["STATE_COV_SCALAR"], ref_cov_scalar=equivariant_train_config["REF_COV_SCALAR"], 
                                            use_des_action_in_reward=equivariant_train_config["USE_DES_ACTION_IN_REWARD"], use_abs_reward_fn=equivariant_train_config["USE_ABS_REWARD_FN"], symmetry_type=equivariant_train_config["SYMMETRY_TYPE"])
    elif args.env == "quadrotor":
        env = SE2xRQuadLissajous()


    # Load model
    # Define the model
    if args.env == "particle":
        pass
        # model = ActorCritic(action_dim=3, activation=train_config.get("ACTIVATION", "tanh"), num_layers=train_config.get("NUM_LAYERS", 3), num_nodes=train_config.get("NUM_NODES", 64), out_activation=train_config.get("OUT_ACTIVATION", "hard_tanh"))
        # model.init(rng, jnp.zeros((1, 3)))
    elif args.env == "astrobee":
        baseline_model = ActorCritic(action_dim=6, activation=baseline_train_config.get("ACTIVATION", "tanh"), num_layers=baseline_train_config.get("NUM_LAYERS", 3), num_nodes=baseline_train_config.get("NUM_NODES", 64))
        baseline_model.init(rng, jnp.zeros(baseline_env.observation_space().shape))
        # save_path_base = os.path.dirname(load_path)
        print("Loading model from: ", baseline_load_path)
        baseline_model_params = orbax_cp.PyTreeCheckpointer().restore(baseline_load_path)[0]['params']['params']
        baseline_model_params = select_seed_params(baseline_model_params, args.train_seed)

        equivariant_model = ActorCritic(action_dim=6, activation=equivariant_train_config.get("ACTIVATION", "tanh"), num_layers=equivariant_train_config.get("NUM_LAYERS", 3), num_nodes=equivariant_train_config.get("NUM_NODES", 64))
        equivariant_model.init(rng, jnp.zeros(equivariant_env.observation_space().shape))
        print("Loading model from: ", equivariant_load_path)
        equivariant_model_params = orbax_cp.PyTreeCheckpointer().restore(equivariant_load_path)[0]['params']['params']
        equivariant_model_params = select_seed_params(equivariant_model_params, args.train_seed)
    elif args.env == "quadrotor":
        pass
        # model = ActorCritic(action_dim=4, activation=train_config.get("ACTIVATION", "tanh"), num_layers=train_config.get("NUM_LAYERS", 3), num_nodes=train_config.get("NUM_NODES", 64))
        # model.init(rng, jnp.zeros((1, 4)))
    else:
        raise ValueError("Invalid Quadrotor model name")


    # reset
    baseline_env_state, baseline_obs = baseline_env.reset(rng)
    equivariant_env_state, equivariant_obs = equivariant_env.reset(rng)

    baseline_init_carry = (baseline_env_state, baseline_model_params, baseline_obs, rng)
    equivariant_init_carry = (equivariant_env_state, equivariant_model_params, equivariant_obs, rng)

    # print("baseline obs shape: ", baseline_obs.shape)
    # print("equivariant obs shape: ", equivariant_obs.shape)
    # print(jnp.shape(baseline_obs)[-1], jnp.shape(equivariant_obs)[-1])

    # rollout
    baseline_carry, (baseline_env_states, baseline_rewards, baseline_dones, baseline_actions) = jax.lax.scan(baseline_rollout_step, baseline_init_carry, jnp.arange(2000))
    equivariant_carry, (equivariant_env_states, equivariant_rewards, equivariant_dones, equivariant_actions) = jax.lax.scan(equivariant_rollout_step, equivariant_init_carry, jnp.arange(2000))

    baseline_pos = baseline_env_states.pos
    baseline_rotm = Rotation.from_matrix(baseline_env_states.rotm).as_quat()
    equivariant_pos = equivariant_env_states.pos
    equivariant_rotm = Rotation.from_matrix(equivariant_env_states.rotm).as_quat()
    ref_pos = equivariant_env_states.ref_pos
    ref_rotm = Rotation.from_matrix(equivariant_env_states.ref_rotm).as_quat()

    save_path = "./ros2_ws/src/trajectory_visualization/trajectory_visualization/data/astrobee/"
    np.save(save_path+"baseline_pos.npy", np.asarray(baseline_pos))
    np.save(save_path+"baseline_rotm.npy", np.asarray(baseline_rotm))
    np.save(save_path+"equivariant_pos.npy", np.asarray(equivariant_pos))
    np.save(save_path+"equivariant_rotm.npy", np.asarray(equivariant_rotm))
    np.save(save_path+"ref_pos.npy", np.asarray(ref_pos))
    np.save(save_path+"ref_rotm.npy", np.asarray(ref_rotm))

