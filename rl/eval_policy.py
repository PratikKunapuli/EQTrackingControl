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
from scipy.spatial.transform import Rotation

from rl.models import ActorCritic
from envs.base_envs import EnvState, PointState
import envs.registry as registry
import argparse
import ast

import pickle
import numpy as np

def fill_defaults(config):
    
    if "REWARD_Q_ROTM" not in config:
        if "astrobee" in config["ENV_NAME"]:
            config["REWARD_Q_ROTM"] = 10.0
        elif "quadrotor" in config["ENV_NAME"]:
            config["REWARD_Q_ROTM"] = 5e-2
        else:
            raise ValueError("Invalid environment name, please use astrobee or quadrotor by default")
        
    if "REWARD_Q_OMEGA" not in config:
        if "astrobee" in config["ENV_NAME"]:
            config["REWARD_Q_ROTM"] = 0.6
        elif "quadrotor" in config["ENV_NAME"]:
            config["REWARD_Q_ROTM"] = 1e-4
        else:
            raise ValueError("Invalid environment name, please use astrobee or quadrotor by default")
        
    if "USE_ABS_REWARD_FN" not in config:
        config["USE_ABS_REWARD_FN"] = False
    if "CLIP_ACTIONS" not in config:
        config["CLIP_ACTIONS"] = True
    if "REWARD_FN_TYPE" not in config:
        config["REWARD_FN_TYPE"] = 1
    
    return config

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
    parser.add_argument("--equivariant", type=int, default=0, help="Whether to use the equivariant version of the environment")
    parser.add_argument("--seed", type=int, default=0, help="Seed to use for the evaluation")
    parser.add_argument("--num-timesteps", type=int, default=5000, help="Number of timesteps to run the evaluation for")
    parser.add_argument("--env-name", type=str, required=True, help="Name of the environment to use for evaluation. position (PointParticlePosition), constant_velocity (PointParticleConstantVelocity), random_walk_velocity (PointParticleRandomWalkVelocity), random_walk_accel (PointParticleRandomWalkAccel)")
    parser.add_argument("--make-animation", action="store_true", help="Make 3D animation")
    parser.add_argument("--symmetry_type", type=int, default=0, dest="SYMMETRY_TYPE", help="Decomposition of the SE(3) group with the tangent space:\n  (0) R3xSO(3), Direct product tangent space\n  (1) R3xSO(3), Semi-direct product tangent space\n  (2) SE(3), Direct product tangent space\n  (3) SE(3), Semi-direct product tangent space")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    data_dict = {}
    rng = jax.random.PRNGKey(0)
    
    if("model" in args.load_path):
        load_path = os.path.abspath(args.load_path)
        checkpoint_folder_path = os.path.dirname(load_path)
    else:
        load_path = os.path.join(os.path.abspath(args.load_path), "model_final/")
        checkpoint_folder_path = os.path.abspath(args.load_path)

    train_config = ast.literal_eval(open(checkpoint_folder_path+"/config.txt", "r").read())
    train_config = fill_defaults(train_config)
    args.equivariant = train_config["EQUIVARIANT"]
    load_path = os.path.abspath(args.load_path)

    # print("\n\n\n")
    # for key, value in train_config.items():
    #     print(f"{key}: {value}")
    # print("\n\n\n")

    # Define the model
    if "particle" in args.env_name:
        model = ActorCritic(action_dim=3, activation=train_config.get("ACTIVATION", "tanh"), num_layers=train_config.get("NUM_LAYERS", 3), num_nodes=train_config.get("NUM_NODES", 64))
    elif "astrobee" in args.env_name:
        model = ActorCritic(action_dim=6, activation=train_config.get("ACTIVATION", "tanh"), num_layers=train_config.get("NUM_LAYERS", 3), num_nodes=train_config.get("NUM_NODES", 64))
    elif "quadrotor" in args.env_name:
        model = ActorCritic(action_dim=4, activation=train_config.get("ACTIVATION", "tanh"), num_layers=train_config.get("NUM_LAYERS", 3), num_nodes=train_config.get("NUM_NODES", 64))
    else:
        raise ValueError("Invalid environment name, please use particle, astrobee or quadrotor")
    #model.init(rng, jnp.zeros((1, 3)))


    save_path_base = os.path.dirname(load_path)
    model_params = orbax_cp.PyTreeCheckpointer().restore(load_path)[0]['params']['params']
    model_params = select_seed_params(model_params)

    # Create environment
    env = registry.get_env_by_name(train_config["env_name"], equivariant=train_config["EQUIVARIANT"], terminate_on_error=train_config["TERMINATE_ON_ERROR"], 
                                   reward_q_pos=train_config["REWARD_Q_POS"], reward_q_vel=train_config["REWARD_Q_VEL"], reward_q_rotm=train_config["REWARD_Q_ROTM"], 
                                   reward_q_omega=train_config["REWARD_Q_OMEGA"], reward_r=train_config["REWARD_R"], reward_reach=train_config["REWARD_REACH"], 
                                   termination_bound=train_config["TERMINATION_BOUND"], terminal_reward=train_config["TERMINAL_REWARD"], 
                                   state_cov_scalar=train_config["STATE_COV_SCALAR"], ref_cov_scalar=train_config["REF_COV_SCALAR"], 
                                   use_des_action_in_reward=train_config["USE_DES_ACTION_IN_REWARD"], use_abs_reward_fn=train_config["USE_ABS_REWARD_FN"], 
                                   symmetry_type=train_config["SYMMETRY_TYPE"])
    #env = registry.get_env_by_name(train_config["env_name"], **train_config)
    
    env_rng = jax.random.split(rng, args.num_envs)
    
    if "quadrotor" in args.env_name:
        traj_indices = jnp.arange(0, args.num_envs, 1)[..., None]
        
    env_states, obs = jax.vmap(env.reset)(env_rng)
    done = False

    init_carry = (env_states, model_params, obs, env_rng)
    carry, (env_states, rewards, dones, actions) = jax.lax.scan(rollout_step, init_carry, None, length=5000)

    # (timesteps for each env, num_envs, data_size)
    pos = env_states.pos # This is of shape (100, 5, 3)
    vel = env_states.vel
    ref_pos = env_states.ref_pos # This is of shape (100, 5, 3)
    ref_vel = env_states.ref_vel
    
    if "quadrotor" in args.env_name:
        rotm = env_states.rotm
        omega = env_states.omega
    
        ref_rotm = env_states.ref_rotm
        ref_omega = env_states.ref_omega
        ref_action = env_states.ref_action
        
        T, B, _, __ = rotm.shape

        eul = Rotation.from_matrix(rotm.reshape(T*B, 3, 3)).as_euler("zyx", degrees=True).reshape(T, B, 3)
        ref_eul = Rotation.from_matrix(ref_rotm.reshape(T*B, 3, 3)).as_euler("zyx", degrees=True).reshape(T, B, 3)

        data_dict = (pos, vel, rotm, omega, ref_pos, ref_vel, ref_rotm, ref_omega, ref_action)

    elif "astrobee" in args.env_name:
        rotm = env_states.rotm
        omega = env_states.omega
    
        ref_rotm = env_states.ref_rotm
        ref_omega = env_states.ref_omega
        
        T, B, _, __ = rotm.shape

        eul = Rotation.from_matrix(rotm.reshape(T*B, 3, 3)).as_euler("zyx", degrees=True).reshape(T, B, 3)
        ref_eul = Rotation.from_matrix(ref_rotm.reshape(T*B, 3, 3)).as_euler("zyx", degrees=True).reshape(T, B, 3)
        ref_F = env_states.ref_F
        ref_tau = env_states.ref_tau
        
        data_dict = (pos, vel, rotm, omega, ref_pos, ref_vel, ref_rotm, ref_omega, ref_F, ref_tau)

    elif "particle" in args.env_name:
        data_dict = (pos, vel, ref_pos, ref_vel)
        
    with open(save_path_base + "/eval_data.pkl", "wb") as f:
        pickle.dump(data_dict, f)
    #anim_pos_data = np.array(pos)
    #anim_ref_pos_data = np.array(ref_pos)
    #np.save(save_path_base + "/pos_data.npy", pos)
    #np.save(save_path_base + "/ref_pos_data.npy", ref_pos)

    import matplotlib.pyplot as plt

    #Animate only 300 samples
    # if args.make_animation:

    #     import matplotlib.animation as animation
    #     from matplotlib.pyplot import cm

    #     anim_pos_data = anim_pos_data[:500, ...]
    #     anim_ref_pos_data = anim_ref_pos_data[:500, ...]

    #     fig = plt.figure()

    #     def animate_scatters(frame, anim_pos_data, anim_ref_pos_data, scatter_points):
            
    #         for i in range(anim_pos_data[0].shape[0]):
    #             scatter_points[0][i]._offsets3d = (anim_pos_data[frame][i,0:1], anim_pos_data[frame][i,1:2], anim_pos_data[frame][i,2:])
    #             scatter_points[1][i]._offsets3d = (anim_ref_pos_data[frame][i,0:1], anim_ref_pos_data[frame][i,1:2], anim_ref_pos_data[frame][i,2:])

    #             dyn_axis_limit = [
    #                 (min(-5., anim_pos_data[frame, :, 0].min()), max(5., anim_pos_data[frame, :, 0].max()) ),
    #                 (min(-5., anim_pos_data[frame, :, 1].min()), max(5., anim_pos_data[frame, :, 1].max()) ),
    #                 (min(-5., anim_pos_data[frame, :, 2].min()), max(5., anim_pos_data[frame, :, 2].max()) ),
    #             ]

    #             scatter_points[0][i]._axes.set_xlim(dyn_axis_limit[0][0], dyn_axis_limit[0][1])
    #             scatter_points[0][i]._axes.set_ylim(dyn_axis_limit[1][0], dyn_axis_limit[1][1])
    #             scatter_points[0][i]._axes.set_zlim(dyn_axis_limit[2][0], dyn_axis_limit[2][1])

    #             scatter_points[1][i]._axes.set_xlim(dyn_axis_limit[0][0], dyn_axis_limit[0][1])
    #             scatter_points[1][i]._axes.set_ylim(dyn_axis_limit[1][0], dyn_axis_limit[1][1])
    #             scatter_points[1][i]._axes.set_zlim(dyn_axis_limit[2][0], dyn_axis_limit[2][1])
    #         return scatter_points

    #     print("Animating trajectories.....")

    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection="3d")

    #     color_list = cm.rainbow(np.linspace(0, 1, anim_pos_data[0].shape[0]))

    #     scatter_points = ([ax.scatter(anim_pos_data[0][i, 0:1], anim_pos_data[0][i, 1:2], anim_pos_data[0][i, 2:], color=color, marker=".", linewidth=0.4) for i, color in zip(range(anim_pos_data[0].shape[0]), color_list)],
    #                       [ax.scatter(anim_ref_pos_data[0][i, 0:1], anim_ref_pos_data[0][i, 1:2], anim_ref_pos_data[0][i, 2:], color=color, marker="*", linewidth=0.4) for i, color in zip(range(anim_ref_pos_data[0].shape[0]), color_list)])
        

    #     # Number of iterations
    #     total_frames = len(anim_pos_data)

    #     # Setting the axes properties
    #     #ax.set_xlim3d([-5, 5])
    #     ax.set_xlabel('x-axis')

    #     #ax.set_ylim3d([-5, 5])
    #     ax.set_ylabel('y-axis')

    #     #ax.set_zlim3d([-5, 5])
    #     ax.set_zlabel('z-axis')

    #     ax.set_title('Particle Trajectory Animation')

    #     ani = animation.FuncAnimation(fig, animate_scatters, total_frames, fargs=(anim_pos_data, anim_ref_pos_data, scatter_points),
    #                                     interval=50, blit=False)

    #     ani.save(save_path_base + "/paticle_animation.mp4", codec="libx264", bitrate=-1, fps=50, dpi=600)

    # # colors = plt.cm.jet(jnp.linspace(0, 1, args.num_envs))
    # default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(args.num_envs):
    #     rollout_end = dones.shape[0]
    #     for t in range(dones.shape[0]):
    #         if dones[t, i]:
    #             rollout_end = t
    #             break
    #     print(i, rollout_end)
        
    #     # color = colors[i]
    #     color = default_colors[i % len(default_colors)]
    #     ax.scatter(pos[0, i, 0], pos[0, i, 1], pos[0, i, 2], label="Particle Start", marker=".", color=color)
    #     ax.plot(pos[:rollout_end, i, 0], pos[:rollout_end, i, 1], pos[:rollout_end, i, 2], label="Particle Position", color=color)
    #     # print("pos start: ", pos[0, i, :])
    #     # print("pos end: ", pos[rollout_end-1, i, :])
    #     if args.env_name == "position":
    #         ax.scatter(ref_pos[0, i, 0], ref_pos[0, i, 1], ref_pos[0, i, 2], label="Reference Position", marker="*", color=color)
    #     elif args.env_name == "constant_velocity":
    #         # Plot beginning and end of reference trajectory
    #         # print("Ref start: ", ref_pos[0, i, :])
    #         # print("Ref end: ", ref_pos[rollout_end-1, i, :])
    #         ax.scatter(ref_pos[0, i, 0], ref_pos[0, i, 1], ref_pos[0, i, 2], label="Reference Position Start", marker='d', color=color)
    #         ax.scatter(ref_pos[rollout_end-1, i, 0], ref_pos[rollout_end-1, i, 1], ref_pos[rollout_end-1, i, 2], label="Reference Position End", marker='*', color=color)
    #     elif args.env_name == "random_walk_position":
    #         # Plot beginning and end of reference trajectory
    #         # print("Ref start: ", ref_pos[0, i, :])
    #         # print("Ref end: ", ref_pos[rollout_end-1, i, :])
    #         ax.scatter(ref_pos[0, i, 0], ref_pos[0, i, 1], ref_pos[0, i, 2], label="Reference Position Start", marker='d', color=color)
    #         ax.scatter(ref_pos[rollout_end-1, i, 0], ref_pos[rollout_end-1, i, 1], ref_pos[rollout_end-1, i, 2], label="Reference Position End", marker='*', color=color)
    #     elif args.env_name == "random_walk_velocity":
    #         # Plot beginning and end of reference trajectory
    #         # print("Ref start: ", ref_pos[0, i, :])
    #         # print("Ref end: ", ref_pos[rollout_end-1, i, :])
    #         ax.scatter(ref_pos[0, i, 0], ref_pos[0, i, 1], ref_pos[0, i, 2], label="Reference Position Start", marker='d', color=color)
    #         ax.scatter(ref_pos[rollout_end-1, i, 0], ref_pos[rollout_end-1, i, 1], ref_pos[rollout_end-1, i, 2], label="Reference Position End", marker='*', color=color)
    #     elif args.env_name == "random_walk_accel":
    #         # Plot beginning and end of reference trajectory
    #         # print("Ref start: ", ref_pos[0, i, :])
    #         # print("Ref end: ", ref_pos[rollout_end-1, i, :])
    #         ax.scatter(ref_pos[0, i, 0], ref_pos[0, i, 1], ref_pos[0, i, 2], label="Reference Position Start", marker='d', color=color)
    #         ax.scatter(ref_pos[rollout_end-1, i, 0], ref_pos[rollout_end-1, i, 1], ref_pos[rollout_end-1, i, 2], label="Reference Position End", marker='*', color=color)
    #     else:
    #         raise ValueError("Invalid environment name")
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # # ax.legend()
    # plt.title(f"Particle Position Rollout for {args.num_envs} Environments \n Equivariant Model: {args.equivariant}")
    # plt.tight_layout()
    # plt.savefig(save_path_base+"/particle_position.png", dpi=1000)
    # # plt.show()


    # Plot position and reference position in 3 axis for each env and save as N unique plots
    # for i in range(args.num_envs):
    #     rollout_end = dones.shape[0]
    #     for t in range(dones.shape[0]):
    #         if dones[t, i]:
    #             rollout_end = t
    #             break
    #     fig = plt.figure()
    #     plt.subplot(3, 1, 1)
    #     plt.plot(jnp.arange(rollout_end), pos[:rollout_end, i, 0], label="Particle Position")
    #     plt.plot(jnp.arange(rollout_end), ref_pos[:rollout_end, i, 0], label="Reference Position")
    #     plt.ylabel("X")
    #     plt.legend(loc="best")
    #     plt.subplot(3, 1, 2)
    #     plt.plot(jnp.arange(rollout_end), pos[:rollout_end, i, 1], label="Particle Position")
    #     plt.plot(jnp.arange(rollout_end), ref_pos[:rollout_end, i, 1], label="Reference Position")
    #     plt.ylabel("Y")
    #     plt.legend(loc="best")
    #     plt.subplot(3, 1, 3)
    #     plt.plot(jnp.arange(rollout_end), pos[:rollout_end, i, 2], label="Particle Position")
    #     plt.plot(jnp.arange(rollout_end), ref_pos[:rollout_end, i, 2], label="Reference Position")
    #     plt.ylabel("Z")
    #     plt.legend(loc="best")
    #     plt.xlabel("Timesteps")
    #     plt.title(f"Particle Position Rollout for Env {i} \n Equivariant Model: {args.equivariant}")
    #     plt.tight_layout()
    #     plt.savefig(save_path_base+f"/particle_position_env_{i}.png", dpi=1000)

    #     plt.figure()
    #     plt.subplot(3,1,1)
    #     plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 0], label="Action X")
    #     plt.legend()
    #     plt.ylabel("Action X")
    #     plt.subplot(3,1,2)
    #     plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 1], label="Action Y")
    #     plt.ylabel("Action Y")
    #     plt.legend()
    #     plt.subplot(3,1,3)
    #     plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 2], label="Action Z")
    #     plt.legend()
    #     plt.xlabel("Timesteps")
    #     plt.ylabel("Action Z")
    #     plt.suptitle(f"Action Curves for Env {i} \n Equivariant Model: {args.equivariant}")
    #     plt.tight_layout()
    #     plt.savefig(save_path_base+f"/actions_env_{i}.png", dpi=1000)
        # plt.show()

    PREFIX_NAME = ""
    if "particle" in args.env_name:
        PREFIX_NAME = "Particle"
    elif "astrobee" in args.env_name:
        PREFIX_NAME = "Astrobee"
    elif "quadrotor" in args.env_name:
        PREFIX_NAME = "Quadrotor"
    else:
        raise ValueError("Invalid environment name, please use particle, astrobee or quadrotor")

    for i in range(args.num_envs):
        rollout_end = dones.shape[0]
        for t in range(dones.shape[0]):
            if dones[t, i]:
                rollout_end = t
                break
        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(jnp.arange(rollout_end), pos[:rollout_end, i, 0], label="Position x")
        plt.plot(jnp.arange(rollout_end), ref_pos[:rollout_end, i, 0], label="Reference Position x")
        plt.grid(True)
        plt.ylabel("X (m)")
        plt.legend(loc="best")
        plt.subplot(3, 1, 2)
        plt.plot(jnp.arange(rollout_end), pos[:rollout_end, i, 1], label="Position y")
        plt.plot(jnp.arange(rollout_end), ref_pos[:rollout_end, i, 1], label="Reference Position y")
        plt.grid(True)
        plt.ylabel("Y (m)")
        plt.legend(loc="best")
        plt.subplot(3, 1, 3)
        plt.plot(jnp.arange(rollout_end), pos[:rollout_end, i, 2], label="Position z")
        plt.plot(jnp.arange(rollout_end), ref_pos[:rollout_end, i, 2], label="Reference Position z")
        plt.grid(True)
        plt.ylabel("Z (m)")
        plt.legend(loc="best")
        plt.xlabel("Timesteps")
        plt.suptitle(f"{PREFIX_NAME} Position Rollout for Env {i} \n Equivariant Model: {args.equivariant} ")
        plt.tight_layout()
        plt.savefig(save_path_base+f"/{PREFIX_NAME}_position_env_{i}.png", dpi=1000)
        plt.close()

        fig = plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(jnp.arange(rollout_end), vel[:rollout_end, i, 0], label="Velocity x")
        plt.plot(jnp.arange(rollout_end), ref_vel[:rollout_end, i, 0], label="Reference Velocity x")
        plt.grid(True)
        plt.ylabel("X (m/s)")
        plt.legend(loc="best")
        plt.subplot(3, 1, 2)
        plt.plot(jnp.arange(rollout_end), vel[:rollout_end, i, 1], label="Velocity y")
        plt.plot(jnp.arange(rollout_end), ref_vel[:rollout_end, i, 1], label="Reference Velocity y")
        plt.grid(True)
        plt.ylabel("Y (m/s)")
        plt.legend(loc="best")
        plt.subplot(3, 1, 3)
        plt.plot(jnp.arange(rollout_end), vel[:rollout_end, i, 2], label="Velocity z")
        plt.plot(jnp.arange(rollout_end), ref_vel[:rollout_end, i, 2], label="Reference Velocity z")
        plt.grid(True)
        plt.ylabel("Z (m/s)")
        plt.legend(loc="best")
        plt.xlabel("Timesteps")
        plt.suptitle(f"{PREFIX_NAME} Velocity Rollout for Env {i} \n Equivariant Model: {args.equivariant}")
        plt.tight_layout()
        plt.savefig(save_path_base+f"/{PREFIX_NAME}_velocity_env_{i}.png", dpi=1000)
        plt.close()
            
        if "astrobee" in args.env_name or "quadrotor" in args.env_name:
            
            fig = plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(jnp.arange(rollout_end), eul[:rollout_end, i, 0], label="Yaw")
            plt.plot(jnp.arange(rollout_end), ref_eul[:rollout_end, i, 0], label="Reference Yaw")
            plt.grid(True)
            plt.ylabel("Yaw (deg)")
            plt.legend(loc="best")
            plt.subplot(3, 1, 2)
            plt.plot(jnp.arange(rollout_end), eul[:rollout_end, i, 1], label="Pitch")
            plt.plot(jnp.arange(rollout_end), ref_eul[:rollout_end, i, 1], label="Reference Pitch")
            plt.grid(True)
            plt.ylabel("Pitch (deg)")
            plt.legend(loc="best")
            plt.subplot(3, 1, 3)
            plt.plot(jnp.arange(rollout_end), eul[:rollout_end, i, 2], label="Roll")
            plt.plot(jnp.arange(rollout_end), ref_eul[:rollout_end, i, 2], label="Reference Roll")
            plt.grid(True)
            plt.ylabel("Roll (deg)")
            plt.legend(loc="best")
            plt.xlabel("Timesteps")
            plt.suptitle(f"{PREFIX_NAME} Orientation Rollout for Env {i} \n Equivariant Model: {args.equivariant}")
            plt.tight_layout()
            plt.savefig(save_path_base+f"/{PREFIX_NAME}_euler_env_{i}.png", dpi=1000)
            plt.close()

            fig = plt.figure()
            plt.subplot(3, 1, 1)
            plt.plot(jnp.arange(rollout_end), omega[:rollout_end, i, 0], label="omega x")
            plt.plot(jnp.arange(rollout_end), ref_omega[:rollout_end, i, 0], label="Reference omega x")
            plt.ylabel("Omega x (rad/s)")
            plt.legend(loc="best")
            plt.grid(True)
            plt.subplot(3, 1, 2)
            plt.plot(jnp.arange(rollout_end), omega[:rollout_end, i, 1], label="omega y")
            plt.plot(jnp.arange(rollout_end), ref_omega[:rollout_end, i, 1], label="Reference omega y")
            plt.ylabel("Omega y (rad/s)")
            plt.legend(loc="best")
            plt.grid(True)
            plt.subplot(3, 1, 3)
            plt.plot(jnp.arange(rollout_end), omega[:rollout_end, i, 2], label="omega z")
            plt.plot(jnp.arange(rollout_end), ref_omega[:rollout_end, i, 2], label="Reference omega z")
            plt.ylabel("Omega z (rad/s)")
            plt.legend(loc="best")
            plt.grid(True)
            plt.xlabel("Timesteps")
            plt.suptitle(f"{PREFIX_NAME} Angular Velocity Rollout for Env {i} \n Equivariant Model: {args.equivariant}")
            plt.tight_layout()
            plt.savefig(save_path_base+f"/{PREFIX_NAME}_omega_env_{i}.png", dpi=1000)
            plt.close()

        if "particle" in args.env_name:
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
            plt.suptitle(f"{PREFIX_NAME} Action Curves for Env {i} \n Equivariant Model: {args.equivariant}")
            plt.tight_layout()
            plt.savefig(save_path_base+f"/{PREFIX_NAME}_actions_env_{i}.png", dpi=1000)
        
        elif "astrobee" in args.env_name:
            plt.figure()
            plt.subplot(3,2,1)
            plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 0], label="Force X")
            plt.plot(jnp.arange(rollout_end), ref_F[:rollout_end, i, 0], label="ref Force X")
            plt.grid(True)
            plt.legend()
            plt.ylabel("Force X")
            plt.subplot(3,2,2)
            plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 1], label="Force Y")
            plt.plot(jnp.arange(rollout_end), ref_F[:rollout_end, i, 1], label="ref Force Y")
            plt.grid(True)
            plt.ylabel("Force Y")
            plt.legend()
            plt.subplot(3,2,3)
            plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 2], label="Force Z")
            plt.plot(jnp.arange(rollout_end), ref_F[:rollout_end, i, 2], label="ref Force Z")
            plt.grid(True)
            plt.ylabel("Force Z")
            plt.xlabel("Timesteps")
            plt.legend()
            plt.subplot(3,2,4)
            plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 3], label="Torque X")
            plt.plot(jnp.arange(rollout_end), ref_tau[:rollout_end, i, 0], label="ref Torque X")
            plt.grid(True)
            plt.ylabel("Torque X")
            plt.legend()
            plt.subplot(3,2,5)
            plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 4], label="Torque Y")
            plt.plot(jnp.arange(rollout_end), ref_tau[:rollout_end, i, 1], label="ref Torque Y")
            plt.grid(True)
            plt.ylabel("Torque Y")
            plt.legend()
            plt.subplot(3,2,6)
            plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 5], label="Torque Z")
            plt.plot(jnp.arange(rollout_end), ref_tau[:rollout_end, i, 2], label="ref Torque Z")
            plt.grid(True)
            plt.legend()
            plt.xlabel("Timesteps")
            plt.ylabel("Torque Z")
            plt.suptitle(f"{PREFIX_NAME} Action Curves for Env {i} \n Equivariant Model: {args.equivariant}")
            plt.tight_layout()
            plt.savefig(save_path_base+f"/{PREFIX_NAME}_actions_env_{i}.png", dpi=1000)
            # plt.show()
            plt.close()
        
        elif "quadrotor" in args.env_name:
            plt.figure()
            plt.subplot(2,2,1)
            plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 0], label="input 1")
            plt.plot(jnp.arange(rollout_end), ref_action[:rollout_end, i, 0], label="ref input 1")
            plt.grid(True)
            plt.legend()
            plt.ylabel("Normalized motor speed squared 1")
            plt.subplot(2,2,2)
            plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 1], label="input 2")
            plt.plot(jnp.arange(rollout_end), ref_action[:rollout_end, i, 1], label="ref input 2")
            plt.grid(True)
            plt.ylabel("Normalized motor speed squared 2")
            plt.xlabel("Timesteps")
            plt.legend()
            plt.subplot(2,2,3)
            plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 2], label="input 3")
            plt.plot(jnp.arange(rollout_end), ref_action[:rollout_end, i, 2], label="ref input 3")
            plt.grid(True)
            plt.ylabel("Normalized motor speed squared 3")
            plt.legend()
            plt.subplot(2,2,4)
            plt.plot(jnp.arange(rollout_end), actions[:rollout_end, i, 3], label="input 4")
            plt.plot(jnp.arange(rollout_end), ref_action[:rollout_end, i, 3], label="ref input 4")
            plt.grid(True)
            plt.ylabel("Normalized motor speed squared 4")
            plt.xlabel("Timesteps")
            plt.legend()
            plt.suptitle(f"{PREFIX_NAME} Action Curves for Env {i} \n Equivariant Model: {args.equivariant}")
            plt.tight_layout()
            plt.savefig(save_path_base+f"/{PREFIX_NAME}_actions_env_{i}.png", dpi=1000)
            # plt.show()
            plt.close()

    # # Make a plot that shows the error between the particle position and the reference position and averages over all envs with mean and std. dev. shown
    # errors = jnp.linalg.norm(pos - ref_pos, axis=-1)
    # mean_errors = jnp.mean(errors, axis=1)
    # std_errors = jnp.std(errors, axis=1)

    # plt.figure()
    # plt.plot(jnp.arange(rollout_end), mean_errors[:rollout_end], label="Mean Error")
    # plt.fill_between(jnp.arange(rollout_end), mean_errors[:rollout_end] - std_errors[:rollout_end], mean_errors[:rollout_end] + std_errors[:rollout_end], alpha=0.5)
    # plt.xlabel("Timesteps")
    # plt.ylabel("Error")
    # plt.legend()
    # plt.title(f"Mean Error Between Particle Position and Reference Position \n  {args.num_envs} Seeds averaged, Equivariant Model: {args.equivariant}")
    # plt.tight_layout()
    # plt.savefig(save_path_base+"/mean_error.png", dpi=1000)
    # # plt.show()


    # # Plot reward curves for each env
    # plt.figure()
    # for i in range(args.num_envs):
    #     rollout_end = dones.shape[0]
    #     for t in range(dones.shape[0]):
    #         if dones[t, i]:
    #             rollout_end = t
    #             break
    #     plt.plot(jnp.arange(rollout_end), rewards[:rollout_end, i], label=f"Env {i}")
    # plt.xlabel("Timesteps")
    # plt.ylabel("Reward")
    # plt.legend()
    # plt.title(f"Reward Curves for {args.num_envs} Environments \n Equivariant Model: {args.equivariant}")
    # plt.tight_layout()
    # plt.savefig(save_path_base+"/rewards.png", dpi=1000)
