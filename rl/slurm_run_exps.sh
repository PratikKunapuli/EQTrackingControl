#!/bin/bash
#SBATCH --mem-per-gpu=60G
#SBATCH --cpus-per-gpu=4
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH --output=./slurm_outs/%A/job_%A_%a.out
#SBATCH --error=./slurm_outs/%A/job_%A_%a.err
#SBATCH --array=0-7

TRAIN_ARGS=(
    "--env-name particle_random_walk_velocity --exp-name RUN_EXPS_particle_random_walk --num-seeds 1 --equivariant 0 --lr 3e-4 --num-envs 16 --num-steps 512 --total-timesteps 10e6 --activation leaky_relu --out-activation hard_tanh --reward_q_pos 1e-2 --reward_q_vel 1e-2 --reward_r 1e-4 --reward_reach --terminal-reward -25.0 --num-layers 3 --num-nodes 64"
    "--env-name particle_random_walk_velocity --exp-name RUN_EXPS_particle_random_walk_p_equivariant --num-seeds 1 --equivariant 1 --lr 3e-4 --num-envs 16 --num-steps 512 --total-timesteps 10e6 --activation leaky_relu --out-activation hard_tanh --reward_q_pos 1e-2 --reward_q_vel 1e-2 --reward_r 1e-4 --reward_reach --terminal-reward -25.0 --num-layers 3 --num-nodes 64"
    "--env-name particle_random_walk_velocity --exp-name RUN_EXPS_particle_random_walk_pv_equivariant --num-seeds 1 --equivariant 3 --lr 3e-4 --num-envs 16 --num-steps 512 --total-timesteps 10e6 --activation leaky_relu --out-activation hard_tanh --reward_q_pos 1e-2 --reward_q_vel 1e-2 --reward_r 1e-4 --reward_reach --terminal-reward -25.0 --num-layers 3 --num-nodes 64"
    "--env-name particle_random_walk_velocity --exp-name RUN_EXPS_particle_random_walk_pva_equivariant --num-seeds 1 --equivariant 4 --lr 3e-4 --num-envs 16 --num-steps 512 --total-timesteps 10e6 --activation leaky_relu --out-activation hard_tanh --reward_q_pos 1e-2 --reward_q_vel 1e-2 --reward_r 1e-4 --reward_reach --terminal-reward -25.0 --num-layers 3 --num-nodes 64"
    "--env-name astrobee_random_walk --exp-name RUN_EXPS_astrobee_random_walk --num-seeds 1 --equivariant 0 --symmetry_type 2 --lr 3e-4 --num-envs 32 --num-steps 512 --total-timesteps 400e6 --activation leaky_relu --out-activation hard_tanh --termination-bound 25.0 --reward_q_pos 5e-2 --reward_q_vel 1e-3 --reward_q_rotm 10.0 --reward_q_omega 5e-2 --reward_r 1e-3 --reward_reach --terminal-reward -25.0 --quad-model FAQuad --num-layers 6 --num-nodes 256"
    "--env-name astrobee_random_walk --exp-name RUN_EXPS_astrobee_random_walk_equivariant --num-seeds 1 --equivariant 1 --symmetry_type 2 --lr 3e-4 --num-envs 32 --num-steps 512 --total-timesteps 400e6 --activation leaky_relu --out-activation hard_tanh --termination-bound 25.0 --reward_q_pos 5e-2 --reward_q_vel 1e-3 --reward_q_rotm 10.0 --reward_q_omega 5e-2 --reward_r 1e-3 --reward_reach --terminal-reward -25.0 --quad-model FAQuad --num-layers 6 --num-nodes 256"
    "--env-name quadrotor_random_walk --exp-name RUN_EXPS_quadrotor_random_walk --num-seeds 1 --equivariant 0 --lr 3e-4 --num-envs 64 --num-steps 512 --total-timesteps 1e9 --activation leaky_relu --out-activation clipped_relu --reward_q_pos 2e-2 --reward_q_vel 1e-3 --reward_q_rotm 0.6 --reward_q_omega 1e-4 --reward_r 5e-4 --reward_reach --terminal-reward -25.0 --num-layers 6 --num-nodes 128"
    "--env-name quadrotor_random_walk --exp-name RUN_EXPS_quadrotor_random_walk_equivariant --num-seeds 1 --equivariant 1 --lr 3e-4 --num-envs 64 --num-steps 512 --total-timesteps 1e9 --activation leaky_relu --out-activation clipped_relu --reward_q_pos 2e-2 --reward_q_vel 1e-3 --reward_q_rotm 0.6 --reward_q_omega 1e-4 --reward_r 5e-4 --reward_reach --terminal-reward -25.0 --num-layers 6 --num-nodes 128"
)

EVAL_ARGS=(
    "--env-name particle_random_walk_velocity --seed 0 --load-path ./checkpoints/RUN_EXPS_particle_random_walk_p_equivariant/model_final --equivariant 1"
    "--env-name particle_random_walk_velocity --seed 0 --load-path ./checkpoints/RUN_EXPS_particle_random_walk/model_final --equivariant 0"
    "--env-name particle_random_walk_velocity --seed 0 --load-path ./checkpoints/RUN_EXPS_particle_random_walk_pv_equivariant/model_final --equivariant 3"
    "--env-name particle_random_walk_velocity --seed 0 --load-path ./checkpoints/RUN_EXPS_particle_random_walk_pva_equivariant/model_final --equivariant 4"
    "--env-name astrobee_random_walk --seed 0 --load-path ./checkpoints/RUN_EXPS_astrobee_random_walk/model_final --equivariant 0 --symmetry_type 2"
    "--env-name astrobee_random_walk --seed 0 --load-path ./checkpoints/RUN_EXPS_astrobee_random_walk_equivariant/model_final --equivariant 1 --symmetry_type 2"
    "--env-name quadrotor_random_walk --seed 0 --load-path ./checkpoints/RUN_EXPS_quadrotor_random_walk/model_final --equivariant 0"
    "--env-name quadrotor_random_walk --seed 0 --load-path ./checkpoints/RUN_EXPS_quadrotor_random_walk_equivariant/model_final --equivariant 1"
)

eval python3 train_policy.py ${TRAIN_ARGS[$SLURM_ARRAY_TASK_ID]}
eval python3 eval_policy.py ${EVAL_ARGS[$SLURM_ARRAY_TASK_ID]}
