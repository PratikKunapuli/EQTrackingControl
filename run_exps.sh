sleep 2
echo "Running Training Experiments...."
echo "==============================================================="
python3 train_policy.py --env-name position --exp-name test_position --add-desc "Non-Eq version, with termination on error and bad, good reward"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name position --exp-name test_position_equivariant --equivariant --add-desc "Eq version, with termination on error and bad, good reward"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name constant_velocity --exp-name test_constant_velocity --add-desc "Non-Eq version, with termination on error and bad, good reward, constant velocity sampled randomly"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name constant_velocity --exp-name test_constant_velocity_equivariant --equivariant --add-desc "Eq version, with termination on error and bad, good reward, constant velocity sampled randomly"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_position --exp-name test_random_walk_position --add-desc "Non-Eq version, with termination on error and bad, good reward and random velocity sampled each time instant"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_position --exp-name test_random_walk_position_equivariant --equivariant --add-desc "Eq version, with termination on error and bad, good reward and random velocity sampled each time instant"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_velocity --exp-name test_random_walk_velocity --add-desc "Non-Eq version, with termination on error and bad, good reward and random accel sampled each time instant"
sleep 2
echo "==============================================================="
python3 train_policy.py --env-name random_walk_velocity --exp-name test_random_walk_velocity_equivariant --equivariant --add-desc "Eq version, with termination on error and bad, good reward and random accel sampled each time instant"
sleep 2
echo "==============================================================="

echo "Running Evaluation Experiments...."
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_position/model_final/ --env-name position --make-animation
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_position_equivariant/model_final/ --env-name position --equivariant --make-animation
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_constant_velocity/model_final/ --env-name constant_velocity --make-animation
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_constant_velocity_equivariant/model_final/ --env-name constant_velocity --equivariant --make-animation
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_walk_position/model_final/ --env-name random_walk_position --make-animation
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_walk_position_equivariant/model_final/ --env-name random_walk_position --equivariant --make-animation
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_walk_velocity/model_final/ --env-name random_walk_velocity --make-animation
sleep 2
echo "==============================================================="
python3 eval_policy.py --seed 0 --load-path ./checkpoints/test_random_walk_velocity_equivariant/model_final/ --env-name random_walk_velocity --equivariant --make-animation
sleep 2
echo "==============================================================="

echo "Done!"
