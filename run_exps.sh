

CUDA_VISIBLE_DEVICES=7 python quick_test_scripts/train_gcbc_pm.py --max_epochs 100 -bs 128 -hd 256 -lr 3e-4 -wb --goal_type zero --run_name goal_zero
CUDA_VISIBLE_DEVICES=7 python quick_test_scripts/train_gcbc_pm.py --max_epochs 100 -bs 128 -hd 256 -lr 3e-4 -wb --goal_type last+start_qpos_qvel_s --run_name last+start_qpos_qvel_s
CUDA_VISIBLE_DEVICES=7 python quick_test_scripts/train_gcbc_pm.py --max_epochs 100 -bs 128 -hd 256 -lr 3e-4 -wb --goal_type last+start_qpos_s --run_name last+start_qpos_s
CUDA_VISIBLE_DEVICES=7 python quick_test_scripts/train_gcbc_pm.py --max_epochs 100 -bs 128 -hd 256 -lr 3e-4 -wb --goal_type last_qpos_s --run_name last_qpos_s
CUDA_VISIBLE_DEVICES=7 python quick_test_scripts/train_gcbc_pm.py --max_epochs 100 -bs 128 -hd 256 -lr 3e-4 -wb --goal_type last+start_qpos_qvel_c --run_name last+start_qpos_qvel_c
CUDA_VISIBLE_DEVICES=7 python quick_test_scripts/train_gcbc_pm.py --max_epochs 100 -bs 128 -hd 256 -lr 3e-4 -wb --goal_type last+start_qpos_c --run_name last+start_qpos_c
CUDA_VISIBLE_DEVICES=7 python quick_test_scripts/train_gcbc_pm.py --max_epochs 100 -bs 128 -hd 256 -lr 3e-4 -wb --goal_type last_qpos_c --run_name last_qpos_c

CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_traphormerbc_pm.py --max_epochs 100 -bs 128 -hd 256 -lr 3e-4 -wb --goal_dim 8 --mask_rate 0.75 --run_name bert_g8_mr_0.75
CUDA_VISIBLE_DEVICES=1 python quick_test_scripts/train_traphormerbc_pm.py --max_epochs 100 -bs 128 -hd 256 -lr 3e-4 -wb --goal_dim 8 --mask_rate 0.9 --run_name bert_g8_mr_0.9
CUDA_VISIBLE_DEVICES=2 python quick_test_scripts/train_traphormerbc_pm.py --max_epochs 100 -bs 128 -hd 256 -lr 3e-4 -wb --goal_dim 8 --mask_rate 0.25 --run_name bert_g8_mr_0.25

CUDA_VISIBLE_DEVICES=4 python quick_test_scripts/train_traphormerbc_pm.py --max_epochs 100 -bs 128 -hd 256 -lr 3e-4 -wb --goal_dim 128 --mask_rate 0.25 --run_name bert_g128_mr_0.25
CUDA_VISIBLE_DEVICES=3 python quick_test_scripts/train_traphormerbc_pm.py --max_epochs 100 -bs 128 -hd 256 -lr 3e-4 -wb --goal_dim 128 --mask_rate 0.9 --run_name bert_g128_mr_0.9

# goal conditioned bc
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 1024 --max_epochs 100 --dataset_path maze2d-open-v0_osil_short_trajs/ --env_name maze2d-open-v0 --frac 1.0 -wb --run_name gcbcv2_maze2d_open_f1.0
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 1024 --max_epochs 200 --dataset_path maze2d-open-v0_osil_short_trajs/ --env_name maze2d-open-v0 --frac 0.5 -wb --run_name gcbcv2_maze2d_open_f0.5
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 1024 --max_epochs 400 --dataset_path maze2d-open-v0_osil_short_trajs/ --env_name maze2d-open-v0 --frac 0.25 -wb --run_name gcbcv2_maze2d_open_f0.25
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 1024 --max_epochs 1000 --dataset_path maze2d-open-v0_osil_short_trajs/ --env_name maze2d-open-v0 --frac 0.1 -wb --run_name gcbcv2_maze2d_open_f0.1
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 1024 --max_epochs 2000 --dataset_path maze2d-open-v0_osil_short_trajs/ --env_name maze2d-open-v0 --frac 0.05 -wb --run_name gcbcv2_maze2d_open_f0.05
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 1024 --max_epochs 10000 --dataset_path maze2d-open-v0_osil_short_trajs/ --env_name maze2d-open-v0 --frac 0.01 -wb --run_name gcbcv2_maze2d_open_f0.01

# supervised osil v1
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 20000 --dataset_path maze2d-open-v0_osil_short_trajs/ --env_name maze2d-open-v0 --frac 1.0 -wb --run_name tosil_maze2d_open_f1.0
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 20000 --dataset_path maze2d-open-v0_osil_short_trajs/ --env_name maze2d-open-v0 --frac 0.5 -wb --run_name tosil_maze2d_open_f0.5
CUDA_VISIBLE_DEVICES=1 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 20000 --dataset_path maze2d-open-v0_osil_short_trajs/ --env_name maze2d-open-v0 --frac 0.25 -wb --run_name tosil_maze2d_open_f0.25
CUDA_VISIBLE_DEVICES=1 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 20000 --dataset_path maze2d-open-v0_osil_short_trajs/ --env_name maze2d-open-v0 --frac 0.1 -wb --run_name tosil_maze2d_open_f0.1
CUDA_VISIBLE_DEVICES=2 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 20000 --dataset_path maze2d-open-v0_osil_short_trajs/ --env_name maze2d-open-v0 --frac 0.05 -wb --run_name tosil_maze2d_open_f0.05
CUDA_VISIBLE_DEVICES=2 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 20000 --dataset_path maze2d-open-v0_osil_short_trajs/ --env_name maze2d-open-v0 --frac 0.01 -wb --run_name tosil_maze2d_open_f0.01