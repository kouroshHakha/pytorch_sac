

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
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots 100 -wb --run_name gcbcv2_maze2d_open_ns_100
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots  50 -wb --run_name gcbcv2_maze2d_open_ns_50
CUDA_VISIBLE_DEVICES=1 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots  25 -wb --run_name gcbcv2_maze2d_open_ns_25
CUDA_VISIBLE_DEVICES=1 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots  10 -wb --run_name gcbcv2_maze2d_open_ns_10
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   5 -wb --run_name gcbcv2_maze2d_open_ns_5
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   4 -wb --run_name gcbcv2_maze2d_open_ns_4
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   3 -wb --run_name gcbcv2_maze2d_open_ns_3
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   2 -wb --run_name gcbcv2_maze2d_open_ns_2

# gcbc with noisy target
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots 100 -wb --run_name gcbcv2_maze2d_open_ns_100_noisy_target
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   5 -wb --run_name gcbcv2_maze2d_open_ns_5_noisy_target
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   4 -wb --run_name gcbcv2_maze2d_open_ns_4_noisy_target
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   3 -wb --run_name gcbcv2_maze2d_open_ns_3_noisy_target
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   2 -wb --run_name gcbcv2_maze2d_open_ns_2_noisy_target

# gcbc -- with 256 dims of goal_dim (fake one)
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --gd 256 --num_shots 100 -wb --run_name gcbcv2_maze2d_open_ns_100_g256
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --gd 256 --num_shots  50 -wb --run_name gcbcv2_maze2d_open_ns_50_g256
CUDA_VISIBLE_DEVICES=1 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --gd 256 --num_shots  25 -wb --run_name gcbcv2_maze2d_open_ns_25_g256
CUDA_VISIBLE_DEVICES=1 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --gd 256 --num_shots  10 -wb --run_name gcbcv2_maze2d_open_ns_10_g256
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --gd 256 --num_shots   5 -wb --run_name gcbcv2_maze2d_open_ns_5_g256
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --gd 256 --num_shots   4 -wb --run_name gcbcv2_maze2d_open_ns_4_g256
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --gd 256 --num_shots   3 -wb --run_name gcbcv2_maze2d_open_ns_3_g256
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --gd 256 --num_shots   2 -wb --run_name gcbcv2_maze2d_open_ns_2_g256

# supervised osil v1
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots 100 -wb --run_name tosil_maze2d_open_ns_100
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots  50 -wb --run_name tosil_maze2d_open_ns_50
CUDA_VISIBLE_DEVICES=1 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots  25 -wb --run_name tosil_maze2d_open_ns_25
CUDA_VISIBLE_DEVICES=1 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots  10 -wb --run_name tosil_maze2d_open_ns_10
CUDA_VISIBLE_DEVICES=2 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   5 -wb --run_name tosil_maze2d_open_ns_5
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   4 -wb --run_name tosil_maze2d_open_ns_4
CUDA_VISIBLE_DEVICES=1 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   3 -wb --run_name tosil_maze2d_open_ns_3
CUDA_VISIBLE_DEVICES=2 python quick_test_scripts/train_tosilv1_pm.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   2 -wb --run_name tosil_maze2d_open_ns_2

# pseudo supervised osil (gcbc as decoder + pretrained encoder)
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_w_osil_embedding.py -bs 128 --max_steps 10000  --dataset_path ./maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --encoder_ckpt wandb_logs/osil/3647wkjm/checkpoints/cgl-step\=6089-valid_loss\=0.0196-epoch\=608.ckpt  --model osil -wb --run_name gcbc+osil_emb_ns100
CUDA_VISIBLE_DEVICES=1 python quick_test_scripts/train_gcbcv2_w_osil_embedding.py -bs 128 --max_steps 10000  --dataset_path ./maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --encoder_ckpt wandb_logs/osil/38tubxrp/checkpoints/cgl-step\=964-valid_loss\=0.1720-epoch\=964.ckpt   --model osil -wb --run_name gcbc+osil_emb_ns5
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_w_osil_embedding.py -bs 128 --max_steps 10000  --dataset_path ./maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --encoder_ckpt wandb_logs/osil/7je3htz1/checkpoints/cgl-step\=865-valid_loss\=0.2645-epoch\=865.ckpt   --model osil -wb --run_name gcbc+osil_emb_ns4
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_w_osil_embedding.py -bs 128 --max_steps 10000  --dataset_path ./maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --encoder_ckpt wandb_logs/osil/2n02w2ux/checkpoints/cgl-step\=162-valid_loss\=0.3441-epoch\=162.ckpt   --model osil -wb --run_name gcbc+osil_emb_ns3
CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_gcbcv2_w_osil_embedding.py -bs 128 --max_steps 10000  --dataset_path ./maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --encoder_ckpt wandb_logs/osil/36m0n5z8/checkpoints/cgl-step\=280-valid_loss\=0.3464-epoch\=280.ckpt   --model osil -wb --run_name gcbc+osil_emb_ns2

# # supervised osil + mtm
# CUDA_VISIBLE_DEVICES=3 python quick_test_scripts/train_tosil_plus_mlm_joint.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots 100 -wb --run_name tosil+mtm_maze2d_open_ns_100
# CUDA_VISIBLE_DEVICES=7 python quick_test_scripts/train_tosil_plus_mlm_joint.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots  50 -wb --run_name tosil+mtm_maze2d_open_ns_50
# CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_tosil_plus_mlm_joint.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots  25 -wb --run_name tosil+mtm_maze2d_open_ns_25
# CUDA_VISIBLE_DEVICES=2 python quick_test_scripts/train_tosil_plus_mlm_joint.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots  10 -wb --run_name tosil+mtm_maze2d_open_ns_10
# CUDA_VISIBLE_DEVICES=5 python quick_test_scripts/train_tosil_plus_mlm_joint.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   5 -wb --run_name tosil+mtm_maze2d_open_ns_5
# CUDA_VISIBLE_DEVICES=2 python quick_test_scripts/train_tosil_plus_mlm_joint.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   4 -wb --run_name tosil+mtm_maze2d_open_ns_4
# CUDA_VISIBLE_DEVICES=3 python quick_test_scripts/train_tosil_plus_mlm_joint.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   3 -wb --run_name tosil+mtm_maze2d_open_ns_3
# CUDA_VISIBLE_DEVICES=6 python quick_test_scripts/train_tosil_plus_mlm_joint.py -bs 128 --max_steps 10000 --dataset_path maze2d-open-v0_osil_short_trajs_v2/ --env_name maze2d-open-v0 --num_shots   2 -wb --run_name tosil+mtm_maze2d_open_ns_2