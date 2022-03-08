
# goal conditioned bc
# CUDA_VISIBLE_DEVICES=0  python quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots  -1 -wb --run_name ns_max
# CUDA_VISIBLE_DEVICES=0  python quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots   5 -wb --run_name ns_5
# CUDA_VISIBLE_DEVICES=0  python quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots   4 -wb --run_name ns_4
# CUDA_VISIBLE_DEVICES=0  python quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots   3 -wb --run_name ns_3
# CUDA_VISIBLE_DEVICES=0  python quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots   2 -wb --run_name ns_2


# CUDA_VISIBLE_DEVICES=0 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented --image --stack_frame 1 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=1 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_encv2_xaviar --image --stack_frame 1 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=2 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 0.01 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_encv2_xaviar_lr1e-2 --image --stack_frame 1 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=4 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 0.01 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_xaviar_lr1e-2 --image --stack_frame 1 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=5 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_xaviar --image --stack_frame 1 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=0 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 4e-4 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_4e-4 --image --stack_frame 1 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=1 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-4 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_1e-4 --image --stack_frame 1 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=2 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_dset10 --image --stack_frame 1 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=5 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_2_g_augmented_dset100 --image --stack_frame 2 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=0 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_bnorm --image --stack_frame 1 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=1 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_bnorm_rm_last_relu --image --stack_frame 1 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=2 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_enc_resnet --image --stack_frame 1 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=0 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_pred_color --image --stack_frame 1 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=0 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_pred_eef --image --stack_frame 1 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=1 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_4_g_augmented_pred_eef --image --stack_frame 4 --start_eval_after 100000 
# CUDA_VISIBLE_DEVICES=0 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_pred_eef_mlp3_tanh --image --stack_frame 1 --start_eval_after 100000
# CUDA_VISIBLE_DEVICES=0 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_augmented_pred_eef_mlp3_tanh_bn --image --stack_frame 1 --start_eval_after 100000
# CUDA_VISIBLE_DEVICES=0 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_next_pred_eef --image --stack_frame 1 --start_eval_after 100000
# CUDA_VISIBLE_DEVICES=1 python /projects/dm_sac/quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name v3_ns_max_stack_1_g_next_pred_eef_mlp1024 -hd 1024 --image --stack_frame 1 --start_eval_after 100000


#### 03/07

# testing overfitting: yes it does overfit
CUDA_VISIBLE_DEVICES=0 taskset -c 1-4   python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --task_size 10 --run_name v5_ns_max_base            
CUDA_VISIBLE_DEVICES=1 taskset -c 5-8   python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --task_size 10 --enc_type normal --run_name v5_ns_max_enc+normal
CUDA_VISIBLE_DEVICES=2 taskset -c 9-12  python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --task_size 10 --enc_type normal_bnorm --run_name v5_ns_max_enc+normal+bnorm  
CUDA_VISIBLE_DEVICES=3 taskset -c 13-16 python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --task_size 10 --use_huber_loss --run_name v5_ns_max_bcloss+huber  
CUDA_VISIBLE_DEVICES=4 taskset -c 17-20 python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --task_size 10 --use_target_eef_loss 1.0 --run_name v5_ns_max_auxloss+eef
CUDA_VISIBLE_DEVICES=5 taskset -c 21-24 python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --task_size 10 --use_target_color_loss 1.0 --run_name v5_ns_max_auxloss+color
CUDA_VISIBLE_DEVICES=6 taskset -c 25-28 python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 1 --start_eval_after 5000 --task_size 10 --run_name v5_ns_max_stack+1
CUDA_VISIBLE_DEVICES=7 taskset -c 29-32 python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 2 --start_eval_after 5000 --task_size 10 --run_name v5_ns_max_stack+2

# normal testing
CUDA_VISIBLE_DEVICES=0 taskset -c 1-4   python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --run_name v5_ns_max_base            
CUDA_VISIBLE_DEVICES=1 taskset -c 5-8   python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --enc_type normal --run_name v5_ns_max_enc+normal
CUDA_VISIBLE_DEVICES=2 taskset -c 9-12  python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --enc_type normal_bnorm --run_name v5_ns_max_enc+normal+bnorm  
CUDA_VISIBLE_DEVICES=3 taskset -c 13-16 python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --use_huber_loss --run_name v5_ns_max_bcloss+huber  
CUDA_VISIBLE_DEVICES=4 taskset -c 17-20 python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --use_target_eef_loss 1.0 --run_name v5_ns_max_auxloss+eef
CUDA_VISIBLE_DEVICES=5 taskset -c 21-24 python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --use_target_color_loss 1.0 --run_name v5_ns_max_auxloss+color
CUDA_VISIBLE_DEVICES=6 taskset -c 25-28 python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 1 --start_eval_after 5000 --run_name v5_ns_max_stack+1
CUDA_VISIBLE_DEVICES=7 taskset -c 29-32 python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 2 --start_eval_after 5000 --run_name v5_ns_max_stack+2

# after the first run
CUDA_VISIBLE_DEVICES=0 taskset -c 1-4 python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --use_huber_loss --enc_type normal_bnorm --run_name v5_ns_max_bcloss+huber_enc+normal+bnorm 
CUDA_VISIBLE_DEVICES=1 taskset -c 5-8 python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --use_huber_loss --enc_type normal --run_name v5_ns_max_bcloss+huber_enc+normal

# debug the resnet slowness
CUDA_VISIBLE_DEVICES=4 taskset -c 9-12 python /projects/dm_sac/quick_test_scripts/train_gcbcv5_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --image --stack_frame 4 --start_eval_after 5000 --use_target_color_loss 1.0 --run_name v5_ns_max_auxloss+color --resume --ckpt wandb_logs/osil/rgfedw3i/checkpoints/last.ckpt


# # different task sizes
# CUDA_VISIBLE_DEVICES=1  python quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --task_size 1000 -wb --run_name ntask_1k
# CUDA_VISIBLE_DEVICES=1  python quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --task_size  500 -wb --run_name ntask_500
# CUDA_VISIBLE_DEVICES=1  python quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --task_size  250 -wb --run_name ntask_250
# CUDA_VISIBLE_DEVICES=1  python quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --task_size  100 -wb --run_name ntask_100
# CUDA_VISIBLE_DEVICES=1  python quick_test_scripts/train_gcbcv2_reacher_2d.py -bs 1024 --lr 1e-3 --max_steps 100000 --task_size   50 -wb --run_name ntask_50

# # goal conditioned bc with last state of the demo as the goal
# # supervised osil v1
# CUDA_VISIBLE_DEVICES=0 python quick_test_scripts/train_tosilv1_reacher_2d.py --max_padding 50 -bs 64 -gd 3 --lr 1e-3 --max_steps 100000 --num_shots -1 -wb --run_name ns_max
