#!/usr/bin/env sh

exp_name=f8F32R256

task_config="train_exps/${exp_name}/task.py"

output_name=${exp_name}
mkdir -p results/${output_name}/

srun -n32 --ntasks-per-node=8 --gres=gpu:8 \
python -u train.py \
--master_port 28281 \
--model MultiScaleNextDiT_2B_GQA \
--patch_sizes 2 2 4 \
--f_patch_sizes 1 2 2 \
--results_dir results/${output_name} \
--lr 2e-4 \
--grad_clip 2.0 \
--data_parallel sdp \
--checkpointing \
--max_steps 100000 \
--ckpt_every 1000 \
--precision bf16 --grad_precision fp32 \
--qk_norm \
--global_seed 20 \
--cache_data_on_disk \
--task_config ${task_config} \
--rope_theta 256.0 \
--t_scale 1000.0 \
--motion_scale 20.0 \
--vd_weight 1.0 \
2>&1 | tee -a results/${output_name}/output.log
