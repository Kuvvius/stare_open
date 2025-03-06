#!/bin/bash
cd /mnt/petrelfs/haoyunzhuo/EMMA/EMMA_iccv

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.8
export PATH=/mnt/petrelfs/share/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="1,2,4,7"

OMP_NUM_THREADS=4 srun \
 --partition=MoE \
 --mpi=pmi2 \
 --job-name=qwen-fold \
 -c 16 \
 -w SH-IDCA1404-10-140-54-36 \
 --ntasks-per-node=1 \
 --kill-on-bad-exit=1 \
 --quotatype=reserved \
nohup python generate_response.py \
--model_path '/mnt/petrelfs/share_data/quxiaoye/models/Qwen2-VL-72B-Instruct' \
--output_path 'results/Qwen2VL_folding_nets_2d_perception_test.json' 1>logs/Qwen2VL_folding_nets_2d_perception_test.log 2>&1 &
# python generate_response.py \
# --subject 'Physics' 'Chemistry' \
# --strategy 'Direct' \
# --rerun \
# --model_path '/mnt/petrelfs/share_data/quxiaoye/models/Qwen2-VL-72B-Instruct' \
# --output_path 'results/open-source/Qwen_Direct.json' 1>logs/Qwen_Direct.log 2>&1 &




# nohup python generate_response.py >logs/test.log 2>&1 &
# nohup python generate_response.py \
# --subject 'Math' 'Physics' 'Chemistry' 'Coding' \
# --strategy 'CoT' \
# --model_path '/mnt/petrelfs/share_data/quxiaoye/models/QVQ-72B-Preview' \
# --output_path 'results/overall/open-source/QVQ_CoT.json' 1>logs/QVQ_CoT.log 2>&1 &
# python scoring.py \
# --subject 'Physics' \
# --total_num 8 \
# --select_num 8 \
# --model_path '/mnt/petrelfs/share_data/quxiaoye/models/InternVL2_5-78B' \
# --output_path 'results/test-time-compute/internvl-best-of-8/InternVL2_5_Physics_8.json' 1>logs/InternVL2_5_Physics_bo8.log 2>&1 &






































