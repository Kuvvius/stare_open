source ~/.bashrc
source ~/anaconda3/bin/activate r1-v

# environment variables
export OMP_NUM_THREADS=1

code_base=/mnt/petrelfs/songmingyang/code/reasoning/others/stare_open/vissim_eval
cd $code_base

export SLURM_JOB_ID=4294232
unset SLURM_JOB_ID

AD_NAME=songmingyang
encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address


# Define an array of datasets
datasets=(
    # vas
    # "3d_va_vissim_test"
    # "3d_va_test"
    # "2d_va_vissim_test"
    # "2d_va_test"
    # # tis
    # "2d_text_instruct_test"
    # "2d_text_instruct_vissim_test"
    # "3d_text_instruct_test"
    # "3d_text_instruct_vissim_test"
    # # folding nets
    # "tangram_puzzle_test"
    # "folding_nets_vissim_test"
    # "folding_nets_3d_perception_test"
    # "folding_nets_2d_perception_test"
    "tangram_puzzle_vissim_test"
#     "folding_nets_test"
#     # others
#     "mvideo"
#     "nperspective"
)


combined_datasets=(
    # # vas
    # "va"
    # "va"
    # "va"
    # "va"
    # # tis
    # "text_instruct"
    # "text_instruct"
    # "text_instruct"
    # "text_instruct"
    # # folding nets
    # "folding"
    # "folding"
    # "folding"
    # "folding"
    # "folding"
    "folding"
    # # others
    # "mvideo"
    # "nperspective"
)


model_short_name=llavaov

cpus=2
gpus=0
quotatype="reserved"
# Loop through both arrays simultaneously
for i in "${!datasets[@]}"; do
    dataset_name="${datasets[$i]}"
    function_type="${combined_datasets[$i]}"
    
    echo "Processing dataset: ${dataset_name}, function type: ${function_type}"
    
    # srun --partition=MoE --job-name="eval_vissim" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
    python  ./evaluator.py  \
    --input_path /mnt/petrelfs/songmingyang/code/reasoning/others/stare_open/vllm_inference/scripts/results/${model_short_name}/${dataset_name}.jsonl \
    --output_path /mnt/petrelfs/songmingyang/code/reasoning/others/stare_open/vissim_eval/scripts/results/${model_short_name}/${dataset_name}.json \
    --dataset_type ${function_type} 
    
    echo "Completed ${dataset_name}"
done
# gpus=0
# cpus=2
# quotatype="reserved"
# CUDA_VISIBLE_DEVICES=0,4,6,7 \
# srun --partition=MoE --job-name="qvq_rollout" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
# -w SH-IDCA1404-10-140-54-67 \
# python  ./inference.py  \
# --input_path VisSim/${dataset_name} \
# --output_path /mnt/petrelfs/songmingyang/code/reasoning/others/stare_open/vllm_inference/scripts/results/qwen25vl72b/${dataset_name}.jsonl \
# --function va \
# --model_path /mnt/petrelfs/share_data/songmingyang/model/mm/Qwen2.5-VL-72B-Instruct \
# --tensor_parallel_size 4 

