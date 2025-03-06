source ~/.bashrc
source ~/anaconda3/bin/activate r1-v

# environment variables
export OMP_NUM_THREADS=1

code_base=/mnt/petrelfs/songmingyang/code/reasoning/others/stare_open/vllm_inference
cd $code_base

export SLURM_JOB_ID=4294232
unset SLURM_JOB_ID

new_proxy_address="http://closeai-proxy.pjlab.org.cn:23128"
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address


dataset_name=folding_nets_test

gpus=0
cpus=2
quotatype="reserved"
CUDA_VISIBLE_DEVICES=0,4,6,7 \
srun --partition=MoE --job-name="qvq_rollout" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
-w SH-IDCA1404-10-140-54-67 \
python  ./inference.py  \
--input_path VisSim/${dataset_name} \
--output_path /mnt/petrelfs/songmingyang/code/reasoning/others/stare_open/vllm_inference/scripts/results/qwen25vl72b/${dataset_name}.jsonl \
--function folding \
--model_path /mnt/petrelfs/share_data/songmingyang/model/mm/Qwen2.5-VL-72B-Instruct \
--tensor_parallel_size 4 



