source ~/.bashrc
source ~/anaconda3/bin/activate r1-v

# environment variables
export OMP_NUM_THREADS=1

code_base=/mnt/petrelfs/songmingyang/code/reasoning/others/stare_open/vllm_inference
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


dataset_full_name=MahtabBg/Video
dataset_short_name=mvideo
model_path=/mnt/petrelfs/songmingyang/songmingyang/model/mm/Qwen2.5-VL-7B-Instruct
model_shortname=qwen25vl7b

gpus=0
cpus=2
quotatype="reserved"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
srun --partition=MoE --job-name="qvq_rollout" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
-w SH-IDCA1404-10-140-54-86 \
python  ./inference.py  \
--input_path ${dataset_full_name} \
--output_path /mnt/petrelfs/songmingyang/code/reasoning/others/stare_open/vllm_inference/scripts/results/${model_shortname}/${dataset_short_name}.jsonl \
--function ${dataset_short_name} \
--model_path ${model_path} \
--tensor_parallel_size 4 \
--max_model_len 96700


dataset_full_name=MahtabBg/NEWPerspective
dataset_short_name=nperspective


gpus=0
cpus=2
quotatype="reserved"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
srun --partition=MoE --job-name="qvq_rollout" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
-w SH-IDCA1404-10-140-54-86 \
python  ./inference.py  \
--input_path ${dataset_full_name} \
--output_path /mnt/petrelfs/songmingyang/code/reasoning/others/stare_open/vllm_inference/scripts/results/${model_shortname}/${dataset_short_name}.jsonl \
--function ${dataset_short_name} \
--model_path ${model_path} \
--tensor_parallel_size 4 \
--max_model_len 96700



