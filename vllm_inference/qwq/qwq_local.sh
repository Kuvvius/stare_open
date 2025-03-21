source ~/.bashrc
source ~/anaconda3/bin/activate vllm2

# environment variables
export OMP_NUM_THREADS=4
AD_NAME=songmingyang
encrypted_password=iWRsYqbwV4EJgJvU8QjLe00CptZc5jBVH3FMo5i6n9mVdOSoUurpyBTmst1Z
new_proxy_address=http://${AD_NAME}:${encrypted_password}@10.1.20.50:23128/
export http_proxy=$new_proxy_address
export https_proxy=$new_proxy_address
export HTTP_PROXY=$new_proxy_address
export HTTPS_PROXY=$new_proxy_address

export HF_ENDPOINT=https://hf-mirror.com
unset HF_ENDPOINT

code_base=/mnt/petrelfs/songmingyang/code/reasoning/others/stare_open/vllm_inference/qwq
cd $code_base


export API_TYPE=openai
export OPENAI_API_URL=https://burn.hair/v1/chat/completions
export OPENAI_API_KEY=sk-2AUzUksigO0e61zt727c74Ce561349D38d1eA8Da4a4550D8
# CUDA_VISIBLE_DEVICES=4,5,6,7
debug_accelerate_config=/mnt/petrelfs/songmingyang/.config/accelerate/1gpu.yaml
run_accelerate_config=./accelerate_4gpus.yaml
# pope,mme,vqav2,gqa,vizwiz_vqa,scienceqa_full,seedbench,seedbench_2,refcoco,refcoco+,mmmu,textvqa,ok_vqa,flickr30k,qbenchs_dev,mmbench_en_dev,mmbench_cn_dev,mmstar 

job_id=4672194


accelerate_config=/mnt/petrelfs/songmingyang/.config/accelerate/4gpus_2.yaml
# accelerate_config=/mnt/petrelfs/songmingyang/.config/accelerate/1gpu.yaml
# accelerate_config=/mnt/petrelfs/songmingyang/.config/accelerate/cpu.yaml
gpus=4
cpus=32
quotatype="reserved"
export API_TYPE=openai
export OPENAI_API_URL=https://burn.hair/v1/chat/completions
export OPENAI_API_KEY=sk-mX5SCQvAm14HZZgx6e23F1Cb84944a38Bc0eF0Cf79Bc3aB2
export VLLM_WORKER_MULTIPROC_METHOD="spawn"


output_path=/mnt/petrelfs/songmingyang/code/reasoning/others/stare_open/vllm_inference/qwq/scripts/qwq_gsm8k.jsonl
input_path=/mnt/petrelfs/songmingyang/songmingyang/data/reasoning/gsm8k
model_path=/mnt/petrelfs/songmingyang/quxiaoye/models/QwQ-32B

gpus=0
cpus=2
quotatype="reserved"
CUDA_VISIBLE_DEVICES="0,1,3,5" srun --partition=MoE --job-name="eval" --mpi=pmi2  --gres=gpu:${gpus} -n1 --ntasks-per-node=1 -c ${cpus} --kill-on-bad-exit=1 --quotatype=${quotatype}  \
-w SH-IDCA1404-10-140-54-5 \
python qwq_infer.py \
--input_path ${input_path} \
--output_path ${output_path} \
--model_path ${model_path} \
--batch_size 100 \
--tensor_parallel_size 4 


