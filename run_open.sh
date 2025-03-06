#!/bin/bash

SUBJECT="Coding" # You can use multiple subjects separated by spaces
STRATEGY="CoT" # CoT or Direct
SUBJECT_FORMATTED=$(echo $SUBJECT | tr ' ' '_')


MODEL_PATH="/mnt/petrelfs/share_data/songmingyang/model/mm/Qwen2.5-VL-72B-Instruct"
MODEL=$(basename "$MODEL_PATH")
DATASET_NAME="luckychao/EMMA"
TEMPERATURE=0
MAX_TOKENS=8192

# Construct output and log file paths
OUTPUT_FILE="results/${MODEL}_${SUBJECT_FORMATTED}_${STRATEGY}.json"
LOG_FILE="logs/${MODEL}_${SUBJECT_FORMATTED}_${STRATEGY}.log"
mkdir -p "$(dirname "$OUTPUT_FILE")"
mkdir -p "$(dirname "$LOG_FILE")"


# Print constructed file paths for debugging
echo "==============================================="
echo "🚀 Starting Script Execution"
echo "-----------------------------------------------"
echo "---- Model:          ${MODEL}"
echo "📁 Output File Path: ${OUTPUT_FILE}"
echo "📝 Log File Path:    ${LOG_FILE}"
echo "==============================================="


export CUDA_HOME=/mnt/petrelfs/share/cuda-11.8
export PATH=/mnt/petrelfs/share/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="1,2,4,7"
OMP_NUM_THREADS=4 srun \
 --partition=MoE \
 --mpi=pmi2 \
 --job-name=qwen-rm \
 --gres=gpu:0 \
 -c 2 \
 -w SH-IDCA1404-10-140-54-36 \
 --ntasks-per-node=1 \
 --kill-on-bad-exit=1 \
 --quotatype=reserved \
nohup python generate_response_ntimes.py \
--dataset_name $DATASET_NAME \
--subject 'Physics' 'Chemistry' \
--strategy 'Direct' \
--rerun \
--max_tokens $MAX_TOKENS \
--temperature $TEMPERATURE \
--model_path $MODEL_PATH \
--output_path $OUTPUT_FILE 1>$LOG_FILE 2>&1 &
# Completion message
echo "✅ Script launched successfully!"
echo "==============================================="



































