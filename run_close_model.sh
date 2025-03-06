#!/bin/bash
cd /mnt/petrelfs/haoyunzhuo/EMMA/EMMA_BAK_108

# Subjects and strategy
SUBJECT="Coding" # You can use multiple subjects separated by spaces
STRATEGY="CoT" # CoT or Direct

# Remote proprietary model selection
# AIzaSyDlmb73omTgAGvw_a9lGxK5pC56fuLtJoQ
# AIzaSyAqnXgoB8uuRTokmbcm8xDfki7c5JjJIt0
MODEL="gemini-2.0-flash-thinking-exp-01-21" # Remote model name
API_KEY="AIzaSyAqnXgoB8uuRTokmbcm8xDfki7c5JjJIt0"

# Default additional parameters
MAX_TOKENS=8192
TEMPERATURE=0
SAVE_EVERY=10
DATASET_NAME="mm-reasoning/EMMA"
SPLIT="test"
CONFIG_PATH="configs/gpt.yaml"

# Replace spaces in subjects with underscores for file names
SUBJECT_FORMATTED=$(echo $SUBJECT | tr ' ' '_')
JOB_NAME="${SUBJECT_FORMATTED}"

# Construct output and log file paths
OUTPUT_FILE="results/All_${MODEL}_${SUBJECT_FORMATTED}_${STRATEGY}.json"
LOG_FILE="logs/All_${MODEL}_${SUBJECT_FORMATTED}_${STRATEGY}.log"

# Print constructed file paths for debugging
echo "==============================================="
echo "🚀 Starting Script Execution"
echo "-----------------------------------------------"
echo "---- Model:          ${MODEL}"
echo "📁 Output File Path: ${OUTPUT_FILE}"
echo "📝 Log File Path:    ${LOG_FILE}"
echo "==============================================="

# Run the script
gpus=0
cpus=1
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE \
--job-name=${JOB_NAME} \
--mpi=pmi2 \
-n1 --ntasks-per-node=1 -c ${cpus} \
--kill-on-bad-exit=1 \
--quotatype=${quotatype}  \
-w SH-IDCA1404-10-140-54-119 \
nohup python generate_response.py  \
  --dataset_name $DATASET_NAME \
  --subject $SUBJECT \
  --split $SPLIT \
  --strategy $STRATEGY \
  --output_path $OUTPUT_FILE \
  --model $MODEL \
  --api_key $API_KEY \
  --model_path "" \
  --config_path $CONFIG_PATH \
  --max_tokens $MAX_TOKENS \
  --temperature $TEMPERATURE \
  --save_every $SAVE_EVERY 1>$LOG_FILE 2>&1 &

# Completion message
echo "✅ Script launched successfully!"
echo "==============================================="
