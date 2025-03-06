#!/bin/bash
cd /mnt/petrelfs/haoyunzhuo/EMMA/EMMA_BAK_108

# Subjects and strategy
# SUBJECT="Chemistry" # You can use multiple subjects separated by spaces
STRATEGY="CoT" # CoT or Direct

# Remote proprietary model selection

# AIzaSyDlmb73omTgAGvw_a9lGxK5pC56fuLtJoQ
# AIzaSyAqnXgoB8uuRTokmbcm8xDfki7c5JjJIt0
MODEL="gemini-2.0-flash-thinking-exp-01-21" # Remote model name
API_KEY="AIzaSyDlmb73omTgAGvw_a9lGxK5pC56fuLtJoQ"

# gemini-2.0-flash-exp # chatgpt-4o-latest
GENERATE_MODEL="gemini-2.0-flash-thinking-exp-01-21"

# Default additional parameters
MAX_TOKENS=4096
TEMPERATURE=0
SAVE_EVERY=1
DATASET_NAME="mm-reasoning/EMMA-mini"
SPLIT="test"
CONFIG_PATH="configs/scoring.yaml"
TOTAL_NUM=4


# Replace spaces in subjects with underscores for file names
SUBJECT_FORMATTED=$(echo $SUBJECT | tr ' ' '_')

# Construct output and log file paths
# OUTPUT_FILE="results/testmini/test-time-compute/thinking-121-Bo${TOTAL_NUM}/${GENERATE_MODEL}_${SUBJECT_FORMATTED}_${TOTAL_NUM}.json"
OUTPUT_FILE="results/testmini/test-time-compute/thinking-121-Bo${TOTAL_NUM}/gemini-2.0-flash-thinking-exp-01-21_all.json$"
LOG_FILE="logs/Scoring/${MODEL}_ALL_Bo${TOTAL_NUM}.log"

# Print constructed file paths for debugging
echo "==============================================="
echo "🚀 Starting Script Execution"
echo "==============================================="
echo "📁 Output File Path: ${OUTPUT_FILE}"
echo "📝 Log File Path:    ${LOG_FILE}"
echo "-----------------------------------------------"
echo "🔧 Configuration Details:"
echo "   - Subjects:       ${SUBJECT}"
echo "   - Strategy:       ${STRATEGY}"
echo "   - Model:          ${MODEL}"
echo "   - Total Num:      ${TOTAL_NUM}"
echo "-----------------------------------------------"

# Run the script
gpus=0
cpus=1
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE \
--job-name="Bo${TOTAL_NUM}" \
--mpi=pmi2 \
-n1 --ntasks-per-node=1 -c ${cpus} \
--kill-on-bad-exit=1 \
--quotatype=${quotatype}  \
-w SH-IDCA1404-10-140-54-119 \
nohup python scoring.py  \
  --dataset_name $DATASET_NAME \
  --subject "Math" "Chemistry" "Physics" "Coding" \
  --split $SPLIT \
  --config_path $CONFIG_PATH \
  --output_path $OUTPUT_FILE \
  --save_every $SAVE_EVERY \
  --total_num $TOTAL_NUM \
  --model $MODEL \
  --api_key $API_KEY \
  --model_path "" \
  --max_tokens $MAX_TOKENS \
  --temperature $TEMPERATURE \
  $RERUN_FLAG 1>$LOG_FILE 2>&1 &

# Completion message
echo "✅ Script launched successfully!"
echo "==============================================="
