#!/bin/bash
cd /mnt/petrelfs/haoyunzhuo/EMMA/EMMA_BAK_108

# Subjects and strategy
SUBJECT="Physics" # You can use multiple subjects separated by spaces
STRATEGY="CoT" # CoT or Direct

# Remote proprietary model selection
# AIzaSyDlmb73omTgAGvw_a9lGxK5pC56fuLtJoQ
# AIzaSyAqnXgoB8uuRTokmbcm8xDfki7c5JjJIt0
MODEL="gemini-2.0-flash-thinking-exp-01-21" # Remote model name
API_KEY="AIzaSyDlmb73omTgAGvw_a9lGxK5pC56fuLtJoQ"
N_TIMES=16
# Default additional parameters
MAX_TOKENS=8192
TEMPERATURE=0.7
SAVE_EVERY=1
SPLIT="test"
CONFIG_PATH="configs/gpt.yaml"

# Replace spaces in subjects with underscores for file names
SUBJECT_FORMATTED=$(echo $SUBJECT | tr ' ' '_')
# Construct output and log file paths
OUTPUT_FILE="results/testmini/test-time-compute/raw_results/${MODEL}_${SUBJECT_FORMATTED}_${N_TIMES}.json"
LOG_FILE="logs/${MODEL}_${SUBJECT_FORMATTED}_${STRATEGY}_${N_TIMES}.log"

JOB_NAME="${SUBJECT_FORMATTED}"

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
echo "-----------------------------------------------"

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
nohup python generate_response_ntimes.py  \
  --n_times $N_TIMES \
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
