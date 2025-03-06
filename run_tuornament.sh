#!/bin/bash
cd /mnt/petrelfs/haoyunzhuo/EMMA/EMMA_BAK_108

# Subjects and strategy
# SUBJECT="Math" # You can use multiple subjects separated by spaces

# Remote proprietary model selection
MODEL="gemini-2.0-flash-thinking-exp-01-21" # Remote model name

# AIzaSyDlmb73omTgAGvw_a9lGxK5pC56fuLtJoQ
# AIzaSyAqnXgoB8uuRTokmbcm8xDfki7c5JjJIt0
API_KEY="AIzaSyDlmb73omTgAGvw_a9lGxK5pC56fuLtJoQ"
TOTAL_NUM=8

# Default additional parameters
MAX_TOKENS=8192
TEMPERATURE=0
SPLIT="test"
CONFIG_PATH="configs/scoring.yaml"

# Replace spaces in subjects with underscores for file names
SUBJECT_FORMATTED=$(echo $SUBJECT | tr ' ' '_')
# Construct output and log file paths
OUTPUT_FILE="results/testmini/test-time-compute/thinking-121-tournament-Bo${TOTAL_NUM}/gemini-2.0-flash-thinking-exp-01-21_all.json"
LOG_FILE="logs/Scoring/${MODEL}_ALL_${TOTAL_NUM}.log"
#OUTPUT_FILE="results/testmini/test-time-compute/thinking-tournament-best-of-${TOTAL_NUM}/${MODEL}_${SUBJECT_FORMATTED}_${TOTAL_NUM}.json"
#LOG_FILE="logs/Scoring/${MODEL}_${SUBJECT_FORMATTED}_${TOTAL_NUM}.log"

# Print constructed file paths for debugging
echo "==============================================="
echo "🚀 Starting Script Execution"
echo "==============================================="
echo "📁 Output File Path: ${OUTPUT_FILE}"
echo "📝 Log File Path:    ${LOG_FILE}"
echo "-----------------------------------------------"
echo "🔧 Configuration Details:"
echo "   - Subjects:       ${SUBJECT}"
echo "   - Model:          ${MODEL}"
echo "-----------------------------------------------"

gpus=0
cpus=1
quotatype="reserved"
OMP_NUM_THREADS=8 srun --partition=MoE \
--job-name="tuor-${TOTAL_NUM}" \
--mpi=pmi2 \
-n1 --ntasks-per-node=1 -c ${cpus} \
--kill-on-bad-exit=1 \
--quotatype=${quotatype}  \
-w SH-IDCA1404-10-140-54-119 \
nohup python tournament_best_of_n.py \
  --total_num $TOTAL_NUM \
  --subject "Math" "Chemistry" "Physics" "Coding" \
  --split $SPLIT \
  --output_path $OUTPUT_FILE \
  --model $MODEL \
  --api_key $API_KEY \
  --config_path $CONFIG_PATH \
  --max_tokens $MAX_TOKENS \
  --temperature $TEMPERATURE 1>$LOG_FILE 2>&1 &

# Completion message
echo "✅ Script launched successfully!"
echo "==============================================="
