
# STARE-OPEN

## Requirements
```bash
git clone https://github.com/Kuvvius/stare_open
cd stare_open
git install -e .
```

## Usage

For Vllm inference:
```bash
python -m vllm_inference \
--input_path VisSim/${dataset_name} \
--output_path ./scripts/results/qwen25vl72b/${dataset_name}.jsonl \
--function folding \
--model_path Qwen/Qwen2.5-VL-72B-Instruct \
--tensor_parallel_size 4 
```

For Vissim results eval:

TBD