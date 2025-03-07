
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
--output_path /mnt/petrelfs/songmingyang/code/reasoning/others/stare_open/vllm_inference/scripts/results/qwen25vl72b/${dataset_name}.jsonl \
--function folding \
--model_path /mnt/petrelfs/share_data/songmingyang/model/mm/Qwen2.5-VL-72B-Instruct \
--tensor_parallel_size 4 
```

For Vissim results eval:

TBD