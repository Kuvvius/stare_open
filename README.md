
# STARE-OPEN

The official repository of **Unfolding Spatial Cognition:Evaluating Multimodal Models on Visual Simulations**.

## Requirements

* vllm>=0.7.3
* torch==2.5.1+cu121
* transformers>=4.49.0
* flash_attn>=2.7.3

You can prepare a VLLM-based environment by your self or reference to `requirements.txt` to prepare an exactly same environment with us.

Then clone this repo and install our package.

```bash
git clone https://github.com/Kuvvius/stare_open
cd stare_open
git install -e .
```

## Usage

Our framework first inference and storage all model responses and subsequently evaluate them on VisSim.

For Vllm inference:
```bash
cd stare_open/vllm_inference
python -m vllm_inference \
--input_path VisSim/${dataset_name} \
--output_path ./scripts/results/qwen25vl72b/${dataset_name}.jsonl \
--function folding \
--model_path Qwen/Qwen2.5-VL-72B-Instruct \
--tensor_parallel_size 4 
# function: ["folding", "va", "text_instruct", "mvideo", "nperspective"] The evaluation type of VisSim, depends on dataset.
```

For Vissim results eval:

```bash
cd stare_open
python  ./evaluator.py  \
--input_path stare_open/vllm_inference/scripts/results/${model_short_name}/${dataset_name}.jsonl \
--output_path vissim_eval/scripts/results/${model_short_name}/${dataset_name}.json \
--dataset_type ${function_type} 
# input_path: The path to your inference results you have generated.
# output_path: The path to your evaluation results
# dataset_type: ["folding", "va", "text_instruct", "mvideo", "nperspective"] same as "function" mentioned above.
```
