

from vllm import LLM, SamplingParams
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
import sys
import os
import datasets
from datasets import load_dataset
from vllm_inference.inferencer import inferencer_dict
from vllm_inference.utils.debug import remote_breakpoint

@dataclass
class VllmInferenceArguments:
    model_path: str = field(default="path/to/Qwen/QvQ")
    function: Optional[str] = field(default="gemini_mmrollout") 
    input_path: Optional[str] = field(default="mr_eval/scripts/data/eval_data.jsonl")
    output_path: Optional[str] = field(default="mr_eval/scripts/data/eval_data.jsonl")
    image_dir_path: Optional[str] = field(default=None)
    dataset_cache_dir: Optional[str] = field(default="/mnt/petrelfs/share_data/songmingyang/data/mm/reasoning/vissim")
    batch_size: Optional[int] = field(default=100)
    gpu_memory_utilization: Optional[float] = field(default=0.8)
    cpu_offload_gb: Optional[int] = field(default=0)
    max_model_len: Optional[int] = field(default=None)
    truncate_prompt_tokens: Optional[int] = field(default=None)
    
    max_images: Optional[int] = field(default = None)
    temperature: Optional[float] = field(default = 0)
    top_p: Optional[float] = field(default = 1)
    top_k: Optional[int] = field(default = -1)
    max_output_tokens: Optional[int] = field(default=65536)
    tensor_parallel_size: Optional[int] = field(default=1)  # 用于多卡推理的GPU数量


def parse_args():
    parser = HfArgumentParser((VllmInferenceArguments,))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, = parser.parse_args_into_dataclasses()
    
    return args

def main():
    args = parse_args()
    function = args.function
    inferencer_cls = inferencer_dict[function]
    inferencer = inferencer_cls(args)
    inferencer.respond_all()

if __name__ == "__main__":
    args = parse_args()
    # remote_breakpoint()
    function = args.function
    inferencer_cls = inferencer_dict[function]
    inferencer = inferencer_cls(args)
    inferencer.respond_all_batches()
    
    

