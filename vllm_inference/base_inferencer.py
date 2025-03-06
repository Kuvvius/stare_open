

from vllm import LLM, SamplingParams
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
import sys
import os
import datasets


@dataclass
class VllmInferenceArguments:
    model_path: str = field(default="path/to/Qwen/QvQ")
    function: Optional[str] = field(default="gemini_mmrollout")  # 可以复用相同的提示模板
    input_path: Optional[str] = field(default="mr_eval/scripts/data/eval_data.jsonl")
    output_path: Optional[str] = field(default="mr_eval/scripts/data/eval_data.jsonl")
    image_dir_path: Optional[str] = field(default="mr_eval/scripts/data/images")
    temperature: Optional[float] = field(default=1)
    top_p: Optional[float] = field(default=0.95)
    top_k: Optional[int] = field(default=40)
    max_output_tokens: Optional[int] = field(default=65536)
    tensor_parallel_size: Optional[int] = field(default=1)  # 用于多卡推理的GPU数量
    batch_size: Optional[int] = field(default=8)
    dataset_type: Optional[str] = field(default="clevr")

    def parse_args():
        parser = HfArgumentParser((VllmInferenceArguments,))
        
        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            args, = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            args, = parser.parse_args_into_dataclasses()
        
        return args



class VLLMInferencer():
    def __init__(self, args):
        self.args = args
        self.model_path = args.model_path
        self.function = args.function
        self.input_path = args.input_path
        self.output_path = args.output_path
        self.image_dir_path = args.image_dir_path
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.max_output_tokens = args.max_output_tokens
        self.tensor_parallel_size = args.tensor_parallel_size
        self.batch_size = args.batch_size
        self.dataset_type = args.dataset_type
    
    def load_data(self):
        dataset = datasets.load_dataset(self.input_path, self.dataset_type)
        return dataset
    
    def inference():
        pass


