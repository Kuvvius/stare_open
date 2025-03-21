
from vllm import LLM, SamplingParams
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
import sys
import os
import datasets
from datasets import load_dataset
import logging
from copy import deepcopy
import re
from r1_p.utils.utils import *

class VLLMQwQInferencer():
    def __init__(self, args):
        self.args = args
        self.model_path = args.model_path

        self.input_path = args.input_path
        self.output_path = args.output_path

        self.batch_size = args.batch_size
        
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.max_output_tokens = args.max_output_tokens
        self.tensor_parallel_size = args.tensor_parallel_size


        # remote_breakpoint()
        self.load_data()
        self.resume_from_ckpt()
        self.load_generation_config()
        self.load_model()
        
        
    def load_model(self):
        """加载模型，使用vllm的LLM类"""
        print(f"Loading model from {self.model_path} with {self.tensor_parallel_size} GPUs")

        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
        )
    
    def load_data(self):
        dataset = load_dataset(self.input_path,"main")["test"]
        self.meta_data = []
        for idx, item in enumerate(dataset):
            item_idx = f"gsm8k_test_{idx}"
            res = deepcopy(item)
            res["id"] = item_idx
            question = item["question"]
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Please think step by step, answer the question, and put your final answer within \\boxed{{}}\n"},
                {"role": "user", "content": f"Question: {question}\nAnswer:"},
            ]
            res["conversations"] = messages
            self.meta_data.append(res)
        print(f"Loaded {len(self.meta_data)} items")
        

    def resume_from_ckpt(self):
        """如果存在输出文件，则从上次中断处继续推理"""
        if os.path.exists(self.output_path):
            print(f"Resuming from checkpoint {self.output_path}")
            self.results = process_jsonl(self.output_path)
            self.processed_ids = {item["id"] for item in self.results}
            renewed_data = [item for item in self.meta_data if item["id"] not in self.processed_ids]
            print(f"Processed: {len(self.processed_ids)}, Remaining: {len(renewed_data)}")
            self.meta_data = renewed_data
        else:
            print("No checkpoint found, starting from scratch")
            
        
    def load_generation_config(self):
        """设置生成参数配置"""
        self.sampling_params = SamplingParams(
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            max_tokens=self.args.max_output_tokens,
        )
    
    def write_output_item(self, item):
        """将推理结果写入输出文件"""
        item.pop("conversations", None)
        append_jsonl(item, self.output_path)
    
    def respond_batch(self, batch) -> None:
        conversations = [item["conversations"] for item in batch]
        try:
            outputs = self.model.chat(
                conversations,
                sampling_params = self.sampling_params,
                use_tqdm = True,
            )
            
            for output_idx, output, batch_item in zip(range(len(outputs)), outputs, batch):
                output_text = output.outputs[0].text
                output_length = len(output.outputs[0].token_ids)  
                input_length = len(output.prompt_token_ids)
                output_ids = output.outputs[0].token_ids
                input_ids = output.prompt_token_ids
                output_meta = output
                batch_item["response"] = output_text
                batch_item["output_length"] = output_length
                batch_item["input_length"] = input_length
                # batch_item["output_meta"] = output_meta
                batch_item["output_ids"] = output_ids
                batch_item["input_ids"] = input_ids
                # breakpoint()
                res = deepcopy(batch_item)
                self.write_output_item(res)
        except Exception as e:
            print(f"Error in responding batch: {e}")
        
        
    def respond_all_batches(self) -> None:
        for i in range(0, len(self.meta_data), self.batch_size):
            batch = self.meta_data[i:i+self.batch_size]
            self.respond_batch(batch)
            
    def respond_all(self,) -> None:
        """批量处理数据进行推理"""
        batch = self.meta_data
        self.respond_batch(batch)
        
@dataclass
class VllmInferenceArguments:
    model_path: str = field(default="path/to/Qwen/QvQ")
    input_path: Optional[str] = field(default="mr_eval/scripts/data/eval_data.jsonl")
    output_path: Optional[str] = field(default="mr_eval/scripts/data/eval_data.jsonl")

    batch_size: Optional[int] = field(default=100)
    

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
    inferencer = VLLMQwQInferencer(args)
    inferencer.respond_all_batches()

if __name__ == "__main__":
    main()
    