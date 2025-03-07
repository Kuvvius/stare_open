

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
from ..utils.utils import *
from ..utils.debug import remote_breakpoint
import re

logger = logging.getLogger(__name__)

class VLLMBaseInferencer():
    def __init__(self, args):
        self.args = args
        self.model_path = args.model_path
        self.function = args.function
        self.input_path = args.input_path
        self.output_path = args.output_path
        self.image_dir_path = args.image_dir_path
        self.dataset_cache_dir = args.dataset_cache_dir
        self.batch_size = args.batch_size
        
        self.truncate_prompt_tokens = args.truncate_prompt_tokens
        self.max_model_len = args.max_model_len
        self.cpu_offload_gb = args.cpu_offload_gb
        self.gpu_memory_utilization = args.gpu_memory_utilization
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.max_output_tokens = args.max_output_tokens
        self.tensor_parallel_size = args.tensor_parallel_size
        self.max_images = {"image": args.max_images}

        # remote_breakpoint()
        self.load_data()
        self.resume_from_ckpt()
        self.load_generation_config()
        self.load_model()
        
        
    def load_model(self):
        """加载模型，使用vllm的LLM类"""
        logger.info(f"Loading model from {self.model_path} with {self.tensor_parallel_size} GPUs")

        self.model = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=True,
            limit_mm_per_prompt=self.max_images,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len = self.max_model_len,
        )
    
    def load_data(self):
        pass

    def resume_from_ckpt(self):
        """如果存在输出文件，则从上次中断处继续推理"""
        if os.path.exists(self.output_path):
            logger.info(f"Resuming from checkpoint {self.output_path}")
            self.results = process_jsonl(self.output_path)
            self.processed_ids = {item["qid"] for item in self.results}
            renewed_data = [item for item in self.meta_data if item["qid"] not in self.processed_ids]
            logger.info(f"Processed: {len(self.processed_ids)}, Remaining: {len(renewed_data)}")
            self.meta_data = renewed_data
        else:
            logger.info("No checkpoint found, starting from scratch")
            
        
    def load_generation_config(self):
        """设置生成参数配置"""
        self.sampling_params = SamplingParams(
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            top_k=self.args.top_k,
            max_tokens=self.args.max_output_tokens,
            truncate_prompt_tokens = self.truncate_prompt_tokens
        )
    
    def write_output_item(self, item):
        """将推理结果写入输出文件"""
        item.pop("conversations", None)
        append_jsonl(item, self.output_path)
    
    def respond_batch(self, batch) -> None:
        conversations = [item["conversations"] for item in batch]
        outputs = self.model.chat(
            conversations,
            sampling_params = self.sampling_params,
            use_tqdm = True,
        )
        
        for output_idx, output, batch_item in zip(range(len(outputs)), outputs, batch):
            output_text = output.outputs[0].text
            batch_item["response"] = output_text
            res = deepcopy(batch_item)
            self.write_output_item(res)
    
    def respond_all_batches(self) -> None:
        for i in range(0, len(self.meta_data), self.batch_size):
            batch = self.meta_data[i:i+self.batch_size]
            self.respond_batch(batch)
            
    def respond_all(self,) -> None:
        """批量处理数据进行推理"""
        batch = self.meta_data
        self.respond_batch(batch)


class FoldingInferencer(VLLMBaseInferencer):
    def load_data(self):
        """加载输入数据"""
        if self.dataset_cache_dir is not None:
            dataset = load_dataset(self.input_path, cache_dir=self.dataset_cache_dir)["train"]
        else:
            dataset = load_dataset(self.input_path)["train"]
        max_images = 0
        self.meta_data = []
        for item in dataset:
            qid = item["qid"]
            images = item["images"]
            question = item["question"]
            answer = item["answer"]
            question_info = item["question_info"]
            item_type = item["type"]
            choices = item["choices"]
            max_images = max(max_images, len(images))
            
            processed_question = question
            query_parts = []

            # Check if the question contains image placeholders
            if "<image_" in processed_question:
                for i, image in enumerate(images):
                    if f"<image_{i}>" in processed_question:
                        prefix, processed_question = processed_question.split(f"<image_{i}>", 1)
                        if prefix:
                            query_parts.append({"type": "text", "text": prefix})
                        query_parts.append({"type": "image_url", "image_url": {"url": pil_to_base64(image)}})
                
                # Add any remaining text
                if processed_question:
                    query_parts.append({"type": "text", "text": processed_question})
            else:
                # If no image placeholders, add the question text first, then images
                if question:
                    query_parts.append({"type": "text", "text": question})
                
                # Add all images
                for image in images:
                    query_parts.append({"type": "image_url", "image_url": {"url": pil_to_base64(image)}})

            # Add instruction to answer with boxed format
            query_parts.append({"type": "text", "text": "Think step-by-step, and then put your final answer in \"\\boxed{}\"."})
            
            initial_user_messages = [{
                    "role": "user",
                    "content": query_parts,
            }]
            
            res = dict(
                qid=qid,
                question=question,
                answer=answer,
                question_info=question_info,
                type=item_type,
                choices=choices,
                conversations=initial_user_messages,
            )
            self.meta_data.append(res)
        if self.max_images is None or \
                (isinstance(self.max_images, dict) and "image"  in self.max_images and self.max_images["image"] is None):
            self.max_images = {"image": max_images}
        
       

class VissimvaInferencer(VLLMBaseInferencer):
    def load_data(self):
        """加载VisSim VA数据集"""
        if self.dataset_cache_dir is not None:
            dataset = load_dataset(self.input_path, cache_dir=self.dataset_cache_dir)["train"]
        else:
            dataset = load_dataset(self.input_path)["train"]
        
        max_images = 0
        self.meta_data = []
        
        for item in dataset:
            qid = item["qid"]
            A_image = item["A_image"]
            B_image = item["B_image"]
            choice_image = item["choices"]
            question_info = json.loads(item["question_info"])
            # breakpoint()
            question = question_info.get("question", "")
            if question is None:
                print(f"Question is None for qid {qid}")
                
            answer = item["answer"]
            difficulty_level = item["difficulty_level"]
            
            # 图片数量计算
            images_count = 3  # A_image, B_image, choice_image
            max_images = max(max_images, images_count)
            
            # 处理问题文本
            prefix, question_part = question.strip().split("<question_image>")
            middle, question_part = question_part.split("<image_for_B>")
            if "<answer_choices>" in question_part:
                suffix, question_part = question_part.split("<answer_choices>")
            else:
                suffix = question_part
                question_part = ""
            contents = [
                    {"type": "text", "text": f"{prefix}"},
                    {"type": "image_url", "image_url": {"url": pil_to_base64(A_image)}},
                    {"type": "text", "text": middle},
                    {"type": "image_url", "image_url": {"url": pil_to_base64(B_image)}},
                    {"type": "text", "text": suffix},
                    {"type": "image_url", "image_url": {"url": pil_to_base64(choice_image)}},
            ]
            if len(question_part) > 0:
                contents.append({"type": "text", "text": question_part})
            contents.append({"type": "text", "text": "Please first solve the problem step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\""})
            initial_user_messages = [{
                "role": "user",
                "content": contents,
            }]  
            
            
            
            # 创建结果字典
            res = dict(
                qid=qid,
                question=question,
                answer=answer,
                question_info=question_info,
                difficulty_level=difficulty_level,
                conversations=initial_user_messages,
                transformations=item["transformations"],
                answer_info = item["answer_info"]
            )
            self.meta_data.append(res)
        
        if self.max_images is None or \
                (isinstance(self.max_images, dict) and "image" in self.max_images and self.max_images["image"] is None):
            self.max_images = {"image": max_images}
        # breakpoint()
        # breakpoint()


class VissimTextinstInferencer(VLLMBaseInferencer):
    def load_data(self):
        """加载VisSim TextInst数据集"""
        if self.dataset_cache_dir is not None:
            dataset = load_dataset(self.input_path, cache_dir=self.dataset_cache_dir)["train"]
        else:
            dataset = load_dataset(self.input_path)["train"]
        
        max_images = 0
        self.meta_data = []
        
        for item in dataset:
            qid = item["qid"]
            images = item["images"][:-1]  # Exclude the last image which is handled separately
            choice_image = item["choices"]
            question = item["question"]
            answer = item["answer"]
            difficulty_level = item["difficulty_level"]
            
            # 计算图片总数
            max_images = max(max_images, len(images) + 1)  # +1 for choice_image
            
            # 处理问题文本和图片
            query_parts = []
            question_text = question
            
            # 使用类似的逻辑处理文本和图片的交替
            for i, image in enumerate(images):
                if i == 0:
                    prefix, question_text = question_text.strip().split("<shapeB_image>", 1)
                else:
                    prefix, question_text = question_text.split(f"<shapeB_step_{i-1}>", 1)
                    
                if prefix:
                    query_parts.append({"type": "text", "text": prefix})
                query_parts.append({"type": "image_url", "image_url": {"url": pil_to_base64(image)}})
            
            # 处理剩余文本，移除任何剩余的占位符
            remaining_text = re.sub(r'<shapeB_step_\d+>', '', question_text)
            if remaining_text:
                query_parts.append({"type": "text", "text": remaining_text})
            
            # 添加选择图片
            query_parts.append({"type": "image_url", "image_url": {"url": pil_to_base64(choice_image)}})
            
            # 添加指令
            query_parts.append({"type": "text", "text": "Please first solve the problem step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\""})
            
            # 构造对话格式
            initial_user_messages = [{
                "role": "user",
                "content": query_parts,
            }]
            
            # 创建结果字典
            res = dict(
                qid=qid,
                question=question,
                answer=answer,
                difficulty_level=difficulty_level,
                conversations=initial_user_messages,
                transformations=item["transformations"],
                question_info=item["question_info"],
                answer_info=item["answer_info"],
            )
            self.meta_data.append(res)
        
        if self.max_images is None or \
                (isinstance(self.max_images, dict) and "image" in self.max_images and self.max_images["image"] is None):
            self.max_images = {"image": max_images}
    
        # breakpoint()
        # breakpoint()

class NperspectiveInferencer(VLLMBaseInferencer):
    def load_data(self):
        """加载NEWPerspective数据集"""
        if self.dataset_cache_dir is not None:
            dataset = load_dataset(self.input_path, cache_dir=self.dataset_cache_dir)["train"]
        else:
            dataset = load_dataset(self.input_path)["train"]
        
        max_images = 0
        self.meta_data = []
        
        for item in dataset:
            qid = item["id"]
            question_image = item["topdown"]
            choice_image = item["choices"]
            # 计算图片总数
            max_images = max(max_images, 2)  # 2 images per example
            
            # 构建内容
            query_parts = [
                {"type": "text", "text": "<Image 1>:"},
                {"type": "image_url", "image_url": {"url": pil_to_base64(question_image)}},
                {"type": "text", "text": "<Image 2>:"},
                {"type": "image_url", "image_url": {"url": pil_to_base64(choice_image)}},
                {"type": "text", "text": "<Image 1> shows an image from the top of a scene with a red square indicating an agent and a red arrow indicating the agent's direction of view.\nSelect from the <Image 2> which image represents the agent's view. Please first solve the problem step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\""}
            ]
            
            # 构造对话格式
            initial_user_messages = [{
                "role": "user",
                "content": query_parts,
            }]
            
            # 创建结果字典
            res = dict(
                qid=qid,
                conversations=initial_user_messages
            )
            
            # 添加其他可能有用的字段
            if "answer" in item:
                res["answer"] = item["answer"]
            
            self.meta_data.append(res)
        
        if self.max_images is None or \
                (isinstance(self.max_images, dict) and "image" in self.max_images and self.max_images["image"] is None):
            self.max_images = {"image": max_images}
            
            
class MvideoInferencer(VLLMBaseInferencer):
    def load_data(self):
        """加载MVideo数据集"""
        if self.dataset_cache_dir is not None:
            dataset = load_dataset(self.input_path, cache_dir=self.dataset_cache_dir)["train"]
        else:
            dataset = load_dataset(self.input_path)["train"]
        
        max_images = 0
        self.meta_data = []
        
        for item in dataset:
            qid = item["id"]
            question_image = item["question"]
            choice_image = item["choices"]
            
            # 计算图片总数
            max_images = max(max_images, 2)  # 2 images per example
            
            # 构建查询部分
            query_parts = [
                {"type": "text", "text": "<Image 1>:"},
                {"type": "image_url", "image_url": {"url": pil_to_base64(question_image)}},
                {"type": "text", "text": "<Image 2>:"},
                {"type": "image_url", "image_url": {"url": pil_to_base64(choice_image)}},
                {"type": "text", "text": "You see 4 sequential frames of a video in <Image 1>, but one is missing (marked with '?').\nChoose which of the images in <Image 2> correctly fills the missing frame.\nRemember, the camera only moves in one direction (left or right) in the video. Please first solve the problem step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\""}
            ]
            
            # 构造对话格式
            initial_user_messages = [{
                "role": "user",
                "content": query_parts,
            }]
            
            # 创建结果字典
            res = dict(
                qid=qid,
                conversations=initial_user_messages
            )
            
            # 添加其他可能有用的字段
            if "answer" in item:
                res["answer"] = item["answer"]
            
            self.meta_data.append(res)
        
        if self.max_images is None or \
                (isinstance(self.max_images, dict) and "image" in self.max_images and self.max_images["image"] is None):
            self.max_images = {"image": max_images}