from vllm_inference.utils.utils import *
from collections import defaultdict
import json
import os
import re
from tqdm import tqdm

import re

def is_number(s: str) -> bool:
    """
    判断一个字符串是否可以解释为数字（整数、浮点数、分数等）
    
    Args:
        s: 要检查的字符串
    
    Returns:
        bool: 如果字符串可以解释为数字则返回True，否则返回False
    """
    # 去除前后空白和可能的引号
    s = s.strip().strip('"\'')
    
    # 如果为空字符串，返回False
    if not s:
        return False
    
    # 尝试检查整数或浮点数
    try:
        # 移除千位分隔符(逗号)
        s = s.replace(',', '')
        # 移除可能的数字间下划线(如1_000_000)
        s = s.replace('_', '')
        float(s)
        return True
    except ValueError:
        # 检查是否为分数格式 (如 "3/4")
        if '/' in s:
            try:
                num, denom = s.split('/', 1)
                float(num) / float(denom)
                return True
            except (ValueError, ZeroDivisionError):
                pass
        
        # 检查是否为百分比
        if s.endswith('%'):
            try:
                float(s[:-1])
                return True
            except ValueError:
                pass
                
        return False


def find_math_answer(text: str) -> str:
    """
    从文本中提取数学问题的答案
    
    Args:
        text: 包含数学答案的文本
    
    Returns:
        str: 提取的数学答案，如果未找到则返回空字符串
    """
    # 删除文本中的LaTeX符号，简化处理
    text = re.sub(r'\$|\\\(|\\\)|\\\[|\\\]', '', text)
    
    # 常见答案引导词模式
    answer_patterns = [
        r'(?:answer|result|value)[^\w\d\.-]* (?:is|=|:)[^\w\d\.-]*([+-]?\d[\d,_]*(\.[\d]+)?)',  # "answer is 42"
        r'(?:答案|结果|等于|为|得)[^\w\d\.-]*(?:是|为|:|\s)[^\w\d\.-]*([+-]?\d[\d,_]*(\.[\d]+)?)',  # "答案是42"
        r'([+-]?\d[\d,_]*(\.[\d]+)?)(?=[^\d]|$)',  # 任何数字
        r'(\d+/\d+)',  # 分数
        r'(\d+\s+\d+/\d+)',  # 带分数
        r'(\d+%)'  # 百分比
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text)
        if matches:
            # 对于单纯的数字，取第一个match的第一个group
            if isinstance(matches[0], tuple):
                candidate = matches[0][0]
            else:
                candidate = matches[0]
            
            # 净化候选答案
            candidate = candidate.strip()
            
            return candidate
    
    # 如果常规搜索失败，尝试查找"最终答案"附近的数字
    final_answer_patterns = [
        r'(?:final answer|finally)[^\d]*([+-]?\d[\d,_]*(\.[\d]+)?)',
        r'(?:最终答案|最终结果)[^\d]*([+-]?\d[\d,_]*(\.[\d]+)?)'
    ]
    
    for pattern in final_answer_patterns:
        matches = re.findall(pattern, text.lower())
        if matches:
            if isinstance(matches[0], tuple):
                return matches[0][0].strip()
            else:
                return matches[0].strip()
    
    return ""

class VissimEvaluator:
    def __init__(
        self,
        input_path: str,
        output_path: str,
        dataset_type: str,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.dataset_type = dataset_type
        self.load_data()
        
        
    def evaluate(self):
        if self.dataset_type == "vissim":
            self.evaluate_vissim()
        else:
            raise NotImplementedError(f"Unsupported dataset type: {self.dataset_type}")


    def load_data(self):
        self.meta_data = process_jsonl(self.input_path)
        
    def extract_answer_from_model_response(self, model_response):
        from mathv_utils import is_number, find_math_answer
        model_answer = model_response.strip()
        model_answer = model_answer.replace("option", "")
        # model_answer = model_answer.replace("textbf", "text")
        
        for c in 'ABCDE':
            if model_answer.endswith(f" {c}.") or model_answer.endswith(f" ({c}).") or model_answer.startswith(f"{c}\n") or model_answer.startswith(f"({c})\n") or model_answer.startswith(f"({c}) {c}\n"):
                model_answer = c
        if is_number(model_answer.split('is ')[-1].rstrip('.')):
            model_answer = model_answer.split('is ')[-1].rstrip('.')
        if 'oxed{' not in model_answer:
            for flag in ['the final answer is', 'the answer is', 'the correct answer is', 'the answer should be']:
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split('\n')[0].split('. ')[0]
                flag = flag.replace('the', 'The')
                raw_model_answer = model_answer
                model_answer = model_answer.split(flag)[-1].strip()
                if flag in raw_model_answer:
                    model_answer = model_answer.split('\n')[0].split('. ')[0]
        elif model_answer.count('oxed{') > 1:
            model_answer = '\\boxed{' + model_answer.split('oxed{')[-1]
            
        model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e').replace('*', '').rstrip('.').lstrip(':').strip()
        return model_answer

    def evaluate_va(self):        
        # Initialize accuracy tracking
        acc_by_type = defaultdict(float)
        acc_by_difficulty = defaultdict(float)
        counts_by_difficulty = defaultdict(int)
        counts_by_type = defaultdict(int)
        
        # Process each example
        for example in tqdm(self.meta_data, total=len(self.meta_data)):
            qid = example['qid']
            difficulty_level = example.get('difficulty_level', 'unknown')
            
            # Extract variant from qid (e.g., "1steps_easy")
            variant = " ".join(qid.split("_")[1:-1])
            
            # Get prediction and ground truth
            response = example.get('response', '')
            pred = self.extract_answer_from_model_response(response)
            # pred = example.get('pred', '')
            gt_ans = example.get('answer', '').lower()
            
            # Update accuracy counters
            if "ans" not in variant:
                acc_by_difficulty[difficulty_level] += (pred.lower() == gt_ans)
                counts_by_difficulty[difficulty_level] += 1
            
            acc_by_type[variant] += (pred.lower() == gt_ans)
            counts_by_type[variant] += 1
        
        # Calculate and print accuracy metrics
        print("Accuracy by difficulty level:")
        difficulty_results = {}
        for k, v in acc_by_difficulty.items():
            accuracy = v / counts_by_difficulty[k] if counts_by_difficulty[k] > 0 else 0
            print(f"{k}: {accuracy:.4f}")
            difficulty_results[k] = accuracy
        
        print("\nAccuracy by variants:")
        variant_results = {}
        for k, v in acc_by_type.items():
            accuracy = v / counts_by_type[k] if counts_by_type[k] > 0 else 0
            print(f"{k}: {accuracy:.4f}")
            variant_results[k] = accuracy
        
        overall_acc = sum(acc_by_difficulty.values()) / sum(counts_by_difficulty.values()) if sum(counts_by_difficulty.values()) > 0 else 0
        print(f"\nOverall accuracy: {overall_acc:.4f}")
        
        # Save results
        results = {
            "overall_accuracy": overall_acc,
            "accuracy_by_difficulty": difficulty_results,
            "accuracy_by_variant": variant_results
        }
        
        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
