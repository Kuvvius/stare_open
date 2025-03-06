import yaml
import json
import random
import re
import logging


def load_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            return yaml_dict
        except yaml.YAMLError as exc:
            print(exc)
            return None


def verify_response(response):
    if isinstance(response, str):
        response = response.strip()
    if response == "" or response is None:
        return False
    if "Response Error" in response:
        return False
    return True


def is_number(s):
    if isinstance(s, dict):
        return False

    try:
        float(s)
        return True
    except ValueError:
        pass

    return False

def build_scoring_query(sample, config, total_num):
    """Build the scoring query by combining the scoring_prompt and query. The <image_n> token is still there"""
    query = sample["query"]
    scoring_prompt = config["Scoring_prompt"]
    res_dict = {}

    scoring_query = query
    for num in range(total_num):
        if sample[f'response_{num}']:
            response = sample[f'response_{num}']
        else:
            response = "no response"
        scoring_query = scoring_query + f"\nResponse{{{num + 1}}}:\n{response}\n"
    scoring_query = scoring_query + scoring_prompt

    res_dict['scoring_query'] = scoring_query.strip()

    # append existing key and value in data
    res_dict.update(sample)
    return res_dict

def extract_score_list(text):

    pattern1 = r"inalscore(?:\{(\d+)\})\{[^\}]+\}\{(\d+)\}"
    pattern2 = r"inalscore(?:(\d+))\{[^\}]+\}\{(\d+)\}"
    json_pattern = r'```json\s*([\s\S]*?)\s*```'

    # 创建列表并初始化为 None，长度为 8（因为最多bo8）
    scores = [0, 0, 0, 0, 0, 0, 0, 0]

    for match in re.finditer(pattern2, text):
        response_number = int(match.group(1)) if match.group(1) else None  # 捕获 response_number
        score = int(match.group(2)) if match.group(2) else None  # 捕获 score
        if response_number:
            scores[response_number - 1] = score  # 将 score 存储在对应的 response_number 位置

    for match in re.finditer(pattern1, text):
        response_number = int(match.group(1)) if match.group(1) else None  # 捕获 response_number
        score = int(match.group(2)) if match.group(2) else None  # 捕获 score
        if response_number is not None:
            scores[response_number - 1] = score  # 将 score 存储在对应的 response_number 位置

    for match in re.finditer(json_pattern, text):
        json_str = match.group(1) if match.group(1) else None
        if json_str is not None:
            try:
                # Parse the JSON string into a Python object
                json_data = json.loads(json_str)
                for item in json_data:
                    if 'response_number' in item and 'final_score' in item and is_number(item['final_score']):
                        scores[item['response_number'] - 1] = int(item['final_score'])
                    elif 'response_number' in item and 'final_score' in item and 'score' in item['final_score'] and is_number(item['final_score']['score']):
                        scores[item['response_number'] - 1] = int(item['final_score']['score'])
            except json.JSONDecodeError as e:
                logging.info("Failed to parse JSON:", e)

    return scores
