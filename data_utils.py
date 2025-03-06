import yaml
import json


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


def build_query(sample, config):
    """Build the text query by combining the context, question and options. The <image_n> token is still there"""

    question = sample['question']
    res_dict = {}
    
    empty_prompt_sample_structure = config['multi_choice_format']
    empty_prompt = empty_prompt_sample_structure
    res_dict['query'] = empty_prompt.format(question=question)

    gt_choice = sample['answer'].lower()
    answer_choices =sample['choices']
    gt_ans = answer_choices[int(ord(gt_choice) - ord('a'))]
    
    res_dict['gt_ans'] = gt_ans
    res_dict['gt_choice'] = gt_choice

    # append existing key and value in data
    res_dict.update(sample)
    return res_dict
