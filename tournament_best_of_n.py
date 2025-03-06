import argparse
import os
import logging
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
import logging
import re
import base64
from io import BytesIO
import time
import random
import json
from scoring_utils import load_yaml, verify_response

from openai import OpenAI
import math


def get_response(args, messages):
    client = OpenAI(api_key=args.api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    attempt = 0

    while attempt < 3:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                # max_tokens=args.max_tokens,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed: {e}")

            if 'error' in str(e) and 'message' in str(e):
                error_message = str(e)
                if 'The server had an error processing your request.' in error_message:
                    sleep_time = 30
                    logging.error(f"Server error, retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                elif 'Please try again in ' in error_message:
                    sleep_time = float(error_message.split('Please try again in ')[1].split('s.')[0])
                    logging.error(f"Rate limit exceeded, retrying in {sleep_time * 2}s...")
                    time.sleep(sleep_time * 2)
                elif 'RESOURCE_EXHAUSTED' in error_message:
                    sleep_time = 30
                    logging.error(f"Gemini rate limit, retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                else:
                    print(e)
                    break
            attempt += 1

    return None

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def create_message(sample, query):
    all_contents = []
    matches = re.findall(r"<(image_\d+)>", query)
    split_text = re.split(r"<image_\d+>", query)
    for i, fragment in enumerate(split_text):
        if fragment.strip():
            all_contents.extend([
                {"type": "text", "text": fragment}
            ])
        if i < len(matches):
            if sample[matches[i]]:
                img_base64 = encode_image_to_base64(sample[matches[i]])
                all_contents.extend([
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    }
                ])
            else:
                logging.error(
                    f"The image token {matches[i]} is in the query, but there is no corresponding image provided by the data")

    messages = [
        {
            "role": "user",
            "content": all_contents
        }
    ]
    return messages


def init_json(input_data):
    for pid, data in input_data.items():
        data['Turn_0'] = []
        for i in range(16):
            if f'response_{i}' in data:
                data['Turn_0'].append(data[f'response_{i}'])
                data.pop(f'response_{i}')

    return input_data


def random_pair_list(responses_list, seed):
    random.seed(seed)
    random.shuffle(responses_list)
    return [responses_list[i:i + 2] for i in range(0, len(responses_list), 2)]


def get_better_response(args, sample, config, res1, res2):
    response_1 = f'\nResponse_1: {res1}\n'
    response_2 = f'Response_2: {res2}\n'
    score_prompt = config['Pairwise_Scoring_Prompt'].format(Query= sample['query'], Response_1=response_1, Response_2=response_2)
    messages = create_message(sample, score_prompt)
    response = get_response(args, messages)
    random.seed(args.seed)
    better_response = random.choice([res1, res2])
    if response:
        if 'oxed{response_1}' in response.lower():
            logging.info('res1 is better')
        elif 'oxed{response_2}' in response.lower():
            logging.info('res2 is better')
            better_response = res2
        elif response.lower().strip().endswith('response_1') or response.lower().strip().endswith('response_1}') or response.lower().strip().endswith('response_1**'):
            logging.info('res1 is better')
        elif response.lower().strip().endswith('response_2') or response.lower().strip().endswith('response_2}') or response.lower().strip().endswith('response_2**'):
            logging.info('res2 is better')
            better_response = res2
        else:
            logging.error(f'Gemini evaluation error:{response}, random select one')
            
        return better_response, response
    else:
        logging.error(f'Gemini does not generate a valid response, random select one')
        return better_response, "Gemini no response, random select one"
        


# dest_dir = '/mnt/petrelfs/haoyunzhuo/EMMA/EMMA_BAK_108/results/testmini/test-time-compute/thinking-121-tournament-Bo16'
# for root, dirs, files in os.walk('/mnt/petrelfs/haoyunzhuo/EMMA/EMMA_BAK_108/results/testmini/test-time-compute/thinking-121-tournament-Bo16'):
#     for file in files:
#         with open(os.path.join(root, file), 'r') as f:
#             data = json.load(f)
#         output_data = init_json(data)
#         with open(os.path.join(dest_dir, file), 'w') as f:
#             f.write(json.dumps(output_data, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='mm-reasoning/EMMA-mini')
    parser.add_argument('--subject', nargs='+', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--config_path', type=str, default="configs/scoring.yaml")
    parser.add_argument('--output_path', type=str,
                        default='results/testmini/test-time-compute/thinking-tournament-best-of-4/gemini-2.0-flash-thinking-exp-1219_Coding_4.json')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer generation')
    parser.add_argument('--total_num', type=int, default=4, help='pass@n')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # Remote model
    parser.add_argument('--model', type=str, default="gemini-2.0-flash-thinking-exp-1219", help='llm engine',
                        choices=['chatgpt-4o-latest', 'claude-3-5-sonnet-latest', 'gemini-2.0-flash-exp',
                                 'gemini-2.0-flash-thinking-exp-1219', 'gemini-2.0-flash-thinking-exp-1219-2', 'gemini-2.0-flash-thinking-exp-01-21'])
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=0)

    args = parser.parse_args()

    if os.path.exists(args.output_path):
        logging.info(f"Reading {args.output_path}")
        with open(args.output_path, 'r') as f:
            results = json.load(f)
    else:
        logging.error(f"{args.output_path} does not exist.")
        exit()

    logging.info(f"Loading dataset {args.dataset_name}, subject: {args.subject}")
    sub_dataset_list = []
    for subj in args.subject:
        sub_dataset = load_dataset(args.dataset_name, subj, split=args.split)
        sub_dataset_list.append(sub_dataset)
    dataset = concatenate_datasets(sub_dataset_list)

    # Load Config
    logging.info(f"Loading config")
    config = load_yaml(args.config_path)

    skip_pids = []
    if not args.rerun and results:
        for pid, problem in results.items():
            if 'best_response' in problem and verify_response(problem['best_response']):
                skip_pids.append(pid)

        if len(skip_pids) > 0:
            logging.info(
                f"Found existing results file with {len(skip_pids)} problems with valid responses. Skipping these problems..."
            )

    logging.info(f"Starting to generate.....")
    for idx, entry in enumerate(tqdm(dataset)):
        pid = entry['pid']
        if skip_pids and pid in skip_pids:
            continue
        sample = results[pid].copy()
        for i in range(1, 6):
            sample[f'image_{i}'] = entry[f'image_{i}']

        logging.info(f"Processing sample {pid}...")
        total_turn = int(math.log2(args.total_num))
        for turn in range(total_turn):
            logging.info(f"Generating {turn}/{total_turn - 1} for pid {pid}")
            response_list = sample[f'Turn_{turn}']
            response_list_pair = random_pair_list(response_list, args.seed)
            if len(response_list_pair) != 1:
                next_turn_list = []
                scoring_content_list = []
                for two_response_list in response_list_pair:
                    better_response, scoring_content  = get_better_response(args, sample, config, two_response_list[0], two_response_list[1])
                    next_turn_list.append(better_response)
                    scoring_content_list.append(scoring_content)
                sample[f'Turn_{turn+1}'] = next_turn_list
                sample[f'scoring_content_list_{turn}'] = scoring_content_list
            elif len(response_list_pair) == 1:
                best_response, scoring_content = get_better_response(args, sample, config, response_list_pair[0][0], response_list_pair[0][1])
                sample['best_response'] = best_response
                sample['scoring_content_fianl_round'] = scoring_content
                break

        if 'best_response' not in sample:
            final_turn = sample.get(f'Turn_{total_turn - 1}', [])
            if len(final_turn) == 1:
                sample['best_response'] = final_turn[0]
            elif len(final_turn) == 2:
                sample['best_response'], scoring_content = get_better_response(args, sample, config, final_turn[0], final_turn[1])
                sample['scoring_content_fianl_round'] = scoring_content

        for i in range(1, 6):
            sample.pop(f'image_{i}')

        results[pid] = sample

        with open(args.output_path, "w") as f:
            f.write(json.dumps(results, indent=2))
            logging.info(f"Partial results saved to {args.output_path}.")

    with open(args.output_path, "w") as f:
        f.write(json.dumps(results, indent=2))
        logging.info(f"Results saved to {args.output_path}.")


if __name__ == "__main__":
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="[%(name)s] %(message)s",
        datefmt="[%X]"
    )
    logger_blocklist = [
        "asyncio",
        "azure",
        "azureml",
        "datasets",
        "httpx",
        "httpcore",
        "filelock",
        "fsspec",
        "msal",
        "msrest",
        "openai",
        "PIL",
        "urllib3",
    ]
    for module in logger_blocklist:
        logging.getLogger(module).setLevel(logging.WARNING)

    main()












