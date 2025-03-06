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

caption_prompt = """
There is a question about the image or figure. Please describe the fine-grained content of the image or figure based 
on this question, including scenes, objects, relationships, and any text present.\nPlease note that you do not need 
to answer this question directly, just describe the information of this picture.\nQuestion:
"""


def get_response(args, messages):
    client = OpenAI(api_key=args.api_key)
    attempt = 0

    while attempt < 5:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
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


def create_message(text_query, pil_image):
    img_base64 = encode_image_to_base64(pil_image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }
                },
                {"type": "text", "text": text_query}
            ]
        }
    ]
    return messages





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='mm-reasoning/EMMA-test100')
    # parser.add_argument('--subject', nargs='+', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--output_path', type=str,
                        default='results/testmini/gpt4o_math_caption.json')
    parser.add_argument('--rerun', action='store_true', help='rerun the answer generation')
    parser.add_argument('--save_every', type=int, default=5, help='save response every n problems')
    # Remote model
    parser.add_argument('--model', type=str, default="chatgpt-4o-latest", help='llm engine',
                        choices=['chatgpt-4o-latest', 'claude-3-5-sonnet-latest', 'gemini-2.0-flash-exp',
                                 'gemini-2.0-flash-thinking-exp-1219'])
    parser.add_argument('--api_key', type=str, default='')
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--temperature', type=float, default=0)

    args = parser.parse_args()

    subject = ['Math']
    logging.info(f"Loading dataset {args.dataset_name}, subject: {subject}")
    sub_dataset_list = []
    for subj in subject:
        sub_dataset = load_dataset(args.dataset_name, subj, split=args.split)
        sub_dataset_list.append(sub_dataset)
    dataset = concatenate_datasets(sub_dataset_list)

    if os.path.exists(args.output_path):
        logging.info("Results already exists.")
        logging.info(f"Reading {args.output_path}")
        with open(args.output_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    skip_pids = []
    if not args.rerun and results:
        for pid, data in results.items():
            if 'caption' in data and verify_response(data['caption']):
                skip_pids.append(pid)

        if len(skip_pids) > 0:
            logging.info(
                f"Found existing results file with {len(skip_pids)} problems with valid captions. Skipping these problems...")

    logging.info(f"Starting to generate.....")

    for idx, entry in enumerate(tqdm(dataset)):
        pid = entry['pid']
        if skip_pids and pid in skip_pids:
            continue
        sample = entry.copy()

        logging.info(f"Generating caption for sample {pid}...")

        text_query = caption_prompt + sample['question'].replace("<image_1>", " ")
        messages = create_message(text_query, sample['image_1'])

        for i in range(1, 6):
            sample.pop(f'image_{i}')

        try:
            response = get_response(args, messages)
            results[pid] = sample
            results[pid]['caption'] = response
        except Exception as e:
            logging.error(f"Error in generating answer for {pid}")
            logging.error(e)
            results[pid] = sample
            results[pid]['error'] = str(e)

        if idx == 2 or (idx % args.save_every == 0 and idx > 0) or idx == len(dataset) - 1:
            try:
                with open(args.output_path, 'w') as f:
                    f.write(json.dumps(results, indent=2))
                logging.info(f"Save results to {args.output_path}")
            except Exception as e:
                logging.info(f"Error in saving {args.output_path}")
                logging.info(e)

    with open(args.output_path, 'w') as f:
        f.write(json.dumps(results, indent=2))
    logging.info(f"Save results to {args.output_path}")

    logging.info("End Generation......")


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












