import fire
from PIL import Image
import os
import json
from google import genai
from google.genai import types

# load API key
with open('../aux_data/credentials/gemini_key.txt', 'r') as f:
    GEMINI_API_KEY = f.read().strip()
client = genai.Client(api_key=GEMINI_API_KEY)

from PIL import Image

def gemini_call_single(query, model='gemini-2.0-flash-exp', temp=0):
    # import cv2
    import requests
    import time
    import json

    # print(query)

    while True:
        print('\n(trying...) ->', )
        try:
            
            content = client.models.generate_content(
                model=model,
                contents=query,
                config = types.GenerateContentConfig(
                    temperature=temp,
                )
            )
            

        except Exception as e_msg:
            content = '[ERROR] ' + str(e_msg)
 
        if isinstance(content, str):
            content = '[ERROR] ' + content.lower()
            if 'exceeded call rate limit' in content or 'exhausted' in content:
                # retry for unacceptable response
                print('\n(retry later in 5 seconds...) ->', content)
                if "thinking" in model:
                    time.sleep(10)
                else:
                    time.sleep(5)
                continue
            else:
                print('\n(retry later...) ->', content)
        elif content.text is None:
            temp += 0.1
            continue
        else:
            break

    
    ########################################
    
    # print(responseJson["choices"][0]["message"]["content"])
    return content.text


def extract_answer_from_model_response(model_response):
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


# convert PIL image to base64
def pil_to_base64(pil_image):
    import io
    import base64
    img_byte_arr = io.BytesIO()
    pil_image.save(img_byte_arr, format='PNG')
    img_encoded_str = base64.b64encode(img_byte_arr.getvalue()).decode('ascii')
    return img_encoded_str


def test_gemini_on_VisSim(dataset_name, output_dir):
    # load hf dataset

    from datasets import load_dataset
    dataset = load_dataset(f"VisSim/{dataset_name}")
    dataset = dataset['train']

    from collections import defaultdict
    acc_by_type = defaultdict(float)
    acc_by_difficulty = defaultdict(float)
    counts_by_difficulty = defaultdict(int)
    counts_by_type = defaultdict(int)

    # create output dir
    os.makedirs(os.path.join(output_dir, dataset_name), exist_ok=True)
    from tqdm import tqdm
    for i, example in tqdm(enumerate(dataset)):
        qid = example['qid']
        difficulty_level = example['difficulty_level']
        if 'transformation_type' not in example:
            transform_type = 'unknown'
        else:
            transform_type = example['transformation_type']

        output_path = os.path.join(output_dir, dataset_name, f"{qid}.json")
        if os.path.exists(output_path):

            output = json.load(open(output_path))
            pred = output['pred']
            gt_ans = output['gt_ans']
        else:
            question_image = example['questions']
            choice_image = example['choices']
            query = [
                "Look at how the shapes change in the first row in <Image 1>. Which option in <image 2> fits in the question mark in second row of <Image 1>?\n",
                "<Image 1>:",
                question_image,
                "<Image 2>:",
                choice_image,
                "Please first solve the problem step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\""
            ]

            response = gemini_call_single(query)

            pred = extract_answer_from_model_response(response)

            gt_ans = example['answer']

            output = {
                "qid": qid,
                "pred": pred,
                "gt_ans": gt_ans.lower(),
                "response": response
            }
            with open(output_path, "w") as f:
                json.dump(output, f)
        acc_by_difficulty[difficulty_level]+= pred.lower() == gt_ans.lower()
        counts_by_difficulty[difficulty_level] += 1
        acc_by_type[transform_type] += pred.lower() == gt_ans.lower()
        counts_by_type[transform_type] += 1

    # print accuracy
    print("Accuracy by difficulty level:")
    for k, v in acc_by_difficulty.items():
        print(f"{k}: {v/counts_by_difficulty[k]}")

    print("Accuracy by transform type:")
    for k, v in acc_by_type.items():
        print(f"{k}: {v/counts_by_type[k]}")
    
    overall_acc = sum(acc_by_difficulty.values())/sum(counts_by_difficulty.values())
    print(f"Overall accuracy: {overall_acc}")


def test_gemini_on_VisSim_va(dataset_name, output_dir, model='gemini-2.0-flash-exp', max_tokens=2048, debug=False):

    from datasets import load_dataset
    dataset = load_dataset(f"VisSim/{dataset_name}")
    dataset = dataset['train']
    d_index = list(range(len(dataset)))
    if debug:
        import random
        random.seed(42)
        random.shuffle(d_index)
        d_index = d_index[:150]
    

    from collections import defaultdict
    acc_by_type = defaultdict(float)
    acc_by_difficulty = defaultdict(float)
    counts_by_difficulty = defaultdict(int)
    counts_by_type = defaultdict(int)

    # create output dir
    os.makedirs(os.path.join(output_dir, model, dataset_name), exist_ok=True)
    from tqdm import tqdm
    for i in tqdm(d_index, total=len(d_index)):
        example = dataset[i]
        qid = example['qid']
        difficulty_level = example['difficulty_level']
        variant = " ".join(qid.split("_")[1:-1])
        output_path = os.path.join(output_dir, model, dataset_name, f"{qid}.json")
        if os.path.exists(output_path):

            output = json.load(open(output_path))
            pred = output['pred']
            gt_ans = output['gt_ans']
        else:
            A_image = example['A_image']
            B_image = example['B_image']
            question_info = json.loads(example['question_info'])
            question = question_info['question']
            choice_image = example['choices']
            query = []
            prefix, question = question.strip().split("<question_image>")
            query.append(prefix)
            query.append(A_image)
            prefix, question = question.split("<image_for_B>")
            query.append(prefix)
            query.append(B_image)
            prefix, question = question.split("<answer_choices>")   
            query.append(prefix)
            query.append(choice_image)
            if len(question) > 0:
                query.append(question)
            query.append("Please first solve the problem step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\"")
            # print(query)
            response = gemini_call_single(query, model=model)

            pred = extract_answer_from_model_response(response)

            gt_ans = example['answer']

            output = {
                "qid": qid,
                "pred": pred,
                "gt_ans": gt_ans.lower(),
                "response": response
            }
            with open(output_path, "w") as f:
                json.dump(output, f)
        if "ans" not in variant:
            acc_by_difficulty[difficulty_level]+= pred.lower() == gt_ans.lower()
            counts_by_difficulty[difficulty_level] += 1
        acc_by_type[variant] += pred.lower() == gt_ans.lower()
        counts_by_type[variant] += 1

    # print accuracy
    print("Accuracy by difficulty level:")
    for k, v in acc_by_difficulty.items():
        print(f"{k}: {v/counts_by_difficulty[k]}")

    print("Accuracy by variants:")
    for k, v in acc_by_type.items():
        print(f"{k}: {v/counts_by_type[k]}")
    
    overall_acc = sum(acc_by_difficulty.values())/sum(counts_by_difficulty.values())
    print(f"Overall accuracy: {overall_acc}")


def test_gemini_on_VisSim_text_inst(dataset_name, output_dir, model='gemini-2.0-flash-exp', max_tokens=2048,debug=False):

    from datasets import load_dataset
    dataset = load_dataset(f"VisSim/{dataset_name}")
    dataset = dataset['train']
    d_index = list(range(len(dataset)))
    if debug:
        import random
        random.seed(42)
        random.shuffle(d_index)
        d_index = d_index[:300]


    from collections import defaultdict
    acc_by_type = defaultdict(float)
    acc_by_difficulty = defaultdict(float)
    counts_by_difficulty = defaultdict(int)
    counts_by_type = defaultdict(int)

    # create output dir
    os.makedirs(os.path.join(output_dir, model, dataset_name), exist_ok=True)
    from tqdm import tqdm
    for i in tqdm(d_index, total=len(d_index)):
        example = dataset[i]
        qid = example['qid']
        difficulty_level = example['difficulty_level']
        variant = " ".join(qid.split("_")[1:-1])
        output_path = os.path.join(output_dir, model, dataset_name, f"{qid}.json")
        if os.path.exists(output_path):

            output = json.load(open(output_path))
            pred = output['pred']
            gt_ans = output['gt_ans']
        else:
            images = example['images'][:-1]
            # question_info = json.loads(example['question_info'])
            question = example['question']
            choice_image = example['choices']

            # use regex to parse the question and place the images in the right spots
            query = []
            for i, image in enumerate(images):
                if i == 0:
                    prefix, question = question.strip().split("<shapeB_image>")
                else:
                    prefix, question = question.split(f"<shapeB_step_{i-1}>")
                query.append(prefix)
                query.append(image)
            
            # replace the remaining <shapeB_image> with "" using regex
            import re
            # using wildcards to match the <shapeB_step_{i}> and replace it with ""
            query.append(re.sub(r'<shapeB_step_\d+>', '', question))

            query.append(choice_image)
            
            query.append("Please first solve the problem step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\"")
            # print(query)
            response = gemini_call_single(query, model=model)

            pred = extract_answer_from_model_response(response)

            gt_ans = example['answer']

            output = {
                "qid": qid,
                "pred": pred,
                "gt_ans": gt_ans.lower(),
                "response": response
            }
            with open(output_path, "w") as f:
                json.dump(output, f)
        if "ans" not in variant:
            acc_by_difficulty[difficulty_level]+= pred.lower() == gt_ans.lower()
            counts_by_difficulty[difficulty_level] += 1
        acc_by_type[variant] += pred.lower() == gt_ans.lower()
        counts_by_type[variant] += 1

    # print accuracy
    print("Accuracy by difficulty level:")
    for k, v in acc_by_difficulty.items():
        print(f"{k}: {v/counts_by_difficulty[k]}")

    print("Accuracy by variants:")
    for k, v in acc_by_type.items():
        print(f"{k}: {v/counts_by_type[k]}")
    
    overall_acc = sum(acc_by_difficulty.values())/sum(counts_by_difficulty.values())
    print(f"Overall accuracy: {overall_acc}")


def test_gemini_on_folding_nets(dataset_name, output_dir, model='gemini-2.0-flash-exp', max_tokens=2048,debug=False):

    from datasets import load_dataset
    dataset = load_dataset(f"VisSim/{dataset_name}")
    dataset = dataset['train']
    d_index = list(range(len(dataset)))
    if debug:
        import random
        random.seed(42)
        random.shuffle(d_index)
        d_index = d_index[:200]


    from collections import defaultdict
    pred_by_type = defaultdict(list)
    pred_by_difficulty = defaultdict(list)

    gt_by_type = defaultdict(list)
    gt_by_difficulty = defaultdict(list)


    # create output dir
    os.makedirs(os.path.join(output_dir, model, dataset_name), exist_ok=True)
    from tqdm import tqdm
    for i in tqdm(d_index, total=len(d_index)):
        example = dataset[i]
        qid = example['qid']
        variant = example['type']
        output_path = os.path.join(output_dir, model, dataset_name, f"{qid}.json")
        if os.path.exists(output_path):

            output = json.load(open(output_path))
            pred = output['pred']
            gt_ans = output['gt_ans']
            gt_choice = output['gt_choice']
        else:
            images = example['images']
            # question_info = json.loads(example['question_info'])
            question = example['question']


            # use regex to parse the question and place the images in the right spots
            query = []
            for i, image in enumerate(images):
                prefix, question = question.split(f"<image_{i}>")
                query.append(prefix)
                query.append(image)
            if len(question) > 0:
                query.append(question + "Think step-by-step, and then put your final answer in \"\\boxed{}\".")
            else:
                query.append("Think step-by-step, and then put your final answer in \"\\boxed{}\".")
            
            # check if the query has at least 1 image after parsing

            # print(query)
            response = gemini_call_single(query, model=model)

            pred = extract_answer_from_model_response(response)

            gt_choice = example['answer'].lower()
            answer_choices = example['choices']

            gt_ans = answer_choices[int(ord(gt_choice) - ord('a'))]

            output = {
                "qid": qid,
                "pred": pred,
                "gt_choice": gt_choice,
                "gt_ans": gt_ans.lower(),
                "response": response,
                "question": [q  if isinstance(q, str) else '<image>' for q in query]
            }
            with open(output_path, "w") as f:
                json.dump(output, f)

        correct = pred.lower() == gt_ans.lower() or pred.lower() == gt_choice.lower()
        pred_by_type[variant].append(pred.lower())
        gt_by_type[variant].append(gt_ans.lower())


    print("F1 by variants:")
    for k in pred_by_type.keys():
        from sklearn.metrics import f1_score
        print(f"{k}: {f1_score(gt_by_type[k], pred_by_type[k], average='weighted')}")

    print("Random Chance F1 by variants:")
    for k in pred_by_type.keys():
        from sklearn.metrics import f1_score
        import random
        random_pred = [random.choice(["yes", "no"]) for _ in range(len(gt_by_type[k]))]
        print(f"{k}: {f1_score(gt_by_type[k],random_pred, average='weighted')}")


def test_gemini_flash_thinking(debug=False):
    model = 'gemini-2.0-flash-thinking-exp-01-21'
    test_gemini_on_folding_nets('folding_nets_test', 'output_dir/gemini_response', model=model, debug=False)
    test_gemini_on_folding_nets('tangram_puzzle_test', 'output_dir/gemini_response', model=model, debug=False)
    test_gemini_on_VisSim_va('2d_va_test', 'output_dir/gemini_response', model=model, debug=False)
    test_gemini_on_VisSim_text_inst('2d_text_instruct_test', 'output_dir/gemini_response', model=model, debug=False)
    test_gemini_on_folding_nets('folding_nets_vissim_test', 'output_dir/gemini_response', model=model, debug=debug)
    test_gemini_on_folding_nets('tangram_puzzle_vissim_test', 'output_dir/gemini_response', model=model, debug=debug)
    test_gemini_on_VisSim_va('2d_va_vissim_test', 'output_dir/gemini_response', model=model, debug=debug)
    test_gemini_on_VisSim_text_inst('2d_text_instruct_vissim_test', 'output_dir/gemini_response', model=model, debug=debug)

def test_gemini_flash_exp(debug=False):
    model = 'gemini-2.0-flash-exp'
    test_gemini_on_folding_nets('folding_nets_test', 'output_dir/gemini_response', model=model, debug=False)
    test_gemini_on_folding_nets('tangram_puzzle_test', 'output_dir/gemini_response', model=model, debug=False)
    test_gemini_on_VisSim_va('2d_va_test', 'output_dir/gemini_response', model=model, debug=False)
    test_gemini_on_VisSim_text_inst('2d_text_instruct_test', 'output_dir/gemini_response', model=model, debug=False)
    test_gemini_on_folding_nets('folding_nets_vissim_test', 'output_dir/gemini_response', model=model, debug=debug)
    test_gemini_on_folding_nets('tangram_puzzle_vissim_test', 'output_dir/gemini_response', model=model, debug=debug)
    test_gemini_on_VisSim_va('2d_va_vissim_test', 'output_dir/gemini_response', model=model, debug=debug)
    test_gemini_on_VisSim_text_inst('2d_text_instruct_vissim_test', 'output_dir/gemini_response', model=model, debug=debug)

'''
Rate limited
'''
# def test_gemini_pro_exp(debug=False):
#     model = 'gemini-2.0-pro-exp-02-05'
#     test_gemini_on_folding_nets('folding_nets', 'output_dir/gemini_response', model=model, debug=False)
#     test_gemini_on_VisSim_va('2d_va', 'output_dir/gemini_response', model=model, debug=debug)
#     test_gemini_on_VisSim_text_inst('2d_text_inst', 'output_dir/gemini_response', model=model, debug=debug)
#     test_gemini_on_VisSim_va('2d_va_2steps', 'output_dir/gemini_response', model=model, debug=debug)
#     test_gemini_on_VisSim_text_inst('2d_text_instruct_2steps', 'output_dir/gemini_response', model=model, debug=debug)
#     test_gemini_on_VisSim_va('2d_va_4steps', 'output_dir/gemini_response', model=model, debug=debug)
#     test_gemini_on_VisSim_text_inst('2d_text_instruct_4steps', 'output_dir/gemini_response', model=model, debug=debug)


if __name__ == '__main__':
    fire.Fire()