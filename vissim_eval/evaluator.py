from vllm_inference.utils.utils import *
from collections import defaultdict
import json
from tqdm import tqdm
from sklearn.metrics import f1_score
import argparse
import numpy as np
from sklearn.metrics import f1_score

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
        if self.dataset_type == "va":
            self.evaluate_va()
        elif self.dataset_type == "text_instruct":
            self.evaluate_text_instruct()
        elif self.dataset_type == "folding":
            self.evaluate_folding_nets()
        elif self.dataset_type == "nperspective":
            self.evaluate_perspective()
        elif self.dataset_type == "mvideo":
            self.evaluate_video()
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
                model_answer = find_math_answer(model_answer).replace('(a)', 'a').replace('(b)', 'b').replace('(c)', 'c').replace('(d)', 'd').replace('(e)', 'e').replace('{a}', 'a').replace('{b}', 'b').replace('{c}', 'c').replace('{d}', 'd').replace('{e}', 'e').replace('*', '').rstrip('.').lstrip(':').strip()
                model_answer = model_answer.replace("{","").replace("}","")
                return model_answer
            
        if is_number(model_answer.split('is ')[-1].rstrip('.')):
            model_answer = model_answer.split('is ')[-1].rstrip('.')
        elif 'oxed{' not in model_answer:
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
        model_answer = model_answer.replace("{","").replace("}","")
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
    
    def evaluate_perspective(self):
        
        # Initialize metrics tracking
        all_gt = []
        all_pred = []
        
        # Process each example
        for example in tqdm(self.meta_data, total=len(self.meta_data)):
            # Get prediction and ground truth
            response = example.get('response', '')
            pred = self.extract_answer_from_model_response(response).lower()
            gt_ans = example.get('answer', '').lower()
            if gt_ans == "up":
                gt_ans = "a"
            elif gt_ans == "right":
                gt_ans = "b"
            elif gt_ans == "down":
                gt_ans = "c"
            elif gt_ans == "left":
                gt_ans = "d"
            else:
                print("Invalid GT answer")
                # import pdb; pdb.set_trace()
            # Ensure prediction is one of the expected answers
            if pred not in ['a', 'b', 'c', 'd']:
                print(f"Warning: Invalid prediction '{pred}', defaulting to 'a'")
                pred = 'a'
            
            # Store predictions and ground truths
            all_pred.append(pred)
            all_gt.append(gt_ans)
        
        # Calculate metrics
        f1_macro = f1_score(all_gt, all_pred, average='macro')
        accuracy = sum(p == g for p, g in zip(all_pred, all_gt)) / len(all_gt) if all_gt else 0
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        
        # Count answers by category
        answer_counts = {}
        for ans in ['a', 'b', 'c', 'd']:
            gt_count = sum(1 for g in all_gt if g == ans)
            pred_count = sum(1 for p in all_pred if p == ans)
            answer_counts[ans] = {"gt": gt_count, "pred": pred_count}
            print(f"Answer {ans.upper()}: GT={gt_count}, Pred={pred_count}")
        
        # Save results
        results = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "answer_counts": answer_counts,
            "total_samples": len(all_gt)
        }
        
        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_text_instruct(self):
        # Initialize accuracy tracking
        acc_by_type = defaultdict(float)
        acc_by_difficulty = defaultdict(float)
        counts_by_difficulty = defaultdict(int)
        counts_by_type = defaultdict(int)
        
        # Process each example
        for example in tqdm(self.meta_data, total=len(self.meta_data)):
            qid = example['qid']
            difficulty_level = example.get('difficulty_level', 'unknown')
            
            # Extract variant from qid
            variant = " ".join(qid.split("_")[2:-1])
            
            # Get prediction and ground truth
            response = example.get('response', '')
            pred = self.extract_answer_from_model_response(response)
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
            print(f"\tCount: {counts_by_type[k]}")
        
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
        
    def evaluate_folding_nets(self):
        # Initialize metrics tracking
        pred_by_type = defaultdict(list)
        gt_by_type = defaultdict(list)
        
        # Process each example
        for example in tqdm(self.meta_data, total=len(self.meta_data)):
            qid = example['qid']
            variant = example.get('type', 'unknown')
            
            # Get prediction and ground truth
            response = example.get('response', '')
            pred = self.extract_answer_from_model_response(response)
            gt_choice = example.get('answer', '').lower()
            
            
            # Get gt_ans from choices when available
            gt_ans = gt_choice
            if 'choices' in example:
                try:
                    gt_ans = example['choices'][ord(gt_choice) - ord('a')].lower()
                    gt_ans = gt_ans.split('.')[0].lower()
                except (IndexError, TypeError):
                    pass
            
            # Store predictions and ground truths by variant
            pred_by_type[variant].append(pred.lower())
            gt_by_type[variant].append(gt_ans.lower())
        
        # Calculate and print metrics
        print(f"F1 scores by variant:")
        variant_results = {}
        all_pred = []
        all_gt = []
        
        
        for variant, preds in pred_by_type.items():
            gts = gt_by_type[variant]
            if variant == 'color':
                gts = [g[0] for g in gts]  # Take first character for color variant
                
            # Calculate F1 score
            try:
                f1 = f1_score(gts, preds, average='weighted')
            except:
                f1 = 0.0
                
            # Calculate accuracy
            acc = sum(p == g for p, g in zip(preds, gts)) / len(gts) if gts else 0
            
            print(f"\tVariant {variant}: F1={f1:.4f}, Acc={acc:.4f}")
            print(f"\t\tSamples: {len(gts)}")
            if variant != 'color':
                yes_count = sum(1 for g in gts if g == 'yes')
                no_count = sum(1 for g in gts if g == 'no')
                print(f"\t\t{yes_count} yes, {no_count} no")
                
            variant_results[variant] = {"f1": f1, "accuracy": acc, "count": len(gts)}
            all_pred.extend(preds)
            all_gt.extend(gts)
        
        # Calculate overall metrics
        overall_f1 = f1_score(all_gt, all_pred, average='weighted')
        overall_acc = sum(p == g for p, g in zip(all_pred, all_gt)) / len(all_gt) if all_gt else 0
        
        print(f"\nOverall F1: {overall_f1:.4f}")
        print(f"Overall Accuracy: {overall_acc:.4f}")
        
        # Save results
        results = {
            "overall_f1": overall_f1,
            "overall_accuracy": overall_acc,
            "metrics_by_variant": variant_results
        }
        
        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_video(self):
        
        # Initialize metrics tracking
        all_gt = []
        all_pred = []
        
        # Process each example
        for example in tqdm(self.meta_data, total=len(self.meta_data)):
            # Get prediction and ground truth
            response = example.get('response', '')
            pred = self.extract_answer_from_model_response(response).lower()
            gt_ans = example.get('answer', '').lower()
            
            # Ensure prediction is valid
            if pred not in ['a', 'b', 'c']:
                print(f"Warning: Invalid prediction '{pred}', defaulting to 'a'")
                pred = 'a'
            
            # Store predictions and ground truths
            all_pred.append(pred)
            all_gt.append(gt_ans)
        
        # Calculate metrics
        f1_macro = f1_score(all_gt, all_pred, average='macro')
        accuracy = sum(p == g for p, g in zip(all_pred, all_gt)) / len(all_gt) if all_gt else 0
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        
        # Count answers by category
        answer_counts = {}
        for ans in ['a', 'b', 'c']:
            gt_count = sum(1 for g in all_gt if g == ans)
            pred_count = sum(1 for p in all_pred if p == ans)
            answer_counts[ans] = {"gt": gt_count, "pred": pred_count}
            print(f"Answer {ans.upper()}: GT={gt_count}, Pred={pred_count}")
        
        # Save results
        results = {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "answer_counts": answer_counts,
            "total_samples": len(all_gt)
        }
        
        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
    

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--dataset_type", type=str, required=True, help="Type of dataset to evaluate (e.g., 'va', 'text_instruct', 'folding_nets')")
    
    args = parser.parse_args()
    
    evaluator = VissimEvaluator(args.input_path, args.output_path, args.dataset_type)
    evaluator.evaluate()
