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
        
        file_name = self.input_path.split("/")[-1].split(".")[0]
        self.is_vissim = "vissim" in file_name
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
        # Initialize metrics tracking
        pred_by_type = defaultdict(list)
        gt_by_type = defaultdict(list)
        pred_by_steps = defaultdict(list)
        gt_by_steps = defaultdict(list)
        pred_by_difficulty = defaultdict(list)
        gt_by_difficulty = defaultdict(list)
        pred_by_transformation = defaultdict(list)
        gt_by_transformation = defaultdict(list)
        
        # Build transformation mapping
        qid2transformation = {}
        for example in self.meta_data:
            qid = example['qid']
            if 'transformations' in example:
                transformations = example['transformations']
                trans = []
                if 'shear' in transformations:
                    trans.append('shear')
                if 'scale' in transformations:
                    trans.append('scale')
                if 'rotate' in transformations:
                    trans.append('rotate')
                if 'translate' in transformations:
                    trans.append('translate')
                if 'flip' in transformations:
                    trans.append('flip')
                qid2transformation[qid] = trans
        
        # Process each example
        overall_gt = []
        overall_pred = []
        valid_samples = 0
        
        
        
        for example in tqdm(self.meta_data, total=len(self.meta_data)):
            qid = example['qid']
            
            # Parse important information from qid
            parts = qid.split("_")
            num_steps = parts[0] if len(parts) > 0 else ""
            difficulty = parts[1] if len(parts) > 1 else "unknown"
            variant = " ".join(parts[2:-1]) if len(parts) > 3 else ""
            
            # Get prediction and ground truth
            response = example.get('response', '')
            pred = self.extract_answer_from_model_response(response).lower()
            gt_ans = example.get('answer', '').lower()
            
            # Skip invalid entries
            if num_steps == "" or variant == "" or difficulty == "":
                print(f"Warning: Empty keys for {qid}")
                continue
            
            valid_samples += 1
            
            # Track overall metrics
            overall_gt.append(gt_ans)
            overall_pred.append(pred)
            
            # Track by type/variant
            pred_by_type[variant].append(pred)
            gt_by_type[variant].append(gt_ans)
            
            # Only track "all" variants for difficulty and steps metrics
            if variant != "all" and self.is_vissim:
                continue
            
            pred_by_difficulty[difficulty].append(pred)
            gt_by_difficulty[difficulty].append(gt_ans)
            pred_by_steps[num_steps].append(pred)
            gt_by_steps[num_steps].append(gt_ans)
            
            # Track by transformation type
            if qid in qid2transformation:
                for transform in qid2transformation[qid]:
                    pred_by_transformation[transform].append(pred)
                    gt_by_transformation[transform].append(gt_ans)
        
        # Calculate and store metrics
        results = {}
        
        # Overall accuracy
        overall_acc = sum(p == g for p, g in zip(overall_pred, overall_gt)) / len(overall_gt) if overall_gt else 0
        results["overall_accuracy"] = overall_acc
        print(f"Overall accuracy: {overall_acc:.4f}")
        
        # Accuracy by variant
        print("\nAccuracy by variants:")
        variant_results = {}
        for variant, preds in pred_by_type.items():
            gt = gt_by_type[variant]
            accuracy = sum(p == g for p, g in zip(preds, gt)) / len(gt) if gt else 0
            print(f"{variant}: {accuracy:.4f}")
            variant_results[variant] = accuracy
        results["accuracy_by_variant"] = variant_results
        
        # Accuracy by difficulty
        print("\nAccuracy by difficulty level:")
        difficulty_results = {}
        for difficulty, preds in pred_by_difficulty.items():
            gt = gt_by_difficulty[difficulty]
            accuracy = sum(p == g for p, g in zip(preds, gt)) / len(gt) if gt else 0
            print(f"{difficulty}: {accuracy:.4f}")
            difficulty_results[difficulty] = accuracy
        results["accuracy_by_difficulty"] = difficulty_results
        
        # Accuracy by steps
        print("\nAccuracy by steps:")
        steps_results = {}
        for steps, preds in pred_by_steps.items():
            gt = gt_by_steps[steps]
            accuracy = sum(p == g for p, g in zip(preds, gt)) / len(gt) if gt else 0
            print(f"{steps}: {accuracy:.4f}")
            steps_results[steps] = accuracy
        results["accuracy_by_steps"] = steps_results
        
        # Accuracy by transformation
        print("\nAccuracy by transformation:")
        transformation_results = {}
        for transform, preds in pred_by_transformation.items():
            gt = gt_by_transformation[transform]
            accuracy = sum(p == g for p, g in zip(preds, gt)) / len(gt) if gt else 0
            print(f"{transform}: {accuracy:.4f}")
            transformation_results[transform] = accuracy
        results["accuracy_by_transformation"] = transformation_results
        
        # Save results
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
        # Initialize metrics tracking
        pred_by_type = defaultdict(list)
        gt_by_type = defaultdict(list)
        pred_by_steps = defaultdict(list)
        gt_by_steps = defaultdict(list)
        pred_by_difficulty = defaultdict(list)
        gt_by_difficulty = defaultdict(list)
        pred_by_transformation = defaultdict(list)
        gt_by_transformation = defaultdict(list)
        
        # Build transformation mapping
        qid2transformation = {}
        for example in self.meta_data:
            qid = example['qid']
            if 'transformations' in example:
                transformations = example['transformations']
                trans = []
                if 'shear' in transformations:
                    trans.append('shear')
                if 'scale' in transformations:
                    trans.append('scale')
                if 'rotate' in transformations:
                    trans.append('rotate')
                if 'translate' in transformations:
                    trans.append('translate')
                if 'flip' in transformations:
                    trans.append('flip')
                qid2transformation[qid] = trans
        
        # Process each example
        overall_gt = []
        overall_pred = []
        valid_samples = 0
        
        for example in tqdm(self.meta_data, total=len(self.meta_data)):
            qid = example['qid']
            
            # Parse important information from qid
            parts = qid.split("_")
            num_steps = parts[0] if len(parts) > 0 else ""
            difficulty = parts[1] if len(parts) > 1 else "unknown"
            variant = " ".join(parts[2:-1]) if len(parts) > 3 else ""
            
            # Get prediction and ground truth
            response = example.get('response', '')
            pred = self.extract_answer_from_model_response(response).lower()
            gt_ans = example.get('answer', '').lower()
            
            # Skip invalid entries
            if num_steps == "" or variant == "" or difficulty == "":
                print(f"Warning: Empty keys for {qid}")
                continue
            
            valid_samples += 1
            
            # Track overall metrics
            overall_gt.append(gt_ans)
            overall_pred.append(pred)
            
            # Track by type/variant
            pred_by_type[variant].append(pred)
            gt_by_type[variant].append(gt_ans)
            
            if variant != "all" and self.is_vissim:
                continue
            
            # Track by difficulty and steps
            pred_by_difficulty[difficulty].append(pred)
            gt_by_difficulty[difficulty].append(gt_ans)
            pred_by_steps[num_steps].append(pred)
            gt_by_steps[num_steps].append(gt_ans)
            
            # Track by transformation type
            if qid in qid2transformation:
                for transform in qid2transformation[qid]:
                    pred_by_transformation[transform].append(pred)
                    gt_by_transformation[transform].append(gt_ans)
        
        # Calculate and store metrics
        results = {}
        
        # Overall accuracy
        overall_acc = sum(p == g for p, g in zip(overall_pred, overall_gt)) / len(overall_gt) if overall_gt else 0
        results["overall_accuracy"] = overall_acc
        print(f"Overall accuracy: {overall_acc:.4f}")
        
        # Accuracy by variant
        print("\nAccuracy by variants:")
        variant_results = {}
        for variant, preds in pred_by_type.items():
            gt = gt_by_type[variant]
            accuracy = sum(p == g for p, g in zip(preds, gt)) / len(gt) if gt else 0
            print(f"{variant}: {accuracy:.4f}")
            variant_results[variant] = accuracy
        results["accuracy_by_variant"] = variant_results
        
        # Accuracy by difficulty
        print("\nAccuracy by difficulty level:")
        difficulty_results = {}
        for difficulty, preds in pred_by_difficulty.items():
            gt = gt_by_difficulty[difficulty]
            accuracy = sum(p == g for p, g in zip(preds, gt)) / len(gt) if gt else 0
            print(f"{difficulty}: {accuracy:.4f}")
            difficulty_results[difficulty] = accuracy
        results["accuracy_by_difficulty"] = difficulty_results
        
        # Accuracy by steps
        print("\nAccuracy by steps:")
        steps_results = {}
        for steps, preds in pred_by_steps.items():
            gt = gt_by_steps[steps]
            accuracy = sum(p == g for p, g in zip(preds, gt)) / len(gt) if gt else 0
            print(f"{steps}: {accuracy:.4f}")
            steps_results[steps] = accuracy
        results["accuracy_by_steps"] = steps_results
        
        # Accuracy by transformation
        print("\nAccuracy by transformation:")
        transformation_results = {}
        for transform, preds in pred_by_transformation.items():
            gt = gt_by_transformation[transform]
            accuracy = sum(p == g for p, g in zip(preds, gt)) / len(gt) if gt else 0
            print(f"{transform}: {accuracy:.4f}")
            transformation_results[transform] = accuracy
        results["accuracy_by_transformation"] = transformation_results
        
        # Save results
        with open(self.output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        return results
        
    def evaluate_folding_nets(self):
        
        # Initialize metrics tracking
        pred_by_type = defaultdict(list)
        gt_by_type = defaultdict(list)
        pred_by_steps = defaultdict(list)
        gt_by_steps = defaultdict(list)
        
        # get task name
        task_name = self.input_path.split("/")[-1].split(".")[0]

        # Try to load dataset from HuggingFace
        try:
            # Create qid to steps mapping
            qid2steps = {}
            for sample in self.meta_data:
                qid = sample['qid']
                if "folding_nets" in task_name:
                    steps = 5
                else:
                    steps = len(json.loads(sample["question_info"])["instructions"])
                qid2steps[qid] = steps
        except Exception as e:
            print(f"Warning: Could not load HuggingFace dataset: {e}")
            # Fallback to default steps
            qid2steps = defaultdict(lambda: 5)
        
        # Process each example
        overall_gt = []
        overall_pred = []
        
        for example in tqdm(self.meta_data, total=len(self.meta_data)):
            qid = example['qid']
            
            # Get prediction and ground truth
            response = example.get('response', '')
            pred = self.extract_answer_from_model_response(response).lower()
            gt_choice = example.get('answer', '').lower()
            
            # Get gt_ans from choices when available
            gt_ans = gt_choice
            if 'choices' in example:
                try:
                    gt_ans = example['choices'][ord(gt_choice) - ord('a')].lower()
                    gt_ans = gt_ans.split('.')[0].lower()
                except (IndexError, TypeError):
                    pass
            
            # Extract variant from qid
            if "folding_nets" in task_name:
                if task_name == "folding_nets_3d_perception_test":
                    variant = qid.split("_")[-1]
                elif task_name == "folding_nets_2d_perception_test":
                    variant = qid.split("_")[2]
                else:
                    variant = " ".join(qid.split("_")[1:])
                    if "all vis" in variant:
                        variant = f"all for valid {gt_ans}"
            else:  # Assuming tangram puzzle or similar
                variant = " ".join(qid.split("_")[3:])
                if "all for valid" in variant:
                    variant = f"all for valid {gt_ans}"
            
            # Skip if variant is empty
            if len(variant.strip()) == 0:
                continue
            
            # Get number of steps for this qid
            steps = qid2steps.get(qid, 5)
            
            # Store predictions and ground truths
            pred_by_type[variant].append(pred)
            gt_by_type[variant].append(gt_ans)
            pred_by_steps[steps].append(pred)
            gt_by_steps[steps].append(gt_ans)
            overall_gt.append(gt_ans)
            overall_pred.append(pred)
        
        # Calculate F1 scores
        def calc_f1(preds, gts):
            try:
                return f1_score(gts, preds, average='weighted')
            except Exception as e:
                print(f"Error calculating F1 score: {e}")
                return 0.0
        
        # Calculate and store metrics
        metrics = {}
        metrics["overall_f1"] = calc_f1(overall_pred, overall_gt)
        
        # Print and store results by variant
        print(f"F1 scores by variant:")
        variant_results = {}
        for variant in pred_by_type:
            f1 = calc_f1(pred_by_type[variant], gt_by_type[variant])
            acc = sum(p == g for p, g in zip(pred_by_type[variant], gt_by_type[variant])) / len(gt_by_type[variant])
            
            print(f"\tVariant {variant}: F1={f1:.4f}, Acc={acc:.4f}")
            print(f"\t\tSamples: {len(gt_by_type[variant])}")
            
            metrics[f"{variant}_f1"] = f1
            variant_results[variant] = {"f1": f1, "accuracy": acc, "count": len(gt_by_type[variant])}
        
        # Print and store results by steps
        print(f"\nF1 scores by steps:")
        steps_results = {}
        for steps in pred_by_steps:
            f1 = calc_f1(pred_by_steps[steps], gt_by_steps[steps])
            acc = sum(p == g for p, g in zip(pred_by_steps[steps], gt_by_steps[steps])) / len(gt_by_steps[steps])
            
            print(f"\t{steps} steps: F1={f1:.4f}, Acc={acc:.4f}")
            print(f"\t\tSamples: {len(gt_by_steps[steps])}")
            
            metrics[f"{steps}_f1"] = f1
            metrics[f"{steps}_count"] = len(gt_by_steps[steps])
            steps_results[steps] = {"f1": f1, "accuracy": acc, "count": len(gt_by_steps[steps])}
        
        # Print overall results
        overall_acc = sum(p == g for p, g in zip(overall_pred, overall_gt)) / len(overall_gt) if overall_gt else 0
        print(f"\nOverall F1: {metrics['overall_f1']:.4f}")
        print(f"Overall Accuracy: {overall_acc:.4f}")
        
        # Save results in the required format
        results = {
            "overall_f1": metrics["overall_f1"],
            "overall_accuracy": overall_acc,
            "metrics_by_variant": variant_results,
            "metrics_by_steps": steps_results,
            "all_metrics": metrics
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
