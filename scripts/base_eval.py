"""
Evaluate the CORE metric for a given model.

Run on a single GPU:
python -m scripts.base_eval

Run with torchrun on e.g. 8 GPUs:
torchrun --nproc_per_node=8 -m scripts.base_eval

The script will print the CORE metric to the console.
"""
import os
import csv
import time
import json
import yaml
import shutil
import random
import zipfile
import tempfile
from contextlib import nullcontext

import torch

from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, autodetect_device_type, download_file_with_lock
from nanochat.tokenizer import HuggingFaceTokenizer
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task

# -----------------------------------------------------------------------------
# nanochat specific function dealing with I/O etc.

# ~162MB of data needed to evaluate the CORE metric
EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"

def place_eval_bundle(file_path):
    # here file_path is the path to the eval_bundle.zip file
    # we need to unzip it and place it in the base directory
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)
        extracted_bundle_dir = os.path.join(tmpdir, "eval_bundle")
        shutil.move(extracted_bundle_dir, eval_bundle_dir)
    print0(f"Placed eval_bundle directory at {eval_bundle_dir}")

def evaluate_model(model, tokenizer, device, max_per_task=-1, num_recur=None):
    """
    Evaluate a base model on the CORE benchmark.
    - max_per_task: crop the data to this many examples per task for testing (-1 = disable)
    - num_recur: number of recurrences for recursive transformer (None = use model default)
    """
    # Load config and task metadata
    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
    # Download the eval bundle to disk (and unzip if needed)
    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(EVAL_BUNDLE_URL, "eval_bundle.zip", postprocess_fn=place_eval_bundle)
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tasks = config['icl_tasks']

    # Load random baseline values from eval metadata
    random_baselines = {}
    with open(eval_meta_data, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row['Eval Task']
            random_baseline = row['Random baseline']
            random_baselines[task_name] = float(random_baseline)

    recur_str = f" (r={num_recur})" if num_recur is not None else ""
    print0(f"Evaluating with{recur_str}...")

    # Evaluate each task
    results = {}
    centered_results = {}
    for task in tasks:
        start_time = time.time()
        label = task['label']
        task_meta = {
            'task_type': task['icl_task_type'],
            'dataset_uri': task['dataset_uri'],
            'num_fewshot': task['num_fewshot'][0],
            'continuation_delimiter': task.get('continuation_delimiter', ' ')
        }
        print0(f"Evaluating: {label} ({task_meta['num_fewshot']}-shot, type: {task_meta['task_type']})... ", end='')

        # Load data for this task
        data_path = os.path.join(data_base_path, task_meta['dataset_uri'])
        with open(data_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line.strip()) for line in f]

        # shuffle the data because in many cases it appears ordered but we want
        # the ability to only run a subset of the data for debugging purposes etc.
        shuffle_rng = random.Random(1337)
        shuffle_rng.shuffle(data)
        if max_per_task > 0:
            data = data[:max_per_task]

        # run the evaluation for this task
        accuracy = evaluate_task(model, tokenizer, data, device, task_meta, num_recur=num_recur)

        results[label] = accuracy
        random_baseline = random_baselines[label]
        centered_result = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline)
        centered_results[label] = centered_result
        end_time = time.time()
        print0(f"accuracy: {accuracy:.4f} | centered: {centered_result:.4f} | time: {end_time - start_time:.2f}s")

    core_metric = sum(centered_results.values()) / len(centered_results)
    out = {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric
    }
    return out

# -----------------------------------------------------------------------------
# HuggingFace loading utilities and light wrappers for a model

class ModelWrapper:
    """Lightweight wrapper for a HuggingFace model"""
    def __init__(self, model, max_seq_len=None):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids):
        outputs = self.model(input_ids)
        logits = outputs.logits
        return logits

def load_hf_model(hf_path: str, device):
    print0(f"Loading model from: {hf_path}")
    # Load the model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(hf_path)
    model.to(device)
    model.eval()
    max_seq_len = 1024 if "openai-community/gpt2" in hf_path else None
    model = ModelWrapper(model, max_seq_len=max_seq_len)
    # Load the tokenizer
    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)
    return model, tokenizer

# -----------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-path', type=str, default=None, help='HuggingFace model path to evaluate')
    parser.add_argument('--max-per-task', type=int, default=-1, help='Max examples per task to evaluate (-1 = disable)')
    parser.add_argument('--num-recur', type=str, default=None,
                        help='Number of recurrences for recursive transformer. Can be single value (e.g. "2") or comma-separated list (e.g. "2,4,8") to evaluate multiple.')
    args = parser.parse_args()

    # Parse num_recur argument - can be single value or comma-separated list
    recur_values = None
    if args.num_recur is not None:
        recur_values = [int(x.strip()) for x in args.num_recur.split(',')]

    # distributed / precision setup
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # Load model and tokenizer from command line or from file system
    if args.hf_path is not None:
        # atm assume that if a path is given, it's a huggingface model path
        hf_path = args.hf_path
        print0(f"Loading huggingface model from: {hf_path}")
        model, tokenizer = load_hf_model(hf_path, device)
        model_name = hf_path # just for logging
        model_slug = hf_path.replace("/", "-") # for the output csv file
    else:
        # load a local model from the file system
        model, tokenizer, meta = load_model("base", device, phase="eval")
        model_name = f"base_model (step {meta['step']})" # just for logging
        model_slug = f"base_model_{meta['step']:06d}" # for the output csv file

    # Evaluate the model - possibly multiple times with different recursion counts
    all_results = {}
    if recur_values is None:
        # Default: evaluate once with model's default recursion count
        recur_values = [None]

    for num_recur in recur_values:
        recur_label = f"r{num_recur}" if num_recur is not None else "default"
        print0(f"\n{'='*80}")
        print0(f"Evaluating with num_recur={num_recur if num_recur is not None else 'default'}")
        print0(f"{'='*80}")

        with autocast_ctx:
            out = evaluate_model(model, tokenizer, device, max_per_task=args.max_per_task, num_recur=num_recur)
        all_results[recur_label] = out

    # Write out the results to csv files
    if ddp_rank == 0:
        base_dir = get_base_dir()

        for recur_label, out in all_results.items():
            suffix = f"_{recur_label}" if recur_label != "default" else ""
            output_csv_path = os.path.join(base_dir, "base_eval", f"{model_slug}{suffix}.csv")
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            results = out["results"]
            centered_results = out["centered_results"]
            core_metric = out["core_metric"]
            with open(output_csv_path, 'w', encoding='utf-8', newline='') as f:
                f.write(f"{'Task':<35}, {'Accuracy':<10}, {'Centered':<10}\n")
                for label in results:
                    f.write(f"{label:<35}, {results[label]:<10.6f}, {centered_results[label]:<10.6f}\n")
                f.write(f"{'CORE':<35}, {'':<10}, {core_metric:<10.6f}\n")
            # Print the content of the csv file to console too
            print0("="*80)
            print0(f"Model: {model_name} ({recur_label})")
            print0("="*80)
            with open(output_csv_path, 'r', encoding='utf-8') as f:
                print0(f.read())

        # Summary comparison if multiple recursion counts were evaluated
        if len(all_results) > 1:
            print0("\n" + "="*80)
            print0("SUMMARY: CORE metric by recursion count")
            print0("="*80)
            for recur_label, out in all_results.items():
                print0(f"  {recur_label}: {out['core_metric']:.4f}")

    # Log to report (use last result for backwards compatibility)
    from nanochat.report import get_report
    last_out = list(all_results.values())[-1]
    get_report().log(section="Base model evaluation", data=[
        {
            "Model": model_name,
            "CORE metric": last_out["core_metric"],
            "Recursion counts evaluated": list(all_results.keys()),
        },
        {f"{k}_core": v["core_metric"] for k, v in all_results.items()},  # summary of all
    ])

    compute_cleanup()

if __name__ == "__main__":
    main()
