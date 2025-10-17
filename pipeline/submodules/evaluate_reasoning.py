import torch
import json
import os
from tqdm import tqdm
from datasets import load_dataset
import re
import pandas as pd
from accelerate import Accelerator
from pipeline.utils.hook_utils import add_hooks
import random

import json
import re
from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def get_adaptive_batch_size(model_base, dataset, max_batch_size, min_batch_size=1, safety_margin=0.8):
    """
    Dynamically determine the batch size based on available GPU memory.

    Args:
        model_base: The model to evaluate.
        dataset: Sample data to test batch size.
        max_batch_size: Maximum allowable batch size.
        min_batch_size: Minimum allowable batch size.
        safety_margin: Fraction of memory usage considered safe.

    Returns:
        int: Adapted batch size.
    """
    batch_size = max_batch_size
    while batch_size >= min_batch_size:
        try:
            # Simulate a forward pass with the current batch size
            batch_dataset = dataset[:batch_size]
            with torch.no_grad():
                model_base.generate_completions(
                    dataset=batch_dataset,
                    batch_size=batch_size,
                    max_new_tokens=128,  # Reduce token length for testing
                    progress=False
                )
            return batch_size  # Return the working batch size
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"Batch size {batch_size} is too large, reducing...")
                batch_size //= 2  # Halve the batch size and retry
                torch.cuda.empty_cache()  # Clear GPU memory
            else:
                raise e  # Re-raise any non-memory-related errors
    return min_batch_size  # Return the smallest batch size if nothing fits

def load_gpqa():
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")  # Update with the correct path
    dataset = dataset["train"]
    return dataset

def prepare_gpqa_prompt(example):
    """
    Prepare the prompt for GPQA with shuffled answer options.

    Args:
        example (dict): A test example from GPQA.

    Returns:
        tuple: Formatted prompt for the model, shuffled options, and correct answer index.
    """

    question = example['Question']
    options = [
        ('Correct Answer', example['Correct Answer']),
        ('Incorrect Answer 1', example['Incorrect Answer 1']),
        ('Incorrect Answer 2', example['Incorrect Answer 2']),
        ('Incorrect Answer 3', example['Incorrect Answer 3'])
    ]

    random.shuffle(options)
    correct_index = next(i for i, (label, _) in enumerate(options) if label == 'Correct Answer')

    prompt = f"{question}\n\nChoose the correct answer and respond only with the corresponding number:\n"
    for idx, (_, option) in enumerate(options, 1):
        prompt += f"{idx}) {option}\n"

    return prompt, correct_index

def extract_answer_gpqa(response):
    """
    Extract the predicted answer from the model response.

    Args:
        response (str): Model's output response.

    Returns:
        int: Extracted answer index (0-based).
    """
    response = response.lower().strip()
    patterns = ['1', '2', '3', '4']
    for i, pattern in enumerate(patterns):
        if re.search(r'\b' + pattern + r'\b', response):
            return i
    return ""

@torch.no_grad()
def evaluate_gpqa(model_base, pre_hooks, hooks, max_batch_size=8, label="GPQA"):
    """
    Evaluate GPQA with adaptive batch sizing.

    Args:
        model_base: The preloaded model.
        pre_hooks: Pre-hooks for ablation or actadd.
        hooks: Hooks for ablation or actadd.
        max_batch_size: Maximum batch size for evaluation.
        label: Label for the evaluation process.

    Returns:
        float: Accuracy on the test set.
    """
    correct = 0
    total = 0

    accelerator = Accelerator()
    model_base = accelerator.prepare(model_base)

    # Load GPQA test set
    test_set = load_gpqa()

    # Prepare prompts
    prompts_data = [prepare_gpqa_prompt(item) for item in test_set]
    prompts = [data[0] for data in prompts_data]
    correct_indices = [data[1] for data in prompts_data]

    dataset = [{"instruction": prompt} for prompt in prompts]
    dataset = accelerator.prepare(dataset)

    # Initialize tqdm progress bar
    pbar = tqdm(total=len(test_set), desc=f"Evaluating {label}")

    i = 0
    while i < len(test_set):
        batch_size = max_batch_size
        while batch_size > 0:  # Try smaller batch sizes on OOM
            try:
                # Process batches of data
                batch_prompts = dataset[i:i + batch_size]
                batch_correct_indices = correct_indices[i:i + batch_size]

                with torch.no_grad():
                    with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=hooks):
                        outputs = model_base.generate_completions(
                            dataset=batch_prompts,
                            fwd_pre_hooks=pre_hooks,
                            fwd_hooks=hooks,
                            batch_size=batch_size,
                            max_new_tokens=128,
                            progress=False
                        )

                    # Process outputs and validate
                    for output, correct_index in zip(outputs, batch_correct_indices):
                        generated_text = output["response"]

                        prediction_index = extract_answer_gpqa(generated_text)

                        if prediction_index == "":
                            total += 1
                        else:
                            if prediction_index == correct_index:
                                correct += 1
                            total += 1

                # Update progress bar and move to the next batch
                pbar.update(batch_size)
                i += batch_size
                break  # Exit the inner loop after successful processing

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()  # Free up GPU memory
                    batch_size //= 2  # Reduce the batch size
                else:
                    raise e  # Re-raise non-memory-related errors

        if batch_size == 0:
            raise RuntimeError("Batch size could not be reduced further. Evaluation failed.")

    pbar.close()  # Close the progress bar

    # Compute accuracy
    accuracy = correct / total * 100
    print(f"Accuracy for {label}: {accuracy:.2f}%")
    return accuracy


def load_ai2_arc():
    """
    Load the AI2 ARC dataset (ARC-Challenge or ARC-Easy subsets).
    """
    dataset = load_dataset("ai2_arc", "ARC-Challenge")
    test_set = dataset["test"]
    return test_set


def prepare_arc_prompt(test_row):

    # Add the target question
    prompt = "Answer the following question by choosing the best answer choice. Answer with only your answer choice. \n"
    prompt += f"Question: {test_row['question']}\n"
    for idx, option in enumerate(test_row["choices"]["text"]):
        prompt += f"{chr(65 + idx)}. {option}\n"
    prompt += "Answer:"
    return prompt


def extract_arc_answer(text):
    """
    Extract the final answer from the model's output.

    Args:
        text (str): The generated text.

    Returns:
        str: The extracted answer, or None if extraction fails.
    """
    #capture a single letter (A-D) surrounded by whitespace or a newline
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)  # Return the captured letter
    return None


def is_correct_ai2_arc(model_completion, gt_answer):
    """
    Check if the model's completion matches the ground truth.

    Args:
        model_completion (str): The model's generated response.
        gt_example (dict): The ground truth example.

    Returns:
        bool: True if the model's output matches the ground truth, False otherwise.
    """
    return extract_arc_answer(model_completion) == gt_answer


@torch.no_grad()
def evaluate_ai2_arc(model_base, pre_hooks, hooks, label, max_tokens=64, max_batch_size=8):
    """
    Evaluate AI2 ARC.

    Args:
        model_base: The preloaded model.
        pre_hooks: Forward pre-hooks for ablation or actadd.
        hooks: Forward hooks for ablation or actadd.
        max_tokens: Maximum tokens for generation.
        batch_size: Batch size for evaluation.

    Returns:
        float: Accuracy on the test set.
    """
    correct = 0
    total = 0

    test_set = load_ai2_arc()


    accelerator = Accelerator()
    model_base = accelerator.prepare(model_base)


    prompts = [prepare_arc_prompt(item) for item in test_set]
    answers = [item["answerKey"] for item in test_set]
    dataset = [{"instruction": prompt} for prompt in prompts]
    dataset = accelerator.prepare(dataset)
    
    batch_size = get_adaptive_batch_size(model_base, dataset, max_batch_size=max_batch_size)

    for i in tqdm(range(0, len(test_set), batch_size), desc=f"Evaluating {label}"):
        batch_dataset = dataset[i:i + batch_size]

        # Generate completions
        with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=hooks):
            outputs = model_base.generate_completions(
                dataset=batch_dataset,
                fwd_pre_hooks=pre_hooks,
                fwd_hooks=hooks,
                batch_size=batch_size,
                max_new_tokens=128,
                progress=False
            )

        for j, output in enumerate(outputs):
            generated_text = output["response"]

            if is_correct_ai2_arc(generated_text, answers[i+j]):
                correct += 1
            total += 1


    accuracy = correct / total * 100
    print(f"AI2 ARC Evaluation Accuracy: {accuracy:.2f}%")
    return accuracy

def evaluate_gpqa_and_arc(cfg, 
                         model_base, 
                         ablation_fwd_pre_hooks, 
                         ablation_fwd_hooks, 
                         actadd_fwd_pre_hooks, 
                         actadd_fwd_hooks,
                         exclude_base = False,
                         batch_size=16):

    benchmark_dir = os.path.join(cfg.artifact_path(), "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)


    # Evaluate gpqa Pro
    accuracy_baseline_gpqa = evaluate_gpqa(
        model_base, pre_hooks=[], hooks=[], max_batch_size=batch_size, label = "gpqa Base"
    ) if not exclude_base else None

    accuracy_ablation_gpqa = evaluate_gpqa(
        model_base, pre_hooks=ablation_fwd_pre_hooks, hooks=ablation_fwd_hooks, max_batch_size=batch_size, label = "gpqa Ablate"
    )

    # Evaluate AI2 ARC
    accuracy_baseline_arc = evaluate_ai2_arc(
        model_base, pre_hooks=[], hooks=[], max_batch_size=batch_size, label = "ARC Base"
    ) if not exclude_base else None
    
    accuracy_ablation_arc = evaluate_ai2_arc(
        model_base, pre_hooks=ablation_fwd_pre_hooks, hooks=ablation_fwd_hooks, max_batch_size=batch_size, label = "ARC Ablate"
    )

    combined_results = {
        "gpqa_baseline": {"accuracy": accuracy_baseline_gpqa},
        "gpqa_ablation": {"accuracy": accuracy_ablation_gpqa},
        "ai2_arc_baseline": {"accuracy": accuracy_baseline_arc},
        "ai2_arc_ablation": {"accuracy": accuracy_ablation_arc}
    }


    # Save results to a JSON file
    with open(os.path.join(benchmark_dir, "gpqa_arc_results.json") , "w") as f:
        json.dump(combined_results, f, indent=4)


def evaluate_truthful_qa(cfg, 
                         model_base, 
                         ablation_fwd_pre_hooks, 
                         ablation_fwd_hooks, 
                         actadd_fwd_pre_hooks, 
                         actadd_fwd_hooks, 
                         truthful_qa_path="dataset/TruthfulQA.csv",
                         batch_size=32):
    """
    Evaluate the model on the TruthfulQA dataset as a multiple-choice task using Accelerate for multi-GPU.

    Args:
        cfg: Configuration object containing paths and settings.
        model_base: The base model object.
        truthful_qa_path: Path to the TruthfulQA dataset CSV file.
        ablation_fwd_pre_hooks: Forward pre-hooks for ablation.
        ablation_fwd_hooks: Forward hooks for ablation.

    Returns:
        None
    """

    accelerator = Accelerator()
    model_base = accelerator.prepare(model_base)

    # Load the TruthfulQA dataset
    truthful_qa_data = pd.read_csv(truthful_qa_path)

    # Create directory for results
    benchmark_dir = os.path.join(cfg.artifact_path(), "benchmark")
    os.makedirs(benchmark_dir, exist_ok=True)

    # Paths for saving results
    results_baseline_path = os.path.join(benchmark_dir, "truthful_qa_accuracy.json")

    # Helper function to format a question as multiple choice
    def format_question_as_multiple_choice(question, correct_answer, incorrect_answer):
        options = [
            {"label": "A", "text": correct_answer, "is_correct": True},
            {"label": "B", "text": incorrect_answer, "is_correct": False},
        ]
        random.shuffle(options)
        prompt = "Answer the following question by choosing the correct answer choice. Answer with only the selected letter. \n"
        prompt += f"Question: {question}\n"
        for option in options:
            prompt += f"{option['label']}. {option['text']}\n"
        prompt += "Answer:"
        correct_label = [opt['label'] for opt in options if opt['is_correct']][0]
        return prompt, correct_label

    # Evaluate the dataset
    def evaluate_dataset(data, pre_hooks, hooks, label):
        correct = 0
        total = 0
        instructions = []

        # Format dataset with prompts and correct labels
        for _, row in data.iterrows():
            question = row["Question"]
            correct_answer = row["Best Answer"]
            incorrect_answer = row["Best Incorrect Answer"]
            prompt, correct_label = format_question_as_multiple_choice(question, correct_answer, incorrect_answer)
            instructions.append({"instruction": prompt, "correct_label": correct_label})

        for i in tqdm(range(0, len(instructions), batch_size), desc=f"Evaluating {label}"):
            batch = instructions[i:i + batch_size]
            prompts = [{"instruction": item["instruction"]} for item in batch]
            correct_labels = [item["correct_label"] for item in batch]
            dataset = accelerator.prepare(prompts)

            with torch.no_grad():
                with add_hooks(module_forward_pre_hooks=pre_hooks, module_forward_hooks=hooks):
                    outputs = model_base.generate_completions(
                        dataset=dataset,
                        fwd_pre_hooks=pre_hooks,
                        fwd_hooks=hooks,
                        batch_size=batch_size,
                        max_new_tokens = 128,
                        progress=False
                    )

            # Process outputs
            for j, output in enumerate(outputs):
                generated_text = output["response"]
                prediction = extract_arc_answer(generated_text)

                # Validate the prediction against the corresponding correct label
                if prediction == correct_labels[j]:
                    correct += 1
                total += 1

        return (correct / total) * 100

    # Evaluate TruthfulQA
    accuracy_baseline = evaluate_dataset(truthful_qa_data, pre_hooks=[], hooks=[], label="TruthfulQA baseline")
    accuracy_ablation = evaluate_dataset(truthful_qa_data, pre_hooks=ablation_fwd_pre_hooks, hooks=ablation_fwd_hooks, label="TruthfulQA ablation")

    # Save results to JSON
    with open(results_baseline_path, "w") as f:
        json.dump({
            "baseline accuracy": accuracy_baseline, 
            "ablation accuracy": accuracy_ablation,
        }, f, indent=4)

    # Print results summary
    print(f"TruthfulQA Results saved:")
    print(f"Baseline Accuracy: {accuracy_baseline:.2f}%")
    print(f"Ablation Accuracy: {accuracy_ablation:.2f}%")
