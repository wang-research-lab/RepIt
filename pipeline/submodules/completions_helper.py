import os
import json
from typing import Optional

from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from dataset.load_dataset import load_dataset_split, load_dataset

def generate_and_save_completions_for_dataset(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label, dataset_name, batch_size = 64, dataset=None, save_path : str = None, use_existing: bool = False):
    
    """Generate and save completions for a dataset."""
    folder_name = save_path if save_path else "completions"

    if os.path.exists(f'{cfg.artifact_path()}/{folder_name}/{dataset_name}_{intervention_label}_completions.json') and use_existing:
        return None

    if not os.path.exists(os.path.join(cfg.artifact_path(), folder_name)):
        os.makedirs(os.path.join(cfg.artifact_path(), folder_name))

    if dataset is None:
        dataset = load_dataset(dataset_name)

    completions = model_base.generate_completions(dataset, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, max_new_tokens=cfg.max_new_tokens, batch_size = batch_size)
    
    with open(f'{cfg.artifact_path()}/{folder_name}/{dataset_name}_{intervention_label}_completions.json', "w") as f:
        json.dump(completions, f, indent=4)


def evaluate_completions_and_save_results_for_dataset(cfg, intervention_label, dataset_name, eval_methodologies, target_categories=None, save_path = False, use_existing = False, batch_size = 32):
    """Evaluate completions and save results for a dataset."""

    folder_name = save_path if save_path else "completions"

    if os.path.exists(f'{cfg.artifact_path()}/{folder_name}/{dataset_name}_{intervention_label}_evaluations.json') and use_existing:
        return None

    with open(os.path.join(cfg.artifact_path(), f'{folder_name}/{dataset_name}_{intervention_label}_completions.json'), 'r') as f:
        completions = json.load(f)

    evaluation = evaluate_jailbreak(
        completions=completions,
        methodologies=eval_methodologies,
        evaluation_path=os.path.join(cfg.artifact_path(), folder_name, f"{dataset_name}_{intervention_label}_evaluations.json"),
        target_categories = target_categories,
        batch_size = batch_size,
    )

    with open(f'{cfg.artifact_path()}/{folder_name}/{dataset_name}_{intervention_label}_evaluations.json', "w") as f:
        json.dump(evaluation, f, indent=4)
    