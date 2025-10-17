import os
import json
import random
import argparse
import hashlib

from typing import Dict, List, Tuple, Union

import torch
import numpy as np
from torch import Tensor
from tqdm import tqdm
from jaxtyping import Float
from collections import defaultdict
from scipy.stats import kurtosis, skew
import torch.nn.functional as F
import torch._dynamo

import gc

# Local imports - Dataset loading
from dataset.load_dataset import load_dataset_split, load_dataset

# Configuration and model setup
from pipeline.config import Config
from pipeline.model_utils.model_factory import construct_model_base

# Hook utilities
from pipeline.utils.hook_utils import (
    add_hooks,
    get_affine_direction_ablation_input_pre_hook,
    get_affine_direction_ablation_output_hook,
)



# FNA pipeline modules
from pipeline.submodules.generate_directions_repit import generate_category_directions
from pipeline.submodules.disentangle_vectors import isolate_target_vectors, generate_completions_for_rhos, evaluate_rhos, filter_exclude_categories
from pipeline.submodules.select_direction_cosmic import select_direction
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak, unload_llamaguard_model
from pipeline.submodules.evaluate_loss import evaluate_loss
from pipeline.submodules.evaluate_reasoning import (
    evaluate_gpqa_and_arc,
    evaluate_truthful_qa,
)
from pipeline.submodules.completions_helper import (
    generate_and_save_completions_for_dataset,
    evaluate_completions_and_save_results_for_dataset
)
# Debugging
import pdb


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse model path argument.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
    parser.add_argument('--target_group', type=str, required=True, help='Target group key (e.g., wmd_bio')
    
    parser.add_argument(
        '--use_existing',
        action='store_true',
        help='If set, uses existing direction generation/selection and rho gridsearch results'
    )
    
    parser.add_argument(
        '--tailweight_ablation_study',
        action='store_true',
        help='If set, runs experiments limiting the target_size'
    )

    parser.add_argument(
        '--target_size_study',
        action='store_true',
        help='If set, runs experiments limiting the target_size'
    )
    #parser.add_argument('--seed', type=int, default = 8, help='Which seed to use to split the data')
    return parser.parse_args()

def _safe_release(*names, scope=None):
    """
    Safely release variables by moving tensors to CPU and deleting them.
    Includes garbage collection and GPU memory clearing, run 3 times to ensure full cleanup.
    """
    import torch, gc
    if scope is None:
        scope = globals()

    for n in names:
        try:
            obj = scope.get(n, None)
            if obj is not None:
                # If it's a tensor or has .cpu(), move off GPU first
                try:
                    if hasattr(obj, "cpu"):
                        obj = obj.cpu()  # Move tensor to CPU
                except Exception:
                    pass
            del scope[n]  # Delete the object
        except Exception:
            pass

    # Perform garbage collection and GPU memory clearing 3 times
    for _ in range(3):
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()  # Ensure memory is freed on all devices


def load_and_sample_beast_datasets(cfg, target_group, target_categories, target_size=None, exclude=None, seed = 8):
    """
    Load datasets and sample them based on proportions.

    Args:
        cfg: configuration with p_train, p_val, p_test attributes (floats between 0 and 1)
        target_group (str): Group name for harmful dataset.
        target_categories (list[str]): Categories to include in target.
        target_size (int, optional): Max number of target-category prompts in training set.
        exclude (list[str], optional): Categories to exclude from harmful data.

    Returns:
        Tuple: (harmful_train, harmless_train, harmful_val, target_val, harmless_val, non_target_val, harmful_test, harmless_test)
    """
    import random
    rng = random.Random(seed)


    base_datasets = ["jailbreakv", "strongreject"]

    harmful_train_all = []
    harmful_val_all = []
    harmful_test_all = []

    for base in base_datasets:
        harmful_train_all += load_dataset_split(harmtype=f'{base}_harmful', split='train', instructions_only=False)
        harmful_val_all   += load_dataset_split(harmtype=f'{base}_harmful', split='val', instructions_only=False)
        harmful_test_all  += load_dataset_split(harmtype=f'{base}_harmful', split='test')



    for split_name, storage in [('train', harmful_train_all), ('val', harmful_val_all), ('test', harmful_test_all)]:
        try:
            extra_data = load_dataset_split(harmtype=f'{target_group}_harmful', split=split_name)
            storage.extend(extra_data)
        except FileNotFoundError:
            raise Exception (f"Could not find {target_group}, split {split_name}. Please note original wmd data is not provided in this repository.")

    harmless_train_all = load_dataset_split(harmtype='harmless', split='train', instructions_only=True)
    harmless_val_all = load_dataset_split(harmtype='harmless', split='val', instructions_only=True)
    harmless_test_all = load_dataset_split(harmtype='harmless', split='test')

    if exclude:
        harmful_train_all = [x for x in harmful_train_all if x["category"] not in exclude]
        harmful_val_all = [x for x in harmful_val_all if x["category"] not in exclude]

    def sample(data, n):
        return rng.sample(data, min(len(data), n))

    harmful_target_train = [x for x in harmful_train_all if x["category"] in target_categories]
    harmful_nontarget_train = [x for x in harmful_train_all if x["category"] not in target_categories]
    total_train = int(len(harmful_train_all) * cfg.p_train)

    if target_size is not None:
        max_target = min(len(harmful_target_train), target_size)
        nontarget_quota = total_train - max_target
        harmful_train = sample(harmful_target_train, max_target) + sample(harmful_nontarget_train, nontarget_quota)
        harmful_val = harmful_train
        nontarget_val = [x for x in harmful_val if x["category"] not in target_categories]
    else:
        harmful_train = sample(harmful_train_all, total_train)
        harmful_val = sample(harmful_val_all, int(len(harmful_val_all) * cfg.p_val))
        nontarget_val = [x for x in harmful_val if x["category"] not in target_categories]
        
    harmful_test = sample(harmful_test_all, int(len(harmful_test_all) * cfg.p_test))

    harmless_train = sample(harmless_train_all, len(harmful_train))
    harmless_val = sample(harmless_val_all, len(harmful_val))
    harmless_test = sample(harmless_test_all, len(harmful_test))

    # Free intermediate memory
    del harmful_train_all
    del harmful_val_all
    del harmful_test_all
    del harmless_train_all
    del harmless_val_all
    del harmless_test_all
    del harmful_target_train
    del harmful_nontarget_train

    return (
        harmful_train,
        harmless_train,
        harmful_val,
        nontarget_val,
        harmless_val,
        harmful_test,
        harmless_test,
    )


def generate_and_save_category_directions(
    cfg,
    model_base,
    harmful_train,
    harmless_train,
    batch_size=128,
    use_existing = False,
    save = True,
    system_prompt=None
):
    """
    Generate and save category direction information 
        - cat_means:       Dict[str, (n, Tensor[pos, layer, d])]
        - harmless_mean:   Tensor[pos, layer, d]


    """
    dir_path = os.path.join(cfg.artifact_path(), 'generate_directions')
    os.makedirs(dir_path, exist_ok=True)

    cat_means_path = os.path.join(dir_path, 'cat_means.pt')
    harmless_ref_path = os.path.join(dir_path, 'harmless_reference.pt')
    pairwise_similarities_path = os.path.join(dir_path, 'pairwise_similarities.json')
    cat_acts_path = os.path.join(dir_path, 'cat_activations.pt')

    # Load if already cached
    if use_existing and os.path.exists(cat_means_path) and os.path.exists(harmless_ref_path):
        cat_means = torch.load(cat_means_path)
        harmless_mean = torch.load(harmless_ref_path)

        return cat_means, harmless_mean

    # Otherwise regenerate
    cat_data, harmless_mean = generate_category_directions(
        model_base,
        harmful_train,
        harmless_train,
        artifact_dir=dir_path,
        system_prompt=system_prompt,
        batch_size=batch_size
    )
    
    if save:
        torch.save(cat_data, cat_means_path)
        torch.save(harmless_mean, harmless_ref_path)
    return cat_data, harmless_mean



def select_and_save_direction(cfg, model_base, harmful_val, harmless_val, candidate_directions, harmless_mean, layers_to_evaluate, batch_size = 64, suffix = None, use_existing = False):
    """Select and save the direction."""

    save_path = f'{cfg.artifact_path()}/direction_metadata{f"_{suffix}" if suffix else ""}.json'

    if not os.path.exists(os.path.join(cfg.artifact_path(), 'select_direction')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'select_direction'))
    elif use_existing:
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                saved_metadata = json.load(f)
                pos = saved_metadata["pos"]
                layer = saved_metadata["layer"]
                direction = candidate_directions[pos, layer]

                harmless_reference = harmless_mean[pos, layer]

            return pos, layer, direction, harmless_reference
        

    pos, layer, direction, harmless_reference = select_direction(
        model_base,
        harmful_val,
        harmless_val,
        candidate_directions,
        harmless_mean,
        layers_to_evaluate,
        artifact_dir=os.path.join(cfg.artifact_path(), "select_direction"), 
        batch_size = batch_size)

    
    with open(save_path, "w") as f:
        json.dump({"pos": pos, 
                   "layer": layer},
                f, indent=4)

    return pos, layer, direction, harmless_reference




    
def evaluate_loss_for_datasets(cfg, model_base, fwd_pre_hooks, fwd_hooks, intervention_label):
    """Evaluate loss on datasets."""
    if not os.path.exists(os.path.join(cfg.artifact_path(), 'loss_evals')):
        os.makedirs(os.path.join(cfg.artifact_path(), 'loss_evals'))

    on_distribution_completions_file_path = os.path.join(cfg.artifact_path(), f'completions/harmless_baseline_completions.json')

    loss_evals = evaluate_loss(model_base, fwd_pre_hooks, fwd_hooks, batch_size=cfg.ce_loss_batch_size, n_batches=cfg.ce_loss_n_batches, completions_file_path=on_distribution_completions_file_path)

    with open(f'{cfg.artifact_path()}/loss_evals/{intervention_label}_loss_eval.json', "w") as f:
        json.dump(loss_evals, f, indent=4)

def run_rho_generation_and_cache(
    cfg,
    model_base,
    category_directions: torch.Tensor,
    harmless_reference: torch.Tensor,
    direction_pos: int,
    direction_layer: int,
    harmful_val: List[Dict],
    target_categories,
    rho_grid: List[float],
    use_existing: bool = False,
    sample_num: int = None,
    batch_size = 32,
    tailweight_ablation_study: bool = False
):
    import os
 
    param_dir = os.path.join(cfg.artifact_path(), "rho_search")
    os.makedirs(param_dir, exist_ok=True)

    def get_missing_rhos():
        missing = []
        for rho in rho_grid:
            tag = f"rho_{rho:.2f}"
            dataset_name = "harmful"
            comp_path = os.path.join(param_dir, f"{dataset_name}_{tag}_completions.json")
            if not os.path.exists(comp_path):
                missing.append(rho)
        return missing

    missing_rhos = get_missing_rhos()

    if use_existing and not missing_rhos:
        print("All completions found for specified rhos — skipping generation.")
        return

    if not missing_rhos:
        print("No rhos to generate — exiting.")
        return
    if not sample_num:
        sample_num = len(harmful_val)

    print(f"Generating completions for missing rhos: {missing_rhos}")
    generate_completions_for_rhos(
        cfg=cfg,
        model_base=model_base,
        harmful_val=harmful_val,
        harmless_reference=harmless_reference,
        category_directions=category_directions,
        direction_pos=direction_pos,
        direction_layer=direction_layer,
        target_categories = target_categories,
        rhos=missing_rhos,
        sample_num=sample_num,
        batch_size = batch_size,
        save_path = "rho_search",
        tailweight_ablation = tailweight_ablation_study
    )


def finalize_rho_and_direction(
    cfg,
    category_directions: torch.Tensor,
    direction_pos: int,
    direction_layer: int,
    target_categories,
    rho_grid: List[float],
    artifact_dir: str,
    nontarget_asr_target=0.1,
    use_existing: bool = False,
    tailweight_ablation_study: bool = False,
    verbose = True,
) -> Tuple[torch.Tensor, float]:

    param_dir = os.path.join(cfg.artifact_path(), "rho_search")
    os.makedirs(param_dir, exist_ok=True)

    rho_path = os.path.join(param_dir, f'best_rho_layer{direction_layer}_pos{direction_pos}.json')



    best_rho = evaluate_rhos(
        cfg=cfg,
        rhos=rho_grid,
        target_categories=target_categories,
        nontarget_asr_target=nontarget_asr_target,
        use_existing =use_existing,
        save_path = "rho_search",
        verbose = verbose
    )
    
    if os.path.exists(rho_path) and use_existing:
        try:
            with open(rho_path, "r") as f:
                data = json.load(f)
            prev_rho_val = data.get("best_rho", None)
            if prev_rho_val != best_rho:
                use_existing = False
                #force reruns of generation
        except Exception as e:
            print(f"Error reading {rho_path}: {e}")

    if best_rho == -1:
        print("Failure to find sufficient rho. Defaulting rho to 0.99 max")
        best_rho = 0.99


    # Build final direction

    original_directions = isolate_target_vectors(
        category_directions,
        targets=target_categories,
        rho=0,
    )

    directions = isolate_target_vectors(
        category_directions,
        targets=target_categories,
        rho=best_rho,
        return_projection = True,
        return_cond = True
    )

    original_direction = original_directions[direction_pos, direction_layer]

    final_direction = directions[0][direction_pos, direction_layer]
    final_projection = directions[1][direction_pos, direction_layer]
    final_cond_cov = directions[2][direction_pos, direction_layer].item()
    final_cond_proj = directions[3][direction_pos, direction_layer].item()

    torch.save(final_projection, f'{cfg.artifact_path()}/rho_search/final_projection.pt')
    torch.save(final_direction, f'{cfg.artifact_path()}/rho_search/final_direction.pt')

    # --- Profiling ---
    nonzero_count = (final_projection.abs() > 1e-6).sum().item()
    total_elements = final_projection.numel()
    sparsity_ratio = nonzero_count / total_elements
    l2_norm = final_projection.norm().item()
    mean_val = final_projection.mean().item()
    std_val = final_projection.std().item()

    cosine_sim = F.cosine_similarity(original_direction.unsqueeze(0), final_direction.unsqueeze(0)).item()
    angle_deg = torch.rad2deg(torch.acos(torch.clamp(torch.tensor(cosine_sim), -1.0, 1.0))).item()

    fraction_l2_removed = final_projection.norm().item() / (original_direction.norm().item() + 1e-6)
    energy_removed_ratio = final_projection.pow(2).sum().item() / (original_direction.pow(2).sum().item() + 1e-6)

    flat_proj = final_projection.cpu().numpy()
    proj_kurtosis = float(kurtosis(flat_proj))
    proj_skewness = float(skew(flat_proj))

    # --- Heavy Tail Analysis ---
    abs_proj = np.abs(flat_proj)
    heavy_threshold = abs_proj.mean() + 2 * abs_proj.std()
    num_heavy = int((abs_proj > heavy_threshold).sum())
    percent_heavy = float(num_heavy) / total_elements

    # --- Gini Impurity ---
    if abs_proj.sum() > 0:
        p = abs_proj / (abs_proj.sum() + 1e-12)
        gini_impurity = 1.0 - float((p ** 2).sum())
    else:
        gini_impurity = 0.0

    # Save rho + profiling to JSON
    with open(rho_path, 'w') as f:
        json.dump({
            "best_rho": best_rho,
            "direction_pos": direction_pos,
            "direction_layer": direction_layer,
            "condition_number_cov": final_cond_cov,
            "condition_number_proj": final_cond_proj,
            "projection_nonzero_count": nonzero_count,
            "projection_total_elements": total_elements,
            "projection_sparsity_ratio": sparsity_ratio,
            "projection_l2_norm": l2_norm,
            "projection_mean": mean_val,
            "projection_std": std_val,
            "cosine_similarity_to_original": cosine_sim,
            "angle_degrees": angle_deg,
            "fraction_l2_removed": fraction_l2_removed,
            "energy_removed_ratio": energy_removed_ratio,
            "kurtosis": proj_kurtosis,
            "skewness": proj_skewness,
            "heavy_tail_count": num_heavy,
            "heavy_tail_fraction": percent_heavy,
            "gini_impurity": gini_impurity
        }, f, indent=4)

    if tailweight_ablation_study:
        tailweight_direction = isolate_target_vectors(
            category_directions,
            targets=target_categories,
            rho=best_rho,
            tailweight_ablation=True
        )
        tailweight_direction = tailweight_direction[direction_pos, direction_layer]
        return tailweight_direction, best_rho, use_existing
    else:
        return final_direction, best_rho, use_existing

def rho_search_phase(
    *,
    cfg,
    max_new_tokens = None,
    model_path: str,
    category_directions: torch.Tensor,
    harmless_reference: torch.Tensor,
    pos: int,
    layer: int,
    val_data,
    target_categories,
    rho_grid: list[float],
    batch_size: int,
    use_existing: bool,
    nontarget_asr_target: float = 0.1,
    refine_points: int = 2,   # 0=skip refinement, 1=midpoint, 2=two interior points,
    tailweight_ablation_study: bool = False,
    verbose: bool = True,
):
    """
    Single phase with optional midpoint refinement:
      - generate/evaluate initial rho_grid
      - bracket the threshold from cached evaluations
      - create midpoints
      - generate/evaluate again on midpoints
      - finalize on the union of rhos
    """

    import os, gc, json
    import torch
    import torch.nn.functional as F
    import numpy as np
    from scipy.stats import kurtosis, skew

    if max_new_tokens is None:
        max_new_tokens = cfg.max_new_tokens

    # --- helpers ---


    def _read_metrics(param_dir: str, rhos: list[float]) -> dict[float, float]:
        """Read cached *_evaluations.json and return rho -> non-target metric."""
        out = {}
        for a in rhos:
            tag = f"rho_{a:.2f}"
            files = [f for f in os.listdir(param_dir) if f.endswith(f"{tag}_evaluations.json")]
            if not files:
                continue
            # prefer PINV_* if present; otherwise take the first
            files.sort(key=lambda s: (not s.startswith("PINV_"), s))
            with open(os.path.join(param_dir, files[0]), "r") as f:
                data = json.load(f)
            m = data.get("is_jailbreak_llamaguard3_nontarget", None)
            if m is not None:
                out[a] = float(m)
        return out

    def _bracket(metrics: dict[float, float], thresh: float) -> tuple[float | None, float | None]:
        """
        Return (lo, hi) such that:
        - lo = rho just below crossing
        - hi = rho just above crossing
        Crossing is defined by metric moving from > thresh to <= thresh (or vice versa).
        If no crossing, return (None, None).
        """
        if not metrics:
            return None, None

        candidates = sorted(metrics.items())  # sorted by rho
        for i in range(len(candidates) - 1):
            rho_i, val_i = candidates[i]
            rho_j, val_j = candidates[i + 1]

            if (val_i > thresh and val_j <= thresh) or (val_i <= thresh and val_j > thresh):
                return rho_i, rho_j

        return None, None

    def _snap_to_grid(x: float, grid: list[float]) -> float:
        return min(grid, key=lambda g: abs(g - x))

    def _interpolated_refine(
        metrics: dict[float, float],
        thresh: float,
        k: int,
        *,
        decimals: int = 2,
        exclude: set[float] | None = None,
    ) -> list[float]:
        """
        Pure secant-style refinement:
        - If bracket exists, interpolate via secant and add midpoints.
        - If no bracket, return [] (no refinement).
        """
        def q(x: float) -> float:
            return float(f"{x:.{decimals}f}")

        if not metrics or k <= 0:
            return []

        exclude = set() if exclude is None else set(exclude)
        lo, hi = _bracket(metrics, thresh)
        refined = []

        if lo is not None and hi is not None and lo < hi:
            v_lo, v_hi = metrics[lo], metrics[hi]
            denom = v_hi - v_lo
            if denom == 0:
                rho_hat = (lo + hi) / 2.0
            else:
                rho_hat = lo + (thresh - v_lo) * (hi - lo) / denom

            # Clip strictly inside bracket
            rho_hat = min(max(rho_hat, lo + 1e-6), hi - 1e-6)

            if k == 1:
                refined = [rho_hat]
            elif k == 2:
                refined = [(lo + rho_hat) / 2.0, (rho_hat + hi) / 2.0]
            else:
                refined = [rho_hat, (lo + rho_hat) / 2.0, (rho_hat + hi) / 2.0]

        # Round, dedupe, and clip to [0,1]
        seen = set()
        final = []
        for r in refined:
            qx = q(min(max(r, 0.0), 1.0))
            if qx not in exclude and qx not in seen:
                final.append(qx)
                seen.add(qx)

        return final




    # ---------- Round 1: generate on initial grid ----------

    global model_base

    cfg.max_new_tokens = max_new_tokens
    run_rho_generation_and_cache(
        cfg=cfg,
        model_base=model_base,
        category_directions=category_directions,
        harmless_reference=harmless_reference,
        direction_pos=pos,
        direction_layer=layer,
        harmful_val=val_data,
        target_categories=target_categories,
        rho_grid=rho_grid,
        use_existing=use_existing,
        batch_size=batch_size,
        tailweight_ablation_study=False,
    )

    # Offload generator before evaluate
    if verbose: print("[RhoSearch] Offloading generation model...")
    try:
        model_base.model = None
        model_base.tokenizer = None
    except Exception:
        pass
    try:
        del model_base
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    # Evaluate initial grid (writes *_evaluations.json)
    if verbose: print("[RhoSearch] Evaluating initial rho grid...")
    _ = evaluate_rhos(
        cfg=cfg,
        rhos=rho_grid,
        target_categories=target_categories,
        nontarget_asr_target=nontarget_asr_target,
        use_existing=False if not use_existing else True,
        save_path="rho_search",
        verbose=verbose,
    )
    unload_llamaguard_model()
    gc.collect()
    torch.cuda.empty_cache()

    # Read metrics and form bracket
    param_dir = os.path.join(cfg.artifact_path(), "rho_search")
    metrics1 = _read_metrics(param_dir, rho_grid)
    if verbose: print(f"[RhoSearch] Initial metrics: {metrics1}")
    lo, hi = _bracket(metrics1, nontarget_asr_target)
    if verbose: print(f"[RhoSearch] Initial bracket: lo={lo}, hi={hi}")

    # Build refinement rhos via interpolation (quantized)
    refine_rhos = []
    if refine_points > 0:
        already = set(metrics1.keys())
        refine_rhos = _interpolated_refine(
            metrics=metrics1,
            thresh=nontarget_asr_target,
            k=refine_points,
            decimals=2,
            exclude=already,
        )

    if verbose:
        if refine_points > 0 and not refine_rhos:
            print("[RhoSearch] No refinement rhos found; consider increasing precision or refine_points.")
        elif refine_rhos:
            print(f"[RhoSearch] Interpolated refinement rhos: {refine_rhos}")



    # ---------- Round 2: generate/evaluate on midpoints (if any) ----------
    if refine_rhos:
        # (Re)load generator for refinement — replace this with your loader if needed
        if verbose: print("[RhoSearch] Reloading generation model for refinement...")
        
        model_base = construct_model_base(model_path)

        run_rho_generation_and_cache(
            cfg=cfg,
            model_base=model_base,
            category_directions=category_directions,
            harmless_reference=harmless_reference,
            direction_pos=pos,
            direction_layer=layer,
            harmful_val=val_data,
            target_categories=target_categories,
            rho_grid=refine_rhos,
            use_existing=use_existing,
            batch_size=batch_size,
            tailweight_ablation_study=False,
        )

        # Offload generator
        if verbose: print("[RhoSearch] Offloading generation model (refinement)...")
        try:
            model_base.model = None
            model_base.tokenizer = None
        except Exception:
            pass
        try:
            del model_base
        except Exception:
            pass
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate refinement rhos
        if verbose: print("[RhoSearch] Evaluating refinement rhos...")
        _ = evaluate_rhos(
            cfg=cfg,
            rhos=refine_rhos,
            target_categories=target_categories,
            nontarget_asr_target=nontarget_asr_target,
            use_existing=True,   # will read generated completions
            save_path="rho_search",
            verbose=verbose,
        )
        unload_llamaguard_model()
        gc.collect()
        torch.cuda.empty_cache()

    # ---------- Finalize using the UNION of initial + refinement ----------
    all_rhos = sorted(set(rho_grid + refine_rhos))
    if verbose: print(f"[RhoSearch] Finalizing with rhos: {all_rhos}")

    final_dir, best_rho, updated_use_existing = finalize_rho_and_direction(
        cfg=cfg,
        category_directions=category_directions,
        direction_pos=pos,
        direction_layer=layer,
        target_categories=target_categories,
        rho_grid=all_rhos,
        artifact_dir=cfg.artifact_path(),
        use_existing=True,  # evaluations already cached
        nontarget_asr_target=nontarget_asr_target, 
        tailweight_ablation_study = tailweight_ablation_study,
        verbose=verbose,
    )

    if best_rho == -1:
        best_rho = max(all_rhos)
        if verbose:
            print("Gridsearch failed to meet threshold; defaulting to strongest rho in grid.")

    return final_dir, best_rho, updated_use_existing

def select_and_cache_filtered_categories(
    cfg,
    category_directions: Dict[str, Tuple[int, torch.Tensor]],
    target_categories: List[str],
    *,
    use_existing: bool = True,
    residual_threshold: float = 0.25,
    min_examples: int = 1,
    verbose: bool = True,
) -> List[str]:
    """
    Compute or load the filtered (dropped) category list.
    Saves/loads at: {cfg.artifact_path()}/generate_directions/filtered_categories.json

    JSON structure:
      {
        "dropped_list": [...],
        "residual_threshold": float,
        "min_examples": int
      }
    """
    dir_path = os.path.join(cfg.artifact_path(), "generate_directions")
    os.makedirs(dir_path, exist_ok=True)
    out_path = os.path.join(dir_path, "filtered_categories.json")

    # Try cache
    if use_existing and os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                data = json.load(f)
            if (
                abs(float(data.get("residual_threshold", -1.0)) - float(residual_threshold)) < 1e-12
                and int(data.get("min_examples", -1)) == int(min_examples)
            ):
                dropped = data.get("dropped_list", [])
                if verbose:
                    print(f"[filtered_categories] Loaded {len(dropped)} from cache: {out_path}")
                return dropped
            else:
                if verbose:
                    print("[filtered_categories] Cache params changed; recomputing...")
        except Exception as e:
            if verbose:
                print(f"[filtered_categories] Failed to read cache ({e}); recomputing...")

    # Compute fresh
    dropped = filter_exclude_categories(
        category_directions,
        target_categories,
        min_examples=min_examples,
        residual_threshold=residual_threshold,
        verbose=verbose,
    )

    # Save
    to_save = {
        "dropped_list": dropped,
        "residual_threshold": float(residual_threshold),
        "min_examples": int(min_examples),
    }
    try:
        with open(out_path, "w") as f:
            json.dump(to_save, f, indent=2)
        if verbose:
            print(f"[filtered_categories] Saved {len(dropped)} to {out_path}")
    except Exception as e:
        if verbose:
            print(f"[filtered_categories] Failed to write cache ({e})")

    return dropped

def compute_and_save_similarity(model, cfg, harmful_data, harmless_data, tokenize_fn,
                                fwd_pre_hooks=[], fwd_hooks=[], batch_size=32,
                                system_prompt=None, fraction=0.10, use_existing=False):
    """
    Computes cosine similarity between mean activations of harmful and harmless data for each layer
    and returns the layers ranked by the least similarities.
    """

    num_layers = model.config.num_hidden_layers
    save_dir = os.path.join(cfg.artifact_path(), 'select_direction')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "COSMIC_evaluation_layers.json")

    if use_existing and os.path.exists(save_path):
        with open(save_path, "r") as f:
            return json.load(f)["ranked_layers"][:int(fraction * num_layers)]

    harmful_outputs = torch.zeros((len(harmful_data), num_layers, model.config.hidden_size), device=model.device)
    harmless_outputs = torch.zeros((len(harmless_data), num_layers, model.config.hidden_size), device=model.device)

    progress = tqdm(total=2, desc="Finding COSMIC Evaluation Layers")

    for data, outputs in zip([harmful_data, harmless_data], [harmful_outputs, harmless_outputs]):
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            tokenized = tokenize_fn(instructions=batch, system=system_prompt)

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                hidden_states = model(
                    input_ids=tokenized.input_ids.to(model.device),
                    attention_mask=tokenized.attention_mask.to(model.device),
                    output_hidden_states=True
                ).hidden_states[1:]  # skip embeddings

            for j, layer_out in enumerate(hidden_states):
                outputs[i:i+batch_size, j] = layer_out[:, -1]

        progress.update(1)

    progress.close()

    normalized_harmful = harmful_outputs.mean(dim=0)
    normalized_harmless = harmless_outputs.mean(dim=0)

    normalized_harmful = normalized_harmful / normalized_harmful.norm(dim=-1, keepdim=True)
    normalized_harmless = normalized_harmless / normalized_harmless.norm(dim=-1, keepdim=True)

    similarities = torch.sum(normalized_harmful * normalized_harmless, dim=-1).cpu().numpy()
    del normalized_harmful, normalized_harmless 
    ranked_layers = np.argsort(similarities).tolist()

    with open(save_path, "w") as f:
        json.dump({"ranked_layers": ranked_layers}, f, indent=4)

    return ranked_layers[:int(fraction * num_layers)]


def run_generations(
    cfg, model_base, fwd_pre_hooks, fwd_hooks,
    tag_prefix, dataset_split, batch_size, save_path=None, use_existing  = False, secondary = True
):
    # Harmful dataset
    generate_and_save_completions_for_dataset(
        cfg,
        model_base,
        fwd_pre_hooks,
        fwd_hooks,
        f'{tag_prefix}',
        'harmful',
        batch_size=batch_size,
        dataset=dataset_split,
        save_path=save_path,
        use_existing = use_existing
    )
    if secondary:
        # Evaluation datasets
        for dataset_name in cfg.evaluation_datasets:
            generate_and_save_completions_for_dataset(
                cfg,
                model_base,
                fwd_pre_hooks,
                fwd_hooks,
                f'ablation_{tag_prefix}',
                dataset_name,
                batch_size=batch_size,
                save_path=save_path,
                use_existing = use_existing
            )

def run_evaluations(
    cfg,
    tag_prefix,
    dataset_split=True,
    eval_methodologies=None,
    target_categories=None,
    save_path=None,
    use_existing=False,
    batch_size = 32
):
    if dataset_split:
        evaluate_completions_and_save_results_for_dataset(
            cfg,
            tag_prefix,
            "harmful",
            eval_methodologies=eval_methodologies,
            target_categories=target_categories,
            save_path=save_path,
            use_existing=use_existing,
            batch_size = batch_size,
        )
    else:
        for dataset_name in cfg.evaluation_datasets:
            evaluate_completions_and_save_results_for_dataset(
                cfg,
                f"ablation_{tag_prefix}",
                dataset_name,
                eval_methodologies=eval_methodologies,
                save_path=save_path,
                use_existing=use_existing,
                batch_size = batch_size,
            )


def get_category_groups_by_name(target_name: str):
    predefined_groups = {
        "wmd_bio": ["WMDP_BIO"],
        "wmd_chem": ["WMDP_CHEM"],
        "wmd_cyber": ["WMDP_CYBER"],
    }

    predefined_exclusions = {
        "wmd_bio": [],
        "wmd_chem": [],
        "wmd_cyber": [
            "Malware",
        ],
    }

    if target_name not in predefined_groups:
        raise ValueError(f"Unknown target category group: {target_name}")
    return predefined_groups[target_name], predefined_exclusions[target_name]


def run_pipeline(model_path, target_group, tailweight_ablation_study, use_existing, target_size_study):
    target_categories, static_excludes = get_category_groups_by_name(target_group)

    global model_base
    model_base = construct_model_base(model_path)

    nontarget_asr_target = 0.1
    batch_size_denom = 1


    with torch.inference_mode():
        print(f"NOW RUNNING TOPIC - {target_group}")
        model_alias = os.path.basename(model_path) + f"-RepIt-{target_group}"
        cfg = Config(model_alias=model_alias, model_path=model_path)


        # --- First pass: static excludes only ---
        harmful_train, harmless_train, harmful_val, nontarget_val, \
        harmless_val, harmful_test, harmless_test = load_and_sample_beast_datasets(
            cfg,
            target_group=target_group,
            exclude=static_excludes,
            target_categories=target_categories,
            seed=8,
        )

        # Generate category directions
        category_directions, harmless_mean = generate_and_save_category_directions(
            cfg, model_base, harmful_train, harmless_train,
            batch_size=128 // batch_size_denom, use_existing=use_existing
        )

        # Similarity scan → choose layers
        layers_to_evaluate = compute_and_save_similarity(
            model_base.model,
            cfg,
            harmful_train,
            harmless_train,
            model_base.tokenize_instructions_fn,
            batch_size=64 // batch_size_denom,
            fwd_pre_hooks=[],
            fwd_hooks=[],
            use_existing=use_existing
        )

        print("Layers to evaluate are", layers_to_evaluate)

        # Get candidate refusal directions
        naive_candidate_directions = isolate_target_vectors(
            category_directions,
            targets=target_categories,
            rho=0.0,
        )

        # Pick best pos/layer from first pass
        pos, layer, unablated_direction, harmless_reference = select_and_save_direction(
            cfg, model_base,
            nontarget_val, harmless_val,
            naive_candidate_directions, harmless_mean,
            layers_to_evaluate,
            batch_size=64 // batch_size_denom,
            use_existing=use_existing
        )


        del naive_candidate_directions, harmful_train, harmless_train, 

        primary_grid = [0.00, 0.33, 0.66, 0.99]

        final_ablate_direction, best_rho, use_existing = rho_search_phase(
            cfg=cfg,
            max_new_tokens = cfg.max_new_tokens,
            model_path=model_path,
            category_directions=category_directions,
            harmless_reference=harmless_reference,
            pos=pos,
            layer=layer,
            val_data=harmful_val,
            target_categories=target_categories,
            rho_grid=primary_grid,
            batch_size=128 // batch_size_denom,
            use_existing=use_existing,
            nontarget_asr_target=nontarget_asr_target,
            refine_points=3,
            verbose=True
        )


        model_base = construct_model_base(model_path)

        nontarget_direction = torch.mean(torch.stack([
            n * direction[pos, layer]
            for cat, (n, direction) in category_directions.items()
            if cat not in target_categories
        ], dim=0), dim=0)


        nontarget_refusal_pre_hooks = [(model_base.model_block_modules[layer], get_affine_direction_ablation_input_pre_hook(direction=nontarget_direction, reference = harmless_reference))]
        nontarget_refusal_hooks = []

        unchanged_refusal_pre_hooks = [(model_base.model_block_modules[layer], get_affine_direction_ablation_input_pre_hook(direction=unablated_direction, reference = harmless_reference))]
        unchanged_refusal_hooks = []

        partial_projection = unablated_direction - final_ablate_direction
        partial_projection_pre_hooks = [(model_base.model_block_modules[layer], get_affine_direction_ablation_input_pre_hook(direction=partial_projection, reference = harmless_reference))]
        partial_projection_hooks = []

        ablation_fwd_pre_hooks = [(model_base.model_block_modules[layer], get_affine_direction_ablation_input_pre_hook(direction=final_ablate_direction, reference = harmless_reference))]
        ablation_fwd_hooks = []

        _safe_release(
            "harmful_val", "nontarget_val", "harmless_val",
            "categories", "best_rho"
        )

        print("Running Baseline")
        run_generations(
            cfg, model_base,
            [], [],
            tag_prefix=f'Baseline_ablation',
            dataset_split = harmful_test,
            batch_size=128 // batch_size_denom,
            save_path = "completions/Baseline",
            use_existing = use_existing,
        )


        print("Running Nontarget Ablation Results")
        run_generations(
            cfg, model_base,
            nontarget_refusal_pre_hooks, nontarget_refusal_hooks,
            tag_prefix=f'Nontarget_ablation',
            dataset_split = harmful_test,
            batch_size=128 // batch_size_denom,
            use_existing = use_existing,
            save_path = "completions/Nontarget",
        )
        # Generate completions for the primary harmful dataset
        print("Running Natural Ablation Results")
        run_generations(
            cfg, model_base,
            unchanged_refusal_pre_hooks, unchanged_refusal_hooks,
            tag_prefix=f'Natural_ablation',
            dataset_split = harmful_test,
            batch_size=128 // batch_size_denom,
            save_path = "completions/Natural",
            use_existing = use_existing,
        )


        # Generate completions for the primary harmful dataset
        print("Running RepIt Ablation Results")
        run_generations(
            cfg, model_base,
            ablation_fwd_pre_hooks, ablation_fwd_hooks,
            tag_prefix=f'RepIt_ablation',
            dataset_split = harmful_test,
            batch_size=128 // batch_size_denom,
            save_path = "completions/RepIt",
            use_existing = use_existing
        )

        print("Running Projection Ablation Results")
        run_generations(
            cfg, model_base,
            partial_projection_pre_hooks, partial_projection_hooks,
            tag_prefix=f'Projection_ablation',
            dataset_split = harmful_test,
            batch_size=128 // batch_size_denom,
            save_path = "completions/Projection",
            use_existing = use_existing
        )

        if tailweight_ablation_study:
            print("Now running tailweight directions")


            tailweight_ablate_direction = isolate_target_vectors(
                category_directions,
                targets=target_categories,
                tailweight_ablation = True,
                rho=best_rho,
            )[pos, layer]

    
            tailweight_fwd_pre_hooks = [(model_base.model_block_modules[layer], get_affine_direction_ablation_input_pre_hook(direction=tailweight_ablate_direction, reference = harmless_reference))]
            tailweight_fwd_hooks = []

            # Generate completions for the harmful dataset with tailweight intervention
            run_generations(
                cfg, model_base,
                tailweight_fwd_pre_hooks, tailweight_fwd_hooks,
                tag_prefix=f'RepIt_tailweight_ablation',
                dataset_split=harmful_test,
                batch_size=128 // batch_size_denom,
                save_path = "completions/RepIt_tail",
                use_existing = use_existing
            )

            _safe_release("tailweight_ablate_direction")

        if target_size_study:
            # optionally subset the test set if it's really large
            target_test = harmful_test

            for seed in range(20, 25):
                for target_size in [12, 24]:
                    print(f"Running seed {seed} with target size {target_size}")
                    harmful_train, harmless_train, _, _, \
                    _, _, _ = load_and_sample_beast_datasets(
                        cfg,
                        target_group=target_group,
                        target_categories=target_categories,
                        target_size = target_size,
                        seed = seed
                    )

                    tgtsize_category_directions, harmless_mean = generate_and_save_category_directions(cfg, model_base, harmful_train, harmless_train, batch_size = 128//batch_size_denom, use_existing = False, save = False)

                    # --- Target Size Direction Ablation ---
                    target_size_direction = isolate_target_vectors(
                        tgtsize_category_directions,
                        targets=target_categories,
                        tailweight_ablation = False, 
                        rho=best_rho,
                    )[pos, layer]

                    ablation_fwd_pre_hooks = [(model_base.model_block_modules[layer], get_affine_direction_ablation_input_pre_hook(direction=target_size_direction, reference = harmless_reference))]

                    save_path = f"completions/target_size_{target_size}"
                    run_generations(
                        cfg, model_base,
                        ablation_fwd_pre_hooks, [],
                        tag_prefix=f'RepIt_ablation_{seed}',
                        dataset_split=target_test,
                        batch_size=128 // batch_size_denom,
                        save_path=save_path,
                        use_existing = use_existing,
                        secondary = False #You can run the secondaries (Advbench, harmbench, etc) if you want
                    )


                    # --- Optional Tailweight Ablation within the loop ---
                    # We don't run this in the actual paper because it just takes super long
                    """if tailweight_ablation_study:
                        tgtsize_tailweight_direction = isolate_target_vectors(
                            tgtsize_category_directions,
                            targets=target_categories,
                            tailweight_ablation = True,
                            rho=best_rho,
                        )[pos, layer]

                        save_path = f"completions/target_size_{target_size}_tail"
                        tailweight_fwd_pre_hooks = [(model_base.model_block_modules[layer], get_affine_direction_ablation_input_pre_hook(direction=tgtsize_tailweight_direction, reference = harmless_reference))]
                        run_generations(
                            cfg, model_base,
                            tailweight_fwd_pre_hooks, [],
                            tag_prefix=f'RepIt_tailweight_{seed}',
                            dataset_split=target_test,
                            batch_size=128 // batch_size_denom,
                            save_path=save_path,
                            use_existing = use_existing
                        )"""

    print("now running evaluations")
    #we load in llamaguard now for evals
    #so we're deleting the model to spare your vram
    print("Clearing memory before evaluations...")

    # Null out fields and references
    model_base.model = None
    model_base.tokenizer = None
    del model_base

    _safe_release(
        "harmless_train", "harmful_train", "harmless_val", "harmful_val",
        "harmless_test", "harmful_test",
        "category_directions", "naive_candidate_directions",
        "pos", "layer", "unablated_direction", "harmless_reference",
        "final_ablate_direction", "best_rho", "tailweight_ablate_direction",
        "tgtsize_category_directions", "target_size_direction", "tgtsize_tailweight_direction"
    )


    print("Memory cleared.")

    run_evaluations(cfg, "Baseline_ablation",
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                    target_categories=target_categories,
                    save_path="completions/Baseline",
                    use_existing=use_existing,
                    batch_size = 128 // (batch_size_denom * 2)
                    )

    run_evaluations(cfg, "Baseline_ablation",
                    dataset_split=False,
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                    save_path="completions/Baseline",
                    use_existing=use_existing,
                    batch_size = 128 // (batch_size_denom * 2)
                    )

    # === Evaluation for Nontarget ===
    run_evaluations(cfg, "Nontarget_ablation",
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                    target_categories=target_categories,
                    save_path="completions/Nontarget",
                    use_existing=use_existing,
                    batch_size = 128 // (batch_size_denom * 2)
                    )

    run_evaluations(cfg, "Nontarget_ablation",
                dataset_split = False, 
                eval_methodologies=cfg.jailbreak_eval_methodologies,
                save_path="completions/Nontarget",
                use_existing=use_existing,
                batch_size = 128 // (batch_size_denom * 2)
                )

    # === Evaluation for Natural ===
    run_evaluations(cfg, "Natural_ablation",
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                    target_categories=target_categories,
                    save_path="completions/Natural",
                    use_existing=use_existing,
                    batch_size = 128 // (batch_size_denom * 2)
                    )

    run_evaluations(cfg, "Natural_ablation",
                    dataset_split=False,
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                    save_path="completions/Natural",
                    use_existing=use_existing,
                    batch_size = 128 // (batch_size_denom * 2)
                    )

    # === Evaluation for standard RepIt ===
    run_evaluations(cfg, "RepIt_ablation",
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                    target_categories=target_categories,
                    save_path="completions/RepIt",
                    use_existing=use_existing,
                    batch_size = 128 // (batch_size_denom * 2)
                    )

    run_evaluations(cfg, "RepIt_ablation",
                    dataset_split=False,
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                    save_path="completions/RepIt",
                    use_existing=use_existing,
                    batch_size = 128 // (batch_size_denom * 2)
                    )

    # === Evaluation for standard RepIt ===
    run_evaluations(cfg, "Projection_ablation",
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                    target_categories=target_categories,
                    save_path="completions/Projection",
                    use_existing=use_existing,
                    batch_size = 128 // (batch_size_denom * 2)
                    )

    run_evaluations(cfg, "Projection_ablation",
                    dataset_split=False,
                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                    save_path="completions/Projection",
                    use_existing=use_existing,
                    batch_size = 128 // (batch_size_denom * 2)
                    )

    # === Evaluation for tailweight ablation ===
    if tailweight_ablation_study:
        run_evaluations(cfg, "RepIt_tailweight_ablation",
                        eval_methodologies=cfg.jailbreak_eval_methodologies,
                        target_categories=target_categories,
                        save_path="completions/RepIt_tail",
                        use_existing=use_existing,
                        batch_size = 128 // (batch_size_denom * 2)
                        )

        run_evaluations(cfg, "RepIt_tailweight_ablation",
                        dataset_split=False,
                        eval_methodologies=cfg.jailbreak_eval_methodologies,
                        save_path="completions/RepIt_tail",
                        use_existing=use_existing,
                        batch_size = 128 // (batch_size_denom * 2)
                        )

    # === Evaluation for target size study ===
    if target_size_study:
        for seed in range(20, 25):
            for target_size in [12, 24]:
                save_path = f"completions/target_size_{target_size}"

                run_evaluations(cfg, f"RepIt_ablation_{seed}",
                                eval_methodologies=cfg.jailbreak_eval_methodologies,
                                target_categories=target_categories,
                                save_path=save_path,
                                use_existing=use_existing,
                                batch_size = 128 // (batch_size_denom * 2)
                                )

                #also not in main study, kept in case others would like it
                """run_evaluations(cfg, f"RepIt_ablation_{seed}",
                                dataset_split=False,
                                eval_methodologies=cfg.jailbreak_eval_methodologies,
                                save_path=save_path,
                                use_existing=use_existing,
                                batch_size = 128 // (batch_size_denom * 2)
                                )   

                if tailweight_ablation_study:
                    save_path = f"completions/target_size_{target_size}_tail"

                    run_evaluations(cfg, f"RepIt_tailweight_{seed}",
                                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                                    target_categories=target_categories,
                                    save_path=save_path,
                                    use_existing=use_existing,
                                    batch_size = 128 // (batch_size_denom * 2)
                                    )

                    run_evaluations(cfg, f"RepIt_tailweight_{seed}",
                                    dataset_split=False,
                                    eval_methodologies=cfg.jailbreak_eval_methodologies,
                                    save_path=save_path,
                                    use_existing=use_existing,
                                    batch_size = 128 // (batch_size_denom * 2)
                                   )"""
    unload_llamaguard_model()
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(model_path=args.model_path, target_group=args.target_group, 
                                                tailweight_ablation_study = args.tailweight_ablation_study,
                                                use_existing = args.use_existing,
                                                target_size_study = args.target_size_study)
