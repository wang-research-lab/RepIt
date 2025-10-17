import os
import gc
import json
import random
import numpy as np
from typing import List, Tuple, Union, Dict, Optional
from collections import defaultdict


import torch, random, gc
from torch import Tensor
import torch.nn.functional as F


from pipeline.utils.hook_utils import add_hooks, get_affine_direction_ablation_input_pre_hook
from pipeline.submodules.completions_helper import (
    generate_and_save_completions_for_dataset,
    evaluate_completions_and_save_results_for_dataset
)

from typing import Dict, Tuple, Union, List

from typing import Dict, Tuple, Union, List
import torch

def isolate_target_vectors(
    cat_vecs: Dict[str, Tuple[int, torch.Tensor]],
    targets: Union[str, List[str]],
    rho: float = 0.75,                 # fraction of non-target energy to remove
    eps: float = 1e-6,
    tailweight_ablation: bool = False,
    return_projection: bool = False,
    ridge_scale: float = 1e-4,         # base ridge factor for covariance
    weight_mode: str = "invnorm",      # "invnorm", "uniform", or "sim"
    return_cond: bool = False          # return condition numbers
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    """
    RepIt disentangling with whitening + QR-based projection:

      1) Build ridge-regularized covariance C = (R^T W R)/n + λI.
      2) Whiten target and non-targets via Cholesky factor of C.
      3) Apply thin QR decomposition on whitened non-targets to obtain an
         explicit orthonormal basis for the residual span.
      4) Project whitened target onto this span and subtract a scaled fraction:
         α = 1 - sqrt(1 - ρ).
      5) Map back (unwhiten) to the original representation space.

    Returns:
      - direction
      - optionally: projection tensor
      - optionally: cond_cov (condition number of covariance matrix C)
      - optionally: cond_proj (condition number of whitened QR basis)
    """


    # --- target set-up ---
    if isinstance(targets, str):
        targets = [targets]
    targets = [t for t in targets if t in cat_vecs]
    if not targets:
        print("[Warning] No valid targets found.")
        outs = [None]
        if return_projection: outs.append(None)
        if return_cond: outs.extend([None, None])
        return tuple(outs) if len(outs) > 1 else outs[0]

    sample = next(iter(cat_vecs.values()))[1]
    p_dim, l_dim, d_dim = sample.shape
    out_dtype = sample.dtype
    fp = torch.float32

    # mean target vector
    tgt_vecs_raw = [cat_vecs[t][1].to(fp) for t in targets]
    tgt_mean = torch.stack(tgt_vecs_raw, dim=0).mean(dim=0)  # [p,l,d]

    # non-target list
    non_targets = [k for k in cat_vecs if k not in targets]
    if not non_targets:
        direction = tgt_mean.to(out_dtype)
        outs = [direction]
        if return_projection: outs.append(torch.zeros_like(direction))
        if return_cond: outs.extend([
            torch.zeros((p_dim, l_dim), dtype=out_dtype),
            torch.zeros((p_dim, l_dim), dtype=out_dtype)
        ])
        return tuple(outs) if len(outs) > 1 else outs[0]

    # --- outputs ---
    cleaned_sum = torch.zeros_like(tgt_mean, dtype=fp)
    final_projection = torch.zeros_like(tgt_mean, dtype=fp) if return_projection else None
    cond_cov = torch.zeros((p_dim, l_dim), dtype=fp) if return_cond else None
    cond_proj = torch.zeros((p_dim, l_dim), dtype=fp) if return_cond else None

    # scaling factor for rho
    alpha = 1.0 - (max(0.0, min(1.0, 1.0 - rho)))**0.5

    def lower_tri_solve(L, X):
        return torch.linalg.solve_triangular(L, X.unsqueeze(1) if X.ndim == 1 else X, upper=False)

    # --- main loop ---
    for vec in tgt_vecs_raw:
        edited = torch.empty_like(vec, dtype=fp)

        for p in range(p_dim):
            for l in range(l_dim):
                v_orig = vec[p, l].to(fp).view(-1)  # [d]
                if torch.allclose(v_orig, torch.zeros_like(v_orig)):
                    edited[p, l] = v_orig
                    if return_projection: final_projection[p, l] += 0.0
                    continue

                # stack non-targets
                rows = [cat_vecs[k][1][p, l].to(fp).view(-1) for k in non_targets]
                R = torch.stack(rows, dim=0)  # [n, d]
                n = R.shape[0]

                # category weights
                if weight_mode == "invnorm":
                    w = 1.0 / (R.norm(dim=1) + eps)
                elif weight_mode == "sim":
                    sims = torch.nn.functional.cosine_similarity(R, v_orig.unsqueeze(0), dim=1)
                    w = sims.abs()
                else:
                    w = torch.ones(n, dtype=fp, device=R.device)
                w = w / (w.sum() + eps)
                Rw = R * w.unsqueeze(1)

                # covariance + ridge
                diag_mean = Rw.pow(2).mean().item()
                lam = ridge_scale * diag_mean + 1e-12
                C = (Rw.t() @ Rw) / max(1, n) + lam * torch.eye(d_dim, dtype=fp, device=R.device)
                try:
                    L = torch.linalg.cholesky(C, upper=False)
                except RuntimeError:
                    lam2 = max(lam * 10.0, 1e-8)
                    C = (Rw.t() @ Rw) / max(1, n) + lam2 * torch.eye(d_dim, dtype=fp, device=R.device)
                    L = torch.linalg.cholesky(C, upper=False)

                # whitening
                v_w = lower_tri_solve(L, v_orig).view(-1)       # [d]
                Rw_cols = lower_tri_solve(L, Rw.t())            # [d, n]

                # QR projection in whitened space
                try:
                    Q, _ = torch.linalg.qr(Rw_cols, mode="reduced")  # [d, r]
                    pv = (Q @ (Q.transpose(0, 1) @ v_w)).view(-1)
                except RuntimeError:
                    pv = torch.zeros_like(v_w)

                # subtraction + unwhiten
                v_w_clean = v_w - alpha * pv
                v_clean = (L @ v_w_clean).view(-1)

                edited[p, l] = v_clean
                if return_projection:
                    final_projection[p, l] += (v_orig - v_clean)
                if return_cond:
                    cond_cov[p, l] += torch.linalg.cond(C).item()
                    cond_proj[p, l] += torch.linalg.cond(Rw_cols).item()

        cleaned_sum += edited

    # average over targets
    cleaned_tensor = cleaned_sum / len(tgt_vecs_raw)
    direction = torch.nan_to_num(cleaned_tensor).to(out_dtype)

    if return_cond:
        cond_cov = (cond_cov / len(tgt_vecs_raw)).to(out_dtype)
        cond_proj = (cond_proj / len(tgt_vecs_raw)).to(out_dtype)

    outs = [direction]
    if return_projection: outs.append((final_projection / len(tgt_vecs_raw)).to(out_dtype))
    if return_cond: outs.extend([cond_cov, cond_proj])
    return tuple(outs) if len(outs) > 1 else outs[0]




def evaluate_rhos(
    cfg,
    target_categories: Union[str, List[str]],
    rhos: List[float],
    nontarget_asr_target=0.1,
    use_existing: bool = False,
    save_path: str = "rho_search",
    verbose: bool = True,
) -> Optional[float]:
    import os, json

    triples = []  # (rho, nontarget_asr, target_asr)

    for rho in rhos:
        if verbose:
            print(f"Evaluating rho = {rho:.2f}")
        tag = f"rho_{rho:.2f}"
        ablation_type = "harmful"
        result_path = os.path.join(cfg.artifact_path(), f"{save_path}/{ablation_type}_{tag}_evaluations.json")

        if not use_existing or not os.path.exists(result_path):
            evaluate_completions_and_save_results_for_dataset(
                cfg=cfg,
                intervention_label=tag,
                dataset_name=ablation_type,
                eval_methodologies=cfg.jailbreak_eval_methodologies,
                target_categories=target_categories,
                save_path=save_path,
            )

        if not os.path.exists(result_path):
            print(f"Missing eval file for rho {rho}")
            continue

        with open(result_path, "r") as f:
            data = json.load(f)

        score_target = data.get("is_jailbreak_llamaguard3_target", None)
        score_nontarget = data.get("is_jailbreak_llamaguard3_nontarget", None)

        if score_target is None or score_nontarget is None:
            if verbose:
                print(f"Rho {rho:.2f}: missing scores, skipping.")
            continue

        if verbose:
            print(f"Rho {rho:.2f}: Target ASR = {score_target:.3f}, Nontarget ASR = {score_nontarget:.3f}")

        triples.append((float(rho), float(score_nontarget), float(score_target)))

    if not triples:
        print("No rhos with valid scores.")
        return None

    # Feasible set: non-target ASR leq the threshold
    feasible = [(a, n, t) for a, n, t in triples if n <= nontarget_asr_target + 0.001]

    if feasible:
        # Select rho with max target ASR; tie-break prefers smaller rho
        best_rho, best_nont, best_targ = max(feasible, key=lambda x: (x[2], -x[0]))
        if verbose:
            print(
                f"Selected rho={best_rho:.4f} with highest target ASR under "
                f"non-target<{nontarget_asr_target:.2f} (non-target={best_nont:.3f}, target={best_targ:.3f})"
            )
        return float(best_rho)

    # If no feasible point, fall back to rho with minimum non-target ASR
    fallback_rho, fallback_nont, fallback_targ = min(triples, key=lambda x: x[1])
    if verbose:
        print(
            f"No rho met non-target<{nontarget_asr_target:.2f}. "
            f"Falling back to rho={fallback_rho:.4f} with lowest non-target ASR "
            f"(non-target={fallback_nont:.3f}, target={fallback_targ:.3f})."
        )
    return float(fallback_rho)



def generate_completions_for_rhos(
    cfg,
    model_base,
    harmful_val,
    harmless_reference,
    direction_pos,
    direction_layer,
    category_directions: Dict[str, Tuple[int, torch.Tensor]],
    target_categories: Union[str, List[str]],
    rhos: List[float] = None,
    sample_num: int = None,
    tailweight_ablation: bool = False,
    batch_size: int = 32,
    save_path: str = "rho_search"
):
    if rhos is None:
        return
    
    if sample_num is None:
        sample_num = len(harmful_val)

    for rho in rhos:
        # --- isolate direction ---
        w = isolate_target_vectors(
            category_directions,
            target_categories,
            rho=rho,
            tailweight_ablation=tailweight_ablation
        )
        direction = w[direction_pos, direction_layer]  # [d]

        print(f"[rho={rho:.2f}] Running generation phase")

        # --- hook setup ---
        pre_hooks = [
            (
                model_base.model_block_modules[direction_layer],
                get_affine_direction_ablation_input_pre_hook(
                    direction=direction,
                    reference=harmless_reference,
                )
            )
        ]

        # --- sample harmful data for this run ---
        sampled_harmful = random.sample(
            harmful_val,
            min(sample_num, len(harmful_val))
        )
        tag = f"rho_{rho:.2f}"

        # --- safe generation ---
        with torch.inference_mode():  
            generate_and_save_completions_for_dataset(
                cfg=cfg,
                model_base=model_base,
                fwd_pre_hooks=pre_hooks,
                fwd_hooks=[],
                intervention_label=tag,
                dataset_name="harmful",
                dataset=sampled_harmful,
                batch_size=batch_size,
                save_path=save_path,
            )

        # --- cleanup aggressively after each rho ---
        del w, direction, pre_hooks, sampled_harmful
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()


def filter_exclude_categories(
    cat_means: Dict[str, Tuple[int, torch.Tensor]],
    target_categories: List[str],
    *,
    min_examples: int = 1,
    residual_threshold: float = 0.25,  # remove if residual < threshold (after unit-norm)
    verbose: bool = True,
) -> List[str]:
    """
    Filter out categories that are conditionally redundant w.r.t. the target set.
    We:
      • Convert each category tensor to a 1D vector by averaging all dims except the last.
      • Normalize vectors.
      • Remove categories whose residual norm after projecting out the target centroid
        is < residual_threshold. (Residual norm == sin(theta) in [0,1].)

    Returns: sorted list of dropped category names.
    """
    eps = 1e-12

    def to_vec(T: torch.Tensor) -> torch.Tensor:
        # Reduce across all dims except the last to get shape (..., D) -> (D)
        if T.ndim == 1:
            return T
        reduce_dims = tuple(range(T.ndim - 1))
        return T.mean(dim=reduce_dims)

    # Flatten centroids -> 1D vectors
    flat_centroids: Dict[str, torch.Tensor] = {}
    for cat, (n, T) in cat_means.items():
        if n < min_examples or T.numel() == 0:
            continue
        v = to_vec(T)
        if v.numel() == 0:
            continue
        flat_centroids[cat] = v

    # Build target centroid
    tgt_vecs = [flat_centroids[c] for c in target_categories if c in flat_centroids]
    if not tgt_vecs:
        raise ValueError("No valid target categories found in cat_means.")
    tgt_centroid = torch.stack(tgt_vecs, dim=0).mean(dim=0)
    tgt_norm = tgt_centroid.norm()
    if tgt_norm.item() < eps:
        raise ValueError("Target centroid has near-zero norm; cannot compute residuals.")
    tgt_c = tgt_centroid / (tgt_norm + eps)

    removed = []
    kept = []

    for cat, v in flat_centroids.items():
        if cat in target_categories:
            continue

        v_norm = v.norm()
        if v_norm.item() < eps:
            # Zero vector: purely redundant
            residual_mag = 0.0
        else:
            v_c = v / (v_norm + eps)
            # cos = <v_c, tgt_c>; residual norm = sqrt(1 - cos^2)
            cos = torch.dot(v_c, tgt_c).clamp(min=-1.0, max=1.0)
            residual_mag = torch.sqrt(torch.clamp(1.0 - cos * cos, min=0.0, max=1.0)).item()

        if residual_mag < residual_threshold:
            removed.append((cat, residual_mag))
        else:
            kept.append((cat, residual_mag))

    if verbose:
        print(f"[filter_by_conditional_redundancy] threshold={residual_threshold:.4f}")
        if removed:
            print(f"Removed {len(removed)} (low unique content): {sorted([c for c, _ in removed])}")
        else:
            print("No categories removed!")

    return sorted([c for c, _ in removed])