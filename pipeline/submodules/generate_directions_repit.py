import os
import json
import collections
from typing import List, Dict, Tuple, Union

import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn.functional as F
from jaxtyping import Float
import gc

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase



# --------------------------- Mean-activation helpers ----------------------------
def get_mean_activations_pre_hook(layer, cache: Float[Tensor, "pos layer d_model"],
                                  n_samples: int, positions: List[int]):
    def hook_fn(module, input_):
        act: Float[Tensor, "batch seq d_model"] = input_[0].clone().to(cache)
        cache[:, layer] += act[:, positions, :].sum(dim=0) / n_samples
    return hook_fn

def get_activations(model, instructions: List[str],
                    tokenize_fn, blocks: List[torch.nn.Module],
                    *, batch_size: int = 32, positions=[-1],
                    system_prompt: str = None) -> Float[Tensor, "n pos layer d_model"]:
    """
    Returns the layer-input activations per prompt (n, pos, layer, d_model).
    """
    torch.cuda.empty_cache()

    n_samples = len(instructions)
    n_pos = len(positions)
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size

    all_acts = torch.zeros((n_samples, n_pos, n_layers, d_model),
                           dtype=torch.float32, device=model.device)

    def make_hook(layer_idx, prompt_idx):
        def hook_fn(module, input_):
            act = input_[0].detach()  # [batch, seq, d]
            for b in range(act.size(0)):
                for p_idx, p in enumerate(positions):
                    all_acts[prompt_idx + b, p_idx, layer_idx] = act[b, p]
        return hook_fn

    for i in range(0, n_samples, batch_size):
        batch_prompts = instructions[i:i + batch_size]
        inputs = tokenize_fn(instructions=batch_prompts,
                             system=system_prompt)

        hooks = [(blocks[l], make_hook(l, i)) for l in range(n_layers)]

        with add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
            model(input_ids=inputs.input_ids.to(model.device),
                  attention_mask=inputs.attention_mask.to(model.device))

    return all_acts


def get_mean_activations(model, instructions: List[str],
                         tokenize_fn, blocks: List[torch.nn.Module],
                         *, batch_size: int = 32, positions=[-1],
                         system_prompt: str = None) -> Float[Tensor, "pos layer d_model"]:
    """
    Returns only the **layer-input** mean activations (pos, layer, d_model).
    """
    torch.cuda.empty_cache()

    n_pos = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    pre_means = torch.zeros((n_pos, n_layers, d_model),
                             dtype=torch.float64, device=model.device)

    pre_hooks = [(blocks[i],
                  get_mean_activations_pre_hook(i, pre_means, n_samples, positions))
                 for i in range(n_layers)]
    with torch.inference_mode():
        for i in range(0, n_samples, batch_size):
            batch_prompts = instructions[i:i + batch_size]
            inputs = tokenize_fn(instructions=batch_prompts,
                                system=system_prompt)
            with add_hooks(module_forward_pre_hooks=pre_hooks,
                        module_forward_hooks=[]):            # empty list, not None
                model(input_ids=inputs.input_ids.to(model.device),
                    attention_mask=inputs.attention_mask.to(model.device))
    return pre_means


# ----------------------------- Grouping helper ----------------------------------
def _group_prompts_by_category(items: List[dict]) -> Dict[str, List[str]]:
    buckets = collections.defaultdict(list)
    for obj in items:
        buckets[obj["category"]].append(obj["instruction"])
    return buckets


# ------------------------ Main per-category routine ----------------------------
def generate_category_directions(
    model_base: ModelBase,
    harmful_instructions: List[dict],
    harmless_instructions: List[str],
    artifact_dir: str,
    *,
    system_prompt: str = None,
    batch_size: int = 32,
) -> Tuple[
    Dict[str, Tuple[int, Float[Tensor, "..."]]],  # shape depends on get_individual_harm
    Float[Tensor, "pos layer d_model"]
]:
    """
    Returns:
        - cat_dirs: category -> (n_prompts, direction_tensor)
            - if get_individual_harm: direction_tensor is [n, pos, layer, d_model]
            - else: direction_tensor is [pos, layer, d_model]
        - harmless_mean: [pos, layer, d_model]
    """
    os.makedirs(artifact_dir, exist_ok=True)

    if len(model_base.eoi_toks) > 5:
        model_base.eoi_toks = model_base.eoi_toks[-5:]
    pos_idx = list(range(-len(model_base.eoi_toks), 0))

    harmless_means = get_mean_activations(
        model_base.model,
        harmless_instructions,
        model_base.tokenize_instructions_fn,
        model_base.model_block_modules,
        batch_size=batch_size,
        positions=pos_idx,
        system_prompt=system_prompt,
    )

    cat_dirs: Dict[str, Tuple[int, Tensor]] = {}

    for cat, prompts in tqdm(_group_prompts_by_category(harmful_instructions).items(),
                             desc="Generating Category Directions"):
        if not prompts:
            continue

        harm_means = get_mean_activations(
            model_base.model,
            prompts,
            model_base.tokenize_instructions_fn,
            model_base.model_block_modules,
            batch_size=batch_size,
            positions=pos_idx,
            system_prompt=system_prompt,
        )  # [pos, layer, d_model]


        direction = harm_means - harmless_means
        cat_dirs[cat] = (len(prompts), direction.to(torch.float32))
        del harm_means
        del direction

        # Free memory
        torch.cuda.empty_cache()
        gc.collect()

    return cat_dirs, harmless_means.to(torch.float32)