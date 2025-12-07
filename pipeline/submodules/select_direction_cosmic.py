import json
import torch
import functools
import math
import matplotlib.pyplot as plt
import os
import pdb

from typing import List, Optional
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
from einops import rearrange
import numpy as np
import torch.nn.functional as F


from pipeline.model_utils.model_base import ModelBase
from pipeline.utils.hook_utils import (
                                    add_hooks, 
                                    get_affine_activation_addition_input_pre_hook,  
                                    get_affine_direction_ablation_input_pre_hook, 
                                    get_all_linear_direction_ablation_hooks,
                                    get_linear_activation_addition_input_pre_hook,  
                                    get_linear_direction_ablation_input_pre_hook
)           

def refusal_score(
    logits: Float[Tensor, 'batch seq d_vocab_out'],
    refusal_toks: Int[Tensor, 'batch seq'],
    epsilon: Float = 1e-8,
):
    logits = logits.to(torch.float64)

    # we only care about the last tok position
    logits = logits[:, -1, :]

    probs = torch.nn.functional.softmax(logits, dim=-1)
    refusal_probs = probs[:, refusal_toks].sum(dim=-1)

    nonrefusal_probs = torch.ones_like(refusal_probs) - refusal_probs
    return torch.log(refusal_probs + epsilon) - torch.log(nonrefusal_probs + epsilon)

def get_refusal_scores(model, instructions, tokenize_instructions_fn, refusal_toks, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32):
    refusal_score_fn = functools.partial(refusal_score, refusal_toks=refusal_toks)

    refusal_scores = torch.zeros(len(instructions), device=model.device)

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
            ).logits

        refusal_scores[i:i+batch_size] = refusal_score_fn(logits=logits)

    return refusal_scores

def get_baseline_vector(model, baseline_instructions, tokenize_instructions_fn, layers_to_evaluate, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32):
    """
    Computes the baseline vector by averaging the last hidden states for baseline instructions across all specified layers.
    """
    num_instructions = len(baseline_instructions)
    raw_outputs = torch.zeros((num_instructions, len(layers_to_evaluate), model.config.hidden_size), device=model.device)

    for i in range(0, num_instructions, batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=baseline_instructions[i:i+batch_size])
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            outputs = model(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
                output_hidden_states=True
            )
        
        hidden_states = outputs.hidden_states[1:]  # Exclude embeddings
        for idx, layer_idx in enumerate(layers_to_evaluate):
            raw_outputs[i:i+batch_size, idx, :] = hidden_states[layer_idx][:, -1, :]
    
    # Compute a single global baseline vector across all layers
    mean_output_directions = raw_outputs.mean(dim=(0, 1))  # Shape: (hidden_size,)
    norm = torch.norm(mean_output_directions, dim=-1, keepdim=True) + 1e-8
    baseline_vector = mean_output_directions / norm

    return baseline_vector  # Shape: (hidden_size,)

def get_refusal_similarity(model, instructions, baseline_vector, tokenize_instructions_fn, layers_to_evaluate, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32):
    """
    Computes a single global cosine similarity between instruction responses and the baseline vector across all evaluated layers.
    """
    num_instructions = len(instructions)
    raw_outputs = torch.zeros((num_instructions, len(layers_to_evaluate), model.config.hidden_size), device=model.device)

    for i in range(0, num_instructions, batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            outputs = model(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
                output_hidden_states=True
            )
        
        hidden_states = outputs.hidden_states[1:]
        for idx, layer_idx in enumerate(layers_to_evaluate):
            raw_outputs[i:i+batch_size, idx, :] = hidden_states[layer_idx][:, -1, :]
    
    # Compute a single global direction across all layers
    mean_output_directions = raw_outputs.mean(dim=(0, 1))  # Shape: (hidden_size,)
    normalized_proposed_direction = mean_output_directions / (torch.norm(mean_output_directions, dim=-1, keepdim=True) + 1e-8)
    
    # Compute global cosine similarity
    cosine_similarity = torch.dot(normalized_proposed_direction, baseline_vector)
    
    return cosine_similarity



def get_last_position_logits(model, tokenizer, instructions, tokenize_instructions_fn, fwd_pre_hooks=[], fwd_hooks=[], batch_size=32) -> Float[Tensor, "n_instructions d_vocab"]:
    last_position_logits = None

    for i in range(0, len(instructions), batch_size):
        tokenized_instructions = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
            logits = model(
                input_ids=tokenized_instructions.input_ids.to(model.device),
                attention_mask=tokenized_instructions.attention_mask.to(model.device),
                use_cache=False
            ).logits

        if last_position_logits is None:
            last_position_logits = logits[:, -1, :]
        else:
            last_position_logits = torch.cat((last_position_logits, logits[:, -1, :]), dim=0)

    return last_position_logits

        

def plot_refusal_scores(
    refusal_scores: Float[Tensor, 'n_pos n_layer'],
    baseline_refusal_score: Optional[float],
    token_labels: List[str],
    title: str,
    artifact_dir: str,
    artifact_name: str,
):
    n_pos, n_layer = refusal_scores.shape

    # Create a figure and an axis
    fig, ax = plt.subplots(figsize=(9, 5))  # width and height in inches

    # Add a trace for each position to extract
    for i in range(-n_pos, 0):
        ax.plot(
            list(range(n_layer)),
            refusal_scores[i].cpu().numpy(),
            label=f'{i}: {repr(token_labels[i])}'
        )

    if baseline_refusal_score is not None:
        # Add a horizontal line for the baseline
        ax.axhline(y=baseline_refusal_score, color='black', linestyle='--')
        ax.annotate('Baseline', xy=(1, baseline_refusal_score), xytext=(8, 10), 
                    xycoords=('axes fraction', 'data'), textcoords='offset points',
                    horizontalalignment='right', verticalalignment='center')

    ax.set_title(title)
    ax.set_xlabel('Layer source of direction (resid_pre)')
    ax.set_ylabel('Cosine Similarity')
    ax.legend(title='Position source of direction', loc='lower left')

    plt.savefig(f"{artifact_dir}/{artifact_name}.png")

# returns True if the direction should be filtered out
def filter_fn(refusal_score, steering_score, kl_div_score, pos, layer, n_layer, bypass_refusal_threshold = None, kl_threshold=None, induce_refusal_threshold=None, prune_layer_percentage=0.20) -> bool:
    if math.isnan(refusal_score) or math.isnan(steering_score) or math.isnan(kl_div_score):
        return True
    if layer >= int(n_layer * (1.0 - prune_layer_percentage)):
        return True
    if kl_threshold is not None and kl_div_score > kl_threshold:
        return True
    if bypass_refusal_threshold is not None and refusal_score > bypass_refusal_threshold:
        return True
    if induce_refusal_threshold is not None and steering_score < induce_refusal_threshold:
        return True
    return False

def select_direction(
    model_base: ModelBase,
    harmful_instructions,
    harmless_instructions,
    candidate_directions: Float[Tensor, 'n_dir n_pos n_layer d_model'],
    harmless_mean: Float[Tensor, 'n_dir n_pos n_layer d_model'],
    layers_to_evaluate,
    artifact_dir,
    affine_steering = True,
    kl_threshold=0.1, # directions larger KL score are filtered out
    induce_refusal_threshold=None, # directions with a lower inducing refusal score are filtered out
    prune_layer_percentage=0.2, # discard the directions extracted from the last 20% of the model
    batch_size=32
):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)

    n_pos, n_layer, d_model = candidate_directions.shape

    baseline_activations_harmful = get_baseline_vector(model_base.model, harmful_instructions, model_base.tokenize_instructions_fn, layers_to_evaluate, fwd_hooks=[], batch_size=batch_size)
    baseline_activations_harmless = get_baseline_vector(model_base.model, harmless_instructions, model_base.tokenize_instructions_fn, layers_to_evaluate, fwd_hooks=[], batch_size=batch_size)

    baseline_refusal = 1 #ideally cosine sim 1 to harmless hidden states
    baseline_harmless = 1 #ideally cosine sim 1 to refusal hidden states

    ablation_kl_div_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)
    ablation_refusal_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)
    steering_refusal_scores = torch.zeros((n_pos, n_layer), device=model_base.model.device, dtype=torch.float64)

    baseline_harmless_logits = get_last_position_logits(
        model=model_base.model,
        tokenizer=model_base.tokenizer,
        instructions=harmless_instructions,
        tokenize_instructions_fn=model_base.tokenize_instructions_fn,
        fwd_pre_hooks=[],
        fwd_hooks=[],
        batch_size=batch_size
    )

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing KL for source position {source_pos}"):

            pre_layer_direction = candidate_directions[source_pos, source_layer]
            pre_layer_reference = harmless_mean[source_pos, source_layer]

            if affine_steering:
                fwd_pre_hooks = [(model_base.model_block_modules[source_layer], get_affine_direction_ablation_input_pre_hook(direction=pre_layer_direction, reference = pre_layer_reference))]
                fwd_hooks = []
            else: 
                fwd_pre_hooks, fwd_hooks = get_all_linear_direction_ablation_hooks(model_base, pre_layer_direction)

            
            intervention_logits: Float[Tensor, "n_instructions 1 d_vocab"] = get_last_position_logits(
                model=model_base.model,
                tokenizer=model_base.tokenizer,
                instructions=harmless_instructions,
                tokenize_instructions_fn=model_base.tokenize_instructions_fn,
                fwd_pre_hooks=fwd_pre_hooks,
                fwd_hooks=fwd_hooks,
                batch_size=batch_size
            )

            ablation_kl_div_scores[source_pos, source_layer] = kl_div_fn(baseline_harmless_logits, intervention_logits, mask=None).mean(dim=0).item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal ablation for source position {source_pos}"):

            pre_layer_direction = candidate_directions[source_pos, source_layer]
            pre_layer_reference = harmless_mean[source_pos, source_layer]

            if affine_steering:
                fwd_pre_hooks = [(model_base.model_block_modules[source_layer], get_affine_direction_ablation_input_pre_hook(direction=pre_layer_direction, reference = pre_layer_reference))]
                fwd_hooks = []
            else: 
                fwd_pre_hooks, fwd_hooks = get_all_linear_direction_ablation_hooks(model_base, pre_layer_direction)

            refusal_scores = get_refusal_similarity(model_base.model, harmful_instructions, baseline_activations_harmless, model_base.tokenize_instructions_fn, layers_to_evaluate, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            ablation_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()

    for source_pos in range(-n_pos, 0):
        for source_layer in tqdm(range(n_layer), desc=f"Computing refusal addition for source position {source_pos}"):

            pre_layer_direction = candidate_directions[source_pos, source_layer]
            pre_layer_reference = harmless_mean[source_pos, source_layer]

            if affine_steering:
                coeff = torch.Tensor([1.0])
                fwd_pre_hooks = [(model_base.model_block_modules[source_layer], get_affine_activation_addition_input_pre_hook(direction=pre_layer_direction, coeff = coeff, reference = pre_layer_reference))]
                fwd_hooks = []
            else: 
                fwd_pre_hooks, fwd_hooks = [(model_base.model_block_modules[source_layer], 
                                                        get_linear_activation_addition_input_pre_hook(vector=pre_layer_direction, 
                                                        coeff=torch.Tensor([1.0])))], []


            refusal_scores = get_refusal_similarity(model_base.model, harmless_instructions, baseline_activations_harmful, model_base.tokenize_instructions_fn, layers_to_evaluate, fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=batch_size)
            steering_refusal_scores[source_pos, source_layer] = refusal_scores.mean().item()



    plot_refusal_scores(
        refusal_scores=ablation_refusal_scores,
        baseline_refusal_score=baseline_refusal,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Ablating direction on harmful instructions',
        artifact_dir=artifact_dir,
        artifact_name='ablation_scores'
    )

    plot_refusal_scores(
        refusal_scores=steering_refusal_scores,
        baseline_refusal_score=baseline_harmless,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='Adding direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='actadd_scores'
    )


    plot_refusal_scores(
        refusal_scores=ablation_kl_div_scores,
        baseline_refusal_score=0.0,
        token_labels=model_base.tokenizer.batch_decode(model_base.eoi_toks),
        title='KL Divergence when ablating direction on harmless instructions',
        artifact_dir=artifact_dir,
        artifact_name='kl_div_scores'
    )


    

    filtered_scores = []
    json_output_all_scores = []
    json_output_filtered_scores = []


    refusal_peaks = torch.argmax(torch.nan_to_num(ablation_refusal_scores[-n_pos:-1, :], nan=-float('inf')), dim=1)
    limit_refusal_layer = torch.nanmedian(refusal_peaks)

    steering_peaks = torch.argmax(torch.nan_to_num(steering_refusal_scores[-n_pos:-1, :], nan=-float('inf')), dim=1)
    cutoff_steering_layer = torch.nanmedian(steering_peaks)
    
    for source_pos in range(-n_pos, 0):
        for source_layer in range(n_layer):

            json_output_all_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'refusal_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'steering_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })

            refusal_score = ablation_refusal_scores[source_pos, source_layer].item()
            steering_score = steering_refusal_scores[source_pos, source_layer].item()
            kl_div_score = ablation_kl_div_scores[source_pos, source_layer].item()


            sorting_score = steering_score + refusal_score

            # we filter out directions if the KL threshold 
            discard_direction = filter_fn(
                refusal_score=refusal_score,
                steering_score=steering_score,
                kl_div_score=kl_div_score,
                pos = source_pos,
                layer=source_layer,
                n_layer=n_layer,
                kl_threshold=kl_threshold,
                induce_refusal_threshold=induce_refusal_threshold,
                prune_layer_percentage=prune_layer_percentage
            )

            if source_pos == -1:
                if limit_refusal_layer < source_layer and cutoff_steering_layer < source_layer:
                    continue
                
            if discard_direction:
                continue

            filtered_scores.append((sorting_score, source_pos, source_layer))

            json_output_filtered_scores.append({
                'position': source_pos,
                'layer': source_layer,
                'refusal_score': ablation_refusal_scores[source_pos, source_layer].item(),
                'steering_score': steering_refusal_scores[source_pos, source_layer].item(),
                'kl_div_score': ablation_kl_div_scores[source_pos, source_layer].item()
            })   

    with open(f"{artifact_dir}/direction_evaluations.json", 'w') as f:
        json.dump(json_output_all_scores, f, indent=4)

    json_output_filtered_scores = sorted(json_output_filtered_scores, 
                                         key=lambda x: x["steering_score"] + x["refusal_score"], reverse=True)

    with open(f"{artifact_dir}/direction_evaluations_filtered.json", 'w') as f:
        json.dump(json_output_filtered_scores, f, indent=4)


    assert len(filtered_scores) > 0, "All scores have been filtered out!"


    filtered_scores = sorted(filtered_scores, key=lambda x: x[0], reverse = True)

    # now return the best position, layer, and direction
    score, pos, layer = filtered_scores[0]

    print(f"Selected direction: position={pos}, layer={layer}")
    print(f"Refusal score: {ablation_refusal_scores[pos, layer]:.4f} (baseline: {baseline_refusal:.4f})")
    print(f"Steering score: {steering_refusal_scores[pos, layer]:.4f} (baseline: {baseline_harmless:.4f})")
    print(f"KL Divergence: {ablation_kl_div_scores[pos, layer]:.4f}")

    selected_directions = candidate_directions[pos, layer]
    harmless_reference = harmless_mean[pos, layer]


    return pos, layer, selected_directions, harmless_reference


def masked_mean(seq, mask = None, dim = 1, keepdim = False):
    if mask is None:
        return seq.mean(dim = dim)

    if seq.ndim == 3:
        mask = rearrange(mask, 'b n -> b n 1')

    masked_seq = seq.masked_fill(~mask, 0.)
    numer = masked_seq.sum(dim = dim, keepdim = keepdim)
    denom = mask.sum(dim = dim, keepdim = keepdim)

    masked_mean = numer / denom.clamp(min = 1e-3)
    masked_mean = masked_mean.masked_fill(denom == 0, 0.)
    return masked_mean

def kl_div_fn(
    logits_a: Float[Tensor, 'batch seq_pos d_vocab'],
    logits_b: Float[Tensor, 'batch seq_pos d_vocab'],
    mask: Int[Tensor, "batch seq_pos"]=None,
    epsilon: Float=1e-6
) -> Float[Tensor, 'batch']:
    """
    Compute the KL divergence loss between two tensors of logits.
    """
    logits_a = logits_a.to(torch.float64)
    logits_b = logits_b.to(torch.float64)

    probs_a = logits_a.softmax(dim=-1)
    probs_b = logits_b.softmax(dim=-1)

    kl_divs = torch.sum(probs_a * (torch.log(probs_a + epsilon) - torch.log(probs_b + epsilon)), dim=-1)

    if mask is None:
        return torch.mean(kl_divs, dim=-1)
    else:
        return masked_mean(kl_divs, mask).mean(dim=-1)