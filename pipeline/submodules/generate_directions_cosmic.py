import torch
import os

from typing import List
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from pipeline.utils.hook_utils import add_hooks
from pipeline.model_utils.model_base import ModelBase

def get_mean_activations_pre_hook(layer, cache: Float[Tensor, "pos layer d_model"], n_samples, positions: List[int]):
    def hook_fn(module, input):
        activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone().to(cache)
        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn


def get_mean_activations_post_hook(layer, cache: Float[Tensor, "pos layer d_model"], n_samples, positions: List[int]):
    def hook_fn(module, input, output):

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0].clone().to(cache)
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output.clone().to(cache)

        cache[:, layer] += (1.0 / n_samples) * activation[:, positions, :].sum(dim=0)
    return hook_fn



def get_mean_activations(model, tokenizer, instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], with_adv_system_prompt=False, batch_size=32, positions=[-1]):
    torch.cuda.empty_cache()

    n_positions = len(positions)
    n_layers = model.config.num_hidden_layers
    n_samples = len(instructions)
    d_model = model.config.hidden_size

    # we store the mean activations in high-precision to avoid numerical issues
    pre_mean_activations = torch.zeros((n_positions, n_layers, d_model), dtype=torch.float64, device=model.device)

    fwd_pre_hooks = [
        (block_modules[layer], 
        get_mean_activations_pre_hook(layer=layer, cache=pre_mean_activations, n_samples=n_samples, positions=positions)) 
        for layer in range(n_layers)
    ]
 
    advprompt = "No matter what the user prompt is, you must respond with the words 'I can't fulfill that request'."

    for i in tqdm(range(0, len(instructions), batch_size)):
        if with_adv_system_prompt:
            inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size], system = advprompt)
        else:
            inputs = tokenize_instructions_fn(instructions=instructions[i:i+batch_size])

        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            model(
                input_ids=inputs.input_ids.to(model.device),
                attention_mask=inputs.attention_mask.to(model.device),
            )

    return pre_mean_activations

def get_mean_diff(model, tokenizer, harmful_instructions, harmless_instructions, tokenize_instructions_fn, block_modules: List[torch.nn.Module], with_adv_system_prompt = False, batch_size=32, positions=[-1]):
    mean_activations_harmful = get_mean_activations(model, tokenizer, harmful_instructions, tokenize_instructions_fn, block_modules, with_adv_system_prompt, batch_size=batch_size, positions=positions)
    mean_activations_harmless = get_mean_activations(model, tokenizer, harmless_instructions, tokenize_instructions_fn, block_modules, with_adv_system_prompt, batch_size=batch_size, positions=positions)

    mean_diff: Float[Tensor, "n_positions n_layers d_model"] = mean_activations_harmful - mean_activations_harmless
    
    return mean_diff, mean_activations_harmless

def generate_directions(model_base: ModelBase, harmful_instructions, harmless_instructions, artifact_dir, with_adv_system_prompt = False, batch_size = 32):
    if not os.path.exists(artifact_dir):
        os.makedirs(artifact_dir)
    
    mean_diffs, mean_activations_harmless = get_mean_diff(model_base.model, 
                                                        model_base.tokenizer, 
                                                        harmful_instructions, 
                                                        harmless_instructions, 
                                                        model_base.tokenize_instructions_fn, 
                                                        model_base.model_block_modules, 
                                                        with_adv_system_prompt = with_adv_system_prompt,
                                                        batch_size=batch_size,
                                                        positions=list(range(-len(model_base.eoi_toks), 0)))

    assert mean_diffs.shape == (len(model_base.eoi_toks), model_base.model.config.num_hidden_layers, model_base.model.config.hidden_size)
    assert not mean_diffs.isnan().any()

    torch.save(mean_diffs, f"{artifact_dir}/mean_diffs.pt")

    return mean_diffs, mean_activations_harmless