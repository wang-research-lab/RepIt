
import torch
import contextlib
import functools

from typing import List, Tuple, Callable
from jaxtyping import Float
from torch import Tensor
import pdb

@contextlib.contextmanager
def add_hooks(
    module_forward_pre_hooks: List[Tuple[torch.nn.Module, Callable]],
    module_forward_hooks: List[Tuple[torch.nn.Module, Callable]],
    **kwargs
):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_forward_pre_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward pre hook on the module
    module_forward_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for module, hook in module_forward_pre_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_pre_hook(partial_hook))
        for module, hook in module_forward_hooks:
            partial_hook = functools.partial(hook, **kwargs)
            handles.append(module.register_forward_hook(partial_hook))
        yield
    finally:
        for h in handles:
            h.remove()

def caa_add_hook(direction: Tensor, multiplier = 1):
    def hook_fn(module, input):
        nonlocal direction, multiplier

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone()
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input.clone()
        
        direction = direction.to(activation)

        activation = activation + multiplier * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_affine_direction_ablation_input_pre_hook(direction: Tensor, 
                                          reference: Tensor | None = None, 
                                          coeff: Tensor| None = None):
    def hook_fn(module, input):
        nonlocal direction, reference, coeff

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone()
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input.clone()

        if reference is None:
            reference = torch.zeros_like(activation[0])
        if coeff is None:
            coeff = torch.Tensor([0])
        
        direction = direction.to(activation)
        reference = reference.to(activation)
        coeff = coeff.to(activation)
        
        # Normalize the direction vector
        normalized_direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute projections
        proj_v = (activation @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(v)
        proj_ref = (reference @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(r^-)

        # Apply affine transformation
        activation = activation - proj_v + proj_ref + coeff * direction

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_affine_direction_ablation_output_hook(direction: Tensor, 
                                        reference: Tensor | None = None):
    def hook_fn(module, input, output):
        nonlocal direction, reference

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0].clone()
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output.clone()

        if reference is None:
            reference = torch.zeros_like(activation[0])

        direction = direction.to(activation)
        reference = reference.to(activation)
        
        # Normalize the direction vector
        normalized_direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute projections
        proj_v = (activation @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(v)
        proj_ref = (reference @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(r^-)

        # Apply affine transformation
        activation = activation - proj_v + proj_ref + direction

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn


def get_affine_activation_addition_input_pre_hook(
    direction: Float[Tensor, "d_model"], 
    coeff: Float[Tensor, ""], 
    reference: Float[Tensor, "d_model"] | None = None
):
    def hook_fn(module, input):
        nonlocal direction, reference, coeff

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0].clone()
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input.clone()

        # Default reference to zero tensor if not provided
        if reference is None:
            reference = torch.zeros_like(activation[0])

        direction = direction.to(activation)
        reference = reference.to(activation)
        coeff = coeff.to(activation)
        
        # Normalize the direction vector
        normalized_direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)


        # Compute projections
        proj_v = (activation @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(v)
        proj_ref = (reference @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(r^-)

        # Apply affine transformation
        modified_activation = activation - proj_v + proj_ref + coeff * direction

        # Return modified activation
        if isinstance(input, tuple):
            return (modified_activation, *input[1:])
        else:
            return modified_activation

    return hook_fn


def get_affine_activation_addition_input_post_hook(
    direction: Float[Tensor, "d_model"], 
    coeff: Float[Tensor, ""], 
    reference: Float[Tensor, "d_model"] | None = None
):
    def hook_fn(module, input, output):
        nonlocal direction, reference, coeff

        if reference is None:
            reference = torch.zeros_like(activation[0])
            

        if isinstance(output, tuple):
            activation = output[0]
            other_outputs = output[1:]
            
            direction = direction.to(activation)
            reference = reference.to(activation)
            coeff = coeff.to(activation)
            
            # Normalize the direction vector
            normalized_direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

            # Compute projections
            proj_v = (activation @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(v)
            proj_ref = (reference @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(r^-)

            # Apply affine transformation
            modified_activation = activation - proj_v + proj_ref + coeff * direction
            
            return (modified_activation,) + other_outputs
        else:
            activation = output

            direction = direction.to(activation)
            reference = reference.to(activation)
            coeff = coeff.to(activation)
            
            # Normalize the direction vector
            normalized_direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

            # Compute projections
            proj_v = (activation @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(v)
            proj_ref = (reference @ normalized_direction).unsqueeze(-1) * normalized_direction  # proj_r(r^-)

            # Apply affine transformation
            modified_activation = activation - proj_v + proj_ref + coeff * direction
            return modified_activation
    return hook_fn

    
def get_linear_direction_ablation_input_pre_hook(direction: Tensor):
    def hook_fn(module, input):
        nonlocal direction

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation) 
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn

def get_linear_direction_ablation_output_hook(direction: Tensor):
    def hook_fn(module, input, output):
        nonlocal direction

        if isinstance(output, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = output[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = output

        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
        direction = direction.to(activation)
        activation -= (activation @ direction).unsqueeze(-1) * direction 

        if isinstance(output, tuple):
            return (activation, *output[1:])
        else:
            return activation

    return hook_fn

def get_all_linear_direction_ablation_hooks(
    model_base,
    direction: Float[Tensor, 'd_model'],
):
    fwd_pre_hooks = [(model_base.model_block_modules[layer], get_linear_direction_ablation_input_pre_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks = [(model_base.model_attn_modules[layer], get_linear_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]
    fwd_hooks += [(model_base.model_mlp_modules[layer], get_linear_direction_ablation_output_hook(direction=direction)) for layer in range(model_base.model.config.num_hidden_layers)]

    return fwd_pre_hooks, fwd_hooks


def get_linear_activation_addition_input_pre_hook(vector: Float[Tensor, "d_model"], coeff: Float[Tensor, ""]):
    def hook_fn(module, input):
        nonlocal vector, coeff

        if isinstance(input, tuple):
            activation: Float[Tensor, "batch_size seq_len d_model"] = input[0]
        else:
            activation: Float[Tensor, "batch_size seq_len d_model"] = input

        vector = vector.to(activation)
        coeff = coeff.to(activation)
        activation += coeff * vector

        if isinstance(input, tuple):
            return (activation, *input[1:])
        else:
            return activation
    return hook_fn