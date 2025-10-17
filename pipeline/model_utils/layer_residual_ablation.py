from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from typing import Optional, Tuple
from transformers.cache_utils import Cache
import torch
from torch import Tensor

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import csv

import torch
from torch import Tensor


#WARNING: These methods are built around transformers==4.44.2 and are very likely incompatible with other versions

def ablateresidual_LlamaDecoderLayer(layer: LlamaDecoderLayer, layer_index, direction, **ablation_kwargs):
    """
    Modifies an existing LlamaDecoderLayer to include ablation logic.
    
    Args:
        layer (LlamaDecoderLayer): The original layer to modify.
        direction (Tensor): The ablation direction tensor.
    """
    # Save the ablation direction as an attribute of the layer
    layer.refusal_directions = direction
    layer.layer_index = layer_index

    #check if we're doing an actadd
    layer.coeff = ablation_kwargs.get("coeff", None)
    
    #check where to log norms
    layer.logfile = ablation_kwargs.get("log_name", None)

    layer.ablate_attn_residual = ablation_kwargs.get("ablate_attn_residual", True)
    layer.ablate_mlp_residual = ablation_kwargs.get("ablate_mlp_residual", True)
    layer.attn_offset = ablation_kwargs.get("attn_offset", -1)
    layer.mlp_offset = ablation_kwargs.get("mlp_offset", 0)



    # Define a new forward method with ablation logic
    def forward_with_residual_ablation(
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual = hidden_states

        hidden_states = layer.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        #tracks magnitudes after residual connections
        projection_magnitudes = []

        # Cast direction_normalized to residual dtype and ablate
        if layer.ablate_attn_residual:
            
            direction = layer.refusal_directions[int(max([layer.layer_index - layer.mlp_offset, 0]))]
            direction_normalized = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            direction_normalized = direction_normalized.to(residual) 
            residual_projection = (residual @ direction_normalized).unsqueeze(-1) * direction_normalized 
            orthogonal_residual = residual - residual_projection

            if layer.coeff is not None:
                #actadd setting
                hidden_states = hidden_states + orthogonal_residual + layer.coeff * direction_normalized
            else:
                hidden_states = orthogonal_residual + hidden_states

                # Calculate the projection of ablated hidden_states on direction
                projection_on_direction = (hidden_states @ direction_normalized.unsqueeze(-1))* direction_normalized

                original_projection_on_direction = ((hidden_states + residual_projection) @ direction_normalized.unsqueeze(-1)) * direction_normalized

                projection_magnitudes.extend([
                                              torch.linalg.vector_norm(original_projection_on_direction).item(), 
                                              torch.linalg.vector_norm(projection_on_direction).item()
                                              ])
        else:
            hidden_states = residual + hidden_states

            #not tracking attention residuals
            projection_magnitudes.extend(["",""])

        # Fully Connected
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)


        # Cast direction_normalized to residual dtype and ablate
        if layer.ablate_mlp_residual:
            direction = layer.refusal_directions[int(max([layer.layer_index - layer.mlp_offset, 0]))]
            direction_normalized = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
            direction_normalized = direction_normalized.to(residual) 
            residual_projection = (residual @ direction_normalized).unsqueeze(-1) * direction_normalized 
            orthogonal_residual = residual - residual_projection

            if layer.coeff is not None:
                #actadd setting
                hidden_states = hidden_states + orthogonal_residual + layer.coeff * direction_normalized
            else:
                hidden_states = orthogonal_residual + hidden_states

                # Calculate the projection of ablated hidden_states on direction
                projection_on_direction = (hidden_states @ direction_normalized.unsqueeze(-1))* direction_normalized

                original_projection_on_direction = ((hidden_states + residual_projection) @ direction_normalized.unsqueeze(-1)) * direction_normalized

                projection_magnitudes.extend([
                                torch.linalg.vector_norm(original_projection_on_direction).item(), 
                                torch.linalg.vector_norm(projection_on_direction).item()
                                ])
        else:
            hidden_states = residual + hidden_states

            #not tracking mlp residuals
            projection_magnitudes.extend(["",""])

            
        if layer.logfile is not None:
            with open(layer.logfile, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(projection_magnitudes) 


        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    # Replace the forward method of the layer
    layer.forward = forward_with_residual_ablation

    return layer
