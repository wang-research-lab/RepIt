
import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, Gemma3ForConditionalGeneration, GenerationConfig, AutoProcessor
from typing import List, Union
from jaxtyping import Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

# Gemma chat template is based on
# - Official Gemma documentation: https://ai.google.dev/gemma/docs/formatting

torch.set_float32_matmul_precision('high')

GEMMA_CHAT_TEMPLATE = """<start_of_turn>user
{instruction}<end_of_turn>
<start_of_turn>model
"""

GEMMA_REFUSAL_TOKS = [235285] # ['I']


import torch._dynamo
torch._dynamo.config.cache_size_limit = float("inf")
#Gemma is really picky abt this
#but tokenizing to consistent lengths in this setup is tough and takes a lot of VRAM

def format_instruction_gemma_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
    if system is not None:
        raise ValueError("System prompts are not supported for Gemma models.")
    else:
        formatted_instruction = GEMMA_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_gemma_chat(
    tokenizer: AutoTokenizer,
    instructions: Union[List[str], List[dict], str, dict],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    if isinstance(instructions, (str, dict)):
        instructions = [instructions]

    # Extract instruction strings
    cleaned_instructions = []
    for item in instructions:
        if isinstance(item, dict):
            instr = str(item.get("instruction", ""))
        else:
            instr = str(item)
        cleaned_instructions.append(instr)

    if system is not None:
        instructions = [system + instruction for instruction in cleaned_instructions]
    else:
        instructions = cleaned_instructions

    if outputs is not None:
        prompts = [
            format_instruction_gemma_chat(
                instruction=instruction,
                output=output,
                system=None,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_gemma_chat(
                instruction=instruction,
                system=None,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result



def orthogonalize_gemma_weights(model: AutoTokenizer, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_gemma_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer-1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer-1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.model.layers[layer-1].mlp.down_proj.bias = torch.nn.Parameter(bias)


class GemmaModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        

        if "3" in model_path:
            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation="eager"
            ).eval()

            model.config.num_hidden_layers = model.config.text_config.num_hidden_layers 
            model.config.hidden_size = model.config.text_config.hidden_size
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map="auto",
                attn_implementation="eager"
            ).eval()

        # disable Dynamo everywhere
        model = torch.compile(model, backend="eager", mode="max-autotune-no-cudagraphs")

        model.forward = torch._dynamo.disable()(model.forward)
        model.generate = torch._dynamo.disable()(model.generate)
        model.requires_grad_(False)
        return model


    def _get_generation_config(self) -> GenerationConfig:
        return GenerationConfig(do_sample=True, temperature = 0.95, attn_implementation = "eager", top_k = 64, top_p = 0.95, cache_implementation = "dynamic")

    def _load_tokenizer(self, model_path):
        if "3" in model_path:
            tokenizer = AutoProcessor.from_pretrained(model_path).tokenizer
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = 'left'

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_gemma_chat, tokenizer=self.tokenizer, system=None, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode(GEMMA_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return GEMMA_REFUSAL_TOKS

    def _get_model_block_modules(self):
        if "3" in self.model_name_or_path:
            return self.model.language_model.layers

        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_gemma_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_gemma_weights, direction=direction, coeff=coeff, layer=layer)
    
    def _get_post_attn_modules(self):
        return torch.nn.ModuleList([block_module.post_attention_layernorm for block_module in self.model_block_modules])
