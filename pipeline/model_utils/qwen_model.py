# https://github.com/andyrdt/refusal_direction/blob/main/pipeline/model_utils/qwen_model.py
import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Union
from torch import Tensor
from jaxtyping import Int, Float
import pdb


from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase

SAMPLE_SYSTEM_PROMPT = """You are a helpful assistant."""

QWEN_CHAT_TEMPLATE_WITH_SYSTEM = """<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_CHAT_TEMPLATE = """<|im_start|>user
{instruction}<|im_end|>
<|im_start|>assistant
"""

QWEN_REFUSAL_TOKS = [40, 2121] # ['I', 'As']

def format_instruction_qwen_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True,
):
  
    if system is not None:
        formatted_instruction = QWEN_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = QWEN_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()
    
    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_qwen_chat(
    tokenizer: AutoTokenizer,
    instructions: Union[List[str], List[dict], str, dict],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
):
    # Ensure `instructions` is a list
    if isinstance(instructions, (str, dict)):
        instructions = [instructions]

    # Extract the instruction string from each element
    cleaned_instructions = []
    for item in instructions:
        if isinstance(item, dict):
            instr = str(item.get("instruction", ""))
        else:
            instr = str(item)
        cleaned_instructions.append(instr)

    # Construct prompt list
    if outputs is not None:
        prompts = [
            format_instruction_qwen_chat(
                instruction=instruction,
                output=output,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction, output in zip(cleaned_instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_qwen_chat(
                instruction=instruction,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace,
            )
            for instruction in cleaned_instructions
        ]

    # Tokenize the prompts
    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

def orthogonalize_qwen_weights(model, direction: Float[Tensor, "d_model"]):
    model.transformer.wte.weight.data = get_orthogonalized_matrix(model.transformer.wte.weight.data, direction)

    for block in model.transformer.h:
        block.attn.c_proj.weight.data = get_orthogonalized_matrix(block.attn.c_proj.weight.data.T, direction).T
        block.mlp.c_proj.weight.data = get_orthogonalized_matrix(block.mlp.c_proj.weight.data.T, direction).T
        

class QwenModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.float16):
        model_kwargs = {}
        model_kwargs.update({"use_flash_attn": True})
        if dtype != "auto":
            model_kwargs.update({
                "bf16": dtype==torch.bfloat16,
                "fp16": dtype==torch.float16,
                "fp32": dtype==torch.float32,
            })

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.requires_grad_(False) 

        return model

    def _get_generation_config(self) -> GenerationConfig:
        return GenerationConfig(do_sample=True, temperature = 0.6, top_p = 0.95)

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )

        tokenizer.padding_side = 'left'
        tokenizer.pad_token = '<|endoftext|>'
        tokenizer.pad_token_id = 151643

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_qwen_chat,
            tokenizer=self.tokenizer,
            include_trailing_whitespace=True
        )


    def _get_eoi_toks(self):
        return self.tokenizer.encode(QWEN_CHAT_TEMPLATE.split("{instruction}")[-1])

    def _get_refusal_toks(self):
        return QWEN_REFUSAL_TOKS

    def _get_model_block_modules(self):
        if "2" in self.model_name_or_path or "3" in self.model_name_or_path:
            return self.model.model.layers
        elif "Chat" in self.model_name_or_path:
            for i, layer in enumerate(self.model.transformer.h):
                self.model.transformer.h[i].self_attn = layer.attn
            return self.model.transformer.h
        else:
            raise AssertionError("model unknown - not 2, 2.5, 3 or Qwen 1")

    def _get_attn_modules(self):
        if "2" in self.model_name_or_path or "3" in self.model_name_or_path:
            return self.model.model.layers
        elif "Chat" in self.model_name_or_path:
            for i, layer in enumerate(self.model.transformer.h):
                self.model.transformer.h[i].self_attn = layer.attn
            return self.model.transformer.h
        else:
            raise AssertionError("model unknown - not 2, 2.5, 3 or Qwen 1")
            
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_post_attn_modules(self):
        if "Chat" in self.model_name_or_path:
            return None
        return torch.nn.ModuleList([block_module.post_attention_layernorm for block_module in self.model_block_modules])
