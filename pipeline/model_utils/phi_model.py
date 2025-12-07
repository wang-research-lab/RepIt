import torch
import functools

from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from typing import List, Union
from jaxtyping import Float
from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase


PHI_REFUSAL_TOKS = [0]  # placeholder
EOI_TEXT = "<|end|>"
def format_instruction_phi_chat(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
) -> List[dict]:
    """
    Return a list of role/content messages for the Phi chat template.
    """
    if not include_trailing_whitespace:
        instruction = instruction.rstrip()

    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": instruction})
    if output is not None:
        messages.append({"role": "assistant", "content": output})
    return messages


def tokenize_instructions_phi_chat(
    tokenizer: AutoTokenizer,
    instructions: Union[List[str], List[dict], str, dict],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True,
    add_generation_prompt: bool = True,
):
    """
    Formats instructions into chat messages and tokenizes using Phi's chat template.
    """
    # Normalize to list
    if isinstance(instructions, (str, dict)):
        instructions = [instructions]

    messages_list = []
    for idx, item in enumerate(instructions):
        instr = item["instruction"] if isinstance(item, dict) else str(item)
        out = outputs[idx] if outputs is not None else None
        messages_list.append(
            format_instruction_phi_chat(
                instr, out, system, include_trailing_whitespace
            )
        )

    # Apply chat template to each conversation
    formatted_prompts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt and outputs is None
        )
        for msgs in messages_list
    ]

    return tokenizer(
        formatted_prompts,
        padding=True,
        truncation=False,
        return_tensors="pt"
    )

def orthogonalize_phi_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)
    for block in model.model.layers:
        block.self_attn.dense.weight.data = get_orthogonalized_matrix(block.self_attn.dense.weight.data.T, direction).T
        block.mlp.fc2.weight.data = get_orthogonalized_matrix(block.mlp.fc2.weight.data.T, direction).T


class PhiModel(ModelBase):
    def _load_model(self, model_path, dtype=torch.float16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto"
        ).eval()
        model.requires_grad_(False)
        return model

    def _get_generation_config(self) -> GenerationConfig:
        return GenerationConfig(do_sample=True, temperature=0.6, top_p=0.95)

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
        tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_phi_chat,
            tokenizer=self.tokenizer,
            system="",
            include_trailing_whitespace=True
        )


    def _get_eoi_toks(self):
        chat_str = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )

        if EOI_TEXT in chat_str:
            # Include the EOI marker itself
            eoi_str = chat_str.split(EOI_TEXT)[-1]
            eoi_str = EOI_TEXT + eoi_str
            return self.tokenizer.encode(eoi_str, add_special_tokens=False)
        else:
            return self.tokenizer.encode(EOI_TEXT, add_special_tokens=False)

    def _get_refusal_toks(self):
        return PHI_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return [layer.self_attn for layer in self.model.model.layers]

    def _get_mlp_modules(self):
        return [layer.mlp for layer in self.model.model.layers]

    def _get_post_attn_modules(self):
        return [layer.input_layernorm for layer in self.model.model.layers]
