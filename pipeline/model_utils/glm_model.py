import torch
import functools

from transformers import AutoTokenizer, Glm4vForConditionalGeneration, GenerationConfig, AutoProcessor
from typing import List, Union

from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase


GLM4_CHAT_TEMPLATE_WITH_SYSTEM = (
    "[gMASK]<sop><|system|>\n{system_prompt}<|user|>\n{user_message}<|assistant|><think>"
)
GLM4_CHAT_TEMPLATE = "[gMASK]<sop><|user|>\n{user_message}<|assistant|><think>"
EOI_TEXT = '<|assistant|>'

def format_instruction_glm4_chat(
    tokenizer: AutoTokenizer,
    messages: List[dict],
    add_generation_prompt: bool = False,
    include_trailing_whitespace: bool = True,
) -> str:
    """Format a conversation for GLM‑4 using the tokenizer's built‑in template."""

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    return prompt if include_trailing_whitespace else prompt.rstrip()


def tokenize_instructions_glm4_chat(
    tokenizer: AutoTokenizer,
    instructions: Union[List[str], List[dict], str, dict],
    outputs: List[str] | None = None,
    system: str | None = None,
    include_trailing_whitespace: bool = True,
):
    """Batch‑tokenize instruction strings or dicts into GLM‑4 chat prompts."""

    # Normalize to list
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

    # Build message threads
    messages_list = []
    for i, instruction in enumerate(cleaned_instructions):
        thread: list[dict] = []
        if system is not None:
            thread.append({"role": "system", "content": system})
        thread.append({"role": "user", "content": instruction})
        if outputs is not None:
            thread.append({"role": "assistant", "content": outputs[i]})
        messages_list.append(thread)

    # Format and tokenize prompts
    prompts = [
        format_instruction_glm4_chat(
            tokenizer=tokenizer,
            messages=thread,
            add_generation_prompt=True,
            include_trailing_whitespace=include_trailing_whitespace,
        )
        for thread in messages_list
    ]

    return tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )


class GLM4Model(ModelBase):
    """Pipeline‑specific thin wrapper around *THUDM/GLM‑4.1V‑9B‑Thinking*."""

    # ------------------------------------------------------------------ loaders
    def _load_model(self, model_path: str, dtype: torch.dtype = torch.bfloat16):
        model = Glm4vForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        model.requires_grad_(False)
        return model

    def _load_tokenizer(self, model_path: str):
        processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        tok = processor.tokenizer
        tok.padding_side = "left"
        tok.pad_token = tok.eos_token
        return tok

    # ---------------------------------------------------------- helpers/aliases
    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_glm4_chat, tokenizer=self.tokenizer)

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
        return []

    def _get_generation_config(self) -> GenerationConfig:
        return GenerationConfig(do_sample=True, temperature = 0.95)
    
    @property
    def _layers(self):
        return self.model.language_model.layers

    def _get_model_block_modules(self):
        return self._layers

    def _get_attn_modules(self):
        out = []
        for blk in self._layers:
            out.append(blk.self_attn)
        return torch.nn.ModuleList(out)

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([blk.mlp for blk in self._layers])

    def _get_post_attn_modules(self):
        # GLM‑4 normally applies RMSNorm *after* attention; the attribute name is
        # ``post_attention_layernorm`` in some ports, ``input_layernorm`` in
        # others.  Try both.
        norms = []
        for blk in self._layers:
            if hasattr(blk, "post_attention_layernorm"):
                norms.append(blk.post_attention_layernorm)
            elif hasattr(blk, "input_layernorm"):
                norms.append(blk.input_layernorm)
        return torch.nn.ModuleList(norms)

    # ----------------------------------------------------- weight mod utilities
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_glm4_weights, direction=direction)

    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_glm4_weights, direction=direction, coeff=coeff, layer=layer)


def _maybe_get(module, *names):
    for n in names:
        if hasattr(module, n):
            return getattr(module, n)
    raise AttributeError(f"None of {names} found in module {module.__class__.__name__}")


def orthogonalize_glm4_weights(model, direction: Float[Tensor, "d_model"]):
    """Project embedding / projection matrices to the nullspace of `direction`."""
    # Embeddings
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
        model.model.embed_tokens.weight.data, direction
    )

    for blk in model.model.layers:
        # Output projection of self‑attention.
        o_proj = _maybe_get(blk, "o_proj", "dense")
        o_proj.weight.data = get_orthogonalized_matrix(o_proj.weight.data.T, direction).T
        # Second / down projection of MLP.
        down_proj = _maybe_get(blk.mlp, "down_proj", "fc2", "dense_4h_to_h")
        down_proj.weight.data = get_orthogonalized_matrix(down_proj.weight.data.T, direction).T


def act_add_glm4_weights(model, direction: Float[Tensor, "d_model"], coeff, layer: int):
    """Add a learned *direction* as a bias term to the MLP down projection."""
    blk = model.model.layers[layer - 1]  # 1‑based indexing for user convenience
    down_proj = _maybe_get(blk.mlp, "down_proj", "fc2", "dense_4h_to_h")
    dtype = down_proj.weight.dtype
    device = down_proj.weight.device
    bias = (coeff * direction).to(dtype=dtype, device=device)
    down_proj.bias = torch.nn.Parameter(bias)