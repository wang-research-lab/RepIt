from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm
from torch import Tensor
from jaxtyping import Int, Float
import torch
import gc

from pipeline.utils.hook_utils import add_hooks

class ModelBase(ABC):
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.model: AutoModelForCausalLM = self._load_model(model_name_or_path)
        self.tokenizer: AutoTokenizer = self._load_tokenizer(model_name_or_path)

        
        self.tokenize_instructions_fn = self._get_tokenize_instructions_fn()
        self.eoi_toks = self._get_eoi_toks()
        self.refusal_toks = self._get_refusal_toks()

        self.model_block_modules = self._get_model_block_modules()
        self.model_attn_modules = self._get_attn_modules()
        self.model_mlp_modules = self._get_mlp_modules()
        self.model_post_attn_modules = self._get_post_attn_modules()
        self.gen_config = self._get_generation_config()

    def del_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model

    @abstractmethod
    def _get_generation_config(self) -> GenerationConfig:
        pass

    @abstractmethod
    def _load_model(self, model_name_or_path: str) -> AutoModelForCausalLM:
        pass

    @abstractmethod
    def _load_tokenizer(self, model_name_or_path: str) -> AutoTokenizer:
        pass

    @abstractmethod
    def _get_tokenize_instructions_fn(self):
        pass

    @abstractmethod
    def _get_eoi_toks(self):
        pass

    @abstractmethod
    def _get_refusal_toks(self):
        pass

    @abstractmethod
    def _get_model_block_modules(self):
        pass

    @abstractmethod
    def _get_attn_modules(self):
        pass

    @abstractmethod
    def _get_mlp_modules(self):
        pass
    
    @abstractmethod
    def _get_post_attn_modules(self):
        pass

    """@abstractmethod
    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        pass

    @abstractmethod
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff: float, layer: int):
        pass"""

    def generate_completions(
        self, dataset, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8,
        max_new_tokens=64, progress=True
    ):
        generation_config = self.gen_config
        generation_config.max_new_tokens = max_new_tokens
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] if 'category' in x else None for x in dataset]

        # tqdm setup
        progress_iter = tqdm(
            range(0, len(dataset), batch_size),
            desc="Generating completions",
            disable=not progress
        )
        
        for i in progress_iter:
            tokenized_instructions = self.tokenize_instructions_fn(
                instructions=instructions[i:i + batch_size]
            )

            with torch.inference_mode():
                with add_hooks(module_forward_pre_hooks=fwd_pre_hooks,
                            module_forward_hooks=fwd_hooks):

                    generation_toks = self.model.generate(
                        input_ids=tokenized_instructions.input_ids.to(self.model.device),
                        attention_mask=tokenized_instructions.attention_mask.to(self.model.device),
                        generation_config=generation_config,
                    )

            # Slice off the prompt tokens
            generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

            # Build completions
            for generation_idx, generation in enumerate(generation_toks):
                completions.append({
                    'category': categories[i + generation_idx],
                    'prompt': instructions[i + generation_idx],
                    'response': self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                })

            # --- cleanup step ---
            del tokenized_instructions, generation_toks
            torch.cuda.empty_cache()
            gc.collect()

        return completions

    def generate_completions_system(self, dataset, system_prompt, fwd_pre_hooks=[], fwd_hooks=[], batch_size=8, max_new_tokens=64, progress = True):

        generation_config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        completions = []
        instructions = [x['instruction'] for x in dataset]
        categories = [x['category'] if 'category' in x else None for x in dataset]

        # Select tqdm functionality based on the tqdm parameter
        progress_iter = tqdm(range(0, len(dataset), batch_size), disable=not progress)

        for i in progress_iter:
            tokenized_instructions = self.tokenize_instructions_fn(instructions=instructions[i:i + batch_size], system=system_prompt)

            with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=fwd_hooks):
                generation_toks = self.model.generate(
                    input_ids=tokenized_instructions.input_ids.to(self.model.device),
                    attention_mask=tokenized_instructions.attention_mask.to(self.model.device),
                    generation_config=generation_config,
                )

                generation_toks = generation_toks[:, tokenized_instructions.input_ids.shape[-1]:]

                for generation_idx, generation in enumerate(generation_toks):
                    completions.append({
                        'category': categories[i + generation_idx],
                        'prompt': instructions[i + generation_idx],
                        'response': self.tokenizer.decode(generation, skip_special_tokens=True).strip()
                    })

        return completions
