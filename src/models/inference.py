import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Optional, Dict, Any
from pathlib import Path
from peft import PeftModel

class SLMInference:
    """
    Unified inference interface for Qwen2.5-3B-Instruct, Mistral-7B-Instruct, and LLaMA-3.2-3B-Instruct.
    Supports model loading, prompt formatting, generation, and decoding.
    """

    MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
        "mistral": {
            "name": "mistralai/Mistral-7B-Instruct-v0.3",
            "chat_template": "mistral",
        },
        "qwen2.5-3b": {
            "name": "Qwen/Qwen2.5-3B-Instruct",
            "chat_template": "qwen",
        },
        "qwen3.5-9b": {
            "name": "Qwen/Qwen3.5-9B-Instruct",
            "chat_template": "qwen",
        },
        "llama": {
            "name": "meta-llama/Llama-3.2-3B-Instruct",
            "chat_template": "llama",
        }, 
    }

    def __init__(
        self,
        model_key: str,
        device: str = "auto",
        load_in_4bit: bool = False,
        trust_remote_code: bool = True,
        revision: Optional[str] = None,
        model_dir: Optional[str] = None,  # <-- add this
        adapter_dir: Optional[str] = None,
    ):
        assert model_key in self.MODEL_CONFIGS, f"Unknown model_key: {model_key}"
        config = self.MODEL_CONFIGS[model_key]
        model_path = Path(model_dir).resolve() if model_dir else None
        use_local_files_only = bool(model_path and model_path.is_dir())
        model_name = str(model_path) if use_local_files_only else config["name"]

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=use_local_files_only,
        )

        # Model loading options
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": device,
            "trust_remote_code": trust_remote_code,
            # "attn_implementation": "flash_attention_2",  # Temporarily disabled due to CUDA/torch version mismatch
        }
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        if revision:
            load_kwargs["revision"] = revision
        if use_local_files_only:
            load_kwargs["local_files_only"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        if adapter_dir:
            print(f"Loading LoRA adapter from {adapter_dir}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_dir)
            self.model = self.model.merge_and_unload()
            
        self.model_key = model_key

    def _format_prompt(self, prompt: str) -> str:
        """
        Formats prompt using the model's chat template.
        """
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        do_sample: bool = True,
        top_p: float = 0.95,
        **kwargs
    ) -> str:
        """
        Generates a response from the model given a prompt.
        """
        formatted_prompt = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                **kwargs,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            else:
                generation_kwargs.setdefault("temperature", None)
                generation_kwargs.setdefault("top_p", None)
                generation_kwargs.setdefault("top_k", None)

            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )

        # Only decode the newly generated tokens
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
        return response.strip()

    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        do_sample: bool = False,
        top_p: float = 0.95,
        **kwargs
    ) -> List[str]:
        """
        Generates responses for a batch of prompts.
        """
        if not prompts:
            return []
            
        formatted_prompts = [self._format_prompt(p) for p in prompts]
        
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.model.device)
        
        with torch.no_grad():
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.pad_token_id,
                **kwargs,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            else:
                generation_kwargs.setdefault("temperature", None)
                generation_kwargs.setdefault("top_p", None)
                generation_kwargs.setdefault("top_k", None)
                
            outputs = self.model.generate(
                **inputs,
                **generation_kwargs
            )
            
        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[:, input_length:]
        
        responses = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        return [r.strip() for r in responses]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Returns model configuration and device info.
        """
        return {
            "model_key": self.model_key,
            "model_name": self.MODEL_CONFIGS[self.model_key]["name"],
            "device": str(self.model.device),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
        }