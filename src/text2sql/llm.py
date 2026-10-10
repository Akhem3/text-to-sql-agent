"""Backends d'inférence interchangeables : même interface generate(messages) -> str.

Les imports lourds (torch, transformers, llama_cpp) sont faits à l'instanciation,
pas à l'import du module : importer ce fichier ne charge aucun modèle.
"""

from typing import Protocol

from .config import MAX_NEW_TOKENS


class Backend(Protocol):
    def generate(self, messages: list[dict]) -> str: ...


class HFBackend:
    """transformers, décodage greedy, adaptateur LoRA optionnel (peft)."""

    def __init__(self, model_id: str, adapter: str | None = None, max_new_tokens: int = MAX_NEW_TOKENS):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, clean_up_tokenization_spaces=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(self.device)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    def generate(self, messages: list[dict]) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class LlamaCppBackend:
    """llama.cpp via llama-cpp-python : modèle GGUF, quantifié ou non."""

    def __init__(self, gguf_path: str, n_ctx: int = 4096, max_new_tokens: int = MAX_NEW_TOKENS):
        from llama_cpp import Llama

        self.llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_gpu_layers=-1, verbose=False)
        self.max_new_tokens = max_new_tokens

    def generate(self, messages: list[dict]) -> str:
        output = self.llm.create_chat_completion(
            messages=messages, temperature=0.0, max_tokens=self.max_new_tokens
        )
        return output["choices"][0]["message"]["content"].strip()


def load_backend(name: str, model: str, adapter: str | None = None) -> Backend:
    if name == "hf":
        return HFBackend(model, adapter=adapter)
    if name == "llamacpp":
        return LlamaCppBackend(model)
    raise ValueError(f"Backend inconnu : {name!r} (attendu : 'hf' ou 'llamacpp')")
