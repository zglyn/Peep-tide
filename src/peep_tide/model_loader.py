"""Load the configured local instruct model. Kept off the rescue import path."""

from peep_tide import LanguageModel, RescueConfig


class _LocalModel:
    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization,
            device_map="auto",
        )

    def generate(self, prompt: str) -> str:
        import torch

        tokens = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            produced = self._model.generate(
                **tokens,
                max_new_tokens=180,
                do_sample=False,
            )
        new_tokens = produced[0, tokens["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)


def load_language_model(config: RescueConfig | None = None) -> LanguageModel:
    chosen = config or RescueConfig()
    return _LocalModel(chosen.model_id)
