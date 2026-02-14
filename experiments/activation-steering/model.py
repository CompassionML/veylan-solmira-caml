"""
Model loading and basic generation functionality.
"""

import gc
from typing import Optional, List, Dict, Any, Tuple

import torch
import numpy as np

from .config import Config


class ModelManager:
    """Manages model loading and basic generation."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize model manager with configuration."""
        self.config = config or Config()
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._num_layers: Optional[int] = None

    @property
    def num_layers(self) -> int:
        """Get number of layers in the model."""
        if self._num_layers is None:
            if self.model is None:
                raise RuntimeError("Model not loaded. Call load_model() first.")
            self._num_layers = len(self.model.model.layers)
        return self._num_layers

    def load_model(self) -> None:
        """Load model and tokenizer with configured settings."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        print(f"Loading model: {self.config.model_id}")
        print(f"Device: {self.device}")

        if self.config.load_in_4bit:
            print("Using 4-bit quantization")
            compute_dtype = torch.float16 if self.config.compute_dtype == "float16" else torch.bfloat16

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                quantization_config=bnb_config,
                torch_dtype=compute_dtype,
                device_map=self.config.device_map
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_id,
                torch_dtype=torch.float16,
                device_map=self.config.device_map
            )

        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            self.model.resize_token_embeddings(len(self.tokenizer))
            print("Added pad token to tokenizer.")

        self.model.eval()
        self._num_layers = len(self.model.model.layers)

        print(f"Model loaded successfully.")
        print(f"  Layers: {self._num_layers}")
        print(f"  Hidden size: {self.model.config.hidden_size}")

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate a response for a single prompt."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.generation_temperature

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        return response

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Generate responses for a batch of prompts."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.generation_temperature

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        responses = []
        for i in range(output_ids.shape[0]):
            response = self.tokenizer.decode(
                output_ids[i][inputs['input_ids'][i].shape[0]:],
                skip_special_tokens=True
            )
            responses.append(response)

        return responses

    def extract_activation(
        self,
        text: str,
        layer_idx: int,
        max_length: int = 256,
    ) -> Optional[np.ndarray]:
        """Extract activation from a specific layer without generation."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_length
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        if layer_idx + 1 < len(outputs.hidden_states):
            hidden_states = outputs.hidden_states[layer_idx + 1].squeeze(0)
            activation = hidden_states.mean(dim=0).cpu().numpy().astype(np.float32)
        else:
            activation = None

        # Cleanup
        del outputs, inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return activation

    def generate_with_activations(
        self,
        prompts: List[str],
        target_layers: List[int],
        last_tokens_to_average: Optional[int] = None,
    ) -> Tuple[List[str], List[Dict[str, np.ndarray]]]:
        """Generate responses and extract activations from specified layers."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        last_tokens_to_average = last_tokens_to_average or self.config.last_tokens_to_average

        # Ensure prompts is a list
        if isinstance(prompts, str):
            prompts = [prompts]

        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=self.config.generation_temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode responses
        responses = []
        for i in range(output_ids.shape[0]):
            input_length = inputs['input_ids'].shape[1]
            response_ids = output_ids[i][input_length:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            responses.append(response)

        # Extract activations
        processed_activations_list = []

        for i in range(output_ids.shape[0]):
            full_sequence_ids = output_ids[i].unsqueeze(0).to(self.model.device)
            attention_mask = (full_sequence_ids != self.tokenizer.pad_token_id).long().to(self.model.device)

            processed_activations = {}

            if full_sequence_ids.shape[1] > 0:
                try:
                    with torch.no_grad():
                        outputs = self.model(
                            input_ids=full_sequence_ids,
                            attention_mask=attention_mask,
                            output_hidden_states=True
                        )

                    for layer_idx in target_layers:
                        if layer_idx + 1 < len(outputs.hidden_states):
                            hidden_states = outputs.hidden_states[layer_idx + 1].squeeze(0)
                            hidden_states = hidden_states.to(torch.float32)

                            # Average over last N tokens
                            start_idx = max(0, hidden_states.shape[0] - last_tokens_to_average)
                            mean_activation = hidden_states[start_idx:].mean(dim=0)
                            processed_activations[f"layer_{layer_idx}"] = mean_activation.detach().cpu().numpy()

                except Exception as e:
                    print(f"Error during forward pass for sample {i}: {e}")
                    processed_activations = {}

            processed_activations_list.append(processed_activations)

        return responses, processed_activations_list

    def format_chat_prompt(
        self,
        system_content: str,
        user_content: str,
    ) -> str:
        """Format a prompt using the model's chat template."""
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def format_simple_prompt(self, question: str) -> str:
        """Format a simple user/assistant prompt."""
        return f"User: {question}\n\nAssistant:"

    def format_evaluation_prompt(
        self,
        question: str,
        system_content: Optional[str] = None,
    ) -> str:
        """
        Format a prompt for evaluation using the configured format.

        IMPORTANT: This should match the format used during vector extraction
        for proper steering effectiveness.

        Args:
            question: The question/user content
            system_content: Optional system instruction (used in chat_template mode)

        Returns:
            Formatted prompt string
        """
        if self.config.prompt_format == "chat_template":
            # Use model's native chat template
            if system_content:
                return self.format_chat_prompt(system_content, question)
            else:
                # No system message, just user message
                messages = [{"role": "user", "content": question}]
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        else:
            # Simple format
            return self.format_simple_prompt(question)

    def cleanup(self) -> None:
        """Free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
