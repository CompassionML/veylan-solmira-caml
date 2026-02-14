"""
Steering functionality using forward hooks.
"""

import asyncio
from typing import Optional, List, Dict, Any, Callable

import torch
import numpy as np

from .config import Config
from .model import ModelManager


class SteeringManager:
    """Manages activation steering via forward hooks."""

    def __init__(
        self,
        model_manager: ModelManager,
        config: Optional[Config] = None,
    ):
        """Initialize steering manager."""
        self.model_manager = model_manager
        self.config = config or model_manager.config
        self._active_hooks: List[Any] = []

    @property
    def model(self):
        return self.model_manager.model

    @property
    def tokenizer(self):
        return self.model_manager.tokenizer

    def create_steering_hook(
        self,
        vector: np.ndarray,
        coefficient: float,
    ) -> Callable:
        """Create a steering hook function."""
        def steering_hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output

            vector_tensor = torch.tensor(
                vector,
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            vector_tensor = vector_tensor.unsqueeze(0).unsqueeze(0)

            # Paper method: h_l = h_l + alpha * v_l
            steered_hidden_states = hidden_states + coefficient * vector_tensor

            if isinstance(output, tuple):
                return (steered_hidden_states,) + output[1:]
            else:
                return steered_hidden_states

        return steering_hook

    def apply_steering_and_generate(
        self,
        prompt: str,
        vector: np.ndarray,
        layer_idx: int,
        coefficient: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """Apply steering at a specific layer during generation."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        coefficient = coefficient or self.config.steering_coefficient
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.generation_temperature

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        target_layer = self.model.model.layers[layer_idx]

        hook = self.create_steering_hook(vector, coefficient)
        hook_handle = target_layer.register_forward_hook(hook)

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            response = self.tokenizer.decode(
                output_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            return response

        except Exception as e:
            print(f"Error during steered generation: {e}")
            return None

        finally:
            hook_handle.remove()

    def generate_steered_batch(
        self,
        prompts: List[str],
        vector: np.ndarray,
        layer_idx: int,
        coefficient: Optional[float] = None,
    ) -> List[str]:
        """Generate steered responses for a batch of prompts."""
        if self.model is None:
            raise RuntimeError("Model not loaded.")

        coefficient = coefficient or self.config.steering_coefficient

        if not prompts:
            return []

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        target_layer = self.model.model.layers[layer_idx]

        # Create hook with model's dtype
        vector_tensor = torch.tensor(
            vector,
            dtype=self.model.dtype,
            device=self.model.device
        ).unsqueeze(0).unsqueeze(0)

        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            steered = hidden_states + coefficient * vector_tensor
            return (steered,) + output[1:] if isinstance(output, tuple) else steered

        handle = target_layer.register_forward_hook(hook)

        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.generation_temperature,
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

        finally:
            handle.remove()

    async def generate_steered_async(
        self,
        prompts: List[str],
        vector: np.ndarray,
        layer_idx: int,
        coefficient: Optional[float] = None,
    ) -> List[str]:
        """Async wrapper for batch steered generation."""
        return self.generate_steered_batch(prompts, vector, layer_idx, coefficient)

    def compare_baseline_vs_steered(
        self,
        prompts: List[str],
        vector: np.ndarray,
        layer_idx: int,
        coefficient: Optional[float] = None,
    ) -> Dict[str, List[str]]:
        """Generate both baseline and steered responses for comparison."""
        baseline_responses = self.model_manager.generate_batch(prompts)
        steered_responses = self.generate_steered_batch(
            prompts, vector, layer_idx, coefficient
        )

        return {
            'baseline': baseline_responses,
            'steered': steered_responses,
        }

    def scale_sweep(
        self,
        prompt: str,
        vector: np.ndarray,
        layer_idx: int,
        scales: Optional[List[float]] = None,
    ) -> Dict[float, str]:
        """Sweep through different steering scales."""
        if scales is None:
            scales = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

        results = {}
        for scale in scales:
            if scale == 0.0:
                response = self.model_manager.generate_response(prompt)
            else:
                response = self.apply_steering_and_generate(
                    prompt, vector, layer_idx, coefficient=scale
                )
            results[scale] = response

        return results
