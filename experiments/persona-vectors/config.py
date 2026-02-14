"""
Configuration settings for persona vector experiments.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass
class Config:
    """Central configuration for persona vector experiments."""

    # ==========================================================================
    # Model Settings
    # ==========================================================================
    model_id: str = "meta-llama/Llama-3.1-70B-Instruct"
    use_vllm: bool = False
    load_in_4bit: bool = True
    compute_dtype: str = "float16"  # "float16" or "bfloat16"
    device_map: str = "auto"

    # ==========================================================================
    # Mode: Controls speed vs reliability tradeoff
    # ==========================================================================
    # 'fast'     -> ~35 model calls, 2-3 min
    # 'balanced' -> ~60 model calls, 3-5 min (RECOMMENDED for 70B)
    # 'robust'   -> ~120 model calls, 5-7 min
    mode: str = "balanced"

    # ==========================================================================
    # Trait Selection
    # ==========================================================================
    test_only_compassion: bool = True
    traits_to_extract: List[str] = field(default_factory=lambda: ["compassion"])

    # ==========================================================================
    # Vector Loading
    # ==========================================================================
    load_vectors_from_hf: bool = False
    hf_vector_repos: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "compassion": {
            "repo_id": "CompassioninMachineLearning/llama-3.1-70b-persona-vector-compassion-layer_9"
        },
        # "open_mindedness": {"repo_id": "..."},
        # "non_helpfulness": {"repo_id": "..."},
    })

    # ==========================================================================
    # Layer Selection (expanded to cover more of the model)
    # ==========================================================================
    target_layers: Optional[List[int]] = None  # None = auto-detect based on model
    layer_range_start: int = 5
    layer_range_end: int = 75  # Expanded from 15 to cover more layers
    layer_search_strategy: str = "sparse"  # "dense", "sparse", or "percentile"

    # ==========================================================================
    # Extraction Settings (increased for statistical validity)
    # ==========================================================================
    num_questions_for_vector: int = 100  # Increased from 20
    last_tokens_to_average: int = 10
    normalize_vectors: bool = True

    # ==========================================================================
    # Prompt Format Settings
    # ==========================================================================
    # 'chat_template' -> Use model's native chat template (system + user roles)
    # 'simple' -> Use "User: {question}\nAssistant:" format
    # IMPORTANT: Must use the same format for extraction AND evaluation!
    prompt_format: str = "chat_template"

    # ==========================================================================
    # Generation Settings
    # ==========================================================================
    generation_temperature: float = 0.3
    max_new_tokens: int = 100

    # ==========================================================================
    # Steering Settings
    # ==========================================================================
    steering_coefficient: float = 2.0
    scale_sweep_values: List[float] = field(default_factory=lambda: [
        0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0
    ])

    # ==========================================================================
    # Evaluation Settings
    # ==========================================================================
    min_difference_rate: float = 0.3
    min_valid_scores_rate: float = 0.6
    min_average_score: float = 3.0

    # Two-stage selection settings (for 'balanced' mode)
    use_two_stage: bool = True
    num_stage1_questions: int = 5
    num_stage2_questions: int = 10
    early_stop_threshold: float = 0.5

    # Single-stage settings (for 'fast' and 'robust' modes)
    num_test_questions: int = 5

    # ==========================================================================
    # HuggingFace Upload Settings
    # ==========================================================================
    upload_to_hf: bool = True
    hf_org: str = "CompassioninMachineLearning"

    # ==========================================================================
    # Analysis Settings
    # ==========================================================================
    analysis_sample_size: int = 500
    analysis_dataset: str = "CompassioninMachineLearning/1k_pretraining_research_documents_Haiku45"

    def __post_init__(self):
        """Configure mode-specific settings after initialization."""
        self._configure_mode()

    def _configure_mode(self):
        """Set parameters based on selected mode."""
        if self.mode == 'fast':
            self.num_questions_for_vector = 50  # Increased from 8
            self.num_test_questions = 15  # Increased from 5
            self.use_two_stage = False
            self.layer_search_strategy = "sparse"
        elif self.mode == 'balanced':
            self.num_questions_for_vector = 100  # Increased from 20
            self.num_stage1_questions = 15  # Increased from 5
            self.num_stage2_questions = 30  # Increased from 10
            self.use_two_stage = True
            self.early_stop_threshold = 0.5
            self.layer_search_strategy = "sparse"
        elif self.mode == 'robust':
            self.num_questions_for_vector = 200  # Use most of available questions
            self.num_test_questions = 50  # Use all test questions
            self.use_two_stage = False
            self.layer_search_strategy = "dense"
        else:
            raise ValueError(f"MODE must be 'fast', 'balanced', or 'robust', got: {self.mode}")

    def get_target_layers(self, num_model_layers: int) -> List[int]:
        """
        Get target layers based on search strategy.

        Strategies:
        - "dense": Test every layer in range
        - "sparse": Sample layers across range (default, ~15 layers)
        - "percentile": Sample at fixed percentages of model depth
        """
        if self.target_layers is not None:
            return self.target_layers

        start = self.layer_range_start
        end = min(self.layer_range_end, num_model_layers)

        if self.layer_search_strategy == "dense":
            return list(range(start, end))

        elif self.layer_search_strategy == "sparse":
            # Sample ~15 layers across the range
            total_range = end - start
            if total_range <= 15:
                return list(range(start, end))
            step = max(1, total_range // 15)
            return list(range(start, end, step))

        elif self.layer_search_strategy == "percentile":
            # Sample at 10%, 20%, ..., 90% of model depth
            percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            return [int(num_model_layers * p) for p in percentiles]

        else:
            return list(range(start, end))

    @classmethod
    def from_env(cls) -> "Config":
        """Create config from environment variables."""
        return cls(
            model_id=os.environ.get("PERSONA_MODEL_ID", cls.model_id),
            mode=os.environ.get("PERSONA_MODE", "balanced"),
            load_vectors_from_hf=os.environ.get("PERSONA_LOAD_FROM_HF", "false").lower() == "true",
            test_only_compassion=os.environ.get("PERSONA_TEST_ONLY_COMPASSION", "true").lower() == "true",
        )


# Default configuration instance
default_config = Config()
