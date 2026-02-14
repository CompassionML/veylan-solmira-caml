"""
Vector extraction functionality for persona vectors.
"""

from typing import Optional, List, Dict, Tuple
import numpy as np

from .config import Config
from .model import ModelManager
from .artifacts import TraitArtifacts, get_artifacts


class VectorExtractor:
    """Extracts persona vectors from contrastive prompts."""

    def __init__(
        self,
        model_manager: ModelManager,
        config: Optional[Config] = None,
    ):
        """Initialize vector extractor."""
        self.model_manager = model_manager
        self.config = config or model_manager.config

    def extract_persona_vector(
        self,
        trait_name: str,
        target_layers: Optional[List[int]] = None,
        num_questions: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """
        Extract persona vectors for a given trait across specified layers.

        Returns:
            Tuple of (vectors_dict, layer_stats_dict)
        """
        artifacts = get_artifacts(trait_name)
        return self._extract_from_artifacts(
            trait_name=trait_name,
            artifacts=artifacts,
            target_layers=target_layers,
            num_questions=num_questions,
        )

    def _extract_from_artifacts(
        self,
        trait_name: str,
        artifacts: TraitArtifacts,
        target_layers: Optional[List[int]] = None,
        num_questions: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict]]:
        """Extract vectors from trait artifacts."""
        print(f"\n--- {trait_name.replace('_', ' ').title()} Vector Extraction ---")

        # Get target layers
        if target_layers is None:
            target_layers = self.config.get_target_layers(self.model_manager.num_layers)

        num_questions = num_questions or self.config.num_questions_for_vector
        questions = artifacts["questions"][:num_questions]
        pos_instructions = artifacts["positive_instructions"]
        neg_instructions = artifacts["negative_instructions"]

        # Initialize activation collectors
        all_pos_activations = {f"layer_{layer}": [] for layer in target_layers}
        all_neg_activations = {f"layer_{layer}": [] for layer in target_layers}

        # Create prompts
        pos_prompts = []
        neg_prompts = []

        for i, question in enumerate(questions):
            pos_instr = pos_instructions[i % len(pos_instructions)]
            neg_instr = neg_instructions[i % len(neg_instructions)]

            pos_prompt = self.model_manager.format_chat_prompt(pos_instr, question)
            neg_prompt = self.model_manager.format_chat_prompt(neg_instr, question)

            pos_prompts.append(pos_prompt)
            neg_prompts.append(neg_prompt)

        print(f"Processing {len(pos_prompts)} {trait_name} questions...")

        # Process positive prompts
        pos_responses, pos_activations_list = self.model_manager.generate_with_activations(
            pos_prompts,
            target_layers=target_layers,
        )

        # Process negative prompts
        neg_responses, neg_activations_list = self.model_manager.generate_with_activations(
            neg_prompts,
            target_layers=target_layers,
        )

        # Collect activations
        for i in range(len(questions)):
            if i < len(pos_activations_list) and pos_activations_list[i]:
                for layer_idx in target_layers:
                    layer_key = f"layer_{layer_idx}"
                    if layer_key in pos_activations_list[i] and pos_activations_list[i][layer_key] is not None:
                        all_pos_activations[layer_key].append(pos_activations_list[i][layer_key])

            if i < len(neg_activations_list) and neg_activations_list[i]:
                for layer_idx in target_layers:
                    layer_key = f"layer_{layer_idx}"
                    if layer_key in neg_activations_list[i] and neg_activations_list[i][layer_key] is not None:
                        all_neg_activations[layer_key].append(neg_activations_list[i][layer_key])

        # Calculate persona vectors
        vectors = {}
        layer_stats = {}

        for layer_key in all_pos_activations.keys():
            if all_pos_activations[layer_key] and all_neg_activations[layer_key]:
                pos_mean = np.mean(all_pos_activations[layer_key], axis=0)
                neg_mean = np.mean(all_neg_activations[layer_key], axis=0)
                persona_vector = pos_mean - neg_mean

                # Optionally normalize
                if self.config.normalize_vectors:
                    norm = np.linalg.norm(persona_vector)
                    if norm > 0:
                        persona_vector = persona_vector / norm

                vectors[layer_key] = persona_vector
                magnitude = np.linalg.norm(persona_vector)
                layer_stats[layer_key] = {
                    'magnitude': magnitude,
                    'num_samples': len(all_pos_activations[layer_key])
                }
                print(f"  {layer_key}: magnitude = {magnitude:.3f}, samples = {layer_stats[layer_key]['num_samples']}")

        print(f"Extracted {trait_name} vectors for {len(vectors)} layers")

        return vectors, layer_stats

    def extract_all_traits(
        self,
        traits: Optional[List[str]] = None,
        target_layers: Optional[List[int]] = None,
    ) -> Dict[str, Tuple[Dict[str, np.ndarray], Dict[str, Dict]]]:
        """Extract vectors for multiple traits."""
        if traits is None:
            traits = self.config.traits_to_extract

        results = {}
        for trait_name in traits:
            vectors, stats = self.extract_persona_vector(
                trait_name=trait_name,
                target_layers=target_layers,
            )
            results[trait_name] = (vectors, stats)

        return results


class VectorLoader:
    """Loads pre-computed vectors from HuggingFace Hub."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize vector loader."""
        self.config = config or Config()

    def load_vector_from_hf(
        self,
        repo_id: str,
        trait_name: str,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Download a vector from HuggingFace Hub."""
        import re
        from huggingface_hub import hf_hub_download

        # Extract layer name from repo_id
        match = re.search(r'-layer_(\d+)', repo_id)
        if not match:
            print(f"Could not extract layer name from repo_id: {repo_id}")
            return None, None

        layer_name = f"layer_{match.group(1)}"
        filename = f"{trait_name}_vector_{layer_name}.npy"

        try:
            vector_path = hf_hub_download(repo_id=repo_id, filename=filename)
            vector = np.load(vector_path)
            print(f"Loaded {trait_name} vector from {repo_id} for {layer_name}")
            return vector, layer_name
        except Exception as e:
            print(f"Error loading {trait_name} vector from {repo_id}: {e}")
            return None, None

    def load_all_vectors(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Load all configured vectors from HuggingFace."""
        all_vectors = {}

        for trait_name, repo_info in self.config.hf_vector_repos.items():
            repo_id = repo_info["repo_id"]
            vector, layer_name = self.load_vector_from_hf(repo_id, trait_name)

            if vector is not None:
                all_vectors[trait_name] = {
                    'best': vector,
                    layer_name: vector,
                }
                print(f"Loaded {trait_name} vector for {layer_name}")

        return all_vectors


class VectorUploader:
    """Uploads vectors to HuggingFace Hub."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize vector uploader."""
        self.config = config or Config()

    def upload_vector(
        self,
        vector: np.ndarray,
        trait_name: str,
        layer_name: str,
        hf_token: Optional[str] = None,
    ) -> bool:
        """Upload a vector to HuggingFace Hub."""
        import os
        from huggingface_hub import HfApi, create_repo, CommitOperationAdd

        try:
            # Get token
            if hf_token is None:
                hf_token = os.environ.get('HF_TOKEN')
                if hf_token is None:
                    try:
                        from google.colab import userdata
                        hf_token = userdata.get('HF_TOKEN')
                    except:
                        pass

            if not hf_token:
                print("HF_TOKEN not found")
                return False

            api = HfApi()
            repo_id = f"{self.config.hf_org}/llama-3.1-70b-persona-vector-{trait_name}-{layer_name}"

            create_repo(repo_id, repo_type="model", exist_ok=True, token=hf_token)

            filename = f"{trait_name}_vector_{layer_name}.npy"
            np.save(filename, vector)

            api.create_commit(
                repo_id=repo_id,
                operations=[CommitOperationAdd(path_in_repo=filename, path_or_fileobj=filename)],
                commit_message=f"Add {trait_name} vector for {layer_name}",
                token=hf_token,
                create_pr=False
            )

            os.remove(filename)
            print(f"Uploaded to {repo_id}")
            return True

        except Exception as e:
            print(f"Upload failed: {e}")
            return False
