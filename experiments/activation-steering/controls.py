"""
Control experiments for validating steering effects.

These controls are CRITICAL for determining whether observed effects
are real steering or artifacts/noise.

Controls implemented:
1. Random vector control - steer with random vector, should show no effect
2. Negative steering control - steer with -coefficient, should show opposite effect
3. Roundtrip control - apply hook with coefficient=0, should show no effect
"""

from typing import Optional, List, Dict, Any
import numpy as np

from .config import Config
from .model import ModelManager
from .steering import SteeringManager
from .evaluation import GeminiJudge
from .artifacts import get_artifacts


class ControlExperiments:
    """Run control experiments to validate steering effects."""

    def __init__(
        self,
        model_manager: ModelManager,
        steering_manager: SteeringManager,
        config: Optional[Config] = None,
    ):
        """Initialize control experiments."""
        self.model_manager = model_manager
        self.steering_manager = steering_manager
        self.config = config or model_manager.config
        self.judge = GeminiJudge()

    def generate_random_vector(
        self,
        hidden_dim: Optional[int] = None,
        seed: int = 42,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate a random unit vector for control experiments.

        Args:
            hidden_dim: Dimension of the vector (default: model's hidden size)
            seed: Random seed for reproducibility
            normalize: Whether to normalize to unit vector

        Returns:
            Random vector of shape (hidden_dim,)
        """
        if hidden_dim is None:
            hidden_dim = self.model_manager.model.config.hidden_size

        np.random.seed(seed)
        random_vec = np.random.randn(hidden_dim).astype(np.float32)

        if normalize:
            random_vec = random_vec / np.linalg.norm(random_vec)

        return random_vec

    def run_random_vector_control(
        self,
        layer_idx: int,
        trait_name: str,
        test_questions: Optional[List[str]] = None,
        coefficient: Optional[float] = None,
        num_random_seeds: int = 3,
    ) -> Dict[str, Any]:
        """
        Test that random vectors don't produce similar effects.

        If random vectors produce similar effects to the target vector,
        the observed effect is likely noise, not real steering.

        Args:
            layer_idx: Layer to apply steering
            trait_name: Trait being evaluated
            test_questions: Questions to test (default: from artifacts)
            coefficient: Steering coefficient (default: from config)
            num_random_seeds: Number of random vectors to test

        Returns:
            Dict with random vector scores and analysis
        """
        if test_questions is None:
            artifacts = get_artifacts(trait_name)
            test_questions = artifacts["test_questions"][:10]

        coefficient = coefficient or self.config.steering_coefficient
        hidden_dim = self.model_manager.model.config.hidden_size

        print(f"\n=== RANDOM VECTOR CONTROL ===")
        print(f"Testing {num_random_seeds} random vectors at layer {layer_idx}")

        all_random_scores = []

        for seed in range(num_random_seeds):
            random_vec = self.generate_random_vector(hidden_dim, seed=seed)
            seed_scores = []

            for q in test_questions:
                prompt = self.model_manager.format_evaluation_prompt(q)
                response = self.steering_manager.apply_steering_and_generate(
                    prompt, random_vec, layer_idx, coefficient
                )

                if response:
                    score = self.judge.judge_response(response, q, trait_name)
                    if score is not None:
                        seed_scores.append(score)

            if seed_scores:
                avg = np.mean(seed_scores)
                all_random_scores.append(avg)
                print(f"  Seed {seed}: avg score = {avg:.2f}")

        results = {
            'random_scores': all_random_scores,
            'random_mean': np.mean(all_random_scores) if all_random_scores else None,
            'random_std': np.std(all_random_scores) if all_random_scores else None,
            'num_seeds': num_random_seeds,
            'num_questions': len(test_questions),
        }

        if results['random_mean']:
            print(f"\nRandom vector mean: {results['random_mean']:.2f} (+/- {results['random_std']:.2f})")

        return results

    def run_negative_steering_control(
        self,
        vector: np.ndarray,
        layer_idx: int,
        trait_name: str,
        test_questions: Optional[List[str]] = None,
        coefficient: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Test that negative steering produces opposite effects.

        If steering is real, negative coefficient should reduce the trait.
        Expected pattern: negative_score < baseline_score < positive_score

        Args:
            vector: The steering vector to test
            layer_idx: Layer to apply steering
            trait_name: Trait being evaluated
            test_questions: Questions to test
            coefficient: Steering coefficient magnitude

        Returns:
            Dict with positive, negative, and baseline scores
        """
        if test_questions is None:
            artifacts = get_artifacts(trait_name)
            test_questions = artifacts["test_questions"][:10]

        coefficient = coefficient or self.config.steering_coefficient

        print(f"\n=== NEGATIVE STEERING CONTROL ===")
        print(f"Testing +{coefficient} vs -{coefficient} at layer {layer_idx}")

        results = {
            'positive': [],
            'negative': [],
            'baseline': [],
        }

        for q in test_questions:
            prompt = self.model_manager.format_evaluation_prompt(q)

            # Baseline (no steering)
            baseline_resp = self.model_manager.generate_response(prompt)
            baseline_score = self.judge.judge_response(baseline_resp, q, trait_name)
            if baseline_score:
                results['baseline'].append(baseline_score)

            # Positive steering
            pos_resp = self.steering_manager.apply_steering_and_generate(
                prompt, vector, layer_idx, coefficient=coefficient
            )
            if pos_resp:
                pos_score = self.judge.judge_response(pos_resp, q, trait_name)
                if pos_score:
                    results['positive'].append(pos_score)

            # Negative steering
            neg_resp = self.steering_manager.apply_steering_and_generate(
                prompt, vector, layer_idx, coefficient=-coefficient
            )
            if neg_resp:
                neg_score = self.judge.judge_response(neg_resp, q, trait_name)
                if neg_score:
                    results['negative'].append(neg_score)

        # Calculate means
        results['baseline_mean'] = np.mean(results['baseline']) if results['baseline'] else None
        results['positive_mean'] = np.mean(results['positive']) if results['positive'] else None
        results['negative_mean'] = np.mean(results['negative']) if results['negative'] else None

        # Check expected pattern
        if all([results['baseline_mean'], results['positive_mean'], results['negative_mean']]):
            expected_pattern = (
                results['negative_mean'] < results['baseline_mean'] < results['positive_mean']
            )
            results['pattern_holds'] = expected_pattern

            print(f"\nResults:")
            print(f"  Negative (-{coefficient}): {results['negative_mean']:.2f}")
            print(f"  Baseline (0):      {results['baseline_mean']:.2f}")
            print(f"  Positive (+{coefficient}): {results['positive_mean']:.2f}")
            print(f"  Expected pattern (neg < base < pos): {'YES' if expected_pattern else 'NO'}")
        else:
            results['pattern_holds'] = None
            print("Insufficient scores to evaluate pattern")

        return results

    def run_roundtrip_control(
        self,
        vector: np.ndarray,
        layer_idx: int,
        trait_name: str,
        test_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Test that the hook mechanism itself doesn't cause effects.

        Apply steering with coefficient=0 (roundtrip only).
        Should produce same results as baseline.

        Args:
            vector: The steering vector (used with coef=0)
            layer_idx: Layer to apply hook
            trait_name: Trait being evaluated
            test_questions: Questions to test

        Returns:
            Dict with roundtrip vs baseline comparison
        """
        if test_questions is None:
            artifacts = get_artifacts(trait_name)
            test_questions = artifacts["test_questions"][:10]

        print(f"\n=== ROUNDTRIP CONTROL ===")
        print(f"Testing coefficient=0 (hook only, no steering)")

        baseline_scores = []
        roundtrip_scores = []

        for q in test_questions:
            prompt = self.model_manager.format_evaluation_prompt(q)

            # Baseline (no hook at all)
            baseline_resp = self.model_manager.generate_response(prompt)
            baseline_score = self.judge.judge_response(baseline_resp, q, trait_name)
            if baseline_score:
                baseline_scores.append(baseline_score)

            # Roundtrip (hook with coef=0)
            roundtrip_resp = self.steering_manager.apply_steering_and_generate(
                prompt, vector, layer_idx, coefficient=0.0
            )
            if roundtrip_resp:
                roundtrip_score = self.judge.judge_response(roundtrip_resp, q, trait_name)
                if roundtrip_score:
                    roundtrip_scores.append(roundtrip_score)

        results = {
            'baseline_scores': baseline_scores,
            'roundtrip_scores': roundtrip_scores,
            'baseline_mean': np.mean(baseline_scores) if baseline_scores else None,
            'roundtrip_mean': np.mean(roundtrip_scores) if roundtrip_scores else None,
        }

        if results['baseline_mean'] and results['roundtrip_mean']:
            diff = abs(results['roundtrip_mean'] - results['baseline_mean'])
            results['difference'] = diff
            results['hook_neutral'] = diff < 0.3  # Threshold for "no effect"

            print(f"\nResults:")
            print(f"  Baseline mean:  {results['baseline_mean']:.2f}")
            print(f"  Roundtrip mean: {results['roundtrip_mean']:.2f}")
            print(f"  Difference:     {diff:.2f}")
            print(f"  Hook is neutral: {'YES' if results['hook_neutral'] else 'NO (possible artifact)'}")

        return results

    def run_all_controls(
        self,
        vector: np.ndarray,
        layer_idx: int,
        trait_name: str,
        test_questions: Optional[List[str]] = None,
        target_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run all control experiments and summarize results.

        Args:
            vector: The steering vector to validate
            layer_idx: Layer where vector is applied
            trait_name: Trait being evaluated
            test_questions: Questions to use for testing
            target_score: The score achieved with the target vector

        Returns:
            Comprehensive control results with pass/fail assessment
        """
        print("\n" + "="*70)
        print("CONTROL EXPERIMENTS")
        print("="*70)

        # Run all controls
        random_results = self.run_random_vector_control(
            layer_idx, trait_name, test_questions
        )

        negative_results = self.run_negative_steering_control(
            vector, layer_idx, trait_name, test_questions
        )

        roundtrip_results = self.run_roundtrip_control(
            vector, layer_idx, trait_name, test_questions
        )

        # Summarize
        print("\n" + "="*70)
        print("CONTROL SUMMARY")
        print("="*70)

        all_results = {
            'random': random_results,
            'negative': negative_results,
            'roundtrip': roundtrip_results,
            'assessments': {},
        }

        # Assessment 1: Target effect > random effect
        if target_score and random_results['random_mean']:
            random_gap = target_score - random_results['random_mean']
            all_results['assessments']['random_control_passes'] = random_gap > 0.5
            print(f"\n1. Random vector control:")
            print(f"   Target score: {target_score:.2f}")
            print(f"   Random mean:  {random_results['random_mean']:.2f}")
            print(f"   Gap: {random_gap:.2f} (need >0.5)")
            print(f"   PASS: {'YES' if random_gap > 0.5 else 'NO'}")

        # Assessment 2: Directionality holds
        if negative_results.get('pattern_holds') is not None:
            all_results['assessments']['directionality_passes'] = negative_results['pattern_holds']
            print(f"\n2. Directionality control:")
            print(f"   Pattern (neg < base < pos): {'HOLDS' if negative_results['pattern_holds'] else 'DOES NOT HOLD'}")
            print(f"   PASS: {'YES' if negative_results['pattern_holds'] else 'NO'}")

        # Assessment 3: Hook is neutral
        if roundtrip_results.get('hook_neutral') is not None:
            all_results['assessments']['roundtrip_passes'] = roundtrip_results['hook_neutral']
            print(f"\n3. Roundtrip control:")
            print(f"   Hook is neutral: {'YES' if roundtrip_results['hook_neutral'] else 'NO'}")
            print(f"   PASS: {'YES' if roundtrip_results['hook_neutral'] else 'NO'}")

        # Overall assessment
        passes = [v for v in all_results['assessments'].values() if v is not None]
        if passes:
            all_pass = all(passes)
            all_results['all_controls_pass'] = all_pass
            print(f"\n{'='*70}")
            print(f"OVERALL: {'ALL CONTROLS PASS' if all_pass else 'SOME CONTROLS FAILED'}")
            print(f"{'='*70}")

        return all_results
