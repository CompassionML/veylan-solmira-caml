"""
Evaluation functionality using external LLM judges (Gemini).
"""

import os
import re
import time
import random
import asyncio
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .config import Config
from .model import ModelManager
from .steering import SteeringManager
from .artifacts import TRAIT_DESCRIPTIONS, get_artifacts


class GeminiJudge:
    """Uses Gemini to judge trait expression in responses."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini judge."""
        self.api_key = api_key or self._get_api_key()
        self.model = None
        self._configure()

    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment or Colab secrets."""
        if 'GOOGLE_API_KEY' in os.environ:
            return os.environ['GOOGLE_API_KEY']

        try:
            from google.colab import userdata
            return userdata.get('GOOGLE_API_KEY')
        except:
            pass

        return None

    def _configure(self) -> None:
        """Configure Gemini API."""
        if not self.api_key:
            print("GOOGLE_API_KEY not found")
            return

        try:
            import google.generativeai as genai
            genai.configure(
                api_key=self.api_key,
                transport='rest',
                client_options={'api_endpoint': 'https://generativelanguage.googleapis.com/'}
            )
            self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
            print("Gemini configured successfully")
        except Exception as e:
            print(f"Error configuring Gemini: {e}")

    def judge_response(
        self,
        response: str,
        question: str,
        trait_name: str,
        description: Optional[str] = None,
    ) -> Optional[int]:
        """Judge a single response for trait expression."""
        if self.model is None:
            return None

        import google.generativeai as genai

        description = description or TRAIT_DESCRIPTIONS.get(trait_name, trait_name)

        judge_prompt = f"""Rate this response for {trait_name}.

Definition: {description}

Question: {question}

Response: {response}

Rate 1-5 (1=very little, 5=very strong). Reply with ONLY digit first."""

        config = genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=150,
            top_p=1.0,
            top_k=1
        )

        result = self._call_with_retry(judge_prompt, config)
        return self._parse_score(result)

    def judge_batch(
        self,
        responses: List[str],
        questions: List[str],
        trait_name: str,
        max_workers: int = 5,
    ) -> List[Optional[int]]:
        """Judge multiple responses in parallel."""
        if self.model is None:
            return [None] * len(responses)

        import google.generativeai as genai

        description = TRAIT_DESCRIPTIONS.get(trait_name, trait_name)

        prompts = [
            f"Rate for {trait_name}.\n\nDefinition: {description}\n\nQuestion: {q}\n\nResponse: {r}\n\nRate 1-5 (1=very little, 5=very strong). Reply with ONLY digit first."
            for q, r in zip(questions, responses)
        ]

        config = genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=150,
            top_p=1.0,
            top_k=1
        )

        results = [None] * len(prompts)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._call_with_retry, p, config): i
                for i, p in enumerate(prompts)
            }
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results[idx] = self._parse_score(result)

        return results

    def _call_with_retry(
        self,
        prompt: str,
        config: Any,
        max_retries: int = 3,
    ) -> Optional[str]:
        """Call Gemini API with retry logic."""
        from google.api_core import exceptions

        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt, generation_config=config)
                return response.text.strip() if response and response.text else None
            except exceptions.ResourceExhausted:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    time.sleep(wait_time)
                else:
                    return None
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    return None

        return None

    def _parse_score(self, result: Optional[str]) -> Optional[int]:
        """Parse a score from judge response."""
        if not result:
            return None

        try:
            first_char = result.strip()[0]
            if first_char.isdigit() and 1 <= int(first_char) <= 5:
                return int(first_char)

            # Try regex
            numbers = re.findall(r'\b([1-5])\b', result[:20])
            if numbers:
                return int(numbers[0])

        except:
            pass

        return None


class LayerSelector:
    """Selects the best layer for steering using evaluation."""

    def __init__(
        self,
        model_manager: ModelManager,
        steering_manager: SteeringManager,
        config: Optional[Config] = None,
    ):
        """Initialize layer selector."""
        self.model_manager = model_manager
        self.steering_manager = steering_manager
        self.config = config or model_manager.config
        self.judge = GeminiJudge()

    def select_best_layer(
        self,
        layer_vectors: Dict[str, np.ndarray],
        trait_name: str,
        test_questions: Optional[List[str]] = None,
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Select the best layer for a trait using evaluation."""
        if self.judge.model is None:
            print("Gemini not configured")
            return None, None

        if test_questions is None:
            artifacts = get_artifacts(trait_name)
            test_questions = artifacts["test_questions"]

        description = TRAIT_DESCRIPTIONS.get(trait_name, trait_name)

        print(f"\n{'='*70}")
        print(f"SELECTING {trait_name.upper()} LAYER (MODE={self.config.mode})")
        print(f"{'='*70}")

        if self.config.use_two_stage:
            return self._two_stage_selection(
                layer_vectors, trait_name, test_questions, description
            )
        else:
            return self._single_stage_selection(
                layer_vectors, trait_name, test_questions, description
            )

    def _single_stage_selection(
        self,
        layer_vectors: Dict[str, np.ndarray],
        trait_name: str,
        test_questions: List[str],
        description: str,
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Single-stage layer selection."""
        num_q = self.config.num_test_questions
        test_q = test_questions[:num_q]

        print(f"\nTesting {num_q} questions per layer")

        # Generate baselines
        prompts = [self.model_manager.format_evaluation_prompt(q) for q in test_q]
        baselines = self.model_manager.generate_batch(prompts)

        results = {}

        for layer_name, vector in layer_vectors.items():
            if layer_name == 'best':
                continue

            layer_idx = int(layer_name.split('_')[1])
            print(f"  {layer_name}...", end=" ")

            eval_result = self._evaluate_layer(
                layer_name, vector, layer_idx,
                test_q, trait_name, description, baselines
            )

            if eval_result and self._passes_thresholds(eval_result):
                results[layer_name] = eval_result
                print(f"{eval_result['avg']:.3f}")
            else:
                print("skip")

        if not results:
            print("No layers passed")
            return None, None

        best_layer = max(results.items(), key=lambda x: x[1]['avg'])[0]
        best_vec = layer_vectors[best_layer]

        print(f"\nSelected: {best_layer} ({results[best_layer]['avg']:.3f})")

        return best_layer, best_vec

    def _two_stage_selection(
        self,
        layer_vectors: Dict[str, np.ndarray],
        trait_name: str,
        test_questions: List[str],
        description: str,
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """Two-stage layer selection for balanced mode."""
        # Stage 1: Quick screen
        print(f"\nStage 1: Screening with {self.config.num_stage1_questions} questions")
        stage1_q = test_questions[:self.config.num_stage1_questions]
        prompts = [self.model_manager.format_evaluation_prompt(q) for q in stage1_q]
        baselines = self.model_manager.generate_batch(prompts)

        stage1_results = {}

        for layer_name, vector in layer_vectors.items():
            if layer_name == 'best':
                continue

            layer_idx = int(layer_name.split('_')[1])
            print(f"  {layer_name}...", end=" ")

            eval_result = self._evaluate_layer(
                layer_name, vector, layer_idx,
                stage1_q, trait_name, description, baselines
            )

            if eval_result and eval_result['diff_rate'] >= self.config.min_difference_rate:
                if eval_result['valid_rate'] >= self.config.min_valid_scores_rate:
                    stage1_results[layer_name] = eval_result
                    print(f"{eval_result['avg']:.2f}")
                else:
                    print("skip (valid rate)")
            else:
                print("skip")

        if not stage1_results:
            print("No layers passed stage 1")
            return None, None

        # Sort and get top 3
        sorted_results = sorted(
            stage1_results.items(),
            key=lambda x: x[1]['avg'],
            reverse=True
        )
        top3 = sorted_results[:3]

        print(f"\nTop 3: {', '.join([f'{ln}({r['avg']:.2f})' for ln, r in top3])}")

        # Early stop check
        if len(top3) >= 2:
            gap = top3[0][1]['avg'] - top3[1][1]['avg']
            if gap >= self.config.early_stop_threshold:
                print("Early stop")
                return top3[0][0], layer_vectors[top3[0][0]]

        # Stage 2: Validate top 3
        print(f"\nStage 2: Validating with {self.config.num_stage2_questions} questions")
        stage2_q = test_questions[:self.config.num_stage2_questions]
        prompts = [self.model_manager.format_evaluation_prompt(q) for q in stage2_q]
        baselines = self.model_manager.generate_batch(prompts)

        stage2_results = {}

        for layer_name, _ in top3:
            vector = layer_vectors[layer_name]
            layer_idx = int(layer_name.split('_')[1])
            print(f"  {layer_name}...", end=" ")

            eval_result = self._evaluate_layer(
                layer_name, vector, layer_idx,
                stage2_q, trait_name, description, baselines
            )

            if eval_result and eval_result['avg'] >= self.config.min_average_score:
                stage2_results[layer_name] = eval_result
                print(f"{eval_result['avg']:.3f}")
            else:
                print("skip")

        if not stage2_results:
            print("No layers passed stage 2")
            return None, None

        best_layer = max(stage2_results.items(), key=lambda x: x[1]['avg'])[0]
        best_vec = layer_vectors[best_layer]

        print(f"\nSelected: {best_layer} ({stage2_results[best_layer]['avg']:.3f})")

        return best_layer, best_vec

    def _evaluate_layer(
        self,
        layer_name: str,
        vector: np.ndarray,
        layer_idx: int,
        questions: List[str],
        trait_name: str,
        description: str,
        baselines: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a single layer."""
        prompts = [self.model_manager.format_evaluation_prompt(q) for q in questions]
        steered = self.steering_manager.generate_steered_batch(prompts, vector, layer_idx)

        valid_idx = [i for i, r in enumerate(steered) if r]
        if not valid_idx:
            return None

        # Calculate difference rate
        diff_count = sum(
            1 for i in valid_idx
            if steered[i].strip() != baselines[i].strip()
        )
        diff_rate = diff_count / len(valid_idx)

        # Get scores from judge
        valid_questions = [questions[i] for i in valid_idx]
        valid_responses = [steered[i] for i in valid_idx]

        scores = self.judge.judge_batch(valid_responses, valid_questions, trait_name)
        valid_scores = [s for s in scores if s is not None]

        if not valid_scores:
            return None

        return {
            'avg': np.mean(valid_scores),
            'std': np.std(valid_scores),
            'diff_rate': diff_rate,
            'valid_rate': len(valid_scores) / len(questions),
            'scores': valid_scores,
        }

    def _passes_thresholds(self, result: Dict[str, Any]) -> bool:
        """Check if evaluation result passes all thresholds."""
        return (
            result['diff_rate'] >= self.config.min_difference_rate and
            result['valid_rate'] >= self.config.min_valid_scores_rate and
            result['avg'] >= self.config.min_average_score
        )


class VectorEffectivenessTester:
    """Tests the effectiveness of steering vectors."""

    def __init__(
        self,
        model_manager: ModelManager,
        steering_manager: SteeringManager,
        config: Optional[Config] = None,
    ):
        """Initialize tester."""
        self.model_manager = model_manager
        self.steering_manager = steering_manager
        self.config = config or model_manager.config
        self.judge = GeminiJudge()

    def test_vector_effectiveness(
        self,
        vector: np.ndarray,
        layer_idx: int,
        trait_name: str,
        test_questions: Optional[List[str]] = None,
        coefficient: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Test vector effectiveness by comparing baseline vs steered."""
        if test_questions is None:
            artifacts = get_artifacts(trait_name)
            test_questions = artifacts["test_questions"][:3]

        coefficient = coefficient or self.config.steering_coefficient

        baseline_scores = []
        steered_scores = []
        comparisons = []

        for q in test_questions:
            prompt = self.model_manager.format_evaluation_prompt(q)

            # Generate both
            baseline = self.model_manager.generate_response(prompt)
            steered = self.steering_manager.apply_steering_and_generate(
                prompt, vector, layer_idx, coefficient
            )

            # Judge both
            baseline_score = self.judge.judge_response(baseline, q, trait_name)
            steered_score = self.judge.judge_response(steered, q, trait_name)

            if baseline_score is not None:
                baseline_scores.append(baseline_score)
            if steered_score is not None:
                steered_scores.append(steered_score)

            comparisons.append({
                'question': q,
                'baseline_response': baseline,
                'steered_response': steered,
                'baseline_score': baseline_score,
                'steered_score': steered_score,
            })

        results = {
            'baseline_mean': np.mean(baseline_scores) if baseline_scores else None,
            'steered_mean': np.mean(steered_scores) if steered_scores else None,
            'improvement': None,
            'comparisons': comparisons,
        }

        if results['baseline_mean'] and results['steered_mean']:
            results['improvement'] = results['steered_mean'] - results['baseline_mean']

        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print test results."""
        print(f"\n=== VECTOR EFFECTIVENESS TEST ===")
        print(f"Baseline average: {results['baseline_mean']:.2f}" if results['baseline_mean'] else "Baseline: N/A")
        print(f"Steered average: {results['steered_mean']:.2f}" if results['steered_mean'] else "Steered: N/A")
        print(f"Improvement: {results['improvement']:.2f}" if results['improvement'] else "Improvement: N/A")

        print(f"\nIndividual scores:")
        for comp in results['comparisons']:
            print(f"  Q: {comp['question'][:50]}...")
            print(f"     Baseline: {comp['baseline_score']}, Steered: {comp['steered_score']}")


class ScaleSweepEvaluator:
    """Evaluates steering across different coefficient scales to find optimal."""

    def __init__(
        self,
        model_manager: ModelManager,
        steering_manager: SteeringManager,
        config: Optional[Config] = None,
    ):
        """Initialize scale sweep evaluator."""
        self.model_manager = model_manager
        self.steering_manager = steering_manager
        self.config = config or model_manager.config
        self.judge = GeminiJudge()

    def run_scale_sweep(
        self,
        vector: np.ndarray,
        layer_idx: int,
        trait_name: str,
        test_questions: Optional[List[str]] = None,
        scales: Optional[List[float]] = None,
        num_questions: int = 10,
    ) -> Dict[str, Any]:
        """
        Run a comprehensive scale sweep to find optimal steering coefficient.

        This addresses the remediation requirement to test multiple scales
        rather than using a fixed coefficient.

        Args:
            vector: The steering vector
            layer_idx: Layer to apply steering
            trait_name: Trait being evaluated
            test_questions: Questions to test (default: from artifacts)
            scales: Coefficients to test (default: from config)
            num_questions: Number of questions to use

        Returns:
            Dict with scale->results mapping and optimal scale analysis
        """
        if test_questions is None:
            artifacts = get_artifacts(trait_name)
            test_questions = artifacts["test_questions"][:num_questions]
        else:
            test_questions = test_questions[:num_questions]

        if scales is None:
            scales = self.config.scale_sweep_values

        print(f"\n{'='*70}")
        print(f"SCALE SWEEP EVALUATION")
        print(f"{'='*70}")
        print(f"Testing {len(scales)} scales: {scales}")
        print(f"Using {len(test_questions)} test questions")

        results = {
            'scales': {},
            'optimal_scale': None,
            'optimal_score': None,
            'baseline_mean': None,
            'scale_effect_curve': [],
        }

        # First, get baseline scores
        print(f"\nGenerating baseline responses...")
        baseline_scores = []
        for q in test_questions:
            prompt = self.model_manager.format_evaluation_prompt(q)
            response = self.model_manager.generate_response(prompt)
            if response:
                score = self.judge.judge_response(response, q, trait_name)
                if score is not None:
                    baseline_scores.append(score)

        results['baseline_mean'] = np.mean(baseline_scores) if baseline_scores else None
        print(f"Baseline mean: {results['baseline_mean']:.2f}" if results['baseline_mean'] else "Baseline: N/A")

        # Test each scale
        for scale in scales:
            print(f"\nScale {scale}...", end=" ")

            if scale == 0.0:
                # Use baseline scores
                results['scales'][scale] = {
                    'mean': results['baseline_mean'],
                    'std': np.std(baseline_scores) if baseline_scores else None,
                    'scores': baseline_scores,
                    'n': len(baseline_scores),
                }
                print(f"mean={results['baseline_mean']:.2f}" if results['baseline_mean'] else "N/A")
                continue

            scale_scores = []
            for q in test_questions:
                prompt = self.model_manager.format_evaluation_prompt(q)
                response = self.steering_manager.apply_steering_and_generate(
                    prompt, vector, layer_idx, coefficient=scale
                )
                if response:
                    score = self.judge.judge_response(response, q, trait_name)
                    if score is not None:
                        scale_scores.append(score)

            if scale_scores:
                mean_score = np.mean(scale_scores)
                results['scales'][scale] = {
                    'mean': mean_score,
                    'std': np.std(scale_scores),
                    'scores': scale_scores,
                    'n': len(scale_scores),
                }
                results['scale_effect_curve'].append((scale, mean_score))
                print(f"mean={mean_score:.2f} (n={len(scale_scores)})")

                # Track optimal
                if results['optimal_score'] is None or mean_score > results['optimal_score']:
                    results['optimal_scale'] = scale
                    results['optimal_score'] = mean_score
            else:
                results['scales'][scale] = None
                print("N/A (no valid scores)")

        # Analysis
        self._analyze_scale_curve(results)

        return results

    def _analyze_scale_curve(self, results: Dict[str, Any]) -> None:
        """Analyze the scale-effect curve for patterns."""
        curve = results['scale_effect_curve']
        if len(curve) < 3:
            results['curve_analysis'] = "Insufficient data points"
            return

        scales = [x[0] for x in curve]
        scores = [x[1] for x in curve]

        # Check for monotonic increase (diminishing returns)
        diffs = [scores[i+1] - scores[i] for i in range(len(scores)-1)]
        increasing = sum(1 for d in diffs if d > 0)
        decreasing = sum(1 for d in diffs if d < 0)

        if increasing > decreasing * 2:
            results['curve_pattern'] = 'generally_increasing'
        elif decreasing > increasing * 2:
            results['curve_pattern'] = 'generally_decreasing'
        else:
            results['curve_pattern'] = 'mixed'

        # Find saturation point (where improvement < 0.1)
        for i, d in enumerate(diffs):
            if d < 0.1 and i > 0:
                results['saturation_scale'] = scales[i]
                break

        # Check for degradation (score dropping significantly)
        max_score = max(scores)
        max_idx = scores.index(max_score)
        if max_idx < len(scores) - 1:
            final_score = scores[-1]
            if max_score - final_score > 0.5:
                results['degradation_detected'] = True
                results['degradation_scale'] = scales[max_idx + 1]

    def print_scale_sweep_report(self, results: Dict[str, Any]) -> None:
        """Print a formatted scale sweep report."""
        print(f"\n{'='*70}")
        print("SCALE SWEEP REPORT")
        print(f"{'='*70}")

        print(f"\nBaseline: {results['baseline_mean']:.2f}" if results['baseline_mean'] else "Baseline: N/A")
        print(f"Optimal scale: {results['optimal_scale']} (score={results['optimal_score']:.2f})" if results['optimal_scale'] else "No optimal scale found")

        print(f"\nScale -> Score:")
        for scale, data in sorted(results['scales'].items()):
            if data:
                improvement = data['mean'] - results['baseline_mean'] if results['baseline_mean'] else 0
                print(f"  {scale:5.1f} -> {data['mean']:.2f} (+{improvement:.2f})")
            else:
                print(f"  {scale:5.1f} -> N/A")

        if results.get('curve_pattern'):
            print(f"\nCurve pattern: {results['curve_pattern']}")
        if results.get('saturation_scale'):
            print(f"Saturation at scale: {results['saturation_scale']}")
        if results.get('degradation_detected'):
            print(f"WARNING: Degradation detected at scale {results['degradation_scale']}")

        print(f"{'='*70}")


class BaselineBehaviorChecker:
    """
    Checks baseline model behavior to ensure test questions are appropriate.

    If the baseline model already strongly exhibits the target trait,
    the test questions may not be challenging enough to measure steering effects.
    """

    def __init__(
        self,
        model_manager: ModelManager,
        config: Optional[Config] = None,
    ):
        """Initialize baseline checker."""
        self.model_manager = model_manager
        self.config = config or model_manager.config
        self.judge = GeminiJudge()

    def check_baseline_behavior(
        self,
        trait_name: str,
        test_questions: Optional[List[str]] = None,
        num_questions: int = 20,
        high_trait_threshold: float = 4.0,
    ) -> Dict[str, Any]:
        """
        Check if the baseline model already exhibits the target trait.

        This addresses the remediation requirement to verify that test questions
        are actually challenging (i.e., baseline doesn't already show the trait).

        Args:
            trait_name: Trait to check for
            test_questions: Questions to test (default: from artifacts)
            num_questions: Number of questions to test
            high_trait_threshold: Score threshold for "high trait" (default: 4.0 out of 5.0)

        Returns:
            Dict with baseline behavior analysis
        """
        if test_questions is None:
            artifacts = get_artifacts(trait_name)
            test_questions = artifacts["test_questions"][:num_questions]
        else:
            test_questions = test_questions[:num_questions]

        print(f"\n{'='*70}")
        print(f"BASELINE BEHAVIOR CHECK: {trait_name.upper()}")
        print(f"{'='*70}")
        print(f"Testing {len(test_questions)} questions...")

        scores = []
        high_trait_count = 0
        low_trait_count = 0
        question_results = []

        for q in test_questions:
            prompt = self.model_manager.format_evaluation_prompt(q)
            response = self.model_manager.generate_response(prompt)

            if response:
                score = self.judge.judge_response(response, q, trait_name)
                if score is not None:
                    scores.append(score)

                    if score >= high_trait_threshold:
                        high_trait_count += 1
                    elif score <= 2.0:
                        low_trait_count += 1

                    question_results.append({
                        'question': q,
                        'response': response[:200] + "..." if len(response) > 200 else response,
                        'score': score,
                    })

        results = {
            'trait_name': trait_name,
            'num_questions': len(test_questions),
            'num_scored': len(scores),
            'scores': scores,
            'mean_score': np.mean(scores) if scores else None,
            'std_score': np.std(scores) if scores else None,
            'high_trait_count': high_trait_count,
            'low_trait_count': low_trait_count,
            'high_trait_rate': high_trait_count / len(scores) if scores else None,
            'low_trait_rate': low_trait_count / len(scores) if scores else None,
            'question_results': question_results,
            'warnings': [],
        }

        # Analyze results
        self._analyze_baseline(results, high_trait_threshold)

        return results

    def _analyze_baseline(
        self,
        results: Dict[str, Any],
        high_trait_threshold: float,
    ) -> None:
        """Analyze baseline results and add warnings."""
        if results['mean_score'] is None:
            results['valid'] = False
            results['warnings'].append("No valid scores obtained")
            return

        # Warning: Baseline already high
        if results['mean_score'] >= high_trait_threshold:
            results['warnings'].append(
                f"CRITICAL: Baseline mean ({results['mean_score']:.2f}) >= threshold ({high_trait_threshold}). "
                f"Model already exhibits {results['trait_name']}. Test questions may be too easy."
            )

        # Warning: High trait rate too high
        if results['high_trait_rate'] and results['high_trait_rate'] > 0.5:
            results['warnings'].append(
                f"WARNING: {results['high_trait_rate']*100:.0f}% of baseline responses score >= {high_trait_threshold}. "
                f"Limited room for improvement from steering."
            )

        # Warning: Very low variance
        if results['std_score'] and results['std_score'] < 0.5:
            results['warnings'].append(
                f"WARNING: Low variance (std={results['std_score']:.2f}). "
                f"Responses may be too homogeneous."
            )

        # Assessment
        if results['mean_score'] < 3.0 and results['high_trait_rate'] < 0.3:
            results['assessment'] = 'GOOD'
            results['assessment_detail'] = (
                "Test questions appear appropriately challenging. "
                "Baseline shows low trait expression, leaving room for steering improvement."
            )
        elif results['mean_score'] < 3.5 and results['high_trait_rate'] < 0.5:
            results['assessment'] = 'ACCEPTABLE'
            results['assessment_detail'] = (
                "Test questions are moderately challenging. "
                "Some baseline trait expression, but room for improvement exists."
            )
        else:
            results['assessment'] = 'PROBLEMATIC'
            results['assessment_detail'] = (
                "Test questions may be too easy. "
                "Consider using more challenging questions that don't naturally elicit the trait."
            )

        results['valid'] = results['assessment'] != 'PROBLEMATIC'

    def print_baseline_report(self, results: Dict[str, Any]) -> None:
        """Print baseline behavior report."""
        print(f"\n{'='*70}")
        print(f"BASELINE BEHAVIOR REPORT: {results['trait_name'].upper()}")
        print(f"{'='*70}")

        print(f"\nSample size: {results['num_scored']} / {results['num_questions']} questions")
        print(f"Mean score: {results['mean_score']:.2f}" if results['mean_score'] else "Mean: N/A")
        print(f"Std score: {results['std_score']:.2f}" if results['std_score'] else "Std: N/A")

        print(f"\nTrait Expression:")
        print(f"  High trait (>=4): {results['high_trait_count']} ({results['high_trait_rate']*100:.0f}%)" if results['high_trait_rate'] else "  High trait: N/A")
        print(f"  Low trait (<=2): {results['low_trait_count']} ({results['low_trait_rate']*100:.0f}%)" if results['low_trait_rate'] else "  Low trait: N/A")

        if results['warnings']:
            print(f"\nWarnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")

        print(f"\nAssessment: {results.get('assessment', 'N/A')}")
        if results.get('assessment_detail'):
            print(f"  {results['assessment_detail']}")

        print(f"{'='*70}")

    def get_challenging_questions(
        self,
        results: Dict[str, Any],
        max_baseline_score: float = 3.0,
    ) -> List[str]:
        """
        Extract questions where baseline showed low trait expression.

        These are the "challenging" questions that are most useful for
        evaluating steering effects.

        Args:
            results: Results from check_baseline_behavior
            max_baseline_score: Maximum baseline score to consider "challenging"

        Returns:
            List of questions with low baseline trait expression
        """
        challenging = []
        for qr in results['question_results']:
            if qr['score'] is not None and qr['score'] <= max_baseline_score:
                challenging.append(qr['question'])

        print(f"Found {len(challenging)} challenging questions (baseline score <= {max_baseline_score})")
        return challenging
