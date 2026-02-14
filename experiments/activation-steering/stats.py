"""
Statistical analysis for evaluating steering effectiveness.

Provides proper statistical testing to determine if observed
effects are significant or could be due to chance.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


def compute_statistics(
    baseline_scores: List[float],
    steered_scores: List[float],
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics comparing baseline vs steered scores.

    Args:
        baseline_scores: Scores without steering
        steered_scores: Scores with steering

    Returns:
        Dictionary with statistical measures
    """
    from scipy import stats as scipy_stats

    baseline = np.array(baseline_scores)
    steered = np.array(steered_scores)

    # Basic descriptive stats
    results = {
        'n': len(baseline_scores),
        'baseline_mean': np.mean(baseline),
        'baseline_std': np.std(baseline),
        'steered_mean': np.mean(steered),
        'steered_std': np.std(steered),
        'mean_difference': np.mean(steered) - np.mean(baseline),
    }

    # Paired t-test (appropriate when same questions used)
    if len(baseline) == len(steered) and len(baseline) >= 2:
        t_stat, p_value = scipy_stats.ttest_rel(steered, baseline)
        results['t_statistic'] = t_stat
        results['p_value'] = p_value
        results['significant_p05'] = p_value < 0.05
        results['significant_p01'] = p_value < 0.01

        # Effect size (Cohen's d for paired samples)
        diff = steered - baseline
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        results['cohens_d'] = cohens_d

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            results['effect_size_interpretation'] = 'negligible'
        elif abs(cohens_d) < 0.5:
            results['effect_size_interpretation'] = 'small'
        elif abs(cohens_d) < 0.8:
            results['effect_size_interpretation'] = 'medium'
        else:
            results['effect_size_interpretation'] = 'large'

        # 95% confidence interval for the difference
        se = scipy_stats.sem(diff)
        ci = scipy_stats.t.interval(
            0.95,
            len(diff) - 1,
            loc=np.mean(diff),
            scale=se
        )
        results['ci_95_low'] = ci[0]
        results['ci_95_high'] = ci[1]

    return results


def compute_power_analysis(
    effect_size: float,
    n: int,
    alpha: float = 0.05,
) -> float:
    """
    Compute statistical power for detecting a given effect size.

    Args:
        effect_size: Expected Cohen's d
        n: Sample size
        alpha: Significance level

    Returns:
        Statistical power (probability of detecting the effect)
    """
    from scipy import stats as scipy_stats

    # For paired t-test
    t_crit = scipy_stats.t.ppf(1 - alpha / 2, n - 1)
    ncp = effect_size * np.sqrt(n)  # Non-centrality parameter
    power = 1 - scipy_stats.nct.cdf(t_crit, n - 1, ncp) + scipy_stats.nct.cdf(-t_crit, n - 1, ncp)

    return power


def required_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """
    Calculate required sample size to detect an effect.

    Args:
        effect_size: Expected Cohen's d
        power: Desired statistical power (default: 0.8)
        alpha: Significance level (default: 0.05)

    Returns:
        Required sample size
    """
    from scipy import stats as scipy_stats

    # Binary search for required n
    for n in range(2, 1000):
        if compute_power_analysis(effect_size, n, alpha) >= power:
            return n

    return 1000  # Maximum


def print_statistics_report(stats: Dict[str, Any]) -> None:
    """Print a formatted statistics report."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS")
    print("="*60)

    print(f"\nSample size: n = {stats['n']}")

    print(f"\nDescriptive Statistics:")
    print(f"  Baseline: {stats['baseline_mean']:.3f} (SD = {stats['baseline_std']:.3f})")
    print(f"  Steered:  {stats['steered_mean']:.3f} (SD = {stats['steered_std']:.3f})")
    print(f"  Difference: {stats['mean_difference']:.3f}")

    if 'p_value' in stats:
        print(f"\nInferential Statistics:")
        print(f"  t-statistic: {stats['t_statistic']:.3f}")
        print(f"  p-value: {stats['p_value']:.4f}")
        print(f"  95% CI: [{stats['ci_95_low']:.3f}, {stats['ci_95_high']:.3f}]")

        print(f"\nEffect Size:")
        print(f"  Cohen's d: {stats['cohens_d']:.3f} ({stats['effect_size_interpretation']})")

        print(f"\nSignificance:")
        print(f"  p < 0.05: {'YES' if stats['significant_p05'] else 'NO'}")
        print(f"  p < 0.01: {'YES' if stats['significant_p01'] else 'NO'}")

        # Power analysis
        if stats['n'] > 0 and stats['cohens_d'] != 0:
            power = compute_power_analysis(abs(stats['cohens_d']), stats['n'])
            print(f"\nPower Analysis:")
            print(f"  Power at current n={stats['n']}: {power:.2%}")

            if power < 0.8:
                needed_n = required_sample_size(abs(stats['cohens_d']))
                print(f"  Sample size needed for 80% power: {needed_n}")

    print("="*60)


class StatisticalValidator:
    """Validates steering results with proper statistical testing."""

    def __init__(self, min_n: int = 30, alpha: float = 0.05):
        """
        Initialize validator.

        Args:
            min_n: Minimum sample size for valid inference
            alpha: Significance level
        """
        self.min_n = min_n
        self.alpha = alpha

    def validate_steering_effect(
        self,
        baseline_scores: List[float],
        steered_scores: List[float],
        random_scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of steering effect.

        Args:
            baseline_scores: Scores without steering
            steered_scores: Scores with target vector steering
            random_scores: Optional scores with random vector steering

        Returns:
            Validation results with pass/fail assessments
        """
        results = {
            'valid': False,
            'issues': [],
            'stats': None,
        }

        # Check sample size
        n = len(baseline_scores)
        if n < self.min_n:
            results['issues'].append(f"Sample size {n} < minimum {self.min_n}")

        if len(baseline_scores) != len(steered_scores):
            results['issues'].append("Baseline and steered sample sizes don't match")
            return results

        # Compute statistics
        stats = compute_statistics(baseline_scores, steered_scores)
        results['stats'] = stats

        # Check significance
        if stats.get('p_value') is not None:
            if stats['p_value'] >= self.alpha:
                results['issues'].append(
                    f"Not significant (p={stats['p_value']:.4f} >= {self.alpha})"
                )

            # Check effect size
            if abs(stats.get('cohens_d', 0)) < 0.2:
                results['issues'].append(
                    f"Negligible effect size (d={stats['cohens_d']:.3f})"
                )

            # Check confidence interval
            if stats.get('ci_95_low', 0) <= 0 <= stats.get('ci_95_high', 0):
                results['issues'].append("95% CI includes zero")

        # Compare against random if provided
        if random_scores:
            random_mean = np.mean(random_scores)
            steered_mean = stats['steered_mean']
            random_gap = steered_mean - random_mean

            results['random_comparison'] = {
                'random_mean': random_mean,
                'steered_mean': steered_mean,
                'gap': random_gap,
            }

            if random_gap < 0.3:
                results['issues'].append(
                    f"Steered not much better than random (gap={random_gap:.2f})"
                )

        # Final verdict
        results['valid'] = len(results['issues']) == 0

        return results

    def print_validation_report(self, results: Dict[str, Any]) -> None:
        """Print validation report."""
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)

        if results['stats']:
            print_statistics_report(results['stats'])

        if results.get('random_comparison'):
            rc = results['random_comparison']
            print(f"\nRandom Vector Comparison:")
            print(f"  Random mean: {rc['random_mean']:.3f}")
            print(f"  Steered mean: {rc['steered_mean']:.3f}")
            print(f"  Gap: {rc['gap']:.3f}")

        print(f"\n{'='*60}")
        if results['valid']:
            print("VALIDATION: PASSED")
        else:
            print("VALIDATION: FAILED")
            print("\nIssues:")
            for issue in results['issues']:
                print(f"  - {issue}")
        print("="*60)
