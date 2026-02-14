"""
Main orchestration script for persona vector experiments.

This script mirrors the flow of the original notebook:
1. Load model
2. Extract or load vectors
3. Select best layer
4. Test effectiveness
5. (Optional) Analyze dataset

Can be run directly or imported for programmatic use.
"""

import os
import warnings
from typing import Optional, Dict, List, Tuple

import numpy as np

warnings.filterwarnings('ignore')


def run_experiment(
    mode: str = "balanced",
    trait: str = "compassion",
    load_from_hf: bool = False,
    model_id: str = "meta-llama/Llama-3.1-70B-Instruct",
    target_layers: Optional[List[int]] = None,
    run_analysis: bool = False,
    analysis_dataset: Optional[str] = None,
) -> Dict:
    """
    Run a complete persona vector experiment.

    Args:
        mode: Speed/reliability tradeoff ('fast', 'balanced', 'robust')
        trait: Trait to extract ('compassion', 'open_mindedness', 'non_helpfulness')
        load_from_hf: Whether to load pre-computed vectors from HuggingFace
        model_id: HuggingFace model ID
        target_layers: Specific layers to extract from (None = auto)
        run_analysis: Whether to run dataset analysis
        analysis_dataset: Dataset to analyze

    Returns:
        Dictionary with experiment results
    """
    from .config import Config
    from .model import ModelManager
    from .steering import SteeringManager
    from .extraction import VectorExtractor, VectorLoader, VectorUploader
    from .evaluation import LayerSelector, VectorEffectivenessTester
    from .artifacts import get_artifacts

    # ==========================================================================
    # Setup
    # ==========================================================================
    print("="*70)
    print("PERSONA VECTORS EXPERIMENT")
    print("="*70)

    config = Config(
        mode=mode,
        model_id=model_id,
        load_vectors_from_hf=load_from_hf,
    )

    if target_layers:
        config.target_layers = target_layers

    print(f"\nConfiguration:")
    print(f"  Mode: {config.mode}")
    print(f"  Model: {config.model_id}")
    print(f"  Trait: {trait}")
    print(f"  Load from HF: {config.load_vectors_from_hf}")

    results = {
        'config': config,
        'trait': trait,
        'vectors': {},
        'best_layer': None,
        'best_vector': None,
        'effectiveness': None,
        'analysis': None,
    }

    # ==========================================================================
    # Load Model
    # ==========================================================================
    print("\n" + "="*70)
    print("LOADING MODEL")
    print("="*70)

    model_manager = ModelManager(config)
    model_manager.load_model()

    steering_manager = SteeringManager(model_manager, config)

    # ==========================================================================
    # Load or Extract Vectors
    # ==========================================================================
    if config.load_vectors_from_hf:
        print("\n" + "="*70)
        print("LOADING VECTORS FROM HUGGINGFACE")
        print("="*70)

        loader = VectorLoader(config)
        all_vectors = loader.load_all_vectors()

        if trait in all_vectors:
            results['vectors'] = all_vectors[trait]
            results['best_vector'] = all_vectors[trait].get('best')

            # Extract layer name
            for key in all_vectors[trait].keys():
                if key.startswith('layer_'):
                    results['best_layer'] = key
                    break
        else:
            print(f"No vector found for trait: {trait}")
    else:
        print("\n" + "="*70)
        print("EXTRACTING VECTORS")
        print("="*70)

        extractor = VectorExtractor(model_manager, config)
        vectors, layer_stats = extractor.extract_persona_vector(trait)
        results['vectors'] = vectors

        # ==========================================================================
        # Select Best Layer
        # ==========================================================================
        if vectors:
            print("\n" + "="*70)
            print("SELECTING BEST LAYER")
            print("="*70)

            selector = LayerSelector(model_manager, steering_manager, config)
            artifacts = get_artifacts(trait)

            best_layer, best_vector = selector.select_best_layer(
                vectors,
                trait,
                artifacts["test_questions"]
            )

            if best_vector is not None:
                results['best_layer'] = best_layer
                results['best_vector'] = best_vector
                results['vectors']['best'] = best_vector

                # Upload to HuggingFace
                if config.upload_to_hf:
                    uploader = VectorUploader(config)
                    uploader.upload_vector(best_vector, trait, best_layer)

    # ==========================================================================
    # Test Effectiveness
    # ==========================================================================
    if results['best_vector'] is not None and results['best_layer'] is not None:
        print("\n" + "="*70)
        print("TESTING VECTOR EFFECTIVENESS")
        print("="*70)

        layer_idx = int(results['best_layer'].split('_')[1])

        tester = VectorEffectivenessTester(model_manager, steering_manager, config)
        effectiveness = tester.test_vector_effectiveness(
            results['best_vector'],
            layer_idx,
            trait,
        )
        tester.print_results(effectiveness)

        results['effectiveness'] = effectiveness

    # ==========================================================================
    # Dataset Analysis (Optional)
    # ==========================================================================
    if run_analysis and results['best_vector'] is not None:
        print("\n" + "="*70)
        print("DATASET ANALYSIS")
        print("="*70)

        analysis_results = run_dataset_analysis(
            model_manager=model_manager,
            vector=results['best_vector'],
            layer_idx=int(results['best_layer'].split('_')[1]),
            dataset_name=analysis_dataset or config.analysis_dataset,
            sample_size=config.analysis_sample_size,
        )
        results['analysis'] = analysis_results

    # ==========================================================================
    # Cleanup
    # ==========================================================================
    print("\n" + "="*70)
    print("DONE")
    print("="*70)

    model_manager.cleanup()

    return results


def run_dataset_analysis(
    model_manager,
    vector: np.ndarray,
    layer_idx: int,
    dataset_name: str,
    sample_size: int = 500,
) -> Dict:
    """
    Run analysis on a dataset using the persona vector.

    Args:
        model_manager: Loaded ModelManager instance
        vector: Persona vector to use for projections
        layer_idx: Layer index for activation extraction
        dataset_name: HuggingFace dataset name
        sample_size: Number of samples to analyze

    Returns:
        Dictionary with analysis results
    """
    import pandas as pd
    from datasets import load_dataset
    from tqdm import tqdm

    print(f"Loading dataset: {dataset_name}")

    try:
        dataset = load_dataset(dataset_name)
        df = pd.DataFrame(
            dataset['train'] if 'train' in dataset else dataset[list(dataset.keys())[0]]
        )
        print(f"Dataset loaded: {df.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

    # Get sample texts
    if 'output' in df.columns:
        sample_texts = df['output'].dropna().astype(str).tolist()
    elif 'text' in df.columns:
        sample_texts = df['text'].dropna().astype(str).tolist()
    else:
        print(f"No 'output' or 'text' column found. Columns: {df.columns.tolist()}")
        return None

    sample_texts = [t for t in sample_texts if t.strip()][:sample_size]
    print(f"Analyzing {len(sample_texts)} samples...")

    # Calculate projections
    projections = []

    for text in tqdm(sample_texts, desc="Processing texts"):
        try:
            activation = model_manager.extract_activation(text, layer_idx)
            if activation is not None:
                projection = np.dot(activation, vector)
            else:
                projection = 0.0
        except Exception as e:
            projection = 0.0

        projections.append({
            'text': text,
            'projection': projection,
        })

    # Calculate statistics
    scores = [p['projection'] for p in projections]

    stats = {
        'mean': np.mean(scores),
        'std': np.std(scores),
        'min': np.min(scores),
        'max': np.max(scores),
        'non_zero': sum(1 for s in scores if s != 0),
        'total': len(scores),
    }

    print(f"\nProjection Statistics:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std: {stats['std']:.3f}")
    print(f"  Min: {stats['min']:.3f}")
    print(f"  Max: {stats['max']:.3f}")
    print(f"  Non-zero: {stats['non_zero']}/{stats['total']}")

    # Find best and worst examples
    sorted_projections = sorted(projections, key=lambda x: x['projection'])

    print(f"\nWorst example (lowest projection):")
    print(f"  Score: {sorted_projections[0]['projection']:.3f}")
    print(f"  Text: {sorted_projections[0]['text'][:200]}...")

    print(f"\nBest example (highest projection):")
    print(f"  Score: {sorted_projections[-1]['projection']:.3f}")
    print(f"  Text: {sorted_projections[-1]['text'][:200]}...")

    return {
        'projections': projections,
        'stats': stats,
        'worst': sorted_projections[0],
        'best': sorted_projections[-1],
    }


def create_visualization(analysis_results: Dict, output_path: Optional[str] = None):
    """Create visualization of analysis results."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    projections = [p['projection'] for p in analysis_results['projections']]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')

    sns.histplot(projections, ax=ax, color='blue', kde=True, stat='density')
    ax.set_title('Persona Vector Projections')
    ax.set_xlabel('Projection Score')
    ax.set_ylabel('Density')

    # Add statistics annotation
    stats = analysis_results['stats']
    stats_text = f"Mean: {stats['mean']:.3f}\nStd: {stats['std']:.3f}"
    ax.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Default experiment: extract compassion vector in balanced mode
    results = run_experiment(
        mode="balanced",
        trait="compassion",
        load_from_hf=False,
        run_analysis=False,
    )

    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)

    if results['best_layer']:
        print(f"Best layer: {results['best_layer']}")

    if results['effectiveness']:
        eff = results['effectiveness']
        if eff['improvement']:
            print(f"Improvement: {eff['improvement']:.2f} points")
