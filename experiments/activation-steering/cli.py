"""
Command-line interface for persona vectors.
"""

import argparse
from typing import Optional


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Persona Vectors: Activation Steering for Trait Expression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract compassion vector and test it
  python -m persona_vectors --trait compassion --mode balanced

  # Load pre-computed vector from HuggingFace
  python -m persona_vectors --load-from-hf --trait compassion

  # Extract vectors for all traits
  python -m persona_vectors --all-traits --mode robust

  # Run analysis on a dataset
  python -m persona_vectors --analyze --dataset your-org/dataset-name
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "robust"],
        default="balanced",
        help="Speed vs reliability tradeoff (default: balanced)"
    )

    # Trait selection
    parser.add_argument(
        "--trait",
        choices=["compassion", "open_mindedness", "non_helpfulness"],
        default="compassion",
        help="Trait to extract/test (default: compassion)"
    )

    parser.add_argument(
        "--all-traits",
        action="store_true",
        help="Extract vectors for all traits"
    )

    # Model settings
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-70B-Instruct",
        help="Model ID to use"
    )

    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable 4-bit quantization"
    )

    # Vector loading/saving
    parser.add_argument(
        "--load-from-hf",
        action="store_true",
        help="Load pre-computed vectors from HuggingFace"
    )

    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Don't upload vectors to HuggingFace"
    )

    parser.add_argument(
        "--vector-path",
        type=str,
        help="Path to load/save vectors locally"
    )

    # Layer selection
    parser.add_argument(
        "--layers",
        type=str,
        help="Comma-separated layer indices or range (e.g., '8,9,10' or '8-15')"
    )

    # Steering settings
    parser.add_argument(
        "--coefficient",
        type=float,
        default=2.0,
        help="Steering coefficient (default: 2.0)"
    )

    # Analysis
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run dataset analysis after extraction"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="HuggingFace dataset to analyze"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of samples for analysis (default: 500)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory for output files"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )

    # Actions
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test existing vector, don't extract"
    )

    parser.add_argument(
        "--scale-sweep",
        action="store_true",
        help="Run scale sweep experiment"
    )

    return parser.parse_args()


def parse_layers(layer_str: str) -> list:
    """Parse layer string into list of indices."""
    if not layer_str:
        return None

    layers = []
    for part in layer_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))

    return sorted(set(layers))


def main():
    """Main entry point for CLI."""
    args = parse_args()

    # Import here to avoid slow startup for --help
    from .config import Config
    from .model import ModelManager
    from .steering import SteeringManager
    from .extraction import VectorExtractor, VectorLoader
    from .evaluation import LayerSelector, VectorEffectivenessTester

    # Build config from args
    config = Config(
        model_id=args.model,
        mode=args.mode,
        load_in_4bit=not args.no_quantize,
        load_vectors_from_hf=args.load_from_hf,
        test_only_compassion=not args.all_traits,
        steering_coefficient=args.coefficient,
        upload_to_hf=not args.no_upload,
        analysis_sample_size=args.sample_size,
    )

    if args.layers:
        config.target_layers = parse_layers(args.layers)

    if args.dataset:
        config.analysis_dataset = args.dataset

    if args.all_traits:
        config.traits_to_extract = ["compassion", "open_mindedness", "non_helpfulness"]
    else:
        config.traits_to_extract = [args.trait]

    print(f"Configuration:")
    print(f"  Mode: {config.mode}")
    print(f"  Model: {config.model_id}")
    print(f"  Traits: {config.traits_to_extract}")
    print(f"  Load from HF: {config.load_vectors_from_hf}")
    print()

    # Initialize components
    print("Loading model...")
    model_manager = ModelManager(config)
    model_manager.load_model()

    steering_manager = SteeringManager(model_manager, config)

    # Load or extract vectors
    vectors_by_trait = {}

    if config.load_vectors_from_hf:
        print("\nLoading vectors from HuggingFace...")
        loader = VectorLoader(config)
        vectors_by_trait = loader.load_all_vectors()
    else:
        print("\nExtracting vectors...")
        extractor = VectorExtractor(model_manager, config)

        for trait_name in config.traits_to_extract:
            vectors, stats = extractor.extract_persona_vector(trait_name)

            if vectors:
                # Select best layer
                selector = LayerSelector(model_manager, steering_manager, config)
                best_layer, best_vector = selector.select_best_layer(
                    vectors, trait_name
                )

                if best_vector is not None:
                    vectors[' best'] = best_vector
                    vectors_by_trait[trait_name] = vectors

    # Test effectiveness
    if not args.test_only or vectors_by_trait:
        print("\n" + "="*70)
        print("TESTING VECTOR EFFECTIVENESS")
        print("="*70)

        tester = VectorEffectivenessTester(model_manager, steering_manager, config)

        for trait_name, vectors in vectors_by_trait.items():
            if 'best' in vectors:
                # Get layer index
                layer_name = [k for k in vectors.keys() if k.startswith('layer_')][0]
                layer_idx = int(layer_name.split('_')[1])

                results = tester.test_vector_effectiveness(
                    vectors['best'],
                    layer_idx,
                    trait_name,
                )
                tester.print_results(results)

    # Scale sweep if requested
    if args.scale_sweep and vectors_by_trait:
        print("\n" + "="*70)
        print("SCALE SWEEP")
        print("="*70)

        for trait_name, vectors in vectors_by_trait.items():
            if 'best' in vectors:
                layer_name = [k for k in vectors.keys() if k.startswith('layer_')][0]
                layer_idx = int(layer_name.split('_')[1])

                from .artifacts import get_artifacts
                artifacts = get_artifacts(trait_name)
                test_q = artifacts["test_questions"][0]
                prompt = model_manager.format_simple_prompt(test_q)

                results = steering_manager.scale_sweep(
                    prompt, vectors['best'], layer_idx
                )

                print(f"\n{trait_name} scale sweep on: {test_q[:50]}...")
                for scale, response in results.items():
                    preview = response[:100] if response else "N/A"
                    print(f"  Scale {scale}: {preview}...")

    # Cleanup
    model_manager.cleanup()
    print("\nDone!")


if __name__ == "__main__":
    main()
