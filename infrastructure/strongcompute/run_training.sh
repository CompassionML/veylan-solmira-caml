#!/bin/bash
# Run probe training on StrongCompute
#
# Usage (on StrongCompute):
#   ./run_training.sh [--activations FILE] [--layers "L1 L2 ..."]
#
# Example:
#   ./run_training.sh
#   ./run_training.sh --activations outputs/activations/activations_layers16_20_24_28.pt

set -e

# Defaults
ACTIVATIONS_FILE="${ACTIVATIONS_FILE:-/workspace/outputs/activations/activations_layers16_20_24_28_31.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/probes}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --activations)
            ACTIVATIONS_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --layers)
            LAYERS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_training.sh [options]"
            echo ""
            echo "Options:"
            echo "  --activations FILE   Path to extracted activations (.pt file)"
            echo "  --output DIR         Output directory for trained probes"
            echo "  --layers LAYERS      Specific layers to train (default: all in file)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== CaML Probe Training ==="
echo ""
echo "Activations: $ACTIVATIONS_FILE"
echo "Output:      $OUTPUT_DIR"
if [[ -n "$LAYERS" ]]; then
    echo "Layers:      $LAYERS"
fi
echo ""

# Check if activations file exists
if [[ ! -f "$ACTIVATIONS_FILE" ]]; then
    echo "Error: Activations file not found: $ACTIVATIONS_FILE"
    echo ""
    echo "Run extraction first:"
    echo "  ./run_extraction.sh"
    exit 1
fi

# Activate venv if available
if [[ -f ~/.venv/bin/activate ]]; then
    source ~/.venv/bin/activate
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command
CMD="python /workspace/experiments/linear-probes/src/train.py \
    --activations $ACTIVATIONS_FILE \
    --output $OUTPUT_DIR"

if [[ -n "$LAYERS" ]]; then
    CMD="$CMD --layers $LAYERS"
fi

# Run training
echo "Starting training..."
echo "─────────────────────────────────────"

eval $CMD

echo ""
echo "Done! Probes saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"

# Display metrics
if [[ -f "$OUTPUT_DIR/compassion_metrics.json" ]]; then
    echo ""
    echo "=== Metrics ==="
    cat "$OUTPUT_DIR/compassion_metrics.json"
fi
