#!/bin/bash
# Run activation extraction on StrongCompute
#
# Usage (on StrongCompute):
#   ./run_extraction.sh [--pairs FILE] [--model MODEL] [--layers "L1 L2 ..."]
#
# Example:
#   ./run_extraction.sh
#   ./run_extraction.sh --pairs /workspace/data/pairs.jsonl --layers "16 20 24 28 31"

set -e

# Defaults
PAIRS_FILE="${PAIRS_FILE:-/workspace/data/contrastive-pairs/pairs.jsonl}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
LAYERS="${LAYERS:-16 20 24 28 31}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/activations}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pairs)
            PAIRS_FILE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --layers)
            LAYERS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_extraction.sh [options]"
            echo ""
            echo "Options:"
            echo "  --pairs FILE     Path to contrastive pairs JSONL"
            echo "  --model MODEL    HuggingFace model name"
            echo "  --layers LAYERS  Space-separated layer numbers (in quotes)"
            echo "  --output DIR     Output directory for activations"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== CaML Activation Extraction ==="
echo ""
echo "Model:  $MODEL"
echo "Pairs:  $PAIRS_FILE"
echo "Layers: $LAYERS"
echo "Output: $OUTPUT_DIR"
echo ""

# Check if pairs file exists
if [[ ! -f "$PAIRS_FILE" ]]; then
    echo "Error: Pairs file not found: $PAIRS_FILE"
    echo ""
    echo "Generate contrastive pairs first, or copy them to the container:"
    echo "  scp pairs.jsonl strongcompute:/workspace/data/contrastive-pairs/"
    exit 1
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
    echo ""
else
    echo "Warning: nvidia-smi not found - running on CPU"
fi

# Activate venv if available
if [[ -f ~/.venv/bin/activate ]]; then
    source ~/.venv/bin/activate
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run extraction
echo "Starting extraction..."
echo "─────────────────────────────────────"

python /workspace/experiments/linear-probes/src/extract.py \
    --model "$MODEL" \
    --pairs "$PAIRS_FILE" \
    --layers $LAYERS \
    --output "$OUTPUT_DIR"

echo ""
echo "Done! Activations saved to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
