#!/bin/bash
# Run full CaML linear probe pipeline on StrongCompute
#
# Prerequisites:
#   1. Contrastive pairs generated and copied to container
#   2. HuggingFace token set (for model access)
#
# Usage:
#   ./run_pipeline.sh
#   ./run_pipeline.sh --pairs /path/to/pairs.jsonl
#   ./run_pipeline.sh --skip-extraction  # Use existing activations

set -e

# Configuration
PAIRS_FILE="${PAIRS_FILE:-/workspace/data/contrastive-pairs/pairs.jsonl}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
LAYERS="${LAYERS:-16 20 24 28 31}"
ACTIVATIONS_DIR="${ACTIVATIONS_DIR:-/workspace/outputs/activations}"
PROBES_DIR="${PROBES_DIR:-/workspace/outputs/probes}"
SKIP_EXTRACTION=false

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
        --skip-extraction)
            SKIP_EXTRACTION=true
            shift
            ;;
        --help|-h)
            echo "CaML Linear Probe Pipeline"
            echo ""
            echo "Usage: ./run_pipeline.sh [options]"
            echo ""
            echo "Options:"
            echo "  --pairs FILE        Path to contrastive pairs JSONL"
            echo "  --model MODEL       HuggingFace model name"
            echo "  --skip-extraction   Skip extraction, use existing activations"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "╔════════════════════════════════════════════════════════════╗"
echo "║           CaML Linear Probe Pipeline                       ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║ Model:  $MODEL"
echo "║ Layers: $LAYERS"
echo "║ Pairs:  $PAIRS_FILE"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Activate venv
if [[ -f ~/.venv/bin/activate ]]; then
    source ~/.venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Check GPU
echo ""
echo "=== GPU Status ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    echo "Warning: No GPU detected"
fi

# Check prerequisites
echo ""
echo "=== Prerequisites Check ==="

if [[ ! -f "$PAIRS_FILE" ]]; then
    echo "✗ Contrastive pairs not found: $PAIRS_FILE"
    echo ""
    echo "Copy pairs to container:"
    echo "  scp pairs.jsonl strongcompute:/workspace/data/contrastive-pairs/"
    exit 1
else
    PAIRS_COUNT=$(wc -l < "$PAIRS_FILE")
    echo "✓ Contrastive pairs: $PAIRS_COUNT pairs"
fi

# Check HuggingFace token
if [[ -z "$HF_TOKEN" && ! -f ~/.cache/huggingface/token ]]; then
    echo ""
    echo "Note: HuggingFace token not set. If model access fails, run:"
    echo "  huggingface-cli login"
fi

echo ""

# Step 1: Activation Extraction
if [[ "$SKIP_EXTRACTION" == true ]]; then
    echo "=== Step 1: Extraction (SKIPPED) ==="
else
    echo "=== Step 1: Activation Extraction ==="
    echo ""

    mkdir -p "$ACTIVATIONS_DIR"

    python /workspace/experiments/linear-probes/src/extract.py \
        --model "$MODEL" \
        --pairs "$PAIRS_FILE" \
        --layers $LAYERS \
        --output "$ACTIVATIONS_DIR"

    echo ""
    echo "✓ Extraction complete"
fi

# Find the activations file
LAYERS_JOINED=$(echo $LAYERS | tr ' ' '_')
ACTIVATIONS_FILE="$ACTIVATIONS_DIR/activations_layers${LAYERS_JOINED}.pt"

if [[ ! -f "$ACTIVATIONS_FILE" ]]; then
    # Try to find any activations file
    ACTIVATIONS_FILE=$(ls -t "$ACTIVATIONS_DIR"/*.pt 2>/dev/null | head -1)
    if [[ -z "$ACTIVATIONS_FILE" ]]; then
        echo "Error: No activations file found in $ACTIVATIONS_DIR"
        exit 1
    fi
fi

echo ""
echo "Using activations: $ACTIVATIONS_FILE"

# Step 2: Probe Training
echo ""
echo "=== Step 2: Probe Training ==="
echo ""

mkdir -p "$PROBES_DIR"

python /workspace/experiments/linear-probes/src/train.py \
    --activations "$ACTIVATIONS_FILE" \
    --output "$PROBES_DIR"

echo ""
echo "✓ Training complete"

# Summary
echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    Pipeline Complete                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Outputs:"
echo "  Activations: $ACTIVATIONS_DIR/"
echo "  Probes:      $PROBES_DIR/"
echo ""

# Display metrics
if [[ -f "$PROBES_DIR/compassion_metrics.json" ]]; then
    echo "=== Results ==="
    cat "$PROBES_DIR/compassion_metrics.json"
    echo ""
fi

echo "Next steps:"
echo "  1. Review metrics in $PROBES_DIR/compassion_metrics.json"
echo "  2. Run AHB evaluation: python src/evaluate.py --probes $PROBES_DIR/compassion_probes.pt"
echo "  3. Copy results to local machine:"
echo "     scp -r strongcompute:/workspace/outputs ."
