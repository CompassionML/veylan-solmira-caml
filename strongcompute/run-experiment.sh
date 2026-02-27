#!/bin/bash
# Quick experiment runners for CaML linear probes
#
# Usage:
#   ./run-experiment.sh extract <layers>    # Extract activations
#   ./run-experiment.sh train               # Train probes on all layers
#   ./run-experiment.sh visualize           # Generate visualizations
#   ./run-experiment.sh full <layers>       # Extract + train + visualize
#
# Examples:
#   ./run-experiment.sh extract "8 12"
#   ./run-experiment.sh extract "4 8 12 16"
#   ./run-experiment.sh train
#   ./run-experiment.sh full "4 8"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_JOB="${SCRIPT_DIR}/run-job.sh"

# Model path on StrongCompute
MODEL_PATH="/data/uds-grave-seasoned-brownie-251009/"
PAIRS_FILE="pairs_v5_best.jsonl"
WORKSPACE="/workspace"

case "${1:-}" in
    extract)
        layers="${2:-8 12 16 20 24 28}"
        echo "Extracting activations for layers: ${layers}"
        "${RUN_JOB}" "cd ${WORKSPACE} && python extract.py --model ${MODEL_PATH} --pairs ${PAIRS_FILE} --layers ${layers} --output outputs/activations/"
        ;;

    train)
        echo "Training probes on all available layers"
        "${RUN_JOB}" "cd ${WORKSPACE} && python train.py --activations 'outputs/activations/activations_layer_*.pt' --output outputs/probes/"
        ;;

    visualize)
        echo "Generating visualizations"
        "${RUN_JOB}" "cd ${WORKSPACE} && python visualize.py --activations outputs/activations/activations_layer_16.pt --probes outputs/probes/compassion_probes.pt --output outputs/figures/"
        ;;

    full)
        layers="${2:-8 12 16 20 24 28}"
        echo "Running full pipeline for layers: ${layers}"
        "${RUN_JOB}" "cd ${WORKSPACE} && python extract.py --model ${MODEL_PATH} --pairs ${PAIRS_FILE} --layers ${layers} --output outputs/activations/ && python train.py --activations 'outputs/activations/activations_layer_*.pt' --output outputs/probes/ && echo 'Pipeline complete!'"
        ;;

    *)
        echo "CaML Linear Probe Experiments"
        echo ""
        echo "Usage:"
        echo "  ./run-experiment.sh extract <layers>  Extract activations"
        echo "  ./run-experiment.sh train             Train probes"
        echo "  ./run-experiment.sh visualize         Generate figures"
        echo "  ./run-experiment.sh full <layers>     Full pipeline"
        echo ""
        echo "Examples:"
        echo "  ./run-experiment.sh extract \"8 12\""
        echo "  ./run-experiment.sh train"
        echo "  ./run-experiment.sh full \"4 8 12 16 20 24 28\""
        echo ""
        echo "Monitor jobs:"
        echo "  ./run-job.sh --status   # List sessions"
        echo "  ./run-job.sh --attach   # Attach to job"
        echo "  ./run-job.sh --logs     # View output"
        ;;
esac
