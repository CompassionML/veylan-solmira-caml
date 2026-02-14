#!/bin/bash
# Build and push CaML research Docker image
#
# Usage:
#   ./build.sh                    # Build only
#   ./build.sh --push             # Build and push to DockerHub
#   ./build.sh --push --tag v2    # Build, tag as v2, and push

set -e

# Configuration
IMAGE_NAME="${DOCKER_IMAGE:-caml/research-env}"
TAG="${TAG:-latest}"
PUSH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Building CaML Research Image ==="
echo "Image: ${IMAGE_NAME}:${TAG}"
echo ""

# Build
docker build \
    -t "${IMAGE_NAME}:${TAG}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}"

echo ""
echo "✓ Build complete: ${IMAGE_NAME}:${TAG}"

# Push if requested
if [ "$PUSH" = true ]; then
    echo ""
    echo "=== Pushing to DockerHub ==="
    docker push "${IMAGE_NAME}:${TAG}"
    echo "✓ Pushed: ${IMAGE_NAME}:${TAG}"
    echo ""
    echo "Import into StrongCompute via Control Plane:"
    echo "  https://cp.strongcompute.ai/organisations/<org-id>/workstations?section=images"
fi
