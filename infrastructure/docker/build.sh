#!/bin/bash
# Build and push CaML research Docker image
#
# Usage:
#   ./build.sh                    # Build for amd64 (StrongCompute)
#   ./build.sh --push             # Build and push to DockerHub
#   ./build.sh --push --tag v2    # Build, tag as v2, and push
#   ./build.sh --local            # Build for local arch (arm64 on M1/M2)
#
# Note: StrongCompute uses amd64, so we default to that platform.

set -e

# Configuration
IMAGE_NAME="${DOCKER_IMAGE:-veylansolmira/caml-env}"
TAG="${TAG:-latest}"
PUSH=false
PLATFORM="linux/amd64"  # Default to StrongCompute's architecture

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
        --local)
            PLATFORM=""  # Use native platform
            shift
            ;;
        --platform)
            PLATFORM="$2"
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
if [[ -n "$PLATFORM" ]]; then
    echo "Platform: ${PLATFORM}"
fi
echo ""

# Build with buildx for cross-platform support
if [[ -n "$PLATFORM" ]]; then
    # Cross-platform build (amd64 on arm64 Mac)
    if [ "$PUSH" = true ]; then
        # Build and push in one step (required for cross-platform)
        echo "Building and pushing for ${PLATFORM}..."
        docker buildx build \
            --platform "${PLATFORM}" \
            -t "${IMAGE_NAME}:${TAG}" \
            -f "${SCRIPT_DIR}/Dockerfile" \
            --push \
            "${SCRIPT_DIR}"
        echo ""
        echo "Build complete and pushed: ${IMAGE_NAME}:${TAG}"
    else
        # Build and load locally (for testing)
        echo "Building for ${PLATFORM}..."
        docker buildx build \
            --platform "${PLATFORM}" \
            -t "${IMAGE_NAME}:${TAG}" \
            -f "${SCRIPT_DIR}/Dockerfile" \
            --load \
            "${SCRIPT_DIR}"
        echo ""
        echo "Build complete: ${IMAGE_NAME}:${TAG}"
    fi
else
    # Native platform build
    docker build \
        -t "${IMAGE_NAME}:${TAG}" \
        -f "${SCRIPT_DIR}/Dockerfile" \
        "${SCRIPT_DIR}"
    echo ""
    echo "Build complete: ${IMAGE_NAME}:${TAG}"

    if [ "$PUSH" = true ]; then
        echo ""
        echo "=== Pushing to DockerHub ==="
        docker push "${IMAGE_NAME}:${TAG}"
        echo "Pushed: ${IMAGE_NAME}:${TAG}"
    fi
fi

echo ""
echo "Import into StrongCompute via Control Plane:"
echo "  https://cp.strongcompute.ai/organisations/<org-id>/workstations?section=images"
