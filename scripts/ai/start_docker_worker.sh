#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if container is already running and responding
if [ $(docker ps -q -f name=medeye3d-ai) ]; then
    echo "Docker container medeye3d-ai is already running."
    exit 0
fi

# Remove any dead/stopped container with the same name
docker rm -f medeye3d-ai 2>/dev/null || true

# Build image if it doesn't exist
if [[ "$(docker images -q medeye3d-ai:latest 2> /dev/null)" == "" ]]; then
    echo "Building MedEye3d AI Docker image..."
    docker build -t medeye3d-ai:latest "$DIR"
fi

# Determine host project paths (works whether run from host or devcontainer)
HOST_PROJECT_DIR="/mnt/big/project_ssd/project_ssd/MedEye3d.jl"
HOST_APP_DIR="$HOST_PROJECT_DIR/scripts/ai"
HOST_INFERENCE_DIR="$HOST_PROJECT_DIR/tmp_inference"

if [ ! -d "$HOST_PROJECT_DIR" ]; then
    HOST_PROJECT_DIR="$( cd "$DIR/../.." &> /dev/null && pwd )"
    HOST_APP_DIR="$DIR"
    HOST_INFERENCE_DIR="$HOST_PROJECT_DIR/tmp_inference"
fi

mkdir -p "$HOST_INFERENCE_DIR"

# Run container in background, expose 5005, and mount the inference temp dir
echo "Starting MedEye3d AI Docker container..."
docker run -d --rm --name medeye3d-ai --gpus all \
    -p 5005:5005 \
    -v "$HOST_INFERENCE_DIR":/tmp/medeye3d_inference \
    -v "$HOST_APP_DIR":/app \
    medeye3d-ai:latest

