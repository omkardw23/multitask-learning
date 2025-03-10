#!/bin/bash

# Define the image name
IMAGE_NAME="multitask-transformer"

# Step 1: Build the Docker image
echo "Building Docker image: $IMAGE_NAME..."
docker build -t $IMAGE_NAME .

# Step 2: Run Training
echo "Starting training inside Docker..."
docker run --gpus all --rm $IMAGE_NAME

# Step 3: Run Inference
echo "Running inference inside Docker..."
docker run --gpus all --rm $IMAGE_NAME python scripts/inference.py

echo "All processes completed successfully!"
