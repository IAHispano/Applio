#!/bin/bash

# Configuration
PROJECT_ID="applio-project"  # Update with actual project ID
REGION="us-central1"
REPOSITORY="applio"
IMAGE_NAME="applio-tts"
VERSION=$(date +%Y%m%d_%H%M%S)

# Ensure environment variables are set
if [ -z "$APPLIO_API_WRITE" ]; then
    echo "Error: APPLIO_API_WRITE environment variable is not set"
    exit 1
fi

# Build and push Docker image
echo "Building Docker image..."
docker build -t gcr.io/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$VERSION .
if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
fi

echo "Pushing Docker image to Google Container Registry..."
docker push gcr.io/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$VERSION
if [ $? -ne 0 ]; then
    echo "Error: Docker push failed"
    exit 1
fi

# Create Vertex AI model and endpoint
echo "Creating Vertex AI model..."
gcloud ai models upload \
    --region=$REGION \
    --display-name=applio-tts-$VERSION \
    --container-image-uri=gcr.io/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$VERSION \
    --container-ports=8080 \
    --container-environment-variables=APPLIO_API_WRITE=$APPLIO_API_WRITE

if [ $? -ne 0 ]; then
    echo "Error: Failed to create Vertex AI model"
    exit 1
fi

echo "Deployment completed successfully!"
