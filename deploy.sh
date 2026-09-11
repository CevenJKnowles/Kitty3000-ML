#!/usr/bin/env bash
set -e

PROJECT=$(gcloud config get-value project 2>/dev/null)
REGION=europe-west1
REPO=kitty3000
SERVICE=kitty3000-api
IMAGE=$REGION-docker.pkg.dev/$PROJECT/$REPO/$SERVICE

echo "project: $PROJECT"
echo "image:   $IMAGE"

mkdir -p models

gcloud services enable run.googleapis.com artifactregistry.googleapis.com
gcloud artifacts repositories create $REPO --repository-format=docker --location=$REGION 2>/dev/null || true
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet

docker build --platform linux/amd64 -t $IMAGE:latest .
docker push $IMAGE:latest

gcloud run deploy $SERVICE \
  --image $IMAGE:latest \
  --region $REGION \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 4 \
  --cpu-boost \
  --allow-unauthenticated

echo
echo "URL:"
gcloud run services describe $SERVICE --region $REGION --format 'value(status.url)'
