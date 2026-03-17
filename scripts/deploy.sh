#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/deploy.sh                    # uses image tag from deployment.yaml
#   ./scripts/deploy.sh v1.2.0             # overrides the image tag

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFESTS_DIR="${REPO_ROOT}/ndbk-bmds"
IMAGE_TAG="${1:-}"
NAMESPACE="ndbk-bmd-app"
IMAGE_REPO="ghcr.io/kyleerwin/ndbk-bmd"

echo "Applying ndbk-bmds manifests..."
kubectl apply -f "${MANIFESTS_DIR}/"

if [[ -n "${IMAGE_TAG}" ]]; then
  echo "Updating deployment image to ${IMAGE_REPO}:${IMAGE_TAG}..."
  kubectl -n "${NAMESPACE}" set image deployment/ndbk-bmd-app \
    ndbk-bmd-app="${IMAGE_REPO}:${IMAGE_TAG}"
fi

echo "Waiting for rollout to complete..."
kubectl -n "${NAMESPACE}" rollout status deployment/ndbk-bmd-app --timeout=120s

echo "Current pod status:"
kubectl -n "${NAMESPACE}" get pods

echo "Deployment complete"
