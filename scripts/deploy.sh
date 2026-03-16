#!/usr/bin/env bash
set -euo pipefail

# CI/CD deployment script — applies manifests and verifies the rollout.
# Intended to be run repeatedly on every new release.
#
# Usage:
#   ./scripts/deploy.sh                    # uses image tag from deployment.yaml
#   ./scripts/deploy.sh v1.2.0             # overrides the image tag

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MANIFESTS_DIR="${REPO_ROOT}/k8s"
IMAGE_TAG="${1:-}"
NAMESPACE="k8-app"
IMAGE_REPO="ghcr.io/kyleerwin/k8"

echo "==> Applying k8s manifests..."
kubectl apply -f "${MANIFESTS_DIR}/"

if [[ -n "${IMAGE_TAG}" ]]; then
  echo "==> Updating deployment image to ${IMAGE_REPO}:${IMAGE_TAG}..."
  kubectl -n "${NAMESPACE}" set image deployment/k8-app \
    k8-app="${IMAGE_REPO}:${IMAGE_TAG}"
fi

echo "==> Waiting for rollout to complete..."
kubectl -n "${NAMESPACE}" rollout status deployment/k8-app --timeout=120s

echo "==> Current pod status:"
kubectl -n "${NAMESPACE}" get pods

echo "==> Deployment complete!"
echo "    To test locally: kubectl -n ${NAMESPACE} port-forward svc/k8-app 8080:80"
echo "    Then: curl http://localhost:8080/hello"
