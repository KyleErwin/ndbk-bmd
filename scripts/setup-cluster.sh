#!/usr/bin/env bash
set -euo pipefail

# One-off script to create a kind cluster and install infrastructure components.
# Safe to re-run — skips steps that are already in place.

CLUSTER_NAME="${KIND_CLUSTER_NAME:-kind}"

echo "==> Checking for kind cluster '${CLUSTER_NAME}'..."
if kind get clusters 2>/dev/null | grep -qx "${CLUSTER_NAME}"; then
  echo "    Cluster '${CLUSTER_NAME}' already exists — skipping creation."
else
  echo "    Creating kind cluster '${CLUSTER_NAME}'..."
  kind create cluster --name "${CLUSTER_NAME}"
fi

echo "==> Checking for NGINX ingress controller..."
if kubectl get namespace ingress-nginx &>/dev/null && \
   kubectl get deploy -n ingress-nginx ingress-nginx-controller &>/dev/null; then
  echo "    NGINX ingress controller already installed — skipping."
else
  echo "    Installing NGINX ingress controller..."
  kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
fi

echo "==> Waiting for ingress controller to be ready..."
kubectl wait --namespace ingress-nginx \
  --for=condition=ready pod \
  --selector=app.kubernetes.io/component=controller \
  --timeout=90s

echo "==> Cluster setup complete!"
