# mlops-pytorch-pipeline

A production-style MLOps pipeline for CIFAR-10 image classification using **PyTorch**, **Docker**, and **Kubernetes**.

[![CI](https://github.com/rock1704/mlops-pytorch-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/rock1704/mlops-pytorch-pipeline/actions/workflows/ci.yml)

---

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │           Kubernetes Cluster             │
                        │           (namespace: ml-training)       │
                        │                                          │
  ┌─────────────┐       │  ┌──────────────┐   ┌───────────────┐   │
  │  Git Push   │──────►│  │  K8s Job     │   │  Deployment   │   │
  │  (CI/CD)    │       │  │  (Training)  │   │  (Serving x2) │   │
  └─────────────┘       │  └──────┬───────┘   └───────┬───────┘   │
                        │         │                    │           │
                        │         ▼                    ▼           │
                        │  ┌──────────────┐   ┌───────────────┐   │
                        │  │  PVC: data   │   │  PVC:         │   │
                        │  │  (CIFAR-10)  │   │  checkpoints  │◄──┤
                        │  └──────────────┘   └───────────────┘   │
                        │                                          │
                        │  ┌──────────────┐   ┌───────────────┐   │
  ┌─────────────┐       │  │  ConfigMap   │   │  Service      │   │
  │   Client    │◄──────┤  │  (YAML cfg)  │   │  port 80→8080 │◄──┤
  │  (curl/app) │       │  └──────────────┘   └───────────────┘   │
  └─────────────┘       └─────────────────────────────────────────┘
```

**Flow:** GitHub CI builds & tests → Docker images pushed → K8s Job trains model → checkpoint written to PVC → Serving Deployment loads checkpoint → exposed via Service.

---

## Project Structure

```
mlops-pytorch-pipeline/
├── .github/workflows/ci.yml       # GitHub Actions: lint, test, Docker build
├── configs/
│   └── training_config.yaml       # Hyperparameters (read by train.py)
├── docker/
│   ├── Dockerfile.train           # Multi-stage training image
│   └── Dockerfile.serve           # Slim serving image (non-root, healthcheck)
├── k8s/
│   ├── namespace.yaml
│   ├── configmap.yaml
│   ├── pvc.yaml                   # PVCs for data & checkpoints
│   ├── training-job.yaml          # K8s Job (+ GPU bonus)
│   ├── serving-deployment.yaml    # 2-replica deployment with probes
│   ├── serving-service.yaml       # ClusterIP service
│   └── hpa.yaml                   # HorizontalPodAutoscaler
├── requirements/
│   ├── train.txt                  # Pinned training deps
│   └── serve.txt                  # Pinned inference-only deps
├── src/
│   ├── model.py                   # ResNet-18 (CIFAR-10 adapted)
│   ├── dataset.py                 # CIFAR-10 DataLoaders
│   ├── train.py                   # Training loop with early stopping
│   └── serve.py                   # FastAPI inference server
└── tests/
    └── test_model.py              # Unit tests
```

---

## Prerequisites

- Python 3.10+
- Docker Desktop
- `kubectl` CLI (for Kubernetes deployment)
- A Kubernetes cluster (Minikube, kind, or Docker Desktop's built-in K8s)

---

## Quick Start (Local Training with Docker)

### 1. Build the training image
```bash
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
```

### 2. Run training (mounts local data and checkpoint directories)
```bash
# Windows PowerShell
docker run --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/checkpoints:/app/checkpoints `
  mlops-train:v1

# Linux / macOS
docker run --rm \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-train:v1
```

Training logs are streamed as JSON lines:
```json
{"epoch": 1, "train_loss": 1.7842, "train_accuracy": 0.3521, "val_loss": 1.6103, "val_accuracy": 0.4012}
```

### 3. Build the serving image
```bash
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
```

### 4. Run the serving container
```bash
# Windows PowerShell
docker run --rm -p 8080:8080 `
  -v ${PWD}/checkpoints:/app/checkpoints `
  mlops-serve:v1

# Linux / macOS
docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1
```

### 5. Test the prediction endpoint
```bash
# Health check
curl http://localhost:8080/health

# Predict
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

Example response:
```json
{
  "predicted_class": "airplane",
  "confidence": 0.8731,
  "probabilities": {
    "airplane": 0.8731, "automobile": 0.0412, "bird": 0.0193,
    "cat": 0.0091, "deer": 0.0089, "dog": 0.0074,
    "frog": 0.0163, "horse": 0.0102, "ship": 0.0093, "truck": 0.0052
  }
}
```

---

## Kubernetes Deployment

> **Note:** Enable Kubernetes in Docker Desktop → Settings → Kubernetes → Enable Kubernetes.

### 1. Apply all manifests
```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
```

### 2. Watch training job progress
```bash
kubectl logs -f job/cifar10-training -n ml-training
```

### 3. Deploy the serving layer (after training completes)
```bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

### 4. Verify pods
```bash
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training
```

### 5. Test the endpoint
```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test_image.png"
```

---

## Running Tests Locally

```bash
pip install pytest torch torchvision pillow numpy
pytest tests/ -v
```

---

## Configuration

Edit `configs/training_config.yaml` to adjust hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.architecture` | `resnet18` | Model backbone |
| `model.num_classes` | `10` | CIFAR-10 classes |
| `training.epochs` | `10` | Max training epochs |
| `training.batch_size` | `64` | Samples per mini-batch |
| `training.learning_rate` | `0.001` | Adam optimizer LR |
| `training.early_stopping_patience` | `3` | Epochs without improvement before stopping |
