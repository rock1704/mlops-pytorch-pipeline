"""FastAPI model serving application for CIFAR-10 classifier.

Endpoints:
  GET  /health   - Returns 200 + status if model is loaded.
  POST /predict  - Accepts an image upload, returns class probabilities.
"""
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent))
from model import get_model


# CIFAR-10 class labels
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Inference pre-processing (no augmentation, only normalise)
_INFER_TRANSFORM = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
])

# Module-level state
_model: torch.nn.Module | None = None
_device: torch.device = torch.device("cpu")


def _load_model() -> None:
    """Load the model checkpoint from CHECKPOINT_PATH env var or default path."""
    global _model, _device
    checkpoint_path = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=_device, weights_only=True)

    architecture = checkpoint.get("architecture", "resnet18")
    num_classes = checkpoint.get("num_classes", 10)
    _model = get_model(architecture=architecture, num_classes=num_classes).to(_device)
    _model.load_state_dict(checkpoint["model_state_dict"])
    _model.eval()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup; clean up on shutdown."""
    _load_model()
    yield


app = FastAPI(title="CIFAR-10 Classifier API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness/readiness probe endpoint.

    Returns HTTP 200 with a status payload if the model is loaded and ready.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> JSONResponse:
    """Run inference on an uploaded image.

    Args:
        image: An image file upload (JPEG, PNG, etc.).

    Returns:
        JSON with the predicted class label and per-class probabilities.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Read and decode the uploaded image
    raw = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    # Pre-process and run inference
    tensor = _INFER_TRANSFORM(pil_image).unsqueeze(0).to(_device)
    with torch.no_grad():
        logits = _model(tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).tolist()

    predicted_idx = int(max(range(len(probabilities)), key=lambda i: probabilities[i]))
    return JSONResponse(
        content={
            "predicted_class": CIFAR10_CLASSES[predicted_idx],
            "confidence": round(probabilities[predicted_idx], 4),
            "probabilities": {
                cls: round(prob, 4)
                for cls, prob in zip(CIFAR10_CLASSES, probabilities)
            },
        }
    )
