"""Tests for the FastAPI serving application.

Nothing in the suite exercised src/serve.py before this module. That gap is why
a missing python-multipart dependency - declared in requirements/serve.txt but
never imported by a test - only surfaced as a crashed container in the cluster.

The app loads its checkpoint during the lifespan startup hook, so every test
that needs a working model goes through `TestClient` as a context manager.
"""
import io
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

# Make src/ importable without installation, as tests/test_model.py does.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import serve  # noqa: E402
from model import get_model  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> Path:
    """Write an untrained checkpoint in the exact shape _load_model expects.

    The weights are random - these tests assert on the response contract, not on
    prediction quality, so an untrained model is sufficient and keeps the fixture
    fast.
    """
    path = tmp_path_factory.mktemp("checkpoints") / "classifier_v1.pt"
    model = get_model(architecture="resnet18", num_classes=10)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "resnet18",
            "num_classes": 10,
        },
        path,
    )
    return path


@pytest.fixture
def client(checkpoint: Path, monkeypatch: pytest.MonkeyPatch):
    """A TestClient with the model loaded via the real lifespan hook."""
    monkeypatch.setenv("CHECKPOINT_PATH", str(checkpoint))
    with TestClient(serve.app) as test_client:
        yield test_client


def _png_bytes(size: tuple[int, int] = (32, 32)) -> bytes:
    """An in-memory PNG, since the endpoint takes a multipart file upload."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(120, 90, 60)).save(buffer, format="PNG")
    return buffer.getvalue()


class TestHealth:
    def test_reports_ok_when_model_loaded(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "model_loaded": True}

    def test_reports_503_when_model_not_loaded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The probe must fail closed, so Kubernetes keeps traffic away."""
        monkeypatch.setattr(serve, "_model", None)
        # No context manager: the lifespan hook does not run, so nothing loads.
        response = TestClient(serve.app).get("/health")
        assert response.status_code == 503


class TestPredict:
    def test_returns_expected_schema(self, client: TestClient) -> None:
        response = client.post(
            "/predict", files={"image": ("test.png", _png_bytes(), "image/png")}
        )
        assert response.status_code == 200

        body = response.json()
        assert set(body) == {"predicted_class", "confidence", "probabilities"}
        assert body["predicted_class"] in serve.CIFAR10_CLASSES
        assert 0.0 <= body["confidence"] <= 1.0

    def test_returns_a_probability_for_every_class(self, client: TestClient) -> None:
        body = client.post(
            "/predict", files={"image": ("test.png", _png_bytes(), "image/png")}
        ).json()

        assert list(body["probabilities"]) == serve.CIFAR10_CLASSES
        # Rounded to 4 dp per class, so allow for accumulated rounding error.
        assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)

    def test_confidence_matches_the_predicted_class(self, client: TestClient) -> None:
        body = client.post(
            "/predict", files={"image": ("test.png", _png_bytes(), "image/png")}
        ).json()

        predicted = body["predicted_class"]
        assert body["confidence"] == body["probabilities"][predicted]
        assert body["probabilities"][predicted] == max(body["probabilities"].values())

    def test_resizes_images_that_are_not_32x32(self, client: TestClient) -> None:
        """The transform chain resizes and centre-crops, so any size is valid."""
        response = client.post(
            "/predict",
            files={"image": ("big.png", _png_bytes(size=(224, 224)), "image/png")},
        )
        assert response.status_code == 200

    def test_rejects_a_payload_that_is_not_an_image(self, client: TestClient) -> None:
        response = client.post(
            "/predict", files={"image": ("notes.txt", b"plain text", "text/plain")}
        )
        assert response.status_code == 400
        assert "Invalid image" in response.json()["detail"]

    def test_requires_the_image_field(self, client: TestClient) -> None:
        """A missing upload is a client error, not a 500."""
        assert client.post("/predict").status_code == 422


class TestStartup:
    def test_startup_fails_when_the_checkpoint_is_absent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Documents why serving pods crash-loop until the training Job finishes.

        _load_model raises during the lifespan hook, so the container exits
        rather than serving a randomly initialised model.
        """
        monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "absent.pt"))
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            with TestClient(serve.app):
                pass
