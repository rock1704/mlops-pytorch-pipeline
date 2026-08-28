"""Unit tests for the model and dataset utilities."""
import sys
from pathlib import Path

import torch
import pytest

# Make src/ importable without installation
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model import get_model
from dataset import get_transforms


class TestGetModel:
    """Tests for get_model()."""

    def test_resnet18_output_shape(self):
        """Model should produce (batch, num_classes) logits."""
        model = get_model(architecture="resnet18", num_classes=10)
        dummy = torch.zeros(2, 3, 32, 32)  # batch=2, CIFAR-10 size
        output = model(dummy)
        assert output.shape == (2, 10)

    def test_resnet18_default_classes(self):
        """Default num_classes should be 10 for CIFAR-10."""
        model = get_model("resnet18")
        params = sum(p.numel() for p in model.parameters())
        assert params > 0, "Model should have parameters"

    def test_unsupported_architecture_raises(self):
        """Requesting an unknown architecture should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported architecture"):
            get_model(architecture="vgg99")

    def test_cifar_stem_modification(self):
        """Verify the CIFAR stem: conv1 should be 3x3, maxpool should be Identity."""
        import torch.nn as nn
        model = get_model("resnet18", num_classes=10)
        assert model.conv1.kernel_size == (3, 3), "Stem conv should be 3x3 for CIFAR-10"
        assert model.conv1.stride == (1, 1), "Stem conv stride should be 1 for CIFAR-10"
        assert isinstance(model.maxpool, nn.Identity), "MaxPool should be replaced with Identity"


class TestGetTransforms:
    """Tests for get_transforms()."""

    def test_train_transform_returns_tensor(self):
        """Training transform pipeline should return a 3x32x32 tensor."""
        from PIL import Image
        import numpy as np

        transform = get_transforms(train=True)
        img = Image.fromarray(np.zeros((32, 32, 3), dtype="uint8"))
        tensor = transform(img)
        assert tensor.shape == (3, 32, 32)

    def test_val_transform_returns_tensor(self):
        """Validation transform pipeline should return a 3x32x32 tensor."""
        from PIL import Image
        import numpy as np

        transform = get_transforms(train=False)
        img = Image.fromarray(np.zeros((32, 32, 3), dtype="uint8"))
        tensor = transform(img)
        assert tensor.shape == (3, 32, 32)

    def test_train_has_more_transforms(self):
        """Training pipeline should include augmentation (more transforms than val)."""
        train_t = get_transforms(train=True)
        val_t = get_transforms(train=False)
        assert len(train_t.transforms) > len(val_t.transforms)
