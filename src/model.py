"""Model definitions for CIFAR-10 image classification.

Supports ResNet-18 from torchvision with modifications for 32x32 input images.
"""
import torch.nn as nn
from torchvision import models


def _adapt_resnet_for_cifar(model: nn.Module) -> nn.Module:
    """Adapt a standard ResNet for CIFAR-10 (32x32 images).

    The default ResNet uses a 7x7 conv with stride 2 followed by a MaxPool,
    which aggressively downsamples 224x224 inputs. For 32x32 CIFAR images
    we replace that stem with a 3x3 conv, stride 1, no MaxPool to preserve
    spatial resolution through the early layers.
    """
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def get_model(architecture: str = "resnet18", num_classes: int = 10) -> nn.Module:
    """Build and return a classification model.

    Args:
        architecture: Model architecture name. Currently supports 'resnet18'.
        num_classes: Number of output classes (10 for CIFAR-10).

    Returns:
        A PyTorch nn.Module ready for training.

    Raises:
        ValueError: If the requested architecture is not supported.
    """
    if architecture == "resnet18":
        # weights=None -> random init, trains from scratch on CIFAR-10
        model = models.resnet18(weights=None, num_classes=num_classes)
        model = _adapt_resnet_for_cifar(model)
        return model

    raise ValueError(
        f"Unsupported architecture: {architecture!r}. "
        "Currently supported: ['resnet18']"
    )
