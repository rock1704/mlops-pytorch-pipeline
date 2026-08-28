"""Dataset loading and transformation utilities for CIFAR-10."""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CIFAR-10 channel statistics (computed over the training set)
_CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_transforms(train: bool = True) -> transforms.Compose:
    """Return image transforms for training or validation.

    Training augmentations (RandomHorizontalFlip + RandomCrop) help
    reduce overfitting on the small CIFAR-10 training set.

    Args:
        train: If True, include data augmentation transforms.

    Returns:
        A composed transform pipeline.
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD),
    ])


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Create and return CIFAR-10 train and validation DataLoaders.

    Args:
        data_dir: Root directory where CIFAR-10 will be downloaded/cached.
        batch_size: Samples per mini-batch.
        num_workers: Parallel worker processes for data loading.

    Returns:
        A tuple of (train_loader, val_loader).
    """
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
