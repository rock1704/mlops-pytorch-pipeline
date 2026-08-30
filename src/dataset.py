"""Dataset loading and transformation utilities for CIFAR-10."""
import torch
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


def resolve_loader_settings(
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> tuple[int, bool]:
    """Resolve DataLoader tuning knobs against the device actually in use.

    Both settings were previously hardcoded for a GPU host, which is the wrong
    default for the training Job: it requests 2 CPUs and no accelerator.

    - num_workers: worker processes are a win when they overlap host-side
      augmentation with device-side compute. On a CPU-only pod they contend
      with the training process for the very same cores, so the default is 0
      (load in the main process) unless CUDA is present.
    - pin_memory: page-locked staging buffers exist to speed host-to-device
      transfers. With no device to transfer to they are pure overhead, so this
      follows CUDA availability rather than being set unconditionally.

    Passing either argument explicitly overrides the derived default.

    Args:
        num_workers: Worker processes, or None to derive from the device.
        pin_memory: Whether to use pinned memory, or None to derive.

    Returns:
        A tuple of (num_workers, pin_memory).
    """
    cuda = torch.cuda.is_available()
    if num_workers is None:
        num_workers = 2 if cuda else 0
    if pin_memory is None:
        pin_memory = cuda
    return num_workers, pin_memory


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create and return CIFAR-10 train and validation DataLoaders.

    Args:
        data_dir: Root directory where CIFAR-10 will be downloaded/cached.
        batch_size: Samples per mini-batch.
        num_workers: Parallel worker processes, or None to derive from the
            device (see resolve_loader_settings).
        pin_memory: Whether to use page-locked memory, or None to derive.

    Returns:
        A tuple of (train_loader, val_loader).
    """
    num_workers, pin_memory = resolve_loader_settings(num_workers, pin_memory)

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
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
