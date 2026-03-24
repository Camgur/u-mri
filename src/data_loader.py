from __future__ import annotations

"""Dataset and dataloader utilities for complex MRI k-space training."""

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from . import config
    from .corruptions import apply_corruption
    from .fft import split_complex_to_channels_np
except ImportError:
    import config
    from corruptions import apply_corruption
    from fft import split_complex_to_channels_np


def normalize_kspace(
    kspace: np.ndarray,
    mode: str = "max_abs",
    eps: float = 1e-8,
) -> tuple[np.ndarray, float]:
    """Normalize complex k-space and return normalized data plus scale factor."""
    if mode == "none":
        return kspace.astype(np.complex64, copy=False), 1.0
    if mode == "max_abs":
        scale = float(np.max(np.abs(kspace)))
    elif mode == "std":
        scale = float(np.std(np.abs(kspace)))
    else:
        raise ValueError(f"Unsupported normalization mode: {mode}")
    scale = max(scale, eps)
    return (kspace / scale).astype(np.complex64), scale


def denormalize_kspace(kspace: np.ndarray, scale: float) -> np.ndarray:
    """Invert normalization using a scalar scale factor."""
    return (kspace * scale).astype(np.complex64)


def discover_npy_files(data_root: str | Path) -> list[Path]:
    """Recursively find ``.npy`` files under ``data_root``."""
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    files = sorted(root.rglob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {root}")
    return files


def split_file_paths(
    file_paths: Iterable[Path],
    train_fraction: float = 0.9,
    seed: int = 42,
) -> tuple[list[Path], list[Path]]:
    """Shuffle and split paths into train/validation subsets."""
    file_paths = list(file_paths)
    if len(file_paths) < 2:
        return file_paths, []

    rng = np.random.default_rng(seed)
    indices = np.arange(len(file_paths))
    rng.shuffle(indices)

    train_count = int(round(len(file_paths) * train_fraction))
    train_count = min(max(train_count, 1), len(file_paths) - 1)

    train_paths = [file_paths[i] for i in indices[:train_count]]
    val_paths = [file_paths[i] for i in indices[train_count:]]
    return train_paths, val_paths


def _pad_last_two_dims(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Pad tensor on its trailing spatial dims to ``target_h`` and ``target_w``."""
    h, w = tensor.shape[-2:]
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"Cannot pad tensor of shape {tuple(tensor.shape)} to ({target_h}, {target_w}).")
    if pad_h == 0 and pad_w == 0:
        return tensor
    return F.pad(tensor, (0, pad_w, 0, pad_h))


def collate_kspace_batch(batch: list[dict]) -> dict:
    """Pad variable-size k-space samples in a batch and stack them."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    max_h = max(int(item["input"].shape[-2]) for item in batch)
    max_w = max(int(item["input"].shape[-1]) for item in batch)
    original_hw = torch.tensor(
        [[int(item["input"].shape[-2]), int(item["input"].shape[-1])] for item in batch],
        dtype=torch.int64,
    )

    collated = {
        "input": torch.stack([_pad_last_two_dims(item["input"], max_h, max_w) for item in batch], dim=0),
        "target_kspace": torch.stack(
            [_pad_last_two_dims(item["target_kspace"], max_h, max_w) for item in batch],
            dim=0,
        ),
        "mask": torch.stack([_pad_last_two_dims(item["mask"], max_h, max_w) for item in batch], dim=0),
        "corrupted_kspace": torch.stack(
            [_pad_last_two_dims(item["corrupted_kspace"], max_h, max_w) for item in batch],
            dim=0,
        ),
        "scale": torch.stack([item["scale"] for item in batch], dim=0),
        "path": [item["path"] for item in batch],
        "corruption_type": [item["corruption_type"] for item in batch],
        "requires_data_consistency": torch.stack(
            [item["requires_data_consistency"] for item in batch],
            dim=0,
        ),
        "original_hw": original_hw,
    }

    if any("noise_map" in item for item in batch):
        noise_maps = []
        for item in batch:
            noise = item.get("noise_map")
            if noise is None:
                h, w = int(item["input"].shape[-2]), int(item["input"].shape[-1])
                noise = torch.zeros((1, h, w), dtype=torch.float32)
            noise_maps.append(_pad_last_two_dims(noise, max_h, max_w))
        collated["noise_map"] = torch.stack(noise_maps, dim=0)

    return collated


class KSpaceDataset(Dataset):
    """PyTorch dataset for loading and corrupting complex k-space slices.

    Each sample returns:
    - ``input``: ``[C,H,W]`` channels (real, imag, mask, optional noise map)
    - ``target_kspace``: ``[2,H,W]`` clean real/imag channels
    - ``mask``: ``[1,H,W]`` binary trust mask
    - ``corrupted_kspace``: ``[2,H,W]`` corrupted real/imag channels
    - optional metadata fields used by the training loop
    """

    def __init__(
        self,
        file_paths: list[Path],
        corruption_strategy: str,
        corruption_params: dict | None = None,
        normalization_mode: str = "max_abs",
        normalization_epsilon: float = 1e-8,
        include_noise_map: bool = True,
        seed: int = 42,
    ) -> None:
        self.file_paths = [Path(p) for p in file_paths]
        self.corruption_strategy = corruption_strategy
        self.corruption_params = corruption_params or {}
        self.normalization_mode = normalization_mode
        self.normalization_epsilon = normalization_epsilon
        self.include_noise_map = include_noise_map
        self.seed = seed

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.file_paths)

    @staticmethod
    def _load_complex_kspace(path: Path) -> np.ndarray:
        """Load a complex k-space array from ``.npy`` file."""
        array = np.load(path)
        if np.iscomplexobj(array):
            return array.astype(np.complex64, copy=False)
        if array.ndim == 3 and array.shape[0] == 2:
            return (array[0].astype(np.float32) + 1j * array[1].astype(np.float32)).astype(np.complex64)
        raise ValueError(f"Unsupported k-space array format at {path}: shape={array.shape}, dtype={array.dtype}")

    def __getitem__(self, index: int) -> dict:
        """Load, normalize, corrupt, and package one training sample."""
        path = self.file_paths[index]
        clean_kspace = self._load_complex_kspace(path)
        clean_norm, scale = normalize_kspace(
            clean_kspace,
            mode=self.normalization_mode,
            eps=self.normalization_epsilon,
        )

        rng = np.random.default_rng(self.seed + index)
        output = apply_corruption(
            clean_norm,
            strategy=self.corruption_strategy,
            rng=rng,
            **self.corruption_params,
        )

        corrupted_2ch = split_complex_to_channels_np(output.corrupted_kspace)
        target_2ch = split_complex_to_channels_np(clean_norm)
        mask = output.mask if output.mask is not None else np.ones(clean_norm.shape, dtype=np.float32)

        inputs = [corrupted_2ch, mask[None, ...].astype(np.float32)]
        if self.include_noise_map and output.noise_map is not None:
            inputs.append(output.noise_map[None, ...].astype(np.float32))
        input_tensor = torch.from_numpy(np.concatenate(inputs, axis=0)).float()

        sample = {
            "input": input_tensor,
            "target_kspace": torch.from_numpy(target_2ch).float(),
            "mask": torch.from_numpy(mask[None, ...].astype(np.float32)),
            "corrupted_kspace": torch.from_numpy(corrupted_2ch).float(),
            "scale": torch.tensor(scale, dtype=torch.float32),
            "path": str(path),
            "corruption_type": self.corruption_strategy,
            "requires_data_consistency": torch.tensor(
                1.0 if output.requires_data_consistency else 0.0,
                dtype=torch.float32,
            ),
        }
        if output.noise_map is not None:
            sample["noise_map"] = torch.from_numpy(output.noise_map[None, ...].astype(np.float32))
        return sample


def build_dataloaders(
    data_root: str | Path | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    train_fraction: float | None = None,
    seed: int | None = None,
    pin_memory: bool | None = None,
):
    """Build train/validation dataloaders using values from config by default."""
    data_root = data_root if data_root is not None else config.data_root
    batch_size = batch_size if batch_size is not None else config.batch_size
    num_workers = num_workers if num_workers is not None else config.num_workers
    train_fraction = train_fraction if train_fraction is not None else config.train_split
    seed = seed if seed is not None else config.seed
    pin_memory = pin_memory if pin_memory is not None else config.pin_memory

    files = discover_npy_files(data_root)
    train_files, val_files = split_file_paths(files, train_fraction=train_fraction, seed=seed)
    corruption_params = config.get_corruption_params(config.corruption_strategy)

    train_dataset = KSpaceDataset(
        file_paths=train_files,
        corruption_strategy=config.corruption_strategy,
        corruption_params=corruption_params,
        normalization_mode=config.normalization_mode,
        normalization_epsilon=config.normalization_epsilon,
        include_noise_map=config.include_noise_map,
        seed=seed,
    )
    val_dataset = KSpaceDataset(
        file_paths=val_files if val_files else train_files,
        corruption_strategy=config.corruption_strategy,
        corruption_params=corruption_params,
        normalization_mode=config.normalization_mode,
        normalization_epsilon=config.normalization_epsilon,
        include_noise_map=config.include_noise_map,
        seed=seed + 10_000,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_kspace_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_kspace_batch,
    )
    return train_loader, val_loader
