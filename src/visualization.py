from __future__ import annotations

"""Visualization helpers for qualitative MRI reconstruction monitoring."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .fft import kspace_magnitude, magnitude_image_from_kspace
except ImportError:
    from fft import kspace_magnitude, magnitude_image_from_kspace


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Detach torch tensor to numpy array when needed."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _ensure_batch(x: torch.Tensor) -> torch.Tensor:
    """Ensure input has a batch dimension."""
    return x.unsqueeze(0) if x.ndim == 3 else x


def save_reconstruction_figure(
    corrupted_kspace: torch.Tensor,
    predicted_kspace: torch.Tensor,
    target_kspace: torch.Tensor,
    save_path: str | Path,
    sample_index: int = 0,
    title: str | None = None,
) -> None:
    """Save a 2x3 panel of k-space, image reconstructions, and error map."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    corrupted_kspace = _ensure_batch(corrupted_kspace)
    predicted_kspace = _ensure_batch(predicted_kspace)
    target_kspace = _ensure_batch(target_kspace)

    corrupted_mag = _to_numpy(kspace_magnitude(corrupted_kspace)[sample_index])
    pred_mag = _to_numpy(kspace_magnitude(predicted_kspace)[sample_index])
    target_mag = _to_numpy(kspace_magnitude(target_kspace)[sample_index])

    pred_img = _to_numpy(magnitude_image_from_kspace(predicted_kspace)[sample_index])
    target_img = _to_numpy(magnitude_image_from_kspace(target_kspace)[sample_index])
    img_error = np.abs(pred_img - target_img)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    if title:
        fig.suptitle(title)

    axes[0, 0].imshow(np.log1p(corrupted_mag), cmap="gray", aspect="auto")
    axes[0, 0].set_title("Corrupted k-space")
    axes[0, 1].imshow(np.log1p(pred_mag), cmap="gray", aspect="auto")
    axes[0, 1].set_title("Predicted k-space")
    axes[0, 2].imshow(np.log1p(target_mag), cmap="gray", aspect="auto")
    axes[0, 2].set_title("Target k-space")

    axes[1, 0].imshow(pred_img, cmap="gray", aspect="auto")
    axes[1, 0].set_title("Predicted image")
    axes[1, 1].imshow(target_img, cmap="gray", aspect="auto")
    axes[1, 1].set_title("Target image")
    im = axes[1, 2].imshow(img_error, cmap="magma", aspect="auto")
    axes[1, 2].set_title("Absolute error")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    for ax in axes.ravel():
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_loss_curves(history: dict[str, list[float]], save_path: str | Path) -> None:
    """Plot train/validation curves for total, k-space, and image losses."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = np.arange(1, len(history.get("train_loss_total", [])) + 1)
    if len(epochs) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [
        ("loss_total", "Total Loss"),
        ("loss_kspace", "K-space Loss"),
        ("loss_image", "Image Loss"),
    ]
    for axis, (key, label) in zip(axes, pairs):
        train_key = f"train_{key}"
        val_key = f"val_{key}"
        axis.plot(epochs, history.get(train_key, []), label="train")
        axis.plot(epochs, history.get(val_key, []), label="val")
        axis.set_title(label)
        axis.set_xlabel("Epoch")
        axis.grid(True, alpha=0.2)
        axis.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
