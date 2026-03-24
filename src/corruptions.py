from __future__ import annotations

"""Corruption operators for complex MRI k-space arrays."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CorruptionOutput:
    """Container returned by all corruption functions.

    Attributes:
        corrupted_kspace: Corrupted complex k-space array with shape ``[H, W]``.
        mask: Optional binary mask with ``1`` for trusted/measured coefficients.
        noise_map: Optional per-pixel confidence/noise guidance map.
        requires_data_consistency: Whether post-network data consistency should be applied.
        metadata: Extra strategy-specific values for logging/debugging.
    """

    corrupted_kspace: np.ndarray
    mask: np.ndarray | None = None
    noise_map: np.ndarray | None = None
    requires_data_consistency: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def _ensure_complex64(kspace: np.ndarray) -> np.ndarray:
    """Validate complex input and cast to ``complex64``."""
    if not np.iscomplexobj(kspace):
        raise ValueError("Expected a complex-valued k-space array.")
    return kspace.astype(np.complex64, copy=False)


def low_frequency_corruption(kspace: np.ndarray, center_fraction: float = 0.18) -> CorruptionOutput:
    """Corrupt central low-frequency coefficients by zeroing a centered region.

    The returned mask has ``1`` in trusted high-frequency areas and ``0`` in
    low-frequency coefficients to reconstruct.
    """

    kspace = _ensure_complex64(kspace)
    h, w = kspace.shape
    center_h = max(1, int(round(h * center_fraction)))
    center_w = max(1, int(round(w * center_fraction)))

    hs = (h - center_h) // 2
    ws = (w - center_w) // 2
    he = hs + center_h
    we = ws + center_w

    mask = np.ones((h, w), dtype=np.float32)
    mask[hs:he, ws:we] = 0.0
    corrupted = kspace * mask
    return CorruptionOutput(
        corrupted_kspace=corrupted,
        mask=mask,
        noise_map=None,
        requires_data_consistency=True,
        metadata={"center_fraction": center_fraction},
    )


def high_frequency_corruption(kspace: np.ndarray, center_fraction: float = 0.22) -> CorruptionOutput:
    """Corrupt high-frequency coefficients and keep only a centered low-frequency region.

    The returned mask has ``1`` in trusted low-frequency areas and ``0`` in
    high-frequency coefficients to reconstruct.
    """

    kspace = _ensure_complex64(kspace)
    h, w = kspace.shape
    center_h = max(1, int(round(h * center_fraction)))
    center_w = max(1, int(round(w * center_fraction)))

    hs = (h - center_h) // 2
    ws = (w - center_w) // 2
    he = hs + center_h
    we = ws + center_w

    mask = np.zeros((h, w), dtype=np.float32)
    mask[hs:he, ws:we] = 1.0
    corrupted = kspace * mask
    return CorruptionOutput(
        corrupted_kspace=corrupted,
        mask=mask,
        noise_map=None,
        requires_data_consistency=True,
        metadata={"center_fraction": center_fraction},
    )


def random_dropout_corruption(
    kspace: np.ndarray,
    dropout_probability: float = 0.55,
    rng: np.random.Generator | None = None,
) -> CorruptionOutput:
    """Randomly drop k-space coefficients with probability ``dropout_probability``."""

    kspace = _ensure_complex64(kspace)
    rng = rng or np.random.default_rng()

    mask = (rng.random(kspace.shape, dtype=np.float32) > dropout_probability).astype(np.float32)
    corrupted = kspace * mask
    return CorruptionOutput(
        corrupted_kspace=corrupted,
        mask=mask,
        noise_map=None,
        requires_data_consistency=True,
        metadata={"dropout_probability": dropout_probability},
    )


def additive_complex_noise(
    kspace: np.ndarray,
    sigma: float = 0.03,
    rng: np.random.Generator | None = None,
    return_noise_map: bool = True,
) -> CorruptionOutput:
    """Add independent Gaussian noise to real and imaginary channels.

    This is a denoising corruption mode. Data consistency is disabled by default.
    """

    kspace = _ensure_complex64(kspace)
    rng = rng or np.random.default_rng()

    noise_real = rng.normal(0.0, sigma, size=kspace.shape).astype(np.float32)
    noise_imag = rng.normal(0.0, sigma, size=kspace.shape).astype(np.float32)
    noise = noise_real + 1j * noise_imag

    corrupted = kspace + noise
    mask = np.ones_like(kspace.real, dtype=np.float32)
    noise_map = np.full_like(kspace.real, fill_value=sigma, dtype=np.float32) if return_noise_map else None
    return CorruptionOutput(
        corrupted_kspace=corrupted.astype(np.complex64),
        mask=mask,
        noise_map=noise_map,
        requires_data_consistency=False,
        metadata={"sigma": sigma},
    )


_CORRUPTION_REGISTRY = {
    "low_frequency_corruption": low_frequency_corruption,
    "high_frequency_corruption": high_frequency_corruption,
    "random_dropout_corruption": random_dropout_corruption,
    "additive_complex_noise": additive_complex_noise,
}


def apply_corruption(
    kspace: np.ndarray,
    strategy: str,
    rng: np.random.Generator | None = None,
    **kwargs: Any,
) -> CorruptionOutput:
    """Apply a named corruption strategy and return structured outputs."""
    if strategy not in _CORRUPTION_REGISTRY:
        valid = ", ".join(sorted(_CORRUPTION_REGISTRY))
        raise ValueError(f"Unsupported corruption strategy: {strategy}. Valid values: {valid}")

    fn = _CORRUPTION_REGISTRY[strategy]
    if "rng" in fn.__code__.co_varnames:
        return fn(kspace, rng=rng, **kwargs)
    return fn(kspace, **kwargs)
