from __future__ import annotations

"""FFT and complex-channel utilities for MRI reconstruction."""

import numpy as np
import torch


def split_complex_to_channels_np(kspace: np.ndarray) -> np.ndarray:
    """Convert complex numpy k-space ``[H,W]`` to two channels ``[2,H,W]``."""
    if not np.iscomplexobj(kspace):
        raise ValueError("Expected a complex-valued numpy array.")
    return np.stack((kspace.real, kspace.imag), axis=0).astype(np.float32, copy=False)


def combine_channels_to_complex_np(kspace_2ch: np.ndarray) -> np.ndarray:
    """Convert two-channel numpy k-space ``[2,H,W]`` to complex ``[H,W]``."""
    if kspace_2ch.shape[0] != 2:
        raise ValueError(f"Expected first channel dim == 2, got shape {kspace_2ch.shape}.")
    return kspace_2ch[0].astype(np.float32) + 1j * kspace_2ch[1].astype(np.float32)


def split_complex_to_channels_torch(kspace: torch.Tensor) -> torch.Tensor:
    """Convert complex torch k-space to real/imag channels."""
    if not torch.is_complex(kspace):
        raise ValueError("Expected a complex-valued torch tensor.")
    channel_dim = 0 if kspace.ndim == 2 else 1
    return torch.stack((kspace.real, kspace.imag), dim=channel_dim).float()


def combine_channels_to_complex_torch(kspace_2ch: torch.Tensor) -> torch.Tensor:
    """Convert two-channel torch k-space to complex tensor."""
    if kspace_2ch.ndim == 3:
        if kspace_2ch.shape[0] != 2:
            raise ValueError(f"Expected shape [2,H,W], got {tuple(kspace_2ch.shape)}.")
        return torch.complex(kspace_2ch[0], kspace_2ch[1])
    if kspace_2ch.ndim == 4:
        if kspace_2ch.shape[1] != 2:
            raise ValueError(f"Expected shape [B,2,H,W], got {tuple(kspace_2ch.shape)}.")
        return torch.complex(kspace_2ch[:, 0], kspace_2ch[:, 1])
    raise ValueError("Expected tensor rank 3 or 4.")


def fftshift2d(x: torch.Tensor) -> torch.Tensor:
    """Apply 2D FFT shift over spatial dimensions."""
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifftshift2d(x: torch.Tensor) -> torch.Tensor:
    """Apply inverse 2D FFT shift over spatial dimensions."""
    return torch.fft.ifftshift(x, dim=(-2, -1))


def ifft2c_torch(kspace_2ch: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Centered inverse FFT from two-channel k-space to complex image."""
    complex_kspace = combine_channels_to_complex_torch(kspace_2ch)
    shifted = ifftshift2d(complex_kspace)
    image = torch.fft.ifft2(shifted, dim=(-2, -1), norm=norm)
    return fftshift2d(image)


def fft2c_torch(image_complex: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Centered forward FFT from complex image to complex k-space."""
    shifted = ifftshift2d(image_complex)
    kspace = torch.fft.fft2(shifted, dim=(-2, -1), norm=norm)
    return fftshift2d(kspace)


def magnitude_image_from_kspace(kspace_2ch: torch.Tensor) -> torch.Tensor:
    """Reconstruct magnitude image from two-channel k-space."""
    image_complex = ifft2c_torch(kspace_2ch)
    return torch.abs(image_complex)


def kspace_magnitude(kspace_2ch: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute magnitude of two-channel k-space."""
    complex_kspace = combine_channels_to_complex_torch(kspace_2ch)
    return torch.sqrt(complex_kspace.real.square() + complex_kspace.imag.square() + eps)


def data_consistency(
    input_kspace_2ch: torch.Tensor,
    pred_kspace_2ch: torch.Tensor,
    mask_1ch: torch.Tensor,
) -> torch.Tensor:
    """Apply hard data consistency: ``mask*input + (1-mask)*prediction``."""
    if mask_1ch.ndim == pred_kspace_2ch.ndim - 1:
        mask_1ch = mask_1ch.unsqueeze(1)
    return mask_1ch * input_kspace_2ch + (1.0 - mask_1ch) * pred_kspace_2ch
