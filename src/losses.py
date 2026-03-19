from __future__ import annotations

"""Loss functions for dual-domain k-space and image-space supervision."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .fft import magnitude_image_from_kspace
except ImportError:
    from fft import magnitude_image_from_kspace


def _base_loss(pred: torch.Tensor, target: torch.Tensor, loss_type: str) -> torch.Tensor:
    """Dispatch to L1 or MSE loss."""
    if loss_type == "l1":
        return F.l1_loss(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"Unsupported loss type: {loss_type}")


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute a lightweight differentiable SSIM loss term."""
    c1 = 0.01**2
    c2 = 0.03**2
    kernel_size = 3
    pad = kernel_size // 2

    mu_x = F.avg_pool2d(pred, kernel_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, kernel_size, stride=1, padding=pad)

    sigma_x = F.avg_pool2d(pred * pred, kernel_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size, stride=1, padding=pad) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / (denominator + 1e-8)
    return 1.0 - ssim_map.mean()


def kspace_loss(
    reconstructed_kspace: torch.Tensor,
    target_kspace: torch.Tensor,
    loss_type: str = "l1",
) -> torch.Tensor:
    """Compute loss directly in complex-channel k-space."""
    return _base_loss(reconstructed_kspace, target_kspace, loss_type=loss_type)


def image_space_loss(
    reconstructed_kspace: torch.Tensor,
    target_kspace: torch.Tensor,
    loss_type: str = "l1",
    ssim_weight: float = 0.0,
) -> torch.Tensor:
    """Compute image-domain loss on IFFT magnitude reconstructions."""
    pred_img = magnitude_image_from_kspace(reconstructed_kspace)
    target_img = magnitude_image_from_kspace(target_kspace)

    if pred_img.ndim == 3:
        pred_img = pred_img.unsqueeze(1)
    if target_img.ndim == 3:
        target_img = target_img.unsqueeze(1)

    pixel_loss = _base_loss(pred_img, target_img, loss_type=loss_type)
    if ssim_weight > 0.0:
        return pixel_loss + ssim_weight * _ssim_loss(pred_img, target_img)
    return pixel_loss


class DualDomainLoss(nn.Module):
    """Weighted combination of k-space and image-space losses."""

    def __init__(
        self,
        lambda_k: float = 1.0,
        lambda_img: float = 1.0,
        kspace_loss_type: str = "l1",
        image_loss_type: str = "l1",
        ssim_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_k = lambda_k
        self.lambda_img = lambda_img
        self.kspace_loss_type = kspace_loss_type
        self.image_loss_type = image_loss_type
        self.ssim_weight = ssim_weight

    def forward(
        self,
        reconstructed_kspace: torch.Tensor,
        target_kspace: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return total loss and detached metric components."""
        loss_k = kspace_loss(
            reconstructed_kspace,
            target_kspace,
            loss_type=self.kspace_loss_type,
        )
        loss_img = image_space_loss(
            reconstructed_kspace,
            target_kspace,
            loss_type=self.image_loss_type,
            ssim_weight=self.ssim_weight,
        )

        total = self.lambda_k * loss_k + self.lambda_img * loss_img
        metrics = {
            "loss_total": total.detach(),
            "loss_kspace": loss_k.detach(),
            "loss_image": loss_img.detach(),
        }
        return total, metrics
