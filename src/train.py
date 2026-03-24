from __future__ import annotations

"""Training entrypoint for k-space reconstruction with dual-domain losses."""

import argparse
import json
import random
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

# treat the program as a package or as a script depending on how it's run
from . import config
from .data_loader import build_dataloaders
from .fft import data_consistency
from .losses import DualDomainLoss
from .model import UNet2D
from .visualization import plot_loss_curves, save_reconstruction_figure



def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_config: str) -> torch.device:
    """Resolve runtime device from config string."""
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_config)


def apply_optional_data_consistency(
    pred_kspace: torch.Tensor,
    corrupted_kspace: torch.Tensor,
    mask: torch.Tensor,
    requires_data_consistency: torch.Tensor | None,
) -> torch.Tensor:
    """Apply data consistency only for samples/modes that require it."""
    if not config.apply_data_consistency:
        return pred_kspace
    if requires_data_consistency is None:
        if not config.is_masking_corruption(config.corruption_strategy):
            return pred_kspace
        return data_consistency(corrupted_kspace, pred_kspace, mask)

    if requires_data_consistency.ndim == 0:
        requires_data_consistency = requires_data_consistency.unsqueeze(0)
    requires = requires_data_consistency.float().view(-1, 1, 1, 1)
    dc_out = data_consistency(corrupted_kspace, pred_kspace, mask)
    return requires * dc_out + (1.0 - requires) * pred_kspace


def compute_batch_loss(
    criterion: DualDomainLoss,
    pred_kspace: torch.Tensor,
    target_kspace: torch.Tensor,
    original_hw: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute loss, optionally cropping each sample to its original spatial size."""
    if original_hw is None:
        loss, metrics = criterion(pred_kspace, target_kspace)
        return loss, {k: float(v) for k, v in metrics.items()}

    if original_hw.ndim == 1:
        original_hw = original_hw.unsqueeze(0)

    losses = []
    metric_totals = {"loss_total": 0.0, "loss_kspace": 0.0, "loss_image": 0.0}
    for i in range(pred_kspace.shape[0]):
        h, w = int(original_hw[i, 0].item()), int(original_hw[i, 1].item())
        sample_pred = pred_kspace[i : i + 1, :, :h, :w]
        sample_target = target_kspace[i : i + 1, :, :h, :w]
        sample_loss, sample_metrics = criterion(sample_pred, sample_target)
        losses.append(sample_loss)
        for key in metric_totals:
            metric_totals[key] += float(sample_metrics[key])

    loss = torch.stack(losses).mean()
    count = float(len(losses))
    metrics = {k: v / count for k, v in metric_totals.items()}
    return loss, metrics


def run_epoch(
    model: torch.nn.Module,
    loader,
    criterion: DualDomainLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    """Run one training or validation epoch and return averaged metrics."""
    training = optimizer is not None
    model.train(training)

    totals = {"loss_total": 0.0, "loss_kspace": 0.0, "loss_image": 0.0}
    num_batches = 0

    for batch in loader:
        # move tensors to device and handle optional data consistency flag
        inputs = batch["input"].to(device, non_blocking=True)
        target_kspace = batch["target_kspace"].to(device, non_blocking=True)
        corrupted_kspace = batch["corrupted_kspace"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        original_hw = batch.get("original_hw")
        requires_dc = batch.get("requires_data_consistency")
        if isinstance(requires_dc, torch.Tensor):
            requires_dc = requires_dc.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            pred_kspace = model(inputs)
            final_kspace = apply_optional_data_consistency(pred_kspace, corrupted_kspace, mask, requires_dc)
            loss, metrics = compute_batch_loss(
                criterion=criterion,
                pred_kspace=final_kspace,
                target_kspace=target_kspace,
                original_hw=original_hw,
            )

            if training:
                loss.backward()
                if config.max_grad_norm is not None and config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

        for key in totals:
            totals[key] += float(metrics[key])
        num_batches += 1

    if num_batches == 0:
        return {key: float("nan") for key in totals}
    return {key: value / num_batches for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    history: dict[str, list[float]],
    best_val: float,
) -> None:
    """Serialize model, optimizer, scheduler, and metric history."""

    def _serialize_config_value(value):
        """Convert config values into checkpoint-safe primitive structures."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (list, tuple)):
            return [_serialize_config_value(item) for item in value]
        if isinstance(value, set):
            return sorted(_serialize_config_value(item) for item in value)
        if isinstance(value, Mapping):
            return {str(k): _serialize_config_value(v) for k, v in value.items()}
        return repr(value)

    config_snapshot = {}
    for key in dir(config):
        if not key.islower() or key.startswith("_"):
            continue
        value = getattr(config, key)
        if callable(value):
            continue
        config_snapshot[key] = _serialize_config_value(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "history": history,
        "best_val_loss": best_val,
        "config": config_snapshot,
    }
    torch.save(payload, path)


def save_epoch_visualization(model, val_loader, device: torch.device, epoch: int) -> None:
    """Save qualitative reconstruction panel for one validation batch."""
    model.eval()
    batch = next(iter(val_loader))
    with torch.no_grad():
        inputs = batch["input"].to(device)
        corrupted = batch["corrupted_kspace"].to(device)
        mask = batch["mask"].to(device)
        requires_dc = batch.get("requires_data_consistency")
        if isinstance(requires_dc, torch.Tensor):
            requires_dc = requires_dc.to(device)

        pred = model(inputs)
        pred_final = apply_optional_data_consistency(pred, corrupted, mask, requires_dc)
        original_hw = batch.get("original_hw")

    if isinstance(original_hw, torch.Tensor) and original_hw.shape[0] > 0:
        h, w = int(original_hw[0, 0].item()), int(original_hw[0, 1].item())
        corrupted_plot = corrupted[0, :, :h, :w].cpu()
        pred_plot = pred_final[0, :, :h, :w].cpu()
        target_plot = batch["target_kspace"][0, :, :h, :w]
    else:
        corrupted_plot = corrupted.cpu()
        pred_plot = pred_final.cpu()
        target_plot = batch["target_kspace"]

    fig_path = config.figure_dir / f"epoch_{epoch:04d}.png"
    save_reconstruction_figure(
        corrupted_kspace=corrupted_plot,
        predicted_kspace=pred_plot,
        target_kspace=target_plot,
        save_path=fig_path,
        title=f"Epoch {epoch}",
    )


def create_scheduler(optimizer: torch.optim.Optimizer):
    """Build LR scheduler from config."""
    if config.scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config.epochs)
    if config.scheduler_name == "step":
        return StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    if config.scheduler_name == "none":
        return None
    raise ValueError(f"Unsupported scheduler_name: {config.scheduler_name}")


def train(data_root: str, epochs: int) -> None:
    """Parse arguments and execute the full training loop."""

    set_seed(config.seed)
    device = resolve_device(config.device)

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.figure_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = build_dataloaders(data_root=data_root)
    sample_batch = next(iter(train_loader))
    in_channels = int(sample_batch["input"].shape[1])

    model = UNet2D(
        in_channels=in_channels,
        out_channels=2,
        base_channels=config.base_channels,
        bilinear=config.bilinear_upsampling,
    ).to(device)

    criterion = DualDomainLoss(
        lambda_k=config.lambda_k,
        lambda_img=config.lambda_img,
        kspace_loss_type=config.kspace_loss_type,
        image_loss_type=config.image_loss_type,
        ssim_weight=config.ssim_weight,
    )
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = create_scheduler(optimizer)

    history = {
        "train_loss_total": [],
        "train_loss_kspace": [],
        "train_loss_image": [],
        "val_loss_total": [],
        "val_loss_kspace": [],
        "val_loss_image": [],
    }
    best_val = float("inf")
    start = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = run_epoch(model, val_loader, criterion, None, device)

        if scheduler is not None:
            scheduler.step()

        history["train_loss_total"].append(train_metrics["loss_total"])
        history["train_loss_kspace"].append(train_metrics["loss_kspace"])
        history["train_loss_image"].append(train_metrics["loss_image"])
        history["val_loss_total"].append(val_metrics["loss_total"])
        history["val_loss_kspace"].append(val_metrics["loss_kspace"])
        history["val_loss_image"].append(val_metrics["loss_image"])

        print(
            f"Epoch {epoch:03d}/{epochs:03d} "
            f"| train_total={train_metrics['loss_total']:.6f} "
            f"| val_total={val_metrics['loss_total']:.6f} "
            f"| train_k={train_metrics['loss_kspace']:.6f} "
            f"| train_img={train_metrics['loss_image']:.6f}"
        )

        if val_metrics["loss_total"] < best_val:
            best_val = val_metrics["loss_total"]
            save_checkpoint(
                config.checkpoint_dir / "best.pt",
                epoch,
                model,
                optimizer,
                scheduler,
                history,
                best_val,
            )

        save_checkpoint(
            config.checkpoint_dir / "last.pt",
            epoch,
            model,
            optimizer,
            scheduler,
            history,
            best_val,
        )

        if epoch % config.visualize_every == 0 or epoch == 1 or epoch == epochs:
            save_epoch_visualization(model, val_loader, device, epoch)

    plot_loss_curves(history, config.figure_dir / "loss_curves.png")
    summary = {
        "best_val_loss": best_val,
        "num_epochs": epochs,
        "elapsed_seconds": time.time() - start,
        "checkpoint_dir": str(config.checkpoint_dir),
        "figure_dir": str(config.figure_dir),
    }
    with open(config.output_root / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Training done. Best validation loss: {best_val:.6f}")


if __name__ == "__main__":
    train(data_root="00_dataset", epochs=100)
