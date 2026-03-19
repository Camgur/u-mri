from __future__ import annotations

"""Vanilla 2D U-Net for k-space reconstruction with rectangular inputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv-BN-ReLU blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the double-convolution block."""
        return self.block(x)


class Down(nn.Module):
    """Downsampling block: max-pool followed by ``DoubleConv``."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample features and increase channel capacity."""
        return self.block(x)


class Up(nn.Module):
    """Upsampling block with skip connection fusion."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.bilinear = bilinear

    def forward(self, x_decoder: torch.Tensor, x_encoder: torch.Tensor) -> torch.Tensor:
        """Upsample decoder features, concatenate skip, and refine."""
        x_decoder = self.up(x_decoder)
        if x_decoder.shape[-2:] != x_encoder.shape[-2:]:
            x_decoder = F.interpolate(x_decoder, size=x_encoder.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x_encoder, x_decoder], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final ``1x1`` projection to output channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project features to prediction channels."""
        return self.conv(x)


class UNet2D(nn.Module):
    """Vanilla U-Net producing clean two-channel k-space predictions."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 2,
        base_channels: int = 32,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear=bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear=bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear=bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear=bilinear)
        self.outc = OutConv(base_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run full U-Net encoder-decoder pass."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
