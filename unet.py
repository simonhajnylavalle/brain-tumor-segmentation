"""
Compact 2D U-Net architecture in PyTorch for binary brain tumor segmentation.

Input shape:  (B, in_channels, H, W)  -> 1 or 2 MRI modalities (e.g. FLAIR, T1gd)
Output shape: (B, 1, H, W)            -> Raw foreground logit per pixel (no sigmoid!)

Default `features=(16, 32, 64, 128)` has ~1.94M parameters (compact and fast on MPS/CPU
for 150-case controlled experiments). Pass `features=(32, 64, 128, 256)` for ~7.76M.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 padded convolutions, each followed by BatchNorm2d and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Compact Encoder-Decoder 2D U-Net with skip connections."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        features: tuple[int, ...] = (16, 32, 64, 128),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (contracting path)
        ch = in_channels
        for feat in features:
            self.downs.append(DoubleConv(ch, feat))
            ch = feat

        # Bottleneck (deepest semantic context)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=dropout)

        # Decoder (expanding path)
        for feat in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feat * 2, feat))

        # Final 1x1 convolution mapping features -> raw binary logit(s)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections: list[torch.Tensor] = []

        # Contracting path
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Expanding path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # ConvTranspose2d upsample (2x H, 2x W)
            skip = skip_connections[idx // 2]

            # Safety guard if input H or W is not evenly divisible by 2**len(features)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat((skip, x), dim=1)  # Concatenate along channel dimension
            x = self.ups[idx + 1](x)         # Fuse skip + upsampled features

        return self.final_conv(x)
