"""
Lightweight U-Net implementation for segmentation.

The architecture is intentionally compact so it can load small checkpoints or
run in CPU-only environments. If you have a different model architecture, you
can swap this file while keeping the same API.
"""

from typing import List

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Matches the ConvBlock naming used in the original 2D-soil-segment checkpoints."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def up_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, num_classes: int = 4, input_channels: int = 3, base_filters: int = 32):
        super().__init__()
        self.num_classes = num_classes

        self.down1 = conv_block(input_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = conv_block(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = conv_block(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bridge = conv_block(base_filters * 4, base_filters * 8)

        self.up3 = up_block(base_filters * 8, base_filters * 4)
        self.dec3 = conv_block(base_filters * 8, base_filters * 4)

        self.up2 = up_block(base_filters * 4, base_filters * 2)
        self.dec2 = conv_block(base_filters * 4, base_filters * 2)

        self.up1 = up_block(base_filters * 2, base_filters)
        self.dec1 = conv_block(base_filters * 2, base_filters)

        self.classifier = nn.Conv2d(base_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.down1(x)
        p1 = self.pool1(c1)

        c2 = self.down2(p1)
        p2 = self.pool2(c2)

        c3 = self.down3(p2)
        p3 = self.pool3(c3)

        bridge = self.bridge(p3)

        u3 = self.up3(bridge)
        d3 = self.dec3(torch.cat([u3, c3], dim=1))

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, c1], dim=1))

        logits = self.classifier(d1)
        return logits


class DummySegmenter(nn.Module):
    """
    Fallback segmenter used when a checkpoint is missing or cannot be loaded.
    Produces a simple foreground mask based on image intensity.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input shape [B, C, H, W]
        gray = x.mean(dim=1, keepdim=True)
        # Expand to num_classes channels with a pseudo logit for background/foreground
        bg = -gray
        fg = gray
        logits = torch.cat([bg, fg], dim=1)
        return logits


class SimpleUNet(nn.Module):
    """
    Architecture matching the 2D-soil-segment training project (features [64,128,256,512]).
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 5, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        channels = in_channels
        for feature in features:
            self.encoder.append(ConvBlock(channels, feature))
            channels = feature

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.decoder.append(ConvBlock(feature * 2, feature))

        self.final_conv = nn.Conv2d(features[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.encoder:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(len(self.upconvs)):
            x = self.upconvs[idx](x)
            skip_connection = skips[idx]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=False)

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx](x)

        return self.final_conv(x)
