"""Hybrid Multi-Scale Attention Network (HMSAN) implementation."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    resnet50,
)

from models.attention import CBAMBlock, SEBlock
from models.grl import GradientReversalLayer


class MultiScaleModule(nn.Module):
    """Parallel multi-scale convolutions with 1x1, 3x3, and 5x5 kernels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch_1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch_3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch_5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm2d(out_channels * 3)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat(
            [self.branch_1x1(x), self.branch_3x3(x), self.branch_5x5(x)], dim=1
        )
        return self.act(self.bn(out))


class ViTContextModule(nn.Module):
    """Transformer encoder over flattened spatial tokens from feature maps."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        depth: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> tokens: [B, H*W, C]
        tokens = x.flatten(2).transpose(1, 2)
        tokens = self.proj(tokens)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        pooled = tokens.mean(dim=1)
        return pooled


class HMSAN(nn.Module):
    """Hybrid Multi-Scale Attention Network for plant disease recognition."""

    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet50",
        pretrained: bool = True,
        transformer_dim: int = 256,
        use_domain_adaptation: bool = False,
        num_domains: int = 2,
        grl_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.use_domain_adaptation = use_domain_adaptation

        self.backbone, backbone_channels = self._build_backbone(backbone, pretrained)

        self.multi_scale = MultiScaleModule(backbone_channels, out_channels=256)
        f2_channels = 256 * 3

        self.se = SEBlock(channels=f2_channels)
        self.cbam = CBAMBlock(channels=f2_channels)

        self.vit = ViTContextModule(
            in_channels=f2_channels,
            embed_dim=transformer_dim,
            num_heads=8,
            depth=2,
            dropout=0.1,
        )

        # F1 pooled + F2 pooled + F3 vector
        fused_dim = backbone_channels + f2_channels + transformer_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        if self.use_domain_adaptation:
            self.grl = GradientReversalLayer(lambd=grl_lambda)
            self.domain_classifier = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(256, num_domains),
            )

    @staticmethod
    def _build_backbone(backbone: str, pretrained: bool) -> tuple[nn.Module, int]:
        if backbone.lower() == "resnet50":
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            net = resnet50(weights=weights)
            layers = [
                net.conv1,
                net.bn1,
                net.relu,
                net.maxpool,
                net.layer1,
                net.layer2,
                net.layer3,
                net.layer4,
            ]
            return nn.Sequential(*layers), 2048

        if backbone.lower() == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = efficientnet_b0(weights=weights)
            return net.features, 1280

        raise ValueError("backbone must be either 'resnet50' or 'efficientnet_b0'")

    def extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # F1 from backbone feature maps
        f1_map = self.backbone(x)
        f1 = F.adaptive_avg_pool2d(f1_map, output_size=1).flatten(1)

        # F2 from multi-scale module
        f2_map = self.multi_scale(f1_map)

        # F4 attention refinement applied to multi-scale features
        f4_map = self.se(f2_map)
        f4_map = self.cbam(f4_map)
        f2 = F.adaptive_avg_pool2d(f4_map, output_size=1).flatten(1)

        # F3 from transformer over attention-refined map
        f3 = self.vit(f4_map)

        fused = torch.cat([f1, f2, f3], dim=1)
        fused = self.fusion(fused)

        return {
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "f_fused": fused,
            "feature_map": f4_map,
        }

    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
    ) -> Dict[str, torch.Tensor]:
        feats = self.extract_features(x)
        logits = self.classifier(feats["f_fused"])

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "probs": torch.softmax(logits, dim=1) if return_probs else logits,
            **feats,
        }

        if self.use_domain_adaptation:
            domain_features = self.grl(feats["f_fused"])
            out["domain_logits"] = self.domain_classifier(domain_features)

        return out
