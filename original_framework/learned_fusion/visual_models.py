"""Lightweight visual candidate-quality and continuous residual fusion model."""
from __future__ import annotations

import torch
from torch import nn


class CandidateVisualEncoder(nn.Module):
    def __init__(self, channels: int = 4, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 12, 3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.SiLU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
        )

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        return self.net(maps)


class VisualQualityResidualFuser(nn.Module):
    """Shared visual backbone, domain heads, quality gate, and bbox residual.

    Domain 0 is chess and domain 1 is TrackVes. The domain indicator is an
    inference-safe dataset property, not a target-derived label.
    """

    def __init__(
        self, feature_dim: int, visual_channels: int = 4,
        visual_dim: int = 32, residual_limit: float = 0.25,
    ):
        super().__init__()
        self.visual = CandidateVisualEncoder(visual_channels, visual_dim)
        self.scalar = nn.Sequential(
            nn.Linear(feature_dim, visual_dim), nn.LayerNorm(visual_dim), nn.SiLU()
        )
        token_dim = visual_dim * 2
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=2, dim_feedforward=96, dropout=0.05,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=1)
        self.quality_heads = nn.ModuleList([nn.Linear(token_dim, 1) for _ in range(2)])
        self.residual_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_dim, 32), nn.SiLU(), nn.Linear(32, 4), nn.Tanh()
            )
            for _ in range(2)
        ])
        self.residual_limit = float(residual_limit)

    def forward_details(
        self, features: torch.Tensor, boxes: torch.Tensor, maps: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        b, k, c, h, w = maps.shape
        visual = self.visual(maps.reshape(b * k, c, h, w)).reshape(b, k, -1)
        tokens = self.context(torch.cat([self.scalar(features), visual], dim=-1))
        # feature 20 is domain_trackves and remains unstandardized.
        domain = (features[:, 0, 20] > 0.5).long()
        logits = torch.empty((b, k), device=features.device, dtype=features.dtype)
        residual_unit = torch.empty((b, 4), device=features.device, dtype=features.dtype)
        pooled = tokens.mean(dim=1)
        for domain_id in (0, 1):
            mask = domain == domain_id
            if mask.any():
                logits[mask] = self.quality_heads[domain_id](tokens[mask]).squeeze(-1)
                residual_unit[mask] = self.residual_heads[domain_id](pooled[mask])
        weights = torch.softmax(logits, dim=1)
        base = (weights.unsqueeze(-1) * boxes).sum(dim=1)
        previous = boxes[:, 3]
        limit = self.residual_limit
        pred = torch.stack(
            [
                base[:, 0] + limit * residual_unit[:, 0] * previous[:, 2],
                base[:, 1] + limit * residual_unit[:, 1] * previous[:, 3],
                base[:, 2] * torch.exp(limit * residual_unit[:, 2]),
                base[:, 3] * torch.exp(limit * residual_unit[:, 3]),
            ],
            dim=1,
        )
        return {
            "pred": pred, "weights": weights,
            "quality": torch.sigmoid(logits), "residual_unit": residual_unit,
        }

    def forward(
        self, features: torch.Tensor, boxes: torch.Tensor, maps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.forward_details(features, boxes, maps)
        return out["pred"], out["weights"]


def make_visual_model(
    feature_dim: int, cfg: dict, visual_channels: int | None = None
) -> VisualQualityResidualFuser:
    model_cfg = cfg["models"]
    return VisualQualityResidualFuser(
        feature_dim=feature_dim,
        visual_channels=(
            int(model_cfg["visual_channels"])
            if visual_channels is None else int(visual_channels)
        ),
        visual_dim=int(model_cfg["visual_dim"]),
        residual_limit=float(model_cfg["visual_residual_limit"]),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
