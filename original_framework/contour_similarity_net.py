"""
轻量级轮廓相似度网络：将单通道轮廓栅格图编码为 L2 归一化向量，
用余弦相似度映射到 [0,1] 作为相似度（同形经旋转/平移/小形变应接近 1）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContourEncoder(nn.Module):
    """小型 CNN 编码器，输入 (B,1,S,S) 轮廓图。"""

    def __init__(
        self,
        image_size: int = 128,
        latent_dim: int = 128,
        base_ch: int = 32,
    ) -> None:
        super().__init__()
        c = base_ch
        self.net = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(c * 4, c * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            nflat = self.net(torch.zeros(1, 1, image_size, image_size)).numel()
        self.fc = nn.Linear(int(nflat), latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).flatten(1)
        z = self.fc(z)
        return F.normalize(z, dim=-1, eps=1e-6)


class ContourSimilarityNet(nn.Module):
    """Siamese：共享 ContourEncoder，输出标量相似度 ∈ [0,1]。"""

    def __init__(
        self,
        image_size: int = 128,
        latent_dim: int = 128,
        base_ch: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = ContourEncoder(
            image_size=image_size,
            latent_dim=latent_dim,
            base_ch=base_ch,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z1 = self.encode(x1)
        z2 = self.encode(x2)
        cos = (z1 * z2).sum(dim=-1).clamp(-1.0 + 1e-5, 1.0 - 1e-5)
        sim = (cos + 1.0) * 0.5
        return sim, z1, z2


@torch.no_grad()
def predict_similarity(
    model: ContourSimilarityNet,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """a,b: (B,1,S,S) float tensor on same device as model."""
    model.eval()
    sim, _, _ = model(a, b)
    return sim
