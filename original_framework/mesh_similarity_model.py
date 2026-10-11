#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云对 → 相似度回归 [0,1]：同 target 不同视点/遮挡 → 高；异 target → 低。

结构：PointNet 风格编码器（每点 MLP + max pool）→ 双支路嵌入；
     可选 L2 归一化嵌入；concat(e1,e2,|e1-e2|) 上可选 dropout 后再 MLP + Sigmoid。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """(B, N, 3) 或 (B, 3, N) → (B, d_out) 全局描述子（未归一化，Head 内可再约束）。"""

    def __init__(self, in_dim: int = 3, d_out: int = 256, width: int = 128):
        super().__init__()
        self.in_dim = int(in_dim)
        self.d_out = int(d_out)
        w = int(width)
        self.backbone = nn.Sequential(
            nn.Conv1d(self.in_dim, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, w, 1),
            nn.BatchNorm1d(w),
            nn.ReLU(inplace=True),
            nn.Conv1d(w, w, 1),
            nn.BatchNorm1d(w),
            nn.ReLU(inplace=True),
            nn.Conv1d(w, self.d_out, 1),
            nn.BatchNorm1d(self.d_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected 3D tensor, got {tuple(x.shape)}")
        if x.shape[1] == self.in_dim:
            pass
        elif x.shape[2] == self.in_dim:
            x = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"need (B,N,{self.in_dim}) or (B,{self.in_dim},N), got {tuple(x.shape)}")
        f = self.backbone(x)
        g = torch.max(f, dim=2)[0]
        return g


class MeshSimilarityNet(nn.Module):
    """
    输入两团点云（已归一化到大致零均值单位球），输出标量相似度 ∈ (0,1)。
    """

    def __init__(
        self,
        in_dim: int = 3,
        embed_dim: int = 256,
        head_hidden: int = 256,
        encoder_width: int = 128,
        *,
        l2_normalize_embed: bool = False,
        head_dropout: float = 0.0,
    ):
        super().__init__()
        self.l2_normalize_embed = bool(l2_normalize_embed)
        self.head_dropout = float(max(0.0, head_dropout))
        self.enc = PointNetEncoder(in_dim=in_dim, d_out=embed_dim, width=int(encoder_width))
        h = int(head_hidden)
        d = int(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(d * 3, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, h // 2),
            nn.ReLU(inplace=True),
            nn.Linear(h // 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, pts: torch.Tensor) -> torch.Tensor:
        return self.enc(pts)

    def forward(self, pts_a: torch.Tensor, pts_b: torch.Tensor) -> torch.Tensor:
        """
        pts_* : (B, N, 3)
        returns: (B,) similarity in [0,1]
        """
        e1 = self.encode(pts_a)
        e2 = self.encode(pts_b)
        if self.l2_normalize_embed:
            e1 = l2_normalize_rows(e1)
            e2 = l2_normalize_rows(e2)
        x = torch.cat([e1, e2, (e1 - e2).abs()], dim=1)
        if self.head_dropout > 0.0:
            x = F.dropout(x, p=self.head_dropout, training=self.training)
        return self.head(x).squeeze(-1)


def l2_normalize_rows(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return z / (z.norm(dim=1, keepdim=True).clamp_min(eps))
