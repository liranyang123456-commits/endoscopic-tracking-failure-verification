# -*- coding: utf-8 -*-
"""
每帧预测 ContourSim 在 coarse/fine/greedy 与大λ融合中的有效权重 contour_fuse_weight ∈ [0,1]。

- 默认使用纯 NumPy 逻辑回归式 sigmoid（无需 ckpt），便于开箱运行。
- 可选 MIS_FUSION_HEAD_NPZ 指向 np.savez 文件，键: W shape (F,), b — z = sigmoid(W·x + b)
- 离线拟合：用 MIS_DEEP_DEBUG_ROWS=1 跑评测生成 nano_deep_rows.csv，再运行
  fit_fusion_weight_head.py 或 deep_stack_one_shot.py
"""
from __future__ import annotations

import os

import numpy as np

_FUSION_FEAT_DIM = 6


def fusion_feature_vector(
    nano_peak_conf: float,
    prior_min: float,
    prior_max: float,
    adaptive_alpha: float,
    contour_hist_mean: float,
    contour_hist_std: float,
) -> np.ndarray:
    """
    与训练脚本 export 时一致的 6 维特征（均已压到 [0,1] 量级）。
    """
    span = float(max(0.0, prior_max - prior_min))
    x = np.array(
        [
            float(np.clip(nano_peak_conf, 0.0, 1.0)),
            float(np.clip(span, 0.0, 1.0)),
            float(np.clip(adaptive_alpha / 2.5, 0.0, 1.0)),
            float(np.clip(contour_hist_mean, 0.0, 1.0)),
            float(np.clip(contour_hist_std / 0.35, 0.0, 1.0)),
            float(
                np.clip((contour_hist_mean - 0.48) / 0.25 + 0.5, 0.0, 1.0)
            ),  # 相对 LOW_CONF_T 的偏移
        ],
        dtype=np.float32,
    )
    assert x.shape[0] == _FUSION_FEAT_DIM
    return x


def default_fusion_head_params() -> tuple[np.ndarray, float]:
    """
    启发式权重（未经离线训练亦可工作）：
    - Nano 置信高、先验尖锐、近期 Contour 可靠 → 提高 contour 权重
    - α 大（已弱化 Nano 先验补偿）→ 降低 contour 权重，更多交给 Hist
    """
    W = np.array(
        [1.25, 0.65, -0.85, 0.95, -0.35, -0.55],
        dtype=np.float32,
    )
    b = -0.35
    return W, float(b)


def load_fusion_head_npz(path: str | None) -> tuple[np.ndarray | None, float | None]:
    if not path or not os.path.isfile(path):
        return None, None
    z = np.load(path)
    W = z["W"].astype(np.float32).reshape(-1)
    b = float(z["b"])
    if W.shape[0] != _FUSION_FEAT_DIM:
        raise ValueError(f"fusion head W dim {W.shape[0]} != {_FUSION_FEAT_DIM}")
    return W, b


def predict_contour_fuse_weight(feat: np.ndarray) -> float:
    """
    feat: shape (6,) float32
    返回 [w_min, w_max] 内的 contour_fuse_weight
    """
    path = os.environ.get("MIS_FUSION_HEAD_NPZ", "").strip()
    Wb, bb = load_fusion_head_npz(path)
    if Wb is None:
        Wb, bb = default_fusion_head_params()
    z = float(np.dot(Wb, feat.astype(np.float32)) + bb)
    # 数值稳定的 sigmoid
    if z >= 0:
        s = 1.0 / (1.0 + np.exp(-z))
    else:
        ez = np.exp(z)
        s = ez / (1.0 + ez)
    w_min = float(os.environ.get("MIS_DF_CONT_W_MIN", "0.12"))
    w_max = float(os.environ.get("MIS_DF_CONT_W_MAX", "1.0"))
    return float(np.clip(s, w_min, w_max))
