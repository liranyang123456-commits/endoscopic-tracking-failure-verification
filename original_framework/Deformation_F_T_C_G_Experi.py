#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deformation_F_T_C_G_Experi.py
──────────────────────────────
Contour-only tracker using SwinUNetLarge (Autoencoder_129.pth) bottleneck features.

Overall logic (copied from MIS_C_F_SIMI_Mesh_Match_Track_3DMatch.py):
  1. Config via core.TrackConfig + environment variables (MIS_FRAMES_DIR, MIS_OUTPUT_DIR, …)
  2. First frame: selectROI / MIS_INIT_BBOX → target bbox
  3. Library: global th68 → crop → RTW augmentation (128 views) → SwinUNetLarge encode
     Library is FIXED for the entire sequence (no update).
  4. Per-frame tracking loop:
       a. global th68 ONCE on full frame
       b. extract expanded-ROI (expand=64) from global th68
       c. sliding-window (step=4) → candidate patches
       d. batch-encode via SwinUNetLarge bottleneck
       e. cosine similarity vs. library → best match
       f. adaptive expansion: if score < 0.50, retry with expand=80
  5. Outputs: frames_with_bbox/, roi/, th_gray/, features/,
             tracking_bboxes_and_scores.json, diag.csv

Model: SwinUNetLarge from Copy12.py, weights D:/Autoencoder_129.pth
"""

from __future__ import annotations

import csv
import glob
import json
import os
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─── 导入轮廓相似度网络 ──────────────────────────────────────────────
from train_contour_similarity import ContourSimilarityNet, _rasterize_contour


def _cosine_sim_safe(enc: np.ndarray, lib: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix (enc @ lib.T) using torch to avoid
    numpy MKL/BLAS stack overflow on large embedding dimensions (e.g. 16384).
    Returns max score per row as float32 ndarray of shape (n_enc,).
    """
    et = torch.from_numpy(np.ascontiguousarray(enc)).float()
    lt = torch.from_numpy(np.ascontiguousarray(lib)).float()
    return (et @ lt.T).numpy().max(axis=1).astype(np.float32)

# ─── SwinUNetLarge (Autoencoder_129.pth 的实际架构) ────────────────────────────
from Copy12 import SwinUNetLarge

# ─── core 工具（来自 MIS_C_F_SIMI_Mesh_Match_Track_3DMatch.py 所依赖的同一模块）──
import MIS_C_Fe_Simi_Mesh_Match_Pose_Estimate_Track as core

# ═════════════════════════════════════════════════════════════════════════════
# 环境变量工具（与 MIS_C_F_SIMI_Mesh_Match_Track_3DMatch.py 保持一致）
# ═════════════════════════════════════════════════════════════════════════════
def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or default).strip().strip('"').strip("'")

def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except Exception:
        return int(default)

def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except Exception:
        return float(default)

def _env_bool(name: str, default: bool) -> bool:
    v = _env(name, "")
    if v == "":
        return bool(default)
    return v.lower() not in ("0", "false", "no", "off")

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _now() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ═════════════════════════════════════════════════════════════════════════════
# 模型加载 (ContourSimilarityNet)
# ═════════════════════════════════════════════════════════════════════════════
_CONTOUR_SIM_MODEL_PATH = r"E:\contour_sim_runs\chess_v1\contour_similarity_best.pt"

# 全局变量，供compare_iscience_tracking.py使用
contour_sim_model = None
def _load_contour_similarity_model(ckpt_path: str) -> ContourSimilarityNet:
    """Load pre-trained ContourSimilarityNet for direct contour similarity scoring."""
    if not os.path.isfile(ckpt_path):
        print(f"[WARN] ContourSimilarity model not found: {ckpt_path}")
        return None
    model = ContourSimilarityNet()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()
    print(f"[ContourSim] Loaded from {ckpt_path}")
    return model

# 加载模型到全局变量
contour_sim_model = _load_contour_similarity_model(_CONTOUR_SIM_MODEL_PATH)
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════════════
def _apply_th68(img_bgr: np.ndarray, th: int = 68) -> np.ndarray:
    """固定阈值 th 二值化（pixel-independent，全局/局部结果相同）。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


# 预计算归一化常量（避免每次创建）
_NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_NORM_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def encode_contour_patches(
    patches_bgr: list,
    model: Any,
    device: Any,
    transform_fn: Any,          # 保留参数兼容旧接口，内部已不使用
    bottleneck_cache: list,
    batch_size: int = 64,
) -> np.ndarray:
    """
    向量化版：消除 PIL/单循环 overhead。
    CUDA fp16: 自动将 batch_t 转为 half，底层一次 float32 转换再 L2。
    """
    _fp16 = getattr(model, "_use_fp16", False)
    encodings = []
    for start in range(0, len(patches_bgr), batch_size):
        chunk = patches_bgr[start: start + batch_size]
        arr = np.stack([cv2.resize(p, (64, 64)) for p in chunk])   # (N,64,64,3) BGR uint8
        arr = arr[:, :, :, ::-1].astype(np.float32) / 255.0         # RGB [0,1]
        arr = (arr - _NORM_MEAN) / _NORM_STD                         # normalize
        batch_t = torch.from_numpy(
            np.ascontiguousarray(arr.transpose(0, 3, 1, 2))          # (N,3,64,64)
        ).to(device)
        if _fp16:
            batch_t = batch_t  # 模型保持 float32，autocast 内部自动处理
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", enabled=_fp16):
                model(batch_t)
        raw = bottleneck_cache[0]
        feat = raw.float().cpu().numpy().reshape(len(chunk), -1).astype(np.float32)
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        encodings.append(feat / norms)
    return np.concatenate(encodings, axis=0)


def _build_rtw_bgr(
    target_roi: np.ndarray,
    mask: np.ndarray,
    num_transforms: int = 128,
) -> list:
    """
    RTW 增广: 旋转·平移·缩放·剪切·透视·翻转·组合 → 最多 num_transforms 张
    输入 target_roi: BGR uint8 (th68 二值图像).
    直接返回 list[BGR ndarray]，不做 tensor 转换，供 encode_contour_patches 直接编码。
    避免 tensor→uint8→tensor 双重归一化导致的特征域错位。
    
    重要：所有变换后，必须确保目标区域（mask白色部分）完整保留在画布内，
    不被边界裁剪。通过扩大画布 + 检查mask完整性来实现。
    """
    h, w = target_roi.shape[:2]
    imgs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    
    # 扩大画布：2倍尺寸，确保旋转/平移后目标不被裁剪
    pad_w, pad_h = w * 2, h * 2
    
    # 将目标放在画布中心
    def _place_in_center(img):
        canvas = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
        cy, cx = pad_h // 2 - h // 2, pad_w // 2 - w // 2
        canvas[cy:cy+h, cx:cx+w] = img
        return canvas
    
    target_centered = _place_in_center(target_roi)
    mask_centered = _place_in_center(np.stack([mask]*3, axis=-1))[:,:,0]  # mask是单通道
    
    # 原始mask的白色像素数（用于完整性检查）
    orig_white = np.count_nonzero(mask > 0)

    def _waff(M):
        warped = cv2.warpAffine(target_centered, M, (pad_w, pad_h))
        warped_mask = cv2.warpAffine(mask_centered, M, (pad_w, pad_h))
        # 检查mask完整性：变换后的白色像素数必须 >= 80% 原始
        if np.count_nonzero(warped_mask > 0) >= orig_white * 0.8:
            imgs.append(warped)
            masks.append(warped_mask)

    def _wpersp(M):
        warped = cv2.warpPerspective(target_centered, M, (pad_w, pad_h))
        warped_mask = cv2.warpPerspective(mask_centered, M, (pad_w, pad_h))
        if np.count_nonzero(warped_mask > 0) >= orig_white * 0.8:
            imgs.append(warped)
            masks.append(warped_mask)

    # 1. Rotation -60…+60 step 5 (25)
    for ang in range(-60, 65, 5):
        _waff(cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0))
    # 2. Scale (13)
    for sc in [0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55]:
        _waff(cv2.getRotationMatrix2D((w / 2, h / 2), 0, sc))
    # 3. Translate X (8)
    for tx in [-0.20,-0.15,-0.10,-0.05,0.05,0.10,0.15,0.20]:
        _waff(np.float32([[1,0,tx*w],[0,1,0]]))
    # 4. Translate Y (8)
    for ty in [-0.20,-0.15,-0.10,-0.05,0.05,0.10,0.15,0.20]:
        _waff(np.float32([[1,0,0],[0,1,ty*h]]))
    # 5. Diagonal translate (8)
    for (tx,ty) in [(-0.15,-0.15),(-0.15,0.15),(0.15,-0.15),(0.15,0.15),
                    (-0.10,-0.10),(-0.10,0.10),(0.10,-0.10),(0.10,0.10)]:
        _waff(np.float32([[1,0,tx*w],[0,1,ty*h]]))
    # 6. Shear X (6)
    for sx in [-0.30,-0.20,-0.10,0.10,0.20,0.30]:
        _waff(np.float32([[1,sx,0],[0,1,0]]))
    # 7. Shear Y (6)
    for sy in [-0.30,-0.20,-0.10,0.10,0.20,0.30]:
        _waff(np.float32([[1,0,0],[sy,1,0]]))
    # 8. Perspective 4×4=16
    src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    for ci in range(4):
        for st in [0.05,0.10,0.15,0.20]:
            dst = src.copy()
            dw, dh = int(st*w), int(st*h)
            if   ci==0: dst[0]=[dw,dh]
            elif ci==1: dst[1]=[w-1-dw,dh]
            elif ci==2: dst[2]=[w-1-dw,h-1-dh]
            else:       dst[3]=[dw,h-1-dh]
            _wpersp(cv2.getPerspectiveTransform(src, dst))
    # 9. Flip (2)
    for fc in [1, 0]:
        flipped = cv2.flip(target_centered, fc)
        flipped_mask = cv2.flip(mask_centered, fc)
        if np.count_nonzero(flipped_mask > 0) >= orig_white * 0.8:
            imgs.append(flipped)
            masks.append(flipped_mask)
    # 10. Rotation+Translation combined (8)
    for (ang,tx,ty) in [(-30,0.10,0),(30,-0.10,0),(0,0.10,0.10),(0,-0.10,-0.10),
                        (-15,0.10,0.10),(15,-0.10,0.10),(-15,-0.10,-0.10),(15,0.10,-0.10)]:
        M = cv2.getRotationMatrix2D((w/2,h/2), ang, 1.0)
        M[0,2] += tx*w; M[1,2] += ty*h
        _waff(M)
    # 11. Extended rotation -90..-63 and 63..90, step 3 (~21)
    for ang in list(range(-90, -60, 3)) + list(range(63, 93, 3)):
        _waff(cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0))
    # 12. Rotation + Scale combined 8 angles × 4 scales (32)
    for ang in [-60, -45, -30, -15, 15, 30, 45, 60]:
        for sc in [0.70, 0.85, 1.15, 1.30]:
            _waff(cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc))
    # 13. Extended translate X/Y ±[0.25, 0.30] (8)
    for tx in [-0.30, -0.25, 0.25, 0.30]:
        _waff(np.float32([[1, 0, tx*w], [0, 1, 0]]))
    for ty in [-0.30, -0.25, 0.25, 0.30]:
        _waff(np.float32([[1, 0, 0], [0, 1, ty*h]]))
    # 14. Perspective extended strengths 4×2 (8)
    for ci in range(4):
        for st in [0.25, 0.30]:
            dst = src.copy()
            dw, dh = int(st*w), int(st*h)
            if   ci==0: dst[0]=[dw,dh]
            elif ci==1: dst[1]=[w-1-dw,dh]
            elif ci==2: dst[2]=[w-1-dw,h-1-dh]
            else:       dst[3]=[dw,h-1-dh]
            _wpersp(cv2.getPerspectiveTransform(src, dst))
    # 15. Shear X+Y combined (12)
    for (sx, sy) in [(-0.20,-0.20),(-0.20, 0.20),(0.20,-0.20),(0.20, 0.20),
                     (-0.15, 0.15),(0.15,-0.15),(-0.10, 0.10),(0.10,-0.10),
                     (-0.25, 0.10),(0.25,-0.10),(-0.10, 0.25),(0.10,-0.25)]:
        _waff(np.float32([[1, sx, 0], [sy, 1, 0]]))
    # 16. Scale extreme values (8)
    for sc in [0.20, 0.25, 0.30, 1.65, 1.75, 1.85, 1.95, 2.0]:
        _waff(cv2.getRotationMatrix2D((w / 2, h / 2), 0, sc))
    # 17. Flip + Rotation combined (22)
    for fc in [1, 0]:
        for ang in [-60, -45, -30, -15, 0, 15, 30, 45, 60, -75, 75]:
            img_f  = cv2.flip(target_centered, fc)
            mask_f = cv2.flip(mask_centered, fc)
            M = cv2.getRotationMatrix2D((pad_w/2, pad_h/2), ang, 1.0)
            warped = cv2.warpAffine(img_f, M, (pad_w, pad_h))
            warped_mask = cv2.warpAffine(mask_f, M, (pad_w, pad_h))
            if np.count_nonzero(warped_mask > 0) >= orig_white * 0.8:
                imgs.append(warped)
                masks.append(warped_mask)
    # 18. Rotation + Translation + Scale combined (24)
    for (ang, tx, ty, sc) in [
        (-45, 0.15, 0.10, 0.90), ( 45,-0.15,-0.10, 0.90),
        (-30, 0.10, 0.15, 1.10), ( 30,-0.10,-0.15, 1.10),
        (-60,-0.10, 0.10, 0.85), ( 60, 0.10,-0.10, 0.85),
        ( 15, 0.20, 0.05, 1.20), (-15,-0.20,-0.05, 1.20),
        ( 30, 0.05, 0.20, 0.80), (-30,-0.05,-0.20, 0.80),
        ( 45,-0.15, 0.15, 1.15), (-45, 0.15,-0.15, 1.15),
        ( 20, 0.12,-0.12, 0.95), (-20,-0.12, 0.12, 0.95),
        ( 10,-0.18, 0.08, 1.05), (-10, 0.18,-0.08, 1.05),
        ( 50, 0.08, 0.08, 0.75), (-50,-0.08,-0.08, 0.75),
        ( 25,-0.05, 0.15, 1.25), (-25, 0.05,-0.15, 1.25),
        ( 35, 0.18,-0.05, 1.00), (-35,-0.18, 0.05, 1.00),
        ( 55,-0.12, 0.12, 0.80), (-55, 0.12,-0.12, 0.80),
    ]:
        M = cv2.getRotationMatrix2D((w/2, h/2), ang, sc)
        M[0,2] += tx*w; M[1,2] += ty*h
        _waff(M)

    imgs = imgs[:num_transforms]
    print(f"[Library] template count: {len(imgs)}")
    return imgs  # raw BGR list — no tensor conversion


def find_best_match_sliding(
    search_plane_th68: np.ndarray,
    lib_encodings: np.ndarray,
    target_h: int,
    target_w: int,
    encode_fn,
    step_size: int = 4,
    spatial_sigma: float = 3.0,
    center_yx: tuple | None = None,
    min_white_ratio: float = 0.03,
) -> dict:
    """
    滑动窗口匹配。
    min_white_ratio: 积分图预过滤——候选中白色像素占比小于此值则跳过模型推理（默认 3%）。
    spatial_sigma:   高斯先验，单位 = target size。
    center_yx:       上一帧 bbox 中心坐标，None=居中。
    """
    eh, ew = search_plane_th68.shape[:2]
    ys = list(range(0, max(1, eh - target_h + 1), step_size))
    xs = list(range(0, max(1, ew - target_w + 1), step_size))
    if ys and (ys[-1] + target_h) < eh: ys.append(eh - target_h)
    if xs and (xs[-1] + target_w) < ew: xs.append(ew - target_w)
    if not ys: ys = [0]
    if not xs: xs = [0]

    # ─ 高斯先验参数 ────────────────────────────────────────
    cy = float(center_yx[0]) if center_yx is not None else eh / 2.0
    cx = float(center_yx[1]) if center_yx is not None else ew / 2.0
    sigma_y = max(1.0, spatial_sigma * target_h)
    sigma_x = max(1.0, spatial_sigma * target_w)

    # ─ 积分图预过滤（SAT）—— O(1) 判断白色像素数 ─────────────────
    min_count = min_white_ratio * target_h * target_w
    plane_bin = (search_plane_th68[:, :, 0] > 0).astype(np.float32) \
                if search_plane_th68.ndim == 3 \
                else (search_plane_th68 > 0).astype(np.float32)
    sat = cv2.integral(plane_bin)    # (H+1, W+1)

    def _white_cnt(wy, wx):
        return (sat[wy + target_h, wx + target_w]
                - sat[wy, wx + target_w]
                - sat[wy + target_h, wx]
                + sat[wy, wx])

    # 过滤后的候选位置
    positions = [(wy, wx) for wy in ys for wx in xs if _white_cnt(wy, wx) >= min_count]
    if not positions:       # 全部被过滤：回退到中心
        positions = [(max(0, min(ys[-1], int(cy - target_h//2))),
                      max(0, min(xs[-1], int(cx - target_w//2))))]
    n_filtered  = len(ys) * len(xs) - len(positions)
    n_total     = len(ys) * len(xs)

    # ─ 提取 patch 列表 ──────────────────────────────────────────
    patches = []
    for (wy, wx) in positions:
        p = search_plane_th68[wy:wy+target_h, wx:wx+target_w]
        if p.shape[0] < target_h or p.shape[1] < target_w:
            p = cv2.copyMakeBorder(p, 0, max(0, target_h-p.shape[0]),
                                   0, max(0, target_w-p.shape[1]),
                                   cv2.BORDER_CONSTANT, value=0)
        patches.append(p)

    W_enc  = encode_fn(patches)                       # (n_pos, D)
    scores = _cosine_sim_safe(W_enc, lib_encodings)     # cosine sim (torch-based, avoids numpy BLAS crash)

    # ─ 向量化高斯先验 ────────────────────────────────────────
    pos_arr  = np.array(positions, dtype=np.float32)   # (n, 2) [wy, wx]
    cand_cy  = pos_arr[:, 0] + target_h / 2.0
    cand_cx  = pos_arr[:, 1] + target_w / 2.0
    dy = (cand_cy - cy) / sigma_y
    dx = (cand_cx - cx) / sigma_x
    spatial_weights = np.exp(-0.5 * (dy**2 + dx**2)).astype(np.float32)
    fused = scores * spatial_weights

    best_i = int(np.argmax(fused))
    best_s = float(scores[best_i])
    by_ret, bx_ret = positions[best_i]
    print(f"[MATCH] total={n_total} filtered={n_filtered} kept={len(positions)} "
          f"best_score={best_s:.4f} sw={spatial_weights[best_i]:.3f} pos=({bx_ret},{by_ret})")
    return {"score": best_s, "x": bx_ret, "y": by_ret, "w": target_w, "h": target_h}


def find_best_match_hierarchical(
    search_plane_th68: np.ndarray,
    lib_encodings: np.ndarray,
    target_h: int,
    target_w: int,
    encode_fn,
    contour_sim_model: nn.Module = None,  # 新增：预训练的轮廓相似度模型
    lib_templates_raw: list = None,  # 新增：原始模板图像列表
    lib_contour_encodings: np.ndarray = None,  # 新增：SwinUNet特征编码
    step_coarse: int = 32,
    step_fine:   int = 8,
    fine_expand: int = 1,      # 精搜窗口 = target 尺寸 × (1+2*fine_expand)，默认取 1，显著减少候选数
    top_k:       int = 3,      # 默认 3 山分平衡准确度与速度
    greedy_step: int = 4,
    greedy_iters: int = 4,
    spatial_sigma: float = 3.0,
    center_yx: tuple | None = None,
    min_white_ratio: float = 0.03,
    early_exit_score: float = 0.90,   # 粗搜中分数达到此阈値直接跳过精搜
) -> dict:
    """
    三阶段层次搜索（深度+广度结合）：

    Stage 1 — 广度粗搜（Breadth）：
        step=step_coarse 滑窗 → SAT 预过滤 → 所有候选的 cosine+高斯优先 fused 分数
        取 Top-K 候选位置

    Stage 2 — 深度精搜（Depth）：
        对每个 Top-K 候选，在其周围 fine_expand 倍目标尺寸的小窗口内
        用 step=step_fine 精细滑动匹配
        取全部精搜结果中的全局最优

    Stage 3 — 贪婪爬坡（Greedy Polish）：
        从精搜最优中心出发，尝试 8 个方向的 greedy_step 个像素
        移动到 score 更高的邻居，重复 greedy_iters 次，实现二维极値爬坡
    """
    fH, fW = search_plane_th68.shape[:2]
    cy = float(center_yx[0]) if center_yx is not None else fH / 2.0
    cx = float(center_yx[1]) if center_yx is not None else fW / 2.0
    # 增大 sigma：全局搜索需要更宽松的高斯先验
    # sigma = 图像尺寸的 1/4，允许在较大范围内均匀搜索
    sigma_y = max(1.0, fH / 4.0)
    sigma_x = max(1.0, fW / 4.0)

    # SAT 预处理（共用）
    _ch = search_plane_th68.shape[2] if search_plane_th68.ndim == 3 else 1
    plane_bin = (search_plane_th68[:, :, 0] > 0).astype(np.float32) \
                if search_plane_th68.ndim == 3 \
                else (search_plane_th68 > 0).astype(np.float32)
    sat = cv2.integral(plane_bin)   # (H+1, W+1)
    min_count = min_white_ratio * target_h * target_w
    
    # 预先计算 lib_tensor（只计算一次，避免重复光栅化）
    lib_tensor_precomputed = None
    if contour_sim_model is not None and lib_templates_raw is not None:
        device = next(contour_sim_model.parameters()).device
        lib_contours = []
        for lib_img in lib_templates_raw:
            if isinstance(lib_img, np.ndarray):
                if lib_img.ndim == 3 and lib_img.shape[2] == 3:
                    lib_gray = cv2.cvtColor(lib_img, cv2.COLOR_BGR2GRAY)
                else:
                    lib_gray = lib_img
                lib_contour = _rasterize_contour(lib_gray, 128)
                lib_contours.append(torch.from_numpy(lib_contour)[None, ...])
        if lib_contours:
            lib_tensor_precomputed = torch.stack(lib_contours).to(device)
            print(f"[PRECOMPUTE] lib_tensor shape={lib_tensor_precomputed.shape}, templates={len(lib_contours)}")

    def _white_cnt(wy, wx):
        wy2 = min(wy + target_h, fH); wx2 = min(wx + target_w, fW)
        return sat[wy2, wx2] - sat[wy, wx2] - sat[wy2, wx] + sat[wy, wx]

    def _grid_positions(step, plane_h, plane_w):
        ys = list(range(0, max(1, plane_h - target_h + 1), step))
        xs = list(range(0, max(1, plane_w - target_w + 1), step))
        if ys and (ys[-1] + target_h) < plane_h: ys.append(plane_h - target_h)
        if xs and (xs[-1] + target_w) < plane_w: xs.append(plane_w - target_w)
        return ys or [0], xs or [0]

    def _score_positions(positions, plane, lib_tensor_precomputed):
        """使用预训练的ContourSimilarityNet计算相似度"""
        scores_contour_sim = []
        
        if contour_sim_model is not None:
            device = next(contour_sim_model.parameters()).device
            lib_tensor = lib_tensor_precomputed  # 使用预先计算的lib_tensor
                
            for (wy, wx) in positions:
                p = plane[wy:wy+target_h, wx:wx+target_w]
                if p.shape[0] < target_h or p.shape[1] < target_w:
                    p = cv2.copyMakeBorder(p, 0, max(0, target_h-p.shape[0]),
                                           0, max(0, target_w-p.shape[1]),
                                           cv2.BORDER_CONSTANT, value=0)
                                
                # 如果是3通道BGR，转为灰度图
                if p.ndim == 3 and p.shape[2] == 3:
                    p_gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
                else:
                    p_gray = p
                                
                patch_contour = _rasterize_contour(p_gray, 128)
                patch_tensor = torch.from_numpy(patch_contour)[None, ...].to(device)  # (1, 128, 128)
                patch_tensor = patch_tensor.unsqueeze(0)  # (1, 1, 128, 128)
                            
                with torch.no_grad():
                    # 批量计算：一次性与所有library模板计算相似度
                    patch_expanded = patch_tensor.expand(lib_tensor.shape[0], -1, -1, -1)  # (N, 1, 128, 128)
                    pred_sim, _, _ = contour_sim_model(patch_expanded, lib_tensor)
                    max_score = float(pred_sim.max().item())
                    scores_contour_sim.append(max_score)
                
            return np.array(scores_contour_sim, dtype=np.float32)
        else:
            return np.zeros(len(positions), dtype=np.float32)

    def _fused(scores, positions, c_y, c_x):
        arr = np.array(positions, dtype=np.float32)
        dy = (arr[:, 0] + target_h / 2.0 - c_y) / sigma_y
        dx = (arr[:, 1] + target_w / 2.0 - c_x) / sigma_x
        return scores * np.exp(-0.5 * (dy**2 + dx**2))

    # ================================================================
    # Stage 1: 广度粗搜 — 全平面 Top-K
    # ================================================================
    ys_c, xs_c = _grid_positions(step_coarse, fH, fW)
    pos_c = [(wy, wx) for wy in ys_c for wx in xs_c
             if 0 <= wy <= fH - target_h and 0 <= wx <= fW - target_w
             and _white_cnt(wy, wx) >= min_count]
    if not pos_c:
        pos_c = [(max(0, int(cy - target_h//2)), max(0, int(cx - target_w//2)))]

    scores_c = _score_positions(pos_c, search_plane_th68, lib_tensor_precomputed)
    fused_c  = _fused(scores_c, pos_c, cy, cx)

    k = min(top_k, len(pos_c))
    topk_idx = np.argpartition(fused_c, -k)[-k:]   # top-K 索引
    _best_ci  = int(topk_idx[np.argmax(fused_c[topk_idx])])
    _coarse_y, _coarse_x  = pos_c[_best_ci]          # 粗搜最优位置（search_plane 局部坐标）
    _coarse_score = float(scores_c[_best_ci])
    print(f"[COARSE] candidates={len(pos_c)} top_k={k} "
          f"best_coarse_score={_coarse_score:.4f}")

    # ================================================================
    # Stage 2: 深度精搜 — 对每个 Top-K 做小窗口精细滑动
    # ================================================================
    best_score = -1.0
    best_gy, best_gx = pos_c[int(topk_idx[np.argmax(fused_c[topk_idx])])]

    # 粗搜高分 Early Exit：直接用粗搜最优，跳过精搜
    # 提高阈值到 0.95，确保只有非常高置信度时才跳过精搜
    if scores_c[_best_ci] >= early_exit_score:
        best_score = float(scores_c[_best_ci])
        print(f"[EARLY_EXIT] coarse score={best_score:.4f} >= {early_exit_score}, skip fine+greedy")
        return {"score": best_score, "x": best_gx, "y": best_gy, "w": target_w, "h": target_h,
                "coarse_x": _coarse_x, "coarse_y": _coarse_y, "coarse_score": best_score}

    for ki in topk_idx:
        ky, kx = pos_c[ki]
        pad_y = int(target_h * fine_expand)
        pad_x = int(target_w * fine_expand)
        fy1 = max(0, ky - pad_y)
        fx1 = max(0, kx - pad_x)
        fy2 = min(fH, ky + target_h + pad_y)
        fx2 = min(fW, kx + target_w + pad_x)
        fine_plane = search_plane_th68[fy1:fy2, fx1:fx2]
        ph, pw = fine_plane.shape[:2]

        ys_f, xs_f = _grid_positions(step_fine, ph, pw)
        # Stage 2 同样用 SAT 预过滤（利用全平面 SAT，坐标射入全平面空间）
        pos_f = [(wy, wx) for wy in ys_f for wx in xs_f
                 if 0 <= wy <= ph - target_h and 0 <= wx <= pw - target_w
                 and _white_cnt(fy1 + wy, fx1 + wx) >= min_count]
        if not pos_f:
            continue

        scores_f = _score_positions(pos_f, fine_plane, lib_tensor_precomputed)
        # 精搜阶段直接用粗搜候选位置为中心的实际坐标（局部坐标）
        fine_cy = ky - fy1 + target_h / 2.0
        fine_cx = kx - fx1 + target_w / 2.0
        fused_f  = _fused(scores_f, pos_f, fine_cy, fine_cx)

        best_fi = int(np.argmax(fused_f))
        if scores_f[best_fi] > best_score:
            best_score = float(scores_f[best_fi])
            best_gy = fy1 + pos_f[best_fi][0]
            best_gx = fx1 + pos_f[best_fi][1]

    print(f"[FINE]   best_score={best_score:.4f} pos=({best_gx},{best_gy})")

    # ================================================================
    # Stage 3: 贪婪爬坡 — 8方向邀居迭代极値
    # ================================================================
    gy, gx = best_gy, best_gx
    # 增大爬坡步长，允许更大范围的调整
    dirs = [(-greedy_step, 0), (greedy_step, 0),
            (0, -greedy_step), (0,  greedy_step),
            (-greedy_step, -greedy_step), (-greedy_step,  greedy_step),
            ( greedy_step, -greedy_step), ( greedy_step,  greedy_step)]

    for _ in range(greedy_iters):
        nbr_positions = []
        for dy_g, dx_g in dirs:
            ny, nx = gy + dy_g, gx + dx_g
            if 0 <= ny <= fH - target_h and 0 <= nx <= fW - target_w:
                nbr_positions.append((ny, nx))
        if not nbr_positions:
            break
        nbr_scores = _score_positions(nbr_positions, search_plane_th68, lib_tensor_precomputed)
        best_ni = int(np.argmax(nbr_scores))
        if nbr_scores[best_ni] > best_score:
            best_score = float(nbr_scores[best_ni])
            gy, gx = nbr_positions[best_ni]
        else:
            break   # 无改善，局部极値已到达

    print(f"[GREEDY] final_score={best_score:.4f} pos=({gx},{gy})")
    return {"score": best_score, "x": gx, "y": gy, "w": target_w, "h": target_h,
            "coarse_x": _coarse_x, "coarse_y": _coarse_y, "coarse_score": _coarse_score}


# ═════════════════════════════════════════════════════════════════════════════
# 轮廓修正外包围方框
# ═════════════════════════════════════════════════════════════════════════════
def _refine_bbox_by_contour(
    th68_crop: np.ndarray,
    gx: int, gy: int,
    frame_w: int, frame_h: int,
    margin: int = 6,
    min_fill_ratio: float = 0.02,
    max_size_factor: float = 1.5,
    orig_w: int | None = None,
    orig_h: int | None = None,
) -> tuple[int, int, int, int]:
    """
    在匹配窗口内（th68 crop）用 findNonZero 取所有前景像素，
    计算紧密外包围方框，使其更准确地反映当前帧中目标的实际轮廓形状。

    这里不用找轮廓过滤，直接对匹配窗口内的所有非零像素做 boundingRect，
    避免轮廓筛选问题引起的背景射爆。

    退化条件：
      - 匹配窗口内非零像素过少 (< min_fill_ratio * crop_area)：不修正。
      - 外包围宽/高 超过 orig*max_size_factor：限制。
    """
    ch, cw = th68_crop.shape[:2]
    ref_w = orig_w if orig_w is not None else cw
    ref_h = orig_h if orig_h is not None else ch

    gray = cv2.cvtColor(th68_crop, cv2.COLOR_BGR2GRAY) if th68_crop.ndim == 3 else th68_crop

    # 直接取所有前景像素的外包围方框 —— 最直接、不依赖轮廓筛选
    nz = cv2.findNonZero(gray)
    if nz is None or len(nz) < min_fill_ratio * cw * ch:
        return gx, gy, ref_w, ref_h

    rx, ry, rw, rh = cv2.boundingRect(nz)

    # 加 margin
    rx = max(0, rx - margin)
    ry = max(0, ry - margin)
    rw = min(cw - rx, rw + 2 * margin)
    rh = min(ch - ry, rh + 2 * margin)

    # 限制宽/高变化幅度（防止异常射爆）
    rw = min(rw, int(ref_w * max_size_factor))
    rh = min(rh, int(ref_h * max_size_factor))

    nx, ny = gx + rx, gy + ry
    refined = core.clamp_bbox(nx, ny, rw, rh, frame_w, frame_h)
    print(f"[REFINE] orig=({gx},{gy},{ref_w},{ref_h}) "
          f"nz_cnt={len(nz)} "
          f"refined=({refined[0]},{refined[1]},{refined[2]},{refined[3]})")
    return refined


# ═════════════════════════════════════════════════════════════════════════════
# Target selection (selectROI / MIS_INIT_BBOX)
# ═════════════════════════════════════════════════════════════════════════════
def _select_roi(frame: np.ndarray, cfg: core.TrackConfig) -> tuple[int,int,int,int]:
    """优先 MIS_INIT_BBOX 环境变量，否则 OpenCV selectROI 鼠标框选。"""
    H, W = frame.shape[:2]
    if cfg.init_bbox:
        roi = core._try_parse_bbox(cfg.init_bbox)
        if roi is not None:
            return core.clamp_bbox(*[int(v) for v in roi], W, H)
    # OpenCV selectROI
    roi = cv2.selectROI("Select target (drag & press ENTER)", frame, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, w, h = [int(v) for v in roi]
    return core.clamp_bbox(x, y, w, h, W, H)


# ═════════════════════════════════════════════════════════════════════════════
# main()
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    cfg = core.TrackConfig()

    # ── 输入/输出目录 ──────────────────────────────────────────────────────────
    default_frames = r"E:\Surgical_Tracking_Datasets\Chess_Box_1_8\captured_photos1"
    default_out    = r"D:\MIS_Pose_Track_Re3D\Exp_Results_Deformation"
    cfg.frames_dir = _env("MIS_FRAMES_DIR", default_frames) or default_frames
    cfg.output_dir = _env("MIS_OUTPUT_DIR", default_out)    or default_out
    cfg.init_bbox  = _env("MIS_INIT_BBOX", "")

    # ── 跟踪搜索参数 ──────────────────────────────────────────────────────────
    EXPAND_BASE     = _env_int  ("MIS_EXPAND_BASE",    128)   # 初始扩展半径
    EXPAND_RETRY    = _env_int  ("MIS_EXPAND_RETRY",   160)   # 低置信度时扩展至此
    ADAPT_THRESH    = _env_float("MIS_ADAPT_THRESH",   0.50)  # 低于此分数触发自适应扩展
    LOSS_THRESH     = _env_float("MIS_LOSS_THRESH",    0.35)  # 低于此视为目标丢失
    SLIDE_STEP      = _env_int  ("MIS_SLIDE_STEP",      1)    # 精搜步长
    COARSE_STEP     = _env_int  ("MIS_COARSE_STEP",    16)    # 粗搜步长（降低到16提高精度）
    FINE_EXPAND     = _env_int  ("MIS_FINE_EXPAND",     2)    # 精搜窗口为目标尺寸的倍数
    NUM_TEMPLATES   = _env_int  ("MIS_NUM_TEMPLATES",  256)   # RTW 增广数量
    SAVE_DEBUG      = _env_bool ("MIS_SAVE_DEBUG",     True)  # 是否保存每帧中间结果
    PRINT_EVERY     = _env_int  ("MIS_PRINT_EVERY",     10)   # 控制台打印间隔

    # ── 分辨率缩放（proc_scale）──────────────────────────────────────────────
    # core._read_frame_bgr 内部已调用 _resize_for_processing(im, cfg) 实现缩放
    cfg.proc_scale = _env_float("MIS_PROC_SCALE", 0.35)  # 默认 0.35：全局压缩分辨率加快处理

    # ── 帧列表 ────────────────────────────────────────────────────────────────
    frames = core.list_frames(cfg.frames_dir)
    if not frames:
        raise RuntimeError(f"未找到任何帧: {cfg.frames_dir}")
    print(f"[配置] frames={len(frames)}  frames_dir={cfg.frames_dir}")
    print(f"[配置] output_dir={cfg.output_dir}  expand={EXPAND_BASE}/{EXPAND_RETRY}  "
          f"adapt_th={ADAPT_THRESH}  slide_step={SLIDE_STEP}  templates={NUM_TEMPLATES}")

    # ── 输出目录结构（与 MIS_C_F_SIMI_Mesh_Match_Track_3DMatch.py 一致）──────
    out_root         = _ensure_dir(cfg.output_dir)
    out_frames       = _ensure_dir(os.path.join(out_root, "frames_with_bbox"))
    out_roi          = _ensure_dir(os.path.join(out_root, "roi"))
    out_th           = _ensure_dir(os.path.join(out_root, "th_gray"))
    out_feats        = _ensure_dir(os.path.join(out_root, "features"))
    out_match_coarse = _ensure_dir(os.path.join(out_root, "match_coarse"))   # 粗搜位置轮廓裁剪 + 标注
    out_match_fine   = _ensure_dir(os.path.join(out_root, "match_fine"))     # 精搜最终位置轮廓裁剪 + 标注
    out_match_cmp    = _ensure_dir(os.path.join(out_root, "match_compare"))  # 双框叠加对比图

    # ── 模型加载 ──────────────────────────────────────────────────────────────
    # 只加载轮廓相似度预训练模型（去除SwinUNetLarge加速）
    contour_sim_model = _load_contour_similarity_model(_CONTOUR_SIM_MODEL_PATH)

    # ════════════════════════════════════════════════════════════════════════
    # 第一帧：目标选择 + Library 构建
    # ════════════════════════════════════════════════════════════════════════
    first = core._read_frame_bgr(frames[0], cfg)
    if first is None:
        raise RuntimeError(f"无法读取第一帧: {frames[0]}")
    H, W = first.shape[:2]
    print(f"[第一帧] 尺寸 {W}x{H}")

    # 鼠标选框 / 环境变量 bbox
    bbox = _select_roi(first, cfg)
    x, y, w, h = bbox
    print(f"[第一帧] 选定 bbox=({x},{y},{w},{h})")

    # 全局 th68 → 截取 target 区域
    th68_first = _apply_th68(first)
    target_th68 = th68_first[y:y+h, x:x+w].copy()
    cv2.imwrite(os.path.join(out_root, "th68_first_frame.jpg"), th68_first)
    cv2.imwrite(os.path.join(out_root, "target_th68_first.jpg"), target_th68)

    # 生成 target mask (bbox 内全白)
    target_mask = np.ones((h, w), dtype=np.uint8) * 255

    # RTW 增广 → 直接获取 raw BGR 列表（不转 tensor，避免双重归一化）
    template_list = _build_rtw_bgr(target_th68, target_mask, NUM_TEMPLATES)
    if SAVE_DEBUG:
        for ki, t in enumerate(template_list[:8]):
            cv2.imwrite(os.path.join(out_feats, f"template_{ki:04d}.jpg"), t)
    
    # 保存原始模板图像列表供 ContourSimilarityNet 使用
    lib_templates_raw = template_list  # 保存原始图像列表

    # ── 保存 Library 到本地 ──────────────────────────────────────────────
    out_lib = _ensure_dir(os.path.join(out_root, "library"))
    # 只保存模板图像，不保存编码（已删除SwinUNet）
    for ki, t in enumerate(template_list):
        cv2.imwrite(os.path.join(out_lib, f"template_{ki:04d}.jpg"), t)
    with open(os.path.join(out_lib, "library_meta.json"), "w", encoding="utf-8") as _f:
        json.dump({
            "num_templates": len(template_list),
            "target_bbox": list(map(int, bbox)),
            "proc_scale": cfg.proc_scale,
        }, _f, indent=2)
    print(f"[Library] 已保存至: {out_lib}")

    # 保存第一帧结果
    if SAVE_DEBUG:
        roi0 = first[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(out_frames, "frame_0000.jpg"),
                    core.draw_bbox(first, bbox, text="f=0"))
        cv2.imwrite(os.path.join(out_roi,    "roi_0000.jpg"), roi0)
        cv2.imwrite(os.path.join(out_th,     "th68_0000.jpg"), target_th68)

    track_rows: list[dict] = [{
        "frame": 0, "image": frames[0],
        "bbox": list(map(int, bbox)),
        "score": 1.0, "dx": 0, "dy": 0,
    }]
    diag_rows: list[dict] = []

    # ════════════════════════════════════════════════════════════════════════
    # 逐帧跟踪循环
    # ════════════════════════════════════════════════════════════════════════
    prev_bbox = bbox
    # 保存第一帧的bbox尺寸作为匹配基准
    base_w, base_h = w, h
    print(f"[BASE] 第一帧基准尺寸: {base_w}x{base_h}")
    for i, img_path in enumerate(frames[1:], start=1):
        t_frame = time.time()
        frame = core._read_frame_bgr(img_path, cfg)
        if frame is None:
            print(f"[警告] 跳过: {img_path}")
            continue

        bx, by, bw, bh = [int(v) for v in prev_bbox]
        cx0, cy0 = bx + bw // 2, by + bh // 2

        if (i % PRINT_EVERY) == 0:
            print(f"[进度] frame={i}/{len(frames)-1}  prev_bbox={list(map(int, prev_bbox))}")

        # ── Step 1: 全局 th68（单次计算）────────────────────────────────────
        th68_curr = _apply_th68(frame)

        # ── Step 2: 全局匹配 — 使用第一帧基准尺寸，避免累积漂移 ─────────────
        # 始终以第一帧的 bbox 尺寸进行匹配
        fH, fW = th68_curr.shape[:2]
        cx_prev = bx + bw // 2
        cy_prev = by + bh // 2
        search_plane = th68_curr  # 全帧，不裁剪
        plane_center_yx = (cy_prev, cx_prev)  # 上一帧中心作为先验
        
        result = find_best_match_hierarchical(
            search_plane, None, base_h, base_w, None,  # lib_encodings和encode_fn不再使用
            contour_sim_model=contour_sim_model,  # 传入预训练模型
            lib_templates_raw=lib_templates_raw,  # 传入原始模板列表
            lib_contour_encodings=None,  # 不再使用SwinUNet特征
            step_coarse=COARSE_STEP,
            step_fine=SLIDE_STEP,
            fine_expand=FINE_EXPAND,
            top_k=_env_int("MIS_TOPK", 3),
            greedy_step=_env_int("MIS_GREEDY_STEP", 4),
            greedy_iters=_env_int("MIS_GREEDY_ITERS", 4),
            spatial_sigma=_env_float("MIS_SPATIAL_SIGMA", 3.0),
            center_yx=plane_center_yx,
            early_exit_score=_env_float("MIS_EARLY_EXIT", 0.95),  # 提高阈值，确保精搜被执行
        )
        
        # Step 3: 全局坐标直接返回（使用第一帧基准尺寸）
        new_x = int(result["x"])
        new_y = int(result["y"])
        new_x, new_y, new_w, new_h = core.clamp_bbox(new_x, new_y, base_w, base_h,
                                                       frame.shape[1], frame.shape[0])

        # Step 4: 轮廓修正 - 基于第一帧基准尺寸，根据实际目标微调
        crop_th68 = th68_curr[new_y:new_y + new_h, new_x:new_x + new_w]
        new_x, new_y, new_w, new_h = _refine_bbox_by_contour(
            crop_th68, new_x, new_y,
            frame.shape[1], frame.shape[0],
            margin=_env_int("MIS_CONTOUR_MARGIN", 10),  # 增大margin到10像素
            min_fill_ratio=_env_float("MIS_MIN_FILL", 0.01),  # 降低阈值到1%
            max_size_factor=_env_float("MIS_MAX_SIZE", 2.0),  # 允许最大2倍变化
            orig_w=base_w, orig_h=base_h,  # 使用第一帧基准尺寸
        )
        new_bbox = (new_x, new_y, new_w, new_h)
        score    = result["score"]
        dx = int((new_x + new_w//2) - cx0)
        dy = int((new_y + new_h//2) - cy0)

        # ── Step 5: 保存调试图像 ─────────────────────────────────────────────
        if SAVE_DEBUG:
            fn = f"frame_{i:04d}"
            cv2.imwrite(os.path.join(out_frames, fn + ".jpg"),
                        core.draw_bbox(frame, new_bbox, text=f"f={i} s={score:.2f}"))
            cv2.imwrite(os.path.join(out_roi, fn + ".jpg"),
                        frame[new_y:new_y+new_h, new_x:new_x+new_w])
            cv2.imwrite(os.path.join(out_th, fn + ".jpg"),
                        th68_curr[new_y:new_y+new_h, new_x:new_x+new_w])
            if (i % max(1, PRINT_EVERY)) == 0:
                cv2.imwrite(os.path.join(out_feats, f"contour_th68_{i:04d}.jpg"),
                            th68_curr)  # save full-frame th68

            # 粗搜 / 精搜轮廓匹配可视化
            coarse_gx = int(result.get("coarse_x", result["x"]))
            coarse_gy = int(result.get("coarse_y", result["y"]))
            coarse_sc = result.get("coarse_score", score)

            # 1) match_coarse: 粗搜 th68 patch + 左上角分数
            crop_c = th68_curr[coarse_gy:coarse_gy+bh, coarse_gx:coarse_gx+bw].copy()
            if crop_c.size > 0:
                cv2.putText(crop_c, f"C:{coarse_sc:.3f}",
                            (2, max(14, crop_c.shape[0] // 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(out_match_coarse, fn + ".jpg"), crop_c)

            # 2) match_fine: 精搜 th68 patch + 左上角分数
            crop_f = th68_curr[new_y:new_y+new_h, new_x:new_x+new_w].copy()
            if crop_f.size > 0:
                cv2.putText(crop_f, f"F:{score:.3f}",
                            (2, max(14, crop_f.shape[0] // 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imwrite(os.path.join(out_match_fine, fn + ".jpg"), crop_f)

            # 3) match_compare: 全帧 th68 双框叠加（黄=粗搜, 绿=精搜）
            vis = th68_curr.copy()
            cv2.rectangle(vis, (coarse_gx, coarse_gy),
                          (coarse_gx + bw, coarse_gy + bh), (0, 255, 255), 2)
            cv2.putText(vis, f"C:{coarse_sc:.2f}",
                        (coarse_gx, max(12, coarse_gy - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(vis, (new_x, new_y),
                          (new_x + new_w, new_y + new_h), (0, 255, 0), 2)
            cv2.putText(vis, f"F:{score:.2f}",
                        (new_x, max(12, new_y - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(out_match_cmp, fn + ".jpg"), vis)

        # ── Step 6: 目标丢失警告 ──────────────────────────────────────────────
        if score < LOSS_THRESH:
            print(f"[LOSS] frame={i} score={score:.3f} < {LOSS_THRESH} (目标可能丢失)")

        elapsed = time.time() - t_frame
        track_rows.append({
            "frame": i,
            "image": img_path,
            "bbox": list(map(int, new_bbox)),
            "score": float(score),
            "dx": int(dx),
            "dy": int(dy),
        })
        diag_rows.append({
            "frame": i,
            "score": float(score),
            "dx": int(dx),
            "dy": int(dy),
            "elapsed_s": float(f"{elapsed:.3f}"),
        })

        prev_bbox = new_bbox

    # ════════════════════════════════════════════════════════════════════════
    # 写出结果
    # ════════════════════════════════════════════════════════════════════════
    result_json = os.path.join(out_root, "tracking_bboxes_and_scores.json")
    with open(result_json, "w", encoding="utf-8") as f:
        json.dump(track_rows, f, ensure_ascii=False, indent=2)

    diag_csv = os.path.join(out_root, "diag.csv")
    if diag_rows:
        with open(diag_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(diag_rows[0].keys()))
            writer.writeheader()
            writer.writerows(diag_rows)

    meta = {
        "time": _now(),
        "task": "contour_only_swinunet_tracking",
        "model": _CONTOUR_SIM_MODEL_PATH,
        "frames_dir": cfg.frames_dir,
        "output_dir": cfg.output_dir,
        "num_frames": len(frames),
        "expand_base": EXPAND_BASE,
        "expand_retry": EXPAND_RETRY,
        "adapt_thresh": ADAPT_THRESH,
        "loss_thresh": LOSS_THRESH,
        "slide_step": SLIDE_STEP,
        "num_templates": NUM_TEMPLATES,
        "proc_scale": float(cfg.proc_scale),
        "device": str(_device),
    }
    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[完成] output_dir={out_root}")
    print(f"[完成] tracking_bboxes_and_scores.json  diag.csv  meta.json")


if __name__ == "__main__":
    main()
