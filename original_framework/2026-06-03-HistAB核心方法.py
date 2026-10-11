#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026-06-03-HistAB核心方法.py [FROZEN 2026-06-03]
───────────────────────────────────────
方案 **A+B（默认 MIS_COARSE_SCHEME=ab）**：HSV(th68 掩膜) 直方图预筛 Top‑M，仅幸存候选跑 CNN；粗排 λ·normalize(轮廓)+(1−λ)·hist，再乘空间先验。
方案 **C**：见 ``Deformation_F_T_C_G_Experi_hist_C.py`` — hist×高斯空间取 Top‑K′，仅对其 CNN；粗排序为轮廓×空间。

运行：``python Deformation_F_T_C_G_Experi_hist_AB.py``

环境变量（增补）：
  MIS_COARSE_SCHEME       ab | c
  MIS_HIST_GATE_T        hist correl→[0,1]；<0 关闭直方图门控退化为「全候选 CNN」（与原版粗搜等价）
  MIS_HIST_TOP_M         A+B 幸存 CNN cap，默认 96
  MIS_HIST_TOPKP_C       方案 C 的 K′，默认 96
  MIS_HIST_H_BINS / MIS_HIST_S_BINS   默认 30 / 32
  MIS_FUSE_LAMBDA        λ 基底，默认 0.65
  MIS_FUSE_ADAPT         1 | 0 自适应 λ（低轮廓分时加大 hist）
  MIS_EXPAND_BASE        局部平面边距像素，默认 128
  MIS_CONTOUR_INFER_BS   补丁轮廓批量 encode 的 batch 大小，默认 48（显存大可试 96）
  MIS_FINE_FUSE_HIST / MIS_GREEDY_FUSE_HIST   默认 1：精搜/贪婪沿用 ColorHist+轮廓融合（否则仅轮廓）
  MIS_EARLY_HIST_MIN     >0 时粗搜尽早退出还须 hist≥该阈值，减轻「高分但色号错」早停
  MIS_EARLY_POLISH       默认 1：早停前一跳融合抛光（当前格+贪婪步邻居），弱化「仅靠轮廓满分早停」误差
  MIS_CONTOUR_SAT_THRESH / MIS_CONTOUR_SAT_SCALE 轮廓分≥阈值时将 λ×scale(<1)，加重 hist（饱和时轮廓难以区分）
其余目录/模板与 ``Deformation_F_T_C_G_Experi.py`` 一致；轮廓模型仍为 ``_CONTOUR_SIM_MODEL_PATH``。
"""

from __future__ import annotations

import os as _os_module

# 全局静默开关：MIS_SILENT=1 时屏蔽所有调试 print
_MIS_SILENT: bool = _os_module.environ.get("MIS_SILENT", "0") == "1"

import csv
import time as _time_mod
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ─── 导入轮廓相似度网络 ──────────────────────────────────────────────
from train_contour_similarity import ContourSimilarityNet, _rasterize_contour


def _cosine_sim_safe(enc: np.ndarray, lib: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix (enc @ lib.T) using torch to avoid
    numpy MKL/BLAS stack overflow on large embedding dimensions (e.g. 16384).
    Pure-torch implementation: avoid numpy .max(axis=1) overhead.
    Returns max score per row as float32 ndarray of shape (n_enc,).
    """
    et = torch.from_numpy(np.ascontiguousarray(enc)).float()
    lt = torch.from_numpy(np.ascontiguousarray(lib)).float()
    sim = et @ lt.T
    return sim.max(dim=1).values.cpu().numpy().astype(np.float32)



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
_CONTOUR_SIM_MODEL_PATH = os.environ.get(
    "MIS_CONTOUR_SIM_CKPT",
    r"E:\contour_sim_runs\chess_v1\contour_similarity_best.pt"
)
# TrackVes 三元组模型：train_contour_sim_trackves_triplet.py → 可设 MIS_CONTOUR_SIM_CKPT 为
# E:\contour_sim_runs\trackves_triplet_v1\contour_similarity_triplet_best.pt

# TrackVes 专用置信度模型（仅用于 NanoTrack 融合触发，不参与候选打分）
_CONTOUR_SIM_CONF_CKPT = os.environ.get(
    "MIS_CONTOUR_SIM_CONF_CKPT",
    r"E:\contour_sim_runs\trackves_v1\contour_similarity_best.pt"
)

# 全局变量，供compare_iscience_tracking.py使用
contour_sim_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_contour_similarity_model(ckpt_path: str) -> ContourSimilarityNet:
    """Load pre-trained ContourSimilarityNet for direct contour similarity scoring."""
    if not os.path.isfile(ckpt_path):
        if not _MIS_SILENT: print(f"[WARN] ContourSimilarity model not found: {ckpt_path}")
        return None
    model = ContourSimilarityNet()
    state = torch.load(ckpt_path, map_location=_device)
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval().to(_device)
    print(f"[ContourSim] Loaded from {ckpt_path} → device={_device}")
    return model

# 加载模型到全局变量
contour_sim_model = _load_contour_similarity_model(_CONTOUR_SIM_MODEL_PATH)


# ═════════════════════════════════════════════════════════════════════════════
# NanoTrack 辅助融合模块
# ─────────────────────────────────────────────────────────────────────────────
# 借鉴 NanoTrack (SiamNet, ~1 MB ONNX) 的 Siamese 模板匹配能力：
#   • 当 HistAB 轮廓分 < ADAPT_THRESH 时，加入 NanoTrack 作为辅助校正
#   • 当 HistAB 分 < LOSS_THRESH 时，由 NanoTrack 接管作为 redetection 并更新搜索中心
# 通过环境变量 MIS_USE_NANO=1 启用，默认关闭（不影响原有流程）
# ═════════════════════════════════════════════════════════════════════════════

_NANO_BACKBONE_DEFAULT = r"D:\MIS_Pose_Track_Re3D\nanotrack_backbone_sim.onnx"
_NANO_NECKHEAD_DEFAULT = r"D:\MIS_Pose_Track_Re3D\nanotrack_head_sim.onnx"


class _NanoTrackHelper:
    """轻量 NanoTrack 包装，用于 HistAB 的自适应融合与 redetection。"""

    def __init__(self, init_frame: np.ndarray, init_bbox: tuple,
                 backbone: str = _NANO_BACKBONE_DEFAULT,
                 neckhead: str = _NANO_NECKHEAD_DEFAULT) -> None:
        self.last_bbox = tuple(map(int, init_bbox))
        self.available = False
        if not (os.path.isfile(backbone) and os.path.isfile(neckhead)):
            print(f"[NanoTrack] ONNX 权重未找到，NanoTrack 融合已禁用")
            return
        try:
            params = cv2.TrackerNano_Params()
            params.backbone = backbone
            params.neckhead = neckhead
            self._tracker = cv2.TrackerNano_create(params)
            self._tracker.init(init_frame, tuple(map(int, init_bbox)))
            self.available = True
            print(f"[NanoTrack] 初始化成功，bbox={init_bbox}")
        except Exception as e:
            print(f"[NanoTrack] 初始化失败: {e}")

    def update(self, frame: np.ndarray):
        """返回 (bbox: tuple[x,y,w,h], score: float)"""
        if not self.available:
            return self.last_bbox, 0.0
        try:
            ok, box = self._tracker.update(frame)
            if ok:
                x, y, w, h = [int(round(float(v))) for v in box]
                h_, w_ = frame.shape[:2]
                x = max(0, min(x, w_ - 1))
                y = max(0, min(y, h_ - 1))
                w = max(1, min(w, w_ - x))
                h = max(1, min(h, h_ - y))
                self.last_bbox = (x, y, w, h)
                score = float(getattr(self._tracker, 'getTrackingScore', lambda: 1.0)())
                return self.last_bbox, score
        except Exception:
            pass
        return self.last_bbox, 0.0


def _fuse_bboxes(bbox_a: tuple, score_a: float,
                 bbox_b: tuple, score_b: float,
                 alpha: float) -> tuple:
    """以 alpha 权重线性融合两个 bbox（cx, cy 加权，w/h 取 score 较高者）。"""
    ax, ay, aw, ah = [float(v) for v in bbox_a]
    bx, by, bw, bh = [float(v) for v in bbox_b]
    cx_a, cy_a = ax + aw / 2, ay + ah / 2
    cx_b, cy_b = bx + bw / 2, by + bh / 2
    cx = alpha * cx_a + (1 - alpha) * cx_b
    cy = alpha * cy_a + (1 - alpha) * cy_b
    # 尺寸取置信度较高者
    if score_a >= score_b:
        w, h = aw, ah
    else:
        w, h = bw, bh
    nx = int(round(cx - w / 2))
    ny = int(round(cy - h / 2))
    return (nx, ny, int(round(w)), int(round(h)))


# ════════════════════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════════════════════
def _apply_th68(img_bgr: np.ndarray, th: int = 68) -> np.ndarray:
    """固定阈值 th 二值化（全局阈值，每帧只需调用一次，搜索区/候选区直接 crop）。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)


# ─── HS 掩膜直方图（方案 A / B / C）─────────────────────────────────────────
def build_ref_masked_hsv_hist(
    roi_bgr: np.ndarray,
    mask_u8: np.ndarray | None,
    h_bins: int,
    s_bins: int,
) -> np.ndarray:
    mk = mask_u8
    if mk is None:
        mk = np.ones(roi_bgr.shape[:2], dtype=np.uint8) * 255
    elif mk.shape[:2] != roi_bgr.shape[:2]:
        mk = cv2.resize(mk, (roi_bgr.shape[1], roi_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    if int(cv2.countNonZero(mk)) < 40:
        return np.zeros((h_bins, s_bins), dtype=np.float32)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], mk, [int(h_bins), int(s_bins)], [0, 180, 0, 256])
    h32 = hist.astype(np.float32)
    cv2.normalize(h32, h32, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return h32


def batched_hist_correl_masked(
    frame_bgr: np.ndarray,
    plane_th68: np.ndarray,
    plane_origin_yx: tuple[int, int],
    positions: list[tuple[int, int]],
    target_h: int,
    target_w: int,
    ref_hist: np.ndarray,
    h_bins: int,
    s_bins: int,
) -> np.ndarray:
    """批量 HSV 直方图相关度。优化：预转换 plane 灰度 + SAT 快速过滤空白候选。"""
    oy, ox = int(plane_origin_yx[0]), int(plane_origin_yx[1])
    hh, ww = frame_bgr.shape[:2]
    n = len(positions)
    out = np.zeros(n, dtype=np.float32)
    # 预转换 plane 为灰度（避免逐 patch 条件判断）
    gray_plane = plane_th68[:, :, 0] if plane_th68.ndim == 3 else plane_th68
    # 预计算 SAT 用于快速统计白色像素
    plane_bin = (gray_plane > 0).astype(np.float32)
    sat = cv2.integral(plane_bin)  # (H+1, W+1)
    min_mask_pixels = 40
    # ref_hist 只需归一化一次（调用方已归一化，此处确保类型）
    rh = ref_hist.astype(np.float32)
    hist_size = [int(h_bins), int(s_bins)]
    hist_ranges = [0, 180, 0, 256]
    for idx, (wy, wx) in enumerate(positions):
        gy_g, gx_g = oy + wy, ox + wx
        gy2, gx2 = gy_g + target_h, gx_g + target_w
        if gy_g < 0 or gx_g < 0 or gy2 > hh or gx2 > ww:
            continue
        # SAT 快速检测白色像素数（替代逐 patch countNonZero）
        wy2_s = min(wy + target_h, gray_plane.shape[0])
        wx2_s = min(wx + target_w, gray_plane.shape[1])
        nz_cnt = sat[wy2_s, wx2_s] - sat[wy, wx2_s] - sat[wy2_s, wx] + sat[wy, wx]
        if nz_cnt < min_mask_pixels:
            continue
        patch_bgr = frame_bgr[gy_g:gy2, gx_g:gx2]
        gray_w = gray_plane[wy : wy + target_h, wx : wx + target_w]
        if gray_w.shape[0] != target_h or gray_w.shape[1] != target_w:
            continue
        mk_u8 = (gray_w > 0).astype(np.uint8) * 255
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        hc = cv2.calcHist([hsv], [0, 1], mk_u8, hist_size, hist_ranges).astype(np.float32)
        cv2.normalize(hc, hc, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        sim = float(cv2.compareHist(rh, hc, cv2.HISTCMP_CORREL))
        out[idx] = max(0.0, min(1.0, sim))
    return out


def _spatial_gauss_xy(
    positions: list[tuple[int, int]],
    target_h: int,
    target_w: int,
    c_y: float,
    c_x: float,
    sigma_y: float,
    sigma_x: float,
) -> np.ndarray:
    arr = np.array(positions, dtype=np.float32)
    cand_cy = arr[:, 0] + float(target_h) / 2.0
    cand_cx = arr[:, 1] + float(target_w) / 2.0
    dy = (cand_cy - c_y) / max(float(sigma_y), 1.0)
    dx = (cand_cx - c_x) / max(float(sigma_x), 1.0)
    return np.exp(-0.5 * (dy * dy + dx * dx)).astype(np.float32)


def pick_survivors_scheme_ab(
    hist01: np.ndarray, T_low: float, cap_m: int, min_keep: int, fallback_m: int
) -> np.ndarray:
    n = int(hist01.size)
    hi = np.argsort(-hist01.astype(np.float64))
    gate = hist01 >= float(T_low)
    surv = np.nonzero(gate)[0]
    if int(surv.size) > int(cap_m):
        order = hist01.argsort()[::-1]
        surv = order[: int(cap_m)].astype(np.int64)
    elif int(surv.size) < int(min_keep):
        extra = hi[: max(int(fallback_m), int(min_keep))]
        surv = np.unique(np.concatenate([surv.astype(np.int64), extra.astype(np.int64)]))
        if surv.size > int(cap_m):
            surv = surv[np.argsort(-hist01[surv])[: int(cap_m)]]
    surv = np.sort(np.unique(np.asarray(surv, dtype=np.int64)))
    if surv.size == 0:
        surv = hi[: max(8, min(32, n))].astype(np.int64)
    return surv


def pick_survivors_scheme_c(pref: np.ndarray, top_k_prime: int) -> np.ndarray:
    n = int(pref.size)
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    k = max(8, min(int(top_k_prime), n))
    order = np.argsort(-pref.astype(np.float64))
    return order[:k].astype(np.int64)


def _effective_fuse_lambda(base: float, max_contour: float, adaptive: bool) -> float:
    b = float(np.clip(base, 0.0, 1.0))
    if not adaptive:
        return b
    t = float(max(0.0, min(1.0, max_contour)))
    lam = b * max(0.05, t)
    return float(np.clip(lam, 0.02, 0.98))


def _fuse_lambda_hist_accuracy(
    fuse_lambda_base: float,
    max_contour: float,
    fuse_adaptive: bool,
    contour_sat_thresh: float,
    contour_sat_lambda_scale: float,
) -> float:
    """融合 λ：在轮廓分饱和（多候选≈1）时压低 λ，使 hist 在排序中占更大权重。"""
    lam = _effective_fuse_lambda(fuse_lambda_base, max_contour, fuse_adaptive)
    st = float(contour_sat_thresh)
    if st > 0.0 and st < 1.01 and float(max_contour) >= st:
        lam *= float(np.clip(contour_sat_lambda_scale, 0.05, 1.0))
    return float(np.clip(lam, 0.03, 0.94))


def _normalize_01(vals: np.ndarray) -> np.ndarray:
    if vals.size == 0:
        return vals.astype(np.float32)
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi <= lo + 1e-8:
        return np.ones_like(vals, dtype=np.float32)
    return np.clip((vals.astype(np.float32) - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def _sz_gray_masked_ncc(ref_g: np.ndarray, cand_g: np.ndarray, fg_thresh: float = 8.0) -> float:
    """轻量伪 3D / 2.5D：参考 ROI 与候选 ROI 灰度的 masked Pearson，映射到 [0,1]。"""
    rg = np.asarray(ref_g, dtype=np.float32)
    cg = np.asarray(cand_g, dtype=np.float32)
    if rg.size <= 0 or cg.size <= 0:
        return 0.0
    if rg.shape != cg.shape:
        cg = cv2.resize(cg, (rg.shape[1], rg.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    m = (rg > fg_thresh) | (cg > fg_thresh)
    if int(np.count_nonzero(m)) < 24:
        m = np.ones_like(rg, dtype=bool)
    a = rg[m]
    b = cg[m]
    a = a - float(np.mean(a))
    b = b - float(np.mean(b))
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da < 1e-6 or db < 1e-6:
        return 0.0
    ncc = float(np.dot(a, b) / (da * db))
    return float(np.clip(0.5 * (ncc + 1.0), 0.0, 1.0))


def _pseudo_depth_u8_from_gray(gray_u8: np.ndarray, fg_thresh: float = 1.0) -> np.ndarray:
    """th_68 前景灰度 → 伪深度 uint8（与 MIS mesh 路径同定义，供 s_z 第二通道）。"""
    g = np.asarray(gray_u8, dtype=np.float32)
    u8 = np.zeros(g.shape, dtype=np.uint8)
    m = g > float(fg_thresh)
    if not np.any(m):
        return u8
    vals = g[m]
    lo, hi = float(np.min(vals)), float(np.max(vals))
    rng = max(hi - lo, 1e-6)
    u8[m] = np.clip(((g[m] - lo) / rng) * 255.0, 0, 255).astype(np.uint8)
    return u8


def _sz_pseudo3d_combined(
    ref_gray: np.ndarray,
    cand_gray: np.ndarray,
    ref_depth_u8: np.ndarray | None,
    cand_depth_u8: np.ndarray | None,
) -> float:
    """伪 3D：灰度 masked NCC + 伪深度 NCC（抑制光照、保留形状）。"""
    w_g = float(np.clip(float(_os_module.environ.get("MIS_SZ_GRAY_WEIGHT", "0.55")), 0.0, 1.0))
    sz_g = _sz_gray_masked_ncc(ref_gray, cand_gray)
    use_p3d = _os_module.environ.get("MIS_SZ_PSEUDO3D", "1").lower() not in ("0", "false", "no", "off")
    if not use_p3d or ref_depth_u8 is None or cand_depth_u8 is None:
        return sz_g
    sz_d = _sz_gray_masked_ncc(ref_depth_u8, cand_depth_u8, fg_thresh=1.0)
    return float(w_g * sz_g + (1.0 - w_g) * sz_d)


def _fusion_disagreement_adjust(
    base_weights: tuple[float, float, float, float],
    fuse_lambda_base: float,
    nano_conf: float,
    hist_at_nano_peak: float,
) -> tuple[tuple[float, float, float, float], float, bool]:
    """模态分歧：调 rerank 权重、粗排 λ 基底、是否建议事后 Nano 框融合。

    - Nano 高 & Hist@Nano 低：信平移，加强 hist 在 rerank/λ 中的比重。
    - Nano 低 & Hist@Nano 高：信颜色+形状，弱化事后 Nano 线性融合。
    """
    if _os_module.environ.get("MIS_FUSION_DISAGREE", "1").lower() in ("0", "false", "no", "off"):
        return base_weights, float(fuse_lambda_base), False
    wn, wc, wh, wz = [float(v) for v in base_weights]
    ns = float(np.clip(float(nano_conf), 0.0, 1.0))
    hh = float(np.clip(float(hist_at_nano_peak), 0.0, 1.0))
    lam = float(fuse_lambda_base)
    suggest_blend = False
    hi_n = float(_os_module.environ.get("MIS_DISAGREE_NANO_HI", "0.55"))
    lo_h = float(_os_module.environ.get("MIS_DISAGREE_HIST_LO", "0.38"))
    lo_n = float(_os_module.environ.get("MIS_DISAGREE_NANO_LO", "0.42"))
    hi_h = float(_os_module.environ.get("MIS_DISAGREE_HIST_HI", "0.52"))
    if ns >= hi_n and hh < lo_h:
        wn *= 1.12
        wh *= 1.22
        wz *= 0.92
        lam *= 0.85
    elif ns < lo_n and hh >= hi_h:
        wh *= 1.18
        wc *= 1.08
        wz *= 1.12
        wn *= 0.88
        lam *= 1.05
        suggest_blend = False
    ws = wn + wc + wh + wz
    if ws <= 1e-9:
        return base_weights, lam, suggest_blend
    return (wn / ws, wc / ws, wh / ws, wz / ws), lam, suggest_blend


def _effective_rerank_topr(
    rerank_topr: int,
    pos_c: list[tuple[int, int]],
    coarse_rank: np.ndarray,
    scores_c_full: np.ndarray,
    nano_prior_map: np.ndarray | None,
    nano_w: np.ndarray,
) -> int:
    """s_n 峰与轮廓峰错位时略增大 Top-R。"""
    base = max(0, int(rerank_topr))
    if base <= 0:
        return 0
    if _os_module.environ.get("MIS_DISAGREE_EXPAND_TOPR", "1").lower() in ("0", "false", "no", "off"):
        return base
    Nc = len(pos_c)
    if Nc <= 1:
        return base
    ci_rank = int(np.argmax(coarse_rank))
    ci_sc = int(np.argmax(scores_c_full))
    ci_nano = int(np.argmax(nano_w)) if nano_w.size else ci_rank
    spread = max(abs(ci_rank - ci_sc), abs(ci_rank - ci_nano))
    step = int(_os_module.environ.get("MIS_COARSE_STEP", "16"))
    if spread * max(1, step) >= int(_os_module.environ.get("MIS_DISAGREE_PEAK_PIX", "32")):
        mul = float(_os_module.environ.get("MIS_DISAGREE_TOPR_MUL", "1.35"))
        return min(Nc, max(base, int(round(base * mul))))
    return base


def _adaptive_rerank_weights(
    base: tuple[float, float, float, float],
    nano_conf: float,
) -> tuple[float, float, float, float]:
    """Nano 置信高 → 加大 s_n；低 → 加大 hist / 伪 3D。"""
    if _os_module.environ.get("MIS_RERANK_ADAPT_W", "1").lower() in ("0", "false", "no", "off"):
        return base
    wn, wc, wh, wz = [float(v) for v in base]
    ns = float(np.clip(float(nano_conf), 0.0, 1.0))
    wn *= 0.60 + 0.75 * ns
    wh *= 0.70 + 0.55 * (1.0 - ns)
    wz *= 0.80 + 0.35 * ns
    ws = wn + wc + wh + wz
    if ws <= 1e-9:
        return base
    return (wn / ws, wc / ws, wh / ws, wz / ws)


def _rerank_fusion_score(
    sn: float,
    sc: float,
    sh: float,
    sz: float,
    weights: tuple[float, float, float, float],
) -> float:
    wn, wc, wh, wz = weights
    mode = (_os_module.environ.get("MIS_RERANK_MODE", "linear") or "linear").strip().lower()
    sn, sc, sh, sz = (
        float(np.clip(sn, 1e-4, 1.0)),
        float(np.clip(sc, 1e-4, 1.0)),
        float(np.clip(sh, 1e-4, 1.0)),
        float(np.clip(sz, 1e-4, 1.0)),
    )
    if mode == "loggeom":
        log_s = 0.0
        for w, v in ((wn, sn), (wc, sc), (wh, sh), (wz, sz)):
            if w > 1e-8:
                log_s += w * float(np.log(v))
        return float(np.exp(log_s))
    return float(wn * sn + wc * sc + wh * sh + wz * sz)


def _sn_at_candidate(
    wy: int,
    wx: int,
    target_h: int,
    target_w: int,
    match_resp: np.ndarray | None,
    nano_prior_map: np.ndarray | None,
    fH: int,
    fW: int,
) -> float:
    """s_n：OpenCV matchTemplate 响应（左上对齐）优先；值 **NaN** 表示 ROI 外未计算，改采 Nano 先验中心；
    否则无先验图时回 0.5。"""
    if match_resp is not None and match_resp.size > 0:
        mr_h, mr_w = int(match_resp.shape[0]), int(match_resp.shape[1])
        if 0 <= wy < mr_h and 0 <= wx < mr_w:
            v = float(match_resp[wy, wx])
            if np.isnan(v):
                pass
            elif not np.isfinite(v):
                return 0.5
            elif -1.01 <= v <= 1.01:
                return float(np.clip(0.5 * (v + 1.0), 0.0, 1.0))
            else:
                return float(np.clip(v, 0.0, 1.0))
    if nano_prior_map is not None:
        cy_c = min(int(wy + target_h // 2), int(nano_prior_map.shape[0]) - 1)
        cx_c = min(int(wx + target_w // 2), int(nano_prior_map.shape[1]) - 1)
        return float(np.clip(float(nano_prior_map[cy_c, cx_c]), 0.0, 1.0))
    return 0.5


def _coarse_rerank_topr_unified(
    *,
    pos_c: list[tuple[int, int]],
    coarse_rank: np.ndarray,
    scores_c_full: np.ndarray,
    hist01: np.ndarray,
    use_hist_gate: bool,
    rerank_topr: int,
    match_resp: np.ndarray | None,
    nano_prior_map: np.ndarray | None,
    frame_bgr_full: np.ndarray | None,
    ref_gray_patch_sz: np.ndarray | None,
    plane_origin_yx: tuple[int, int],
    target_h: int,
    target_w: int,
    fH: int,
    fW: int,
    weights: tuple[float, float, float, float],
    hist_run_indices: np.ndarray | None = None,
    ref_depth_patch_sz: np.ndarray | None = None,
    nano_confidence: float = 0.0,
) -> int:
    """粗网格 Top‑R：s_n(Nano/模板) + 轮廓 + HistColor + s_z(灰度+伪深度 NCC)。

    hist_run_indices: 实际参与 batched_hist_correl 的粗格索引；其余索引上 hist01 保持为 0（Nano 门控跳过），
    再打分时不应把「未计算」当成极低 hist——对该类索引令 sh=0.5 中性。"""
    Nc = len(pos_c)
    if Nc <= 0:
        return 0
    take = max(1, min(int(rerank_topr), Nc))
    order = np.argsort(-coarse_rank.astype(np.float64))
    cand_ix = order[:take].astype(np.int64)
    wn, wc, wh, wz = _adaptive_rerank_weights(tuple(float(v) for v in weights), float(nano_confidence))
    oy, ox = int(plane_origin_yx[0]), int(plane_origin_yx[1])
    fh_img = int(frame_bgr_full.shape[0]) if frame_bgr_full is not None else 0
    fw_img = int(frame_bgr_full.shape[1]) if frame_bgr_full is not None else 0
    gray_plane_full = None
    if frame_bgr_full is not None and fh_img > 0 and fw_img > 0:
        gray_plane_full = (
            cv2.cvtColor(frame_bgr_full, cv2.COLOR_BGR2GRAY)
            if frame_bgr_full.ndim == 3
            else np.asarray(frame_bgr_full, dtype=np.uint8)
        )

    hist_ix_set: set[int] | None = None
    if use_hist_gate and hist_run_indices is not None and int(hist_run_indices.size) > 0:
        hist_ix_set = {int(j) for j in hist_run_indices.tolist()}

    best_i = int(cand_ix[0])
    best_s = -1e18
    for ii in cand_ix.tolist():
        wy, wx = pos_c[int(ii)]
        sn = _sn_at_candidate(wy, wx, target_h, target_w, match_resp, nano_prior_map, fH, fW)
        sc = float(np.clip(float(scores_c_full[int(ii)]), 0.0, 1.0))
        if use_hist_gate:
            if hist_ix_set is not None and int(ii) not in hist_ix_set:
                sh = 0.5
            else:
                sh = float(np.clip(float(hist01[int(ii)]), 0.0, 1.0))
        else:
            sh = 0.5
        sz = 0.5
        if gray_plane_full is not None and ref_gray_patch_sz is not None:
            gy_g = oy + wy
            gx_g = ox + wx
            gy2 = gy_g + int(target_h)
            gx2 = gx_g + int(target_w)
            if gy_g >= 0 and gx_g >= 0 and gy2 <= fh_img and gx2 <= fw_img:
                cand_gray = gray_plane_full[gy_g:gy2, gx_g:gx2]
                cand_depth = (
                    _pseudo_depth_u8_from_gray(cand_gray)
                    if ref_depth_patch_sz is not None
                    else None
                )
                sz = _sz_pseudo3d_combined(
                    ref_gray_patch_sz, cand_gray, ref_depth_patch_sz, cand_depth
                )
        score_u = _rerank_fusion_score(sn, sc, sh, sz, (wn, wc, wh, wz))
        if score_u > best_s:
            best_s = score_u
            best_i = int(ii)
    return best_i


def _build_lib_contour_stack_and_z(
    contour_sim_model: nn.Module | None,
    lib_templates_raw: list | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """序列级一次：模板光栅 (N,1,S,S) + 归一化编码向量 lib_z (N,D)。"""
    if contour_sim_model is None or not lib_templates_raw:
        return None, None
    enc = getattr(contour_sim_model, "encode", None)
    if not callable(enc):
        return None, None
    device = next(contour_sim_model.parameters()).device
    contours: list[torch.Tensor] = []
    for lib_img in lib_templates_raw:
        if not isinstance(lib_img, np.ndarray):
            continue
        if lib_img.ndim == 3 and lib_img.shape[2] == 3:
            lib_gray = cv2.cvtColor(lib_img, cv2.COLOR_BGR2GRAY)
        else:
            lib_gray = lib_img
        lib_contour = _rasterize_contour(lib_gray, 128)
        contours.append(
            torch.from_numpy(np.asarray(lib_contour, dtype=np.float32))[None, None, :, :],
        )
    if not contours:
        return None, None
    lib_tp = torch.cat(contours, dim=0).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        lib_z = enc(lib_tp).to(dtype=torch.float32)
    return lib_tp, lib_z


def _cos_to_sim01(cos: torch.Tensor) -> torch.Tensor:
    return ((cos.clamp(-1.0 + 1e-5, 1.0 - 1e-5) + 1.0) * 0.5).to(dtype=torch.float32)


def _raster_stack_from_plane(
    positions: list[tuple[int, int]],
    plane: np.ndarray,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """P×128×128 轮廓栅格，与逐点 _rasterize_contour 一致。
    优化：预转换灰度图 + 跳过空白 patch（<8 非零像素则 _rasterize_contour 返回零矩阵）。"""
    p_n = len(positions)
    s = 128
    stack = np.zeros((p_n, s, s), dtype=np.float32)

    # 预转换整幅 plane 为灰度（避免逐 patch cvtColor）
    if plane.ndim == 3 and plane.shape[2] == 3:
        plane_gray = cv2.cvtColor(plane, cv2.COLOR_BGR2GRAY)
    elif plane.ndim == 3:
        plane_gray = plane[:, :, 0]
    else:
        plane_gray = plane

    # 预计算 plane_gray 的积分图用于快速统计非零像素数
    plane_bin = (plane_gray > 0).astype(np.float32)
    sat = cv2.integral(plane_bin)  # (H+1, W+1)

    for i, (wy, wx) in enumerate(positions):
        # 快速检测：非零像素 < 8 则跳过（_rasterize_contour 返回零矩阵）
        wy2 = min(wy + target_h, plane_gray.shape[0])
        wx2 = min(wx + target_w, plane_gray.shape[1])
        nz_cnt = sat[wy2, wx2] - sat[wy, wx2] - sat[wy2, wx] + sat[wy, wx]
        if nz_cnt < 8:
            continue  # stack[i] 已经是零矩阵

        # 直接从 plane_gray 裁剪（无需逐 patch cvtColor）
        p_gray = plane_gray[wy:wy + target_h, wx:wx + target_w]
        if p_gray.shape[0] < target_h or p_gray.shape[1] < target_w:
            p_gray = cv2.copyMakeBorder(
                p_gray,
                0, max(0, target_h - p_gray.shape[0]),
                0, max(0, target_w - p_gray.shape[1]),
                cv2.BORDER_CONSTANT, value=0,
            )
        stack[i] = np.asarray(_rasterize_contour(p_gray, 128), dtype=np.float32)
    return stack


def _score_positions_batched_embeddings(
    positions: list[tuple[int, int]],
    plane: np.ndarray,
    target_h: int,
    target_w: int,
    contour_sim_model: nn.Module,
    lib_z: torch.Tensor,
    infer_bs: int,
) -> np.ndarray:
    if not positions:
        return np.zeros((0,), dtype=np.float32)
    device = next(contour_sim_model.parameters()).device
    enc = getattr(contour_sim_model, "encode", None)
    if lib_z is None or not callable(enc):
        return np.zeros((len(positions),), dtype=np.float32)

    rz = _raster_stack_from_plane(positions, plane, target_h, target_w)
    p_n = rz.shape[0]
    lz = lib_z.to(device=device, dtype=torch.float32)
    out = np.empty((p_n,), dtype=np.float32)
    bs = max(1, int(infer_bs))
    rz_gpu = torch.as_tensor(rz, dtype=torch.float32, device=device)

    with torch.inference_mode():
        for s0 in range(0, p_n, bs):
            s1 = min(p_n, s0 + bs)
            xb = rz_gpu[s0:s1].unsqueeze(1).contiguous()
            zp = enc(xb).to(dtype=torch.float32)
            cos = zp @ lz.transpose(0, 1)
            sim = _cos_to_sim01(cos)
            out[s0:s1] = sim.max(dim=1).values.float().cpu().numpy()
    return out


def _score_positions_legacy_loop(
    positions: list[tuple[int, int]],
    plane: np.ndarray,
    target_h: int,
    target_w: int,
    contour_sim_model: nn.Module,
    lib_tp: torch.Tensor,
) -> np.ndarray:
    device = next(contour_sim_model.parameters()).device
    lib_tensor = lib_tp.to(device=device)
    out_list: list[float] = []
    for (wy, wx) in positions:
        p = plane[wy : wy + target_h, wx : wx + target_w]
        if p.shape[0] < target_h or p.shape[1] < target_w:
            p = cv2.copyMakeBorder(
                p,
                0,
                max(0, target_h - p.shape[0]),
                0,
                max(0, target_w - p.shape[1]),
                cv2.BORDER_CONSTANT,
                value=0,
            )
        if p.ndim == 3 and p.shape[2] == 3:
            p_gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        elif p.ndim == 2:
            p_gray = p
        else:
            p_gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        patch_contour = _rasterize_contour(p_gray, 128)
        patch_tensor = (
            torch.from_numpy(np.asarray(patch_contour, dtype=np.float32))[None, ...]
            .to(device)
            .unsqueeze(0)
        )
        with torch.inference_mode():
            patch_expanded = patch_tensor.expand(lib_tensor.shape[0], -1, -1, -1).contiguous()
            pred_sim, _, _ = contour_sim_model(patch_expanded, lib_tensor)
            out_list.append(float(pred_sim.max().item()))
    return np.asarray(out_list, dtype=np.float32)


def _build_rtw_bgr(
    target_roi: np.ndarray,
    mask: np.ndarray,
    num_transforms: int = 128,
) -> list:
    """
    RTW 增广: 旋转·平移·缩放·剪切·透视·翻转·组合 → 最多 num_transforms 张
    优化：ThreadPoolExecutor 并行执行 warpAffine/warpPerspective（OpenCV 释放 GIL）。
    """
    from concurrent.futures import ThreadPoolExecutor

    h, w = target_roi.shape[:2]

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
    mask_thresh = orig_white * 0.8

    # ── 阶段1：收集所有变换任务 ──────────────────────────────────
    # 任务格式: (M, is_persp, img_src, mask_src)
    #   is_persp=False → warpAffine; is_persp=True → warpPerspective
    _tc, _mc = target_centered, mask_centered
    tasks: list[tuple] = []

    def _add_affine(M):
        tasks.append((M, False, _tc, _mc))

    def _add_persp(M):
        tasks.append((M, True, _tc, _mc))

    # 1. Rotation -60…+60 step 5 (25)
    for ang in range(-60, 65, 5):
        _add_affine(cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0))
    # 2. Scale (13)
    for sc in [0.35,0.45,0.55,0.65,0.75,0.85,0.95,1.05,1.15,1.25,1.35,1.45,1.55]:
        _add_affine(cv2.getRotationMatrix2D((w / 2, h / 2), 0, sc))
    # 3. Translate X (8)
    for tx in [-0.20,-0.15,-0.10,-0.05,0.05,0.10,0.15,0.20]:
        _add_affine(np.float32([[1,0,tx*w],[0,1,0]]))
    # 4. Translate Y (8)
    for ty in [-0.20,-0.15,-0.10,-0.05,0.05,0.10,0.15,0.20]:
        _add_affine(np.float32([[1,0,0],[0,1,ty*h]]))
    # 5. Diagonal translate (8)
    for (tx,ty) in [(-0.15,-0.15),(-0.15,0.15),(0.15,-0.15),(0.15,0.15),
                    (-0.10,-0.10),(-0.10,0.10),(0.10,-0.10),(0.10,0.10)]:
        _add_affine(np.float32([[1,0,tx*w],[0,1,ty*h]]))
    # 6. Shear X (6)
    for sx in [-0.30,-0.20,-0.10,0.10,0.20,0.30]:
        _add_affine(np.float32([[1,sx,0],[0,1,0]]))
    # 7. Shear Y (6)
    for sy in [-0.30,-0.20,-0.10,0.10,0.20,0.30]:
        _add_affine(np.float32([[1,0,0],[sy,1,0]]))
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
            _add_persp(cv2.getPerspectiveTransform(src, dst))
    # 9. Flip (2) — 直接加入任务列表（is_persp=False, M=None 表示仅翻转）
    for fc in [1, 0]:
        flipped = cv2.flip(target_centered, fc)
        flipped_mask = cv2.flip(mask_centered, fc)
        tasks.append((None, False, flipped, flipped_mask))
    # 10. Rotation+Translation combined (8)
    for (ang,tx,ty) in [(-30,0.10,0),(30,-0.10,0),(0,0.10,0.10),(0,-0.10,-0.10),
                        (-15,0.10,0.10),(15,-0.10,0.10),(-15,-0.10,-0.10),(15,0.10,-0.10)]:
        M = cv2.getRotationMatrix2D((w/2,h/2), ang, 1.0)
        M[0,2] += tx*w; M[1,2] += ty*h
        _add_affine(M)
    # 11. Extended rotation -90..-63 and 63..90, step 3 (~21)
    for ang in list(range(-90, -60, 3)) + list(range(63, 93, 3)):
        _add_affine(cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0))
    # 12. Rotation + Scale combined 8 angles × 4 scales (32)
    for ang in [-60, -45, -30, -15, 15, 30, 45, 60]:
        for sc in [0.70, 0.85, 1.15, 1.30]:
            _add_affine(cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc))
    # 13. Extended translate X/Y ±[0.25, 0.30] (8)
    for tx in [-0.30, -0.25, 0.25, 0.30]:
        _add_affine(np.float32([[1, 0, tx*w], [0, 1, 0]]))
    for ty in [-0.30, -0.25, 0.25, 0.30]:
        _add_affine(np.float32([[1, 0, 0], [0, 1, ty*h]]))
    # 14. Perspective extended strengths 4×2 (8)
    for ci in range(4):
        for st in [0.25, 0.30]:
            dst = src.copy()
            dw, dh = int(st*w), int(st*h)
            if   ci==0: dst[0]=[dw,dh]
            elif ci==1: dst[1]=[w-1-dw,dh]
            elif ci==2: dst[2]=[w-1-dw,h-1-dh]
            else:       dst[3]=[dw,h-1-dh]
            _add_persp(cv2.getPerspectiveTransform(src, dst))
    # 15. Shear X+Y combined (12)
    for (sx, sy) in [(-0.20,-0.20),(-0.20, 0.20),(0.20,-0.20),(0.20, 0.20),
                     (-0.15, 0.15),(0.15,-0.15),(-0.10, 0.10),(0.10,-0.10),
                     (-0.25, 0.10),(0.25,-0.10),(-0.10, 0.25),(0.10,-0.25)]:
        _add_affine(np.float32([[1, sx, 0], [sy, 1, 0]]))
    # 16. Scale extreme values (8)
    for sc in [0.20, 0.25, 0.30, 1.65, 1.75, 1.85, 1.95, 2.0]:
        _add_affine(cv2.getRotationMatrix2D((w / 2, h / 2), 0, sc))
    # 17. Flip + Rotation combined (22) — 预计算翻转，避免重复 cv2.flip
    for fc in [1, 0]:
        img_f  = cv2.flip(target_centered, fc)
        mask_f = cv2.flip(mask_centered, fc)
        for ang in [-60, -45, -30, -15, 0, 15, 30, 45, 60, -75, 75]:
            M = cv2.getRotationMatrix2D((pad_w/2, pad_h/2), ang, 1.0)
            tasks.append((M, False, img_f, mask_f))
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
        M = cv2.getRotationMatrix2D((w/2,h/2), ang, sc)
        M[0,2] += tx*w; M[1,2] += ty*h
        _add_affine(M)

    # ── 阶段2：并行执行变换 ──────────────────────────────────
    def _apply_task(task):
        """单个变换：warp + mask完整性检查，返回 BGR ndarray 或 None。"""
        M, is_persp, img_src, mask_src = task
        if M is None:
            # 翻转直通（item 9）：无 warp，直接检查 mask
            warped, warped_mask = img_src, mask_src
        elif is_persp:
            warped = cv2.warpPerspective(img_src, M, (pad_w, pad_h))
            warped_mask = cv2.warpPerspective(mask_src, M, (pad_w, pad_h))
        else:
            warped = cv2.warpAffine(img_src, M, (pad_w, pad_h))
            warped_mask = cv2.warpAffine(mask_src, M, (pad_w, pad_h))
        if np.count_nonzero(warped_mask > 0) >= mask_thresh:
            return warped
        return None

    n_workers = min(4, max(1, (os.cpu_count() or 1)))
    imgs: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_apply_task, tasks):
            if result is not None:
                imgs.append(result)

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
    contour_sim_model: nn.Module | None = None,
    lib_templates_raw: list | None = None,
    lib_contour_encodings: np.ndarray | None = None,
    step_coarse: int = 32,
    step_fine: int = 8,
    fine_expand: int = 1,
    top_k: int = 3,
    greedy_step: int = 4,
    greedy_iters: int = 4,
    spatial_sigma: float = 3.0,
    center_yx: tuple | None = None,
    min_white_ratio: float = 0.03,
    early_exit_score: float = 0.90,
    *,
    coarse_scheme: str = "ab",
    frame_bgr_full: np.ndarray | None = None,
    plane_origin_yx: tuple[int, int] = (0, 0),
    ref_hist: np.ndarray | None = None,
    hist_h_bins: int = 30,
    hist_s_bins: int = 32,
    hist_gate_T: float = 0.08,
    hist_top_M_cap: int = 96,
    hist_min_keep: int = 24,
    hist_fallback_take: int = 48,
    hist_topkp_c: int = 96,
    fuse_lambda_base: float = 0.65,
    fuse_adaptive: bool = True,
    fine_fuse_hist: bool = True,
    greedy_fuse_hist: bool = True,
    early_exit_hist_min: float = 0.0,
    early_exit_polish: bool = True,
    contour_sat_thresh: float = 0.97,
    contour_sat_lambda_scale: float = 1.0,
    contour_infer_bs: int = 48,
    cached_lib_tensor: torch.Tensor | None = None,
    cached_lib_z: torch.Tensor | None = None,
    max_time_s: float = 0.0,
    nano_prior_map: "np.ndarray | None" = None,
    contour_fuse_weight: float = 1.0,
    nano_fusion: str = "product",
    nano_prior_beta: float = 0.45,
    nano_hist_coupling: float = 0.5,
    rerank_topr: int = 0,
    match_resp: "np.ndarray | None" = None,
    ref_gray_patch_sz: "np.ndarray | None" = None,
    rerank_weights: tuple[float, float, float, float] | None = None,
    ref_depth_patch_sz: "np.ndarray | None" = None,
    nano_confidence: float = 0.0,
    contour_fuse_lambda_override: float | None = None,
) -> dict:
    """三阶段层次搜索；粗搜可选用方案 A+B（hist 预筛+CNN）或 C（hist×空间 Top‑K′ 再 CNN）。

    max_time_s: >0 时启用每帧超时保护；超时后立即返回当前最优结果，避免卡死。
    nano_prior_map: 可选 (fH, fW) float32 空间先验图（与 search_plane_th68 同坐标系）。
        值越大表示 NanoTrack 认为目标越可能在该位置。用于在 hist 门控前对候选加权，
        实现 NanoTrack 深度特征 × HistColor × ContourSim 的乘积式三路融合。
    nano_fusion: \"product\"（旧：hist×nano 双重压制）或 \"log\"（推荐：hist 软耦合 + rank 上乘 nano**β）。
    nano_prior_beta: log 模式下 rank 中 nano 权重指数，典型 0.35–0.65。
    nano_hist_coupling: log 模式下 hist 门控对 nano 的耦合强度 γ，hist_eff ∝ hist × nano^γ。
    rerank_topr: >0 时在粗网格上对 Top‑R 候选做二次加权排序（模板响应 / Nano / 轮廓 / Hist / s_z）。
        若启用 Nano hist 门控：未参与 batched_hist_correl 的格点在再打分中 hist 取中性 0.5（hist01 仍为 0）。
    match_resp: cv2.matchTemplate(..., TM_CCOEFF_NORMED) 输出，形状 (fH−th+1, fW−tw+1)，与候选左上角 (wy,wx) 对齐采样。
        允许部分为 NaN（例如 Scheme C 仅在局部 ROI 内算相关、域外未填），该处 ``_sn_at_candidate`` 会回落到 ``nano_prior_map``。
    ref_gray_patch_sz: 首帧目标灰度小块 (target_h, target_w)，供 s_z masked NCC。
    ref_depth_patch_sz: 首帧伪深度 u8，与 ref_gray 同尺寸；MIS_SZ_PSEUDO3D=1 时参与 s_z。
    nano_confidence: 当前帧 NanoTrack score，供 MIS_RERANK_ADAPT_W 调节 Top‑R 权重。
    rerank_weights: (w_sn, w_sc, w_sh, w_sz)，默认内部均衡归一化。
    contour_fuse_weight: ∈[0,1]，将 ContourSim 在 λ 融合中的有效系数降为 λ·weight；
        0 表示 coarse/fine/greedy 中仅用 Hist（缓解实例混淆）；1 为原始行为。
    """
    _ = spatial_sigma
    scheme_in = (coarse_scheme or "ab").strip().lower()
    scheme = scheme_in if scheme_in in ("ab", "c") else "ab"
    _t0 = _time_mod.perf_counter() if max_time_s > 0 else 0.0
    _timed_out = False

    def _check_timeout() -> bool:
        nonlocal _timed_out
        if max_time_s <= 0 or _timed_out:
            return False
        if _time_mod.perf_counter() - _t0 >= max_time_s:
            _timed_out = True
            return True
        return False

    fH, fW = search_plane_th68.shape[:2]
    cy = float(center_yx[0]) if center_yx is not None else fH / 2.0
    cx = float(center_yx[1]) if center_yx is not None else fW / 2.0

    w_cf = float(np.clip(float(contour_fuse_weight), 0.0, 1.0))

    def _lam_effective(lam: float) -> float:
        return float(np.clip(float(lam) * w_cf, 0.0, 1.0))

    def _blend_contour_hist(lam: float, nn_sub: np.ndarray, hh_sub: np.ndarray) -> np.ndarray:
        le = _lam_effective(lam)
        return (
            le * _normalize_01(nn_sub) + (1.0 - le) * hh_sub.astype(np.float32)
        ).astype(np.float32)

    sigma_y = max(1.0, fH / 4.0)
    sigma_x = max(1.0, fW / 4.0)

    plane_bin = (
        search_plane_th68[:, :, 0].astype(np.float32) > 0
        if search_plane_th68.ndim == 3
        else (search_plane_th68.astype(np.float32) > 0)
    ).astype(np.float32)
    sat = cv2.integral(plane_bin)
    min_count = min_white_ratio * target_h * target_w

    lib_tensor_precomputed = cached_lib_tensor
    lib_z_precomputed = cached_lib_z
    if contour_sim_model is not None and lib_templates_raw:
        enc_fn = getattr(contour_sim_model, "encode", None)
        if lib_tensor_precomputed is None:
            device = next(contour_sim_model.parameters()).device
            lib_contours: list[torch.Tensor] = []
            for lib_img in lib_templates_raw:
                if isinstance(lib_img, np.ndarray):
                    if lib_img.ndim == 3 and lib_img.shape[2] == 3:
                        lib_gray = cv2.cvtColor(lib_img, cv2.COLOR_BGR2GRAY)
                    else:
                        lib_gray = lib_img
                    lib_contour = _rasterize_contour(lib_gray, 128)
                    lib_contours.append(
                        torch.from_numpy(np.asarray(lib_contour, dtype=np.float32))[None, None, :, :],
                    )
            if lib_contours:
                lib_tensor_precomputed = torch.cat(lib_contours, dim=0).to(device=device, dtype=torch.float32)
                print(f"[PRECOMPUTE] lib_tensor shape={lib_tensor_precomputed.shape}, templates={len(lib_contours)}")
        if (
            lib_tensor_precomputed is not None
            and lib_z_precomputed is None
            and callable(enc_fn)
        ):
            with torch.inference_mode():
                lib_z_precomputed = enc_fn(lib_tensor_precomputed).to(dtype=torch.float32)
    elif lib_tensor_precomputed is None:
        lib_z_precomputed = None

    lib_z_torch = lib_z_precomputed

    def _score_positions_inner(pos_list: list[tuple[int, int]], plane_np: np.ndarray) -> np.ndarray:
        if contour_sim_model is None or lib_tensor_precomputed is None:
            return np.zeros(len(pos_list), dtype=np.float32)
        lz = lib_z_torch
        enc_fn = getattr(contour_sim_model, "encode", None)
        if callable(enc_fn) and lz is not None:
            return _score_positions_batched_embeddings(
                pos_list,
                plane_np,
                target_h,
                target_w,
                contour_sim_model,
                lz,
                contour_infer_bs,
            )
        return _score_positions_legacy_loop(
            pos_list,
            plane_np,
            target_h,
            target_w,
            contour_sim_model,
            lib_tensor_precomputed,
        )

    def _white_cnt(wy: int, wx: int):
        wy2 = min(wy + target_h, fH)
        wx2 = min(wx + target_w, fW)
        return sat[wy2, wx2] - sat[wy, wx2] - sat[wy2, wx] + sat[wy, wx]

    def _grid_positions(step: int, plane_h: int, plane_w: int):
        ys = list(range(0, max(1, plane_h - target_h + 1), step))
        xs = list(range(0, max(1, plane_w - target_w + 1), step))
        if ys and (ys[-1] + target_h) < plane_h:
            ys.append(plane_h - target_h)
        if xs and (xs[-1] + target_w) < plane_w:
            xs.append(plane_w - target_w)
        return ys or [0], xs or [0]

    def _fused(scores, positions, c_y, c_x):
        arr = np.array(positions, dtype=np.float32)
        dy = (arr[:, 0] + target_h / 2.0 - c_y) / sigma_y
        dx = (arr[:, 1] + target_w / 2.0 - c_x) / sigma_x
        return scores * np.exp(-0.5 * (dy**2 + dx**2)).astype(np.float32)

    greedy_dirs = (
        (-greedy_step, 0),
        (greedy_step, 0),
        (0, -greedy_step),
        (0, greedy_step),
        (-greedy_step, -greedy_step),
        (-greedy_step, greedy_step),
        (greedy_step, -greedy_step),
        (greedy_step, greedy_step),
    )

    _fuse_lam_base_eff = float(fuse_lambda_base)
    if contour_fuse_lambda_override is not None:
        _fuse_lam_base_eff = float(contour_fuse_lambda_override)
    _rw_agg: tuple[float, float, float, float] = (
        tuple(float(x) for x in rerank_weights)
        if rerank_weights is not None
        else (0.30, 0.32, 0.22, 0.16)
    )
    _suggest_nano_blend = False

    def _fuse_lam(mx_c: float) -> float:
        return _fuse_lambda_hist_accuracy(
            _fuse_lam_base_eff,
            mx_c,
            fuse_adaptive,
            contour_sat_thresh,
            contour_sat_lambda_scale,
        )

    ys_c, xs_c = _grid_positions(step_coarse, fH, fW)
    pos_c = [
        (wy, wx)
        for wy in ys_c
        for wx in xs_c
        if 0 <= wy <= fH - target_h and 0 <= wx <= fW - target_w and _white_cnt(wy, wx) >= min_count
    ]
    if not pos_c:
        pos_c = [(max(0, int(cy - target_h // 2)), max(0, int(cx - target_w // 2)))]

    Nc = len(pos_c)
    _fuse_mode = (nano_fusion or "product").strip().lower()
    _beta = float(np.clip(float(nano_prior_beta), 0.0, 2.0))
    _gamma = float(np.clip(float(nano_hist_coupling), 0.0, 1.0))

    use_hist_gate = (
        hist_gate_T is not None
        and float(hist_gate_T) >= 0.0
        and frame_bgr_full is not None
        and ref_hist is not None
        and scheme in ("ab", "c")
    )

    coarse_rank = np.full(Nc, -1e9, dtype=np.float32)
    scores_c_full = np.zeros(Nc, dtype=np.float32)
    hist01 = np.zeros(Nc, dtype=np.float32)

    # ── NanoTrack 空间注意力先验 ────────────────────────────────────────────
    # nano_prior_map 为 (fH, fW) float32，与 search_plane_th68 同坐标系。
    # 对每个候选位置采样其中心点的 NanoTrack 前景概率，归一化为 [0,1]。
    # 深度融合策略：
    #   1. 用先验预筛选：只对高先验候选计算昂贵的 hist_correl（大目标加速关键）
    #   2. 乘积融合：coarse_rank ∝ nano_prior × hist_color × contour_sim × spatial
    nano_w = np.ones(Nc, dtype=np.float32)
    # hist_correl 计算的候选子集（默认全部，有 nano_prior 时按先验过滤）
    hist_run_indices = np.arange(Nc)
    if nano_prior_map is not None:
        nmap_h, nmap_w = nano_prior_map.shape[:2]
        for i, (wy, wx) in enumerate(pos_c):
            cy_c = min(int(wy + target_h // 2), nmap_h - 1)
            cx_c = min(int(wx + target_w // 2), nmap_w - 1)
            nano_w[i] = float(nano_prior_map[cy_c, cx_c])
        nmax = float(nano_w.max())
        if nmax > 1e-6:
            nano_w = nano_w / nmax   # 归一化：最优候选 nano_w=1.0

        # 预筛选：只对先验 >= 5% 的位置计算 hist_correl（大目标核心加速）
        # 最少保留 hist_min_keep 个，防止 NanoTrack 不确定时完全失效
        _NANO_GATE = 0.05
        # 最少保留数量：不超过实际候选数 Nc（防止 Nc 很小时 argpartition 越界）
        _NANO_HIST_MIN = min(max(int(hist_min_keep), 16), Nc)
        gate_mask = nano_w >= _NANO_GATE
        if int(gate_mask.sum()) < _NANO_HIST_MIN:
            # NanoTrack 先验分散（不确定）→ 保留先验最高的 _NANO_HIST_MIN 个
            keep_idx = np.argpartition(nano_w, -_NANO_HIST_MIN)[-_NANO_HIST_MIN:]
            gate_mask = np.zeros(Nc, dtype=bool)
            gate_mask[keep_idx] = True
        hist_run_indices = np.where(gate_mask)[0]
    # ────────────────────────────────────────────────────────────────────────

    if use_hist_gate:
        pos_c_hist = [pos_c[int(i)] for i in hist_run_indices]
        hist01_sub = batched_hist_correl_masked(
            frame_bgr_full,
            search_plane_th68,
            plane_origin_yx,
            pos_c_hist,
            target_h,
            target_w,
            ref_hist,
            hist_h_bins,
            hist_s_bins,
        )
        hist01[hist_run_indices] = hist01_sub
        if nano_prior_map is not None and Nc > 0:
            _ni = int(np.argmax(nano_w))
            _h_at_nano = float(hist01[_ni]) if 0 <= _ni < Nc else 0.5
            _rw_agg, _fuse_lam_base_eff, _suggest_nano_blend = _fusion_disagreement_adjust(
                _rw_agg,
                _fuse_lam_base_eff,
                float(nano_confidence),
                _h_at_nano,
            )
        # 未参与 hist_correl 的位置保持 0（被 nano_prior 排除）
        sp_all = _spatial_gauss_xy(pos_c, target_h, target_w, cy, cx, sigma_y, sigma_x)
        # product：hist × nano（旧）；log：hist × nano^γ 软耦合，避免 hist 被 nano 双重乘进 rank
        if nano_prior_map is not None and _fuse_mode == "log":
            hist01_eff = hist01 * np.power(np.clip(nano_w, 1e-4, 1.0), _gamma)
        else:
            hist01_eff = hist01 * nano_w
        if scheme == "ab":
            surv = pick_survivors_scheme_ab(
                hist01_eff, float(hist_gate_T), int(hist_top_M_cap),
                int(hist_min_keep), int(hist_fallback_take),
            )
            sub = [pos_c[int(i)] for i in surv.tolist()]
            nn_sub = _score_positions_inner(sub, search_plane_th68)
            max_nn = float(np.max(nn_sub)) if nn_sub.size else 0.0
            lam_e = _fuse_lam(max_nn)
            # 融合：ContourSim × HistColor × Spatial × Nano（nano 在 log 模式下仅幂次乘入 rank）
            if nano_prior_map is not None and _fuse_mode == "log":
                hh_blend = hist01[surv].astype(np.float32)
                fused_mix = _blend_contour_hist(lam_e, nn_sub, hh_blend)
                nw_r = np.power(np.maximum(nano_w[surv], 1e-4), _beta)
                rank_surv = (fused_mix * sp_all[surv] * nw_r).astype(np.float32)
            else:
                fused_mix = _blend_contour_hist(lam_e, nn_sub, hist01_eff[surv])
                rank_surv = (fused_mix * sp_all[surv] * nano_w[surv]).astype(np.float32)
            scores_c_full[surv] = nn_sub
            coarse_rank[surv] = rank_surv
            if not _MIS_SILENT: print(f"[COARSE/AB] coarse={Nc} survivors={surv.size} λ={lam_e:.3f} hist_T={hist_gate_T}")
        else:
            pref = hist01_eff * sp_all
            surv = pick_survivors_scheme_c(pref, int(hist_topkp_c))
            sub = [pos_c[int(i)] for i in surv.tolist()]
            nn_sub = _score_positions_inner(sub, search_plane_th68)
            nn_n = _normalize_01(nn_sub)
            mix_c = (w_cf * nn_n + (1.0 - w_cf) * hist01_eff[surv].astype(np.float32)).astype(
                np.float32
            )
            if nano_prior_map is not None and _fuse_mode == "log":
                nw_r = np.power(np.maximum(nano_w[surv], 1e-4), _beta)
                rank_surv = (mix_c * sp_all[surv] * nw_r).astype(np.float32)
            else:
                rank_surv = (mix_c * sp_all[surv] * nano_w[surv]).astype(np.float32)
            scores_c_full[surv] = nn_sub
            coarse_rank[surv] = rank_surv
            if not _MIS_SILENT: print(f"[COARSE/C] coarse={Nc} topkp={int(hist_topkp_c)} survivors={surv.size}")
    else:
        scores_c_full[:] = _score_positions_inner(pos_c, search_plane_th68)
        fused_base = _fused(scores_c_full, pos_c, cy, cx)
        if nano_prior_map is not None and _fuse_mode == "log":
            coarse_rank[:] = fused_base * np.power(np.maximum(nano_w, 1e-4), _beta).astype(np.float32)
        else:
            coarse_rank[:] = fused_base * nano_w

    kk = max(1, min(int(top_k), Nc))
    topk_local = np.argpartition(coarse_rank, -kk)[-kk:]
    _best_ci = int(topk_local[np.argmax(coarse_rank[topk_local])])

    _rerank_topr_eff = _effective_rerank_topr(
        int(rerank_topr), pos_c, coarse_rank, scores_c_full, nano_prior_map, nano_w
    )
    if int(_rerank_topr_eff) > 0:
        try:
            best_rr = _coarse_rerank_topr_unified(
                pos_c=pos_c,
                coarse_rank=coarse_rank,
                scores_c_full=scores_c_full,
                hist01=hist01,
                use_hist_gate=use_hist_gate,
                rerank_topr=int(_rerank_topr_eff),
                match_resp=match_resp,
                nano_prior_map=nano_prior_map,
                frame_bgr_full=frame_bgr_full,
                ref_gray_patch_sz=ref_gray_patch_sz,
                plane_origin_yx=plane_origin_yx,
                target_h=target_h,
                target_w=target_w,
                fH=fH,
                fW=fW,
                weights=_rw_agg,
                hist_run_indices=hist_run_indices,
                ref_depth_patch_sz=ref_depth_patch_sz,
                nano_confidence=float(nano_confidence),
            )
            _best_ci = int(best_rr)
            tl = topk_local.tolist()
            if _best_ci not in tl:
                worst_j = int(np.argmin(np.asarray([float(coarse_rank[j]) for j in tl])))
                tl[worst_j] = _best_ci
            topk_local = np.asarray(tl, dtype=np.int64)
            if not _MIS_SILENT:
                print(f"[RERANK] topr={int(_rerank_topr_eff)} chose_ci={_best_ci} weights={_rw_agg}")
        except Exception as _e_r:
            if not _MIS_SILENT:
                print(f"[RERANK] skipped: {_e_r}")

    topk_idx = topk_local
    _coarse_y, _coarse_x = pos_c[_best_ci]
    _coarse_score = float(scores_c_full[_best_ci])

    if scores_c_full[_best_ci] >= early_exit_score:
        hist_ok = True
        if float(early_exit_hist_min) > 0.0 and scheme == "ab" and use_hist_gate:
            hist_ok = float(hist01[_best_ci]) >= float(early_exit_hist_min)
        if hist_ok:
            py_final, px_final = int(pos_c[_best_ci][0]), int(pos_c[_best_ci][1])
            best_score_out = float(scores_c_full[_best_ci])
            hh_dbg = float(hist01[_best_ci]) if use_hist_gate else -1.0
            if (
                early_exit_polish
                and greedy_fuse_hist
                and scheme == "ab"
                and use_hist_gate
                and frame_bgr_full is not None
                and ref_hist is not None
                and contour_sim_model is not None
            ):
                cand_e: list[tuple[int, int]] = [(py_final, px_final)]
                for dy_g, dx_g in greedy_dirs:
                    ny, nx = py_final + dy_g, px_final + dx_g
                    if 0 <= ny <= fH - target_h and 0 <= nx <= fW - target_w:
                        cand_e.append((ny, nx))
                sc_e = _score_positions_inner(cand_e, search_plane_th68)
                hh_e = batched_hist_correl_masked(
                    frame_bgr_full,
                    search_plane_th68,
                    plane_origin_yx,
                    cand_e,
                    target_h,
                    target_w,
                    ref_hist,
                    hist_h_bins,
                    hist_s_bins,
                )
                mx_e = float(np.max(sc_e)) if sc_e.size else 0.0
                lam_e_pol = _fuse_lam(mx_e)
                mix_e = _blend_contour_hist(lam_e_pol, sc_e, hh_e.astype(np.float32))
                be = int(np.argmax(mix_e))
                py_final, px_final = cand_e[be]
                best_score_out = float(sc_e[be])
            print(
                f"[EARLY_EXIT] coarse contour={best_score_out:.4f}>={early_exit_score} "
                f"hist={hh_dbg:.3f} thresh_hist_min={early_exit_hist_min}",
            )
            return {
                "score": float(best_score_out),
                "x": int(px_final),
                "y": int(py_final),
                "w": target_w,
                "h": target_h,
                "coarse_x": int(_coarse_x),
                "coarse_y": int(_coarse_y),
                "coarse_score": float(scores_c_full[_best_ci]),
                "hist_sim": float(hh_dbg) if use_hist_gate else -1.0,
            }
        print(
            f"[EARLY_DEFER_FINE] contour={scores_c_full[_best_ci]:.4f} but "
            f"hist={float(hist01[_best_ci]):.3f} < {early_exit_hist_min} → run fine/greedy",
        )

    print(
        f"[COARSE] candidates={Nc} anchors={kk} best_mix={float(coarse_rank[_best_ci]):.4f} "
        f"best_contour={_coarse_score:.4f}",
    )

    best_score = float(scores_c_full[_best_ci])
    best_gy, best_gx = int(pos_c[_best_ci][0]), int(pos_c[_best_ci][1])

    for ki in topk_idx.tolist():
        if _check_timeout():
            print(f"[TIMEOUT] Fine search interrupted after {_time_mod.perf_counter()-_t0:.1f}s")
            break
        ky, kx = pos_c[int(ki)]
        pad_y = int(target_h * fine_expand)
        pad_x = int(target_w * fine_expand)
        fy1 = max(0, ky - pad_y)
        fx1 = max(0, kx - pad_x)
        fy2 = min(fH, ky + target_h + pad_y)
        fx2 = min(fW, kx + target_w + pad_x)
        fine_plane = search_plane_th68[fy1:fy2, fx1:fx2]
        ph, pw = fine_plane.shape[:2]
        ys_f, xs_f = _grid_positions(step_fine, ph, pw)
        pos_f = [
            (wy, wx)
            for wy in ys_f
            for wx in xs_f
            if 0 <= wy <= ph - target_h
            and 0 <= wx <= pw - target_w
            and _white_cnt(fy1 + wy, fx1 + wx) >= min_count
        ]
        if not pos_f:
            continue
        scores_f = _score_positions_inner(pos_f, fine_plane)
        fine_cy = ky - fy1 + target_h / 2.0
        fine_cx = kx - fx1 + target_w / 2.0
        rank_scores = scores_f.astype(np.float32)
        if (
            fine_fuse_hist
            and scheme == "ab"
            and use_hist_gate
            and frame_bgr_full is not None
            and ref_hist is not None
        ):
            fh, fw = frame_bgr_full.shape[:2]
            y2c, x2c = min(fy2, fh), min(fx2, fw)
            if y2c > fy1 and x2c > fx1:
                fine_bgr = frame_bgr_full[fy1:y2c, fx1:x2c]
                if fine_bgr.shape[:2] == fine_plane.shape[:2]:
                    hh_f = batched_hist_correl_masked(
                        fine_bgr,
                        fine_plane,
                        plane_origin_yx,
                        pos_f,
                        target_h,
                        target_w,
                        ref_hist,
                        hist_h_bins,
                        hist_s_bins,
                    )
                    mx = float(np.max(scores_f)) if scores_f.size else 0.0
                    lam_f = _fuse_lam(mx)
                    rank_scores = _blend_contour_hist(lam_f, scores_f, hh_f.astype(np.float32))
        fused_f = _fused(rank_scores, pos_f, fine_cy, fine_cx)
        best_fi = int(np.argmax(fused_f))
        if scores_f[best_fi] > best_score:
            best_score = float(scores_f[best_fi])
            best_gy = fy1 + pos_f[best_fi][0]
            best_gx = fx1 + pos_f[best_fi][1]

    print(f"[FINE]   best_score={best_score:.4f} pos=({best_gx},{best_gy})")

    gy, gx = best_gy, best_gx
    dirs = list(greedy_dirs)
    for gi in range(greedy_iters):
        if _check_timeout():
            print(f"[TIMEOUT] Greedy search interrupted after {_time_mod.perf_counter()-_t0:.1f}s")
            break
        nbr_positions: list[tuple[int, int]] = []
        for dy_g, dx_g in dirs:
            ny, nx = gy + dy_g, gx + dx_g
            if 0 <= ny <= fH - target_h and 0 <= nx <= fW - target_w:
                nbr_positions.append((ny, nx))
        if not nbr_positions:
            break
        use_hist_greedy = (
            greedy_fuse_hist
            and scheme == "ab"
            and use_hist_gate
            and frame_bgr_full is not None
            and ref_hist is not None
        )
        if use_hist_greedy:
            cand = [(gy, gx)] + nbr_positions
            sc_all = _score_positions_inner(cand, search_plane_th68)
            hh_all = batched_hist_correl_masked(
                frame_bgr_full,
                search_plane_th68,
                plane_origin_yx,
                cand,
                target_h,
                target_w,
                ref_hist,
                hist_h_bins,
                hist_s_bins,
            )
            lam_g = _fuse_lam(float(np.max(sc_all)))
            mix_all = _blend_contour_hist(lam_g, sc_all, hh_all.astype(np.float32))
            best_k = int(np.argmax(mix_all))
            if best_k == 0:
                break
            gy, gx = cand[best_k]
            best_score = float(sc_all[best_k])
        else:
            nbr_scores = _score_positions_inner(nbr_positions, search_plane_th68)
            best_ni = int(np.argmax(nbr_scores))
            if nbr_scores[best_ni] > best_score:
                best_score = float(nbr_scores[best_ni])
                gy, gx = nbr_positions[best_ni]
            else:
                break

    print(f"[GREEDY] final_score={best_score:.4f} pos=({gx},{gy})" +
          (f" [TIMEOUT after {max_time_s:.0f}s]" if _timed_out else ""))
    # ── 计算最终位置的直方图相似度 ──
    _final_hist_sim = -1.0
    if use_hist_gate and ref_hist is not None and frame_bgr_full is not None:
        _cand_hist = batched_hist_correl_masked(
            frame_bgr_full,
            search_plane_th68,
            plane_origin_yx,
            [(gy, gx)],
            target_h,
            target_w,
            ref_hist,
            hist_h_bins,
            hist_s_bins,
        )
        _final_hist_sim = float(_cand_hist[0])

    _trust_hist = False
    if use_hist_gate and _final_hist_sim >= 0.0:
        _hi_h = float(_os_module.environ.get("MIS_DISAGREE_HIST_HI", "0.52"))
        _lo_n = float(_os_module.environ.get("MIS_DISAGREE_NANO_LO", "0.42"))
        _trust_hist = float(nano_confidence) < _lo_n and float(_final_hist_sim) >= _hi_h

    return {
        "score": float(best_score),
        "x": int(gx),
        "y": int(gy),
        "w": target_w,
        "h": target_h,
        "coarse_x": int(_coarse_x),
        "coarse_y": int(_coarse_y),
        "coarse_score": float(_coarse_score),
        "hist_sim": _final_hist_sim,
        "trust_hist_fusion": bool(_trust_hist),
        "suggest_nano_post_blend": bool(_suggest_nano_blend),
    }

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
    if not _MIS_SILENT: print(f"[REFINE] orig=({gx},{gy},{ref_w},{ref_h}) "
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
# main() / run_main()
# ═════════════════════════════════════════════════════════════════════════════
def run_main(coarse_scheme: str = "ab") -> None:
    coarse_scheme_eff = (coarse_scheme or "ab").strip().lower()
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
    CONTOUR_INFER_BS = _env_int("MIS_CONTOUR_INFER_BS", 48)   # A+B：补丁轮廓批量 encode batch
    FINE_FUSE_HIST = _env_bool("MIS_FINE_FUSE_HIST", True)
    GREEDY_FUSE_HIST = _env_bool("MIS_GREEDY_FUSE_HIST", True)
    EARLY_HIST_MIN_COARSE = _env_float("MIS_EARLY_HIST_MIN", 0.0)
    EARLY_POLISH = _env_bool("MIS_EARLY_POLISH", True)
    CONTOUR_SAT_THRESH = _env_float("MIS_CONTOUR_SAT_THRESH", 0.97)
    CONTOUR_SAT_SCALE = _env_float("MIS_CONTOUR_SAT_SCALE", 0.73)
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
    if not _MIS_SILENT: print(f"[配置] frames={len(frames)}  frames_dir={cfg.frames_dir}")
    if not _MIS_SILENT: print(f"[配置] output_dir={cfg.output_dir}  expand={EXPAND_BASE}/{EXPAND_RETRY}  "
          f"adapt_th={ADAPT_THRESH}  slide_step={SLIDE_STEP}  templates={NUM_TEMPLATES}")
    if not _MIS_SILENT: print(f"[配置] MIS_COARSE_SCHEME={coarse_scheme_eff}")

    # ── 输出目录结构（与 MIS_C_F_SIMI_Mesh_Match_Track_3DMatch.py 一致）──────
    out_root         = _ensure_dir(cfg.output_dir)
    out_frames       = _ensure_dir(os.path.join(out_root, "frames_with_bbox"))
    out_roi          = _ensure_dir(os.path.join(out_root, "roi"))
    out_th           = _ensure_dir(os.path.join(out_root, "th_gray"))
    out_feats        = _ensure_dir(os.path.join(out_root, "features"))
    out_match_coarse = _ensure_dir(os.path.join(out_root, "match_coarse"))   # 粗搜位置轮廓裁剪 + 标注
    out_match_fine   = _ensure_dir(os.path.join(out_root, "match_fine"))     # 精搜最终位置轮廓裁剪 + 标注
    out_match_cmp    = _ensure_dir(os.path.join(out_root, "match_compare"))  # 双框叠加对比图

    # contour_sim_model 已在模块级加载（line ~119），直接使用全局变量

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

    hist_gate_T = _env_float("MIS_HIST_GATE_T", 0.08)
    hist_h_bins = _env_int("MIS_HIST_H_BINS", 30)
    hist_s_bins = _env_int("MIS_HIST_S_BINS", 32)
    _mr = cv2.cvtColor(target_th68, cv2.COLOR_BGR2GRAY)
    ref_hist = build_ref_masked_hsv_hist(
        first[y : y + h, x : x + w].copy(),
        ((_mr > 0).astype(np.uint8) * 255),
        hist_h_bins,
        hist_s_bins,
    )
    ref_hist_f32 = ref_hist.astype(np.float32)  # 预转换，避免每帧重复 .astype()
    print(f"[HIST] bins={hist_h_bins}x{hist_s_bins} gate_T={hist_gate_T} coarse_scheme={coarse_scheme_eff}")
    lib_tp_cached, lib_z_cached = _build_lib_contour_stack_and_z(contour_sim_model, lib_templates_raw)
    if lib_tp_cached is not None and lib_z_cached is not None:
        print(f"[LIB_CACHE] templates={lib_tp_cached.shape[0]} device={lib_tp_cached.device} encoder_batch=ON")

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
    # NanoTrack 融合模块初始化（MIS_USE_NANO=1 时启用）
    # ════════════════════════════════════════════════════════════════════════
    USE_NANO        = _env_bool("MIS_USE_NANO", False)
    NANO_CONF_THRESH = _env_float("MIS_NANO_CONF", 0.4)   # NanoTrack接管所需最低分
    NANO_BLEND_ALPHA = _env_float("MIS_NANO_ALPHA", 0.6)  # 融合时 HistAB 权重（0.6→40%NanoTrack）
    nano_tracker: _NanoTrackHelper | None = None
    _nano_takes = 0   # 统计NanoTrack接管次数
    _nano_blends = 0  # 统计融合次数

    if USE_NANO:
        # NanoTrack 在原始分辨率下运行（不做 proc_scale 缩放）
        first_full = cv2.imread(str(frames[0]))
        if first_full is None:
            first_full = first
        # 将 bbox 映射回原始分辨率
        inv_scale = 1.0 / cfg.proc_scale if cfg.proc_scale > 0 else 1.0
        nano_bbox_init = (
            int(x * inv_scale), int(y * inv_scale),
            int(w * inv_scale), int(h * inv_scale)
        )
        nano_tracker = _NanoTrackHelper(
            first_full, nano_bbox_init,
            backbone=_env("MIS_NANO_BACKBONE", _NANO_BACKBONE_DEFAULT),
            neckhead=_env("MIS_NANO_NECKHEAD", _NANO_NECKHEAD_DEFAULT),
        )
        print(f"[NanoTrack-Fusion] enabled: conf_thresh={NANO_CONF_THRESH:.2f} "
              f"blend_alpha={NANO_BLEND_ALPHA:.2f}")
    else:
        print("[NanoTrack-Fusion] disabled (set MIS_USE_NANO=1 to enable)")

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

        cx_prev = bx + bw // 2
        cy_prev = by + bh // 2
        sy1 = max(0, by - EXPAND_BASE)
        sx1 = max(0, bx - EXPAND_BASE)
        sy2 = min(th68_curr.shape[0], by + bh + EXPAND_BASE)
        sx2 = min(th68_curr.shape[1], bx + bw + EXPAND_BASE)
        search_plane = th68_curr[sy1:sy2, sx1:sx2]
        plane_bgr_crop = frame[sy1:sy2, sx1:sx2]
        plane_center_yx = (float(cy_prev - sy1), float(cx_prev - sx1))

        result = find_best_match_hierarchical(
            search_plane, None, base_h, base_w, None,
            contour_sim_model=contour_sim_model,
            lib_templates_raw=lib_templates_raw,
            lib_contour_encodings=None,
            step_coarse=COARSE_STEP,
            step_fine=SLIDE_STEP,
            fine_expand=FINE_EXPAND,
            top_k=_env_int("MIS_TOPK", 3),
            greedy_step=_env_int("MIS_GREEDY_STEP", 4),
            greedy_iters=_env_int("MIS_GREEDY_ITERS", 4),
            spatial_sigma=_env_float("MIS_SPATIAL_SIGMA", 3.0),
            center_yx=plane_center_yx,
            early_exit_score=_env_float("MIS_EARLY_EXIT", 0.95),
            coarse_scheme=coarse_scheme_eff,
            frame_bgr_full=plane_bgr_crop,
            plane_origin_yx=(0, 0),
            ref_hist=ref_hist_f32,
            hist_h_bins=hist_h_bins,
            hist_s_bins=hist_s_bins,
            hist_gate_T=hist_gate_T,
            hist_top_M_cap=_env_int("MIS_HIST_TOP_M", 96),
            hist_min_keep=_env_int("MIS_HIST_MIN_KEEP", 24),
            hist_fallback_take=_env_int("MIS_HIST_FALLBACK_M", 48),
            hist_topkp_c=_env_int("MIS_HIST_TOPKP_C", 96),
            fuse_lambda_base=_env_float("MIS_FUSE_LAMBDA", 0.65),
            fuse_adaptive=_env_bool("MIS_FUSE_ADAPT", True),
            fine_fuse_hist=FINE_FUSE_HIST,
            greedy_fuse_hist=GREEDY_FUSE_HIST,
            early_exit_hist_min=EARLY_HIST_MIN_COARSE,
            early_exit_polish=EARLY_POLISH,
            contour_sat_thresh=CONTOUR_SAT_THRESH,
            contour_sat_lambda_scale=CONTOUR_SAT_SCALE,
            contour_infer_bs=CONTOUR_INFER_BS,
            cached_lib_tensor=lib_tp_cached,
            cached_lib_z=lib_z_cached,
        )

        off_x = int(sx1)
        off_y = int(sy1)
        # Step 3: 裁剪平面坐标 → 全帧
        new_x = int(result["x"]) + off_x
        new_y = int(result["y"]) + off_y
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

        # ── NanoTrack 融合（MIS_USE_NANO=1）──────────────────────────────────
        _nano_source = "ours"
        if nano_tracker is not None and nano_tracker.available:
            # NanoTrack 在原始分辨率帧上更新
            frame_full = cv2.imread(str(img_path))
            if frame_full is None:
                frame_full = frame
            nano_bbox_full, nano_score = nano_tracker.update(frame_full)
            # 将 NanoTrack bbox 缩放到 proc_scale 坐标系
            scale = cfg.proc_scale if cfg.proc_scale > 0 else 1.0
            nb_x = int(nano_bbox_full[0] * scale)
            nb_y = int(nano_bbox_full[1] * scale)
            nb_w = max(1, int(nano_bbox_full[2] * scale))
            nb_h = max(1, int(nano_bbox_full[3] * scale))
            nb_x, nb_y, nb_w, nb_h = core.clamp_bbox(nb_x, nb_y, nb_w, nb_h,
                                                       frame.shape[1], frame.shape[0])
            nano_bbox_proc = (nb_x, nb_y, nb_w, nb_h)

            if score < LOSS_THRESH and nano_score > NANO_CONF_THRESH:
                # HistAB 丢失目标，NanoTrack 接管
                new_bbox = nano_bbox_proc
                new_x, new_y, new_w, new_h = new_bbox
                score = nano_score * 0.8   # 降档标记为 NanoTrack 接管
                _nano_takes += 1
                _nano_source = "nano_takeover"
                # 更新 HistAB 下一帧搜索中心
                bx, by = new_x, new_y
            elif LOSS_THRESH <= score < ADAPT_THRESH and nano_score > NANO_CONF_THRESH:
                # 低置信度：加权融合
                alpha = min(0.9, score / ADAPT_THRESH)  # HistAB 可信度越高权重越大
                alpha = max(NANO_BLEND_ALPHA, alpha)
                new_bbox = _fuse_bboxes(new_bbox, score, nano_bbox_proc, nano_score, alpha)
                new_x, new_y, new_w, new_h = new_bbox
                _nano_blends += 1
                _nano_source = f"fused(a={alpha:.2f})"

            if (i % PRINT_EVERY) == 0:
                print(f"  [Nano] score_ours={result['score']:.3f}  nano={nano_score:.3f}  "
                      f"src={_nano_source}")
        # ─────────────────────────────────────────────────────────────────────

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
            coarse_gx = int(result.get("coarse_x", result["x"])) + off_x
            coarse_gy = int(result.get("coarse_y", result["y"])) + off_y
            coarse_sc = result.get("coarse_score", score)

            # 1) match_coarse: 粗搜 th68 patch + 左上角分数
            crop_c = th68_curr[coarse_gy:coarse_gy + base_h, coarse_gx:coarse_gx + base_w].copy()
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
                          (coarse_gx + base_w, coarse_gy + base_h), (0, 255, 255), 2)
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
        "task": "contour_hist_coarse_tracking",
        "model": _CONTOUR_SIM_MODEL_PATH,
        "frames_dir": cfg.frames_dir,
        "output_dir": cfg.output_dir,
        "num_frames": len(frames),
        "coarse_scheme": coarse_scheme_eff,
        "hist_gate_T": hist_gate_T,
        "contour_infer_bs": CONTOUR_INFER_BS,
        "fine_fuse_hist": FINE_FUSE_HIST,
        "greedy_fuse_hist": GREEDY_FUSE_HIST,
        "early_exit_hist_min": EARLY_HIST_MIN_COARSE,
        "early_exit_polish": EARLY_POLISH,
        "contour_sat_thresh": CONTOUR_SAT_THRESH,
        "contour_sat_lambda_scale": CONTOUR_SAT_SCALE,
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

    if USE_NANO and nano_tracker is not None:
        print(f"\n[NanoTrack-Fusion] 总计 NanoTrack 接管帧={_nano_takes}  融合帧={_nano_blends}"
              f"  共{len(frames)-1}帧")
    print(f"\n[完成] output_dir={out_root}")
    print(f"[完成] tracking_bboxes_and_scores.json  diag.csv  meta.json")


def main() -> None:
    sch = (_env("MIS_COARSE_SCHEME", "ab") or "ab").strip().lower()
    run_main(sch)


if __name__ == "__main__":
    main()


# 评测跟踪器兼容别名
_NanoTrackHelper = _NanoTrackHelper
