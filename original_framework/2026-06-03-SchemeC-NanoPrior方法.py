#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026-06-03-SchemeC-NanoPrior方法.py [FROZEN 2026-06-03]
==========================================
**方案 C（NanoTrack 空间先验图）独立入口**：在局部搜索平面内构造与 ``search_plane_th68``
同尺寸的二维先验图，传入 ``find_best_match_hierarchical(..., nano_prior_map=...)``，
使粗排在候选中心采样 Nano 置信度场，并用改进的 **log 融合**（见父模块参数
``nano_fusion=\"log\"``）替代 hist×nano 的双重乘积压制。

与原脚本差异概要
----------------
- **必选**：NanoTrack（OpenCV TrackerNano）在每个 proc 帧先更新，bbox 映射回搜索窗口，
  生成 **高斯先验图**（可调 sigma / floor / score 增益）。
- **粗排**：父模块内 ``nano_fusion=log`` 时 — hist 门控用 ``hist × nano^γ``（γ 可调），
  轮廓×hist 混合后再乘 ``spatial × nano^β``，避免同一 nano 因子在 hist 与 rank 上连乘两次。
- **事后框融合**：默认 **关闭**（``MIS_SCHEME_C_POST_BLEND=0``），以免抵消先验粗排收益；
  需要可与旧行为一致时设 ``MIS_SCHEME_C_POST_BLEND=1``。

运行示例（建议在仓库根目录执行，以便导入同目录模块）::

    cd /d D:\\MIS_Pose_Track_Re3D
    python Deformation_hist_AB_scheme_C_nanoprior.py

环境变量沿用 ``Deformation_F_T_C_G_Experi_hist_AB.py``（帧目录、输出、hist、粗搜步长等），
增补::

    MIS_SCHEME_C_POST_BLEND     0 | 1   是否在 HistAB 结果后再做 Nano 线性框融合（默认 0）
    MIS_NANO_PRIOR_SIGMA_SCALE  高斯 σ = scale * max(w,h)，默认 0.55
    MIS_NANO_PRIOR_FLOOR        先验均匀下限 ∈[0,1)，默认 0.08
    MIS_NANO_FUSION             log | product  （默认 log，传给粗排）
    MIS_NANO_PRIOR_BETA         log 模式下 rank 中 nano 幂指数 β，默认 0.48
    MIS_NANO_HIST_COUPLING      log 模式下 hist 门控耦合 γ，默认 0.52

**B → A → s_z（已实现最小闭环）**
- **B / s_n**：OpenCV Python **无法读取 TrackerNano 内部特征相关图**，默认用首帧目标灰度模板对
  当前 ``plane_bgr_crop`` 做 ``cv2.matchTemplate(..., TM_CCOEFF_NORMED)``，在粗候选 **(wy,wx)**
  处采样响应作为 s_n（与 Siamese 相关面同用途、非同一套权重）。默认仅在 **上一帧框 ∪ Nano 框（平面坐标）**
  外包邻域内计算响应并嵌入全尺寸图（域外为 NaN，再打分回落 Nano 先验）；有效半径至少
  ``max(MIS_MATCH_RESP_MARGIN, 2·MIS_COARSE_STEP)``，保证粗网格边界可被覆盖。
  设 ``MIS_MATCH_RESP_MARGIN=0`` 则退回整幅平面匹配。
- **A**：``MIS_RERANK_TOPR>0`` 时，父函数在粗网格上对 Top‑R 用
  ``w_n·s_n + w_c·contour + w_h·hist + w_z·s_z`` **再选锚点**（接入 ``find_best_match_hierarchical``）。
- **s_z**：首帧灰度 + **伪深度**（th 前景 min-max）双通道 masked NCC；``MIS_SZ_PSEUDO3D`` / ``MIS_SZ_GRAY_WEIGHT``。
- **Top‑R 融合**：``MIS_RERANK_MODE=loggeom`` 为对数几何平均；``MIS_RERANK_ADAPT_W=1`` 按 Nano 置信调节 w_sn/w_h/w_z。

**ContourSimilarityNet + TrackVes 数据微调（离线）**
1. 从 TrackVes（或你的手术跟踪数据）采样一批 **目标轮廓 patch**（可只抽部分帧以控时）。
2. 训练：``python train_contour_sim_trackves_triplet.py`` 或 ``python train_contour_sim_trackves.py``（见仓库内脚本及参数）。
3. 将导出的 checkpoint 设为环境变量 ``MIS_CONTOUR_SIM_CKPT`` 后再跑本脚本。

粗排二次融合相关环境变量::

    MIS_RERANK_TOPR             粗网格 Top‑R 再打分，0=关闭，默认 48
    MIS_RERANK_WEIGHTS          逗号分隔 w_sn,w_sc,w_sh,w_sz，默认 0.28,0.34,0.24,0.14
    MIS_MATCH_RESP_ENABLE       1|0 是否计算 matchTemplate 响应（RERANK>0 时建议 1），默认 1
    MIS_MATCH_RESP_MARGIN       ROI 半边扩展（像素），在上一帧位置附近算 s_n；**0**=全图匹配，默认 112
    MIS_MATCH_RESP_FULL_RATIO   ROI 覆盖格数占全平面比例 ≥ 此值则直接全图匹配（省镶嵌），默认 0.92

**性能（保精度前提下加速，默认开启）**::

    th68：每帧 ``_apply_th68`` 一次，搜索平面与 refine 均从 ``th68_curr`` 裁剪（勿对子区域重复 threshold）。
    MIS_NANO_PRIOR_CACHE        1|0  Nano 高斯先验图在相邻帧中心微动时复用（默认 1）
    MIS_NANO_PRIOR_CACHE_PIX    复用允许的中心平移（像素），默认 8
    MIS_MATCH_RESP_COND         1|0  条件计算 matchTemplate（默认 1）
    MIS_MATCH_RESP_MIN_NANO     条件模式下 Nano 置信低于此则跳过 s_n，默认 0.55
    MIS_MATCH_RESP_SKIP_MOTION_PX  上一帧中心位移低于此且上一帧 score 高则跳过 s_n，默认 6
    MIS_MATCH_RESP_SKIP_SCORE   与 SKIP_MOTION 联用的 score 阈值，默认 0.88
    MIS_ADAPT_LIB               1|0  静止时减少轮廓库参与粗搜的模板数（默认 1）
    MIS_ADAPT_LIB_MOTION_PX     超过此位移则用全库，默认 12
    MIS_ADAPT_LIB_MIN           静止时最少保留模板数，默认 64
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

# 允许从任意当前工作目录启动：将本脚本所在目录加入 sys.path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np

# 依赖原实验模块（轮廓模型、层次匹配、Nano 封装等）
import importlib as _importlib
dab = _importlib.import_module("2026-06-03-HistAB核心方法")

import MIS_C_Fe_Simi_Mesh_Match_Pose_Estimate_Track as core


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name, default) or default).strip().strip('"').strip("'")


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    v = _env(name, "")
    if v == "":
        return bool(default)
    return v.lower() not in ("0", "false", "no", "off")


def apply_scheme_c_fusion_defaults() -> None:
    """Scheme C 推荐融合超参（仅当环境变量未设置时写入）。"""
    _defaults = {
        "MIS_NANO_FUSION": "log",
        "MIS_NANO_PRIOR_BETA": "0.55",
        "MIS_NANO_HIST_COUPLING": "0.45",
        "MIS_RERANK_TOPR": "64",
        "MIS_RERANK_WEIGHTS": "0.28,0.30,0.24,0.18",
        "MIS_NANO_SCORE_MARGIN": "0.06",
        "MIS_RERANK_MODE": "loggeom",
        "MIS_RERANK_ADAPT_W": "1",
        "MIS_SZ_PSEUDO3D": "1",
        "MIS_SZ_GRAY_WEIGHT": "0.55",
        "MIS_FUSION_DISAGREE": "1",
        "MIS_DISAGREE_EXPAND_TOPR": "1",
        "MIS_NANO_RECOVERY": "1",
        "MIS_NANO_ANCHOR_CENTER": "1",
        "MIS_NANO_ANCHOR_MOTION_PX": "6",
        "MIS_NANO_RECOVERY_IOU": "0.54",
        "MIS_NANO_DISAGREE_IOU": "0.60",
        "MIS_NANO_SNAP_IOU": "0.50",
        "MIS_NANO_KEEP_MIS_IOU": "0.64",
        "MIS_NANO_HIST_DISAGREE_IOU": "0.56",
        "MIS_NANO_ARBITRATE": "1",
        "MIS_NANO_PICK_IOU": "0.55",
        "MIS_NANO_PICK_HIST_LO": "0.50",
        "MIS_NANO_ARBITRATE": "1",
        "MIS_NANO_PICK_IOU": "0.55",
        "MIS_NANO_PICK_HIST_LO": "0.50",
        "MIS_NANO_RECOVERY_HARD_IOU": "0.12",
        "MIS_NANO_SNAP_MOTION_PX": "16",
        "MIS_NANO_SNAP_ALPHA": "0.90",
        "MIS_NANO_PROPAGATE_ALPHA": "0.55",
        "MIS_NANO_PROPAGATE_ALPHA": "0.55",
        "MIS_SCHEME_C_POST_BLEND": "1",
        "MIS_LOSS_THRESH": "0.42",
        "MIS_ADAPT_THRESH": "0.58",
        "MIS_EVAL_PROPAGATE_NANO": "1",
        "MIS_EVAL_UPDATE_ALL_FRAMES": "0",
        "MIS_NANO_LITE_ENABLE": "1",
        "MIS_NANO_LITE_NS": "0.82",
        "MIS_NANO_LITE_MOTION_PX": "8",
        "MIS_NANO_LITE_PREV_SCORE": "0.78",
        "MIS_NANO_LITE_PREV_IOU": "0.50",
        "MIS_RERANK_TOPR_LITE": "28",
        "MIS_MATCH_RESP_LITE_NS": "0.68",
        "MIS_SILENT": "1",
    }
    for _k, _v in _defaults.items():
        if _env(_k, "") == "":
            os.environ[_k] = _v


def apply_cascade_fusion_defaults() -> None:
    """Nano 级联融合：Nano 为主路径，Hist/Contour 仅作门控与局部纠错（大幅降耗时）。"""
    apply_scheme_c_fusion_defaults()
    _cascade = {
        "MIS_NANO_CASCADE": "1",
        "MIS_NANO_LITE_ENABLE": "1",
        "MIS_NANO_LITE_NS": "0.68",
        "MIS_NANO_LITE_MOTION_PX": "22",
        "MIS_NANO_LITE_PREV_SCORE": "0.55",
        "MIS_NANO_LITE_PREV_IOU": "0.38",
        "MIS_CASCADE_PRIMARY_NS": "0.58",
        "MIS_CASCADE_PRIMARY_HIST": "0.34",
        "MIS_CASCADE_PRIMARY_MOTION": "28",
        "MIS_CASCADE_LOCAL_NS": "0.42",
        "MIS_CASCADE_LOCAL_HIST": "0.28",
        "MIS_CASCADE_LOCAL_EXPAND": "72",
        "MIS_CASCADE_LOCAL_COARSE": "22",
        "MIS_CASCADE_LOCAL_TOPR": "16",
        "MIS_CASCADE_LOCAL_MAX_T": "0.45",
        "MIS_CASCADE_FULL_NS": "0.38",
        "MIS_CASCADE_FULL_HIST": "0.22",
        "MIS_CASCADE_FULL_MOTION": "55",
        "MIS_MATCH_RESP_ENABLE": "0",
        "MIS_RERANK_TOPR": "40",
        "MIS_COARSE_STEP": "18",
        "MIS_MAX_TIME_S": "0.9",
        "MIS_ADAPT_LIB": "1",
        "MIS_ADAPT_LIB_MIN": "48",
        "MIS_SCHEME_C_POST_BLEND": "0",
    }
    for _k, _v in _cascade.items():
        if _env(_k, "") == "":
            os.environ[_k] = _v


def fast_bbox_hist_correl(
    frame_bgr: np.ndarray,
    ref_hist: np.ndarray,
    bbox: tuple[int, int, int, int],
    h_bins: int,
    s_bins: int,
) -> float:
    """在 bbox ROI 上快速计算与 ref_hist 的 HSV 相关度（~1ms，用于 Nano 级联门控）。"""
    import cv2
    import numpy as np

    x, y, w, h = (int(v) for v in bbox)
    H, W = frame_bgr.shape[:2]
    if w <= 2 or h <= 2:
        return 0.0
    x2, y2 = min(W, x + w), min(H, y + h)
    x, y = max(0, x), max(0, y)
    roi = frame_bgr[y:y2, x:x2]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mk = (gray > 8).astype(np.uint8) * 255
    if int(cv2.countNonZero(mk)) < 40:
        return 0.0
    cand = dab.build_ref_masked_hsv_hist(roi, mk, int(h_bins), int(s_bins))
    rh = np.asarray(ref_hist, dtype=np.float32)
    if rh.size == 0 or cand.size == 0:
        return 0.0
    sim = float(cv2.compareHist(rh, cand, cv2.HISTCMP_CORREL))
    return max(0.0, min(1.0, sim))


def cascade_tier_for_frame(
    *,
    nano_score: float,
    hist_sim: float,
    motion_px: float,
    iou_prev_nano: float,
    prev_track_score: float,
) -> str:
    """返回 primary | local | full — 决定本帧计算预算。"""
    if not _env_bool("MIS_NANO_CASCADE", True):
        return "full"
    p_ns = _env_float("MIS_CASCADE_PRIMARY_NS", 0.58)
    p_hist = _env_float("MIS_CASCADE_PRIMARY_HIST", 0.34)
    p_motion = _env_float("MIS_CASCADE_PRIMARY_MOTION", 28.0)
    l_ns = _env_float("MIS_CASCADE_LOCAL_NS", 0.42)
    l_hist = _env_float("MIS_CASCADE_LOCAL_HIST", 0.28)
    f_ns = _env_float("MIS_CASCADE_FULL_NS", 0.38)
    f_hist = _env_float("MIS_CASCADE_FULL_HIST", 0.22)
    f_motion = _env_float("MIS_CASCADE_FULL_MOTION", 55.0)

    if float(nano_score) >= p_ns and motion_px < p_motion:
        if float(hist_sim) >= p_hist or (
            float(nano_score) >= _env_float("MIS_CASCADE_PRIMARY_NS_STRONG", 0.76)
            and iou_prev_nano >= _env_float("MIS_CASCADE_PRIMARY_IOU", 0.32)
        ):
            if float(prev_track_score) >= _env_float("MIS_CASCADE_PRIMARY_PREV", 0.45):
                return "primary"
    if float(nano_score) >= l_ns and float(hist_sim) >= l_hist:
        return "local"
    if float(nano_score) < f_ns or float(hist_sim) < f_hist or motion_px >= f_motion:
        return "full"
    if float(nano_score) >= _env_float("MIS_CASCADE_LOCAL_NS", 0.42):
        return "local"
    return "full"


def _ncc_u8(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    aa = np.asarray(a_u8, dtype=np.float32).reshape(-1)
    bb = np.asarray(b_u8, dtype=np.float32).reshape(-1)
    if aa.size <= 4 or bb.size != aa.size:
        return 0.0
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    da = float(np.linalg.norm(aa))
    db = float(np.linalg.norm(bb))
    if da < 1e-6 or db < 1e-6:
        return 0.0
    cc = float(np.dot(aa, bb) / (da * db))
    return max(0.0, min(1.0, 0.5 * (cc + 1.0)))


def fast_pseudo3d_similarity(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    ref_gray_patch_sz: np.ndarray,
    ref_depth_patch_sz: np.ndarray,
    *,
    gray_weight: float = 0.55,
) -> float:
    x, y, w, h = (int(v) for v in bbox)
    H, W = frame_bgr.shape[:2]
    if w <= 2 or h <= 2:
        return 0.0
    x2, y2 = min(W, x + w), min(H, y + h)
    x, y = max(0, x), max(0, y)
    roi = frame_bgr[y:y2, x:x2]
    if roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hh, ww = int(ref_gray_patch_sz.shape[0]), int(ref_gray_patch_sz.shape[1])
    if hh <= 2 or ww <= 2:
        return 0.0
    gray_r = cv2.resize(gray, (ww, hh), interpolation=cv2.INTER_AREA)
    dep_r = dab._pseudo_depth_u8_from_gray(gray_r)
    rg = np.asarray(ref_gray_patch_sz, dtype=np.uint8)
    rz = np.asarray(ref_depth_patch_sz, dtype=np.uint8)
    gw = float(np.clip(gray_weight, 0.0, 1.0))
    return float(gw * _ncc_u8(gray_r, rg) + (1.0 - gw) * _ncc_u8(dep_r, rz))


def cascade_risk_score(
    *,
    nano_score: float,
    hist_sim: float,
    motion_px: float,
    iou_prev_nano: float,
    prev_track_score: float,
) -> float:
    ns = float(np.clip(nano_score, 0.0, 1.0))
    hs = float(np.clip(hist_sim, 0.0, 1.0))
    mot = float(np.clip(motion_px / max(1.0, _env_float("MIS_CASCADE_FULL_MOTION", 55.0)), 0.0, 1.0))
    iou = float(np.clip(iou_prev_nano, 0.0, 1.0))
    prev = float(np.clip(prev_track_score, 0.0, 1.0))
    return float(np.clip(0.34 * (1.0 - ns) + 0.28 * (1.0 - hs) + 0.20 * mot + 0.12 * (1.0 - iou) + 0.06 * (1.0 - prev), 0.0, 1.0))


def cascade_tier_with_temporal_hysteresis(
    *,
    base_tier: str,
    risk_score: float,
    risk_ema: float,
    bad_streak: int,
) -> str:
    tier = str(base_tier)
    if int(bad_streak) >= _env_int("MIS_CASCADE_BAD_STREAK_FULL", 6):
        return "full"
    if float(risk_ema) >= _env_float("MIS_CASCADE_RISK_EMA_FULL", 0.62):
        return "full"
    if float(risk_score) >= _env_float("MIS_CASCADE_RISK_FULL", 0.58) and tier == "primary":
        return "local"
    return tier


def high_level_policy_mode(
    *,
    nano_score: float,
    hist_sim: float,
    sz_sim: float,
    motion_px: float,
    iou_prev_nano: float,
    prev_track_score: float,
    rel_hist: float,
    rel_sz: float,
    rel_nano: float,
    risk_ema: float,
    bad_streak: int,
) -> tuple[str, str]:
    base_tier = cascade_tier_for_frame(
        nano_score=float(nano_score),
        hist_sim=float(hist_sim),
        motion_px=float(motion_px),
        iou_prev_nano=float(iou_prev_nano),
        prev_track_score=float(prev_track_score),
    )
    risk = cascade_risk_score(
        nano_score=float(nano_score),
        hist_sim=float(hist_sim),
        motion_px=float(motion_px),
        iou_prev_nano=float(iou_prev_nano),
        prev_track_score=float(prev_track_score),
    )
    score = 0.58 * float(risk) + 0.42 * float(np.clip(risk_ema, 0.0, 1.0))
    t_stable = _env_float("MIS_POLICY_T_STABLE", 0.30)
    t_appear = _env_float("MIS_POLICY_T_APPEAR", 0.48)
    t_geom = _env_float("MIS_POLICY_T_GEOM", 0.66)
    mode = "stable"
    if (
        float(motion_px) >= _env_float("MIS_POLICY_MOTION_HIGH", 38.0)
        or float(iou_prev_nano) < _env_float("MIS_POLICY_IOU_LOW", 0.30)
        or float(rel_nano) < 0.45
        or int(bad_streak) >= _env_int("MIS_CASCADE_BAD_STREAK_FULL", 6)
        or score >= t_geom
    ):
        mode = "geometry_risk"
    elif (
        float(hist_sim) < _env_float("MIS_POLICY_HIST_LOW", 0.26)
        and float(sz_sim) >= _env_float("MIS_POLICY_SZ_LOW", 0.30)
    ) or score >= t_appear:
        mode = "appearance_shift"
    elif score <= t_stable:
        mode = "stable"
    tier = base_tier
    if mode == "geometry_risk":
        tier = "full" if score >= t_geom else "local"
    elif mode == "appearance_shift" and tier == "primary":
        tier = "local"
    tier = cascade_tier_with_temporal_hysteresis(
        base_tier=tier,
        risk_score=float(risk),
        risk_ema=float(risk_ema),
        bad_streak=int(bad_streak),
    )
    return mode, tier


def bbox_center_motion_px(
    bbox_a: tuple[int, int, int, int],
    bbox_b: tuple[int, int, int, int],
) -> float:
    ax, ay, aw, ah = (int(v) for v in bbox_a)
    bx, by, bw, bh = (int(v) for v in bbox_b)
    acx, acy = ax + aw * 0.5, ay + ah * 0.5
    bcx, bcy = bx + bw * 0.5, by + bh * 0.5
    return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)


def should_compute_match_response(
    *,
    rerank_topr: int,
    match_resp_enable: bool,
    last_ns: float,
    motion_px: float,
    prev_track_score: float,
) -> bool:
    if rerank_topr <= 0 or not match_resp_enable:
        return False
    if not _env_bool("MIS_MATCH_RESP_COND", True):
        return True
    if float(last_ns) < _env_float("MIS_MATCH_RESP_MIN_NANO", 0.55):
        return False
    if motion_px < _env_float("MIS_MATCH_RESP_SKIP_MOTION_PX", 6.0):
        if float(prev_track_score) >= _env_float("MIS_MATCH_RESP_SKIP_SCORE", 0.88):
            return False
    return True


def get_or_build_nano_prior_map(
    plane_h: int,
    plane_w: int,
    nano_bbox_proc: tuple[int, int, int, int],
    crop_xy_tl: tuple[int, int],
    nano_score: float,
    template_wh: tuple[int, int],
    cache: dict | None,
    *,
    sigma_scale: float,
    uniform_floor: float,
) -> tuple[np.ndarray | None, dict]:
    """带帧间复用的 Nano 先验图；cache 由调用方持久化。"""
    c: dict = cache if cache is not None else {}
    if float(nano_score) < _env_float("MIS_NANO_CONF", 0.4):
        c.clear()
        return None, c

    nx, ny, nw, nh = (int(v) for v in nano_bbox_proc)
    sx1, sy1 = int(crop_xy_tl[0]), int(crop_xy_tl[1])
    cx = float(nx + max(1, nw) * 0.5 - sx1)
    cy = float(ny + max(1, nh) * 0.5 - sy1)

    if _env_bool("MIS_NANO_PRIOR_CACHE", True):
        max_pix = _env_float("MIS_NANO_PRIOR_CACHE_PIX", 8.0)
        prev_map = c.get("map")
        if prev_map is not None and int(c.get("ph", -1)) == int(plane_h) and int(c.get("pw", -1)) == int(plane_w):
            dcx = abs(cx - float(c.get("cx", -1e9)))
            dcy = abs(cy - float(c.get("cy", -1e9)))
            dns = abs(float(nano_score) - float(c.get("ns", -1.0)))
            if dcx <= max_pix and dcy <= max_pix and dns < 0.08:
                return prev_map, c

    prior = build_nano_prior_map_on_plane(
        plane_h,
        plane_w,
        nano_bbox_proc,
        crop_xy_tl,
        float(nano_score),
        template_wh,
        sigma_scale=sigma_scale,
        uniform_floor=uniform_floor,
    )
    c["map"] = prior
    c["ph"] = int(plane_h)
    c["pw"] = int(plane_w)
    c["cx"] = cx
    c["cy"] = cy
    c["ns"] = float(nano_score)
    return prior, c


def adapt_contour_lib_cache(
    lib_tp,
    lib_z,
    motion_px: float,
):
    """目标近似静止时仅用前 N 个增广模板，降低 CNN 粗搜开销。"""
    if lib_tp is None:
        return lib_tp, lib_z
    if not _env_bool("MIS_ADAPT_LIB", True):
        return lib_tp, lib_z
    if float(motion_px) >= _env_float("MIS_ADAPT_LIB_MOTION_PX", 12.0):
        return lib_tp, lib_z
    n_full = int(lib_tp.shape[0])
    n_keep = min(n_full, max(8, _env_int("MIS_ADAPT_LIB_MIN", 64)))
    if n_keep >= n_full:
        return lib_tp, lib_z
    return lib_tp[:n_keep], (lib_z[:n_keep] if lib_z is not None else None)


def build_nano_prior_map_on_plane(
    plane_h: int,
    plane_w: int,
    nano_bbox_xywh_proc: tuple[int, int, int, int],
    crop_xy_tl_proc: tuple[int, int],
    nano_score: float,
    template_wh: tuple[int, int],
    *,
    sigma_scale: float = 0.55,
    uniform_floor: float = 0.08,
) -> np.ndarray:
    """
    在局部搜索平面坐标系内构造 float32 先验图 [0, 1]，与 ``search_plane_th68[:,:,0]`` 对齐。
    nano_bbox_xywh_proc / crop_xy_tl_proc 均在 **proc_scale 帧坐标系**下。
    """
    ph = max(1, int(plane_h))
    pw = max(1, int(plane_w))
    sx1, sy1 = int(crop_xy_tl_proc[0]), int(crop_xy_tl_proc[1])
    nx, ny, nw, nh = [int(v) for v in nano_bbox_xywh_proc]
    cx = float(nx + max(1, nw) * 0.5 - sx1)
    cy = float(ny + max(1, nh) * 0.5 - sy1)
    cx = float(np.clip(cx, 0.0, pw - 1.0))
    cy = float(np.clip(cy, 0.0, ph - 1.0))

    yy, xx = np.indices((ph, pw), dtype=np.float32)
    tw, th = max(1, int(template_wh[0])), max(1, int(template_wh[1]))
    sigma = float(max(4.0, sigma_scale * float(max(tw, th))))
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    g = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
    gmax = float(max(g.max(), 1e-9))
    g = g / gmax

    amp = float(np.clip(max(nano_score, 0.12), 0.0, 1.0))
    uf = float(np.clip(uniform_floor, 0.0, 0.95))
    prior = uf + (1.0 - uf) * g * amp
    return np.asarray(prior, dtype=np.float32)


def _plane_gray_u8(plane_bgr: np.ndarray) -> np.ndarray:
    if plane_bgr.ndim == 2:
        return np.asarray(plane_bgr, dtype=np.uint8)
    return cv2.cvtColor(plane_bgr, cv2.COLOR_BGR2GRAY)


def _match_template_ccoeff_normed_u8(pg: np.ndarray, templ_gray_u8: np.ndarray) -> np.ndarray:
    tg = np.asarray(templ_gray_u8, dtype=np.uint8)
    if tg.ndim != 2:
        raise ValueError("templ_gray must be 2D")
    out = cv2.matchTemplate(pg, tg, cv2.TM_CCOEFF_NORMED)
    return np.asarray(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0), dtype=np.float32)


def compute_template_match_response(plane_bgr: np.ndarray, templ_gray: np.ndarray) -> np.ndarray:
    """s_n 用：归一化互相关响应，形状 (Ph-th+1, Pw-tw+1)，与粗候选左上角 (wy,wx) 对齐。"""
    tg = np.asarray(templ_gray, dtype=np.uint8)
    if tg.ndim != 2:
        raise ValueError("templ_gray must be 2D")
    pg = _plane_gray_u8(plane_bgr)
    return _match_template_ccoeff_normed_u8(pg, tg)


def compute_template_match_response_roi_embedded(
    plane_bgr: np.ndarray,
    templ_gray: np.ndarray,
    *,
    py_tl: int,
    px_tl: int,
    bbox_h: int,
    bbox_w: int,
    margin: int,
    coarse_step: int = 16,
    full_plane_ratio: float = 0.92,
    nano_plane_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """在全平面尺寸的响应图上仅在局部邻域计算 matchTemplate，域外填 NaN（``_sn_at_candidate`` 回落 Nano）。

    ROI = **上一帧框 ∪ Nano 框**（均在平面坐标）各自向外扩展 ``max(margin, 2*coarse_step)`` 后的外包矩形，
    避免 Nano 与 MIS 分歧时峰值落在 ROI 外，同时保证粗步长网格边缘可被覆盖。
    """
    tg = np.asarray(templ_gray, dtype=np.uint8)
    if tg.ndim != 2:
        raise ValueError("templ_gray must be 2D")
    th, tw = int(tg.shape[0]), int(tg.shape[1])
    pg = _plane_gray_u8(plane_bgr)
    Ph, Pw = int(pg.shape[0]), int(pg.shape[1])
    full_h, full_w = Ph - th + 1, Pw - tw + 1
    if full_h <= 0 or full_w <= 0:
        return np.zeros((max(0, full_h), max(0, full_w)), dtype=np.float32)
    if margin <= 0:
        return compute_template_match_response(plane_bgr, templ_gray)

    margin_eff = max(int(margin), int(coarse_step) * 2)
    boxes: list[tuple[int, int, int, int]] = [
        (int(py_tl), int(px_tl), int(bbox_h), int(bbox_w)),
    ]
    if nano_plane_box is not None:
        ny, nx, nh, nw = (int(v) for v in nano_plane_box)
        if nh > 0 and nw > 0:
            boxes.append((ny, nx, nh, nw))

    y0 = max(0, min(b[0] - margin_eff for b in boxes))
    x0 = max(0, min(b[1] - margin_eff for b in boxes))
    y1 = min(Ph, max(b[0] + b[2] + margin_eff for b in boxes))
    x1 = min(Pw, max(b[1] + b[3] + margin_eff for b in boxes))
    if y1 - y0 < th or x1 - x0 < tw:
        return compute_template_match_response(plane_bgr, templ_gray)

    rh = y1 - y0 - th + 1
    rw = x1 - x0 - tw + 1
    if rh <= 0 or rw <= 0:
        return compute_template_match_response(plane_bgr, templ_gray)

    plane_cells = float(max(1, full_h * full_w))
    if (rh * rw) >= float(full_plane_ratio) * plane_cells:
        return compute_template_match_response(plane_bgr, templ_gray)

    roi = pg[y0:y1, x0:x1]
    resp_roi = _match_template_ccoeff_normed_u8(roi, tg)

    full = np.full((full_h, full_w), np.nan, dtype=np.float32)
    rh_a, rw_a = int(resp_roi.shape[0]), int(resp_roi.shape[1])
    y_end = min(y0 + rh_a, full_h)
    x_end = min(x0 + rw_a, full_w)
    full[y0:y_end, x0:x_end] = resp_roi[: y_end - y0, : x_end - x0]
    return full


def _parse_rerank_weights(text: str) -> tuple[float, float, float, float]:
    raw = (text or "").strip()
    if not raw:
        return (0.28, 0.34, 0.24, 0.14)
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 4:
        return (0.28, 0.34, 0.24, 0.14)
    return tuple(float(x) for x in parts)


def _plane_gray_u8(plane_bgr: np.ndarray) -> np.ndarray:
    if plane_bgr.ndim == 2:
        return np.asarray(plane_bgr, dtype=np.uint8)
    return cv2.cvtColor(plane_bgr, cv2.COLOR_BGR2GRAY)


def _match_template_ccoeff_normed_u8(pg: np.ndarray, templ_gray_u8: np.ndarray) -> np.ndarray:
    tg = np.asarray(templ_gray_u8, dtype=np.uint8)
    if tg.ndim != 2:
        raise ValueError("templ_gray must be 2D")
    out = cv2.matchTemplate(pg, tg, cv2.TM_CCOEFF_NORMED)
    return np.asarray(np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0), dtype=np.float32)


def compute_template_match_response(plane_bgr: np.ndarray, templ_gray: np.ndarray) -> np.ndarray:
    """s_n 用：归一化互相关响应，形状 (Ph-th+1, Pw-tw+1)，与粗候选左上角 (wy,wx) 对齐。"""
    tg = np.asarray(templ_gray, dtype=np.uint8)
    if tg.ndim != 2:
        raise ValueError("templ_gray must be 2D")
    pg = _plane_gray_u8(plane_bgr)
    return _match_template_ccoeff_normed_u8(pg, tg)


def compute_template_match_response_roi_embedded(
    plane_bgr: np.ndarray,
    templ_gray: np.ndarray,
    *,
    py_tl: int,
    px_tl: int,
    bbox_h: int,
    bbox_w: int,
    margin: int,
    coarse_step: int = 16,
    full_plane_ratio: float = 0.92,
    nano_plane_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """在全平面尺寸的响应图上仅在局部邻域计算 matchTemplate，域外填 NaN（``_sn_at_candidate`` 回落 Nano）。

    ROI = **上一帧框 ∪ Nano 框**（均在平面坐标）各自向外扩展 ``max(margin, 2*coarse_step)`` 后的外包矩形，
    避免 Nano 与 MIS 分歧时峰值落在 ROI 外，同时保证粗步长网格边缘可被覆盖。
    """
    tg = np.asarray(templ_gray, dtype=np.uint8)
    if tg.ndim != 2:
        raise ValueError("templ_gray must be 2D")
    th, tw = int(tg.shape[0]), int(tg.shape[1])
    pg = _plane_gray_u8(plane_bgr)
    Ph, Pw = int(pg.shape[0]), int(pg.shape[1])
    full_h, full_w = Ph - th + 1, Pw - tw + 1
    if full_h <= 0 or full_w <= 0:
        return np.zeros((max(0, full_h), max(0, full_w)), dtype=np.float32)
    if margin <= 0:
        return compute_template_match_response(plane_bgr, templ_gray)

    margin_eff = max(int(margin), int(coarse_step) * 2)
    boxes: list[tuple[int, int, int, int]] = [
        (int(py_tl), int(px_tl), int(bbox_h), int(bbox_w)),
    ]
    if nano_plane_box is not None:
        ny, nx, nh, nw = (int(v) for v in nano_plane_box)
        if nh > 0 and nw > 0:
            boxes.append((ny, nx, nh, nw))

    y0 = max(0, min(b[0] - margin_eff for b in boxes))
    x0 = max(0, min(b[1] - margin_eff for b in boxes))
    y1 = min(Ph, max(b[0] + b[2] + margin_eff for b in boxes))
    x1 = min(Pw, max(b[1] + b[3] + margin_eff for b in boxes))
    if y1 - y0 < th or x1 - x0 < tw:
        return compute_template_match_response(plane_bgr, templ_gray)

    rh = y1 - y0 - th + 1
    rw = x1 - x0 - tw + 1
    if rh <= 0 or rw <= 0:
        return compute_template_match_response(plane_bgr, templ_gray)

    plane_cells = float(max(1, full_h * full_w))
    if (rh * rw) >= float(full_plane_ratio) * plane_cells:
        return compute_template_match_response(plane_bgr, templ_gray)

    roi = pg[y0:y1, x0:x1]
    resp_roi = _match_template_ccoeff_normed_u8(roi, tg)

    full = np.full((full_h, full_w), np.nan, dtype=np.float32)
    rh_a, rw_a = int(resp_roi.shape[0]), int(resp_roi.shape[1])
    y_end = min(y0 + rh_a, full_h)
    x_end = min(x0 + rw_a, full_w)
    full[y0:y_end, x0:x_end] = resp_roi[: y_end - y0, : x_end - x0]
    return full


def _parse_rerank_weights(text: str) -> tuple[float, float, float, float]:
    raw = (text or "").strip()
    if not raw:
        return (0.28, 0.34, 0.24, 0.14)
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 4:
        return (0.28, 0.34, 0.24, 0.14)
    return tuple(float(x) for x in parts)


def run_scheme_c_main(coarse_scheme: str = "ab") -> None:
    apply_scheme_c_fusion_defaults()
    apply_scheme_c_fusion_defaults()
    coarse_scheme_eff = (coarse_scheme or "ab").strip().lower()
    cfg = core.TrackConfig()

    default_frames = r"E:\Surgical_Tracking_Datasets\Chess_Box_1_8\captured_photos1"
    default_out = r"F:\MIS_Pose_Track_Re3D\Exp_Results_Deformation_scheme_C_nano_prior"
    cfg.frames_dir = _env("MIS_FRAMES_DIR", default_frames) or default_frames
    cfg.output_dir = _env("MIS_OUTPUT_DIR", default_out) or default_out
    cfg.init_bbox = _env("MIS_INIT_BBOX", "")

    EXPAND_BASE = _env_int("MIS_EXPAND_BASE", 128)
    ADAPT_THRESH = _env_float("MIS_ADAPT_THRESH", 0.50)
    LOSS_THRESH = _env_float("MIS_LOSS_THRESH", 0.35)
    SLIDE_STEP = _env_int("MIS_SLIDE_STEP", 1)
    COARSE_STEP = _env_int("MIS_COARSE_STEP", 16)
    FINE_EXPAND = _env_int("MIS_FINE_EXPAND", 2)
    CONTOUR_INFER_BS = _env_int("MIS_CONTOUR_INFER_BS", 48)
    FINE_FUSE_HIST = _env_bool("MIS_FINE_FUSE_HIST", True)
    GREEDY_FUSE_HIST = _env_bool("MIS_GREEDY_FUSE_HIST", True)
    EARLY_HIST_MIN_COARSE = _env_float("MIS_EARLY_HIST_MIN", 0.0)
    EARLY_POLISH = _env_bool("MIS_EARLY_POLISH", True)
    CONTOUR_SAT_THRESH = _env_float("MIS_CONTOUR_SAT_THRESH", 0.97)
    CONTOUR_SAT_SCALE = _env_float("MIS_CONTOUR_SAT_SCALE", 0.73)
    NUM_TEMPLATES = _env_int("MIS_NUM_TEMPLATES", 256)
    SAVE_DEBUG = _env_bool("MIS_SAVE_DEBUG", True)
    PRINT_EVERY = _env_int("MIS_PRINT_EVERY", 10)
    cfg.proc_scale = _env_float("MIS_PROC_SCALE", 0.35)

    # ── 架构级优化：NanoTrack响应图引导快速定位 ─────────────────────────────
    # 问题：原方案仍然在128px范围内暴力搜索（115600像素 × 120次CNN推理）
    # 解决：用NanoTrack的响应峰值直接定位，仅在峰值±16px内精细搜索
    # 效果：搜索面积从115600降到1024像素（减少99%）
    
    NANO_RESPONSE_GUIDED = _env_bool("MIS_NANO_RESPONSE_GUIDED", True)  # 默认启用
    NANO_RESPONSE_MARGIN = _env_int("MIS_NANO_RESPONSE_MARGIN", 16)  # 响应峰值周围搜索范围
    FAST_MODE_THRESH = _env_float("MIS_FAST_MODE_THRESH", 0.65)  # Nano score高于此值走快速路径
    
    POST_BLEND = _env_bool("MIS_SCHEME_C_POST_BLEND", False)
    NANO_CONF_THRESH = _env_float("MIS_NANO_CONF", 0.4)
    NANO_BLEND_ALPHA = _env_float("MIS_NANO_ALPHA", 0.6)

    nano_sigma_scale = _env_float("MIS_NANO_PRIOR_SIGMA_SCALE", 0.55)
    nano_floor = _env_float("MIS_NANO_PRIOR_FLOOR", 0.08)
    nano_fusion = _env("MIS_NANO_FUSION", "log").strip().lower() or "log"
    nano_prior_beta = _env_float("MIS_NANO_PRIOR_BETA", 0.48)
    nano_hist_gamma = _env_float("MIS_NANO_HIST_COUPLING", 0.52)

    frames = core.list_frames(cfg.frames_dir)
    if not frames:
        raise RuntimeError(f"未找到任何帧: {cfg.frames_dir}")

    print(f"[Scheme-C] frames={len(frames)} coarse_scheme={coarse_scheme_eff}")
    print(f"[Scheme-C] nano_fusion={nano_fusion} beta={nano_prior_beta:.3f} gamma={nano_hist_gamma:.3f} "
          f"sigma_scale={nano_sigma_scale:.3f} floor={nano_floor:.3f} post_blend={int(POST_BLEND)}")
    print(f"[Scheme-C][架构优化] NANO_RESPONSE_GUIDED={NANO_RESPONSE_GUIDED} margin={NANO_RESPONSE_MARGIN}px "
          f"fast_thresh={FAST_MODE_THRESH:.2f}")

    out_root = dab._ensure_dir(cfg.output_dir)
    out_frames = dab._ensure_dir(os.path.join(out_root, "frames_with_bbox"))
    out_roi = dab._ensure_dir(os.path.join(out_root, "roi"))
    out_th = dab._ensure_dir(os.path.join(out_root, "th_gray"))
    out_feats = dab._ensure_dir(os.path.join(out_root, "features"))
    out_prior = dab._ensure_dir(os.path.join(out_root, "nano_prior"))
    out_match_coarse = dab._ensure_dir(os.path.join(out_root, "match_coarse"))
    out_match_fine = dab._ensure_dir(os.path.join(out_root, "match_fine"))
    out_match_cmp = dab._ensure_dir(os.path.join(out_root, "match_compare"))

    contour_sim_model = dab.contour_sim_model

    first_disk = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first_disk is None:
        raise RuntimeError(f"无法读取第一帧: {frames[0]}")
    first = core._resize_for_processing(first_disk, cfg)
    H, W = first.shape[:2]

    bbox = dab._select_roi(first, cfg)
    x, y, w, h = bbox

    th68_first = dab._apply_th68(first)
    target_th68 = th68_first[y : y + h, x : x + w].copy()
    cv2.imwrite(os.path.join(out_root, "th68_first_frame.jpg"), th68_first)
    cv2.imwrite(os.path.join(out_root, "target_th68_first.jpg"), target_th68)

    target_mask = np.ones((h, w), dtype=np.uint8) * 255
    template_list = dab._build_rtw_bgr(target_th68, target_mask, NUM_TEMPLATES)
    lib_templates_raw = template_list

    hist_gate_T = _env_float("MIS_HIST_GATE_T", 0.08)
    hist_h_bins = _env_int("MIS_HIST_H_BINS", 30)
    hist_s_bins = _env_int("MIS_HIST_S_BINS", 32)
    _mr = cv2.cvtColor(target_th68, cv2.COLOR_BGR2GRAY)
    ref_hist = dab.build_ref_masked_hsv_hist(
        first[y : y + h, x : x + w].copy(),
        ((_mr > 0).astype(np.uint8) * 255),
        hist_h_bins,
        hist_s_bins,
    )
    ref_hist_f32 = ref_hist.astype(np.float32)
    ref_gray_patch_sz = cv2.cvtColor(first[y : y + h, x : x + w].copy(), cv2.COLOR_BGR2GRAY)
    ref_depth_patch_sz = dab._pseudo_depth_u8_from_gray(ref_gray_patch_sz)
    ref_depth_patch_sz = dab._pseudo_depth_u8_from_gray(ref_gray_patch_sz)

    rerank_topr = _env_int("MIS_RERANK_TOPR", 48)
    rerank_weights = _parse_rerank_weights(_env("MIS_RERANK_WEIGHTS", ""))
    match_resp_enable = _env_bool("MIS_MATCH_RESP_ENABLE", True)
    match_resp_margin = _env_int("MIS_MATCH_RESP_MARGIN", 112)
    match_resp_full_ratio = float(np.clip(_env_float("MIS_MATCH_RESP_FULL_RATIO", 0.92), 0.5, 1.0))
    print(f"[Scheme-C] rerank_topr={int(rerank_topr)} match_resp_enable={int(match_resp_enable)} "
          f"match_resp_margin={int(match_resp_margin)} full_ratio={match_resp_full_ratio:.2f} "
          f"weights={tuple(round(float(w), 4) for w in rerank_weights)}")
    match_resp_margin = _env_int("MIS_MATCH_RESP_MARGIN", 112)
    match_resp_full_ratio = float(np.clip(_env_float("MIS_MATCH_RESP_FULL_RATIO", 0.92), 0.5, 1.0))
    print(f"[Scheme-C] rerank_topr={int(rerank_topr)} match_resp_enable={int(match_resp_enable)} "
          f"match_resp_margin={int(match_resp_margin)} full_ratio={match_resp_full_ratio:.2f} "
          f"weights={tuple(round(float(w), 4) for w in rerank_weights)}")

    lib_tp_cached, lib_z_cached = dab._build_lib_contour_stack_and_z(contour_sim_model, lib_templates_raw)

    out_lib = dab._ensure_dir(os.path.join(out_root, "library"))
    for ki, t in enumerate(template_list):
        cv2.imwrite(os.path.join(out_lib, f"template_{ki:04d}.jpg"), t)
    with open(os.path.join(out_lib, "library_meta.json"), "w", encoding="utf-8") as _f:
        json.dump(
            {
                "num_templates": len(template_list),
                "target_bbox": list(map(int, bbox)),
                "scheme": "C_nano_prior",
                "rerank_topr": int(rerank_topr),
                "rerank_weights": list(map(float, rerank_weights)),
                "match_resp_enable": bool(match_resp_enable),
                "match_resp_margin": int(match_resp_margin),
                "match_resp_full_ratio": float(match_resp_full_ratio),
                "coarse_step": int(COARSE_STEP),
            },
            _f,
            indent=2,
        )

    track_rows: list[dict] = [{"frame": 0, "image": frames[0], "bbox": list(map(int, bbox)), "score": 1.0, "dx": 0, "dy": 0}]
    diag_rows: list[dict] = []

    first_full = first_disk
    inv_scale = 1.0 / cfg.proc_scale if cfg.proc_scale > 0 else 1.0
    nano_bbox_init = (int(x * inv_scale), int(y * inv_scale), int(w * inv_scale), int(h * inv_scale))
    nano_tracker = dab._NanoTrackHelper(
        first_full,
        nano_bbox_init,
        backbone=_env("MIS_NANO_BACKBONE", dab._NANO_BACKBONE_DEFAULT),
        neckhead=_env("MIS_NANO_NECKHEAD", dab._NANO_NECKHEAD_DEFAULT),
    )
    if not nano_tracker.available:
        print("[Scheme-C][WARN] NanoTrack 不可用（缺少 ONNX 或初始化失败），将不使用 nano_prior_map（退回纯 HistAB）")

    prev_bbox = bbox
    base_w, base_h = w, h

    _nano_prior_none = 0
    _plane_shape_warned = False
    _nano_prior_cache: dict = {}
    _prev_track_score = 1.0
    for i, img_path in enumerate(frames[1:], start=1):
        t_frame = time.time()
        img_full = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_full is None:
            print(f"[警告] 跳过: {img_path}")
            continue
        frame = core._resize_for_processing(img_full, cfg)

        bx, by, bw, bh = [int(v) for v in prev_bbox]
        cx0, cy0 = bx + bw // 2, by + bh // 2

        if (i % PRINT_EVERY) == 0:
            print(f"[进度] frame={i}/{len(frames)-1} prev_bbox={list(map(int, prev_bbox))}")

        cx_prev = bx + bw // 2
        cy_prev = by + bh // 2
        fH, fW = frame.shape[:2]
        sy1 = max(0, by - EXPAND_BASE)
        sx1 = max(0, bx - EXPAND_BASE)
        sy2 = min(fH, by + bh + EXPAND_BASE)
        sx2 = min(fW, bx + bw + EXPAND_BASE)
        th68_curr = dab._apply_th68(frame)
        search_plane = th68_curr[sy1:sy2, sx1:sx2]
        plane_bgr_crop = frame[sy1:sy2, sx1:sx2]
        if search_plane.shape[:2] != plane_bgr_crop.shape[:2]:
            if not _plane_shape_warned:
                print(
                    "[Scheme-C][WARN] search_plane 与 plane_bgr_crop 尺寸不一致，match_resp / coarse 对齐可能异常：",
                    tuple(search_plane.shape[:2]),
                    "vs",
                    tuple(plane_bgr_crop.shape[:2]),
                )
                _plane_shape_warned = True
        if search_plane.shape[:2] != plane_bgr_crop.shape[:2]:
            if not _plane_shape_warned:
                print(
                    "[Scheme-C][WARN] search_plane 与 plane_bgr_crop 尺寸不一致，match_resp / coarse 对齐可能异常：",
                    tuple(search_plane.shape[:2]),
                    "vs",
                    tuple(plane_bgr_crop.shape[:2]),
                )
                _plane_shape_warned = True
        plane_center_yx = (float(cy_prev - sy1), float(cx_prev - sx1))

        ph, pw = search_plane.shape[:2]
        nano_prior_map = None
        nano_bbox_proc = None
        nano_score = 0.0
        last_nb_proc = None
        last_ns = 0.0
        motion_px = 0.0

        if nano_tracker.available:
            nano_bbox_full, nano_score = nano_tracker.update(img_full)
            scale = cfg.proc_scale if cfg.proc_scale > 0 else 1.0
            nb_x = int(nano_bbox_full[0] * scale)
            nb_y = int(nano_bbox_full[1] * scale)
            nb_w = max(1, int(nano_bbox_full[2] * scale))
            nb_h = max(1, int(nano_bbox_full[3] * scale))
            nano_bbox_proc = tuple(core.clamp_bbox(nb_x, nb_y, nb_w, nb_h, frame.shape[1], frame.shape[0]))
            last_nb_proc = nano_bbox_proc
            last_ns = float(nano_score)
            motion_px = bbox_center_motion_px(prev_bbox, nano_bbox_proc)

            if nano_score >= NANO_CONF_THRESH:
                nano_prior_map, _nano_prior_cache = get_or_build_nano_prior_map(
                    ph,
                    pw,
                    nano_bbox_proc,
                    (sx1, sy1),
                    nano_score,
                    (base_w, base_h),
                    _nano_prior_cache,
                    sigma_scale=nano_sigma_scale,
                    uniform_floor=nano_floor,
                )
            else:
                _nano_prior_none += 1
                _nano_prior_cache.clear()

        lib_tp_use, lib_z_use = adapt_contour_lib_cache(lib_tp_cached, lib_z_cached, motion_px)

        match_resp = None
        if should_compute_match_response(
            rerank_topr=rerank_topr,
            match_resp_enable=match_resp_enable,
            last_ns=last_ns,
            motion_px=motion_px,
            prev_track_score=_prev_track_score,
        ):
            try:
                py_tl_p = max(0, by - sy1)
                px_tl_p = max(0, bx - sx1)
                nano_plane_box = None
                if nano_bbox_proc is not None:
                    nbx, nby, nbw, nbh = [int(v) for v in nano_bbox_proc]
                    npy1, npx1 = nby - sy1, nbx - sx1
                    npy2 = min(ph, npy1 + nbh)
                    npx2 = min(pw, npx1 + nbw)
                    npy1 = max(0, npy1)
                    npx1 = max(0, npx1)
                    if npy2 > npy1 and npx2 > npx1:
                        nano_plane_box = (npy1, npx1, npy2 - npy1, npx2 - npx1)
                match_resp = compute_template_match_response_roi_embedded(
                    plane_bgr_crop,
                    ref_gray_patch_sz,
                    py_tl=py_tl_p,
                    px_tl=px_tl_p,
                    bbox_h=bh,
                    bbox_w=bw,
                    margin=int(match_resp_margin),
                    coarse_step=int(COARSE_STEP),
                    full_plane_ratio=float(match_resp_full_ratio),
                    nano_plane_box=nano_plane_box,
                )
            except Exception:
                match_resp = None

        # ── 架构级优化：NanoTrack响应图引导快速定位 ─────────────────────
        pw_old, ph_old = pw, ph  # 保存原始尺寸用于对比
        
        if NANO_RESPONSE_GUIDED and nano_tracker.available and nano_score >= FAST_MODE_THRESH:
            # ── 快速路径：NanoTrack已经可靠 → 仅在其周围小范围搜索 ─────────
            nbx, nby, nbw, nbh = [int(v) for v in nano_bbox_proc]
            
            # 搜索范围：NanoTrack bbox ± 16px（vs 原来的±128px）
            margin = NANO_RESPONSE_MARGIN
            sy1_fast = max(0, nby - margin)
            sx1_fast = max(0, nbx - margin)
            sy2_fast = min(fH, nby + nbh + margin)
            sx2_fast = min(fW, nbx + nbw + margin)
            
            # 裁剪搜索平面（面积减少99%）
            search_plane = th68_curr[sy1_fast:sy2_fast, sx1_fast:sx2_fast]
            plane_bgr_crop = frame[sy1_fast:sy2_fast, sx1_fast:sx2_fast]
            
            # 调整prev_bbox到快速坐标系
            prev_bbox_fast = (
                max(0, bx - sx1_fast),
                max(0, by - sy1_fast),
                bw, bh
            )
            cx_fast = prev_bbox_fast[0] + bw // 2
            cy_fast = prev_bbox_fast[1] + bh // 2
            plane_center_yx = (float(cy_fast), float(cx_fast))
            
            # 缩小搜索步长（因为搜索范围已经很小）
            step_coarse_fast = max(4, COARSE_STEP // 2)  # 16→8
            expand_base_fast = 0  # 不再需要大范围搜索
            
            ph, pw = search_plane.shape[:2]
            
            # 构建Nano先验图（仍然需要，但范围更小）
            if nano_score >= NANO_CONF_THRESH:
                nano_prior_map_fast, _nano_prior_cache = get_or_build_nano_prior_map(
                    ph, pw,
                    (nbx - sx1_fast, nby - sy1_fast, nbw, nbh),  # 转换到快速坐标系
                    (sx1_fast, sy1_fast),
                    nano_score, (base_w, base_h),
                    _nano_prior_cache,
                    sigma_scale=nano_sigma_scale,
                    uniform_floor=nano_floor,
                )
            else:
                nano_prior_map_fast = None
            
            # 调用层次搜索（范围小、步长大、速度极快）
            result = dab.find_best_match_hierarchical(
                search_plane,
                None, base_h, base_w, None,
                contour_sim_model=contour_sim_model,
                lib_templates_raw=lib_templates_raw,
                lib_contour_encodings=None,
                step_coarse=step_coarse_fast,  # 8（vs 16）
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
                cached_lib_tensor=lib_tp_use,
                cached_lib_z=lib_z_use,
                nano_prior_map=nano_prior_map_fast,
                nano_fusion=nano_fusion,
                nano_prior_beta=nano_prior_beta,
                nano_hist_coupling=nano_hist_gamma,
                rerank_topr=rerank_topr,
                match_resp=None,  # 快速模式下跳过matchTemplate
                ref_gray_patch_sz=ref_gray_patch_sz,
                ref_depth_patch_sz=ref_depth_patch_sz,
                nano_confidence=float(last_ns),
                rerank_weights=rerank_weights,
            )
            
            # 转换回全局坐标系
            new_x = int(result["x"]) + sx1_fast
            new_y = int(result["y"]) + sy1_fast
            
            if (i % PRINT_EVERY) == 0:
                print(f"[快速路径] frame={i} Nano={nano_score:.2f} 搜索范围={pw}×{ph} (vs {pw_old}×{ph_old})")
        else:
            # ── 标准路径：NanoTrack不可靠 → 原始大范围搜索 ──────────────
            if NANO_RESPONSE_GUIDED:
                pw_old, ph_old = pw, ph  # 保存原始尺寸用于对比
            
            result = dab.find_best_match_hierarchical(
                search_plane,
                None,
                base_h,
                base_w,
                None,
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
                cached_lib_tensor=lib_tp_use,
                cached_lib_z=lib_z_use,
                nano_prior_map=nano_prior_map,
                nano_fusion=nano_fusion,
                nano_prior_beta=nano_prior_beta,
                nano_hist_coupling=nano_hist_gamma,
                rerank_topr=rerank_topr,
                match_resp=match_resp,
                ref_gray_patch_sz=ref_gray_patch_sz,
                ref_depth_patch_sz=ref_depth_patch_sz,
                nano_confidence=float(last_ns),
                rerank_weights=rerank_weights,
            )
            
            new_x = int(result["x"]) + sx1
            new_y = int(result["y"]) + sy1
        new_x, new_y, new_w, new_h = core.clamp_bbox(new_x, new_y, base_w, base_h, frame.shape[1], frame.shape[0])

        crop_th68 = th68_curr[new_y : new_y + new_h, new_x : new_x + new_w]
        new_x, new_y, new_w, new_h = dab._refine_bbox_by_contour(
            crop_th68,
            new_x,
            new_y,
            frame.shape[1],
            frame.shape[0],
            margin=_env_int("MIS_CONTOUR_MARGIN", 10),
            min_fill_ratio=_env_float("MIS_MIN_FILL", 0.01),
            max_size_factor=_env_float("MIS_MAX_SIZE", 2.0),
            orig_w=base_w,
            orig_h=base_h,
        )
        new_bbox = (new_x, new_y, new_w, new_h)
        score = float(result["score"])
        _prev_track_score = score

        # 可选：旧版事后线性框融合（默认关）；复用本帧首次 NanoTrack.update，避免重复 step
        if POST_BLEND and nano_tracker.available and last_nb_proc is not None:
            nb_proc = last_nb_proc
            ns = float(last_ns)
            if score < LOSS_THRESH and ns > NANO_CONF_THRESH:
                new_bbox = nb_proc
                new_x, new_y, new_w, new_h = new_bbox
                score = float(ns * 0.8)
            elif LOSS_THRESH <= score < ADAPT_THRESH and ns > NANO_CONF_THRESH:
                alpha = max(NANO_BLEND_ALPHA, min(0.9, score / ADAPT_THRESH))
                new_bbox = dab._fuse_bboxes(new_bbox, score, nb_proc, ns, alpha)
                new_x, new_y, new_w, new_h = new_bbox

        dx = int((new_x + new_w // 2) - cx0)
        dy = int((new_y + new_h // 2) - cy0)

        if SAVE_DEBUG:
            fn = f"frame_{i:04d}"
            cv2.imwrite(os.path.join(out_frames, fn + ".jpg"), core.draw_bbox(frame, new_bbox, text=f"f={i} s={score:.2f}"))
            cv2.imwrite(os.path.join(out_roi, fn + ".jpg"), frame[new_y : new_y + new_h, new_x : new_x + new_w])
            cv2.imwrite(os.path.join(out_th, fn + ".jpg"), th68_curr[new_y : new_y + new_h, new_x : new_x + new_w])
            if nano_prior_map is not None:
                vis = (nano_prior_map * 255.0).astype(np.uint8)
                vis_clr = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(out_prior, fn + ".png"), vis_clr)

        track_rows.append({"frame": i, "image": img_path, "bbox": list(map(int, new_bbox)), "score": float(score), "dx": dx, "dy": dy})
        diag_rows.append({"frame": i, "score": float(score), "dx": dx, "dy": dy, "elapsed_s": float(f"{time.time() - t_frame:.3f}")})

        prev_bbox = new_bbox

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
        "scheme": "C_nano_prior_log_fusion",
        "frames_dir": cfg.frames_dir,
        "output_dir": cfg.output_dir,
        "nano_fusion": nano_fusion,
        "nano_prior_beta": nano_prior_beta,
        "nano_hist_coupling": nano_hist_gamma,
        "mis_scheme_c_post_blend": int(POST_BLEND),
        "nano_prior_none_frames_low_conf": _nano_prior_none,
        "rerank_topr": int(rerank_topr),
        "rerank_weights": list(map(float, rerank_weights)),
        "match_resp_enable": bool(match_resp_enable),
        "match_resp_margin": int(match_resp_margin),
        "match_resp_full_ratio": float(match_resp_full_ratio),
        "coarse_step": int(COARSE_STEP),
        "expand_base": int(EXPAND_BASE),
        "proc_scale": float(cfg.proc_scale),
        "coarse_scheme": coarse_scheme_eff,
    }
    with open(os.path.join(out_root, "meta_scheme_C.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Scheme-C][完成] 输出目录: {out_root}")
    print(f"[Scheme-C][完成] Nano 低置信未送先验帧计数≈{_nano_prior_none}（score<{NANO_CONF_THRESH}）")


if __name__ == "__main__":
    scheme = _env("MIS_COARSE_SCHEME", "ab").strip().lower()
    run_scheme_c_main(coarse_scheme=scheme)
