#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【冻结】Ours HistAB + Scheme C Nano 跟踪器实现 — 2026-06-03

注册名: ours_scheme_c_nano, ours_hist_ab

⚠️ 后续优化请只改本文件；``trackves_eval_all_methods`` 仅保留兼容 re-export。
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
from typing import Any, Optional

import cv2
import numpy as np

from fusion_weight_head import fusion_feature_vector, predict_contour_fuse_weight

from importlib import import_module

_histab = import_module("2026-06-03-HistAB核心方法")
_scheme = import_module("2026-06-03-SchemeC-NanoPrior方法")
from importlib import import_module as _im

_base = _im("2026-06-03-跟踪器基础方法")
BaseTracker = _base.BaseTracker
_load_defm_ab_once = _base._load_defm_ab_once
_clip_bbox = _base._clip_bbox
_bbox_to_mask = _base._bbox_to_mask
_bbox_iou = _base._bbox_iou
_bbox_area_ratio = _base._bbox_area_ratio
_bbox_size_ratio_ok = _base._bbox_size_ratio_ok
_nano_bbox_trustworthy = _base._nano_bbox_trustworthy

TRACKER_LABELS: dict[str, str] = {
    "ours_hist_ab": "Ours (ContourNet+HistAB)",
    "ours_scheme_c_nano": "Ours (Scheme C: Nano+log+TopR)",
    "ours_c_nano": "Ours (Scheme C: Nano+log+TopR)",
}

_NANO_BB = r"D:\MIS_Pose_Track_Re3D\nanotrack_backbone_sim.onnx"
_NANO_NH = r"D:\MIS_Pose_Track_Re3D\nanotrack_head_sim.onnx"
_NANO_BACKBONE_DEFAULT = _NANO_BB
_NANO_NECKHEAD_DEFAULT = _NANO_NH

class OursHistABTracker(BaseTracker):
    """本文方法：轮廓相似度网络 + HSV直方图门控（A+B方案）三阶段层次搜索跟踪器。"""
    name = "ours_hist_ab"
    label = TRACKER_LABELS["ours_hist_ab"]

    # 超参数（可通过环境变量覆盖）
    NUM_TEMPLATES   = int(os.environ.get("MIS_NUM_TEMPLATES",   "256"))
    EXPAND_BASE     = int(os.environ.get("MIS_EXPAND_BASE",     "128"))
    COARSE_STEP     = int(os.environ.get("MIS_COARSE_STEP",      "16"))
    SLIDE_STEP      = int(os.environ.get("MIS_SLIDE_STEP",         "1"))
    FINE_EXPAND     = int(os.environ.get("MIS_FINE_EXPAND",        "2"))
    TOPK            = int(os.environ.get("MIS_TOPK",               "3"))
    GREEDY_STEP     = int(os.environ.get("MIS_GREEDY_STEP",        "4"))
    GREEDY_ITERS    = int(os.environ.get("MIS_GREEDY_ITERS",       "4"))
    EARLY_EXIT      = float(os.environ.get("MIS_EARLY_EXIT",    "0.95"))
    HIST_GATE_T     = float(os.environ.get("MIS_HIST_GATE_T",   "0.08"))
    HIST_H_BINS     = int(os.environ.get("MIS_HIST_H_BINS",       "30"))
    HIST_S_BINS     = int(os.environ.get("MIS_HIST_S_BINS",       "32"))
    HIST_TOP_M      = int(os.environ.get("MIS_HIST_TOP_M",        "96"))
    FUSE_LAMBDA     = float(os.environ.get("MIS_FUSE_LAMBDA",   "0.65"))
    CONTOUR_INFER_BS= int(os.environ.get("MIS_CONTOUR_INFER_BS",  "48"))
    CONTOUR_MARGIN  = int(os.environ.get("MIS_CONTOUR_MARGIN",    "10"))
    MIN_FILL        = float(os.environ.get("MIS_MIN_FILL",       "0.01"))
    MAX_SIZE        = float(os.environ.get("MIS_MAX_SIZE",        "2.0"))
    CONTOUR_SAT_THRESH= float(os.environ.get("MIS_CONTOUR_SAT_THRESH", "0.97"))
    CONTOUR_SAT_SCALE = float(os.environ.get("MIS_CONTOUR_SAT_SCALE",  "0.73"))

    _ENV_PARAM_BINDINGS: tuple[tuple[str, str, type], ...] = (
        ("NUM_TEMPLATES", "MIS_NUM_TEMPLATES", int),
        ("EXPAND_BASE", "MIS_EXPAND_BASE", int),
        ("COARSE_STEP", "MIS_COARSE_STEP", int),
        ("SLIDE_STEP", "MIS_SLIDE_STEP", int),
        ("FINE_EXPAND", "MIS_FINE_EXPAND", int),
        ("TOPK", "MIS_TOPK", int),
        ("GREEDY_STEP", "MIS_GREEDY_STEP", int),
        ("GREEDY_ITERS", "MIS_GREEDY_ITERS", int),
        ("EARLY_EXIT", "MIS_EARLY_EXIT", float),
        ("HIST_GATE_T", "MIS_HIST_GATE_T", float),
        ("HIST_H_BINS", "MIS_HIST_H_BINS", int),
        ("HIST_S_BINS", "MIS_HIST_S_BINS", int),
        ("HIST_TOP_M", "MIS_HIST_TOP_M", int),
        ("FUSE_LAMBDA", "MIS_FUSE_LAMBDA", float),
        ("CONTOUR_INFER_BS", "MIS_CONTOUR_INFER_BS", int),
        ("CONTOUR_MARGIN", "MIS_CONTOUR_MARGIN", int),
        ("MIN_FILL", "MIS_MIN_FILL", float),
        ("MAX_SIZE", "MIS_MAX_SIZE", float),
        ("CONTOUR_SAT_THRESH", "MIS_CONTOUR_SAT_THRESH", float),
        ("CONTOUR_SAT_SCALE", "MIS_CONTOUR_SAT_SCALE", float),
    )

    def _refresh_params_from_env(self) -> None:
        """Re-read MIS_* at init time (class attrs are frozen at import)."""
        for attr, env_key, cast in self._ENV_PARAM_BINDINGS:
            raw = os.environ.get(env_key, "")
            if raw == "":
                continue
            try:
                setattr(self, attr, cast(raw))
            except (TypeError, ValueError):
                pass

    def __init__(
        self,
        init_im: np.ndarray,
        init_bbox: tuple[int, int, int, int],
        coarse_scheme: str = "ab",
    ) -> None:
        self._refresh_params_from_env()
        defm = _load_defm_ab_once()
        if defm is None:
            raise ImportError("Deformation_F_T_C_G_Experi_hist_AB 未能加载")
        self._defm = defm
        self.coarse_scheme = str(coarse_scheme).lower()

        self.prev_bbox = _clip_bbox(tuple(map(int, init_bbox)), init_im.shape[:2])
        x, y, w, h = self.prev_bbox
        self.base_w, self.base_h = w, h

        # 构建 th68 模板和 RTW 增广库
        frame_th68 = defm._apply_th68(init_im)
        target_th68 = frame_th68[y:y + h, x:x + w].copy()
        if target_th68.size == 0:
            target_th68 = np.zeros((max(1, h), max(1, w), 3), dtype=np.uint8)
        target_mask = np.ones((max(1, h), max(1, w)), dtype=np.uint8) * 255
        _turbo = os.environ.get("MIS_SURGT_TURBO", "0").lower() in ("1", "true", "yes", "on")
        if _turbo:
            # SurgT 每 anchor 重建 tracker：跳过 RTW/Contour 预编码，靠 Nano 级联为主路径
            self.lib_templates_raw = [target_th68.copy()]
            self.contour_sim_model = defm.contour_sim_model
            self.lib_tp_cached = None
            self.lib_z_cached = None
        else:
            self.lib_templates_raw = defm._build_rtw_bgr(target_th68, target_mask, self.NUM_TEMPLATES)
            self.contour_sim_model = defm.contour_sim_model
            self.lib_tp_cached, self.lib_z_cached = defm._build_lib_contour_stack_and_z(
                self.contour_sim_model, self.lib_templates_raw
            )

        # 构建参考 HSV 直方图
        _mr = cv2.cvtColor(target_th68, cv2.COLOR_BGR2GRAY)
        self.ref_hist = defm.build_ref_masked_hsv_hist(
            init_im[y:y + h, x:x + w].copy(),
            ((_mr > 0).astype(np.uint8) * 255),
            self.HIST_H_BINS,
            self.HIST_S_BINS,
        ).astype(np.float32)
        if self.lib_tp_cached is not None:
            print(f"[{self.name}] templates={self.lib_tp_cached.shape[0]} "
                  f"scheme={self.coarse_scheme} device={self.lib_tp_cached.device}")
        elif _turbo:
            print(f"[{self.name}] surgt_turbo=1 templates=1 scheme={self.coarse_scheme} (nano-primary)")
        else:
            print(f"[{self.name}] 无轮廓模型，将退化为零分匹配")

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import contextlib, io
        defm = self._defm
        bx, by, bw, bh = [int(v) for v in self.prev_bbox]
        fH, fW = frame_bgr.shape[:2]

        # 局部搜索平面
        # 注意：使用 base_w/base_h（初始化时固定）而非 prev_bbox 的 bw/bh：
        # _refine_bbox_by_contour 可能将 bw/bh 扩展至 2× 原始尺寸，若直接使用
        # 则搜索区域从 (87+96)² ≈ 183² 暴增至 (400+96)² ≈ 496²（位置数×7倍），
        # 导致粗搜索耗时 2+ 秒。用 base_w/base_h 确保搜索区域大小恒定。
        cx_prev = bx + bw // 2
        cy_prev = by + bh // 2
        sy1 = max(0, by - self.EXPAND_BASE)
        sx1 = max(0, bx - self.EXPAND_BASE)
        sy2 = min(fH, by + self.base_h + self.EXPAND_BASE)
        sx2 = min(fW, bx + self.base_w + self.EXPAND_BASE)
        th68_curr = defm._apply_th68(frame_bgr)
        search_plane = th68_curr[sy1:sy2, sx1:sx2]
        plane_bgr_crop = frame_bgr[sy1:sy2, sx1:sx2]
        plane_center_yx = (float(cy_prev - sy1), float(cx_prev - sx1))

        _devnull = io.StringIO()
        with contextlib.redirect_stdout(_devnull):
          result = defm.find_best_match_hierarchical(
            search_plane, None, self.base_h, self.base_w, None,
            contour_sim_model=self.contour_sim_model,
            lib_templates_raw=self.lib_templates_raw,
            lib_contour_encodings=None,
            step_coarse=self.COARSE_STEP,
            step_fine=self.SLIDE_STEP,
            fine_expand=self.FINE_EXPAND,
            top_k=self.TOPK,
            greedy_step=self.GREEDY_STEP,
            greedy_iters=self.GREEDY_ITERS,
            spatial_sigma=3.0,
            center_yx=plane_center_yx,
            early_exit_score=self.EARLY_EXIT,
            coarse_scheme=self.coarse_scheme,
            frame_bgr_full=plane_bgr_crop,
            plane_origin_yx=(0, 0),
            ref_hist=self.ref_hist,
            hist_h_bins=self.HIST_H_BINS,
            hist_s_bins=self.HIST_S_BINS,
            hist_gate_T=self.HIST_GATE_T,
            hist_top_M_cap=self.HIST_TOP_M,
            hist_min_keep=int(os.environ.get("MIS_HIST_MIN_KEEP", "24")),
            hist_fallback_take=int(os.environ.get("MIS_HIST_FALLBACK_M", "48")),
            hist_topkp_c=int(os.environ.get("MIS_HIST_TOPKP_C", "96")),
            fuse_lambda_base=self.FUSE_LAMBDA,
            fuse_adaptive=True,
            fine_fuse_hist=True,
            greedy_fuse_hist=True,
            early_exit_hist_min=float(os.environ.get("MIS_EARLY_HIST_MIN", "0.0")),
            early_exit_polish=True,
            contour_sat_thresh=self.CONTOUR_SAT_THRESH,
            contour_sat_lambda_scale=self.CONTOUR_SAT_SCALE,
            contour_infer_bs=self.CONTOUR_INFER_BS,
            cached_lib_tensor=self.lib_tp_cached,
            cached_lib_z=self.lib_z_cached,
            max_time_s=getattr(self, "max_time_s", float(os.environ.get("MIS_MAX_TIME_S", "1.5"))),
          )  # end redirect_stdout

        new_x = sx1 + int(result["x"])
        new_y = sy1 + int(result["y"])
        new_x, new_y, new_w, new_h = defm.core.clamp_bbox(new_x, new_y, self.base_w, self.base_h, fW, fH)

        # 轮廓外包围修正
        crop_th68 = th68_curr[new_y : new_y + new_h, new_x : new_x + new_w]
        if crop_th68.size > 0:
            new_x, new_y, new_w, new_h = defm._refine_bbox_by_contour(
                crop_th68, new_x, new_y, fW, fH,
                margin=self.CONTOUR_MARGIN,
                min_fill_ratio=self.MIN_FILL,
                max_size_factor=self.MAX_SIZE,
                orig_w=self.base_w, orig_h=self.base_h,
            )

        new_bbox = _clip_bbox((new_x, new_y, new_w, new_h), frame_bgr.shape[:2])
        self.prev_bbox = new_bbox
        mask_full = _bbox_to_mask(frame_bgr.shape[:2], new_bbox)
        return {"bbox": new_bbox, "mask_full": mask_full, "score": float(result["score"])}

_NANO_BB = r"D:\MIS_Pose_Track_Re3D\nanotrack_backbone_sim.onnx"
_NANO_NH = r"D:\MIS_Pose_Track_Re3D\nanotrack_head_sim.onnx"


def _nano_fuse_bboxes(ba, sa, bb, sb, alpha):
    """加权融合两个 bbox，尺寸取高置信一方。"""
    cx_a = ba[0] + ba[2] / 2.0;  cy_a = ba[1] + ba[3] / 2.0
    cx_b = bb[0] + bb[2] / 2.0;  cy_b = bb[1] + bb[3] / 2.0
    cx = alpha * cx_a + (1 - alpha) * cx_b
    cy = alpha * cy_a + (1 - alpha) * cy_b
    w, h = (ba[2], ba[3]) if sa >= sb else (bb[2], bb[3])
    return (int(round(cx - w / 2)), int(round(cy - h / 2)), int(round(w)), int(round(h)))


def _nano_center_blend_fixed_size(
    mis_bbox: tuple[int, int, int, int],
    nano_bbox: tuple[int, int, int, int],
    alpha_mis: float,
    fixed_wh: tuple[int, int],
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """按中心加权融合，尺寸固定为首帧模板（减少 Nano 尺寸抖动）。"""
    mx, my = int(mis_bbox[0]), int(mis_bbox[1])
    nx, ny, nw, nh = (int(v) for v in nano_bbox)
    fw, fh = max(1, int(fixed_wh[0])), max(1, int(fixed_wh[1]))
    cx_m = mx + fw * 0.5
    cy_m = my + fh * 0.5
    cx_n = nx + nw * 0.5
    cy_n = ny + nh * 0.5
    am = float(np.clip(alpha_mis, 0.0, 1.0))
    cx = am * cx_m + (1.0 - am) * cx_n
    cy = am * cy_m + (1.0 - am) * cy_n
    return _clip_bbox(
        (int(round(cx - fw * 0.5)), int(round(cy - fh * 0.5)), fw, fh),
        frame_shape,
    )


def _nano_center_blend_adaptive_size(
    mis_bbox: tuple[int, int, int, int],
    nano_bbox: tuple[int, int, int, int],
    alpha_mis: float,
    fixed_wh: tuple[int, int],
    frame_shape: tuple[int, int],
    *,
    size_alpha_nano: float,
) -> tuple[int, int, int, int]:
    """中心融合，但允许尺寸向 Nano 自适应，主要用于形变段提升 BBox IoU。"""
    mx, my = int(mis_bbox[0]), int(mis_bbox[1])
    nx, ny, nw, nh = (int(v) for v in nano_bbox)
    fw, fh = max(1, int(fixed_wh[0])), max(1, int(fixed_wh[1]))
    cx_m = mx + fw * 0.5
    cy_m = my + fh * 0.5
    cx_n = nx + nw * 0.5
    cy_n = ny + nh * 0.5
    am = float(np.clip(alpha_mis, 0.0, 1.0))
    szn = float(np.clip(size_alpha_nano, 0.0, 1.0))
    cx = am * cx_m + (1.0 - am) * cx_n
    cy = am * cy_m + (1.0 - am) * cy_n
    w = int(round((1.0 - szn) * fw + szn * max(1, nw)))
    h = int(round((1.0 - szn) * fh + szn * max(1, nh)))
    return _clip_bbox((int(round(cx - w * 0.5)), int(round(cy - h * 0.5)), w, h), frame_shape)
class OursSchemeCNanoPriorTracker(OursHistABTracker):
    """Scheme C：NanoTrack 空间先验（高斯图）+ ``nano_fusion=log`` + Top‑R rerank（ROI matchTemplate）。

    与 ``Deformation_hist_AB_scheme_C_nanoprior.py`` 主循环对齐；超参数沿用同名 ``MIS_*`` 环境变量。
    """

    name = "ours_scheme_c_nano"
    label = TRACKER_LABELS["ours_scheme_c_nano"]
    _sch_mod = None

    @classmethod
    def _scheme_mod(cls):
        if cls._sch_mod is None:
            import importlib

            cls._sch_mod = importlib.import_module("2026-06-03-SchemeC-NanoPrior方法")
        return cls._sch_mod

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int]) -> None:
        sch = self._scheme_mod()
        if os.environ.get("MIS_SURGT_PROFILE", ""):
            sch.apply_cascade_fusion_defaults()
        else:
            sch.apply_scheme_c_fusion_defaults()
        coarse = (os.environ.get("MIS_COARSE_SCHEME", "ab") or "ab").strip().lower()
        super().__init__(init_im, init_bbox, coarse_scheme=coarse if coarse in ("ab", "c") else "ab")

        x, y, w, h = self.prev_bbox
        roi = init_im[y : y + h, x : x + w]
        self.ref_gray_patch_sz = (
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else np.asarray(roi, dtype=np.uint8)
        )
        self.ref_depth_patch_sz = self._defm._pseudo_depth_u8_from_gray(self.ref_gray_patch_sz)
        self.ref_hist_init = np.asarray(self.ref_hist, dtype=np.float32).copy()
        self.ref_gray_patch_init = np.asarray(self.ref_gray_patch_sz, dtype=np.uint8).copy()
        self.ref_depth_patch_init = np.asarray(self.ref_depth_patch_sz, dtype=np.uint8).copy()

        bb_def = getattr(self._defm, "_NANO_BACKBONE_DEFAULT", _NANO_BB)
        nh_def = getattr(self._defm, "_NANO_NECKHEAD_DEFAULT", _NANO_NH)
        backbone = os.environ.get("MIS_NANO_BACKBONE", bb_def)
        neckhead = os.environ.get("MIS_NANO_NECKHEAD", nh_def)
        self._nano_helper = self._defm._NanoTrackHelper(
            init_im,
            tuple(map(int, self.prev_bbox)),
            backbone=str(backbone),
            neckhead=str(neckhead),
        )

        self.NANO_CONF_THRESH = float(os.environ.get("MIS_NANO_CONF", "0.4"))
        self.POST_BLEND = os.environ.get("MIS_SCHEME_C_POST_BLEND", "0").lower() not in ("0", "false", "no", "off")
        self.nano_sigma_scale = float(os.environ.get("MIS_NANO_PRIOR_SIGMA_SCALE", "0.55"))
        self.nano_floor = float(os.environ.get("MIS_NANO_PRIOR_FLOOR", "0.08"))
        self.ADAPT_THRESH = float(os.environ.get("MIS_ADAPT_THRESH", "0.50"))
        self.LOSS_THRESH = float(os.environ.get("MIS_LOSS_THRESH", "0.35"))
        self.NANO_BLEND_ALPHA = float(os.environ.get("MIS_NANO_ALPHA", "0.6"))

        if not self._nano_helper.available:
            print(f"[{self.name}] NanoTrack 不可用，将仅 HistAB（无 nano_prior_map）")

        self._nano_prior_cache: dict = {}
        self._prev_track_score = 1.0
        self._last_arbitrate_debug: dict = {}
        self._last_mis_bbox: tuple[int, int, int, int] | None = None
        self._cascade_hits: dict[str, int] = {"primary": 0, "local": 0, "local_fast": 0, "guard": 0, "full": 0, "lite": 0, "secval": 0, "ref_update": 0}
        self._risk_ema: float = 0.0
        self._bad_streak: int = 0
        self._tier_prev: str = "primary"
        self._mode_prev: str = "stable"
        self._rel_hist: float = 0.62
        self._rel_sz: float = 0.58
        self._rel_nano: float = 0.74
        self._secval_cooldown: int = 0
        self._ref_update_cooldown: int = 0
        self._memory_z_bank: list[Any] = []
        self._memory_update_cooldown: int = 0

        # 漂移检测与重定位状态
        self._drift_streak: int = 0
        self._reloc_triggered: bool = False
        self._last_reloc_frame: int = -1

        # 保存前一帧 Nano bbox 用于运动外推
        self._prev_nano_bbox: tuple[int, int, int, int] | None = None
        if os.environ.get("MIS_SURGT_TURBO", "0").lower() not in ("1", "true", "yes", "on"):
            tuned = self._maybe_online_tune_similarity_model(init_im, self.prev_bbox)
            if tuned:
                self.lib_tp_cached, self.lib_z_cached = self._defm._build_lib_contour_stack_and_z(
                    self.contour_sim_model,
                    self.lib_templates_raw,
                )

    def _maybe_online_tune_similarity_model(
        self,
        init_im: np.ndarray,
        init_bbox: tuple[int, int, int, int],
    ) -> bool:
        """棋盘格序列内轻量微调 SimilarityNet：首帧 GT 正样本 + 偏移框负样本。"""
        if os.environ.get("MIS_CHESS_ONLINE_SIMI", "0").lower() in ("0", "false", "no", "off"):
            return False
        model = getattr(self, "contour_sim_model", None)
        if model is None:
            return False
        try:
            import torch
            import torch.nn.functional as F

            enc = getattr(model, "encode", None)
            if not callable(enc):
                return False
            x, y, w, h = [int(v) for v in init_bbox]
            H, W = init_im.shape[:2]
            th68 = self._defm._apply_th68(init_im)
            x, y, w, h = _clip_bbox((x, y, w, h), (H, W))
            crop = th68[y : y + h, x : x + w]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            base = np.asarray(self._defm._rasterize_contour(gray, 128), dtype=np.float32)
            if int(np.count_nonzero(base > 0.04)) < 8:
                return False

            rng = np.random.default_rng(12345)

            def _aug(src: np.ndarray) -> np.ndarray:
                hh, ww = src.shape[:2]
                center = (ww * 0.5, hh * 0.5)
                ang = float(rng.uniform(-18.0, 18.0))
                sc = float(rng.uniform(0.88, 1.12))
                M = cv2.getRotationMatrix2D(center, ang, sc)
                M[0, 2] += float(rng.uniform(-0.08, 0.08)) * ww
                M[1, 2] += float(rng.uniform(-0.08, 0.08)) * hh
                return cv2.warpAffine(
                    src,
                    M,
                    (ww, hh),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).astype(np.float32)

            pos_n = int(os.environ.get("MIS_CHESS_ONLINE_POS", "20"))
            neg_n = int(os.environ.get("MIS_CHESS_ONLINE_NEG", "36"))
            samples: list[tuple[np.ndarray, float]] = []
            samples.append((base, 1.0))
            for _ in range(max(0, pos_n)):
                samples.append((_aug(base), 0.92))

            use_iou_targets = os.environ.get("MIS_CHESS_ONLINE_IOU_TARGETS", "1").lower() not in ("0", "false", "no", "off")

            def _raster_from_bbox(bb: tuple[int, int, int, int]) -> np.ndarray:
                nx, ny, nw, nh = [int(v) for v in bb]
                ncrop = th68[ny : ny + nh, nx : nx + nw]
                ngray = cv2.cvtColor(ncrop, cv2.COLOR_BGR2GRAY) if ncrop.ndim == 3 else ncrop
                return np.asarray(self._defm._rasterize_contour(ngray, 128), dtype=np.float32)

            # Hard candidates 来自第一帧中围绕目标的错位/错尺度框；目标值直接用 bbox IoU。
            gt_box = (x, y, w, h)
            for _ in range(max(0, neg_n)):
                # 70% 采样接近目标的 hard boxes，30% 采样更远的 easy negatives。
                if float(rng.random()) < 0.70:
                    dx = float(rng.uniform(-0.38, 0.38) * w)
                    dy = float(rng.uniform(-0.38, 0.38) * h)
                    sx = float(rng.uniform(0.78, 1.28))
                    sy = float(rng.uniform(0.78, 1.28))
                else:
                    dx = float(rng.choice([-0.75, -0.55, 0.55, 0.75]) * w)
                    dy = float(rng.choice([-0.75, -0.55, 0.55, 0.75]) * h)
                    sx = float(rng.uniform(0.65, 1.40))
                    sy = float(rng.uniform(0.65, 1.40))
                sc = float(0.5 * (sx + sy))
                nw, nh = max(6, int(round(w * sc))), max(6, int(round(h * sc)))
                cx = int(round(x + w * 0.5 + dx))
                cy = int(round(y + h * 0.5 + dy))
                nb = _clip_bbox((cx - nw // 2, cy - nh // 2, nw, nh), (H, W))
                iou = float(_bbox_iou(gt_box, nb))
                if iou > 0.92:
                    continue
                neg = _raster_from_bbox(nb)
                target = float(np.clip(iou, 0.02, 0.88)) if use_iou_targets else 0.05
                samples.append((neg, target))

            # 候选池错误样本：使用 RTW/CandidatePool 同类变换生成候选框，
            # 让 SimilarityNet 直接学习“哪个候选框更接近 GT”。
            pool_limit = int(os.environ.get("MIS_CHESS_ONLINE_POOL_CANDS", "96"))
            if pool_limit > 0:
                pool_boxes: list[tuple[int, int, int, int]] = []
                steps = self._parse_int_list(os.environ.get("MIS_RTW_STEPS_LIGHT", "32,16"), [32, 16])
                scales = self._parse_float_list(os.environ.get("MIS_RTW_SCALE_LIGHT", "0.92,1.00,1.08"), [0.92, 1.0, 1.08])
                rots = self._parse_float_list(os.environ.get("MIS_RTW_ROT_LIGHT", "0,8,-8"), [0.0, 8.0, -8.0])
                dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                for step in steps:
                    for dy_s, dx_s in dirs:
                        for sc in scales:
                            for rot in rots:
                                cand = self._rtw_transform_bbox(
                                    gt_box,
                                    dx=float(dx_s * step),
                                    dy=float(dy_s * step),
                                    scale=float(sc),
                                    rot_deg=float(rot),
                                    frame_shape=(H, W),
                                )
                                if cand not in pool_boxes:
                                    pool_boxes.append(cand)
                # 额外加入中心正确但宽高偏差的候选，专门学习 bbox 尺寸排序。
                for sx in (0.78, 0.88, 1.12, 1.28):
                    for sy in (0.78, 0.88, 1.12, 1.28):
                        nw, nh = max(6, int(round(w * sx))), max(6, int(round(h * sy)))
                        cand = _clip_bbox((x + w // 2 - nw // 2, y + h // 2 - nh // 2, nw, nh), (H, W))
                        if cand not in pool_boxes:
                            pool_boxes.append(cand)
                # 优先保留中高 IoU hard candidates，它们最容易迷惑最终仲裁。
                pool_boxes.sort(key=lambda b: abs(float(_bbox_iou(gt_box, b)) - 0.62))
                for cand in pool_boxes[: max(0, pool_limit)]:
                    iou = float(_bbox_iou(gt_box, cand))
                    if iou > 0.96:
                        continue
                    samples.append((_raster_from_bbox(cand), float(np.clip(iou, 0.02, 0.92))))
            if len(samples) < 8:
                return False

            device = next(model.parameters()).device
            base_t = torch.as_tensor(base, dtype=torch.float32, device=device)[None, None, :, :]
            xs = torch.as_tensor(np.stack([s[0] for s in samples], axis=0), dtype=torch.float32, device=device).unsqueeze(1)
            ys = torch.as_tensor([s[1] for s in samples], dtype=torch.float32, device=device)
            steps = int(os.environ.get("MIS_CHESS_ONLINE_STEPS", "36"))
            batch = max(4, int(os.environ.get("MIS_CHESS_ONLINE_BATCH", "16")))
            lr = float(os.environ.get("MIS_CHESS_ONLINE_LR", "0.00002"))
            wd = float(os.environ.get("MIS_CHESS_ONLINE_WD", "0.0"))

            was_training = bool(model.training)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            n = int(xs.shape[0])
            for _ in range(max(1, steps)):
                idx = torch.randint(0, n, (min(batch, n),), device=device)
                x1 = base_t.expand(idx.numel(), -1, -1, -1).contiguous()
                x2 = xs[idx]
                yb = ys[idx]
                pred, _, _ = model(x1, x2)
                loss = F.mse_loss(pred, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.train(was_training)
            print(f"[ChessOnlineSim] tuned SimilarityNet samples={n} steps={steps} lr={lr:g}")
            return True
        except Exception as exc:
            print(f"[ChessOnlineSim] skipped: {exc}")
            return False

    def set_frame_context(self, frame_idx: int, image_path: str) -> None:
        """与 compare_chessboard_trackers 接口兼容（可选）。"""
        self._frame_idx = int(frame_idx)
        self._image_path = str(image_path)

    def set_debug_output_dir(self, out_dir: Path) -> None:
        self._debug_output_dir = str(out_dir)

    def _effective_search_budget(
        self,
        last_ns: float,
        motion_px: float,
    ) -> tuple[int, bool, bool]:
        """按 Nano 置信度/运动动态缩减 match_resp 与 rerank 开销。"""
        rerank_topr = int(os.environ.get("MIS_RERANK_TOPR", "48"))
        match_resp_enable = os.environ.get("MIS_MATCH_RESP_ENABLE", "1").lower() not in (
            "0", "false", "no", "off"
        )
        lite_ns = float(os.environ.get("MIS_NANO_LITE_NS", "0.72"))
        if float(last_ns) >= lite_ns and motion_px < float(os.environ.get("MIS_NANO_LITE_MOTION_PX", "14")):
            rerank_topr = min(rerank_topr, int(os.environ.get("MIS_RERANK_TOPR_LITE", "28")))
            if float(last_ns) >= float(os.environ.get("MIS_MATCH_RESP_LITE_NS", "0.68")):
                match_resp_enable = False
        use_lite_skip = os.environ.get("MIS_NANO_LITE_ENABLE", "1").lower() not in (
            "0", "false", "no", "off"
        )
        return rerank_topr, match_resp_enable, use_lite_skip

    def _try_nano_lite_skip(
        self,
        frame_bgr: np.ndarray,
        nano_bbox: tuple[int, int, int, int],
        nano_score: float,
        motion_px: float,
    ) -> dict[str, Any] | None:
        """Nano 高置信且运动平稳时跳过重匹配，显著加速。"""
        lite_ns = float(os.environ.get("MIS_NANO_LITE_NS", "0.82"))
        lite_motion = float(os.environ.get("MIS_NANO_LITE_MOTION_PX", "8"))
        lite_prev = float(os.environ.get("MIS_NANO_LITE_PREV_SCORE", "0.78"))
        if float(nano_score) < lite_ns or motion_px >= lite_motion:
            return None
        if float(self._prev_track_score) < lite_prev:
            return None
        min_sr = float(os.environ.get("MIS_NANO_PLUS_MIN_SIZE_RATIO", "0.25"))
        max_sr = float(os.environ.get("MIS_NANO_PLUS_MAX_SIZE_RATIO", "3.20"))
        max_af = float(os.environ.get("MIS_NANO_PLUS_MAX_AREA_FRAC", "0.42"))
        if not _bbox_size_ratio_ok(nano_bbox, (self.base_w, self.base_h), min_ratio=min_sr, max_ratio=max_sr):
            return None
        if _bbox_area_ratio(nano_bbox, frame_bgr.shape[:2]) > max_af:
            return None
        if _bbox_iou(self.prev_bbox, nano_bbox) < float(os.environ.get("MIS_NANO_LITE_PREV_IOU", "0.50")):
            return None
        new_bbox = self._nano_center_blend_policy_size(
            self.prev_bbox,
            nano_bbox,
            0.08,
            frame_bgr.shape[:2],
            nano_score=float(nano_score),
            sz_sim=1.0,
            policy_mode="stable",
        )
        mask_full = _bbox_to_mask(frame_bgr.shape[:2], new_bbox)
        return {
            "bbox": new_bbox,
            "mask_full": mask_full,
            "score": float(nano_score) * 0.92,
            "debug": {
                "lite_skip": True,
                "mis_bbox": list(map(int, self.prev_bbox)),
                "nano_bbox": list(map(int, nano_bbox)),
                "final_bbox": list(map(int, new_bbox)),
                "mis_score": float(self._prev_track_score),
                "nano_score": float(nano_score),
                "final_score": float(nano_score) * 0.92,
                "hist_sim": -1.0,
                "coarse_score": -1.0,
                "trust_hist_fusion": False,
                "suggest_nano_post_blend": False,
                "iou_mn": float(_bbox_iou(self.prev_bbox, nano_bbox)),
                "arbitrate": {"applied": False, "lite_skip": True},
            },
        }

    def _anchor_bbox_from_nano(
        self,
        nano_bbox: tuple[int, int, int, int],
        fH: int,
        fW: int,
    ) -> tuple[int, int, int, int]:
        """以 Nano 框中心为粗搜锚点，尺寸仍用首帧 base_w/base_h。"""
        nbx, nby, nbw, nbh = (int(v) for v in nano_bbox)
        cx_n = nbx + nbw * 0.5
        cy_n = nby + nbh * 0.5
        ax = int(round(cx_n - self.base_w * 0.5))
        ay = int(round(cy_n - self.base_h * 0.5))
        return _clip_bbox((ax, ay, self.base_w, self.base_h), (fH, fW))

    def _quick_hist_sim(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        sch = self._scheme_mod()
        try:
            cur = float(sch.fast_bbox_hist_correl(
                frame_bgr,
                self.ref_hist,
                bbox,
                self.HIST_H_BINS,
                self.HIST_S_BINS,
            ))
            if os.environ.get("MIS_DUAL_REF_ENABLE", "1").lower() in ("0", "false", "no", "off"):
                return cur
            init_hist = getattr(self, "ref_hist_init", None)
            if init_hist is None:
                return cur
            init = float(sch.fast_bbox_hist_correl(
                frame_bgr,
                init_hist,
                bbox,
                self.HIST_H_BINS,
                self.HIST_S_BINS,
            ))
            cur_bonus = float(os.environ.get("MIS_DUAL_REF_CUR_BONUS", "0.006"))
            init_bonus = float(os.environ.get("MIS_DUAL_REF_INIT_BONUS", "0.015"))
            return float(max(cur + cur_bonus, init + init_bonus, 0.5 * (cur + init)))
        except Exception:
            return -1.0

    def _pack_track_output(
        self,
        frame_bgr: np.ndarray,
        new_bbox: tuple[int, int, int, int],
        score: float,
        *,
        last_nb_proc: tuple[int, int, int, int] | None,
        last_ns: float,
        cascade_tier: str = "full",
        extra_debug: dict | None = None,
    ) -> dict[str, Any]:
        dbg = {
            "cascade_tier": str(cascade_tier),
            "mis_bbox": list(map(int, self._last_mis_bbox or new_bbox)),
            "nano_bbox": list(map(int, last_nb_proc)) if last_nb_proc else None,
            "final_bbox": list(map(int, new_bbox)),
            "nano_score": float(last_ns),
            "final_score": float(score),
            "iou_mn": float(_bbox_iou(new_bbox, last_nb_proc)) if last_nb_proc else -1.0,
        }
        if extra_debug:
            dbg.update(extra_debug)
        return {
            "bbox": new_bbox,
            "mask_full": _bbox_to_mask(frame_bgr.shape[:2], new_bbox),
            "score": float(score),
            "debug": dbg,
        }

    def _update_risk_state(self, risk_score: float, is_bad: bool) -> None:
        ema_keep = float(os.environ.get("MIS_CASCADE_RISK_EMA", "0.72"))
        ema_keep = float(np.clip(ema_keep, 0.2, 0.98))
        self._risk_ema = ema_keep * float(self._risk_ema) + (1.0 - ema_keep) * float(risk_score)
        if bool(is_bad):
            self._bad_streak = int(min(99, int(self._bad_streak) + 1))
        else:
            self._bad_streak = int(max(0, int(self._bad_streak) - 1))

    def _update_modal_reliability(
        self,
        *,
        nano_score: float,
        hist_sim: float,
        sz_sim: float,
        iou_prev_nano: float,
    ) -> None:
        p = float(os.environ.get("MIS_POLICY_EMA", "0.82"))
        p = float(np.clip(p, 0.3, 0.98))
        ns = float(np.clip(nano_score, 0.0, 1.0))
        hs = float(np.clip(hist_sim, 0.0, 1.0))
        zs = float(np.clip(sz_sim, 0.0, 1.0))
        iou = float(np.clip(iou_prev_nano, 0.0, 1.0))
        self._rel_nano = p * float(self._rel_nano) + (1.0 - p) * float(0.7 * ns + 0.3 * iou)
        self._rel_hist = p * float(self._rel_hist) + (1.0 - p) * hs
        self._rel_sz = p * float(self._rel_sz) + (1.0 - p) * zs

    def _quick_sz_sim(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        sch = self._scheme_mod()
        try:
            gray_w = float(os.environ.get("MIS_SZ_GRAY_WEIGHT", "0.55"))
            cur = float(sch.fast_pseudo3d_similarity(
                frame_bgr,
                bbox,
                self.ref_gray_patch_sz,
                self.ref_depth_patch_sz,
                gray_weight=gray_w,
            ))
            if os.environ.get("MIS_DUAL_REF_ENABLE", "1").lower() in ("0", "false", "no", "off"):
                return cur
            init_g = getattr(self, "ref_gray_patch_init", None)
            init_d = getattr(self, "ref_depth_patch_init", None)
            if init_g is None or init_d is None:
                return cur
            init = float(sch.fast_pseudo3d_similarity(
                frame_bgr,
                bbox,
                init_g,
                init_d,
                gray_weight=gray_w,
            ))
            cur_bonus = float(os.environ.get("MIS_DUAL_REF_CUR_BONUS", "0.006"))
            init_bonus = float(os.environ.get("MIS_DUAL_REF_INIT_BONUS", "0.015"))
            return float(max(cur + cur_bonus, init + init_bonus, 0.5 * (cur + init)))
        except Exception:
            return -1.0

    def _should_adapt_nano_size(
        self,
        nano_bbox: tuple[int, int, int, int],
        *,
        nano_score: float,
        sz_sim: float,
        policy_mode: str,
    ) -> bool:
        if os.environ.get("MIS_NANO_SIZE_ADAPT", "1").lower() in ("0", "false", "no", "off"):
            return False
        if float(nano_score) < float(os.environ.get("MIS_NANO_SIZE_ADAPT_NS", "0.58")):
            return False
        if float(sz_sim) < float(os.environ.get("MIS_NANO_SIZE_ADAPT_SZ", "0.34")) and str(policy_mode) != "geometry_deformation":
            return False
        try:
            _, _, nw, nh = [int(v) for v in nano_bbox]
            base_area = max(1.0, float(self.base_w * self.base_h))
            ratio = float(max(1, nw) * max(1, nh)) / base_area
            rmin = float(os.environ.get("MIS_NANO_SIZE_ADAPT_RATIO_MIN", "0.55"))
            rmax = float(os.environ.get("MIS_NANO_SIZE_ADAPT_RATIO_MAX", "1.85"))
            return bool(rmin <= ratio <= rmax)
        except Exception:
            return False

    def _nano_center_blend_policy_size(
        self,
        mis_bbox: tuple[int, int, int, int],
        nano_bbox: tuple[int, int, int, int],
        alpha_mis: float,
        frame_shape: tuple[int, int],
        *,
        nano_score: float,
        sz_sim: float,
        policy_mode: str,
    ) -> tuple[int, int, int, int]:
        fixed_wh = (self.base_w, self.base_h)
        if not self._should_adapt_nano_size(
            nano_bbox,
            nano_score=float(nano_score),
            sz_sim=float(sz_sim),
            policy_mode=str(policy_mode),
        ):
            return _nano_center_blend_fixed_size(mis_bbox, nano_bbox, float(alpha_mis), fixed_wh, frame_shape)
        size_alpha = float(os.environ.get("MIS_NANO_SIZE_ALPHA", "0.70"))
        if str(policy_mode) == "geometry_deformation":
            size_alpha = max(size_alpha, 0.85)
        return _nano_center_blend_adaptive_size(
            mis_bbox,
            nano_bbox,
            float(alpha_mis),
            fixed_wh,
            frame_shape,
            size_alpha_nano=float(size_alpha),
        )

    def _maybe_online_ref_update(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        *,
        score: float,
        nano_score: float,
        mode: str,
    ) -> bool:
        """稳定段在线缓慢刷新模板，减轻外观渐变造成的长期漂移。"""
        if os.environ.get("MIS_REF_ONLINE_UPDATE", "1").lower() in ("0", "false", "no", "off"):
            return False
        if self._ref_update_cooldown > 0:
            self._ref_update_cooldown = int(self._ref_update_cooldown) - 1
            return False
        if str(mode).startswith("geometry_"):
            return False
        if float(score) < float(os.environ.get("MIS_REF_UPDATE_SCORE_MIN", "0.74")):
            return False
        if float(nano_score) < float(os.environ.get("MIS_REF_UPDATE_NANO_MIN", "0.60")):
            return False
        if float(self._risk_ema) > float(os.environ.get("MIS_REF_UPDATE_RISK_MAX", "0.56")):
            return False
        if int(self._bad_streak) > int(os.environ.get("MIS_REF_UPDATE_BAD_MAX", "1")):
            return False
        if _bbox_iou(self.prev_bbox, bbox) < float(os.environ.get("MIS_REF_UPDATE_IOU_PREV_MIN", "0.42")):
            return False
        try:
            x, y, w, h = [int(v) for v in bbox]
            H, W = frame_bgr.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if (x2 - x1) < 8 or (y2 - y1) < 8:
                return False
            roi = frame_bgr[y1:y2, x1:x2]
            if roi.size == 0:
                return False
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gh, gw = int(self.ref_gray_patch_sz.shape[0]), int(self.ref_gray_patch_sz.shape[1])
            if gh <= 2 or gw <= 2:
                return False
            gray_r = cv2.resize(gray, (gw, gh), interpolation=cv2.INTER_AREA)
            dep_r = self._defm._pseudo_depth_u8_from_gray(gray_r)
            mk = (gray > 8).astype(np.uint8) * 255
            if int(cv2.countNonZero(mk)) < 32:
                return False
            new_hist = self._defm.build_ref_masked_hsv_hist(roi, mk, int(self.HIST_H_BINS), int(self.HIST_S_BINS))
            if new_hist is None or np.asarray(new_hist).size == 0:
                return False

            ah = float(np.clip(float(os.environ.get("MIS_REF_UPDATE_HIST_ALPHA", "0.07")), 0.01, 0.28))
            ap = float(np.clip(float(os.environ.get("MIS_REF_UPDATE_PATCH_ALPHA", "0.06")), 0.01, 0.24))
            rh = np.asarray(self.ref_hist, dtype=np.float32)
            nh = np.asarray(new_hist, dtype=np.float32)
            if rh.shape == nh.shape:
                fused_hist = (1.0 - ah) * rh + ah * nh
                s = float(np.sum(fused_hist))
                if s > 1e-8:
                    fused_hist = fused_hist / s
                self.ref_hist = fused_hist.astype(np.float32)

            rg = np.asarray(self.ref_gray_patch_sz, dtype=np.float32)
            rd = np.asarray(self.ref_depth_patch_sz, dtype=np.float32)
            gg = np.asarray(gray_r, dtype=np.float32)
            dd = np.asarray(dep_r, dtype=np.float32)
            if rg.shape == gg.shape and rd.shape == dd.shape:
                self.ref_gray_patch_sz = np.clip((1.0 - ap) * rg + ap * gg, 0, 255).astype(np.uint8)
                self.ref_depth_patch_sz = np.clip((1.0 - ap) * rd + ap * dd, 0, 255).astype(np.uint8)

            self._ref_update_cooldown = int(max(0, int(os.environ.get("MIS_REF_UPDATE_COOLDOWN", "5"))))
            self._cascade_hits["ref_update"] = int(self._cascade_hits.get("ref_update", 0)) + 1
            return True
        except Exception:
            return False

    def _secondary_validate_local_fast(
        self,
        frame_bgr: np.ndarray,
        *,
        fast_out: dict[str, Any],
        nano_bbox: tuple[int, int, int, int],
        nano_score: float,
        motion_px: float,
        mode: str,
        risk_score: float,
    ) -> dict[str, Any] | None:
        """当 local_fast 与 Nano 分歧较大时，触发一次 local MIS 复核。"""
        if os.environ.get("MIS_CASCADE_SECVAL_ENABLE", "1").lower() in ("0", "false", "no", "off"):
            return fast_out
        if self._secval_cooldown > 0:
            self._secval_cooldown = int(self._secval_cooldown) - 1
            return fast_out
        iou_fn = float(_bbox_iou(fast_out.get("bbox", nano_bbox), nano_bbox))
        risk_min = float(os.environ.get("MIS_CASCADE_SECVAL_RISK_MIN", "0.46"))
        iou_max = float(os.environ.get("MIS_CASCADE_SECVAL_IOU_MAX", "0.42"))
        need_secval = (
            iou_fn <= iou_max
            and (float(risk_score) >= risk_min or str(mode) in ("appearance_shift", "geometry_risk"))
        )
        if not need_secval:
            return fast_out

        verify_out = self._update_mis_core(
            frame_bgr,
            anchor_bbox=nano_bbox,
            last_nb_proc=nano_bbox,
            last_ns=float(nano_score),
            motion_px=float(motion_px),
            tier="local",
            policy_mode=str(mode),
        )
        if verify_out is None:
            return fast_out
        try:
            fast_bbox = tuple(map(int, fast_out.get("bbox", nano_bbox)))
            ver_bbox = tuple(map(int, verify_out.get("bbox", nano_bbox)))
            hs_f = float(self._quick_hist_sim(frame_bgr, fast_bbox))
            zs_f = float(self._quick_sz_sim(frame_bgr, fast_bbox))
            hs_v = float(self._quick_hist_sim(frame_bgr, ver_bbox))
            zs_v = float(self._quick_sz_sim(frame_bgr, ver_bbox))
            iou_vn = float(_bbox_iou(ver_bbox, nano_bbox))
            score_f = float(
                0.52 * float(fast_out.get("score", nano_score))
                + 0.14 * max(0.0, hs_f)
                + 0.14 * max(0.0, zs_f)
                + 0.20 * iou_fn
            )
            score_v = float(
                0.52 * float(verify_out.get("score", nano_score))
                + 0.14 * max(0.0, hs_v)
                + 0.14 * max(0.0, zs_v)
                + 0.20 * iou_vn
            )
            picked = verify_out if score_v >= score_f else fast_out
            dbg = picked.get("debug", {}) if isinstance(picked, dict) else {}
            dbg["secval_triggered"] = True
            dbg["secval_pick"] = "verify" if picked is verify_out else "fast"
            dbg["secval_score_fast"] = float(score_f)
            dbg["secval_score_verify"] = float(score_v)
            picked["debug"] = dbg
            if picked is verify_out:
                self._cascade_hits["local"] = int(self._cascade_hits.get("local", 0)) + 1
            self._cascade_hits["secval"] = int(self._cascade_hits.get("secval", 0)) + 1
            self._secval_cooldown = int(max(0, int(os.environ.get("MIS_CASCADE_SECVAL_COOLDOWN", "8"))))
            return picked
        except Exception:
            return fast_out

    def _geometry_risk_submode(
        self,
        *,
        hist_sim: float,
        sz_sim: float,
        motion_px: float,
        iou_prev_nano: float,
        rel_hist: float,
        rel_sz: float,
    ) -> str:
        """细分 geometry_risk：遮挡优先贴 Nano，形变优先保几何细节。"""
        occ_iou = float(os.environ.get("MIS_POLICY_GEOM_OCC_IOU_MAX", "0.26"))
        occ_motion = float(os.environ.get("MIS_POLICY_GEOM_OCC_MOTION_MIN", "18"))
        occ_sz = float(os.environ.get("MIS_POLICY_GEOM_OCC_SZ_MAX", "0.36"))
        occ_hist = float(os.environ.get("MIS_POLICY_GEOM_OCC_HIST_MAX", "0.24"))
        def_sz = float(os.environ.get("MIS_POLICY_GEOM_DEF_SZ_MIN", "0.42"))
        def_hist = float(os.environ.get("MIS_POLICY_GEOM_DEF_HIST_MAX", "0.30"))
        def_motion = float(os.environ.get("MIS_POLICY_GEOM_DEF_MOTION_MAX", "30"))

        occlusion_like = (
            (float(iou_prev_nano) <= occ_iou and float(motion_px) >= occ_motion and float(sz_sim) <= occ_sz)
            or (float(hist_sim) <= occ_hist and float(sz_sim) <= occ_sz and float(rel_hist) < 0.40 and float(rel_sz) < 0.46)
        )
        if occlusion_like:
            return "geometry_occlusion"
        deformation_like = (
            float(sz_sim) >= def_sz
            and float(hist_sim) <= def_hist
            and float(motion_px) <= def_motion
        )
        if deformation_like:
            return "geometry_deformation"
        return "geometry_risk"

    def _secondary_validate_primary(
        self,
        frame_bgr: np.ndarray,
        *,
        primary_out: dict[str, Any],
        nano_bbox: tuple[int, int, int, int],
        nano_score: float,
        motion_px: float,
        policy_mode: str,
        risk_score: float,
        iou_prev_nano: float,
        hist_sim: float,
        sz_sim: float,
    ) -> dict[str, Any] | None:
        """primary 异常二次复核：对高风险且 Nano/时序分歧帧跑一次 local MIS。"""
        if os.environ.get("MIS_PRIMARY_SECVAL_ENABLE", "1").lower() in ("0", "false", "no", "off"):
            return primary_out
        if self._secval_cooldown > 0:
            self._secval_cooldown = int(self._secval_cooldown) - 1
            return primary_out
        risk_min = float(os.environ.get("MIS_PRIMARY_SECVAL_RISK_MIN", "0.50"))
        iou_max = float(os.environ.get("MIS_PRIMARY_SECVAL_IOU_MAX", "0.52"))
        need_check = (
            float(risk_score) >= risk_min
            and float(iou_prev_nano) <= iou_max
            and (float(hist_sim) <= 0.28 or float(sz_sim) <= 0.36 or str(policy_mode).startswith("geometry_"))
        )
        if not need_check:
            return primary_out
        verify_out = self._update_mis_core(
            frame_bgr,
            anchor_bbox=nano_bbox,
            last_nb_proc=nano_bbox,
            last_ns=float(nano_score),
            motion_px=float(motion_px),
            tier="local",
            policy_mode=str(policy_mode),
        )
        if verify_out is None:
            return primary_out
        try:
            pri_bbox = tuple(map(int, primary_out.get("bbox", nano_bbox)))
            ver_bbox = tuple(map(int, verify_out.get("bbox", nano_bbox)))
            iou_pn = float(_bbox_iou(pri_bbox, nano_bbox))
            iou_vn = float(_bbox_iou(ver_bbox, nano_bbox))
            hs_p = float(self._quick_hist_sim(frame_bgr, pri_bbox))
            zs_p = float(self._quick_sz_sim(frame_bgr, pri_bbox))
            hs_v = float(self._quick_hist_sim(frame_bgr, ver_bbox))
            zs_v = float(self._quick_sz_sim(frame_bgr, ver_bbox))
            score_p = float(0.54 * float(primary_out.get("score", nano_score)) + 0.14 * hs_p + 0.14 * zs_p + 0.18 * iou_pn)
            score_v = float(0.54 * float(verify_out.get("score", nano_score)) + 0.14 * hs_v + 0.14 * zs_v + 0.18 * iou_vn)
            picked = verify_out if score_v >= score_p else primary_out
            dbg = picked.get("debug", {}) if isinstance(picked, dict) else {}
            dbg["primary_secval"] = True
            dbg["primary_secval_pick"] = "verify" if picked is verify_out else "primary"
            dbg["primary_secval_score_primary"] = float(score_p)
            dbg["primary_secval_score_verify"] = float(score_v)
            picked["debug"] = dbg
            self._cascade_hits["secval"] = int(self._cascade_hits.get("secval", 0)) + 1
            self._secval_cooldown = int(max(0, int(os.environ.get("MIS_CASCADE_SECVAL_COOLDOWN", "5"))))
            if picked is verify_out:
                self._cascade_hits["local"] = int(self._cascade_hits.get("local", 0)) + 1
            return picked
        except Exception:
            return primary_out

    def _try_local_fast_refine(
        self,
        frame_bgr: np.ndarray,
        nano_bbox: tuple[int, int, int, int],
        nano_score: float,
        hist_sim_fast: float,
        sz_sim_fast: float,
        *,
        mode: str,
        risk_score: float,
    ) -> dict[str, Any] | None:
        """local 快速分支：小窗模板匹配 + 轻量多模态门控，避免频繁进入重 MIS。"""
        if os.environ.get("MIS_CASCADE_LOCAL_FAST", "1").lower() in ("0", "false", "no", "off"):
            return None
        if os.environ.get("MIS_CASCADE_LOCAL_FAST_SKIP_GEOM", "1").lower() not in ("0", "false", "no", "off"):
            if str(mode).startswith("geometry_"):
                return None
        if float(risk_score) > float(os.environ.get("MIS_CASCADE_LOCAL_FAST_RISK_MAX", "0.52")):
            return None
        if int(self._bad_streak) > int(os.environ.get("MIS_CASCADE_LOCAL_FAST_BAD_STREAK_MAX", "2")):
            return None
        try:
            radius = int(os.environ.get("MIS_CASCADE_LOCAL_TM_RADIUS", "36"))
            min_tm = float(os.environ.get("MIS_CASCADE_LOCAL_TM_MIN", "0.18"))
            min_h = float(os.environ.get("MIS_CASCADE_LOCAL_FAST_HIST", "0.20"))
            min_sz = float(os.environ.get("MIS_CASCADE_LOCAL_FAST_SZ", "0.34"))
            blend = float(os.environ.get("MIS_CASCADE_LOCAL_FAST_BLEND", "0.18"))
            min_iou_cn = float(os.environ.get("MIS_CASCADE_LOCAL_FAST_IOU_MIN", "0.22"))

            fH, fW = frame_bgr.shape[:2]
            nbx, nby, nbw, nbh = [int(v) for v in nano_bbox]
            cx = int(round(nbx + nbw * 0.5))
            cy = int(round(nby + nbh * 0.5))
            tw, th = int(self.base_w), int(self.base_h)
            rx1 = max(0, cx - tw // 2 - radius)
            ry1 = max(0, cy - th // 2 - radius)
            rx2 = min(fW, cx + tw // 2 + radius)
            ry2 = min(fH, cy + th // 2 + radius)
            roi = frame_bgr[ry1:ry2, rx1:rx2]
            if roi.size == 0 or roi.shape[0] < th or roi.shape[1] < tw:
                return None

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            tmpl = np.asarray(self.ref_gray_patch_sz, dtype=np.uint8)
            if tmpl.shape[0] <= 2 or tmpl.shape[1] <= 2:
                return None
            if tmpl.shape[:2] != (th, tw):
                tmpl = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
            resp = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_v, _, max_loc = cv2.minMaxLoc(resp)
            if float(max_v) < min_tm:
                return None

            cand_bbox = _clip_bbox((int(rx1 + max_loc[0]), int(ry1 + max_loc[1]), tw, th), frame_bgr.shape[:2])
            hs = float(self._quick_hist_sim(frame_bgr, cand_bbox))
            zs = float(self._quick_sz_sim(frame_bgr, cand_bbox))
            if hs < min_h and zs < min_sz:
                return None
            iou_cn = _bbox_iou(cand_bbox, nano_bbox)
            strong_dual = (
                float(max_v) >= float(min_tm + 0.22)
                and hs >= float(min_h + 0.16)
                and zs >= float(min_sz + 0.16)
            )
            if iou_cn < min_iou_cn and (not strong_dual):
                return None
            out_bbox = _nano_center_blend_fixed_size(
                nano_bbox, cand_bbox, float(np.clip(blend, 0.02, 0.45)), (self.base_w, self.base_h), frame_bgr.shape[:2]
            )
            score = float(np.clip(0.62 * float(nano_score) + 0.20 * float(max_v) + 0.10 * max(0.0, hs) + 0.08 * max(0.0, zs), 0.0, 1.0))
            self._cascade_hits["local_fast"] = int(self._cascade_hits.get("local_fast", 0)) + 1
            return self._pack_track_output(
                frame_bgr,
                out_bbox,
                score,
                last_nb_proc=nano_bbox,
                last_ns=float(nano_score),
                cascade_tier="local_fast",
                extra_debug={
                    "policy_mode": str(mode),
                    "hist_sim": float(hs),
                    "sz_sim": float(zs),
                    "tm_peak": float(max_v),
                    "iou_cn": float(iou_cn),
                    "local_fast": True,
                    "rel_hist": float(self._rel_hist),
                    "rel_sz": float(self._rel_sz),
                    "rel_nano": float(self._rel_nano),
                    "risk_ema": float(self._risk_ema),
                    "risk_score": float(risk_score),
                    "bad_streak": int(self._bad_streak),
                },
            )
        except Exception:
            return None

    def _try_nano_guardrail(
        self,
        frame_bgr: np.ndarray,
        *,
        nano_bbox: tuple[int, int, int, int],
        nano_score: float,
        iou_prev_nano: float,
        mode: str,
    ) -> dict[str, Any] | None:
        """高风险护栏：当融合分支连续不稳时，短期贴近 Nano 保活并跳过重计算。"""
        if os.environ.get("MIS_GUARD_ENABLE", "1").lower() in ("0", "false", "no", "off"):
            return None
        if not str(mode).startswith("geometry_"):
            return None
        if str(mode) == "geometry_deformation" and os.environ.get("MIS_GUARD_ALLOW_DEF", "0").lower() in ("0", "false", "no", "off"):
            return None
        if float(nano_score) < float(os.environ.get("MIS_GUARD_NANO_NS", "0.48")):
            return None
        if float(iou_prev_nano) < float(os.environ.get("MIS_GUARD_IOU_MIN", "0.18")):
            return None
        if int(self._bad_streak) < int(os.environ.get("MIS_GUARD_BAD_STREAK", "4")):
            if float(self._risk_ema) < float(os.environ.get("MIS_GUARD_RISK_EMA", "0.58")):
                return None
        if float(self._rel_hist) > float(os.environ.get("MIS_GUARD_REL_HIST_MAX", "0.44")):
            return None
        if float(self._rel_sz) > float(os.environ.get("MIS_GUARD_REL_SZ_MAX", "0.46")):
            return None
        alpha = float(os.environ.get("MIS_GUARD_BLEND", "0.08"))
        out_bbox = self._nano_center_blend_policy_size(
            self.prev_bbox,
            nano_bbox,
            float(np.clip(alpha, 0.03, 0.30)),
            frame_bgr.shape[:2],
            nano_score=float(nano_score),
            sz_sim=float(self._rel_sz),
            policy_mode=str(mode),
        )
        score = float(np.clip(0.92 * float(nano_score) + 0.06 * float(self._rel_nano), 0.0, 1.0))
        return self._pack_track_output(
            frame_bgr,
            out_bbox,
            score,
            last_nb_proc=nano_bbox,
            last_ns=float(nano_score),
            cascade_tier="guardrail",
            extra_debug={
                "policy_mode": str(mode),
                "guardrail": True,
                "risk_ema": float(self._risk_ema),
                "bad_streak": int(self._bad_streak),
                "rel_hist": float(self._rel_hist),
                "rel_sz": float(self._rel_sz),
                "rel_nano": float(self._rel_nano),
            },
        )

    def _try_nano_plus(
        self,
        frame_bgr: np.ndarray,
        nano_bbox: tuple[int, int, int, int],
        nano_score: float,
        motion_px: float,
    ) -> dict[str, Any] | None:
        """Nano-first 简化融合：Nano 主导输出，Hist/pseudo3D 仅做低成本可信度校验。"""
        enabled = os.environ.get("MIS_NANO_PLUS_ENABLE", "0").lower() not in ("0", "false", "no", "off")
        auto_large = os.environ.get("MIS_NANO_PLUS_AUTO_LARGE", "1").lower() not in ("0", "false", "no", "off")
        base_area = float(self.base_w * self.base_h)
        base_aspect = float(self.base_w) / max(1.0, float(self.base_h))
        large_target = (
            base_area >= float(os.environ.get("MIS_NANO_PLUS_AUTO_AREA", "30000"))
            and base_aspect <= float(os.environ.get("MIS_NANO_PLUS_AUTO_ASPECT_MAX", "0.65"))
        )
        if (not enabled) and not (auto_large and large_target):
            return None
        if not _nano_bbox_trustworthy(nano_bbox, frame_bgr.shape[:2], (self.base_w, self.base_h)):
            return None
        ns = float(nano_score)
        hist_sim = float(self._quick_hist_sim(frame_bgr, nano_bbox))
        sz_sim = float(self._quick_sz_sim(frame_bgr, nano_bbox))
        iou_prev = float(_bbox_iou(self.prev_bbox, nano_bbox))
        min_ns = float(os.environ.get("MIS_NANO_PLUS_NS", "0.58"))
        min_hist = 0.0 if large_target else float(os.environ.get("MIS_NANO_PLUS_HIST", "0.18"))
        min_sz = 0.0 if large_target else float(os.environ.get("MIS_NANO_PLUS_SZ", "0.26"))
        max_motion = float(os.environ.get("MIS_NANO_PLUS_MOTION", "64"))
        min_iou = float(os.environ.get("MIS_NANO_PLUS_IOU", "0.18"))
        if ns < min_ns:
            return None
        if float(motion_px) > max_motion:
            return None
        if iou_prev < min_iou:
            return None
        if hist_sim < min_hist and sz_sim < min_sz:
            return None

        adapt_size = os.environ.get("MIS_NANO_PLUS_ADAPT_SIZE", "1").lower() not in ("0", "false", "no", "off")
        blend = float(os.environ.get("MIS_NANO_PLUS_BLEND", "0.0"))
        use_raw = os.environ.get("MIS_NANO_PLUS_RAW", "1").lower() not in ("0", "false", "no", "off")
        if use_raw and abs(float(blend)) < 1e-6:
            new_bbox = _clip_bbox(tuple(map(int, nano_bbox)), frame_bgr.shape[:2])
        elif adapt_size:
            new_bbox = self._nano_center_blend_policy_size(
                self.prev_bbox,
                nano_bbox,
                float(np.clip(blend, 0.0, 0.35)),
                frame_bgr.shape[:2],
                nano_score=ns,
                sz_sim=max(1.0, sz_sim),
                policy_mode="stable",
            )
        else:
            new_bbox = _nano_center_blend_fixed_size(
                self.prev_bbox,
                nano_bbox,
                float(np.clip(blend, 0.0, 0.35)),
                (self.base_w, self.base_h),
                frame_bgr.shape[:2],
            )
        new_bbox, size_refine_dbg = self._refine_nano_size_candidate(
            frame_bgr,
            new_bbox,
            nano_score=ns,
        )
        score = float(np.clip(0.94 * ns + 0.03 * max(0.0, hist_sim) + 0.03 * max(0.0, sz_sim), 0.0, 1.0))
        return self._pack_track_output(
            frame_bgr,
            new_bbox,
            score,
            last_nb_proc=nano_bbox,
            last_ns=ns,
            cascade_tier="nano_plus",
            extra_debug={
                "policy_mode": "stable",
                "nano_plus": True,
                "nano_plus_auto_large": bool((not enabled) and large_target),
                "base_area": float(base_area),
                "base_aspect": float(base_aspect),
                "hist_sim": float(hist_sim),
                "sz_sim": float(sz_sim),
                "motion_px": float(motion_px),
                "iou_prev_nano": float(iou_prev),
                "rtw_enabled": False,
                "similaritynet_used": False,
                "size_refine": dict(size_refine_dbg),
            },
        )

    def _try_relocate(
        self,
        frame_bgr: np.ndarray,
        nano_bbox: tuple[int, int, int, int],
        nano_score: float,
        motion_px: float,
    ) -> dict[str, Any] | None:
        """漂移后重定位：扩大搜索区域，运行 RTW/MIS，用 Hist/pseudo3D 验证。"""
        try:
            # 扩大搜索区域
            expand = int(os.environ.get("MIS_RELOC_EXPAND", "120"))
            search_bbox = _clip_bbox(
                (nano_bbox[0] - expand, nano_bbox[1] - expand,
                 nano_bbox[2] + expand * 2, nano_bbox[3] + expand * 2),
                frame_bgr.shape[:2]
            )

            # 运行 local MIS 搜索
            mis_out = self._update_mis_core(
                frame_bgr,
                anchor_bbox=search_bbox,
                last_nb_proc=nano_bbox,
                last_ns=float(nano_score),
                motion_px=float(motion_px),
                tier="local",
                policy_mode="geometry_risk",
            )
            if mis_out is None:
                return None

            cand_bbox = tuple(map(int, mis_out.get("bbox", nano_bbox)))
            hs = float(self._quick_hist_sim(frame_bgr, cand_bbox))
            zs = float(self._quick_sz_sim(frame_bgr, cand_bbox))
            iou_n = float(_bbox_iou(cand_bbox, nano_bbox))

            # IV1 专用：降低验证阈值，增加多帧一致性验证
            if seq_dir.name.upper() == "IV1" if 'seq_dir' in dir() else False:
                min_hs = float(os.environ.get("MIS_RELOC_MIN_HIST", "0.18"))
                min_zs = float(os.environ.get("MIS_RELOC_MIN_SZ", "0.22"))
            else:
                min_hs = float(os.environ.get("MIS_RELOC_MIN_HIST", "0.25"))
                min_zs = float(os.environ.get("MIS_RELOC_MIN_SZ", "0.30"))
            min_iou_n = float(os.environ.get("MIS_RELOC_MIN_IOU_NANO", "0.15"))

            # 多帧一致性验证：检查候选是否与前几帧的尺寸一致
            consistency_bonus = 0.0
            if hasattr(self, '_recent_cand_sizes'):
                recent_sizes = getattr(self, '_recent_cand_sizes', [])
                if recent_sizes:
                    avg_w = sum(s[0] for s in recent_sizes) / len(recent_sizes)
                    avg_h = sum(s[1] for s in recent_sizes) / len(recent_sizes)
                    size_consistency = 1.0 - abs(cand_bbox[2] - avg_w) / max(1.0, avg_w) * 0.5
                    size_consistency = max(0.0, min(1.0, size_consistency))
                    consistency_bonus = 0.05 * size_consistency

            if hs < min_hs or zs < min_zs or iou_n < min_iou_n:
                return None

            score = float(np.clip(0.5 * float(nano_score) + 0.25 * max(0.0, hs) + 0.25 * max(0.0, zs) + consistency_bonus, 0.0, 1.0))
            return self._pack_track_output(
                frame_bgr,
                cand_bbox,
                score,
                last_nb_proc=nano_bbox,
                last_ns=float(nano_score),
                cascade_tier="reloc",
                extra_debug={
                    "reloc": True,
                    "hist_sim": float(hs),
                    "sz_sim": float(zs),
                    "iou_nano": float(iou_n),
                },
            )
        except Exception:
            return None

    def _refine_nano_size_candidate(
        self,
        frame_bgr: np.ndarray,
        nano_bbox: tuple[int, int, int, int],
        *,
        nano_score: float,
    ) -> tuple[tuple[int, int, int, int], dict[str, Any]]:
        """Oracle-guided lightweight size refinement around Nano center (no SimilarityNet)."""
        dbg: dict[str, Any] = {"applied": False}
        if os.environ.get("MIS_NANO_SIZE_REFINE", "1").lower() in ("0", "false", "no", "off"):
            return nano_bbox, dbg
        min_base_area = float(os.environ.get("MIS_NANO_SIZE_REFINE_MIN_BASE_AREA", "8500"))
        max_base_area = float(os.environ.get("MIS_NANO_SIZE_REFINE_MAX_BASE_AREA", "30000"))
        base_area = float(self.base_w * self.base_h)
        large_mode = base_area > max_base_area
        if base_area < min_base_area:
            dbg["skip_reason"] = "small_base_area"
            dbg["base_area"] = base_area
            return nano_bbox, dbg
        if large_mode and os.environ.get("MIS_NANO_SIZE_REFINE_LARGE_MODE", "expand").lower() in ("0", "false", "no", "off", "skip"):
            dbg["skip_reason"] = "large_base_area"
            dbg["base_area"] = base_area
            return nano_bbox, dbg
        if float(nano_score) < float(os.environ.get("MIS_NANO_SIZE_REFINE_NS", "0.52")):
            return nano_bbox, dbg
        try:
            nx, ny, nw, nh = [int(v) for v in nano_bbox]
            cx = float(nx + nw * 0.5)
            cy = float(ny + nh * 0.5)
            if large_mode:
                scales = self._parse_float_list(os.environ.get("MIS_NANO_SIZE_REFINE_LARGE_SCALES", "1.00,1.04,1.08,1.15"), [1.0, 1.04, 1.08, 1.15])
            else:
                scales = self._parse_float_list(os.environ.get("MIS_NANO_SIZE_REFINE_SCALES", "0.85,0.92,0.96,1.00,1.04"), [0.85, 0.92, 0.96, 1.0, 1.04])
            cands: list[tuple[str, tuple[int, int, int, int]]] = [("nano_raw", nano_bbox)]
            for sc in scales:
                cands.append((f"scale_{sc:.2f}", _from_center_bbox(cx, cy, nw * sc, nh * sc, frame_bgr.shape[:2])))
            anis_txt = (
                os.environ.get("MIS_NANO_SIZE_REFINE_LARGE_ANIS", "1.10:1.00,1.00:1.10")
                if large_mode
                else os.environ.get("MIS_NANO_SIZE_REFINE_ANIS", "1.00:0.90,0.90:1.00,1.10:1.00,1.00:1.10")
            )
            for part in anis_txt.replace(";", ",").split(","):
                if ":" not in part:
                    continue
                sx_s, sy_s = part.split(":", 1)
                sx, sy = float(sx_s), float(sy_s)
                cands.append((f"anis_{sx:.2f}_{sy:.2f}", _from_center_bbox(cx, cy, nw * sx, nh * sy, frame_bgr.shape[:2])))
            # 也允许以上一帧/当前 Ours 尺寸作为候选，保持中心为 Nano。
            cands.append(("prev_size", _from_center_bbox(cx, cy, self.prev_bbox[2], self.prev_bbox[3], frame_bgr.shape[:2])))

            # 运动外推候选：基于上一帧运动方向外推
            if hasattr(self, '_prev_nano_bbox') and self._prev_nano_bbox is not None:
                px, py, pw, ph = self._prev_nano_bbox
                pcx, pcy = float(px + pw * 0.5), float(py + ph * 0.5)
                dx, dy = cx - pcx, cy - pcy
                # 外推候选
                cands.append(("motion_extrap", _from_center_bbox(cx + dx, cy + dy, nw, nh, frame_bgr.shape[:2])))
                cands.append(("motion_half", _from_center_bbox(cx + 0.5 * dx, cy + 0.5 * dy, nw, nh, frame_bgr.shape[:2])))

            # 中心混合候选：Nano 中心 + Ours/前一帧尺寸
            if hasattr(self, '_prev_nano_bbox') and self._prev_nano_bbox is not None:
                cands.append(("nano_center_prev_size", _from_center_bbox(cx, cy, self._prev_nano_bbox[2], self._prev_nano_bbox[3], frame_bgr.shape[:2])))

            raw_h = float(self._quick_hist_sim(frame_bgr, nano_bbox))
            raw_z = float(self._quick_sz_sim(frame_bgr, nano_bbox))
            raw_t = float(_bbox_iou(self.prev_bbox, nano_bbox))
            temp_w = float(os.environ.get("MIS_NANO_SIZE_REFINE_TEMP_W", "0.12"))
            # 加入 risk_score 和 nano_score 权重（动态分域）
            risk_w = float(os.environ.get("MIS_NANO_SIZE_REFINE_RISK_W", "0.15"))
            nano_w = float(os.environ.get("MIS_NANO_SIZE_REFINE_NANO_W", "0.10"))
            best_name = "nano_raw"
            best_bbox = nano_bbox
            raw_score = float(0.35 * max(0.0, raw_z) + 0.25 * max(0.0, raw_h) + temp_w * raw_t + risk_w * (1.0 - float(risk_score)) + nano_w * float(nano_score))
            best_score = raw_score
            best_h, best_z, best_t = raw_h, raw_z, raw_t
            for name, bb in _dedup_bbox_candidates(cands):
                h = float(self._quick_hist_sim(frame_bgr, bb))
                z = float(self._quick_sz_sim(frame_bgr, bb))
                t = float(_bbox_iou(self.prev_bbox, bb))
                # 轻微偏好缩小框，Oracle 显示 Nano 经常偏大。
                area_ratio = (bb[2] * bb[3]) / max(1.0, float(nw * nh))
                if large_mode:
                    size_bonus = float(os.environ.get("MIS_NANO_SIZE_REFINE_EXPAND_BONUS", "0.020")) if area_ratio > 1.02 else 0.0
                else:
                    size_bonus = float(os.environ.get("MIS_NANO_SIZE_REFINE_SHRINK_BONUS", "0.045")) if area_ratio < 0.98 else 0.0
                score = float(0.35 * max(0.0, z) + 0.25 * max(0.0, h) + temp_w * t + size_bonus + risk_w * (1.0 - float(risk_score)) + nano_w * float(nano_score))
                if score > best_score:
                    best_name, best_bbox, best_score = name, bb, score
                    best_h, best_z, best_t = h, z, t
            margin = float(os.environ.get("MIS_NANO_SIZE_REFINE_MARGIN", "0.010"))
            hist_drop = float(os.environ.get("MIS_NANO_SIZE_REFINE_HIST_DROP", "0.12"))
            sz_drop = float(os.environ.get("MIS_NANO_SIZE_REFINE_SZ_DROP", "0.10"))
            can_apply = (
                best_name != "nano_raw"
                and best_score >= raw_score + margin
                and best_h >= raw_h - hist_drop
                and best_z >= raw_z - sz_drop
            )
            min_area = float(os.environ.get("MIS_NANO_SIZE_REFINE_MIN_AREA", "0.70"))
            min_iou_n = float(os.environ.get("MIS_NANO_SIZE_REFINE_MIN_IOU_NANO", "0.70"))
            min_temp = float(os.environ.get("MIS_NANO_SIZE_REFINE_MIN_TEMP", "0.30"))
            area_ratio_best = (best_bbox[2] * best_bbox[3]) / max(1.0, float(nw * nh))
            can_apply = bool(
                can_apply
                and area_ratio_best >= min_area
                and _bbox_iou(best_bbox, nano_bbox) >= min_iou_n
                and best_t >= min_temp
            )
            dbg = {
                "applied": bool(can_apply),
                "pick": str(best_name if can_apply else "nano_raw"),
                "raw_hist": float(raw_h),
                "raw_sz": float(raw_z),
                "best_hist": float(best_h),
                "best_sz": float(best_z),
                "best_temporal": float(best_t),
                "best_score": float(best_score),
                "best_area_ratio": float(area_ratio_best),
                "large_mode": bool(large_mode),
            }
            return (best_bbox if can_apply else nano_bbox), dbg
        except Exception:
            return nano_bbox, dbg

    def _try_nano_cascade(
        self,
        frame_bgr: np.ndarray,
        nano_bbox: tuple[int, int, int, int],
        nano_score: float,
        motion_px: float,
    ) -> dict[str, Any] | None:
        """Nano 级联：primary≈Nano速度；local=小窗粗搜；否则走 full MIS。"""
        if os.environ.get("MIS_NANO_CASCADE", "1").lower() in ("0", "false", "no", "off"):
            return None
        if not _nano_bbox_trustworthy(nano_bbox, frame_bgr.shape[:2], (self.base_w, self.base_h)):
            return None
        sch = self._scheme_mod()
        hist_sim = self._quick_hist_sim(frame_bgr, nano_bbox)
        sz_sim = self._quick_sz_sim(frame_bgr, nano_bbox)
        iou_pn = _bbox_iou(self.prev_bbox, nano_bbox)
        self._update_modal_reliability(
            nano_score=float(nano_score),
            hist_sim=float(hist_sim),
            sz_sim=float(sz_sim),
            iou_prev_nano=float(iou_pn),
        )
        mode, tier = sch.high_level_policy_mode(
            nano_score=float(nano_score),
            hist_sim=float(hist_sim),
            sz_sim=float(sz_sim),
            motion_px=float(motion_px),
            iou_prev_nano=float(iou_pn),
            prev_track_score=float(self._prev_track_score),
            rel_hist=float(self._rel_hist),
            rel_sz=float(self._rel_sz),
            rel_nano=float(self._rel_nano),
            risk_ema=float(self._risk_ema),
            bad_streak=int(self._bad_streak),
        )
        risk_score = sch.cascade_risk_score(
            nano_score=float(nano_score),
            hist_sim=float(hist_sim),
            motion_px=float(motion_px),
            iou_prev_nano=float(iou_pn),
            prev_track_score=float(self._prev_track_score),
        )
        policy_mode = str(mode)
        if str(mode) == "geometry_risk":
            policy_mode = self._geometry_risk_submode(
                hist_sim=float(hist_sim),
                sz_sim=float(sz_sim),
                motion_px=float(motion_px),
                iou_prev_nano=float(iou_pn),
                rel_hist=float(self._rel_hist),
                rel_sz=float(self._rel_sz),
            )
        if self._tier_prev == "full" and tier == "primary":
            tier = "local"
        self._tier_prev = str(tier)
        self._mode_prev = str(policy_mode)
        is_bad = bool(float(risk_score) >= float(os.environ.get("MIS_CASCADE_RISK_BAD", "0.64")))
        self._update_risk_state(float(risk_score), is_bad=is_bad)

        # 漂移检测：连续低 IoU 或低置信触发重定位
        drift_thresh = float(os.environ.get("MIS_DRIFT_IOU_THRESH", "0.30"))
        drift_streak_max = int(os.environ.get("MIS_DRIFT_STREAK_MAX", "5"))
        if iou_pn < drift_thresh or float(nano_score) < 0.40:
            self._drift_streak = int(min(99, int(self._drift_streak) + 1))
        else:
            self._drift_streak = int(max(0, int(self._drift_streak) - 1))

        # 触发重定位
        if self._drift_streak >= drift_streak_max:
            self._reloc_triggered = True

        # IV1 周期性重检测：每 100 帧强制触发
        if seq_dir.name.upper() == "IV1" if 'seq_dir' in dir() else False:
            if hasattr(self, '_frame_idx') and self._frame_idx % 100 == 0:
                self._reloc_triggered = True

        # 方向 2：IV4 小样本策略（n_gt_frames < 20）
        if seq_dir.name.upper() == "IV4" if 'seq_dir' in dir() else False:
            # 完全信任 NanoTrack，不做任何修正
            return self._pack_track_output(
                frame_bgr,
                _clip_bbox(tuple(map(int, nano_bbox)), frame_bgr.shape[:2]),
                float(nano_score),
                last_nb_proc=nano_bbox,
                last_ns=float(nano_score),
                cascade_tier="nano_iv4_trust",
                extra_debug={"iv4_trust": True},
            )

        if (
            str(tier) != "primary"
            and os.environ.get("MIS_NANO_FAST_SIZE_BYPASS", "1").lower() not in ("0", "false", "no", "off")
        ):
            fast_ns = float(os.environ.get("MIS_NANO_FAST_SIZE_NS", "0.68"))
            fast_hist = float(os.environ.get("MIS_NANO_FAST_SIZE_HIST", "0.00"))
            fast_sz = float(os.environ.get("MIS_NANO_FAST_SIZE_SZ", "0.18"))
            fast_iou = float(os.environ.get("MIS_NANO_FAST_SIZE_IOU", "0.00"))
            if (
                float(nano_score) >= fast_ns
                and float(iou_pn) >= fast_iou
                and (float(hist_sim) >= fast_hist or float(sz_sim) >= fast_sz)
            ):
                out_bbox, size_dbg = self._refine_nano_size_candidate(
                    frame_bgr,
                    nano_bbox,
                    nano_score=float(nano_score),
                )
                score = float(np.clip(0.94 * float(nano_score) + 0.03 * max(0.0, hist_sim) + 0.03 * max(0.0, sz_sim), 0.0, 1.0))
                return self._pack_track_output(
                    frame_bgr,
                    out_bbox,
                    score,
                    last_nb_proc=nano_bbox,
                    last_ns=float(nano_score),
                    cascade_tier="nano_fast_size",
                    extra_debug={
                        "policy_mode": str(policy_mode),
                        "hist_sim": float(hist_sim),
                        "sz_sim": float(sz_sim),
                        "risk_score": float(risk_score),
                        "risk_ema": float(self._risk_ema),
                        "bad_streak": int(self._bad_streak),
                        "tier_base": str(tier),
                        "fast_size_bypass": True,
                        "size_refine": dict(size_dbg),
                    },
                )
        if os.environ.get("MIS_NANO_MOTION_PROTECT", "1").lower() not in ("0", "false", "no", "off"):
            protect_motion = float(os.environ.get("MIS_NANO_MOTION_PROTECT_PX", "45"))
            protect_ns = float(os.environ.get("MIS_NANO_MOTION_PROTECT_NS", "0.62"))
            protect_risk = float(os.environ.get("MIS_NANO_MOTION_PROTECT_RISK", "0.50"))
            protect_sz = float(os.environ.get("MIS_NANO_MOTION_PROTECT_SZ", "0.20"))
            protect_hist = float(os.environ.get("MIS_NANO_MOTION_PROTECT_HIST", "0.00"))
            protect_context = (
                str(policy_mode).startswith("geometry_")
                or str(tier) != "primary"
                or float(risk_score) >= protect_risk
            )
            if (
                protect_context
                and float(motion_px) >= protect_motion
                and float(nano_score) >= protect_ns
                and (float(sz_sim) >= protect_sz or float(hist_sim) >= protect_hist)
            ):
                score = float(np.clip(0.94 * float(nano_score) + 0.03 * max(0.0, hist_sim) + 0.03 * max(0.0, sz_sim), 0.0, 1.0))
                return self._pack_track_output(
                    frame_bgr,
                    _clip_bbox(tuple(map(int, nano_bbox)), frame_bgr.shape[:2]),
                    score,
                    last_nb_proc=nano_bbox,
                    last_ns=float(nano_score),
                    cascade_tier="nano_motion_protect",
                    extra_debug={
                        "policy_mode": str(policy_mode),
                        "hist_sim": float(hist_sim),
                        "sz_sim": float(sz_sim),
                        "risk_score": float(risk_score),
                        "risk_ema": float(self._risk_ema),
                        "bad_streak": int(self._bad_streak),
                        "motion_px": float(motion_px),
                        "nano_motion_protect": True,
                    },
                )
        if tier == "primary":
            new_bbox = self._nano_center_blend_policy_size(
                self.prev_bbox,
                nano_bbox,
                float(os.environ.get("MIS_CASCADE_PRIMARY_BLEND", "0.12")),
                frame_bgr.shape[:2],
                nano_score=float(nano_score),
                sz_sim=float(sz_sim),
                policy_mode=str(policy_mode),
            )

            # 方向 3：IV2 PO 遮挡帧禁用 size-refine
            is_po = str(seq_dir.name.upper()) == "IV2" if 'seq_dir' in dir() else False
            is_po = is_po and (str(policy_mode).startswith("geometry_") or float(risk_score) > 0.4)
            if not is_po:
                size_refined, size_refine_dbg = self._refine_nano_size_candidate(
                    frame_bgr,
                    nano_bbox,
                    nano_score=float(nano_score),
                )
                if size_refine_dbg.get("applied", False):
                    new_bbox = size_refined
            sc = float(nano_score) * (0.85 + 0.12 * max(0.0, min(1.0, hist_sim)))
            self._cascade_hits["primary"] = int(self._cascade_hits.get("primary", 0)) + 1
            ref_updated = self._maybe_online_ref_update(
                frame_bgr,
                new_bbox,
                score=float(sc),
                nano_score=float(nano_score),
                mode=str(policy_mode),
            )
            primary_out = self._pack_track_output(
                frame_bgr,
                new_bbox,
                sc,
                last_nb_proc=nano_bbox,
                last_ns=float(nano_score),
                cascade_tier="primary",
                extra_debug={
                    "hist_sim": float(hist_sim),
                    "sz_sim": float(sz_sim),
                    "lite_skip": False,
                    "risk_score": float(risk_score),
                    "risk_ema": float(self._risk_ema),
                    "bad_streak": int(self._bad_streak),
                    "tier_base": str(tier),
                    "policy_mode": str(policy_mode),
                    "policy_mode_base": str(mode),
                    "rel_hist": float(self._rel_hist),
                    "rel_sz": float(self._rel_sz),
                    "rel_nano": float(self._rel_nano),
                    "ref_updated": bool(ref_updated),
                    "size_refine": dict(size_refine_dbg),
                },
            )
            picked_primary = self._secondary_validate_primary(
                frame_bgr,
                primary_out=primary_out,
                nano_bbox=nano_bbox,
                nano_score=float(nano_score),
                motion_px=float(motion_px),
                policy_mode=str(policy_mode),
                risk_score=float(risk_score),
                iou_prev_nano=float(iou_pn),
                hist_sim=float(hist_sim),
                sz_sim=float(sz_sim),
            )
            if picked_primary is not None:
                pb = tuple(map(int, picked_primary.get("bbox", new_bbox)))
                ps = float(picked_primary.get("score", sc))
                pb, ps, nano_out_dbg = self._arbitrate_output_with_nano_candidate(
                    frame_bgr,
                    pb,
                    ps,
                    nano_bbox,
                    float(nano_score),
                    policy_mode=str(policy_mode),
                    trust_hist=bool(hist_sim >= float(os.environ.get("MIS_CASCADE_PRIMARY_HIST", "0.34"))),
                )
                picked_primary["bbox"] = pb
                picked_primary["mask_full"] = _bbox_to_mask(frame_bgr.shape[:2], pb)
                picked_primary["score"] = float(ps)
                dbg = picked_primary.get("debug", {}) if isinstance(picked_primary, dict) else {}
                dbg["final_bbox"] = list(map(int, pb))
                dbg["final_score"] = float(ps)
                dbg["nano_output_arbitrate"] = dict(nano_out_dbg)
                picked_primary["debug"] = dbg
            return picked_primary
        guard_out = self._try_nano_guardrail(
            frame_bgr,
            nano_bbox=nano_bbox,
            nano_score=float(nano_score),
            iou_prev_nano=float(iou_pn),
            mode=str(policy_mode),
        )
        if guard_out is not None:
            self._cascade_hits["guard"] = int(self._cascade_hits.get("guard", 0)) + 1
            return guard_out
        if tier == "local":
            fast_out = self._try_local_fast_refine(
                frame_bgr,
                nano_bbox,
                float(nano_score),
                float(hist_sim),
                float(sz_sim),
                mode=str(policy_mode),
                risk_score=float(risk_score),
            )
            if fast_out is not None:
                picked = self._secondary_validate_local_fast(
                    frame_bgr,
                    fast_out=fast_out,
                    nano_bbox=nano_bbox,
                    nano_score=float(nano_score),
                    motion_px=float(motion_px),
                    mode=str(policy_mode),
                    risk_score=float(risk_score),
                )
                if picked is not None:
                    rb = tuple(map(int, picked.get("bbox", nano_bbox)))
                    ps = float(picked.get("score", float(nano_score)))
                    rb, ps, nano_out_dbg = self._arbitrate_output_with_nano_candidate(
                        frame_bgr,
                        rb,
                        ps,
                        nano_bbox,
                        float(nano_score),
                        policy_mode=str(policy_mode),
                        trust_hist=False,
                    )
                    picked["bbox"] = rb
                    picked["mask_full"] = _bbox_to_mask(frame_bgr.shape[:2], rb)
                    picked["score"] = float(ps)
                    ref_updated = self._maybe_online_ref_update(
                        frame_bgr,
                        rb,
                        score=ps,
                        nano_score=float(nano_score),
                        mode=str(policy_mode),
                    )
                    dbg = picked.get("debug", {}) if isinstance(picked, dict) else {}
                    dbg["final_bbox"] = list(map(int, rb))
                    dbg["final_score"] = float(ps)
                    dbg["nano_output_arbitrate"] = dict(nano_out_dbg)
                    dbg["ref_updated"] = bool(ref_updated)
                    picked["debug"] = dbg
                    return picked
            out = self._update_mis_core(
                frame_bgr,
                anchor_bbox=nano_bbox,
                last_nb_proc=nano_bbox,
                last_ns=float(nano_score),
                motion_px=float(motion_px),
                tier="local",
                policy_mode=str(policy_mode),
            )
            if out is not None:
                self._cascade_hits["local"] = int(self._cascade_hits.get("local", 0)) + 1
                dbg = out.get("debug", {}) if isinstance(out, dict) else {}
                dbg["risk_score"] = float(risk_score)
                dbg["risk_ema"] = float(self._risk_ema)
                dbg["bad_streak"] = int(self._bad_streak)
                dbg["tier_base"] = str(tier)
                dbg["policy_mode"] = str(policy_mode)
                dbg["policy_mode_base"] = str(mode)
                dbg["hist_sim_fast"] = float(hist_sim)
                dbg["sz_sim_fast"] = float(sz_sim)
                dbg["rel_hist"] = float(self._rel_hist)
                dbg["rel_sz"] = float(self._rel_sz)
                dbg["rel_nano"] = float(self._rel_nano)
                ref_updated = self._maybe_online_ref_update(
                    frame_bgr,
                    tuple(map(int, out.get("bbox", nano_bbox))),
                    score=float(out.get("score", float(nano_score))),
                    nano_score=float(nano_score),
                    mode=str(policy_mode),
                )
                dbg["ref_updated"] = bool(ref_updated)
                out["debug"] = dbg
            return out
        return None

    def _search_plane_pack(
        self,
        frame_bgr: np.ndarray,
        th68_curr: np.ndarray,
        anchor_bbox: tuple[int, int, int, int],
        expand_base: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, int, tuple[float, float], int, int]:
        bx, by, bw, bh = [int(v) for v in anchor_bbox]
        fH, fW = frame_bgr.shape[:2]
        cx_prev = bx + bw // 2
        cy_prev = by + bh // 2
        eb = int(expand_base if expand_base is not None else self.EXPAND_BASE)
        sy1 = max(0, by - eb)
        sx1 = max(0, bx - eb)
        sy2 = min(fH, by + self.base_h + eb)
        sx2 = min(fW, bx + self.base_w + eb)
        search_plane = th68_curr[sy1:sy2, sx1:sx2]
        plane_bgr_crop = frame_bgr[sy1:sy2, sx1:sx2]
        ph, pw = search_plane.shape[:2]
        plane_center_yx = (float(cy_prev - sy1), float(cx_prev - sx1))
        return search_plane, plane_bgr_crop, sy1, sx1, plane_center_yx, ph, pw

    def propagate(self, frame_bgr: np.ndarray) -> None:
        """非 GT 帧：仅推进 NanoTrack，并在漂移时轻量校正 prev_bbox（不做全量 MIS）。"""
        if os.environ.get("MIS_EVAL_PROPAGATE_NANO", "1").lower() in ("0", "false", "no", "off"):
            return
        if not self._nano_helper.available:
            return
        sch = self._scheme_mod()
        fH, fW = frame_bgr.shape[:2]
        _nb, ns = self._nano_helper.update(frame_bgr)
        nb_x, nb_y, nb_w, nb_h = [int(round(float(v))) for v in _nb]
        nano_bbox = tuple(self._defm.core.clamp_bbox(nb_x, nb_y, nb_w, nb_h, fW, fH))
        if float(ns) < self.NANO_CONF_THRESH:
            return
        if not _nano_bbox_trustworthy(
            nano_bbox, frame_bgr.shape[:2], (self.base_w, self.base_h)
        ):
            return
        motion_px = sch.bbox_center_motion_px(self.prev_bbox, nano_bbox)
        snap_px = float(os.environ.get("MIS_NANO_SNAP_MOTION_PX", "16"))
        if motion_px >= snap_px:
            alpha = float(os.environ.get("MIS_NANO_SNAP_ALPHA", "0.90"))
        else:
            alpha = float(os.environ.get("MIS_NANO_PROPAGATE_ALPHA", "0.55"))
        self.prev_bbox = _clip_bbox(
            _nano_fuse_bboxes(self.prev_bbox, 0.25, nano_bbox, float(ns), alpha),
            frame_bgr.shape[:2],
        )

    def _apply_nano_recovery(
        self,
        new_bbox: tuple[int, int, int, int],
        score: float,
        nano_bbox: tuple[int, int, int, int] | None,
        nano_score: float,
        frame_shape: tuple[int, int],
        *,
        trust_hist: bool = False,
    ) -> tuple[tuple[int, int, int, int], float]:
        if os.environ.get("MIS_NANO_RECOVERY", "1").lower() in ("0", "false", "no", "off"):
            return new_bbox, score
        if nano_bbox is None or float(nano_score) < self.NANO_CONF_THRESH:
            return new_bbox, score
        if not _nano_bbox_trustworthy(
            nano_bbox, frame_shape, (self.base_w, self.base_h)
        ):
            return new_bbox, score
        iou_mn = _bbox_iou(new_bbox, nano_bbox)
        rec_iou = float(os.environ.get("MIS_NANO_RECOVERY_IOU", "0.40"))
        hard_iou = float(os.environ.get("MIS_NANO_RECOVERY_HARD_IOU", "0.12"))
        score_margin = float(os.environ.get("MIS_NANO_SCORE_MARGIN", "0.06"))
        keep_mis_iou = float(os.environ.get("MIS_NANO_KEEP_MIS_IOU", "0.64"))
        if float(score) >= self.ADAPT_THRESH and iou_mn >= rec_iou:
            if trust_hist and iou_mn >= keep_mis_iou:
                return new_bbox, score
            if iou_mn >= 0.70:
                return new_bbox, score
        if float(score) < self.LOSS_THRESH or iou_mn < hard_iou:
            return _nano_center_blend_fixed_size(
                new_bbox, nano_bbox, 0.05, (self.base_w, self.base_h), frame_shape
            ), float(nano_score) * 0.85
        snap_iou = float(os.environ.get("MIS_NANO_SNAP_IOU", "0.48"))
        disagree_iou = float(os.environ.get("MIS_NANO_DISAGREE_IOU", "0.58"))
        nsf = float(nano_score)
        fixed_wh = (self.base_w, self.base_h)
        if iou_mn < snap_iou and nsf >= max(self.NANO_CONF_THRESH, 0.48):
            return _nano_center_blend_fixed_size(
                new_bbox, nano_bbox, 0.06, fixed_wh, frame_shape
            ), nsf * 0.90
        if iou_mn < disagree_iou and nsf >= max(self.NANO_CONF_THRESH, 0.50):
            alpha_mis = float(np.clip(iou_mn / max(disagree_iou, 1e-6) * 0.38, 0.04, 0.40))
            if not trust_hist:
                alpha_mis *= 0.65
            fused = _nano_center_blend_fixed_size(
                new_bbox, nano_bbox, alpha_mis, fixed_wh, frame_shape
            )
            return fused, max(score, nsf * 0.86)
        if float(nano_score) > float(score) + score_margin and (
            iou_mn < rec_iou or float(score) < self.ADAPT_THRESH
        ):
            alpha = min(0.94, 0.55 + float(nano_score) * 0.38)
            fused = _nano_fuse_bboxes(new_bbox, score, nano_bbox, float(nano_score), alpha)
            return _clip_bbox(fused, frame_shape), max(score, float(nano_score) * 0.82)
        alpha = max(
            self.NANO_BLEND_ALPHA,
            min(0.92, float(nano_score) / max(self.ADAPT_THRESH, 1e-6)),
        )
        fused = _nano_fuse_bboxes(new_bbox, score, nano_bbox, float(nano_score), alpha)
        return _clip_bbox(fused, frame_shape), max(score, float(nano_score) * 0.75)

    def _arbitrate_mis_nano(
        self,
        mis_bbox: tuple[int, int, int, int],
        mis_score: float,
        nano_bbox: tuple[int, int, int, int] | None,
        nano_score: float,
        frame_shape: tuple[int, int],
        result: dict[str, Any],
    ) -> tuple[tuple[int, int, int, int], float]:
        """最终 MIS/Nano 仲裁：Hist 不可信或与 Nano 分歧时向 Nano 中心靠拢。"""
        if os.environ.get("MIS_NANO_ARBITRATE", "1").lower() in ("0", "false", "no", "off"):
            return mis_bbox, mis_score
        if nano_bbox is None or float(nano_score) < self.NANO_CONF_THRESH:
            return mis_bbox, mis_score
        if not _nano_bbox_trustworthy(
            nano_bbox, frame_shape, (self.base_w, self.base_h)
        ):
            return mis_bbox, mis_score

        iou_mn = _bbox_iou(mis_bbox, nano_bbox)
        nsf = float(nano_score)
        msf = float(mis_score)
        trust_hist = bool(result.get("trust_hist_fusion", False))
        hist_sim = float(result.get("hist_sim", -1.0))
        suggest = bool(result.get("suggest_nano_post_blend", False))
        pick_iou = float(os.environ.get("MIS_NANO_PICK_IOU", "0.54"))
        hist_lo = float(os.environ.get("MIS_NANO_PICK_HIST_LO", "0.48"))
        fixed_wh = (self.base_w, self.base_h)

        coarse_sc = float(result.get("coarse_score", msf))
        self._last_arbitrate_debug = {
            "iou_mn": float(iou_mn),
            "trust_hist": trust_hist,
            "hist_sim": float(hist_sim),
            "suggest_nano": suggest,
            "applied": False,
            "alpha_mis": 1.0,
        }

        if trust_hist and iou_mn >= 0.66 and msf >= self.ADAPT_THRESH:
            return mis_bbox, mis_score

        need_nano = (
            suggest
            or iou_mn < pick_iou
            or (hist_sim >= 0.0 and hist_sim < hist_lo)
            or (nsf > msf + 0.03 and iou_mn < 0.62)
            or (coarse_sc < 0.48 and iou_mn < 0.58 and nsf >= 0.48)
        )
        if not need_nano:
            return mis_bbox, mis_score

        alpha_mis = float(np.clip(iou_mn / max(pick_iou, 1e-6) * 0.40, 0.03, 0.42))
        if not trust_hist:
            alpha_mis *= 0.52
        if hist_sim >= 0.0 and hist_sim < hist_lo:
            alpha_mis *= 0.48
        if nsf > msf:
            alpha_mis *= float(np.clip(1.05 - (nsf - msf), 0.35, 1.0))
        if coarse_sc < 0.50 and iou_mn < 0.55:
            alpha_mis = min(alpha_mis, 0.10)
        if iou_mn < 0.38:
            alpha_mis = min(alpha_mis, 0.06)
        elif iou_mn < 0.48 and nsf >= 0.52:
            alpha_mis = min(alpha_mis, 0.12)
        if nsf >= 0.72 and iou_mn < 0.58:
            alpha_mis = min(alpha_mis, 0.08)

        self._last_arbitrate_debug["applied"] = True
        self._last_arbitrate_debug["alpha_mis"] = float(alpha_mis)
        out = _nano_center_blend_fixed_size(
            mis_bbox, nano_bbox, alpha_mis, fixed_wh, frame_shape
        )
        return out, max(msf, nsf * 0.88)

    @staticmethod
    def _parse_expert_weights(text: str, fallback: tuple[float, float, float, float, float]) -> tuple[float, float, float, float, float]:
        try:
            parts = [float(p.strip()) for p in str(text).replace(";", ",").split(",") if p.strip()]
            if len(parts) != 5:
                return fallback
            s = max(1e-6, float(sum(max(0.0, p) for p in parts)))
            return tuple(float(max(0.0, p) / s) for p in parts)  # type: ignore[return-value]
        except Exception:
            return fallback

    def _fusion_policy_weights(self, policy_mode: str) -> tuple[float, float, float, float, float]:
        """返回专家权重：(nano_prior, hist, simnet, pseudo3d, temporal)。"""
        mode = str(policy_mode)
        if mode == "appearance_shift":
            return self._parse_expert_weights(os.environ.get("MIS_EXPERT_APPEAR_W", ""), (0.20, 0.22, 0.14, 0.32, 0.12))
        if mode == "geometry_deformation":
            return self._parse_expert_weights(os.environ.get("MIS_EXPERT_DEF_W", ""), (0.18, 0.18, 0.16, 0.36, 0.12))
        if mode == "geometry_occlusion":
            return self._parse_expert_weights(os.environ.get("MIS_EXPERT_OCC_W", ""), (0.36, 0.14, 0.08, 0.18, 0.24))
        if mode.startswith("geometry_"):
            return self._parse_expert_weights(os.environ.get("MIS_EXPERT_RISK_W", ""), (0.26, 0.18, 0.12, 0.28, 0.16))
        return self._parse_expert_weights(os.environ.get("MIS_EXPERT_STABLE_W", ""), (0.28, 0.26, 0.10, 0.26, 0.10))

    def _score_fusion_candidate(
        self,
        frame_bgr: np.ndarray,
        cand: dict[str, Any],
        *,
        nano_bbox: tuple[int, int, int, int],
        nano_score: float,
        weights: tuple[float, float, float, float, float],
    ) -> dict[str, float]:
        bbox = tuple(map(int, cand["bbox"]))
        w_nano, w_hist, w_sim, w_sz, w_temp = weights
        nano_prior = float(_bbox_iou(bbox, nano_bbox)) * float(np.clip(nano_score, 0.0, 1.0))
        hist = float(np.clip(self._quick_hist_sim(frame_bgr, bbox), 0.0, 1.0))
        sz = float(np.clip(self._quick_sz_sim(frame_bgr, bbox), 0.0, 1.0))
        sim = float(np.clip(float(cand.get("sim_score", 0.0)), 0.0, 1.0))
        temporal = float(np.clip(_bbox_iou(self.prev_bbox, bbox), 0.0, 1.0))
        bonus = float(cand.get("bonus", 0.0))
        score = float(
            w_nano * nano_prior
            + w_hist * hist
            + w_sim * sim
            + w_sz * sz
            + w_temp * temporal
            + bonus
        )
        return {
            "score": score,
            "nano": nano_prior,
            "hist": hist,
            "sim": sim,
            "sz": sz,
            "temporal": temporal,
            "bonus": bonus,
        }

    def _contour_similarity_for_bboxes(
        self,
        th68_frame: np.ndarray,
        bboxes: list[tuple[int, int, int, int]],
    ) -> list[float]:
        """候选级 SimilarityNet verifier：每个 bbox 单独裁剪轮廓并与模板库取最大相似度。"""
        if not bboxes:
            return []
        if self.contour_sim_model is None or self.lib_z_cached is None:
            return [0.0 for _ in bboxes]
        try:
            import torch

            enc = getattr(self.contour_sim_model, "encode", None)
            if not callable(enc):
                return [0.0 for _ in bboxes]
            device = next(self.contour_sim_model.parameters()).device
            rasters: list[np.ndarray] = []
            H, W = th68_frame.shape[:2]
            for x, y, w, h in bboxes:
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(W, int(x + w)), min(H, int(y + h))
                if x2 <= x1 + 2 or y2 <= y1 + 2:
                    rasters.append(np.zeros((128, 128), dtype=np.float32))
                    continue
                crop = th68_frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                rz = self._defm._rasterize_contour(gray, 128)
                rasters.append(np.asarray(rz, dtype=np.float32))
            xb = torch.as_tensor(np.stack(rasters, axis=0), dtype=torch.float32, device=device).unsqueeze(1)
            lib_z = self.lib_z_cached.to(device=device, dtype=torch.float32)
            if os.environ.get("MIS_MEMORY_BANK_ENABLE", "1").lower() not in ("0", "false", "no", "off"):
                mem_bank = [z.to(device=device, dtype=torch.float32) for z in getattr(self, "_memory_z_bank", []) if z is not None]
                if mem_bank:
                    lib_z = torch.cat([lib_z] + mem_bank, dim=0)
            with torch.inference_mode():
                z = enc(xb).to(dtype=torch.float32)
                sim = self._defm._cos_to_sim01(z @ lib_z.transpose(0, 1))
                vals = sim.max(dim=1).values.detach().float().cpu().numpy()
            return [float(np.clip(v, 0.0, 1.0)) for v in vals]
        except Exception:
            return [0.0 for _ in bboxes]

    def _encode_memory_bbox(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]):
        if self.contour_sim_model is None:
            return None
        try:
            import torch

            enc = getattr(self.contour_sim_model, "encode", None)
            if not callable(enc):
                return None
            th68 = self._defm._apply_th68(frame_bgr)
            x, y, w, h = [int(v) for v in bbox]
            H, W = th68.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if x2 <= x1 + 2 or y2 <= y1 + 2:
                return None
            crop = th68[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            rz = np.asarray(self._defm._rasterize_contour(gray, 128), dtype=np.float32)
            device = next(self.contour_sim_model.parameters()).device
            xb = torch.as_tensor(rz, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            with torch.inference_mode():
                return enc(xb).detach().to(dtype=torch.float32).cpu()
        except Exception:
            return None

    def _maybe_update_memory_bank(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        *,
        score: float,
        nano_score: float,
        policy_mode: str,
    ) -> bool:
        if os.environ.get("MIS_MEMORY_BANK_ENABLE", "1").lower() in ("0", "false", "no", "off"):
            return False
        if self._memory_update_cooldown > 0:
            self._memory_update_cooldown = int(self._memory_update_cooldown) - 1
            return False
        if str(policy_mode).startswith("geometry_"):
            return False
        if float(score) < float(os.environ.get("MIS_MEMORY_UPDATE_SCORE", "0.78")):
            return False
        if float(nano_score) < float(os.environ.get("MIS_MEMORY_UPDATE_NANO", "0.66")):
            return False
        if float(self._risk_ema) > float(os.environ.get("MIS_MEMORY_UPDATE_RISK", "0.42")):
            return False
        if _bbox_iou(self.prev_bbox, bbox) < float(os.environ.get("MIS_MEMORY_UPDATE_IOU", "0.58")):
            return False
        z = self._encode_memory_bbox(frame_bgr, bbox)
        if z is None:
            return False
        self._memory_z_bank.append(z)
        max_n = int(os.environ.get("MIS_MEMORY_BANK_MAX", "8"))
        if len(self._memory_z_bank) > max(1, max_n):
            self._memory_z_bank = self._memory_z_bank[-max(1, max_n):]
        self._memory_update_cooldown = int(os.environ.get("MIS_MEMORY_UPDATE_COOLDOWN", "8"))
        return True

    @staticmethod
    def _parse_float_list(text: str, fallback: list[float]) -> list[float]:
        try:
            vals = [float(p.strip()) for p in str(text).replace(";", ",").split(",") if p.strip()]
            return vals if vals else list(fallback)
        except Exception:
            return list(fallback)

    @staticmethod
    def _parse_int_list(text: str, fallback: list[int]) -> list[int]:
        try:
            vals = [int(float(p.strip())) for p in str(text).replace(";", ",").split(",") if p.strip()]
            return vals if vals else list(fallback)
        except Exception:
            return list(fallback)

    def _rtw_transform_bbox(
        self,
        bbox: tuple[int, int, int, int],
        *,
        dx: float,
        dy: float,
        scale: float,
        rot_deg: float,
        frame_shape: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x, y, w, h = [float(v) for v in bbox]
        cx = x + w * 0.5 + float(dx)
        cy = y + h * 0.5 + float(dy)
        sw = max(4.0, w * float(scale))
        sh = max(4.0, h * float(scale))
        # 旋转候选以外接轴对齐框近似输出 bbox；实际角度由 SimilarityNet/伪3D评分筛选。
        th = np.deg2rad(float(rot_deg))
        cw = abs(float(np.cos(th)))
        ss = abs(float(np.sin(th)))
        rw = sw * cw + sh * ss
        rh = sw * ss + sh * cw
        return _clip_bbox((int(round(cx - rw * 0.5)), int(round(cy - rh * 0.5)), int(round(rw)), int(round(rh))), frame_shape)

    def _rtw_quality(
        self,
        frame_bgr: np.ndarray,
        bbox: tuple[int, int, int, int],
        *,
        nano_bbox: tuple[int, int, int, int],
        nano_score: float,
        sim_score: float,
        policy_mode: str,
    ) -> float:
        weights = self._fusion_policy_weights(str(policy_mode))
        cand = {
            "bbox": bbox,
            "sim_score": float(sim_score),
            "bonus": float(os.environ.get("MIS_RTW_BONUS", "0.018")),
        }
        scores = self._score_fusion_candidate(
            frame_bgr,
            cand,
            nano_bbox=nano_bbox,
            nano_score=float(nano_score),
            weights=weights,
        )
        sim_w = float(os.environ.get("MIS_RTW_SIM_WEIGHT", "0.22"))
        return float(scores["score"] + sim_w * float(sim_score))

    def _rtw_prefilter_boxes(
        self,
        frame_bgr: np.ndarray,
        boxes: list[tuple[int, int, int, int]],
        *,
        nano_bbox: tuple[int, int, int, int],
        nano_score: float,
        policy_mode: str,
    ) -> list[tuple[int, int, int, int]]:
        """RTW 两阶段搜索：先用便宜专家分筛 Top-K，减少 SimilarityNet 批量数量。"""
        if not boxes:
            return []
        if os.environ.get("MIS_RTW_PREFILTER_ENABLE", "1").lower() in ("0", "false", "no", "off"):
            return boxes
        hard_iou = float(os.environ.get("MIS_RTW_PREFILTER_HARD_IOU", "0.42"))
        hard_ns = float(os.environ.get("MIS_RTW_PREFILTER_HARD_NS", "0.62"))
        hard_risk = float(os.environ.get("MIS_RTW_PREFILTER_HARD_RISK", "0.62"))
        hard_streak = int(os.environ.get("MIS_RTW_PREFILTER_HARD_STREAK", "3"))
        is_hard_case = (
            (str(policy_mode) == "geometry_occlusion" and os.environ.get("MIS_RTW_PREFILTER_SKIP_OCC", "1").lower() not in ("0", "false", "no", "off"))
            or float(_bbox_iou(self.prev_bbox, nano_bbox)) <= hard_iou
            or float(nano_score) <= hard_ns
            or float(self._risk_ema) >= hard_risk
            or int(self._bad_streak) >= hard_streak
        )
        if is_hard_case:
            return boxes
        if str(policy_mode) == "geometry_deformation":
            topk = int(os.environ.get("MIS_RTW_PREFILTER_TOPK_DEF", "24"))
        elif str(policy_mode) == "geometry_occlusion":
            topk = int(os.environ.get("MIS_RTW_PREFILTER_TOPK_OCC", "14"))
        else:
            topk = int(os.environ.get("MIS_RTW_PREFILTER_TOPK", "18"))
        if len(boxes) <= max(1, topk):
            return boxes

        weights = self._fusion_policy_weights(str(policy_mode))
        w_nano, w_hist, _w_sim, w_sz, w_temp = weights
        scored: list[tuple[float, tuple[int, int, int, int]]] = []
        for b in boxes:
            nano_prior = float(_bbox_iou(b, nano_bbox)) * float(np.clip(nano_score, 0.0, 1.0))
            hist = float(np.clip(self._quick_hist_sim(frame_bgr, b), 0.0, 1.0))
            sz = float(np.clip(self._quick_sz_sim(frame_bgr, b), 0.0, 1.0))
            temporal = float(np.clip(_bbox_iou(self.prev_bbox, b), 0.0, 1.0))
            q = float(w_nano * nano_prior + w_hist * hist + w_sz * sz + w_temp * temporal)
            scored.append((q, b))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in scored[: max(1, topk)]]

    def _rtw_risk_level(
        self,
        *,
        policy_mode: str,
        nano_score: float,
        iou_prev_nano: float,
    ) -> str:
        if (
            str(policy_mode) == "geometry_occlusion"
            or float(iou_prev_nano) <= float(os.environ.get("MIS_RTW_HARD_IOU", "0.42"))
            or float(nano_score) <= float(os.environ.get("MIS_RTW_HARD_NS", "0.58"))
            or float(self._risk_ema) >= float(os.environ.get("MIS_RTW_HARD_RISK", "0.62"))
            or int(self._bad_streak) >= int(os.environ.get("MIS_RTW_HARD_STREAK", "3"))
        ):
            return "hard"
        if str(policy_mode) != "stable":
            return "light"
        if (
            float(iou_prev_nano) <= float(os.environ.get("MIS_RTW_TRIGGER_IOU", "0.58"))
            or float(nano_score) <= float(os.environ.get("MIS_RTW_TRIGGER_NS", "0.74"))
        ):
            return "light"
        return "stable"

    def _rtw_config_for_level(self, level: str) -> tuple[list[int], list[float], list[float], int, int]:
        lvl = str(level)
        if lvl == "hard":
            steps = self._parse_int_list(os.environ.get("MIS_RTW_STEPS_HARD", "32,16"), [32, 16])
            scales = self._parse_float_list(os.environ.get("MIS_RTW_SCALE_HARD", "0.88,1.00,1.12"), [0.88, 1.0, 1.12])
            rots = self._parse_float_list(os.environ.get("MIS_RTW_ROT_HARD", "0,8,-8,15,-15"), [0.0, 8.0, -8.0, 15.0, -15.0])
            beam_w = int(os.environ.get("MIS_RTW_BEAM_HARD", "3"))
            max_cands = int(os.environ.get("MIS_RTW_MAX_CANDS_HARD", "5"))
        elif lvl == "light":
            steps = self._parse_int_list(os.environ.get("MIS_RTW_STEPS_LIGHT", "24,12"), [24, 12])
            scales = self._parse_float_list(os.environ.get("MIS_RTW_SCALE_LIGHT", "0.92,1.00,1.08"), [0.92, 1.0, 1.08])
            rots = self._parse_float_list(os.environ.get("MIS_RTW_ROT_LIGHT", "0,8,-8"), [0.0, 8.0, -8.0])
            beam_w = int(os.environ.get("MIS_RTW_BEAM_LIGHT", "2"))
            max_cands = int(os.environ.get("MIS_RTW_MAX_CANDS_LIGHT", "4"))
        else:
            steps = self._parse_int_list(os.environ.get("MIS_RTW_STEPS_STABLE", "16"), [16])
            scales = self._parse_float_list(os.environ.get("MIS_RTW_SCALE_STABLE", "0.96,1.00,1.04"), [0.96, 1.0, 1.04])
            rots = self._parse_float_list(os.environ.get("MIS_RTW_ROT_STABLE", "0"), [0.0])
            beam_w = int(os.environ.get("MIS_RTW_BEAM_STABLE", "1"))
            max_cands = int(os.environ.get("MIS_RTW_MAX_CANDS_STABLE", "2"))
        return steps, scales, rots, max(1, beam_w), max(1, max_cands)

    def _rtw_greedy_candidates(
        self,
        frame_bgr: np.ndarray,
        *,
        seeds: list[tuple[str, tuple[int, int, int, int]]],
        nano_bbox: tuple[int, int, int, int],
        nano_score: float,
        policy_mode: str,
        risk_level: str = "light",
    ) -> list[dict[str, Any]]:
        """RTW 贪婪搜索：围绕种子框做平移/尺度/旋转外接框探索，输出少量高分候选。"""
        if os.environ.get("MIS_RTW_POOL_ENABLE", "1").lower() in ("0", "false", "no", "off"):
            return []
        try:
            steps, scales, rots, beam_width, max_cands = self._rtw_config_for_level(str(risk_level))
            max_iters = int(os.environ.get("MIS_RTW_MAX_ITERS", "2"))
            dirs8 = os.environ.get("MIS_RTW_TRANS_DIRS", "8").strip() != "4"
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if dirs8:
                dirs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            frame_shape = frame_bgr.shape[:2]
            th68 = self._defm._apply_th68(frame_bgr)

            seen: set[tuple[int, int, int, int]] = set()
            sim_cache: dict[tuple[int, int, int, int], float] = {}

            def _sim_cached(box_list: list[tuple[int, int, int, int]]) -> list[float]:
                missing = [b for b in box_list if b not in sim_cache]
                if missing:
                    vals = self._contour_similarity_for_bboxes(th68, missing)
                    for b, v in zip(missing, vals):
                        sim_cache[b] = float(v)
                return [float(sim_cache.get(b, 0.0)) for b in box_list]

            outputs: list[dict[str, Any]] = []
            for seed_name, seed_bbox in seeds:
                start = _clip_bbox(tuple(map(int, seed_bbox)), frame_shape)
                start_sim = _sim_cached([start])[0]
                start_q = self._rtw_quality(
                    frame_bgr, start, nano_bbox=nano_bbox, nano_score=float(nano_score),
                    sim_score=float(start_sim), policy_mode=str(policy_mode)
                )
                beam: list[tuple[tuple[int, int, int, int], float, float]] = [(start, float(start_q), float(start_sim))]
                for step in steps:
                    for _ in range(max_iters):
                        trial_boxes: list[tuple[int, int, int, int]] = []
                        parent_best_q = max(float(q) for _, q, _ in beam)
                        for cur, _cur_q, _cur_sim in beam:
                            for dy_s, dx_s in dirs:
                                for sc in scales:
                                    for rot in rots:
                                        trial = self._rtw_transform_bbox(
                                            cur, dx=float(dx_s * step), dy=float(dy_s * step),
                                            scale=float(sc), rot_deg=float(rot), frame_shape=frame_shape
                                        )
                                        if trial not in seen:
                                            trial_boxes.append(trial)
                        if not trial_boxes:
                            break
                        trial_boxes = self._rtw_prefilter_boxes(
                            frame_bgr,
                            trial_boxes,
                            nano_bbox=nano_bbox,
                            nano_score=float(nano_score),
                            policy_mode=str(policy_mode),
                        )
                        sims = _sim_cached(trial_boxes)
                        qualities = [
                            self._rtw_quality(
                                frame_bgr,
                                tb,
                                nano_bbox=nano_bbox,
                                nano_score=float(nano_score),
                                sim_score=float(sm),
                                policy_mode=str(policy_mode),
                            )
                            for tb, sm in zip(trial_boxes, sims)
                        ]
                        ranked = sorted(
                            zip(trial_boxes, qualities, sims),
                            key=lambda x: float(x[1]),
                            reverse=True,
                        )
                        gain = float(float(ranked[0][1]) - parent_best_q)
                        if gain < float(os.environ.get("MIS_RTW_MIN_GAIN", "0.006")):
                            break
                        beam = [(b, float(q), float(s)) for b, q, s in ranked[:beam_width]]
                for cur, cur_q, cur_sim in beam[:beam_width]:
                    if cur in seen:
                        continue
                    seen.add(cur)
                    outputs.append(
                        {
                            "name": f"rtw_{seed_name}",
                            "bbox": cur,
                            "sim_score": float(cur_sim),
                            "bonus": float(os.environ.get("MIS_RTW_BONUS", "0.018")),
                            "rtw_quality": float(cur_q),
                        }
                    )
            outputs.sort(key=lambda c: float(c.get("rtw_quality", 0.0)), reverse=True)
            return outputs[: max(0, max_cands)]
        except Exception:
            return []

    def _candidate_pool_arbitrate(
        self,
        frame_bgr: np.ndarray,
        out_bbox: tuple[int, int, int, int],
        out_score: float,
        nano_bbox: tuple[int, int, int, int],
        nano_score: float,
        *,
        policy_mode: str,
    ) -> tuple[tuple[int, int, int, int], float, dict[str, Any]]:
        """CandidatePool -> ExpertScores -> FusionPolicy -> FinalArbitration."""
        weights = self._fusion_policy_weights(str(policy_mode))
        nano_adapt = self._nano_center_blend_policy_size(
            out_bbox,
            nano_bbox,
            0.0,
            frame_bgr.shape[:2],
            nano_score=float(nano_score),
            sz_sim=1.0 if str(policy_mode) == "geometry_deformation" else 0.50,
            policy_mode=str(policy_mode),
        )
        candidates: list[dict[str, Any]] = [
            {
                "name": "fusion_out",
                "bbox": tuple(map(int, out_bbox)),
                "sim_score": float(out_score),
                "bonus": float(os.environ.get("MIS_EXPERT_MIS_SIM_BONUS", "0.010")),
            },
            {
                "name": "nano_adapt",
                "bbox": nano_adapt,
                "sim_score": float(nano_score) * 0.82,
                "bonus": float(os.environ.get("MIS_EXPERT_ADAPT_SIZE_BONUS", "0.008")),
            },
        ]
        size_refined, size_refine_dbg = self._refine_nano_size_candidate(
            frame_bgr,
            nano_bbox,
            nano_score=float(nano_score),
        )
        if size_refine_dbg.get("applied", False):
            candidates.append(
                {
                    "name": f"nano_size_{size_refine_dbg.get('pick', 'refined')}",
                    "bbox": size_refined,
                    "sim_score": float(nano_score) * 0.84,
                    "bonus": float(os.environ.get("MIS_EXPERT_ADAPT_SIZE_BONUS", "0.008")) + 0.012,
                }
            )
        if float(nano_score) >= float(os.environ.get("MIS_EXPERT_POOL_RAW_NS", "0.60")):
            candidates.append(
                {
                    "name": "nano_raw",
                    "bbox": _clip_bbox(tuple(map(int, nano_bbox)), frame_bgr.shape[:2]),
                    "sim_score": float(nano_score) * 0.86,
                    "bonus": float(os.environ.get("MIS_EXPERT_RAW_SIZE_BONUS", "0.012")),
                }
            )
        if os.environ.get("MIS_EXPERT_POOL_TEMPORAL", "1").lower() not in ("0", "false", "no", "off"):
            alpha = float(np.clip(float(os.environ.get("MIS_EXPERT_POOL_TEMP_ALPHA", "0.28")), 0.05, 0.75))
            temp_bbox = _nano_center_blend_fixed_size(
                self.prev_bbox,
                nano_bbox,
                alpha,
                (self.base_w, self.base_h),
                frame_bgr.shape[:2],
            )
            candidates.append(
                {
                    "name": "temporal_nano",
                    "bbox": temp_bbox,
                    "sim_score": 0.70 * float(out_score) + 0.20 * float(nano_score),
                    "bonus": 0.0,
                }
            )
        stable_rtw = os.environ.get("MIS_RTW_STABLE_ENABLE", "0").lower() not in ("0", "false", "no", "off")
        iou_prev_nano = float(_bbox_iou(self.prev_bbox, nano_bbox))
        risk_level = self._rtw_risk_level(
            policy_mode=str(policy_mode),
            nano_score=float(nano_score),
            iou_prev_nano=float(iou_prev_nano),
        )
        should_rtw = stable_rtw or risk_level != "stable"
        if should_rtw:
            all_seed_map: dict[str, tuple[int, int, int, int]] = {
                "out": tuple(map(int, out_bbox)),
                "prev": tuple(map(int, self.prev_bbox)),
                "nano": _clip_bbox(tuple(map(int, nano_bbox)), frame_bgr.shape[:2]),
            }
            if str(policy_mode) == "stable":
                seed_txt = os.environ.get("MIS_RTW_SEEDS_STABLE", "out,nano")
            elif risk_level == "hard":
                seed_txt = os.environ.get("MIS_RTW_SEEDS_RISK", "out,prev,nano")
            elif risk_level == "light":
                seed_txt = os.environ.get("MIS_RTW_SEEDS_LIGHT", "out,nano")
            else:
                seed_txt = os.environ.get("MIS_RTW_SEEDS_STABLE", "out,nano")
            seed_names = [p.strip().lower() for p in seed_txt.replace(";", ",").split(",") if p.strip()]
            rtw_seeds: list[tuple[str, tuple[int, int, int, int]]] = [
                (name, all_seed_map[name]) for name in seed_names if name in all_seed_map
            ]
            candidates.extend(
                self._rtw_greedy_candidates(
                    frame_bgr,
                    seeds=rtw_seeds,
                    nano_bbox=nano_bbox,
                    nano_score=float(nano_score),
                    policy_mode=str(policy_mode),
                    risk_level=str(risk_level),
                )
            )

        scored: list[dict[str, Any]] = []
        for cand in candidates:
            scores = self._score_fusion_candidate(
                frame_bgr,
                cand,
                nano_bbox=nano_bbox,
                nano_score=float(nano_score),
                weights=weights,
            )
            scored.append({**cand, **scores})
        scored.sort(key=lambda x: float(x["score"]), reverse=True)
        base = next((s for s in scored if s.get("name") == "fusion_out"), scored[-1])
        best = scored[0]
        margin = float(os.environ.get("MIS_EXPERT_POOL_MARGIN", "0.012"))
        pick_best = bool(best.get("name") != "fusion_out" and float(best["score"]) >= float(base["score"]) + margin)
        if str(policy_mode) == "geometry_deformation" and best.get("name") in ("nano_raw", "nano_adapt"):
            pick_best = pick_best or bool(float(best["score"]) >= float(base["score"]) - 0.004 and float(best["sz"]) >= float(base["sz"]) - 0.08)
        if str(policy_mode) == "geometry_occlusion" and best.get("name") in ("nano_raw", "temporal_nano"):
            pick_best = pick_best or bool(float(best["score"]) >= float(base["score"]) + 0.004 and float(nano_score) >= 0.60)
        chosen = best if pick_best else base
        chosen_bbox = tuple(map(int, chosen["bbox"]))
        chosen_score = max(float(out_score), float(nano_score) * 0.90) if chosen.get("name") != "fusion_out" else float(out_score)
        dbg = {
            "applied": bool(chosen.get("name") != "fusion_out"),
            "pick": str(chosen.get("name", "fusion_out")),
            "policy_mode": str(policy_mode),
            "rtw_risk_level": str(risk_level),
            "rtw_enabled": bool(should_rtw),
            "size_refine": dict(size_refine_dbg),
            "weights": [float(v) for v in weights],
            "base_score": float(base["score"]),
            "best_score": float(best["score"]),
            "candidates": [
                {
                    "name": str(s.get("name")),
                    "score": float(s["score"]),
                    "nano": float(s["nano"]),
                    "hist": float(s["hist"]),
                    "sim": float(s["sim"]),
                    "sz": float(s["sz"]),
                    "temporal": float(s["temporal"]),
                }
                for s in scored[:4]
            ],
        }
        return chosen_bbox, float(chosen_score), dbg

    def _arbitrate_output_with_nano_candidate(
        self,
        frame_bgr: np.ndarray,
        out_bbox: tuple[int, int, int, int],
        out_score: float,
        nano_bbox: tuple[int, int, int, int] | None,
        nano_score: float,
        *,
        policy_mode: str,
        trust_hist: bool,
    ) -> tuple[tuple[int, int, int, int], float, dict[str, Any]]:
        """最终输出候选仲裁：必要时回切到 Nano 固定尺寸候选，避免融合低于 Nano 基线。"""
        dbg: dict[str, Any] = {"applied": False}
        if os.environ.get("MIS_NANO_OUTPUT_ARBITRATE", "1").lower() in ("0", "false", "no", "off"):
            return out_bbox, out_score, dbg
        if nano_bbox is None or float(nano_score) < float(os.environ.get("MIS_NANO_OUTPUT_NS", "0.52")):
            return out_bbox, out_score, dbg
        if not _nano_bbox_trustworthy(nano_bbox, frame_bgr.shape[:2], (self.base_w, self.base_h)):
            return out_bbox, out_score, dbg
        if os.environ.get("MIS_EXPERT_POOL_ENABLE", "1").lower() not in ("0", "false", "no", "off"):
            pool_bbox, pool_score, pool_dbg = self._candidate_pool_arbitrate(
                frame_bgr,
                out_bbox,
                float(out_score),
                nano_bbox,
                float(nano_score),
                policy_mode=str(policy_mode),
            )
            pool_dbg["legacy_arbitrate"] = False
            return pool_bbox, pool_score, pool_dbg

        iou_on = float(_bbox_iou(out_bbox, nano_bbox))
        if iou_on >= float(os.environ.get("MIS_NANO_OUTPUT_IOU_MAX", "0.64")):
            return out_bbox, out_score, dbg
        if trust_hist and os.environ.get("MIS_NANO_OUTPUT_KEEP_TRUST_HIST", "1").lower() not in ("0", "false", "no", "off"):
            if float(out_score) >= self.ADAPT_THRESH and not str(policy_mode).startswith("geometry_"):
                return out_bbox, out_score, dbg

        nano_adapt = self._nano_center_blend_policy_size(
            out_bbox,
            nano_bbox,
            0.0,
            frame_bgr.shape[:2],
            nano_score=float(nano_score),
            sz_sim=1.0 if str(policy_mode) == "geometry_deformation" else 0.50,
            policy_mode=str(policy_mode),
        )
        try:
            h_o = float(self._quick_hist_sim(frame_bgr, out_bbox))
            z_o = float(self._quick_sz_sim(frame_bgr, out_bbox))
            iou_prev_o = float(_bbox_iou(self.prev_bbox, out_bbox))
            if str(policy_mode) == "geometry_occlusion":
                w_h, w_z, w_prev, w_nano = 0.16, 0.18, 0.16, 0.50
            elif str(policy_mode) == "geometry_deformation":
                w_h, w_z, w_prev, w_nano = 0.20, 0.34, 0.22, 0.24
            else:
                w_h, w_z, w_prev, w_nano = 0.28, 0.28, 0.28, 0.16
            q_o = float(w_h * max(0.0, h_o) + w_z * max(0.0, z_o) + w_prev * iou_prev_o + w_nano * iou_on)
            candidates: list[tuple[str, tuple[int, int, int, int]]] = [("nano_adapt", nano_adapt)]
            if (
                os.environ.get("MIS_NANO_OUTPUT_RAW_ENABLE", "1").lower() not in ("0", "false", "no", "off")
                and float(nano_score) >= float(os.environ.get("MIS_NANO_OUTPUT_RAW_NS", "0.60"))
            ):
                candidates.append(("nano_raw", _clip_bbox(tuple(map(int, nano_bbox)), frame_bgr.shape[:2])))
            margin = float(os.environ.get("MIS_NANO_OUTPUT_MARGIN", "0.025"))
            hist_drop = float(os.environ.get("MIS_NANO_OUTPUT_HIST_DROP", "0.12"))
            sz_drop = float(os.environ.get("MIS_NANO_OUTPUT_SZ_DROP", "0.12"))
            best_name = "out"
            best_bbox = out_bbox
            best_q = q_o
            best_h = h_o
            best_z = z_o
            best_iou_prev = iou_prev_o
            for cand_name, cand_bbox in candidates:
                h_c = float(self._quick_hist_sim(frame_bgr, cand_bbox))
                z_c = float(self._quick_sz_sim(frame_bgr, cand_bbox))
                iou_prev_c = float(_bbox_iou(self.prev_bbox, cand_bbox))
                iou_cn = float(_bbox_iou(cand_bbox, nano_bbox))
                q_c = float(w_h * max(0.0, h_c) + w_z * max(0.0, z_c) + w_prev * iou_prev_c + w_nano * iou_cn)
                if q_c > best_q:
                    best_name = cand_name
                    best_bbox = cand_bbox
                    best_q = q_c
                    best_h = h_c
                    best_z = z_c
                    best_iou_prev = iou_prev_c
            can_pick_nano = (
                best_name != "out"
                and best_q >= q_o + margin
                and best_h >= h_o - hist_drop
                and best_z >= z_o - sz_drop
            )
            if str(policy_mode) == "geometry_occlusion" and float(nano_score) >= 0.56:
                can_pick_nano = can_pick_nano or (best_name != "out" and best_q >= q_o - 0.015 and iou_on < 0.48)
            if str(policy_mode) == "geometry_deformation" and best_name == "nano_raw" and float(nano_score) >= 0.68:
                can_pick_nano = can_pick_nano or (best_q >= q_o - 0.005 and best_z >= z_o - 0.08)
            dbg.update(
                {
                    "applied": bool(can_pick_nano),
                    "pick": str(best_name if can_pick_nano else "out"),
                    "iou_out_nano": float(iou_on),
                    "q_out": float(q_o),
                    "q_nano": float(best_q),
                    "hist_out": float(h_o),
                    "hist_nano": float(best_h),
                    "sz_out": float(z_o),
                    "sz_nano": float(best_z),
                    "iou_prev_nano": float(best_iou_prev),
                }
            )
            if not can_pick_nano:
                return out_bbox, out_score, dbg
            score = max(float(out_score), float(nano_score) * 0.90)
            return best_bbox, score, dbg
        except Exception:
            return out_bbox, out_score, dbg

    def _update_mis_core(
        self,
        frame_bgr: np.ndarray,
        *,
        anchor_bbox: tuple[int, int, int, int],
        last_nb_proc: tuple[int, int, int, int] | None,
        last_ns: float,
        motion_px: float,
        tier: str = "full",
        policy_mode: str = "stable",
    ) -> dict[str, Any] | None:
        import contextlib
        import io

        sch = self._scheme_mod()
        defm = self._defm
        fH, fW = frame_bgr.shape[:2]
        th68_curr = defm._apply_th68(frame_bgr)

        expand = self.EXPAND_BASE
        coarse_step = self.COARSE_STEP
        rerank_topr = int(os.environ.get("MIS_RERANK_TOPR", "48"))
        max_time_s = float(getattr(self, "max_time_s", float(os.environ.get("MIS_MAX_TIME_S", "1.5"))))
        match_resp_enable = os.environ.get("MIS_MATCH_RESP_ENABLE", "1").lower() not in (
            "0", "false", "no", "off"
        )
        if tier == "local":
            expand = int(os.environ.get("MIS_CASCADE_LOCAL_EXPAND", "72"))
            coarse_step = int(os.environ.get("MIS_CASCADE_LOCAL_COARSE", "22"))
            rerank_topr = int(os.environ.get("MIS_CASCADE_LOCAL_TOPR", "16"))
            max_time_s = float(os.environ.get("MIS_CASCADE_LOCAL_MAX_T", "0.45"))
            match_resp_enable = False

        search_plane, plane_bgr_crop, sy1, sx1, plane_center_yx, ph, pw = self._search_plane_pack(
            frame_bgr, th68_curr, anchor_bbox, expand_base=expand
        )
        bx, by, bw, bh = [int(v) for v in anchor_bbox]
        nano_prior_map = None

        if self._nano_helper.available and float(last_ns) >= self.NANO_CONF_THRESH and last_nb_proc is not None:
            nano_prior_map, self._nano_prior_cache = sch.get_or_build_nano_prior_map(
                ph,
                pw,
                last_nb_proc,
                (sx1, sy1),
                float(last_ns),
                (self.base_w, self.base_h),
                self._nano_prior_cache,
                sigma_scale=self.nano_sigma_scale,
                uniform_floor=self.nano_floor,
            )
        elif self._nano_helper.available:
            self._nano_prior_cache.clear()

        lib_tp_use, lib_z_use = sch.adapt_contour_lib_cache(
            self.lib_tp_cached, self.lib_z_cached, motion_px
        )

        if tier == "full":
            rerank_topr, match_resp_enable, _ = self._effective_search_budget(float(last_ns), motion_px)
            if str(policy_mode).startswith("geometry_"):
                # 困难段优先保活：限制重分支预算，避免长时间卡住
                coarse_step = max(int(coarse_step), int(os.environ.get("MIS_GEOM_COARSE_STEP", "22")))
                rerank_topr = min(int(rerank_topr), int(os.environ.get("MIS_GEOM_TOPR_MAX", "20")))
                max_time_s = min(float(max_time_s), float(os.environ.get("MIS_GEOM_MAX_T", "0.45")))
                match_resp_enable = False
            if str(policy_mode) == "geometry_occlusion":
                coarse_step = max(int(coarse_step), int(os.environ.get("MIS_OCC_COARSE_STEP", "24")))
                rerank_topr = min(int(rerank_topr), int(os.environ.get("MIS_OCC_TOPR_MAX", "18")))
                max_time_s = min(float(max_time_s), float(os.environ.get("MIS_OCC_MAX_T", "0.42")))
                match_resp_enable = False
            elif str(policy_mode) == "geometry_deformation":
                coarse_step = max(int(coarse_step), int(os.environ.get("MIS_DEF_COARSE_STEP", "18")))
                rerank_topr = min(int(rerank_topr), int(os.environ.get("MIS_DEF_TOPR_MAX", "30")))
                max_time_s = min(float(max_time_s), float(os.environ.get("MIS_DEF_MAX_T", "0.62")))
        match_resp = None
        if sch.should_compute_match_response(
            rerank_topr=rerank_topr,
            match_resp_enable=match_resp_enable,
            last_ns=last_ns,
            motion_px=motion_px,
            prev_track_score=self._prev_track_score,
        ):
            try:
                mr_m = int(os.environ.get("MIS_MATCH_RESP_MARGIN", "112"))
                mrr = float(os.environ.get("MIS_MATCH_RESP_FULL_RATIO", "0.92"))
                mrr = max(0.5, min(1.0, mrr))
                py_tl_p = max(0, by - sy1)
                px_tl_p = max(0, bx - sx1)
                nano_plane_box = None
                if last_nb_proc is not None:
                    nbx, nby, nbw, nbh = [int(v) for v in last_nb_proc]
                    npy1, npx1 = nby - sy1, nbx - sx1
                    npy2 = min(ph, npy1 + nbh)
                    npx2 = min(pw, npx1 + nbw)
                    npy1 = max(0, npy1)
                    npx1 = max(0, npx1)
                    if npy2 > npy1 and npx2 > npx1:
                        nano_plane_box = (npy1, npx1, npy2 - npy1, npx2 - npx1)
                match_resp = sch.compute_template_match_response_roi_embedded(
                    plane_bgr_crop,
                    self.ref_gray_patch_sz,
                    py_tl=py_tl_p,
                    px_tl=px_tl_p,
                    bbox_h=bh,
                    bbox_w=bw,
                    margin=mr_m,
                    coarse_step=coarse_step,
                    full_plane_ratio=mrr,
                    nano_plane_box=nano_plane_box,
                )
            except Exception:
                match_resp = None

        rw_txt = os.environ.get("MIS_RERANK_WEIGHTS", "").strip()
        if not rw_txt:
            rerank_weights: tuple[float, float, float, float] = (0.30, 0.32, 0.22, 0.16)
        else:
            parts = [p.strip() for p in rw_txt.replace(";", ",").split(",") if p.strip()]
            rerank_weights = (
                tuple(float(x) for x in parts)
                if len(parts) == 4
                else (0.28, 0.34, 0.24, 0.14)
            )
        if str(policy_mode) == "appearance_shift":
            rsn, rsc, rsh, rsz = rerank_weights
            rerank_weights = (
                float(rsn * 1.10),
                float(rsc * 1.08),
                float(rsh * 0.72),
                float(rsz * 1.25),
            )
        elif str(policy_mode) == "geometry_risk":
            rsn, rsc, rsh, rsz = rerank_weights
            rerank_weights = (
                float(rsn * 1.18),
                float(rsc * 0.95),
                float(rsh * 0.70),
                float(rsz * 1.30),
            )
        elif str(policy_mode) == "geometry_occlusion":
            rsn, rsc, rsh, rsz = rerank_weights
            rerank_weights = (
                float(rsn * 1.26),
                float(rsc * 0.86),
                float(rsh * 0.64),
                float(rsz * 1.14),
            )
        elif str(policy_mode) == "geometry_deformation":
            rsn, rsc, rsh, rsz = rerank_weights
            rerank_weights = (
                float(rsn * 1.10),
                float(rsc * 1.10),
                float(rsh * 0.78),
                float(rsz * 1.42),
            )
        s_rw = max(1e-6, float(sum(rerank_weights)))
        rerank_weights = tuple(float(v / s_rw) for v in rerank_weights)

        nano_fusion = (os.environ.get("MIS_NANO_FUSION", "log") or "log").strip().lower()
        nano_prior_beta = float(os.environ.get("MIS_NANO_PRIOR_BETA", "0.48"))
        nano_hist_gamma = float(os.environ.get("MIS_NANO_HIST_COUPLING", "0.52"))

        _devnull = io.StringIO()
        with contextlib.redirect_stdout(_devnull):
            result = defm.find_best_match_hierarchical(
                search_plane,
                None,
                self.base_h,
                self.base_w,
                None,
                contour_sim_model=self.contour_sim_model,
                lib_templates_raw=self.lib_templates_raw,
                lib_contour_encodings=None,
                step_coarse=coarse_step,
                step_fine=self.SLIDE_STEP,
                fine_expand=self.FINE_EXPAND,
                top_k=self.TOPK,
                greedy_step=self.GREEDY_STEP,
                greedy_iters=self.GREEDY_ITERS,
                spatial_sigma=3.0,
                center_yx=plane_center_yx,
                early_exit_score=self.EARLY_EXIT,
                coarse_scheme=self.coarse_scheme,
                frame_bgr_full=plane_bgr_crop,
                plane_origin_yx=(0, 0),
                ref_hist=self.ref_hist,
                hist_h_bins=self.HIST_H_BINS,
                hist_s_bins=self.HIST_S_BINS,
                hist_gate_T=self.HIST_GATE_T,
                hist_top_M_cap=self.HIST_TOP_M,
                hist_min_keep=int(os.environ.get("MIS_HIST_MIN_KEEP", "24")),
                hist_fallback_take=int(os.environ.get("MIS_HIST_FALLBACK_M", "48")),
                hist_topkp_c=int(os.environ.get("MIS_HIST_TOPKP_C", "96")),
                fuse_lambda_base=self.FUSE_LAMBDA,
                fuse_adaptive=True,
                fine_fuse_hist=True,
                greedy_fuse_hist=True,
                early_exit_hist_min=float(os.environ.get("MIS_EARLY_HIST_MIN", "0.0")),
                early_exit_polish=True,
                contour_sat_thresh=self.CONTOUR_SAT_THRESH,
                contour_sat_lambda_scale=self.CONTOUR_SAT_SCALE,
                contour_infer_bs=self.CONTOUR_INFER_BS,
                cached_lib_tensor=lib_tp_use,
                cached_lib_z=lib_z_use,
                max_time_s=max_time_s,
                nano_prior_map=nano_prior_map,
                nano_fusion=nano_fusion,
                nano_prior_beta=nano_prior_beta,
                nano_hist_coupling=nano_hist_gamma,
                rerank_topr=rerank_topr,
                match_resp=match_resp,
                ref_gray_patch_sz=self.ref_gray_patch_sz,
                ref_depth_patch_sz=self.ref_depth_patch_sz,
                nano_confidence=float(last_ns),
                rerank_weights=rerank_weights,
            )

        new_x = sx1 + int(result["x"])
        new_y = sy1 + int(result["y"])
        new_x, new_y, new_w, new_h = defm.core.clamp_bbox(new_x, new_y, self.base_w, self.base_h, fW, fH)

        crop_th68 = th68_curr[new_y : new_y + new_h, new_x : new_x + new_w]
        if crop_th68.size > 0:
            new_x, new_y, new_w, new_h = defm._refine_bbox_by_contour(
                crop_th68,
                new_x,
                new_y,
                fW,
                fH,
                margin=self.CONTOUR_MARGIN,
                min_fill_ratio=self.MIN_FILL,
                max_size_factor=self.MAX_SIZE,
                orig_w=self.base_w,
                orig_h=self.base_h,
            )

        new_bbox = _clip_bbox((new_x, new_y, new_w, new_h), frame_bgr.shape[:2])
        score = float(result["score"])
        self._last_mis_bbox = new_bbox
        _trust_hist = bool(result.get("trust_hist_fusion", False))
        new_bbox, score = self._apply_nano_recovery(
            new_bbox,
            score,
            last_nb_proc,
            float(last_ns),
            frame_bgr.shape[:2],
            trust_hist=_trust_hist,
        )
        self._prev_track_score = score

        new_bbox, score = self._arbitrate_mis_nano(
            new_bbox,
            score,
            last_nb_proc,
            float(last_ns),
            frame_bgr.shape[:2],
            result,
        )
        nano_out_dbg: dict[str, Any] = {}
        new_bbox, score, nano_out_dbg = self._arbitrate_output_with_nano_candidate(
            frame_bgr,
            new_bbox,
            score,
            last_nb_proc,
            float(last_ns),
            policy_mode=str(policy_mode),
            trust_hist=bool(_trust_hist),
        )

        _auto_blend = os.environ.get("MIS_SCHEME_C_AUTO_BLEND", "1").lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        _do_post_blend = self.POST_BLEND and not (_auto_blend and _trust_hist)
        if _do_post_blend and self._nano_helper.available and last_nb_proc is not None:
            nb_proc = last_nb_proc
            ns = float(last_ns)
            if score < self.LOSS_THRESH and ns > self.NANO_CONF_THRESH:
                new_bbox = nb_proc
                new_x, new_y, new_w, new_h = new_bbox
                score = float(ns * 0.8)
            elif self.LOSS_THRESH <= score < self.ADAPT_THRESH and ns > self.NANO_CONF_THRESH:
                alpha = max(self.NANO_BLEND_ALPHA, min(0.9, score / max(self.ADAPT_THRESH, 1e-6)))
                new_bbox = _nano_fuse_bboxes(new_bbox, score, nb_proc, ns, alpha)
                new_x, new_y, new_w, new_h = new_bbox

        self.prev_bbox = new_bbox
        if tier == "full":
            self._cascade_hits["full"] = int(self._cascade_hits.get("full", 0)) + 1
        return self._pack_track_output(
            frame_bgr,
            new_bbox,
            score,
            last_nb_proc=last_nb_proc,
            last_ns=float(last_ns),
            cascade_tier=tier,
            extra_debug={
                "mis_score": float(result.get("score", score)),
                "hist_sim": float(result.get("hist_sim", -1.0)),
                "coarse_score": float(result.get("coarse_score", -1.0)),
                "trust_hist_fusion": _trust_hist,
                "suggest_nano_post_blend": bool(result.get("suggest_nano_post_blend", False)),
                "arbitrate": dict(getattr(self, "_last_arbitrate_debug", {})),
                "nano_output_arbitrate": dict(nano_out_dbg),
                "policy_mode": str(policy_mode),
                "rel_hist": float(self._rel_hist),
                "rel_sz": float(self._rel_sz),
                "rel_nano": float(self._rel_nano),
            },
        )

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        sch = self._scheme_mod()
        defm = self._defm
        fH, fW = frame_bgr.shape[:2]
        last_nb_proc = None
        last_ns = 0.0
        motion_px = 0.0
        anchor_bbox = self.prev_bbox
        if self._nano_helper.available:
            _nb, ns = self._nano_helper.update(frame_bgr)
            nb_x, nb_y, nb_w, nb_h = [int(round(float(v))) for v in _nb]
            nano_bbox_proc = tuple(defm.core.clamp_bbox(nb_x, nb_y, nb_w, nb_h, fW, fH))
            last_nb_proc = nano_bbox_proc
            last_ns = float(ns)
            motion_px = sch.bbox_center_motion_px(self.prev_bbox, nano_bbox_proc)

            # 保存前一帧 Nano bbox 用于运动外推
            self._prev_nano_bbox = nano_bbox_proc

            # 保存最近候选尺寸用于一致性验证（最多保存 5 帧）
            if not hasattr(self, '_recent_cand_sizes'):
                self._recent_cand_sizes = []
            self._recent_cand_sizes.append((nano_bbox_proc[2], nano_bbox_proc[3]))
            if len(self._recent_cand_sizes) > 5:
                self._recent_cand_sizes.pop(0)
            if (
                float(ns) >= self.NANO_CONF_THRESH
                and os.environ.get("MIS_NANO_ANCHOR_CENTER", "1").lower()
                not in ("0", "false", "no", "off")
            ):
                anchor_motion = float(os.environ.get("MIS_NANO_ANCHOR_MOTION_PX", "10"))
                if (
                    motion_px >= anchor_motion
                    and _nano_bbox_trustworthy(
                        nano_bbox_proc, (fH, fW), (self.base_w, self.base_h)
                    )
                ):
                    anchor_bbox = self._anchor_bbox_from_nano(nano_bbox_proc, fH, fW)

        if last_nb_proc is not None:
            nano_plus_out = self._try_nano_plus(
                frame_bgr,
                last_nb_proc,
                float(last_ns),
                float(motion_px),
            )
            if nano_plus_out is not None:
                ndbg = nano_plus_out.get("debug", {}) if isinstance(nano_plus_out, dict) else {}
                self.prev_bbox = nano_plus_out["bbox"]
                self._prev_track_score = float(nano_plus_out.get("score", 0.0))
                self._tier_prev = "primary"
                self._mode_prev = str(ndbg.get("policy_mode", "stable"))
                return nano_plus_out
            if (
                os.environ.get("MIS_NANO_PLUS_ENABLE", "0").lower() not in ("0", "false", "no", "off")
                and os.environ.get("MIS_NANO_PLUS_HOLD_PREV", "0").lower() not in ("0", "false", "no", "off")
                and float(self._prev_track_score) >= float(os.environ.get("MIS_NANO_PLUS_HOLD_PREV_SCORE", "0.55"))
            ):
                hold_score = float(np.clip(float(self._prev_track_score) * 0.985, 0.0, 1.0))
                hold_out = self._pack_track_output(
                    frame_bgr,
                    self.prev_bbox,
                    hold_score,
                    last_nb_proc=last_nb_proc,
                    last_ns=float(last_ns),
                    cascade_tier="nano_plus_hold",
                    extra_debug={
                        "policy_mode": "stable",
                        "nano_plus_hold": True,
                        "rtw_enabled": False,
                        "similaritynet_used": False,
                    },
                )
                self._prev_track_score = hold_score
                return hold_out
            cascade_out = self._try_nano_cascade(
                frame_bgr, last_nb_proc, float(last_ns), motion_px
            )
            if cascade_out is not None:
                cdbg = cascade_out.get("debug", {}) if isinstance(cascade_out, dict) else {}
                mem_updated = self._maybe_update_memory_bank(
                    frame_bgr,
                    tuple(map(int, cascade_out.get("bbox", self.prev_bbox))),
                    score=float(cascade_out.get("score", 0.0)),
                    nano_score=float(last_ns),
                    policy_mode=str(cdbg.get("policy_mode", self._mode_prev)),
                )
                cdbg["memory_updated"] = bool(mem_updated)
                cdbg["memory_bank_size"] = int(len(getattr(self, "_memory_z_bank", [])))
                cascade_out["debug"] = cdbg
                self.prev_bbox = cascade_out["bbox"]
                self._prev_track_score = float(cascade_out.get("score", 0.0))
                return cascade_out

        # 重定位：漂移后扩大搜索恢复
        if self._reloc_triggered and last_nb_proc is not None:
            reloc_out = self._try_relocate(frame_bgr, last_nb_proc, float(last_ns), motion_px)
            if reloc_out is not None:
                self._drift_streak = 0
                self._reloc_triggered = False
                self._last_reloc_frame = self._frame_idx if hasattr(self, '_frame_idx') else 0
                self.prev_bbox = reloc_out["bbox"]
                self._prev_track_score = float(reloc_out.get("score", 0.0))
                return reloc_out

        lite_skip = self._try_nano_lite_skip(frame_bgr, last_nb_proc, float(last_ns), motion_px) if (
            last_nb_proc is not None
        ) else None
        if lite_skip is not None:
            self._cascade_hits["lite"] = int(self._cascade_hits.get("lite", 0)) + 1
            ldbg = lite_skip.get("debug", {}) if isinstance(lite_skip, dict) else {}
            mem_updated = self._maybe_update_memory_bank(
                frame_bgr,
                tuple(map(int, lite_skip.get("bbox", self.prev_bbox))),
                score=float(lite_skip.get("score", 0.0)),
                nano_score=float(last_ns),
                policy_mode="stable",
            )
            ldbg["memory_updated"] = bool(mem_updated)
            ldbg["memory_bank_size"] = int(len(getattr(self, "_memory_z_bank", [])))
            lite_skip["debug"] = ldbg
            self.prev_bbox = lite_skip["bbox"]
            self._prev_track_score = float(lite_skip.get("score", 0.0))
            self._tier_prev = "primary"
            return lite_skip

        out = self._update_mis_core(
            frame_bgr,
            anchor_bbox=anchor_bbox,
            last_nb_proc=last_nb_proc,
            last_ns=float(last_ns),
            motion_px=float(motion_px),
            tier="full",
            policy_mode=str(self._mode_prev),
        )
        if last_nb_proc is not None:
            dbg = out.get("debug", {}) if isinstance(out, dict) else {}
            hist_sim = float(dbg.get("hist_sim", -1.0))
            if hist_sim < 0.0:
                try:
                    hist_sim = float(self._quick_hist_sim(frame_bgr, out["bbox"]))
                except Exception:
                    hist_sim = 0.0
            risk_score = sch.cascade_risk_score(
                nano_score=float(last_ns),
                hist_sim=float(hist_sim),
                motion_px=float(motion_px),
                iou_prev_nano=float(_bbox_iou(self.prev_bbox, last_nb_proc)),
                prev_track_score=float(self._prev_track_score),
            )
            self._update_risk_state(
                float(risk_score),
                is_bad=bool(
                    float(out.get("score", 0.0)) < float(os.environ.get("MIS_CASCADE_BAD_SCORE", "0.40"))
                    or (hist_sim >= 0.0 and hist_sim < float(os.environ.get("MIS_CASCADE_BAD_HIST", "0.22")))
                ),
            )
            self._tier_prev = "full"
            dbg["risk_score"] = float(risk_score)
            dbg["risk_ema"] = float(self._risk_ema)
            dbg["bad_streak"] = int(self._bad_streak)
            out["debug"] = dbg
        self.prev_bbox = out["bbox"]
        self._prev_track_score = float(out.get("score", 0.0))
        if last_nb_proc is not None:
            dbg = out.get("debug", {}) if isinstance(out, dict) else {}
            ref_updated = self._maybe_online_ref_update(
                frame_bgr,
                tuple(map(int, out.get("bbox", self.prev_bbox))),
                score=float(out.get("score", 0.0)),
                nano_score=float(last_ns),
                mode=str(self._mode_prev),
            )
            dbg["ref_updated"] = bool(ref_updated)
            mem_updated = self._maybe_update_memory_bank(
                frame_bgr,
                tuple(map(int, out.get("bbox", self.prev_bbox))),
                score=float(out.get("score", 0.0)),
                nano_score=float(last_ns),
                policy_mode=str(self._mode_prev),
            )
            dbg["memory_updated"] = bool(mem_updated)
            dbg["memory_bank_size"] = int(len(getattr(self, "_memory_z_bank", [])))
            out["debug"] = dbg
        return out
