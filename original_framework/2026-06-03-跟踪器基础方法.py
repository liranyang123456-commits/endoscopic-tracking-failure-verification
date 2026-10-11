#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""【冻结】Ours 跟踪器基础类型与 bbox 工具 — 2026-06-03"""
from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

def _clip_bbox(bbox: tuple[int, int, int, int], shape_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    x, y, bw, bh = [int(v) for v in bbox]
    x = max(0, min(x, max(0, w - 1)))
    y = max(0, min(y, max(0, h - 1)))
    bw = max(1, min(max(1, bw), max(1, w - x)))
    bh = max(1, min(max(1, bh), max(1, h - y)))
    return x, y, bw, bh


def _bbox_to_mask(shape_hw: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = np.zeros((int(shape_hw[0]), int(shape_hw[1])), dtype=np.uint8)
    x, y, w, h = [int(v) for v in bbox]
    out[y:y + h, x:x + w] = 255
    return out


def _from_center_bbox(
    cx: float,
    cy: float,
    w: float,
    h: float,
    shape_hw: tuple[int, int],
) -> tuple[int, int, int, int]:
    return _clip_bbox(
        (int(round(float(cx) - float(w) * 0.5)), int(round(float(cy) - float(h) * 0.5)), int(round(w)), int(round(h))),
        shape_hw,
    )


def _dedup_bbox_candidates(
    cands: list[tuple[str, tuple[int, int, int, int]]],
) -> list[tuple[str, tuple[int, int, int, int]]]:
    seen: set[tuple[int, int, int, int]] = set()
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for name, bbox in cands:
        bb = tuple(map(int, bbox))
        if bb in seen:
            continue
        seen.add(bb)
        out.append((str(name), bb))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 评估指标
# ═══════════════════════════════════════════════════════════════════════════════
def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = [int(v) for v in a]
    bx, by, bw, bh = [int(v) for v in b]
    ix1 = max(ax, bx); iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw); iy2 = min(ay + ah, by + bh)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = max(1, aw * ah + bw * bh - inter)
    return float(inter) / float(union)


def _bbox_area_ratio(
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
) -> float:
    _, _, bw, bh = (int(v) for v in bbox)
    fH, fW = int(frame_shape[0]), int(frame_shape[1])
    return float(max(0, bw) * max(0, bh)) / float(max(1, fH * fW))


def _bbox_size_ratio_ok(
    bbox: tuple[int, int, int, int],
    ref_wh: tuple[int, int],
    *,
    min_ratio: float = 0.30,
    max_ratio: float = 3.0,
) -> bool:
    _, _, bw, bh = (int(v) for v in bbox)
    rw, rh = max(1, int(ref_wh[0])), max(1, int(ref_wh[1]))
    return (
        min_ratio < bw / rw < max_ratio
        and min_ratio < bh / rh < max_ratio
    )


def _nano_bbox_trustworthy(
    nano_bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int],
    ref_wh: tuple[int, int],
    *,
    max_area_frac: float = 0.42,
) -> bool:
    if not _bbox_size_ratio_ok(nano_bbox, ref_wh):
        return False
    return _bbox_area_ratio(nano_bbox, frame_shape) <= max_area_frac


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    ma = np.asarray(a, dtype=np.uint8) > 0
    mb = np.asarray(b, dtype=np.uint8) > 0
    inter = int(np.count_nonzero(ma & mb))
    union = int(np.count_nonzero(ma | mb))
    return float(inter) / float(union) if union > 0 else 0.0


def _center_error(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    return np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# 基础跟踪器接口
# ═══════════════════════════════════════════════════════════════════════════════
class BaseTracker:
    name: str = "base"
    label: str = "Base"

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError

    def reset(self) -> None:
        """在序列切换时重置状态（可选覆盖）。"""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Ours — Deformation_F_T_C_G_Experi_hist_AB 封装
# ═══════════════════════════════════════════════════════════════════════════════
# 模块级懒加载单例（避免多次导入和重复加载模型）
_DEFM_AB_MODULE: Any = None
_DEFM_AB_LOAD_DONE: bool = False


def _load_defm_ab_once() -> Any:
    global _DEFM_AB_MODULE, _DEFM_AB_LOAD_DONE
    if _DEFM_AB_LOAD_DONE:
        return _DEFM_AB_MODULE
    try:
        import importlib

        defm_ab = importlib.import_module("2026-06-03-HistAB核心方法")
        _DEFM_AB_MODULE = defm_ab
        print("[LOAD] Deformation_F_T_C_G_Experi_hist_AB 加载成功")
    except Exception as e:
        print(f"[WARN] 无法加载 Deformation_F_T_C_G_Experi_hist_AB: {e}")
        _DEFM_AB_MODULE = None
    _DEFM_AB_LOAD_DONE = True
    return _DEFM_AB_MODULE
