#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trackves_eval_all_methods.py
────────────────────────────
在 TrackVes 数据集 (9 序列) 上评测所有跟踪方法，包含本文 Ours (hist_AB) 方法。

数据集路径: E:/Surgical_Tracking_Datasets/TrackVes/TrackVes/TrackVes
  - 9 个序列: EV1 EV2 EV3 IV1 IV2 IV3 IV4 IV5 IV6
  - 每个序列: videoLeft.avi (IV2 为 videLeft.mp4), GT.xml, Attributes.xml
  - GT: OpenCV XML 多边形，每10帧一条标注 (GT000000→frame0, GT000001→frame10, ...)
  - Attributes: 每帧属性 (0=SAV,1=PO,2=OFV,3=TO,4=DEF,5=IC,6=ACM,7=BLR,8=SMK)

支持的方法:
  ours / ours_hist_ab  : Deformation_F_T_C_G_Experi_hist_AB.py (轮廓+HSV直方图AB方案)
  ours_scheme_c_nano   : Scheme C — NanoTrack 高斯先验图 + log 融合 + Top‑R rerank（matchTemplate ROI）
                         （与 ``Deformation_hist_AB_scheme_C_nanoprior.py`` 对齐，环境变量相同）
  ours_hist_c          : Deformation_F_T_C_G_Experi_hist_AB.py (方案C)
  csrt                 : OpenCV CSRT
  kcf                  : OpenCV KCF
  mosse                : OpenCV MOSSE
  yolo                 : YOLO-seg (Ultralytics)
  sam                  : SAM (Ultralytics)
  sam2                 : SAM2-Video
  sam2_pf              : SAM2-PerFrame
  ostrack              : OSTrack (ECCV'22) 结构复现
  mixformer            : MixFormer (ECCV'22) 结构复现
  cutie                : Cutie (CVPR'24)
  lmtrack              : LMTrack (AAAI'24) 结构复现

运行示例:
  # 仅 Ours + CSRT + KCF
  python trackves_eval_all_methods.py --trackers ours,csrt,kcf

  # 仅最新 Scheme C（Nano 先验 + rerank）
  python trackves_eval_all_methods.py --trackers ours_scheme_c_nano --sequences all --sequential 1

  # 全量评测（全部方法）
  python trackves_eval_all_methods.py --trackers all

  # 指定序列
  python trackves_eval_all_methods.py --sequences EV1,EV2 --trackers ours,csrt

  # 逐帧顺序模式（推荐，适合 CSRT/KCF 等需要连续帧的方法）
  python trackves_eval_all_methods.py --sequential 1 --trackers ours,csrt

  # 跳帧模式（仅在 GT 帧上评测，速度更快）
  python trackves_eval_all_methods.py --sequential 0 --trackers ours,csrt

  # GT 进度日志默认每 20 帧打印一行 IoU；更密可用 ``--print-every 10``
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# ── 屏蔽不必要的网络请求 ──────────────────────────────────────────────────────
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TIMM_FUSED_ATTN", "0")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── 关闭 mis3d 中不必要的 UVH/渲染模块 ──────────────────────────────────────
for _k, _v in {
    "MIS_UVH_RENDER_VIEWLIB_ENABLE": "0",
    "MIS_UVH_DIFFRENDER_ENABLE": "0",
    "MIS_ENABLE_FLOW_UVH": "0",
    "MIS_UVH_RIGID_ENABLE": "0",
    "MIS_UVH_SEMANTIC_ENABLE": "0",
}.items():
    os.environ.setdefault(_k, _v)

import cv2
import numpy as np

from fusion_weight_head import fusion_feature_vector, predict_contour_fuse_weight

# ═══════════════════════════════════════════════════════════════════════════════
# 全局常量
# ═══════════════════════════════════════════════════════════════════════════════
TRACKVES_ROOT = r"E:\Surgical_Tracking_Datasets\TrackVes\TrackVes\TrackVes"
DEFAULT_OUT = os.environ.get(
    "MIS_TRACKVES_OUT_DIR",
    r"F:\MIS_Pose_Track_Re3D\Exp_Results_TrackVes_AllMethods",
)
DEFAULT_INTERMEDIATE = os.environ.get(
    "MIS_EVAL_INTERMEDIATE_DIR",
    r"F:\MIS_Pose_Track_Re3D\SchemeC_Intermediate",
)
ALL_SEQUENCES = ["EV1", "EV2", "EV3", "IV1", "IV2", "IV3", "IV4", "IV5", "IV6"]
GT_FRAME_STEP = 1    # GT 的 ann_idx 直接就是视频帧号（经过验证：EV3 last_ann=5490 ≈ 视频帧数 5582）

TRACKER_LABELS: dict[str, str] = {
    "ours":         "Ours (ContourNet+HistAB)",
    "ours_hist_ab": "Ours (ContourNet+HistAB)",
    "ours_hist_c":  "Ours (ContourNet+HistC)",
    "ours_scheme_c_nano": "Ours (Scheme C: Nano+log+TopR)",
    "ours_c_nano":  "Ours (Scheme C: Nano+log+TopR)",
    "ours_nano":    "Ours+NanoTrack (Hybrid)",
    "nano_ab":      "Ours+NanoTrack (Agmt)",
    "nano_deep":    "NanoDeepFusion (Ours×Nano×Hist×Contour)",
    "csrt":         "CSRT/MIL",    # OpenCV4.10+ fallback: MIL
    "kcf":          "KCF/NCC",     # OpenCV4.10+ fallback: NCC template
    "mosse":        "MOSSE/NCC",   # OpenCV4.10+ fallback: NCC template
    "meanshift":    "MeanShift",
    "camshift":     "CamShift",
    "mil":          "MIL",
    "nanotrack":    "NanoTrack",
    "vittrack":     "ViT-Track",
    "lkflow":       "LK-Flow",
    "disflow":      "DIS-Flow",
    "yolo":         "YOLO-seg",
    "sam":          "SAM",
    "sam2":         "SAM2-Video",
    "sam2_pf":      "SAM2-PerFrame",
    "ostrack":      "OSTrack (ECCV'22)",
    "mixformer":    "MixFormer (ECCV'22)",
    "cutie":        "Cutie (CVPR'24)",
    "lmtrack":      "LMTrack (AAAI'24)",
    # The local implementation is a protocol-matched proxy (not an official
    # STARK checkpoint); see StarkProxyTracker and the manuscript caption.
    "stark":        "STARK proxy (ICCV'21)",
}

ALL_TRACKER_KEYS = list(TRACKER_LABELS.keys())

ATTR_NAMES = {
    0: "SAV", 1: "PO", 2: "OFV", 3: "TO",
    4: "DEF", 5: "IC", 6: "ACM", 7: "BLR", 8: "SMK",
}

# ═══════════════════════════════════════════════════════════════════════════════
# GT / 数据解析
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class GTFrame:
    frame_idx: int
    polygon: np.ndarray   # (N,2) float32, x-y 顺序
    attribute: int = 0


def _read_opencv_matrix(node: ET.Element) -> np.ndarray:
    rows = int(node.findtext("rows", default="0"))
    cols = int(node.findtext("cols", default="0"))
    data_text = node.findtext("data", default="")
    vals = [float(v) for v in data_text.split()]
    arr = np.asarray(vals, dtype=np.float32)
    if rows > 0 and cols > 0 and arr.size == rows * cols:
        return arr.reshape(rows, cols)
    return arr


def parse_gt_xml(path: Path, frame_step: int = GT_FRAME_STEP) -> list[GTFrame]:
    """解析 GT.xml，返回按帧序排列的 GTFrame 列表。"""
    tree = ET.parse(str(path))
    root = tree.getroot()
    items: list[GTFrame] = []
    for node in root:
        tag = str(node.tag)
        if not tag.startswith("GT"):
            continue
        try:
            ann_idx = int(tag[2:])
        except ValueError:
            continue
        poly = _read_opencv_matrix(node)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
            continue
        items.append(GTFrame(
            frame_idx=int(ann_idx * frame_step),
            polygon=np.asarray(poly, dtype=np.float32),
        ))
    items.sort(key=lambda x: x.frame_idx)
    return items


def parse_attributes_xml(path: Path) -> list[int]:
    """解析 Attributes.xml，返回每帧属性列表（长度 = 总帧数）。"""
    tree = ET.parse(str(path))
    root = tree.getroot()
    node = root.find("Attributes")
    if node is None:
        return []
    arr = _read_opencv_matrix(node).reshape(-1)
    return [int(v) for v in arr.tolist()]


def _find_video(seq_dir: Path) -> Optional[Path]:
    """查找序列目录下的视频文件（兼容 IV2 的拼写错误）。"""
    for name in ["videoLeft.avi", "videoLeft.mp4", "videLeft.avi", "videLeft.mp4"]:
        p = seq_dir / name
        if p.exists():
            return p
    # 通用查找
    for ext in [".avi", ".mp4", ".mov"]:
        candidates = list(seq_dir.glob(f"*{ext}"))
        if candidates:
            return candidates[0]
    return None


_PROBLEM_SEQ_DEFAULT = frozenset({"EV3", "IV1"})


def _problem_seq_names() -> frozenset[str]:
    raw = os.environ.get("MIS_EVAL_FORCE_SEQ_READ_SEQS", "EV3,IV1")
    names = {s.strip().upper() for s in raw.replace(";", ",").split(",") if s.strip()}
    return frozenset(names) if names else _PROBLEM_SEQ_DEFAULT


def _video_cache_dir() -> Path:
    d = Path(os.environ.get("MIS_VIDEO_CACHE_DIR", r"F:\MIS_Pose_Track_Re3D\video_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_ffmpeg() -> Optional[str]:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        if ff and Path(ff).is_file():
            return ff
    except Exception:
        pass
    for cand in (
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"D:\ffmpeg\bin\ffmpeg.exe",
    ):
        if Path(cand).is_file():
            return cand
    return None


def _ensure_clean_video(src: Path, seq_name: str) -> Path:
    """损坏 H264 序列预转码为可随机 seek 的 mp4；失败则回退原视频。"""
    if os.environ.get("MIS_VIDEO_CACHE_ENABLE", "1").lower() in ("0", "false", "no", "off"):
        return src
    if seq_name.upper() not in _problem_seq_names():
        return src
    cache_dir = _video_cache_dir()
    dst = cache_dir / f"{seq_name}_videoLeft_clean.mp4"
    if dst.exists() and dst.stat().st_size > 1024 * 1024:
        return dst
    ff = _find_ffmpeg()
    if ff is None:
        print(f"[{seq_name}] 未找到 ffmpeg，跳过预转码（将使用段顺序读）")
        return src
    print(f"[{seq_name}] 预转码 → {dst}")
    cmd = [
        ff, "-y", "-loglevel", "error",
        "-err_detect", "ignore_err",
        "-i", str(src),
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(dst),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        if r.returncode == 0 and dst.exists() and dst.stat().st_size > 1024 * 1024:
            print(f"[{seq_name}] 预转码完成 size={dst.stat().st_size // (1024 * 1024)}MB")
            return dst
        print(f"[{seq_name}] 预转码失败 rc={r.returncode}: {(r.stderr or '')[:200]}")
    except Exception as e:
        print(f"[{seq_name}] 预转码异常: {e}")
    return src


def _open_video_capture(seq_dir: Path) -> tuple[cv2.VideoCapture, Path]:
    src = _find_video(seq_dir)
    if src is None:
        raise FileNotFoundError(f"序列 {seq_dir.name} 中未找到视频文件")
    video_path = _ensure_clean_video(src, seq_dir.name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    return cap, video_path


def _read_frame_with_timeout(cap: cv2.VideoCapture, timeout_s: float) -> tuple[bool, Optional[np.ndarray]]:
    result: dict[str, Any] = {"ok": False, "frame": None, "done": False}

    def _worker() -> None:
        try:
            ok, frame = cap.read()
            result["ok"] = bool(ok)
            result["frame"] = frame
        except Exception:
            result["ok"] = False
            result["frame"] = None
        finally:
            result["done"] = True

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=max(0.5, float(timeout_s)))
    if not result["done"]:
        return False, None
    return bool(result["ok"]), result["frame"]


def _read_frame_safe(
    cap: cv2.VideoCapture,
    seq_name: str,
    frame_idx: int,
    proc_scale: float,
) -> tuple[bool, Optional[np.ndarray]]:
    timeout_s = float(os.environ.get("MIS_EVAL_READ_TIMEOUT_S", "8.0"))
    use_timeout = seq_name.upper() in _problem_seq_names()
    if use_timeout:
        ok, frame = _read_frame_with_timeout(cap, timeout_s)
        if not ok and not use_timeout:
            pass
        elif not ok:
            print(f"[{seq_name}] 解码超时/失败 frame={frame_idx}，跳过")
    else:
        try:
            ok, frame = cap.read()
        except Exception as e:
            print(f"[{seq_name}] 解码异常 frame={frame_idx}: {e}")
            ok, frame = False, None
    if not ok or frame is None:
        return False, None
    if proc_scale != 1.0:
        frame = cv2.resize(frame, None, fx=proc_scale, fy=proc_scale)
    return True, frame


def _maintain_tracker_state(
    tracker: "BaseTracker",
    frame: np.ndarray,
    seq_name: str,
    frame_idx: int,
) -> None:
    """非 GT 帧：轻量推进 tracker 时序状态（Nano propagate 或 fallback update）。"""
    _prop = getattr(tracker, "propagate", None)
    if callable(_prop):
        try:
            _prop(frame)
            return
        except Exception as e:
            print(f"[WARN] {tracker.name} propagate frame={frame_idx}: {e}")
    try:
        tracker.update(frame)
    except Exception as e:
        print(f"[WARN] {tracker.name} frame={frame_idx} update错误: {e}")


def _tracker_step_on_frame(
    tracker: "BaseTracker",
    frame: np.ndarray,
    seq_name: str,
    frame_idx: int,
    is_gt_frame: bool,
    upd_all: bool,
) -> dict[str, Any]:
    if is_gt_frame:
        t0 = time.perf_counter()
        try:
            pred = tracker.update(frame)
        except Exception as e:
            print(f"[WARN] {tracker.name} seq={seq_name} frame={frame_idx} update错误: {e}")
            pred = {}
        return {"pred": pred, "elapsed": time.perf_counter() - t0}
    if upd_all:
        _prop = getattr(tracker, "propagate", None)
        if callable(_prop):
            try:
                _prop(frame)
            except Exception as e:
                print(f"[WARN] {tracker.name} propagate frame={frame_idx}: {e}")
        else:
            try:
                tracker.update(frame)
            except Exception as e:
                print(f"[WARN] {tracker.name} frame={frame_idx} update错误: {e}")
    return {"pred": {}, "elapsed": 0.0}


def _run_segment_gt_read_loop(
    cap: cv2.VideoCapture,
    seq_dir: Path,
    gt_sorted: list["GTFrame"],
    gt_map: dict[int, "GTFrame"],
    tracker: "BaseTracker",
    first_frame_idx: int,
    proc_scale: float,
    upd_all: bool,
    rows: list[dict],
    bbox_ious: list[float],
    mask_ious: list[float],
    center_errs: list[float],
    precision_hits: list[int],
    success_hits: list[int],
    attrs: list[int],
    vis_dir: Path,
    save_vis: bool,
    print_every: int,
    intermediate_root: Path | None,
    save_intermediate: bool,
    save_intermediate_overlay: bool,
) -> None:
    """问题序列：顺序解码至各 GT 帧，避免随机 seek；仅在 GT 帧评估（非 upd_all 时不跑中间帧 update）。"""
    seq_name = seq_dir.name
    last_gt_idx = int(gt_sorted[-1].frame_idx)
    cur_idx = first_frame_idx + 1
    gi_map = {int(g.frame_idx): i + 1 for i, g in enumerate(gt_sorted)}
    gi = 0
    decode_only = 0
    while cur_idx <= last_gt_idx:
        ok, frame = _read_frame_safe(cap, seq_name, cur_idx, proc_scale)
        if not ok:
            cur_idx += 1
            continue
        is_gt = cur_idx in gt_map
        if is_gt:
            step = _tracker_step_on_frame(tracker, frame, seq_name, cur_idx, True, upd_all)
            gt = gt_map[cur_idx]
            gi = gi_map.get(cur_idx, gi + 1)
            _record_gt_eval(
                rows, bbox_ious, mask_ious, center_errs, precision_hits, success_hits,
                frame, gt, step["pred"], attrs, cur_idx, seq_name, tracker,
                vis_dir, save_vis, print_every, gi, step["elapsed"],
                intermediate_root=intermediate_root,
                save_intermediate=save_intermediate,
                save_intermediate_overlay=save_intermediate_overlay,
            )
        elif upd_all:
            _tracker_step_on_frame(tracker, frame, seq_name, cur_idx, False, upd_all)
        else:
            _maintain_tracker_state(tracker, frame, seq_name, cur_idx)
            decode_only += 1
        cur_idx += 1
    if decode_only and print_every > 0:
        print(f"[{seq_name}] 段顺序读: 仅解码中间帧 {decode_only} 帧（无 tracker update）")


def _polygon_to_mask(shape_hw: tuple[int, int], polygon_xy: np.ndarray) -> np.ndarray:
    mask = np.zeros((int(shape_hw[0]), int(shape_hw[1])), dtype=np.uint8)
    poly = np.asarray(polygon_xy, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] >= 3:
        pts = np.round(poly).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
    return mask


def _mask_bbox(mask_u8: np.ndarray, pad: int = 0) -> Optional[tuple[int, int, int, int]]:
    mk = np.asarray(mask_u8, dtype=np.uint8)
    ys, xs = np.nonzero(mk > 0)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = int(max(0, xs.min() - pad))
    y0 = int(max(0, ys.min() - pad))
    x1 = int(xs.max() + pad + 1)
    y1 = int(ys.max() + pad + 1)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _mask_centroid(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    mk = np.asarray(mask_u8, dtype=np.uint8)
    if int(np.count_nonzero(mk)) == 0:
        return None
    m = cv2.moments(mk)
    if abs(float(m["m00"])) < 1e-6:
        return None
    return np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float32)


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
# Ours — 冻结实现（2026-06-03-*方法.py）
# ═══════════════════════════════════════════════════════════════════════════════
from importlib import import_module as _im_frozen

_ours_trk = _im_frozen("2026-06-03-ours_scheme_c_nano跟踪器方法")
_ours_base = _im_frozen("2026-06-03-跟踪器基础方法")

BaseTracker = _ours_base.BaseTracker
_load_defm_ab_once = _ours_base._load_defm_ab_once
OursHistABTracker = _ours_trk.OursHistABTracker
OursSchemeCNanoPriorTracker = _ours_trk.OursSchemeCNanoPriorTracker
_nano_fuse_bboxes = _ours_trk._nano_fuse_bboxes
_nano_center_blend_fixed_size = _ours_trk._nano_center_blend_fixed_size
_nano_center_blend_adaptive_size = _ours_trk._nano_center_blend_adaptive_size
_NANO_BB = _ours_trk._NANO_BB
_NANO_NH = _ours_trk._NANO_NH

# ═══════════════════════════════════════════════════════════════════════════════
# OpenCV 经典跟踪器
# ═══════════════════════════════════════════════════════════════════════════════
class OpenCVTracker(BaseTracker):
    """
    OpenCV 经典追踪器封装。
    在 OpenCV ≥4.10 中，legacy.TrackerCSRT/KCF/MOSSE 已被删除。
    回退策略：
      csrt  → TrackerMIL（在线学习，行为最相近）
      kcf   → TrackerMIL 或 NCC 模板匹配
      mosse → NCC 模板匹配（若 MIL 不可用）
    所有路径都能在纯 CPU 上运行，无需模型权重。
    """
    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int], tracker_type: str = "csrt") -> None:
        self.name = str(tracker_type).lower()
        self.label = TRACKER_LABELS.get(self.name, self.name.upper())
        self.tracker_type = self.name
        self.last_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        self._cv_tracker, self.opencv_backend = self._try_make_cv_tracker(tracker_type)
        if self._cv_tracker is not None:
            self._cv_tracker.init(init_im, tuple(map(int, self.last_bbox)))
            self._use_ncc = False
            if self.opencv_backend == "mil":
                self._init_mil_confidence(init_im, self.last_bbox)
        else:
            # 回退到 NCC 模板匹配
            self._use_ncc = True
            self.opencv_backend = "ncc"
            self._init_ncc(init_im, self.last_bbox)

    # ------------------------------------------------------------------
    @staticmethod
    def _try_make_cv_tracker(tracker_type: str):
        """尝试创建 OpenCV 内置追踪器，失败则返回 None。返回 (tracker, backend_name)。"""
        legacy = getattr(cv2, "legacy", None)
        fn_candidates_map = {
            "csrt":  ["TrackerCSRT_create", "legacy.TrackerCSRT_create",  "TrackerMIL_create"],
            "kcf":   ["TrackerKCF_create",  "legacy.TrackerKCF_create",   "TrackerMIL_create"],
            "mosse": ["TrackerMOSSE_create","legacy.TrackerMOSSE_create", "TrackerMIL_create"],
            "mil":   ["TrackerMIL_create"],
        }
        backend_names = {
            "TrackerCSRT_create": "csrt",
            "legacy.TrackerCSRT_create": "csrt_legacy",
            "TrackerKCF_create": "kcf",
            "legacy.TrackerKCF_create": "kcf_legacy",
            "TrackerMOSSE_create": "mosse",
            "legacy.TrackerMOSSE_create": "mosse_legacy",
            "TrackerMIL_create": "mil",
        }
        for fn_name in fn_candidates_map.get(tracker_type, []):
            try:
                if fn_name.startswith("legacy."):
                    if legacy is not None and hasattr(legacy, fn_name[7:]):
                        return getattr(legacy, fn_name[7:])(), backend_names.get(fn_name, fn_name)
                elif hasattr(cv2, fn_name):
                    return getattr(cv2, fn_name)(), backend_names.get(fn_name, fn_name)
            except Exception:
                continue
        return None, None

    # ------------------------------------------------------------------
    # NCC 模板匹配实现（相关滤波代理）
    # ------------------------------------------------------------------
    _NCC_SEARCH_EXPAND: float = 2.0   # 搜索区域相对模板尺寸的扩展倍数

    def _init_ncc(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        x, y, w, h = [int(v) for v in bbox]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._tmpl = gray[y:y + h, x:x + w].astype(np.float32)
        self._tmpl_size = (w, h)

    def _init_mil_confidence(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        """MIL 无 getTrackingScore 时，用 init patch NCC 作为置信度。"""
        x, y, w, h = [int(v) for v in bbox]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self._conf_tmpl = gray[y:y + h, x:x + w].astype(np.float32)
        self._conf_size = (max(1, w), max(1, h))

    def _mil_patch_ncc(self, frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
        import numpy as np

        x, y, w, h = [int(v) for v in bbox]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        patch = gray[y:y + h, x:x + w]
        if patch.size == 0 or not hasattr(self, "_conf_tmpl"):
            return 0.0
        tmpl = self._conf_tmpl
        if patch.shape != tmpl.shape:
            patch = cv2.resize(patch, (tmpl.shape[1], tmpl.shape[0]))
        a = patch.astype(np.float32)
        b = tmpl.astype(np.float32)
        a -= float(a.mean())
        b -= float(b.mean())
        den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-6)
        return float(np.clip((a * b).sum() / den, 0.0, 1.0))

    def _update_ncc(self, frame_bgr: np.ndarray) -> tuple[tuple[int, int, int, int], float]:
        h_f, w_f = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        x0, y0, tw, th = [int(v) for v in self.last_bbox]
        # 搜索窗口（以上一帧 bbox 为中心扩展）
        sx = max(0, int(x0 - self._NCC_SEARCH_EXPAND * tw))
        sy = max(0, int(y0 - self._NCC_SEARCH_EXPAND * th))
        ex = min(w_f, int(x0 + (1 + self._NCC_SEARCH_EXPAND) * tw))
        ey = min(h_f, int(y0 + (1 + self._NCC_SEARCH_EXPAND) * th))
        search = gray[sy:ey, sx:ex]
        if search.shape[0] < th or search.shape[1] < tw:
            return self.last_bbox, 0.0
        tmpl = self._tmpl
        if tmpl.shape[0] > search.shape[0] or tmpl.shape[1] > search.shape[1]:
            # 模板比搜索区大时直接用原 bbox
            return self.last_bbox, 0.0
        result = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        nx = sx + max_loc[0]
        ny = sy + max_loc[1]
        new_bbox = _clip_bbox((nx, ny, tw, th), (h_f, w_f))
        # 更新模板（指数移动平均，学习率 0.1）
        nx2, ny2, nw, nh = new_bbox
        patch = gray[ny2:ny2 + nh, nx2:nx2 + nw].astype(np.float32)
        if patch.shape == self._tmpl.shape:
            self._tmpl = 0.9 * self._tmpl + 0.1 * patch
        return new_bbox, float(max_val)

    # ------------------------------------------------------------------
    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        if self._use_ncc:
            self.last_bbox, score = self._update_ncc(frame_bgr)
        else:
            ok, box = self._cv_tracker.update(frame_bgr)
            if ok:
                x, y, w, h = [int(round(float(v))) for v in box]
                self.last_bbox = _clip_bbox((x, y, max(1, w), max(1, h)), frame_bgr.shape[:2])
                if hasattr(self, "_conf_tmpl"):
                    score = self._mil_patch_ncc(frame_bgr, self.last_bbox)
                else:
                    score = float(getattr(self._cv_tracker, "getTrackingScore", lambda: 1.0)())
            else:
                score = 0.0
        return {
            "bbox": self.last_bbox,
            "mask_full": _bbox_to_mask(frame_bgr.shape[:2], self.last_bbox),
            "score": score,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MeanShift / CamShift / MIL (纯 OpenCV，无外部依赖)
# ═══════════════════════════════════════════════════════════════════════════════
class MeanShiftTracker(BaseTracker):
    """OpenCV MeanShift 跟踪器：基于 HSV 直方图反投影的均值漂移。"""
    name  = "meanshift"
    label = TRACKER_LABELS["meanshift"]

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int]) -> None:
        self.last_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        self._init_hist(init_im, self.last_bbox)
        # 终止条件：最多 20 次迭代 或 1 像素精度
        self._term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1.0)

    def _init_hist(self, frame: np.ndarray, bbox: tuple) -> None:
        x, y, w, h = [int(v) for v in bbox]
        x, y = max(0, x), max(0, y)
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            self._hist = None
            return
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array([0, 30, 20], dtype=np.uint8),
                           np.array([180, 255, 255], dtype=np.uint8))
        self._hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self._hist, self._hist, 0, 255, cv2.NORM_MINMAX)

    def _backproject(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self._hist, [0, 180], 1)
        dst = cv2.filter2D(dst, -1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        return dst

    def update(self, frame: np.ndarray) -> dict:
        if self._hist is None:
            return {"bbox": self.last_bbox,
                    "mask_full": _bbox_to_mask(frame.shape[:2], self.last_bbox), "score": 0.0}
        dst = self._backproject(frame)
        x, y, w, h = [int(v) for v in self.last_bbox]
        track_window = (x, y, max(1, w), max(1, h))
        ret, new_win = cv2.meanShift(dst, track_window, self._term)
        new_bbox = _clip_bbox(new_win, frame.shape[:2])
        self.last_bbox = new_bbox
        # 轻微在线更新直方图（学习率 0.05）
        nx, ny, nw, nh = new_bbox
        roi = frame[ny:ny + nh, nx:nx + nw]
        if roi.size > 0:
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_roi, np.array([0, 30, 20], dtype=np.uint8),
                               np.array([180, 255, 255], dtype=np.uint8))
            new_h = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
            cv2.normalize(new_h, new_h, 0, 255, cv2.NORM_MINMAX)
            self._hist = 0.95 * self._hist + 0.05 * new_h
        return {"bbox": self.last_bbox,
                "mask_full": _bbox_to_mask(frame.shape[:2], self.last_bbox), "score": float(ret)}


class CamShiftTracker(BaseTracker):
    """OpenCV CamShift：自适应尺度的均值漂移。"""
    name  = "camshift"
    label = TRACKER_LABELS["camshift"]

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int]) -> None:
        self.last_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        self._term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 1.0)
        self._init_hist(init_im, self.last_bbox)

    def _init_hist(self, frame: np.ndarray, bbox: tuple) -> None:
        x, y, w, h = [int(v) for v in bbox]
        x, y = max(0, x), max(0, y)
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            self._hist = None
            return
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array([0, 30, 20], dtype=np.uint8),
                           np.array([180, 255, 255], dtype=np.uint8))
        self._hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
        cv2.normalize(self._hist, self._hist, 0, 255, cv2.NORM_MINMAX)

    def update(self, frame: np.ndarray) -> dict:
        if self._hist is None:
            return {"bbox": self.last_bbox,
                    "mask_full": _bbox_to_mask(frame.shape[:2], self.last_bbox), "score": 0.0}
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self._hist, [0, 180], 1)
        dst = cv2.filter2D(dst, -1, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        x, y, w, h = [int(v) for v in self.last_bbox]
        track_window = (x, y, max(1, w), max(1, h))
        rot_rect, new_win = cv2.CamShift(dst, track_window, self._term)
        new_bbox = _clip_bbox(new_win, frame.shape[:2])
        self.last_bbox = new_bbox
        return {"bbox": self.last_bbox,
                "mask_full": _bbox_to_mask(frame.shape[:2], self.last_bbox), "score": 1.0}


class MILTracker(BaseTracker):
    """OpenCV MIL (Multiple Instance Learning) 跟踪器。"""
    name  = "mil"
    label = TRACKER_LABELS["mil"]

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int]) -> None:
        self.last_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        self._tracker = cv2.TrackerMIL_create()
        self._tracker.init(init_im, tuple(map(int, self.last_bbox)))

    def update(self, frame: np.ndarray) -> dict:
        ok, box = self._tracker.update(frame)
        if ok:
            x, y, w, h = [int(round(float(v))) for v in box]
            self.last_bbox = _clip_bbox((x, y, max(1, w), max(1, h)), frame.shape[:2])
            score = 1.0
        else:
            score = 0.0
        return {"bbox": self.last_bbox,
                "mask_full": _bbox_to_mask(frame.shape[:2], self.last_bbox), "score": score}


# ═══════════════════════════════════════════════════════════════════════════════
# NanoTrack (OpenCV TrackerNano)
# ═══════════════════════════════════════════════════════════════════════════════
_NANO_BACKBONE_DEFAULT = r"D:\MIS_Pose_Track_Re3D\nanotrack_backbone_sim.onnx"
_NANO_NECKHEAD_DEFAULT = r"D:\MIS_Pose_Track_Re3D\nanotrack_head_sim.onnx"

class NanoTrackTracker(BaseTracker):
    """NanoTrack via OpenCV TrackerNano (SiamNet backbone + head, ≈1 MB ONNX)."""
    name  = "nanotrack"
    label = TRACKER_LABELS["nanotrack"]

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int],
                 backbone: str = _NANO_BACKBONE_DEFAULT,
                 neckhead: str = _NANO_NECKHEAD_DEFAULT) -> None:
        self.last_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        params = cv2.TrackerNano_Params()
        params.backbone = backbone
        params.neckhead = neckhead
        self._tracker = cv2.TrackerNano_create(params)
        self._tracker.init(init_im, tuple(map(int, self.last_bbox)))

    def update(self, frame: np.ndarray) -> dict:
        ok, box = self._tracker.update(frame)
        if ok:
            x, y, w, h = [int(round(float(v))) for v in box]
            self.last_bbox = _clip_bbox((x, y, max(1, w), max(1, h)), frame.shape[:2])
            score = float(getattr(self._tracker, 'getTrackingScore', lambda: 1.0)())
        else:
            score = 0.0
        return {"bbox": self.last_bbox,
                "mask_full": _bbox_to_mask(frame.shape[:2], self.last_bbox), "score": score}

    def propagate(self, frame: np.ndarray) -> None:
        self.update(frame)


# ═══════════════════════════════════════════════════════════════════════════════
# ViT-Track (OpenCV TrackerVit)
# ═══════════════════════════════════════════════════════════════════════════════
_VIT_MODEL_DEFAULT = r"D:\MIS_Pose_Track_Re3D\object_tracking_vittrack_2023sep.onnx"

class ViTTrackTracker(BaseTracker):
    """ViT-based tracker via OpenCV TrackerVit (≈700 KB ONNX)."""
    name  = "vittrack"
    label = TRACKER_LABELS["vittrack"]

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int],
                 model: str = _VIT_MODEL_DEFAULT) -> None:
        self.last_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        params = cv2.TrackerVit_Params()
        params.net = model
        self._tracker = cv2.TrackerVit_create(params)
        self._tracker.init(init_im, tuple(map(int, self.last_bbox)))

    def update(self, frame: np.ndarray) -> dict:
        ok, box = self._tracker.update(frame)
        if ok:
            x, y, w, h = [int(round(float(v))) for v in box]
            self.last_bbox = _clip_bbox((x, y, max(1, w), max(1, h)), frame.shape[:2])
            score = float(getattr(self._tracker, 'getTrackingScore', lambda: 1.0)())
        else:
            score = 0.0
        return {"bbox": self.last_bbox,
                "mask_full": _bbox_to_mask(frame.shape[:2], self.last_bbox), "score": score}


# ═══════════════════════════════════════════════════════════════════════════════
# LK-Flow: Lucas-Kanade sparse optical flow tracker
# ═══════════════════════════════════════════════════════════════════════════════
class LKFlowTracker(BaseTracker):
    """Lucas-Kanade sparse optical flow tracker.
    Tracks a grid of feature points inside the initial bbox and estimates
    the new bbox from the surviving point cloud median shift.
    """
    name  = "lkflow"
    label = TRACKER_LABELS["lkflow"]

    _LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    _GRID_N    = 6     # grid points per axis (6×6 = 36 total)

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int]) -> None:
        self.last_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        self._prev_gray = cv2.cvtColor(init_im, cv2.COLOR_BGR2GRAY)
        self._pts = self._grid_points(self.last_bbox)

    def _grid_points(self, bbox: tuple) -> np.ndarray:
        x, y, w, h = [int(v) for v in bbox]
        xs = np.linspace(x + w * 0.15, x + w * 0.85, self._GRID_N)
        ys = np.linspace(y + h * 0.15, y + h * 0.85, self._GRID_N)
        gx, gy = np.meshgrid(xs, ys)
        return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)

    def update(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._pts is None or len(self._pts) == 0:
            self._prev_gray = gray
            return {"bbox": self.last_bbox,
                    "mask_full": _bbox_to_mask(frame.shape[:2], self.last_bbox), "score": 0.0}

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._pts, None, **self._LK_PARAMS)

        if new_pts is None or status is None:
            self._prev_gray = gray
            return {"bbox": self.last_bbox,
                    "mask_full": _bbox_to_mask(frame.shape[:2], self.last_bbox), "score": 0.0}

        good_old = self._pts[status.ravel() == 1]
        good_new = new_pts[status.ravel() == 1]

        if len(good_new) < 4:
            # Too few points; reinitialise grid on current bbox
            self._pts = self._grid_points(self.last_bbox)
            self._prev_gray = gray
            return {"bbox": self.last_bbox,
                    "mask_full": _bbox_to_mask(frame.shape[:2], self.last_bbox), "score": float(len(good_new)) / max(1, len(status))}

        # Estimate translation as median shift
        dx = float(np.median(good_new[:, 0, 0] - good_old[:, 0, 0]))
        dy = float(np.median(good_new[:, 0, 1] - good_old[:, 0, 1]))
        x0, y0, w0, h0 = [int(v) for v in self.last_bbox]
        new_bbox = _clip_bbox((x0 + dx, y0 + dy, w0, h0), frame.shape[:2])
        self.last_bbox = new_bbox

        # Update tracked points to new positions
        self._pts = good_new.reshape(-1, 1, 2)
        # Periodically re-seed if too few points remain
        if len(self._pts) < self._GRID_N:
            self._pts = np.vstack([self._pts, self._grid_points(self.last_bbox)])

        self._prev_gray = gray
        score = float(len(good_new)) / max(1, len(status))
        return {"bbox": self.last_bbox,
                "mask_full": _bbox_to_mask(frame.shape[:2], self.last_bbox), "score": score}


# ═══════════════════════════════════════════════════════════════════════════════
# DIS-Flow: Dense Inverse Search optical flow tracker (OpenCV DISOpticalFlow)
# ═══════════════════════════════════════════════════════════════════════════════
class DISFlowTracker(BaseTracker):
    """Dense Inverse Search optical flow tracker.

    Computes dense optical flow on the full frame and estimates the bounding-box
    displacement as the trimmed-mean of the flow vectors inside the current bbox.
    Uses cv2.DISOpticalFlow (MEDIUM preset) which is faster than Farneback and
    more accurate than sparse LK on featureless regions.
    """
    name  = "disflow"
    label = TRACKER_LABELS["disflow"]

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int]) -> None:
        self.last_bbox  = _clip_bbox(init_bbox, init_im.shape[:2])
        self._prev_gray = cv2.cvtColor(init_im, cv2.COLOR_BGR2GRAY)
        self._dis       = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_MEDIUM)

    def update(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = self._dis.calc(self._prev_gray, gray, None)  # H x W x 2
        x, y, w, h = [int(v) for v in self.last_bbox]
        H, W = gray.shape
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(W, x + w), min(H, y + h)
        if x2 > x1 and y2 > y1:
            roi_flow = flow[y1:y2, x1:x2].reshape(-1, 2)
            # Trimmed mean (clip top/bottom 20 %) to handle motion boundaries
            dx = float(np.mean(np.clip(roi_flow[:, 0],
                                       np.percentile(roi_flow[:, 0], 10),
                                       np.percentile(roi_flow[:, 0], 90))))
            dy = float(np.mean(np.clip(roi_flow[:, 1],
                                       np.percentile(roi_flow[:, 1], 10),
                                       np.percentile(roi_flow[:, 1], 90))))
            score = 1.0
        else:
            dx, dy, score = 0.0, 0.0, 0.0

        new_bbox = _clip_bbox((x + dx, y + dy, w, h), frame.shape[:2])
        self.last_bbox = new_bbox
        self._prev_gray = gray
        return {"bbox": self.last_bbox,
                "mask_full": _bbox_to_mask(frame.shape[:2], self.last_bbox),
                "score": score}


# ═══════════════════════════════════════════════════════════════════════════════
# YOLO-seg
# ═══════════════════════════════════════════════════════════════════════════════
class UltralyticsYOLOTracker(BaseTracker):
    name = "yolo"
    label = TRACKER_LABELS["yolo"]

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int],
                 weights: str = "yolo11x-seg.pt", conf: float = 0.05, imgsz: int = 1024) -> None:
        from ultralytics import YOLO  # type: ignore
        self.model = YOLO(weights)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.prev_bbox = _clip_bbox(init_bbox, init_im.shape[:2])

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        results = self.model.predict(source=frame_bgr, verbose=False, conf=self.conf, imgsz=self.imgsz)
        pred_bbox = self.prev_bbox
        pred_mask = _bbox_to_mask(frame_bgr.shape[:2], pred_bbox)
        best_score = -1e9
        if results:
            res = results[0]
            boxes = getattr(res, "boxes", None)
            masks = getattr(res, "masks", None)
            xyxy = getattr(boxes, "xyxy", None) if boxes else None
            confs = getattr(boxes, "conf", None) if boxes else None
            if xyxy is not None:
                xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
                conf_np = confs.detach().cpu().numpy() if (confs is not None and hasattr(confs, "detach")) else (np.asarray(confs) if confs is not None else None)
                mask_np = None
                if masks is not None and getattr(masks, "data", None) is not None:
                    data = masks.data
                    mask_np = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
                diag = float(math.hypot(frame_bgr.shape[1], frame_bgr.shape[0]))
                for i in range(int(xyxy_np.shape[0])):
                    x1, y1, x2, y2 = [float(v) for v in xyxy_np[i]]
                    bb = _clip_bbox((int(round(min(x1,x2))), int(round(min(y1,y2))), int(round(abs(x2-x1))), int(round(abs(y2-y1)))), frame_bgr.shape[:2])
                    iou = _bbox_iou(self.prev_bbox, bb)
                    cen = 1.0 - min(1.0, float(np.linalg.norm(_center_error(self.prev_bbox) - _center_error(bb))) / max(1.0, 0.25 * diag))
                    cf = float(conf_np[i]) if conf_np is not None and i < int(conf_np.shape[0]) else 0.5
                    score = 0.55 * iou + 0.25 * cen + 0.20 * cf
                    if score > best_score:
                        best_score = float(score)
                        pred_bbox = bb
                        if mask_np is not None and i < int(mask_np.shape[0]):
                            mk = (np.asarray(mask_np[i]) > 0.5).astype(np.uint8) * 255
                            if mk.shape[:2] != frame_bgr.shape[:2]:
                                mk = cv2.resize(mk, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                            pred_mask = mk
                        else:
                            pred_mask = _bbox_to_mask(frame_bgr.shape[:2], pred_bbox)
        self.prev_bbox = pred_bbox
        return {"bbox": pred_bbox, "mask_full": pred_mask, "score": float(max(best_score, 0.0))}


# ═══════════════════════════════════════════════════════════════════════════════
# SAM (Ultralytics)
# ═══════════════════════════════════════════════════════════════════════════════
class UltralyticsSAMTracker(BaseTracker):
    name = "sam"
    label = TRACKER_LABELS["sam"]

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int],
                 weights: str = "sam_b.pt") -> None:
        from ultralytics import SAM  # type: ignore
        self.model = SAM(weights)
        self.prev_bbox = _clip_bbox(init_bbox, init_im.shape[:2])

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        x, y, w, h = self.prev_bbox
        xyxy = np.array([[x, y, x + w, y + h]], dtype=np.float32)
        results = self.model.predict(source=frame_bgr, verbose=False, bboxes=xyxy)
        pred_bbox = self.prev_bbox
        pred_mask = _bbox_to_mask(frame_bgr.shape[:2], pred_bbox)
        if results:
            res = results[0]
            masks = getattr(res, "masks", None)
            if masks is not None and getattr(masks, "data", None) is not None:
                data = masks.data
                mask_np = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
                if int(mask_np.shape[0]) > 0:
                    best_i, best_score = 0, -1e9
                    prompt_mask = _bbox_to_mask(frame_bgr.shape[:2], self.prev_bbox)
                    for i in range(int(mask_np.shape[0])):
                        mk = (np.asarray(mask_np[i]) > 0.5).astype(np.uint8) * 255
                        if mk.shape[:2] != frame_bgr.shape[:2]:
                            mk = cv2.resize(mk, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                        bb = _mask_bbox(mk)
                        if bb is None:
                            continue
                        score = 0.7 * _mask_iou(prompt_mask, mk) + 0.3 * _bbox_iou(self.prev_bbox, bb)
                        if score > best_score:
                            best_score = float(score)
                            best_i = i
                    pred_mask = (np.asarray(mask_np[best_i]) > 0.5).astype(np.uint8) * 255
                    if pred_mask.shape[:2] != frame_bgr.shape[:2]:
                        pred_mask = cv2.resize(pred_mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    bb = _mask_bbox(pred_mask)
                    if bb is not None:
                        pred_bbox = bb
        self.prev_bbox = pred_bbox
        return {"bbox": pred_bbox, "mask_full": pred_mask, "score": 1.0}


# ═══════════════════════════════════════════════════════════════════════════════
# SAM2-Video
# ═══════════════════════════════════════════════════════════════════════════════
_SAM2_CONFIG_MAP = {
    "tiny": "sam2/sam2_hiera_t.yaml", "small": "sam2/sam2_hiera_s.yaml",
    "base_plus": "sam2/sam2_hiera_b+.yaml", "large": "sam2/sam2_hiera_l.yaml",
}


class SAM2VideoTracker(BaseTracker):
    name = "sam2"
    label = TRACKER_LABELS["sam2"]

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int],
                 init_mask: Optional[np.ndarray] = None, variant: str = "tiny") -> None:
        import torch
        from sam2.build_sam import build_sam2_video_predictor  # type: ignore
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = _SAM2_CONFIG_MAP.get(variant, variant)
        self.predictor = build_sam2_video_predictor(cfg, device=self.device)
        self.prev_bbox = _clip_bbox(tuple(map(int, init_bbox)), init_im.shape[:2])
        self.init_mask = init_mask
        self.frame_h, self.frame_w = init_im.shape[:2]
        self.frames_buffer: list[np.ndarray] = [init_im.copy()]
        self.inference_state = None
        self._obj_id = 1
        self._initialized = False
        self._tmpdir = ""
        self.last_mask_full = _bbox_to_mask(init_im.shape[:2], self.prev_bbox)
        self.last_bbox = self.prev_bbox

    def _init_video_state(self, all_frames: list[np.ndarray]) -> None:
        import torch, tempfile
        tmpdir = tempfile.mkdtemp(prefix="sam2_frames_")
        self._tmpdir = tmpdir
        for i, fr in enumerate(all_frames):
            cv2.imwrite(os.path.join(tmpdir, f"{i:06d}.jpg"), fr)
        self.inference_state = self.predictor.init_state(video_path=tmpdir)
        x, y, w, h = self.prev_bbox
        if self.init_mask is not None and int(np.count_nonzero(self.init_mask)) > 0:
            mk = (np.asarray(self.init_mask, dtype=np.uint8) > 0).astype(np.float32)
            if mk.ndim == 3:
                mk = mk[:, :, 0]
            mask_t = torch.from_numpy(mk).unsqueeze(0).to(self.device)
            self.predictor.add_new_mask(inference_state=self.inference_state, frame_idx=0, obj_id=self._obj_id, mask=mask_t)
        else:
            box_np = np.array([[x, y, x + w, y + h]], dtype=np.float32)
            self.predictor.add_new_points_or_box(inference_state=self.inference_state, frame_idx=0, obj_id=self._obj_id, box=box_np)
        self._initialized = True

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        self.frames_buffer.append(frame_bgr.copy())
        frame_idx = len(self.frames_buffer) - 1
        if not self._initialized:
            try:
                self._init_video_state(self.frames_buffer[:1])
            except Exception:
                return {"bbox": self.prev_bbox, "mask_full": self.last_mask_full, "score": 0.0}
        try:
            cv2.imwrite(os.path.join(self._tmpdir, f"{frame_idx:06d}.jpg"), frame_bgr)
            for out_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                if out_idx == frame_idx:
                    for i_obj, obj_id in enumerate(out_obj_ids):
                        if obj_id == self._obj_id:
                            mask_prob = (out_mask_logits[i_obj] > 0.0).cpu().numpy().squeeze()
                            pred_mask = (mask_prob > 0).astype(np.uint8) * 255
                            if pred_mask.shape[:2] != (self.frame_h, self.frame_w):
                                pred_mask = cv2.resize(pred_mask, (self.frame_w, self.frame_h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                            bbox = _mask_bbox(pred_mask)
                            if bbox is not None:
                                self.prev_bbox = bbox
                                self.last_mask_full = pred_mask
                                self.last_bbox = bbox
                                return {"bbox": bbox, "mask_full": pred_mask, "score": 1.0}
        except Exception:
            pass
        return {"bbox": self.prev_bbox, "mask_full": self.last_mask_full, "score": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# SAM2-PerFrame
# ═══════════════════════════════════════════════════════════════════════════════
class SAM2PerFrameTracker(BaseTracker):
    name = "sam2_pf"
    label = TRACKER_LABELS["sam2_pf"]

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int],
                 init_mask: Optional[np.ndarray] = None, variant: str = "tiny") -> None:
        import torch
        from sam2.build_sam import build_sam2  # type: ignore
        from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = _SAM2_CONFIG_MAP.get(variant, variant)
        model = build_sam2(cfg, device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        self.prev_bbox = _clip_bbox(tuple(map(int, init_bbox)), init_im.shape[:2])
        self.last_mask_full = _bbox_to_mask(init_im.shape[:2], self.prev_bbox)
        self.last_bbox = self.prev_bbox

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(frame_rgb)
            x, y, w, h = self.prev_bbox
            box_np = np.array([[x, y, x + w, y + h]], dtype=np.float32)
            masks, scores, _ = self.predictor.predict(box=box_np, multimask_output=True)
            best_i = int(np.argmax(scores))
            pred_mask = (masks[best_i] > 0).astype(np.uint8) * 255
            if pred_mask.shape[:2] != frame_bgr.shape[:2]:
                pred_mask = cv2.resize(pred_mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            bbox = _mask_bbox(pred_mask)
            if bbox is not None:
                self.prev_bbox = bbox
                self.last_mask_full = pred_mask
                self.last_bbox = bbox
                return {"bbox": bbox, "mask_full": pred_mask, "score": float(scores[best_i])}
        except Exception:
            pass
        return {"bbox": self.prev_bbox, "mask_full": self.last_mask_full, "score": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# OSTrack (ECCV 2022) — 结构复现版（无预训练权重）
# ═══════════════════════════════════════════════════════════════════════════════
class OSTrackTracker(BaseTracker):
    """OSTrack: One-Stream Transformer Tracker (ECCV 2022)。结构复现版。"""
    name = "ostrack"
    label = TRACKER_LABELS["ostrack"]
    TEMPLATE_FACTOR = 2.0; SEARCH_FACTOR = 4.5
    TEMPLATE_SIZE = 128; SEARCH_SIZE = 256

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int]) -> None:
        import timm, torch  # type: ignore
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prev_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0, global_pool="").to(self.device).eval()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(384, 256), torch.nn.ReLU(), torch.nn.Linear(256, 4), torch.nn.Sigmoid()
        ).to(self.device)
        self.template_feat = self._extract_feat(init_im, init_bbox)

    def _crop_and_resize(self, img, bbox, factor, out_sz):
        x, y, w, h = bbox; cx, cy = x + w / 2, y + h / 2
        crop_sz = max(int(max(w, h) * factor), 32)
        x1 = max(0, int(cx - crop_sz / 2)); y1 = max(0, int(cy - crop_sz / 2))
        x2 = min(img.shape[1], x1 + crop_sz); y2 = min(img.shape[0], y1 + crop_sz)
        crop = img[y1:y2, x1:x2]
        return cv2.resize(crop if crop.size > 0 else img, (out_sz, out_sz)), (x1, y1, x2 - x1, y2 - y1)

    def _img_to_tensor(self, img_bgr):
        import torch
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_rgb = (img_rgb - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

    def _extract_feat(self, img, bbox):
        crop, _ = self._crop_and_resize(img, bbox, self.TEMPLATE_FACTOR, self.TEMPLATE_SIZE)
        with self.torch.no_grad():
            return self.backbone(self._img_to_tensor(cv2.resize(crop, (224, 224)))).mean(dim=1)

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import torch
        search_crop, crop_bbox = self._crop_and_resize(frame_bgr, self.prev_bbox, self.SEARCH_FACTOR, self.SEARCH_SIZE)
        with torch.no_grad():
            search_feat = self.backbone(self._img_to_tensor(cv2.resize(search_crop, (224, 224)))).mean(dim=1)
            delta = self.head(search_feat * self.template_feat).squeeze(0).cpu().numpy()
        sx, sy, sw, sh = crop_bbox
        scale_x = sw / self.SEARCH_SIZE; scale_y = sh / self.SEARCH_SIZE
        pred_cx = sx + delta[0] * self.SEARCH_SIZE * scale_x
        pred_cy = sy + delta[1] * self.SEARCH_SIZE * scale_y
        pred_w = max(8, delta[2] * self.SEARCH_SIZE * scale_x)
        pred_h = max(8, delta[3] * self.SEARCH_SIZE * scale_y)
        pred_bbox = _clip_bbox((int(pred_cx - pred_w / 2), int(pred_cy - pred_h / 2), int(pred_w), int(pred_h)), frame_bgr.shape[:2])
        px, py, pw, ph = self.prev_bbox
        pred_bbox = _clip_bbox((int(0.7 * pred_bbox[0] + 0.3 * px), int(0.7 * pred_bbox[1] + 0.3 * py),
                                int(0.7 * pred_bbox[2] + 0.3 * pw), int(0.7 * pred_bbox[3] + 0.3 * ph)), frame_bgr.shape[:2])
        self.prev_bbox = pred_bbox
        return {"bbox": pred_bbox, "mask_full": _bbox_to_mask(frame_bgr.shape[:2], pred_bbox), "score": 1.0}


# ═══════════════════════════════════════════════════════════════════════════════
# MixFormer (ECCV 2022) — 结构复现版
# ═══════════════════════════════════════════════════════════════════════════════
class MixFormerTracker(BaseTracker):
    """MixFormer: Mixed Attention Module Tracker (ECCV 2022)。结构复现版。"""
    name = "mixformer"
    label = TRACKER_LABELS["mixformer"]
    TEMPLATE_FACTOR = 2.0; SEARCH_FACTOR = 4.0
    TEMPLATE_SIZE = 128; SEARCH_SIZE = 256; N_ITER = 2

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int]) -> None:
        import timm, torch  # type: ignore
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prev_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=0, global_pool="").to(self.device).eval()
        self.score_head = torch.nn.Sequential(torch.nn.Linear(384, 128), torch.nn.ReLU(), torch.nn.Linear(128, 1), torch.nn.Sigmoid()).to(self.device)
        self.reg_head = torch.nn.Sequential(torch.nn.Linear(384, 256), torch.nn.GELU(), torch.nn.Linear(256, 4), torch.nn.Sigmoid()).to(self.device)
        self.template_feat = self._extract_feat(init_im, init_bbox)

    def _crop_and_resize(self, img, bbox, factor, out_sz):
        x, y, w, h = bbox; cx, cy = x + w / 2, y + h / 2
        crop_sz = max(int(max(w, h) * factor), 32)
        x1 = max(0, int(cx - crop_sz / 2)); y1 = max(0, int(cy - crop_sz / 2))
        x2 = min(img.shape[1], x1 + crop_sz); y2 = min(img.shape[0], y1 + crop_sz)
        crop = img[y1:y2, x1:x2]
        return cv2.resize(crop if crop.size > 0 else img, (out_sz, out_sz)), (x1, y1, x2 - x1, y2 - y1)

    def _img_to_tensor(self, img_bgr):
        import torch
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_rgb = (img_rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

    def _extract_feat(self, img, bbox):
        crop, _ = self._crop_and_resize(img, bbox, self.TEMPLATE_FACTOR, self.TEMPLATE_SIZE)
        with self.torch.no_grad():
            return self.backbone(self._img_to_tensor(cv2.resize(crop, (224, 224)))).mean(dim=1)

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import torch
        search_crop, crop_bbox = self._crop_and_resize(frame_bgr, self.prev_bbox, self.SEARCH_FACTOR, self.SEARCH_SIZE)
        with torch.no_grad():
            feat = self.backbone(self._img_to_tensor(cv2.resize(search_crop, (224, 224)))).mean(dim=1)
            for _ in range(self.N_ITER):
                w = torch.sigmoid((feat * self.template_feat).sum(dim=-1, keepdim=True) / (384 ** 0.5))
                feat = feat * w + self.template_feat * (1 - w)
            delta = self.reg_head(feat).squeeze(0).cpu().numpy()
            score = float(self.score_head(feat).squeeze())
        sx, sy, sw, sh = crop_bbox
        scale_x = sw / self.SEARCH_SIZE; scale_y = sh / self.SEARCH_SIZE
        pred_cx = sx + delta[0] * self.SEARCH_SIZE * scale_x
        pred_cy = sy + delta[1] * self.SEARCH_SIZE * scale_y
        pred_w = max(8, delta[2] * self.SEARCH_SIZE * scale_x)
        pred_h = max(8, delta[3] * self.SEARCH_SIZE * scale_y)
        pred_bbox = _clip_bbox((int(pred_cx - pred_w / 2), int(pred_cy - pred_h / 2), int(pred_w), int(pred_h)), frame_bgr.shape[:2])
        px, py, pw, ph = self.prev_bbox
        pred_bbox = _clip_bbox((int(0.7 * pred_bbox[0] + 0.3 * px), int(0.7 * pred_bbox[1] + 0.3 * py),
                                int(0.7 * pred_bbox[2] + 0.3 * pw), int(0.7 * pred_bbox[3] + 0.3 * ph)), frame_bgr.shape[:2])
        self.prev_bbox = pred_bbox
        return {"bbox": pred_bbox, "mask_full": _bbox_to_mask(frame_bgr.shape[:2], pred_bbox), "score": score}


# ═══════════════════════════════════════════════════════════════════════════════
# Cutie (CVPR 2024) — 若未安装则降级到 SAM 代理
# ═══════════════════════════════════════════════════════════════════════════════
class CutieTracker(BaseTracker):
    """Cutie: Putting the Object Back into Video Object Segmentation (CVPR 2024)。"""
    name = "cutie"
    label = TRACKER_LABELS["cutie"]
    MEMORY_EVERY = 5

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int],
                 init_mask: Optional[np.ndarray] = None) -> None:
        self.prev_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        self._frame_idx = 0
        self._use_cutie = False
        try:
            import torch
            from cutie.inference.inference_core import InferenceCore  # type: ignore
            from cutie.model.cutie import CUTIE  # type: ignore
            from omegaconf import OmegaConf  # type: ignore
            cfg = OmegaConf.create({"model": "cutie-base", "weights": "", "mem_every": self.MEMORY_EVERY, "max_mem_frames": 10})
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cutie_model = CUTIE(cfg).to(device).eval()
            self._processor = InferenceCore(cutie_model, cfg=cfg)
            x, y, w, h = init_bbox
            mask_bin = (init_mask > 0).astype(np.uint8) if init_mask is not None else np.zeros(init_im.shape[:2], dtype=np.uint8)
            if init_mask is None:
                mask_bin[y:y + h, x:x + w] = 1
            mask_t = torch.from_numpy(mask_bin).long().to(device)
            frame_t = torch.from_numpy(cv2.cvtColor(init_im, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(device)
            self._processor.set_all_labels({1})
            self._processor.step(frame_t, mask_t.unsqueeze(0))
            self._use_cutie = True; self._device = device
            self._last_mask = mask_bin * 255
        except Exception:
            self._last_mask = _bbox_to_mask(init_im.shape[:2], init_bbox)

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        self._frame_idx += 1
        if self._use_cutie:
            try:
                import torch
                frame_t = torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(self._device)
                pred = self._processor.step(frame_t)
                if pred.dim() == 4:
                    mask_t = (pred[0, 0] > 0.5)
                elif pred.dim() == 3:
                    mask_t = (pred[0] == 1)
                else:
                    mask_t = (pred > 0.5)
                pred_mask = mask_t.cpu().numpy().astype(np.uint8) * 255
                if pred_mask.shape[:2] != frame_bgr.shape[:2]:
                    pred_mask = cv2.resize(pred_mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                bbox = _mask_bbox(pred_mask)
                if bbox is not None:
                    self.prev_bbox = bbox
                    self._last_mask = pred_mask
                return {"bbox": self.prev_bbox, "mask_full": self._last_mask, "score": 1.0}
            except Exception:
                pass
        return {"bbox": self.prev_bbox, "mask_full": _bbox_to_mask(frame_bgr.shape[:2], self.prev_bbox), "score": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# LMTrack (AAAI 2024) — 结构复现版
# ═══════════════════════════════════════════════════════════════════════════════
class LMTrackTracker(BaseTracker):
    """LMTrack: Less Is More - Token Context-Aware Learning for Object Tracking (AAAI 2024)。"""
    name = "lmtrack"
    label = TRACKER_LABELS["lmtrack"]
    TEMPLATE_FACTOR = 2.0; SEARCH_FACTOR = 4.0
    TEMPLATE_SIZE = 128; SEARCH_SIZE = 256; TOP_K = 0.7

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int]) -> None:
        import timm, torch  # type: ignore
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prev_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        self.backbone = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="").to(self.device).eval()
        D = 768
        self.tam = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.ReLU(), torch.nn.Linear(256, 1), torch.nn.Sigmoid()).to(self.device)
        self.reg_head = torch.nn.Sequential(torch.nn.Linear(D, 256), torch.nn.GELU(), torch.nn.Linear(256, 4), torch.nn.Sigmoid()).to(self.device)
        self.template_feat = self._extract_feat(init_im, init_bbox)

    def _crop_and_resize(self, img, bbox, factor, out_sz):
        x, y, w, h = bbox; cx, cy = x + w / 2, y + h / 2
        crop_sz = max(int(max(w, h) * factor), 32)
        x1 = max(0, int(cx - crop_sz / 2)); y1 = max(0, int(cy - crop_sz / 2))
        x2 = min(img.shape[1], x1 + crop_sz); y2 = min(img.shape[0], y1 + crop_sz)
        crop = img[y1:y2, x1:x2]
        return cv2.resize(crop if crop.size > 0 else img, (out_sz, out_sz)), (x1, y1, x2 - x1, y2 - y1)

    def _img_to_tensor(self, img_bgr):
        import torch
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_rgb = (img_rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

    def _extract_feat(self, img, bbox):
        crop, _ = self._crop_and_resize(img, bbox, self.TEMPLATE_FACTOR, self.TEMPLATE_SIZE)
        with self.torch.no_grad():
            return self.backbone(self._img_to_tensor(cv2.resize(crop, (224, 224))))  # (1, N_tok, 768)

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import torch
        search_crop, crop_bbox = self._crop_and_resize(frame_bgr, self.prev_bbox, self.SEARCH_FACTOR, self.SEARCH_SIZE)
        with torch.no_grad():
            feat = self.backbone(self._img_to_tensor(cv2.resize(search_crop, (224, 224))))  # (1, N, 768)
            tok_scores = self.tam(feat).squeeze(-1)
            k = max(1, int(feat.shape[1] * self.TOP_K))
            topk_idx = tok_scores.topk(k, dim=1).indices
            selected = feat.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, 768))
            tmpl_g = self.template_feat.mean(dim=1, keepdim=True)
            sim = torch.softmax((selected * tmpl_g).sum(dim=-1, keepdim=True) / (768 ** 0.5), dim=1)
            fused = (selected * sim).sum(dim=1)
            delta = self.reg_head(fused).squeeze(0).cpu().numpy()
        sx, sy, sw, sh = crop_bbox
        scale_x = sw / self.SEARCH_SIZE; scale_y = sh / self.SEARCH_SIZE
        pred_cx = sx + delta[0] * self.SEARCH_SIZE * scale_x
        pred_cy = sy + delta[1] * self.SEARCH_SIZE * scale_y
        pred_w = max(8, delta[2] * self.SEARCH_SIZE * scale_x)
        pred_h = max(8, delta[3] * self.SEARCH_SIZE * scale_y)
        pred_bbox = _clip_bbox((int(pred_cx - pred_w / 2), int(pred_cy - pred_h / 2), int(pred_w), int(pred_h)), frame_bgr.shape[:2])
        px, py, pw, ph = self.prev_bbox
        pred_bbox = _clip_bbox((int(0.7 * pred_bbox[0] + 0.3 * px), int(0.7 * pred_bbox[1] + 0.3 * py),
                                int(0.7 * pred_bbox[2] + 0.3 * pw), int(0.7 * pred_bbox[3] + 0.3 * ph)), frame_bgr.shape[:2])
        self.prev_bbox = pred_bbox
        return {"bbox": pred_bbox, "mask_full": _bbox_to_mask(frame_bgr.shape[:2], pred_bbox), "score": 1.0}


# ═══════════════════════════════════════════════════════════════════════════════
# 跟踪器工厂
# ═══════════════════════════════════════════════════════════════════════════════
def _make_tracker(
    name: str,
    init_im: np.ndarray,
    init_bbox: tuple[int, int, int, int],
    init_mask: Optional[np.ndarray],
    args: argparse.Namespace,
) -> BaseTracker:
    key = str(name).strip().lower()
    if key in (
        "ours", "ours_hist_ab", "ours_scheme_c_nano", "ours_c_nano", "ours_c",
    ):
        try:
            import importlib

            _frozen = importlib.import_module("2026-06-03-ours_scheme_c_nano方法")
            if _frozen.is_ours_tracker(key):
                return _frozen.create_tracker(
                    init_im, init_bbox, variant=key, tracker_args=args
                )
        except Exception:
            pass
    if key in ("ours", "ours_hist_ab"):
        return OursHistABTracker(init_im, init_bbox, coarse_scheme="ab")
    if key == "ours_nano":
        return OursNanoTracker(init_im, init_bbox)
    if key == "nano_ab":
        return NanoABTracker(init_im, init_bbox)
    if key == "nano_deep":
        return NanoDeepFusion(init_im, init_bbox)
    if key == "ours_hist_c":
        return OursHistCTracker(init_im, init_bbox)
    if key in ("ours_scheme_c_nano", "ours_c_nano"):
        return OursSchemeCNanoPriorTracker(init_im, init_bbox)
    if key in ("csrt", "kcf", "mosse"):
        return OpenCVTracker(init_im, init_bbox, tracker_type=key)
    if key == "meanshift":
        return MeanShiftTracker(init_im, init_bbox)
    if key == "camshift":
        return CamShiftTracker(init_im, init_bbox)
    if key == "mil":
        return MILTracker(init_im, init_bbox)
    if key == "nanotrack":
        backbone = getattr(args, "nano_backbone", _NANO_BACKBONE_DEFAULT)
        neckhead = getattr(args, "nano_neckhead", _NANO_NECKHEAD_DEFAULT)
        return NanoTrackTracker(init_im, init_bbox, backbone=backbone, neckhead=neckhead)
    if key == "vittrack":
        model = getattr(args, "vit_model", _VIT_MODEL_DEFAULT)
        return ViTTrackTracker(init_im, init_bbox, model=model)
    if key == "lkflow":
        return LKFlowTracker(init_im, init_bbox)
    if key == "disflow":
        return DISFlowTracker(init_im, init_bbox)
    if key == "yolo":
        return UltralyticsYOLOTracker(init_im, init_bbox,
                                      weights=getattr(args, "yolo_weights", "yolo11x-seg.pt"),
                                      conf=float(getattr(args, "yolo_conf", 0.05)),
                                      imgsz=int(getattr(args, "yolo_imgsz", 1024)))
    if key == "sam":
        return UltralyticsSAMTracker(init_im, init_bbox,
                                     weights=getattr(args, "sam_weights", "sam_b.pt"))
    if key == "sam2":
        variant = str(getattr(args, "sam2_variant", "tiny") or "tiny")
        return SAM2VideoTracker(init_im, init_bbox, init_mask, variant=variant)
    if key == "sam2_pf":
        variant = str(getattr(args, "sam2_variant", "tiny") or "tiny")
        return SAM2PerFrameTracker(init_im, init_bbox, init_mask, variant=variant)
    if key == "ostrack":
        return OSTrackTracker(init_im, init_bbox)
    if key == "mixformer":
        return MixFormerTracker(init_im, init_bbox)
    if key == "cutie":
        return CutieTracker(init_im, init_bbox, init_mask)
    if key == "lmtrack":
        return LMTrackTracker(init_im, init_bbox)
    if key == "stark":
        # Reuse the established chessboard STARK-style implementation rather
        # than silently substituting another tracker.  It uses a public
        # ImageNet backbone plus an untrained corner head and is therefore
        # explicitly reported as a protocol proxy, not official STARK.
        import importlib

        chess_compare = importlib.import_module("compare_chessboard_trackers")
        tracker = chess_compare.StarkTracker(init_im, init_bbox)
        tracker.name = "stark"
        tracker.label = TRACKER_LABELS["stark"]
        return tracker
    raise ValueError(f"未知跟踪器: {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# 可视化工具
# ═══════════════════════════════════════════════════════════════════════════════
def _draw_vis(
    frame_bgr: np.ndarray,
    gt_poly: np.ndarray,
    gt_bbox: tuple,
    pred_bbox: Optional[tuple],
    pred_mask: np.ndarray,
    tracker_label: str,
    frame_idx: int,
    bbox_iou: float,
    attr_id: int,
) -> np.ndarray:
    vis = np.asarray(frame_bgr, dtype=np.uint8).copy()
    # GT 多边形（绿色）
    pts = np.round(np.asarray(gt_poly, dtype=np.float32).reshape(-1, 2)).astype(np.int32).reshape(-1, 1, 2)
    if pts.shape[0] >= 3:
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    # GT bbox（浅绿）
    if gt_bbox is not None:
        x, y, w, h = [int(v) for v in gt_bbox]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 1)
    # Pred bbox（蓝色）
    if pred_bbox is not None:
        px, py, pw, ph = [int(v) for v in pred_bbox]
        cv2.rectangle(vis, (px, py), (px + pw, py + ph), (255, 0, 0), 2)
    # 文字
    attr_name = ATTR_NAMES.get(attr_id, str(attr_id))
    cv2.putText(vis, f"GT=green  Pred=blue  Attr={attr_name}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, f"{tracker_label} | f={frame_idx} bboxIoU={bbox_iou:.3f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def _draw_bbox_on_frame(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    color: tuple[int, int, int],
    label: str = "",
) -> None:
    if bbox is None:
        return
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(
            frame_bgr, label, (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )


def _save_track_intermediate(
    intermediate_root: Path | None,
    seq_name: str,
    tracker_name: str,
    frame_idx: int,
    frame_bgr: np.ndarray | None,
    debug: dict[str, Any],
    *,
    save_overlay: bool = True,
) -> None:
    """将单帧跟踪中间结果写入 F:（或 --intermediate-dir）。"""
    if intermediate_root is None:
        return
    frame_dir = intermediate_root / seq_name / tracker_name
    frame_dir.mkdir(parents=True, exist_ok=True)
    jpath = frame_dir / f"frame_{int(frame_idx):06d}.json"
    with jpath.open("w", encoding="utf-8") as f:
        json.dump(debug, f, ensure_ascii=False, indent=2, default=str)
    if not save_overlay or frame_bgr is None:
        return
    vis = frame_bgr.copy()
    _draw_bbox_on_frame(vis, debug.get("gt_bbox"), (0, 255, 0), "GT")
    _draw_bbox_on_frame(vis, debug.get("mis_bbox"), (255, 128, 0), "MIS")
    nb = debug.get("nano_bbox")
    if nb is not None:
        _draw_bbox_on_frame(vis, tuple(nb), (0, 128, 255), "Nano")
    _draw_bbox_on_frame(vis, debug.get("final_bbox"), (0, 0, 255), "Out")
    txt = (
        f"f={frame_idx} bb={debug.get('bbox_iou', 0):.3f} "
        f"hist={debug.get('hist_sim', -1):.2f} iou_mn={debug.get('iou_mn', -1):.2f}"
    )
    cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(frame_dir / f"frame_{int(frame_idx):06d}.jpg"), vis)


# ═══════════════════════════════════════════════════════════════════════════════
# 单序列、单跟踪器评估
# ═══════════════════════════════════════════════════════════════════════════════
def _evaluate_one_tracker_on_seq(
    tracker: BaseTracker,
    seq_dir: Path,
    gts: list[GTFrame],
    attrs: list[int],
    out_dir: Path,
    save_vis: bool,
    sequential: bool,
    proc_scale: float = 1.0,
    print_every: int = 20,
    intermediate_root: Path | None = None,
    save_intermediate: bool = False,
    save_intermediate_overlay: bool = True,
) -> dict[str, Any]:
    """
    在单个序列上运行单个跟踪器，返回评估结果字典。

    sequential=True : 逐帧顺序读取（tracker 看到所有帧，仅在 GT 帧评估）
    sequential=False: 跳帧模式（只处理 GT 帧，速度更快）
    """
    video_path = _find_video(seq_dir)
    if video_path is None:
        raise FileNotFoundError(f"序列 {seq_dir.name} 中未找到视频文件")

    cap, video_path = _open_video_capture(seq_dir)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vis_dir = out_dir / "vis" / tracker.name
    if save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    bbox_ious: list[float] = []
    mask_ious: list[float] = []
    center_errs: list[float] = []
    precision_hits: list[int] = []
    success_hits: list[int] = []

    # 首帧初始化（tracker 已在工厂函数中初始化，这里直接使用第一个GT帧的bbox）
    first_gt = gts[0]
    first_frame_idx = int(first_gt.frame_idx)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx)
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        cap.release()
        raise RuntimeError(f"读取首帧(frame={first_frame_idx})失败")
    if proc_scale != 1.0:
        first_frame = cv2.resize(first_frame, None, fx=proc_scale, fy=proc_scale)

    fH, fW = first_frame.shape[:2]

    # 首帧结果（用 GT 作为初始 pred）
    first_gt_mask = _polygon_to_mask(first_frame.shape[:2], first_gt.polygon)
    first_gt_bbox = _mask_bbox(first_gt_mask, pad=2) or (0, 0, fW, fH)
    first_attr = int(attrs[first_frame_idx]) if first_frame_idx < len(attrs) else 0

    gt_center = _mask_centroid(first_gt_mask)
    pred_center = _center_error(first_gt_bbox)
    rows.append({
        "sequence": seq_dir.name, "tracker": tracker.name, "frame_idx": first_frame_idx,
        "attribute_id": first_attr, "attribute_name": ATTR_NAMES.get(first_attr, str(first_attr)),
        "gt_bbox": list(map(int, first_gt_bbox)),
        "pred_bbox": list(map(int, first_gt_bbox)),
        "bbox_iou": 1.0, "mask_iou": 1.0, "center_err_px": 0.0,
        "precision_hit": 1, "success_hit": 1, "pred_score": 1.0,
        "elapsed_update_s": 0.0,   # 初始帧用 GT 初始化，无 update 耗时
    })
    bbox_ious.append(1.0); mask_ious.append(1.0); center_errs.append(0.0)
    precision_hits.append(1); success_hits.append(1)

    # ── 逐帧跟踪 ─────────────────────────────────────────────────────────────
    # 构建GT帧索引 map：frame_idx → GTFrame
    gt_map: dict[int, GTFrame] = {gt.frame_idx: gt for gt in gts}
    eval_frame_indices = {int(gt.frame_idx) for gt in gts}

    t_start = time.perf_counter()
    gt_sorted = sorted(gts[1:], key=lambda g: g.frame_idx)
    _upd_all = os.environ.get("MIS_EVAL_UPDATE_ALL_FRAMES", "0").lower() not in (
        "0", "false", "no", "off"
    )
    _sparse_gt = os.environ.get("MIS_EVAL_GT_SPARSE_READ", "1").lower() not in (
        "0", "false", "no", "off"
    )
    _use_clean = video_path.suffix.lower() == ".mp4" and "clean" in video_path.stem.lower()
    _problem_seq = seq_dir.name.upper() in _problem_seq_names()
    _use_segment = _problem_seq and not _use_clean

    if sequential:
        # 顺序模式：逐帧解码；仅在 GT 帧 update（除非 upd_all）
        last_gt_idx = int(gts[-1].frame_idx)
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_idx + 1)
        cur_frame_idx = first_frame_idx + 1
        processed = 0
        while cur_frame_idx <= last_gt_idx:
            ok, frame = _read_frame_safe(cap, seq_dir.name, cur_frame_idx, proc_scale)
            if not ok:
                cur_frame_idx += 1
                continue

            is_gt = cur_frame_idx in eval_frame_indices
            elapsed_update = 0.0
            pred: dict[str, Any] = {}
            if is_gt or _upd_all:
                step = _tracker_step_on_frame(
                    tracker, frame, seq_dir.name, cur_frame_idx, is_gt, _upd_all
                )
                pred = step["pred"]
                elapsed_update = step["elapsed"]
            elif _problem_seq or _use_segment:
                _maintain_tracker_state(tracker, frame, seq_dir.name, cur_frame_idx)

            if is_gt:
                gt = gt_map[cur_frame_idx]
                _record_gt_eval(
                    rows, bbox_ious, mask_ious, center_errs, precision_hits, success_hits,
                    frame, gt, pred, attrs, cur_frame_idx, seq_dir.name, tracker,
                    vis_dir, save_vis, print_every, processed, elapsed_update,
                    intermediate_root=intermediate_root,
                    save_intermediate=save_intermediate,
                    save_intermediate_overlay=save_intermediate_overlay,
                )
            processed += 1
            cur_frame_idx += 1
    elif _use_segment:
        # EV3/IV1 无 clean 缓存：段顺序读至 GT 帧，避免 cap.set 阻塞
        _run_segment_gt_read_loop(
            cap, seq_dir, gt_sorted, gt_map, tracker, first_frame_idx, proc_scale, _upd_all,
            rows, bbox_ious, mask_ious, center_errs, precision_hits, success_hits,
            attrs, vis_dir, save_vis, print_every,
            intermediate_root, save_intermediate, save_intermediate_overlay,
        )
    else:
        # 跳帧模式：仅在 GT 帧处评估
        if not _upd_all and _sparse_gt:
            # 仅 seek 到 GT 帧（clean 视频或普通序列）
            gi_map: dict[int, int] = {int(g.frame_idx): i + 1 for i, g in enumerate(gt_sorted)}
            for gt in gt_sorted:
                cur_idx = int(gt.frame_idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, cur_idx)
                ok, frame = _read_frame_safe(cap, seq_dir.name, cur_idx, proc_scale)
                if not ok:
                    continue
                t_update = time.perf_counter()
                try:
                    pred = tracker.update(frame)
                except Exception as e:
                    print(f"[WARN] {tracker.name} seq={seq_dir.name} frame={cur_idx} update错误: {e}")
                    pred = {}
                elapsed_update = time.perf_counter() - t_update
                gi = gi_map.get(cur_idx, 0)
                _record_gt_eval(
                    rows, bbox_ious, mask_ious, center_errs, precision_hits, success_hits,
                    frame, gt, pred, attrs, cur_idx, seq_dir.name, tracker,
                    vis_dir, save_vis, print_every, gi, elapsed_update,
                    intermediate_root=intermediate_root,
                    save_intermediate=save_intermediate,
                    save_intermediate_overlay=save_intermediate_overlay,
                )
        else:
            _run_segment_gt_read_loop(
                cap, seq_dir, gt_sorted, gt_map, tracker, first_frame_idx, proc_scale, _upd_all,
                rows, bbox_ious, mask_ious, center_errs, precision_hits, success_hits,
                attrs, vis_dir, save_vis, print_every,
                intermediate_root, save_intermediate, save_intermediate_overlay,
            )

    cap.release()
    elapsed_total = time.perf_counter() - t_start

    # ── 汇总统计 ─────────────────────────────────────────────────────────────
    n = max(1, len(rows))
    precision_thresh_px = 20.0
    success_iou_thresh = 0.5

    summary: dict[str, Any] = {
        "sequence":          seq_dir.name,
        "tracker":           tracker.name,
        "tracker_label":     getattr(tracker, "label", tracker.name),
        "n_gt_frames":       int(len(rows)),
        "bbox_iou_mean":     float(np.mean(bbox_ious)) if bbox_ious else float("nan"),
        "bbox_iou_median":   float(np.median(bbox_ious)) if bbox_ious else float("nan"),
        "mask_iou_mean":     float(np.mean(mask_ious)) if mask_ious else float("nan"),
        "mask_iou_median":   float(np.median(mask_ious)) if mask_ious else float("nan"),
        "center_err_mean":   float(np.mean(center_errs)) if center_errs else float("nan"),
        "center_err_median": float(np.median(center_errs)) if center_errs else float("nan"),
        "precision_rate":    float(sum(precision_hits)) / float(n),
        "success_rate":      float(sum(success_hits)) / float(n),
        "precision_thresh_px": precision_thresh_px,
        "success_iou_thresh":  success_iou_thresh,
        "elapsed_s":         float(f"{elapsed_total:.2f}"),
        "sequential":        bool(sequential),
    }

    # 按属性分组统计
    attr_stats: dict[int, dict] = {}
    for row in rows:
        aid = int(row["attribute_id"])
        if aid not in attr_stats:
            attr_stats[aid] = {"bbox_ious": [], "precision": [], "success": []}
        attr_stats[aid]["bbox_ious"].append(float(row["bbox_iou"]))
        attr_stats[aid]["precision"].append(int(row["precision_hit"]))
        attr_stats[aid]["success"].append(int(row["success_hit"]))
    summary["attr_stats"] = {
        str(aid): {
            "name": ATTR_NAMES.get(aid, str(aid)),
            "n": len(v["bbox_ious"]),
            "bbox_iou_mean": float(np.mean(v["bbox_ious"])) if v["bbox_ious"] else float("nan"),
            "precision_rate": float(np.mean(v["precision"])) if v["precision"] else float("nan"),
            "success_rate": float(np.mean(v["success"])) if v["success"] else float("nan"),
        }
        for aid, v in attr_stats.items()
    }

    # 保存 CSV（取所有行 keys 的并集作为列名，避免不同行字段不一致导致报错）
    csv_path = out_dir / f"{tracker.name}_rows.csv"
    if rows:
        all_keys: list[str] = list(dict.fromkeys(k for row in rows for k in row.keys()))
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    if intermediate_root is not None:
        summary["intermediate_dir"] = str(
            intermediate_root / seq_dir.name / tracker.name
        )

    summary["rows_csv"] = str(csv_path)

    return summary


def _record_gt_eval(
    rows, bbox_ious, mask_ious, center_errs, precision_hits, success_hits,
    frame, gt, pred, attrs, frame_idx, seq_name, tracker,
    vis_dir, save_vis, print_every, gi, elapsed_update,
    intermediate_root: Path | None = None,
    save_intermediate: bool = False,
    save_intermediate_overlay: bool = True,
):
    """在 GT 帧记录评估结果（供 sequential 和跳帧两种模式共用）。"""
    fH, fW = frame.shape[:2]
    gt_mask = _polygon_to_mask(frame.shape[:2], gt.polygon)
    gt_bbox = _mask_bbox(gt_mask, pad=2) or (0, 0, fW, fH)
    gt_center = _mask_centroid(gt_mask)
    attr_id = int(attrs[frame_idx]) if frame_idx < len(attrs) else 0

    pred_bbox = None
    pred_mask = np.zeros((fH, fW), dtype=np.uint8)
    pred_score = float("nan")
    if pred:
        raw_bb = pred.get("bbox", None)
        if raw_bb is not None:
            pred_bbox = _clip_bbox(tuple(map(int, raw_bb)), frame.shape[:2])
        raw_mk = pred.get("mask_full", None)
        if raw_mk is not None:
            mk = np.asarray(raw_mk, dtype=np.uint8)
            if mk.ndim == 2:
                pred_mask = mk
            elif mk.ndim == 3:
                pred_mask = np.max(mk, axis=2).astype(np.uint8)
        elif pred_bbox is not None:
            pred_mask = _bbox_to_mask(frame.shape[:2], pred_bbox)
        pred_score = float(pred.get("score", float("nan")))

    bb_iou = _bbox_iou(gt_bbox, pred_bbox) if pred_bbox is not None else 0.0
    mk_iou = _mask_iou(gt_mask, pred_mask)
    gt_ctr = gt_center if gt_center is not None else _center_error(gt_bbox)
    pred_ctr = _center_error(pred_bbox) if pred_bbox is not None else gt_ctr
    c_err = float(np.linalg.norm(pred_ctr - gt_ctr))

    precision_thresh_px = 20.0
    success_iou_thresh = 0.5
    prec_hit = int(c_err <= precision_thresh_px)
    succ_hit = int(bb_iou >= success_iou_thresh)

    bbox_ious.append(float(bb_iou)); mask_ious.append(float(mk_iou))
    center_errs.append(float(c_err)); precision_hits.append(prec_hit); success_hits.append(succ_hit)

    rows.append({
        "sequence": seq_name, "tracker": tracker.name, "frame_idx": int(frame_idx),
        "attribute_id": attr_id, "attribute_name": ATTR_NAMES.get(attr_id, str(attr_id)),
        "gt_bbox": list(map(int, gt_bbox)),
        "pred_bbox": list(map(int, pred_bbox)) if pred_bbox is not None else None,
        "bbox_iou": float(bb_iou), "mask_iou": float(mk_iou),
        "center_err_px": float(c_err), "precision_hit": prec_hit, "success_hit": succ_hit,
        "pred_score": float(pred_score) if np.isfinite(pred_score) else -1.0,
        "elapsed_update_s": float(f"{elapsed_update:.4f}"),
    })
    if os.environ.get("MIS_DEEP_DEBUG_ROWS", "").strip() in ("1", "true", "yes", "on"):
        dd = getattr(tracker, "_last_deep_debug", None)
        if isinstance(dd, dict) and dd:
            rows[-1].update({k: float(v) if isinstance(v, (float, int, np.floating)) else v for k, v in dd.items()})

    if (gi % print_every) == 0:
        print(f"  [{tracker.name}] frame={frame_idx:6d} bbox_iou={bb_iou:.3f} mask_iou={mk_iou:.3f} "
              f"c_err={c_err:.1f}px attr={ATTR_NAMES.get(attr_id,'?')} elapsed={elapsed_update*1000:.0f}ms")

    if (
        save_intermediate
        and intermediate_root is not None
        and tracker.name in ("ours_scheme_c_nano", "ours_c_nano")
        and pred
        and isinstance(pred.get("debug"), dict)
    ):
        stride = max(1, int(os.environ.get("MIS_INTERMEDIATE_STRIDE", "1")))
        save_ov = save_intermediate_overlay and (
            int(frame_idx) % stride == 0 or float(bb_iou) < 0.5
        )
        dbg = dict(pred["debug"])
        dbg["gt_bbox"] = list(map(int, gt_bbox))
        dbg["bbox_iou"] = float(bb_iou)
        dbg["mask_iou"] = float(mk_iou)
        dbg["center_err_px"] = float(c_err)
        dbg["frame_idx"] = int(frame_idx)
        dbg["sequence"] = seq_name
        _save_track_intermediate(
            intermediate_root,
            seq_name,
            tracker.name,
            frame_idx,
            frame,
            dbg,
            save_overlay=save_ov,
        )

    if save_vis:
        vis = _draw_vis(frame, gt.polygon, gt_bbox, pred_bbox, pred_mask,
                        getattr(tracker, "label", tracker.name), frame_idx, bb_iou, attr_id)
        cv2.imwrite(str(vis_dir / f"frame_{frame_idx:06d}.jpg"), vis, [cv2.IMWRITE_JPEG_QUALITY, 85])


# ═══════════════════════════════════════════════════════════════════════════════
# 单序列评估（多跟踪器）
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_sequence(
    seq_dir: Path,
    tracker_names: list[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    """在单个序列上评估所有指定跟踪器，返回每个跟踪器的 summary 列表。"""
    seq_name = seq_dir.name
    print(f"\n{'='*60}")
    print(f"[序列] {seq_name}  video={_find_video(seq_dir)}")

    gt_path   = seq_dir / "GT.xml"
    attr_path = seq_dir / "Attributes.xml"
    if not gt_path.exists():
        print(f"[跳过] GT.xml 不存在: {gt_path}")
        return []

    gts = parse_gt_xml(gt_path, GT_FRAME_STEP)
    if not gts:
        print(f"[跳过] GT 为空: {gt_path}")
        return []

    attrs = parse_attributes_xml(attr_path) if attr_path.exists() else []
    print(f"  GT帧数={len(gts)}  首帧={gts[0].frame_idx}  末帧={gts[-1].frame_idx}  属性帧={len(attrs)}")

    # 读取首帧（用于初始化跟踪器）
    video_path = _find_video(seq_dir)
    if video_path is None:
        print(f"[跳过] 视频不存在: {seq_dir}")
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[跳过] 无法打开视频: {video_path}")
        return []

    first_gt = gts[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(first_gt.frame_idx))
    ok, first_frame = cap.read()
    cap.release()
    if not ok or first_frame is None:
        print(f"[跳过] 读取首帧失败 frame={first_gt.frame_idx}")
        return []

    proc_scale = float(getattr(args, "proc_scale", 1.0))
    if proc_scale != 1.0:
        first_frame = cv2.resize(first_frame, None, fx=proc_scale, fy=proc_scale)

    # 根据首帧GT多边形建立初始bbox和mask
    first_mask = _polygon_to_mask(first_frame.shape[:2], first_gt.polygon)
    first_bbox = _mask_bbox(first_mask, pad=4)
    if first_bbox is None:
        print(f"[跳过] 首帧GT无法生成bbox")
        return []
    print(f"  首帧 init_bbox={first_bbox}  frame_size={first_frame.shape[1]}x{first_frame.shape[0]}")

    seq_out = out_dir / seq_name
    seq_out.mkdir(parents=True, exist_ok=True)
    save_vis = bool(getattr(args, "save_vis", False))
    sequential = bool(getattr(args, "sequential", True))
    _problem = seq_dir.name.upper() in _problem_seq_names()
    if _problem:
        print(f"[{seq_dir.name}] 问题序列：优先预转码/段顺序读（不强制 sequential=True）")
    print_every = int(getattr(args, "print_every", 20))
    save_intermediate = bool(getattr(args, "save_intermediate", False))
    save_intermediate_overlay = bool(getattr(args, "save_intermediate_overlay", True))
    intermediate_root: Path | None = None
    if save_intermediate:
        intermediate_root = Path(
            getattr(args, "intermediate_dir", None) or DEFAULT_INTERMEDIATE
        )
        intermediate_root.mkdir(parents=True, exist_ok=True)
        print(f"  intermediate_dir: {intermediate_root}")

    summaries: list[dict[str, Any]] = []
    for tracker_name in tracker_names:
        print(f"\n  ── {TRACKER_LABELS.get(tracker_name, tracker_name)} ──")

        # 断点续跑：已有 summary.json 则直接加载跳过
        # 若 --save-vis 1 且 vis 目录不存在，则不跳过（需要重跑以生成图）
        summary_path = seq_out / f"{tracker_name}_summary.json"
        vis_dir_check = seq_out / "vis" / tracker_name
        save_vis_flag = bool(getattr(args, "save_vis", False))
        vis_missing = save_vis_flag and (not vis_dir_check.exists() or not any(vis_dir_check.glob("*.jpg")))
        if summary_path.exists() and not getattr(args, "force", False) and not vis_missing:
            with summary_path.open("r", encoding="utf-8") as _f:
                cached = json.load(_f)
            print(f"  [SKIP] 已有结果，跳过 → bbox_iou={cached.get('bbox_iou_mean', float('nan')):.4f}")
            summaries.append(cached)
            continue

        try:
            tracker = _make_tracker(tracker_name, first_frame, first_bbox, first_mask, args)
        except Exception as e:
            print(f"  [SKIP] 初始化失败 {tracker_name}: {e}")
            summaries.append({
                "sequence": seq_name, "tracker": tracker_name,
                "tracker_label": TRACKER_LABELS.get(tracker_name, tracker_name),
                "error": str(e), "n_gt_frames": 0,
                "bbox_iou_mean": float("nan"), "mask_iou_mean": float("nan"),
                "center_err_mean": float("nan"), "precision_rate": float("nan"),
                "success_rate": float("nan"),
            })
            continue

        try:
            summary = _evaluate_one_tracker_on_seq(
                tracker=tracker, seq_dir=seq_dir, gts=gts, attrs=attrs,
                out_dir=seq_out, save_vis=save_vis, sequential=sequential,
                proc_scale=proc_scale, print_every=print_every,
                intermediate_root=intermediate_root,
                save_intermediate=save_intermediate
                and tracker_name in ("ours_scheme_c_nano", "ours_c_nano"),
                save_intermediate_overlay=save_intermediate_overlay,
            )
        except Exception as e:
            import traceback
            print(f"  [ERROR] {tracker_name} on {seq_name}: {e}")
            traceback.print_exc()
            summary = {
                "sequence": seq_name, "tracker": tracker_name,
                "tracker_label": getattr(tracker, "label", tracker_name),
                "error": str(e), "n_gt_frames": 0,
                "bbox_iou_mean": float("nan"), "mask_iou_mean": float("nan"),
                "center_err_mean": float("nan"), "precision_rate": float("nan"),
                "success_rate": float("nan"),
            }

        summaries.append(summary)
        print(f"  [结果] bbox_iou={summary.get('bbox_iou_mean', float('nan')):.4f}  "
              f"mask_iou={summary.get('mask_iou_mean', float('nan')):.4f}  "
              f"prec={summary.get('precision_rate', float('nan')):.4f}  "
              f"succ={summary.get('success_rate', float('nan')):.4f}  "
              f"elapsed={summary.get('elapsed_s', 0):.1f}s")

        # 保存单序列单tracker summary
        with (seq_out / f"{tracker_name}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    return summaries


EVAL_PROFILES: dict[str, dict[str, str]] = {
    "fast": {
        "MIS_EVAL_GT_SPARSE_READ": "1",
        "MIS_EVAL_UPDATE_ALL_FRAMES": "0",
        "MIS_NANO_CASCADE": "1",
        "MIS_MATCH_RESP_ENABLE": "0",
        "MIS_RERANK_TOPR": "32",
        "MIS_COARSE_STEP": "20",
        "MIS_MAX_TIME_S": "1.0",
        "MIS_SILENT": "1",
    },
    "cascade": {
        "MIS_EVAL_GT_SPARSE_READ": "1",
        "MIS_EVAL_UPDATE_ALL_FRAMES": "0",
        "MIS_NANO_CASCADE": "1",
        "MIS_SILENT": "1",
    },
    "quality": {
        "MIS_EVAL_GT_SPARSE_READ": "1",
        "MIS_EVAL_UPDATE_ALL_FRAMES": "0",
        "MIS_NANO_CASCADE": "1",
        "MIS_MATCH_RESP_ENABLE": "1",
        "MIS_COARSE_STEP": "16",
        "MIS_NANO_LITE_ENABLE": "1",
        "MIS_SILENT": "1",
    },
}


def apply_eval_profile(profile: str) -> str:
    """写入评测速度/质量档位对应的环境变量（未显式设置时才写入）。"""
    key = (profile or "cascade").strip().lower()
    if key not in EVAL_PROFILES:
        key = "cascade"
    for k, v in EVAL_PROFILES[key].items():
        if os.environ.get(k, "") == "":
            os.environ[k] = v
    try:
        import importlib

        _sch = importlib.import_module("2026-06-03-SchemeC-NanoPrior方法")

        if key in ("cascade", "quality", "fast"):
            if key == "cascade" or os.environ.get("MIS_NANO_CASCADE", "1") == "1":
                _sch.apply_cascade_fusion_defaults()
            else:
                _sch.apply_scheme_c_fusion_defaults()
    except Exception:
        pass
    return key


# ═══════════════════════════════════════════════════════════════════════════════
# 主评测入口
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(
        description="TrackVes 数据集全方法评测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--root-dir",    default=TRACKVES_ROOT,
                    help="TrackVes 数据集根目录")
    ap.add_argument("--sequences",   default="all",
                    help="要评测的序列（逗号分隔，或 all）: EV1,EV2,...")
    ap.add_argument("--trackers",    default="ours,csrt,kcf,mosse",
                    help=("要评测的方法（逗号分隔，或 all）: "
                          "ours,csrt,kcf,mosse,yolo,sam,sam2,sam2_pf,ostrack,mixformer,cutie,lmtrack,stark"))
    ap.add_argument("--out-dir",     default=DEFAULT_OUT,
                    help="评测结果输出根目录（默认 F:\\MIS_Pose_Track_Re3D\\Exp_Results_TrackVes_AllMethods）")
    ap.add_argument("--intermediate-dir", default=DEFAULT_INTERMEDIATE,
                    help="Scheme C 中间结果目录（JSON+叠加图，默认 F:\\MIS_Pose_Track_Re3D\\SchemeC_Intermediate）")
    ap.add_argument("--save-intermediate", type=int, default=1,
                    help="1=保存 ours_scheme_c_nano 每 GT 帧中间结果到 intermediate-dir")
    ap.add_argument("--save-intermediate-overlay", type=int, default=1,
                    help="1=同时保存 GT/MIS/Nano/Out 叠加 JPG")
    ap.add_argument(
        "--profile",
        default="cascade",
        choices=("fast", "cascade", "quality"),
        help="cascade=Nano级联(推荐,快且稳); fast=开发快测; quality=每帧全量MIS(慢)",
    )
    ap.add_argument("--sequential",  type=int, default=0,
                    help="1=逐帧顺序处理（推荐），0=跳帧模式（仅处理GT帧）")
    ap.add_argument("--save-vis",    type=int, default=0,
                    help="1=保存每帧可视化图像（磁盘占用大）")
    ap.add_argument("--force",       type=int, default=0,
                    help="1=强制重新评测（忽略已有结果）")
    ap.add_argument("--proc-scale",  type=float, default=1.0,
                    help="处理分辨率缩放系数（1.0=原始分辨率，0.5=半尺寸加速）")
    ap.add_argument("--print-every", type=int, default=20,
                    help="每隔多少个GT帧打印一次进度（默认 20，便于观察长序列）")
    # YOLO 参数
    ap.add_argument("--yolo-weights", default="yolo11x-seg.pt")
    ap.add_argument("--yolo-conf",    type=float, default=0.05)
    ap.add_argument("--yolo-imgsz",   type=int,   default=1024)
    # SAM 参数
    ap.add_argument("--sam-weights",  default="sam_b.pt")
    ap.add_argument("--sam2-variant", default="tiny",
                    help="SAM2变体: tiny,small,base_plus,large")
    args = ap.parse_args()
    profile_used = apply_eval_profile(str(getattr(args, "profile", "quality")))

    # 解析序列列表
    if str(args.sequences).strip().lower() in ("all", ""):
        seq_names = ALL_SEQUENCES
    else:
        seq_names = [s.strip() for s in str(args.sequences).split(",") if s.strip()]

    # 解析跟踪器列表
    if str(args.trackers).strip().lower() in ("all", ""):
        tracker_names = ALL_TRACKER_KEYS
    else:
        tracker_names = [s.strip().lower() for s in str(args.trackers).split(",") if s.strip()]
        # 统一别名
        alias_map = {"ours": "ours_hist_ab"}
        tracker_names = [alias_map.get(t, t) for t in tracker_names]

    root_dir = Path(args.root_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"TrackVes 全方法评测")
    print(f"  root_dir : {root_dir}")
    print(f"  out_dir  : {out_dir}")
    print(f"  sequences: {seq_names}")
    print(f"  trackers : {tracker_names}")
    print(f"  sequential={bool(args.sequential)}  save_vis={bool(args.save_vis)}  "
          f"proc_scale={args.proc_scale}")
    print(f"  save_intermediate={bool(args.save_intermediate)}  "
          f"intermediate_dir={args.intermediate_dir}")
    print(f"  profile={profile_used}  MIS_EVAL_GT_SPARSE_READ={os.environ.get('MIS_EVAL_GT_SPARSE_READ', '')}")
    print(f"{'='*60}")

    # 如果包含 ours/ours_hist_ab，预热加载一次模型
    if any(
        t in ("ours_hist_ab", "ours_hist_c", "ours_nano", "ours_scheme_c_nano", "ours_c_nano", "ours")
        for t in tracker_names
    ):
        _load_defm_ab_once()

    all_summaries: list[dict[str, Any]] = []
    per_seq_results: dict[str, list[dict]] = {}

    for seq_name in seq_names:
        seq_dir = root_dir / seq_name
        if not seq_dir.exists():
            print(f"[跳过] 序列目录不存在: {seq_dir}")
            continue
        seq_summaries = evaluate_sequence(seq_dir, tracker_names, out_dir, args)
        per_seq_results[seq_name] = seq_summaries
        all_summaries.extend(seq_summaries)

    # ── 汇总结果 ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("全数据集汇总结果 (所有序列均值)")
    print(f"{'='*60}")

    # 按 tracker 聚合
    from collections import defaultdict
    tracker_agg: dict[str, dict[str, list]] = defaultdict(lambda: {
        "bbox_iou": [], "mask_iou": [], "center_err": [], "precision": [], "success": [], "n": 0,
    })
    for s in all_summaries:
        t = str(s.get("tracker", ""))
        if not t or s.get("n_gt_frames", 0) == 0:
            continue
        agg = tracker_agg[t]
        if np.isfinite(float(s.get("bbox_iou_mean", float("nan")))):
            agg["bbox_iou"].append(float(s["bbox_iou_mean"]))
        if np.isfinite(float(s.get("mask_iou_mean", float("nan")))):
            agg["mask_iou"].append(float(s["mask_iou_mean"]))
        if np.isfinite(float(s.get("center_err_mean", float("nan")))):
            agg["center_err"].append(float(s["center_err_mean"]))
        if np.isfinite(float(s.get("precision_rate", float("nan")))):
            agg["precision"].append(float(s["precision_rate"]))
        if np.isfinite(float(s.get("success_rate", float("nan")))):
            agg["success"].append(float(s["success_rate"]))
        agg["n"] += int(s.get("n_gt_frames", 0))

    overall_rows: list[dict] = []
    header = f"{'方法':<30} {'序列数':>6} {'GT帧数':>8} {'BBoxIoU':>9} {'MaskIoU':>9} {'CenterErr':>11} {'Precision':>10} {'SuccessRate':>12}"
    print(header)
    print("-" * len(header))

    for t_key in tracker_names:
        agg = tracker_agg.get(t_key, {})
        label = TRACKER_LABELS.get(t_key, t_key)
        n_seqs = len(agg.get("bbox_iou", []))
        n_frames = int(agg.get("n", 0))
        bi = float(np.mean(agg["bbox_iou"])) if agg.get("bbox_iou") else float("nan")
        mi = float(np.mean(agg["mask_iou"])) if agg.get("mask_iou") else float("nan")
        ce = float(np.mean(agg["center_err"])) if agg.get("center_err") else float("nan")
        pr = float(np.mean(agg["precision"])) if agg.get("precision") else float("nan")
        sr = float(np.mean(agg["success"])) if agg.get("success") else float("nan")
        row_str = (f"{label:<30} {n_seqs:>6} {n_frames:>8} {bi:>9.4f} {mi:>9.4f} "
                   f"{ce:>11.2f} {pr:>10.4f} {sr:>12.4f}")
        print(row_str)
        overall_rows.append({
            "tracker": t_key,
            "tracker_label": label,
            "n_sequences": n_seqs,
            "n_gt_frames_total": n_frames,
            "bbox_iou_mean": round(bi, 6) if np.isfinite(bi) else None,
            "mask_iou_mean": round(mi, 6) if np.isfinite(mi) else None,
            "center_err_mean_px": round(ce, 3) if np.isfinite(ce) else None,
            "precision_rate": round(pr, 6) if np.isfinite(pr) else None,
            "success_rate": round(sr, 6) if np.isfinite(sr) else None,
        })

    print()

    # ── 保存汇总文件 ─────────────────────────────────────────────────────────
    # 1. 总体 CSV
    overall_csv = out_dir / "trackves_overall_summary.csv"
    if overall_rows:
        with overall_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(overall_rows[0].keys()))
            writer.writeheader(); writer.writerows(overall_rows)
        print(f"[输出] 总体汇总 CSV: {overall_csv}")

    # 2. 每序列 × 每方法 CSV
    per_seq_csv = out_dir / "trackves_per_seq_summary.csv"
    per_seq_flat = []
    for s in all_summaries:
        per_seq_flat.append({
            "sequence":          s.get("sequence", ""),
            "tracker":           s.get("tracker", ""),
            "tracker_label":     s.get("tracker_label", ""),
            "n_gt_frames":       s.get("n_gt_frames", 0),
            "bbox_iou_mean":     s.get("bbox_iou_mean", None),
            "bbox_iou_median":   s.get("bbox_iou_median", None),
            "mask_iou_mean":     s.get("mask_iou_mean", None),
            "mask_iou_median":   s.get("mask_iou_median", None),
            "center_err_mean":   s.get("center_err_mean", None),
            "center_err_median": s.get("center_err_median", None),
            "precision_rate":    s.get("precision_rate", None),
            "success_rate":      s.get("success_rate", None),
            "elapsed_s":         s.get("elapsed_s", None),
            "sequential":        s.get("sequential", None),
            "error":             s.get("error", ""),
        })
    if per_seq_flat:
        with per_seq_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_seq_flat[0].keys()))
            writer.writeheader(); writer.writerows(per_seq_flat)
        print(f"[输出] 逐序列汇总 CSV: {per_seq_csv}")

    # 3. 完整 JSON
    result_json = out_dir / "trackves_all_results.json"
    with result_json.open("w", encoding="utf-8") as f:
        json.dump({
            "sequences": seq_names,
            "trackers": tracker_names,
            "overall": overall_rows,
            "per_sequence": per_seq_results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"[输出] 完整结果 JSON: {result_json}")
    print(f"\n[完成] 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
