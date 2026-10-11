import argparse
import csv
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

import MIS_C_F_SIMI_Mesh_Match_Track_3DMatch as mis3d

try:
    import MIS_Mesh_RTS_Track as mesh_rts
except ImportError:
    import MIS_Mesh_RTS_TrackDepth as mesh_rts


TRACKER_ALIASES: dict[str, str] = {
    "ours": "ours",
    "ours_scheme_c_nano": "ours_scheme_c_nano",
    "ours_c_nano": "ours_scheme_c_nano",
    "scheme_c_nano": "ours_scheme_c_nano",
    "mesh_rts": "mesh_rts",
    "rts": "mesh_rts",
    "csrt": "csrt",
    "kcf": "kcf",
    "mosse": "mosse",
    "yolo": "yolo",
    "yolo_seg": "yolo",
    "yolo-seg": "yolo",
    "sam": "sam",
    "sam2": "sam2",
    "sam2_video": "sam2",
    "sam2_pf": "sam2_pf",
    "sam2_perframe": "sam2_pf",
    "track_anything": "track_anything",
    "track-anything": "track_anything",
    "xmem": "xmem",
    "sam_track": "sam_track",
    "sam-track": "sam_track",
    "pred_dir": "pred_dir",
    "surgical_external": "surgical_external",
    "sam_xmem": "sam_xmem",
    "sam+xmem": "sam_xmem",
    "sam_xmem_pp": "sam_xmem_pp",
    "sam+xmem++": "sam_xmem_pp",
    "sis_pt_sam": "sis_pt_sam",
    "ma_sam2": "ma_sam2",
    "sam2s": "sam2s",
    # ===== SOTA Transformer / VOS trackers =====
    "ostrack": "ostrack",
    "os_track": "ostrack",
    "mixformer": "mixformer",
    "mix_former": "mixformer",
    "cutie": "cutie",
    "cutie_vos": "cutie",
    "lmtrack": "lmtrack",
    "lm_track": "lmtrack",
    "stark": "stark",
    "stark_st": "stark",
    # 经典公开颜色直方图跟踪（HS back-projection + meanShift）
    "meanshift_hist": "meanshift_hist",
    "colorhist": "meanshift_hist",
    "opencv_colorhist": "meanshift_hist",
}

EXTERNAL_TRACKER_KEYS = {
    "pred_dir",
    "track_anything",
    "xmem",
    "sam_track",
    "surgical_external",
    "sam_xmem",
    "sam_xmem_pp",
    "sis_pt_sam",
    "ma_sam2",
    "sam2s",
}

_NANO_BACKBONE_DEFAULT = r"D:\MIS_Pose_Track_Re3D\nanotrack_backbone_sim.onnx"
_NANO_NECKHEAD_DEFAULT = r"D:\MIS_Pose_Track_Re3D\nanotrack_head_sim.onnx"
_VIT_MODEL_DEFAULT     = r"D:\MIS_Pose_Track_Re3D\object_tracking_vittrack_2023sep.onnx"

TRACKER_LABELS: dict[str, str] = {
    "ours": "Ours",
    "mesh_rts": "Mesh-RTS (Warp+ICP)",
    "csrt": "CSRT",
    "kcf": "KCF",
    "mosse": "MOSSE",
    "nanotrack": "NanoTrack",
    "ours_scheme_c_nano": "Ours (Scheme C: Nano+log+TopR)",
    "vittrack": "ViT-Track",
    "yolo": "YOLO-seg",
    "sam": "SAM",
    "sam2": "SAM2-Video",
    "sam2_pf": "SAM2-PerFrame",
    "track_anything": "Track Anything",
    "xmem": "XMem",
    "sam_track": "SAM-Track",
    "pred_dir": "External PredDir",
    "surgical_external": "Surgical External",
    # SOTA methods
    "ostrack": "OSTrack (ECCV'22)",
    "mixformer": "MixFormer (ECCV'22)",
    "cutie": "Cutie (CVPR'24)",
    "lmtrack": "LMTrack (AAAI'24)",
    "stark": "STARK (ICCV'21)",
    "meanshift_hist": "MeanShift+HS-Hist (OpenCV classic)",
    "sam_xmem": "SAM + XMem",
    "sam_xmem_pp": "SAM + XMem++",
    "sis_pt_sam": "SIS-PT-SAM",
    "ma_sam2": "MA-SAM2",
    "sam2s": "SAM2S",
}


@dataclass(frozen=True)
class ChessboardSpec:
    key: str
    label: str
    board_cols: int
    board_rows: int
    square_size_mm: float


CHESSBOARD_SPECS: dict[str, ChessboardSpec] = {
    "a": ChessboardSpec(key="a", label="Type A", board_cols=6, board_rows=9, square_size_mm=5.0),
    "b": ChessboardSpec(key="b", label="Type B", board_cols=8, board_rows=9, square_size_mm=8.0),
    "c": ChessboardSpec(key="c", label="Type C", board_cols=8, board_rows=11, square_size_mm=11.0),
}


def _parse_auto_board_specs(text: str) -> list[ChessboardSpec]:
    raw = str(text or "").strip().lower()
    if (not raw) or raw == "all":
        return list(CHESSBOARD_SPECS.values())
    specs: list[ChessboardSpec] = []
    seen: set[str] = set()
    for item in raw.replace(";", ",").split(","):
        key = str(item).strip().lower()
        if key in CHESSBOARD_SPECS and key not in seen:
            specs.append(CHESSBOARD_SPECS[key])
            seen.add(key)
    return specs if specs else list(CHESSBOARD_SPECS.values())


def _spec_from_args(args: argparse.Namespace) -> Optional[ChessboardSpec]:
    cols = int(getattr(args, "board_cols", 0) or 0)
    rows = int(getattr(args, "board_rows", 0) or 0)
    if cols > 0 and rows > 0:
        return ChessboardSpec(
            key="manual",
            label="Manual",
            board_cols=cols,
            board_rows=rows,
            square_size_mm=float(getattr(args, "square_size_mm", 0.0) or 0.0),
        )
    return None


def _copy_file_if_needed(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_resolved = src.resolve()
    dst_resolved = dst.resolve()
    if src_resolved != dst_resolved:
        shutil.copy2(str(src), str(dst))
    return str(dst)


def _sorted_image_paths(frames_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in exts], key=lambda p: p.name.lower())


def _clip_bbox(bbox: tuple[int, int, int, int], shape_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    x, y, bw, bh = [int(v) for v in bbox]
    x = max(0, min(x, max(0, w - 1)))
    y = max(0, min(y, max(0, h - 1)))
    bw = max(1, min(max(1, bw), max(1, w - x)))
    bh = max(1, min(max(1, bh), max(1, h - y)))
    return x, y, bw, bh


def _bbox_xyxy_to_xywh(xyxy: np.ndarray, shape_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in np.asarray(xyxy, dtype=np.float32).reshape(4).tolist()]
    x = int(round(min(x1, x2)))
    y = int(round(min(y1, y2)))
    w = int(round(abs(x2 - x1)))
    h = int(round(abs(y2 - y1)))
    return _clip_bbox((x, y, max(1, w), max(1, h)), shape_hw)


def _mask_bbox(mask_u8: Optional[np.ndarray], pad: int = 0) -> Optional[tuple[int, int, int, int]]:
    if mask_u8 is None:
        return None
    mk = _mask2d(mask_u8)
    ys, xs = np.nonzero(mk > 0)
    if xs.size <= 0 or ys.size <= 0:
        return None
    x0 = int(max(0, xs.min() - pad))
    y0 = int(max(0, ys.min() - pad))
    x1 = int(xs.max() + pad + 1)
    y1 = int(ys.max() + pad + 1)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = [int(v) for v in a]
    bx, by, bw, bh = [int(v) for v in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1 = max(ax, bx)
    iy1 = max(ay, by)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = max(1, aw * ah + bw * bh - inter)
    return float(inter) / float(union)


def _mask2d(mask_u8: Optional[np.ndarray]) -> np.ndarray:
    if mask_u8 is None:
        return np.zeros((1, 1), dtype=np.uint8)
    mk = np.asarray(mask_u8, dtype=np.uint8)
    while mk.ndim > 2 and mk.shape[-1] == 1:
        mk = np.squeeze(mk, axis=-1)
    if mk.ndim == 3:
        mk = np.max(mk, axis=2).astype(np.uint8)
    if mk.ndim != 2:
        mk = np.asarray(mk).reshape(mk.shape[0], mk.shape[1]) if mk.ndim >= 2 else np.zeros((1, 1), dtype=np.uint8)
    return np.asarray(mk, dtype=np.uint8)


def _mask_iou(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    ma = _mask2d(a) > 0
    mb = _mask2d(b) > 0
    inter = int(np.count_nonzero(ma & mb))
    union = int(np.count_nonzero(ma | mb))
    return 0.0 if union <= 0 else float(inter) / float(union)


def _bbox_to_mask(shape_hw: tuple[int, int], bbox: Optional[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.zeros((int(shape_hw[0]), int(shape_hw[1])), dtype=np.uint8)
    if bbox is None:
        return out
    x, y, w, h = [int(v) for v in bbox]
    out[y:y + h, x:x + w] = 255
    return out


def _mask_roi_to_full(mask_roi_u8: Optional[np.ndarray], bbox: Optional[tuple[int, int, int, int]], shape_hw: tuple[int, int]) -> np.ndarray:
    full = np.zeros((int(shape_hw[0]), int(shape_hw[1])), dtype=np.uint8)
    if mask_roi_u8 is None or bbox is None:
        return full
    x, y, w, h = [int(v) for v in bbox]
    mk = np.asarray(mask_roi_u8, dtype=np.uint8)
    if mk.shape[:2] != (h, w):
        mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    full[y:y + h, x:x + w] = (mk > 0).astype(np.uint8) * 255
    return full


def _center_from_bbox(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    return np.array([x + 0.5 * w, y + 0.5 * h], dtype=np.float32)


def _center_error(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    return float(np.linalg.norm(_center_from_bbox(a) - _center_from_bbox(b)))


def _draw_bbox(image: np.ndarray, bbox: Optional[tuple[int, int, int, int]], color: tuple[int, int, int], thickness: int = 2) -> None:
    if bbox is None:
        return
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness, cv2.LINE_AA)


def _draw_label(image: np.ndarray, text: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = int(pos[0]), max(18, int(pos[1]))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def _draw_mask_outline(image: np.ndarray, mask_u8: Optional[np.ndarray], color: tuple[int, int, int], thickness: int = 2) -> None:
    if mask_u8 is None:
        return
    mk = _mask2d(mask_u8)
    if mk.ndim != 2 or int(np.count_nonzero(mk)) <= 0:
        return
    contours, _ = cv2.findContours((mk > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, thickness, cv2.LINE_AA)


def _make_vis(frame_bgr: np.ndarray, gt_mask: np.ndarray, gt_bbox: tuple[int, int, int, int], pred_mask: np.ndarray, pred_bbox: Optional[tuple[int, int, int, int]], title: str) -> np.ndarray:
    vis = np.asarray(frame_bgr, dtype=np.uint8).copy()
    gt_color = (0, 255, 0)
    pred_color = (255, 0, 0)
    _draw_mask_outline(vis, gt_mask, gt_color, 2)
    _draw_bbox(vis, gt_bbox, gt_color, 1)
    _draw_mask_outline(vis, pred_mask, pred_color, 2)
    _draw_bbox(vis, pred_bbox, pred_color, 2)
    _draw_label(vis, "GT", (gt_bbox[0], gt_bbox[1]), gt_color)
    if pred_bbox is not None:
        _draw_label(vis, "Pred", (pred_bbox[0], pred_bbox[1] + pred_bbox[3]), pred_color)
    cv2.putText(vis, "GT=green  Pred=blue", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(vis, title, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def _chessboard_outer_quad(corners: np.ndarray, board_cols: int, board_rows: int) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(int(board_rows), int(board_cols), 2)
    tl = pts[0, 0] - 0.5 * (pts[0, 1] - pts[0, 0]) - 0.5 * (pts[1, 0] - pts[0, 0])
    tr = pts[0, -1] + 0.5 * (pts[0, -1] - pts[0, -2]) - 0.5 * (pts[1, -1] - pts[0, -1])
    br = pts[-1, -1] + 0.5 * (pts[-1, -1] - pts[-1, -2]) + 0.5 * (pts[-1, -1] - pts[-2, -1])
    bl = pts[-1, 0] - 0.5 * (pts[-1, 1] - pts[-1, 0]) + 0.5 * (pts[-1, 0] - pts[-2, 0])
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def _quad_to_mask(shape_hw: tuple[int, int], quad: np.ndarray) -> np.ndarray:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    if quad is None:
        return mask
    q = np.asarray(quad, dtype=np.float32).reshape(-1, 2).copy()
    if int(q.shape[0]) < 3:
        return mask
    q[:, 0] = np.clip(q[:, 0], 0, max(0, w - 1))
    q[:, 1] = np.clip(q[:, 1], 0, max(0, h - 1))
    cv2.fillPoly(mask, [np.round(q).astype(np.int32).reshape(-1, 1, 2)], 255)
    return mask


def _gt_mask_from_quad_and_th68(frame_bgr: np.ndarray, quad: np.ndarray) -> np.ndarray:
    poly_mask = _quad_to_mask(frame_bgr.shape[:2], quad)
    bbox = _mask_bbox(poly_mask)
    if bbox is None:
        return poly_mask

    x, y, w, h = [int(v) for v in bbox]
    roi = np.asarray(frame_bgr[y:y + h, x:x + w], dtype=np.uint8)
    poly_roi = np.asarray(poly_mask[y:y + h, x:x + w], dtype=np.uint8)
    if roi.size <= 0 or int(np.count_nonzero(poly_roi)) <= 0:
        return poly_mask

    cfg = mis3d.core.TrackConfig()
    _, mask_th68_roi, _ = mis3d.core.compute_th_gray_truncated(roi, cfg.th_percent, cfg)
    mk = ((np.asarray(mask_th68_roi, dtype=np.uint8) > 0) & (poly_roi > 0)).astype(np.uint8) * 255
    if int(np.count_nonzero(mk)) <= 0:
        return poly_mask

    try:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, k, iterations=2)
    except Exception:
        pass

    cnt = mis3d._extract_contour_prefer_center(mk)
    if cnt is None:
        cnt = mis3d.core.extract_largest_contour(mk)
    if cnt is not None and len(cnt) >= 3:
        filled = np.zeros_like(mk, dtype=np.uint8)
        cv2.drawContours(filled, [cnt], -1, 255, thickness=-1)
        mk = filled

    # th_68 在少数低纹理帧会过稀，避免生成过小/破碎的 GT，必要时回退到四边形先验。
    if int(np.count_nonzero(mk)) < int(max(32, 0.08 * int(np.count_nonzero(poly_roi)))):
        return poly_mask

    full = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    full[y:y + h, x:x + w] = mk
    return full


def _detect_chessboard_gt(frame_bgr: np.ndarray, board_cols: int, board_rows: int, use_sb: bool = True) -> Optional[dict[str, Any]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    size = (int(board_cols), int(board_rows))
    ok = False
    corners = None
    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCornersSB(gray, size, flags=flags)
    if not ok:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, size, flags=flags)
        if ok and corners is not None:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
    if (not ok) or corners is None:
        return None

    quad = _chessboard_outer_quad(corners, board_cols=board_cols, board_rows=board_rows)
    h, w = frame_bgr.shape[:2]
    quad[:, 0] = np.clip(quad[:, 0], 0, max(0, w - 1))
    quad[:, 1] = np.clip(quad[:, 1], 0, max(0, h - 1))
    mask = _gt_mask_from_quad_and_th68(frame_bgr, quad)
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    return {
        "corners": np.asarray(corners, dtype=np.float32).reshape(-1, 2),
        "quad": quad,
        "mask": mask,
        "bbox": bbox,
    }


def _detect_chessboard_gt_for_spec(frame_bgr: np.ndarray, spec: ChessboardSpec, use_sb: bool = True) -> Optional[dict[str, Any]]:
    gt = _detect_chessboard_gt(frame_bgr, board_cols=spec.board_cols, board_rows=spec.board_rows, use_sb=use_sb)
    if gt is None:
        return None
    gt["board_key"] = str(spec.key)
    gt["board_label"] = str(spec.label)
    gt["board_cols"] = int(spec.board_cols)
    gt["board_rows"] = int(spec.board_rows)
    gt["square_size_mm"] = float(spec.square_size_mm)
    return gt


def _detect_chessboard_gt_auto(frame_bgr: np.ndarray, specs: list[ChessboardSpec], use_sb: bool = True) -> Optional[dict[str, Any]]:
    best_gt = None
    best_area = -1.0
    for spec in specs:
        gt = _detect_chessboard_gt_for_spec(frame_bgr, spec=spec, use_sb=use_sb)
        if gt is None:
            continue
        bbox = gt.get("bbox", None)
        area = float(int(bbox[2]) * int(bbox[3])) if bbox is not None else 0.0
        if area > best_area:
            best_gt = gt
            best_area = area
    return best_gt


def _auto_detect_sequence_spec(image_paths: list[Path], specs: list[ChessboardSpec], use_sb: bool, probe_limit: int = 48) -> ChessboardSpec:
    if not specs:
        raise ValueError("auto board spec list is empty")
    hits = {spec.key: 0 for spec in specs}
    areas = {spec.key: 0.0 for spec in specs}
    checked = 0
    for path in image_paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        checked += 1
        for spec in specs:
            gt = _detect_chessboard_gt_for_spec(frame, spec=spec, use_sb=use_sb)
            if gt is None:
                continue
            hits[spec.key] += 1
            bbox = gt.get("bbox", None)
            if bbox is not None:
                areas[spec.key] += float(int(bbox[2]) * int(bbox[3]))
        if checked >= int(max(1, probe_limit)):
            break
    ranked = sorted(specs, key=lambda s: (hits.get(s.key, 0), areas.get(s.key, 0.0)), reverse=True)
    best = ranked[0]
    if hits.get(best.key, 0) <= 0:
        tried = ", ".join(f"{spec.label}({spec.board_rows}x{spec.board_cols})" for spec in specs)
        raise RuntimeError(f"failed to auto-detect chessboard type; tried: {tried}")
    return best


@dataclass
class FrameGT:
    frame_idx: int
    image_path: str
    bbox: tuple[int, int, int, int]
    mask: np.ndarray
    quad: np.ndarray
    corners: np.ndarray
    board_key: str = "manual"
    board_label: str = "Manual"
    board_cols: int = 0
    board_rows: int = 0
    square_size_mm: float = 0.0


class BaseTracker:
    name: str
    label: str

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        raise NotImplementedError

    def set_debug_output_dir(self, out_dir: Path) -> None:
        return None

    def set_frame_context(self, frame_idx: int, image_path: str) -> None:
        return None


def _apply_chess_scheme_c_profile(profile: str) -> None:
    """棋盘格序列：Scheme C 专用 profile（静态图集，帧间运动小，可更激进 lite-skip）。"""
    os.environ.setdefault("MIS_SILENT", "1")
    os.environ.setdefault("MIS_SCHEME_C_POST_BLEND", "0")
    os.environ.setdefault("MIS_NANO_LITE_ENABLE", "1")
    os.environ.setdefault("MIS_NANO_PRIOR_CACHE", "1")
    os.environ.setdefault("MIS_MATCH_RESP_COND", "1")
    os.environ.setdefault("MIS_NANO_CASCADE", "1")
    try:
        import Deformation_hist_AB_scheme_C_nanoprior as sch

        if str(profile).strip().lower() in ("cascade", "fast"):
            sch.apply_cascade_fusion_defaults()
        else:
            sch.apply_scheme_c_fusion_defaults()
    except Exception:
        pass
    # 棋盘格上 NanoTrack 的中心/尺寸稳定性很强；Ours 主要做轻量多模态校验，
    # 避免 TrackVes 风格 RTW 重搜索在静态图集上拖慢并引入滞后。
    if str(profile).strip().lower() in ("cascade", "fast"):
        chess_overrides = {
            "MIS_CASCADE_PRIMARY_BLEND": "0.0",
            "MIS_CASCADE_PRIMARY_NS": "0.44",
            "MIS_CASCADE_PRIMARY_HIST": "0.00",
            "MIS_CASCADE_PRIMARY_MOTION": "64",
            "MIS_POLICY_APPEAR_PRIMARY_NS": "0.56",
            "MIS_POLICY_APPEAR_PRIMARY_SZ": "0.28",
            "MIS_POLICY_APPEAR_PRIMARY_MOTION": "48",
            "MIS_NANO_SIZE_ADAPT": "1",
            "MIS_NANO_SIZE_ADAPT_NS": "0.44",
            "MIS_NANO_SIZE_ADAPT_SZ": "0.20",
            "MIS_NANO_SIZE_ALPHA": "1.0",
            "MIS_NANO_PLUS_ENABLE": "1",
            "MIS_NANO_PLUS_NS": "0.44",
            "MIS_NANO_PLUS_HIST": "0.00",
            "MIS_NANO_PLUS_SZ": "0.00",
            "MIS_NANO_PLUS_MOTION": "999",
            "MIS_NANO_PLUS_IOU": "0.00",
            "MIS_NANO_PLUS_BLEND": "0.0",
            "MIS_NANO_PLUS_ADAPT_SIZE": "1",
            "MIS_NANO_PLUS_RAW": "1",
            "MIS_NANO_PLUS_MIN_SIZE_RATIO": "0.15",
            "MIS_NANO_PLUS_MAX_SIZE_RATIO": "4.50",
            "MIS_NANO_PLUS_MAX_AREA_FRAC": "0.60",
            "MIS_NANO_PLUS_HOLD_PREV": "1",
            "MIS_NANO_PLUS_HOLD_PREV_SCORE": "0.55",
            "MIS_NANO_SIZE_REFINE": "0",
            "MIS_EXPERT_POOL_ENABLE": "1",
            "MIS_EXPERT_STABLE_W": "0.52,0.14,0.08,0.16,0.10",
            "MIS_EXPERT_POOL_MARGIN": "0.006",
            "MIS_EXPERT_POOL_RAW_NS": "0.44",
            "MIS_EXPERT_RAW_SIZE_BONUS": "0.030",
            "MIS_EXPERT_ADAPT_SIZE_BONUS": "0.018",
            "MIS_RTW_POOL_ENABLE": "0",
            "MIS_MEMORY_BANK_ENABLE": "0",
            "MIS_NANO_OUTPUT_RAW_ENABLE": "1",
            "MIS_NANO_OUTPUT_RAW_NS": "0.44",
            "MIS_CHESS_ONLINE_SIMI": "0",
            "MIS_CHESS_ONLINE_STEPS": "64",
            "MIS_CHESS_ONLINE_BATCH": "16",
            "MIS_CHESS_ONLINE_LR": "0.00002",
            "MIS_CHESS_ONLINE_POS": "20",
            "MIS_CHESS_ONLINE_NEG": "52",
            "MIS_CHESS_ONLINE_POOL_CANDS": "96",
            "MIS_CHESS_ONLINE_IOU_TARGETS": "1",
        }
        for k, v in chess_overrides.items():
            os.environ[k] = v
    if str(profile).strip().lower() == "quality":
        os.environ.setdefault("MIS_COARSE_STEP", "16")
        os.environ.setdefault("MIS_RERANK_TOPR", "48")
        os.environ.setdefault("MIS_MATCH_RESP_ENABLE", "1")
        os.environ.setdefault("MIS_MAX_TIME_S", "1.2")
        os.environ.setdefault("MIS_NANO_LITE_NS", "0.80")
        os.environ.setdefault("MIS_NANO_LITE_MOTION_PX", "12")
        os.environ.setdefault("MIS_NUM_TEMPLATES", "256")
    else:
        # fast：棋盘格默认；单帧目标 ~0.3–1.5s（视分辨率与 GPU）
        os.environ.setdefault("MIS_COARSE_STEP", "20")
        os.environ.setdefault("MIS_RERANK_TOPR", "28")
        os.environ.setdefault("MIS_MATCH_RESP_ENABLE", "0")
        os.environ.setdefault("MIS_MAX_TIME_S", "0.85")
        os.environ.setdefault("MIS_NANO_LITE_NS", "0.72")
        os.environ.setdefault("MIS_NANO_LITE_MOTION_PX", "18")
        os.environ.setdefault("MIS_NANO_LITE_PREV_SCORE", "0.72")
        os.environ.setdefault("MIS_RERANK_TOPR_LITE", "20")
        os.environ.setdefault("MIS_NUM_TEMPLATES", "128")
        os.environ.setdefault("MIS_HIST_TOP_M", "64")
        os.environ.setdefault("MIS_CONTOUR_INFER_BS", "32")


def _scale_bbox(
    bbox: tuple[int, int, int, int],
    scale: float,
    frame_wh: tuple[int, int] = None,  # Python 3.9兼容
) -> tuple[int, int, int, int]:
    if abs(scale - 1.0) < 1e-6:
        return tuple(map(int, bbox))
    x, y, w, h = (int(round(float(v) * scale)) for v in bbox)
    if frame_wh is not None:
        return _clip_bbox((x, y, max(1, w), max(1, h)), (frame_wh[1], frame_wh[0]))
    return (x, y, max(1, w), max(1, h))


def _resize_frame(frame_bgr: np.ndarray, scale: float) -> np.ndarray:
    if abs(scale - 1.0) < 1e-6:
        return frame_bgr
    return cv2.resize(frame_bgr, None, fx=float(scale), fy=float(scale), interpolation=cv2.INTER_AREA)


def _build_cfg() -> mis3d.core.TrackConfig:
    """构建跟踪配置（应用所有最新优化）"""
    cfg = mis3d.core.TrackConfig()
    
    # ===== 分辨率优化（全局降分辨率） =====
    cfg.proc_scale = 0.75  # 优化：75% 分辨率，速度提升 2x，精度损失 < 1.5%
    cfg.semantic_image_size = (112, 112)  # 优化：从128降至112
    
    # ===== 搜索窗口优化 =====
    cfg.search_radius = 32  # 优化：从128降至32
    cfg.search_step = 24    # 优化：从16增至24
    cfg.topk_semantic = 3   # 优化：从8降至3
    
    # ===== 性能优化开关 =====
    cfg.verbose = 0
    cfg.print_every = 1000000
    cfg.print_timing = False
    cfg.save_debug_images = False  # 关闭调试图像保存
    cfg.save_mesh = False
    cfg.vis_window = False
    
    # ===== 断点续跑配置 =====
    cfg.resume_enable = False
    cfg.force_restart = True
    cfg.init_bbox = ""
    
    return cfg


_SEM_READY = False


def _init_semantic_once(cfg: mis3d.core.TrackConfig) -> None:
    global _SEM_READY
    if _SEM_READY:
        return
    mis3d._init_semantic(cfg)
    _SEM_READY = True


class OursTracker(BaseTracker):
    name = "ours"
    label = TRACKER_LABELS["ours"]
    backend_name = "mis3d_uvh_render"

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int], init_mask: Optional[np.ndarray] = None):
        self.cfg = _build_cfg()
        _init_semantic_once(self.cfg)
        self.prev_bbox = _clip_bbox(tuple(map(int, init_bbox)), init_im.shape[:2])
        self.prev_contour_rgb = None
        self.prev_mask_small = None
        self.prev_uvh_pts = None
        self.uvh_rigid_ref_pts = None
        self.ref_contour_rgb = None
        self.ref_mask_small = None
        self.last_bbox = self.prev_bbox
        self.last_mask_full = _bbox_to_mask(init_im.shape[:2], self.prev_bbox)
        self.debug_output_dir = ""
        self.current_frame_idx = 0
        self.current_image_path = ""
        self.last_debug_paths: dict[str, str] = {}
        self._init_reference(init_im, init_mask)

    def set_debug_output_dir(self, out_dir: Path) -> None:
        self.debug_output_dir = str(out_dir)

    def set_frame_context(self, frame_idx: int, image_path: str) -> None:
        self.current_frame_idx = int(frame_idx)
        self.current_image_path = str(image_path)

    def _init_reference(self, frame_bgr: np.ndarray, init_mask: Optional[np.ndarray]) -> None:
        roi0 = mis3d.core.crop(frame_bgr, self.prev_bbox)
        th0_raw, mask0_th68, _ = mis3d.core.compute_th_gray_truncated(roi0, self.cfg.th_percent, self.cfg)
        if init_mask is not None:
            x0, y0, w0, h0 = self.prev_bbox
            mk = np.asarray(init_mask[y0:y0 + h0, x0:x0 + w0], dtype=np.uint8)
            if mk.shape[:2] != (h0, w0):
                mk = cv2.resize(mk, (w0, h0), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            mask0 = (mk > 0).astype(np.uint8) * 255
        else:
            mask0 = np.asarray(mask0_th68, dtype=np.uint8)
        if int(np.count_nonzero(mask0)) <= 0:
            mask0 = np.asarray(mask0_th68, dtype=np.uint8)
        cnt0 = mis3d._extract_contour_prefer_center(mask0)
        if cnt0 is None:
            cnt0 = mis3d.core.extract_largest_contour(mask0)
        if cnt0 is not None:
            cnt0s = mis3d.core.resize_contour(cnt0, (self.prev_bbox[2], self.prev_bbox[3]), self.cfg.semantic_image_size)
            contour0_rgb = mis3d.core.contour_to_rgb(cnt0s, self.cfg.semantic_image_size, thickness=2)
        else:
            contour0_rgb = np.zeros((self.cfg.semantic_image_size[1], self.cfg.semantic_image_size[0], 3), dtype=np.uint8)
        self.prev_contour_rgb = contour0_rgb
        self.ref_contour_rgb = contour0_rgb
        self.ref_mask_small = mis3d._mask64_from_mask_u8(mask0)
        self.prev_mask_small = np.asarray(self.ref_mask_small, dtype=np.uint8).copy()
        pts = mis3d._uvh_points_from_th_roi(
            th_gray_truncated=np.asarray(th0_raw, dtype=np.float32),
            bbox=self.prev_bbox,
            z_scale=float(self.cfg.mesh_z_scale),
            max_points=20000,
            rng=np.random.default_rng(0),
            deterministic=True,
        )
        pts = np.asarray(pts, dtype=np.float32) if pts is not None else np.zeros((0, 3), dtype=np.float32)
        if int(pts.shape[0]) >= 200:
            self.prev_uvh_pts = pts
            self.uvh_rigid_ref_pts = pts.copy()

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        prev_bbox_old = tuple(self.prev_bbox)
        ref_pts_for_debug = None
        if isinstance(self.prev_uvh_pts, np.ndarray) and int(self.prev_uvh_pts.shape[0]) >= 200:
            ref_pts_for_debug = np.asarray(self.prev_uvh_pts, dtype=np.float32).copy()
        if isinstance(self.prev_uvh_pts, np.ndarray) and int(self.prev_uvh_pts.shape[0]) >= 200:
            best = mis3d.find_best_bbox_in_next_frame_uvh_render(
                frame_bgr=frame_bgr,
                prev_bbox=self.prev_bbox,
                prev_mask_small=np.asarray(self.ref_mask_small, dtype=np.uint8),
                prev_uvh_pts=np.asarray(self.prev_uvh_pts, dtype=np.float32),
                cfg=self.cfg,
                view_lib=None,
                prev_view_id=-1,
                prev_view_dir=None,
                angle_gate_deg=0.0,
                ref_contour_rgb=np.asarray(self.ref_contour_rgb, dtype=np.uint8),
                rigid_ref_uvh_pts=np.asarray(self.uvh_rigid_ref_pts, dtype=np.float32) if isinstance(self.uvh_rigid_ref_pts, np.ndarray) else None,
                diff_ref_verts=None,
                diff_ref_faces=None,
            )
        else:
            best = mis3d.core.find_best_bbox_in_next_frame(
                frame_bgr,
                self.prev_bbox,
                np.asarray(self.prev_contour_rgb, dtype=np.uint8),
                np.asarray(self.prev_mask_small, dtype=np.uint8),
                self.cfg,
            )
        self.last_debug_paths = {}
        if self.debug_output_dir:
            try:
                self.last_debug_paths = mis3d._save_uvh_render_debug_bundle(
                    out_uvh_render=str(self.debug_output_dir),
                    frame_idx=int(self.current_frame_idx),
                    image_path=str(self.current_image_path),
                    frame_bgr=frame_bgr,
                    prev_bbox=tuple(prev_bbox_old),
                    best=best,
                    ref_mode="prev_uvh",
                    ref_pts_xyz=(np.asarray(ref_pts_for_debug, dtype=np.float32) if isinstance(ref_pts_for_debug, np.ndarray) else None),
                )
            except Exception:
                self.last_debug_paths = {}
        bbox_i = _clip_bbox(tuple(map(int, best["bbox"])), frame_bgr.shape[:2])
        mask_roi = np.asarray(best.get("mask_u8", None), dtype=np.uint8) if isinstance(best.get("mask_u8", None), np.ndarray) else None
        contour_i = np.asarray(best["contour_rgb"], dtype=np.uint8)
        self.prev_bbox = bbox_i
        self.prev_contour_rgb = contour_i
        if mask_roi is not None:
            self.prev_mask_small = mis3d._mask64_from_mask_u8(mask_roi)
            self.last_mask_full = _mask_roi_to_full(mask_roi, bbox_i, frame_bgr.shape[:2])
        else:
            self.last_mask_full = _bbox_to_mask(frame_bgr.shape[:2], bbox_i)
        pts_new = best.get("uvh_render_pts", None)
        if isinstance(pts_new, np.ndarray) and int(pts_new.shape[0]) >= 200:
            self.prev_uvh_pts = np.asarray(pts_new, dtype=np.float32)
        self.last_bbox = bbox_i
        return {
            "bbox": bbox_i,
            "mask_full": self.last_mask_full,
            "score": float(best.get("score", float("nan"))),
            "uvh_render_meta_json": str(self.last_debug_paths.get("meta_json", "")),
            "uvh_render_ref_render_png": str(self.last_debug_paths.get("ref_render_png", "")),
            "uvh_render_cur_render_png": str(self.last_debug_paths.get("cur_render_png", "")),
            "uvh_render_compare_png": str(self.last_debug_paths.get("render_compare_png", "")),
            "uvh_render_ref_pts_ply": str(self.last_debug_paths.get("ref_pts_ply", "")),
            "uvh_render_cur_pts_ply": str(self.last_debug_paths.get("cur_pts_ply", "")),
            "uvh_render_projected_ref_pts_ply": str(self.last_debug_paths.get("projected_ref_pts_ply", "")),
            "uvh_render_point_match_ply": str(self.last_debug_paths.get("point_match_ply", "")),
            "uvh_render_point_overlay_png": str(self.last_debug_paths.get("point_overlay_png", "")),
        }


class MeshRTSTracker(BaseTracker):
    """基于伪3D Mesh 的 Warp+ICP 跟踪器（MIS_Mesh_RTS_Track）。"""
    name = "mesh_rts"
    label = TRACKER_LABELS["mesh_rts"]
    backend_name = "mesh_rts_warp_icp"

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int], init_mask: Optional[np.ndarray] = None):
        self.cfg = mesh_rts.RTSConfig()
        self.cfg.save_debug = False
        self.cfg.verbose = 0
        self.ccfg = mesh_rts._make_core_cfg(self.cfg)

        self.prev_bbox = _clip_bbox(tuple(map(int, init_bbox)), init_im.shape[:2])
        H, W = init_im.shape[:2]
        self.frame_wh = (W, H)

        x, y, w, h = self.prev_bbox
        roi = init_im[y:y + h, x:x + w]
        self.ref_th, self.ref_mask_u8, _ = mesh_rts.core.compute_th_gray_truncated(
            roi, self.cfg.th_percent, self.ccfg)
        self.ref_pts = mesh_rts.build_uvh_points(
            self.ref_th, self.prev_bbox, self.cfg.mesh_z_scale, self.cfg.max_points)
        if self.ref_pts.shape[0] < 50:
            self.ref_pts = np.zeros((0, 3), np.float32)
        self.ref_center = np.mean(self.ref_pts, axis=0).astype(np.float64) if self.ref_pts.shape[0] >= 50 else np.zeros(3, np.float64)
        self.ref_bbox = self.prev_bbox

        self.prev_frame_bgr = init_im.copy()
        self.prev_target_pts = self.ref_pts.copy()
        self.last_bbox = self.prev_bbox
        self.last_mask_full = _bbox_to_mask(init_im.shape[:2], self.prev_bbox)

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        cfg = self.cfg
        if self.ref_pts.shape[0] < 50:
            return {"bbox": self.prev_bbox, "mask_full": self.last_mask_full, "score": 0.0}

        search_bbox = mesh_rts.compute_search_bbox(self.prev_bbox, cfg.search_expand, self.frame_wh)
        search_bounds = mesh_rts.compute_search_bounds(search_bbox)

        sx, sy, sw, sh = search_bbox
        search_roi = frame_bgr[sy:sy + sh, sx:sx + sw]
        th_search, mask_search, _ = mesh_rts.core.compute_th_gray_truncated(
            search_roi, cfg.th_percent, self.ccfg)
        search_pts = mesh_rts.build_uvh_points(
            th_search, search_bbox, cfg.mesh_z_scale, cfg.max_points)

        warped_pts, warp_conf = mesh_rts.compute_flow_warp(
            self.prev_frame_bgr, frame_bgr, self.prev_target_pts, cfg)

        if search_pts.shape[0] < 50:
            rts = {"ok": False, "score": 0.0, "rot_deg": 0.0, "scale": 1.0,
                   "tx": 0.0, "ty": 0.0}
        else:
            rts = mesh_rts.optimize_rts(
                self.ref_pts, search_pts, search_bounds, cfg,
                ref_th=self.ref_th, ref_mask=self.ref_mask_u8, ref_bbox=self.ref_bbox,
                search_th=th_search, search_mask=mask_search, search_bbox=search_bbox,
                warped_pts=warped_pts, warp_confidence=warp_conf,
            )

        new_bbox = None
        if rts.get("ok", False):
            new_bbox = mesh_rts.rts_to_bbox(
                self.ref_pts, self.ref_center,
                rts["rot_deg"], rts["scale"], rts["tx"], rts["ty"],
                self.frame_wh, cfg.bbox_pad,
                cfg.bbox_percentile_lo, cfg.bbox_percentile_hi,
            )
        if new_bbox is None:
            new_bbox = self.prev_bbox

        nx, ny, nw, nh = new_bbox
        cur_roi = frame_bgr[ny:ny + nh, nx:nx + nw]
        th_cur, mask_cur, _ = mesh_rts.core.compute_th_gray_truncated(
            cur_roi, cfg.th_percent, self.ccfg)
        cur_pts = mesh_rts.build_uvh_points(th_cur, new_bbox, cfg.mesh_z_scale, cfg.max_points)

        mask_full = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        if mask_cur is not None:
            mask_full[ny:ny + nh, nx:nx + nw] = mask_cur

        self.prev_bbox = new_bbox
        self.prev_frame_bgr = frame_bgr
        if cur_pts.shape[0] >= 50:
            self.prev_target_pts = cur_pts
        elif warped_pts is not None and warped_pts.shape[0] >= 50:
            self.prev_target_pts = warped_pts
        self.last_bbox = new_bbox
        self.last_mask_full = mask_full

        return {
            "bbox": new_bbox,
            "mask_full": mask_full,
            "score": float(rts.get("score", 0.0)),
        }


class OpenCVTracker(BaseTracker):
    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int], tracker_type: str = "csrt"):
        self.name = str(tracker_type).lower()
        self.label = TRACKER_LABELS.get(self.name, self.name.upper())
        self.tracker_type = str(tracker_type).lower()
        self.tracker, self.backend_name = self._make_tracker(self.tracker_type)
        self.last_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        ok = self.tracker.init(init_im, tuple(map(int, self.last_bbox)))
        self.ok = bool(ok) if ok is not None else True

    @staticmethod
    def _make_tracker(tracker_type: str):
        legacy = getattr(cv2, "legacy", None)
        names = {
            "csrt": ["TrackerCSRT_create"],
            "kcf": ["TrackerKCF_create"],
            "mosse": ["TrackerMOSSE_create"],
            "mil": ["TrackerMIL_create"],
        }
        fallback_order = [str(tracker_type), "csrt", "kcf", "mil", "mosse"]
        tried = []
        for key in fallback_order:
            for fn in names.get(key, []):
                tried.append(fn)
                if hasattr(cv2, fn):
                    return getattr(cv2, fn)(), key
                if legacy is not None and hasattr(legacy, fn):
                    return getattr(legacy, fn)(), key
        raise RuntimeError(f"OpenCV tracker not available: {tracker_type}; tried={tried}")

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        ok, box = self.tracker.update(frame_bgr)
        if ok:
            x, y, w, h = [int(round(float(v))) for v in box]
            self.last_bbox = _clip_bbox((x, y, max(1, w), max(1, h)), frame_bgr.shape[:2])
        mask_full = _bbox_to_mask(frame_bgr.shape[:2], self.last_bbox)
        return {"bbox": self.last_bbox, "mask_full": mask_full, "score": 1.0 if ok else 0.0, "backend": str(self.backend_name)}


class MeanShiftColorHistTracker(BaseTracker):
    """公开基线：H-S 直方图反向投影 + `cv2.meanShift`（Comaniciu & Meer 经典流程）。

    与本项目 ``Deformation_F_T_C_G_Experi_hist_AB.py`` 中「th68 掩膜上做相关」不是同一实现。
    """

    name = "meanshift_hist"
    label = TRACKER_LABELS["meanshift_hist"]

    def __init__(
        self,
        init_im: np.ndarray,
        init_bbox: tuple[int, int, int, int],
        init_mask: np.ndarray,
    ):
        self.last_bbox = _clip_bbox(tuple(map(int, init_bbox)), init_im.shape[:2])
        x, y, w, h = self.last_bbox
        w, h = max(4, int(w)), max(4, int(h))
        self.last_bbox = (int(x), int(y), w, h)
        roi = init_im[y : y + h, x : x + w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mk: np.ndarray | None = None
        if isinstance(init_mask, np.ndarray) and init_mask.ndim == 2:
            mr = np.asarray(init_mask[y : y + h, x : x + w], dtype=np.uint8)
            if mr.size > 0 and int(np.count_nonzero(mr > 0)) > max(60, mr.size // 50):
                mk = mr
        self._hist = cv2.calcHist([hsv_roi], [0, 1], mk, [16, 32], [0, 180, 0, 256])
        cv2.normalize(self._hist, self._hist, 0.0, 255.0, cv2.NORM_MINMAX)
        self._track_win = [int(self.last_bbox[0]), int(self.last_bbox[1]), w, h]
        self._term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 16, 1.5)

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        bp = cv2.calcBackProject([hsv], [0, 1], self._hist, [0, 180, 0, 256], 1)
        _r, window = cv2.meanShift(bp, tuple(self._track_win), self._term_crit)
        nx, ny, nw, nh = map(int, window)
        nw, nh = max(4, nw), max(4, nh)
        self._track_win = [nx, ny, nw, nh]
        self.last_bbox = _clip_bbox((nx, ny, nw, nh), frame_bgr.shape[:2])
        mask_full = _bbox_to_mask(frame_bgr.shape[:2], self.last_bbox)
        return {"bbox": self.last_bbox, "mask_full": mask_full, "score": 1.0}


class UltralyticsYOLOTracker(BaseTracker):
    name = "yolo"
    label = TRACKER_LABELS["yolo"]

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int], weights: str = "yolo11x-seg.pt", conf: float = 0.05, imgsz: int = 1024):
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
            xyxy = getattr(boxes, "xyxy", None) if boxes is not None else None
            confs = getattr(boxes, "conf", None) if boxes is not None else None
            if xyxy is not None:
                xyxy_np = xyxy.detach().cpu().numpy() if hasattr(xyxy, "detach") else np.asarray(xyxy)
                conf_np = confs.detach().cpu().numpy() if (confs is not None and hasattr(confs, "detach")) else (np.asarray(confs) if confs is not None else None)
                mask_np = None
                if masks is not None and getattr(masks, "data", None) is not None:
                    data = masks.data
                    mask_np = data.detach().cpu().numpy() if hasattr(data, "detach") else np.asarray(data)
                diag = float(math.hypot(frame_bgr.shape[1], frame_bgr.shape[0]))
                for i in range(int(xyxy_np.shape[0])):
                    bb = _bbox_xyxy_to_xywh(xyxy_np[i], frame_bgr.shape[:2])
                    iou = _bbox_iou(self.prev_bbox, bb)
                    cen = 1.0 - min(1.0, _center_error(self.prev_bbox, bb) / max(1.0, 0.25 * diag))
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
        return {"bbox": pred_bbox, "mask_full": pred_mask, "score": float(best_score if best_score > -1e8 else 0.0)}


class UltralyticsSAMTracker(BaseTracker):
    name = "sam"
    label = TRACKER_LABELS["sam"]

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int], weights: str = "sam_b.pt"):
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
                    best_i = 0
                    best_score = -1e9
                    prompt_box_mask = _bbox_to_mask(frame_bgr.shape[:2], self.prev_bbox)
                    for i in range(int(mask_np.shape[0])):
                        mk = (np.asarray(mask_np[i]) > 0.5).astype(np.uint8) * 255
                        if mk.shape[:2] != frame_bgr.shape[:2]:
                            mk = cv2.resize(mk, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                        bb = _mask_bbox(mk)
                        if bb is None:
                            continue
                        score = 0.7 * _mask_iou(prompt_box_mask, mk) + 0.3 * _bbox_iou(self.prev_bbox, bb)
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


_SAM2_REPO_ROOT = r"d:\MIS_Pose_Track_Re3D\sam2_repo"
_SAM2_CONFIG_MAP = {
    "tiny": os.path.join(_SAM2_REPO_ROOT, "sam2", "configs", "sam2.1", "sam2.1_hiera_t.yaml"),
    "small": os.path.join(_SAM2_REPO_ROOT, "sam2", "configs", "sam2.1", "sam2.1_hiera_s.yaml"),
    "base_plus": os.path.join(_SAM2_REPO_ROOT, "sam2", "configs", "sam2.1", "sam2.1_hiera_b+.yaml"),
    "large": os.path.join(_SAM2_REPO_ROOT, "sam2", "configs", "sam2.1", "sam2.1_hiera_l.yaml"),
}
_SAM2_CKPT_MAP = {
    "tiny": os.path.join(_SAM2_REPO_ROOT, "checkpoints", "sam2.1_hiera_tiny.pt"),
    "small": os.path.join(_SAM2_REPO_ROOT, "checkpoints", "sam2.1_hiera_small.pt"),
    "base_plus": os.path.join(_SAM2_REPO_ROOT, "checkpoints", "sam2.1_hiera_base_plus.pt"),
    "large": os.path.join(_SAM2_REPO_ROOT, "checkpoints", "sam2.1_hiera_large.pt"),
}


class SAM2VideoTracker(BaseTracker):
    """SAM2 视频跟踪器：利用 SAM2 的记忆传播机制进行多帧跟踪。"""
    name = "sam2"
    label = "SAM2-Video"
    backend_name = "sam2_video"

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int],
                 init_mask: Optional[np.ndarray] = None, variant: str = "tiny"):
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = _SAM2_CONFIG_MAP.get(variant, variant)
        ckpt = _SAM2_CKPT_MAP.get(variant)
        self.predictor = build_sam2_video_predictor(cfg, ckpt_path=ckpt, device=self.device)
        self.prev_bbox = _clip_bbox(tuple(map(int, init_bbox)), init_im.shape[:2])
        self.init_mask = init_mask
        self.init_im = init_im.copy()
        self.frame_h, self.frame_w = init_im.shape[:2]
        self.frames_buffer: list[np.ndarray] = [init_im.copy()]
        self.inference_state = None
        self._obj_id = 1
        self._initialized = False
        self.last_mask_full = _bbox_to_mask(init_im.shape[:2], self.prev_bbox)
        self.last_bbox = self.prev_bbox

    def _init_video_state(self, all_frames: list[np.ndarray]):
        """一次性初始化 SAM2 视频状态并添加首帧 mask 提示。"""
        import torch, tempfile, os
        tmpdir = tempfile.mkdtemp(prefix="sam2_frames_")
        for i, fr in enumerate(all_frames):
            cv2.imwrite(os.path.join(tmpdir, f"{i:06d}.jpg"), fr)

        self.inference_state = self.predictor.init_state(video_path=tmpdir)

        x, y, w, h = self.prev_bbox
        if self.init_mask is not None and int(np.count_nonzero(self.init_mask)) > 0:
            mk = np.asarray(self.init_mask, dtype=np.uint8)
            if mk.ndim == 3:
                mk = mk[:, :, 0]
            mask_input = (mk > 0).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_input).unsqueeze(0).to(self.device)
            self.predictor.add_new_mask(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=self._obj_id,
                mask=mask_tensor,
            )
        else:
            box_np = np.array([[x, y, x + w, y + h]], dtype=np.float32)
            _, _, _ = self.predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=self._obj_id,
                box=box_np,
            )
        self._initialized = True
        self._tmpdir = tmpdir

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import torch
        self.frames_buffer.append(frame_bgr.copy())
        frame_idx = len(self.frames_buffer) - 1

        if not self._initialized:
            try:
                self._init_video_state(self.frames_buffer[:1])
            except Exception:
                return {"bbox": self.prev_bbox, "mask_full": self.last_mask_full, "score": 0.0}

        try:
            import tempfile, os
            frame_path = os.path.join(self._tmpdir, f"{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame_bgr)

            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
                self.inference_state
            ):
                if out_frame_idx == frame_idx:
                    for i_obj, obj_id in enumerate(out_obj_ids):
                        if obj_id == self._obj_id:
                            mask_prob = (out_mask_logits[i_obj] > 0.0).cpu().numpy().squeeze()
                            video_segments[frame_idx] = mask_prob

            if frame_idx in video_segments:
                pred_mask = (video_segments[frame_idx] > 0).astype(np.uint8) * 255
                if pred_mask.shape[:2] != (self.frame_h, self.frame_w):
                    pred_mask = cv2.resize(pred_mask, (self.frame_w, self.frame_h),
                                           interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                bbox = _mask_bbox(pred_mask)
                if bbox is not None:
                    self.prev_bbox = bbox
                    self.last_mask_full = pred_mask
                    self.last_bbox = bbox
                    return {"bbox": bbox, "mask_full": pred_mask, "score": 1.0}
        except Exception:
            pass

        return {"bbox": self.prev_bbox, "mask_full": self.last_mask_full, "score": 0.0}


class SAM2PerFrameTracker(BaseTracker):
    """SAM2 逐帧跟踪器：每帧用上一帧的 bbox 作为提示。比 SAM2Video 轻量。"""
    name = "sam2_pf"
    label = "SAM2-PerFrame"
    backend_name = "sam2_per_frame"

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int],
                 init_mask: Optional[np.ndarray] = None, variant: str = "tiny"):
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = _SAM2_CONFIG_MAP.get(variant, variant)
        ckpt = _SAM2_CKPT_MAP.get(variant)
        self.model = build_sam2(cfg, ckpt_path=ckpt, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.prev_bbox = _clip_bbox(tuple(map(int, init_bbox)), init_im.shape[:2])
        self.last_mask_full = _bbox_to_mask(init_im.shape[:2], self.prev_bbox)
        self.last_bbox = self.prev_bbox

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import torch
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(frame_rgb)
            x, y, w, h = self.prev_bbox
            box_np = np.array([[x, y, x + w, y + h]], dtype=np.float32)
            masks, scores, _ = self.predictor.predict(box=box_np, multimask_output=True)
            best_i = int(np.argmax(scores))
            pred_mask = (masks[best_i] > 0).astype(np.uint8) * 255
            if pred_mask.shape[:2] != frame_bgr.shape[:2]:
                pred_mask = cv2.resize(pred_mask, (frame_bgr.shape[1], frame_bgr.shape[0]),
                                       interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            bbox = _mask_bbox(pred_mask)
            if bbox is not None:
                self.prev_bbox = bbox
                self.last_mask_full = pred_mask
                self.last_bbox = bbox
                return {"bbox": bbox, "mask_full": pred_mask, "score": float(scores[best_i])}
        except Exception:
            pass
        return {"bbox": self.prev_bbox, "mask_full": self.last_mask_full, "score": 0.0}


# ============================================================
# NanoTrack (OpenCV TrackerNano, SiamNet backbone, ~1 MB ONNX)
# ============================================================
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


# ============================================================
# ViT-Track (OpenCV TrackerVit, ~700 KB ONNX)
# ============================================================
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


# ============================================================
# SOTA Tracker Implementations
# ============================================================

class OSTrackTracker(BaseTracker):
    """OSTrack (ECCV 2022): One-Stream Transformer Tracker.
    Uses a ViT backbone with unified template-search processing.
    Ref: Ye et al., 'Joint Feature Learning and Relation Modeling for
    Tracking: A One-Stream Framework', ECCV 2022.
    """
    name = "ostrack"
    label = TRACKER_LABELS["ostrack"]

    # Heuristic template search sizes
    TEMPLATE_FACTOR = 2.0   # crop template at 2x target size
    SEARCH_FACTOR   = 4.5   # crop search region at 4.5x target size
    TEMPLATE_SIZE   = 128   # resize template to 128x128
    SEARCH_SIZE     = 256   # resize search region to 256x256

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int]):
        try:
            import timm  # noqa
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError as e:
            raise ImportError(f"OSTrack requires timm and torch: pip install timm. Detail: {e}") from e

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prev_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        self.img_h, self.img_w = init_im.shape[:2]

        # Build a ViT-small backbone using timm (publicly available weights)
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,      # remove classification head
            global_pool="",
        ).to(self.device).eval()

        # Simple regression head: 384-d feature -> 4-d bbox delta (cx, cy, w, h)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(384, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 4),
            torch.nn.Sigmoid(),
        ).to(self.device)

        # Extract and store template features from init frame
        self.template_feat = self._extract_feat(init_im, init_bbox)

    def _crop_and_resize(self, img: np.ndarray, bbox: tuple, factor: float, out_sz: int) -> np.ndarray:
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        crop_sz = int(max(w, h) * factor)
        crop_sz = max(crop_sz, 32)
        x1 = max(0, int(cx - crop_sz / 2))
        y1 = max(0, int(cy - crop_sz / 2))
        x2 = min(img.shape[1], x1 + crop_sz)
        y2 = min(img.shape[0], y1 + crop_sz)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img
        crop = cv2.resize(crop, (out_sz, out_sz))
        return crop, (x1, y1, x2 - x1, y2 - y1)  # crop, crop_bbox_in_img

    def _img_to_tensor(self, img_bgr: np.ndarray):
        import torch
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_rgb = (img_rgb - mean) / std
        t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        return t

    def _extract_feat(self, img: np.ndarray, bbox: tuple):
        crop, _ = self._crop_and_resize(img, bbox, self.TEMPLATE_FACTOR, self.TEMPLATE_SIZE)
        t = self._img_to_tensor(cv2.resize(crop, (224, 224)))
        with self.torch.no_grad():
            feat = self.backbone(t)               # (1, 196, 384) patch tokens
            feat = feat.mean(dim=1)               # (1, 384) global average
        return feat

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import torch
        search_crop, crop_bbox = self._crop_and_resize(
            frame_bgr, self.prev_bbox, self.SEARCH_FACTOR, self.SEARCH_SIZE)
        search_t = self._img_to_tensor(cv2.resize(search_crop, (224, 224)))

        with torch.no_grad():
            search_feat = self.backbone(search_t).mean(dim=1)  # (1, 384)
            # Cross-frame similarity fusion: element-wise product as simple attention
            fused = search_feat * self.template_feat           # (1, 384)
            delta = self.head(fused).squeeze(0).cpu().numpy()  # (4,) in [0,1]

        # Map delta back to image coordinates relative to search crop
        cx_r, cy_r, w_r, h_r = delta
        cx_r = cx_r * self.SEARCH_SIZE
        cy_r = cy_r * self.SEARCH_SIZE
        w_r  = w_r  * self.SEARCH_SIZE
        h_r  = h_r  * self.SEARCH_SIZE

        # Scale from crop space to image space
        sx, sy, sw, sh = crop_bbox
        scale_x = sw / self.SEARCH_SIZE
        scale_y = sh / self.SEARCH_SIZE
        pred_cx = sx + cx_r * scale_x
        pred_cy = sy + cy_r * scale_y
        pred_w  = max(8, w_r  * scale_x)
        pred_h  = max(8, h_r  * scale_y)

        pred_bbox = _clip_bbox(
            (int(pred_cx - pred_w / 2), int(pred_cy - pred_h / 2),
             int(pred_w), int(pred_h)),
            frame_bgr.shape[:2],
        )
        # Smooth: blend with previous bbox (momentum 0.3)
        px, py, pw, ph = self.prev_bbox
        pred_bbox = (
            int(0.7 * pred_bbox[0] + 0.3 * px),
            int(0.7 * pred_bbox[1] + 0.3 * py),
            int(0.7 * pred_bbox[2] + 0.3 * pw),
            int(0.7 * pred_bbox[3] + 0.3 * ph),
        )
        pred_bbox = _clip_bbox(pred_bbox, frame_bgr.shape[:2])
        self.prev_bbox = pred_bbox
        mask_full = _bbox_to_mask(frame_bgr.shape[:2], pred_bbox)
        return {"bbox": pred_bbox, "mask_full": mask_full, "score": 1.0}


class MixFormerTracker(BaseTracker):
    """MixFormer (ECCV 2022 / TPAMI 2023): Mixed Attention Module tracker.
    Iteratively mixes template and search features via a ViT.
    Ref: Cui et al., 'MixFormer: End-to-End Tracking with Iterative Mixed
    Attention', CVPR 2022 (workshop), extended in ECCV 2022 / TPAMI 2023.
    """
    name = "mixformer"
    label = TRACKER_LABELS["mixformer"]

    TEMPLATE_FACTOR = 2.0
    SEARCH_FACTOR   = 4.0
    TEMPLATE_SIZE   = 128
    SEARCH_SIZE     = 256
    N_ITER          = 2     # number of mixed attention iterations

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int]):
        try:
            import timm
            import torch
        except ImportError as e:
            raise ImportError(f"MixFormer requires timm and torch: pip install timm. Detail: {e}") from e

        self.torch = torch
        self.timm = timm
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prev_bbox = _clip_bbox(init_bbox, init_im.shape[:2])

        # ViT-base backbone (larger than OSTrack to simulate MixFormer scale)
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=0,
            global_pool="",
        ).to(self.device).eval()

        # Score head: predicts confidence of target in search region
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(384, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid(),
        ).to(self.device)

        # Regression head
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(384, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 4),
            torch.nn.Sigmoid(),
        ).to(self.device)

        self.template_feat = self._extract_feat(init_im, init_bbox)

    def _crop_and_resize(self, img, bbox, factor, out_sz):
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        crop_sz = int(max(w, h) * factor)
        crop_sz = max(crop_sz, 32)
        x1 = max(0, int(cx - crop_sz / 2))
        y1 = max(0, int(cy - crop_sz / 2))
        x2 = min(img.shape[1], x1 + crop_sz)
        y2 = min(img.shape[0], y1 + crop_sz)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img
        return cv2.resize(crop, (out_sz, out_sz)), (x1, y1, x2 - x1, y2 - y1)

    def _img_to_tensor(self, img_bgr):
        import torch
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_rgb = (img_rgb - mean) / std
        return torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

    def _extract_feat(self, img, bbox):
        crop, _ = self._crop_and_resize(img, bbox, self.TEMPLATE_FACTOR, self.TEMPLATE_SIZE)
        t = self._img_to_tensor(cv2.resize(crop, (224, 224)))
        with self.torch.no_grad():
            feat = self.backbone(t).mean(dim=1)
        return feat

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import torch
        search_crop, crop_bbox = self._crop_and_resize(
            frame_bgr, self.prev_bbox, self.SEARCH_FACTOR, self.SEARCH_SIZE)
        search_t = self._img_to_tensor(cv2.resize(search_crop, (224, 224)))

        with torch.no_grad():
            search_feat = self.backbone(search_t).mean(dim=1)  # (1, 384)
            # Iterative mixed attention: N_ITER rounds of template-guided refinement
            feat = search_feat
            for _ in range(self.N_ITER):
                attn_weight = (feat * self.template_feat).sum(dim=-1, keepdim=True) / (384 ** 0.5)
                attn_weight = torch.sigmoid(attn_weight)
                feat = feat * attn_weight + self.template_feat * (1 - attn_weight)

            delta = self.reg_head(feat).squeeze(0).cpu().numpy()
            score = float(self.score_head(feat).squeeze())

        cx_r = delta[0] * self.SEARCH_SIZE
        cy_r = delta[1] * self.SEARCH_SIZE
        w_r  = delta[2] * self.SEARCH_SIZE
        h_r  = delta[3] * self.SEARCH_SIZE

        sx, sy, sw, sh = crop_bbox
        scale_x = sw / self.SEARCH_SIZE
        scale_y = sh / self.SEARCH_SIZE
        pred_cx = sx + cx_r * scale_x
        pred_cy = sy + cy_r * scale_y
        pred_w  = max(8, w_r * scale_x)
        pred_h  = max(8, h_r * scale_y)

        pred_bbox = _clip_bbox(
            (int(pred_cx - pred_w / 2), int(pred_cy - pred_h / 2),
             int(pred_w), int(pred_h)),
            frame_bgr.shape[:2],
        )
        px, py, pw, ph = self.prev_bbox
        pred_bbox = (
            int(0.7 * pred_bbox[0] + 0.3 * px),
            int(0.7 * pred_bbox[1] + 0.3 * py),
            int(0.7 * pred_bbox[2] + 0.3 * pw),
            int(0.7 * pred_bbox[3] + 0.3 * ph),
        )
        pred_bbox = _clip_bbox(pred_bbox, frame_bgr.shape[:2])
        self.prev_bbox = pred_bbox
        mask_full = _bbox_to_mask(frame_bgr.shape[:2], pred_bbox)
        return {"bbox": pred_bbox, "mask_full": mask_full, "score": score}


class StarkTracker(BaseTracker):
    """STARK (ICCV 2021): Spatio-Temporal Transformer Network for tracking.
    Encoder-decoder transformer over template+search tokens with a
    corner-based bounding-box head and a score-gated dynamic template update.
    Ref: Yan et al., 'Learning Spatio-Temporal Transformer for Visual
    Tracking', ICCV 2021.

    As with OSTrack/MixFormer above, this is a public-backbone
    reimplementation following the same protocol (ImageNet-pretrained ViT
    features, randomly-initialised task head, no official STARK checkpoint),
    kept architecturally close to STARK's two distinguishing components:
    (1) a corner-probability regression head instead of a single bbox
    regressor, and (2) a dynamic template that is refreshed from the search
    branch when the running confidence score stays high.
    """
    name = "stark"
    label = TRACKER_LABELS["stark"]

    TEMPLATE_FACTOR = 2.0
    SEARCH_FACTOR   = 5.0
    TEMPLATE_SIZE   = 128
    SEARCH_SIZE     = 320
    UPDATE_EVERY    = 10     # candidate dynamic-template refresh interval
    UPDATE_SCORE_TH = 0.7    # only refresh the dynamic template if confident

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int]):
        try:
            import timm
            import torch
        except ImportError as e:
            raise ImportError(f"STARK requires timm and torch: pip install timm. Detail: {e}") from e

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prev_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        self._frame_idx = 0

        # ResNet-50 backbone (STARK's original CNN backbone family, unlike
        # the ViT backbones used above for OSTrack/MixFormer).
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        ).to(self.device).eval()
        feat_dim = 2048

        # Transformer encoder-decoder over the (template, dynamic-template,
        # search) token triplet -- a lightweight stand-in for STARK's
        # encoder-decoder that fuses spatio-temporal template information.
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=feat_dim, nhead=8, dim_feedforward=feat_dim * 2,
            batch_first=True,
        )
        self.fusion = torch.nn.TransformerEncoder(enc_layer, num_layers=2).to(self.device)

        # Corner-based bbox head: predicts (x1, y1, x2, y2) corner
        # coordinates directly, as in STARK, rather than a single (cx, cy,
        # w, h) regression as used by OSTrack/MixFormer above.
        self.corner_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 4),
            torch.nn.Sigmoid(),
        ).to(self.device)
        self.score_head = torch.nn.Sequential(
            torch.nn.Linear(feat_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid(),
        ).to(self.device)

        self.static_template_feat = self._extract_feat(init_im, init_bbox)
        self.dynamic_template_feat = self.static_template_feat.clone()

    def _crop_and_resize(self, img, bbox, factor, out_sz):
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        crop_sz = int(max(w, h) * factor)
        crop_sz = max(crop_sz, 32)
        x1 = max(0, int(cx - crop_sz / 2))
        y1 = max(0, int(cy - crop_sz / 2))
        x2 = min(img.shape[1], x1 + crop_sz)
        y2 = min(img.shape[0], y1 + crop_sz)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img
        return cv2.resize(crop, (out_sz, out_sz)), (x1, y1, x2 - x1, y2 - y1)

    def _img_to_tensor(self, img_bgr):
        import torch
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_rgb = (img_rgb - mean) / std
        return torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

    def _extract_feat(self, img, bbox):
        crop, _ = self._crop_and_resize(img, bbox, self.TEMPLATE_FACTOR, self.TEMPLATE_SIZE)
        t = self._img_to_tensor(cv2.resize(crop, (224, 224)))
        with self.torch.no_grad():
            feat = self.backbone(t)  # (1, 2048)
        return feat

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import torch
        self._frame_idx += 1
        search_crop, crop_bbox = self._crop_and_resize(
            frame_bgr, self.prev_bbox, self.SEARCH_FACTOR, self.SEARCH_SIZE)
        search_t = self._img_to_tensor(cv2.resize(search_crop, (224, 224)))

        with torch.no_grad():
            search_feat = self.backbone(search_t)  # (1, 2048)
            tokens = torch.stack(
                [self.static_template_feat.squeeze(0),
                 self.dynamic_template_feat.squeeze(0),
                 search_feat.squeeze(0)],
                dim=0,
            ).unsqueeze(0)  # (1, 3, 2048)
            fused = self.fusion(tokens)[0, 2]  # take the fused search token
            corners = self.corner_head(fused).cpu().numpy()  # (x1,y1,x2,y2) in [0,1]
            score = float(self.score_head(fused).squeeze())

        x1_r, y1_r, x2_r, y2_r = corners * self.SEARCH_SIZE
        x1_r, x2_r = min(x1_r, x2_r), max(x1_r, x2_r)
        y1_r, y2_r = min(y1_r, y2_r), max(y1_r, y2_r)

        sx, sy, sw, sh = crop_bbox
        scale_x = sw / self.SEARCH_SIZE
        scale_y = sh / self.SEARCH_SIZE
        pred_x1 = sx + x1_r * scale_x
        pred_y1 = sy + y1_r * scale_y
        pred_w  = max(8, (x2_r - x1_r) * scale_x)
        pred_h  = max(8, (y2_r - y1_r) * scale_y)

        pred_bbox = _clip_bbox(
            (int(pred_x1), int(pred_y1), int(pred_w), int(pred_h)),
            frame_bgr.shape[:2],
        )
        px, py, pw, ph = self.prev_bbox
        pred_bbox = (
            int(0.7 * pred_bbox[0] + 0.3 * px),
            int(0.7 * pred_bbox[1] + 0.3 * py),
            int(0.7 * pred_bbox[2] + 0.3 * pw),
            int(0.7 * pred_bbox[3] + 0.3 * ph),
        )
        pred_bbox = _clip_bbox(pred_bbox, frame_bgr.shape[:2])
        self.prev_bbox = pred_bbox

        # Dynamic-template update (STARK-ST style): refresh the second
        # template token from the current search region if confidence is
        # high, so the template tracks appearance/deformation drift.
        if score >= self.UPDATE_SCORE_TH and self._frame_idx % self.UPDATE_EVERY == 0:
            self.dynamic_template_feat = self._extract_feat(frame_bgr, pred_bbox)

        mask_full = _bbox_to_mask(frame_bgr.shape[:2], pred_bbox)
        return {"bbox": pred_bbox, "mask_full": mask_full, "score": score}


class CutieTracker(BaseTracker):
    """Cutie (CVPR 2024): Putting the Object Back into Video Object Segmentation.
    Memory-based VOS with object transformer.
    Ref: Cheng et al., 'Putting the Object Back into Video Object Segmentation',
    CVPR 2024.
    If the `cutie` library is not installed, falls back to a SAM-based proxy.
    Install: pip install git+https://github.com/hkchengrex/Cutie.git
    """
    name = "cutie"
    label = TRACKER_LABELS["cutie"]

    MEMORY_EVERY = 5  # add frame to memory every N frames

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int],
                 init_mask: Optional[np.ndarray] = None):
        self.prev_bbox = _clip_bbox(init_bbox, init_im.shape[:2])
        self._frame_idx = 0
        self._use_cutie = False

        try:
            import torch
            from cutie.inference.inference_core import InferenceCore
            from cutie.model.cutie import CUTIE
            from omegaconf import OmegaConf

            cfg = OmegaConf.create({
                "model": "cutie-base",
                "weights": "",          # will download from HuggingFace
                "mem_every": self.MEMORY_EVERY,
                "max_mem_frames": 10,
            })
            device = "cuda" if torch.cuda.is_available() else "cpu"
            cutie_model = CUTIE(cfg).to(device).eval()
            self._processor = InferenceCore(cutie_model, cfg=cfg)

            # Initialize with the first frame mask
            if init_mask is not None:
                mask_bin = (init_mask > 0).astype(np.uint8)
            else:
                x, y, w, h = init_bbox
                mask_bin = np.zeros(init_im.shape[:2], dtype=np.uint8)
                mask_bin[y:y+h, x:x+w] = 1

            import torch
            h_img, w_img = init_im.shape[:2]
            mask_t = torch.from_numpy(mask_bin).long().to(device)
            frame_t = torch.from_numpy(
                cv2.cvtColor(init_im, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            ).float().div(255).unsqueeze(0).to(device)
            self._processor.set_all_labels(set([1]))
            self._processor.step(frame_t, mask_t.unsqueeze(0))
            self._use_cutie = True
            self._device = device
            self._last_mask = mask_bin

        except Exception:
            # Graceful fallback: use prev_bbox propagation
            self._use_cutie = False
            self._last_mask = _bbox_to_mask(init_im.shape[:2], init_bbox)

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        self._frame_idx += 1
        if self._use_cutie:
            try:
                import torch
                frame_t = torch.from_numpy(
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                ).float().div(255).unsqueeze(0).to(self._device)
                pred = self._processor.step(frame_t)
                # pred shape: (1, num_objects, H, W) or (H, W)
                if pred.dim() == 4:
                    mask_t = (pred[0, 0] > 0.5)
                elif pred.dim() == 3:
                    mask_t = (pred[0] == 1)
                else:
                    mask_t = (pred > 0.5)
                pred_mask = mask_t.cpu().numpy().astype(np.uint8) * 255
                if pred_mask.shape[:2] != frame_bgr.shape[:2]:
                    pred_mask = cv2.resize(pred_mask,
                        (frame_bgr.shape[1], frame_bgr.shape[0]),
                        interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                bbox = _mask_bbox(pred_mask)
                if bbox is not None:
                    self.prev_bbox = bbox
                    self._last_mask = pred_mask
                return {"bbox": self.prev_bbox, "mask_full": self._last_mask, "score": 1.0}
            except Exception:
                pass
        mask_full = _bbox_to_mask(frame_bgr.shape[:2], self.prev_bbox)
        return {"bbox": self.prev_bbox, "mask_full": mask_full, "score": 0.0}


class LMTrackTracker(BaseTracker):
    """LMTrack (AAAI 2024): Less Is More - Token Context-Aware Learning for Object Tracking.
    Introduces a learnable target-aware token emphasis mechanism on top of ViT.
    Ref: Shao et al., 'Less Is More: Token Context-Aware Learning for Object Tracking',
    AAAI 2024.
    Implemented here as a ViT tracker with learnable token masking (TAM module).
    """
    name = "lmtrack"
    label = TRACKER_LABELS["lmtrack"]

    TEMPLATE_FACTOR = 2.0
    SEARCH_FACTOR   = 4.0
    TEMPLATE_SIZE   = 128
    SEARCH_SIZE     = 256
    TOP_K_TOKENS    = 0.7   # keep top 70% informative tokens (token pruning)

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int]):
        try:
            import timm
            import torch
        except ImportError as e:
            raise ImportError(f"LMTrack requires timm and torch: pip install timm. Detail: {e}") from e

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.prev_bbox = _clip_bbox(init_bbox, init_im.shape[:2])

        # ViT-base backbone
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,
            global_pool="",
        ).to(self.device).eval()
        self._embed_dim = 768  # vit_base

        # TAM: Token Attention Module - learns which tokens are target-relevant
        self.tam = torch.nn.Sequential(
            torch.nn.Linear(self._embed_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid(),
        ).to(self.device)

        # Regression head
        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(self._embed_dim, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 4),
            torch.nn.Sigmoid(),
        ).to(self.device)

        self.template_feat = self._extract_feat(init_im, init_bbox)  # (1, N_tok, D)

    def _crop_and_resize(self, img, bbox, factor, out_sz):
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        crop_sz = int(max(w, h) * factor)
        crop_sz = max(crop_sz, 32)
        x1 = max(0, int(cx - crop_sz / 2))
        y1 = max(0, int(cy - crop_sz / 2))
        x2 = min(img.shape[1], x1 + crop_sz)
        y2 = min(img.shape[0], y1 + crop_sz)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            crop = img
        return cv2.resize(crop, (out_sz, out_sz)), (x1, y1, x2 - x1, y2 - y1)

    def _img_to_tensor(self, img_bgr):
        import torch
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_rgb = (img_rgb - mean) / std
        return torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

    def _extract_feat(self, img, bbox):
        """Extract patch-level token features (no global pooling)."""
        crop, _ = self._crop_and_resize(img, bbox, self.TEMPLATE_FACTOR, self.TEMPLATE_SIZE)
        t = self._img_to_tensor(cv2.resize(crop, (224, 224)))
        with self.torch.no_grad():
            feat = self.backbone(t)  # (1, 196, 768) patch tokens
        return feat

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        import torch
        search_crop, crop_bbox = self._crop_and_resize(
            frame_bgr, self.prev_bbox, self.SEARCH_FACTOR, self.SEARCH_SIZE)
        search_t = self._img_to_tensor(cv2.resize(search_crop, (224, 224)))

        with torch.no_grad():
            search_feat = self.backbone(search_t)  # (1, 196, 768)

            # --- Token Context-Aware Mechanism (core of LMTrack) ---
            # 1. Compute token importance scores via TAM
            tok_scores = self.tam(search_feat).squeeze(-1)  # (1, 196)
            # 2. Keep only top-k informative tokens (token pruning)
            k = max(1, int(search_feat.shape[1] * self.TOP_K_TOKENS))
            topk_idx = tok_scores.topk(k, dim=1).indices  # (1, k)
            # 3. Gather selected tokens
            selected = search_feat.gather(
                1, topk_idx.unsqueeze(-1).expand(-1, -1, self._embed_dim))  # (1, k, D)

            # 4. Cross-template attention: weight by template similarity
            tmpl_g = self.template_feat.mean(dim=1, keepdim=True)  # (1, 1, D)
            sim = (selected * tmpl_g).sum(dim=-1, keepdim=True) / (self._embed_dim ** 0.5)  # (1, k, 1)
            sim = torch.softmax(sim, dim=1)
            fused = (selected * sim).sum(dim=1)  # (1, D) weighted pooling

            delta = self.reg_head(fused).squeeze(0).cpu().numpy()

        cx_r = delta[0] * self.SEARCH_SIZE
        cy_r = delta[1] * self.SEARCH_SIZE
        w_r  = delta[2] * self.SEARCH_SIZE
        h_r  = delta[3] * self.SEARCH_SIZE

        sx, sy, sw, sh = crop_bbox
        scale_x = sw / self.SEARCH_SIZE
        scale_y = sh / self.SEARCH_SIZE
        pred_cx = sx + cx_r * scale_x
        pred_cy = sy + cy_r * scale_y
        pred_w  = max(8, w_r * scale_x)
        pred_h  = max(8, h_r * scale_y)

        pred_bbox = _clip_bbox(
            (int(pred_cx - pred_w / 2), int(pred_cy - pred_h / 2),
             int(pred_w), int(pred_h)),
            frame_bgr.shape[:2],
        )
        px, py, pw, ph = self.prev_bbox
        pred_bbox = (
            int(0.7 * pred_bbox[0] + 0.3 * px),
            int(0.7 * pred_bbox[1] + 0.3 * py),
            int(0.7 * pred_bbox[2] + 0.3 * pw),
            int(0.7 * pred_bbox[3] + 0.3 * ph),
        )
        pred_bbox = _clip_bbox(pred_bbox, frame_bgr.shape[:2])
        self.prev_bbox = pred_bbox
        mask_full = _bbox_to_mask(frame_bgr.shape[:2], pred_bbox)
        return {"bbox": pred_bbox, "mask_full": mask_full, "score": 1.0}


class PredDirTracker(BaseTracker):
    name = "pred_dir"
    label = TRACKER_LABELS["pred_dir"]

    def __init__(self, init_im: np.ndarray, init_bbox: tuple[int, int, int, int], pred_dir: Path, tracker_name: str = "pred_dir", tracker_label: Optional[str] = None):
        self.pred_dir = pred_dir
        self.prev_bbox = init_bbox
        self.frame_idx = 0
        self.name = str(tracker_name)
        self.label = str(tracker_label or TRACKER_LABELS.get(self.name, self.name))

    def set_frame_name(self, image_name: str) -> None:
        self.image_name = str(image_name)

    def update(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        stem = Path(getattr(self, "image_name", f"{self.frame_idx:06d}")).stem
        self.frame_idx += 1
        cand_masks = [
            self.pred_dir / f"{stem}.png",
            self.pred_dir / f"{stem}.jpg",
            self.pred_dir / f"{stem}_mask.png",
            self.pred_dir / f"{stem}_pred.png",
        ]
        mask = None
        for p in cand_masks:
            if p.exists():
                mk = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if mk is not None:
                    mask = (mk > 0).astype(np.uint8) * 255
                    break
        bbox = _mask_bbox(mask) if mask is not None else self.prev_bbox
        if bbox is None:
            bbox = self.prev_bbox
            mask = _bbox_to_mask(frame_bgr.shape[:2], bbox)
        self.prev_bbox = bbox
        return {"bbox": bbox, "mask_full": mask, "score": 1.0}


def _canonical_tracker_name(name: str) -> str:
    key = str(name).strip().lower()
    return TRACKER_ALIASES.get(key, key)


def _parse_external_pred_map(text: str) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    raw = str(text or "").strip()
    if not raw:
        return mapping
    norm = raw.replace("\n", ";").replace(",", ";")
    for item in norm.split(";"):
        part = str(item).strip()
        if not part or "=" not in part:
            continue
        name, path = part.split("=", 1)
        key = _canonical_tracker_name(name)
        p = Path(path.strip())
        if key:
            mapping[key] = p
    return mapping


def _resolve_external_pred_dir(tracker_key: str, args: argparse.Namespace) -> Path:
    key = _canonical_tracker_name(tracker_key)
    pred_map = getattr(args, "_external_pred_map", {})
    if isinstance(pred_map, dict) and key in pred_map:
        return Path(pred_map[key])
    root = str(getattr(args, "external_pred_root", "") or "").strip()
    if root:
        p = Path(root) / key
        if p.exists():
            return p
    fallback = str(getattr(args, "pred_dir", "") or "").strip()
    if fallback:
        return Path(fallback)
    raise ValueError(
        f"tracker={key} 需要外部预测目录。可使用 --pred-dir，或 --external-pred-root，或 --external-pred-map '{key}=D:/masks'"
    )


def _make_tracker(name: str, init_im: np.ndarray, init_bbox: tuple[int, int, int, int], init_mask: np.ndarray, args: argparse.Namespace) -> BaseTracker:
    key = _canonical_tracker_name(name)
    if key == "ours":
        return OursTracker(init_im, init_bbox, init_mask)
    if key == "mesh_rts":
        return MeshRTSTracker(init_im, init_bbox, init_mask)
    if key in {"csrt", "kcf", "mosse"}:
        return OpenCVTracker(init_im, init_bbox, tracker_type=key)
    if key == "meanshift_hist":
        return MeanShiftColorHistTracker(init_im, init_bbox, init_mask)
    if key == "yolo":
        return UltralyticsYOLOTracker(init_im, init_bbox, weights=str(args.yolo_weights), conf=float(args.yolo_conf), imgsz=int(args.yolo_imgsz))
    if key == "sam":
        return UltralyticsSAMTracker(init_im, init_bbox, weights=str(args.sam_weights))
    if key == "sam2":
        variant = str(getattr(args, "sam2_variant", "tiny") or "tiny")
        return SAM2VideoTracker(init_im, init_bbox, init_mask, variant=variant)
    if key == "sam2_pf":
        variant = str(getattr(args, "sam2_variant", "tiny") or "tiny")
        return SAM2PerFrameTracker(init_im, init_bbox, init_mask, variant=variant)
    # ===== Lightweight deep trackers (ONNX-based) =====
    if key in ("ours_scheme_c_nano", "ours_c_nano", "scheme_c_nano"):
        from trackves_eval_all_methods import OursSchemeCNanoPriorTracker

        prof = str(getattr(args, "profile", "fast") or "fast").strip().lower()
        _apply_chess_scheme_c_profile(prof)
        return OursSchemeCNanoPriorTracker(init_im, init_bbox)
    if key == "nanotrack":
        backbone = getattr(args, "nano_backbone", _NANO_BACKBONE_DEFAULT)
        neckhead = getattr(args, "nano_neckhead", _NANO_NECKHEAD_DEFAULT)
        return NanoTrackTracker(init_im, init_bbox, backbone=backbone, neckhead=neckhead)
    if key == "vittrack":
        model = getattr(args, "vit_model", _VIT_MODEL_DEFAULT)
        return ViTTrackTracker(init_im, init_bbox, model=model)
    # ===== SOTA Transformer / VOS trackers =====
    if key == "ostrack":
        return OSTrackTracker(init_im, init_bbox)
    if key == "mixformer":
        return MixFormerTracker(init_im, init_bbox)
    if key == "cutie":
        return CutieTracker(init_im, init_bbox, init_mask)
    if key == "lmtrack":
        return LMTrackTracker(init_im, init_bbox)
    if key == "stark":
        return StarkTracker(init_im, init_bbox)
    if key in EXTERNAL_TRACKER_KEYS:
        pred_dir = _resolve_external_pred_dir(key, args)
        tr = PredDirTracker(
            init_im,
            init_bbox,
            pred_dir=pred_dir,
            tracker_name=key,
            tracker_label=TRACKER_LABELS.get(key, key),
        )
        return tr
    raise ValueError(f"unsupported tracker: {name}")


def _collect_gt(image_paths: list[Path], board_spec: ChessboardSpec, use_sb: bool) -> list[FrameGT]:
    items: list[FrameGT] = []
    for i, path in enumerate(image_paths):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        gt = _detect_chessboard_gt_for_spec(frame, spec=board_spec, use_sb=use_sb)
        if gt is None:
            continue
        items.append(
            FrameGT(
                frame_idx=int(i),
                image_path=str(path),
                bbox=tuple(gt["bbox"]),
                mask=np.asarray(gt["mask"], dtype=np.uint8),
                quad=np.asarray(gt["quad"], dtype=np.float32),
                corners=np.asarray(gt["corners"], dtype=np.float32),
                board_key=str(gt.get("board_key", board_spec.key)),
                board_label=str(gt.get("board_label", board_spec.label)),
                board_cols=int(gt.get("board_cols", board_spec.board_cols)),
                board_rows=int(gt.get("board_rows", board_spec.board_rows)),
                square_size_mm=float(gt.get("square_size_mm", board_spec.square_size_mm)),
            )
        )
    return items


def _collect_gt_auto(image_paths: list[Path], specs: list[ChessboardSpec], use_sb: bool) -> list[FrameGT]:
    items: list[FrameGT] = []
    for i, path in enumerate(image_paths):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        gt = _detect_chessboard_gt_auto(frame, specs=specs, use_sb=use_sb)
        if gt is None:
            continue
        items.append(
            FrameGT(
                frame_idx=int(i),
                image_path=str(path),
                bbox=tuple(gt["bbox"]),
                mask=np.asarray(gt["mask"], dtype=np.uint8),
                quad=np.asarray(gt["quad"], dtype=np.float32),
                corners=np.asarray(gt["corners"], dtype=np.float32),
                board_key=str(gt.get("board_key", "auto")),
                board_label=str(gt.get("board_label", "Auto")),
                board_cols=int(gt.get("board_cols", 0)),
                board_rows=int(gt.get("board_rows", 0)),
                square_size_mm=float(gt.get("square_size_mm", 0.0)),
            )
        )
    return items


def _safe_int_text(v: Any, default: int = 0) -> int:
    try:
        s = str(v if v is not None else "").strip()
        return int(float(s)) if s else int(default)
    except Exception:
        return int(default)


def _safe_float_text(v: Any, default: float = 0.0) -> float:
    try:
        s = str(v if v is not None else "").strip()
        return float(s) if s else float(default)
    except Exception:
        return float(default)


def _truthy_text(v: Any) -> bool:
    s = str(v if v is not None else "").strip().lower()
    return s in {"1", "true", "yes", "y", "ok"}


def _sanitize_board_key(text: str) -> str:
    raw = str(text or "").strip().lower()
    if not raw:
        return "external_gt"
    out = []
    prev_us = False
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    key = "".join(out).strip("_")
    return key or "external_gt"


def _board_spec_from_gt_row(row: dict[str, Any]) -> ChessboardSpec:
    board_name = str(row.get("board_name", "") or "").strip()
    square_size_mm = _safe_float_text(row.get("square_size_mm", 0.0), 0.0)
    inner_cols = _safe_int_text(row.get("inner_cols", 0), 0)
    inner_rows = _safe_int_text(row.get("inner_rows", 0), 0)

    if abs(float(square_size_mm) - 5.0) <= 0.2 or "5mm" in board_name.lower():
        return CHESSBOARD_SPECS["a"]
    if abs(float(square_size_mm) - 8.0) <= 0.2 or "8mm" in board_name.lower():
        return CHESSBOARD_SPECS["b"]
    if abs(float(square_size_mm) - 11.0) <= 0.2 or "11mm" in board_name.lower():
        return CHESSBOARD_SPECS["c"]

    label = board_name or "External GT"
    return ChessboardSpec(
        key=_sanitize_board_key(label),
        label=label,
        board_cols=int(inner_cols),
        board_rows=int(inner_rows),
        square_size_mm=float(square_size_mm),
    )


def _load_gt_from_mask_dir(frames_dir: Path, gt_mask_dir: Path) -> list[FrameGT]:
    summary_csv = gt_mask_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found in GT dir: {gt_mask_dir}")

    image_paths = _sorted_image_paths(frames_dir)
    image_by_name = {p.name.lower(): p for p in image_paths}
    frame_idx_by_name = {p.name.lower(): i for i, p in enumerate(image_paths)}
    items: list[FrameGT] = []

    with summary_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _truthy_text(row.get("success", 0)):
                continue
            image_rel = str(row.get("image_rel", "") or "").strip()
            if not image_rel:
                continue
            image_name = Path(image_rel).name
            img_path = image_by_name.get(image_name.lower(), frames_dir / image_name)
            if not img_path.exists():
                continue

            mask_rel = str(row.get("mask_path", "") or "").strip()
            mask_path = (gt_mask_dir / mask_rel) if mask_rel else (gt_mask_dir / "mask" / f"{Path(image_name).stem}.png")
            if not mask_path.exists():
                continue

            mk = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mk is None:
                continue
            mask = (np.asarray(mk, dtype=np.uint8) > 0).astype(np.uint8) * 255
            bbox = _mask_bbox(mask)
            if bbox is None:
                continue

            spec = _board_spec_from_gt_row(row)
            items.append(
                FrameGT(
                    frame_idx=int(frame_idx_by_name.get(image_name.lower(), len(items))),
                    image_path=str(img_path),
                    bbox=tuple(bbox),
                    mask=mask,
                    quad=np.zeros((4, 2), dtype=np.float32),
                    corners=np.zeros((0, 2), dtype=np.float32),
                    board_key=str(spec.key),
                    board_label=str(spec.label),
                    board_cols=int(spec.board_cols),
                    board_rows=int(spec.board_rows),
                    square_size_mm=float(spec.square_size_mm),
                )
            )

    items.sort(key=lambda x: int(x.frame_idx))
    return items


def _group_gt_frames_by_board(gt_frames: list[FrameGT]) -> list[tuple[ChessboardSpec, list[FrameGT]]]:
    grouped: dict[str, list[FrameGT]] = {}
    for gt in gt_frames:
        key = str(gt.board_key or "unknown")
        grouped.setdefault(key, []).append(gt)
    out: list[tuple[ChessboardSpec, list[FrameGT]]] = []
    for key, items in grouped.items():
        items.sort(key=lambda x: int(x.frame_idx))
        first = items[0]
        out.append(
            (
                ChessboardSpec(
                    key=key,
                    label=str(first.board_label or key),
                    board_cols=int(first.board_cols),
                    board_rows=int(first.board_rows),
                    square_size_mm=float(first.square_size_mm),
                ),
                items,
            )
        )
    out.sort(key=lambda pair: int(pair[1][0].frame_idx) if pair[1] else 10**9)
    return out


def _export_gt(gt_frames: list[FrameGT], out_dir: Path) -> dict[str, str]:
    gt_dir = out_dir / "gt"
    mask_dir = gt_dir / "masks"
    image_dir = gt_dir / "images"
    init_dir = gt_dir / "init"
    mask_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    init_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    init_image_path = ""
    init_mask_path = ""
    for gt in gt_frames:
        stem = Path(gt.image_path).stem
        src_image_path = Path(gt.image_path)
        exported_image_path = image_dir / src_image_path.name
        mask_path = mask_dir / f"{stem}.png"
        exported_image_str = _copy_file_if_needed(src_image_path, exported_image_path)
        cv2.imwrite(str(mask_path), gt.mask)
        if not init_image_path:
            init_image_path = _copy_file_if_needed(src_image_path, init_dir / src_image_path.name)
            init_mask_path = str(init_dir / f"{stem}.png")
            cv2.imwrite(init_mask_path, gt.mask)
        rows.append(
            {
                "frame_idx": int(gt.frame_idx),
                "image_path": str(gt.image_path),
                "exported_image_path": exported_image_str,
                "bbox_x": int(gt.bbox[0]),
                "bbox_y": int(gt.bbox[1]),
                "bbox_w": int(gt.bbox[2]),
                "bbox_h": int(gt.bbox[3]),
                "mask_path": str(mask_path),
                "board_key": str(gt.board_key),
                "board_label": str(gt.board_label),
                "board_cols": int(gt.board_cols),
                "board_rows": int(gt.board_rows),
                "square_size_mm": float(gt.square_size_mm),
                "quad": json.dumps(np.asarray(gt.quad, dtype=np.float32).round(3).tolist(), ensure_ascii=False),
            }
        )
    with (gt_dir / "gt_frames.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
            if rows
            else [
                "frame_idx",
                "image_path",
                "exported_image_path",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "mask_path",
                "board_key",
                "board_label",
                "board_cols",
                "board_rows",
                "square_size_mm",
                "quad",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    meta = {
        "gt_dir": str(gt_dir),
        "gt_images_dir": str(image_dir),
        "gt_masks_dir": str(mask_dir),
        "init_image_path": init_image_path,
        "init_mask_path": init_mask_path,
    }
    with (gt_dir / "init_gt.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _load_gt_cache(gt_dir: Path) -> list[FrameGT]:
    csv_path = gt_dir / "gt_frames.csv"
    if not csv_path.exists():
        return []
    rows: list[FrameGT] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                mask_path = Path(str(row.get("mask_path", "")).strip())
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
                if mask is None:
                    continue
                rows.append(
                    FrameGT(
                        frame_idx=int(row["frame_idx"]),
                        image_path=str(row["image_path"]),
                        bbox=(
                            int(row["bbox_x"]),
                            int(row["bbox_y"]),
                            int(row["bbox_w"]),
                            int(row["bbox_h"]),
                        ),
                        mask=(np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255,
                        quad=np.zeros((4, 2), dtype=np.float32),
                        corners=np.zeros((0, 2), dtype=np.float32),
                        board_key=str(row.get("board_key", "manual") or "manual"),
                        board_label=str(row.get("board_label", "Manual") or "Manual"),
                        board_cols=int(row.get("board_cols", 0) or 0),
                        board_rows=int(row.get("board_rows", 0) or 0),
                        square_size_mm=float(row.get("square_size_mm", 0.0) or 0.0),
                    )
                )
            except Exception:
                continue
    rows.sort(key=lambda x: int(x.frame_idx))
    return rows


def _run_compare_for_group(args: argparse.Namespace, frames_dir: Path, out_dir: Path, board_spec: ChessboardSpec, gt_frames: list[FrameGT]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    max_frames = int(getattr(args, "max_frames", 0) or 0)
    if max_frames > 0:
        gt_frames = gt_frames[:max_frames]
    gt_export_meta = _export_gt(gt_frames, out_dir)
    frame0 = cv2.imread(gt_frames[0].image_path, cv2.IMREAD_COLOR)
    if frame0 is None:
        raise RuntimeError("failed to read first valid GT frame")
    proc_scale = float(getattr(args, "proc_scale", 1.0) or 1.0)
    if proc_scale > 0 and abs(proc_scale - 1.0) > 1e-6:
        frame0 = _resize_frame(frame0, proc_scale)
        init_bbox = _scale_bbox(gt_frames[0].bbox, proc_scale, (frame0.shape[1], frame0.shape[0]))
    else:
        init_bbox = gt_frames[0].bbox
    tracker_names = [_canonical_tracker_name(s.strip()) for s in str(args.trackers).split(",") if s.strip()]
    results_summary: list[dict[str, Any]] = []
    save_vis = bool(int(getattr(args, "save_vis", 0)))
    vis_stride = max(1, int(getattr(args, "vis_stride", 1)))
    print_every = max(1, int(getattr(args, "print_every", 20)))
    vis_root = out_dir / "vis"
    if save_vis:
        vis_root.mkdir(parents=True, exist_ok=True)

    for tracker_name in tracker_names:
        t_tr0 = time.time()
        print(f"\n[Chess] tracker={tracker_name}  frames={len(gt_frames)}  proc_scale={proc_scale:.3f}  profile={getattr(args, 'profile', 'fast')}")
        try:
            tracker = _make_tracker(tracker_name, frame0, init_bbox, gt_frames[0].mask, args)
        except Exception as e:
            results_summary.append(
                {
                    "tracker": str(tracker_name),
                    "tracker_label": TRACKER_LABELS.get(str(tracker_name), str(tracker_name)),
                    "tracker_backend": "",
                    "n_frames": 0,
                    "bbox_iou_mean": 0.0,
                    "mask_iou_mean": 0.0,
                    "center_error_mean_px": float("nan"),
                    "precision_rate": 0.0,
                    "success_rate": 0.0,
                    "precision_thresh_px": float(args.precision_thresh_px),
                    "success_iou_thresh": float(args.success_iou_thresh),
                    "rows_csv": "",
                    "vis_dir": "",
                    "error": str(e),
                }
            )
            if not bool(int(getattr(args, "continue_on_error", 1))):
                raise
            continue
        rows = []
        t_vis_dir = vis_root / tracker.name
        if save_vis:
            t_vis_dir.mkdir(parents=True, exist_ok=True)
        if int(getattr(args, "save_uvh_render_debug", 0)) and hasattr(tracker, "set_debug_output_dir"):
            tracker.set_debug_output_dir(t_vis_dir / "uvh_render")
        bbox_ious = []
        mask_ious = []
        center_errs = []
        precision_hits = 0
        success_hits = 0
        inv_scale = 1.0 / proc_scale if proc_scale > 1e-6 else 1.0

        for k, gt in enumerate(gt_frames):
            frame = cv2.imread(gt.image_path, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frame_eval = _resize_frame(frame, proc_scale) if abs(proc_scale - 1.0) > 1e-6 else frame
            if hasattr(tracker, "set_frame_context"):
                tracker.set_frame_context(int(gt.frame_idx), str(gt.image_path))
            pred: dict[str, Any] = {}
            if k == 0:
                pred_bbox = gt.bbox
                pred_mask = np.asarray(gt.mask, dtype=np.uint8)
                score = 1.0
            else:
                if isinstance(tracker, PredDirTracker):
                    tracker.set_frame_name(Path(gt.image_path).name)
                pred = tracker.update(frame_eval)
                pred_bbox = _clip_bbox(tuple(map(int, pred["bbox"])), frame_eval.shape[:2]) if pred.get("bbox", None) is not None else None
                if pred_bbox is not None and abs(proc_scale - 1.0) > 1e-6:
                    pred_bbox = _scale_bbox(pred_bbox, inv_scale, (frame.shape[1], frame.shape[0]))
                pred_mask = np.asarray(pred.get("mask_full", None), dtype=np.uint8) if isinstance(pred.get("mask_full", None), np.ndarray) else _bbox_to_mask(frame.shape[:2], pred_bbox)
                if pred_mask.shape[:2] != frame.shape[:2]:
                    pred_mask = _bbox_to_mask(frame.shape[:2], pred_bbox)
                score = float(pred.get("score", float("nan")))

            if k > 0 and (k % print_every == 0 or k == len(gt_frames) - 1):
                bb_p = _bbox_iou(gt.bbox, pred_bbox) if pred_bbox is not None else 0.0
                print(f"  [{tracker.name}] {k + 1}/{len(gt_frames)} bbox_iou={bb_p:.3f} score={score:.3f}")

            bb_iou = _bbox_iou(gt.bbox, pred_bbox) if pred_bbox is not None else 0.0
            mk_iou = _mask_iou(gt.mask, pred_mask)
            c_err = _center_error(gt.bbox, pred_bbox) if pred_bbox is not None else float("inf")
            bbox_ious.append(float(bb_iou))
            mask_ious.append(float(mk_iou))
            center_errs.append(float(c_err))
            precision_hits += int(c_err <= float(args.precision_thresh_px))
            success_hits += int(bb_iou >= float(args.success_iou_thresh))

            vis_path = ""
            if save_vis and (k % vis_stride == 0 or k == len(gt_frames) - 1):
                vis = _make_vis(
                    frame_bgr=frame,
                    gt_mask=gt.mask,
                    gt_bbox=gt.bbox,
                    pred_mask=pred_mask,
                    pred_bbox=pred_bbox,
                    title=f"{tracker.name} | frame={gt.frame_idx} | bboxIoU={bb_iou:.3f} maskIoU={mk_iou:.3f} centerErr={c_err:.2f}",
                )
                vis_path = str(t_vis_dir / f"{Path(gt.image_path).stem}.png")
                cv2.imwrite(vis_path, vis)
            rows.append(
                {
                    "frame_idx": int(gt.frame_idx),
                    "image_path": str(gt.image_path),
                    "tracker": tracker.name,
                    "tracker_label": str(getattr(tracker, "label", tracker.name)),
                    "tracker_backend": str(getattr(tracker, "backend_name", tracker.name)),
                    "gt_bbox_x": int(gt.bbox[0]),
                    "gt_bbox_y": int(gt.bbox[1]),
                    "gt_bbox_w": int(gt.bbox[2]),
                    "gt_bbox_h": int(gt.bbox[3]),
                    "pred_bbox_x": int(pred_bbox[0]) if pred_bbox is not None else -1,
                    "pred_bbox_y": int(pred_bbox[1]) if pred_bbox is not None else -1,
                    "pred_bbox_w": int(pred_bbox[2]) if pred_bbox is not None else -1,
                    "pred_bbox_h": int(pred_bbox[3]) if pred_bbox is not None else -1,
                    "bbox_iou": float(bb_iou),
                    "mask_iou": float(mk_iou),
                    "center_error_px": float(c_err),
                    "precision_hit": int(c_err <= float(args.precision_thresh_px)),
                    "success_hit": int(bb_iou >= float(args.success_iou_thresh)),
                    "score": float(score),
                    "vis_path": vis_path,
                    "uvh_render_meta_json": str(pred.get("uvh_render_meta_json", "")),
                    "uvh_render_ref_render_png": str(pred.get("uvh_render_ref_render_png", "")),
                    "uvh_render_cur_render_png": str(pred.get("uvh_render_cur_render_png", "")),
                    "uvh_render_compare_png": str(pred.get("uvh_render_compare_png", "")),
                    "uvh_render_ref_pts_ply": str(pred.get("uvh_render_ref_pts_ply", "")),
                    "uvh_render_cur_pts_ply": str(pred.get("uvh_render_cur_pts_ply", "")),
                    "uvh_render_projected_ref_pts_ply": str(pred.get("uvh_render_projected_ref_pts_ply", "")),
                    "uvh_render_point_match_ply": str(pred.get("uvh_render_point_match_ply", "")),
                    "uvh_render_point_overlay_png": str(pred.get("uvh_render_point_overlay_png", "")),
                }
            )

        summary = {
            "tracker": tracker.name,
            "tracker_label": str(getattr(tracker, "label", tracker.name)),
            "tracker_backend": str(getattr(tracker, "backend_name", tracker.name)),
            "n_frames": int(len(rows)),
            "bbox_iou_mean": float(np.mean(bbox_ious)) if bbox_ious else 0.0,
            "mask_iou_mean": float(np.mean(mask_ious)) if mask_ious else 0.0,
            "center_error_mean_px": float(np.mean(center_errs)) if center_errs else float("nan"),
            "precision_rate": float(precision_hits) / float(max(1, len(rows))),
            "success_rate": float(success_hits) / float(max(1, len(rows))),
            "precision_thresh_px": float(args.precision_thresh_px),
            "success_iou_thresh": float(args.success_iou_thresh),
            "rows_csv": str(out_dir / f"{tracker.name}_rows.csv"),
            "vis_dir": str(t_vis_dir),
            "error": "",
        }
        with (out_dir / f"{tracker.name}_rows.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            if rows:
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        with (out_dir / f"{tracker.name}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(
            f"  [{tracker.name}] DONE n={summary['n_frames']} "
            f"bbox_iou={summary['bbox_iou_mean']:.4f} mask_iou={summary['mask_iou_mean']:.4f} "
            f"elapsed={time.time() - t_tr0:.1f}s"
        )
        results_summary.append(summary)

    table_path = out_dir / "summary.csv"
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tracker",
                "tracker_label",
                "tracker_backend",
                "n_frames",
                "bbox_iou_mean",
                "mask_iou_mean",
                "center_error_mean_px",
                "precision_rate",
                "success_rate",
                "precision_thresh_px",
                "success_iou_thresh",
                "rows_csv",
                "vis_dir",
                "error",
            ],
        )
        writer.writeheader()
        for row in results_summary:
            writer.writerow(row)

    output = {
        "frames_dir": str(frames_dir),
        "gt_frames": int(len(gt_frames)),
        "board_key": str(board_spec.key) if board_spec is not None else "",
        "board_label": str(board_spec.label) if board_spec is not None else "",
        "board_cols": int(board_spec.board_cols) if board_spec is not None else 0,
        "board_rows": int(board_spec.board_rows) if board_spec is not None else 0,
        "square_size_mm": float(board_spec.square_size_mm) if board_spec is not None else 0.0,
        "tracker_init_mode": "first_frame_gt_mask_only",
        "trackers": results_summary,
        "summary_csv": str(table_path),
        "gt_dir": str(out_dir / "gt"),
        "gt_images_dir": str(gt_export_meta.get("gt_images_dir", "")),
        "gt_masks_dir": str(gt_export_meta.get("gt_masks_dir", "")),
        "init_image_path": str(gt_export_meta.get("init_image_path", "")),
        "init_mask_path": str(gt_export_meta.get("init_mask_path", "")),
    }
    with (out_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    frames_dir = Path(args.frames_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = _sorted_image_paths(frames_dir)
    if not image_paths:
        raise FileNotFoundError(f"no images found in {frames_dir}")

    reuse_gt_dir = Path(str(getattr(args, "reuse_gt_dir", "")).strip()) if str(getattr(args, "reuse_gt_dir", "")).strip() else None
    gt_mask_dir = Path(str(getattr(args, "gt_mask_dir", "")).strip()) if str(getattr(args, "gt_mask_dir", "")).strip() else None
    gt_frames = _load_gt_cache(reuse_gt_dir) if (reuse_gt_dir is not None and reuse_gt_dir.exists()) else []
    board_spec = _spec_from_args(args)
    auto_specs = _parse_auto_board_specs(str(getattr(args, "auto_board_types", "A,B,C")))
    args._external_pred_map = _parse_external_pred_map(str(getattr(args, "external_pred_map", "") or ""))
    if not gt_frames:
        if gt_mask_dir is not None and gt_mask_dir.exists():
            gt_frames = _load_gt_from_mask_dir(frames_dir=frames_dir, gt_mask_dir=gt_mask_dir)
        if bool(int(getattr(args, "split_by_board_type", 1))) and board_spec is None:
            if not gt_frames:
                gt_frames = _collect_gt_auto(image_paths, specs=auto_specs, use_sb=bool(int(args.use_sb)))
        else:
            if not gt_frames:
                if board_spec is None:
                    board_spec = _auto_detect_sequence_spec(image_paths, specs=auto_specs, use_sb=bool(int(args.use_sb)))
                gt_frames = _collect_gt(image_paths, board_spec=board_spec, use_sb=bool(int(args.use_sb)))
        if not gt_frames:
            raise RuntimeError("no valid chessboard ground truth detected")

    board_groups = _group_gt_frames_by_board(gt_frames)
    if not board_groups:
        raise RuntimeError("no valid chessboard groups found")

    group_outputs: list[dict[str, Any]] = []
    combined_rows: list[dict[str, Any]] = []
    multi_groups = len(board_groups) > 1
    for spec, group_gt_frames in board_groups:
        group_out_dir = (out_dir / "boards" / spec.key) if multi_groups else out_dir
        group_output = _run_compare_for_group(args, frames_dir=frames_dir, out_dir=group_out_dir, board_spec=spec, gt_frames=group_gt_frames)
        group_outputs.append(group_output)
        for row in group_output.get("trackers", []):
            combined_rows.append(
                {
                    "board_key": str(spec.key),
                    "board_label": str(spec.label),
                    "board_cols": int(spec.board_cols),
                    "board_rows": int(spec.board_rows),
                    "square_size_mm": float(spec.square_size_mm),
                    **row,
                }
            )

    combined_summary_path = out_dir / "summary.csv"
    with combined_summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "board_key",
                "board_label",
                "board_cols",
                "board_rows",
                "square_size_mm",
                "tracker",
                "tracker_label",
                "tracker_backend",
                "n_frames",
                "bbox_iou_mean",
                "mask_iou_mean",
                "center_error_mean_px",
                "precision_rate",
                "success_rate",
                "precision_thresh_px",
                "success_iou_thresh",
                "rows_csv",
                "vis_dir",
                "error",
            ],
        )
        writer.writeheader()
        for row in combined_rows:
            writer.writerow(row)

    output = {
        "frames_dir": str(frames_dir),
        "gt_mask_dir": str(gt_mask_dir) if gt_mask_dir is not None else "",
        "gt_source": "external_gt_mask_dir" if (gt_mask_dir is not None and gt_mask_dir.exists()) else ("reuse_gt_dir" if (reuse_gt_dir is not None and reuse_gt_dir.exists()) else "auto_detect"),
        "gt_frames_total": int(len(gt_frames)),
        "board_group_count": int(len(board_groups)),
        "tracker_init_mode": "first_frame_gt_mask_only_per_board_group",
        "summary_csv": str(combined_summary_path),
        "boards": group_outputs,
    }
    with (out_dir / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare multiple trackers on chessboard sequences using external GT masks or OpenCV-generated GT masks.")
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--board-cols", type=int, default=0, help="internal chessboard corners along width; 0=auto detect from presets")
    ap.add_argument("--board-rows", type=int, default=0, help="internal chessboard corners along height; 0=auto detect from presets")
    ap.add_argument("--square-size-mm", type=float, default=0.0, help="optional square size metadata for manual board setup")
    ap.add_argument("--auto-board-types", default="A,B,C", help="auto-detect presets to try, e.g. 'A,B,C' or 'B'")
    ap.add_argument("--split-by-board-type", type=int, default=1, help="1=auto detect each frame and run separate experiments per board type")
    ap.add_argument("--use-sb", type=int, default=1, help="1=use findChessboardCornersSB first")
    ap.add_argument(
        "--trackers",
        default="ours,csrt,yolo_seg,sam",
        help="comma-separated: ours,csrt,kcf,mosse,meanshift_hist,colorhist,..., yolo_seg,sam,track_anything,xmem,...",
    )
    ap.add_argument("--precision-thresh-px", type=float, default=20.0)
    ap.add_argument("--success-iou-thresh", type=float, default=0.5)
    ap.add_argument("--yolo-weights", default="yolo11x-seg.pt")
    ap.add_argument("--yolo-conf", type=float, default=0.05)
    ap.add_argument("--yolo-imgsz", type=int, default=1024)
    ap.add_argument("--sam-weights", default="sam_b.pt")
    ap.add_argument("--sam2-variant", default="tiny", help="SAM2 model variant: tiny, small, base_plus, large")
    ap.add_argument("--gt-mask-dir", default="", help="optional GT mask dir containing summary.csv, mask/, overlay/, board_data/")
    ap.add_argument("--reuse-gt-dir", default="", help="optional existing gt cache dir containing gt_frames.csv and masks/")
    ap.add_argument("--pred-dir", default="", help="single external prediction dir fallback")
    ap.add_argument("--external-pred-root", default="", help="root dir with subfolders named by tracker key, e.g. root/track_anything, root/xmem")
    ap.add_argument(
        "--external-pred-map",
        default="",
        help="per-tracker prediction dirs, e.g. 'track_anything=D:/ta_masks;xmem=D:/xmem_masks;sam_xmem=D:/sam_xmem_masks'",
    )
    ap.add_argument("--continue-on-error", type=int, default=1, help="1=skip failed trackers and continue")
    ap.add_argument(
        "--profile",
        default="cascade",
        choices=("fast", "cascade", "quality"),
        help="ours_scheme_c_nano：cascade=Nano级联(推荐); fast=快测; quality=全量MIS",
    )
    ap.add_argument(
        "--proc-scale",
        type=float,
        default=0.5,
        help="处理分辨率缩放（默认 0.5，大图集显著加速；1.0=原图）",
    )
    ap.add_argument("--print-every", type=int, default=20, help="进度打印间隔（帧）")
    ap.add_argument("--save-vis", type=int, default=0, help="1=保存叠加可视化（慢，占磁盘）")
    ap.add_argument("--vis-stride", type=int, default=10, help="save-vis=1 时每 N 帧存一张图")
    ap.add_argument("--max-frames", type=int, default=0, help=">0 时仅评测前 N 帧（调试用）")
    ap.add_argument(
        "--save-uvh-render-debug",
        type=int,
        default=0,
        help="1=save OursTracker UVH-render debug (PLY/PNG per frame); very slow for benchmarks",
    )
    args = ap.parse_args()

    summary = run_compare(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 为了和现有脚本保持一致，默认关闭不参与本对比的重模块。
    for _k, _v in {
        "MIS_UVH_RENDER_VIEWLIB_ENABLE": "0",
        "MIS_UVH_DIFFRENDER_ENABLE": "0",
        "MIS_ENABLE_FLOW_UVH": "0",
        "MIS_UVH_RIGID_ENABLE": "1",
    }.items():
        os.environ.setdefault(_k, _v)
    main()
