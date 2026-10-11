"""Minimal TrackVes I/O helpers extracted from the original project file
``trackves_eval_all_methods.py`` so that the evaluation code runs without the
rest of the historical framework. Behaviour is identical to the original
functions (GT annotation index equals the video frame index).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

GT_FRAME_STEP = 1


@dataclass
class GTFrame:
    frame_idx: int
    polygon: np.ndarray  # (N, 2) float32, x-y order
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
    """Parse TrackVes GT.xml and return GTFrame records sorted by frame index."""
    root = ET.parse(str(path)).getroot()
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
        items.append(GTFrame(frame_idx=int(ann_idx * frame_step), polygon=np.asarray(poly, dtype=np.float32)))
    items.sort(key=lambda x: x.frame_idx)
    return items


def _find_video(seq_dir: Path) -> Optional[Path]:
    """Locate the left-view video of a TrackVes sequence (tolerates the IV2 misspelling)."""
    for name in ["videoLeft.avi", "videoLeft.mp4", "videLeft.avi", "videLeft.mp4"]:
        p = seq_dir / name
        if p.exists():
            return p
    for ext in [".avi", ".mp4", ".mov"]:
        candidates = list(seq_dir.glob(f"*{ext}"))
        if candidates:
            return candidates[0]
    return None


def _clip_bbox(bbox: tuple[int, int, int, int], shape_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = int(shape_hw[0]), int(shape_hw[1])
    x, y, bw, bh = [int(v) for v in bbox]
    x = max(0, min(x, max(0, w - 1)))
    y = max(0, min(y, max(0, h - 1)))
    bw = max(1, min(max(1, bw), max(1, w - x)))
    bh = max(1, min(max(1, bh), max(1, h - y)))
    return x, y, bw, bh
