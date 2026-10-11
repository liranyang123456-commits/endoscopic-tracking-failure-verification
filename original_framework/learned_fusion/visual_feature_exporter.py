"""Synchronously export template/ROI visual response maps for every candidate.

This is independent of frozen tracker internals. All seven boxes for a sample
are evaluated against exactly the same decoded image and sequence template.
Channels are: grayscale ROI, edge response, template absolute difference, and
HSV template back-projection. GT is used only to crop the first-frame template.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from candidate_exporter import build_from_config


MAP_SIZE = 24
CHANNEL_NAMES = ("gray", "edge", "template_absdiff", "hsv_backprojection")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _find_video(root: Path, sequence: str) -> Path:
    seq_dir = root / sequence
    preferred = (
        "videoLeft_clean.mp4", "videoLeft.avi", "videoLeft.mp4",
        "videLeft.mp4", "video.avi", "video.mp4",
    )
    for name in preferred:
        path = seq_dir / name
        if path.is_file():
            return path
    videos = sorted([
        p for p in seq_dir.iterdir()
        if p.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}
    ])
    if not videos:
        raise FileNotFoundError(f"No video found for {sequence}: {seq_dir}")
    return videos[0]


def _crop(image: np.ndarray, box: np.ndarray, context: float = 1.5) -> np.ndarray:
    x, y, w, h = map(float, box)
    cx, cy = x + w / 2.0, y + h / 2.0
    w, h = max(4.0, w * context), max(4.0, h * context)
    ih, iw = image.shape[:2]
    x1, y1 = max(0, int(round(cx - w / 2))), max(0, int(round(cy - h / 2)))
    x2, y2 = min(iw, int(round(cx + w / 2))), min(ih, int(round(cy + h / 2)))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
    return cv2.resize(
        image[y1:y2, x1:x2], (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_AREA
    )


def _responses(patch: np.ndarray, template: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 40, 120)
    difference = cv2.absdiff(gray, template_gray)
    hsv_template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_template], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    backprojection = cv2.calcBackProject([hsv_patch], [0, 1], hist, [0, 180, 0, 256], 1)
    return np.stack([gray, edge, difference, backprojection]).astype(np.uint8)


class _FrameReader:
    def __init__(self, trackves_root: Path):
        self.trackves_root = trackves_root
        self.cap: cv2.VideoCapture | None = None
        self.sequence = ""
        self.next_frame = -1

    def read(self, domain: str, sequence: str, frame_idx: int, source_path: str) -> np.ndarray | None:
        if domain == "chess":
            return cv2.imread(source_path, cv2.IMREAD_COLOR)
        if sequence != self.sequence:
            if self.cap is not None:
                self.cap.release()
            video = _find_video(self.trackves_root, sequence)
            self.cap = cv2.VideoCapture(str(video))
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open {video}")
            self.sequence, self.next_frame = sequence, -1
        assert self.cap is not None
        if frame_idx != self.next_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, image = self.cap.read()
        self.next_frame = frame_idx + 1
        if not ok or image is None:
            # A second seek handles occasional H.264/OpenCV decoder drift.
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, image = self.cap.read()
            self.next_frame = frame_idx + 1
        return image if ok else None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


def export(config_path: Path, force: bool = False) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.parent
    artifacts = root / cfg["paths"]["artifacts"]
    candidate_path = artifacts / "candidate_dataset.npz"
    if not candidate_path.is_file():
        build_from_config(config_path)
    output = artifacts / "visual_responses.npz"
    if output.is_file() and not force:
        return output
    data = np.load(candidate_path, allow_pickle=False)
    n, k = data["boxes"].shape[:2]
    maps = np.zeros((n, k, len(CHANNEL_NAMES), MAP_SIZE, MAP_SIZE), dtype=np.uint8)
    valid = np.zeros(n, dtype=np.uint8)
    templates: dict[str, np.ndarray] = {}
    reader = _FrameReader(Path(cfg["paths"]["trackves_root"]))
    try:
        for i in range(n):
            domain, sequence, unit = map(str, (
                data["domain"][i], data["sequence"][i], data["unit"][i]
            ))
            image = reader.read(
                domain, sequence, int(data["frame_idx"][i]), str(data["source_path"][i])
            )
            if image is None:
                continue
            template_key = f"{domain}:{unit}"
            if template_key not in templates:
                templates[template_key] = _crop(image, data["gt"][i], context=1.0)
            template = templates[template_key]
            for candidate_idx, box in enumerate(data["boxes"][i]):
                maps[i, candidate_idx] = _responses(_crop(image, box), template)
            valid[i] = 1
            if (i + 1) % 250 == 0:
                print(f"[{i + 1}/{n}] visual responses", flush=True)
    finally:
        reader.close()
    np.savez_compressed(
        output, maps=maps, valid=valid,
        channel_names=np.asarray(CHANNEL_NAMES), map_size=np.asarray(MAP_SIZE),
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "samples": n, "valid_samples": int(valid.sum()),
        "candidate_count": k, "channels": CHANNEL_NAMES, "map_size": MAP_SIZE,
        "candidate_dataset_sha256": _sha256(candidate_path),
        "visual_responses_sha256": _sha256(output),
        "template_gt_use": "First-frame crop only; never an inference-time feature.",
        "synchronization": "All candidate maps use one decoded frame per sample.",
    }
    (artifacts / "visual_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    print(export(Path(args.config).resolve(), args.force))
