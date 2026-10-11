"""Probe global template/feature re-detection on TrackVes GT frames.

Only the first annotated frame supplies the template. Later GT boxes are used
for evaluation, never by candidate generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from candidate_exporter import box_iou
from visual_feature_exporter import _FrameReader


def _clip(box: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    x, y, bw, bh = map(float, box)
    bw, bh = min(max(4.0, bw), w), min(max(4.0, bh), h)
    return np.asarray([
        min(max(0.0, x), max(0.0, w - bw)),
        min(max(0.0, y), max(0.0, h - bh)), bw, bh,
    ], dtype=np.float32)


def _template_crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    x, y, w, h = map(int, np.round(box))
    return image[max(0, y):min(image.shape[0], y + h),
                 max(0, x):min(image.shape[1], x + w)].copy()


def _norm_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(2.0, (8, 8)).apply(gray)


def template_candidate(
    image: np.ndarray, template: np.ndarray, scales: tuple[float, ...],
    work_scale: float = 0.5,
) -> tuple[np.ndarray, float]:
    frame = cv2.resize(image, None, fx=work_scale, fy=work_scale)
    frame_gray = _norm_gray(frame)
    template_gray = _norm_gray(template)
    best_score, best_box = -1.0, np.zeros(4, dtype=np.float32)
    for scale in scales:
        tw = max(8, int(round(template.shape[1] * scale * work_scale)))
        th = max(8, int(round(template.shape[0] * scale * work_scale)))
        if tw >= frame_gray.shape[1] or th >= frame_gray.shape[0]:
            continue
        resized = cv2.resize(template_gray, (tw, th), interpolation=cv2.INTER_AREA)
        response = cv2.matchTemplate(frame_gray, resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(response)
        if score > best_score:
            best_score = float(score)
            best_box = np.asarray([
                location[0] / work_scale, location[1] / work_scale,
                tw / work_scale, th / work_scale,
            ], dtype=np.float32)
    return _clip(best_box, image.shape[:2]), best_score


class OrbRedetector:
    def __init__(self, template: np.ndarray):
        self.template = template
        self.orb = cv2.ORB_create(nfeatures=1200, scaleFactor=1.15, nlevels=10)
        self.kp_template, self.desc_template = self.orb.detectAndCompute(
            _norm_gray(template), None
        )
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        kp_image, desc_image = self.orb.detectAndCompute(_norm_gray(image), None)
        if self.desc_template is None or desc_image is None:
            return np.zeros(4, dtype=np.float32), 0.0
        pairs = self.matcher.knnMatch(self.desc_template, desc_image, k=2)
        good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
        if len(good) < 4:
            return np.zeros(4, dtype=np.float32), 0.0
        source = np.float32([self.kp_template[m.queryIdx].pt for m in good])
        target = np.float32([kp_image[m.trainIdx].pt for m in good])
        matrix, inliers = cv2.estimateAffinePartial2D(
            source, target, method=cv2.RANSAC, ransacReprojThreshold=4.0
        )
        if matrix is None or inliers is None:
            return np.zeros(4, dtype=np.float32), 0.0
        h, w = self.template.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])[None]
        mapped = cv2.transform(corners, matrix)[0]
        x, y, bw, bh = cv2.boundingRect(mapped)
        inlier_count = int(inliers.sum())
        confidence = (inlier_count / max(len(good), 1)) * min(1.0, inlier_count / 12.0)
        return _clip(np.asarray([x, y, bw, bh]), image.shape[:2]), float(confidence)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/candidate_dataset.npz"))
    parser.add_argument("--sequences", default="EV3,IV1,IV2")
    parser.add_argument("--max-samples-per-sequence", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/global_redetection_probe.npz"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = np.load(args.dataset, allow_pickle=False)
    requested = {item.strip() for item in args.sequences.split(",") if item.strip()}
    reader = _FrameReader(Path(cfg["paths"]["trackves_root"]))
    indices, candidates, scores = [], [], []
    summaries = []
    try:
        for sequence in sorted(requested):
            sequence_indices = np.flatnonzero(
                (data["domain"] == "trackves") & (data["sequence"] == sequence)
            )
            if args.max_samples_per_sequence > 0:
                sequence_indices = sequence_indices[:args.max_samples_per_sequence]
            if len(sequence_indices) == 0:
                continue
            first = int(sequence_indices[0])
            first_image = reader.read(
                "trackves", sequence, int(data["frame_idx"][first]), ""
            )
            if first_image is None:
                continue
            template = _template_crop(first_image, data["gt"][first])
            orb = OrbRedetector(template)
            sequence_ious = {"nano": [], "template": [], "orb": [], "oracle": []}
            for count, index in enumerate(sequence_indices):
                index = int(index)
                image = reader.read(
                    "trackves", sequence, int(data["frame_idx"][index]), ""
                )
                if image is None:
                    continue
                template_box, template_score = template_candidate(
                    image, template, (0.65, 0.8, 1.0, 1.2, 1.4)
                )
                orb_box, orb_score = orb.detect(image)
                nano = data["boxes"][index, 1].astype(np.float32)
                gt = data["gt"][index]
                boxes = np.stack([nano, template_box, orb_box])
                ious = [box_iou(box, gt) for box in boxes]
                for key, value in zip(("nano", "template", "orb"), ious):
                    sequence_ious[key].append(value)
                sequence_ious["oracle"].append(max(ious))
                indices.append(index)
                candidates.append(boxes)
                scores.append([1.0, template_score, orb_score])
                if (count + 1) % 100 == 0:
                    print(f"[{sequence}] {count + 1}/{len(sequence_indices)}", flush=True)
            summaries.append({
                "sequence": sequence,
                **{key: float(np.mean(value)) for key, value in sequence_ious.items()},
            })
    finally:
        reader.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, indices=np.asarray(indices), boxes=np.asarray(candidates),
        scores=np.asarray(scores, dtype=np.float32),
    )
    report = args.output.with_suffix(".json")
    report.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
