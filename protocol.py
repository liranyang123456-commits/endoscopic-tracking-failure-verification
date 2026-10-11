from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import wilcoxon


import os


def _env_path(name: str, default: str | Path) -> Path:
    """Data locations are configurable through environment variables (see README.md);
    the defaults are the authors' local layout."""
    return Path(os.environ.get(name, str(default)))


HERE = Path(__file__).resolve().parent
PROJECT = _env_path("MFE_PROJECT_ROOT", HERE.parents[2] if len(HERE.parents) > 2 else HERE)
LEARNED = _env_path("MFE_LEARNED_FUSION_ROOT", PROJECT / "BSPC_Submission_v2" / "06_OursSchemeC_LearnedFusion")
CHESS_ROOT = _env_path(
    "MFE_CHESS_ROOT",
    r"F:\MIS_Pose_Track_Re3D\BenchmarkSuite\Results\Chessboard\Exp_Results_Chessboard_Final_Optimize_v3",
)
BASE_ROOT = _env_path("MFE_LEGACY_BASELINE_ROOT", PROJECT / "Exp_Compare_iScience_GTMask")
TRACKVES_ROOT = _env_path("MFE_TRACKVES_ROOT", r"E:\Surgical_Tracking_Datasets\TrackVes\TrackVes\TrackVes")
TRACKVES_RESULTS = _env_path("MFE_TRACKVES_LEGACY_RESULTS", PROJECT / "Exp_Rerun_TrackVes_Paper_20260717" / "full_9seq_10methods")
SEGMENTATION_ROOT = _env_path("MFE_SEGMENTATION_ROOT", r"D:\reloc3r")
SURGT_ROOT = _env_path("MFE_SURGT_ROOT", r"E:\Surgical_Tracking_Datasets\Benchmarks\SurgT_benchmarking\data")
NANOTRACK_DIR = _env_path("MFE_NANOTRACK_DIR", HERE / "weights" if (HERE / "weights" / "nanotrack_backbone_sim.onnx").is_file() else PROJECT)
Box = tuple[float, float, float, float]
Key = tuple[str, str, int]


@dataclass(frozen=True)
class GroundTruth:
    sequence: str
    unit: str
    frame: int
    box: Box
    image_path: str
    is_initialization: bool = False

    @property
    def key(self) -> Key:
        return self.sequence, self.unit, self.frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def parse_box(row: Mapping[str, str], prefix: str) -> Box | None:
    values = [row[f"{prefix}{axis}"] for axis in ("x", "y", "w", "h")]
    if any(value.strip() == "" for value in values):
        return None
    box = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in box):
        raise ValueError(f"Non-finite box: {values}")
    if box[2] <= 0 or box[3] <= 0:
        return None
    return box


def box_iou(gt: Box, pred: Box | None) -> float:
    if pred is None:
        return 0.0
    gx, gy, gw, gh = gt
    px, py, pw, ph = pred
    intersection = max(0.0, min(gx + gw, px + pw) - max(gx, px)) * max(
        0.0, min(gy + gh, py + ph) - max(gy, py)
    )
    return intersection / (gw * gh + pw * ph - intersection)


def center_error(gt: Box, pred: Box | None) -> float | None:
    if pred is None:
        return None
    return math.hypot(
        gt[0] + gt[2] / 2 - pred[0] - pred[2] / 2,
        gt[1] + gt[3] / 2 - pred[1] - pred[3] / 2,
    )


def canonical_chess() -> list[GroundTruth]:
    records: dict[Key, GroundTruth] = {}
    for path in sorted(CHESS_ROOT.rglob("nanotrack_rows.csv")):
        parts = path.relative_to(CHESS_ROOT).parts
        sequence = parts[0]
        if "boards" in parts:
            unit = parts[2]
        else:
            if list((CHESS_ROOT / sequence / "boards").glob("*\\nanotrack_rows.csv")):
                continue
            unit = "sequence_target"
        for row in read_csv(path):
            gt = parse_box(row, "gt_bbox_")
            if gt is None:
                raise ValueError(f"Invalid canonical GT in {path}: {row['frame_idx']}")
            record = GroundTruth(
                sequence, unit, int(row["frame_idx"]), gt, row["image_path"]
            )
            if record.key in records:
                raise ValueError(f"Duplicate canonical GT: {record.key}")
            records[record.key] = record
    if len({(r.sequence, r.unit) for r in records.values()}) != 22:
        raise ValueError("Expected exactly 22 canonical board tracks")
    if len({r.sequence for r in records.values()}) != 11:
        raise ValueError("Expected exactly 11 canonical chess sequences")
    result = sorted(records.values(), key=lambda r: r.key)
    first_keys = {}
    for record in result:
        first_keys.setdefault((record.sequence, record.unit), record.key)
    return [
        replace(record, is_initialization=record.key == first_keys[record.sequence, record.unit])
        for record in result
    ]


def canonical_trackves() -> list[GroundTruth]:
    import cv2

    from legacy.trackves_io import _find_video, parse_gt_xml

    records = []
    for sequence in ("EV1", "EV2", "EV3", "IV1", "IV2", "IV3", "IV4", "IV5", "IV6"):
        directory = TRACKVES_ROOT / sequence
        video = _find_video(directory)
        if video is None:
            raise FileNotFoundError(f"No TrackVes video: {directory}")
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        annotations = parse_gt_xml(directory / "GT.xml")
        if not annotations:
            raise ValueError(f"No valid TrackVes annotations: {directory}")
        for annotation in annotations:
            polygon = np.round(annotation.polygon).astype(np.int32)
            x, y, w, h = cv2.boundingRect(polygon)
            x2, y2 = min(width, x + w + 2), min(height, y + h + 2)
            x, y = max(0, x - 2), max(0, y - 2)
            if x2 <= x or y2 <= y:
                raise ValueError(f"GT polygon outside image: {sequence} {annotation.frame_idx}")
            records.append(GroundTruth(
                sequence, "roi", int(annotation.frame_idx),
                (float(x), float(y), float(x2 - x), float(y2 - y)),
                str(video), annotation is annotations[0],
            ))
    if len({r.key for r in records}) != len(records):
        raise ValueError("Duplicate TrackVes annotation key")
    return sorted(records, key=lambda r: r.key)


def align_predictions(
    ground_truth: Iterable[GroundTruth], predictions: Mapping[Key, Box | None]
) -> list[dict]:
    gt = list(ground_truth)
    if not gt or len({item.key for item in gt}) != len(gt):
        raise ValueError("GT must be nonempty and have unique keys")
    extra = set(predictions) - {item.key for item in gt}
    if extra:
        raise ValueError(f"Predictions outside canonical evaluation set: {sorted(extra)[:3]}")
    rows = []
    for item in gt:
        pred = predictions.get(item.key)
        error = center_error(item.box, pred)
        rows.append({
            "sequence": item.sequence, "unit": item.unit, "frame_idx": item.frame,
            "is_initialization": int(item.is_initialization),
            "gt_x": item.box[0], "gt_y": item.box[1],
            "gt_w": item.box[2], "gt_h": item.box[3],
            "pred_x": "" if pred is None else pred[0],
            "pred_y": "" if pred is None else pred[1],
            "pred_w": "" if pred is None else pred[2],
            "pred_h": "" if pred is None else pred[3],
            "prediction_present": int(pred is not None),
            "bbox_iou": box_iou(item.box, pred),
            "center_error_px": "" if error is None else error,
            "success_05": int(box_iou(item.box, pred) >= 0.5),
            "precision_20": int(error is not None and error <= 20),
            "status": "missing_prediction" if pred is None else "predicted",
        })
    return rows


def holm_adjust(pvalues: Iterable[float]) -> list[float]:
    p = np.asarray(list(pvalues), dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("P values must be finite and between zero and one")
    order = np.argsort(p)
    corrected = np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1))
    result = np.empty(len(p))
    result[order] = np.minimum(1.0, corrected)
    return result.tolist()


def paired_statistics(
    reference: Mapping[str, float], comparator: Mapping[str, float],
    *, seed: int = 20261011, resamples: int = 20000,
) -> dict:
    if set(reference) != set(comparator):
        raise ValueError("Paired statistics require identical independent-group keys")
    names = sorted(reference)
    if len(names) < 2:
        raise ValueError("At least two independent groups are required")
    delta = np.asarray([reference[n] - comparator[n] for n in names])
    if not np.isfinite(delta).all():
        raise ValueError("Non-finite paired differences")
    rng = np.random.default_rng(seed)
    boot = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(axis=1)
    p = 1.0 if np.all(delta == 0) else float(
        wilcoxon(delta, alternative="two-sided", zero_method="wilcox", method="auto").pvalue
    )
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "n_groups": len(names), "groups": names,
        "mean_paired_difference": float(delta.mean()),
        "ci95_low": float(low), "ci95_high": float(high),
        "p_two_sided": p, "improved": int((delta > 0).sum()),
        "degraded": int((delta < 0).sum()), "ties": int((delta == 0).sum()),
        "bootstrap_resamples": resamples, "bootstrap_seed": seed,
    }


def validate_split(
    train: Iterable[str], validation: Iterable[str], test: Iterable[str]
) -> None:
    sets = [set(train), set(validation), set(test)]
    if any(not values for values in sets):
        raise ValueError("Train, validation and test groups must all be nonempty")
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Sequence leakage between training, validation and test")


def trackeval_data(rows: list[dict]) -> dict:
    gt_ids, tracker_ids, similarity = [], [], []
    present = 0
    for row in rows:
        exists = bool(int(row["prediction_present"]))
        gt_ids.append(np.asarray([0], dtype=np.int64))
        tracker_ids.append(np.asarray([0] if exists else [], dtype=np.int64))
        similarity.append(
            np.asarray([[float(row["bbox_iou"])]]) if exists else np.zeros((1, 0))
        )
        present += int(exists)
    return {
        "num_gt_ids": 1, "num_tracker_ids": int(present > 0),
        "num_gt_dets": len(rows), "num_tracker_dets": present,
        "num_timesteps": len(rows), "gt_ids": gt_ids,
        "tracker_ids": tracker_ids, "similarity_scores": similarity,
    }
