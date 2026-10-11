from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from contour_verifier import ContourVerifier
from protocol import HERE, SURGT_ROOT, box_iou, sha256, write_csv, write_json
from run_official_trackers import OfficialTracker, OpenCVBaseline


DATA = SURGT_ROOT  # override with the MFE_SURGT_ROOT environment variable


class TupleSafeLoader(yaml.SafeLoader):
    pass


TupleSafeLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node, deep=True)),
)


def load_sequence(directory: Path) -> tuple[dict, dict]:
    info = yaml.safe_load((directory / "info.yaml").read_text(encoding="utf-8"))
    if len(info["name_ground_truth"]) != 1:
        raise ValueError(f"Expected one declared target, not selective target filtering: {directory}")
    path = directory / info["name_ground_truth"][0]
    annotations = yaml.load(path.read_text(encoding="utf-8"), Loader=TupleSafeLoader)
    if not isinstance(annotations, dict) or not annotations:
        raise ValueError(f"Empty external GT: {path}")
    if any(not isinstance(key, int) or key < 0 for key in annotations):
        raise ValueError(f"Invalid external frame indices: {path}")
    return info, annotations


def rectification(directory: Path, info: dict) -> tuple[np.ndarray, np.ndarray]:
    storage = cv2.FileStorage(str(directory / info["name_calib"]), cv2.FILE_STORAGE_READ)
    if not storage.isOpened():
        raise OSError(f"Cannot read stereo calibration: {directory}")
    try:
        values = {name: storage.getNode(name).mat() for name in ("R", "T", "M1", "D1", "M2", "D2")}
        if any(value is None for value in values.values()):
            raise ValueError(f"Incomplete calibration: {directory}")
        size = (int(info["resolution"]["width"]), int(info["resolution"]["height"]))
        rotation, _, projection, _, _, _, _ = cv2.stereoRectify(
            values["M1"].astype(float), values["D1"].reshape(-1).astype(float),
            values["M2"].astype(float), values["D2"].reshape(-1).astype(float),
            size, values["R"].astype(float), values["T"].reshape(-1).astype(float),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
        return cv2.initUndistortRectifyMap(
            values["M1"], values["D1"], rotation, projection, size, cv2.CV_32FC1
        )
    finally:
        storage.release()


def left_view(frame: np.ndarray, info: dict, maps: tuple) -> np.ndarray:
    width, height = int(info["resolution"]["width"]), int(info["resolution"]["height"])
    if info["video_stack"] == "vertical":
        if frame.shape[0] != height * 2 or frame.shape[1] != width:
            raise ValueError("Stereo video shape differs from declared vertical stack")
        left = frame[:height]
    elif info["video_stack"] == "horizontal":
        if frame.shape[0] != height or frame.shape[1] != width * 2:
            raise ValueError("Stereo video shape differs from declared horizontal stack")
        left = frame[:, :width]
    else:
        raise ValueError(f"Unknown stereo stack: {info['video_stack']}")
    return cv2.remap(left, maps[0], maps[1], cv2.INTER_LINEAR)


def scoreable(annotation: tuple, info: dict) -> bool:
    visible, difficult, boxes = annotation
    if not visible or difficult or boxes is None or boxes[0] is None:
        return False
    box = tuple(float(x) for x in boxes[0])
    width, height = info["resolution"]["width"], info["resolution"]["height"]
    return (
        len(box) == 4 and np.isfinite(box).all() and min(box[2:]) > 0
        and box[0] >= 0 and box[1] >= 0
        and box[0] + box[2] <= width and box[1] + box[3] <= height
    )


def run(split: str, proposer: str, output: Path, smoke: bool = False) -> None:
    protocol_path = HERE / "verification_protocol.json"
    if split == "test":
        freeze = output / "selected_verification.json"
        if not freeze.is_file():
            raise RuntimeError("External test must wait until validation-only protocol selection is frozen")
        selected = __import__("json").loads(freeze.read_text(encoding="utf-8"))
        if selected["protocol_sha256"] != sha256(protocol_path):
            raise RuntimeError("Protocol changed after validation selection")
    directories = sorted((DATA / split).glob("*\\*\\info.yaml"))
    if not directories:
        raise FileNotFoundError(f"External split not available: {DATA / split}")
    if smoke:
        directories = directories[:1]
    if proposer in ("stark", "ostrack"):
        tracker = OfficialTracker(proposer, "surgt")
    else:
        tracker = OpenCVBaseline("nanotrack")
    all_rows, metadata = [], []
    for item in directories:
        directory = item.parent
        info, annotations = load_sequence(directory)
        sequence = "_".join(directory.relative_to(DATA / split).parts)
        first_frame = min(index for index, value in annotations.items() if scoreable(value, info))
        maps = rectification(directory, info)
        capture = cv2.VideoCapture(
            str(directory / info["name_video"]), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        if not capture.isOpened():
            raise OSError(f"Cannot open external video: {directory}")
        hashes, elapsed_monitor, elapsed_tracker = {}, 0.0, 0.0
        expected = sum(index > first_frame and scoreable(value, info) for index, value in annotations.items())
        rows, update_count = [], 0
        verifier = None
        try:
            last_frame = min(max(annotations), first_frame + 12) if smoke else max(annotations)
            for index in range(last_frame + 1):
                ok, packed = capture.read()
                if not ok:
                    raise OSError(f"Cannot decode annotated external frame: {sequence}/{index}")
                if index < first_frame:
                    continue
                image = left_view(packed, info, maps)
                hashes[str(index)] = hashlib.sha256(image.tobytes()).hexdigest()
                if index == first_frame:
                    initial_box = tuple(float(value) for value in annotations[index][2][0])
                    tracker.initialize(image, initial_box)
                    verifier = ContourVerifier(image, initial_box)
                    continue
                start = time.perf_counter()
                predicted = tracker.update(image)
                elapsed_tracker += time.perf_counter() - start
                update_count += 1
                if index not in annotations or not scoreable(annotations[index], info):
                    continue
                ground_truth = tuple(float(value) for value in annotations[index][2][0])
                risk_start = time.perf_counter()
                if predicted is None:
                    contour = (1.0, 1.0, 1.0)
                    rank = color = confidence = 1.0
                else:
                    measured = verifier.measure(image, predicted, tracker.last_confidence)
                    contour = measured.contour_distance
                    rank, color, confidence = (
                        measured.rank_disagreement, measured.color_disagreement,
                        measured.native_confidence_risk,
                    )
                elapsed_monitor += time.perf_counter() - risk_start
                iou = box_iou(ground_truth, predicted)
                rows.append({
                    "split": split, "sequence": sequence,
                    "case": directory.parent.name, "frame_idx": index, "proposer": proposer,
                    "box_iou": iou, "failure": int(iou < .5),
                    "prediction_present": int(predicted is not None),
                    "contour_50": contour[0], "contour_68": contour[1], "contour_80": contour[2],
                    "rank_risk": rank, "color_risk": color, "native_risk": confidence,
                })
        finally:
            capture.release()
        if not smoke and len(rows) != expected:
            raise ValueError(f"External scoring denominator mismatch: {sequence} {len(rows)} != {expected}")
        all_rows.extend(rows)
        metadata.append({
            "sequence": sequence, "case": directory.parent.name,
            "video_sha256": sha256(directory / info["name_video"]),
            "ground_truth_sha256": sha256(directory / info["name_ground_truth"][0]),
            "first_supplied_box_frame": first_frame, "scored_frames": len(rows),
            "post_initialization_updates": update_count, "input_frame_sha256": hashes,
            "proposer_time_s": elapsed_tracker, "monitor_time_s": elapsed_monitor,
        })
        print(f"{split}/{proposer}/{sequence}: {len(rows)} scored frames; "
              f"monitor {elapsed_monitor / max(len(rows), 1) * 1000:.2f}ms/frame", flush=True)
    prefix = output / ("smoke_external" if smoke else "external_verification") / split
    write_csv(prefix / f"{proposer}.csv", all_rows)
    write_json(prefix / f"{proposer}.manifest.json", {
        "split": split, "proposer": proposer, "proposer_metadata": tracker.metadata,
        "protocol_sha256": sha256(protocol_path),
        "no_local_weights_fitted": True, "proposer_box_unchanged": True,
        "stereo_view": "left; official calibration rectification",
        "scoring": "visible/non-difficult valid GT; no GT resets; initial box not scored",
        "sequences": metadata,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--proposer", choices=("stark", "nanotrack", "ostrack"), required=True)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    run(args.split, args.proposer, args.output.resolve(), args.smoke)


if __name__ == "__main__":
    main()
