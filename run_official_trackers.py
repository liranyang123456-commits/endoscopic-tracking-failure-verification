from __future__ import annotations

import argparse
import importlib
import hashlib
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from protocol import (
    HERE, GroundTruth, align_predictions, canonical_chess,
    canonical_trackves, sha256, write_csv, write_json,
)


class FrameReader:
    def __init__(self) -> None:
        self.capture = None
        self.video = None
        self.next_frame = 0

    def read(self, record: GroundTruth) -> np.ndarray:
        path = Path(record.image_path)
        if path.suffix.lower() in (".avi", ".mp4", ".mov"):
            cached = HERE / "results" / "frozen_trackves_frames" / record.sequence / f"{record.frame:06d}.png"
            if cached.is_file():
                image = cv2.imread(str(cached))
                if image is None:
                    raise OSError(f"Cannot decode frozen evaluation frame: {cached}")
                return image
        if path.suffix.lower() not in (".avi", ".mp4", ".mov"):
            image = cv2.imread(str(path))
            if image is None:
                raise OSError(f"Cannot decode image: {path}")
            return image
        if self.video != path:
            self.close()
            self.capture = cv2.VideoCapture(str(path))
            if not self.capture.isOpened():
                raise OSError(f"Cannot open video: {path}")
            self.video, self.next_frame = path, 0
        if record.frame < self.next_frame:
            raise ValueError("Video evaluation must visit frames in increasing order")
        while self.next_frame < record.frame:
            if not self.capture.grab():
                raise OSError(f"Cannot advance to annotated frame: {path} {record.frame}")
            self.next_frame += 1
        ok, image = self.capture.read()
        self.next_frame += 1
        if not ok:
            raise OSError(f"Cannot decode annotated frame: {path} {record.frame}")
        return image

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = self.video = None
        self.next_frame = 0


class OfficialTracker:
    def __init__(self, name: str, dataset: str):
        repository = HERE / "external" / ("OSTrack" if name == "ostrack" else "STARK")
        sys.path.insert(0, str(repository))
        weight = HERE / "weights" / (
            "ostrack_vitb256_ce.pth.tar" if name == "ostrack" else "stark_st50.pth.tar"
        )
        if not weight.is_file():
            raise FileNotFoundError(f"Author-released checkpoint missing: {weight}")
        safe_checkpoint = weight.with_suffix(".state.pt")
        if not safe_checkpoint.is_file():
            local_file = repository / "lib" / "train" / "admin" / "local.py"
            if name == "stark" and not local_file.is_file():
                environment = importlib.import_module("lib.train.admin.environment")
                environment.create_default_local_file()
            # Author-released archives contain training metadata in addition to tensors.
            payload = torch.load(weight, map_location="cpu", weights_only=False)
            torch.save({"net": payload["net"]}, safe_checkpoint)
        if name == "ostrack":
            config_module = importlib.import_module("lib.config.ostrack.config")
            config_path = repository / "experiments" / "ostrack" / "vitb_256_mae_ce_32x4_ep300.yaml"
            tracker_class = importlib.import_module("lib.test.tracker.ostrack").OSTrack
        else:
            config_module = importlib.import_module("lib.config.stark_st2.config")
            config_path = repository / "experiments" / "stark_st2" / "baseline.yaml"
            tracker_class = importlib.import_module("lib.test.tracker.stark_st").STARK_ST
        config_module.update_config_from_file(str(config_path))
        cfg = config_module.cfg
        params = SimpleNamespace(
            cfg=cfg, checkpoint=str(safe_checkpoint), debug=0, save_all_boxes=False,
            template_factor=cfg.TEST.TEMPLATE_FACTOR, template_size=cfg.TEST.TEMPLATE_SIZE,
            search_factor=cfg.TEST.SEARCH_FACTOR, search_size=cfg.TEST.SEARCH_SIZE,
        )
        self.tracker = tracker_class(params, dataset)
        self.last_confidence = 1.0
        self._last_response = None
        if name == "ostrack":
            # OSTrack exposes no scalar confidence. Record the maximum of the
            # Hann-windowed classification score map that its own box head
            # already consumes; the box computation itself is untouched.
            head = self.tracker.network.box_head
            original_cal_bbox = head.cal_bbox

            def recording_cal_bbox(score_map, size_map, offset_map, *args, **kwargs):
                self._last_response = float(score_map.max().item())
                return original_cal_bbox(score_map, size_map, offset_map, *args, **kwargs)

            head.cal_bbox = recording_cal_bbox
        self.metadata = {
            "repository": str(repository),
            "commit": subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
            ).strip(),
            "checkpoint": str(weight), "checkpoint_sha256": sha256(weight),
            "configuration": str(config_path), "config_sha256": sha256(config_path),
            "all_network_keys_loaded_strictly": True,
            "compatibility_patches": "torch._six aliases and torchvision URL-loader import only",
            "local_dataset_fine_tuning": False,
            "input": "RGB; official crop, normalization, box head and update rule",
            "native_confidence": (
                "maximum of the Hann-windowed classification score map consumed by the official box head"
                if name == "ostrack" else "official conf_score (sigmoid of the reliability logit)"
            ),
        }

    def initialize(self, image: np.ndarray, box: tuple) -> None:
        self.tracker.initialize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), {"init_bbox": list(box)})

    def update(self, image: np.ndarray) -> tuple[float, float, float, float] | None:
        self._last_response = None
        output = self.tracker.track(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if "conf_score" in output:
            self.last_confidence = float(output["conf_score"])
        elif self._last_response is not None:
            self.last_confidence = float(np.clip(self._last_response, 0.0, 1.0))
        else:
            self.last_confidence = 1.0
        values = tuple(float(x) for x in output["target_bbox"])
        if len(values) != 4 or not all(np.isfinite(values)):
            raise ValueError(f"Official tracker produced invalid box: {values}")
        return values if values[2] > 0 and values[3] > 0 else None


class OpenCVBaseline:
    def __init__(self, name: str):
        self.name = name
        self.tracker = None
        self.last_confidence = 1.0
        if name == "nanotrack":
            from legacy.trackves_io import _clip_bbox
            from protocol import NANOTRACK_DIR
            self._clip_bbox = _clip_bbox
            backbone = NANOTRACK_DIR / "nanotrack_backbone_sim.onnx"
            head = NANOTRACK_DIR / "nanotrack_head_sim.onnx"
            if not backbone.is_file() or not head.is_file():
                raise FileNotFoundError(f"NanoTrack ONNX models not found in {NANOTRACK_DIR}; see weights/README.md")
            self.metadata = {
                "backend": "OpenCV TrackerNano", "opencv_version": cv2.__version__,
                "backbone_sha256": sha256(backbone), "head_sha256": sha256(head),
                "local_dataset_fine_tuning": False,
                "output_box_policy": "integer rounding and clipping identical to the gate's Nano wrapper",
            }
        else:
            self.metadata = {
                "backend": "OpenCV TrackerCSRT", "opencv_version": cv2.__version__,
                "local_dataset_fine_tuning": False,
            }

    def initialize(self, image: np.ndarray, box: tuple) -> None:
        if self.name == "nanotrack":
            from protocol import NANOTRACK_DIR
            params = cv2.TrackerNano_Params()
            params.backbone = str(NANOTRACK_DIR / "nanotrack_backbone_sim.onnx")
            params.neckhead = str(NANOTRACK_DIR / "nanotrack_head_sim.onnx")
            self.tracker = cv2.TrackerNano_create(params)
        else:
            factory = getattr(cv2, "TrackerCSRT_create", None)
            if factory is None:
                factory = cv2.legacy.TrackerCSRT_create
            self.tracker = factory()
        self.tracker.init(image, tuple(int(round(value)) for value in box))

    def update(self, image: np.ndarray):
        ok, box = self.tracker.update(image)
        self.last_confidence = (
            float(self.tracker.getTrackingScore())
            if ok and hasattr(self.tracker, "getTrackingScore") else float(ok)
        )
        if not ok:
            return None
        if self.name == "nanotrack":
            rounded = tuple(int(round(float(value))) for value in box)
            clipped = self._clip_bbox(
                (rounded[0], rounded[1], max(1, rounded[2]), max(1, rounded[3])),
                image.shape[:2],
            )
            return tuple(float(value) for value in clipped)
        return tuple(float(value) for value in box)


def run(name: str, dataset: str, output: Path, smoke_frames: int = 0) -> None:
    gt = canonical_chess() if dataset == "chess" else canonical_trackves()
    units = defaultdict(list)
    for record in gt:
        units[record.sequence, record.unit].append(record)
    if smoke_frames:
        first = next(iter(sorted(units)))
        units = {first: units[first][:smoke_frames]}
        gt = units[first]
    tracker = (
        OpenCVBaseline(name) if name in ("nanotrack", "csrt")
        else OfficialTracker(name, dataset)
    )
    predictions, timings, input_hashes, confidences = {}, [], {}, {}
    for number, (key, records) in enumerate(sorted(units.items()), 1):
        reader = FrameReader()
        start = time.perf_counter()
        try:
            image = reader.read(records[0])
            input_hashes[str(records[0].key)] = hashlib.sha256(image.tobytes()).hexdigest()
            tracker.initialize(image, records[0].box)
            predictions[records[0].key] = records[0].box
            confidences[records[0].key] = 1.0
            for record in records[1:]:
                image = reader.read(record)
                input_hashes[str(record.key)] = hashlib.sha256(image.tobytes()).hexdigest()
                predictions[record.key] = tracker.update(image)
                confidences[record.key] = float(tracker.last_confidence)
        finally:
            reader.close()
        seconds = time.perf_counter() - start
        timings.append({
            "sequence": key[0], "unit": key[1], "frames": len(records),
            "elapsed_s_including_io_initialization": seconds,
            "fps_including_io_initialization": len(records) / seconds,
        })
        print(f"{name} {dataset} [{number}/{len(units)}] {key}: "
              f"{len(records)} frames, {seconds:.1f}s", flush=True)
    directory = output / ("smoke" if smoke_frames else "official") / dataset
    aligned = align_predictions(gt, predictions)
    for row in aligned:
        row["native_confidence"] = confidences.get(
            (row["sequence"], row["unit"], int(row["frame_idx"])), ""
        )
    write_csv(directory / f"{name}.csv", aligned)
    write_json(directory / f"{name}.manifest.json", {
        **tracker.metadata, "dataset": dataset, "protocol": "shared first-GT initialization; GT-frame updates",
        "unit_count": len(units), "gt_frame_count": len(gt),
        "scored_frame_count": sum(not r.is_initialization for r in gt),
        "initialization_frames_excluded_from_scores": True, "timings": timings,
        "input_frame_sha256": input_hashes,
        "smoke_only": bool(smoke_frames),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=("ostrack", "stark", "nanotrack", "csrt"), required=True)
    parser.add_argument("--dataset", choices=("chess", "trackves", "both"), default="both")
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--smoke-frames", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261011)
    datasets = ("chess", "trackves") if args.dataset == "both" else (args.dataset,)
    for dataset in datasets:
        run(args.tracker, dataset, args.output, args.smoke_frames)


if __name__ == "__main__":
    main()
