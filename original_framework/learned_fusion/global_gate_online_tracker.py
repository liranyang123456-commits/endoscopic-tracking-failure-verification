"""Online Nano + global re-detection tracker using a held-out fold gate."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from global_redetection_gate_oof import _features
from global_redetection_probe import OrbRedetector, template_candidate
from global_redetection_visual_gate_oof import VisualQualityGate
from visual_feature_exporter import _crop, _responses


class GlobalGateOnlineTracker:
    name = "ours_global_visual_gate_oof"
    label = "Ours Global Visual Gate (LOSO)"
    backend_name = name

    def __init__(
        self, init_image: np.ndarray, init_bbox: tuple[int, int, int, int],
        checkpoint: str | Path, frame_indices: list[int] | np.ndarray | None = None,
    ):
        from trackves_eval_all_methods import NanoTrackTracker
        self.nano = NanoTrackTracker(init_image, init_bbox)
        self.template = self._raw_crop(init_image, np.asarray(init_bbox, dtype=np.float32))
        self.template_map = cv2.resize(
            self.template, (24, 24), interpolation=cv2.INTER_AREA
        )
        self.orb = OrbRedetector(self.template)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = VisualQualityGate(int(payload["scalar_dim"]))
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.mean = np.asarray(payload["scalar_mean"], dtype=np.float32)
        self.std = np.asarray(payload["scalar_std"], dtype=np.float32)
        self.threshold = float(payload["threshold"])
        self.frame_idx = 0
        self.frame_indices = (
            [] if frame_indices is None else list(map(int, frame_indices))
        )
        self.update_index = 1
        self.last_route = 0

    @staticmethod
    def _raw_crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
        x, y, w, h = map(int, np.round(box))
        patch = image[
            max(0, y):min(image.shape[0], y + h),
            max(0, x):min(image.shape[1], x + w),
        ]
        if patch.size == 0:
            raise ValueError("Initial template crop is empty")
        return patch.copy()

    def set_frame_context(self, frame_idx: int, _image_path: str = "") -> None:
        self.frame_idx = int(frame_idx)

    def propagate(self, frame: np.ndarray) -> None:
        self.nano.update(frame)

    @torch.no_grad()
    def update(self, frame: np.ndarray) -> dict[str, Any]:
        if self.update_index < len(self.frame_indices):
            self.frame_idx = self.frame_indices[self.update_index]
            self.update_index += 1
        nano_output = self.nano.update(frame)
        nano_box = np.asarray(nano_output["bbox"], dtype=np.float32)
        template_box, template_score = template_candidate(
            frame, self.template, (0.65, 0.8, 1.0, 1.2, 1.4)
        )
        orb_box, orb_score = self.orb.detect(frame)
        boxes = np.stack([nano_box, template_box, orb_box]).astype(np.float32)
        scores = np.asarray([[1.0, template_score, orb_score]], dtype=np.float32)
        nano_score = np.asarray([
            float(nano_output.get("score", nano_output.get("nano_score", 0.0)))
        ], dtype=np.float32)
        scalar = _features(
            boxes[None], scores, nano_score,
            np.asarray([self.frame_idx], dtype=np.float32),
        )
        scalar = (scalar - self.mean) / self.std
        maps = np.stack([
            _responses(_crop(frame, box), self.template_map) for box in boxes
        ])[None].astype(np.float32) / 255.0
        quality = self.model(
            torch.from_numpy(maps), torch.from_numpy(scalar).float()
        )[0].numpy()
        winner = int(np.argmax(quality))
        gain = float(quality[winner] - quality[0])
        choice = winner if gain > self.threshold else 0
        self.last_route = choice
        selected = boxes[choice]
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x, y, bw, bh = map(int, np.round(selected))
        mask[max(0, y):min(h, y + bh), max(0, x):min(w, x + bw)] = 255
        output = dict(nano_output)
        output.update({
            "bbox": selected.tolist(), "mask_full": mask,
            "score": float(quality[choice]),
            "global_gate": {
                "choice": ("nano", "template", "orb")[choice],
                "quality": quality.tolist(), "gain": gain,
                "threshold": self.threshold,
                "template_score": template_score, "orb_score": orb_score,
            },
        })
        return output
