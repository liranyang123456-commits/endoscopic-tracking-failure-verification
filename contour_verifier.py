from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def ordinal_image(gray: np.ndarray) -> np.ndarray:
    values = np.asarray(gray)
    if values.ndim != 2 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Ordinal image requires a finite nonempty grayscale array")
    _, inverse, counts = np.unique(values.reshape(-1), return_inverse=True, return_counts=True)
    before = np.cumsum(counts) - counts
    midranks = (before + (counts - 1) / 2) / max(values.size - 1, 1)
    return midranks[inverse].reshape(values.shape).astype(np.float32)


def crop_resize(image: np.ndarray, box: tuple[float, float, float, float], size: int = 64) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Verifier input must be a BGR image")
    x, y, w, h = box
    if not np.isfinite(box).all() or w <= 0 or h <= 0:
        raise ValueError(f"Invalid verifier ROI: {box}")
    x1, y1 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    x2, y2 = min(image.shape[1], int(np.ceil(x + w))), min(image.shape[0], int(np.ceil(y + h)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Verifier ROI is outside image: {box}")
    return cv2.resize(image[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)


def contour_map(rank: np.ndarray, percentile: float) -> np.ndarray:
    if not 0 < percentile < 1:
        raise ValueError("Contour percentile must be between zero and one")
    smooth = cv2.GaussianBlur(rank, (3, 3), 0.8)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(dx, dy)
    interior = magnitude[2:-2, 2:-2]
    threshold = float(np.quantile(interior, percentile))
    result = (magnitude > max(threshold, 1e-6)).astype(np.uint8)
    result[:2] = result[-2:] = 0
    result[:, :2] = result[:, -2:] = 0
    return result


def symmetric_trimmed_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Contour distance requires matching raster shapes")
    if not a.any() or not b.any():
        return 1.0
    da = cv2.distanceTransform((1 - a).astype(np.uint8), cv2.DIST_L2, 3)
    db = cv2.distanceTransform((1 - b).astype(np.uint8), cv2.DIST_L2, 3)
    distances = np.concatenate([da[b > 0], db[a > 0]])
    limit = float(np.quantile(distances, 0.9))
    return float(np.clip(np.minimum(distances, limit).mean() / 8.0, 0, 1))


def color_histogram(patch: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256]).reshape(-1)
    total = float(hist.sum())
    if total == 0:
        raise ValueError("Empty color histogram")
    return (hist / total).astype(np.float32)


@dataclass(frozen=True)
class VerificationResult:
    contour_distance: tuple[float, ...]
    rank_disagreement: float
    color_disagreement: float
    native_confidence_risk: float
    alignment_response: float

    def vector(self, percentile_index: int) -> np.ndarray:
        return np.asarray([
            self.contour_distance[percentile_index], self.rank_disagreement,
            self.color_disagreement, self.native_confidence_risk,
        ], dtype=np.float32)


class ContourVerifier:
    percentiles = (0.5, 0.68, 0.8)

    def __init__(self, initial_image: np.ndarray, initial_box: tuple):
        patch = crop_resize(initial_image, initial_box)
        self.reference_rank = ordinal_image(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
        self.reference_hist = color_histogram(patch)
        self.reference_contours = [
            contour_map(self.reference_rank, percentile) for percentile in self.percentiles
        ]
        self.window = cv2.createHanningWindow((64, 64), cv2.CV_32F)

    def measure(self, image: np.ndarray, box: tuple, confidence: float) -> VerificationResult:
        if not np.isfinite(confidence):
            raise ValueError("Proposer confidence must be finite")
        patch = crop_resize(image, box)
        rank = ordinal_image(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))
        shift, response = cv2.phaseCorrelate(
            self.reference_rank.copy(), rank.copy(), self.window.copy()
        )
        if np.isfinite(shift).all() and max(abs(shift[0]), abs(shift[1])) <= 12 and response > 0.05:
            transform = np.float32([[1, 0, -shift[0]], [0, 1, -shift[1]]])
            aligned = cv2.warpAffine(rank, transform, (64, 64), borderMode=cv2.BORDER_REFLECT_101)
        else:
            aligned = rank
        a, b = self.reference_rank[4:-4, 4:-4].reshape(-1), aligned[4:-4, 4:-4].reshape(-1)
        ac, bc = a - a.mean(), b - b.mean()
        denominator = float(np.linalg.norm(ac) * np.linalg.norm(bc))
        correlation = float(np.dot(ac, bc) / denominator) if denominator > 1e-8 else 0.0
        rank_risk = float(np.clip((1 - correlation) / 2, 0, 1))
        current_hist = color_histogram(patch)
        coefficient = float(np.sqrt(self.reference_hist * current_hist).sum())
        color_risk = float(np.clip(1 - coefficient, 0, 1))
        distances = tuple(
            symmetric_trimmed_distance(reference, contour_map(aligned, percentile))
            for reference, percentile in zip(self.reference_contours, self.percentiles)
        )
        return VerificationResult(
            distances, rank_risk, color_risk,
            float(1 - np.clip(confidence, 0, 1)), float(response),
        )
