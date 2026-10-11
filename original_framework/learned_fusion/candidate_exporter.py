"""Build a unified, GT-supervised candidate dataset without changing trackers.

Inference features never include GT. GT is stored separately and is used only
by the offline trainer/evaluator. Candidate order is fixed:
  fusion_out, nano_raw, hist_mis, previous, blend25, blend50, blend75.
"""
from __future__ import annotations

import ast
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


CANDIDATE_NAMES = (
    "fusion_out", "nano_raw", "hist_mis", "previous",
    "blend25", "blend50", "blend75",
)
FEATURE_NAMES = (
    "dx", "dy", "log_w", "log_h", "source_final", "source_nano",
    "source_hist", "source_previous", "score", "nano_score", "hist_score",
    "contour_score", "pseudo3d_score", "size_score", "temporal_score", "risk", "reliability",
    "iou_nano_mis", "motion", "area_ratio", "domain_trackves", "available",
)


def _box(value: Any) -> np.ndarray | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    a = np.asarray(value, dtype=np.float32).reshape(-1)
    if a.size != 4 or not np.all(np.isfinite(a)) or a[2] <= 0 or a[3] <= 0:
        return None
    return a


def _safe(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if np.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def _blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    return ((1.0 - alpha) * a + alpha * b).astype(np.float32)


def _normalize_box(box: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    x, y, w, h = box
    ax, ay, aw, ah = anchor
    return np.asarray(
        [(x - ax) / max(aw, 1.0), (y - ay) / max(ah, 1.0),
         np.log(max(w, 1.0) / max(aw, 1.0)),
         np.log(max(h, 1.0) / max(ah, 1.0))],
        dtype=np.float32,
    )


def _candidate_scores(debug: dict[str, Any], name: str) -> dict[str, float]:
    arb = debug.get("nano_output_arbitrate") or {}
    for item in arb.get("candidates") or []:
        if item.get("name") == name:
            return {k: _safe(item.get(k)) for k in ("score", "nano", "hist", "sim", "sz", "temporal")}
    return {}


def _token(
    box: np.ndarray,
    anchor: np.ndarray,
    source: str,
    scores: dict[str, float],
    global_values: dict[str, float],
    domain_trackves: float,
    available: float = 1.0,
) -> np.ndarray:
    geom = _normalize_box(box, anchor)
    source_flags = [
        float(source == "final"), float(source == "nano"),
        float(source == "hist"), float(source == "previous"),
    ]
    return np.asarray(
        [
            *geom, *source_flags,
            scores.get("score", 0.0), scores.get("nano", 0.0),
            scores.get("hist", 0.0), scores.get("sim", 0.0),
            global_values.get("pseudo3d", 0.0),
            scores.get("sz", 0.0), scores.get("temporal", 0.0),
            global_values.get("risk", 0.0), global_values.get("reliability", 0.0),
            global_values.get("iou_nano_mis", 0.0), global_values.get("motion", 0.0),
            float((box[2] * box[3]) / max(anchor[2] * anchor[3], 1.0)),
            domain_trackves, available,
        ],
        dtype=np.float32,
    )


@dataclass
class Sample:
    domain: str
    sequence: str
    unit: str
    frame_idx: int
    source_path: str
    features: np.ndarray
    boxes: np.ndarray
    gt: np.ndarray
    base_index: int = 0
    nano_index: int = 1


def _assemble(
    *,
    domain: str,
    sequence: str,
    unit: str,
    frame_idx: int,
    source_path: str,
    final: np.ndarray,
    nano: np.ndarray,
    hist: np.ndarray,
    previous: np.ndarray,
    gt: np.ndarray,
    final_scores: dict[str, float],
    nano_scores: dict[str, float],
    hist_scores: dict[str, float],
    global_values: dict[str, float],
) -> Sample:
    boxes = [
        final, nano, hist, previous,
        _blend(nano, hist, 0.25), _blend(nano, hist, 0.50), _blend(nano, hist, 0.75),
    ]
    score_mix = lambda a: {
        k: (1.0 - a) * nano_scores.get(k, 0.0) + a * hist_scores.get(k, 0.0)
        for k in ("score", "nano", "hist", "sim", "sz", "temporal")
    }
    scores = [
        final_scores, nano_scores, hist_scores,
        {"temporal": 1.0},
        score_mix(0.25), score_mix(0.50), score_mix(0.75),
    ]
    sources = ["final", "nano", "hist", "previous", "blend", "blend", "blend"]
    anchor = previous
    features = np.stack([
        _token(b, anchor, s, sc, global_values, float(domain == "trackves"))
        for b, s, sc in zip(boxes, sources, scores)
    ])
    return Sample(
        domain=domain, sequence=sequence, unit=unit, frame_idx=frame_idx,
        source_path=source_path, features=features, boxes=np.stack(boxes), gt=gt,
    )


def load_trackves(intermediate_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for seq_dir in sorted(intermediate_root.iterdir()):
        frames_dir = seq_dir / "ours_scheme_c_nano"
        if not frames_dir.is_dir():
            continue
        previous: np.ndarray | None = None
        for path in sorted(frames_dir.glob("frame_*.json")):
            debug = json.loads(path.read_text(encoding="utf-8"))
            final, nano, hist, gt = map(_box, (
                debug.get("final_bbox"), debug.get("nano_bbox"),
                debug.get("mis_bbox"), debug.get("gt_bbox"),
            ))
            if final is None or nano is None or hist is None or gt is None:
                continue
            previous = final.copy() if previous is None else previous
            final_sc = _candidate_scores(debug, "fusion_out") or {
                "score": _safe(debug.get("final_score")),
                "hist": _safe(debug.get("hist_sim")),
                "sim": _safe(debug.get("final_score")),
                "sz": _safe(debug.get("sz_sim")),
            }
            nano_sc = _candidate_scores(debug, "nano_raw") or {
                "score": _safe(debug.get("nano_score")),
                "nano": _safe(debug.get("nano_score")),
                "hist": _safe(debug.get("hist_sim")),
                "sz": _safe(debug.get("sz_sim")),
            }
            hist_sc = {
                "score": _safe(debug.get("final_score")),
                "hist": _safe(debug.get("hist_sim")),
                "sim": _safe(debug.get("final_score")),
                "sz": _safe(debug.get("sz_sim")),
                "temporal": _candidate_scores(debug, "fusion_out").get("temporal", 0.0),
            }
            rel = np.mean([
                _safe(debug.get("rel_hist")), _safe(debug.get("rel_sz")),
                _safe(debug.get("rel_nano")),
            ])
            global_values = {
                "risk": _safe(debug.get("risk_ema", debug.get("risk_score"))),
                "pseudo3d": _safe(debug.get(
                    "pseudo3d_score",
                    debug.get("uvh_score", debug.get("render_score")),
                )),
                "reliability": float(rel),
                "iou_nano_mis": _safe(debug.get("iou_mn"), box_iou(nano, hist)),
                "motion": float(np.linalg.norm(
                    (nano[:2] + nano[2:] / 2) - (previous[:2] + previous[2:] / 2)
                ) / max(np.hypot(previous[2], previous[3]), 1.0)),
            }
            samples.append(_assemble(
                domain="trackves", sequence=seq_dir.name, unit=seq_dir.name,
                frame_idx=int(debug.get("frame_idx", 0)), final=final, nano=nano,
                source_path="",
                hist=hist, previous=previous, gt=gt, final_scores=final_sc,
                nano_scores=nano_sc, hist_scores=hist_sc,
                global_values=global_values,
            ))
            previous = final.copy()
    return samples


def _read_csv(path: Path) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {int(float(r["frame_idx"])): r for r in csv.DictReader(f)}


def _chess_hist_path(hist_root: Path, rel: Path) -> Path | None:
    p = hist_root / rel
    if p.is_file():
        return p
    # Historical HistAB key is ours_hist_ab.
    p = p.with_name("ours_hist_ab_rows.csv")
    return p if p.is_file() else None


def load_chess(ours_root: Path, hist_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    ours_files = sorted(ours_root.glob("**/ours_scheme_c_nano_rows.csv"))
    for ours_path in ours_files:
        rel = ours_path.relative_to(ours_root)
        if "unknown_from_threshold" in rel.parts:
            continue
        seq = rel.parts[0]
        # Avoid duplicate top-level files when board-level files exist.
        if "boards" not in rel.parts and (ours_root / seq / "boards").is_dir():
            continue
        unit = "/".join(rel.parts[:-1])
        nano_path = ours_path.with_name("nanotrack_rows.csv")
        hist_path = _chess_hist_path(hist_root, rel)
        ours_rows = _read_csv(ours_path)
        nano_rows = _read_csv(nano_path) if nano_path.is_file() else {}
        hist_rows = _read_csv(hist_path) if hist_path else {}
        previous: np.ndarray | None = None
        for frame_idx in sorted(ours_rows):
            row = ours_rows[frame_idx]
            nrow = nano_rows.get(frame_idx, {})
            hrow = hist_rows.get(frame_idx, {})
            final = _box([row.get(f"pred_bbox_{k}") for k in "xywh"])
            nano = _box([nrow.get(f"pred_bbox_{k}") for k in "xywh"])
            hist = _box([hrow.get(f"pred_bbox_{k}") for k in "xywh"])
            nano = final if nano is None else nano
            hist = final if hist is None else hist
            gt = _box([row.get(f"gt_bbox_{k}") for k in "xywh"])
            if final is None or nano is None or hist is None or gt is None:
                continue
            previous = final.copy() if previous is None else previous
            fs = {"score": _safe(row.get("score")), "sim": _safe(row.get("score"))}
            ns = {"score": _safe(nrow.get("score")), "nano": _safe(nrow.get("score"))}
            hs = {"score": _safe(hrow.get("score")), "hist": _safe(hrow.get("score"))}
            globals_ = {
                "risk": 0.0, "pseudo3d": 0.0,
                "reliability": max(fs["score"], ns["score"], hs["score"]),
                "iou_nano_mis": box_iou(nano, hist),
                "motion": float(np.linalg.norm(
                    (nano[:2] + nano[2:] / 2) - (previous[:2] + previous[2:] / 2)
                ) / max(np.hypot(previous[2], previous[3]), 1.0)),
            }
            samples.append(_assemble(
                domain="chess", sequence=seq, unit=unit, frame_idx=frame_idx,
                source_path=str(row.get("image_path", "")),
                final=final, nano=nano, hist=hist, previous=previous, gt=gt,
                final_scores=fs, nano_scores=ns, hist_scores=hs,
                global_values=globals_,
            ))
            previous = final.copy()
    return samples


def save_npz(samples: Iterable[Sample], path: Path) -> None:
    samples = list(samples)
    if not samples:
        raise RuntimeError("No candidate samples were built")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.stack([s.features for s in samples]).astype(np.float32),
        boxes=np.stack([s.boxes for s in samples]).astype(np.float32),
        gt=np.stack([s.gt for s in samples]).astype(np.float32),
        domain=np.asarray([s.domain for s in samples]),
        sequence=np.asarray([s.sequence for s in samples]),
        unit=np.asarray([s.unit for s in samples]),
        frame_idx=np.asarray([s.frame_idx for s in samples], dtype=np.int32),
        source_path=np.asarray([s.source_path for s in samples]),
        candidate_names=np.asarray(CANDIDATE_NAMES),
        feature_names=np.asarray(FEATURE_NAMES),
    )


def build_from_config(config_path: Path) -> Path:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = config_path.parent.parent
    paths = cfg["paths"]
    trackves = load_trackves(Path(paths["trackves_intermediate"]))
    chess = load_chess(Path(paths["chess_ours_root"]), Path(paths["chess_hist_root"]))
    out = root / paths["artifacts"] / "candidate_dataset.npz"
    save_npz([*trackves, *chess], out)
    manifest = {
        "trackves_samples": len(trackves), "chess_samples": len(chess),
        "trackves_sequences": sorted({s.sequence for s in trackves}),
        "chess_sequences": sorted({s.sequence for s in chess}),
        "candidate_names": CANDIDATE_NAMES, "feature_names": FEATURE_NAMES,
        "gt_in_features": False,
    }
    (out.parent / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args()
    print(build_from_config(Path(args.config).resolve()))
