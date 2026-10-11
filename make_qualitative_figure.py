"""Qualitative SurgT test cases: thumbnails with GT/proposal boxes and risk time lines.

Predicted boxes were not retained in the frozen CSVs, so the chosen sequences
are re-tracked with the identical frozen proposer. The recomputed IoU and risk
values are compared frame by frame with the frozen CSV; any mismatch aborts.
Nothing is selected or tuned here.
"""
from __future__ import annotations

import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from contour_verifier import ContourVerifier
from evaluate_verification import risk_values
from protocol import HERE, box_iou, read_csv, write_json
from run_external_verification import DATA, left_view, load_sequence, rectification, scoreable
from run_official_trackers import OfficialTracker, OpenCVBaseline
from supplementary_analyses import episodes


ROOT = HERE / "results"
CASES = (
    ("nanotrack", "case_2_3", "Large gain: native confidence 0.49, fused 1.00"),
    ("nanotrack", "case_1_1", "Gain with false alarms: specificity 0.56"),
    ("nanotrack", "case_5_4", "Weak case: fused AUROC 0.60"),
    ("stark", "case_1_2", "Fused below native confidence: 0.77 vs 0.99"),
)


def retrack(proposer: str, sequence: str, selected: dict, frozen_rows: list[dict]) -> dict:
    case, index = sequence.rsplit("_", 1)
    directory = DATA / "test" / case / index
    info, annotations = load_sequence(directory)
    first = min(i for i, v in annotations.items() if scoreable(v, info))
    maps = rectification(directory, info)
    tracker = OfficialTracker(proposer, "surgt") if proposer == "stark" else OpenCVBaseline("nanotrack")
    capture = cv2.VideoCapture(str(directory / info["name_video"]), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    frozen = {int(r["frame_idx"]): r for r in frozen_rows if r["sequence"] == sequence}
    frames, records, verifier = {}, [], None
    try:
        for i in range(max(annotations) + 1):
            ok, packed = capture.read()
            if not ok:
                raise OSError(f"Cannot decode {sequence}/{i}")
            if i < first:
                continue
            image = left_view(packed, info, maps)
            if i == first:
                box = tuple(float(v) for v in annotations[i][2][0])
                tracker.initialize(image, box)
                verifier = ContourVerifier(image, box)
                frames[i] = (image.copy(), box, box)
                continue
            predicted = tracker.update(image)
            if i not in annotations or not scoreable(annotations[i], info):
                continue
            gt = tuple(float(v) for v in annotations[i][2][0])
            measured = verifier.measure(image, predicted, tracker.last_confidence)
            row = {
                "frame_idx": i, "box_iou": box_iou(gt, predicted),
                "contour_80": measured.contour_distance[2], "rank_risk": measured.rank_disagreement,
                "color_risk": measured.color_disagreement, "native_risk": measured.native_confidence_risk,
            }
            saved = frozen[i]
            for key in ("box_iou", "contour_80", "rank_risk", "color_risk", "native_risk"):
                if abs(float(saved[key]) - row[key]) > 1e-4:
                    raise ValueError(f"Re-tracking does not reproduce the frozen CSV: {sequence} frame {i} {key} {saved[key]} vs {row[key]}")
            records.append(row)
            frames[i] = (image, gt, predicted)
    finally:
        capture.release()
    risks = risk_values(records, selected["percentile"], selected["weights"])
    native = np.asarray([r["native_risk"] for r in records])
    iou = np.asarray([r["box_iou"] for r in records])
    idx = np.asarray([r["frame_idx"] for r in records])
    y = (iou < 0.5).astype(int)
    runs = [r for r in episodes(y) if r[1] - r[0] >= 5]
    picks = [idx[0]]
    if runs:
        start, end = runs[0]
        picks += [idx[start], idx[min(end - 1, start + (end - start) // 2)]]
    picks.append(idx[-1])
    picks = sorted(dict.fromkeys(int(p) for p in picks))
    return {
        "frames": {p: frames[p] for p in picks}, "idx": idx, "iou": iou, "risk": risks, "native": native,
        "first": first, "threshold": selected["alarm_thresholds"][proposer]["verification"],
        "native_threshold": selected["alarm_thresholds"][proposer]["native_confidence"],
    }


def crop_with_boxes(image: np.ndarray, gt: tuple, pred: tuple | None) -> np.ndarray:
    h, w = image.shape[:2]
    boxes = [gt] + ([pred] if pred is not None else [])
    cx = np.mean([b[0] + b[2] / 2 for b in boxes])
    cy = np.mean([b[1] + b[3] / 2 for b in boxes])
    size = max(max(b[2] for b in boxes), max(b[3] for b in boxes)) * 3.2
    size = max(size, 160)
    x1, y1 = int(max(0, cx - size / 2)), int(max(0, cy - size / 2))
    x2, y2 = int(min(w, cx + size / 2)), int(min(h, cy + size / 2))
    view = image[y1:y2, x1:x2].copy()
    thickness = max(2, int(size / 120))
    gx, gy, gw, gh = gt
    cv2.rectangle(view, (int(gx - x1), int(gy - y1)), (int(gx + gw - x1), int(gy + gh - y1)), (80, 200, 80), thickness)
    if pred is not None:
        px, py, pw, ph = pred
        cv2.rectangle(view, (int(px - x1), int(py - y1)), (int(px + pw - x1), int(py + ph - y1)), (40, 60, 230), thickness)
    view = cv2.resize(view, (288, 288), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(view, cv2.COLOR_BGR2RGB)


def main() -> None:
    import torch
    torch.set_num_threads(4)
    cv2.setNumThreads(1)  # identical runtime settings to the frozen external runs
    selected = json.loads((ROOT / "selected_verification.json").read_text())
    rows = {p: read_csv(ROOT / "external_verification" / "test" / f"{p}.csv") for p in ("nanotrack", "stark")}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    fig = plt.figure(figsize=(10.5, 2.75 * len(CASES)), constrained_layout=True)
    grid = fig.add_gridspec(len(CASES), 8)
    manifest = {}
    for row_index, (proposer, sequence, note) in enumerate(CASES):
        data = retrack(proposer, sequence, selected, rows[proposer])
        picks = sorted(data["frames"])
        for k, frame_id in enumerate(picks[:4]):
            axis = fig.add_subplot(grid[row_index, k])
            image, gt, pred = data["frames"][frame_id]
            axis.imshow(crop_with_boxes(image, gt, pred))
            iou_here = box_iou(gt, pred) if pred is not None else 0.0
            axis.set_title(f"frame {frame_id}  IoU {iou_here:.2f}", fontsize=7)
            axis.axis("off")
        axis = fig.add_subplot(grid[row_index, 4:])
        idx, iou, risk, native = data["idx"], data["iou"], data["risk"], data["native"]
        failing = iou < 0.5
        axis.fill_between(idx, 0, 1, where=failing, color="#f2d7d5", step="mid", label="localization failure (IoU<0.5)")
        axis.plot(idx, iou, color="#222222", lw=0.9, label="IoU with GT")
        axis.plot(idx, risk, color="#2f4f6f", lw=1.1, label="fused risk $r_t$")
        axis.plot(idx, native, color="#9a9a9a", lw=0.8, label="native-confidence risk $d_N$")
        axis.axhline(data["threshold"], color="#2f4f6f", ls="--", lw=0.8, label=r"frozen $\tau$ (fused)")
        axis.axhline(0.5, color="#222222", ls=":", lw=0.7)
        alarms = risk > data["threshold"]
        axis.scatter(idx[alarms], np.full(alarms.sum(), 1.03), s=3, color="#c0392b", marker="|", label="warning")
        for frame_id in picks[:4]:
            axis.axvline(frame_id, color="#888888", lw=0.5, ls=":")
        axis.set(xlim=(idx[0], idx[-1]), ylim=(0, 1.08), ylabel="IoU / risk")
        axis.set_title(f"{'NanoTrack' if proposer == 'nanotrack' else 'STARK-ST50'} / {sequence.replace('_', ' ')}: {note}", fontsize=8, loc="left")
        if row_index == len(CASES) - 1:
            axis.set_xlabel("video frame index")
        if row_index == 0:
            axis.legend(fontsize=6, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.32), framealpha=0.9)
        axis.grid(alpha=0.2)
        manifest[f"{proposer}/{sequence}"] = {
            "thumbnail_frames": picks[:4], "scored_frames": int(len(idx)),
            "frozen_csv_reproduced_within_1e-4": True, "note": note,
        }
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / "Verification_Qualitative.pdf", dpi=200)
    plt.close(fig)
    write_json(ROOT / "figures" / "Verification_Qualitative.manifest.json", {
        "cases": manifest, "boxes": "green = supplied ground truth, red = unchanged proposer box",
        "re_tracking": "identical frozen proposer; frozen CSV values reproduced frame by frame",
    })
    print("QUALITATIVE_FIGURE_WRITTEN", manifest)


if __name__ == "__main__":
    main()
