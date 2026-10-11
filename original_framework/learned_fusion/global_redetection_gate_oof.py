"""Strict sequence-LOSO safety gate for Nano/global re-detection candidates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from candidate_exporter import box_iou


def _relative(box: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    x, y, w, h = box.T
    ax, ay, aw, ah = anchor.T
    return np.stack([
        (x - ax) / np.maximum(aw, 1.0),
        (y - ay) / np.maximum(ah, 1.0),
        np.log(np.maximum(w, 1.0) / np.maximum(aw, 1.0)),
        np.log(np.maximum(h, 1.0) / np.maximum(ah, 1.0)),
    ], axis=1)


def _iou_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.asarray([box_iou(x, y) for x, y in zip(a, b)], dtype=np.float32)


def _features(
    boxes: np.ndarray, scores: np.ndarray, nano_score: np.ndarray,
    frame_idx: np.ndarray,
) -> np.ndarray:
    nano, template, orb = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    agreements = np.stack([
        _iou_vector(nano, template), _iou_vector(nano, orb),
        _iou_vector(template, orb),
    ], axis=1)
    return np.concatenate([
        nano_score[:, None], scores[:, 1:3], agreements,
        _relative(template, nano), _relative(orb, nano),
        (scores[:, 1] * nano_score)[:, None],
        (scores[:, 2] * nano_score)[:, None],
        (np.log1p(frame_idx) / 10.0)[:, None],
    ], axis=1).astype(np.float32)


def _balanced_weights(sequence: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(sequence), dtype=np.float32)
    for name in np.unique(sequence):
        mask = sequence == name
        weights[mask] = 1.0 / max(int(mask.sum()), 1)
    return weights * len(weights) / max(float(weights.sum()), 1e-8)


def _fit_predict(
    x: np.ndarray, targets: np.ndarray, train: np.ndarray, test: np.ndarray,
    sequence: np.ndarray,
) -> np.ndarray:
    prediction = np.zeros((int(test.sum()), targets.shape[1]), dtype=np.float32)
    weights = _balanced_weights(sequence[train])
    for candidate in range(targets.shape[1]):
        model = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=180, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=0.2, random_state=20260718,
        )
        model.fit(x[train], targets[train, candidate], sample_weight=weights)
        prediction[:, candidate] = model.predict(x[test])
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/candidate_dataset.npz"))
    parser.add_argument("--redetection", type=Path, default=Path("artifacts/global_redetection_all.npz"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/global_redetection_gate_oof.csv"))
    args = parser.parse_args()
    data = np.load(args.dataset, allow_pickle=False)
    redetection = np.load(args.redetection, allow_pickle=False)
    indices = redetection["indices"].astype(int)
    boxes = redetection["boxes"].astype(np.float32)
    scores = redetection["scores"].astype(np.float32)
    sequence = data["sequence"][indices].astype(str)
    gt = data["gt"][indices]
    frame_idx = data["frame_idx"][indices].astype(np.float32)
    nano_score = data["features"][indices, 1, 9].astype(np.float32)
    x = _features(boxes, scores, nano_score, frame_idx)
    targets = np.stack([
        _iou_vector(boxes[:, candidate], gt) for candidate in range(3)
    ], axis=1)
    sequences = sorted(np.unique(sequence))
    selected = np.zeros(len(indices), dtype=int)
    predicted_quality = np.zeros_like(targets)
    thresholds: dict[str, float] = {}
    for fold, test_sequence in enumerate(sequences):
        validation_sequence = sequences[(fold + 1) % len(sequences)]
        test = sequence == test_sequence
        validation = sequence == validation_sequence
        train = ~(test | validation)
        val_prediction = _fit_predict(x, targets, train, validation, sequence)
        test_prediction = _fit_predict(x, targets, train, test, sequence)
        best_threshold, best_score = 0.0, -1.0
        for threshold in np.linspace(0.0, 0.35, 36):
            best = np.argmax(val_prediction, axis=1)
            gain = val_prediction[np.arange(len(best)), best] - val_prediction[:, 0]
            choice = np.where(gain > threshold, best, 0)
            score = float(np.mean(targets[validation][np.arange(len(choice)), choice]))
            if score > best_score:
                best_score, best_threshold = score, float(threshold)
        best = np.argmax(test_prediction, axis=1)
        gain = test_prediction[np.arange(len(best)), best] - test_prediction[:, 0]
        selected[test] = np.where(gain > best_threshold, best, 0)
        predicted_quality[test] = test_prediction
        thresholds[test_sequence] = best_threshold
    chosen_iou = targets[np.arange(len(targets)), selected]
    rows = []
    for position, index in enumerate(indices):
        rows.append({
            "dataset_index": int(index), "sequence": sequence[position],
            "frame_idx": int(frame_idx[position]), "selected": int(selected[position]),
            "nano_iou": float(targets[position, 0]),
            "template_iou": float(targets[position, 1]),
            "orb_iou": float(targets[position, 2]),
            "selected_iou": float(chosen_iou[position]),
            "template_score": float(scores[position, 1]),
            "orb_score": float(scores[position, 2]),
            "predicted_quality": json.dumps(predicted_quality[position].tolist()),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for name in sequences:
        mask = sequence == name
        summary.append({
            "sequence": name, "threshold": thresholds[name],
            "nano_iou": float(targets[mask, 0].mean()),
            "gated_iou": float(chosen_iou[mask].mean()),
            "template_rate": float(np.mean(selected[mask] == 1)),
            "orb_rate": float(np.mean(selected[mask] == 2)),
        })
    report = {
        "protocol": "strict sequence LOSO; next sequence validation",
        "macro_nano_iou": float(np.mean([row["nano_iou"] for row in summary])),
        "macro_gated_iou": float(np.mean([row["gated_iou"] for row in summary])),
        "sequences": summary,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
