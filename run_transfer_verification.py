"""Transfer of the frozen verifier to TrackVes and the controlled benchmark.

The frozen contour percentile and fusion weights are applied without any
target-dataset tuning. Native proposer confidence comes from the confidence-
recording reruns in ``results/with_confidence``; their boxes are checked frame
by frame against the frozen official CSVs before they are used, so the
localization record of the fair-tracking control is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from contour_verifier import ContourVerifier
from evaluate_verification import grouped_metrics, risk_values
from protocol import (
    HERE, canonical_chess, canonical_trackves, paired_statistics, parse_box,
    read_csv, sha256, write_csv, write_json,
)
from run_official_trackers import FrameReader


METHODS = {
    "native_confidence": [0, 0, 0, 1],
    "rank_only": [0, 1, 0, 0],
    "contour_only": [1, 0, 0, 0],
    "color_only": [0, 0, 1, 0],
}


def confidence_rows(root: Path, domain: str, proposer: str) -> tuple[dict, bool]:
    frozen = {(r["sequence"], r["unit"], int(r["frame_idx"])): r for r in read_csv(root / "official" / domain / f"{proposer}.csv")}
    rerun_path = root / "with_confidence" / "official" / domain / f"{proposer}.csv"
    rerun = {(r["sequence"], r["unit"], int(r["frame_idx"])): r for r in read_csv(rerun_path)}
    if set(frozen) != set(rerun):
        raise ValueError(f"Confidence rerun covers a different GT inventory: {domain}/{proposer}")
    identical = True
    for key, row in frozen.items():
        other = rerun[key]
        if parse_box(row, "pred_") != parse_box(other, "pred_"):
            identical = False
            break
    return rerun, identical


def main() -> None:
    root = HERE / "results"
    frozen = json.loads((root / "selected_verification.json").read_text())
    if frozen["protocol_sha256"] != sha256(HERE / "verification_protocol.json"):
        raise ValueError("Transfer evaluation cannot alter the external-validation frozen protocol")
    summaries, repeated_measure_error = {}, []
    appearance = np.asarray(frozen["weights"], dtype=float)
    appearance[3] = 0
    appearance /= appearance.sum()
    for domain, ground_truth in (("trackves", canonical_trackves()), ("chess", canonical_chess())):
        sequence_gt = {}
        for record in ground_truth:
            sequence_gt.setdefault((record.sequence, record.unit), []).append(record)
        summaries[domain] = {}
        for proposer in ("nanotrack", "stark"):
            predictions, boxes_identical = confidence_rows(root, domain, proposer)
            output = []
            for (sequence, unit), records in sorted(sequence_gt.items()):
                reader = FrameReader()
                try:
                    initial = reader.read(records[0])
                    verifier = ContourVerifier(initial, records[0].box)
                    for record in records[1:]:
                        row = predictions[record.key]
                        image = reader.read(record)
                        box = parse_box(row, "pred_")
                        confidence = float(row["native_confidence"]) if row.get("native_confidence", "") != "" else 0.0
                        if box is None:
                            contours, rank, color, native = (1.0, 1.0, 1.0), 1.0, 1.0, 1.0
                        else:
                            measure = verifier.measure(image, box, confidence)
                            contours, rank, color = measure.contour_distance, measure.rank_disagreement, measure.color_disagreement
                            native = measure.native_confidence_risk
                            if not repeated_measure_error:
                                repeat = verifier.measure(image, box, confidence)
                                repeated_measure_error.append(float(np.max(np.abs(measure.vector(2) - repeat.vector(2)))))
                        iou = float(row["bbox_iou"])
                        output.append({
                            "sequence": sequence, "case": sequence, "unit": unit,
                            "frame_idx": record.frame, "proposer": proposer,
                            "box_iou": iou, "failure": int(iou < .5),
                            "contour_50": contours[0], "contour_68": contours[1],
                            "contour_80": contours[2], "rank_risk": rank,
                            "color_risk": color, "native_risk": native,
                        })
                finally:
                    reader.close()
            expected = sum(not record.is_initialization for record in ground_truth)
            if len(output) != expected:
                raise ValueError(f"Transfer GT denominator changed: {domain}/{proposer}")
            scores = {
                **{name: risk_values(output, frozen["percentile"], w) for name, w in METHODS.items()},
                "appearance_fusion": risk_values(output, frozen["percentile"], appearance.tolist()),
                "verification": risk_values(output, frozen["percentile"], frozen["weights"]),
            }
            results, per_sequence = {}, {}
            for name, risk in scores.items():
                groups = grouped_metrics(output, risk, 1.0)
                auc = [g["failure_auroc"] for g in groups.values() if g["failure_auroc"] is not None]
                per_sequence[name] = {s: g["failure_auroc"] for s, g in groups.items()}
                results[name] = {
                    "AUROC_macro": float(np.mean(auc)) if auc else None,
                    "AUROC_sd": float(np.std(auc, ddof=1)) if len(auc) > 1 else None,
                    "AUROC_n_sequences": len(auc), "all_sequences": groups,
                    "no_transfer_threshold_calibration": True,
                }
            paired = {}
            for comparator in ("native_confidence", "rank_only", "contour_only", "color_only", "appearance_fusion"):
                eligible = [s for s in per_sequence["verification"] if per_sequence["verification"][s] is not None and per_sequence[comparator][s] is not None]
                if len(eligible) >= 2:
                    paired[comparator] = paired_statistics(
                        {s: per_sequence["verification"][s] for s in eligible},
                        {s: per_sequence[comparator][s] for s in eligible},
                    )
            write_csv(root / "transfer_verification" / f"{domain}_{proposer}.csv", output)
            summaries[domain][proposer] = {
                "methods": results, "scored_frames": len(output),
                "verification_minus_comparator_auroc": paired,
                "native_confidence_source": "confidence-recording rerun of the identical frozen tracker",
                "rerun_boxes_identical_to_frozen_official_csv": boxes_identical,
                "appearance_weights": appearance.tolist(),
            }
            print(domain, proposer, "boxes identical:", boxes_identical, {
                name: round(values["AUROC_macro"], 3) if values["AUROC_macro"] is not None else None
                for name, values in results.items()
            }, flush=True)
    write_json(root / "transfer_verification_summary.json", {
        "frozen_validation_manifest_sha256": sha256(root / "selected_verification.json"),
        "calibration_dataset": "SurgT validation only",
        "no_transfer_sequence_used_for_parameter_selection": True,
        "four_cue_monitor_with_native_confidence": True,
        "repeated_deterministic_descriptor_max_error": max(repeated_measure_error),
        "domains": summaries,
    })
    print("TRANSFER_WRITTEN")


if __name__ == "__main__":
    main()
