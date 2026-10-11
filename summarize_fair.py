from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_results import summarize_tracking
from protocol import (
    HERE, align_predictions, canonical_chess, canonical_trackves, holm_adjust,
    paired_statistics, parse_box, read_csv, sha256, validate_split, write_csv, write_json,
)


def load_aligned(paths: list[Path], gt) -> list[dict]:
    predictions, initialization, given_gt = {}, {}, {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"A complete run is required, not a partial result: {path}")
        for row in read_csv(path):
            key = row["sequence"], row["unit"], int(row["frame_idx"])
            if key in predictions:
                raise ValueError(f"Duplicate output key: {path} {key}")
            predictions[key] = parse_box(row, "pred_")
            initialization[key] = int(row["is_initialization"])
            given_gt[key] = parse_box(row, "gt_")
    target = {record.key: record for record in gt}
    if set(predictions) != set(target):
        raise ValueError("Run omitted GT records or introduced unscored records")
    for key, record in target.items():
        if initialization[key] != int(record.is_initialization):
            raise ValueError(f"Initialization role differs across methods: {key}")
        if given_gt[key] is None or not np.allclose(given_gt[key], record.box, atol=1e-6):
            raise ValueError(f"Run used different GT coordinates: {key}")
    return align_predictions(gt, predictions)


def run(root: Path) -> dict:
    results = {}
    frozen_hashes = json.loads(
        (root / "frozen_trackves_frames" / "manifest.json").read_text(encoding="utf-8")
    )["input_frame_sha256"]
    for domain, gt in (("chess", canonical_chess()), ("trackves", canonical_trackves())):
        summaries = {}
        methods = ("nanotrack", "ostrack", "stark", "csrt")
        if domain == "trackves":
            methods += ("learnedgate",)
        for name in methods:
            path = root / "official" / domain / f"{name}.csv"
            rows = load_aligned([path], gt)
            manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            if domain == "trackves" and manifest["input_frame_sha256"] != frozen_hashes:
                raise ValueError(f"Decoded input frames were not exactly identical: {name}")
            summary, _, _ = summarize_tracking(rows)
            summary["output_sha256"] = sha256(path)
            summaries[name] = summary
        if domain == "trackves":
            ungated = load_aligned(
                [root / "official" / domain / "gate_ungated_nano.csv"], gt
            )
            nano = load_aligned([root / "official" / domain / "nanotrack.csv"], gt)
            for a, b in zip(ungated, nano):
                if any(a[key] != b[key] for key in ("pred_x", "pred_y", "pred_w", "pred_h")):
                    raise ValueError("Gate and baseline used different underlying NanoTrack proposals")
        sequences = sorted({record.sequence for record in gt})
        paths = [root / "disjoint" / domain / f"{name}.csv" for name in sequences]
        rows = load_aligned(paths, gt)
        summaries["ours_disjoint"] = summarize_tracking(rows)[0]
        ours_hashes = {}
        for sequence in sequences:
            manifest_path = root / "disjoint" / domain / f"{sequence}.manifest.json"
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
            test = f"{domain}:{sequence}"
            validate_split(metadata["training_sequences"], metadata["validation_sequences"], [test])
            checkpoint = root / "contour_folds" / f"{domain}_{sequence}.pt"
            if sha256(checkpoint) != metadata["checkpoint_sha256"]:
                raise ValueError(f"Evaluation checkpoint changed: {test}")
            if not metadata["fold_checkpoint_exactly_verified_at_inference"]:
                raise ValueError(f"Unverified inference checkpoint: {test}")
            if domain == "trackves":
                ours_hashes.update(metadata["input_frame_sha256"])
        if domain == "trackves" and ours_hashes != frozen_hashes:
            raise ValueError("Disjoint contour inference used different input images")
        tests = []
        references = ["ours_disjoint"] + (["learnedgate"] if domain == "trackves" else [])
        for reference in references:
            for comparator in ("nanotrack", "ostrack", "stark", "csrt"):
                for metric in ("bbox_iou", "HOTA", "AssA"):
                    tests.append({
                        "reference": reference, "comparator": comparator, "metric": metric,
                        **paired_statistics(
                            summaries[reference]["sequence_metrics"][metric],
                            summaries[comparator]["sequence_metrics"][metric],
                        ),
                    })
        corrected = holm_adjust(row["p_two_sided"] for row in tests)
        for row, p in zip(tests, corrected):
            row["p_holm"] = p
        results[domain] = {
            "methods": summaries, "paired_tests": tests,
            "GT_records": len(gt), "initialization_frames": sum(r.is_initialization for r in gt),
            "scored_frames": sum(not r.is_initialization for r in gt),
            "canonical_units": len({(r.sequence, r.unit) for r in gt}),
            "units_with_post_initialization_observations": summaries["nanotrack"]["n_units"],
            "missing_prediction_policy": "zero overlap/precision and TrackEval false negatives",
            "Holm_family": "all declared reference-by-comparator-by-metric tests within this dataset",
        }
        sequence_rows = []
        for name, values in summaries.items():
            for sequence in sequences:
                sequence_rows.append({
                    "method": name, "sequence": sequence,
                    **{metric: values["sequence_metrics"][metric][sequence]
                       for metric in ("bbox_iou", "success_05", "precision_20", "HOTA", "AssA")},
                })
        write_csv(root / f"{domain}_fair_per_sequence.csv", sequence_rows)
    write_json(root / "fair_tracking_summary.json", results)
    for domain, data in results.items():
        print(domain.upper())
        for name, value in data["methods"].items():
            print(name, value["bbox_iou"], value["pooled_official_trackeval"], flush=True)
    return results


if __name__ == "__main__":
    run(HERE / "results")
