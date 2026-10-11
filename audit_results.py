from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from protocol import (
    BASE_ROOT, CHESS_ROOT, HERE, PROJECT, SEGMENTATION_ROOT,
    align_predictions, canonical_chess, holm_adjust, paired_statistics,
    parse_box, read_csv, sha256, trackeval_data, write_csv, write_json,
)

sys.path.insert(0, str(HERE / "external" / "TrackEval"))
import numpy as _np  # noqa: E402
# The pinned TrackEval commit still uses the removed alias ``np.float`` inside HOTA.
# Restoring the alias is a pure compatibility shim; no metric equation is touched.
if not hasattr(_np, "float"):
    _np.float = float  # type: ignore[attr-defined]
from trackeval.metrics import CLEAR, HOTA  # noqa: E402


def summarize_tracking(rows: list[dict]) -> tuple[dict, dict[str, float], list[dict]]:
    rows = [row for row in rows if not int(row.get("is_initialization", 0))]
    if not rows:
        raise ValueError("No evaluable frames after excluding GT-supplied initialization")
    units: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        units[row["sequence"], row["unit"]].append(row)
    hota = HOTA()
    clear = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False})
    hota_units, clear_units, unit_rows = {}, {}, []
    for key, values in sorted(units.items()):
        data = trackeval_data(values)
        hota_units[str(key)] = hota.eval_sequence(data)
        clear_units[str(key)] = clear.eval_sequence(data)
        valid_errors = [
            float(r["center_error_px"]) for r in values if r["center_error_px"] != ""
        ]
        unit_rows.append({
            "sequence": key[0], "unit": key[1], "n_gt_frames": len(values),
            "n_predictions": sum(int(r["prediction_present"]) for r in values),
            "bbox_iou": float(np.mean([float(r["bbox_iou"]) for r in values])),
            "success_05": float(np.mean([float(r["success_05"]) for r in values])),
            "precision_20": float(np.mean([float(r["precision_20"]) for r in values])),
            "center_error_valid_predictions": (
                float(np.mean(valid_errors)) if valid_errors else None
            ),
            "HOTA": float(np.mean(hota_units[str(key)]["HOTA"])),
            "AssA": float(np.mean(hota_units[str(key)]["AssA"])),
        })
    by_sequence: dict[str, list[dict]] = defaultdict(list)
    for row in unit_rows:
        by_sequence[row["sequence"]].append(row)
    metrics = ("bbox_iou", "success_05", "precision_20", "HOTA", "AssA")
    sequence_values = {
        key: {
            sequence: float(np.mean([r[key] for r in values]))
            for sequence, values in sorted(by_sequence.items())
        }
        for key in metrics
    }
    result = {
        "n_sequences": len(by_sequence), "n_units": len(units),
        "n_gt_frames": len(rows),
        "n_missing_predictions": sum(not int(r["prediction_present"]) for r in rows),
        "aggregation": "equal-unit mean within sequence, equal-sequence macro mean",
        "center_error": "valid predictions only; absent predictions are misses for overlap/precision",
        "sequence_metrics": sequence_values,
    }
    for key, values in sequence_values.items():
        result[key] = {
            "mean": float(np.mean(list(values.values()))),
            "sample_sd": float(np.std(list(values.values()), ddof=1)),
        }
    pooled_hota = hota.combine_sequences(hota_units)
    pooled_clear = clear.combine_sequences(clear_units)
    result["pooled_official_trackeval"] = {
        key: float(np.mean(pooled_hota[key])) for key in ("HOTA", "DetA", "AssA", "LocA")
    } | {key: float(pooled_clear[key]) for key in ("MOTA", "MOTP")}
    return result, sequence_values["bbox_iou"], unit_rows


def audit_chess(output: Path) -> dict:
    gt = canonical_chess()
    write_csv(output / "chess_gt_manifest.csv", [
        {"sequence": r.sequence, "unit": r.unit, "frame_idx": r.frame,
         "image_path": r.image_path, **dict(zip(("gt_x", "gt_y", "gt_w", "gt_h"), r.box))}
        for r in gt
    ])
    target_keys = {r.key for r in gt}
    methods = {
        "NanoTrack": (CHESS_ROOT, "nanotrack", "frozen official OpenCV model"),
        "Ours-Legacy": (CHESS_ROOT, "ours_scheme_c_nano", "legacy; learning provenance not outer-fold-isolated"),
        "CSRT": (BASE_ROOT, "csrt", "OpenCV baseline"),
        "SAM": (BASE_ROOT, "sam", "public segmentation checkpoint with tracking wrapper"),
        "YOLO": (BASE_ROOT, "yolo", "public segmentation checkpoint with tracking wrapper"),
        "Cutie": (BASE_ROOT, "cutie", "public checkpoint with tracking wrapper"),
        "OSTrack-proxy": (BASE_ROOT, "ostrack", "untrained task head; appendix only"),
        "MixFormer-proxy": (BASE_ROOT, "mixformer", "untrained task head; appendix only"),
        "LMTrack-proxy": (BASE_ROOT, "lmtrack", "untrained task head; appendix only"),
        "STARK-proxy": (BASE_ROOT, "stark", "untrained task head; appendix only"),
    }
    summaries = {}
    for name, (root, method, provenance) in methods.items():
        predictions = {}
        source_hashes = {}
        missing_files = []
        outside = 0
        gt_mismatches = 0
        canonical_by_key = {r.key: r.box for r in gt}
        for sequence, unit in sorted({(r.sequence, r.unit) for r in gt}):
            path = (
                root / sequence / f"{method}_rows.csv"
                if unit == "sequence_target" else
                root / sequence / "boards" / unit / f"{method}_rows.csv"
            )
            if not path.is_file():
                missing_files.append(str(path))
                continue
            source_hashes[str(path)] = sha256(path)
            for row in read_csv(path):
                key = sequence, unit, int(row["frame_idx"])
                if key not in target_keys:
                    outside += 1
                    continue
                if key in predictions:
                    raise ValueError(f"Duplicate prediction: {name} {key}")
                row_gt = parse_box(row, "gt_bbox_")
                if row_gt is None or not np.allclose(row_gt, canonical_by_key[key], atol=1e-6):
                    gt_mismatches += 1
                predictions[key] = parse_box(row, "pred_bbox_")
        aligned = align_predictions(gt, predictions)
        summary, _, units = summarize_tracking(aligned)
        summary.update({
            "provenance": provenance, "source_sha256": source_hashes,
            "missing_prediction_files": missing_files,
            "predictions_outside_fixed_protocol_not_scored": outside,
            "source_gt_coordinate_mismatches": gt_mismatches,
            "shared_gt_protocol_verified": gt_mismatches == 0,
        })
        summaries[name] = summary
        write_csv(output / "chess_aligned" / f"{name}.csv", aligned)
        write_csv(output / "chess_units" / f"{name}.csv", units)
        print(f"{name}: all 22 units, missing={summary['n_missing_predictions']}, "
              f"IoU={summary['bbox_iou']['mean']:.6f}", flush=True)
    tests = []
    reference = summaries["Ours-Legacy"]["sequence_metrics"]
    for name in ("NanoTrack", "CSRT", "SAM", "YOLO", "Cutie"):
        for metric in ("bbox_iou", "HOTA", "AssA"):
            tests.append({
                "reference": "Ours-Legacy", "comparator": name, "metric": metric,
                **paired_statistics(reference[metric], summaries[name]["sequence_metrics"][metric]),
            })
    for row, p in zip(tests, holm_adjust(r["p_two_sided"] for r in tests)):
        row["p_holm"] = p
    result = {"methods": summaries, "paired_tests": tests}
    write_json(output / "chess_fair_legacy.json", result)
    return result


def audit_segmentation(output: Path) -> dict:
    batch = SEGMENTATION_ROOT / "eval_contour_CC_outputs_batch" / "archive_soft_tissue" / "summary.csv"
    sources = {
        "Ours-Contour": (
            SEGMENTATION_ROOT / "eval_contour_CC_outputs_filled_from_contour"
            / "archive_soft_tissue" / "summary.csv", "ours_th68_fill"
        ),
        "MedSAM": (
            SEGMENTATION_ROOT / "eval_medsam_outputs_batch" / "archive_soft_tissue"
            / "summary.csv", "medsam_vit_b"
        ),
        "SAM2.1": (batch, "ultra_sam_sam2.1_b"),
        "YOLO11m-seg": (batch, "yolo_yolo11m-seg"),
    }
    universe_rows = [r for r in read_csv(batch) if r["method"] == "ours_th68"]
    universe = {r["rel_path"] for r in universe_rows}
    if len(universe) != 8080 or len(universe_rows) != 8080:
        raise ValueError("Expected unique canonical 8080-frame CholecSeg8k set")
    summaries, sequence_scores, per_video_rows = {}, {}, []
    frame_rows = []
    for name, (path, method) in sources.items():
        records = {}
        for row in read_csv(path):
            if row["method"] != method:
                continue
            key = row["rel_path"]
            if key in records:
                raise ValueError(f"Duplicate segmentation record: {name} {key}")
            records[key] = row
        if set(records) - universe:
            raise ValueError(f"Unexpected segmentation frames for {name}")
        by_video = defaultdict(list)
        missing = 0
        for key in sorted(universe):
            row = records.get(key)
            present = row is not None and row["has_mask"] == "1"
            if present:
                dice, iou = float(row["dice"]), float(row["iou"])
                if not (0 <= dice <= 1 and 0 <= iou <= 1):
                    raise ValueError(f"Invalid segmentation metric: {name} {key}")
            else:
                dice = iou = 0.0
                missing += 1
            video = key.replace("\\", "/").split("/")[0]
            by_video[video].append((dice, iou))
            frame_rows.append({
                "method": name, "video": video, "rel_path": key,
                "dice": dice, "iou": iou, "mask_present": int(present),
            })
        scores = {
            metric: {
                video: float(np.mean(np.asarray(values)[:, index]))
                for video, values in sorted(by_video.items())
            }
            for index, metric in enumerate(("dice", "iou"))
        }
        if len(by_video) != 17:
            raise ValueError("Expected 17 video groups; patient mapping is not inferred")
        sequence_scores[name] = scores
        summaries[name] = {
            "n_frames": 8080, "n_video_groups": 17, "missing_masks": missing,
            "source_sha256": sha256(path),
            **{
                metric: {
                    "video_macro_mean": float(np.mean(list(values.values()))),
                    "video_sample_sd": float(np.std(list(values.values()), ddof=1)),
                    "frame_mean": float(np.mean([x[index] for v in by_video.values() for x in v])),
                }
                for index, (metric, values) in enumerate(scores.items())
            },
        }
        for video in sorted(by_video):
            per_video_rows.append({
                "method": name, "video": video, "n_frames": len(by_video[video]),
                "dice": scores["dice"][video], "iou": scores["iou"][video],
            })
    tests = []
    for name in ("MedSAM", "SAM2.1", "YOLO11m-seg"):
        for metric in ("dice", "iou"):
            tests.append({
                "reference": "Ours-Contour", "comparator": name, "metric": metric,
                **paired_statistics(sequence_scores["Ours-Contour"][metric], sequence_scores[name][metric]),
            })
    for row, p in zip(tests, holm_adjust(t["p_two_sided"] for t in tests)):
        row["p_holm"] = p
    result = {
        "grouping": "17 video IDs, not verified patient identities",
        "primary_summary": "equal-video macro mean; missing masks scored zero",
        "confidence_intervals": "paired video-cluster percentile bootstrap",
        "multiple_comparisons": "Holm over six predeclared comparator-by-metric tests",
        "methods": summaries, "paired_tests": tests,
    }
    write_csv(output / "cholecseg_per_video.csv", per_video_rows)
    write_csv(output / "cholecseg_aligned_frames.csv", frame_rows)
    write_json(output / "cholecseg_grouped.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    audit_chess(args.output)
    result = audit_segmentation(args.output)
    print("GROUPED CHOLECSEG RESULTS", result["methods"], flush=True)
    print("GROUPED CHOLECSEG TESTS", result["paired_tests"], flush=True)


if __name__ == "__main__":
    main()
