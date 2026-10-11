from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from protocol import HERE, holm_adjust, paired_statistics, read_csv, sha256, write_csv, write_json


PROPOSERS = ("stark", "nanotrack")


def risk_values(rows: list[dict], percentile: float, weights: list[float]) -> np.ndarray:
    column = f"contour_{round(percentile * 100)}"
    x = np.asarray([
        [float(row[column]), float(row["rank_risk"]),
         float(row["color_risk"]), float(row["native_risk"])]
        for row in rows
    ])
    w = np.asarray(weights, dtype=float)
    if w.shape != (4,) or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError("Risk fusion weights must be nonnegative, finite and sum to one")
    return x @ w


def area_under_risk_coverage(y: np.ndarray, score: np.ndarray) -> float:
    if not len(y) or len(y) != len(score):
        raise ValueError("Risk-coverage requires nonempty matched observations")
    order = np.argsort(score, kind="stable")
    cumulative = np.cumsum(y[order])
    thresholds = np.flatnonzero(np.r_[score[order][1:] != score[order][:-1], True]) + 1
    risk = cumulative[thresholds - 1] / thresholds
    coverage = thresholds / len(y)
    return float(np.trapezoid(np.r_[risk[0], risk], np.r_[0.0, coverage]))


def metrics(rows: list[dict], risks: np.ndarray, threshold: float) -> dict:
    y = np.asarray([int(row["failure"]) for row in rows], dtype=int)
    alarm = risks > threshold
    failed, reliable = int(y.sum()), int((y == 0).sum())
    return {
        "n_frames": len(rows), "n_failed": failed, "n_reliable": reliable,
        "failure_prevalence": float(y.mean()),
        "failure_auroc": float(roc_auc_score(y, risks)) if failed and reliable else None,
        "failure_average_precision": float(average_precision_score(y, risks)) if failed else None,
        "sensitivity": float(alarm[y == 1].mean()) if failed else None,
        "specificity": float((~alarm[y == 0]).mean()) if reliable else None,
        "alarm_rate": float(alarm.mean()),
        "risk_coverage_area": area_under_risk_coverage(y, risks),
        "box_iou_unchanged": float(np.mean([float(row["box_iou"]) for row in rows])),
    }


def grouped_metrics(rows: list[dict], risks: np.ndarray, threshold: float) -> dict[str, dict]:
    positions = defaultdict(list)
    for index, row in enumerate(rows):
        positions[row["sequence"]].append(index)
    return {
        name: {
            **metrics([rows[i] for i in indices], risks[indices], threshold),
            "case": rows[indices[0]]["case"],
        }
        for name, indices in sorted(positions.items())
    }


def validate_shared_inputs(root: Path, split: str) -> dict:
    manifests = {
        name: json.loads((root / "external_verification" / split / f"{name}.manifest.json").read_text())
        for name in PROPOSERS
    }
    reference = manifests["stark"]
    reference_frames = {
        record["sequence"]: record["input_frame_sha256"] for record in reference["sequences"]
    }
    for name, manifest in manifests.items():
        hashes = {record["sequence"]: record["input_frame_sha256"] for record in manifest["sequences"]}
        if hashes != reference_frames:
            raise ValueError(f"Input pixel mismatch in external {split} run: {name}")
        if not manifest["proposer_box_unchanged"]:
            raise ValueError("Reliability monitoring must not rewrite proposer boxes")
    return manifests


def select(root: Path) -> dict:
    output = root / "selected_verification.json"
    if output.exists():
        raise FileExistsError("Validation selection is already frozen; do not silently retune it")
    config_path = HERE / "verification_protocol.json"
    config = json.loads(config_path.read_text())
    validate_shared_inputs(root, "validation")
    rows = {
        name: read_csv(root / "external_verification" / "validation" / f"{name}.csv")
        for name in PROPOSERS
    }
    candidates = []
    for percentile in config["contour_percentile_candidates"]:
        for name, weights in config["fusion_candidates"].items():
            per_proposer = {}
            for proposer in PROPOSERS:
                risks = risk_values(rows[proposer], percentile, weights)
                groups = grouped_metrics(rows[proposer], risks, .5)
                values = [v["failure_auroc"] for v in groups.values() if v["failure_auroc"] is not None]
                if not values:
                    raise ValueError("No validation sequences contain both localization classes")
                per_proposer[proposer] = float(np.mean(values))
            candidates.append({
                "percentile": percentile, "fusion_name": name, "weights": weights,
                "validation_auroc_by_proposer": per_proposer,
                "objective": float(np.mean(list(per_proposer.values()))),
            })
    chosen = max(candidates, key=lambda c: c["objective"])
    thresholds = {}
    comparators = {
        "native_confidence": [0.0, 0.0, 0.0, 1.0],
        "rank_only": [0.0, 1.0, 0.0, 0.0],
        "contour_only": [1.0, 0.0, 0.0, 0.0],
        "verification": chosen["weights"],
    }
    for proposer in PROPOSERS:
        labels = np.asarray([int(row["failure"]) for row in rows[proposer]])
        if not np.any(labels == 0):
            raise ValueError(f"No reliable validation frames for alarm calibration: {proposer}")
        thresholds[proposer] = {
            name: float(np.quantile(
                risk_values(rows[proposer], chosen["percentile"], weights)[labels == 0], .9
            ))
            for name, weights in comparators.items()
        }
    payload = {
        **chosen, "protocol_sha256": sha256(config_path),
        "comparators": comparators, "alarm_thresholds": thresholds,
        "selection_split": "SurgT supplied validation partition",
        "test_labels_used_for_selection": False,
        "test_predictions_used_for_selection": False,
        "validation_source_sha256": {
            name: sha256(root / "external_verification" / "validation" / f"{name}.csv")
            for name in PROPOSERS
        },
        "all_validation_candidates": candidates,
    }
    write_json(output, payload)
    print("VALIDATION_ONLY_SELECTED", json.dumps(chosen), flush=True)
    print("VALIDATION_ALARM_THRESHOLDS", thresholds, flush=True)
    return payload


def evaluate(root: Path) -> dict:
    freeze = root / "selected_verification.json"
    config = json.loads(freeze.read_text())
    if config["protocol_sha256"] != sha256(HERE / "verification_protocol.json"):
        raise ValueError("Protocol changed after validation freeze")
    manifests = validate_shared_inputs(root, "test")
    summary, tests, table_rows = {}, [], []
    for proposer in PROPOSERS:
        rows = read_csv(root / "external_verification" / "test" / f"{proposer}.csv")
        results = {}
        for name, weights in config["comparators"].items():
            risks = risk_values(rows, config["percentile"], weights)
            threshold = config["alarm_thresholds"][proposer][name]
            groups = grouped_metrics(rows, risks, threshold)
            aggregate = {}
            for metric in (
                "failure_auroc", "failure_average_precision", "sensitivity",
                "specificity", "alarm_rate", "risk_coverage_area",
            ):
                available = [g[metric] for g in groups.values() if g[metric] is not None]
                aggregate[metric] = {
                    "mean": float(np.mean(available)) if available else None,
                    "sample_sd": float(np.std(available, ddof=1)) if len(available) > 1 else None,
                    "n_sequences": len(available),
                }
            results[name] = {
                "threshold_frozen_from_validation": threshold,
                "pooled": metrics(rows, risks, threshold),
                "sequence_macro": aggregate, "sequences": groups,
            }
            table_rows.extend([
                {"proposer": proposer, "method": name, "sequence": sequence, **values}
                for sequence, values in groups.items()
            ])
        reference = results["verification"]["sequences"]
        for comparator in ("native_confidence", "rank_only", "contour_only"):
            other = results[comparator]["sequences"]
            for metric in ("failure_auroc", "risk_coverage_area"):
                eligible = [n for n in reference if reference[n][metric] is not None and other[n][metric] is not None]
                if len(eligible) < 2:
                    raise ValueError(f"Insufficient external sequences for paired {metric}")
                effect = paired_statistics(
                    {n: reference[n][metric] for n in eligible},
                    {n: other[n][metric] for n in eligible},
                )
                cases = sorted({reference[n]["case"] for n in eligible})
                case_effect = {
                    case: float(np.mean([
                        reference[n][metric] - other[n][metric]
                        for n in eligible if reference[n]["case"] == case
                    ]))
                    for case in cases
                }
                cluster = paired_statistics(
                    case_effect, {case: 0.0 for case in cases}, seed=20261012
                ) if len(cases) >= 2 else None
                tests.append({
                    "proposer": proposer, "comparator": comparator, "metric": metric,
                    **effect, "case_cluster_sensitivity": cluster,
                    "positive_favors_verification": metric == "failure_auroc",
                })
        metadata = manifests[proposer]["sequences"]
        summary[proposer] = {
            "methods": results, "sequence_count": len(reference),
            "case_count": len({v["case"] for v in reference.values()}),
            "scored_frames": len(rows),
            "mean_proposer_ms_per_update": (
                sum(v["proposer_time_s"] for v in metadata)
                / sum(v["post_initialization_updates"] for v in metadata) * 1000
            ),
            "mean_monitor_ms_per_scored_frame": sum(v["monitor_time_s"] for v in metadata) / len(rows) * 1000,
        }
    for row, adjusted in zip(tests, holm_adjust(row["p_two_sided"] for row in tests)):
        row["p_holm"] = adjusted
    result = {
        "protocol_sha256": config["protocol_sha256"],
        "selection_manifest_sha256": sha256(freeze),
        "parameter_selection": "validation only, frozen before external test runs",
        "proposers": summary, "paired_tests": tests,
        "test_sources_sha256": {
            name: sha256(root / "external_verification" / "test" / f"{name}.csv")
            for name in PROPOSERS
        },
        "localization_boxes_unchanged_by_monitor": True,
        "undefined_auroc_sequences": "Single-class sequences remain in operating-point scores; AUROC is explicitly undefined",
    }
    write_json(root / "external_verification_summary.json", result)
    write_csv(root / "external_verification_per_sequence.csv", table_rows)
    for proposer, values in summary.items():
        print(proposer, {
            name: result["sequence_macro"] for name, result in values["methods"].items()
        }, flush=True)
    print("PAIRED_EXTERNAL_TESTS", tests, flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("select", "evaluate"))
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    (select if args.stage == "select" else evaluate)(args.output.resolve())


if __name__ == "__main__":
    main()
