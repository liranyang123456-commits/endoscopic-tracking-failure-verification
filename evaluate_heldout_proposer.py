"""Evaluate the frozen verifier around a proposer that took no part in selection.

The contour percentile and fusion weights are read from the frozen
``selected_verification.json`` and are not changed. The only proposer-specific
quantity, the alarm threshold, is calibrated on the held-out proposer's
validation reliable frames exactly as for the two selection proposers.
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from evaluate_verification import grouped_metrics, metrics
from protocol import HERE, holm_adjust, paired_statistics, read_csv, sha256, write_csv, write_json
from supplementary_analyses import (
    comparator_risks, episodes, labels, macro_auroc, sequences_of, validation_threshold,
)


ROOT = HERE / "results"
COMPARATORS = ("native_confidence", "rank_only", "contour_only", "color_only", "verification")


def shared_frames(proposer: str, split: str) -> dict:
    reference = json.loads((ROOT / "external_verification" / split / "stark.manifest.json").read_text())
    candidate = json.loads((ROOT / "external_verification" / split / f"{proposer}.manifest.json").read_text())
    ref = {r["sequence"]: r["input_frame_sha256"] for r in reference["sequences"]}
    new = {r["sequence"]: r["input_frame_sha256"] for r in candidate["sequences"]}
    if ref != new:
        raise ValueError(f"Held-out proposer did not see the identical rectified frames on {split}")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposer", default="ostrack")
    args = parser.parse_args()
    proposer = args.proposer
    selected = json.loads((ROOT / "selected_verification.json").read_text())
    if selected["protocol_sha256"] != sha256(HERE / "verification_protocol.json"):
        raise ValueError("Frozen protocol hash differs")
    if proposer in selected["validation_auroc_by_proposer"]:
        raise ValueError("This script is only for proposers that were excluded from selection")
    manifest = shared_frames(proposer, "test")
    shared_frames(proposer, "validation")
    val = read_csv(ROOT / "external_verification" / "validation" / f"{proposer}.csv")
    test = read_csv(ROOT / "external_verification" / "test" / f"{proposer}.csv")
    if len(test) != 12056 or len(val) != 7666:
        raise ValueError(f"Held-out proposer has a different scored-frame inventory: {len(val)} / {len(test)}")
    y = labels(test)
    results, per_sequence_rows = {}, []
    per_sequence_auc = {}
    for name in COMPARATORS:
        risks_val = comparator_risks(val, selected, name)
        risks = comparator_risks(test, selected, name)
        threshold = validation_threshold(val, risks_val)
        groups = grouped_metrics(test, risks, threshold)
        aggregate = {}
        for metric in ("failure_auroc", "failure_average_precision", "sensitivity", "specificity", "alarm_rate", "risk_coverage_area"):
            available = [g[metric] for g in groups.values() if g[metric] is not None]
            aggregate[metric] = {
                "mean": float(np.mean(available)) if available else None,
                "sample_sd": float(np.std(available, ddof=1)) if len(available) > 1 else None,
                "n_sequences": len(available),
            }
        validation_macro = macro_auroc(val, risks_val)
        results[name] = {
            "threshold_frozen_from_validation": threshold,
            "validation_sequence_macro_auroc": validation_macro[0],
            "validation_n_sequences": validation_macro[1],
            "pooled": metrics(test, risks, threshold),
            "sequence_macro": aggregate, "sequences": groups,
        }
        per_sequence_auc[name] = {s: g["failure_auroc"] for s, g in groups.items()}
        per_sequence_rows.extend({"proposer": proposer, "method": name, "sequence": s, **g} for s, g in groups.items())
    tests = []
    reference = per_sequence_auc["verification"]
    for comparator in COMPARATORS[:-1]:
        other = per_sequence_auc[comparator]
        eligible = [s for s in reference if reference[s] is not None and other[s] is not None]
        effect = paired_statistics({s: reference[s] for s in eligible}, {s: other[s] for s in eligible})
        delta = np.asarray([reference[s] - other[s] for s in eligible])
        signs = np.asarray(list(itertools.product([-1, 1], repeat=len(delta))), dtype=float)
        null = (signs * delta).mean(axis=1)
        cases = np.asarray([results["verification"]["sequences"][s]["case"] for s in eligible])
        unique = sorted(set(cases))
        rng = np.random.default_rng(20261011)
        boots = []
        for _ in range(20000):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            sample = np.concatenate([rng.choice(delta[cases == c], size=int((cases == c).sum()), replace=True) for c in chosen])
            boots.append(sample.mean())
        tests.append({
            "proposer": proposer, "comparator": comparator, "metric": "failure_auroc", **effect,
            "exact_sign_flip_p_two_sided": float((np.abs(null) >= abs(delta.mean()) - 1e-12).mean()),
            "hierarchical_bootstrap_ci95": [float(v) for v in np.quantile(boots, [0.025, 0.975])],
        })
    for row, adjusted in zip(tests, holm_adjust(t["p_two_sided"] for t in tests)):
        row["p_holm_heldout_family"] = adjusted
    # detection latency for the first failure episode (>= 5 scored frames)
    latency = {}
    for name in COMPARATORS:
        risks = comparator_risks(test, selected, name)
        alarm = risks > results[name]["threshold_frozen_from_validation"]
        found, missed = [], 0
        for seq, idx in sequences_of(test).items():
            runs = [r for r in episodes(y[idx]) if r[1] - r[0] >= 5]
            if not runs:
                continue
            start, end = runs[0]
            hits = np.flatnonzero(alarm[idx[start:end]])
            if len(hits):
                found.append(int(hits[0]))
            else:
                missed += 1
        latency[name] = {
            "detected_episodes": len(found), "missed_episodes": missed,
            "median_latency_scored_frames": float(np.median(found)) if found else None,
            "detected_within_10": int(sum(l < 10 for l in found)),
        }
    sequences = manifest["sequences"]
    pooled = results["verification"]["pooled"]
    summary = {
        "proposer": proposer,
        "role": "held-out proposer; excluded from percentile/weight selection",
        "selection_manifest_sha256": sha256(ROOT / "selected_verification.json"),
        "proposer_metadata": manifest["proposer_metadata"],
        "test_source_sha256": sha256(ROOT / "external_verification" / "test" / f"{proposer}.csv"),
        "validation_source_sha256": sha256(ROOT / "external_verification" / "validation" / f"{proposer}.csv"),
        "scored_frames": len(test), "validation_scored_frames": len(val),
        "sequence_count": len(sequences), "case_count": len({r["case"] for r in sequences}),
        "box_iou_unchanged": pooled["box_iou_unchanged"], "failure_prevalence": pooled["failure_prevalence"],
        "n_failed": pooled["n_failed"], "n_reliable": pooled["n_reliable"],
        "mean_proposer_ms_per_update": sum(r["proposer_time_s"] for r in sequences) / sum(r["post_initialization_updates"] for r in sequences) * 1000,
        "mean_monitor_ms_per_scored_frame": sum(r["monitor_time_s"] for r in sequences) / len(test) * 1000,
        "methods": results, "paired_tests": tests, "detection_latency": latency,
        "localization_boxes_unchanged_by_monitor": True,
    }
    write_json(ROOT / f"heldout_{proposer}_summary.json", summary)
    write_csv(ROOT / f"heldout_{proposer}_per_sequence.csv", per_sequence_rows)
    print(proposer, "box IoU", round(pooled["box_iou_unchanged"], 3), "prevalence", round(pooled["failure_prevalence"], 3))
    for name in COMPARATORS:
        macro = results[name]["sequence_macro"]["failure_auroc"]
        print(f"  {name}: test AUROC {macro['mean']:.3f} (n={macro['n_sequences']}), validation {results[name]['validation_sequence_macro_auroc']:.3f}")
    for t in tests:
        print(f"  verification - {t['comparator']}: {t['mean_paired_difference']:+.3f} [{t['ci95_low']:+.3f}, {t['ci95_high']:+.3f}] p={t['p_two_sided']:.4f} Holm={t['p_holm_heldout_family']:.4f}")
    print("HELDOUT_EVALUATED")


if __name__ == "__main__":
    main()
