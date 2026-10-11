"""Post-hoc supplementary analyses requested in internal peer review.

Everything here is recomputed from the retained per-frame CSVs of the frozen
SurgT validation/test runs. No tracker is re-run, no frozen parameter is
changed, and every analysis that uses test labels for anything other than
scoring is labelled as exploratory/post hoc in the output JSON.
"""
from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from contour_verifier import symmetric_trimmed_distance
from evaluate_verification import (
    area_under_risk_coverage, grouped_metrics, metrics, risk_values,
)
from protocol import HERE, holm_adjust, paired_statistics, read_csv, sha256, write_json


ROOT = HERE / "results"
FIGURES = ROOT / "figures"
PROPOSERS = ("nanotrack", "stark")
LABELS = {
    "native_confidence": "Native confidence", "rank_only": "Rank only",
    "contour_only": "Contour only", "verification": "Fused verification",
    "color_only": "Color only", "appearance_only": "Appearance only (no confidence)",
    "rank_contour": "Rank + contour", "delta_confidence": "|Delta confidence|",
}
TITLES = {"nanotrack": "NanoTrack", "stark": "STARK-ST50"}


def load(split: str, proposer: str) -> list[dict]:
    return read_csv(ROOT / "external_verification" / split / f"{proposer}.csv")


def labels(rows: list[dict], threshold: float = 0.5) -> np.ndarray:
    return np.asarray([float(r["box_iou"]) < threshold for r in rows], dtype=int)


def sequences_of(rows: list[dict]) -> dict[str, np.ndarray]:
    positions = defaultdict(list)
    for index, row in enumerate(rows):
        positions[row["sequence"]].append(index)
    return {name: np.asarray(idx) for name, idx in sorted(positions.items())}


def macro_auroc(rows: list[dict], risks: np.ndarray, y: np.ndarray | None = None) -> tuple[float | None, int, dict]:
    y = labels(rows) if y is None else y
    per = {}
    for name, idx in sequences_of(rows).items():
        yy = y[idx]
        per[name] = float(roc_auc_score(yy, risks[idx])) if 0 < yy.sum() < len(yy) else None
    values = [v for v in per.values() if v is not None]
    return (float(np.mean(values)) if values else None), len(values), per


def delta_confidence(rows: list[dict]) -> np.ndarray:
    native = np.asarray([float(r["native_risk"]) for r in rows])
    out = np.zeros(len(rows))
    for _, idx in sequences_of(rows).items():
        d = np.abs(np.diff(native[idx]))
        out[idx[1:]] = d
    return out


def validation_threshold(rows: list[dict], risks: np.ndarray, quantile: float = 0.9) -> float:
    y = labels(rows)
    return float(np.quantile(risks[y == 0], quantile))


def comparator_risks(rows: list[dict], selected: dict, name: str) -> np.ndarray:
    weights = {
        "native_confidence": [0, 0, 0, 1], "rank_only": [0, 1, 0, 0],
        "contour_only": [1, 0, 0, 0], "verification": selected["weights"],
        "color_only": [0, 0, 1, 0],
        "appearance_only": (lambda w: (np.asarray(w[:3] + [0.0]) / sum(w[:3])).tolist())(selected["weights"]),
        "rank_contour": [0.5, 0.5, 0, 0],
    }
    if name == "delta_confidence":
        return delta_confidence(rows)
    return risk_values(rows, selected["percentile"], weights[name])


def extra_comparators(selected: dict) -> dict:
    out = {"note": "Post-hoc comparators evaluated with the frozen percentile; thresholds use the validation reliable-frame 90th percentile exactly as for the frozen comparators. They were not part of the pre-registered selection grid."}
    for proposer in PROPOSERS:
        val, test = load("validation", proposer), load("test", proposer)
        out[proposer] = {}
        verification = comparator_risks(test, selected, "verification")
        _, _, per_verification = macro_auroc(test, verification)
        for name in ("native_confidence", "rank_only", "contour_only", "verification",
                     "color_only", "appearance_only", "rank_contour", "delta_confidence"):
            risks = comparator_risks(test, selected, name)
            threshold = validation_threshold(val, comparator_risks(val, selected, name))
            groups = grouped_metrics(test, risks, threshold)
            mean, n, per = macro_auroc(test, risks)
            rc = [g["risk_coverage_area"] for g in groups.values()]
            entry = {
                "sequence_macro_auroc": mean, "n_sequences": n,
                "sequence_macro_auroc_sd": float(np.std([v for v in per.values() if v is not None], ddof=1)),
                "pooled": metrics(test, risks, threshold),
                "sequence_macro_risk_coverage": float(np.mean(rc)),
                "threshold_from_validation": threshold,
            }
            if name != "verification":
                eligible = [s for s in per if per[s] is not None and per_verification[s] is not None]
                entry["verification_minus_this_auroc"] = paired_statistics(
                    {s: per_verification[s] for s in eligible}, {s: per[s] for s in eligible}
                )
            out[proposer][name] = entry
    return out


def episodes(y: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for i, v in enumerate(y):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(y)))
    return runs


def detection_latency(selected: dict, minimum_episode: int = 5, horizons=(1, 5, 10, 30)) -> dict:
    out = {
        "definition": (
            f"A failure episode is a maximal run of >= {minimum_episode} consecutive scored frames with IoU<0.5. "
            "Latency is the number of scored frames from the first frame of the first episode to the first frame "
            "of that episode at which the risk exceeds the validation-frozen threshold; episodes without an alarm "
            "are censored at their length and reported as missed. False alarms before the first episode are counted "
            "on reliable frames preceding it."
        ),
    }
    for proposer in PROPOSERS:
        val, test = load("validation", proposer), load("test", proposer)
        y = labels(test)
        out[proposer] = {}
        for name in ("native_confidence", "rank_only", "contour_only", "verification", "color_only"):
            risks = comparator_risks(test, selected, name)
            threshold = validation_threshold(val, comparator_risks(val, selected, name))
            alarm = risks > threshold
            latencies, missed, pre_false, pre_total, details = [], 0, 0, 0, {}
            for seq, idx in sequences_of(test).items():
                runs = [r for r in episodes(y[idx]) if r[1] - r[0] >= minimum_episode]
                if not runs:
                    details[seq] = {"episode": None}
                    continue
                start, end = runs[0]
                reliable_before = idx[:start][y[idx[:start]] == 0]
                pre_total += len(reliable_before)
                pre_false += int(alarm[reliable_before].sum())
                hits = np.flatnonzero(alarm[idx[start:end]])
                if len(hits):
                    latencies.append(int(hits[0]))
                    details[seq] = {"episode": [int(test[idx[start]]["frame_idx"]), int(test[idx[end - 1]]["frame_idx"])],
                                    "latency_scored_frames": int(hits[0]), "detected": True}
                else:
                    missed += 1
                    details[seq] = {"episode": [int(test[idx[start]]["frame_idx"]), int(test[idx[end - 1]]["frame_idx"])],
                                    "latency_scored_frames": None, "detected": False, "episode_length": int(end - start)}
            n_episodes = len(latencies) + missed
            out[proposer][name] = {
                "sequences_with_episode": n_episodes,
                "detected_episodes": len(latencies), "missed_episodes": missed,
                "median_latency_scored_frames": float(np.median(latencies)) if latencies else None,
                "mean_latency_scored_frames": float(np.mean(latencies)) if latencies else None,
                "detected_within": {
                    str(h): int(sum(l < h for l in latencies)) for h in horizons
                },
                "false_alarm_rate_before_first_episode": (pre_false / pre_total) if pre_total else None,
                "reliable_frames_before_first_episode": pre_total,
                "per_sequence": details,
            }
    return out


def temporal_specificity(selected: dict) -> dict:
    out = {"definition": "Scored frames of each sequence are split into three equal-length thirds by position; specificity on reliable frames and alarm rate on reliable frames are macro-averaged over sequences that contain reliable frames in that third."}
    for proposer in PROPOSERS:
        val, test = load("validation", proposer), load("test", proposer)
        y = labels(test)
        out[proposer] = {}
        for name in ("native_confidence", "verification"):
            risks = comparator_risks(test, selected, name)
            alarm = risks > validation_threshold(val, comparator_risks(val, selected, name))
            thirds = {0: [], 1: [], 2: []}
            for seq, idx in sequences_of(test).items():
                bounds = np.linspace(0, len(idx), 4).astype(int)
                for k in range(3):
                    part = idx[bounds[k]:bounds[k + 1]]
                    reliable = part[y[part] == 0]
                    if len(reliable):
                        thirds[k].append(float((~alarm[reliable]).mean()))
            out[proposer][name] = {
                f"third_{k + 1}": {"specificity_mean": float(np.mean(v)), "n_sequences": len(v)}
                for k, v in thirds.items()
            }
    return out


def sensitivity(selected: dict) -> dict:
    out = {"note": "Post-hoc sensitivity analyses on the test partition; they inform robustness only and were not used to choose any parameter."}
    comparators = ("native_confidence", "rank_only", "contour_only", "verification")
    # IoU failure threshold
    iou = {}
    for proposer in PROPOSERS:
        test = load("test", proposer)
        iou[proposer] = {}
        for thr in (0.3, 0.5, 0.7):
            y = labels(test, thr)
            iou[proposer][str(thr)] = {
                "failure_prevalence": float(y.mean()),
                **{name: macro_auroc(test, comparator_risks(test, selected, name), y)[0] for name in comparators},
                "n_sequences": macro_auroc(test, comparator_risks(test, selected, "verification"), y)[1],
            }
    out["iou_threshold"] = iou
    # contour percentile
    alpha = {}
    for proposer in PROPOSERS:
        test = load("test", proposer)
        alpha[proposer] = {}
        for pct in (0.5, 0.68, 0.8):
            alpha[proposer][str(pct)] = {
                "contour_only": macro_auroc(test, risk_values(test, pct, [1, 0, 0, 0]))[0],
                "verification": macro_auroc(test, risk_values(test, pct, selected["weights"]))[0],
            }
    out["contour_percentile"] = alpha
    # weight simplex scan (step 0.05)
    step = 0.05
    grid = []
    units = int(round(1 / step))
    for a in range(units + 1):
        for b in range(units + 1 - a):
            for c in range(units + 1 - a - b):
                d = units - a - b - c
                grid.append([a * step, b * step, c * step, d * step])
    simplex = {"step": step, "n_points": len(grid)}
    for proposer in PROPOSERS:
        test = load("test", proposer)
        y = labels(test)
        seqs = sequences_of(test)
        features = np.asarray([
            [float(r[f"contour_{round(selected['percentile'] * 100)}"]), float(r["rank_risk"]),
             float(r["color_risk"]), float(r["native_risk"])] for r in test
        ])
        two_class = {s: idx for s, idx in seqs.items() if 0 < y[idx].sum() < len(idx)}
        values = []
        for w in grid:
            risks = features @ np.asarray(w)
            values.append(float(np.mean([roc_auc_score(y[idx], risks[idx]) for idx in two_class.values()])))
        values = np.asarray(values)
        frozen = float(np.mean([
            roc_auc_score(y[idx], (features @ np.asarray(selected["weights"]))[idx]) for idx in two_class.values()
        ]))
        best = int(np.argmax(values))
        simplex[proposer] = {
            "frozen_auroc": frozen, "grid_min": float(values.min()), "grid_median": float(np.median(values)),
            "grid_max": float(values.max()), "best_weights": grid[best],
            "frozen_percentile_rank": float((values <= frozen).mean()),
            "fraction_within_0.01_of_best": float((values >= values.max() - 0.01).mean()),
            "fraction_within_0.02_of_best": float((values >= values.max() - 0.02).mean()),
        }
    out["weight_simplex"] = simplex
    # alarm quantile
    quantiles = {}
    for proposer in PROPOSERS:
        val, test = load("validation", proposer), load("test", proposer)
        quantiles[proposer] = {}
        for name in ("native_confidence", "verification"):
            risks_val = comparator_risks(val, selected, name)
            risks_test = comparator_risks(test, selected, name)
            quantiles[proposer][name] = {}
            for q in (0.8, 0.9, 0.95):
                thr = validation_threshold(val, risks_val, q)
                groups = grouped_metrics(test, risks_test, thr)
                sens = [g["sensitivity"] for g in groups.values() if g["sensitivity"] is not None]
                spec = [g["specificity"] for g in groups.values() if g["specificity"] is not None]
                quantiles[proposer][name][str(q)] = {
                    "threshold": thr, "sensitivity_macro": float(np.mean(sens)),
                    "specificity_macro": float(np.mean(spec)),
                    "alarm_rate_macro": float(np.mean([g["alarm_rate"] for g in groups.values()])),
                    "pooled": metrics(test, risks_test, thr),
                }
    out["alarm_quantile"] = quantiles
    return out


def weighting_and_inference(selected: dict, summary: dict) -> dict:
    out = {}
    for proposer in PROPOSERS:
        test = load("test", proposer)
        y = labels(test)
        seqs = sequences_of(test)
        per = {}
        for name in ("native_confidence", "verification"):
            risks = comparator_risks(test, selected, name)
            per[name] = macro_auroc(test, risks)[2]
        eligible = [s for s in seqs if per["verification"][s] is not None and per["native_confidence"][s] is not None]
        n_frames = np.asarray([len(seqs[s]) for s in eligible], dtype=float)
        delta = np.asarray([per["verification"][s] - per["native_confidence"][s] for s in eligible])
        cases = np.asarray([test[seqs[s][0]]["case"] for s in eligible])
        weighted = {}
        for label, w in (("equal", np.ones_like(n_frames)), ("sqrt_frames", np.sqrt(n_frames)), ("frames", n_frames)):
            w = w / w.sum()
            weighted[label] = {
                "verification": float(np.sum(w * [per["verification"][s] for s in eligible])),
                "native_confidence": float(np.sum(w * [per["native_confidence"][s] for s in eligible])),
                "difference": float(np.sum(w * delta)),
            }
        # exact sequence-level sign-flip permutation test on the mean difference
        n = len(delta)
        signs = np.asarray(list(itertools.product([-1, 1], repeat=n)), dtype=float)
        null = (signs * delta).mean(axis=1)
        observed = delta.mean()
        p_exact = float((np.abs(null) >= abs(observed) - 1e-12).mean())
        # hierarchical bootstrap: cases, then sequences within case
        rng = np.random.default_rng(20261011)
        unique_cases = sorted(set(cases))
        by_case = {c: delta[cases == c] for c in unique_cases}
        boots = []
        for _ in range(20000):
            chosen = rng.choice(unique_cases, size=len(unique_cases), replace=True)
            sample = np.concatenate([
                rng.choice(by_case[c], size=len(by_case[c]), replace=True) for c in chosen
            ])
            boots.append(sample.mean())
        lo, hi = np.quantile(boots, [0.025, 0.975])
        out[proposer] = {
            "n_sequences": n, "weighted_means": weighted,
            "exact_sign_flip_p_two_sided": p_exact, "sign_flip_permutations": int(2 ** n),
            "hierarchical_bootstrap_ci95": [float(lo), float(hi)],
            "hierarchical_bootstrap_note": "cases resampled with replacement, then sequences within each drawn case",
        }
    # Holm families
    tests = summary["paired_tests"]
    primary = [t for t in tests if t["comparator"] == "native_confidence"]
    exploratory = [t for t in tests if t["comparator"] != "native_confidence"]
    for group, label in ((primary, "primary_vs_native_confidence"), (exploratory, "exploratory_vs_single_cues")):
        adjusted = holm_adjust(t["p_two_sided"] for t in group)
        out[label] = [
            {"proposer": t["proposer"], "comparator": t["comparator"], "metric": t["metric"],
             "p_raw": t["p_two_sided"], "p_holm_family": a, "p_holm_all12": t["p_holm"]}
            for t, a in zip(group, adjusted)
        ]
    return out


def conformal_guarantee(selected: dict) -> dict:
    out = {
        "statement": (
            "Split-conformal sequence-level calibration: let q_j be the 0.90 quantile of fused risk over the reliable "
            "frames of validation sequence j (j=1..n). For a miscoverage level delta, tau_delta is the ceil((n+1)(1-delta))-th "
            "smallest q_j. If a new sequence is exchangeable with the validation sequences, P(q_new <= tau_delta) >= 1-delta, "
            "i.e. with probability at least 1-delta its reliable-frame specificity at tau_delta is at least 0.90."
        ),
    }
    for proposer in PROPOSERS:
        val, test = load("validation", proposer), load("test", proposer)
        yv, yt = labels(val), labels(test)
        out[proposer] = {}
        for name in ("native_confidence", "verification"):
            rv = comparator_risks(val, selected, name)
            rt = comparator_risks(test, selected, name)
            q = []
            for seq, idx in sequences_of(val).items():
                reliable = idx[yv[idx] == 0]
                if len(reliable) >= 10:
                    q.append(float(np.quantile(rv[reliable], 0.9)))
            q = np.sort(q)
            n = len(q)
            entry = {"n_validation_sequences": n, "frame_pooled_threshold": validation_threshold(val, rv)}
            for delta in (0.1, 0.2, 0.3):
                k = math.ceil((n + 1) * (1 - delta))
                if k > n:
                    entry[str(delta)] = {"attainable": False, "required_order_statistic": k}
                    continue
                tau = float(q[k - 1])
                spec, sens, covered = [], [], 0
                for seq, idx in sequences_of(test).items():
                    reliable, failed = idx[yt[idx] == 0], idx[yt[idx] == 1]
                    if len(reliable) >= 10:
                        s = float((rt[reliable] <= tau).mean())
                        spec.append(s)
                        covered += int(s >= 0.9)
                    if len(failed):
                        sens.append(float((rt[failed] > tau).mean()))
                entry[str(delta)] = {
                    "attainable": True, "order_statistic": k, "threshold": tau,
                    "guaranteed_sequence_coverage": 1 - delta,
                    "test_sequences_evaluated": len(spec),
                    "test_sequences_with_specificity_at_least_0.9": covered,
                    "empirical_sequence_coverage": covered / len(spec) if spec else None,
                    "test_specificity_macro": float(np.mean(spec)) if spec else None,
                    "test_sensitivity_macro": float(np.mean(sens)) if sens else None,
                }
            out[proposer][name] = entry
    return out


def contour_chance_level(seed: int = 20261011) -> dict:
    rng = np.random.default_rng(seed)
    out = {"description": "Monte Carlo chance level of the trimmed symmetric contour discrepancy between two independent random binary interior masks of the stated density on the 64x64 raster (2-pixel border empty), and its value between a mask and its translated copy."}
    for alpha in (0.5, 0.68, 0.8):
        density = 1 - alpha
        values = []
        for _ in range(300):
            a = np.zeros((64, 64), np.uint8)
            b = np.zeros((64, 64), np.uint8)
            a[2:-2, 2:-2] = rng.random((60, 60)) < density
            b[2:-2, 2:-2] = rng.random((60, 60)) < density
            values.append(symmetric_trimmed_distance(a, b))
        shifted = {}
        base = np.zeros((64, 64), np.uint8)
        base[2:-2, 2:-2] = rng.random((60, 60)) < density
        for shift in (1, 2, 4, 8, 16):
            moved = np.zeros_like(base)
            moved[:, shift:] = base[:, :-shift]
            shifted[str(shift)] = symmetric_trimmed_distance(base, moved)
        out[str(alpha)] = {
            "density": density, "random_pair_mean": float(np.mean(values)),
            "random_pair_q05": float(np.quantile(values, 0.05)), "random_pair_q95": float(np.quantile(values, 0.95)),
            "identical": symmetric_trimmed_distance(base, base), "translated_copy": shifted,
        }
    return out


def validation_test_gap(selected: dict) -> dict:
    out = {}
    for proposer in PROPOSERS:
        val, test = load("validation", proposer), load("test", proposer)
        out[proposer] = {}
        for name in ("native_confidence", "rank_only", "contour_only", "verification", "color_only"):
            v, nv, _ = macro_auroc(val, comparator_risks(val, selected, name))
            t, nt, _ = macro_auroc(test, comparator_risks(test, selected, name))
            out[proposer][name] = {"validation": v, "n_validation": nv, "test": t, "n_test": nt, "gap": (v - t) if v is not None and t is not None else None}
    return out


def failure_overlap_and_coverage() -> dict:
    nano, stark = load("test", "nanotrack"), load("test", "stark")
    key = lambda r: (r["sequence"], int(r["frame_idx"]))
    fn = {key(r) for r in nano if int(r["failure"])}
    fs = {key(r) for r in stark if int(r["failure"])}
    per_sequence = {}
    manifests = {
        p: json.loads((ROOT / "external_verification" / "test" / f"{p}.manifest.json").read_text())
        for p in PROPOSERS
    }
    for record in manifests["nanotrack"]["sequences"]:
        seq = record["sequence"]
        a = {k for k in fn if k[0] == seq}
        b = {k for k in fs if k[0] == seq}
        per_sequence[seq] = {
            "scored_frames": record["scored_frames"],
            "post_initialization_updates": record["post_initialization_updates"],
            "scored_fraction": record["scored_frames"] / record["post_initialization_updates"],
            "failure_jaccard": (len(a & b) / len(a | b)) if (a | b) else None,
        }
    native = {}
    for proposer, rows in (("nanotrack", nano), ("stark", stark)):
        c = 1 - np.asarray([float(r["native_risk"]) for r in rows])
        y = labels(rows)
        native[proposer] = {
            "confidence_mean_reliable": float(c[y == 0].mean()), "confidence_sd_reliable": float(c[y == 0].std(ddof=1)),
            "confidence_mean_failed": float(c[y == 1].mean()), "confidence_sd_failed": float(c[y == 1].std(ddof=1)),
            "fraction_confidence_above_0.9_failed": float((c[y == 1] > 0.9).mean()),
            "fraction_confidence_above_0.9_reliable": float((c[y == 0] > 0.9).mean()),
        }
    return {
        "pooled_failure_jaccard": len(fn & fs) / len(fn | fs),
        "per_sequence": per_sequence,
        "scored_fraction_mean": float(np.mean([v["scored_fraction"] for v in per_sequence.values()])),
        "scored_fraction_min": float(np.min([v["scored_fraction"] for v in per_sequence.values()])),
        "native_confidence_distribution": native,
    }


def figures(selected: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    cues = ("native_confidence", "rank_only", "contour_only", "color_only", "verification")
    colors = {"native_confidence": "#9a9a9a", "rank_only": "#7fa6c9", "contour_only": "#6fa36f",
              "color_only": "#d9a441", "verification": "#2f4f6f"}
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.2), constrained_layout=True)
    for column, proposer in enumerate(PROPOSERS):
        test = load("test", proposer)
        y = labels(test)
        for name in cues:
            risks = comparator_risks(test, selected, name)
            fpr, tpr, _ = roc_curve(y, risks)
            axes[0, column].plot(fpr, tpr, color=colors[name], lw=1.6 if name == "verification" else 1.1,
                                 label=f"{LABELS[name]} (pooled AUROC {roc_auc_score(y, risks):.3f})")
            order = np.argsort(risks, kind="stable")
            coverage = np.arange(1, len(y) + 1) / len(y)
            risk = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
            axes[1, column].plot(coverage, risk, color=colors[name], lw=1.6 if name == "verification" else 1.1,
                                 label=f"{LABELS[name]} (area {area_under_risk_coverage(y, risks):.3f})")
        axes[0, column].plot([0, 1], [0, 1], ls="--", color="#bbbbbb", lw=0.8)
        axes[0, column].set(title=f"{TITLES[proposer]}: pooled ROC", xlabel="False alarm rate on reliable frames",
                            ylabel="Failure detection rate", xlim=(0, 1), ylim=(0, 1))
        axes[1, column].set(title=f"{TITLES[proposer]}: risk-coverage", xlabel="Coverage (fraction of frames retained, lowest risk first)",
                            ylabel="Failure rate among retained frames", xlim=(0, 1), ylim=(0, 1))
        for axis in axes[:, column]:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, loc="lower right" if axis is axes[0, column] else "upper left")
    fig.savefig(FIGURES / "Verification_Curves.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    for axis, proposer in zip(axes, PROPOSERS):
        test = load("test", proposer)
        seqs = sequences_of(test)
        native = macro_auroc(test, comparator_risks(test, selected, "native_confidence"))[2]
        fused = macro_auroc(test, comparator_risks(test, selected, "verification"))[2]
        cases = sorted({test[idx[0]]["case"] for idx in seqs.values()})
        palette = dict(zip(cases, plt.cm.tab10.colors))
        for seq, idx in seqs.items():
            if native[seq] is None or fused[seq] is None:
                continue
            axis.scatter(native[seq], fused[seq], s=12 + len(idx) / 12, color=palette[test[idx[0]]["case"]],
                         alpha=0.85, edgecolor="white", linewidth=0.5)
        axis.plot([0, 1], [0, 1], ls="--", color="#888888", lw=0.8)
        axis.set(title=f"{TITLES[proposer]}: per-sequence failure AUROC", xlabel="Native confidence",
                 ylabel="Fused verification", xlim=(0, 1.02), ylim=(0, 1.02))
        axis.grid(alpha=0.25)
        handles = [plt.Line2D([], [], marker="o", ls="", color=palette[c], label=c.replace("_", " ")) for c in cases]
        axis.legend(handles=handles, fontsize=7, loc="lower right", title="marker size ~ frames", title_fontsize=7)
    fig.savefig(FIGURES / "Verification_PairedSequences.pdf")
    plt.close(fig)


def main() -> None:
    selected = json.loads((ROOT / "selected_verification.json").read_text())
    summary = json.loads((ROOT / "external_verification_summary.json").read_text())
    result = {
        "selection_manifest_sha256": sha256(ROOT / "selected_verification.json"),
        "external_summary_sha256": sha256(ROOT / "external_verification_summary.json"),
        "all_analyses_post_hoc_on_frozen_outputs": True,
        "extra_comparators": extra_comparators(selected),
        "detection_latency": detection_latency(selected),
        "temporal_specificity": temporal_specificity(selected),
        "sensitivity": sensitivity(selected),
        "weighting_and_inference": weighting_and_inference(selected, summary),
        "conformal": conformal_guarantee(selected),
        "contour_chance_level": contour_chance_level(),
        "validation_test_gap": validation_test_gap(selected),
        "overlap_and_coverage": failure_overlap_and_coverage(),
    }
    figures(selected)
    write_json(ROOT / "supplementary_analyses.json", result)
    for proposer in PROPOSERS:
        print(proposer, "extra:", {k: round(v["sequence_macro_auroc"], 3) for k, v in result["extra_comparators"][proposer].items()})
        print(proposer, "latency:", {k: (v["median_latency_scored_frames"], v["missed_episodes"]) for k, v in result["detection_latency"][proposer].items()})
        print(proposer, "conformal:", result["conformal"][proposer]["verification"])
        print(proposer, "inference:", {k: v for k, v in result["weighting_and_inference"][proposer].items() if k != "weighted_means"})
    print("SUPPLEMENTARY_ANALYSES_WRITTEN")


if __name__ == "__main__":
    main()
