"""Publication figures in the house style of the earlier manuscripts.

Palette and conventions follow ``Sensors_Submission/make_headline_figures.py``:
two series colours (blue / orange), black edge for the proposed method, green
edge for NanoTrack, value labels on bars, (a)/(b)/(c) panels, dashed reference
lines. Every number is read from the result JSON files.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import roc_auc_score, roc_curve

from evaluate_verification import area_under_risk_coverage
from protocol import HERE
from supplementary_analyses import comparator_risks, labels, load, macro_auroc, sequences_of


ROOT = HERE / "results"
OUT = ROOT / "figures"
C_A, C_B = "#4C78A8", "#F58518"
C_OURS_EDGE, C_NANO_EDGE = "#1B1B1B", "#2A6F3B"
C_RED = "#E45756"
BLUES = ["#C6DBEF", "#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#08519C"]
PROPOSER_NAMES = {"nanotrack": "NanoTrack", "stark": "STARK-ST50", "ostrack": "OSTrack (held-out)"}
CUE_NAMES = {
    "native_confidence": "Native conf.", "delta_confidence": "|Δ conf.|", "color_only": "Color",
    "contour_only": "Contour", "rank_only": "Rank", "rank_contour": "Rank+contour",
    "appearance_only": "Appear. (no conf.)", "verification": "Fused (ours)",
}


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
        "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.28, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.dpi": 300,
    })


def annotate(ax, bars, fmt="{:.3f}", dy=0.012, fontsize=7.0):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=fontsize, color="#222222")


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------- framework banner
def framework(selected: dict) -> None:
    style()
    fig, ax = plt.subplots(figsize=(12.6, 6.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    palette = {
        "A": ("#EAF2FB", "#4C78A8"), "B": ("#F1EAF7", "#7B5EA7"), "C": ("#EAF6EE", "#2A6F3B"),
        "D": ("#F3F3F3", "#8A8A8A"),
    }

    def panel(x, y, w, h, key, title, dashed=False):
        face, edge = palette[key]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.0", fc=face, ec=edge,
                                    lw=1.6, ls="--" if dashed else "-", zorder=1))
        ax.add_patch(FancyBboxPatch((x + 0.6, y + h - 3.4), w - 1.2, 2.8, boxstyle="round,pad=0.1,rounding_size=0.6",
                                    fc=edge, ec=edge, lw=0, zorder=2))
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="center", fontsize=9.4, color="white", weight="bold", zorder=3)

    def step(x, y, w, h, number, head, detail, key):
        _, edge = palette[key]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=0.7", fc="white", ec=edge, lw=1.1, zorder=3))
        ax.text(x + 1.3, y + h / 2, number, ha="center", va="center", fontsize=10, color=edge, weight="bold", zorder=4)
        ax.text(x + 2.7, y + h / 2 + (1.0 if detail else 0), head, ha="left", va="center", fontsize=8.0, color="#1F2D3D", weight="bold", zorder=4)
        if detail:
            ax.text(x + 2.7, y + h / 2 - 1.05, detail, ha="left", va="center", fontsize=6.9, color="#4A4A4A", zorder=4)

    def arrow(p, q, color, lw=1.6, ls="-"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, ls=ls, zorder=5))

    ax.text(50, 48.8, "Ordinal-gradient appearance verification around a frozen region-level tracker",
            ha="center", va="center", fontsize=12.5, weight="bold", color="#2F3E4E")
    ax.text(50, 46.9, "online path A → C and B → C at every frame  ·  D is used only offline  ·  the tracker box is never modified",
            ha="center", va="center", fontsize=8.4, color="#5A6A7A")
    top, bottom = 45.2, 13.2
    # Panel A: reference cues (stacked steps)
    panel(0.5, bottom, 30.5, top - bottom, "A", "A   Reference cues (once)")
    ys = [37.6, 32.6, 27.6, 22.6]
    step(1.6, ys[0], 28.3, 3.9, "①", "Initial frame $I_0$ and supplied ROI $R_0$", "the only supervision available online", "A")
    step(1.6, ys[1], 28.3, 3.9, "②", "Canonical crop 64 × 64, grayscale", "area interpolation inside the image", "A")
    step(1.6, ys[2], 28.3, 3.9, "③", "Midranks $U_0$  (Eq. 1)", "invariant to monotone photometric change", "A")
    step(1.6, ys[3], 28.3, 3.9, "④", f"Contour $C_0$ (α = {selected['percentile']:.2f}, Eq. 2) and HSV $h_0$", "fixed-density gradient set · 16×16 histogram", "A")
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow((15.75, y0), (15.75, y1 + 3.9), palette["A"][1])
    ax.text(15.75, 18.4, "fixed reference $(U_0, C_0, h_0)$\nno online update", ha="center", va="center", fontsize=7.6, color=palette["A"][1], style="italic")
    # Panel B: frozen proposer
    panel(33.0, bottom, 24.0, top - bottom, "B", "B   Frozen proposer")
    step(34.1, 32.6, 21.8, 4.6, "①", "Official tracker $\\mathcal{T}$ (each frame $t$)", "NanoTrack · STARK-ST50 · OSTrack\nno reset, no fine-tuning", "B")
    step(34.1, 26.4, 21.8, 3.9, "②", "Proposal $R_t$, confidence $c_t$", "box is forwarded downstream unchanged", "B")
    arrow((45.0, 32.6), (45.0, 30.3), palette["B"][1])
    ax.add_patch(FancyBboxPatch((34.1, 18.6), 21.8, 3.6, boxstyle="round,pad=0.15,rounding_size=0.6", fc="white", ec=palette["B"][1], lw=1.4, zorder=3))
    ax.text(45.0, 20.4, "output 1:  box $R_t$ — unchanged", ha="center", va="center", fontsize=8.0, color=palette["B"][1], weight="bold", zorder=4)
    arrow((45.0, 26.4), (45.0, 22.2), palette["B"][1])
    # Panel C: verifier (stacked steps)
    panel(59.5, bottom, 40.0, top - bottom, "C", "C   Non-invasive verifier (≈ 1.4 ms per frame)")
    cx = 60.6
    step(cx, ys[0], 37.8, 3.9, "①", "Candidate cues $U_t, C_t, h_t$ from the patch under $R_t$", "same operators as Panel A", "C")
    step(cx, ys[1], 37.8, 3.9, "②", "Bounded alignment of $U_t, C_t$ to the reference", "phase correlation, |shift| ≤ 12 px, response > 0.05  (Eq. 3)", "C")
    step(cx, ys[2], 37.8, 3.9, "③", "Discrepancies $d_C, d_U, d_H, d_N \\in [0, 1]$", "trimmed contour distance · 1 − Spearman · Hellinger · 1 − $c_t$  (Eqs. 4–5)", "C")
    step(cx, ys[3], 37.8, 3.9, "④", "Risk $r_t = \\mathbf{w}^{\\top}\\mathbf{d}_t$,  warning $\\omega_t = \\mathbf{1}\\{r_t > \\tau\\}$", "convex weights (Eq. 6); τ frame-pooled or conformal $\\tau_\\delta$ (Prop. 2)", "C")
    for y0, y1 in zip(ys[:-1], ys[1:]):
        arrow((79.5, y0), (79.5, y1 + 3.9), palette["C"][1])
    ax.add_patch(FancyBboxPatch((cx, 16.6), 37.8, 3.6, boxstyle="round,pad=0.15,rounding_size=0.6", fc="white", ec=palette["C"][1], lw=1.4, zorder=3))
    ax.text(79.5, 18.4, "output 2:  risk $r_t$ and warning $\\omega_t$ (box untouched)", ha="center", va="center", fontsize=8.0, color=palette["C"][1], weight="bold", zorder=4)
    arrow((79.5, 22.6), (79.5, 20.2), palette["C"][1])
    # cross-panel arrows
    arrow((30.0, 39.55), (60.6, 39.55), palette["A"][1], lw=2.0)
    ax.text(45.3, 40.1, "reference descriptors $(U_0, C_0, h_0)$", ha="center", va="bottom", fontsize=7.4, color=palette["A"][1])
    arrow((56.0, 28.35), (60.6, 28.35), palette["B"][1], lw=2.0)
    ax.text(58.3, 29.0, "$R_t, c_t$", ha="center", va="bottom", fontsize=7.4, color=palette["B"][1])
    # Panel D: offline
    panel(0.5, 0.6, 99, 10.6, "D", "D   Offline only — never read by the online verifier", dashed=True)
    step(1.6, 1.7, 31.2, 5.0, "①", "Validation partition (12 SurgT sequences)", "selects α and $\\mathbf{w}$ on a 9-point grid\ncalibrates τ on reliable frames", "D")
    step(34.3, 1.7, 31.2, 5.0, "②", "Conformal threshold $\\tau_\\delta$ (Prop. 2)", "per-sequence 0.90 risk quantiles → order statistic\nspecificity ≥ 0.90 with probability ≥ 1 − δ", "D")
    step(67.0, 1.7, 31.5, 5.0, "③", "Test partition (20 sequences, frozen)", "failure = IoU($R_t$, $G_t$) < 0.5\n12 056 scored frames per proposer", "D")
    save(fig, "Fig_Framework")


# ----------------------------------------------------------------------------- headline
def headline(results: dict, heldout: dict, supplementary: dict, transfer: dict) -> None:
    style()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})
    # (a) AUROC native vs fused for three proposers
    ax = axes[0]
    props = ("nanotrack", "stark", "ostrack")
    data = {p: (results["proposers"][p]["methods"] if p != "ostrack" else heldout["methods"]) for p in props}
    x = np.arange(3)
    w = 0.38
    nat = [data[p]["native_confidence"]["sequence_macro"]["failure_auroc"]["mean"] for p in props]
    nat_sd = [data[p]["native_confidence"]["sequence_macro"]["failure_auroc"]["sample_sd"] for p in props]
    fus = [data[p]["verification"]["sequence_macro"]["failure_auroc"]["mean"] for p in props]
    fus_sd = [data[p]["verification"]["sequence_macro"]["failure_auroc"]["sample_sd"] for p in props]
    b1 = ax.bar(x - w / 2, nat, w, yerr=nat_sd, capsize=3, color=C_A, label="Native confidence", zorder=3, error_kw={"lw": 0.9})
    b2 = ax.bar(x + w / 2, fus, w, yerr=fus_sd, capsize=3, color=C_B, label="Fused verification (ours)", zorder=3, error_kw={"lw": 0.9})
    for b in b2:
        b.set_edgecolor(C_OURS_EDGE)
        b.set_linewidth(1.4)
    b1[0].set_edgecolor(C_NANO_EDGE)
    b1[0].set_linewidth(1.2)
    annotate(ax, b1, dy=0.03)
    annotate(ax, b2, dy=0.03)
    ax.axhline(0.5, ls="--", color="#999999", lw=0.9)
    ax.text(2.45, 0.515, "chance", fontsize=7, color="#777777", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(["NanoTrack", "STARK-ST50", "OSTrack\n(held-out)"])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Failure AUROC (sequence mean ± SD)")
    ax.set_title("(a) SurgT test: failure ranking\n20 sequences, frozen parameters")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    # (b) first-episode detection
    ax = axes[1]
    lat = supplementary["detection_latency"]
    det_nat = [lat[p]["native_confidence"]["detected_episodes"] for p in ("nanotrack", "stark")] + [heldout["detection_latency"]["native_confidence"]["detected_episodes"]]
    det_fus = [lat[p]["verification"]["detected_episodes"] for p in ("nanotrack", "stark")] + [heldout["detection_latency"]["verification"]["detected_episodes"]]
    tot = [lat[p]["verification"]["sequences_with_episode"] for p in ("nanotrack", "stark")] + [heldout["detection_latency"]["verification"]["detected_episodes"] + heldout["detection_latency"]["verification"]["missed_episodes"]]
    med_nat = [lat[p]["native_confidence"]["median_latency_scored_frames"] for p in ("nanotrack", "stark")] + [heldout["detection_latency"]["native_confidence"]["median_latency_scored_frames"]]
    med_fus = [lat[p]["verification"]["median_latency_scored_frames"] for p in ("nanotrack", "stark")] + [heldout["detection_latency"]["verification"]["median_latency_scored_frames"]]
    frac_nat = np.array(det_nat) / np.array(tot)
    frac_fus = np.array(det_fus) / np.array(tot)
    b1 = ax.bar(x - w / 2, frac_nat, w, color=C_A, zorder=3, label="Native confidence")
    b2 = ax.bar(x + w / 2, frac_fus, w, color=C_B, zorder=3, label="Fused verification (ours)")
    for b in b2:
        b.set_edgecolor(C_OURS_EDGE)
        b.set_linewidth(1.4)
    for i, (bn, bf) in enumerate(zip(b1, b2)):
        ax.text(bn.get_x() + bn.get_width() / 2, bn.get_height() + 0.015, f"{det_nat[i]}/{tot[i]}\nmed. {med_nat[i]:.0f} fr", ha="center", va="bottom", fontsize=6.6)
        ax.text(bf.get_x() + bf.get_width() / 2, bf.get_height() + 0.015, f"{det_fus[i]}/{tot[i]}\nmed. {med_fus[i]:.1f} fr", ha="center", va="bottom", fontsize=6.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["NanoTrack", "STARK-ST50", "OSTrack\n(held-out)"])
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("First failure episode detected (fraction)")
    ax.set_title("(b) Detection of the first failure episode\nlabels: detected/episodes, median latency")
    # (c) transfer
    ax = axes[2]
    groups = [("trackves", "nanotrack"), ("trackves", "stark"), ("chess", "nanotrack"), ("chess", "stark")]
    xt = np.arange(4)
    nat = [transfer["domains"][d][p]["methods"]["native_confidence"]["AUROC_macro"] for d, p in groups]
    fus = [transfer["domains"][d][p]["methods"]["verification"]["AUROC_macro"] for d, p in groups]
    b1 = ax.bar(xt - w / 2, nat, w, color=C_A, zorder=3, label="Native confidence")
    b2 = ax.bar(xt + w / 2, fus, w, color=C_B, zorder=3, label="Four-cue verification (ours)")
    for b in b2:
        b.set_edgecolor(C_OURS_EDGE)
        b.set_linewidth(1.4)
    annotate(ax, b1)
    annotate(ax, b2)
    ax.set_xticks(xt)
    ax.set_xticklabels(["TrackVes\nNanoTrack", "TrackVes\nSTARK-ST50", "Controlled\nNanoTrack", "Controlled\nSTARK-ST50"], fontsize=8)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Failure AUROC (sequence mean)")
    ax.set_title("(c) Transfer without recalibration\nfrozen α, w; no target-dataset tuning")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    fig.suptitle("Headline results (black edge: ours; green edge: NanoTrack native). Higher is better.", fontsize=10, y=1.02)
    fig.tight_layout()
    save(fig, "Fig_Headline")


# ----------------------------------------------------------------------------- cue ablation
def cue_ablation(results: dict, heldout: dict, supplementary: dict) -> None:
    style()
    extra = supplementary["extra_comparators"]
    order = ["native_confidence", "delta_confidence", "color_only", "contour_only", "rank_only", "rank_contour", "appearance_only", "verification"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), sharey=True)
    for ax, proposer in zip(axes, ("nanotrack", "stark", "ostrack")):
        if proposer == "ostrack":
            names = [n for n in order if n in heldout["methods"]]
            means = [heldout["methods"][n]["sequence_macro"]["failure_auroc"]["mean"] for n in names]
            sds = [heldout["methods"][n]["sequence_macro"]["failure_auroc"]["sample_sd"] for n in names]
        else:
            names = order
            means = [extra[proposer][n]["sequence_macro_auroc"] for n in names]
            sds = [extra[proposer][n]["sequence_macro_auroc_sd"] for n in names]
        colors = [BLUES[min(i, len(BLUES) - 1)] for i in range(len(names) - 1)] + [C_RED]
        bars = ax.bar(np.arange(len(names)), means, yerr=sds, capsize=3, color=colors, zorder=3, error_kw={"lw": 0.9})
        bars[-1].set_edgecolor(C_OURS_EDGE)
        bars[-1].set_linewidth(1.4)
        annotate(ax, bars, dy=0.03, fontsize=6.6)
        singles = [i for i, n in enumerate(names) if n in ("color_only", "contour_only", "rank_only")]
        best_single = max(means[i] for i in singles)
        ax.axhline(best_single, ls="--", color="#777777", lw=0.9)
        ax.text(-0.35, best_single + 0.015, f"best single appearance cue = {best_single:.3f}", fontsize=6.6, color="#555555", ha="left", va="bottom")
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels([CUE_NAMES[n] for n in names], fontsize=7.2, rotation=18, ha="right")
        ax.set_ylim(0, 1.2)
        ax.set_title(f"({'abc'[['nanotrack', 'stark', 'ostrack'].index(proposer)]}) {PROPOSER_NAMES[proposer]}")
    axes[0].set_ylabel("Failure AUROC (sequence mean ± SD)")
    fig.suptitle("Cue ablation on the SurgT test partition: where the information lies (red: fused verifier)", fontsize=10.5)
    fig.tight_layout()
    save(fig, "Fig_CueAblation")


# ----------------------------------------------------------------------------- curves and paired
def curves(selected: dict) -> None:
    style()
    cues = ("native_confidence", "color_only", "contour_only", "rank_only", "verification")
    colors = {"native_confidence": "#9A9A9A", "color_only": "#F58518", "contour_only": "#54A24B", "rank_only": "#4C78A8", "verification": C_RED}
    names = {"native_confidence": "Native confidence", "color_only": "Color only", "contour_only": "Contour only", "rank_only": "Rank only", "verification": "Fused (ours)"}
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.4))
    for col, proposer in enumerate(("nanotrack", "stark")):
        test = load("test", proposer)
        y = labels(test)
        for cue in cues:
            risks = comparator_risks(test, selected, cue)
            fpr, tpr, _ = roc_curve(y, risks)
            axes[0, col].plot(fpr, tpr, color=colors[cue], lw=1.9 if cue == "verification" else 1.2, label=f"{names[cue]} ({roc_auc_score(y, risks):.3f})")
            order = np.argsort(risks, kind="stable")
            coverage = np.arange(1, len(y) + 1) / len(y)
            risk = np.cumsum(y[order]) / np.arange(1, len(y) + 1)
            axes[1, col].plot(coverage, risk, color=colors[cue], lw=1.9 if cue == "verification" else 1.2, label=f"{names[cue]} (area {area_under_risk_coverage(y, risks):.3f})")
        axes[0, col].plot([0, 1], [0, 1], ls="--", color="#BBBBBB", lw=0.9)
        axes[0, col].set(title=f"({'ab'[col]}) {PROPOSER_NAMES[proposer]}: pooled ROC", xlabel="False-alarm rate on reliable frames", ylabel="Detection rate on failed frames", xlim=(0, 1), ylim=(0, 1.02))
        axes[1, col].set(title=f"({'cd'[col]}) {PROPOSER_NAMES[proposer]}: risk–coverage", xlabel="Coverage (lowest-risk frames retained)", ylabel="Failure rate among retained frames", xlim=(0, 1), ylim=(0, 1))
        axes[0, col].legend(loc="lower right", fontsize=7.3)
        axes[1, col].legend(loc="upper left", fontsize=7.3)
    fig.tight_layout()
    save(fig, "Fig_Curves")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
    for ax, proposer in zip(axes, ("nanotrack", "stark", "ostrack")):
        test = load("test", proposer)
        seqs = sequences_of(test)
        native = macro_auroc(test, comparator_risks(test, selected, "native_confidence"))[2]
        fused = macro_auroc(test, comparator_risks(test, selected, "verification"))[2]
        cases = sorted({test[idx[0]]["case"] for idx in seqs.values()})
        palette = dict(zip(cases, ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]))
        above = total = 0
        for seq, idx in seqs.items():
            if native[seq] is None or fused[seq] is None:
                continue
            total += 1
            above += fused[seq] > native[seq]
            ax.scatter(native[seq], fused[seq], s=14 + len(idx) / 10, color=palette[test[idx[0]]["case"]], alpha=0.9, edgecolor="white", linewidth=0.6, zorder=3)
        ax.plot([0, 1], [0, 1], ls="--", color="#888888", lw=0.9)
        ax.set(title=f"({'abc'[list(axes).index(ax)]}) {PROPOSER_NAMES[proposer]}: {above}/{total} above diagonal", xlabel="Native confidence AUROC", ylabel="Fused verification AUROC", xlim=(0, 1.03), ylim=(0, 1.03))
        handles = [plt.Line2D([], [], marker="o", ls="", color=palette[c], label=c.replace("_", " ")) for c in cases]
        ax.legend(handles=handles, fontsize=7, loc="lower right", title="marker size ∝ frames", title_fontsize=7)
    fig.tight_layout()
    save(fig, "Fig_PairedSequences")


# ----------------------------------------------------------------------------- sensitivity + conformal
def sensitivity(supplementary: dict) -> None:
    style()
    sens = supplementary["sensitivity"]
    conf = supplementary["conformal"]
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 3.7))
    cue_colors = {"native_confidence": "#9A9A9A", "rank_only": "#4C78A8", "contour_only": "#54A24B", "verification": C_RED}
    cue_names = {"native_confidence": "Native", "rank_only": "Rank", "contour_only": "Contour", "verification": "Fused (ours)"}
    markers = {"nanotrack": "o", "stark": "s"}
    ax = axes[0]
    for proposer in ("nanotrack", "stark"):
        for cue in cue_colors:
            ys = [sens["iou_threshold"][proposer][t][cue] for t in ("0.3", "0.5", "0.7")]
            ax.plot([0.3, 0.5, 0.7], ys, marker=markers[proposer], color=cue_colors[cue], lw=1.6 if cue == "verification" else 1.1, ls="-" if proposer == "nanotrack" else "--",
                    label=f"{cue_names[cue]} ({PROPOSER_NAMES[proposer].split()[0]})" if True else None)
    ax.set(xlabel="failure definition: IoU threshold", ylabel="Failure AUROC (sequence mean)", title="(a) Failure-definition sensitivity", xticks=[0.3, 0.5, 0.7], ylim=(0.4, 1.0))
    ax.legend(fontsize=6.3, ncol=2, loc="lower left")
    ax = axes[1]
    for proposer in ("nanotrack", "stark"):
        for cue, color in (("contour_only", "#54A24B"), ("verification", C_RED)):
            ys = [sens["contour_percentile"][proposer][a][cue] for a in ("0.5", "0.68", "0.8")]
            ax.plot([0.5, 0.68, 0.8], ys, marker=markers[proposer], color=color, lw=1.6 if cue == "verification" else 1.1, ls="-" if proposer == "nanotrack" else "--", label=f"{cue_names[cue]} ({PROPOSER_NAMES[proposer].split()[0]})")
    ax.axvline(0.8, ls=":", color="#888888", lw=0.9)
    ax.text(0.795, 0.42, "frozen α", fontsize=7, ha="right", color="#666666")
    ax.set(xlabel="contour percentile α", title="(b) Contour-percentile sensitivity", xticks=[0.5, 0.68, 0.8], ylim=(0.4, 1.0))
    ax.legend(fontsize=6.6, loc="lower left")
    ax = axes[2]
    for proposer, ls in (("nanotrack", "-"), ("stark", "--")):
        qs = ("0.8", "0.9", "0.95")
        se = [sens["alarm_quantile"][proposer]["verification"][q]["sensitivity_macro"] for q in qs]
        sp = [sens["alarm_quantile"][proposer]["verification"][q]["specificity_macro"] for q in qs]
        ax.plot([0.8, 0.9, 0.95], se, marker=markers[proposer], color=C_B, ls=ls, label=f"sensitivity ({PROPOSER_NAMES[proposer].split()[0]})")
        ax.plot([0.8, 0.9, 0.95], sp, marker=markers[proposer], color=C_A, ls=ls, label=f"specificity ({PROPOSER_NAMES[proposer].split()[0]})")
    ax.axvline(0.9, ls=":", color="#888888", lw=0.9)
    ax.set(xlabel="validation reliable-frame quantile for τ", title="(c) Alarm-quantile trade-off (fused)", xticks=[0.8, 0.9, 0.95], ylim=(0.3, 1.0))
    ax.legend(fontsize=6.6, loc="lower left")
    ax = axes[3]
    xs = np.arange(3)
    w = 0.38
    for k, (proposer, color) in enumerate((("nanotrack", C_A), ("stark", C_B))):
        emp = [conf[proposer]["verification"][d]["empirical_sequence_coverage"] for d in ("0.1", "0.2", "0.3")]
        bars = ax.bar(xs + (k - 0.5) * w, emp, w, color=color, zorder=3, label=f"{PROPOSER_NAMES[proposer]}: empirical")
        annotate(ax, bars, fmt="{:.2f}", dy=0.01, fontsize=6.8)
    ax.plot(xs, [0.9, 0.8, 0.7], "k_", markersize=26, mew=1.6, label="guaranteed 1 − δ", zorder=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(["δ = 0.1", "δ = 0.2", "δ = 0.3"])
    ax.set(ylabel="test sequences with specificity ≥ 0.90", title="(d) Conformal threshold coverage (Prop. 2)", ylim=(0, 1.32))
    ax.legend(fontsize=6.6, loc="upper left", ncol=1)
    fig.tight_layout()
    save(fig, "Fig_Sensitivity")


# ----------------------------------------------------------------------------- appendix fair localization
def fair_localization(tracking: dict) -> None:
    style()
    names_map = {"nanotrack": "NanoTrack", "csrt": "CSRT", "ostrack": "OSTrack", "stark": "STARK-ST50", "learnedgate": "LearnedGate", "ours_disjoint": "Prior Ours-SD"}
    order = {"chess": ("csrt", "ostrack", "stark", "nanotrack", "ours_disjoint"), "trackves": ("csrt", "ostrack", "stark", "nanotrack", "learnedgate", "ours_disjoint")}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.0), gridspec_kw={"width_ratios": [5, 6]})
    for ax, (domain, names) in zip(axes, order.items()):
        data = tracking[domain]["methods"]
        x = np.arange(len(names))
        w = 0.38
        iou = [data[n]["bbox_iou"]["mean"] for n in names]
        sd = [data[n]["bbox_iou"]["sample_sd"] for n in names]
        hota = [data[n]["pooled_official_trackeval"]["HOTA"] for n in names]
        b1 = ax.bar(x - w / 2, iou, w, yerr=sd, capsize=3, color=C_A, label="Box IoU (mean ± SD)", zorder=3, error_kw={"lw": 0.9})
        b2 = ax.bar(x + w / 2, hota, w, color=C_B, label="Official HOTA (pooled)", zorder=3)
        for bars in (b1, b2):
            for b, n in zip(bars, names):
                if n in ("ours_disjoint", "learnedgate"):
                    b.set_edgecolor(C_OURS_EDGE)
                    b.set_linewidth(1.4)
                elif n == "nanotrack":
                    b.set_edgecolor(C_NANO_EDGE)
                    b.set_linewidth(1.2)
        annotate(ax, b1, dy=0.025)
        annotate(ax, b2)
        ax.axhline(data["nanotrack"]["bbox_iou"]["mean"], ls="--", color=C_A, lw=0.9, alpha=0.7)
        ax.axhline(data["nanotrack"]["pooled_official_trackeval"]["HOTA"], ls="--", color=C_B, lw=0.9, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([names_map[n] for n in names], rotation=18, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title("(a) Controlled benchmark (11 sequences)" if domain == "chess" else "(b) TrackVes (9 sequences)")
    axes[0].set_ylabel("score")
    axes[1].legend(loc="upper right")
    fig.suptitle("Complete-denominator localization control (black edge: prior fusion variants; green edge: NanoTrack; dashed: NanoTrack level)", fontsize=10)
    fig.tight_layout()
    save(fig, "Fig_FairLocalization")


def main() -> None:
    results = json.loads((ROOT / "external_verification_summary.json").read_text())
    heldout = json.loads((ROOT / "heldout_ostrack_summary.json").read_text())
    supplementary = json.loads((ROOT / "supplementary_analyses.json").read_text())
    transfer = json.loads((ROOT / "transfer_verification_summary.json").read_text())
    tracking = json.loads((ROOT / "fair_tracking_summary.json").read_text())
    selected = json.loads((ROOT / "selected_verification.json").read_text())
    framework(selected)
    headline(results, heldout, supplementary, transfer)
    cue_ablation(results, heldout, supplementary)
    curves(selected)
    sensitivity(supplementary)
    fair_localization(tracking)
    print("REVISED_FIGURES_WRITTEN", sorted(p.name for p in OUT.glob("Fig_*.pdf")))


if __name__ == "__main__":
    main()
