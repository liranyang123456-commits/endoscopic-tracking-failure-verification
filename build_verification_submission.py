"""Assemble the revised verification manuscript, cover letter and figures.

Prose and table layouts live in ``manuscript_text.py``; this module draws the
pipeline figure, copies analysis figures, and writes the assembly manifest.
"""
from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

from build_fair_submission import ORIGINAL
from manuscript_text import TITLE, TRACKER_LABELS, cover, manuscript
from protocol import HERE, sha256, write_json


DEST = HERE.parent / "Mathematics_Verification_Submission_20261011"
LABELS = {
    "native_confidence": "Native confidence", "rank_only": "Rank only",
    "contour_only": "Contour only", "verification": "Fused verification",
}
PROPOSERS = {"stark": "STARK-ST50", "nanotrack": "NanoTrack"}
ANALYSIS_FIGURES = (
    "Fig_Framework.pdf", "Fig_Headline.pdf", "Fig_CueAblation.pdf", "Fig_Curves.pdf",
    "Fig_PairedSequences.pdf", "Fig_Sensitivity.pdf", "Fig_FairLocalization.pdf",
    "Verification_Qualitative.pdf",
)
STALE_FIGURES = (
    "Verification_Algorithm.pdf", "Verification_Ablation.pdf", "Verification_Transfer.pdf",
    "Verification_Pipeline.pdf", "Verification_Curves.pdf", "Verification_PairedSequences.pdf",
    "Fair_Localization.pdf",
)


def plots(results: dict, transfer: dict) -> None:
    """Draw the pipeline overview (Figure 1); result figures come from the analysis scripts."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axis = plt.subplots(figsize=(14.0, 8.0))
    axis.axis("off")
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 8)
    colors = {
        "reference": ("#eaf3fb", "#4c83ad"),
        "tracker": ("#f0ebfa", "#7c69a8"),
        "verifier": ("#e9f5ef", "#52876a"),
        "evaluation": ("#f5f5f5", "#777777"),
    }

    def panel(x, y, width, height, title, color, linestyle="-"):
        face, edge = colors[color]
        patch = Rectangle(
            (x, y), width, height, linewidth=1.8, edgecolor=edge,
            facecolor=face, linestyle=linestyle, zorder=0,
        )
        axis.add_patch(patch)
        axis.text(
            x + 0.16, y + height - 0.22, title,
            ha="left", va="top", fontsize=13, fontweight="bold", color=edge,
        )

    def block(x, y, width, height, title, detail, color, fontsize=11):
        _, edge = colors[color]
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            linewidth=1.3, edgecolor=edge, facecolor="white", zorder=2,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2, y + height * 0.64, title,
            ha="center", va="center", fontsize=fontsize, fontweight="bold", color=edge,
        )
        axis.text(
            x + width / 2, y + height * 0.32, detail,
            ha="center", va="center", fontsize=fontsize - 0.5, color="#263442",
        )

    def arrow(start, end, color="#5b6670", style="-", connectionstyle="arc3,rad=0"):
        axis.annotate(
            "", xy=end, xytext=start,
            arrowprops={
                "arrowstyle": "-|>", "lw": 1.4, "color": color,
                "linestyle": style, "connectionstyle": connectionstyle,
                "shrinkA": 2, "shrinkB": 2,
            },
            zorder=1,
        )

    axis.text(
        7, 7.83, "Non-invasive failure verification for region-level endoscopic tracking",
        ha="center", va="center", fontsize=16, fontweight="bold", color="#34495e",
    )

    panel(0.35, 4.48, 4.25, 2.95, "A   INITIAL REFERENCE (once)", "reference")
    block(0.62, 5.92, 1.02, 0.92, "Frame $I_0$", "ROI $R_0$", "reference", fontsize=9)
    block(1.87, 5.92, 1.02, 0.92, "Canonical crop", "resize to $64^2$", "reference", fontsize=8.5)
    block(3.12, 5.92, 1.20, 0.92, "Reference cues", "$U_0,C_0,h_0$", "reference", fontsize=8.8)
    arrow((1.65, 6.38), (1.82, 6.38), colors["reference"][1])
    arrow((2.90, 6.38), (3.07, 6.38), colors["reference"][1])
    axis.text(
        2.48, 5.30,
        "Fixed for the sequence:\nmidrank template, contour set,\nHSV histogram",
        ha="center", va="center", fontsize=10, color="#34495e",
    )
    axis.text(
        2.48, 4.76,
        "No online reference update",
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=colors["reference"][1],
    )

    panel(0.35, 1.55, 4.25, 2.55, "B   FROZEN PROPOSER (each frame $t$)", "tracker")
    block(0.65, 2.68, 1.10, 0.90, "Current frame", "$I_t$", "tracker", fontsize=8.8)
    block(2.05, 2.68, 1.14, 0.90, "Official tracker", "frozen weights", "tracker", fontsize=8.2)
    block(3.48, 2.68, 0.85, 0.90, "Proposal", "$R_t,c_t$", "tracker", fontsize=8.5)
    arrow((1.77, 3.13), (2.00, 3.13), colors["tracker"][1])
    arrow((3.21, 3.13), (3.43, 3.13), colors["tracker"][1])
    axis.text(
        2.48, 2.15,
        "The box is forwarded unchanged;\nno reset, suppression, or overwrite",
        ha="center", va="center", fontsize=8.8, color="#34495e",
    )

    panel(4.95, 1.55, 8.70, 5.88, "C   NON-INVASIVE VERIFIER (fixed descriptors; no verifier network)", "verifier")
    block(5.30, 4.86, 2.20, 0.94, "Current ROI evidence", "$I_t[R_t]$, rank, contour,\nHSV and confidence $c_t$", "verifier", fontsize=10)
    block(8.12, 4.86, 2.10, 0.94, "Compare with reference", "$d_C, d_U, d_H, d_N$", "verifier", fontsize=9.5)
    block(10.86, 4.86, 2.38, 0.94, "Fuse and threshold", "$r_t=\\mathbf{w}^{\\top}\\mathbf{d}_t$;\nwarn if $r_t>\\tau$", "verifier", fontsize=9.5)
    arrow((7.54, 5.33), (8.06, 5.33), colors["verifier"][1])
    arrow((10.27, 5.33), (10.80, 5.33), colors["verifier"][1])

    # The reference path runs above the current-cue row to avoid crossing either input branch.
    axis.plot([4.34, 7.54, 9.16], [6.82, 6.82, 6.82],
              color=colors["reference"][1], lw=1.3, zorder=1)
    arrow((9.16, 6.82), (9.16, 5.82), colors["reference"][1])
    axis.text(6.95, 6.88, "fixed $U_0,C_0,h_0$", ha="center", va="bottom",
              fontsize=9.5, color=colors["reference"][1])

    block(8.00, 3.55, 4.92, 0.82, "LOCKED ON VALIDATION BEFORE TEST",
          "contour percentile, fusion weights $\\mathbf{w}$, and reliable-frame alarm threshold $\\tau$",
          "verifier", fontsize=9)
    arrow((10.46, 4.41), (12.0, 4.81), colors["verifier"][1], style="--")

    block(5.45, 2.25, 3.05, 0.92, "TRACKER BOX OUTPUT", "return the original $R_t$ unchanged", "tracker", fontsize=9.5)
    block(9.16, 2.25, 3.00, 0.92, "VERIFIER OUTPUT", "$r_t$ and optional warning $\\mathbf{1}\\{r_t>\\tau\\}$", "verifier", fontsize=9.5)
    arrow((4.34, 3.13), (5.38, 2.88), colors["tracker"][1], connectionstyle="arc3,rad=-.06")
    axis.plot([13.24, 13.48, 13.48, 12.20], [5.33, 5.33, 2.71, 2.71],
              color=colors["verifier"][1], lw=1.35, zorder=1)
    arrow((12.20, 2.71), (12.13, 2.71), colors["verifier"][1])
    axis.plot([1.20, 1.20, 4.75], [3.58, 4.34, 4.34],
              color=colors["tracker"][1], lw=1.25, linestyle=(0, (3, 2)), zorder=1)
    arrow((4.75, 4.34), (5.42, 4.82), colors["tracker"][1], style="--")
    axis.text(3.00, 4.20, "current-frame image branch", ha="center", va="center",
              fontsize=8.7, color=colors["tracker"][1])

    panel(0.35, 0.20, 13.30, 1.05, "OFFLINE LABEL USE  |  not available to the online verifier", "evaluation", "--")
    axis.text(
        7.0, 0.70,
        "Validation GT selects $\\mathbf{w}$ and $\\tau$; held-out test GT defines failure by $\\mathrm{IoU}(R_t,G_t)<0.5$.",
        ha="center", va="center", fontsize=10, color="#454545",
    )
    axis.text(
        7.0, 0.40,
        "Neither validation nor test ground truth is read during online descriptor extraction or warning generation.",
        ha="center", va="center", fontsize=9, color="#555555",
    )
    fig.savefig(DEST / "images" / "Verification_Pipeline.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    # The former algorithm-overview image is superseded by the two-phase
    # algorithm environment in the manuscript; a stale copy must not be shipped.
    (DEST / "images" / "Verification_Algorithm.pdf").unlink(missing_ok=True)


def fair_localization_figure(tracking: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    order = {
        "chess": ("nanotrack", "csrt", "ostrack", "stark", "ours_disjoint"),
        "trackves": ("nanotrack", "csrt", "ostrack", "stark", "learnedgate", "ours_disjoint"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [5, 6]})
    for axis, (domain, names) in zip(axes, order.items()):
        data = tracking[domain]["methods"]
        x = np.arange(len(names))
        axis.bar(x, [data[n]["bbox_iou"]["mean"] for n in names], yerr=[data[n]["bbox_iou"]["sample_sd"] for n in names],
                 color="#7fa6c9", capsize=3, label="box IoU (equal-sequence mean, SD)")
        axis.plot(x, [data[n]["pooled_official_trackeval"]["HOTA"] for n in names], "o", color="#2f4f6f", label="pooled official HOTA")
        axis.set_xticks(x)
        axis.set_xticklabels([TRACKER_LABELS[n].replace(" (official)", "\n(official)") for n in names], fontsize=7.5)
        axis.set_ylim(0, 1)
        axis.set_title("Controlled benchmark" if domain == "chess" else "TrackVes")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("score")
    axes[1].legend(fontsize=7.5, loc="upper right")
    fig.savefig(DEST / "images" / "Fair_Localization.pdf")
    plt.close(fig)


def cpu_name() -> str:
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command",
                                       "(Get-CimInstance Win32_Processor).Name"], text=True, timeout=30).strip()
        if out:
            return re.sub(r"\s+", " ", out).replace("(R)", "").replace("(TM)", "").strip()
    except Exception:
        pass
    return platform.processor() or "unspecified"


def main() -> None:
    root = HERE / "results"
    results = json.loads((root / "external_verification_summary.json").read_text())
    transfer = json.loads((root / "transfer_verification_summary.json").read_text())
    segmentation = json.loads((root / "cholecseg_grouped.json").read_text())
    tracking = json.loads((root / "fair_tracking_summary.json").read_text())
    selected = json.loads((root / "selected_verification.json").read_text())
    supplementary = json.loads((root / "supplementary_analyses.json").read_text())
    heldout = json.loads((root / "heldout_ostrack_summary.json").read_text())
    if not transfer.get("four_cue_monitor_with_native_confidence"):
        raise ValueError("Transfer summary predates the confidence-logging reruns; run run_transfer_verification.py first")
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / "images").mkdir(exist_ok=True)
    shutil.copytree(ORIGINAL / "Definitions", DEST / "Definitions", dirs_exist_ok=True)
    archived = DEST / "images" / "archived"
    if archived.is_dir():
        shutil.rmtree(archived)
    for stale in STALE_FIGURES:
        (DEST / "images" / stale).unlink(missing_ok=True)
    for name in ANALYSIS_FIGURES:
        source = root / "figures" / name
        if not source.is_file():
            raise FileNotFoundError(f"Figure missing; run make_revised_figures.py / make_qualitative_figure.py: {source}")
        shutil.copy2(source, DEST / "images" / name)
    bibliography = (ORIGINAL / "mybibfile.bib").read_text(encoding="utf-8")
    bibliography += "\n" + (HERE / "verification_references.bib").read_text(encoding="utf-8")
    marker = "@article{cartucho2024surgt,"
    first = bibliography.index(marker)
    last = bibliography.index("\n}", first)
    if "doi=" not in bibliography[first:last].lower():
        bibliography = bibliography[:last] + ",\n  doi={10.1016/j.media.2023.102985}" + bibliography[last:]
        bibliography = bibliography.replace("publisher={Elsevier},,", "publisher={Elsevier},")
    (DEST / "mybibfile.bib").write_text(bibliography, encoding="utf-8")
    text = manuscript(results, transfer, segmentation, tracking, selected, supplementary, heldout, cpu_name())
    text = re.sub(r"\\\\(cite|ref|eqref)(?=\{)", r"\\\1", text)
    for stale in ("highest HOTA", "highest observed mean", "submitted to Sensors", "declined", "Highlights"):
        if stale in text:
            raise ValueError(f"Stale claim or removed section in final manuscript: {stale}")
    (DEST / "Manuscript_Mathematics.tex").write_text(text, encoding="utf-8")
    (DEST / "Cover_Letter_Mathematics.tex").write_text(cover(results, heldout), encoding="utf-8")
    for stale in STALE_FIGURES:
        (DEST / "images" / stale).unlink(missing_ok=True)
    write_json(DEST / "verification_assembly_manifest.json", {
        "title": TITLE,
        "manuscript_sha256": sha256(DEST / "Manuscript_Mathematics.tex"),
        "external_results_sha256": sha256(root / "external_verification_summary.json"),
        "selected_protocol_sha256": sha256(root / "selected_verification.json"),
        "transfer_results_sha256": sha256(root / "transfer_verification_summary.json"),
        "supplementary_analyses_sha256": sha256(root / "supplementary_analyses.json"),
        "heldout_proposer_sha256": sha256(root / "heldout_ostrack_summary.json"),
        "negative_tracking_results_retained": True, "AI_assistance_disclosed": True,
        "main_claim": "failure verification around unchanged proposals, not improved tracking localization",
    })
    print(DEST)


if __name__ == "__main__":
    main()
