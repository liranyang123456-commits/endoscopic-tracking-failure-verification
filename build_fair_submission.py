from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from protocol import HERE, read_csv, sha256, write_json


REVISION = HERE.parent
ORIGINAL = REVISION / "Mathematics_Submission"
DESTINATION = REVISION / "Mathematics_Fair_Submission_20261011"
TITLE = "Annotation-Free Contour Verification for Region-Level Tracking of Deforming Soft Tissue in Endoscopy"


def replace_span(text: str, start: str, end: str, replacement: str) -> str:
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"Ambiguous document assembly anchors: {start!r} / {end!r}")
    first = text.index(start)
    last = text.index(end, first)
    return text[:first] + replacement + "\n\n" + text[last:]


def table(caption: str, label: str, headings: list[str], rows: list[list[str]]) -> str:
    return (
        "\\begin{table}[htbp]\n\\centering\n"
        + f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        + "\\footnotesize\n\\setlength{\\tabcolsep}{3pt}\n"
        + "\\resizebox{\\linewidth}{!}{%\n"
        + "\\begin{tabular}{" + "l" * len(headings) + "}\n\\toprule\n"
        + " & ".join(headings) + r" \\" + "\n\\midrule\n"
        + "\n".join(" & ".join(row) + r" \\" for row in rows)
        + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
    )


def pvalue(value: float) -> str:
    return f"{value:.4f}" if value >= 0.0001 else f"{value:.2e}"


def mean_sd(values: dict) -> str:
    return f"${values['mean']:.3f}\\pm{values['sample_sd']:.3f}$"


def plot_results(segmentation: dict, tracking: dict) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    names = ["Ours-Contour", "MedSAM", "SAM2.1", "YOLO11m-seg"]
    fig, axis = plt.subplots(figsize=(7.5, 3.8), constrained_layout=True)
    axis.bar(
        names, [segmentation["methods"][name]["dice"]["video_macro_mean"] for name in names],
        yerr=[segmentation["methods"][name]["dice"]["video_sample_sd"] for name in names],
        color="#577a98", capsize=4,
    )
    axis.set(ylim=(0, 1), ylabel="Dice: equal-video mean +/- sample SD",
             title="CholecSeg8k: all 8080 frames; missing masks score zero")
    axis.grid(axis="y", alpha=0.2)
    fig.savefig(DESTINATION / "images" / "Fair_CholecSeg.pdf")
    plt.close(fig)
    for domain in ("chess", "trackves"):
        names = ["nanotrack", "ostrack", "stark", "csrt", "ours_disjoint"]
        if domain == "trackves":
            names.append("learnedgate")
        labels = ["NanoTrack", "OSTrack\n(official)", "STARK-ST50\n(official)", "CSRT", "Ours-SD"]
        if domain == "trackves":
            labels.append("LearnedGate\n(LOSO)")
        data = tracking[domain]["methods"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
        axes[0].bar(
            labels, [data[n]["bbox_iou"]["mean"] for n in names],
            yerr=[data[n]["bbox_iou"]["sample_sd"] for n in names],
            color="#577a98", capsize=3,
        )
        axes[0].set(ylim=(0, 1.04), ylabel="Box IoU: sequence mean +/- sample SD")
        x = np.arange(len(names))
        axes[1].bar(x - 0.18, [data[n]["pooled_official_trackeval"]["HOTA"] for n in names],
                    width=0.36, label="HOTA", color="#577a98")
        axes[1].bar(x + 0.18, [data[n]["pooled_official_trackeval"]["AssA"] for n in names],
                    width=0.36, label="AssA", color="#8ea480")
        axes[1].set_xticks(x, labels)
        axes[1].set(ylim=(0, 1.04), ylabel="Pooled official TrackEval score")
        axes[1].legend(frameon=False)
        for axis in axes:
            axis.grid(axis="y", alpha=0.2)
        fig.suptitle("Controlled chessboard benchmark" if domain == "chess" else "Public TrackVes benchmark")
        fig.savefig(DESTINATION / "images" / f"Fair_{domain}.pdf")
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6)
    axis.axis("off")
    nodes = [
        (1.7, 4.8, "Offline gradient quantile\nq = empirical 68th percentile\nContour delineation / auxiliary study"),
        (1.7, 1.6, "Online raster\nfixed gray cutoff = 68\nImage-derived contour templates"),
        (5.0, 1.6, "NanoTrack proposals\n+ Siamese contour similarity\n+ HSV / spatial evidence"),
        (8.2, 1.6, "Conservative arbitration\nROI box trajectory\nnot a dense deformation field"),
        (6.5, 4.8, "Separate TrackVes learned gate\nNano + template NCC + ORB\nsequence-LOSO quality prediction"),
    ]
    for x, y, label in nodes:
        axis.text(x, y, label, ha="center", va="center", fontsize=10,
                  bbox={"boxstyle": "round,pad=0.7", "fc": "#edf2f5", "ec": "#577a98"})
    for x1, x2 in ((3.0, 3.5), (6.5, 6.8)):
        axis.annotate("", xy=(x2, 1.6), xytext=(x1, 1.6),
                      arrowprops={"arrowstyle": "->", "lw": 1.5})
    axis.text(5, 0.35, "No arrow connects the offline gradient/autoencoder branch to online inference.",
              ha="center", fontsize=10)
    fig.savefig(DESTINATION / "images" / "Fair_Pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def experiments(segmentation: dict, tracking: dict, legacy: dict) -> str:
    section = r"""
\section{Experiments and Results}
\label{sec:experiments}
The fairness audit uses the public CholecSeg8k and TrackVes datasets and the released controlled chessboard sequence inventory. New tracking runs were performed locally on an NVIDIA GeForce RTX 5090 Laptop GPU, not on the historical A100 environment. Model configurations, author-release checkpoint hashes, fold manifests, per-frame outputs, and statistical scripts are recorded in the accompanying reproducibility package.

\subsection{Fixed evaluation sets and independent statistical units}
CholecSeg8k contains 8080 evaluated frames grouped by 17 video IDs. Video IDs, rather than unverified patient identities or individual frames, are the independent units for paired tests and confidence intervals. A missing predicted mask receives Dice and IoU zero. The primary summary is the equal-video mean and sample SD; frame-weighted means are secondary descriptive values. MedSAM uses the retained box-prompted protocol, whereas the other segmentation configurations use their retained zero-shot protocols; this is not a matched prompting-budget experiment.

The controlled inventory contains 11 sequences, 22 target trajectories, and 3848 GT records. Each supplied first target box is an initialization instruction rather than an accuracy observation; removing all 22 initialization observations leaves 3826 scored records. Three targets have only an initialization record and no subsequent observation, so 19 targets contribute tracking scores. They remain in the inventory and are not reported as tracking successes. Boards are averaged equally within each sequence and sequences equally across the benchmark. A rigid chessboard scene tests motion, perspective, scale, and association mechanics; it does not establish soft-tissue biomechanics or clinical navigation validity.

TrackVes contains nine sequences and 5931 GT records, of which 5922 are scored after the nine supplied initialization boxes. Its source videos produce H.264 decoding warnings. A single sequential, single-thread decoding pass was therefore saved as lossless PNG frames, with every annotated frame retained and every PNG round trip checked. All newly reported methods use this identical frozen image set; pixel hashes are verified against the common manifest. GT polygons are rasterized to bounding boxes with the retained two-pixel padding convention. Updates follow the same GT-frame protocol for all methods, not an assertion of dense every-video-frame tracking.

\subsection{Contour delineation with grouped inference}
\label{sec:contour-comparison}
\label{sec:medseg-stats}
The Ours-Contour statistics use saved gradient-percentile contour responses followed by the documented contour-fill postprocessing. The fill script's binarization level on the saved response image is not the adaptive gradient quantile and is not the online tracker gray cutoff. Table~\ref{tab:MedicalSegStats} reports all frames, including failures. In particular, SAM2.1 has 2470 missing masks and YOLO11m-seg 1373; omitting these frames would incorrectly increase their scores. No supervised U-Net value from a different test set is inserted into this comparison.
"""
    rows = []
    for name in ("Ours-Contour", "MedSAM", "SAM2.1", "YOLO11m-seg"):
        values = segmentation["methods"][name]
        rows.append([
            name,
            f"${values['dice']['video_macro_mean']:.3f}\\pm{values['dice']['video_sample_sd']:.3f}$",
            f"${values['iou']['video_macro_mean']:.3f}\\pm{values['iou']['video_sample_sd']:.3f}$",
            f"{values['dice']['frame_mean']:.3f}",
            str(values["missing_masks"]),
        ])
    section += table(
        "CholecSeg8k: all 8080 frames and 17 video groups. The first two score columns are equal-video mean and sample SD; frame Dice is descriptive. Missing masks score zero. Prompting and pretraining differ across configurations.",
        "tab:MedicalSegStats",
        ["Configuration", "Video Dice", "Video IoU", "Frame Dice", "Missing masks"], rows,
    )
    tests = [
        [
            row["comparator"], row["metric"].capitalize(),
            f"{row['mean_paired_difference']:+.3f}",
            f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]",
            pvalue(row["p_two_sided"]), pvalue(row["p_holm"]),
        ]
        for row in segmentation["paired_tests"]
    ]
    section += table(
        "Paired equal-video effects for Ours-Contour minus each comparator. Confidence intervals resample 17 video groups 20000 times with fixed seed 20261011. Two-sided Wilcoxon tests are Holm-adjusted over the six comparator-by-metric tests. These are retrospective audit analyses, not preregistered confirmatory tests.",
        "tab:GroupedSegTests", ["Comparator", "Metric", "Difference", "95\\% CI", "$p$", "$p_{\\mathrm{Holm}}$"], tests,
    )
    medsam = next(row for row in segmentation["paired_tests"] if row["comparator"] == "MedSAM" and row["metric"] == "dice")
    section += (
        f"Ours-Contour has video-macro Dice {segmentation['methods']['Ours-Contour']['dice']['video_macro_mean']:.3f} "
        f"versus {segmentation['methods']['MedSAM']['dice']['video_macro_mean']:.3f} for MedSAM. "
        f"The paired video-level difference is {medsam['mean_paired_difference']:.3f} "
        f"(95\\% CI [{medsam['ci95_low']:.3f}, {medsam['ci95_high']:.3f}]; "
        f"Holm-adjusted $p={pvalue(medsam['p_holm'])}$). "
        "This delineation result does not establish improved temporal tracking or general segmentation superiority.\n"
        r"""
\begin{figure}[htbp]
\centering\includegraphics[width=.9\linewidth]{images/Fair_CholecSeg.pdf}
\caption{Video-grouped CholecSeg8k results from the same complete frame set as Table~\ref{tab:MedicalSegStats}. Error bars are sample SD across video means, not a confidence interval over independent frames.}
\label{fig:FigResultsMedSeg}
\end{figure}

\subsection{Official tracking checkpoints and fully held-out contour verification}
\label{sec:tracking-qual}
\label{sec:iou-tracking}
\label{sec:auc-hota}
The primary tracking comparison uses author-released OSTrack ViT-B-256 with candidate elimination and STARK-ST50 checkpoints, loaded with strict matching of all backbone and task-head keys. NanoTrack uses the public OpenCV ONNX pair; CSRT uses an isolated OpenCV-contrib build with the same version as the other runs. The official tracker crop, RGB normalization, box head, and update logic are retained. No baseline is locally fine-tuned. Old untrained-head proxies are excluded from this comparison and are documented separately in Appendix~\ref{sec:legacy-audit}.

Ours-SD uses a distinct, randomly initialized ContourSimilarityNet checkpoint for every outer held-out sequence. Each fold has 17 training sequence groups, two validation groups (one per domain), and one test group; no test sequence enters training, validation selection, normalization, or checkpoint initialization. Training inputs are image-derived gray-threshold rasters inside training ROI boxes, not human polygon masks. The auxiliary encoder--decoder and externally fitted fusion heads are not loaded, and online neural fine-tuning is disabled. Each inference run verifies that the loaded contour and confidence checkpoint matches the designated held-out fold. This comparison has different local training budgets from the frozen baselines and should not be called a matched-training-budget SOTA experiment.
"""
    )
    labels = {
        "nanotrack": "NanoTrack", "ostrack": "OSTrack (official)",
        "stark": "STARK-ST50 (official)", "csrt": "CSRT",
        "ours_disjoint": "Ours-SD", "learnedgate": "LearnedGate (LOSO)",
    }
    for domain in ("chess", "trackves"):
        if domain == "trackves":
            section += r"""
\subsection{Public TrackVes results and active-component isolation}
\label{sec:trackves}
\label{sec:learnedgate}
The TrackVes LearnedGate is a separate NanoTrack/template-NCC/ORB candidate-quality implementation: it does not use ContourSimilarityNet or the auxiliary encoder--decoder. Its only locally learned inference component is the 17073-parameter VisualQualityGate. For each of nine LOSO folds, seven sequences train the gate, one validates both checkpoint and intervention margin, and the remaining sequence is evaluated online. Scalar normalization is fitted on training groups only. The feature arrays are rebuilt from the deterministic shared image set rather than inherited from legacy fusion candidates. Thus, its active components are sequence-isolated without assuming that a frozen contour encoder is present.
"""
        data = tracking[domain]
        methods = data["methods"]
        order = ["nanotrack", "ostrack", "stark", "csrt", "ours_disjoint"]
        if domain == "trackves":
            order.append("learnedgate")
        rows = [
            [
                labels[name], mean_sd(methods[name]["bbox_iou"]),
                mean_sd(methods[name]["success_05"]), mean_sd(methods[name]["precision_20"]),
                f"{methods[name]['pooled_official_trackeval']['HOTA']:.3f}",
                f"{methods[name]['pooled_official_trackeval']['AssA']:.3f}",
                str(methods[name]["n_missing_predictions"]),
            ]
            for name in order
        ]
        label = "tab:TrackingIoU" if domain == "chess" else "tab:TrackVesIoU"
        title = "Controlled 11-sequence benchmark" if domain == "chess" else "TrackVes nine-sequence benchmark"
        section += table(
            title + f": {data['scored_frames']} scored GT observations after all initialization frames are removed. "
            "IoU, Success@0.5 and Precision@20 are sequence mean and sample SD. HOTA and AssA are pooled official TrackEval scores. "
            "Missing predictions count as misses; no tracker-specific failed sequence is removed.",
            label,
            ["Method", "Box IoU", "Success@0.5", "Precision@20", "HOTA", "AssA", "Missing"],
            rows,
        )
        tests = [
            [
                labels[row["reference"]], labels[row["comparator"]],
                "Box IoU" if row["metric"] == "bbox_iou" else row["metric"],
                f"{row['mean_paired_difference']:+.3f}",
                f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]",
                pvalue(row["p_holm"]),
            ]
            for row in data["paired_tests"]
        ]
        section += table(
            title + " paired sequence effects. Positive differences favor the reference. HOTA/AssA effects use sequence summaries, not subtraction of pooled scores. "
            "Percentile CIs resample complete sequences 20000 times; Holm adjustment covers all declared tests within this dataset. "
            "Small samples, overlapping cross-fit training folds, and retrospective model development limit confirmatory interpretation.",
            "tab:FairTests" + domain,
            ["Reference", "Comparator", "Metric", "Difference", "95\\% CI", "$p_{\\mathrm{Holm}}$"],
            tests,
        )
        ours, nano = methods["ours_disjoint"], methods["nanotrack"]
        section += (
            f"Ours-SD has sequence-macro box IoU {ours['bbox_iou']['mean']:.3f}, "
            f"compared with {nano['bbox_iou']['mean']:.3f} for NanoTrack, "
            f"{methods['ostrack']['bbox_iou']['mean']:.3f} for official OSTrack, "
            f"and {methods['stark']['bbox_iou']['mean']:.3f} for official STARK-ST50. "
            "The official-checkpoint rows supersede the old proxy ranking; no claim of overall tracking superiority is made.\n"
            + "\\begin{figure}[htbp]\n\\centering"
            + f"\\includegraphics[width=\\linewidth]{{images/Fair_{domain}.pdf}}\n"
            + "\\caption{New runs on the fixed GT set. IoU bars show sequence mean and sample SD; "
            + "HOTA/AssA bars are pooled official TrackEval scores. All methods read the same image bytes.}\n"
            + f"\\label{{fig:Fair{domain}}}\n\\end{{figure}}\n"
        )
        if domain == "trackves":
            learned = methods["learnedgate"]
            effect = next(r for r in data["paired_tests"] if r["reference"] == "learnedgate" and r["comparator"] == "nanotrack" and r["metric"] == "bbox_iou")
            section += (
                f"LearnedGate has box IoU {learned['bbox_iou']['mean']:.3f}; its paired gain over NanoTrack is "
                f"{effect['mean_paired_difference']:+.3f} (95\\% CI [{effect['ci95_low']:+.3f}, {effect['ci95_high']:+.3f}]; "
                f"Holm-adjusted two-sided $p={pvalue(effect['p_holm'])}$). "
                f"{effect['improved']} sequences improve, {effect['degraded']} degrade, and {effect['ties']} tie. "
                "This supervised extension does not establish a generalizable improvement.\n"
            )
    section += r"""
\subsection{Contribution isolation and failure analysis}
\label{sec:ablation}
The NanoTrack versus Ours-SD comparison isolates the aggregate addition of contour verification and arbitration under the common GT protocol; it does not isolate every fusion coefficient or establish that the auxiliary latent encoder improves tracking. The historical contour+color and contour+latent configurations used different learned assets and are not mixed into the new sequence-isolated main table. A controlled leave-one-cue retraining experiment remains necessary for causal attribution to individual cues.

Failure modes include contour fragmentation from specular highlights, loss of discriminative texture, abrupt scale change, and occlusion. Official pretrained trackers can outperform the proposed configuration on these settings. Video freezing removes algorithm-to-algorithm input differences but cannot reconstruct information already corrupted in a source video. All reported intervals condition on the available sequences and a single training seed per fold; they do not measure between-seed or prospective clinical uncertainty.
"""
    return section


def assemble(segmentation: dict, tracking: dict, legacy: dict) -> None:
    source_path = ORIGINAL / "Manuscript_Mathematics.tex"
    source = source_path.read_text(encoding="utf-8")
    source_hash = sha256(source_path)
    shutil.copytree(ORIGINAL / "Definitions", DESTINATION / "Definitions", dirs_exist_ok=True)
    for name in ("Fig4a_NetEncoderDecoder.pdf", "Fig4b_NetContourSim.pdf"):
        shutil.copy2(ORIGINAL / "images" / name, DESTINATION / "images" / name)
    shutil.copy2(ORIGINAL / "mybibfile.bib", DESTINATION / "mybibfile.bib")
    figure_caption = (
        r"\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{images/Fair_Pipeline.pdf}"
        "\n" + r"\caption{Separated offline and online paths. Adaptive gradient quantiles produce offline contour cues; "
        r"the online tracker uses fixed-gray rasters and a fold-specific Siamese verifier. "
        r"The separate TrackVes learned gate uses NanoTrack/NCC/ORB and no contour network.}"
        "\n" + r"\label{fig:Fig1}\end{figure}"
    )
    prefix = source[:source.index(r"\section{Experiments and Results}")]
    group_dice = segmentation["methods"]["Ours-Contour"]["dice"]
    tv = tracking["trackves"]["methods"]
    abstract = (
        "Region-level tracking of deforming soft tissue in endoscopy is challenged by weak texture, "
        "specular highlights, and costly pixel labels. We study contour verification as an arbitration "
        "layer around NanoTrack, distinguishing offline gradient-percentile delineation from online "
        "fixed-gray contour rasters. Image-derived contour labels require no manual pixel masks, "
        "although ROI boxes and the learned quality gate are supervised. "
        "All active learned components are trained and selected without their held-out test sequence. "
        "Evaluation retains every scored GT frame, counts missing outputs as misses, and uses "
        "video-level paired effects, bootstrap intervals, and Holm-adjusted Wilcoxon tests. "
        f"On CholecSeg8k, Ours-Contour obtains video-macro Dice {group_dice['video_macro_mean']:.3f} "
        f"(SD {group_dice['video_sample_sd']:.3f}) over 17 videos and 8080 frames. "
        "Official pretrained OSTrack and STARK-ST50 replace untrained-head proxies in the tracking comparison. "
        f"On nine TrackVes sequences, Ours-SD and the LOSO quality gate obtain mean box IoU "
        f"{tv['ours_disjoint']['bbox_iou']['mean']:.3f} and {tv['learnedgate']['bbox_iou']['mean']:.3f}, "
        f"respectively, versus {tv['nanotrack']['bbox_iou']['mean']:.3f} for NanoTrack. "
        "The results delimit where verification helps and fails; they do not establish uniform tracking "
        "superiority, annotation-free initialization, or dense deformation estimation."
    )
    prefix = replace_span(prefix, r"\abstract{", r"\keyword{", "\\abstract{" + abstract + "}")
    highlights = (
        r"\section*{Highlights}" + "\n"
        r"\noindent\textbf{What are the main findings?}\begin{itemize}[leftmargin=*]" + "\n"
        r"\item Video-grouped contour delineation retains all 8080 CholecSeg8k frames, including missing-mask failures." + "\n"
        r"\item Official tracking checkpoints and sequence-isolated learned components replace proxy and in-domain rankings; no overall superiority claim is made." + "\n"
        r"\end{itemize}\noindent\textbf{What are the implications?}\begin{itemize}[leftmargin=*]" + "\n"
        r"\item Contour generation without human pixel masks does not remove ROI initialization or supervised gate training." + "\n"
        r"\item Shared GT denominators, deterministic image inputs, and independent-group statistics are essential to interpreting verification layers." + "\n"
        r"\end{itemize}"
    )
    prefix = replace_span(prefix, r"\section*{Highlights}", r"\section{Introduction}", highlights)
    introduction = r"""
\section{Introduction}
\label{sec:introduction}
Keeping a region of interest attached to a deforming soft-tissue target is relevant to endoscopic safety-area overlays. Weak texture, specular reflections, occlusion, and non-rigid appearance changes can make both the target boundary and its temporal location ambiguous. Dense manual pixel labels are costly, and contour cues alone need not provide reliable tracking.

This study examines whether contour and color verification can arbitrate proposals from a lightweight tracker rather than replacing its temporal localization mechanism. The problem is explicitly ROI-level: an initial target box is supplied, and the output is a box trajectory, not a displacement field, biomechanical estimate, or dense point correspondence. The public TrackVes safety-area setting~\cite{PENZA201813,penza2017trackvesdata} provides a relevant soft-tissue evaluation domain, while controlled chessboard sequences serve only as an engineering motion test.

The implementation has distinct paths. An empirical gradient quantile generates offline contour cues for region delineation and an auxiliary reconstruction architecture. Online tracking instead rasterizes intensity with a fixed gray cutoff and uses a Siamese contour representation plus color/spatial evidence around NanoTrack. A separate supervised TrackVes quality gate scores NanoTrack, template-matching, and ORB candidates without loading the contour network. These distinctions matter when assessing both mathematical motivation and supervision.

The contributions are: (i) a contour-verification/arbitration formulation with an explicit separation of pixel-label generation from ROI supervision; (ii) an implementation and rerun with author-released pretrained baselines and held-out-sequence training of every active learned component; and (iii) a grouped evaluation retaining missing outputs, paired effect sizes and confidence intervals, corrected significance tests, and negative findings. Gaussian smoothing, percentile thresholding, Siamese encoding, and standard statistical tests are established tools, not claimed here as new standalone theories. The study assesses the integrated configuration and its limitations rather than claiming general tracking dominance.
"""
    prefix = replace_span(prefix, r"\section{Introduction}", r"\section{Related Work}", introduction.strip())
    prefix = prefix.replace(
        "we include coverage-aware IoU comparisons against several of these trackers in Section",
        "we compare author-released tracking checkpoints on a fixed GT set in Section",
    )
    methods = r"""
\section{Materials and Methods}
\paragraph{Notation and scope.}
An RGB image is $\mathbf{I}\in\mathbb{R}^{3\times H\times W}$, and $R_t$ is the predicted ROI box. The offline empirical gradient quantile $t_{0.68}^{\mathrm{grad}}$ is distinct from the online intensity cutoff $t_{\mathrm{gray}}=68$. Both produce image-derived cues without human pixel masks, but the sequence-isolated contour training uses annotated ROI boxes. The learned gate is supervised by training-sequence box quality. Evaluation GT masks are never supplied as inference inputs beyond deriving the initial box.
"""
    prefix = replace_span(prefix, r"\section{Materials and Methods}", r"\subsection{Adaptive gradient-percentile contour extraction}", methods.strip() + "\n" + figure_caption)
    prefix = prefix.replace(
        "and as contour templates for the online verifier, removing manual pixel annotation from the pipeline.",
        "for offline analysis; the online verifier instead uses the fixed-gray raster described below.",
    )
    quantile = r"""
\subsection{Adaptive gradient-percentile contour extraction}
\label{sec:gradient-detector}
The archived CPU delineation implementation first scales grayscale intensities to $[0,1]$, applies bilateral smoothing (diameter 7, intensity scale 0.1, spatial scale 3.0), and computes horizontal/vertical $3\times3$ Sobel responses. Let $m_p$ be the resulting min--max-normalized gradient magnitude and define
\begin{equation}
\widehat F_m(x)=\frac{1}{HW}\sum_p\mathbf{1}\{m_p\leq x\},
\qquad t_\alpha^{\mathrm{grad}}=\inf\{x:\widehat F_m(x)\geq\alpha\}.
\end{equation}
Ignoring histogram quantization, this is the order statistic $m_{(\lceil\alpha HW\rceil)}$. The released implementation uses 1024 histogram bins over $[0,1]$ and the center of the first bin whose cumulative count reaches the fixed percentile $\alpha=0.68$. Gradients are converted to 8-bit intensity, responses below the quantile are truncated, and retained nonzero responses are rescaled before contour extraction. The contour-fill postprocessor acts on the saved response image. Its binarization value 68 is a response-map setting, not a raw-image intensity threshold or a new gradient quantile.

This is a single-image contour-cue extractor, not temporal deformation estimation. Bilateral smoothing is nonlinear, and signed linear-gradient null models do not establish a Gaussian law for the observed nonnegative magnitudes. More generally, for a linear response $S=\sum_p w_p X_p$, its variance includes spatial covariance:
\begin{equation}
\mathrm{Var}(S)=\sum_p\sum_q w_p w_q\mathrm{Cov}(X_p,X_q).
\end{equation}
Dropping off-diagonal terms would require an additional independence approximation, not established for endoscopic pixels. Accordingly, the method uses the empirical histogram rather than a calibrated Gaussian probability. The 68th percentile is a heuristic setting; it is not identified with the central $\pm1\sigma$ probability or claimed as an optimally derived deformation threshold.

For online tracking, the binary cue is instead $B_t(p)=\mathbf{1}\{I_{\mathrm{gray},t}(p)>68\}$ for 8-bit intensities. The gray value is fixed rather than an empirical percentile. The contour-fill postprocessor on saved offline response maps also has its own documented binarization setting; it must not be confused with either raw-image cutoff or gradient quantile.
"""
    prefix = replace_span(
        prefix, r"\subsection{Adaptive gradient-percentile contour extraction}",
        r"\subsection{Auxiliary latent encoding of contour regions}", quantile.strip()
    )
    auxiliary = r"""
\subsection{Auxiliary latent encoding of contour regions}
\label{sec:latent-encoding}
The residual/self-attention encoder--decoder is an auxiliary architecture for reconstructing image-derived contour responses. It is not loaded by Ours-SD, LearnedGate, or any official tracking baseline, and no unverified legacy latent-ablation result is used to support the new tracking comparison. Its reference design uses five encoder widths $64,128,256,512,1024$, residual blocks and windowed attention, followed by convolutional upsampling and skip connections. Figure~\ref{fig:FigEncoderDecoder} separates this architectural reference from the active online Siamese encoder.
\begin{figure}[htbp]
\centering
\begin{subfigure}{\linewidth}\centering\includegraphics[width=.85\linewidth]{images/Fig4a_NetEncoderDecoder.pdf}
\subcaption{Auxiliary offline architectural reference; not a primary-evaluation component.}\end{subfigure}\\
\begin{subfigure}{\linewidth}\centering\includegraphics[width=.85\linewidth]{images/Fig4b_NetContourSim.pdf}
\subcaption{The online shared contour encoder used by Ours-SD.}\end{subfigure}
\caption{Offline reconstruction and online Siamese verification are separate modules, not an inference chain. The new held-out-sequence tracker loads only the contour verifier.}
\label{fig:FigEncoderDecoder}
\end{figure}
"""
    prefix = replace_span(prefix, r"\subsection{Auxiliary latent encoding of contour regions}", r"\subsection{Implementation details and reproducibility}", auxiliary.strip())
    implementation = r"""
\subsection{Implementation details and reproducibility}
\label{sec:impl}
ContourSimilarityNet has four stride-2 stages with channels $32,64,128,128$ and a 128-dimensional L2-normalized embedding for a $128\times128$ input. Similarity is $s_c=(\cos(\mathbf z_1,\mathbf z_2)+1)/2$. Each of 20 outer folds starts from random weights, without the legacy Chess/TrackVes checkpoints. Image-derived binary contour rasters are sampled from at most 64 annotated ROI boxes per training sequence. Synthetic rotation, translation, scale and occlusion pairs provide similarity targets; polygon masks are not training targets.

Every fold excludes its test sequence from both domains' training and model selection. The held-out domain supplies one validation sequence and the other domain another; the remaining 17 of the 20 total sequence groups train the model. AdamW uses learning rate $3\times10^{-4}$, weight decay $10^{-4}$, batch size 64, 1024 sampled pairs per epoch, and 60 epochs. The checkpoint with minimum validation MSE is used; the held-out test never selects an epoch. Seeds are fixed and listed per fold. The runtime contour/confidence weights are explicitly set to this checkpoint and checked by tensor equality; online neural fine-tuning and external learned fusion-weight files are disabled.

The independent TrackVes VisualQualityGate uses four-channel response images and inference-safe scalar features. Nine separate folds use seven training, one validation and one test sequence. AdamW uses learning rate $8\times10^{-4}$, weight decay $10^{-4}$, batch size 256 and 18 epochs, with sequence-balanced sampling. Training-group normalization, validation-only checkpoint/margin selection, and all active model dependencies are recorded. This branch does not load ContourSimilarityNet or the offline encoder--decoder.

Official baselines retain the authors' trained task heads. OSTrack uses the author's ViT-B-256 candidate-elimination configuration; STARK uses the author's ST50 configuration and published checkpoint. All keys load strictly. Only dependency-import compatibility adjustments were made (removed PyTorch type aliases and the moved torchvision URL loader), without changes to inference mathematics. The official TrackEval HOTA implementation retains its metric equations, with NumPy float-alias compatibility changes. Immutable source commits and checkpoint hashes are included in the supplementary manifests. No official baseline was fine-tuned on these datasets.
"""
    prefix = replace_span(prefix, r"\subsection{Implementation details and reproducibility}", r"\subsection{ROI tracking with multi-feature fusion (scheme-C/nano)}", implementation.strip())
    naming_start = r"\textbf{Naming convention.}"
    naming_end = r"\paragraph{Initialization}"
    naming = r"""
\textbf{Naming convention.} Ours-SD denotes scheme-C/nano with an outer-fold-specific, sequence-isolated contour verifier. Ours-Contour denotes offline gradient-percentile contour-fill delineation. LearnedGate denotes the separate supervised TrackVes Nano/NCC/ORB quality gate. They do not all share one front-end or one checkpoint. Legacy Ours, conservative and latent variants are archival configurations, not evidence for the new primary ranking.

NanoTrack provides a candidate box and temporal localization, while hierarchical HistAB search verifies contour, color and spatial evidence. The offline gradient quantile does not enter this online inference path.
"""
    prefix = replace_span(prefix, naming_start, naming_end, naming.strip())
    prefix = prefix.replace(
        "Top-$R$ candidates (default $R{\\approx}48$, scheme-C defaults up to $64$) are re-ranked by the linear fusion",
        "Top-$R$ candidates are re-ranked by score fusion. A configurable linear form is",
    )
    prefix = prefix.replace(
        "with optional log-geometric or confidence-adaptive variants.",
        "while the strict rerun uses the released deterministic log-geometric/confidence-adaptive profile, "
        "not a newly learned fusion-weight model: $\\log s=\\sum_k w_k\\log(\\max(s_k,\\epsilon))$. "
        "The manifest records the enabled profile; the linear expression is not claimed as its exact runtime formula.",
    )
    prefix = prefix.replace(r"\caption{Ours (scheme-C/nano) online tracking}", r"\caption{Ours-SD: scheme-C/nano with fold-specific contour verification}")
    prefix = prefix.replace(
        "Memory-bank updates are available in code but disabled in the chessboard evaluation profiles used "
        "for Tables~\\ref{tab:TrackingIoU}--\\ref{tab:AblationModules}.",
        "Memory-bank updates and online neural fine-tuning are disabled in the sequence-isolated rerun; "
        "appearance/histogram updates use no post-initialization GT.",
    )
    supervised = r"""
\paragraph{Separate supervised TrackVes extension}
\label{sec:learnedgate-method}
LearnedGate proposes NanoTrack, global five-scale template-NCC, and ORB-affine candidate boxes. A shared CNN predicts each candidate's quality from synchronized template/current-ROI responses and scalar geometry, confidence and agreement cues. A non-Nano candidate is selected only when its predicted gain exceeds a validation-selected margin. The template is derived from the supplied initial box, and no subsequent test GT enters features or online decisions. All local learned weights and normalization are fold-specific; official NanoTrack weights remain frozen. This is a separate quality-arbitration experiment, not evidence that the online contour network or auxiliary latent encoder caused its changes.
"""
    prefix = replace_span(prefix, r"\paragraph{Conservative non-learned extension (Ours-Conservative)}", r"\subsection{Evaluation measures}", supervised.strip())
    measures = r"""
\subsection{Evaluation measures}
Standard Jaccard box IoU, Success@0.5, and Precision@20 pixels are primary localization measures. Missing, nonpositive-size or absent predictions score zero IoU, zero success and zero precision; a completely failed target still contributes all its post-initialization GT observations as misses. Center error is descriptive only for valid predictions and is not made artificially finite for missing detections.

Official TrackEval HOTA/DetA/AssA/CLEAR metrics use the same complete GT set and explicit empty prediction arrays for misses. Each board/ROI has a fixed single identity. Consequently, AssA here reflects match continuity under this single-object representation, not a demonstrated cross-object re-identification capability. Pooled HOTA is reported separately from per-sequence summaries; paired effects and bootstrap intervals use the independent sequence summaries.

Two-sided Wilcoxon tests align sequence/video names, not row positions or the intersection of surviving outputs. Paired mean effects and 95\% percentile bootstrap intervals resample whole video/sequence units with 20000 repetitions and seed 20261011. Holm correction covers the fixed comparator-by-metric family within each dataset. Frame-level significance markers, one-sided selected-variant tests, coverage-aware rewards for oversized boxes, and simplified HOTA proxies are not used as primary evidence.
"""
    prefix = prefix[:prefix.index(r"\subsection{Evaluation measures}")] + measures.strip() + "\n"
    body = experiments(segmentation, tracking, legacy)
    discussion = r"""
\section{Discussion}
The rerun separates contour delineation, contour-assisted tracking, and supervised proposal-quality gating. Video-grouped delineation remains favorable under the retained prompting configurations, but that result does not transfer automatically to temporal tracking. Official pretrained trackers provide substantially stronger comparisons than untrained task-head proxies; the latter cannot support a modern-baseline superiority claim.

The strict contour folds exclude every held-out sequence from locally learned model fitting and selection, and the separate gate has its own isolated active dependencies. These controls address direct sequence overlap, not all possible generalization limitations. The supervised methods have different training budgets, and retrospective architecture/profile choices were informed by earlier experiments on the same benchmark family. Small video groups, overlapping outer-fold training data, single-seed fitting, and a rigid engineering control limit statistical and clinical interpretation.

Several limitations remain explicit: no new independent patient cohort, no matched fine-tuning of official baselines, no controlled retraining ablation for every cue weight, and no restored information for already-corrupted source video regions. Equal-video weighting answers a different question from equal-frame weighting; both are reported with their definitions. The source code and lossless frame hashes make the evaluation auditable, but do not certify rights to redistribute controlled images or clinical outcomes. Prospective external validation, repeated seeds and matched supervision are needed before drawing clinical navigation conclusions.

\section{Conclusions}
Contour verification can be formulated as a conservative arbitration layer around an existing tracker. This study distinguishes adaptive offline gradient quantiles from fixed-gray online rasters and supplies an implementation with sequence-isolated active learned components, official pretrained tracking baselines, complete GT denominators, and grouped statistical uncertainty. The resulting evidence supports a bounded contour-delineation contribution and an honest evaluation of tracking limitations, not uniform superiority, annotation-free initialization, dense deformation fields, or demonstrated clinical navigation accuracy.
"""
    backmatter = source[source.index(r"\authorcontributions{"):]
    ai_note = (
        "An AI assistant using Copilot SDK in VS Code assisted substantive language revisions of the "
        "abstract, cover letter, methods and conclusions, and implementation of evaluation, statistical "
        "and plotting scripts. Results in this revision were computed from the recorded local model "
        "outputs and datasets, not supplied as AI-invented experimental observations. The authors remain "
        "responsible for reviewing the code, numbers, citations and final text before submission."
    )
    backmatter = re.sub(
        r"\\acknowledgments\{[^}]*\}",
        lambda _: "\\acknowledgments{The authors thank the public dataset providers. " + ai_note + "}",
        backmatter, count=1,
    )
    data_statement = (
        r"\dataavailability{The fairness revision's code, checkpoint/source hashes, split manifests, "
        r"complete prediction-denominator audit, video/sequence-level statistics and plotted aggregate "
        r"results accompany this submission as supplementary reproducibility material. The pre-existing "
        r"\href{https://github.com/liranyang123456-commits/Accurate-Deformation-Tracking-of-Soft-Tissue-Surfaces-in-Endoscopic-Surgical-Navigation}"
        r"{public repository} contains earlier implementation assets; the new local fairness revision is "
        r"not claimed to have been uploaded there. Controlled source images, GT annotations and crops are "
        r"not redistributed without documented permission. Public datasets must be obtained from their original providers.}"
    )
    backmatter = replace_span(backmatter, r"\dataavailability{", r"\acknowledgments{", data_statement)
    appendix_rows = []
    for name in ("OSTrack-proxy", "MixFormer-proxy", "LMTrack-proxy", "STARK-proxy"):
        data = legacy["methods"][name]
        appendix_rows.append([
            name, mean_sd(data["bbox_iou"]),
            f"{data['pooled_official_trackeval']['HOTA']:.3f}",
            str(data["n_missing_predictions"]),
            "Untrained task head; not official reproduction",
        ])
    appendix = (
        r"\appendix" + "\n" + r"\section{Legacy proxy and denominator audit}" + "\n"
        + r"\label{sec:legacy-audit}" + "\n"
        + "The archival untrained-head configurations are retained only to explain why the earlier "
        + "ranking was invalid. Their boxes are aligned to the canonical GT set, failed outputs are "
        + "kept as misses, and initialization observations are removed. Their absolute scores are not "
        + "estimates of the published methods and are not included in the new primary ranking. "
        + "Historical in-domain contour and latent checkpoints likewise do not supply main-table evidence.\n"
        + table(
            "Archival proxy diagnostics on the controlled benchmark. These rows are not officially trained OSTrack, MixFormer, LMTrack or STARK.",
            "tab:ProxyAppendix", ["Archived configuration", "Box IoU", "HOTA", "Missing", "Validity"], appendix_rows,
        )
        + r"\section{Generative-AI assistance and author responsibility}" + "\n"
        + ai_note + "\n"
        + "Synthetic augmentation used in contour fitting is explicit numerical image transformation, "
        + "not generative-AI production of patient observations. Authors must verify study provenance, "
        + "publication history, permissions, author approvals and all disclosures before submission.\n"
    )
    backmatter = backmatter.replace(r"\reftitle{References}", appendix + "\n" + r"\reftitle{References}")
    final = prefix + "\n" + body + "\n" + discussion.strip() + "\n\\vspace{6pt}\n" + backmatter
    prohibited = ("highest official TrackEval HOTA", "all four share the same gradient",
                  "An earlier version was submitted", "frame-level Wilcoxon $p<")
    if any(value.lower() in final.lower() for value in prohibited):
        raise ValueError("Stale superiority, shared-front-end or submission-history claim remained")
    (DESTINATION / "Manuscript_Mathematics.tex").write_text(final, encoding="utf-8")
    cover = (ORIGINAL / "Cover_Letter_Mathematics.tex").read_text(encoding="utf-8")
    first = cover.index("The article addresses")
    last = cover.index("This manuscript has not previously been submitted")
    replacement = (
        "The article examines ROI verification in endoscopic imagery, explicitly separating offline "
        "gradient-percentile contour delineation from online fixed-gray rasterization and Siamese/HSV "
        "arbitration. No manual pixel masks are required to generate contour cues, but the tracker uses "
        "a supplied initial box and the learned quality gate uses supervised training-sequence targets.\n\n"
        "The mathematical connection to this Special Issue is empirical gradient order statistics, "
        "Siamese metric representation, and score-level proposal arbitration, with official HOTA/AssA "
        "evaluation and video/sequence-grouped paired effects, bootstrap confidence intervals and "
        "Holm-adjusted two-sided Wilcoxon tests. These established tools are integrated rather than "
        "presented as a new edge-detection theorem.\n\n"
        f"CholecSeg8k video-macro Dice is {group_dice['video_macro_mean']:.3f} over 17 videos and all 8080 "
        "frames, including absent masks. Tracking comparisons now use author-released OSTrack and "
        "STARK-ST50 task weights, a fixed GT denominator, identical decoded image bytes, and held-out "
        "training of every active learned component. The results do not establish uniform superiority "
        "over official trackers or general clinical navigation accuracy. Negative findings and "
        "retrospective-study limitations are reported explicitly.\n\n"
        "The fairness revision's code and audit/statistical manifests accompany the submission as "
        "supplementary material. Generative-AI assistance with text and evaluation code is disclosed "
        "in the manuscript; experimental results were obtained by the documented local computations.\n\n"
    )
    cover = cover[:first] + replacement + cover[last:]
    (DESTINATION / "Cover_Letter_Mathematics.tex").write_text(cover, encoding="utf-8")
    write_json(DESTINATION / "assembly_manifest.json", {
        "source_manuscript_sha256": source_hash,
        "new_manuscript_sha256": sha256(DESTINATION / "Manuscript_Mathematics.tex"),
        "grouped_results_sha256": sha256(HERE / "results" / "cholecseg_grouped.json"),
        "tracking_results_sha256": sha256(HERE / "results" / "fair_tracking_summary.json"),
        "AI_assistance_disclosed": True,
        "official_tracker_weights_used": True,
        "old_proxy_main_tables_removed": True,
    })


def main() -> None:
    segmentation = json.loads((HERE / "results" / "cholecseg_grouped.json").read_text(encoding="utf-8"))
    tracking = json.loads((HERE / "results" / "fair_tracking_summary.json").read_text(encoding="utf-8"))
    legacy = json.loads((HERE / "results" / "chess_fair_legacy.json").read_text(encoding="utf-8"))
    DESTINATION.mkdir(parents=True, exist_ok=True)
    (DESTINATION / "images").mkdir(exist_ok=True)
    assemble(segmentation, tracking, legacy)
    plot_results(segmentation, tracking)
    print(DESTINATION)


if __name__ == "__main__":
    main()
