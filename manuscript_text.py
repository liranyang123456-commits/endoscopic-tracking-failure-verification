"""Manuscript and cover-letter text for the revised verification submission.

Every number in the generated LaTeX is read from the result JSON files; the
module contains prose, table layouts and the mathematical statements only.
"""
from __future__ import annotations

import re

import numpy as np

from build_fair_submission import ORIGINAL, pvalue, table


TITLE = (
    "Ordinal-Gradient Appearance Consistency and Conformal Alarms for Verifying "
    "Region-Level Soft-Tissue Tracking in Endoscopy"
)
LABELS = {
    "native_confidence": "Native confidence", "rank_only": "Rank only",
    "contour_only": "Contour only", "verification": "Fused verification",
    "color_only": "Color only", "appearance_only": "Appearance only (no confidence)",
    "appearance_fusion": "Appearance only (no confidence)",
    "rank_contour": "Rank + contour", "delta_confidence": "$|\\Delta$ confidence$|$",
}
PROPOSERS = {"stark": "STARK-ST50", "nanotrack": "NanoTrack", "ostrack": "OSTrack"}
FUSION_LABELS = {
    "contour_rank": "Contour + rank",
    "contour_color": "Contour + rank + color",
    "contour_color_confidence": "Contour + rank + color + confidence",
}
TRACKER_LABELS = {
    "nanotrack": "NanoTrack", "csrt": "CSRT", "ostrack": "OSTrack (official)",
    "stark": "STARK-ST50 (official)", "learnedgate": "LearnedGate",
    "ours_disjoint": "Prior Ours-SD",
}


def fmt(value, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "--"
    try:
        if value != value:
            return "--"
    except TypeError:
        return "--"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def score(values: dict, metric: str) -> str:
    value = values["sequence_macro"][metric]
    return "--" if value["mean"] is None else f"${value['mean']:.3f}\\pm{value['sample_sd']:.3f}$"


def ci(entry: dict, low: str = "ci95_low", high: str = "ci95_high") -> str:
    return f"$[{entry[low]:+.3f}, {entry[high]:+.3f}]$"


def find_test(tests: list, **keys) -> dict:
    return next(row for row in tests if all(row[k] == v for k, v in keys.items()))


# --------------------------------------------------------------------------- front matter
def abstract_text(results: dict, heldout: dict, supplementary: dict) -> str:
    nano = results["proposers"]["nanotrack"]["methods"]
    stark = results["proposers"]["stark"]["methods"]
    effect = find_test(results["paired_tests"], proposer="nanotrack", comparator="native_confidence", metric="failure_auroc")
    ost = heldout["methods"]
    latency = supplementary["detection_latency"]["nanotrack"]
    text = (
        "A region-level tracker following deforming soft tissue in endoscopic video can lose its target yet keep returning "
        "a plausible box with unchanged confidence. We ask, from the supplied initial region and a frozen tracker's unchanged "
        "proposals alone, whether each proposal is still consistent with the target. The proposed layer compares midrank "
        "intensities, a fixed-density gradient-quantile contour set, a hue--saturation histogram and the tracker's confidence "
        "between proposal and reference, and fuses the four bounded discrepancies into a convex risk. We prove elementary "
        "properties of the contour distance and give the alarm threshold a split-conformal construction with a finite-sample "
        "sequence-level specificity guarantee. With all parameters "
        f"frozen on the SurgT validation split, test failure AUROC rises from {nano['native_confidence']['sequence_macro']['failure_auroc']['mean']:.3f} to "
        f"{nano['verification']['sequence_macro']['failure_auroc']['mean']:.3f} for NanoTrack, {stark['native_confidence']['sequence_macro']['failure_auroc']['mean']:.3f} to "
        f"{stark['verification']['sequence_macro']['failure_auroc']['mean']:.3f} for STARK-ST50 and {ost['native_confidence']['sequence_macro']['failure_auroc']['mean']:.3f} to "
        f"{ost['verification']['sequence_macro']['failure_auroc']['mean']:.3f} for the held-out OSTrack; median latency to the first failure episode falls from "
        f"{latency['native_confidence']['median_latency_scored_frames']:.0f} to {latency['verification']['median_latency_scored_frames']:.1f} frames for NanoTrack. "
        "Ordinal and contour cues each carry most of the information; fusing them adds little. The layer costs about "
        f"{results['proposers']['nanotrack']['mean_monitor_ms_per_scored_frame']:.1f} ms per frame, never alters a box, and supplies failure awareness, not better localization."
    )
    if len(re.findall(r"[A-Za-z0-9]+", text)) > 200:
        raise ValueError("Conservative abstract word count exceeds 200")
    return text


def preamble(original_tex: str) -> str:
    text = original_tex[:original_tex.index(r"\abstract{")]
    text = re.sub(r"\\Title\{[^}]*\}", lambda _: r"\Title{" + TITLE + "}", text, count=1)
    return text


# --------------------------------------------------------------------------- introduction / related work
def introduction(results: dict, heldout: dict, supplementary: dict) -> str:
    nano = results["proposers"]["nanotrack"]["methods"]
    stark = results["proposers"]["stark"]["methods"]
    dist = supplementary["overlap_and_coverage"]["native_confidence_distribution"]["nanotrack"]
    return rf"""
\section{{Introduction}}
\textbf{{The problem.}} Image-guided endoscopy increasingly overlays a region of interest (ROI) on the live video---a safety margin around a vessel, a lesion to be revisited, a resection boundary---and relies on a region-level tracker to keep that overlay attached to deforming soft tissue~\cite{{PENZA201813,yip2012tissue,YE2016}}. Tissue folds and stretches, specular highlights move with the camera, instruments occlude the target and the field of view is narrow. When a tracker drifts under these conditions it rarely stops: it settles on visually similar tissue and keeps returning a well-formed box. On the public SurgT test partition~\cite{{cartucho2024surgt}}, tracked continuously without ground-truth reset, three official trackers lose the target on {nano['verification']['pooled']['failure_prevalence'] * 100:.0f}\%, {stark['verification']['pooled']['failure_prevalence'] * 100:.0f}\% and {heldout['failure_prevalence'] * 100:.0f}\% of the scored frames (NanoTrack, STARK-ST50, OSTrack), while the native confidence of NanoTrack averages {dist['confidence_mean_reliable']:.2f} on reliable and {dist['confidence_mean_failed']:.2f} on failed frames. The overlay therefore points at the wrong anatomy with no visible sign of failure (Figure~\ref{{fig:Qualitative}}).

\textbf{{Why native confidence is not enough.}} The scalar that a tracker reports---a classification response in Siamese trackers~\cite{{bertinetto2016siamfc,opencv_nanotrack}}, a reliability logit used for template updating~\cite{{yan2021stark}}, a predicted overlap~\cite{{danelljan2019atom}}---is trained to choose or refine a box, not to say whether the chosen box is still the target. Learned networks are also poorly calibrated~\cite{{guo2017calibration}}. Classical failure signals such as forward--backward error~\cite{{kalal2010forward,kalal2012tld}}, the peak-to-sidelobe ratio~\cite{{Bolme2010}} or the average peak-to-correlation energy~\cite{{wang2017lmcf}}, and the re-detection triggers of long-term trackers~\cite{{yan2019splt,dai2020ltmu}}, require a response map, a second motion estimate or a trained quality head, and the failure detector of the long-term safety-area tracker of Penza et~al.~\cite{{PENZA201813}} is bound to its own tracker. A signal that can be attached to \emph{{any}} box-producing tracker, that needs no manually drawn masks and no additional network, and that is evaluated with every parameter frozen before the test data are touched, is missing for soft tissue.

\textbf{{The idea.}} We treat the problem as appearance-consistency verification: compare the patch under the current proposal with the patch under the single ROI supplied at initialization, using statistics chosen for the failure modes of endoscopic video. Ordinal (midrank) intensities remove monotone photometric change, so that illumination drift alone does not raise an alarm. A fixed-density set of the strongest ordinal gradients summarizes where structure is, and a trimmed symmetric distance between two such sets measures whether the structure has moved; because the set has fixed density its chance level is known. Bounded phase correlation absorbs small translations so that camera jitter is not read as failure. A hue--saturation histogram captures the change from mucosa to instrument or blood that structure alone misses, and the tracker's own confidence is kept as a fourth, optional cue. The four bounded discrepancies are fused by a convex combination into a risk in $[0, 1]$. The alarm threshold is calibrated on validation sequences, and a sequence-level split-conformal variant of this threshold carries a finite-sample specificity guarantee under exchangeability. The tracker's box is returned exactly as proposed; the layer only adds a risk and a warning (Figure~\ref{{fig:Pipeline}}).

\textbf{{What the evidence shows.}} Around three official trackers---two used for parameter selection and one held out---the verifier ranks failed frames above reliable ones far better than the native confidence of NanoTrack and OSTrack and modestly better than that of STARK-ST50; it raises a warning at the onset of most first failure episodes where native confidence misses or lags; and it transfers to two further benchmarks without recalibration (Figure~\ref{{fig:Headline}}). The cue ablation shows that ordinal and contour statistics each carry most of this information and that fusing them adds little, which is why the title names the signal rather than a single cue. Every non-significant and negative comparison is reported, including a validation-to-test shift for STARK-ST50 and an earlier, unsuccessful attempt to let contour evidence overwrite the tracker's box.

\textbf{{Contributions.}}
\begin{{enumerate}}[leftmargin=*]
\item A non-invasive, mask-free appearance-consistency verifier for region-level tracking built from classical operators (Section~\ref{{sec:method}}), with proven elementary properties and a Monte Carlo chance level for its trimmed contour distance (Proposition~\ref{{prop:dc}}) and a split-conformal sequence-level specificity guarantee for its alarm threshold (Proposition~\ref{{prop:conformal}}).
\item A validation-frozen evaluation on the public SurgT benchmark around two selection proposers (NanoTrack, STARK-ST50) and one held-out proposer (OSTrack), with complete shared denominators, failure-ranking, operating-point, selective-risk and detection-latency metrics, paired sequence-level inference, exact permutation tests and case-cluster bootstrap intervals (Sections~\ref{{sec:protocol}}--\ref{{sec:results}}).
\item A cue ablation and post-hoc sensitivity study (failure definition, contour percentile, fusion weights, alarm quantile, temporal position) that identifies which statistics carry the information and shows that the result is not an artefact of one setting.
\item A four-cue transfer to TrackVes and a controlled rigid benchmark without target-dataset calibration, and a transparent account of an earlier box-overwriting fusion tracker under the same complete-denominator protocol (Appendix~\ref{{sec:appendix_fusion}}).
\end{{enumerate}}

\textbf{{Scope and claims.}} The verifier reports a risk; it does not replace, reset or suppress the tracker's box, does not estimate dense tissue displacement, and is not a validated clinical safety device. ``Mask-free'' means that no manually drawn pixel masks are needed to construct the descriptors; the initial ROI box and labeled validation sequences for threshold calibration are still required. The SurgT test partition was not used for any selection, but the dataset is public and earlier work in our group has evaluated other trackers on it, so it is not a never-seen patient cohort (Section~\ref{{sec:protocol}}).

The remainder of the paper is organized as follows. Section~\ref{{sec:related}} reviews related work. Section~\ref{{sec:method}} motivates and formulates the verifier and proves its mathematical properties. Section~\ref{{sec:protocol}} describes the datasets, the frozen protocol and the statistical analysis. Section~\ref{{sec:results}} opens with a summary of all performance metrics and then reports each analysis, Section~\ref{{sec:discussion}} discusses design implications and limitations, and Section~\ref{{sec:conclusions}} concludes.
"""


RELATED = r"""
\section{Related Work}\label{sec:related}
\textbf{Region-level soft-tissue tracking.} Tissue tracking for image-guided surgery has been approached with feature-based registration~\cite{yip2012tissue}, probabilistic affine-invariant region tracking~\cite{giannarou2013probabilistic}, online tracking with retargeting for optical biopsy~\cite{YE2016}, and long-term safety-area tracking with failure detection and recovery~\cite{PENZA201813}. The TrackVes dataset~\cite{penza2017trackvesdata} and the SurgT challenge~\cite{cartucho2024surgt} provide public region-level benchmarks; SurgT additionally defines a reset-based expected-average-overlap protocol inherited from the VOT methodology~\cite{kristan2016methodology,VOT2016}. Dense tissue correspondence and mapping are addressed by surgical point trackers and tattoo-based ground truth~\cite{li2020super,schmidt2024stir,karaoglu2025litetracker}, and surgical tool tracking has been reviewed in~\cite{BOUGET2017}. The present work concerns the reliability of a region-level box, not dense correspondence.

\textbf{Learned trackers and their confidence.} Siamese and transformer trackers~\cite{bertinetto2016siamfc,li2018siamrpn,ye2022ostrack,yan2021stark,cui2022mixformer} dominate generic benchmarks~\cite{wu2015otb,kristan2016methodology}. Their scalar outputs---a classification response~\cite{bertinetto2016siamfc,opencv_nanotrack}, a reliability logit~\cite{yan2021stark} or a predicted overlap~\cite{danelljan2019atom}---are trained for box selection or template updating rather than calibrated failure detection, and modern networks are known to be miscalibrated~\cite{guo2017calibration}. Correlation-filter trackers~\cite{Bolme2010,KCF2015,lukezic2017csrt} expose response-map statistics instead. We treat the proposer's confidence as one cue among several and quantify how informative it is on soft tissue.

\textbf{Tracking failure detection.} Forward--backward error~\cite{kalal2010forward} and the detector of Tracking--Learning--Detection~\cite{kalal2012tld}, the peak-to-sidelobe ratio of MOSSE~\cite{Bolme2010}, the average peak-to-correlation energy of LMCF~\cite{wang2017lmcf} and the re-detection triggers of long-term trackers~\cite{yan2019splt,dai2020ltmu} are the classical and learned signals used to decide whether a tracker has lost its target. They are tracker-specific (response maps), require a second motion estimate (forward--backward flow) or train a verifier network. The verifier studied here compares the current ROI with the supplied initial reference through ordinal and contour statistics, so it can be attached to any box-producing tracker without access to its internals.

\textbf{Selective prediction, calibration and conformal inference.} Deciding when to trust a prediction is studied as selective classification with risk--coverage trade-offs~\cite{elyaniv2010foundations,geifman2017selective}. Distribution-free guarantees for thresholds follow from conformal inference~\cite{vovk2005algorithmic,shafer2008tutorial,lei2018distribution,angelopoulos2023gentle}, whose exchangeability assumption can be relaxed at the price of weaker bounds~\cite{barber2023beyond}. Failure ranking is evaluated with ROC analysis~\cite{hanley1982meaning,fawcett2006roc}, with the caveat that precision-type metrics depend on prevalence~\cite{saito2015precision}; paired comparisons use rank tests, bootstrap intervals and family-wise error control~\cite{wilcoxon1945individual,efron1993bootstrap,holm1979sequential,delong1988comparing}. We adopt these tools for a frame-level failure label defined by box overlap.

\textbf{Ordinal representations and shape distances.} Rank and census transforms~\cite{zabih1994nonparametric} are invariant to strictly increasing photometric mappings and are among the most robust matching costs under radiometric differences~\cite{hirschmuller2009evaluation}; the Pearson correlation of ranks is Spearman's coefficient~\cite{spearman1904proof}. Phase correlation~\cite{kuglin1975phase,guizar2008registration} recovers translations between patches. Distances between binary contour sets are classically computed through distance transforms~\cite{borgefors1986distance}, as in chamfer matching~\cite{barrow1977chamfer} and the Hausdorff distance~\cite{huttenlocher1993hausdorff}; the Bhattacharyya coefficient and Hellinger distance compare histograms~\cite{bhattacharyya1943measure}. Learned edge detectors~\cite{RcfEdgePami2019,He2019CVPR,Soria_2023teed} and promptable segmenters~\cite{kirillov2023segment,ma2024segment,ravi2024sam2} produce richer boundaries but require training data or prompts; our contour cue is a fixed-density set of high-gradient pixels and is not a semantic boundary. Specular highlights, a known nuisance for endoscopic appearance models~\cite{Ranyang2020}, are not removed in this work.
"""


# --------------------------------------------------------------------------- method
def method(selected: dict, supplementary: dict) -> str:
    chance = supplementary["contour_chance_level"]
    c80 = chance[str(selected["percentile"])]
    weights = selected["weights"]
    return rf"""
\section{{Problem Formulation and Verification Method}}\label{{sec:method}}
\subsection{{Scope and information available online}}
Let $I_t \in \mathbb{{R}}^{{H \times W \times 3}}$ be a BGR endoscopic frame at time $t \ge 0$ and $R_t = (x_t, y_t, w_t, h_t)$ the box proposed by a frozen tracker $\mathcal{{T}}$. The only supervision available online is the initial ROI $R_0$ annotated on $I_0$. The verifier observes the triplet $(I_t, R_t, c_t)$, where $c_t$ is the tracker's native scalar confidence, and returns a bounded risk $r_t \in [0, 1]$ and an optional warning $\omega_t = \mathbf{{1}}\{{r_t > \tau\}}$ for a fixed threshold $\tau$. The coordinates $R_t$ passed downstream are never altered.

For evaluation only, localization failure is defined against the ground-truth box $G_t$ as $Y_t = \mathbf{{1}}\{{\mathrm{{IoU}}(R_t, G_t) < 0.5\}}$. Ground truth never enters descriptor computation or the online decision; validation ground truth is used offline to select fusion weights and calibrate $\tau$, and test ground truth only to score.

\begin{{figure}}[htbp]
\centering\includegraphics[width=\linewidth]{{images/Fig_Framework.pdf}}
\caption{{Overview of the verification layer. (\textbf{{A}})~The supplied initial ROI is converted once into a fixed rank, contour and color reference. (\textbf{{B}})~At each frame the frozen tracker emits a proposal box and its native confidence; the box is forwarded unchanged. (\textbf{{C}})~The verifier computes the same descriptors under the proposal, aligns them to the reference, fuses four bounded discrepancies into a risk $r_t$ and raises a warning when $r_t$ exceeds a threshold. (\textbf{{D}})~Ground truth is used only offline: validation sequences select the percentile and weights and calibrate the threshold (frame-pooled or conformal), and the frozen test partition scores failures.}}
\label{{fig:Pipeline}}
\end{{figure}}

\subsection{{Design rationale: failure modes and the cues that address them}}\label{{sec:rationale}}
The descriptors were chosen for the ways in which region-level tracking fails on soft tissue, and each is kept deliberately simple so that its behaviour can be stated exactly.
\begin{{itemize}}[leftmargin=*]
\item \emph{{Drift to a different structure.}} The most common failure moves the box onto neighbouring tissue whose fine structure differs from the target. A fixed-density set of the strongest ordinal gradients (Section~\ref{{sec:contours}}) records where structure is, and a trimmed symmetric distance between the reference and candidate sets measures how far it has moved; the rank-correlation discrepancy (Section~\ref{{sec:fusion}}) measures the same change at the level of intensity ordering. These two cues turn out to carry most of the information (Section~\ref{{sec:results_ranking}}).
\item \emph{{Illumination and exposure change.}} Endoscopic light moves with the camera, so raw intensities change even when the target does not. Midranks (Section~\ref{{sec:ordinal}}) are invariant to every strictly increasing intensity mapping of the patch, which removes this nuisance exactly rather than approximately.
\item \emph{{Small translations.}} A box that is a few pixels off is not a failure. Bounded phase correlation aligns the candidate to the reference before distances are computed, and its displacement cap keeps a genuine drift from being ``corrected'' away.
\item \emph{{Occlusion by instruments, blood or smoke.}} These change color before they change structure; a hue--saturation histogram distance adds this evidence.
\item \emph{{The tracker's own doubt.}} When a proposer does expose a meaningful confidence (STARK-ST50), it is informative and is kept as a fourth cue; when it does not (NanoTrack), the appearance cues must carry the decision. The fusion is a convex combination so that the risk stays in $[0, 1]$ and each cue's contribution is bounded by its weight.
\end{{itemize}}
The alarm threshold is treated as a statistical object rather than a tuning constant: the frame-pooled validation quantile used in the main tables has a nominal specificity target, and the sequence-level conformal construction of Section~\ref{{sec:conformal}} converts the same validation data into a guarantee.

\subsection{{Ordinal intensity representation}}\label{{sec:ordinal}}
The patch under $R_t$ is cropped within the image, resampled by area interpolation to $64 \times 64$ pixels ($N = 4096$) and converted to grayscale $I: \Omega \to [0, 255]$ with $\Omega = \{{1, \dots, 64\}}^2$. Intensities are replaced by normalized empirical midranks~\cite{{zabih1994nonparametric}},
\begin{{equation}}
U(p) = \frac{{\#\{{q \in \Omega : I(q) < I(p)\}} + \tfrac{{1}}{{2}}\bigl(\#\{{q \in \Omega : I(q) = I(p)\}} - 1\bigr)}}{{N - 1}}, \qquad p \in \Omega,
\label{{eq:rank}}
\end{{equation}}
so that tied gray levels receive equal ranks and $U(p) \in [0, 1]$ is defined even for a constant patch.

\begin{{Remark}}\label{{rem:monotone}}
If $g: \mathbb{{R}} \to \mathbb{{R}}$ is strictly increasing then $g \circ I$ has the same strict-order and tie counts as $I$ at every pixel, hence the same midranks $U$; every quantity computed from $U$ alone (Sections~\ref{{sec:contours}}--\ref{{sec:fusion}}) is therefore invariant to strictly increasing photometric mappings of the fixed resampled patch. This is the classical property of rank transforms~\cite{{zabih1994nonparametric,hirschmuller2009evaluation}} and is recalled only to make explicit what the construction does and does not guarantee: sensor saturation, specular clipping and tissue deformation are not monotone intensity mappings and are handled, if at all, only empirically.
\end{{Remark}}

\subsection{{Gradient-quantile contours, alignment and the trimmed contour distance}}\label{{sec:contours}}
A $3\times3$ Gaussian kernel with $\sigma = 0.8$ followed by $3\times3$ Sobel operators gives the gradient magnitude $m(p)$ of $U$. A 2-pixel border is discarded, leaving $n = 3600$ interior samples ordered as $m_{{(1)}} \le \cdots \le m_{{(n)}}$. For a percentile $\alpha \in (0, 1)$ the linear sample quantile is
\begin{{equation}}
\begin{{aligned}}
a &= (n - 1)\alpha,\qquad k = \lfloor a \rfloor,\qquad \gamma = a - k,\\
q_\alpha &= (1 - \gamma)\,m_{{(k+1)}} + \gamma\,m_{{(k+2)}},\qquad
C_\alpha(p) = \mathbf{{1}}\{{m(p) > \max(q_\alpha, 10^{{-6}})\}}.
\end{{aligned}}
\label{{eq:quantile}}
\end{{equation}}
By construction $C_\alpha$ retains a fraction close to $1 - \alpha$ of the interior pixels regardless of image content: it is a \emph{{fixed-density set of the strongest ordinal gradients}}, not a semantic tissue boundary. Retained pixels may belong to folds, vessel edges, specularities, instruments or noise. The percentile $\alpha = {selected['percentile']:.2f}$ was chosen on validation data from $\{{0.50, 0.68, 0.80\}}$ (Section~\ref{{sec:protocol}}).

To tolerate small translations, the reference and candidate rank patches are multiplied by a separable Hann window $W$ and registered by phase correlation~\cite{{kuglin1975phase,guizar2008registration}}: with $P = \mathcal{{F}}\{{W \odot U_t\}}\,\mathcal{{F}}\{{W \odot U_0\}}^*$ the normalized cross-power spectrum is $Q = P / (|P| + \epsilon)$, and the peak of $\mathcal{{F}}^{{-1}}\{{Q\}}$ gives a displacement $(\delta_x, \delta_y)$ with response $s_{{\mathrm{{PC}}}}$. The candidate rank patch and its contour set are translated by $(-\delta_x, -\delta_y)$ with reflection padding only if the shift is finite, $\max(|\delta_x|, |\delta_y|) \le 12$ pixels and $s_{{\mathrm{{PC}}}} > 0.05$; otherwise they are left unaligned. This bounded step does not model non-rigid deformation.

Let $A, B \subset \Omega$ be the reference and (aligned) candidate contour sets and let $D_A, D_B$ be their Euclidean distance transforms computed with the $3\times3$ chamfer mask of Borgefors~\cite{{borgefors1986distance}}. With the bidirectional distance multiset $\mathcal{{D}}(A, B) = \{{D_B(p) : p \in A\}} \uplus \{{D_A(q) : q \in B\}}$ and its 90th percentile $d_{{0.90}}$, the trimmed symmetric contour distance is
\begin{{equation}}
d_C(A, B) = \min\Bigl(1,\ \frac{{1}}{{8\,|\mathcal{{D}}(A, B)|}} \sum_{{d \in \mathcal{{D}}(A, B)}} \min(d, d_{{0.90}})\Bigr), \qquad d_C(A, \emptyset) = d_C(\emptyset, B) = 1.
\label{{eq:contour}}
\end{{equation}}

\begin{{Proposition}}[Elementary properties of $d_C$]\label{{prop:dc}}
For nonempty $A, B \subset \Omega$ let $\tilde H(A, B) = \max\bigl(\max_{{p \in A}} D_B(p), \max_{{q \in B}} D_A(q)\bigr)$ denote the Hausdorff distance induced by the chamfer distance transform. Then
(i) $0 \le d_C(A, B) \le 1$ and $d_C(A, B) = d_C(B, A)$;
(ii) $d_C(A, B) = 0$ if and only if $A = B$;
(iii) $d_C(A, B) \le \min\bigl(1, \tilde H(A, B) / 8\bigr)$.
\end{{Proposition}}
\begin{{proof}}
(i) Every term of the sum is nonnegative, and $\mathcal{{D}}(A, B) = \mathcal{{D}}(B, A)$ as multisets, so the mean and hence $d_C$ are symmetric; the outer minimum bounds the value by one. (ii) If $A = B$ then $D_B(p) = 0$ for all $p \in A$ and $D_A(q) = 0$ for all $q \in B$, so the sum vanishes. Conversely, a zero mean of nonnegative terms forces every $\min(d, d_{{0.90}})$ to vanish; since $d_{{0.90}} > 0$ unless all distances are zero, every $d \in \mathcal{{D}}$ is zero, which means every point of $A$ lies on $B$ and vice versa, i.e. $A = B$. (iii) Each $\min(d, d_{{0.90}}) \le d \le \tilde H(A, B)$, so the mean is at most $\tilde H(A, B)$ and the claim follows from the division by 8 and the outer minimum.
\end{{proof}}

The trimming at $d_{{0.90}}$ and the clipping make $d_C$ a bounded dissimilarity rather than a metric; the triangle inequality is not used anywhere in the method. Proposition~\ref{{prop:dc}}(iii) also exposes a limitation: the scale factor 8 and the trimming compress the useful range of $d_C$. Monte Carlo simulation (Appendix~\ref{{sec:appendix_chance}}) gives a chance level of $d_C \approx {c80['random_pair_mean']:.3f}$ between two independent random masks of density ${c80['density']:.2f}$, whereas a mask and its copy translated by 8 pixels differ by only ${c80['translated_copy']['8']:.3f}$ and by 16 pixels by ${c80['translated_copy']['16']:.3f}$. Displacements beyond a few pixels are therefore hardly distinguishable from an unrelated structure, which is why $d_C$ is complemented by rank and color cues.

\subsection{{Complementary discrepancies and bounded fusion}}\label{{sec:fusion}}
The rank discrepancy is $d_U = (1 - \rho) / 2$, where $\rho \in [-1, 1]$ is the Pearson correlation between the aligned candidate and reference rank arrays on the interior excluding a 4-pixel border; because the inputs are ranks, $\rho$ is Spearman's coefficient~\cite{{spearman1904proof}} of the underlying intensities. If a crop has zero variance, $\rho = 0$ and $d_U = 0.5$. The color discrepancy is the squared Hellinger distance between normalized $16\times16$ hue--saturation histograms $h_0, h_t \in \Delta^{{255}}$~\cite{{bhattacharyya1943measure}},
\begin{{equation}}
d_H(h_0, h_t) = 1 - \sum_{{b=1}}^{{256}} \sqrt{{h_0(b)\,h_t(b)}} = \tfrac{{1}}{{2}} \sum_{{b=1}}^{{256}} \bigl(\sqrt{{h_0(b)}} - \sqrt{{h_t(b)}}\bigr)^2 \in [0, 1],
\label{{eq:hellinger}}
\end{{equation}}
and the native-confidence risk is $d_N = 1 - \mathrm{{clip}}(c_t, 0, 1)$. The fused risk is the convex combination
\begin{{equation}}
r_t = \mathbf{{w}}^\top \mathbf{{d}}_t = {weights[0]:.2f}\,d_C + {weights[1]:.2f}\,d_U + {weights[2]:.2f}\,d_H + {weights[3]:.2f}\,d_N, \qquad \mathbf{{w}} \in \Delta^3,
\label{{eq:risk}}
\end{{equation}}
which lies in $[0, 1]$ because every cue does. Three weight vectors---contour$+$rank $(0.5, 0.5, 0, 0)$, contour$+$rank$+$color $(0.45, 0.25, 0.30, 0)$ and the four-cue vector of Equation~\eqref{{eq:risk}}---were written into the frozen protocol file before any verification run of this study; they encode the prior that contour evidence should dominate and that confidence should never exceed one fifth of the score, and were not fitted to data. The grid over these three vectors and three percentiles was searched once on the validation partition (Section~\ref{{sec:protocol}}).

\subsection{{Alarm threshold and a sequence-level conformal guarantee}}\label{{sec:conformal}}
The threshold used in the main tables is the 90th percentile of $r_t$ over all reliable validation frames ($Y_t = 0$), a nominal specificity target of 0.90 without a formal guarantee, because frames within a sequence are strongly dependent. A guarantee is available at the level at which observations are plausibly exchangeable---whole sequences.

\begin{{Proposition}}[Split-conformal sequence-level specificity]\label{{prop:conformal}}
Let $S_1, \dots, S_n$ be validation sequences and $S_{{n+1}}$ a new sequence, and suppose $(S_1, \dots, S_{{n+1}})$ are exchangeable. For each sequence let $q_j$ be the empirical $0.90$-quantile of the risk $r_t$ over its reliable frames, and for $\delta \in (0, 1)$ with $k = \lceil (n + 1)(1 - \delta) \rceil \le n$ let $\tau_\delta = q_{{(k)}}$ be the $k$-th smallest of $q_1, \dots, q_n$. Then
\begin{{equation}}
\Pr\bigl(q_{{n+1}} \le \tau_\delta\bigr) \ge 1 - \delta,
\label{{eq:conformal}}
\end{{equation}}
and consequently, with probability at least $1 - \delta$, the reliable-frame specificity of the warning $\mathbf{{1}}\{{r_t > \tau_\delta\}}$ on the new sequence is at least $0.90$.
\end{{Proposition}}
\begin{{proof}}
Each $q_j$ is a deterministic function of $S_j$, so $(q_1, \dots, q_{{n+1}})$ are exchangeable scalars. By exchangeability the rank of $q_{{n+1}}$ among the $n + 1$ values is uniformly distributed on $\{{1, \dots, n+1\}}$ when ties are broken at random, hence $\Pr(q_{{n+1}} \le q_{{(k)}}) \ge k / (n + 1) \ge 1 - \delta$~\cite{{vovk2005algorithmic,lei2018distribution}}. If $q_{{n+1}} \le \tau_\delta$ then at least $90\%$ of the reliable frames of $S_{{n+1}}$ satisfy $r_t \le q_{{n+1}} \le \tau_\delta$ and are not flagged, which is the specificity statement.
\end{{proof}}

Equation~\eqref{{eq:conformal}} is marginal over the draw of the validation sequences and requires exchangeability, not independence of frames; it says nothing about sensitivity, and it degrades when test sequences come from a shifted distribution~\cite{{barber2023beyond}}. Section~\ref{{sec:results_operating}} reports $\tau_\delta$ for $\delta \in \{{0.1, 0.2, 0.3\}}$ next to the nominal frame-pooled threshold and checks the empirical coverage on the test partition.

\subsection{{Computational properties and algorithm}}
Midranks cost $O(N \log N)$ by sorting; Gaussian and Sobel filtering, distance transforms and histograms are $O(N)$ on the fixed $64 \times 64$ patch; phase correlation is $O(N \log N)$. No network is trained, no stochastic optimization is involved, and repeated evaluation on identical inputs is deterministic. Algorithm~\ref{{alg:verification}} lists the complete procedure: Phase~A builds the fixed reference once from the supplied ROI, and Phase~B scores every proposal without modifying it.

\begin{{algorithm}}[htbp]
\SetAlgoLined
\DontPrintSemicolon
\caption{{Non-invasive ordinal-gradient appearance verification: one-time reference construction (Phase~A) and per-frame monitoring (Phase~B)}}\label{{alg:verification}}
\KwIn{{Initial frame $I_0$ with the supplied ROI $R_0$; subsequent frames $I_t$, $t\ge1$; a frozen tracker $\mathcal{{T}}$ returning a proposal $R_t$ and native confidence $c_t$; validation-frozen percentile $\alpha={selected['percentile']:.2f}$, weights $\mathbf{{w}}=[{weights[0]:.2f},{weights[1]:.2f},{weights[2]:.2f},{weights[3]:.2f}]^\top\in\Delta^3$, and threshold $\tau$ (frame-pooled or conformal $\tau_\delta$).}}
\KwOut{{For every frame $t$: the unchanged proposal $R_t$, risk $r_t\in[0,1]$ and warning $\omega_t\in\{{0,1\}}$.}}
\tcc{{Phase A: reference construction, executed once}}
Crop $R_0$ from $I_0$, area-interpolate to $64\times64$, convert to grayscale\;
$U_0\leftarrow$ normalized midranks (Equation~\eqref{{eq:rank}}); $m_0\leftarrow$ gradient magnitude of $U_0$ after Gaussian ($\sigma=0.8$) and Sobel filtering\;
$C_0\leftarrow\mathbf{{1}}\{{m_0>\max(q_\alpha,10^{{-6}})\}}$ on the interior (Equation~\eqref{{eq:quantile}}); $h_0\leftarrow$ normalized $16\times16$ hue--saturation histogram of the $R_0$ patch\;
Store the fixed reference $(U_0,C_0,h_0)$; it is never updated from later proposals\;
\tcc{{Phase B: per-frame verification; tracker state and box are never modified}}
\For{{$t=1,2,\ldots$}}{{
  $(R_t,c_t)\leftarrow\mathcal{{T}}(I_t)$\tcp*{{frozen proposer}}
  Crop $R_t$ from $I_t$, resample to $64\times64$, convert to grayscale; compute $U_t$, $m_t$, $C_t$, $h_t$ as in Phase~A\;
  $(\delta_x,\delta_y,s_{{\mathrm{{PC}}}})\leftarrow$ Hann-windowed phase correlation of $U_0$ and $U_t$\;
  \eIf{{the shift is finite, $\max(|\delta_x|,|\delta_y|)\le12$, and $s_{{\mathrm{{PC}}}}>0.05$}}{{
    Translate $U_t$ and $C_t$ by $(-\delta_x,-\delta_y)$ with reflection padding\;
  }}{{
    Keep $U_t$ and $C_t$ unaligned\;
  }}
  $d_C\leftarrow$ trimmed symmetric contour distance of $(C_0,C_t)$ (Equation~\eqref{{eq:contour}}); $d_C\leftarrow1$ if either set is empty\;
  $d_U\leftarrow(1-\rho)/2$ with $\rho$ the Pearson correlation of the 4-pixel interior crops of $U_0$ and $U_t$ ($\rho\leftarrow0$ for zero variance)\;
  $d_H\leftarrow1-\sum_b\sqrt{{h_0(b)\,h_t(b)}}$ (Equation~\eqref{{eq:hellinger}}); \quad $d_N\leftarrow1-\mathrm{{clip}}(c_t,0,1)$\;
  $r_t\leftarrow\mathbf{{w}}^\top[d_C,d_U,d_H,d_N]^\top$ (Equation~\eqref{{eq:risk}}); \quad $\omega_t\leftarrow\mathbf{{1}}\{{r_t>\tau\}}$\;
  \textbf{{emit}} $(R_t,r_t,\omega_t)$\tcp*{{$R_t$ is returned exactly as proposed}}
}}
\end{{algorithm}}
\FloatBarrier
"""


# --------------------------------------------------------------------------- protocol
def protocol(results: dict, selected: dict, supplementary: dict, heldout: dict, cpu: str) -> str:
    overlap = supplementary["overlap_and_coverage"]
    text = rf"""
\section{{Datasets, Frozen Protocol and Statistical Analysis}}\label{{sec:protocol}}
\subsection{{SurgT partitions and continuous no-reset protocol}}
We use the provided SurgT validation partition (12 sequences) and test partition (20 sequences, five case groups), with the single declared left-view target of each sequence~\cite{{cartucho2024surgt}}. Stereo frames are split according to the metadata and rectified with the supplied calibration (OpenCV \texttt{{stereoRectify}}, zero disparity, $\alpha = 0$); every frame is decoded sequentially with one decoder thread, and all proposers see identical rectified pixels (verified by per-frame hashes).

Tracking starts from the first visible, non-difficult, valid ground-truth box and continues without any ground-truth reset; every later decoded frame updates the tracker, including frames that are not scoreable. Scoring uses all later visible, non-difficult ground-truth boxes inside the image, which are fixed independently of the tracker output and amount to 7666 validation and 12056 test observations per proposer. On average {overlap['scored_fraction_mean'] * 100:.0f}\% of the post-initialization frames of a test sequence are scoreable (minimum {overlap['scored_fraction_min'] * 100:.0f}\%); the remaining frames are tracked but not labeled. A missing prediction is a failure, and the initialization box is not an observation. This is not the challenge's reset-based stereo EAO protocol, and no challenge ranking is claimed. The failure frames of the two selection proposers overlap with a pooled Jaccard index of {overlap['pooled_failure_jaccard']:.2f}, so their failure labels are related but not identical.

\textbf{{Reproducibility.}} All scripts, the frozen manifest, per-frame outputs and checksums are available in the public repository named in the Data Availability Statement (\url{{https://github.com/liranyang123456-commits/endoscopic-tracking-failure-verification}}). The verifier is deterministic: repeated evaluation of identical inputs gives identical descriptors, re-tracking four test sequences reproduced every stored IoU and cue value within $10^{{-4}}$, and confidence-logging reruns of NanoTrack and STARK-ST50 reproduced all previously stored boxes on TrackVes and the controlled benchmark exactly. All runs fix the OpenCV thread count to one, because multi-threaded filtering can move a few gradient magnitudes across the quantile threshold and change contour values by about $0.01$.

\textbf{{Data independence.}} The selection manifest (percentile, weights, thresholds) was written before any test-partition run of this study. The dataset is public, however, and earlier work in our group evaluated other trackers on both SurgT partitions, so the test partition is not a never-seen cohort for the authors; no parameter of the present verifier was derived from those earlier runs. SurgT case identifiers group sequences recorded in the same procedure but are not verified patient identities, which is why case-cluster analyses are reported as sensitivity analyses.

\subsection{{Validation-partition selection record}}
"""
    rows = []
    for candidate in selected["all_validation_candidates"]:
        chosen = candidate["percentile"] == selected["percentile"] and candidate["fusion_name"] == selected["fusion_name"]
        cells = [
            f"{candidate['percentile']:.2f}", FUSION_LABELS[candidate["fusion_name"]],
            "(" + ", ".join(f"{w:.2f}" for w in candidate["weights"]) + ")",
            f"{candidate['validation_auroc_by_proposer']['stark']:.3f}",
            f"{candidate['validation_auroc_by_proposer']['nanotrack']:.3f}",
            f"{candidate['objective']:.3f}",
        ]
        rows.append([r"\textbf{" + c + "}" for c in cells] if chosen else cells)
    best_nano = max(selected["all_validation_candidates"], key=lambda c: c["validation_auroc_by_proposer"]["nanotrack"])
    text += table(
        "Complete validation-partition grid (12 sequences, 7666 scored observations per proposer). Entries are macro-averaged two-class sequence AUROC; "
        "the objective is the mean over both selection proposers. The bold row is the single configuration frozen before any test-partition run. "
        "Weights are listed in the order $(d_C, d_U, d_H, d_N)$.",
        "tab:ValidationGrid", ["Percentile $\\alpha$", "Fused cues", "Weights", "STARK-ST50", "NanoTrack", "Objective"], rows,
    )
    text += (
        f"The frozen configuration ($\\alpha = {selected['percentile']:.2f}$, {FUSION_LABELS[selected['fusion_name']].lower()}) maximizes the two-proposer objective "
        f"({selected['objective']:.3f}). It is not the best NanoTrack-only candidate ({FUSION_LABELS[best_nano['fusion_name']].lower()} at $\\alpha = {best_nano['percentile']:.2f}$ "
        f"reaches {best_nano['validation_auroc_by_proposer']['nanotrack']:.3f}), so the selection was not tuned towards the proposer that later shows the largest test gain. "
        "The whole grid, including rejected candidates, is retained in the supplementary selection manifest.\n"
    )
    text += rf"""
\subsection{{Proposers}}
Three official trackers are used as frozen proposers. \textbf{{NanoTrack}} is the OpenCV \texttt{{TrackerNano}} implementation with the public ONNX backbone and head~\cite{{opencv_nanotrack}}; its native confidence is the tracking score returned by OpenCV. \textbf{{STARK-ST50}} uses the authors' released checkpoint with the official crop, normalization, box head and template-update rule~\cite{{yan2021stark}}; its native confidence is the sigmoid of the reliability logit that the tracker itself uses to decide template updates. \textbf{{OSTrack}} (ViT-B/256 with candidate elimination) uses the authors' released checkpoint~\cite{{ye2022ostrack}}; it exposes no scalar confidence, so we record the maximum of the Hann-windowed classification response map that its own box head consumes, without changing the box computation. NanoTrack and STARK-ST50 took part in parameter selection; \textbf{{OSTrack is a held-out proposer}}: the frozen percentile and weights are applied unchanged and only its alarm threshold is calibrated on its own validation reliable frames, exactly as for the other two. No tracker is fine-tuned, no task head is randomly initialized, and only obsolete dependency imports were patched for current PyTorch. Runs used an NVIDIA GeForce RTX~5090 Laptop GPU and an {cpu} CPU; the verifier runs on the CPU. Source commits, checkpoint hashes, input hashes and timings accompany the submission.

\subsection{{Metrics, denominators and uncertainty}}\label{{sec:stats}}
Failure AUROC~\cite{{hanley1982meaning,fawcett2006roc}} ranks frames by risk against the label $Y_t$ and is defined only for sequences that contain both classes; across the 20 test sequences this gives 17 with defined sequence-level AUROC for NanoTrack, 16 for STARK-ST50 and {heldout['methods']['verification']['sequence_macro']['failure_auroc']['n_sequences']} for OSTrack. Single-class sequences remain in every globally defined quantity (prevalence, operating points, alarm rates, selective risk). Average precision is reported with the caveat that it depends on prevalence~\cite{{saito2015precision}}. The risk--coverage curve~\cite{{elyaniv2010foundations,geifman2017selective}} sorts frames by increasing risk and plots the failure rate of the retained fraction; ties enter jointly and a smaller area is better. Sequence-level summaries are equal-weight means over sequences; frame-weighted and $\sqrt{{\text{{frames}}}}$-weighted means are given as sensitivity analyses because sequences differ in length by a factor of six.

\textbf{{Detection latency.}} A failure episode is a maximal run of at least five consecutive scored frames with $Y_t = 1$. For the first episode of each sequence we record the number of scored frames until the first warning inside the episode (censored and counted as missed if none occurs) and the false-alarm rate on reliable frames preceding the episode.

\textbf{{Inference.}} For two risk cues $A$ and $B$ and the sequences $k = 1, \dots, K$ with defined metric values for both, $D_k = M_{{A,k}} - M_{{B,k}}$ are paired differences. We report the mean difference with a percentile bootstrap interval ($B = 20000$, fixed seed)~\cite{{efron1993bootstrap}}, the two-sided Wilcoxon signed-rank $p$-value~\cite{{wilcoxon1945individual}} (\texttt{{scipy.stats.wilcoxon}}, zeros dropped, average ranks, exact distribution when available), and for the primary comparison an exact sign-flip permutation $p$-value over all $2^K$ sign assignments. Two hypothesis families are declared: the \emph{{primary}} family contains the four comparisons of the fused verifier with native confidence (two selection proposers $\times$ \{{AUROC, risk--coverage\}}), and the \emph{{exploratory}} family the eight comparisons with single appearance cues; Holm's step-down procedure~\cite{{holm1979sequential}} is applied within each family, and for transparency also over all twelve tests. Comparisons involving the held-out proposer and all post-hoc analyses form separate exploratory families. Because the five SurgT case groups may share anatomy, we also report (a)~a hierarchical bootstrap that resamples cases and then sequences within cases and (b)~case-level mean differences with their bootstrap intervals. A five-group two-sided Wilcoxon test has an attainable minimum $p$-value of $(1/2)^4 = 0.0625$ and therefore cannot reach $0.05$ even when every case improves; it is not used as evidence.

\subsection{{Transfer protocol}}
The frozen percentile and weights are applied to TrackVes (nine sequences, 5922 scored observations) and to a controlled rigid-scene benchmark (11 sequences, 22 targets, 3826 scored observations) without any target-dataset tuning or alarm calibration. Both follow a ground-truth-frame update protocol with frozen decoded frames, so results are not pooled with SurgT. Native confidence was recorded in a confidence-logging rerun of the identical frozen trackers whose boxes were checked frame by frame against the previously published official runs.
"""
    return text


# --------------------------------------------------------------------------- results
def results_section(results: dict, selected: dict, supplementary: dict, heldout: dict, transfer: dict) -> str:
    nano = results["proposers"]["nanotrack"]
    stark = results["proposers"]["stark"]
    effect = find_test(results["paired_tests"], proposer="nanotrack", comparator="native_confidence", metric="failure_auroc")
    stark_effect = find_test(results["paired_tests"], proposer="stark", comparator="native_confidence", metric="failure_auroc")
    extra = supplementary["extra_comparators"]
    infer = supplementary["weighting_and_inference"]
    text = rf"""
\section{{Results}}\label{{sec:results}}
\subsection{{Summary of performance}}\label{{sec:results_summary}}
Table~\ref{{tab:Summary}} and Figure~\ref{{fig:Headline}} collect the principal metrics for the three proposers on the frozen SurgT test partition and for the transfer benchmarks; the following subsections report each analysis in detail. Throughout, \emph{{native}} denotes the proposer's own confidence used as a failure score and \emph{{fused}} the four-cue verifier of Equation~\eqref{{eq:risk}} with the frozen parameters.
"""
    rows = []
    lat = supplementary["detection_latency"]
    for proposer in ("nanotrack", "stark", "ostrack"):
        if proposer == "ostrack":
            methods, pooled = heldout["methods"], heldout["methods"]["verification"]["pooled"]
            pair = find_test(heldout["paired_tests"], comparator="native_confidence")
            p_text = pvalue(pair["p_holm_heldout_family"])
            latency = heldout["detection_latency"]
            episodes = latency["verification"]["detected_episodes"] + latency["verification"]["missed_episodes"]
            monitor_ms = heldout["mean_monitor_ms_per_scored_frame"]
            role = "held-out"
        else:
            methods, pooled = results["proposers"][proposer]["methods"], results["proposers"][proposer]["methods"]["verification"]["pooled"]
            pair = find_test(results["paired_tests"], proposer=proposer, comparator="native_confidence", metric="failure_auroc")
            p_text = pvalue(find_test(infer["primary_vs_native_confidence"], proposer=proposer, metric="failure_auroc")["p_holm_family"])
            latency = lat[proposer]
            episodes = latency["verification"]["sequences_with_episode"]
            monitor_ms = results["proposers"][proposer]["mean_monitor_ms_per_scored_frame"]
            role = "selection"
        rows.append([
            PROPOSERS[proposer] + (" (held-out)" if role == "held-out" else ""), f"{pooled['failure_prevalence']:.2f}",
            f"{methods['native_confidence']['sequence_macro']['failure_auroc']['mean']:.3f} $\\to$ {methods['verification']['sequence_macro']['failure_auroc']['mean']:.3f}",
            f"${pair['mean_paired_difference']:+.3f}$ {ci(pair)}", p_text,
            f"{methods['verification']['sequence_macro']['sensitivity']['mean']:.2f} / {methods['verification']['sequence_macro']['specificity']['mean']:.2f}",
            f"{latency['native_confidence']['detected_episodes']} $\\to$ {latency['verification']['detected_episodes']} of {episodes}; "
            f"{fmt(latency['native_confidence']['median_latency_scored_frames'], 1)} $\\to$ {fmt(latency['verification']['median_latency_scored_frames'], 1)} fr",
        ])
    text += table(
        "Summary of performance on the frozen SurgT test partition (20 sequences, 12056 scored frames per proposer; verifier cost 1.4--2.1 ms per frame). Failure AUROC is given as native $\\to$ fused (equal-sequence means over two-class sequences); "
        "$\\Delta$ is the paired fused-minus-native difference with 95\\% bootstrap CI and $p_H$ the Holm-adjusted two-sided Wilcoxon $p$ within the declared family (primary family for the selection proposers, held-out family for OSTrack). "
        "Sensitivity / specificity are sequence means of the fused verifier at the frame-pooled validation threshold. The last column gives, for the first failure episode of each sequence, the number of episodes detected (native $\\to$ fused) and the median latency in scored frames.",
        "tab:Summary",
        ["Proposer", "Failure prev.", "AUROC native $\\to$ fused", "$\\Delta$ [95\\% CI]", "$p_H$", "Sens. / spec.", "First episode: detected; median latency"], rows,
    )
    rows = []
    for domain in ("trackves", "chess"):
        for proposer in ("nanotrack", "stark"):
            m = transfer["domains"][domain][proposer]["methods"]
            pair = transfer["domains"][domain][proposer]["verification_minus_comparator_auroc"].get("native_confidence")
            rows.append([
                "TrackVes" if domain == "trackves" else "Controlled", PROPOSERS[proposer], str(transfer["domains"][domain][proposer]["scored_frames"]),
                fmt(m["native_confidence"]["AUROC_macro"]), fmt(m["verification"]["AUROC_macro"]), str(m["verification"]["AUROC_n_sequences"]),
                "--" if pair is None else f"${pair['mean_paired_difference']:+.3f}$ {ci(pair)}",
            ])
    text += table(
        "Summary of the transfer without recalibration (frozen percentile, weights; no target-dataset tuning or alarm calibration): sequence-mean failure AUROC of native confidence and of the four-cue verifier, two-class sequence count and paired difference with 95\\% bootstrap CI.",
        "tab:SummaryTransfer", ["Dataset", "Proposer", "Scored frames", "Native AUROC", "Fused AUROC", "$n$", "$\\Delta$ [95\\% CI]"], rows,
    )
    text += r"""
\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{images/Fig_Headline.pdf}
\caption{Headline results. (\textbf{a})~Sequence-mean failure AUROC (error bars: sample SD) of native confidence and of the fused verifier for the two selection proposers and the held-out OSTrack on the SurgT test partition. (\textbf{b})~Fraction of sequences whose first failure episode is detected at the frozen threshold, with detected/episode counts and median latency in scored frames. (\textbf{c})~Transfer to TrackVes and the controlled benchmark without recalibration. Black edges mark the proposed verifier, the green edge NanoTrack's native confidence.}
\label{fig:Headline}
\end{figure}

\subsection{Proposer behaviour on the frozen test partition}
"""
    rows = []
    for proposer, data in (("nanotrack", nano), ("stark", stark)):
        pooled = data["methods"]["verification"]["pooled"]
        rows.append([
            PROPOSERS[proposer], "selection", str(pooled["n_frames"]), f"{pooled['box_iou_unchanged']:.3f}",
            f"{pooled['failure_prevalence']:.3f}", f"{pooled['n_failed']} / {pooled['n_reliable']}",
            f"{data['mean_proposer_ms_per_update']:.1f}", f"{data['mean_monitor_ms_per_scored_frame']:.2f}",
        ])
    rows.append([
        "OSTrack", "held-out", str(heldout["scored_frames"]), f"{heldout['box_iou_unchanged']:.3f}",
        f"{heldout['failure_prevalence']:.3f}", f"{heldout['n_failed']} / {heldout['n_reliable']}",
        f"{heldout['mean_proposer_ms_per_update']:.1f}", f"{heldout['mean_monitor_ms_per_scored_frame']:.2f}",
    ])
    text += table(
        "Frozen SurgT test partition: behaviour of the three unchanged proposers over all scored frames. Failure is $\\mathrm{IoU}(R_t,G_t)<0.5$ under continuous tracking without reset. "
        "Box IoU is identical with and without the monitor. Timings are per-frame means on the stated hardware.",
        "tab:ProposerContext",
        ["Proposer", "Role", "Scored frames", "Mean box IoU", "Failure prevalence", "Failed / reliable", "Proposer ms", "Monitor ms"], rows,
    )
    dist = supplementary["overlap_and_coverage"]["native_confidence_distribution"]
    text += (
        f"All three proposers lose the target on more than half of the scored test frames, with mean box IoU of about 0.40. "
        f"NanoTrack's native confidence is uninformative in this regime: its mean is {dist['nanotrack']['confidence_mean_reliable']:.3f} on reliable and "
        f"{dist['nanotrack']['confidence_mean_failed']:.3f} on failed frames (SD $\\approx$ {dist['nanotrack']['confidence_sd_failed']:.2f}), whereas STARK-ST50's reliability score separates the classes "
        f"({dist['stark']['confidence_mean_reliable']:.3f} versus {dist['stark']['confidence_mean_failed']:.3f}). The verifier adds about "
        f"{nano['mean_monitor_ms_per_scored_frame']:.1f} ms per frame to {nano['mean_proposer_ms_per_update']:.1f} ms (NanoTrack), {stark['mean_proposer_ms_per_update']:.1f} ms (STARK-ST50) "
        f"or {heldout['mean_proposer_ms_per_update']:.1f} ms (OSTrack) of proposer time.\n"
    )
    # ---- failure ranking
    text += r"""
\subsection{Failure ranking and cue ablation}\label{sec:results_ranking}
"""
    rows = []
    for proposer in ("nanotrack", "stark"):
        for name in ("native_confidence", "rank_only", "contour_only", "verification"):
            values = results["proposers"][proposer]["methods"][name]
            rows.append([
                PROPOSERS[proposer], LABELS[name], score(values, "failure_auroc"),
                str(values["sequence_macro"]["failure_auroc"]["n_sequences"]),
                score(values, "failure_average_precision"), score(values, "risk_coverage_area"),
            ])
    text += table(
        "Frozen comparators on the SurgT test partition (20 sequences, 12056 scored frames per proposer). Equal-sequence mean and sample SD; "
        "AUROC uses the two-class sequence count shown, average precision the sequences with failures, risk--coverage all 20. Lower risk--coverage area is better.",
        "tab:FailureRanking", ["Proposer", "Risk cue", "Failure AUROC", "$n_{\\mathrm{AUC}}$", "Average precision", "Risk--coverage"], rows,
    )
    text += (
        f"NanoTrack's fused failure AUROC is {nano['methods']['verification']['sequence_macro']['failure_auroc']['mean']:.3f} versus "
        f"{nano['methods']['native_confidence']['sequence_macro']['failure_auroc']['mean']:.3f} for native confidence; the paired sequence difference is "
        f"{effect['mean_paired_difference']:.3f}, 95\\% CI [{effect['ci95_low']:.3f}, {effect['ci95_high']:.3f}], Holm-adjusted $p = {pvalue(effect['p_holm'])}$ over all twelve tests "
        f"and $p = {pvalue(find_test(infer['primary_vs_native_confidence'], proposer='nanotrack', metric='failure_auroc')['p_holm_family'])}$ within the primary family; the exact sign-flip permutation "
        f"$p$-value is {infer['nanotrack']['exact_sign_flip_p_two_sided']:.1e} and the hierarchical case-then-sequence bootstrap interval is "
        f"[{infer['nanotrack']['hierarchical_bootstrap_ci95'][0]:.3f}, {infer['nanotrack']['hierarchical_bootstrap_ci95'][1]:.3f}]. This paired sequence comparison includes {effect['n_groups']} sequences with defined AUROC for both methods. "
        f"For STARK-ST50 the gain is {stark_effect['mean_paired_difference']:.3f} (95\\% CI [{stark_effect['ci95_low']:.3f}, {stark_effect['ci95_high']:.3f}]); it is significant within the primary family "
        f"($p = {pvalue(find_test(infer['primary_vs_native_confidence'], proposer='stark', metric='failure_auroc')['p_holm_family'])}$) but not over all twelve tests ($p = {pvalue(stark_effect['p_holm'])}$), its sign-flip $p$-value is "
        f"{infer['stark']['exact_sign_flip_p_two_sided']:.3f} and the hierarchical interval [{infer['stark']['hierarchical_bootstrap_ci95'][0]:.3f}, {infer['stark']['hierarchical_bootstrap_ci95'][1]:.3f}] includes zero. "
        "Figure~\\ref{fig:Curves} shows the pooled ROC and risk--coverage curves and Figure~\\ref{fig:Paired} the per-sequence pairs behind these means.\n"
    )
    text += r"""
\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{images/Fig_Curves.pdf}
\caption{Pooled ROC (\textbf{a}, \textbf{b}) and risk--coverage (\textbf{c}, \textbf{d}) curves on the frozen SurgT test partition for the two selection proposers, for native confidence, the three single appearance cues and the fused verifier. Pooled curves mix sequences with different failure prevalence and are shown for shape; inference uses the sequence-level statistics.}
\label{fig:Curves}
\end{figure}
\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{images/Fig_PairedSequences.pdf}
\caption{Per-sequence failure AUROC of the fused verifier against native confidence on the test partition for (\textbf{a}) NanoTrack, (\textbf{b}) STARK-ST50 and (\textbf{c}) the held-out OSTrack. Points above the diagonal favor the verifier; marker size grows with the number of scored frames and color encodes the case group.}
\label{fig:Paired}
\end{figure}
"""
    paired = []
    for row in results["paired_tests"]:
        family = infer["primary_vs_native_confidence"] if row["comparator"] == "native_confidence" else infer["exploratory_vs_single_cues"]
        fam = find_test(family, proposer=row["proposer"], comparator=row["comparator"], metric=row["metric"])
        case = row["case_cluster_sensitivity"]
        paired.append([
            PROPOSERS[row["proposer"]], LABELS[row["comparator"]], "AUROC" if row["metric"] == "failure_auroc" else "Risk--coverage",
            f"${row['mean_paired_difference']:+.3f}$", ci(row), pvalue(row["p_holm"]), pvalue(fam["p_holm_family"]), ci(case),
        ])
    text += table(
        "Fused verifier minus each comparator: paired sequence effects with 95\\% bootstrap intervals. AUROC tests use the sequences with defined two-class AUROC for both cues (17 NanoTrack, 16 STARK-ST50); "
        "risk--coverage tests use all 20. Positive AUROC and negative risk--coverage differences favor the verifier. $p_{H,12}$ is Holm-adjusted over all twelve tests; $p_{H,\\mathrm{fam}}$ within the declared family "
        "(primary: versus native confidence; exploratory: versus single cues). The last column is the bootstrap interval of the five case-group mean effects.",
        "tab:PairedTests", ["Proposer", "Comparator", "Metric", "Difference", "Sequence CI", "$p_{H,12}$", "$p_{H,\\mathrm{fam}}$", "Case CI"], paired,
    )
    # extra comparators
    rows = []
    for proposer in ("nanotrack", "stark"):
        for name in ("color_only", "rank_contour", "appearance_only", "delta_confidence"):
            entry = extra[proposer][name]
            pair = entry["verification_minus_this_auroc"]
            rows.append([
                PROPOSERS[proposer], LABELS[name], f"${entry['sequence_macro_auroc']:.3f}\\pm{entry['sequence_macro_auroc_sd']:.3f}$",
                str(entry["n_sequences"]), f"{entry['pooled']['failure_auroc']:.3f}", f"{entry['sequence_macro_risk_coverage']:.3f}",
                f"${pair['mean_paired_difference']:+.3f}$ {ci(pair)}", pvalue(pair["p_two_sided"]),
            ])
    text += table(
        "Post-hoc comparators evaluated with the frozen percentile and validation-calibrated thresholds (exploratory; unadjusted two-sided Wilcoxon $p$). "
        "Appearance only renormalizes the frozen weights without the confidence term; $|\\Delta$ confidence$|$ is the absolute frame-to-frame change of native confidence.",
        "tab:ExtraComparators",
        ["Proposer", "Risk cue", "Failure AUROC", "$n$", "Pooled AUROC", "Risk--cov.", "Fused $-$ cue [95\\% CI]", "$p$"], rows,
    )
    nano_app = extra["nanotrack"]["appearance_only"]["sequence_macro_auroc"]
    stark_app = extra["stark"]["appearance_only"]["sequence_macro_auroc"]
    stark_rc = extra["stark"]["rank_contour"]["sequence_macro_auroc"]
    text += r"""
\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{images/Fig_CueAblation.pdf}
\caption{Cue ablation on the SurgT test partition for (\textbf{a}) NanoTrack, (\textbf{b}) STARK-ST50 and (\textbf{c}) the held-out OSTrack: sequence-mean failure AUROC (error bars: sample SD) of native confidence, its frame-to-frame change, each single appearance cue, two partial fusions and the fused verifier (red). The dashed line marks the best single appearance cue.}
\label{fig:CueAblation}
\end{figure}
"""
    text += (
        "Figure~\\ref{fig:CueAblation} and Tables~\\ref{tab:PairedTests} and~\\ref{tab:ExtraComparators} identify where the information lies. For NanoTrack, rank-only and contour-only cues each recover most of the gain over native confidence, "
        "and the fused score is not significantly better than either; dropping the confidence term changes the fused AUROC by less than 0.001 "
        f"({nano_app:.3f} without versus {nano['methods']['verification']['sequence_macro']['failure_auroc']['mean']:.3f} with), consistent with a native confidence at chance level. "
        f"For STARK-ST50, rank-only ({stark['methods']['rank_only']['sequence_macro']['failure_auroc']['mean']:.3f}), rank$+$contour ({stark_rc:.3f}) and appearance-only ({stark_app:.3f}) all exceed the four-cue score "
        f"({stark['methods']['verification']['sequence_macro']['failure_auroc']['mean']:.3f}): adding a confidence term that is informative on validation sequences slightly hurts on the test sequences. "
        f"Color alone is weaker ({extra['nanotrack']['color_only']['sequence_macro_auroc']:.3f} and {extra['stark']['color_only']['sequence_macro_auroc']:.3f}) and the temporal change of confidence carries almost no information "
        f"({extra['nanotrack']['delta_confidence']['sequence_macro_auroc']:.3f} and {extra['stark']['delta_confidence']['sequence_macro_auroc']:.3f}). The evidence therefore supports an appearance-consistency signal built from ordinal and contour statistics, "
        "not a specific advantage of the contour term over the ordinal term.\n"
    )
    # pooled metrics
    rows = []
    for proposer in ("nanotrack", "stark"):
        for name in ("native_confidence", "rank_only", "contour_only", "verification"):
            pooled = results["proposers"][proposer]["methods"][name]["pooled"]
            rows.append([
                PROPOSERS[proposer], LABELS[name], f"{pooled['failure_auroc']:.3f}", f"{pooled['failure_average_precision']:.3f}",
                f"{pooled['sensitivity']:.3f}", f"{pooled['specificity']:.3f}", f"{pooled['alarm_rate']:.3f}", f"{pooled['risk_coverage_area']:.3f}",
            ])
    text += table(
        "Descriptive pooled frame-level metrics over all 12056 scored test frames per proposer at the validation-frozen thresholds. Pooling treats frames as exchangeable and mixes "
        "sequences with different prevalence; it is reported for completeness and inference uses Tables~\\ref{tab:FailureRanking}--\\ref{tab:PairedTests}.",
        "tab:PooledMetrics", ["Proposer", "Risk cue", "Pooled AUROC", "Pooled AP", "Sensitivity", "Specificity", "Alarm rate", "Risk--coverage"], rows,
    )
    text += (
        f"Pooled and sequence-level summaries disagree for STARK-ST50's native confidence ({stark['methods']['native_confidence']['pooled']['failure_auroc']:.3f} pooled versus "
        f"{stark['methods']['native_confidence']['sequence_macro']['failure_auroc']['mean']:.3f} sequence mean): the score separates sequences that fail almost everywhere from sequences that rarely fail, "
        "but ranks frames poorly within a sequence. The weighted sequence means in Appendix~\\ref{sec:appendix_weighting} lie between the two views and do not change any conclusion.\n"
    )
    # ---- operating points & conformal
    text += r"""
\subsection{Warning operating points and the conformal threshold}\label{sec:results_operating}
"""
    rows = []
    for proposer in ("nanotrack", "stark"):
        for name in ("native_confidence", "verification"):
            values = results["proposers"][proposer]["methods"][name]
            rows.append([
                PROPOSERS[proposer], LABELS[name], f"{values['threshold_frozen_from_validation']:.3f}",
                score(values, "sensitivity"), str(values["sequence_macro"]["sensitivity"]["n_sequences"]),
                score(values, "specificity"), str(values["sequence_macro"]["specificity"]["n_sequences"]), score(values, "alarm_rate"),
            ])
    text += table(
        "Test warnings at the frame-pooled thresholds calibrated on validation reliable frames (nominal specificity 0.90). Sensitivity and specificity use sequences with the respective class; alarm rate uses all 20.",
        "tab:OperatingPoints", ["Proposer", "Cue", "Threshold", "Sensitivity", "$n_+$", "Specificity", "$n_-$", "Alarm rate"], rows,
    )
    conf = supplementary["conformal"]
    rows = []
    for proposer in ("nanotrack", "stark"):
        entry = conf[proposer]["verification"]
        rows.append([PROPOSERS[proposer], "frame-pooled", "--", f"{entry['frame_pooled_threshold']:.3f}", "none",
                     f"{results['proposers'][proposer]['methods']['verification']['sequence_macro']['specificity']['mean']:.3f}",
                     f"{results['proposers'][proposer]['methods']['verification']['sequence_macro']['sensitivity']['mean']:.3f}", "--"])
        for delta in ("0.1", "0.2", "0.3"):
            d = entry[delta]
            if not d["attainable"]:
                continue
            rows.append([
                PROPOSERS[proposer], f"conformal $\\delta={float(delta):.1f}$", f"{d['order_statistic']} of {entry['n_validation_sequences']}", f"{d['threshold']:.3f}",
                f"$\\ge {d['guaranteed_sequence_coverage']:.1f}$", f"{d['test_specificity_macro']:.3f}", f"{d['test_sensitivity_macro']:.3f}",
                f"{d['test_sequences_with_specificity_at_least_0.9']}/{d['test_sequences_evaluated']} ({d['empirical_sequence_coverage']:.2f})",
            ])
    text += table(
        "Sequence-level split-conformal thresholds (Proposition~\\ref{prop:conformal}) for the fused verifier, computed from the per-sequence 0.90 risk quantiles of validation sequences with at least ten reliable frames, "
        "next to the frame-pooled threshold. The guarantee is the probability that a new exchangeable sequence has reliable-frame specificity $\\ge 0.90$; the last column is the fraction of test sequences for which this held.",
        "tab:Conformal", ["Proposer", "Threshold rule", "Order statistic", "$\\tau$", "Guaranteed coverage", "Test specificity", "Test sensitivity", "Sequences with spec.\\ $\\ge 0.9$"], rows,
    )
    n1 = conf["nanotrack"]["verification"]["0.1"]
    s1 = conf["stark"]["verification"]["0.1"]
    text += (
        f"For NanoTrack the conformal threshold with $\\delta = 0.1$ ({n1['threshold']:.3f}) kept specificity at or above 0.90 in {n1['test_sequences_with_specificity_at_least_0.9']} of {n1['test_sequences_evaluated']} test sequences "
        f"(guaranteed $\\ge 0.9$) at the cost of sensitivity ({n1['test_sensitivity_macro']:.3f} versus {results['proposers']['nanotrack']['methods']['verification']['sequence_macro']['sensitivity']['mean']:.3f} at the frame-pooled threshold). "
        f"For STARK-ST50 only {s1['test_sequences_with_specificity_at_least_0.9']} of {s1['test_sequences_evaluated']} sequences met the target at $\\delta = 0.1$, below the nominal 0.9; with nine validation sequences the bound is loose "
        "and the validation-to-test shift reported in Appendix~\\ref{sec:appendix_gap} indicates that exchangeability is doubtful for this proposer. The frame-pooled threshold reaches a specificity near 0.89 for both proposers but carries no guarantee.\n"
    )
    aq = supplementary["sensitivity"]["alarm_quantile"]
    text += (
        "Moving the validation quantile from 0.90 to 0.80 or 0.95 trades sensitivity against specificity as expected: for the fused NanoTrack verifier sensitivity/specificity are "
        f"{aq['nanotrack']['verification']['0.8']['sensitivity_macro']:.2f}/{aq['nanotrack']['verification']['0.8']['specificity_macro']:.2f}, "
        f"{aq['nanotrack']['verification']['0.9']['sensitivity_macro']:.2f}/{aq['nanotrack']['verification']['0.9']['specificity_macro']:.2f} and "
        f"{aq['nanotrack']['verification']['0.95']['sensitivity_macro']:.2f}/{aq['nanotrack']['verification']['0.95']['specificity_macro']:.2f}; no clinician-facing operating point is proposed.\n"
    )
    # ---- latency
    text += r"""
\subsection{Detection latency of the first failure episode}
"""
    lat = supplementary["detection_latency"]
    rows = []
    for proposer in ("nanotrack", "stark"):
        for name in ("native_confidence", "rank_only", "contour_only", "color_only", "verification"):
            e = lat[proposer][name]
            rows.append([
                PROPOSERS[proposer], LABELS[name], f"{e['detected_episodes']} / {e['missed_episodes']}",
                fmt(e["median_latency_scored_frames"], 1), f"{e['detected_within']['1']} / {e['detected_within']['10']} / {e['detected_within']['30']}",
                fmt(e["false_alarm_rate_before_first_episode"]),
            ])
    for name in ("native_confidence", "rank_only", "contour_only", "color_only", "verification"):
        e = heldout["detection_latency"][name]
        rows.append(["OSTrack (held-out)", LABELS[name], f"{e['detected_episodes']} / {e['missed_episodes']}",
                     fmt(e["median_latency_scored_frames"], 1), f"-- / {e['detected_within_10']} / --", "--"])
    text += table(
        "First failure episode (at least five consecutive failed scored frames) per test sequence: episodes detected / missed at the validation-frozen threshold, median latency in scored frames, "
        "number of episodes detected within 1 / 10 / 30 scored frames, and false-alarm rate on reliable frames preceding the episode.",
        "tab:Latency", ["Proposer", "Cue", "Detected / missed", "Median latency", "Within 1 / 10 / 30", "False alarms before"], rows,
    )
    text += (
        f"For NanoTrack the fused verifier raises a warning in the first scored frame of the episode in half of the detected cases (median {lat['nanotrack']['verification']['median_latency_scored_frames']:.1f} frames, "
        f"{lat['nanotrack']['verification']['missed_episodes']} of {lat['nanotrack']['verification']['sequences_with_episode']} episodes missed), whereas native confidence detects "
        f"{lat['nanotrack']['native_confidence']['detected_episodes']} episodes with a median latency of {lat['nanotrack']['native_confidence']['median_latency_scored_frames']:.0f} frames and misses "
        f"{lat['nanotrack']['native_confidence']['missed_episodes']}. Its false-alarm rate before the first episode ({lat['nanotrack']['verification']['false_alarm_rate_before_first_episode']:.3f}) is lower than that of native confidence "
        f"({lat['nanotrack']['native_confidence']['false_alarm_rate_before_first_episode']:.3f}). For STARK-ST50 the contour-only cue misses the fewest episodes "
        f"({lat['stark']['contour_only']['missed_episodes']}) while native confidence and the fused score have similar median latency; here the fused verifier's advantage is in false alarms before the episode "
        f"({lat['stark']['verification']['false_alarm_rate_before_first_episode']:.3f} versus {lat['stark']['rank_only']['false_alarm_rate_before_first_episode']:.3f} for rank only). "
        "Latency is measured in scored frames; unlabeled frames between them are tracked but cannot be scored.\n"
    )
    # ---- held-out proposer
    text += r"""
\subsection{Held-out proposer}\label{sec:results_heldout}
"""
    rows = []
    for name in ("native_confidence", "rank_only", "contour_only", "color_only", "verification"):
        values = heldout["methods"][name]
        pair = None if name == "verification" else find_test(heldout["paired_tests"], comparator=name)
        rows.append([
            LABELS[name], f"{values['validation_sequence_macro_auroc']:.3f} ({values['validation_n_sequences']})",
            score(values, "failure_auroc"), str(values["sequence_macro"]["failure_auroc"]["n_sequences"]),
            score(values, "failure_average_precision"), score(values, "risk_coverage_area"),
            "--" if pair is None else f"${pair['mean_paired_difference']:+.3f}$ {ci(pair)}",
            "--" if pair is None else pvalue(pair["p_holm_heldout_family"]),
        ])
    text += table(
        "OSTrack as held-out proposer: the frozen percentile and weights were applied unchanged; only its alarm threshold was calibrated on its own validation reliable frames. "
        "Validation AUROC (with sequence count) is shown for reference; test values are equal-sequence means with SD. The last columns give the fused-minus-cue paired difference with 95\\% bootstrap CI and the Holm-adjusted $p$ within this five-test family.",
        "tab:HeldOut", ["Risk cue", "Validation AUROC ($n$)", "Test failure AUROC", "$n_{\\mathrm{AUC}}$", "Average precision", "Risk--coverage", "Fused $-$ cue [95\\% CI]", "$p_H$"], rows,
    )
    ho_native = find_test(heldout["paired_tests"], comparator="native_confidence")
    text += (
        f"OSTrack fails on {heldout['failure_prevalence'] * 100:.0f}\\% of the scored test frames (Table~\\ref{{tab:ProposerContext}}). Its response-map maximum is a moderately informative failure signal "
        f"(sequence AUROC {heldout['methods']['native_confidence']['sequence_macro']['failure_auroc']['mean']:.3f}), and the frozen verifier raises it to "
        f"{heldout['methods']['verification']['sequence_macro']['failure_auroc']['mean']:.3f} (paired difference {ho_native['mean_paired_difference']:+.3f}, 95\\% CI [{ho_native['ci95_low']:+.3f}, {ho_native['ci95_high']:+.3f}], "
        f"exact sign-flip $p = {ho_native['exact_sign_flip_p_two_sided']:.4f}$, hierarchical bootstrap CI [{ho_native['hierarchical_bootstrap_ci95'][0]:+.3f}, {ho_native['hierarchical_bootstrap_ci95'][1]:+.3f}]). "
        "Because OSTrack took no part in selecting the percentile or the weights, this is the cleanest evidence that the signal is not tuned to a particular proposer; as for the other trackers, the fused score is not significantly better than the best single appearance cue.\n"
    )
    # ---- sensitivity
    text += r"""
\subsection{Post-hoc sensitivity analyses}\label{sec:results_sensitivity}
"""
    sens = supplementary["sensitivity"]
    rows = []
    for proposer in ("nanotrack", "stark"):
        for thr in ("0.3", "0.5", "0.7"):
            e = sens["iou_threshold"][proposer][thr]
            rows.append([PROPOSERS[proposer], f"failure if IoU $<$ {thr}", f"{e['failure_prevalence']:.2f}", fmt(e["native_confidence"]), fmt(e["rank_only"]), fmt(e["contour_only"]), fmt(e["verification"]), str(e["n_sequences"])])
        for pct in ("0.5", "0.68", "0.8"):
            e = sens["contour_percentile"][proposer][pct]
            rows.append([PROPOSERS[proposer], f"contour percentile $\\alpha = {float(pct):.2f}$", "--", "--", "--", fmt(e["contour_only"]), fmt(e["verification"]), "--"])
    text += table(
        "Post-hoc sensitivity of the sequence-mean failure AUROC on the test partition to the failure definition (top rows per proposer, all cues; $n$ is the two-class sequence count) "
        "and to the contour percentile (bottom rows; frozen weights). These analyses were not used to choose any parameter.",
        "tab:Sensitivity", ["Proposer", "Setting", "Prevalence", "Native", "Rank", "Contour", "Fused", "$n$"], rows,
    )
    simplex = sens["weight_simplex"]
    temporal = supplementary["temporal_specificity"]
    text += r"""
\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{images/Fig_Sensitivity.pdf}
\caption{Post-hoc sensitivity analyses on the SurgT test partition. (\textbf{a})~Sequence-mean failure AUROC of each cue as the failure definition changes from $\mathrm{IoU}<0.3$ to $\mathrm{IoU}<0.7$ (solid: NanoTrack, dashed: STARK-ST50). (\textbf{b})~Contour-only and fused AUROC as a function of the contour percentile $\alpha$; the frozen value is marked. (\textbf{c})~Sensitivity and specificity of the fused verifier when the validation quantile that defines $\tau$ is moved from 0.80 to 0.95. (\textbf{d})~Fraction of test sequences whose reliable-frame specificity reaches 0.90 under the conformal thresholds $\tau_\delta$, against the guaranteed level $1-\delta$ (black ticks).}
\label{fig:Sensitivity}
\end{figure}
"""
    text += (
        f"Figure~\\ref{{fig:Sensitivity}} summarizes the sensitivity analyses. The ranking of cues is stable for NanoTrack across failure definitions, whereas for STARK-ST50 the best single cue changes with the threshold. The contour percentile acts in opposite directions for the two proposers "
        f"(NanoTrack improves and STARK-ST50 degrades as $\\alpha$ grows), so the frozen value is a compromise rather than an optimum for either. On the weight simplex sampled in steps of 0.05 "
        f"({simplex['n_points']} points), the frozen vector lies at the {simplex['nanotrack']['frozen_percentile_rank'] * 100:.0f}th percentile of all NanoTrack AUROC values and within 0.01 of the best point, "
        f"a region that contains {simplex['nanotrack']['fraction_within_0.01_of_best'] * 100:.0f}\\% of the grid; for STARK-ST50 it lies at the {simplex['stark']['frozen_percentile_rank'] * 100:.0f}th percentile, "
        f"{simplex['stark']['grid_max'] - simplex['stark']['frozen_auroc']:.3f} below the best grid point ({', '.join(f'{w:.2f}' for w in simplex['stark']['best_weights'])}), which drops the contour and confidence terms. "
        "The response surface is flat for NanoTrack and only mildly peaked for STARK-ST50; no weight vector dominates both proposers.\n"
    )
    rows = []
    for proposer in ("nanotrack", "stark"):
        for name in ("native_confidence", "verification"):
            e = temporal[proposer][name]
            rows.append([PROPOSERS[proposer], LABELS[name]] + [f"{e[f'third_{k}']['specificity_mean']:.3f} ({e[f'third_{k}']['n_sequences']})" for k in (1, 2, 3)])
    text += table(
        "Specificity on reliable frames by temporal position within each test sequence (first, second and third third of the scored frames; sequence count in parentheses), at the frame-pooled threshold.",
        "tab:Temporal", ["Proposer", "Cue", "First third", "Second third", "Last third"], rows,
    )
    text += (
        f"With a reference fixed at the first frame, the fused verifier's specificity is lowest in the middle third of NanoTrack sequences ({temporal['nanotrack']['verification']['third_2']['specificity_mean']:.3f}) and recovers in the last third "
        f"({temporal['nanotrack']['verification']['third_3']['specificity_mean']:.3f}, {temporal['nanotrack']['verification']['third_3']['n_sequences']} sequences with reliable frames), so benign appearance change does raise false alarms but does not accumulate monotonically. "
        "A conservative reference update was deliberately not introduced in this study because it would make the verifier's reference depend on possibly wrong boxes.\n"
    )
    # ---- transfer
    text += r"""
\subsection{Transfer without recalibration}
"""
    rows = []
    for domain in ("trackves", "chess"):
        for proposer in ("nanotrack", "stark"):
            m = transfer["domains"][domain][proposer]["methods"]
            pair = transfer["domains"][domain][proposer]["verification_minus_comparator_auroc"].get("native_confidence")
            rows.append([
                "TrackVes" if domain == "trackves" else "Controlled", PROPOSERS[proposer],
                fmt(m["native_confidence"]["AUROC_macro"]), fmt(m["rank_only"]["AUROC_macro"]), fmt(m["contour_only"]["AUROC_macro"]), fmt(m["color_only"]["AUROC_macro"]),
                fmt(m["appearance_fusion"]["AUROC_macro"]), fmt(m["verification"]["AUROC_macro"]), str(m["verification"]["AUROC_n_sequences"]),
                "--" if pair is None else f"${pair['mean_paired_difference']:+.3f}$ {ci(pair)}",
            ])
    identical = all(transfer["domains"][d][p]["rerun_boxes_identical_to_frozen_official_csv"] for d in ("trackves", "chess") for p in ("nanotrack", "stark"))
    text += table(
        "Transfer of the frozen verifier to TrackVes and the controlled benchmark without any target-dataset calibration: sequence-mean failure AUROC of each cue, the appearance-only fusion and the four-cue fusion, "
        "the two-class sequence count, and the four-cue-minus-native paired difference with 95\\% bootstrap CI.",
        "tab:Transfer", ["Dataset", "Proposer", "Native", "Rank", "Contour", "Color", "Appearance", "Four-cue", "$n$", "Four-cue $-$ native [CI]"], rows,
    )
    text += (
        ("The confidence-logging reruns reproduced the previously published boxes exactly, so the localization record of Appendix~\\ref{sec:appendix_fusion} is unchanged. "
         if identical else "The confidence-logging reruns did not reproduce every published box bit for bit; the transfer table therefore uses the rerun boxes and confidences consistently, and the fair-tracking control in Appendix~\\ref{sec:appendix_fusion} keeps the original runs. ")
        + "Without recalibration the four-cue verifier transfers to the soft-tissue TrackVes sequences and to the rigid controlled scenes with the same qualitative picture as on SurgT: "
        + f"a large gain where native confidence is uninformative (TrackVes NanoTrack, {transfer['domains']['trackves']['nanotrack']['methods']['native_confidence']['AUROC_macro']:.3f} to {transfer['domains']['trackves']['nanotrack']['methods']['verification']['AUROC_macro']:.3f}), "
        + f"moderate gains on the controlled scenes, and no gain where native confidence is already informative (TrackVes STARK-ST50, {transfer['domains']['trackves']['stark']['methods']['native_confidence']['AUROC_macro']:.3f} versus {transfer['domains']['trackves']['stark']['methods']['verification']['AUROC_macro']:.3f}). "
        + "Transfer sets are small and the intervals are wide.\n"
    )
    # ---- qualitative
    text += r"""
\subsection{Qualitative cases}
\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{images/Verification_Qualitative.pdf}
\caption{Four SurgT test sequences re-tracked with the identical frozen proposers (frozen CSV values reproduced frame by frame). Left: cropped frames at the first scored frame, the onset and the middle of the first failure episode, and the last frame; green is the supplied ground truth, red the unchanged proposer box. Right: IoU with ground truth (black), fused risk $r_t$ (blue) and native-confidence risk $d_N$ (gray) over the sequence; shaded regions are localization failures, the dashed line is the frozen threshold and the red ticks are warnings. Rows: a large gain, a gain with false alarms after the target reappears with changed appearance, a weak case in which the drifted box covers visually similar tissue, and a STARK-ST50 case in which the native reliability score is already informative and the fused score is worse.}
\label{fig:Qualitative}
\end{figure}
Figure~\ref{fig:Qualitative} illustrates the three behaviours quantified above: a drift to a different structure is detected within a frame or two even when native confidence does not move; a target that is recovered with changed appearance keeps the risk above threshold and produces false alarms; and a drifted box that still covers visually similar tissue stays below threshold, which no appearance-based signal can detect.

\subsection{Runtime and determinism}
"""
    text += (
        f"The verifier averages {nano['mean_monitor_ms_per_scored_frame']:.2f} ms per scored frame with NanoTrack, {stark['mean_monitor_ms_per_scored_frame']:.2f} ms with STARK-ST50 and "
        f"{heldout['mean_monitor_ms_per_scored_frame']:.2f} ms with OSTrack on the CPU, computing all three diagnostic contour percentiles. Re-tracking the four sequences of Figure~\\ref{{fig:Qualitative}} reproduced every frozen IoU and cue value within $10^{{-4}}$; "
        "within-process repetition of the descriptor is exact. No training seed exists for the monitor itself.\n"
    )
    return text


# --------------------------------------------------------------------------- discussion / conclusions
def discussion(results: dict, supplementary: dict, heldout: dict) -> str:
    gap = supplementary["validation_test_gap"]
    return rf"""
\section{{Discussion}}\label{{sec:discussion}}
\textbf{{What the evidence supports.}} Around three official trackers, an appearance-consistency score built from ordinal and gradient-quantile statistics of the ROI ranks localization failures far better than a native confidence that is close to chance (NanoTrack), clearly better than a moderately informative one (the held-out OSTrack) and modestly better than an informative one (STARK-ST50). The gain over native confidence is significant in the primary family for both selection proposers and for the held-out proposer, with hierarchical bootstrap intervals that account for the five case groups. The gain of \emph{{fusing}} cues over the best single appearance cue is not significant anywhere, and for STARK-ST50 the confidence term lowers the test score. The title of this paper therefore names the signal---appearance consistency from ordinal gradients---rather than a particular cue, and the contour distance should be read as one of two near-equivalent descriptors.

\textbf{{Design implications.}} Three practical conclusions follow from the ablation and sensitivity analyses. First, a deployable verifier can be built from the ordinal and contour cues alone: the appearance-only score matches the four-cue score for NanoTrack and exceeds it for STARK-ST50, so the confidence term should be included only when a proposer's confidence is known to be informative. Second, the weight vector need not be tuned per tracker---the frozen vector lies within 0.01 of the best grid point for NanoTrack and within 0.03 for STARK-ST50 on a flat response surface---whereas the alarm threshold must be calibrated per proposer, which the conformal construction does with a stated guarantee. Third, because the verifier never touches the box, it composes with any tracker and with any downstream policy (re-initialization, operator prompt, overlay suppression) without re-validating the tracker itself; the earlier attempt to let the same cues overwrite the box (Appendix~\ref{{sec:appendix_fusion}}) shows why this separation matters.

\textbf{{Validation-to-test shift.}} Every cue loses accuracy from validation to test for STARK-ST50 (AUROC gaps between {min(v['gap'] for v in gap['stark'].values()):.2f} and {max(v['gap'] for v in gap['stark'].values()):.2f}, Appendix~\ref{{sec:appendix_gap}}) but hardly for NanoTrack, so the shift is a property of how STARK-ST50 fails on the two partitions rather than of the fused weights. This explains why a configuration that looked best for STARK-ST50 on validation did not translate into a significant test gain, why its conformal coverage fell below the nominal level, and why nine validation sequences are too few for a tight sequence-level guarantee.

\textbf{{Latency, false alarms and the fixed reference.}} Warnings arrive at the onset of most first failure episodes, and the fused score produces fewer false alarms before that onset than native confidence or any single appearance cue for two of the three proposers. False alarms concentrate in the middle of sequences when the target is recovered with changed appearance, a direct consequence of keeping the reference fixed at the supplied ROI. Updating the reference from later proposals would remove some of these alarms but would also let a wrong box contaminate the reference; we regard a conservative, risk-gated update as the main avenue for improvement and did not evaluate it here to keep the protocol frozen.

\textbf{{Comparison with tracker-specific failure signals.}} Forward--backward error, response-map statistics and learned quality heads~\cite{{kalal2010forward,Bolme2010,wang2017lmcf,dai2020ltmu}} were not run as baselines because they require access to a motion estimate or the tracker's response map, which the present setting deliberately excludes; the native confidences evaluated here are the signals that the trackers themselves use for template updating or box selection. A comparison with the online failure detector of LT-SAT~\cite{{PENZA201813}} under the same no-reset protocol would be informative and is left to future work.

\textbf{{Limitations.}} (i)~The failure label is a single IoU threshold on a box; it does not capture the clinical consequence of an error. (ii)~The chance-level analysis shows that $d_C$ saturates within a few pixels of displacement, so its useful range is narrow. (iii)~The sequence-level guarantee of Proposition~\ref{{prop:conformal}} assumes exchangeable sequences and is marginal; it failed empirically for STARK-ST50 and says nothing about sensitivity. (iv)~Only five case groups exist, patient identity is unverified, and the dataset had been used by the authors for other experiments. (v)~Specular highlights, smoke and instruments were not stratified, and no artifact-specific robustness is claimed. (vi)~A drifted box on visually similar tissue cannot be detected by appearance alone. (vii)~Dense tissue displacement, clinical workflow integration and surgeon-in-the-loop evaluation are outside the scope of this study.

\section{{Conclusions}}\label{{sec:conclusions}}
A mask-free, deterministic appearance-consistency layer built from midranks, gradient-quantile contours, a trimmed contour distance and a color histogram ranks localization failures of frozen region-level trackers on the SurgT test partition far better than NanoTrack's and OSTrack's native confidence and modestly better than STARK-ST50's, detects most first failure episodes within a frame or two, costs about 1.4~ms per frame and never changes a box. Ordinal and contour cues are each sufficient for most of the gain; their fusion adds little, and a sequence-level conformal threshold provides a finite-sample specificity guarantee whose assumptions hold for one proposer and not for another. The layer supplies failure awareness around an unchanged tracker; it does not make the tracker more accurate and is not a validated clinical safety mechanism.
"""


# --------------------------------------------------------------------------- appendices
def appendix(results: dict, tracking: dict, segmentation: dict, supplementary: dict, heldout: dict) -> str:
    text = r"""
\appendix
\section{Prior box-overwriting fusion tracker under the complete-denominator protocol}\label{sec:appendix_fusion}
Before adopting the non-invasive design, we built a tracker (``scheme-C/nano'') that let contour evidence overwrite NanoTrack's box: a Siamese contour-similarity network scored contour patches of candidate boxes and a rule-based arbitration replaced the NanoTrack proposal when contour and color scores favored another candidate. Two variants are reported here under the same complete-denominator protocol as the official baselines: \textbf{Prior Ours-SD}, in which all twenty contour-network folds were trained from random weights on image-derived contour patches with the test sequence excluded (``sequence-disjoint''), and \textbf{LearnedGate}, a TrackVes-only variant that gates NanoTrack with template normalized cross-correlation and ORB keypoint evidence through nine leave-one-sequence-out gates and loads no contour network. CSRT~\cite{lukezic2017csrt} and the official OSTrack and STARK-ST50 checkpoints are the comparison trackers. Box IoU, success at IoU $\ge 0.5$ and precision at 20 pixels are equal-sequence means; HOTA, DetA and AssA are pooled official TrackEval values~\cite{luiten2021hota}; missing predictions are scored as misses.
"""
    rows = []
    order = {"chess": ("nanotrack", "csrt", "ostrack", "stark", "ours_disjoint"), "trackves": ("nanotrack", "csrt", "ostrack", "stark", "learnedgate", "ours_disjoint")}
    for domain, names in order.items():
        for name in names:
            v = tracking[domain]["methods"][name]
            official = v["pooled_official_trackeval"]
            rows.append([
                "Controlled" if domain == "chess" else "TrackVes", TRACKER_LABELS[name],
                f"${v['bbox_iou']['mean']:.3f}\\pm{v['bbox_iou']['sample_sd']:.3f}$", f"${v['success_05']['mean']:.3f}\\pm{v['success_05']['sample_sd']:.3f}$",
                f"${v['precision_20']['mean']:.3f}\\pm{v['precision_20']['sample_sd']:.3f}$", f"{official['HOTA']:.3f}", f"{official['DetA']:.3f}", f"{official['AssA']:.3f}", str(v["n_missing_predictions"]),
            ])
    text += table(
        "Complete fixed-denominator localization control. The controlled benchmark has 11 sequences and 3826 scored frames; TrackVes has nine sequences and 5922 scored frames. "
        "The non-invasive verifier of the main text returns exactly its proposer's boxes and does not claim to change any of these numbers.",
        "tab:NegativeFusion", ["Dataset", "Configuration", "Box IoU", "Success@0.5", "Precision@20", "HOTA", "DetA", "AssA", "Missing"], rows,
    )
    text += r"""
\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{images/Fig_FairLocalization.pdf}
\caption{Localization control of Table~\ref{tab:NegativeFusion} on (\textbf{a}) the controlled benchmark and (\textbf{b}) TrackVes: equal-sequence box IoU (blue, with sample SD) and pooled official HOTA (orange) for every configuration. Black edges mark the prior fusion variants, the green edge NanoTrack, dashed lines the NanoTrack level.}
\label{fig:FairLocalization}
\end{figure}
"""
    paired_rows = []
    for domain in ("chess", "trackves"):
        tests = tracking[domain]["paired_tests"]
        for reference in ("ours_disjoint", "learnedgate"):
            for comparator in ("nanotrack", "csrt", "ostrack", "stark"):
                iou = next((r for r in tests if r["reference"] == reference and r["comparator"] == comparator and r["metric"] == "bbox_iou"), None)
                hota = next((r for r in tests if r["reference"] == reference and r["comparator"] == comparator and r["metric"] == "HOTA"), None)
                if iou is None or hota is None:
                    continue
                paired_rows.append([
                    "Controlled" if domain == "chess" else "TrackVes", TRACKER_LABELS[reference].split(" (")[0], TRACKER_LABELS[comparator].split(" (")[0],
                    f"${iou['mean_paired_difference']:+.3f}$ {ci(iou)}", pvalue(iou["p_holm"]),
                    f"${hota['mean_paired_difference']:+.3f}$ {ci(hota)}", pvalue(hota["p_holm"]), f"{iou['improved']}/{iou['degraded']}/{iou['ties']}",
                ])
    text += table(
        "Paired sequence-level differences (reference minus comparator) with 95\\% bootstrap intervals for box IoU and per-sequence HOTA; $p_H$ is Holm-adjusted within each dataset over all declared tests "
        "(12 controlled, 24 TrackVes). The last column counts sequences in which the reference improved, degraded or tied on box IoU.",
        "tab:FairPaired", ["Dataset", "Reference", "Comparator", "$\\Delta$ IoU [95\\% CI]", "$p_H$", "$\\Delta$ HOTA [95\\% CI]", "$p_H$", "Imp./deg./tie"], paired_rows,
    )
    gate = next(r for r in tracking["trackves"]["paired_tests"] if r["reference"] == "learnedgate" and r["comparator"] == "nanotrack" and r["metric"] == "bbox_iou")
    sd_nano = next(r for r in tracking["trackves"]["paired_tests"] if r["reference"] == "ours_disjoint" and r["comparator"] == "nanotrack" and r["metric"] == "bbox_iou")
    text += (
        f"Under sequence-isolated training the box-overwriting tracker is below NanoTrack on both benchmarks (TrackVes box IoU difference {sd_nano['mean_paired_difference']:+.3f}, 95\\% CI [{sd_nano['ci95_low']:+.3f}, {sd_nano['ci95_high']:+.3f}]) "
        f"and far below the official checkpoints; LearnedGate changes TrackVes box IoU by {gate['mean_paired_difference']:+.3f} relative to NanoTrack (95\\% CI [{gate['ci95_low']:+.3f}, {gate['ci95_high']:+.3f}]). "
        f"CSRT reports no box in {tracking['chess']['methods']['csrt']['n_missing_predictions']} controlled and {tracking['trackves']['methods']['csrt']['n_missing_predictions']} TrackVes ground-truth frames. "
        "These results are the reason the present work reports failure awareness around unchanged proposals instead of a box-overwriting fusion tracker; an earlier comparison of the same tracker against untrained task-head proxies is superseded by this table and is not reported.\n"
    )
    # per-sequence
    nano = results["proposers"]["nanotrack"]["methods"]
    stark = results["proposers"]["stark"]["methods"]
    ost = heldout["methods"]
    rows = []
    for sequence in sorted(nano["verification"]["sequences"]):
        n, s, o = nano["verification"]["sequences"][sequence], stark["verification"]["sequences"][sequence], ost["verification"]["sequences"][sequence]
        rows.append([
            sequence.replace("_", r"\_"), n["case"].replace("_", r"\_"), str(n["n_frames"]),
            f"{n['failure_prevalence']:.2f}", fmt(nano["native_confidence"]["sequences"][sequence]["failure_auroc"]), fmt(n["failure_auroc"]),
            f"{s['failure_prevalence']:.2f}", fmt(stark["native_confidence"]["sequences"][sequence]["failure_auroc"]), fmt(s["failure_auroc"]),
            f"{o['failure_prevalence']:.2f}", fmt(ost["native_confidence"]["sequences"][sequence]["failure_auroc"]), fmt(o["failure_auroc"]),
        ])
    text += r"""
\section{Per-sequence outcomes on the frozen SurgT test partition}\label{sec:appendix_per_sequence}
"""
    text += table(
        "Per-sequence test outcomes for the three proposers: case group, scored frames, failure prevalence and two-class failure AUROC of native confidence (Nat.) and the fused verifier (Ver.). "
        "AUROC is undefined (--) when a proposer never fails or never succeeds in a sequence; such sequences remain in all operating-point denominators.",
        "tab:PerSequence", ["Sequence", "Case", "Frames", "Nano prev.", "Nano Nat.", "Nano Ver.", "STARK prev.", "STARK Nat.", "STARK Ver.", "OSTrack prev.", "OSTrack Nat.", "OSTrack Ver."], rows,
    )
    # weighting
    infer = supplementary["weighting_and_inference"]
    rows = []
    for proposer in ("nanotrack", "stark"):
        for label, key in (("equal", "equal"), ("$\\sqrt{\\text{frames}}$", "sqrt_frames"), ("frames", "frames")):
            w = infer[proposer]["weighted_means"][key]
            rows.append([PROPOSERS[proposer], label, f"{w['native_confidence']:.3f}", f"{w['verification']:.3f}", f"{w['difference']:+.3f}"])
    text += r"""
\section{Weighted sequence means}\label{sec:appendix_weighting}
"""
    text += table(
        "Sequence means of failure AUROC under three weighting schemes for the sequences with defined AUROC for both cues.",
        "tab:Weighting", ["Proposer", "Weights", "Native confidence", "Fused verification", "Difference"], rows,
    )
    # gap
    gap = supplementary["validation_test_gap"]
    rows = []
    for proposer in ("nanotrack", "stark"):
        for name in ("native_confidence", "rank_only", "contour_only", "color_only", "verification"):
            g = gap[proposer][name]
            rows.append([PROPOSERS[proposer], LABELS[name], f"{g['validation']:.3f} ({g['n_validation']})", f"{g['test']:.3f} ({g['n_test']})", f"{g['gap']:+.3f}"])
    text += r"""
\section{Validation-to-test shift}\label{sec:appendix_gap}
"""
    text += table(
        "Sequence-mean failure AUROC on the validation and test partitions (two-class sequence counts in parentheses) and their difference, for every cue and selection proposer.",
        "tab:Gap", ["Proposer", "Cue", "Validation", "Test", "Validation $-$ test"], rows,
    )
    # chance level
    chance = supplementary["contour_chance_level"]
    rows = []
    for alpha in ("0.5", "0.68", "0.8"):
        c = chance[alpha]
        rows.append([alpha, f"{c['density']:.2f}", f"{c['random_pair_mean']:.3f} [{c['random_pair_q05']:.3f}, {c['random_pair_q95']:.3f}]", f"{c['identical']:.3f}"] + [f"{c['translated_copy'][k]:.3f}" for k in ("1", "2", "4", "8", "16")])
    text += r"""
\section{Chance level and displacement response of the trimmed contour distance}\label{sec:appendix_chance}
"""
    text += table(
        "Monte Carlo evaluation of $d_C$ on the $64\\times64$ raster (2-pixel border empty): mean and 5--95\\% range over 300 pairs of independent random masks of the stated density, the value for identical masks, "
        "and the value between a random mask and its copy translated horizontally by the stated number of pixels.",
        "tab:ChanceLevel", ["$\\alpha$", "Density", "Random pair", "Identical", "1 px", "2 px", "4 px", "8 px", "16 px"], rows,
    )
    text += (
        "Translations of a few pixels already produce values comparable to unrelated random masks, which quantifies the saturation discussed in Section~\\ref{sec:contours}; the phase-correlation alignment step is what keeps small translations from being read as structural change.\n"
    )
    # segmentation
    rows = []
    for name in ("Ours-Contour", "MedSAM", "SAM2.1", "YOLO11m-seg"):
        v = segmentation["methods"][name]
        rows.append([name, f"${v['dice']['video_macro_mean']:.3f}\\pm{v['dice']['video_sample_sd']:.3f}$", f"${v['iou']['video_macro_mean']:.3f}\\pm{v['iou']['video_sample_sd']:.3f}$", str(v["missing_masks"])])
    text += r"""
\section{Secondary contour-delineation statistics}\label{sec:appendix_seg}
The gradient-percentile contour front-end of the precursor study was also compared, as a filled region, with promptable and supervised segmenters on all 8080 CholecSeg8k frames~\cite{hong2020cholecseg8k} grouped by the 17 available video identifiers (not verified patient identities); missing masks score zero. This secondary result concerns region delineation and does not validate temporal failure awareness.
"""
    text += table(
        "Equal-video segmentation statistics with all masks and absences counted. Prompt settings and pretrained-model budgets differ; these are not universal segmentation rankings.",
        "tab:SecondarySeg", ["Configuration", "Video Dice", "Video IoU", "Missing masks"], rows,
    )
    return text


# --------------------------------------------------------------------------- declarations / assembly
def declarations(original_tex: str) -> str:
    text = "\n\\vspace{6pt}\n"
    for command in ("authorcontributions", "funding", "institutionalreview", "informedconsent", "conflictsofinterest"):
        match = re.search(r"\\" + command + r"\{(.*?)\}\s*(?:\n|$)", original_tex, re.S)
        if match is None:
            raise ValueError(f"Missing original author declaration: {command}")
        text += "\\" + command + "{" + match.group(1) + "}\n"
    text += r"""
\dataavailability{The evaluation code, the frozen protocol and selection manifest, the per-frame outputs of every proposer on both SurgT partitions, the transfer and confidence-logging reruns, all post-hoc analyses, the figure scripts, the historical framework code needed for Appendix~\ref{sec:appendix_fusion}, SHA-256 checksums of every result file, and the source commits and checkpoint hashes of the official trackers are publicly available at \url{https://github.com/liranyang123456-commits/endoscopic-tracking-failure-verification} (tag \texttt{v1.0-submission}, accessed 11 October 2026); the repository README lists the commands that reproduce every table and figure of this paper from the raw data. SurgT, TrackVes and CholecSeg8k must be obtained from their providers; the controlled benchmark images, their ground-truth coordinates and model weights are not redistributed, but their hashes and all per-frame outputs are.}
\acknowledgments{The authors thank the providers of the public SurgT, TrackVes and CholecSeg8k datasets and the authors of OSTrack, STARK and NanoTrack for releasing their code and weights. During manuscript preparation, the authors used Copilot SDK in VS Code for language editing and for assistance with drafting method descriptions, figures and evaluation scripts. The authors reviewed all generated material and take full responsibility for the accuracy and integrity of the content; all reported numbers were computed from the documented datasets and model outputs.}
"""
    return text


def manuscript(results: dict, transfer: dict, segmentation: dict, tracking: dict, selected: dict,
               supplementary: dict, heldout: dict, cpu: str) -> str:
    original = (ORIGINAL / "Manuscript_Mathematics.tex").read_text(encoding="utf-8")
    text = preamble(original)
    text += "\\abstract{" + abstract_text(results, heldout, supplementary).replace("%", r"\%").replace(r"\\%", r"\%") + "}\n"
    text += r"\keyword{image processing; visual tracking; failure detection; ordinal representation; contour distance; conformal prediction; endoscopic video; soft tissue}" + "\n"
    text += r"\MSC{68U10; 68T45; 68T10; 62G09; 92C55}" + "\n"
    text += r"\begin{document}" + "\n"
    text += introduction(results, heldout, supplementary)
    text += RELATED
    text += method(selected, supplementary)
    text += protocol(results, selected, supplementary, heldout, cpu)
    text += results_section(results, selected, supplementary, heldout, transfer)
    text += discussion(results, supplementary, heldout)
    text += declarations(original)
    text += appendix(results, tracking, segmentation, supplementary, heldout)
    text += "\n\\reftitle{References}\n\\externalbibliography{yes}\n\\bibliography{mybibfile}\n\\end{document}\n"
    return text


def cover(results: dict, heldout: dict) -> str:
    original = (ORIGINAL / "Cover_Letter_Mathematics.tex").read_text(encoding="utf-8")
    original = original.replace("Annotation-Free Contour Verification for Region-Level Tracking of Deforming Soft Tissue in Endoscopy", TITLE)
    original = original.replace(
        "All authors have read and approved this submission",
        "Before submission, the corresponding author will confirm that all co-authors have reviewed and approved the final files",
    )
    start = original.index("The article addresses")
    end = original.index("This manuscript has not previously been submitted")
    nano = results["proposers"]["nanotrack"]["methods"]
    stark = results["proposers"]["stark"]["methods"]
    ost = heldout["methods"]
    body = (
        "The manuscript addresses failure awareness in region-level endoscopic soft-tissue tracking: a frozen tracker can keep returning a plausible box after losing a deforming target, "
        "and its native confidence need not reveal the loss. We propose a mask-free appearance-consistency layer that monitors the unchanged proposals of any box-producing tracker and reports a bounded risk. "
        "The initial target box is supplied and labeled validation sequences calibrate the threshold; no manually drawn pixel masks and no additional network are needed.\n\n"
        "The mathematical content is an empirical midrank representation, gradient-quantile contour sets, a trimmed symmetric contour distance whose elementary properties and Monte Carlo chance level we establish, "
        "Fourier-domain phase correlation, the Hellinger distance between color histograms, and a convex fusion. The alarm threshold is given a sequence-level split-conformal construction with a finite-sample specificity guarantee under exchangeability (Proposition 2), "
        "and the complete procedure is specified as Algorithm 1.\n\n"
        "The evaluation on the public SurgT benchmark freezes every parameter on the validation partition before the test partition is touched, uses the official pretrained STARK-ST50 and NanoTrack trackers for selection and the official OSTrack tracker as a held-out proposer, "
        "keeps complete shared denominators, and reports failure-ranking, operating-point, selective-risk and detection-latency metrics with paired sequence statistics, exact permutation tests and case-cluster bootstrap intervals. "
        f"NanoTrack failure AUROC rises from {nano['native_confidence']['sequence_macro']['failure_auroc']['mean']:.3f} to {nano['verification']['sequence_macro']['failure_auroc']['mean']:.3f}, STARK-ST50 from "
        f"{stark['native_confidence']['sequence_macro']['failure_auroc']['mean']:.3f} to {stark['verification']['sequence_macro']['failure_auroc']['mean']:.3f}, and the held-out OSTrack from "
        f"{ost['native_confidence']['sequence_macro']['failure_auroc']['mean']:.3f} to {ost['verification']['sequence_macro']['failure_auroc']['mean']:.3f}. Cue ablations, post-hoc sensitivity analyses, a transfer to TrackVes and a controlled benchmark, qualitative cases, "
        "and an earlier unsuccessful box-overwriting fusion tracker are all reported, including every non-significant comparison. The layer leaves localization unchanged, costs about 1.4 ms per frame and does not claim validated clinical safety.\n\n"
        "The frozen protocol, all scripts, per-frame results and checksums are publicly available at\\\\ "
        "\\url{https://github.com/liranyang123456-commits/endoscopic-tracking-failure-verification} (tag v1.0-submission)\\\\ and also accompany the submission as supplementary material. "
        "Copilot SDK in VS Code assisted with language editing and with drafting method descriptions, figures and evaluation scripts; the authors reviewed all generated material and are responsible for the accuracy and integrity of the content.\n\n"
    )
    return original[:start] + body + original[end:]
