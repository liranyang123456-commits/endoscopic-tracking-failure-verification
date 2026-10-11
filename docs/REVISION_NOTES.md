# Mathematics fairness revision

This directory implements and executes a retrospective fairness audit, not a
promise of improved tracker accuracy. Old manuscript packages are not inputs to
the new result tables. New numbers are assembled only after the full expected
evaluation inventory has passed validation.

## Current verification-focused submission

The later optimization changes the endpoint from overriding tracker boxes to
monitoring unreliable localization. `contour_verifier.py` uses ordinal grayscale
patches, empirical gradient quantiles, trimmed contour distance, rank/color
discrepancy and native tracker confidence. It returns a risk score without
altering the tracker box or internal state. No additional neural model is fitted,
and no training-repeat claims are needed for this deterministic descriptor.

The frozen candidate grid is in `verification_protocol.json`. Parameter choice
and warning thresholds use only the supplied SurgT validation partition. The
test partition runs after `selected_verification.json` is persisted; no later
test-driven threshold adjustment is performed. Rectified input pixel hashes and
scored frame identities match across official STARK-ST50 and NanoTrack.

```powershell
& $python run_external_verification.py --split validation --proposer stark
& $python run_external_verification.py --split validation --proposer nanotrack
& $python evaluate_verification.py select
& $python run_external_verification.py --split test --proposer stark
& $python run_external_verification.py --split test --proposer nanotrack
& $python evaluate_verification.py evaluate
& $python run_transfer_verification.py
& $python -m pytest test_protocol.py test_contour_verifier.py test_verification_statistics.py -q
& $python build_verification_submission.py
```

After compiling the manuscript and cover letter in
`..\Mathematics_Verification_Submission_20261011`:

```powershell
& $python verify_verification_submission.py
& $python publish_submission.py --verification --stage-only
```

`--stage-only` rebuilds the LaTeX-source and supplementary archives beside the
verified PDFs and does not modify `Mathematics_Submission_Final_20261011`.
Review the generated archives and obtain author confirmation of submission
history and approval before using the separate, non-staging publish command.

The exact SurgT scoring inventory contains 7666 validation and 12056 test
post-initialization observations per proposer. All later video frames update the
tracker; only visible, non-difficult, valid left-view GT boxes are scored under
the stated no-GT-reset protocol. This is not the challenge's stereo/reset EAO
leaderboard protocol.

### Verified results and limits

- On the fixed test split, NanoTrack failure AUROC is 0.8818 for fused
  verification versus 0.5011 for native confidence. The paired sequence
  difference is 0.3806, 95% interval [0.2572, 0.5028]; Holm-adjusted
  sequence p = 0.000366.
- Five-case two-sided Wilcoxon p = 0.0625 is not significant at 0.05. The case
  identities are not established independent patient identifiers.
- Sequence-level AUROC is defined for 17/20 NanoTrack test sequences and 16/20
  STARK-ST50 test sequences. Paired AUROC tests use those eligible sequences;
  the separate five-case-group analysis is a sensitivity analysis, not the
  sequence-level test or verified patient-level inference. Single-class
  sequences remain in the full test/frame inventory and in metrics defined for
  their outcomes.
- STARK-ST50 fused AUROC is 0.7635 versus native 0.7097, but its corrected
  difference is not significant and rank-only has a larger mean AUROC.
- Fused verification does not significantly outperform every single appearance
  cue. All ablations, thresholds, false-alarm operating points and denominators
  are retained.
- The monitor adds approximately 1.4 ms per scored frame, excluding the frozen
  proposer and including all three diagnostic contour percentiles.
- The appearance-only transfer branch obtains TrackVes NanoTrack AUROC 0.9306.
  The native confidence was not saved in retained proposal CSVs, so that branch
  explicitly removes and renormalizes the confidence weight rather than
  inventing confidence or comparing it with the full monitor.
- Historical work in this workspace involved SurgT. The newly frozen protocol
  is not claimed to use a never-seen independent patient cohort. Video-file
  hashes differ across the provided splits, which does not prove patient
  independence.

The earlier unfavorable sequence-isolated contour-fusion results remain in an
appendix. New findings support an empirical failure-verification contribution,
not a uniformly more accurate tracker or validated clinical safety.

### Narrative and figure revision (11 Oct 2026, 23:00–23:40)

- Title: *Ordinal-Gradient Appearance Consistency and Conformal Alarms for
  Verifying Region-Level Soft-Tissue Tracking in Endoscopy*.
- `make_revised_figures.py` regenerates every figure in the house style of the
  earlier manuscripts (blue/orange series, black edge = ours, green edge =
  NanoTrack, value labels, lettered panels): `Fig_Framework` (portrait banner
  with panels A–D), `Fig_Headline`, `Fig_CueAblation`, `Fig_Curves`,
  `Fig_PairedSequences`, `Fig_Sensitivity` (incl. conformal coverage) and
  `Fig_FairLocalization`; `make_qualitative_figure.py` supplies the case figure.
- Results open with a consolidated performance table (`tab:Summary`) and a
  transfer summary (`tab:SummaryTransfer`). The Introduction follows
  problem → why native confidence fails → idea with per-failure-mode rationale →
  evidence → contributions → scope; Methods gained §3.2 *Design rationale*;
  Discussion gained *Design implications*.

### Peer-review-driven revision (11 Oct 2026, 22:00–23:00)

Four simulated reviews (`..\SIMULATED_PEER_REVIEW_20261011.md`) were addressed:

- Title changed to *Ordinal-Gradient Appearance Consistency for Failure
  Verification in Region-Level Endoscopic Soft-Tissue Tracking*; "annotation-free"
  replaced by "mask-free" throughout.
- `run_external_verification.py --proposer ostrack` adds the official OSTrack
  checkpoint as a **held-out proposer** (native confidence = maximum of the
  Hann-windowed classification response, recorded without touching the box
  head). `evaluate_heldout_proposer.py` applies the frozen percentile/weights
  and calibrates only its threshold on its own validation reliable frames.
- `supplementary_analyses.py` recomputes, from the retained per-frame CSVs:
  extra comparators (color-only, appearance-only, rank+contour, |Δconfidence|),
  detection latency of the first failure episode, pooled ROC / risk–coverage
  curves and per-sequence paired plots, temporal-third specificity, sensitivity
  to the IoU failure threshold / contour percentile / weight simplex / alarm
  quantile, weighted sequence means, exact sign-flip permutation tests,
  hierarchical case→sequence bootstrap, primary vs exploratory Holm families,
  the sequence-level split-conformal threshold, the Monte Carlo chance level of
  the trimmed contour distance, the validation-to-test gap and the failure-set
  overlap between proposers.
- `run_official_trackers.py` now records native confidence; reruns in
  `results/with_confidence` reproduced the frozen boxes exactly and feed the
  four-cue transfer in `run_transfer_verification.py`.
- `make_qualitative_figure.py` re-tracks four test sequences (frozen CSV values
  reproduced within 1e-4) and draws thumbnails with boxes plus risk time lines.
- `manuscript_text.py` holds the rewritten prose: numbered contributions, a
  consolidated scope-and-claims paragraph, a five-paragraph related-work section
  (61 references), Remark 1 (monotone invariance), Proposition 1 (properties of
  d_C), Proposition 2 (split-conformal sequence-level specificity), and the
  removal of the Highlights block and of the archived precursor figures
  (Appendix A keeps the complete fair control with a regenerated figure).

### Manuscript assembly notes (11 Oct 2026, evening)

- The verifier is typeset as a single two-phase `algorithm2e` block
  (Algorithm 1: Phase A reference construction, Phase B per-frame monitoring).
  The former `Verification_Algorithm.pdf` overview image is no longer generated
  or referenced; `verify_verification_submission.py` fails if it reappears.
- `build_verification_submission.py` now also emits, from the result JSONs:
  the full validation selection grid, proposer context on the frozen test
  split, descriptive pooled frame-level metrics, the complete fair tracking
  control (CSRT, LearnedGate, Success@0.5, Precision@20, DetA, AssA, missing
  predictions) with paired Holm tests, and per-sequence SurgT outcomes.
- Legacy SurgT challenge-protocol runs (`F:\...\BenchmarkSuite\Results\SurgT_*`,
  reset-based EAO with the old scheme-C tracker), the incomplete EndoVis2017
  run and the Cholec80 pseudo-GT runs are intentionally **not** reported: they
  use a different protocol, several trackers in that sweep share identical
  numbers (fallback suspected), and they are not part of the frozen inventory.

## Protocol

- Official author-released OSTrack ViT-B-256 CE and STARK-ST50 task checkpoints
  replace untrained task-head proxies in the primary comparison.
- The fixed controlled inventory has 11 sequences, 22 targets and 3848 GT records.
  The 22 supplied initialization observations are not scored. Three
  initialization-only targets have no post-initialization observations; they
  remain in the inventory rather than being counted as successful tracking.
- TrackVes has nine sequences and 5931 GT records, with 5922 scored observations
  after initialization. Every evaluated method uses the same two-pixel GT-box
  padding and GT-frame update protocol.
- H.264 source-video errors are not concealed. A sequential single-thread decode
  pass freezes every annotated frame to lossless PNG. Every round trip and
  per-method decoded-pixel hash is checked. Freezing equalizes inputs; it cannot
  restore information damaged in the original videos.
- Missing predictions score zero overlap, success and precision, and become
  explicit TrackEval false negatives. No method-specific failed target or
  sequence is removed.
- All 20 contour folds start from random weights. A fold has 17 training
  sequence groups, two validation groups and one test group. Contour patches are
  computed from image intensities inside annotated ROI boxes, not from human
  polygon-mask targets. ROI-box supervision is therefore still present.
- The separate TrackVes LearnedGate uses NanoTrack, template NCC and ORB. It does
  **not** load ContourSimilarityNet or the auxiliary autoencoder. All nine gates,
  scalar normalization, epoch selection and intervention margins exclude their
  held-out test sequence.
- CholecSeg8k comparisons keep all 8080 frames, including absent masks, and use
  the 17 available video IDs as clusters. Patient identities are not inferred.
  Paired effects, video/sequence bootstrap intervals and two-sided Wilcoxon tests
  are reported; Holm adjustment uses a fixed comparator-by-metric family.
- Confidence intervals condition on the available clusters and one fitting seed
  per fold. Retrospective architecture/profile selection and overlapping
  cross-fit training folds are disclosed; these are not preregistered
  confirmatory clinical tests.

## Local execution

The available runtime used for this revision is:

```powershell
$python = 'D:\MIS_Pose_Track_Re3D\.venv_surgt_llm\Scripts\python.exe'
Set-Location 'D:\MIS_Pose_Track_Re3D\BSPC_Submission_v2\08_Resubmission_Revision_20260917\Mathematics_Fair_Evaluation'
& $python -m pytest test_protocol.py -q
& $python audit_results.py
& $python freeze_trackves_frames.py
& $python run_official_trackers.py --tracker ostrack --dataset both
& $python run_official_trackers.py --tracker stark --dataset both
& $python run_official_trackers.py --tracker nanotrack --dataset both
& $python run_disjoint.py all --epochs 60 --samples-per-epoch 1024
& $python run_gate.py all
& $python summarize_fair.py
& $python build_fair_submission.py
```

If the shared virtual-environment launcher becomes unavailable, the original
base interpreter is `D:\anaconda\python.exe` (Python 3.13.9 with the same
system-site packages). It was used successfully for the final 20-test
regression run, numeric/PDF verification and packaging. No experiment outcome,
model choice or warning threshold was changed when switching the validation
interpreter; no shared runtime was repaired or replaced.

`protocol.py` declares the actual data and legacy-code locations. Reproduction
requires obtaining the public datasets, the controlled benchmark with
permission, the original framework code and the author-released weights.
Do not replace a missing asset with invented outputs or untrained task heads.
The new supplementary package contains audit scripts and aggregate results;
restricted source images, controlled GT coordinates and model weights are not
redistributed.

CSRT was unavailable in the shared OpenCV build. A matching
`opencv-contrib-python==4.13.0.92` was installed under
`opencv_contrib_runtime`, without replacing the shared installation:

```powershell
& $python -c "import sys,runpy; sys.path.insert(0,r'D:\MIS_Pose_Track_Re3D\BSPC_Submission_v2\08_Resubmission_Revision_20260917\Mathematics_Fair_Evaluation\opencv_contrib_runtime'); sys.argv=['run_official_trackers.py','--tracker','csrt','--dataset','both']; runpy.run_path('run_official_trackers.py',run_name='__main__')"
```

## Source and checkpoint provenance

- OSTrack: <https://github.com/botaoye/OSTrack>, source commit
  `33b5e12586216b7fd0e95d255bd01ba44cbec759`; author model folder
  <https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy>;
  selected file ID `1dySfPrSk-knAEo0lTXqZ1kMk3BCvVofV`.
- STARK: <https://github.com/researchmm/Stark>, source commit
  `162e1d37a38f92251cb380e90ccdb59608f2df3c`; author model folder
  <https://drive.google.com/drive/folders/1fSgll53ZnVKeUn22W37Nijk-b9LGhMdN>;
  selected file ID `1sV_idlYLyxeCIO2o4AQDvUFO5b-X-E4w`.
- TrackEval: <https://github.com/JonathonLuiten/TrackEval>, source commit
  `12c8791b303e0a0b50f753af204249e622d0281a`.
- OpenCV NanoTrack: local author-distributed ONNX backbone/head; their exact
  file hashes appear in each run manifest.

Only obsolete dependency imports/type aliases were patched in the official
sources: `torch._six`, `torchvision.models.utils`, and `np.float` in HOTA.
No tracker architecture, trained head, HOTA equation, or model score is modified
to obtain favorable results. Run manifests contain checkpoint and configuration
hashes. Runtime package versions and input hashes are part of the audit outputs.

## Submission assembly and AI assistance

`build_fair_submission.py` creates a separate, updated manuscript and cover
letter from the historical source and verified new statistics. It removes old
proxy rankings, frame-level significance claims, incompatible legacy comparisons
and claims that all variants share an adaptive-gradient front-end. Old results
remain explicitly labeled archival diagnostics.

The manuscript acknowledges that Copilot SDK in VS Code was used solely for
language editing and style polishing of the manuscript text. Experimental
results are computed outputs of the released code, not AI-generated
observations. Authors must review scientific claims, code, citations,
permissions, submission history and author approvals before submitting.
