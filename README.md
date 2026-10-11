# Endoscopic tracking failure verification

Code, frozen protocol, per-frame outputs and figures for

> **Ordinal-Gradient Appearance Consistency and Conformal Alarms for Verifying Region-Level Soft-Tissue Tracking in Endoscopy**
> Ranyang Li, Nan Wei, Wufeng Liu, Chao Fan, Zhipeng Lin. Submitted to *Mathematics* (MDPI), Special Issue
> "Mathematical Methods for Image Processing and Understanding, 2nd Edition", October 2026.

The verifier monitors the unchanged box proposals of a frozen region-level tracker and reports a bounded
failure risk built from midrank intensities, a fixed-density gradient-quantile contour set, a hue–saturation
histogram and the tracker's own confidence. Fusion weights and alarm thresholds were frozen on the SurgT
validation partition before the test partition was touched; a sequence-level split-conformal threshold carries
a finite-sample specificity guarantee. The tracker's box is never modified.

| Proposer (SurgT test, 20 sequences) | Native AUROC | Fused AUROC | Δ [95% CI] |
|---|---|---|---|
| NanoTrack (selection) | 0.501 | 0.882 | +0.381 [+0.257, +0.503] |
| STARK-ST50 (selection) | 0.710 | 0.763 | +0.054 [-0.005, +0.119] |
| OSTrack (held-out) | 0.743 | 0.921 | +0.178 [+0.087, +0.300] |

## What is in this repository

| Path | Content |
|---|---|
| `*.py` | Evaluation, analysis, figure and manuscript-assembly scripts (entry points listed below) |
| `legacy/` | Minimal TrackVes I/O helpers extracted from the historical project |
| `original_framework/` | Historical tracker code needed only for the Appendix A experiments (`run_disjoint.py`, `run_gate.py`) |
| `verification_protocol.json` | Frozen protocol: candidate grid, selection objective, threshold rule |
| `results/selected_verification.json` | Validation-only selection manifest (frozen before any test run) |
| `results/external_verification/{validation,test}/` | Per-frame cues, IoU and failure labels for NanoTrack, STARK-ST50 and OSTrack, with input-frame hashes |
| `results/external_verification_summary.json`, `results/heldout_ostrack_summary.json` | Test-partition statistics for the selection and held-out proposers |
| `results/supplementary_analyses.json` | Post-hoc analyses: extra comparators, detection latency, sensitivity, hierarchical bootstrap, conformal thresholds, chance level |
| `results/transfer_verification*`, `results/with_confidence/` | Transfer to TrackVes and the controlled benchmark; confidence-logging reruns of the official trackers |
| `results/official/`, `results/fair_tracking_summary.json` | Complete-denominator localization control (Appendix A) |
| `results/figures/` | All manuscript figures (PDF + PNG) and the qualitative-case manifest |
| `results/CHECKSUMS.sha256` | SHA-256 of every result file in this repository |
| `docs/REVISION_NOTES.md` | Internal revision log of the study |

Not included (size or licence): dataset frames, decoded TrackVes PNGs (hashes in
`results/frozen_trackves_frames.manifest.json`), model weights (see `weights/README.md`), and the
OSTrack / STARK / TrackEval source trees (see `external/README.md`; exact commits are pinned there and in every
run manifest).

## Setup

```bash
git clone https://github.com/liranyang123456-commits/endoscopic-tracking-failure-verification.git
cd endoscopic-tracking-failure-verification
python -m venv .venv && .venv/Scripts/activate      # Python 3.13 was used
pip install -r requirements.txt                       # opencv-contrib-python provides TrackerNano and CSRT
# third-party trackers (pinned commits, see external/README.md)
git clone https://github.com/botaoye/OSTrack external/OSTrack && git -C external/OSTrack checkout 33b5e12586216b7fd0e95d255bd01ba44cbec759
git clone https://github.com/researchmm/Stark external/STARK && git -C external/STARK checkout 162e1d37a38f92251cb380e90ccdb59608f2df3c
git clone https://github.com/JonathonLuiten/TrackEval external/TrackEval && git -C external/TrackEval checkout 12c8791b303e0a0b50f753af204249e622d0281a
```

Download the author-released checkpoints and the NanoTrack ONNX models into `weights/` (`weights/README.md`
lists the sources and SHA-256 values recorded in the run manifests). Point the scripts at the data with
environment variables:

| Variable | Meaning |
|---|---|
| `MFE_SURGT_ROOT` | SurgT benchmarking `data/` folder (contains `validation/` and `test/`) |
| `MFE_TRACKVES_ROOT` | TrackVes root with `EV1 … IV6` sequence folders |
| `MFE_CHESS_ROOT` | Controlled benchmark root (restricted; GT and image hashes are in `results/chess_gt_manifest.csv`) |
| `MFE_NANOTRACK_DIR` | Folder holding `nanotrack_backbone_sim.onnx` and `nanotrack_head_sim.onnx` (default `weights/`) |
| `MFE_SEGMENTATION_ROOT`, `MFE_LEGACY_BASELINE_ROOT`, `MFE_TRACKVES_LEGACY_RESULTS`, `MFE_LEARNED_FUSION_ROOT` | Only for the appendix (CholecSeg8k statistics and historical baselines) |

## Reproducing the paper

Run from the repository root; each step writes into `results/` and refuses to overwrite a frozen selection.

```bash
python -m pytest test_protocol.py test_contour_verifier.py test_verification_statistics.py -q

# 1. SurgT runs with the frozen proposers (GPU for STARK/OSTrack; ~15–25 min per proposer and split)
python run_external_verification.py --split validation --proposer stark
python run_external_verification.py --split validation --proposer nanotrack
python evaluate_verification.py select                      # writes results/selected_verification.json (already frozen here)
python run_external_verification.py --split test --proposer stark
python run_external_verification.py --split test --proposer nanotrack
python evaluate_verification.py evaluate                    # external_verification_summary.json
python run_external_verification.py --split validation --proposer ostrack
python run_external_verification.py --split test --proposer ostrack
python evaluate_heldout_proposer.py --proposer ostrack      # heldout_ostrack_summary.json

# 2. Post-hoc analyses, transfer and figures
python supplementary_analyses.py                            # supplementary_analyses.json + curves
python run_official_trackers.py --tracker nanotrack --dataset both --output results/with_confidence
python run_official_trackers.py --tracker stark --dataset both --output results/with_confidence
python run_transfer_verification.py                         # transfer_verification_summary.json
python make_qualitative_figure.py                           # re-tracks four sequences and checks them against the frozen CSVs
python make_revised_figures.py

# 3. Appendix A (historical fusion tracker, sequence-disjoint folds; needs original_framework/ and the controlled benchmark)
python run_official_trackers.py --tracker ostrack --dataset both
python run_official_trackers.py --tracker stark --dataset both
python run_official_trackers.py --tracker nanotrack --dataset both
python run_official_trackers.py --tracker csrt --dataset both
python run_disjoint.py all --epochs 60 --samples-per-epoch 1024
python run_gate.py all
python summarize_fair.py
```

`verify_verification_submission.py` recomputes every pooled statistic from the per-frame CSVs and checks the
compiled manuscript against them; `build_verification_submission.py` regenerates the LaTeX source from the JSON
files so that no number is typed by hand.

## Reproducibility evidence and caveats

* **Determinism of the verifier.** The descriptor is deterministic; repeated evaluation on identical inputs
  gives a maximum difference of 0.0 (`transfer_verification_summary.json`). Re-tracking four test sequences with
  the identical frozen proposers reproduced every stored IoU and cue value within 1e-4
  (`results/figures/Verification_Qualitative.manifest.json`), and the confidence-logging reruns of NanoTrack and
  STARK-ST50 reproduced all previously published boxes on TrackVes and the controlled benchmark exactly
  (`rerun_boxes_identical_to_frozen_official_csv` in the transfer summary).
* **Thread settings matter.** All runs use `cv2.setNumThreads(1)` and `torch.set_num_threads(4)`; with
  multi-threaded OpenCV filtering the contour values can differ by ~0.01 because a few gradient magnitudes move
  across the quantile threshold. The scripts set these values themselves.
* **GPU trackers.** STARK-ST50 was reproduced bit-for-bit on an RTX 5090 Laptop GPU; OSTrack was run once. Other
  GPUs/cuDNN versions may change a small number of boxes; the stored per-frame CSVs are the reference.
* **Frozen selection.** `results/selected_verification.json` carries the SHA-256 of the protocol file; the test
  runs refuse to start if it is missing or changed, and `evaluate_verification.py select` refuses to overwrite it.
* **Statistics.** Bootstrap and permutation procedures use fixed seeds (20261011/20261012) and are listed in
  the JSON outputs together with their denominators.
* **Data.** SurgT, TrackVes and CholecSeg8k are public; the controlled rigid benchmark is not redistributed, so
  only its hashes, GT layout and all per-frame outputs are provided.

## Licence and citation

Our code is released under the MIT licence (`LICENSE`). OSTrack, STARK, TrackEval, NanoTrack and OpenCV remain
under their own licences and are not redistributed here. If you use this code or the frozen protocol, please
cite the paper above and the SurgT benchmark (Cartucho et al., *Medical Image Analysis* 2024).
