"""Assemble the public reproducibility repository for the verification study.

Copies the evaluation code, frozen protocol, aggregate results, per-frame
outputs, figures and the historical framework files needed by the appendix
experiments into a clean staging directory. Restricted assets (dataset frames,
model weights, third-party source trees) are not copied; their locations,
commits and hashes are documented instead.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
from pathlib import Path

from protocol import HERE, LEARNED, PROJECT

STAGE = Path(r"D:\MIS_Pose_Track_Re3D\github_verification_release")
REPO_NAME = "endoscopic-tracking-failure-verification"
TITLE = "Ordinal-Gradient Appearance Consistency and Conformal Alarms for Verifying Region-Level Soft-Tissue Tracking in Endoscopy"

CODE = [
    "protocol.py", "contour_verifier.py", "run_official_trackers.py", "run_external_verification.py",
    "evaluate_verification.py", "evaluate_heldout_proposer.py", "supplementary_analyses.py",
    "run_transfer_verification.py", "make_qualitative_figure.py", "make_revised_figures.py",
    "freeze_trackves_frames.py", "audit_results.py", "run_disjoint.py", "run_gate.py", "summarize_fair.py",
    "build_fair_submission.py", "build_verification_submission.py", "manuscript_text.py",
    "verify_verification_submission.py", "publish_submission.py", "release_to_github.py",
    "test_protocol.py", "test_contour_verifier.py", "test_verification_statistics.py",
    "verification_protocol.json", "verification_references.bib", "requirements.txt",
]
RESULT_FILES = [
    "selected_verification.json", "external_verification_summary.json", "external_verification_per_sequence.csv",
    "heldout_ostrack_summary.json", "heldout_ostrack_per_sequence.csv", "supplementary_analyses.json",
    "transfer_verification_summary.json", "fair_tracking_summary.json", "cholecseg_grouped.json",
    "cholecseg_per_video.csv", "cholecseg_aligned_frames.csv", "chess_fair_legacy.json",
    "chess_fair_per_sequence.csv", "trackves_fair_per_sequence.csv", "chess_gt_manifest.csv",
    "gate_split_manifest.json", "gate_route_audit.csv", "gate_training_inputs.manifest.json",
    "image_derived_contour_patches.manifest.json", "ostrack_validation_run.log", "ostrack_test_run.log",
]
RESULT_DIRS = {
    "external_verification": ("*.csv", "*.manifest.json"),
    "transfer_verification": ("*.csv",),
    "official": ("*.csv", "*.manifest.json"),
    "with_confidence/official": ("*.csv", "*.manifest.json"),
    "disjoint": ("*.csv", "*.manifest.json"),
    "contour_folds": ("*.manifest.json",),
    "gate_folds": ("*.manifest.json",),
    "chess_aligned": ("*.csv",),
    "chess_units": ("*.csv",),
    "figures": ("*.pdf", "*.png", "*.json"),
}
FRAMEWORK_ENTRY_POINTS = (
    "contour_similarity_net.py", "train_contour_similarity.py", "trackves_eval_all_methods.py",
    "2026-06-03-ours_scheme_c_nano方法.py", "2026-06-03-ours_scheme_c_nano跟踪器方法.py",
    "2026-06-03-HistAB核心方法.py", "2026-06-03-SchemeC-NanoPrior方法.py", "2026-06-03-跟踪器基础方法.py",
    "fusion_weight_head.py", "surgt_profiles.py", "MIS_C_Fe_Simi_Mesh_Match_Pose_Estimate_Track.py",
    "Deformation_F_T_C_G_Experi_hist_AB.py", "Deformation_F_T_C_G_Experi.py",
    "surgt_contour_ckpt.py", "surgt_contour_ckpt_resolve.py",
)
LEARNED_FILES = (
    "global_gate_online_tracker.py", "global_redetection_gate_oof.py", "global_redetection_probe.py",
    "global_redetection_visual_gate_oof.py", "visual_feature_exporter.py", "candidate_exporter.py",
    "dataset_builder.py", "visual_models.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def framework_closure() -> set[str]:
    pending, names = list(FRAMEWORK_ENTRY_POINTS), set()
    while pending:
        name = pending.pop()
        if name in names:
            continue
        path = PROJECT / name
        if not path.is_file():
            raise FileNotFoundError(path)
        names.add(name)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import):
                modules = [e.name.split(".")[0] for e in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module.split(".")[0]]
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                value = node.args[0].value
                if isinstance(value, str) and value.endswith(".py"):
                    modules = [value[:-3]]
            for module in modules:
                dependency = module + ".py"
                if (PROJECT / dependency).is_file() and dependency not in names:
                    pending.append(dependency)
    return names


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_readmes(manifests: dict) -> None:
    readme = f"""# Endoscopic tracking failure verification

Code, frozen protocol, per-frame outputs and figures for

> **{TITLE}**
> Ranyang Li, Nan Wei, Wufeng Liu, Chao Fan, Zhipeng Lin. Submitted to *Mathematics* (MDPI), Special Issue
> "Mathematical Methods for Image Processing and Understanding, 2nd Edition", October 2026.

The verifier monitors the unchanged box proposals of a frozen region-level tracker and reports a bounded
failure risk built from midrank intensities, a fixed-density gradient-quantile contour set, a hue–saturation
histogram and the tracker's own confidence. Fusion weights and alarm thresholds were frozen on the SurgT
validation partition before the test partition was touched; a sequence-level split-conformal threshold carries
a finite-sample specificity guarantee. The tracker's box is never modified.

| Proposer (SurgT test, 20 sequences) | Native AUROC | Fused AUROC | Δ [95% CI] |
|---|---|---|---|
| NanoTrack (selection) | {manifests['nano_native']:.3f} | {manifests['nano_fused']:.3f} | {manifests['nano_delta']} |
| STARK-ST50 (selection) | {manifests['stark_native']:.3f} | {manifests['stark_fused']:.3f} | {manifests['stark_delta']} |
| OSTrack (held-out) | {manifests['ost_native']:.3f} | {manifests['ost_fused']:.3f} | {manifests['ost_delta']} |

## What is in this repository

| Path | Content |
|---|---|
| `*.py` | Evaluation, analysis, figure and manuscript-assembly scripts (entry points listed below) |
| `legacy/` | Minimal TrackVes I/O helpers extracted from the historical project |
| `original_framework/` | Historical tracker code needed only for the Appendix A experiments (`run_disjoint.py`, `run_gate.py`) |
| `verification_protocol.json` | Frozen protocol: candidate grid, selection objective, threshold rule |
| `results/selected_verification.json` | Validation-only selection manifest (frozen before any test run) |
| `results/external_verification/{{validation,test}}/` | Per-frame cues, IoU and failure labels for NanoTrack, STARK-ST50 and OSTrack, with input-frame hashes |
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
git clone https://github.com/liranyang123456-commits/{REPO_NAME}.git
cd {REPO_NAME}
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
"""
    (STAGE / "README.md").write_text(readme, encoding="utf-8")
    (STAGE / "weights" / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (STAGE / "weights" / "README.md").write_text(f"""# Model weights (not redistributed)

Place the following files in this folder. The SHA-256 values are those recorded in the run manifests under
`results/external_verification/*/*.manifest.json` and `results/official/*/*.manifest.json`.

| File | Source | SHA-256 |
|---|---|---|
| `ostrack_vitb256_ce.pth.tar` | OSTrack author model folder https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy (file ID `1dySfPrSk-knAEo0lTXqZ1kMk3BCvVofV`, `vitb_256_mae_ce_32x4_ep300`) | `{manifests['ostrack_sha']}` |
| `stark_st50.pth.tar` | STARK author model folder https://drive.google.com/drive/folders/1fSgll53ZnVKeUn22W37Nijk-b9LGhMdN (file ID `1sV_idlYLyxeCIO2o4AQDvUFO5b-X-E4w`, `STARK-ST50`) | `{manifests['stark_sha']}` |
| `nanotrack_backbone_sim.onnx` | https://github.com/HonglinChu/SiamTrackers (NanoTrack V2 ONNX, as used by OpenCV `TrackerNano`) | `{manifests['nano_backbone_sha']}` |
| `nanotrack_head_sim.onnx` | same | `{manifests['nano_head_sha']}` |

`run_official_trackers.py` converts the author archives once into `*.state.pt` files containing only the
network tensors and loads them with `strict=True`.
""", encoding="utf-8")
    (STAGE / "external" / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (STAGE / "external" / "README.md").write_text("""# Third-party source trees (clone at the pinned commits)

| Repository | Commit | Local folder |
|---|---|---|
| https://github.com/botaoye/OSTrack | `33b5e12586216b7fd0e95d255bd01ba44cbec759` | `external/OSTrack` |
| https://github.com/researchmm/Stark | `162e1d37a38f92251cb380e90ccdb59608f2df3c` | `external/STARK` |
| https://github.com/JonathonLuiten/TrackEval | `12c8791b303e0a0b50f753af204249e622d0281a` | `external/TrackEval` |

The only local modifications made during the study were compatibility shims for current PyTorch/NumPy
(`torch._six` aliases, the torchvision URL-loader import, and `np.float` in TrackEval's HOTA). No architecture,
trained head, scoring rule or HOTA equation was changed. `run_official_trackers.py` records the commit and the
checkpoint hash in every run manifest.
""", encoding="utf-8")
    (STAGE / "LICENSE").write_text("""MIT License

Copyright (c) 2026 Ranyang Li, Nan Wei, Wufeng Liu, Chao Fan, Zhipeng Lin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""", encoding="utf-8")
    (STAGE / ".gitignore").write_text("""__pycache__/
*.pyc
.pytest_cache/
.venv/
external/OSTrack/
external/STARK/
external/TrackEval/
weights/*.pt
weights/*.pth
weights/*.tar
weights/*.onnx
results/frozen_trackves_frames/*/
results/smoke*/
*.npz
""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, default=STAGE)
    args = parser.parse_args()
    stage = args.stage
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    root = HERE / "results"
    for name in CODE:
        copy(HERE / name, stage / name)
    for path in (HERE / "legacy").glob("*.py"):
        copy(path, stage / "legacy" / path.name)
    copy(HERE / "README.md", stage / "docs" / "REVISION_NOTES.md")
    for name in RESULT_FILES:
        copy(root / name, stage / "results" / name)
    copy(root / "frozen_trackves_frames" / "manifest.json", stage / "results" / "frozen_trackves_frames.manifest.json")
    for directory, patterns in RESULT_DIRS.items():
        for pattern in patterns:
            for path in sorted((root / directory).rglob(pattern)):
                copy(path, stage / "results" / directory / path.relative_to(root / directory))
    for name in sorted(framework_closure()):
        copy(PROJECT / name, stage / "original_framework" / name)
    for name in LEARNED_FILES:
        copy(LEARNED / name, stage / "original_framework" / "learned_fusion" / name)
    # checksums of every result file
    lines = []
    for path in sorted((stage / "results").rglob("*")):
        if path.is_file():
            lines.append(f"{sha256(path)}  {path.relative_to(stage).as_posix()}")
    (stage / "results" / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # numbers for the README
    results = json.loads((root / "external_verification_summary.json").read_text())
    heldout = json.loads((root / "heldout_ostrack_summary.json").read_text())
    nano, stark = results["proposers"]["nanotrack"]["methods"], results["proposers"]["stark"]["methods"]

    def delta(tests, **keys):
        row = next(r for r in tests if all(r[k] == v for k, v in keys.items()))
        return f"{row['mean_paired_difference']:+.3f} [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]"

    ostrack_manifest = json.loads((root / "external_verification" / "test" / "ostrack.manifest.json").read_text())["proposer_metadata"]
    stark_manifest = json.loads((root / "external_verification" / "test" / "stark.manifest.json").read_text())["proposer_metadata"]
    nano_manifest = json.loads((root / "external_verification" / "test" / "nanotrack.manifest.json").read_text())["proposer_metadata"]
    manifests = {
        "nano_native": nano["native_confidence"]["sequence_macro"]["failure_auroc"]["mean"],
        "nano_fused": nano["verification"]["sequence_macro"]["failure_auroc"]["mean"],
        "nano_delta": delta(results["paired_tests"], proposer="nanotrack", comparator="native_confidence", metric="failure_auroc"),
        "stark_native": stark["native_confidence"]["sequence_macro"]["failure_auroc"]["mean"],
        "stark_fused": stark["verification"]["sequence_macro"]["failure_auroc"]["mean"],
        "stark_delta": delta(results["paired_tests"], proposer="stark", comparator="native_confidence", metric="failure_auroc"),
        "ost_native": heldout["methods"]["native_confidence"]["sequence_macro"]["failure_auroc"]["mean"],
        "ost_fused": heldout["methods"]["verification"]["sequence_macro"]["failure_auroc"]["mean"],
        "ost_delta": delta(heldout["paired_tests"], comparator="native_confidence"),
        "ostrack_sha": ostrack_manifest["checkpoint_sha256"], "stark_sha": stark_manifest["checkpoint_sha256"],
        "nano_backbone_sha": nano_manifest["backbone_sha256"], "nano_head_sha": nano_manifest["head_sha256"],
    }
    write_readmes(manifests)
    total = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    print(f"STAGED {stage} files={sum(1 for p in stage.rglob('*') if p.is_file())} size_mb={total / 1e6:.1f}")


if __name__ == "__main__":
    main()
