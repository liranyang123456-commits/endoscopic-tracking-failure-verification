from __future__ import annotations

import json
import ast
import argparse
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from build_fair_submission import DESTINATION as FAIR_DESTINATION
from build_verification_submission import TITLE
from protocol import HERE, LEARNED, PROJECT, sha256, write_json


FINAL = HERE.parent / "00_MDPI_Mathematics_Submission_Ready"


def inspect_pdf(path: Path) -> str:
    text = subprocess.check_output(["pdftotext", "-layout", str(path), "-"], encoding="utf-8")
    if not text.strip() or "??" in text:
        raise ValueError(f"Unreadable PDF or unresolved reference: {path}")
    if "Sensors" in text or "declined" in text:
        raise ValueError(f"False prior-submission wording in PDF: {path}")
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verification", action="store_true")
    parser.add_argument(
        "--stage-only", action="store_true",
        help="Build and validate verification archives in the assembly directory without touching the upload folder",
    )
    args = parser.parse_args()
    if args.stage_only and not args.verification:
        parser.error("--stage-only is supported only with --verification")
    destination = (
        HERE.parent / "Mathematics_Verification_Submission_20261011"
        if args.verification else FAIR_DESTINATION
    )
    manifest_name = "verification_assembly_manifest.json" if args.verification else "assembly_manifest.json"
    assembly = json.loads((destination / manifest_name).read_text(encoding="utf-8"))
    source = destination / "Manuscript_Mathematics.tex"
    source_hash_key = "manuscript_sha256" if args.verification else "new_manuscript_sha256"
    if sha256(source) != assembly[source_hash_key]:
        raise ValueError("Manuscript changed after validated result assembly")
    if args.verification:
        checks = {
            "external_verification_summary.json": "external_results_sha256",
            "selected_verification.json": "selected_protocol_sha256",
            "transfer_verification_summary.json": "transfer_results_sha256",
        }
        for name, key in checks.items():
            if sha256(HERE / "results" / name) != assembly[key]:
                raise ValueError(f"Verification evidence changed after manuscript assembly: {name}")
    else:
        if sha256(HERE / "results" / "fair_tracking_summary.json") != assembly["tracking_results_sha256"]:
            raise ValueError("Tracking results changed after manuscript assembly")
        if sha256(HERE / "results" / "cholecseg_grouped.json") != assembly["grouped_results_sha256"]:
            raise ValueError("Grouped statistics changed after manuscript assembly")
    manuscript = inspect_pdf(destination / "Manuscript_Mathematics.pdf")
    letter = inspect_pdf(destination / "Cover_Letter_Mathematics.pdf")
    if "not previously been submitted to an MDPI journal" not in " ".join(letter.split()):
        raise ValueError("Cover letter's no-prior-MDPI statement is missing")
    if "Copilot SDK" not in manuscript or "Ours-SD" not in manuscript:
        raise ValueError("Required AI disclosure or updated method naming is absent from PDF")
    canonical_text = source.read_text(encoding="utf-8")
    abstract = re.search(r"\\abstract\{(.*?)\}\s*\\keyword", canonical_text, flags=re.S)
    if abstract is None or len(re.findall(r"[A-Za-z0-9]+", abstract.group(1))) > 200:
        raise ValueError("Abstract exceeds 200 words under conservative component counting")
    for name in ("Manuscript_Mathematics", "Cover_Letter_Mathematics"):
        log = (destination / f"{name}.log").read_text(encoding="utf-8", errors="replace")
        if any(line.startswith("!") for line in log.splitlines()):
            raise ValueError(f"Fatal compilation error: {name}")
        if re.search(r"(Citation|Reference) .{0,180}undefined", log):
            raise ValueError(f"Undefined citation/reference requires inspection: {name}")
        if "Overfull \\hbox" in log:
            raise ValueError(f"Visible layout overflow: {name}")

    names = (
        "Manuscript_Mathematics.pdf", "Cover_Letter_Mathematics.pdf",
        "Manuscript_Mathematics_LaTeX_Source.zip",
    )
    source_zip = destination / "Manuscript_Mathematics_LaTeX_Source.zip"
    with zipfile.ZipFile(source_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("Manuscript_Mathematics.tex", "Manuscript_Mathematics.bbl", "mybibfile.bib"):
            archive.write(destination / name, name)
        for directory in ("Definitions", "images"):
            for path in sorted((destination / directory).rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(destination))
    with zipfile.ZipFile(source_zip) as archive:
        if archive.read("Manuscript_Mathematics.tex") != source.read_bytes():
            raise ValueError("LaTeX ZIP does not contain the exact compiled manuscript source")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt source archive entry: {bad}")

    supplementary = destination / "Supplementary_Fair_Evaluation.zip"
    with zipfile.ZipFile(supplementary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(HERE.glob("*.py")):
            archive.write(path, Path("evaluation_code") / path.name)
        for name in ("README.md", "requirements.txt"):
            archive.write(HERE / name, Path("evaluation_code") / name)
        if args.verification:
            archive.write(HERE / "verification_protocol.json", "verification/protocol.json")
            archive.write(HERE / "verification_references.bib", "verification/references.bib")
            archive.write(
                HERE / "verification_protocol.json",
                "evaluation_code/verification_protocol.json",
            )
            archive.write(
                HERE / "verification_references.bib",
                "evaluation_code/verification_references.bib",
            )
            for name in (
                "selected_verification.json", "external_verification_summary.json",
                "external_verification_per_sequence.csv", "transfer_verification_summary.json",
                "supplementary_analyses.json", "heldout_ostrack_summary.json",
                "heldout_ostrack_per_sequence.csv",
            ):
                archive.write(HERE / "results" / name, Path("verification") / name)
            archive.write(
                HERE / "results" / "figures" / "Verification_Qualitative.manifest.json",
                "verification/qualitative_cases.manifest.json",
            )
            for path in sorted((HERE / "results" / "with_confidence" / "official").glob("*\\*.csv")):
                archive.write(path, Path("verification") / "confidence_reruns" / path.relative_to(HERE / "results" / "with_confidence" / "official"))
            for path in sorted((HERE / "results" / "with_confidence" / "official").glob("*\\*.manifest.json")):
                archive.write(path, Path("verification") / "confidence_reruns" / path.relative_to(HERE / "results" / "with_confidence" / "official"))
            for path in sorted((HERE / "results" / "external_verification").glob("*\\*.manifest.json")):
                archive.write(path, Path("verification") / path.relative_to(HERE / "results"))
            for path in sorted((HERE / "results" / "external_verification").glob("*\\*.csv")):
                archive.write(path, Path("verification") / path.relative_to(HERE / "results"))
            for path in sorted((HERE / "results" / "transfer_verification").glob("*.csv")):
                archive.write(path, Path("verification") / path.relative_to(HERE / "results"))
        for name in (
            "fair_tracking_summary.json", "cholecseg_grouped.json",
            "chess_fair_legacy.json", "chess_fair_per_sequence.csv",
            "trackves_fair_per_sequence.csv", "cholecseg_per_video.csv",
            "gate_split_manifest.json", "image_derived_contour_patches.manifest.json",
        ):
            path = HERE / "results" / name
            archive.write(path, Path("aggregate_results") / name)
        archive.write(
            HERE / "results" / "frozen_trackves_frames" / "manifest.json",
            "provenance/frozen_frame_hashes.json",
        )
        for directory in ("contour_folds", "gate_folds"):
            for path in sorted((HERE / "results" / directory).glob("*.manifest.json")):
                archive.write(path, Path("provenance") / directory / path.name)
        for path in sorted((HERE / "results" / "official").glob("*\\*.manifest.json")):
            archive.write(path, Path("provenance") / path.relative_to(HERE / "results"))
        project_names = (
            "contour_similarity_net.py", "train_contour_similarity.py",
            "trackves_eval_all_methods.py", "2026-06-03-ours_scheme_c_nano方法.py",
            "2026-06-03-ours_scheme_c_nano跟踪器方法.py", "2026-06-03-HistAB核心方法.py",
            "2026-06-03-SchemeC-NanoPrior方法.py", "2026-06-03-跟踪器基础方法.py",
            "fusion_weight_head.py", "surgt_profiles.py",
            "MIS_C_Fe_Simi_Mesh_Match_Pose_Estimate_Track.py",
            "Deformation_F_T_C_G_Experi_hist_AB.py", "Deformation_F_T_C_G_Experi.py",
            "surgt_contour_ckpt.py", "surgt_contour_ckpt_resolve.py",
        )
        pending = list(project_names)
        dependency_names = set()
        while pending:
            name = pending.pop()
            if name in dependency_names:
                continue
            path = PROJECT / name
            if not path.is_file():
                raise FileNotFoundError(f"Required framework dependency missing: {path}")
            dependency_names.add(name)
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                modules = []
                if isinstance(node, ast.Import):
                    modules = [entry.name.split(".")[0] for entry in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules = [node.module.split(".")[0]]
                elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                    value = node.args[0].value
                    if isinstance(value, str) and value.endswith(".py"):
                        modules = [value[:-3]]
                    elif (
                        isinstance(value, str) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "import_module"
                    ):
                        modules = [value]
                for module in modules:
                    dependency = module + ".py"
                    if (PROJECT / dependency).is_file() and dependency not in dependency_names:
                        pending.append(dependency)
        for name in sorted(dependency_names):
            path = PROJECT / name
            if not path.is_file():
                raise FileNotFoundError(f"Required original framework source missing: {path}")
            archive.write(path, Path("original_framework") / name)
        for name in (
            "global_gate_online_tracker.py", "global_redetection_gate_oof.py",
            "global_redetection_probe.py", "global_redetection_visual_gate_oof.py",
            "visual_feature_exporter.py", "candidate_exporter.py", "dataset_builder.py",
            "visual_models.py",
        ):
            archive.write(LEARNED / name, Path("original_framework") / "learned_fusion" / name)
        for repository in ("OSTrack", "STARK", "TrackEval"):
            directory = HERE / "external" / repository
            for path in sorted(directory.rglob("*")):
                if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
                    continue
                if path.suffix not in (".py", ".yaml", ".yml", ".sh") and path.name not in (
                    "LICENSE", "requirements.txt", "README.md", "MODEL_ZOO.md"
                ):
                    continue
                if path.name == "local.py":
                    continue
                archive.write(path, Path("official_sources") / repository / path.relative_to(directory))
    with zipfile.ZipFile(supplementary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Supplementary archive failed integrity check")
        if any(name.endswith((".npz", ".pt", ".pth", ".onnx", ".png", ".jpg")) for name in archive.namelist()):
            raise ValueError("Restricted image/model payload accidentally entered supplementary package")
    if args.stage_only:
        staged = {
            "verification_focused_staging_only": True,
            "upload_folder_modified": False,
            "source_archive_sha256": sha256(source_zip),
            "supplementary_archive_sha256": sha256(supplementary),
            "manuscript_source_sha256": sha256(source),
            "external_evidence_sha256": sha256(HERE / "results" / "external_verification_summary.json"),
            "validation_report_embedded_in_supplementary": False,
            "reason_validation_report_is_external": "The local verification report records archive status and cannot self-embed without becoming stale.",
        }
        write_json(destination / "staged_submission_archives_manifest.json", staged)
        print(destination)
        return

    FINAL.mkdir(parents=True, exist_ok=True)
    backup = HERE / ("pre_verification_submission_backup" if args.verification else "historical_submission_backup")
    backup.mkdir(exist_ok=True)
    for name in (*names, "Supplementary_Fair_Evaluation.zip"):
        previous = FINAL / name
        if previous.is_file() and not (backup / name).exists():
            shutil.copy2(previous, backup / name)
    for name in names[:2]:
        source_path = destination / name
        shutil.copy2(source_path, FINAL / name)
        if sha256(source_path) != sha256(FINAL / name):
            raise ValueError(f"Copied upload PDF differs from validated PDF: {name}")
    shutil.copy2(source_zip, FINAL / names[2])
    shutil.copy2(supplementary, FINAL / supplementary.name)
    subprocess.run([
        "pdftotext", "-layout", str(destination / "Cover_Letter_Mathematics.pdf"),
        str(destination / "Cover_Letter_Mathematics.txt"),
    ], check=True)
    write_json(destination / "final_package_manifest.json", {
        "title": TITLE, "upload_folder": str(FINAL),
        "primary_files": {name: sha256(FINAL / name) for name in names},
        "supplementary_file": supplementary.name,
        "supplementary_sha256": sha256(FINAL / supplementary.name),
        "old_results_backed_up": str(backup),
        "original_framework_main_sources_not_overwritten": True,
        "raw_controlled_images_gt_coordinates_and_model_weights_redistributed": False,
        "AI_assistance_disclosed": True,
        "verification_focused_revision": args.verification,
        "external_validation_test_outcomes_included": args.verification,
    })
    print(FINAL)


if __name__ == "__main__":
    main()
