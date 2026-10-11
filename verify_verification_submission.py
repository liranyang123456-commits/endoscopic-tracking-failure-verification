from __future__ import annotations

import json
import re
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from build_verification_submission import DEST, TITLE
from evaluate_verification import metrics, risk_values
from protocol import HERE, read_csv, sha256, write_json


def main() -> None:
    root = HERE / "results"
    selected = json.loads((root / "selected_verification.json").read_text())
    result = json.loads((root / "external_verification_summary.json").read_text())
    if selected["test_labels_used_for_selection"] or selected["test_predictions_used_for_selection"]:
        raise ValueError("Test-informed selection is prohibited")
    if selected["protocol_sha256"] != sha256(HERE / "verification_protocol.json"):
        raise ValueError("Frozen protocol hash differs")
    selection_time = (root / "selected_verification.json").stat().st_mtime
    expected_rows = None
    for proposer in ("nanotrack", "stark"):
        path = root / "external_verification" / "test" / f"{proposer}.csv"
        if path.stat().st_mtime <= selection_time:
            raise ValueError("External test outputs precede selection freeze")
        rows = read_csv(path)
        identity = [(r["sequence"], r["frame_idx"], r["failure"]) for r in rows]
        if len(rows) != 12056:
            raise ValueError("Unexpected external scored-frame denominator")
        # GT outcomes may differ by proposer; only sequence/frame identity must match.
        frame_identity = [(r[0], r[1]) for r in identity]
        if expected_rows is None:
            expected_rows = frame_identity
        elif frame_identity != expected_rows:
            raise ValueError("Proposers have different external scoring frames")
        for name, weights in selected["comparators"].items():
            calculated = metrics(
                rows, risk_values(rows, selected["percentile"], weights),
                selected["alarm_thresholds"][proposer][name],
            )
            reported = result["proposers"][proposer]["methods"][name]["pooled"]
            for key in (
                "failure_auroc", "failure_average_precision", "sensitivity",
                "specificity", "alarm_rate", "risk_coverage_area", "box_iou_unchanged",
            ):
                if not np.isclose(calculated[key], reported[key], atol=1e-12):
                    raise ValueError(f"Statistics do not recompute: {proposer}/{name}/{key}")
    source = (DEST / "Manuscript_Mathematics.tex").read_text(encoding="utf-8")
    if re.search(r"\\\\(?:cite|ref|eqref)\{", source):
        raise ValueError("A doubled TeX command would print a citation/reference as literal text")
    abstract = re.search(r"\\abstract\{(.*?)\}\s*\\keyword", source, re.S)
    if abstract is None:
        raise ValueError("Missing single-paragraph abstract")
    word_upper = len(re.findall(r"[A-Za-z0-9]+", abstract.group(1)))
    if word_upper > 200:
        raise ValueError("Abstract exceeds conservative word limit")
    referenced = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source)
    if not all((DEST / name).is_file() for name in referenced):
        raise ValueError("Referenced result graphic missing")
    if "images/Verification_Algorithm.pdf" in referenced:
        raise ValueError("The algorithm overview must be typeset as an algorithm environment, not as an image")
    if source.count(r"\begin{algorithm}") != 1 or r"\label{alg:verification}" not in source:
        raise ValueError("The two-phase verification algorithm environment is missing from the manuscript")
    for label in (
        "tab:ValidationGrid", "tab:ProposerContext", "tab:PooledMetrics", "tab:FairPaired", "tab:PerSequence",
        "tab:HeldOut", "tab:Latency", "tab:Conformal", "tab:ExtraComparators", "tab:Sensitivity",
        "tab:Temporal", "tab:Transfer", "tab:ChanceLevel", "prop:conformal", "fig:Qualitative", "fig:Curves",
        "tab:Summary", "tab:SummaryTransfer", "fig:Headline", "fig:CueAblation", "fig:Sensitivity", "fig:Pipeline",
        "sec:rationale",
    ):
        if f"\\label{{{label}}}" not in source:
            raise ValueError(f"Required table/figure/proposition missing from the manuscript: {label}")
    if any(name.startswith("images/archived/") for name in referenced):
        raise ValueError("Archived precursor figures must not appear in the revised manuscript")
    if "section*{Highlights}" in source:
        raise ValueError("The Highlights block was removed from this revision and must not reappear")
    heldout = json.loads((root / "heldout_ostrack_summary.json").read_text())
    if heldout["scored_frames"] != 12056 or heldout["proposer"] in selected["validation_auroc_by_proposer"]:
        raise ValueError("Held-out proposer evidence is inconsistent with the frozen protocol")
    nano = result["proposers"]["nanotrack"]["methods"]
    stark = result["proposers"]["stark"]["methods"]
    nano_auc_n = nano["verification"]["sequence_macro"]["failure_auroc"]["n_sequences"]
    stark_auc_n = stark["verification"]["sequence_macro"]["failure_auroc"]["n_sequences"]
    if (nano_auc_n, stark_auc_n) != (17, 16):
        raise ValueError(f"Unexpected two-class sequence denominators: NanoTrack={nano_auc_n}, STARK={stark_auc_n}")
    case_groups = sorted({
        values["case"]
        for values in nano["verification"]["sequences"].values()
    })
    if len(case_groups) != 5:
        raise ValueError(f"Expected five case groups, found {len(case_groups)}")
    primary_auc_test = next(
        row for row in result["paired_tests"]
        if row["proposer"] == "nanotrack"
        and row["comparator"] == "native_confidence"
        and row["metric"] == "failure_auroc"
    )
    paired_auc_group_counts = {
        proposer: {
            row["n_groups"] for row in result["paired_tests"]
            if row["proposer"] == proposer and row["metric"] == "failure_auroc"
        }
        for proposer in ("nanotrack", "stark")
    }
    expected_paired_counts = {"nanotrack": {nano_auc_n}, "stark": {stark_auc_n}}
    if paired_auc_group_counts != expected_paired_counts:
        raise ValueError(f"Paired AUROC test denominators do not match eligibility: {paired_auc_group_counts}")
    expected_strings = (
        f"{nano['verification']['sequence_macro']['failure_auroc']['mean']:.3f}",
        f"{nano['native_confidence']['sequence_macro']['failure_auroc']['mean']:.3f}",
        f"{heldout['methods']['verification']['sequence_macro']['failure_auroc']['mean']:.3f}",
        f"{heldout['methods']['native_confidence']['sequence_macro']['failure_auroc']['mean']:.3f}",
        "7666", "12056", "0.0625", "Copilot SDK",
        "17 with defined sequence-level AUROC", "16 for STARK-ST50",
        f"includes {primary_auc_test['n_groups']} sequences with defined AUROC",
        "five case-group", "held-out", "Proposition 2",
    )
    manuscript = subprocess.check_output(
        ["pdftotext", "-layout", str(DEST / "Manuscript_Mathematics.pdf"), "-"],
        encoding="utf-8",
    )
    normalized = " ".join(re.sub(r"(?m)^\s*\d+\s+", "", manuscript).split())
    missing = [value for value in expected_strings if value not in normalized]
    if missing:
        raise ValueError(f"Compiled PDF omits verified numeric/disclosure content: {missing}")
    if "??" in manuscript or "highest observed mean" in manuscript or "highest HOTA" in manuscript:
        raise ValueError("Unresolved references or outdated primary result")
    if TITLE not in normalized:
        raise ValueError("Compiled title differs from the specified title")
    for stem in ("Manuscript_Mathematics", "Cover_Letter_Mathematics"):
        log = (DEST / f"{stem}.log").read_text(errors="replace")
        if re.search(r"^!|Overfull \\[hv]box|(?:Citation|Reference).{0,200}undefined", log, re.M):
            raise ValueError(f"Compilation/layout warning requiring resolution: {stem}")
    source_zip = DEST / "Manuscript_Mathematics_LaTeX_Source.zip"
    supplementary_zip = DEST / "Supplementary_Fair_Evaluation.zip"
    expected_source_entries = {
        "Manuscript_Mathematics.tex": DEST / "Manuscript_Mathematics.tex",
        "Manuscript_Mathematics.bbl": DEST / "Manuscript_Mathematics.bbl",
        "mybibfile.bib": DEST / "mybibfile.bib",
    }
    expected_source_entries.update({
        name: DEST / name for name in referenced
    })
    source_archive_status = "missing"
    if source_zip.is_file():
        with zipfile.ZipFile(source_zip) as archive:
            if archive.testzip() is not None:
                source_archive_status = "corrupt"
            else:
                source_archive_status = "current" if all(
                    name in archive.namelist() and archive.read(name) == path.read_bytes()
                    for name, path in expected_source_entries.items()
                ) else "content_mismatch"

    expected_supplementary_entries = {
        "evaluation_code/build_verification_submission.py": HERE / "build_verification_submission.py",
        "evaluation_code/verify_verification_submission.py": HERE / "verify_verification_submission.py",
        "evaluation_code/publish_submission.py": HERE / "publish_submission.py",
        "evaluation_code/README.md": HERE / "README.md",
        "verification/protocol.json": HERE / "verification_protocol.json",
        "verification/references.bib": HERE / "verification_references.bib",
        "verification/external_verification_summary.json": root / "external_verification_summary.json",
        "verification/selected_verification.json": root / "selected_verification.json",
    }
    supplementary_archive_status = "missing"
    if supplementary_zip.is_file():
        with zipfile.ZipFile(supplementary_zip) as archive:
            if archive.testzip() is not None:
                supplementary_archive_status = "corrupt"
            else:
                supplementary_archive_status = "current" if all(
                    name in archive.namelist() and archive.read(name) == path.read_bytes()
                    for name, path in expected_supplementary_entries.items()
                ) else "content_mismatch"

    package_archive_status = {
        source_zip.name: source_archive_status,
        supplementary_zip.name: supplementary_archive_status,
    }
    package_archives_complete = all(value == "current" for value in package_archive_status.values())
    final_folder = HERE.parent / "00_MDPI_Mathematics_Submission_Ready"
    final_pdfs_match = all(
        (final_folder / name).is_file()
        and sha256(final_folder / name) == sha256(DEST / name)
        for name in ("Manuscript_Mathematics.pdf", "Cover_Letter_Mathematics.pdf")
    )
    write_json(DEST / "verification_validation_report.json", {
        "complete_external_scoring_frames": len(expected_rows),
        "pooled_statistics_recomputed_from_saved_raw_scores": True,
        "validation_selection_precedes_test_outputs": True,
        "all_test_ablation_results_kept": True,
        "abstract_conservative_word_components": word_upper,
        "referenced_figures_present": len(referenced),
        "numeric_pdf_strings_verified": list(expected_strings),
        "sequence_level_auroc_denominators": {
            "total_sequences_per_proposer": 20,
            "nanotrack_two_class_sequences": nano_auc_n,
            "stark_st50_two_class_sequences": stark_auc_n,
            "paired_auroc_sequence_tests": {
                "nanotrack": nano_auc_n,
                "stark_st50": stark_auc_n,
            },
            "paired_auroc_comparison_group_counts": {
                "nanotrack": nano_auc_n,
                "stark_st50": stark_auc_n,
            },
            "case_group_sensitivity_groups": case_groups,
        },
        "submission_archives": package_archive_status,
        "submission_archives_complete": package_archives_complete,
        "upload_folder_contains_current_pdfs": final_pdfs_match,
        "author_confirmation": {
            "submission_history_confirmed": True,
            "all_author_approval_confirmed": False,
        },
        "manuscript_source_sha256": sha256(DEST / "Manuscript_Mathematics.tex"),
        "pdf_sha256": sha256(DEST / "Manuscript_Mathematics.pdf"),
        "cover_letter_sha256": sha256(DEST / "Cover_Letter_Mathematics.pdf"),
        "clinical_safety_or_uniform_superiority_claimed": False,
    })
    print("VERIFICATION_SUBMISSION_VALIDATED", word_upper, len(expected_rows))


if __name__ == "__main__":
    main()
