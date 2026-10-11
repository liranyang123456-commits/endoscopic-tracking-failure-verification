"""SurgT / Scheme C runtime profiles (shared by surgt_mis_tracker and ablation runs)."""
from __future__ import annotations

import os


def _setenv(key: str, val: str, *, force: bool = False) -> None:
    if force or os.environ.get(key, "") == "":
        os.environ[key] = str(val)


def _apply_cascade_base() -> None:
    try:
        import importlib

        sch = importlib.import_module("2026-06-03-SchemeC-NanoPrior方法")
        sch.apply_cascade_fusion_defaults()
    except Exception:
        pass


def apply_profile(name: str, *, segment_frames: int = 0) -> str:
    """Apply MIS_* env for Scheme C. Returns resolved profile name."""
    key = str(name or "standard").strip().lower()
    if key in ("legacy", "standard", "surgt"):
        key = "standard"
    elif key in ("chess", "v3_chess", "cascade", "chess_cascade"):
        key = "v3_chess"
    elif key in ("lite", "anchor_lite", "anchor-lite"):
        key = "anchor_lite"
    elif key in ("tuned", "surgt_tuned", "robust"):
        key = "surgt_tuned"
    elif key in ("continuous", "surgt_continuous", "cont"):
        key = "surgt_continuous"
    elif key in ("nano_first", "surgt_nano_first", "nano-first"):
        key = "surgt_nano_first"

    os.environ["MIS_SURGT_PROFILE"] = key
    os.environ.setdefault("MIS_SILENT", "1")

    if key == "turbo":
        _apply_turbo()
        os.environ["MIS_SURGT_PROFILE"] = key
        return key
    if key == "v3_chess":
        _apply_v3_chess()
        os.environ["MIS_SURGT_PROFILE"] = key
        return key
    if key == "anchor_lite":
        _apply_anchor_lite(segment_frames)
        os.environ["MIS_SURGT_PROFILE"] = key
        return key
    if key == "surgt_tuned":
        _apply_surgt_tuned(segment_frames)
        os.environ["MIS_SURGT_PROFILE"] = key
        return key
    if key == "surgt_continuous":
        _apply_surgt_continuous(segment_frames)
        os.environ["MIS_SURGT_PROFILE"] = key
        return key
    if key == "surgt_nano_first":
        _apply_surgt_nano_first(segment_frames)
        os.environ["MIS_SURGT_PROFILE"] = key
        return key
    _apply_standard()
    os.environ["MIS_SURGT_PROFILE"] = key
    return key


def _apply_standard() -> None:
    """Previous SurgT normal profile (64 templates)."""
    os.environ.pop("MIS_SURGT_TURBO", None)
    for k, v in {
        "MIS_SCHEME_C_POST_BLEND": "0",
        "MIS_NANO_LITE_ENABLE": "1",
        "MIS_NANO_PRIOR_CACHE": "1",
        "MIS_NANO_CASCADE": "1",
        "MIS_NUM_TEMPLATES": "64",
        "MIS_COARSE_STEP": "18",
        "MIS_RERANK_TOPR": "24",
        "MIS_RERANK_TOPR_LITE": "18",
        "MIS_MATCH_RESP_ENABLE": "0",
        "MIS_MAX_TIME_S": "0.65",
        "MIS_HIST_TOP_M": "48",
        "MIS_CONTOUR_INFER_BS": "24",
        "MIS_RTW_POOL_ENABLE": "0",
        "MIS_MEMORY_BANK_ENABLE": "0",
        "MIS_NANO_LITE_NS": "0.70",
        "MIS_NANO_LITE_MOTION_PX": "16",
        "MIS_NANO_LITE_PREV_SCORE": "0.68",
    }.items():
        os.environ[k] = v
    _apply_cascade_base()


def _apply_turbo() -> None:
    os.environ["MIS_SURGT_TURBO"] = "1"
    _setenv("MIS_NUM_TEMPLATES", "1", force=True)
    _setenv("MIS_SCHEME_C_POST_BLEND", "0", force=True)
    _setenv("MIS_NANO_LITE_ENABLE", "1", force=True)
    _setenv("MIS_NANO_PRIOR_CACHE", "1", force=True)
    _setenv("MIS_NANO_CASCADE", "1", force=True)
    _setenv("MIS_MATCH_RESP_ENABLE", "0", force=True)
    _setenv("MIS_MAX_TIME_S", "0.35", force=True)
    _setenv("MIS_COARSE_STEP", "24", force=True)
    _setenv("MIS_RERANK_TOPR", "12", force=True)
    _setenv("MIS_RERANK_TOPR_LITE", "8", force=True)
    _setenv("MIS_HIST_TOP_M", "24", force=True)
    _setenv("MIS_CONTOUR_INFER_BS", "8", force=True)
    _setenv("MIS_RTW_POOL_ENABLE", "0", force=True)
    _setenv("MIS_MEMORY_BANK_ENABLE", "0", force=True)
    _setenv("MIS_CHESS_ONLINE_SIMI", "0", force=True)
    _setenv("MIS_NANO_LITE_NS", "0.55", force=True)
    _setenv("MIS_NANO_PLUS_ENABLE", "1", force=True)
    _setenv("MIS_NANO_PLUS_RAW", "1", force=True)
    _setenv("MIS_NANO_OUTPUT_RAW_ENABLE", "1", force=True)
    _setenv("MIS_CASCADE_PRIMARY_HIST", "0.00", force=True)
    _apply_cascade_base()


def _apply_v3_chess() -> None:
    """Chessboard v3 cascade profile (compare_chessboard_trackers --profile cascade)."""
    _setenv("MIS_SCHEME_C_POST_BLEND", "0", force=True)
    _setenv("MIS_NANO_LITE_ENABLE", "1", force=True)
    _setenv("MIS_NANO_PRIOR_CACHE", "1", force=True)
    _setenv("MIS_MATCH_RESP_COND", "1", force=True)
    _setenv("MIS_NANO_CASCADE", "1", force=True)
    _apply_cascade_base()
    overrides = {
        "MIS_CASCADE_PRIMARY_BLEND": "0.0",
        "MIS_CASCADE_PRIMARY_NS": "0.44",
        "MIS_CASCADE_PRIMARY_HIST": "0.00",
        "MIS_CASCADE_PRIMARY_MOTION": "64",
        "MIS_POLICY_APPEAR_PRIMARY_NS": "0.56",
        "MIS_POLICY_APPEAR_PRIMARY_SZ": "0.28",
        "MIS_POLICY_APPEAR_PRIMARY_MOTION": "48",
        "MIS_NANO_SIZE_ADAPT": "1",
        "MIS_NANO_SIZE_ADAPT_NS": "0.44",
        "MIS_NANO_SIZE_ADAPT_SZ": "0.20",
        "MIS_NANO_PLUS_ENABLE": "1",
        "MIS_NANO_PLUS_NS": "0.44",
        "MIS_NANO_PLUS_HIST": "0.00",
        "MIS_NANO_PLUS_RAW": "1",
        "MIS_NANO_PLUS_HOLD_PREV": "1",
        "MIS_NANO_PLUS_HOLD_PREV_SCORE": "0.55",
        "MIS_NANO_OUTPUT_RAW_ENABLE": "1",
        "MIS_NANO_OUTPUT_RAW_NS": "0.44",
        "MIS_RTW_POOL_ENABLE": "0",
        "MIS_MEMORY_BANK_ENABLE": "0",
        "MIS_COARSE_STEP": "20",
        "MIS_RERANK_TOPR": "28",
        "MIS_RERANK_TOPR_LITE": "20",
        "MIS_MATCH_RESP_ENABLE": "0",
        "MIS_MAX_TIME_S": "0.85",
        "MIS_NUM_TEMPLATES": "128",
        "MIS_HIST_TOP_M": "64",
        "MIS_CONTOUR_INFER_BS": "32",
    }
    for k, v in overrides.items():
        os.environ[k] = v


def _short_segment(segment_frames: int) -> bool:
    thresh = int(os.environ.get("MIS_SURGT_SHORT_SEGMENT_THRESH", "180"))
    return segment_frames > 0 and segment_frames < thresh


def _apply_anchor_lite(segment_frames: int) -> None:
    """Nano-first anchor mode; optional pure-Nano on short sub-sequences."""
    if _short_segment(segment_frames):
        _apply_turbo()
        os.environ["MIS_SURGT_ANCHOR_MODE"] = "nano_only_short"
        return
    os.environ.pop("MIS_SURGT_TURBO", None)
    os.environ["MIS_SURGT_ANCHOR_MODE"] = "anchor_lite"
    _setenv("MIS_SCHEME_C_POST_BLEND", "0", force=True)
    _setenv("MIS_NANO_LITE_ENABLE", "1", force=True)
    _setenv("MIS_NANO_PRIOR_CACHE", "1", force=True)
    _setenv("MIS_NANO_CASCADE", "1", force=True)
    _setenv("MIS_NUM_TEMPLATES", "12", force=True)
    _setenv("MIS_COARSE_STEP", "22", force=True)
    _setenv("MIS_RERANK_TOPR", "16", force=True)
    _setenv("MIS_RERANK_TOPR_LITE", "12", force=True)
    _setenv("MIS_MATCH_RESP_ENABLE", "0", force=True)
    _setenv("MIS_MAX_TIME_S", "0.40", force=True)
    _setenv("MIS_HIST_TOP_M", "24", force=True)
    _setenv("MIS_CONTOUR_INFER_BS", "12", force=True)
    _setenv("MIS_RTW_POOL_ENABLE", "0", force=True)
    _setenv("MIS_NANO_LITE_NS", "0.52", force=True)
    _setenv("MIS_NANO_LITE_MOTION_PX", "28", force=True)
    _setenv("MIS_NANO_LITE_PREV_SCORE", "0.50", force=True)
    _setenv("MIS_NANO_PLUS_ENABLE", "1", force=True)
    _setenv("MIS_NANO_PLUS_NS", "0.38", force=True)
    _setenv("MIS_NANO_PLUS_HIST", "0.00", force=True)
    _setenv("MIS_NANO_PLUS_RAW", "1", force=True)
    _setenv("MIS_NANO_PLUS_HOLD_PREV", "1", force=True)
    _setenv("MIS_NANO_PLUS_HOLD_PREV_SCORE", "0.42", force=True)
    _setenv("MIS_NANO_OUTPUT_RAW_ENABLE", "1", force=True)
    _setenv("MIS_NANO_OUTPUT_RAW_NS", "0.35", force=True)
    _setenv("MIS_CASCADE_PRIMARY_BLEND", "0.0", force=True)
    _setenv("MIS_CASCADE_PRIMARY_NS", "0.36", force=True)
    _setenv("MIS_CASCADE_PRIMARY_HIST", "0.00", force=True)
    _setenv("MIS_CASCADE_LOCAL_HIST", "0.18", force=True)
    _apply_cascade_base()


def _apply_surgt_tuned(segment_frames: int) -> None:
    """SurgT-tuned: maximize Rob2D (avoid failures) while keeping Nano+ quality."""
    if _short_segment(segment_frames):
        _apply_turbo()
        os.environ["MIS_SURGT_ANCHOR_MODE"] = "tuned_short"
        return
    os.environ.pop("MIS_SURGT_TURBO", None)
    os.environ["MIS_SURGT_ANCHOR_MODE"] = "tuned_long"
    _apply_v3_chess()
    tuned = {
        "MIS_NUM_TEMPLATES": "32",
        "MIS_COARSE_STEP": "20",
        "MIS_RERANK_TOPR": "20",
        "MIS_MAX_TIME_S": "0.50",
        "MIS_NANO_LITE_NS": "0.48",
        "MIS_NANO_LITE_PREV_SCORE": "0.45",
        "MIS_NANO_PLUS_HOLD_PREV_SCORE": "0.40",
        "MIS_NANO_OUTPUT_RAW_NS": "0.32",
        "MIS_CASCADE_PRIMARY_NS": "0.40",
        "MIS_CASCADE_LOCAL_HIST": "0.15",
        "MIS_SURGT_TEMPLATE_CACHE": "1",
    }
    for k, v in tuned.items():
        os.environ[k] = v


def _apply_surgt_continuous(segment_frames: int) -> None:
    """SurgT-Continuous: surgt_tuned + stronger Nano/secval for long single-init tracks."""
    _apply_surgt_tuned(segment_frames)
    cont = {
        "MIS_NUM_TEMPLATES": "24",
        "MIS_RERANK_TOPR": "18",
        "MIS_MAX_TIME_S": "0.55",
        "MIS_NANO_LITE_NS": "0.44",
        "MIS_NANO_LITE_PREV_SCORE": "0.42",
        "MIS_NANO_PLUS_NS": "0.36",
        "MIS_NANO_PLUS_HOLD_PREV_SCORE": "0.38",
        "MIS_NANO_OUTPUT_RAW_NS": "0.30",
        "MIS_CASCADE_PRIMARY_NS": "0.36",
        "MIS_CASCADE_SECVAL_ENABLE": "1",
        "MIS_CASCADE_SECVAL_RISK_MIN": "0.40",
        "MIS_CASCADE_SECVAL_IOU_MAX": "0.45",
        "MIS_DRIFT_STREAK_MAX": "4",
        "MIS_RELOC_EXPAND": "140",
    }
    for k, v in cont.items():
        os.environ[k] = v


def _apply_surgt_nano_first(segment_frames: int) -> None:
    """SurgT-Continuous: Nano-dominant cascade (target case_1 Rob2D gaps vs CSRT)."""
    _apply_anchor_lite(segment_frames)
    os.environ["MIS_SURGT_ANCHOR_MODE"] = "nano_first"
    nano = {
        "MIS_NUM_TEMPLATES": "8",
        "MIS_COARSE_STEP": "24",
        "MIS_RERANK_TOPR": "12",
        "MIS_RERANK_TOPR_LITE": "8",
        "MIS_MAX_TIME_S": "0.35",
        "MIS_CASCADE_PRIMARY_BLEND": "0.90",
        "MIS_CASCADE_PRIMARY_NS": "0.28",
        "MIS_CASCADE_PRIMARY_HIST": "0.99",
        "MIS_NANO_LITE_NS": "0.40",
        "MIS_NANO_LITE_PREV_SCORE": "0.38",
        "MIS_NANO_PLUS_NS": "0.30",
        "MIS_NANO_PLUS_HOLD_PREV_SCORE": "0.32",
        "MIS_NANO_OUTPUT_RAW_NS": "0.22",
        "MIS_POLICY_T_STABLE": "0.22",
        "MIS_POLICY_T_APPEAR": "0.88",
        "MIS_POLICY_T_GEOM": "0.94",
        "MIS_POLICY_APPEAR_PRIMARY_NS": "0.28",
        "MIS_POLICY_APPEAR_PRIMARY_MOTION": "72",
        "MIS_NANO_PRIOR_BETA": "0.62",
        "MIS_RERANK_WEIGHTS": "0.52,0.28,0.12,0.08",
        "MIS_DRIFT_STREAK_MAX": "3",
        "MIS_RELOC_EXPAND": "160",
    }
    for k, v in nano.items():
        os.environ[k] = v
