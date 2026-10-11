#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
【冻结】Ours Scheme C + HistAB — 统一入口（2026-06-03）

文件布局:
  - ``2026-06-03-HistAB核心方法.py``     形变/粗搜核心
  - ``2026-06-03-SchemeC-NanoPrior方法.py``  Nano 先验 + log 融合
  - ``2026-06-03-跟踪器基础方法.py``       BaseTracker + bbox 工具
  - ``2026-06-03-ours_scheme_c_nano跟踪器方法.py``  跟踪器类（本方法主体）

后续优化 **只改上述冻结文件**；评测脚本通过本模块创建跟踪器。
"""
from __future__ import annotations

import argparse
import importlib
import os
from typing import Any, Literal

METHOD_DATE = "2026-06-03"
METHOD_ID = "ours_scheme_c_nano"
METHOD_LABEL = "Ours (Scheme C: Nano+log+TopR) [frozen 2026-06-03]"
HISTAB_METHOD_ID = "ours_hist_ab"
FROZEN_TRACKER_MODULE = "2026-06-03-ours_scheme_c_nano跟踪器方法"

TRACKER_KEYS = frozenset({
    "ours_scheme_c_nano", "ours_c_nano", "ours_c", "ours_hist_ab", "ours", "ours_nano",
})


def _tracker_mod():
    return importlib.import_module(FROZEN_TRACKER_MODULE)


def _scheme_mod():
    return importlib.import_module("2026-06-03-SchemeC-NanoPrior方法")


def _histab_mod():
    return importlib.import_module("2026-06-03-HistAB核心方法")


def apply_method_defaults(profile: str = "v3_chess", *, segment_frames: int = 0) -> str:
    from surgt_profiles import apply_profile

    key = (profile or "v3_chess").strip().lower()
    if key in ("chess", "cascade", "v3_chess", "chess_cascade"):
        key = "v3_chess"
    resolved = apply_profile(key, segment_frames=segment_frames)
    os.environ["MIS_SURGT_PROFILE"] = resolved
    os.environ.setdefault("MIS_COARSE_SCHEME", "ab")
    os.environ.setdefault("MIS_SCHEME_C_POST_BLEND", "0")
    return resolved


def create_tracker(init_im: Any, init_bbox: tuple[int, int, int, int], *, variant: str = "ours_scheme_c_nano") -> Any:
    mod = _tracker_mod()
    key = str(variant).strip().lower()
    if key in ("ours_c", "ours_c_nano"):
        key = "ours_scheme_c_nano"
    if key in ("ours_scheme_c_nano",):
        return mod.OursSchemeCNanoPriorTracker(init_im, init_bbox)
    if key in ("ours", "ours_hist_ab"):
        return mod.OursHistABTracker(init_im, init_bbox)
    raise ValueError(f"Unknown Ours variant: {variant!r}")


def get_tracker_classes() -> dict[str, type]:
    mod = _tracker_mod()
    return {
        "ours_hist_ab": mod.OursHistABTracker,
        "ours_scheme_c_nano": mod.OursSchemeCNanoPriorTracker,
        "ours_c_nano": mod.OursSchemeCNanoPriorTracker,
    }


def is_ours_tracker(name: str) -> bool:
    return str(name).strip().lower() in TRACKER_KEYS


if __name__ == "__main__":
    print(f"[{METHOD_DATE}] frozen Ours package")
    print("  HistAB   : 2026-06-03-HistAB核心方法.py")
    print("  SchemeC  : 2026-06-03-SchemeC-NanoPrior方法.py")
    print("  Tracker  :", FROZEN_TRACKER_MODULE)
