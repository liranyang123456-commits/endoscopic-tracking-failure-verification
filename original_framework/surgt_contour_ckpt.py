#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""运行时切换 ContourSimilarityNet 权重（不改冻结 HistAB 文件）。"""
from __future__ import annotations

import importlib
import os
from pathlib import Path

_APPLIED: str = ""


def apply_contour_checkpoint(ckpt_path: str | Path) -> bool:
    global _APPLIED
    path = Path(ckpt_path)
    if not path.is_file():
        return False
    os.environ["MIS_CONTOUR_SIM_CKPT"] = str(path.resolve())
    mod = importlib.import_module("2026-06-03-HistAB核心方法")
    model = mod._load_contour_similarity_model(str(path))
    mod.contour_sim_model = model
    _APPLIED = str(path)
    if not os.environ.get("MIS_SILENT"):
        print(f"[ContourSim] runtime ckpt -> {path}")
    return model is not None


def get_applied_checkpoint() -> str:
    return _APPLIED or os.environ.get("MIS_CONTOUR_SIM_CKPT", "")
