#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""解析并加载 ContourSimilarityNet 权重（SurgT > 环境变量 > TrackVes > Chess）。"""
from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_CHAIN = (
    Path(r"E:\contour_sim_runs\surgt_v1\contour_similarity_best.pt"),
    Path(r"E:\contour_sim_runs\trackves_triplet_v1\contour_similarity_triplet_best.pt"),
    Path(r"E:\contour_sim_runs\chess_v1\contour_similarity_best.pt"),
)


def resolve_contour_ckpt(explicit: str | Path | None = None) -> Path | None:
    if explicit is not None:
        p = Path(explicit)
        if p.is_file():
            return p
    env = os.environ.get("MIS_CONTOUR_SIM_CKPT", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for p in _DEFAULT_CHAIN:
        if p.is_file():
            return p
    return None


def ensure_surgt_contour_ckpt(explicit: str | Path | None = None) -> str:
    """运行时注入 HistAB.contour_sim_model；返回实际使用的 ckpt 路径或空串。"""
    path = resolve_contour_ckpt(explicit)
    if path is None:
        return ""
    from surgt_contour_ckpt import apply_contour_checkpoint

    apply_contour_checkpoint(path)
    return str(path.resolve())
