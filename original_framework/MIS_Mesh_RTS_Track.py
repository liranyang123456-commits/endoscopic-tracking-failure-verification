#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入兼容层：训练/对比脚本使用 ``import MIS_Mesh_RTS_Track``，而仓库内主实现文件名为
带空格/编号的副本（无法作为合法 Python 模块名直接 import）。

本文件将该实现注册为 ``sys.modules['MIS_Mesh_RTS_Track']``，供 IDE 与 ``import MIS_Mesh_RTS_Track`` 使用。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SUB = Path(__file__).resolve().parent / "MIS_Mesh_RTS_Track copy3Dmesh copy 3.py"

if __name__ == "__main__":
    import runpy

    runpy.run_path(str(_SUB), run_name="__main__")
else:
    if not _SUB.is_file():
        raise ImportError(
            f"MIS_Mesh_RTS_Track: 需要同目录下的实现文件 {_SUB.name!s}（路径 {_SUB}）"
        )
    _spec = importlib.util.spec_from_file_location(__name__, _SUB)
    if _spec is None or _spec.loader is None:
        raise ImportError(_SUB)
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[__name__] = _mod
    _spec.loader.exec_module(_mod)
