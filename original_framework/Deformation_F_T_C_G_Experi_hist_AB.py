#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兼容 shim → 实现已迁至 ``2026-06-03-HistAB核心方法.py``（冻结 2026-06-03）。"""
from importlib import import_module as _im
import sys as _sys

_mod = _im("2026-06-03-HistAB核心方法")
_sys.modules[__name__] = _mod
globals().update({k: getattr(_mod, k) for k in dir(_mod) if not k.startswith("__")})
