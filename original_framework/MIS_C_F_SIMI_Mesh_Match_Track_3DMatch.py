#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIS_C_F_SIMI_Mesh_Match_Track_3DMatch.py

基于 `MIS_C_Fe_Simi_Mesh_Match_Pose_Estimate_Track.py` 的“直接复用版”跟踪脚本：

================================================================================
总体逻辑（鸟瞰）
================================================================================
在图像序列上估计**每一帧目标的矩形跟踪框 bbox**。不输出相机/物体在真实世界中的
累计位姿（RT）；若开启光流或 uvh 刚体等分支，其产物仅用于**诊断或辅助打分**，默认
不改变“只做 bbox 跟踪”的主目标。

数据流可以概括为：**首帧锁定目标外观与（可选）伪 3D 表示 → 后续帧在上一帧位置附近
搜索候选框 → 用统一前处理与多种相似度/渲染度量精排 → 取最高分框作为本帧结果并
更新状态**，循环直到序列结束。

================================================================================
核心步骤（与 ``main()`` 主流程对齐）
================================================================================
1. **配置与 IO**  
   构造 ``core.TrackConfig``，用环境变量 ``MIS_*`` 覆盖帧目录、输出目录、搜索半径/步长、
   匹配模式等；枚举 ``frames``；创建 ``out_root`` 下各子目录（bbox 可视化、mask、
   contour、uvh_render 等，视模式与开关而定）。

2. **首帧目标初始化**  
   交互式涂抹 mask（或 ``MIS_INIT_BBOX`` 矩形）确定目标区域；在 ROI 上执行 **th_68**
   灰度截断，与用户 mask 结合得到首帧 ``th_gray``、``mask``、``contour_rgb``；为
   **uvh_render / lib** 等模式从 ``th_gray`` 采样 **伪 3D 点云 (u,v,h)** 作为参考形状。

3. **可选的一次性准备（非每帧必跑）**  
   - 构建或加载 **多视点目标库** ``target_library/``（上半球虚拟相机渲染的轮廓模板 +
     每视点 ``T_w2c``）。  
   - 可选：首两帧 **光流 → uvh 匹配 → RANSAC/IRLS**，结果写入 ``init_pair_0000_0001/``
    （校验用，不驱动主跟踪决策）。  
   - 可选：将默认/库视点下的 **相机内参 K、外参 T_w2c** 等写入 ``camera_init/`` 便于核对
     （见 ``MIS_SAVE_CAMERA_INIT_DEBUG``）。

4. **逐帧跟踪循环（frame ≥ 1）**  
   读入当前帧；以 **上一帧 bbox 为中心** 在局部窗内生成多个候选位置/尺度；**quick mask
   IoU** 保留 top-K；再按 ``MIS_TRACK_MATCH_MODE`` 进入 **2d / lib / uvh_render / uvh_icp**
   之一做精排（轮廓网络或回退相似度、库模板匹配、伪 3D 渲染对齐、Open3D ICP 等，见下文
   “重要说明”）；得到本帧最优 ``bbox`` 及中间图；更新 ``prev_bbox``、``prev_mask_small``、
   （可选）``prev_uvh_pts``、库视点门控状态等。

5. **可选并行诊断**  
   ``MIS_ENABLE_FLOW_UVH=1`` 时，每帧在跟踪决策**之后**额外算光流与 uvh 刚体，写入
   ``pair_results/``，**不参与** bbox 选取。

6. **收尾与断点**  
   写出 ``tracking_bboxes_and_scores.json``、``diag_track_only.csv``、``meta_track_only.json``；
   若启用 ``MIS_RESUME_ENABLE``，按间隔写 checkpoint，便于续跑。

================================================================================
以下为能力说明与参数索引（与上文步骤交叉引用）
================================================================================

- 保留原始的跟踪逻辑：th_68（灰度截断）-> 轮廓 -> 轮廓相似度(可选，ContourSimilarityNet 或回退) -> 滑动窗口搜索 bbox
- 保留原始的保存逻辑：每帧保存 bbox 可视化、ROI、mask、th_gray、contour_rgb、对比图等
- 断点续跑（checkpoint）：默认关闭；设置环境变量 ``MIS_RESUME_ENABLE=1`` 可启用（与 ``MIS_FORCE_RESTART=0`` 配合）
- 仅将“跟踪流程”与“位姿/RT 输出”解耦：本脚本默认只跑跟踪并不导出 RT 结果；RT/ICP/flow/mesh 配准相关代码在原脚本中保持完整保留（本脚本复用其函数实现，但不作为主流程输出）。

重要说明（与原 RT 脚本的关系）：
- 本脚本主目标是 bbox 跟踪；
  - `MIS_TRACK_MATCH_MODE=2d`：mask IoU 粗筛 + 单模板轮廓相似度（ContourSimilarityNet 或回退）精排 + 空间先验（与原脚本最接近）。
  - `MIS_TRACK_MATCH_MODE=lib`：构建上半球多视点目标库（如 1024 视点），精排为“候选轮廓 ↔ 库模板”向量化 IoU+相关性匹配，并记录 `view_id`。
  - `MIS_TRACK_MATCH_MODE=uvh_render`（**本脚本代码默认**）：伪3D 点云(uvh) + 可选 `MIS_UVH_RENDER_VIEWLIB_ENABLE` 做多视角粗检索 + 平面 (s,θ,tx,ty) 迭代渲染匹配；对比图在 `uvh_render/`。
  - `MIS_TRACK_MATCH_MODE=uvh_icp`：需 open3d；不可用时脚本会回退到 `uvh_render`。
- 同时，为了便于你确认“光流 + 伪3D(uvh)匹配”确实在工作，本脚本支持在跟踪过程中额外运行：
  Farneback 光流 $\rightarrow$ uvh 点对 $\rightarrow$ RANSAC/IRLS 刚体拟合，并保存中间结果与匹配可视化。
  该诊断流程**默认关闭**；设置 `MIS_ENABLE_FLOW_UVH=1` 开启。
- 本脚本不做 mesh-ICP 的三角网格配准与位姿累计导出（这是原 RT 脚本的功能）；这里的 uvh 刚体仅作为“伪3D匹配/诊断”结果保存。

提速相关（默认值已略偏快；均可通过环境变量恢复旧行为）：
- 搜索：`MIS_SEARCH_RADIUS` / `MIS_SEARCH_STEP` / `MIS_TOPK_SEMANTIC`；`MIS_PROC_SCALE`（core，<1 降分辨率）。
- uvh_render：`MIS_UVH_FRAME_TH68_ENABLE`（默认 1：本帧整图只算一次 th_68，搜索子窗与 rebuild 从全图裁剪；设 0 恢复「每子窗重算 th_68」，与旧行为统计上可能略有差异）、
  `MIS_UVH_RENDER_TWO_STAGE_ENABLE`（粗步长 + 细步长局部）、`MIS_UVH_WARM_START_ENABLE`（帧间中心位移缩小窗）、
  `MIS_UVH_RENDER_BBOX_SCALES` / `MIS_UVH_RENDER_MAX_POINTS` / `MIS_UVH_ALIGN_*` / `MIS_UVH_ALIGN_EARLY_STOP_SIM_DELTA`、
  `MIS_UVH_SEMANTIC_ENABLE` / `MIS_UVH_RENDER_VIEWLIB_ENABLE` / `MIS_UVH_VIEWLIB_TOPK`、
  `MIS_UVH_RIGID_ENABLE` / `MIS_UVH_RENDERFIELD_ENABLE` / `MIS_UVH_DIFFRENDER_ENABLE`、
  `MIS_SAVE_DEBUG`（默认 1：写入 `frames_with_bbox/`、`roi/`、`mask/`、`th_gray/`、`contour_rgb/`、`vis/`、`contour_compare/` 等；设 0 关闭）、
  `MIS_UVH_RENDER_SAVE_DEBUG`（默认随 `MIS_SAVE_DEBUG` 开启 uvh 目录下 ply/meta 等重包；仅关此项可设 0）、`MIS_UVH_RENDER_SAVE_DEBUG_STRIDE`、
  `MIS_ENABLE_FLOW_UVH`、目标库 `MIS_LIBRARY_NUM_VIEWS`（勿随意 `MIS_LIBRARY_FORCE_REBUILD=1`）。
- 可选 JIT：`pip install numba` 后设 `MIS_UVH_NUMBA_ENABLE=1` 加速 mask IoU。
- 相机初始化调试：默认在输出目录写入 `camera_init/`（`uvh_topdown_camera.npz`、`camera_init_meta.json` 等）；设 `MIS_SAVE_CAMERA_INIT_DEBUG=0` 可关闭。

多视点目标库输出（`target_library/`）：
- `view_library_templates.npz`：模板二值轮廓(用于向量化匹配) + 每视点相机外参 `T_w2c`
- `renders/contour_rgb/`：每视点渲染轮廓图
- `renders/depth_vis/`：每视点深度可视化（伪渲染图）
- `view_points_cam/`：每视点相机坐标系下的点云（作为“每个 target 的 3D 模型结果”）
- `init_pair_0000_0001/`：首两帧 flow->uvh->RANSAC/IRLS 的 RT 估计与可视化（用于渲染/验证）

ROI 获取方式：
- 默认使用 OpenCV 的 `selectROI`（鼠标拖拽框选；可理解为“鼠标点击获得跟踪区域”）
- 若 HighGUI 不可用，可回退 Tkinter 选框（沿用原脚本）
- 也支持用环境变量 `MIS_INIT_BBOX="x,y,w,h"` 直接指定初始框（沿用原脚本）

推荐运行方式（PowerShell）：

```powershell
$env:MIS_RT_MODE="roi"  # 对本脚本无强制作用，但沿用同名配置以保持一致
python d:/MIS_Pose_Track_Re3D/MIS_C_F_SIMI_Mesh_Match_Track_3DMatch.py
```
"""

from __future__ import annotations

import contextlib
import os
import json
import time
import csv
import math
from dataclasses import dataclass
from typing import Any, Optional
from functools import lru_cache  # 核心优化：LRU 缓存

import numpy as np
import cv2

# 直接复用原始实现（不修改原文件）
import MIS_C_Fe_Simi_Mesh_Match_Pose_Estimate_Track as core

# bit-pack 配置（用于大规模 target_library，例如 64k）
_MASK_BITORDER = "little"  # 只要 pack/unpack 一致即可；popcount 与 bitorder 无关
_POPCOUNT_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


def _make_heightfield_mesh_arrays(
    *,
    th_gray_truncated: np.ndarray,
    bbox: tuple[int, int, int, int],
    z_scale: float,
    z_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    从 th_gray_truncated（0~255, float/uint）生成伪3D高度场 mesh：
      - verts: (N,3) float32，坐标在整帧坐标系 (u+bx, v+by, h)
      - faces: (M,3) int32
      - colors: (N,3) uint8 或 None
    不依赖 open3d，便于在无 open3d 环境也能保存 mesh 结果。
    """
    bx, by, w, h = [int(v) for v in bbox]
    img = np.asarray(th_gray_truncated, dtype=np.float32)
    if img.ndim != 2:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), None
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    m = np.isfinite(img) & (img > float(z_eps))
    if int(np.count_nonzero(m)) < 50:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), None

    vid = -np.ones((h, w), dtype=np.int32)
    idx = np.flatnonzero(m.reshape(-1))
    vid.reshape(-1)[idx] = np.arange(int(idx.size), dtype=np.int32)

    ys, xs = np.nonzero(m)
    z = (img[ys, xs] / 255.0) * float(z_scale)
    vx = xs.astype(np.float32) + float(bx)
    vy = ys.astype(np.float32) + float(by)
    verts = np.stack([vx, vy, z.astype(np.float32)], axis=1).astype(np.float32, copy=False)

    faces: list[list[int]] = []
    for yy in range(h - 1):
        vrow = vid[yy]
        vrow2 = vid[yy + 1]
        for xx in range(w - 1):
            v00 = int(vrow[xx])
            v10 = int(vrow[xx + 1])
            v01 = int(vrow2[xx])
            v11 = int(vrow2[xx + 1])
            if v00 < 0 or v10 < 0 or v01 < 0 or v11 < 0:
                continue
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    if len(faces) < 50:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), None

    faces_a = np.asarray(faces, dtype=np.int32)
    # height color (z normalized)
    try:
        z0 = verts[:, 2].astype(np.float32)
        zmin = float(np.min(z0))
        zmax = float(np.max(z0))
        den = float((zmax - zmin) if (zmax - zmin) > 1e-9 else 1.0)
        zn = (z0 - float(zmin)) / float(den)
        col = np.stack([zn, 1.0 - zn, 0.2 + 0.0 * zn], axis=1)
        col = np.clip(col * 255.0, 0, 255).astype(np.uint8)
    except Exception:
        col = None
    return verts, faces_a, col


def _save_ply_ascii(
    path: str,
    *,
    verts: np.ndarray,
    faces: np.ndarray,
    colors_u8: np.ndarray | None,
) -> None:
    """保存一个简单的 ASCII PLY（三角面）。"""
    v = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if v.size == 0 or f.size == 0:
        return
    c = None
    if colors_u8 is not None:
        c2 = np.asarray(colors_u8, dtype=np.uint8).reshape(-1, 3)
        if c2.shape[0] == v.shape[0]:
            c = c2

    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {int(v.shape[0])}\n")
        fp.write("property float x\nproperty float y\nproperty float z\n")
        if c is not None:
            fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fp.write(f"element face {int(f.shape[0])}\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")
        if c is None:
            for (x, y, z) in v:
                fp.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")
        else:
            for (x, y, z), (r, g, b) in zip(v, c):
                fp.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f} {int(r)} {int(g)} {int(b)}\n")
        for (a, b, c3) in f:
            fp.write(f"3 {int(a)} {int(b)} {int(c3)}\n")


def _save_mesh_npz(path: str, *, verts: np.ndarray, faces: np.ndarray, colors_u8: np.ndarray | None) -> None:
    v = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
    if v.size == 0 or f.size == 0:
        return
    if colors_u8 is not None:
        c = np.asarray(colors_u8, dtype=np.uint8).reshape(-1, 3)
    else:
        c = None
    np.savez_compressed(path, verts=v.astype(np.float16), faces=f.astype(np.int32), colors=(c.astype(np.uint8) if c is not None else None))


def _save_point_cloud_ply_ascii(
    path: str,
    *,
    pts_xyz: np.ndarray,
    colors_u8: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> None:
    pts = np.asarray(pts_xyz, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return
    cols = None
    if colors_u8 is not None:
        c2 = np.asarray(colors_u8, dtype=np.uint8).reshape(-1, 3)
        if c2.shape[0] == pts.shape[0]:
            cols = c2
    nrm = None
    if normals is not None:
        n2 = np.asarray(normals, dtype=np.float32).reshape(-1, 3)
        if n2.shape[0] == pts.shape[0]:
            nrm = n2

    with open(path, "w", encoding="utf-8", newline="\n") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {int(pts.shape[0])}\n")
        fp.write("property float x\nproperty float y\nproperty float z\n")
        if nrm is not None:
            fp.write("property float nx\nproperty float ny\nproperty float nz\n")
        if cols is not None:
            fp.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fp.write("element face 0\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")
        if cols is None and nrm is None:
            for (x, y, z) in pts:
                fp.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f}\n")
        elif cols is None and nrm is not None:
            for (x, y, z), (nx, ny, nz) in zip(pts, nrm):
                fp.write(
                    f"{float(x):.6f} {float(y):.6f} {float(z):.6f} "
                    f"{float(nx):.6f} {float(ny):.6f} {float(nz):.6f}\n"
                )
        elif cols is not None and nrm is None:
            for (x, y, z), (r, g, b) in zip(pts, cols):
                fp.write(f"{float(x):.6f} {float(y):.6f} {float(z):.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for (x, y, z), (nx, ny, nz), (r, g, b) in zip(pts, nrm, cols):
                fp.write(
                    f"{float(x):.6f} {float(y):.6f} {float(z):.6f} "
                    f"{float(nx):.6f} {float(ny):.6f} {float(nz):.6f} "
                    f"{int(r)} {int(g)} {int(b)}\n"
                )


def _solid_point_colors(n: int, rgb: tuple[int, int, int]) -> np.ndarray:
    rr, gg, bb = [int(max(0, min(255, v))) for v in rgb]
    return np.tile(np.asarray([[rr, gg, bb]], dtype=np.uint8), (int(max(0, n)), 1))


def _colorize_points_by_height(
    pts_xyz: np.ndarray,
    *,
    low_rgb: tuple[int, int, int] = (60, 120, 255),
    high_rgb: tuple[int, int, int] = (255, 220, 80),
) -> np.ndarray:
    pts = np.asarray(pts_xyz, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    z = pts[:, 2].astype(np.float32)
    zmin = float(np.nanmin(z)) if z.size > 0 else 0.0
    zmax = float(np.nanmax(z)) if z.size > 0 else 1.0
    den = float((zmax - zmin) if abs(zmax - zmin) > 1e-9 else 1.0)
    t = np.clip((z - float(zmin)) / float(den), 0.0, 1.0).reshape(-1, 1)
    c0 = np.asarray(low_rgb, dtype=np.float32).reshape(1, 3)
    c1 = np.asarray(high_rgb, dtype=np.float32).reshape(1, 3)
    cols = (1.0 - t) * c0 + t * c1
    return np.clip(cols, 0, 255).astype(np.uint8)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, "").strip()
        return int(v) if v else int(default)
    except Exception:
        return int(default)


# 环境变量缓存（避免重复读取和解析）
_env_cache: dict[str, Any] = {}


def _env_int_cached(name: str, default: int) -> int:
    """缓存的环境变量读取，避免重复解析（性能优化）"""
    if name not in _env_cache:
        _env_cache[name] = _env_int(name, default)
    return int(_env_cache[name])


def _env_float_cached(name: str, default: float) -> float:
    """缓存的环境变量读取，避免重复解析（性能优化）"""
    if name not in _env_cache:
        _env_cache[name] = _env_float(name, default)
    return float(_env_cache[name])


def _env_bool_cached(name: str, default: bool) -> bool:
    """缓存的环境变量读取，避免重复解析（性能优化）"""
    if name not in _env_cache:
        _env_cache[name] = _env_bool(name, default)
    return bool(_env_cache[name])


def _env_float(name: str, default: float) -> float:
    try:
        v = os.environ.get(name, "").strip()
        return float(v) if v else float(default)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    try:
        v = os.environ.get(name, "").strip()
        if not v:
            return bool(default)
        return bool(int(v))
    except Exception:
        return bool(default)


try:
    from numba import njit  # type: ignore
except ImportError:
    njit = None  # type: ignore

_UVH_IOU_NUMBA_FN: Any = None


def _uvh_iou_u8_numba_jit():
    """Numba 加速的 IoU（可选）；失败或未安装 numba 时返回 None。默认开启（若 numba 可用）"""
    if njit is None or not bool(_env_bool("MIS_UVH_NUMBA_ENABLE", True)):  # 默认开启
        return None
    try:

        @njit(cache=True, fastmath=False)
        def _iou(a: np.ndarray, b: np.ndarray) -> float:
            h, w = a.shape
            inter = 0
            union = 0
            for i in range(h):
                for j in range(w):
                    aa = a[i, j] > 0
                    bb = b[i, j] > 0
                    if aa and bb:
                        inter += 1
                    if aa or bb:
                        union += 1
            if union <= 0:
                return 0.0
            return float(inter) / float(union + 1e-6)

        return _iou
    except Exception:
        return None


def _uvh_iou_u8_numba_fn():
    global _UVH_IOU_NUMBA_FN
    if _UVH_IOU_NUMBA_FN is None and njit is not None:
        _UVH_IOU_NUMBA_FN = _uvh_iou_u8_numba_jit()
    return _UVH_IOU_NUMBA_FN


_RENDERFIELD_CACHE: dict[str, Any] = {
    "attempted": False,
    "checkpoint": "",
    "pack": None,
    "error": "",
}


def _get_renderfield_refiner(*, cfg: core.TrackConfig | None = None) -> dict[str, Any] | None:
    enable = bool(_env_bool("MIS_UVH_RENDERFIELD_ENABLE", False))
    if not enable:
        return None
    checkpoint_path = (
        os.environ.get(
            "MIS_UVH_RENDERFIELD_CHECKPOINT",
            r"E:\uvh_point_rt_runs\all_datasets_firstframe_v1\point_rt_renderfield_best.pt",
        )
        or ""
    ).strip().strip('"').strip("'")
    if not checkpoint_path:
        return None
    global _RENDERFIELD_CACHE
    if (
        bool(_RENDERFIELD_CACHE.get("attempted", False))
        and str(_RENDERFIELD_CACHE.get("checkpoint", "")) == str(checkpoint_path)
    ):
        pack = _RENDERFIELD_CACHE.get("pack", None)
        return pack if isinstance(pack, dict) else None
    _RENDERFIELD_CACHE = {
        "attempted": True,
        "checkpoint": str(checkpoint_path),
        "pack": None,
        "error": "",
    }
    if not os.path.isfile(checkpoint_path):
        _RENDERFIELD_CACHE["error"] = "checkpoint_not_found"
        _log_once(
            "uvh_rf_missing_ckpt",
            f"[UVH-RF][Skip] checkpoint 不存在: {checkpoint_path}",
            cfg=cfg,
            level=1,
        )
        return None
    try:
        import torch  # type: ignore
        import uvh_point_rt_renderfield as upr  # type: ignore
        import uvh_rt_regression as urr  # type: ignore

        device_req = (os.environ.get("MIS_UVH_RENDERFIELD_DEVICE", "cuda") or "cuda").strip().lower()
        ckpt_pack = upr.load_point_rt_renderfield_checkpoint(checkpoint_path, device=device_req)
        model = ckpt_pack["model"]
        device_t = ckpt_pack["device"]
        pack = {
            "torch": torch,
            "upr": upr,
            "urr": urr,
            "model": model,
            "device": device_t,
            "checkpoint_path": str(checkpoint_path),
            "image_size": int(getattr(model, "image_size", 96) or 96),
            "point_in_dim": int(getattr(model, "point_in_dim", 3) or 3),
        }
        _RENDERFIELD_CACHE["pack"] = pack
        _log_once(
            "uvh_rf_ready",
            f"[UVH-RF] ready checkpoint={checkpoint_path} image_size={int(pack['image_size'])} device={device_t}",
            cfg=cfg,
            level=1,
        )
        return pack
    except Exception as e:
        _RENDERFIELD_CACHE["error"] = f"{e}"
        _log_once(
            "uvh_rf_init_fail",
            f"[UVH-RF][Skip] 初始化失败: {e}",
            cfg=cfg,
            level=1,
        )
        return None


def _build_renderfield_observation(
    *,
    th_gray: np.ndarray,
    mask_u8: np.ndarray,
    image_size: int,
) -> np.ndarray | None:
    img = int(max(16, image_size))
    dep = np.asarray(th_gray, dtype=np.float32)
    mk = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if dep.ndim != 2 or mk.ndim != 2:
        return None
    if dep.shape != mk.shape:
        mk = cv2.resize(mk, (int(dep.shape[1]), int(dep.shape[0])), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    dep = np.where(mk > 0, dep, 0.0).astype(np.float32)
    dep_r = cv2.resize(dep, (img, img), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    mk_r = cv2.resize(mk, (img, img), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    valid = (mk_r > 0) & np.isfinite(dep_r)
    depth = np.zeros((img, img), dtype=np.float32)
    if int(np.count_nonzero(valid)) <= 0:
        return None
    vals = dep_r[valid]
    vmin = float(np.percentile(vals, 2.0)) if vals.size >= 8 else float(np.min(vals))
    vmax = float(np.percentile(vals, 98.0)) if vals.size >= 8 else float(np.max(vals))
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or abs(vmax - vmin) <= 1e-6:
        depth[valid] = 1.0
    else:
        depth_v = np.clip((vals - float(vmin)) / float(max(1e-6, vmax - vmin)), 0.0, 1.0).astype(np.float32)
        depth[valid] = depth_v
    depth = np.where(valid, depth, 0.0).astype(np.float32)
    dep_blur = cv2.GaussianBlur(depth, (3, 3), sigmaX=0.0)
    dzdx = cv2.Sobel(dep_blur, cv2.CV_32F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(dep_blur, cv2.CV_32F, 0, 1, ksize=3)
    normals = np.stack([-dzdx, -dzdy, np.ones_like(dep_blur, dtype=np.float32)], axis=2).astype(np.float32)
    den = np.linalg.norm(normals, axis=2, keepdims=True)
    den = np.maximum(den, 1e-6)
    normals = normals / den
    normals = np.where(valid[..., None], normals, 0.0).astype(np.float32)
    normal01 = np.clip((normals + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    normal01 = normal01 * valid[..., None].astype(np.float32)
    render = np.concatenate([depth[..., None], normal01], axis=2).astype(np.float32)
    return np.transpose(render, (2, 0, 1)).astype(np.float32, copy=False)


def _refine_pose_bbox_with_renderfield(
    *,
    ref_points: np.ndarray,
    init_T_cur_ref: np.ndarray,
    cand: dict[str, Any],
    frame_wh: tuple[int, int],
    cfg: core.TrackConfig,
) -> dict[str, Any]:
    res = {
        "ok": False,
        "best_loss": float("nan"),
        "steps": int(max(1, _env_int("MIS_UVH_RENDERFIELD_STEPS", 40))),
        "lr": float(_env_float("MIS_UVH_RENDERFIELD_LR", 5e-3)),
        "bbox": None,
        "projected_ref_pts": None,
        "pose_vec": None,
        "error": "",
    }
    pack = _get_renderfield_refiner(cfg=cfg)
    if not isinstance(pack, dict):
        res["error"] = str(_RENDERFIELD_CACHE.get("error", "renderfield_unavailable"))
        return res
    point_in_dim = int(pack.get("point_in_dim", 3))
    pts_raw = np.asarray(ref_points, dtype=np.float32)
    if pts_raw.ndim == 1:
        pts_raw = pts_raw.reshape(-1, 3)
    if int(pts_raw.shape[0]) < 80:
        res["error"] = "too_few_ref_points"
        return res
    pts = _ensure_renderfield_point_dim(pts_raw, point_in_dim)
    obs_np = _build_renderfield_observation(
        th_gray=np.asarray(cand.get("th_gray", None), dtype=np.float32),
        mask_u8=np.asarray(cand.get("mask_u8", None), dtype=np.uint8),
        image_size=int(pack["image_size"]),
    )
    if not isinstance(obs_np, np.ndarray):
        res["error"] = "invalid_render_observation"
        return res
    try:
        torch_mod = pack["torch"]
        urr = pack["urr"]
        upr = pack["upr"]
        init_pose_np = urr._transform_to_pose_vec(np.asarray(init_T_cur_ref, dtype=np.float32).reshape(4, 4))
        opt = upr.optimize_pose_from_render_observation(
            model=pack["model"],
            points=torch_mod.from_numpy(pts.astype(np.float32, copy=False)),
            render_obs=torch_mod.from_numpy(obs_np.astype(np.float32, copy=False)),
            init_pose_vec=torch_mod.from_numpy(np.asarray(init_pose_np, dtype=np.float32)),
            steps=int(res["steps"]),
            lr=float(res["lr"]),
            device=pack["device"],
        )
        best_pose = opt.get("best_pose_vec", None)
        if best_pose is None:
            res["error"] = "missing_best_pose_vec"
            return res
        best_pose_t = best_pose if hasattr(best_pose, "detach") else torch_mod.as_tensor(best_pose, dtype=torch_mod.float32)
        best_pose_t = best_pose_t.detach().to(device=pack["device"], dtype=torch_mod.float32)
        if best_pose_t.ndim == 1:
            best_pose_t = best_pose_t.unsqueeze(0)
        T_best = urr._pose_vec_to_transform_tensor(best_pose_t).detach().cpu().numpy()[0].astype(np.float32)
        pts_refined = _transform_uvh_points(
            pts_xyz=pts[:, :3],
            T_dst_src=T_best,
        )
        bbox_rf = _bbox_from_transformed_uvh_points(
            pts_xyz=pts_refined,
            frame_wh=frame_wh,
            fallback_bbox=tuple(cand.get("bbox", (0, 0, 0, 0))),
        )
        res["ok"] = bbox_rf is not None
        res["bbox"] = tuple(bbox_rf) if bbox_rf is not None else None
        res["projected_ref_pts"] = np.asarray(pts_refined, dtype=np.float32)
        res["pose_vec"] = best_pose_t.detach().cpu().numpy()[0].astype(np.float32)
        res["best_loss"] = float(opt.get("best_loss", float("nan")))
        if bbox_rf is None:
            res["error"] = "invalid_refined_bbox"
        return res
    except Exception as e:
        res["error"] = f"{e}"
        return res


def _renderfield_common_init_transform(
    *,
    points_xyz: np.ndarray,
    view_lib: ViewLibrary | None,
    prev_view_id: int,
) -> np.ndarray | None:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if int(pts.shape[0]) < 8:
        return None
    mode = str(os.environ.get("MIS_UVH_RENDERFIELD_INIT_MODE", "viewlib")).strip().lower()
    override_view_id = _env_int("MIS_UVH_RENDERFIELD_INIT_VIEW_ID", -1)
    use_view_id = int(override_view_id) if int(override_view_id) >= 0 else int(prev_view_id)
    if str(mode) != "topdown":
        try:
            if (view_lib is not None) and (0 <= int(use_view_id) < int(view_lib.T_w2c.shape[0])):
                return np.asarray(view_lib.T_w2c[int(use_view_id)], dtype=np.float32).reshape(4, 4)
        except Exception:
            pass
    pmin = np.nanmin(pts, axis=0)
    pmax = np.nanmax(pts, axis=0)
    center = 0.5 * (pmin + pmax)
    radius = float(max(1e-3, 0.5 * np.linalg.norm((pmax - pmin).astype(np.float32))))
    dist_mult = float(max(1.0, _env_float("MIS_UVH_RENDERFIELD_INIT_DIST_MULT", 3.0)))
    eye = center + np.asarray([0.0, 0.0, float(dist_mult) * radius], dtype=np.float32)
    return _look_at_T_w2c(eye_w=eye, target_w=center).astype(np.float32)


def _estimate_heightfield_normals(pts: np.ndarray) -> np.ndarray:
    """Estimate per-point unit normals for UVH heightfield points via grid rasterisation.

    The points are assumed to originate from a 2.5-D heightfield z = f(u, v).
    We rasterise them onto a regular 2-D grid, compute finite-difference
    gradients of the height, derive the surface normal (-dz/du, -dz/dv, 1),
    and look up the per-point normal from the nearest grid cell.

    Parameters
    ----------
    pts : (N, 3+) float32 – first three columns are (u, v, h).

    Returns
    -------
    normals : (N, 3) float32 – unit normals.
    """
    pts = np.asarray(pts, dtype=np.float32)
    N = int(pts.shape[0])
    if N < 3:
        out = np.zeros((N, 3), dtype=np.float32)
        out[:, 2] = 1.0
        return out
    u, v, h = pts[:, 0].copy(), pts[:, 1].copy(), pts[:, 2].copy()
    grid_res = int(max(16, min(128, int(np.sqrt(N)))))
    u_mn, u_mx = float(u.min()), float(u.max())
    v_mn, v_mx = float(v.min()), float(v.max())
    u_rng = max(float(u_mx - u_mn), 1e-6)
    v_rng = max(float(v_mx - v_mn), 1e-6)
    gi = np.clip(((u - u_mn) / u_rng * (grid_res - 1)).astype(np.intp), 0, grid_res - 1)
    gj = np.clip(((v - v_mn) / v_rng * (grid_res - 1)).astype(np.intp), 0, grid_res - 1)

    h_sum = np.zeros((grid_res, grid_res), dtype=np.float64)
    h_cnt = np.zeros((grid_res, grid_res), dtype=np.float64)
    np.add.at(h_sum, (gi, gj), h.astype(np.float64))
    np.add.at(h_cnt, (gi, gj), 1.0)
    valid = h_cnt > 0
    hmap = np.zeros((grid_res, grid_res), dtype=np.float32)
    hmap[valid] = (h_sum[valid] / h_cnt[valid]).astype(np.float32)

    du = np.zeros_like(hmap)
    dv = np.zeros_like(hmap)
    du[1:-1, :] = (hmap[2:, :] - hmap[:-2, :]) * 0.5
    du[0, :] = hmap[1, :] - hmap[0, :]
    du[-1, :] = hmap[-1, :] - hmap[-2, :]
    dv[:, 1:-1] = (hmap[:, 2:] - hmap[:, :-2]) * 0.5
    dv[:, 0] = hmap[:, 1] - hmap[:, 0]
    dv[:, -1] = hmap[:, -1] - hmap[:, -2]
    su = float(u_rng) / float(grid_res)
    sv = float(v_rng) / float(grid_res)
    du = du / max(su, 1e-8)
    dv = dv / max(sv, 1e-8)

    nx = -du
    ny = -dv
    nz = np.ones_like(du)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2).clip(min=1e-8)
    nx /= norm
    ny /= norm
    nz /= norm

    normals = np.empty((N, 3), dtype=np.float32)
    normals[:, 0] = nx[gi, gj]
    normals[:, 1] = ny[gi, gj]
    normals[:, 2] = nz[gi, gj]
    return normals


def _ensure_renderfield_point_dim(
    pts: np.ndarray,
    point_in_dim: int,
) -> np.ndarray:
    """Pad / truncate *pts* so that its last dimension equals *point_in_dim*.

    When the model was trained with ``point_in_dim=6`` (xyz + normals) but
    the caller only has ``(N, 3)`` xyz data, this helper estimates heightfield
    normals and concatenates them automatically.
    """
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    cur_dim = int(pts.shape[-1])
    if cur_dim >= int(point_in_dim):
        return pts[:, :int(point_in_dim)].copy()
    if int(point_in_dim) == 6 and cur_dim == 3:
        normals = _estimate_heightfield_normals(pts[:, :3])
        return np.concatenate([pts[:, :3], normals], axis=1).astype(np.float32)
    pad = np.zeros((int(pts.shape[0]), int(point_in_dim) - cur_dim), dtype=np.float32)
    return np.concatenate([pts, pad], axis=1).astype(np.float32)


def _canonicalize_uvh_points(
    *,
    points_xyz: np.ndarray,
) -> dict[str, Any] | None:
    pts = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    if int(pts.shape[0]) < 8:
        return None
    valid = np.all(np.isfinite(pts), axis=1)
    pts = pts[valid]
    if int(pts.shape[0]) < 8:
        return None
    center = np.nanmean(pts, axis=0).astype(np.float32)
    pts0 = (pts - center[None, :]).astype(np.float32)
    scale_mode = str(os.environ.get("MIS_UVH_RENDERFIELD_CANONICAL_SCALE", "rms")).strip().lower()
    if scale_mode == "bboxdiag":
        pmin = np.nanmin(pts0, axis=0)
        pmax = np.nanmax(pts0, axis=0)
        scale = float(np.linalg.norm((pmax - pmin).astype(np.float32)))
    else:
        scale = float(np.sqrt(np.mean(np.sum(pts0.astype(np.float64) ** 2, axis=1))))
    scale = float(max(scale, 1e-3))
    ptsn = (pts0 / float(scale)).astype(np.float32)
    return {
        "points": np.asarray(ptsn, dtype=np.float32),
        "center": np.asarray(center, dtype=np.float32),
        "scale": float(scale),
    }


def _renderfield_compare_render_pair(
    *,
    ref_render: np.ndarray,
    cur_render: np.ndarray,
) -> dict[str, float]:
    ref = np.asarray(ref_render, dtype=np.float32)
    cur = np.asarray(cur_render, dtype=np.float32)
    if ref.ndim == 4:
        ref = ref[0]
    if cur.ndim == 4:
        cur = cur[0]
    if ref.ndim != 3 or cur.ndim != 3 or int(ref.shape[0]) < 4 or int(cur.shape[0]) < 4:
        return {
            "loss": float("inf"),
            "depth_loss": float("inf"),
            "normal_loss": float("inf"),
            "mask_iou": 0.0,
            "valid_ratio": 0.0,
        }
    dep_ref = np.nan_to_num(ref[0], nan=0.0, posinf=0.0, neginf=0.0)
    dep_cur = np.nan_to_num(cur[0], nan=0.0, posinf=0.0, neginf=0.0)
    mk_ref = dep_ref > 1e-6
    mk_cur = dep_cur > 1e-6
    inter = mk_ref & mk_cur
    union = mk_ref | mk_cur
    inter_n = int(np.count_nonzero(inter))
    union_n = int(np.count_nonzero(union))
    if inter_n <= 0 or union_n <= 0:
        return {
            "loss": float("inf"),
            "depth_loss": float("inf"),
            "normal_loss": float("inf"),
            "mask_iou": 0.0,
            "valid_ratio": 0.0,
        }
    depth_loss = float(np.mean(np.abs(dep_cur[inter] - dep_ref[inter])))
    normal_ref = np.clip(ref[1:4] * 2.0 - 1.0, -1.0, 1.0)
    normal_cur = np.clip(cur[1:4] * 2.0 - 1.0, -1.0, 1.0)
    nr_den = np.maximum(np.linalg.norm(normal_ref, axis=0, keepdims=False), 1e-6)
    nc_den = np.maximum(np.linalg.norm(normal_cur, axis=0, keepdims=False), 1e-6)
    normal_ref = normal_ref / nr_den[None, ...]
    normal_cur = normal_cur / nc_den[None, ...]
    ncos = np.clip(np.sum(normal_ref * normal_cur, axis=0), -1.0, 1.0)
    normal_loss = float(np.mean(1.0 - ncos[inter]))
    mask_iou = float(inter_n) / float(max(1, union_n))
    valid_ratio = float(inter_n) / float(max(1, dep_ref.size))
    iou_pen = 1.0 - float(mask_iou)
    total_loss = float(depth_loss + normal_loss + 0.25 * iou_pen)
    return {
        "loss": total_loss,
        "depth_loss": float(depth_loss),
        "normal_loss": float(normal_loss),
        "mask_iou": float(mask_iou),
        "valid_ratio": float(valid_ratio),
    }


def _renderfield_predict_render(
    *,
    pack: dict[str, Any],
    points_xyz: np.ndarray,
    pose_vec: np.ndarray,
) -> np.ndarray:
    torch_mod = pack["torch"]
    upr = pack["upr"]
    render = upr.predict_render_from_pose(
        model=pack["model"],
        points=torch_mod.from_numpy(np.asarray(points_xyz, dtype=np.float32)),
        pose_vec=torch_mod.from_numpy(np.asarray(pose_vec, dtype=np.float32)),
        device=pack["device"],
    )
    return np.asarray(render, dtype=np.float32)


def _renderfield_optimize_pose_to_reference(
    *,
    pack: dict[str, Any],
    points_xyz: np.ndarray,
    ref_render: np.ndarray,
    init_pose_vec: np.ndarray,
) -> dict[str, Any]:
    res = {
        "ok": False,
        "best_loss": float("inf"),
        "best_pose_vec": np.asarray(init_pose_vec, dtype=np.float32).reshape(-1),
        "best_render": None,
        "depth_loss": float("inf"),
        "normal_loss": float("inf"),
        "mask_iou": 0.0,
        "valid_ratio": 0.0,
        "steps": int(max(1, _env_int("MIS_UVH_RENDERFIELD_SEARCH_ITERS", 4))),
        "error": "",
    }
    pts_in = np.asarray(points_xyz, dtype=np.float32)
    if pts_in.ndim == 1:
        pts_in = pts_in.reshape(-1, 3)
    pts = pts_in
    if int(pts.shape[0]) < 80:
        res["error"] = "too_few_cur_points"
        return res
    urr = pack["urr"]
    upr = pack["upr"]
    try:
        best_pose = np.asarray(init_pose_vec, dtype=np.float32).reshape(-1)
        best_render = _renderfield_predict_render(pack=pack, points_xyz=pts, pose_vec=best_pose)
        best_cmp = _renderfield_compare_render_pair(ref_render=ref_render, cur_render=best_render)
        best_loss = float(best_cmp.get("loss", float("inf")))
        rot_step_deg = float(max(0.25, _env_float("MIS_UVH_RENDERFIELD_ROT_STEP_DEG", 10.0)))
        trans_step = float(max(0.25, _env_float("MIS_UVH_RENDERFIELD_TRANS_STEP", 6.0)))
        for _ in range(int(res["steps"])):
            improved = False
            T_best = urr._pose_vec_to_transform_tensor(
                pack["torch"].from_numpy(np.asarray(best_pose, dtype=np.float32)).reshape(1, -1)
            ).detach().cpu().numpy()[0].astype(np.float32)
            candidates: list[np.ndarray] = [np.asarray(T_best, dtype=np.float32)]
            for axis in (
                np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            ):
                for sign in (-1.0, 1.0):
                    delta = np.eye(4, dtype=np.float32)
                    delta[:3, :3] = upr._axis_angle_to_matrix_np(axis, math.radians(float(sign) * float(rot_step_deg)))
                    candidates.append((delta @ T_best).astype(np.float32))
                    delta_t = np.eye(4, dtype=np.float32)
                    delta_t[:3, 3] = axis * float(sign) * float(trans_step)
                    candidates.append((delta_t @ T_best).astype(np.float32))
            for Tc in candidates:
                pose_c = urr._transform_to_pose_vec(np.asarray(Tc, dtype=np.float32).reshape(4, 4))
                render_c = _renderfield_predict_render(pack=pack, points_xyz=pts, pose_vec=pose_c)
                cmp_c = _renderfield_compare_render_pair(ref_render=ref_render, cur_render=render_c)
                loss_c = float(cmp_c.get("loss", float("inf")))
                if loss_c + 1e-6 < float(best_loss):
                    best_loss = float(loss_c)
                    best_pose = np.asarray(pose_c, dtype=np.float32).reshape(-1)
                    best_render = np.asarray(render_c, dtype=np.float32)
                    best_cmp = dict(cmp_c)
                    improved = True
            rot_step_deg = float(max(0.25, rot_step_deg * (0.85 if improved else 0.5)))
            trans_step = float(max(0.25, trans_step * (0.85 if improved else 0.5)))
        res["ok"] = np.isfinite(float(best_loss))
        res["best_loss"] = float(best_loss)
        res["best_pose_vec"] = np.asarray(best_pose, dtype=np.float32).reshape(-1)
        res["best_render"] = np.asarray(best_render, dtype=np.float32) if isinstance(best_render, np.ndarray) else None
        res["depth_loss"] = float(best_cmp.get("depth_loss", float("inf")))
        res["normal_loss"] = float(best_cmp.get("normal_loss", float("inf")))
        res["mask_iou"] = float(best_cmp.get("mask_iou", 0.0))
        res["valid_ratio"] = float(best_cmp.get("valid_ratio", 0.0))
        return res
    except Exception as e:
        res["error"] = f"{e}"
        return res


def _build_renderfield_reference_context(
    *,
    ref_points: np.ndarray,
    view_lib: ViewLibrary | None,
    prev_view_id: int,
    cfg: core.TrackConfig,
) -> dict[str, Any] | None:
    pack = _get_renderfield_refiner(cfg=cfg)
    if not isinstance(pack, dict):
        return None
    pts_raw = np.asarray(ref_points, dtype=np.float32).reshape(-1, 3)
    if int(pts_raw.shape[0]) < 80:
        return None
    canon = _canonicalize_uvh_points(points_xyz=pts_raw)
    if not isinstance(canon, dict):
        return None
    pts3 = np.asarray(canon.get("points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    if int(pts3.shape[0]) < 80:
        return None
    point_in_dim = int(pack.get("point_in_dim", 3))
    pts = _ensure_renderfield_point_dim(pts3, point_in_dim)
    init_T = _renderfield_common_init_transform(
        points_xyz=pts3,
        view_lib=view_lib,
        prev_view_id=int(prev_view_id),
    )
    if init_T is None:
        return None
    try:
        init_pose_vec = pack["urr"]._transform_to_pose_vec(np.asarray(init_T, dtype=np.float32).reshape(4, 4))
        ref_render = _renderfield_predict_render(
            pack=pack,
            points_xyz=pts,
            pose_vec=np.asarray(init_pose_vec, dtype=np.float32),
        )
        return {
            "pack": pack,
            "ref_points_raw": np.asarray(pts_raw, dtype=np.float32),
            "ref_points": np.asarray(pts, dtype=np.float32),
            "ref_center": np.asarray(canon.get("center", np.zeros(3, dtype=np.float32)), dtype=np.float32),
            "ref_scale": float(canon.get("scale", 1.0)),
            "init_T_w2c": np.asarray(init_T, dtype=np.float32).reshape(4, 4),
            "init_pose_vec": np.asarray(init_pose_vec, dtype=np.float32).reshape(-1),
            "ref_render": np.asarray(ref_render, dtype=np.float32),
            "image_size": int(pack.get("image_size", 96)),
            "point_in_dim": point_in_dim,
        }
    except Exception as e:
        _log_once(
            "uvh_rf_refctx_fail",
            f"[UVH-RF][Skip] reference context 初始化失败: {e}",
            cfg=cfg,
            level=1,
        )
        return None


def _match_candidate_with_renderfield_reference(
    *,
    renderfield_ref_ctx: dict[str, Any],
    cand: dict[str, Any],
    cfg: core.TrackConfig,
    frame_wh: tuple[int, int] | None = None,
) -> dict[str, Any]:
    res = {
        "ok": False,
        "best_loss": float("inf"),
        "best_pose_vec": None,
        "best_render": None,
        "delta_pose_vec": None,
        "delta_T_w2c": None,
        "T_cur_ref": None,
        "projected_ref_pts": None,
        "bbox": None,
        "depth_loss": float("inf"),
        "normal_loss": float("inf"),
        "mask_iou": 0.0,
        "valid_ratio": 0.0,
        "sim": 0.0,
        "steps": int(max(1, _env_int("MIS_UVH_RENDERFIELD_SEARCH_ITERS", 4))),
        "error": "",
    }
    pack = renderfield_ref_ctx.get("pack", None)
    if not isinstance(pack, dict):
        res["error"] = "missing_renderfield_pack"
        return res
    try:
        pts_cur = _uvh_points_from_th_mask_roi(
            th_gray_truncated=np.asarray(cand.get("th_gray", None), dtype=np.float32),
            mask_u8=np.asarray(cand.get("mask_u8", None), dtype=np.uint8),
            bbox=tuple(cand.get("bbox", (0, 0, 0, 0))),
            z_scale=float(cfg.mesh_z_scale),
            max_points=int(max(1500, _env_int("MIS_UVH_RENDERFIELD_MAX_POINTS", 48000))),
            rng=np.random.default_rng(0),
            fill_mask=True,
            min_height=float(_env_float("MIS_UVH_RENDERFIELD_MIN_HEIGHT", 1.0)),
            deterministic=True,
        )
    except Exception:
        pts_cur = np.zeros((0, 3), dtype=np.float32)
    if int(np.asarray(pts_cur).shape[0]) < 80:
        res["error"] = "too_few_cur_points"
        return res
    canon_cur = _canonicalize_uvh_points(points_xyz=np.asarray(pts_cur, dtype=np.float32))
    if not isinstance(canon_cur, dict):
        res["error"] = "canonicalize_cur_failed"
        return res
    pts_cur_canon3 = np.asarray(canon_cur.get("points", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32).reshape(-1, 3)
    if int(pts_cur_canon3.shape[0]) < 80:
        res["error"] = "too_few_cur_points_after_canonical"
        return res
    point_in_dim = int(pack.get("point_in_dim", 3))
    pts_cur_canon = _ensure_renderfield_point_dim(pts_cur_canon3, point_in_dim)
    opt = _renderfield_optimize_pose_to_reference(
        pack=pack,
        points_xyz=np.asarray(pts_cur_canon, dtype=np.float32),
        ref_render=np.asarray(renderfield_ref_ctx.get("ref_render", None), dtype=np.float32),
        init_pose_vec=np.asarray(renderfield_ref_ctx.get("init_pose_vec", None), dtype=np.float32),
    )
    res["steps"] = int(opt.get("steps", res["steps"]) or res["steps"])
    res["best_loss"] = float(opt.get("best_loss", float("inf")))
    res["depth_loss"] = float(opt.get("depth_loss", float("inf")))
    res["normal_loss"] = float(opt.get("normal_loss", float("inf")))
    res["mask_iou"] = float(opt.get("mask_iou", 0.0))
    res["valid_ratio"] = float(opt.get("valid_ratio", 0.0))
    res["best_pose_vec"] = opt.get("best_pose_vec", None)
    res["best_render"] = opt.get("best_render", None)
    res["error"] = str(opt.get("error", ""))
    loss_scale = float(max(0.1, _env_float("MIS_UVH_RENDERFIELD_LOSS_SCALE", 4.0)))
    if np.isfinite(float(res["best_loss"])):
        res["sim"] = float(1.0 / (1.0 + loss_scale * float(res["best_loss"])))
    try:
        best_pose_vec = np.asarray(res["best_pose_vec"], dtype=np.float32).reshape(-1)
        urr = pack["urr"]
        torch_mod = pack["torch"]
        T_best = urr._pose_vec_to_transform_tensor(torch_mod.from_numpy(best_pose_vec).reshape(1, -1)).detach().cpu().numpy()[0].astype(np.float32)
        T_init = np.asarray(renderfield_ref_ctx.get("init_T_w2c", np.eye(4, dtype=np.float32)), dtype=np.float32).reshape(4, 4)
        delta_T = np.asarray(T_best @ np.linalg.inv(T_init), dtype=np.float32).reshape(4, 4)
        res["delta_T_w2c"] = delta_T
        res["delta_pose_vec"] = np.asarray(urr._transform_to_pose_vec(delta_T), dtype=np.float32).reshape(-1)
    except Exception:
        pass
    res["ok"] = bool(opt.get("ok", False)) and np.isfinite(float(res["best_loss"]))
    return res


def _parse_float_list(env_name: str, default_csv: str) -> list[float]:
    s = (os.environ.get(env_name, default_csv) or "").strip()
    if not s:
        s = str(default_csv)
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    out: list[float] = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            continue
    return out


_LOG_ONCE_KEYS: set[str] = set()


def _log(msg: str, *, cfg: Any | None = None, level: int = 1) -> None:
    """
    控制台日志（避免刷屏）：
    - level=1：关键环节
    - level=2：更详细的 debug
    使用 cfg.verbose（来自 core.TrackConfig/MIS_VERBOSE）控制输出等级。
    """
    try:
        v = int(getattr(cfg, "verbose", _env_int("MIS_VERBOSE", 1))) if cfg is not None else int(_env_int("MIS_VERBOSE", 1))
        if v >= int(level):
            print(str(msg), flush=True)
    except Exception:
        print(str(msg), flush=True)


def _log_once(key: str, msg: str, *, cfg: Any | None = None, level: int = 1) -> None:
    try:
        k = str(key)
    except Exception:
        k = "key"
    if k in _LOG_ONCE_KEYS:
        return
    _LOG_ONCE_KEYS.add(k)
    _log(msg, cfg=cfg, level=int(level))


def _fibonacci_hemisphere_dirs(n: int, *, z_bias: float = 1.0, theta_offset: float = 0.0) -> np.ndarray:
    """
    Fibonacci 上半球采样方向（z>=0），返回 (n,3) 单位向量。
    - z_bias=1.0：近似均匀面积分布（默认）
    - z_bias>1.0：更偏向 z→0（更“贴近地平线/大倾角”）
    - z_bias<1.0：更偏向 z→1（更“俯视/小倾角”）
    """
    n = int(max(0, n))
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    # uniform in z => uniform area distribution on sphere surface bands
    i = np.arange(n, dtype=np.float64)
    z = (i + 0.5) / float(n)  # (0,1)
    try:
        zb = float(z_bias)
        if not np.isfinite(zb) or zb <= 0:
            zb = 1.0
    except Exception:
        zb = 1.0
    if abs(float(zb) - 1.0) > 1e-9:
        z = np.clip(z, 1e-9, 1.0) ** float(zb)
    golden = (1.0 + math.sqrt(5.0)) * 0.5
    theta = 2.0 * math.pi * (i / golden) + float(theta_offset)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    dirs = np.stack([x, y, z], axis=1).astype(np.float32)
    # normalize (numerical safety)
    den = np.linalg.norm(dirs, axis=1, keepdims=True)
    den = np.where(den > 1e-9, den, 1.0)
    return (dirs / den).astype(np.float32)


def _look_at_T_w2c(
    *,
    eye_w: np.ndarray,
    target_w: np.ndarray,
    up_w: np.ndarray | None = None,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """
    构造世界到相机的外参 T_w2c（4x4）：
      p_c = R_cw (p_w - eye)
    其中相机坐标约定：
      - z 轴为“朝向 target 的 forward”
      - x 轴为 right
      - y 轴为 up
    """
    eye = np.asarray(eye_w, dtype=np.float64).reshape(3)
    tgt = np.asarray(target_w, dtype=np.float64).reshape(3)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up_w is None else np.asarray(up_w, dtype=np.float64).reshape(3)

    f = (tgt - eye).astype(np.float64)
    fn = float(np.linalg.norm(f))
    if fn <= 1e-9:
        f = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        f = f / fn

    r = np.cross(f, up).astype(np.float64)
    rn = float(np.linalg.norm(r))
    if rn <= 1e-9:
        # f 与 up 近共线时，换一个 up
        up2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        r = np.cross(f, up2).astype(np.float64)
        rn = float(np.linalg.norm(r))
    if rn <= 1e-9:
        r = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        r = r / rn

    u = np.cross(r, f).astype(np.float64)
    un = float(np.linalg.norm(u))
    if un > 1e-9:
        u = u / un

    # roll around forward axis (f): rotate (r,u) plane
    try:
        rd = float(roll_deg)
    except Exception:
        rd = 0.0
    if np.isfinite(rd) and abs(rd) > 1e-9:
        a = float(rd) * math.pi / 180.0
        ca = float(math.cos(a))
        sa = float(math.sin(a))
        r2 = (ca * r + sa * u).astype(np.float64)
        u2 = (-sa * r + ca * u).astype(np.float64)
        r, u = r2, u2

    # columns are camera axes in world (x=right, y=up, z=forward)
    R_wc = np.stack([r, u, f], axis=1).astype(np.float64)  # 3x3
    R_cw = R_wc.T
    t = (-R_cw @ eye).astype(np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_cw
    T[:3, 3] = t
    return T


def _default_K_for_library(image_wh: tuple[int, int]) -> np.ndarray:
    w, h = int(image_wh[0]), int(image_wh[1])
    # 经验：用较窄 FOV 让“轮廓形状”更稳定；尺度变化主要靠距离 dist
    fx = float(max(16.0, 0.90 * w))
    fy = float(max(16.0, 0.90 * h))
    cx = (w - 1) * 0.5
    cy = (h - 1) * 0.5
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return K


def _uvh_topdown_camera_from_points(
    *,
    pts_w: np.ndarray,
    image_wh: tuple[int, int],
    render_dist_mult: float | None = None,
) -> dict[str, Any]:
    """
    uvh_render **无 view_lib** 时与 ``find_best_bbox_in_next_frame_uvh_render`` 一致的相机初始化：
    - 世界系为整帧像素 (u,v) + 伪高度 z（uvh 点云）
    - 用点云 AABB 中心 ``center``、半对角线尺度 ``radius``
    - 相机光心 ``eye = center + (0,0,1) * (render_dist_mult * radius)``（沿 +Z 后退）
    - ``T_w2c = _look_at_T_w2c(eye, center)``：相机 z 轴指向 target
    - 内参 ``K`` 来自 ``_default_K_for_library(image_wh)``
    """
    pts = np.asarray(pts_w, dtype=np.float32).reshape(-1, 3)
    dm = float(render_dist_mult) if render_dist_mult is not None else float(_env_float("MIS_UVH_RENDER_DIST_MULT", 3.0))
    dm = float(max(1e-3, dm))
    if int(pts.shape[0]) < 1:
        raise ValueError("empty pts_w")
    pmin = np.nanmin(pts, axis=0)
    pmax = np.nanmax(pts, axis=0)
    center = (0.5 * (pmin + pmax)).astype(np.float64)
    radius = float(0.5 * np.linalg.norm((pmax - pmin).astype(np.float64)))
    radius = float(max(1e-3, radius))
    dist = float(max(1e-3, dm * radius))
    eye = (center + np.array([0.0, 0.0, 1.0], dtype=np.float64) * dist).astype(np.float64)
    T_w2c = np.asarray(_look_at_T_w2c(eye_w=eye, target_w=center), dtype=np.float64)
    K = np.asarray(_default_K_for_library((int(image_wh[0]), int(image_wh[1]))), dtype=np.float32)
    return {
        "image_wh": (int(image_wh[0]), int(image_wh[1])),
        "render_dist_mult": float(dm),
        "pmin": pmin.astype(np.float32),
        "pmax": pmax.astype(np.float32),
        "center": center.astype(np.float32),
        "radius": float(radius),
        "dist_camera": float(dist),
        "eye": eye.astype(np.float32),
        "K": K,
        "T_w2c": T_w2c.astype(np.float32),
    }


def _save_uvh_camera_init_debug(
    *,
    out_root: str,
    cfg: core.TrackConfig,
    prev_uvh_pts: np.ndarray,
    bbox_ref: tuple[int, int, int, int],
    lib: ViewLibrary | None,
    uvh_viewlib_enable: bool,
    prev_view_id: int,
    renderfield_ref_ctx: dict[str, Any] | None,
) -> str:
    """
    将「相机初始化」相关中间量写入 ``<out_root>/camera_init/``，便于离线查看。
    默认开启；设 ``MIS_SAVE_CAMERA_INIT_DEBUG=0`` 可关闭。
    """
    if not bool(_env_bool("MIS_SAVE_CAMERA_INIT_DEBUG", True)):
        return ""
    cam_dir = _ensure_dir(os.path.join(out_root, "camera_init"))
    meta: dict[str, Any] = {
        "time": _now(),
        "note_zh": (
            "uvh_render 默认（无 view_lib）：俯视类相机由参考点云 AABB 与 MIS_UVH_RENDER_DIST_MULT 决定 eye，"
            "再用 _look_at_T_w2c 得到 T_w2c。有 view_lib 时对齐/渲染用的外参取自目标库 T_w2c[view_id]（每视点预生成）。"
        ),
        "semantic_image_size": [int(cfg.semantic_image_size[0]), int(cfg.semantic_image_size[1])],
        "bbox_ref": [int(v) for v in bbox_ref],
        "mesh_z_scale": float(cfg.mesh_z_scale),
    }
    try:
        wh = (int(cfg.semantic_image_size[0]), int(cfg.semantic_image_size[1]))
        pack = _uvh_topdown_camera_from_points(pts_w=prev_uvh_pts, image_wh=wh, render_dist_mult=None)
        np.savez_compressed(
            os.path.join(cam_dir, "uvh_topdown_camera.npz"),
            K=pack["K"],
            T_w2c=pack["T_w2c"],
            center=pack["center"],
            eye=pack["eye"],
            pmin=pack["pmin"],
            pmax=pack["pmax"],
            radius=np.asarray([pack["radius"]], dtype=np.float32),
            dist_camera=np.asarray([pack["dist_camera"]], dtype=np.float32),
            render_dist_mult=np.asarray([pack["render_dist_mult"]], dtype=np.float32),
        )
        meta["uvh_topdown"] = {
            "image_wh": list(pack["image_wh"]),
            "render_dist_mult": float(pack["render_dist_mult"]),
            "radius": float(pack["radius"]),
            "dist_camera": float(pack["dist_camera"]),
            "center": pack["center"].reshape(-1).tolist(),
            "eye": pack["eye"].reshape(-1).tolist(),
            "pmin": pack["pmin"].reshape(-1).tolist(),
            "pmax": pack["pmax"].reshape(-1).tolist(),
            "K": pack["K"].reshape(-1).tolist(),
            "T_w2c_rowmajor": pack["T_w2c"].reshape(-1).tolist(),
            "files": {"npz": "uvh_topdown_camera.npz"},
        }
    except Exception as e:
        meta["uvh_topdown_error"] = str(e)

    meta["viewlib_enabled_for_uvh"] = int(bool(lib is not None and uvh_viewlib_enable))
    if lib is not None and 0 <= int(prev_view_id) < int(lib.ids.size):
        try:
            Tw = np.asarray(lib.T_w2c[int(prev_view_id)], dtype=np.float32).reshape(4, 4)
            d = np.asarray(lib.dirs[int(prev_view_id)], dtype=np.float32).reshape(-1)
            np.save(os.path.join(cam_dir, "viewlib_prev_T_w2c.npy"), Tw)
            meta["viewlib_prev_view"] = {
                "prev_view_id": int(prev_view_id),
                "dir": d.tolist(),
                "dist": float(lib.dists[int(prev_view_id)]) if int(lib.dists.shape[0]) > int(prev_view_id) else float("nan"),
                "roll_deg": float(lib.rolls[int(prev_view_id)]) if int(lib.rolls.shape[0]) > int(prev_view_id) else float("nan"),
                "T_w2c_rowmajor": Tw.reshape(-1).tolist(),
                "file_npy": "viewlib_prev_T_w2c.npy",
            }
        except Exception as e:
            meta["viewlib_prev_view_error"] = str(e)

    if isinstance(renderfield_ref_ctx, dict):
        try:
            init_T = np.asarray(renderfield_ref_ctx.get("init_T_w2c", None), dtype=np.float32).reshape(4, 4)
            np.save(os.path.join(cam_dir, "renderfield_init_T_w2c.npy"), init_T)
            meta["renderfield"] = {
                "init_T_w2c_rowmajor": init_T.reshape(-1).tolist(),
                "ref_scale": float(renderfield_ref_ctx.get("ref_scale", float("nan"))),
                "ref_center": np.asarray(renderfield_ref_ctx.get("ref_center", np.zeros(3)), dtype=np.float32).reshape(-1).tolist(),
                "image_size": int(renderfield_ref_ctx.get("image_size", 0)),
                "file_npy": "renderfield_init_T_w2c.npy",
            }
        except Exception as e:
            meta["renderfield_error"] = str(e)

    try:
        p = os.path.join(cam_dir, "camera_init_meta.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return cam_dir


def _uvh_points_from_th_roi(
    *,
    th_gray_truncated: np.ndarray,
    bbox: tuple[int, int, int, int],
    z_scale: float,
    max_points: int,
    rng: np.random.Generator,
    deterministic: bool = True,
) -> np.ndarray:
    """从 ROI 的 th_gray_truncated 生成 uvh 点云（整帧坐标系）。"""
    # 核心优化：智能缓存（基于 bbox 和 th_gray 的哈希）
    cache_key = None
    if deterministic:
        # 使用 bbox + th_gray 的简单哈希作为缓存键
        th_hash = int(np.sum(th_gray_truncated) * 1000) % 10000000 if th_gray_truncated.size > 0 else 0
        cache_key = (bbox, th_hash, z_scale, max_points)
        
        # 检查缓存
        if hasattr(_uvh_points_from_th_roi, '_cache') and cache_key in _uvh_points_from_th_roi._cache:
            return _uvh_points_from_th_roi._cache[cache_key].copy()
    
    bx, by, w, h = [int(v) for v in bbox]
    th = np.asarray(th_gray_truncated, dtype=np.float32)
    if th.ndim != 2:
        return np.zeros((0, 3), dtype=np.float32)
    if th.shape[0] != h or th.shape[1] != w:
        th = cv2.resize(th, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    m = th > 0
    ys, xs = np.nonzero(m)
    if xs.size <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    z = (th[ys, xs] / 255.0) * float(z_scale)
    u = xs.astype(np.float32) + float(bx)
    v = ys.astype(np.float32) + float(by)
    pts = np.stack([u, v, z.astype(np.float32)], axis=1).astype(np.float32)
    n = int(pts.shape[0])
    mp = int(max(0, max_points))
    if mp > 0 and n > mp:
        if deterministic:
            try:
                idx = np.linspace(0, n - 1, num=int(mp), dtype=np.int32)
                pts = pts[idx]
            except Exception:
                pts = pts[: int(mp)]
        else:
            try:
                idx = rng.choice(n, size=int(mp), replace=False)
                pts = pts[idx]
            except Exception:
                pts = pts[: int(mp)]
    
    # 核心优化：缓存结果
    if cache_key is not None:
        if not hasattr(_uvh_points_from_th_roi, '_cache'):
            _uvh_points_from_th_roi._cache = {}
        # 限制缓存大小（LRU 策略）
        if len(_uvh_points_from_th_roi._cache) > 64:
            # 删除最旧的 32 个条目
            keys_to_remove = list(_uvh_points_from_th_roi._cache.keys())[:32]
            for k in keys_to_remove:
                del _uvh_points_from_th_roi._cache[k]
        _uvh_points_from_th_roi._cache[cache_key] = pts.copy()
    
    return pts


def _mesh_to_points(
    mesh: Any,
    *,
    max_points: int,
    rng: np.random.Generator,
    z_eps: float = 1e-6,
    deterministic: bool = True,
) -> np.ndarray:
    """从 open3d TriangleMesh 取点（默认用 vertices），并做随机下采样。"""
    try:
        v = np.asarray(getattr(mesh, "vertices", np.zeros((0, 3))), dtype=np.float32).reshape(-1, 3)
    except Exception:
        v = np.zeros((0, 3), dtype=np.float32)
    if v.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    m = np.isfinite(v).all(axis=1)
    if float(z_eps) > 0:
        m = m & (v[:, 2] > float(z_eps))
    v = v[m]
    if v.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    mp = int(max(0, int(max_points)))
    if mp > 0 and int(v.shape[0]) > mp:
        n = int(v.shape[0])
        if deterministic:
            try:
                idx = np.linspace(0, n - 1, num=int(mp), dtype=np.int32)
                v = v[idx]
            except Exception:
                v = v[: int(mp)]
        else:
            try:
                idx = rng.choice(n, size=int(mp), replace=False)
                v = v[idx]
            except Exception:
                v = v[: int(mp)]
    return v.astype(np.float32, copy=False)


def _render_mask_depth_from_points(
    *,
    pts_w: np.ndarray,
    T_w2c: np.ndarray,
    K: np.ndarray,
    image_wh: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    简易“投影渲染”：
      - 输入世界点云 pts_w (N,3)
      - 输出 mask_u8 (h,w) / depth (h,w, float32, inf 表示背景)
    """
    w, h = int(image_wh[0]), int(image_wh[1])
    pts = np.asarray(pts_w, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        return np.zeros((h, w), dtype=np.uint8), np.full((h, w), np.inf, dtype=np.float32)

    T = np.asarray(T_w2c, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    pc = (R @ pts.T).T + t[None, :]
    z = pc[:, 2].astype(np.float64)
    m = np.all(np.isfinite(pc), axis=1) & np.isfinite(z) & (z > 1e-6)
    if not np.any(m):
        return np.zeros((h, w), dtype=np.uint8), np.full((h, w), np.inf, dtype=np.float32)

    pc = pc[m]
    z = z[m]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = pc[:, 0] / z
    y = pc[:, 1] / z
    u = fx * x + cx
    v = fy * (-y) + cy  # y-up -> v-down
    uv_ok = np.isfinite(u) & np.isfinite(v)
    if not np.any(uv_ok):
        return np.zeros((h, w), dtype=np.uint8), np.full((h, w), np.inf, dtype=np.float32)
    u = u[uv_ok]
    v = v[uv_ok]
    z = z[uv_ok]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui = ui[inb]
    vi = vi[inb]
    zf = z[inb].astype(np.float32)
    if ui.size <= 0:
        return np.zeros((h, w), dtype=np.uint8), np.full((h, w), np.inf, dtype=np.float32)

    depth = np.full((h, w), np.inf, dtype=np.float32)
    np.minimum.at(depth, (vi, ui), zf)
    mask = (np.isfinite(depth)).astype(np.uint8) * 255
    return mask, depth


def _mask_to_contour_rgb(mask_u8: np.ndarray, image_wh: tuple[int, int], *, thickness: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """将 mask 转为 contour_rgb，并返回一个“更稳定的填充 mask”（便于 debug）。"""
    w, h = int(image_wh[0]), int(image_wh[1])
    mk = np.asarray(mask_u8, dtype=np.uint8)
    if mk.ndim != 2:
        mk = np.zeros((h, w), dtype=np.uint8)
    if mk.shape[0] != h or mk.shape[1] != w:
        mk = cv2.resize(mk, (w, h), interpolation=cv2.INTER_NEAREST)

    # 形态学增强连通性（点云投影会有小孔）
    try:
        k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mk2 = cv2.dilate(mk, k1, iterations=1)
        mk2 = cv2.morphologyEx(mk2, cv2.MORPH_CLOSE, k2, iterations=2)
    except Exception:
        mk2 = mk

    cnt = core.extract_largest_contour(mk2)
    if cnt is None or len(cnt) < 3:
        return np.zeros((h, w, 3), dtype=np.uint8), mk2

    filled = np.zeros((h, w), dtype=np.uint8)
    try:
        cv2.drawContours(filled, [cnt], -1, 255, thickness=-1)
    except Exception:
        filled = mk2

    cnt2 = core.extract_largest_contour(filled)
    if cnt2 is None or len(cnt2) < 3:
        return np.zeros((h, w, 3), dtype=np.uint8), filled

    rgb = core.contour_to_rgb(cnt2, (w, h), thickness=int(max(1, thickness)))
    return rgb, filled


def _extract_contour_prefer_center(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    """
    在 ROI 内提取“更像目标”的轮廓：不是只取最大面积，而是用“面积 + 距离 ROI 中心”的组合打分，
    能显著减少 ROI 内背景边缘/大轮廓导致的漂移。
    """
    mk = np.asarray(mask_u8, dtype=np.uint8)
    if mk.ndim != 2 or mk.size <= 0:
        return None
    cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    H, W = mk.shape[:2]
    cx0 = 0.5 * float(W - 1)
    cy0 = 0.5 * float(H - 1)
    norm = float(math.hypot(cx0, cy0) + 1e-6)
    area_min = float(_env_float("MIS_CONTOUR_AREA_MIN", 5.0))
    alpha = float(_env_float("MIS_CONTOUR_CENTER_ALPHA", 12.0))  # 越大越偏向中心

    best = None
    best_s = -1.0
    for c in cnts:
        try:
            a = float(cv2.contourArea(c))
        except Exception:
            continue
        if a <= area_min:
            continue
        try:
            M = cv2.moments(c)
            if abs(float(M.get("m00", 0.0))) > 1e-9:
                cx = float(M.get("m10", 0.0)) / float(M.get("m00", 1.0))
                cy = float(M.get("m01", 0.0)) / float(M.get("m00", 1.0))
            else:
                x, y, w, h = cv2.boundingRect(c)
                cx = float(x) + 0.5 * float(w)
                cy = float(y) + 0.5 * float(h)
        except Exception:
            x, y, w, h = cv2.boundingRect(c)
            cx = float(x) + 0.5 * float(w)
            cy = float(y) + 0.5 * float(h)

        d = float(math.hypot(cx - cx0, cy - cy0) / norm)  # 0~1
        # 组合打分：面积越大越好，离中心越近越好
        s = float(a) / float(1.0 + alpha * d)
        if s > best_s:
            best_s = float(s)
            best = c
    if best is None:
        return None
    if float(cv2.contourArea(best)) <= area_min:
        return None
    return best


def _mask_to_filled_u8(mask_u8: np.ndarray) -> np.ndarray:
    """将二值 mask 转为“填充后的目标区域 mask”（尽量保持单连通、填洞）。"""
    mk = np.asarray(mask_u8, dtype=np.uint8)
    if mk.ndim != 2 or mk.size <= 0:
        return np.zeros((1, 1), dtype=np.uint8)
    mk2 = (mk > 0).astype(np.uint8) * 255
    try:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mk2 = cv2.morphologyEx(mk2, cv2.MORPH_CLOSE, k, iterations=2)
    except Exception:
        pass
    try:
        cnt = _extract_contour_prefer_center(mk2)
        if cnt is None:
            cnt = core.extract_largest_contour(mk2)
        filled = np.zeros_like(mk2, dtype=np.uint8)
        if cnt is not None and len(cnt) >= 3:
            cv2.drawContours(filled, [cnt], -1, 255, thickness=-1)
            return filled
    except Exception:
        pass
    return mk2


def _mask64_from_mask_u8(mask_u8: np.ndarray) -> np.ndarray:
    mk = np.asarray(mask_u8, dtype=np.uint8)
    if mk.ndim != 2 or mk.size <= 0:
        return np.zeros((64, 64), dtype=np.uint8)
    ms = cv2.resize(mk, (64, 64), interpolation=cv2.INTER_NEAREST)
    ms = _mask_to_filled_u8(ms)
    if ms.shape != (64, 64):
        ms = cv2.resize(ms, (64, 64), interpolation=cv2.INTER_NEAREST)
    return ms.astype(np.uint8, copy=False)


def _quick_mask64_from_roi(roi_bgr: np.ndarray, roi_wh: tuple[int, int], cfg: core.TrackConfig) -> np.ndarray:
    """
    为 quick 粗筛构造 64x64 mask，保持与滑窗候选的 quick 流程一致（先 resize ROI 再做 th_68）。
    """
    ww, hh = int(roi_wh[0]), int(roi_wh[1])
    qs = float(max(0.1, min(1.0, float(cfg.quick_scale))))
    qw = int(max(16, round(float(ww) * qs)))
    qh = int(max(16, round(float(hh) * qs)))
    qmin = int(max(16, int(cfg.quick_min_size)))
    if min(qw, qh) < qmin:
        s2 = float(qmin) / float(max(1, min(qw, qh)))
        qw = int(round(float(qw) * s2))
        qh = int(round(float(qh) * s2))
    roi_small = cv2.resize(np.asarray(roi_bgr, dtype=np.uint8), (int(qw), int(qh)), interpolation=cv2.INTER_AREA)
    _, mask_small_u8, _ = core.compute_th_gray_truncated(roi_small, cfg.th_percent, cfg)
    ms = cv2.resize(np.asarray(mask_small_u8, dtype=np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST)
    return ms


def _pca_angle_sigma_xy(xy: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
    """
    返回 (angle_rad, sigma_rms)，用于估计 in-plane 旋转与尺度。
    - angle_rad: 第一主轴方向角
    - sigma_rms: sqrt(E[||x-mean||^2])，代表“尺度”
    """
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if int(pts.shape[0]) < 10:
        return 0.0, 1.0
    if w is not None:
        ww = np.asarray(w, dtype=np.float64).reshape(-1)
        if int(ww.size) == int(pts.shape[0]):
            ww = np.clip(ww, 0.0, np.inf)
        else:
            ww = None
    else:
        ww = None
    if ww is None:
        mu = np.nanmean(pts, axis=0)
        d = pts - mu[None, :]
        d2 = np.sum(d * d, axis=1)
        sigma = float(math.sqrt(float(np.nanmean(d2)) + 1e-12))
        C = (d.T @ d) / float(max(1, pts.shape[0]))
    else:
        s = float(np.nansum(ww))
        if (not np.isfinite(s)) or s <= 1e-9:
            mu = np.nanmean(pts, axis=0)
            d = pts - mu[None, :]
            d2 = np.sum(d * d, axis=1)
            sigma = float(math.sqrt(float(np.nanmean(d2)) + 1e-12))
            C = (d.T @ d) / float(max(1, pts.shape[0]))
        else:
            wn = ww / s
            mu = np.nansum(pts * wn[:, None], axis=0)
            d = pts - mu[None, :]
            d2 = np.sum(d * d, axis=1)
            sigma = float(math.sqrt(float(np.nansum(d2 * wn)) + 1e-12))
            C = (d.T * wn[None, :]) @ d
    C = np.asarray(C, dtype=np.float64).reshape(2, 2)
    if not np.isfinite(C).all():
        return 0.0, float(max(1e-3, sigma))
    try:
        vals, vecs = np.linalg.eigh(C)
        j = int(np.argmax(vals))
        v = vecs[:, j]
        ang = float(math.atan2(float(v[1]), float(v[0])))
    except Exception:
        ang = 0.0
    return float(ang), float(max(1e-3, sigma))


def _uvh_iou_u8(a_u8: np.ndarray, b_u8: np.ndarray) -> float:
    aa = np.asarray(a_u8, dtype=np.uint8)
    bb = np.asarray(b_u8, dtype=np.uint8)
    if aa.shape == bb.shape:
        fn = _uvh_iou_u8_numba_fn()
        if fn is not None:
            try:
                return float(fn(aa, bb))
            except Exception:
                pass
    a = aa > 0
    b = bb > 0
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    if union <= 0:
        return 0.0
    return float(inter) / float(union + 1e-6)


def _uvh_depth_corr01(depth_a: np.ndarray, depth_b: np.ndarray) -> float:
    da = np.asarray(depth_a, dtype=np.float32)
    db = np.asarray(depth_b, dtype=np.float32)
    if da.shape != db.shape or da.size <= 0:
        return 0.0
    m = np.isfinite(da) & np.isfinite(db)
    n = int(np.count_nonzero(m))
    if n < 64:
        return 0.0
    aa = da[m].astype(np.float32, copy=False)
    bb = db[m].astype(np.float32, copy=False)
    am = float(np.mean(aa))
    bm = float(np.mean(bb))
    a0 = aa - am
    b0 = bb - bm
    av = float(np.mean(a0 * a0))
    bv = float(np.mean(b0 * b0))
    if av <= 1e-10 or bv <= 1e-10:
        return 0.0
    corr = float(np.mean(a0 * b0) / float(math.sqrt(av * bv) + 1e-12))
    corr = max(-1.0, min(1.0, corr))
    return float(0.5 * (corr + 1.0))


def _uvh_render_sim_from_mask_depth(
    *,
    ref_filled_u8: np.ndarray,
    ref_depth: np.ndarray,
    cur_filled_u8: np.ndarray,
    cur_depth: np.ndarray,
) -> tuple[float, float, float]:
    """
    返回 (sim, iou, depth_corr01)
    - sim: IoU + depth 相关性融合（更贴近“伪3D mesh”匹配）
    """
    w_iou = float(_env_float("MIS_UVH_SIM_W_IOU", 0.65))
    w_dep = float(_env_float("MIS_UVH_SIM_W_DEPTHCORR", 0.35))
    wsum = float(abs(w_iou) + abs(w_dep))
    if wsum <= 1e-9:
        w_iou, w_dep = 0.65, 0.35
        wsum = 1.0
    w_iou = float(w_iou) / wsum
    w_dep = float(w_dep) / wsum
    iou = float(_uvh_iou_u8(ref_filled_u8, cur_filled_u8))
    dc = float(_uvh_depth_corr01(ref_depth, cur_depth))
    sim = float(w_iou * iou + w_dep * dc)
    sim = 0.0 if (not np.isfinite(sim)) else float(max(0.0, min(1.0, sim)))
    return float(sim), float(iou), float(dc)


def _align_uvh_points_iterative(
    *,
    pts_ref: np.ndarray,
    pts_cur: np.ndarray,
    cam_T_w2c: np.ndarray,
    cam_K: np.ndarray,
    image_wh: tuple[int, int],
    ref_filled_u8: np.ndarray,
    ref_depth: np.ndarray,
) -> dict[str, Any]:
    """
    轻量“迭代优化”(coordinate descent) 的伪3D对齐：
    优化参数：in-plane scale/rotation/translation，使渲染 mask+depth 相似度最大。
    """
    xy_ref = np.asarray(pts_ref, dtype=np.float32).reshape(-1, 3)[:, :2]
    xy_cur = np.asarray(pts_cur, dtype=np.float32).reshape(-1, 3)[:, :2]
    z_cur = np.asarray(pts_cur, dtype=np.float32).reshape(-1, 3)[:, 2:3]
    if int(xy_ref.shape[0]) < 50 or int(xy_cur.shape[0]) < 50:
        return {"sim": 0.0, "iou": 0.0, "depth_corr": 0.0, "scale": 1.0, "rot_deg": 0.0, "tx": 0.0, "ty": 0.0, "render_rgb": None, "render_depth": None}

    # 参数范围
    rot_max = float(_env_float("MIS_UVH_ALIGN_ROT_MAX_DEG", 48.0))
    scl_min = float(_env_float("MIS_UVH_ALIGN_SCALE_MIN", 0.68))
    scl_max = float(_env_float("MIS_UVH_ALIGN_SCALE_MAX", 1.52))
    tr_max = float(_env_float("MIS_UVH_ALIGN_TRANS_MAX_PX", 26.0))

    # 初始估计：PCA 方向 + RMS 尺度 + 重心平移
    w_ref = np.asarray(pts_ref, dtype=np.float32).reshape(-1, 3)[:, 2]
    w_cur = np.asarray(pts_cur, dtype=np.float32).reshape(-1, 3)[:, 2]
    a_ref, s_ref = _pca_angle_sigma_xy(xy_ref, w_ref)
    a_cur, s_cur = _pca_angle_sigma_xy(xy_cur, w_cur)
    rot0 = float((a_ref - a_cur) * 180.0 / math.pi)
    # wrap to [-180,180]
    while rot0 > 180.0:
        rot0 -= 360.0
    while rot0 < -180.0:
        rot0 += 360.0
    rot0 = float(max(-rot_max, min(rot_max, rot0)))
    scale0 = float(s_ref / max(1e-6, s_cur))
    scale0 = float(max(scl_min, min(scl_max, scale0)))
    c_ref = np.nanmean(xy_ref.astype(np.float64), axis=0).astype(np.float64)
    c_cur = np.nanmean(xy_cur.astype(np.float64), axis=0).astype(np.float64)

    iters = int(max(0, _env_int("MIS_UVH_ALIGN_ITERS", 3)))
    rot_step = float(max(0.1, _env_float("MIS_UVH_ALIGN_ROT_STEP_DEG", 15.0)))
    scl_step = float(max(0.001, _env_float("MIS_UVH_ALIGN_SCALE_STEP", 0.12)))
    tr_step = float(max(0.1, _env_float("MIS_UVH_ALIGN_TRANS_STEP_PX", 8.0)))
    shrink = float(max(0.25, min(0.95, _env_float("MIS_UVH_ALIGN_SHRINK", 0.60))))

    def _eval(scale: float, rot_deg: float, tx: float, ty: float) -> dict[str, Any]:
        scale = float(max(scl_min, min(scl_max, scale)))
        rot_deg = float(max(-rot_max, min(rot_max, rot_deg)))
        tx = float(max(-tr_max, min(tr_max, tx)))
        ty = float(max(-tr_max, min(tr_max, ty)))
        th = float(rot_deg) * math.pi / 180.0
        ct = float(math.cos(th))
        st = float(math.sin(th))
        R = np.array([[ct, -st], [st, ct]], dtype=np.float64)
        xy = (xy_cur.astype(np.float64) - c_cur[None, :]) @ R.T
        xy = float(scale) * xy + c_ref[None, :]
        xy[:, 0] = xy[:, 0] + float(tx)
        xy[:, 1] = xy[:, 1] + float(ty)
        pts_al = np.concatenate([xy.astype(np.float32), z_cur.astype(np.float32)], axis=1)
        mk, depth = _render_mask_depth_from_points(pts_w=pts_al, T_w2c=cam_T_w2c, K=cam_K, image_wh=image_wh)
        rgb, filled = _mask_to_contour_rgb(mk, image_wh, thickness=2)
        sim, iou, dc = _uvh_render_sim_from_mask_depth(ref_filled_u8=ref_filled_u8, ref_depth=ref_depth, cur_filled_u8=filled, cur_depth=depth)
        return {
            "sim": float(sim),
            "iou": float(iou),
            "depth_corr": float(dc),
            "scale": float(scale),
            "rot_deg": float(rot_deg),
            "tx": float(tx),
            "ty": float(ty),
            "render_rgb": rgb,
            "render_depth": depth,
            "render_filled": filled,
        }

    best = _eval(scale0, rot0, 0.0, 0.0)
    if iters <= 0:
        return best

    scale = float(best["scale"])
    rot_deg = float(best["rot_deg"])
    tx = float(best["tx"])
    ty = float(best["ty"])
    
    # 核心优化：增强迭代 Early Stop 机制
    early_stop_d = float(max(0.0, _env_float("MIS_UVH_ALIGN_EARLY_STOP_SIM_DELTA", 0.003)))
    min_converged_iters = int(_env_int("MIS_UVH_ALIGN_MIN_CONVERGED_ITERS", 2))  # 连续收敛次数
    consecutive_converged = 0  # 连续收敛计数
    
    for iter_idx in range(int(iters)):
        sim_before_iter = float(best["sim"])
        # rotation
        for d in (-rot_step, rot_step):
            cand = _eval(scale, rot_deg + float(d), tx, ty)
            if float(cand["sim"]) > float(best["sim"]):
                best = cand
                scale, rot_deg, tx, ty = float(best["scale"]), float(best["rot_deg"]), float(best["tx"]), float(best["ty"])
        # scale
        for m in (1.0 - scl_step, 1.0 + scl_step):
            cand = _eval(scale * float(m), rot_deg, tx, ty)
            if float(cand["sim"]) > float(best["sim"]):
                best = cand
                scale, rot_deg, tx, ty = float(best["scale"]), float(best["rot_deg"]), float(best["tx"]), float(best["ty"])
        # translation x/y
        for d in (-tr_step, tr_step):
            cand = _eval(scale, rot_deg, tx + float(d), ty)
            if float(cand["sim"]) > float(best["sim"]):
                best = cand
                scale, rot_deg, tx, ty = float(best["scale"]), float(best["rot_deg"]), float(best["tx"]), float(best["ty"])
        for d in (-tr_step, tr_step):
            cand = _eval(scale, rot_deg, tx, ty + float(d))
            if float(cand["sim"]) > float(best["sim"]):
                best = cand
                scale, rot_deg, tx, ty = float(best["scale"]), float(best["rot_deg"]), float(best["tx"]), float(best["ty"])

        rot_step = float(max(0.1, rot_step * shrink))
        scl_step = float(max(0.0005, scl_step * shrink))
        tr_step = float(max(0.1, tr_step * shrink))
        
        # 增强的 Early Stop 检查
        sim_improvement = float(best["sim"]) - sim_before_iter
        if early_stop_d > 0.0 and sim_improvement <= early_stop_d:
            consecutive_converged += 1
            if consecutive_converged >= min_converged_iters:
                # 连续多次迭代改进很小，认为已收敛
                break
        else:
            consecutive_converged = 0  # 重置收敛计数
        
        # 原有的步长检查
        if rot_step <= 0.15 and scl_step <= 0.002 and tr_step <= 0.15:
            break

    return best


def _uvh_points_from_th_mask_roi(
    *,
    th_gray_truncated: np.ndarray,
    mask_u8: np.ndarray,
    bbox: tuple[int, int, int, int],
    z_scale: float,
    max_points: int,
    rng: np.random.Generator,
    fill_mask: bool = True,
    min_height: float = 1.0,
    deterministic: bool = True,
) -> np.ndarray:
    """
    为点云刚体匹配准备“更稠密”的 uvh 点：
    - 使用 th_68 灰度图作为高度
    - 用填充后的 mask 决定支撑区域
    - 在支撑区域内给一个最小高度，避免只剩稀疏边缘点
    """
    th = np.asarray(th_gray_truncated, dtype=np.float32)
    mk = np.asarray(mask_u8, dtype=np.uint8)
    if th.ndim != 2 or mk.ndim != 2 or th.size <= 0 or mk.size <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if mk.shape != th.shape:
        mk = cv2.resize(mk, (int(th.shape[1]), int(th.shape[0])), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    if fill_mask:
        mk = _mask_to_filled_u8(mk)
        if mk.shape != th.shape:
            mk = cv2.resize(mk, (int(th.shape[1]), int(th.shape[0])), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    th2 = np.where(mk > 0, np.maximum(th, float(min_height)), 0.0).astype(np.float32)
    return _uvh_points_from_th_roi(
        th_gray_truncated=th2,
        bbox=bbox,
        z_scale=float(z_scale),
        max_points=int(max_points),
        rng=rng,
        deterministic=bool(deterministic),
    )


def _uvh_pcd_preprocess_for_fpfh(
    *,
    pts_xyz: np.ndarray,
    voxel: float,
) -> tuple[Any, Any, Any] | None:
    o3d = getattr(core, "o3d", None)
    if o3d is None:
        return None
    pts = np.asarray(pts_xyz, dtype=np.float64).reshape(-1, 3)
    if int(pts.shape[0]) < 50:
        return None
    try:
        pcd_full = o3d.geometry.PointCloud()
        pcd_full.points = o3d.utility.Vector3dVector(pts)
        if float(voxel) > 1e-9:
            pcd = pcd_full.voxel_down_sample(voxel_size=float(voxel))
        else:
            pcd = pcd_full
        if len(pcd.points) < 50:
            return None
        normal_rad = float(max(float(voxel) * 2.0, _env_float("MIS_UVH_RIGID_NORMAL_RADIUS", 0.0)))
        fpfh_rad = float(max(float(voxel) * 5.0, _env_float("MIS_UVH_RIGID_FPFH_RADIUS", 0.0)))
        if normal_rad <= 1e-9:
            normal_rad = max(1.0, float(voxel) * 2.0)
        if fpfh_rad <= 1e-9:
            fpfh_rad = max(2.0, float(voxel) * 5.0)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(normal_rad), max_nn=30))
        pcd_full.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(normal_rad), max_nn=30))
        feat = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=float(fpfh_rad), max_nn=100),
        )
        return pcd_full, pcd, feat
    except Exception:
        return None


def _uvh_pcd_match_stats(
    *,
    ref_pcd: Any,
    cur_pcd_aligned: Any,
    max_corr: float,
) -> dict[str, float]:
    """
    统计点云配准质量：
    - inlier_ratio / overlap
    - mean_dist / median_dist
    - normal_consistency
    """
    o3d = getattr(core, "o3d", None)
    if o3d is None or ref_pcd is None or cur_pcd_aligned is None:
        return {
            "inlier_ratio": float("nan"),
            "overlap": float("nan"),
            "mean_dist": float("nan"),
            "median_dist": float("nan"),
            "normal_consistency": float("nan"),
        }
    try:
        pts_cur = np.asarray(cur_pcd_aligned.points, dtype=np.float64).reshape(-1, 3)
        pts_ref = np.asarray(ref_pcd.points, dtype=np.float64).reshape(-1, 3)
        if int(pts_cur.shape[0]) <= 0 or int(pts_ref.shape[0]) <= 0:
            raise RuntimeError("empty_points")
        tree = o3d.geometry.KDTreeFlann(ref_pcd)
        dists: list[float] = []
        ncos: list[float] = []
        ref_n = np.asarray(ref_pcd.normals, dtype=np.float64).reshape(-1, 3) if ref_pcd.has_normals() else None
        cur_n = np.asarray(cur_pcd_aligned.normals, dtype=np.float64).reshape(-1, 3) if cur_pcd_aligned.has_normals() else None
        max_corr2 = float(max(1e-9, max_corr)) ** 2
        for i, p in enumerate(pts_cur):
            try:
                k, idx, dist2 = tree.search_knn_vector_3d(p.astype(np.float64), 1)
            except Exception:
                continue
            if int(k) < 1 or len(idx) <= 0 or len(dist2) <= 0:
                continue
            d2 = float(dist2[0])
            d = float(math.sqrt(max(0.0, d2)))
            dists.append(d)
            if (ref_n is not None) and (cur_n is not None) and i < int(cur_n.shape[0]) and int(idx[0]) < int(ref_n.shape[0]):
                nr = ref_n[int(idx[0])]
                nc = cur_n[int(i)]
                nrn = float(np.linalg.norm(nr))
                ncn = float(np.linalg.norm(nc))
                if nrn > 1e-9 and ncn > 1e-9:
                    c = float(np.dot(nr / nrn, nc / ncn))
                    ncos.append(abs(max(-1.0, min(1.0, c))))
        if len(dists) <= 0:
            raise RuntimeError("empty_distances")
        da = np.asarray(dists, dtype=np.float64)
        inlier = da <= float(max_corr)
        inlier_ratio = float(np.mean(inlier)) if da.size > 0 else float("nan")
        overlap = float(inlier_ratio)
        mean_dist = float(np.mean(da))
        median_dist = float(np.median(da))
        normal_consistency = float(np.mean(ncos)) if len(ncos) > 0 else float("nan")
        return {
            "inlier_ratio": float(inlier_ratio),
            "overlap": float(overlap),
            "mean_dist": float(mean_dist),
            "median_dist": float(median_dist),
            "normal_consistency": float(normal_consistency),
        }
    except Exception:
        return {
            "inlier_ratio": float("nan"),
            "overlap": float("nan"),
            "mean_dist": float("nan"),
            "median_dist": float("nan"),
            "normal_consistency": float("nan"),
        }


@contextlib.contextmanager
def _o3d_quiet_down_to_error(o3d: Any) -> Any:
    """
    临时将 Open3D 日志降到 Error，避免 ``registration_ransac_based_on_feature_matching``
    在 mutual filter 后对应点仍「偏多」但低于内部阈时反复打印
    ``Too few correspondences ... fall back to original`` 的 WARNING（库仍会正常回退）。
    """
    restore: Any = None
    util = getattr(o3d, "utility", None)
    if util is None or not hasattr(util, "set_verbosity_level"):
        yield
        return
    try:
        VL = util.VerbosityLevel
        if hasattr(util, "get_verbosity_level"):
            try:
                restore = util.get_verbosity_level()
            except Exception:
                restore = None
        if restore is None:
            restore = getattr(VL, "Warning", VL.Info)
        util.set_verbosity_level(VL.Error)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            util.set_verbosity_level(restore)
        except Exception:
            pass


def _estimate_uvh_rigid_pose_from_points(
    *,
    pts_ref: np.ndarray,
    pts_cur: np.ndarray,
    cfg: core.TrackConfig,
) -> dict[str, Any]:
    """
    工程化刚体匹配：
    1. 点云化/下采样
    2. FPFH + RANSAC 粗配准
    3. ICP 两阶段精配准
    4. 用 inlier_ratio / overlap / median_dist / normal_consistency 打分

    环境变量（补充）：
    - ``MIS_UVH_RIGID_O3D_QUIET_RANSAC=1``（默认）：RANSAC 调用期间暂降 Open3D 日志，抑制 mutual-filter 回退类 WARNING。
    - ``MIS_UVH_RIGID_RANSAC_MUTUAL_FILTER=0``：关闭互最近邻对应滤除，可减少上述分支（对应点可能更噪，靠 RANSAC checker 约束）。
    """
    o3d = getattr(core, "o3d", None)
    if o3d is None:
        return {"ok": False, "reason": "open3d_unavailable"}

    pref = np.asarray(pts_ref, dtype=np.float64).reshape(-1, 3)
    pcur = np.asarray(pts_cur, dtype=np.float64).reshape(-1, 3)
    if int(pref.shape[0]) < 80 or int(pcur.shape[0]) < 80:
        return {"ok": False, "reason": "too_few_points"}

    try:
        span = np.nanmax(pref, axis=0) - np.nanmin(pref, axis=0)
        L = float(max(1.0, np.nanmax(span)))
    except Exception:
        L = 32.0
    voxel = float(_env_float("MIS_UVH_RIGID_VOXEL", 0.8))
    if not np.isfinite(voxel) or voxel <= 1e-6:
        voxel = float(max(1.0, L / 40.0))

    pp_ref = _uvh_pcd_preprocess_for_fpfh(pts_xyz=pref, voxel=float(voxel))
    pp_cur = _uvh_pcd_preprocess_for_fpfh(pts_xyz=pcur, voxel=float(voxel))
    if pp_ref is None or pp_cur is None:
        return {"ok": False, "reason": "pcd_preprocess_failed"}
    ref_full, ref_ds, ref_fpfh = pp_ref
    cur_full, cur_ds, cur_fpfh = pp_cur
    if len(ref_ds.points) < 30 or len(cur_ds.points) < 30:
        return {"ok": False, "reason": "too_few_downsampled_points"}

    try:
        max_corr_ransac = float(_env_float("MIS_UVH_RIGID_RANSAC_MAX_CORR", max(2.0, voxel * 3.0)))
        max_corr_icp_coarse = float(_env_float("MIS_UVH_RIGID_ICP_MAX_CORR_COARSE", max(2.0, voxel * 3.0)))
        max_corr_icp_fine = float(_env_float("MIS_UVH_RIGID_ICP_MAX_CORR_FINE", max(1.0, voxel * 1.5)))
        allow_scale = bool(_env_bool("MIS_UVH_RIGID_ALLOW_SCALE", True))
        ransac_max_iter = int(max(1000, _env_int("MIS_UVH_RIGID_RANSAC_MAX_ITER", 20000)))
        ransac_conf = float(_env_float("MIS_UVH_RIGID_RANSAC_CONFIDENCE", 0.999))
        icp_max_iter_coarse = int(max(10, _env_int("MIS_UVH_RIGID_ICP_MAX_ITER_COARSE", 60)))
        icp_max_iter_fine = int(max(10, _env_int("MIS_UVH_RIGID_ICP_MAX_ITER_FINE", 40)))
        use_mutual = bool(_env_bool("MIS_UVH_RIGID_RANSAC_MUTUAL_FILTER", True))
        quiet_ransac = bool(_env_bool("MIS_UVH_RIGID_O3D_QUIET_RANSAC", True))

        _ransac_ctx = (
            _o3d_quiet_down_to_error(o3d) if quiet_ransac else contextlib.nullcontext()
        )
        with _ransac_ctx:
            ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                cur_ds,
                ref_ds,
                cur_fpfh,
                ref_fpfh,
                bool(use_mutual),
                float(max_corr_ransac),
                o3d.pipelines.registration.TransformationEstimationPointToPoint(bool(allow_scale)),
                4,
                [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(float(max_corr_ransac)),
                ],
                o3d.pipelines.registration.RANSACConvergenceCriteria(int(ransac_max_iter), float(ransac_conf)),
            )
        T0 = np.asarray(getattr(ransac, "transformation", np.eye(4)), dtype=np.float64).reshape(4, 4)
        if not np.isfinite(T0).all():
            return {"ok": False, "reason": "invalid_ransac_T"}

        est_type = (os.environ.get("MIS_UVH_RIGID_ICP_TYPE", "point_to_plane") or "point_to_plane").strip().lower()
        if est_type == "point_to_point":
            est = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
        else:
            est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
            est_type = "point_to_plane"

        icp0 = o3d.pipelines.registration.registration_icp(
            cur_full,
            ref_full,
            float(max_corr_icp_coarse),
            T0,
            est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(icp_max_iter_coarse)),
        )
        T1 = np.asarray(getattr(icp0, "transformation", T0), dtype=np.float64).reshape(4, 4)
        icp1 = o3d.pipelines.registration.registration_icp(
            cur_full,
            ref_full,
            float(max_corr_icp_fine),
            T1,
            est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(icp_max_iter_fine)),
        )
        T = np.asarray(getattr(icp1, "transformation", T1), dtype=np.float64).reshape(4, 4)
        if not np.isfinite(T).all():
            return {"ok": False, "reason": "invalid_icp_T"}

        cur_aligned = o3d.geometry.PointCloud(cur_full)
        cur_aligned.transform(T)
        stats = _uvh_pcd_match_stats(
            ref_pcd=ref_full,
            cur_pcd_aligned=cur_aligned,
            max_corr=float(max_corr_icp_fine),
        )
        median_dist = float(stats.get("median_dist", float("nan")))
        mean_dist = float(stats.get("mean_dist", float("nan")))
        overlap = float(stats.get("overlap", float("nan")))
        inlier_ratio = float(stats.get("inlier_ratio", float("nan")))
        normal_consistency = float(stats.get("normal_consistency", float("nan")))
        if (not np.isfinite(median_dist)) and (not np.isfinite(mean_dist)):
            return {"ok": False, "reason": "empty_distance_stats"}

        fitness = float(getattr(icp1, "fitness", 0.0))
        try:
            corr_count = int(len(getattr(ransac, "correspondence_set", [])))
        except Exception:
            corr_count = 0
        rmse = float(getattr(icp1, "inlier_rmse", float("inf")))
        if not np.isfinite(inlier_ratio):
            inlier_ratio = float(max(0.0, min(1.0, fitness)))
        if not np.isfinite(overlap):
            overlap = float(max(0.0, min(1.0, fitness)))
        if not np.isfinite(normal_consistency):
            normal_consistency = float("nan")
        dist_ref = median_dist if np.isfinite(median_dist) else mean_dist
        dist_score = float(math.exp(-float(dist_ref) / float(max(1e-6, max_corr_icp_fine)))) if np.isfinite(dist_ref) else 0.0
        w_in = float(_env_float("MIS_UVH_RIGID_W_INLIER", 0.34))
        w_ov = float(_env_float("MIS_UVH_RIGID_W_OVERLAP", 0.26))
        w_dst = float(_env_float("MIS_UVH_RIGID_W_DIST", 0.20))
        w_nrm = float(_env_float("MIS_UVH_RIGID_W_NORMAL", 0.20))
        ws = float(abs(w_in) + abs(w_ov) + abs(w_dst) + abs(w_nrm))
        if ws <= 1e-9:
            w_in, w_ov, w_dst, w_nrm, ws = 0.34, 0.26, 0.20, 0.20, 1.0
        nrm_score = float(normal_consistency) if np.isfinite(normal_consistency) else float(max(0.0, min(1.0, fitness)))
        rigid_score = float((w_in * inlier_ratio + w_ov * overlap + w_dst * dist_score + w_nrm * nrm_score) / ws)
        corr_soft_min = int(max(16, _env_int("MIS_UVH_RIGID_CORR_SOFT_MIN", 192)))
        corr_pen_w = float(max(0.0, _env_float("MIS_UVH_RIGID_CORR_PEN_W", 0.20)))
        lowcorr_pen = 0.0
        if int(corr_count) > 0 and int(corr_count) < int(corr_soft_min):
            lowcorr_pen = float(corr_soft_min - int(corr_count)) / float(max(1, corr_soft_min))
            rigid_score = float(rigid_score) - float(corr_pen_w) * float(lowcorr_pen)
        rigid_score = float(max(0.0, min(1.0, rigid_score)))
        return {
            "ok": True,
            "reason": "",
            "method": f"fpfh_ransac+icp_{est_type}",
            "T_ref_cur": T,
            "fitness": float(fitness),
            "inlier_ratio": float(inlier_ratio),
            "rmse": float(rmse if np.isfinite(rmse) else float("nan")),
            "median_dist": float(median_dist if np.isfinite(median_dist) else float("nan")),
            "mean_dist": float(mean_dist if np.isfinite(mean_dist) else float("nan")),
            "overlap": float(overlap),
            "normal_consistency": float(normal_consistency if np.isfinite(normal_consistency) else float("nan")),
            "score": float(rigid_score),
            "voxel": float(voxel),
            "corr_count": int(corr_count),
            "lowcorr_pen": float(lowcorr_pen),
            "max_corr_ransac": float(max_corr_ransac),
            "max_corr_icp": float(max_corr_icp_fine),
        }
    except Exception as e:
        return {"ok": False, "reason": f"exception:{e}"}


def _transform_uvh_points(
    *,
    pts_xyz: np.ndarray,
    T_dst_src: np.ndarray,
) -> np.ndarray:
    pts = np.asarray(pts_xyz, dtype=np.float64).reshape(-1, 3)
    if int(pts.shape[0]) <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    T = np.asarray(T_dst_src, dtype=np.float64).reshape(4, 4)
    ones = np.ones((int(pts.shape[0]), 1), dtype=np.float64)
    pts4 = np.concatenate([pts, ones], axis=1)
    out = (T @ pts4.T).T[:, :3]
    if out.size <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def _bbox_point_coverage_ratio(
    *,
    pts_xyz: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    pts = np.asarray(pts_xyz, dtype=np.float32).reshape(-1, 3)
    if int(pts.shape[0]) <= 0:
        return float("nan")
    bx, by, bw, bh = [int(v) for v in bbox]
    if bw <= 0 or bh <= 0:
        return float("nan")
    xy = np.asarray(pts[:, :2], dtype=np.float32)
    good = np.isfinite(xy).all(axis=1)
    xy = xy[good]
    if int(xy.shape[0]) <= 0:
        return float("nan")
    xin = (xy[:, 0] >= float(bx)) & (xy[:, 0] <= float(bx + bw - 1))
    yin = (xy[:, 1] >= float(by)) & (xy[:, 1] <= float(by + bh - 1))
    return float(np.mean(xin & yin))


def _bbox_from_transformed_uvh_points(
    *,
    pts_xyz: np.ndarray,
    frame_wh: tuple[int, int],
    fallback_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    pts = np.asarray(pts_xyz, dtype=np.float32).reshape(-1, 3)
    if int(pts.shape[0]) < 20:
        return None
    xy = np.asarray(pts[:, :2], dtype=np.float32)
    good = np.isfinite(xy).all(axis=1)
    xy = xy[good]
    if int(xy.shape[0]) < 20:
        return None
    # 默认改为更保守的“全覆盖外扩框”，避免 percentile + 小 pad 把目标边缘裁掉
    q_lo = float(max(0.0, min(20.0, _env_float("MIS_UVH_RIGID_BBOX_Q_LO", 0.0))))
    q_hi = float(max(80.0, min(100.0, _env_float("MIS_UVH_RIGID_BBOX_Q_HI", 100.0))))
    pad = float(max(0.0, _env_float("MIS_UVH_RIGID_BBOX_PAD", 10.0)))
    try:
        x0 = float(np.percentile(xy[:, 0], q_lo)) - pad
        x1 = float(np.percentile(xy[:, 0], q_hi)) + pad
        y0 = float(np.percentile(xy[:, 1], q_lo)) - pad
        y1 = float(np.percentile(xy[:, 1], q_hi)) + pad
    except Exception:
        x0 = float(np.min(xy[:, 0])) - pad
        x1 = float(np.max(xy[:, 0])) + pad
        y0 = float(np.min(xy[:, 1])) - pad
        y1 = float(np.max(xy[:, 1])) + pad
    fw, fh = int(frame_wh[0]), int(frame_wh[1])
    w = int(max(4, round(x1 - x0)))
    h = int(max(4, round(y1 - y0)))
    x = int(round(x0))
    y = int(round(y0))
    x, y, w, h = core.clamp_bbox(x, y, w, h, fw, fh)
    if w < 4 or h < 4:
        return None
    try:
        fb = tuple(int(v) for v in fallback_bbox)
        area_f = float(max(1, fb[2] * fb[3]))
        area_n = float(max(1, w * h))
        r = float(area_n / area_f)
        rmin = float(_env_float("MIS_UVH_RIGID_BBOX_AREA_RATIO_MIN", 0.60))
        rmax = float(_env_float("MIS_UVH_RIGID_BBOX_AREA_RATIO_MAX", 3.20))
        if (not np.isfinite(r)) or r < rmin or r > rmax:
            return None
    except Exception:
        pass
    return int(x), int(y), int(w), int(h)


def _rebuild_uvh_candidate_from_bbox(
    *,
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    cfg: core.TrackConfig,
    ref_mask_small: np.ndarray | None = None,
    search_ctx: dict[str, Any] | None = None,
    cache: dict[tuple[int, int, int, int], dict[str, Any]] | None = None,
    frame_th_gray: np.ndarray | None = None,
    frame_mask_u8: np.ndarray | None = None,
    frame_th_uint8: float | None = None,
) -> dict[str, Any] | None:
    x, y, w, h = [int(v) for v in bbox]
    if w < 4 or h < 4:
        return None
    cache_key = (int(x), int(y), int(w), int(h))
    if isinstance(cache, dict):
        cached = cache.get(cache_key, None)
        if isinstance(cached, dict):
            return dict(cached)

    th_gray = None
    mask_u8 = None
    th_uint8 = float("nan")
    if isinstance(search_ctx, dict):
        try:
            sx0 = int(search_ctx.get("sx0", 0))
            sy0 = int(search_ctx.get("sy0", 0))
            sw = int(search_ctx.get("sw", 0))
            sh = int(search_ctx.get("sh", 0))
            th_search = np.asarray(search_ctx.get("th_gray", None), dtype=np.float32)
            mask_search = np.asarray(search_ctx.get("mask_u8", None), dtype=np.uint8)
            if th_search.ndim == 2 and mask_search.ndim == 2 and th_search.shape == mask_search.shape:
                xi = int(x - sx0)
                yi = int(y - sy0)
                if xi >= 0 and yi >= 0 and (xi + int(w) <= sw) and (yi + int(h) <= sh):
                    th_gray = th_search[yi:yi + int(h), xi:xi + int(w)].astype(np.float32, copy=True)
                    mask_u8 = mask_search[yi:yi + int(h), xi:xi + int(w)].astype(np.uint8, copy=True)
                    th_uint8 = float(search_ctx.get("th_uint8", float("nan")))
        except Exception:
            th_gray = None
            mask_u8 = None

    if (not isinstance(th_gray, np.ndarray) or not isinstance(mask_u8, np.ndarray)) and (
        isinstance(frame_th_gray, np.ndarray) and isinstance(frame_mask_u8, np.ndarray)
    ):
        fH, fW = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
        if (
            int(frame_th_gray.shape[0]) == fH
            and int(frame_th_gray.shape[1]) == fW
            and int(frame_mask_u8.shape[0]) == fH
            and int(frame_mask_u8.shape[1]) == fW
            and (y + h) <= fH
            and (x + w) <= fW
            and y >= 0
            and x >= 0
        ):
            th_gray = np.asarray(frame_th_gray[y : y + h, x : x + w], dtype=np.float32).copy()
            mask_u8 = np.asarray(frame_mask_u8[y : y + h, x : x + w], dtype=np.uint8).copy()
            th_uint8 = float(frame_th_uint8) if (frame_th_uint8 is not None and np.isfinite(float(frame_th_uint8))) else float("nan")

    if not isinstance(th_gray, np.ndarray) or not isinstance(mask_u8, np.ndarray):
        roi = np.asarray(frame_bgr[y:y + h, x:x + w], dtype=np.uint8)
        if roi.ndim != 3 or roi.shape[0] < 4 or roi.shape[1] < 4:
            return None
        th_gray, mask_u8, th_uint8 = core.compute_th_gray_truncated(roi, cfg.th_percent, cfg)
    mask_u8 = _apply_reference_mask_guidance(
        th_gray=np.asarray(th_gray, dtype=np.float32),
        mask_u8=np.asarray(mask_u8, dtype=np.uint8),
        ref_mask_small=ref_mask_small,
    )
    cnt = _extract_contour_prefer_center(mask_u8)
    if cnt is None:
        cnt = core.extract_largest_contour(mask_u8)
    if cnt is not None:
        cnt2 = core.resize_contour(cnt, (w, h), cfg.semantic_image_size)
        contour_rgb = core.contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)
    else:
        contour_rgb = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
    pts_cur = _uvh_points_from_th_roi(
        th_gray_truncated=np.asarray(th_gray, dtype=np.float32),
        bbox=(int(x), int(y), int(w), int(h)),
        z_scale=float(cfg.mesh_z_scale),
        max_points=int(_env_int("MIS_UVH_RENDER_MAX_POINTS", 5000)),  # 优化：从10000降至5000提速
        rng=np.random.default_rng(0),
        deterministic=True,
    )
    pack = {
        "bbox": (int(x), int(y), int(w), int(h)),
        "th_gray": np.asarray(th_gray, dtype=np.float32),
        "mask_u8": np.asarray(mask_u8, dtype=np.uint8),
        "th_uint8": float(th_uint8),
        "contour_rgb": contour_rgb,
        "uvh_render_pts": np.asarray(pts_cur, dtype=np.float32),
    }
    if isinstance(cache, dict):
        cache[cache_key] = pack
    return dict(pack)


def _apply_reference_mask_guidance(
    *,
    th_gray: np.ndarray,
    mask_u8: np.ndarray,
    ref_mask_small: np.ndarray | None,
) -> np.ndarray:
    """
    用首帧 reference mask 作为弱先验，对当前 th_68 mask 做“补边而非硬替换”：
    - 保留当前 th_68 的高置信区域
    - 仅在 reference 软区域内，且当前 th_gray 仍有响应时扩边
    """
    base = np.asarray(mask_u8, dtype=np.uint8)
    th = np.asarray(th_gray, dtype=np.float32)
    if not bool(_env_bool("MIS_UVH_REFMASK_GUIDE_ENABLE", True)):
        return base
    if not isinstance(ref_mask_small, np.ndarray):
        return base
    if th.ndim != 2 or base.ndim != 2 or th.shape != base.shape:
        return base
    h, w = th.shape[:2]
    if h < 4 or w < 4:
        return base

    try:
        ref64 = np.asarray(ref_mask_small, dtype=np.uint8)
        if ref64.shape != (64, 64):
            ref64 = cv2.resize(ref64, (64, 64), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        ref_soft = cv2.resize(ref64, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    except Exception:
        return base

    pos = th[th > 1e-6]
    if pos.size <= 0:
        return base
    vmax = float(np.percentile(pos, 98.0)) if pos.size >= 8 else float(np.max(pos))
    if not np.isfinite(vmax) or vmax <= 1e-6:
        return base

    guide_thr = float(max(0.0, min(1.0, _env_float("MIS_UVH_REFMASK_GUIDE_THR", 0.18))))
    strong_thr = float(max(guide_thr, min(1.0, _env_float("MIS_UVH_REFMASK_GUIDE_STRONG_THR", 0.35))))
    low_resp = float(max(0.0, _env_float("MIS_UVH_REFMASK_TH_LOW_RATIO", 0.08))) * vmax
    high_resp = float(max(0.0, _env_float("MIS_UVH_REFMASK_TH_HIGH_RATIO", 0.18))) * vmax

    # 先把 reference prior 稍微放宽一点，允许补上 th_68 漏掉的边缘。
    k = int(max(1, round(min(h, w) * float(max(0.01, _env_float("MIS_UVH_REFMASK_DILATE_RATIO", 0.03))))))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    ref_bin = (ref_soft >= guide_thr).astype(np.uint8) * 255
    ref_bin = cv2.dilate(ref_bin, kernel, iterations=int(max(1, _env_int("MIS_UVH_REFMASK_DILATE_ITERS", 1))))
    ref_soft2 = np.maximum(ref_soft, ref_bin.astype(np.float32) / 255.0)

    extra_soft = (ref_soft2 >= guide_thr) & (th >= float(low_resp))
    extra_strong = (ref_soft2 >= strong_thr) & (th >= float(high_resp))
    merged = np.where((base > 0) | extra_soft | extra_strong, 255, 0).astype(np.uint8)

    close_k = int(max(1, round(min(h, w) * float(max(0.01, _env_float("MIS_UVH_REFMASK_CLOSE_RATIO", 0.025))))))
    if close_k % 2 == 0:
        close_k += 1
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, close_kernel)
    return merged.astype(np.uint8)


def _bbox_from_mask_in_frame(
    *,
    mask_u8: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_wh: tuple[int, int],
    pad: int = 0,
) -> tuple[int, int, int, int] | None:
    mk = np.asarray(mask_u8, dtype=np.uint8)
    if mk.ndim != 2 or mk.size <= 0:
        return None
    ys, xs = np.nonzero(mk > 0)
    if xs.size <= 0 or ys.size <= 0:
        return None
    bx, by, _bw, _bh = [int(v) for v in bbox]
    x0 = int(bx + int(xs.min()) - int(max(0, pad)))
    y0 = int(by + int(ys.min()) - int(max(0, pad)))
    x1 = int(bx + int(xs.max()) + 1 + int(max(0, pad)))
    y1 = int(by + int(ys.max()) + 1 + int(max(0, pad)))
    fw, fh = int(frame_wh[0]), int(frame_wh[1])
    return core.clamp_bbox(x0, y0, max(4, x1 - x0), max(4, y1 - y0), fw, fh)


def _bbox_union(
    *,
    frame_wh: tuple[int, int],
    boxes: list[tuple[int, int, int, int] | None],
) -> tuple[int, int, int, int] | None:
    pts: list[tuple[int, int, int, int]] = []
    for bb in boxes:
        if not (isinstance(bb, tuple) and len(bb) == 4):
            continue
        x, y, w, h = [int(v) for v in bb]
        if w < 4 or h < 4:
            continue
        pts.append((x, y, w, h))
    if not pts:
        return None
    x0 = min(int(v[0]) for v in pts)
    y0 = min(int(v[1]) for v in pts)
    x1 = max(int(v[0] + v[2]) for v in pts)
    y1 = max(int(v[1] + v[3]) for v in pts)
    fw, fh = int(frame_wh[0]), int(frame_wh[1])
    return core.clamp_bbox(x0, y0, max(4, x1 - x0), max(4, y1 - y0), fw, fh)


def _bbox_area(bbox: tuple[int, int, int, int] | None) -> float:
    if not (isinstance(bbox, tuple) and len(bbox) == 4):
        return 0.0
    try:
        return float(max(1, int(bbox[2]) * int(bbox[3])))
    except Exception:
        return 0.0


def _enforce_full_coverage_bbox(
    *,
    frame_wh: tuple[int, int],
    chosen_bbox: tuple[int, int, int, int],
    support_boxes: list[tuple[int, int, int, int] | None],
    area_ratio_min: float,
    side_ratio_min: float,
    expand_scale: float,
) -> tuple[tuple[int, int, int, int], str]:
    union_bbox = _bbox_union(frame_wh=frame_wh, boxes=[chosen_bbox, *support_boxes])
    if union_bbox is None:
        return tuple(chosen_bbox), ""

    cw, ch = int(chosen_bbox[2]), int(chosen_bbox[3])
    uw, uh = int(union_bbox[2]), int(union_bbox[3])
    chosen_area = _bbox_area(chosen_bbox)
    union_area = _bbox_area(union_bbox)
    if union_area <= chosen_area + 1.0:
        return tuple(chosen_bbox), ""

    need_union = False
    if float(chosen_area) < float(area_ratio_min) * float(union_area):
        need_union = True
    if float(cw) < float(side_ratio_min) * float(max(1, uw)):
        need_union = True
    if float(ch) < float(side_ratio_min) * float(max(1, uh)):
        need_union = True
    if not need_union:
        return tuple(chosen_bbox), ""

    grown = _scale_bbox_about_center(
        bbox=tuple(union_bbox),
        scale=float(max(1.0, expand_scale)),
        frame_wh=frame_wh,
    )
    return tuple(grown), "anti_shrink_union"


def _scale_bbox_about_center(
    *,
    bbox: tuple[int, int, int, int],
    scale: float,
    frame_wh: tuple[int, int],
) -> tuple[int, int, int, int]:
    x, y, w, h = [int(v) for v in bbox]
    s = float(scale) if np.isfinite(float(scale)) else 1.0
    s = float(max(0.5, min(3.0, s)))
    cx = float(x) + 0.5 * float(w)
    cy = float(y) + 0.5 * float(h)
    nw = int(max(4, round(float(w) * s)))
    nh = int(max(4, round(float(h) * s)))
    nx = int(round(cx - 0.5 * float(nw)))
    ny = int(round(cy - 0.5 * float(nh)))
    fw, fh = int(frame_wh[0]), int(frame_wh[1])
    return core.clamp_bbox(nx, ny, nw, nh, fw, fh)


def _bbox_edge_touch_score(
    *,
    inner_bbox: tuple[int, int, int, int] | None,
    outer_bbox: tuple[int, int, int, int],
    tol_px: int = 2,
) -> float:
    if inner_bbox is None:
        return 0.0
    ix, iy, iw, ih = [int(v) for v in inner_bbox]
    ox, oy, ow, oh = [int(v) for v in outer_bbox]
    left = 1.0 if abs(int(ix) - int(ox)) <= int(tol_px) else 0.0
    top = 1.0 if abs(int(iy) - int(oy)) <= int(tol_px) else 0.0
    right = 1.0 if abs(int(ix + iw) - int(ox + ow)) <= int(tol_px) else 0.0
    bottom = 1.0 if abs(int(iy + ih) - int(oy + oh)) <= int(tol_px) else 0.0
    return float(0.25 * (left + top + right + bottom))


def _postprocess_bbox_for_full_coverage(
    *,
    frame_bgr: np.ndarray,
    cand: dict[str, Any],
    ref_mask_small: np.ndarray,
    ref_contour_rgb: np.ndarray | None,
    projected_ref_pts: np.ndarray | None,
    cfg: core.TrackConfig,
    search_ctx: dict[str, Any] | None = None,
    cache: dict[tuple[int, int, int, int], dict[str, Any]] | None = None,
    frame_th_gray: np.ndarray | None = None,
    frame_mask_u8: np.ndarray | None = None,
    frame_th_uint8: float | None = None,
) -> dict[str, Any]:
    """
    最终 bbox 修正层：
    - 汇合当前 mask bbox / 刚体投影点 bbox / 原 bbox
    - 对 union bbox 再做若干外扩候选
    - 用 quick + contour + coverage 选择“更完整但不过度发散”的框
    """
    if not bool(_env_bool("MIS_UVH_POST_BBOX_ENABLE", True)):
        return cand
    bbox0 = tuple(int(v) for v in cand.get("bbox", (0, 0, 0, 0)))
    if bbox0[2] < 4 or bbox0[3] < 4:
        return cand
    H, W = frame_bgr.shape[:2]
    frame_wh = (int(W), int(H))
    mask_pad = int(max(0, _env_int("MIS_UVH_POST_MASK_PAD", 6)))
    point_pad = float(max(0.0, _env_float("MIS_UVH_POST_POINT_PAD", 14.0)))
    edge_tol = int(max(0, _env_int("MIS_UVH_POST_EDGE_TOL", 2)))
    cov_soft_min = float(max(0.0, min(1.0, _env_float("MIS_UVH_POST_COVERAGE_SOFT_MIN", 0.95))))
    score_eps = float(max(0.0, _env_float("MIS_UVH_POST_ACCEPT_EPS", 1e-4)))
    scales = _parse_float_list("MIS_UVH_POST_EXPAND_SCALES", "1.00,1.08,1.16,1.28,1.42,1.58")
    if not scales:
        scales = [1.0]
    union_area_ratio_min = float(max(0.50, min(1.0, _env_float("MIS_UVH_POST_UNION_AREA_RATIO_MIN", 0.90))))
    union_side_ratio_min = float(max(0.50, min(1.0, _env_float("MIS_UVH_POST_UNION_SIDE_RATIO_MIN", 0.94))))
    union_expand_scale = float(max(1.0, _env_float("MIS_UVH_POST_UNION_EXPAND_SCALE", 1.04)))

    ref_mask64 = np.asarray(ref_mask_small, dtype=np.uint8)
    if ref_mask64.shape != (64, 64):
        ref_mask64 = cv2.resize(ref_mask64, (64, 64), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

    mask_src = cand.get("mask_u8", None)
    if isinstance(mask_src, np.ndarray):
        mask_bbox = _bbox_from_mask_in_frame(
            mask_u8=np.asarray(mask_src, dtype=np.uint8),
            bbox=bbox0,
            frame_wh=frame_wh,
            pad=int(mask_pad),
        )
    else:
        mask_bbox = None
    pts_bbox = None
    pts_cov0 = float("nan")
    pts_arr = None
    try:
        if isinstance(projected_ref_pts, np.ndarray) and int(np.asarray(projected_ref_pts).shape[0]) >= 20:
            pts_arr = np.asarray(projected_ref_pts, dtype=np.float32).reshape(-1, 3)
            pts_bbox = _bbox_from_transformed_uvh_points(
                pts_xyz=pts_arr,
                frame_wh=frame_wh,
                fallback_bbox=bbox0,
            )
            if pts_bbox is not None and float(point_pad) > 1e-6:
                pts_bbox = _scale_bbox_about_center(
                    bbox=pts_bbox,
                    scale=float(
                        max(
                            1.0,
                            math.sqrt(
                                float(max(1.0, (pts_bbox[2] + 2.0 * point_pad) * (pts_bbox[3] + 2.0 * point_pad)))
                                / float(max(1, pts_bbox[2] * pts_bbox[3]))
                            ),
                        )
                    ),
                    frame_wh=frame_wh,
                )
            pts_cov0 = _bbox_point_coverage_ratio(pts_xyz=pts_arr, bbox=bbox0)
    except Exception:
        pts_bbox = None
        pts_cov0 = float("nan")

    touch0 = _bbox_edge_touch_score(inner_bbox=mask_bbox, outer_bbox=bbox0, tol_px=int(edge_tol))
    need_expand = False
    if np.isfinite(pts_cov0) and float(pts_cov0) < float(cov_soft_min):
        need_expand = True
    if float(touch0) > 0.20:
        need_expand = True
    if (mask_bbox is not None) and (mask_bbox[2] * mask_bbox[3] > bbox0[2] * bbox0[3]):
        need_expand = True
    if not need_expand:
        cand["uvh_post_bbox_used"] = 0
        cand["uvh_post_bbox_score"] = float("nan")
        cand["uvh_post_bbox_source"] = ""
        return cand

    union_bbox = _bbox_union(frame_wh=frame_wh, boxes=[bbox0, mask_bbox, pts_bbox])
    unique_boxes: dict[tuple[int, int, int, int], str] = {bbox0: "orig"}
    if mask_bbox is not None:
        unique_boxes[tuple(mask_bbox)] = "mask"
    if pts_bbox is not None:
        unique_boxes[tuple(pts_bbox)] = "points"
    if union_bbox is not None:
        unique_boxes[tuple(union_bbox)] = "union"
        for sc in scales:
            try:
                bb = _scale_bbox_about_center(bbox=union_bbox, scale=float(sc), frame_wh=frame_wh)
                unique_boxes[tuple(bb)] = f"union_x{float(sc):.2f}"
            except Exception:
                continue

    w_quick = float(max(0.0, _env_float("MIS_UVH_POST_W_QUICK", 0.40)))
    w_sem = float(max(0.0, _env_float("MIS_UVH_POST_W_SEM", 0.12)))
    w_cov = float(max(0.0, _env_float("MIS_UVH_POST_W_COVERAGE", 0.34)))
    w_touch = float(max(0.0, _env_float("MIS_UVH_POST_W_TOUCH", 0.20)))
    pen_under = float(max(0.0, _env_float("MIS_UVH_POST_PEN_UNDER", 0.30)))
    pen_over = float(max(0.0, _env_float("MIS_UVH_POST_PEN_OVER", 0.05)))
    pen_shift = float(max(0.0, _env_float("MIS_UVH_POST_PEN_SHIFT", 0.04)))
    area0 = float(max(1, bbox0[2] * bbox0[3]))
    cx0 = float(bbox0[0]) + 0.5 * float(bbox0[2])
    cy0 = float(bbox0[1]) + 0.5 * float(bbox0[3])

    best_score = -1e9
    orig_post_score = float("nan")
    best_pack = None
    for bb, src in unique_boxes.items():
        rebuilt = _rebuild_uvh_candidate_from_bbox(
            frame_bgr=frame_bgr,
            bbox=bb,
            cfg=cfg,
            ref_mask_small=ref_mask_small,
            search_ctx=search_ctx,
            cache=cache,
            frame_th_gray=frame_th_gray,
            frame_mask_u8=frame_mask_u8,
            frame_th_uint8=frame_th_uint8,
        )
        if not isinstance(rebuilt, dict):
            continue
        try:
            qmask = _mask64_from_mask_u8(np.asarray(rebuilt["mask_u8"], dtype=np.uint8))
            quick = float(core.quick_mask_score(ref_mask64, qmask))
        except Exception:
            quick = 0.0
        sem = float("nan")
        if isinstance(ref_contour_rgb, np.ndarray):
            try:
                sem = float(
                    core.semantic_similarity(
                        ref_contour_rgb,
                        np.asarray(rebuilt["contour_rgb"], dtype=np.uint8),
                        cfg,
                    )
                )
            except Exception:
                try:
                    sem = float(
                        core.fallback_similarity(
                            ref_contour_rgb,
                            np.asarray(rebuilt["contour_rgb"], dtype=np.uint8),
                        )
                    )
                except Exception:
                    sem = float("nan")
        cov = float("nan")
        if isinstance(pts_arr, np.ndarray) and int(pts_arr.shape[0]) >= 20:
            try:
                cov = _bbox_point_coverage_ratio(pts_xyz=pts_arr, bbox=tuple(rebuilt["bbox"]))
            except Exception:
                cov = float("nan")
        mask_bbox_i = _bbox_from_mask_in_frame(
            mask_u8=np.asarray(rebuilt["mask_u8"], dtype=np.uint8),
            bbox=tuple(rebuilt["bbox"]),
            frame_wh=frame_wh,
            pad=int(mask_pad),
        )
        touch_i = _bbox_edge_touch_score(inner_bbox=mask_bbox_i, outer_bbox=tuple(rebuilt["bbox"]), tol_px=int(edge_tol))
        bx, by, bw, bh = [int(v) for v in rebuilt["bbox"]]
        area_ratio = float(max(1.0, bw * bh)) / float(area0)
        under_pen = float(max(0.0, math.log(max(1e-6, 1.0 / max(1e-6, area_ratio))))) if area_ratio < 1.0 else 0.0
        over_pen = float(max(0.0, math.log(max(1e-6, area_ratio)))) if area_ratio > 1.0 else 0.0
        shift_pen = float(math.hypot((float(bx) + 0.5 * float(bw)) - cx0, (float(by) + 0.5 * float(bh)) - cy0) / float(max(1.0, 0.5 * (bbox0[2] + bbox0[3]))))
        score = (
            float(w_quick) * float(max(0.0, min(1.0, quick)))
            + float(w_sem) * float(max(0.0, min(1.0, sem if np.isfinite(sem) else 0.0)))
            + float(w_cov) * float(max(0.0, min(1.0, cov if np.isfinite(cov) else 0.0)))
            + float(w_touch) * float(max(0.0, 1.0 - touch_i))
            - float(pen_under) * float(under_pen)
            - float(pen_over) * float(over_pen)
            - float(pen_shift) * float(shift_pen)
        )
        if float(score) > float(best_score):
            best_score = float(score)
            best_pack = (rebuilt, src, cov, touch_i, area_ratio, score, mask_bbox_i)
        if tuple(bb) == tuple(bbox0):
            orig_post_score = float(score)

    if best_pack is None:
        cand["uvh_post_bbox_used"] = 0
        cand["uvh_post_bbox_score"] = float("nan")
        cand["uvh_post_bbox_source"] = ""
        return cand

    rebuilt, src, cov, touch_i, area_ratio, post_score, mask_bbox_best = best_pack
    choose_new = False
    if tuple(rebuilt["bbox"]) != bbox0:
        if np.isfinite(float(cov)) and np.isfinite(float(pts_cov0)):
            choose_new = bool(float(cov) > float(pts_cov0) + float(score_eps))
        elif np.isfinite(float(orig_post_score)):
            choose_new = bool(float(post_score) > float(orig_post_score) + float(score_eps))
        else:
            choose_new = bool(float(post_score) > 1e-6)

    chosen_bbox = tuple(rebuilt["bbox"]) if choose_new else tuple(bbox0)
    chosen_src = str(src) if choose_new else "orig"
    chosen_mask_bbox = mask_bbox_best if choose_new else mask_bbox
    final_bbox, anti_src = _enforce_full_coverage_bbox(
        frame_wh=frame_wh,
        chosen_bbox=chosen_bbox,
        support_boxes=[chosen_mask_bbox, pts_bbox, bbox0],
        area_ratio_min=float(union_area_ratio_min),
        side_ratio_min=float(union_side_ratio_min),
        expand_scale=float(union_expand_scale),
    )
    apply_post = bool(choose_new or (tuple(final_bbox) != tuple(bbox0)))
    if not apply_post:
        cand["uvh_post_bbox_used"] = 0
        cand["uvh_post_bbox_score"] = float(post_score)
        cand["uvh_post_bbox_source"] = str(chosen_src)
        return cand

    if tuple(final_bbox) == tuple(rebuilt["bbox"]):
        final_pack = rebuilt
    else:
        final_pack = _rebuild_uvh_candidate_from_bbox(
            frame_bgr=frame_bgr,
            bbox=tuple(final_bbox),
            cfg=cfg,
            ref_mask_small=ref_mask_small,
            search_ctx=search_ctx,
            cache=cache,
            frame_th_gray=frame_th_gray,
            frame_mask_u8=frame_mask_u8,
            frame_th_uint8=frame_th_uint8,
        )
        if not isinstance(final_pack, dict):
            final_pack = rebuilt if choose_new else cand

    cand = dict(cand)
    cand["bbox"] = tuple(final_pack["bbox"])
    cand["th_gray"] = final_pack["th_gray"]
    cand["mask_u8"] = final_pack["mask_u8"]
    cand["th_uint8"] = float(final_pack["th_uint8"])
    cand["contour_rgb"] = final_pack["contour_rgb"]
    cand["uvh_render_pts"] = final_pack["uvh_render_pts"]
    try:
        bx, by, bw, bh = [int(v) for v in cand["bbox"]]
        cand["dx"] = int((bx + bw // 2) - cx0)
        cand["dy"] = int((by + bh // 2) - cy0)
        cand["quick"] = float(core.quick_mask_score(ref_mask64, _mask64_from_mask_u8(np.asarray(cand["mask_u8"], dtype=np.uint8))))
        cand["uvh_bbox_scale"] = float(math.sqrt((float(bw) * float(bh)) / float(max(1.0, float(bbox0[2]) * float(bbox0[3]) + 1e-6))))
    except Exception:
        pass
    if np.isfinite(float(cov)):
        cand["uvh_coverage_ratio"] = float(cov)
    cand["uvh_post_bbox_used"] = 1
    cand["uvh_post_bbox_score"] = float(post_score)
    cand["uvh_post_bbox_source"] = str(chosen_src if not anti_src else f"{chosen_src}+{anti_src}")
    cand["uvh_post_bbox_touch"] = float(touch_i)
    cand["uvh_post_bbox_area_ratio"] = float(area_ratio)
    cand["uvh_decision_stage"] = "post"
    return cand


def _refine_bbox_by_local_mask_search(
    *,
    frame_bgr: np.ndarray,
    init_bbox: tuple[int, int, int, int],
    ref_mask_small: np.ndarray,
    ref_contour_rgb: np.ndarray | None,
    cfg: core.TrackConfig,
    search_ctx: dict[str, Any] | None = None,
    cache: dict[tuple[int, int, int, int], dict[str, Any]] | None = None,
    frame_th_gray: np.ndarray | None = None,
    frame_mask_u8: np.ndarray | None = None,
    frame_th_uint8: float | None = None,
) -> dict[str, Any] | None:
    """
    在一个较小的局部窗口内，对刚体反推后的 bbox 再做一次边界细化：
    - 使用当前 ROI 内 mask 的紧致包围框作为候选锚点
    - 对平移/尺度做小范围枚举
    - 以 quick mask + contour semantic 为主进行重评分
    """
    x0, y0, w0, h0 = [int(v) for v in init_bbox]
    if w0 < 4 or h0 < 4:
        return None
    H, W = frame_bgr.shape[:2]

    base = _rebuild_uvh_candidate_from_bbox(
        frame_bgr=frame_bgr,
        bbox=init_bbox,
        cfg=cfg,
        ref_mask_small=ref_mask_small,
        search_ctx=search_ctx,
        cache=cache,
        frame_th_gray=frame_th_gray,
        frame_mask_u8=frame_mask_u8,
        frame_th_uint8=frame_th_uint8,
    )
    if not isinstance(base, dict):
        return None

    mk0 = np.asarray(base.get("mask_u8", None), dtype=np.uint8)
    anchor_bbox = tuple(int(v) for v in init_bbox)
    try:
        mk_fill = _mask_to_filled_u8(mk0)
        cnt = _extract_contour_prefer_center(mk_fill)
        if cnt is None:
            cnt = core.extract_largest_contour(mk_fill)
        if cnt is not None and len(cnt) >= 3:
            rx, ry, rw, rh = cv2.boundingRect(cnt)
            tight_pad = float(max(0.0, _env_float("MIS_UVH_LOCAL_TIGHT_PAD", 8.0)))
            ax = int(round(x0 + rx - tight_pad))
            ay = int(round(y0 + ry - tight_pad))
            aw = int(round(rw + 2.0 * tight_pad))
            ah = int(round(rh + 2.0 * tight_pad))
            anchor_bbox = core.clamp_bbox(ax, ay, aw, ah, W, H)
    except Exception:
        anchor_bbox = tuple(int(v) for v in init_bbox)

    move_rad = int(max(0, _env_int("MIS_UVH_LOCAL_MOVE_RAD", 6)))
    move_step = int(max(1, _env_int("MIS_UVH_LOCAL_MOVE_STEP", 2)))
    scales = [float(v) for v in _parse_float_list("MIS_UVH_LOCAL_SCALE_LIST", "0.95,1.00,1.08,1.16,1.28,1.40,1.55")]
    if not scales:
        scales = [1.0]
    aspect_mults = [float(v) for v in _parse_float_list("MIS_UVH_LOCAL_ASPECT_LIST", "0.92,1.00,1.08")]
    if not aspect_mults:
        aspect_mults = [1.0]

    try:
        ref_mask64 = np.asarray(ref_mask_small, dtype=np.uint8)
        if ref_mask64.shape != (64, 64):
            ref_mask64 = cv2.resize(ref_mask64, (64, 64), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    except Exception:
        ref_mask64 = np.zeros((64, 64), dtype=np.uint8)

    use_sem = bool(_env_bool("MIS_UVH_LOCAL_USE_SEM", False))
    sem_w = float(max(0.0, min(1.0, _env_float("MIS_UVH_LOCAL_W_SEM", 0.15 if use_sem else 0.0))))
    quick_w = float(max(0.0, min(1.0, _env_float("MIS_UVH_LOCAL_W_QUICK", 0.60 if (not use_sem) else 0.45))))
    center_w = float(max(0.0, _env_float("MIS_UVH_LOCAL_W_CENTER", 0.06)))
    area_w = float(max(0.0, _env_float("MIS_UVH_LOCAL_W_AREA", 0.08)))
    aspect_w = float(max(0.0, _env_float("MIS_UVH_LOCAL_W_ASPECT", 0.05)))
    under_w = float(max(0.0, _env_float("MIS_UVH_LOCAL_W_UNDERCOVER", 0.22)))
    over_w = float(max(0.0, _env_float("MIS_UVH_LOCAL_W_OVERSIZE", 0.03)))

    unique_boxes: dict[tuple[int, int, int, int], None] = {}

    def _add_bbox(bb: tuple[int, int, int, int]) -> None:
        bx, by, bw, bh = [int(v) for v in bb]
        if bw >= 4 and bh >= 4:
            unique_boxes[(bx, by, bw, bh)] = None

    _add_bbox(tuple(int(v) for v in init_bbox))
    _add_bbox(tuple(int(v) for v in anchor_bbox))

    ax, ay, aw, ah = [int(v) for v in anchor_bbox]
    cx_a = ax + aw * 0.5
    cy_a = ay + ah * 0.5
    area0 = float(max(1, w0 * h0))
    aspect0 = float(max(1e-6, float(w0) / float(max(1, h0))))
    for sc in scales:
        for am in aspect_mults:
            am = float(max(0.5, min(2.0, am)))
            # 保持总体尺度由 sc 控制，同时允许长宽比独立调整
            bw = int(max(4, round(float(aw) * float(sc) * math.sqrt(am))))
            bh = int(max(4, round(float(ah) * float(sc) / math.sqrt(am))))
            for dy in range(-move_rad, move_rad + 1, move_step):
                for dx in range(-move_rad, move_rad + 1, move_step):
                    bx = int(round(cx_a + float(dx) - 0.5 * float(bw)))
                    by = int(round(cy_a + float(dy) - 0.5 * float(bh)))
                    _add_bbox(core.clamp_bbox(bx, by, bw, bh, W, H))

    best = None
    best_score = -1e9
    for bb in unique_boxes.keys():
        rebuilt = _rebuild_uvh_candidate_from_bbox(
            frame_bgr=frame_bgr,
            bbox=bb,
            cfg=cfg,
            ref_mask_small=ref_mask_small,
            search_ctx=search_ctx,
            cache=cache,
            frame_th_gray=frame_th_gray,
            frame_mask_u8=frame_mask_u8,
            frame_th_uint8=frame_th_uint8,
        )
        if not isinstance(rebuilt, dict):
            continue
        try:
            qmask = _mask64_from_mask_u8(np.asarray(rebuilt["mask_u8"], dtype=np.uint8))
            quick = float(core.quick_mask_score(ref_mask64, qmask))
        except Exception:
            quick = 0.0
        sem = float("nan")
        if float(sem_w) > 1e-12 and isinstance(ref_contour_rgb, np.ndarray):
            try:
                if use_sem:
                    sem = float(
                        core.semantic_similarity(
                            ref_contour_rgb,
                            np.asarray(rebuilt["contour_rgb"], dtype=np.uint8),
                            cfg,
                        )
                    )
                else:
                    sem = float(
                        core.fallback_similarity(
                            ref_contour_rgb,
                            np.asarray(rebuilt["contour_rgb"], dtype=np.uint8),
                        )
                    )
            except Exception:
                if use_sem:
                    try:
                        sem = float(
                            core.fallback_similarity(
                                ref_contour_rgb,
                                np.asarray(rebuilt["contour_rgb"], dtype=np.uint8),
                            )
                        )
                    except Exception:
                        sem = float("nan")
                else:
                    sem = float("nan")
        bx, by, bw, bh = [int(v) for v in rebuilt["bbox"]]
        cdx = float((bx + 0.5 * bw) - (x0 + 0.5 * w0))
        cdy = float((by + 0.5 * bh) - (y0 + 0.5 * h0))
        center_pen = float(math.hypot(cdx, cdy) / float(max(1.0, 0.5 * (w0 + h0))))
        area_ratio = float((bw * bh) / area0)
        area_pen = float(abs(math.log(max(1e-6, area_ratio))))
        under_pen = float(max(0.0, math.log(max(1e-6, 1.0 / max(1e-6, area_ratio))))) if area_ratio < 1.0 else 0.0
        over_pen = float(max(0.0, math.log(max(1e-6, area_ratio)))) if area_ratio > 1.0 else 0.0
        aspect_ratio = float(max(1e-6, float(bw) / float(max(1, bh))))
        aspect_pen = float(abs(math.log(max(1e-6, aspect_ratio / aspect0))))
        score = float(
            quick_w * quick
            + sem_w * (sem if np.isfinite(sem) else 0.0)
            - center_w * center_pen
            - area_w * area_pen
            - under_w * under_pen
            - over_w * over_pen
            - aspect_w * aspect_pen
        )
        rebuilt["uvh_local_bbox_score"] = float(score)
        rebuilt["uvh_local_quick"] = float(quick)
        rebuilt["uvh_local_sem"] = float(sem) if np.isfinite(sem) else float("nan")
        if float(score) > float(best_score):
            best_score = float(score)
            best = rebuilt
    return best


def _try_import_kaolin() -> tuple[Any, Any] | None:
    try:
        import torch  # type: ignore
        import kaolin  # type: ignore

        return torch, kaolin
    except Exception:
        return None


def _kaolin_pick_backend(*, kaolin_mod: Any, device: Any, height: int, width: int) -> str:
    """
    优先使用 nvdiffrast（速度快/可微），否则回退 kaolin 自带 cuda backend。
    注意：nvdiffrast(CUDA) 要求分辨率是 8 的倍数。
    """
    try:
        h8 = (int(height) % 8) == 0
        w8 = (int(width) % 8) == 0
        if h8 and w8:
            try:
                from kaolin.render.mesh import nvdiffrast_context  # type: ignore

                if bool(getattr(nvdiffrast_context, "nvdiffrast_is_available", lambda: False)()):
                    try:
                        getattr(nvdiffrast_context, "nvdiffrast_use_cuda", lambda: None)()
                    except Exception:
                        pass
                    return "nvdiffrast"
            except Exception:
                pass
    except Exception:
        pass
    return "cuda"


def _torch_orbit_camera_rot_trans(
    *,
    torch_mod: Any,
    yaw: Any,
    pitch: Any,
    roll: Any,
    dist: Any,
    dtype: Any,
    device: Any,
) -> tuple[Any, Any]:
    """
    构造 orbit camera（看向原点）：
      - yaw/pitch/roll: 弧度（可微）
      - dist: 正数（可微）
    返回 world->camera 的 (R,t)：
      p_cam = R @ p_world + t
    约定：前方点 z 为负（OpenGL-style）。
    """
    # direction on upper hemisphere
    cy = torch_mod.cos(yaw)
    sy = torch_mod.sin(yaw)
    cp = torch_mod.cos(pitch)
    sp = torch_mod.sin(pitch)
    d = torch_mod.stack([cp * cy, cp * sy, sp], dim=0).to(device=device, dtype=dtype)  # (3,)
    d = d / (torch_mod.norm(d) + 1e-9)
    eye = d * dist
    target = torch_mod.zeros((3,), device=device, dtype=dtype)
    up = torch_mod.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)

    # look-at (world->camera), with camera looking along -Z
    z_cam = eye - target
    z_cam = z_cam / (torch_mod.norm(z_cam) + 1e-9)  # camera backward axis
    # 指定 dim，避免 torch.cross 的未来行为变化警告
    x_cam = torch_mod.cross(up, z_cam, dim=0)
    x_cam = x_cam / (torch_mod.norm(x_cam) + 1e-9)
    y_cam = torch_mod.cross(z_cam, x_cam, dim=0)
    y_cam = y_cam / (torch_mod.norm(y_cam) + 1e-9)
    R = torch_mod.stack([x_cam, y_cam, z_cam], dim=0)  # rows
    t = -R @ eye

    # roll around camera Z axis (in camera coords): left-multiply
    cr = torch_mod.cos(roll)
    sr = torch_mod.sin(roll)
    Rz = torch_mod.stack(
        [
            torch_mod.stack([cr, -sr, torch_mod.zeros_like(cr)], dim=0),
            torch_mod.stack([sr, cr, torch_mod.zeros_like(cr)], dim=0),
            torch_mod.stack([torch_mod.zeros_like(cr), torch_mod.zeros_like(cr), torch_mod.ones_like(cr)], dim=0),
        ],
        dim=0,
    )
    R2 = Rz @ R
    t2 = Rz @ t
    return R2.unsqueeze(0), t2.unsqueeze(0)


def _torch_soft_iou(*, torch_mod: Any, a: Any, b: Any, eps: float = 1e-6) -> Any:
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    inter = torch_mod.sum(aa * bb)
    union = torch_mod.sum(aa) + torch_mod.sum(bb) - inter
    return inter / (union + float(eps))


def _torch_weighted_depth_corr01(*, torch_mod: Any, da: Any, db: Any, w: Any, eps: float = 1e-6) -> Any:
    ww = w.reshape(-1)
    aa = da.reshape(-1)
    bb = db.reshape(-1)
    ws = torch_mod.sum(ww) + float(eps)
    ma = torch_mod.sum(ww * aa) / ws
    mb = torch_mod.sum(ww * bb) / ws
    a0 = aa - ma
    b0 = bb - mb
    va = torch_mod.sum(ww * a0 * a0) / ws
    vb = torch_mod.sum(ww * b0 * b0) / ws
    cov = torch_mod.sum(ww * a0 * b0) / ws
    den = torch_mod.sqrt(torch_mod.clamp(va * vb, min=float(eps)))
    corr = cov / den
    corr = torch_mod.clamp(corr, -1.0, 1.0)
    return 0.5 * (corr + 1.0)


def _torch_softmask_to_contour_normed(
    *,
    torch_mod: Any,
    softmask: Any,
    edge_scale: float = 25.0,
) -> Any:
    """
    softmask(B,H,W) -> contour_rgb tensor(B,3,H,W) in [-1,1]（匹配语义模型 Normalize(mean=0.5,std=0.5) 的输出范围）
    """
    m = softmask
    if m.dim() == 2:
        m = m.unsqueeze(0)
    if m.dim() == 3:
        m = m.unsqueeze(1)  # (B,1,H,W)

    kx = torch_mod.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], device=m.device, dtype=m.dtype).view(1, 1, 3, 3) / 8.0
    ky = torch_mod.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], device=m.device, dtype=m.dtype).view(1, 1, 3, 3) / 8.0
    gx = torch_mod.nn.functional.conv2d(m, kx, padding=1)
    gy = torch_mod.nn.functional.conv2d(m, ky, padding=1)
    edge = torch_mod.sqrt(gx * gx + gy * gy + 1e-12)
    edge = torch_mod.tanh(edge * float(max(1e-6, edge_scale)))
    rgb01 = edge.repeat(1, 3, 1, 1)  # (B,3,H,W) in ~[0,1]
    rgb = rgb01 * 2.0 - 1.0  # [-1,1]
    return rgb


def _kaolin_dibr_render_softmask_depth(
    *,
    torch_mod: Any,
    kaolin_mod: Any,
    verts: Any,
    faces: Any,
    height: int,
    width: int,
    fovy_rad: float,
    camera_rot: Any,
    camera_trans: Any,
    backend: str,
    sigmainv: float = 7000.0,
    boxlen: float = 0.02,
    knum: int = 50,
) -> tuple[Any, Any]:
    """
    kaolin DIB-R 可微渲染：
      - 输出 softmask (B,H,W) 与 depth (B,H,W)
    """
    ratio = float(width) / float(max(1, int(height)))
    camera_proj = kaolin_mod.render.camera.generate_perspective_projection(float(fovy_rad), ratio=ratio, dtype=verts.dtype).to(device=verts.device)
    # vertices: (B,V,3) / faces: (F,3)
    if verts.dim() == 2:
        verts_b = verts.unsqueeze(0)
    else:
        verts_b = verts
    if faces.dim() != 2 or int(faces.shape[1]) != 3:
        raise ValueError("faces must be (F,3)")

    face_vertices_camera, face_vertices_image, _face_normals = kaolin_mod.render.mesh.prepare_vertices(
        verts_b, faces, camera_proj, camera_rot=camera_rot, camera_trans=camera_trans
    )
    # nvdiffrast 对输入的 contiguous 性要求更严格，这里统一显式 contiguous，避免运行时报错
    face_vertices_image = face_vertices_image.contiguous()
    face_vertices_z = face_vertices_camera[..., 2].contiguous()  # (B,F,3)
    face_features = face_vertices_z.unsqueeze(-1).contiguous()  # depth feature: (B,F,3,1)
    # 不做 backface culling（避免高度场面片朝向不一致导致缺面）
    B = int(face_vertices_z.shape[0])
    F = int(face_vertices_z.shape[1])
    face_normals_z = torch_mod.ones((B, F), device=verts_b.device, dtype=verts_b.dtype).contiguous()

    feat, softmask, _face_idx = kaolin_mod.render.mesh.dibr_rasterization(
        int(height),
        int(width),
        face_vertices_z,
        face_vertices_image,
        face_features,
        face_normals_z,
        sigmainv=float(sigmainv),
        boxlen=float(boxlen),
        knum=int(knum),
        rast_backend=str(backend),
    )
    depth = feat[..., 0]  # (B,H,W)
    softmask = torch_mod.clamp(softmask, 0.0, 1.0)
    return softmask, depth


def _uvh_diffrefine_with_kaolin(
    *,
    ref_verts_np: np.ndarray,
    ref_faces_np: np.ndarray,
    cur_verts_np: np.ndarray,
    cur_faces_np: np.ndarray,
    init_view_dir: Optional[np.ndarray],
    init_view_roll_deg: float,
    init_view_dist: float,
    init_scale: float,
    init_rot_deg: float,
    init_tx: float,
    init_ty: float,
    bbox_scale: float,
    cfg: core.TrackConfig,
) -> dict[str, Any]:
    """
    真 3D 可微渲染（Kaolin DIB-R）+ 反向优化视点(yaw/pitch/roll/dist) 与 in-plane (s,theta,tx,ty)。
    返回：包含 refined 分数与优化后的参数（用于候选 bbox 重排序/诊断）。
    """
    imp = _try_import_kaolin()
    if imp is None:
        return {"ok": False, "reason": "kaolin_not_available"}
    torch_mod, kaolin_mod = imp

    device_str = os.environ.get("MIS_UVH_DIFF_DEVICE", "cuda").strip().lower()
    device = torch_mod.device(device_str if (device_str in ("cuda", "cpu")) else "cuda")
    if device.type == "cuda" and (not torch_mod.cuda.is_available()):
        device = torch_mod.device("cpu")

    # render size: prefer semantic size (typically 128x128, multiple-of-8)
    W = int(cfg.semantic_image_size[0])
    H = int(cfg.semantic_image_size[1])
    if (H % 8 != 0) or (W % 8 != 0):
        # nvdiffrast 需要 8 的倍数；不满足时回退 cuda backend
        pass
    backend = _kaolin_pick_backend(kaolin_mod=kaolin_mod, device=device, height=H, width=W)

    dtype = torch_mod.float32
    ref_verts = torch_mod.as_tensor(np.asarray(ref_verts_np, dtype=np.float32), device=device, dtype=dtype)
    ref_faces = torch_mod.as_tensor(np.asarray(ref_faces_np, dtype=np.int64), device=device, dtype=torch_mod.int64)
    cur_verts0 = torch_mod.as_tensor(np.asarray(cur_verts_np, dtype=np.float32), device=device, dtype=dtype)
    cur_faces = torch_mod.as_tensor(np.asarray(cur_faces_np, dtype=np.int64), device=device, dtype=torch_mod.int64)

    if int(ref_verts.shape[0]) < 50 or int(ref_faces.shape[0]) < 50 or int(cur_verts0.shape[0]) < 50 or int(cur_faces.shape[0]) < 50:
        return {"ok": False, "reason": "mesh_too_small"}

    # centers (xy only)
    c_ref = torch_mod.mean(ref_verts[:, :2], dim=0, keepdim=False)
    c_cur = torch_mod.mean(cur_verts0[:, :2], dim=0, keepdim=False)
    c_ref3 = torch_mod.stack([c_ref[0], c_ref[1], torch_mod.tensor(0.0, device=device, dtype=dtype)], dim=0)
    ref_verts_c = ref_verts - c_ref3[None, :]

    # init view angles from dir
    if init_view_dir is None or (not np.isfinite(np.asarray(init_view_dir)).all()):
        d0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        d0 = np.asarray(init_view_dir, dtype=np.float32).reshape(3)
        dn = float(np.linalg.norm(d0))
        d0 = d0 / float(max(1e-9, dn))
    yaw0 = float(math.atan2(float(d0[1]), float(d0[0])))
    pitch0 = float(math.atan2(float(d0[2]), float(math.sqrt(float(d0[0] * d0[0] + d0[1] * d0[1])) + 1e-9)))
    roll0 = float(init_view_roll_deg) * math.pi / 180.0
    dist0 = float(init_view_dist)
    if not np.isfinite(dist0) or dist0 <= 1e-6:
        # crude fallback: 3 * radius
        pmin = torch_mod.amin(ref_verts_c, dim=0)
        pmax = torch_mod.amax(ref_verts_c, dim=0)
        rad = float(0.5 * torch_mod.linalg.norm((pmax - pmin)).item())
        dist0 = float(max(1e-3, 3.0 * rad))

    # init in-plane transform
    s0 = float(init_scale) if (np.isfinite(init_scale) and init_scale > 1e-6) else 1.0
    th0 = float(init_rot_deg) * math.pi / 180.0 if np.isfinite(init_rot_deg) else 0.0
    tx0 = float(init_tx) if np.isfinite(init_tx) else 0.0
    ty0 = float(init_ty) if np.isfinite(init_ty) else 0.0

    # optimization hyperparams
    iters = int(max(0, _env_int("MIS_UVH_DIFF_ITERS", 20)))
    lr = float(_env_float("MIS_UVH_DIFF_LR", 0.05))
    fovy_deg = float(_env_float("MIS_UVH_DIFF_FOVY_DEG", 55.0))
    fovy_rad = float(fovy_deg) * math.pi / 180.0

    dyaw_max = float(_env_float("MIS_UVH_DIFF_YAW_MAX_DEG", 35.0)) * math.pi / 180.0
    dpitch_max = float(_env_float("MIS_UVH_DIFF_PITCH_MAX_DEG", 35.0)) * math.pi / 180.0
    droll_max = float(_env_float("MIS_UVH_DIFF_ROLL_MAX_DEG", 45.0)) * math.pi / 180.0
    logd_max = float(_env_float("MIS_UVH_DIFF_LOGDIST_MAX", 0.7))  # exp(±0.7) ~= x2

    logs_max = float(_env_float("MIS_UVH_DIFF_LOGSCALE_MAX", 0.35))  # exp(±0.35) ~= x1.42
    dth_max = float(_env_float("MIS_UVH_DIFF_THETA_MAX_DEG", 25.0)) * math.pi / 180.0
    dxy_max = float(_env_float("MIS_UVH_DIFF_TRANS_MAX_PX", 18.0))

    # weights (normalized)；可微分支仅 IoU + depth 相关（无额外神经网络语义项）
    w_iou = float(_env_float("MIS_UVH_DIFF_W_IOU", 0.64))
    w_dep = float(_env_float("MIS_UVH_DIFF_W_DEPTHCORR", 0.36))
    ws = float(max(1e-9, abs(w_iou) + abs(w_dep)))
    w_iou, w_dep = float(w_iou) / ws, float(w_dep) / ws

    # penalties
    pen_area = float(max(0.0, _env_float("MIS_UVH_DIFF_PEN_W_AREA", 0.12)))
    pen_scale = float(max(0.0, _env_float("MIS_UVH_DIFF_PEN_W_SCALE", 0.08)))
    pen_bbox = float(max(0.0, _env_float("MIS_UVH_DIFF_PEN_W_BBOXSCALE", 0.0)))
    pen_view = float(max(0.0, _env_float("MIS_UVH_DIFF_PEN_W_VIEW", 0.02)))
    pen_trans = float(max(0.0, _env_float("MIS_UVH_DIFF_PEN_W_TRANS", 0.01)))

    # learnable deltas (centered at init)
    p_yaw = torch_mod.zeros((), device=device, dtype=dtype, requires_grad=True)
    p_pitch = torch_mod.zeros((), device=device, dtype=dtype, requires_grad=True)
    p_roll = torch_mod.zeros((), device=device, dtype=dtype, requires_grad=True)
    p_logd = torch_mod.zeros((), device=device, dtype=dtype, requires_grad=True)

    p_logs = torch_mod.zeros((), device=device, dtype=dtype, requires_grad=True)
    p_th = torch_mod.zeros((), device=device, dtype=dtype, requires_grad=True)
    p_tx = torch_mod.zeros((), device=device, dtype=dtype, requires_grad=True)
    p_ty = torch_mod.zeros((), device=device, dtype=dtype, requires_grad=True)

    opt = torch_mod.optim.Adam([p_yaw, p_pitch, p_roll, p_logd, p_logs, p_th, p_tx, p_ty], lr=float(max(1e-6, lr)))

    best = None
    best_sim = -1e9

    for _ in range(int(iters)):
        yaw = torch_mod.tensor(yaw0, device=device, dtype=dtype) + float(dyaw_max) * torch_mod.tanh(p_yaw)
        pitch = torch_mod.tensor(pitch0, device=device, dtype=dtype) + float(dpitch_max) * torch_mod.tanh(p_pitch)
        pitch = torch_mod.clamp(pitch, 0.0, 0.5 * math.pi)  # upper hemisphere
        roll = torch_mod.tensor(roll0, device=device, dtype=dtype) + float(droll_max) * torch_mod.tanh(p_roll)
        logd = float(logd_max) * torch_mod.tanh(p_logd)
        dist = torch_mod.tensor(dist0, device=device, dtype=dtype) * torch_mod.exp(logd)

        logs = float(logs_max) * torch_mod.tanh(p_logs)
        scale = torch_mod.tensor(s0, device=device, dtype=dtype) * torch_mod.exp(logs)
        th = torch_mod.tensor(th0, device=device, dtype=dtype) + float(dth_max) * torch_mod.tanh(p_th)
        tx = torch_mod.tensor(tx0, device=device, dtype=dtype) + float(dxy_max) * torch_mod.tanh(p_tx)
        ty = torch_mod.tensor(ty0, device=device, dtype=dtype) + float(dxy_max) * torch_mod.tanh(p_ty)

        # transform current verts to ref coords (xy only), then center by c_ref
        xy = cur_verts0[:, :2] - c_cur[None, :]
        ct = torch_mod.cos(th)
        st = torch_mod.sin(th)
        R2 = torch_mod.stack([torch_mod.stack([ct, -st]), torch_mod.stack([st, ct])], dim=0)  # (2,2)
        xy2 = (xy @ R2.T) * scale + c_ref[None, :]
        xy2 = xy2 + torch_mod.stack([tx, ty])[None, :]
        z2 = cur_verts0[:, 2:3]
        cur_verts_al = torch_mod.cat([xy2, z2], dim=1) - c_ref3[None, :]

        cam_R, cam_t = _torch_orbit_camera_rot_trans(torch_mod=torch_mod, yaw=yaw, pitch=pitch, roll=roll, dist=dist, dtype=dtype, device=device)

        m_ref, d_ref = _kaolin_dibr_render_softmask_depth(
            torch_mod=torch_mod,
            kaolin_mod=kaolin_mod,
            verts=ref_verts_c,
            faces=ref_faces,
            height=H,
            width=W,
            fovy_rad=float(fovy_rad),
            camera_rot=cam_R,
            camera_trans=cam_t,
            backend=str(backend),
        )
        m_cur, d_cur = _kaolin_dibr_render_softmask_depth(
            torch_mod=torch_mod,
            kaolin_mod=kaolin_mod,
            verts=cur_verts_al,
            faces=cur_faces,
            height=H,
            width=W,
            fovy_rad=float(fovy_rad),
            camera_rot=cam_R,
            camera_trans=cam_t,
            backend=str(backend),
        )

        iou = _torch_soft_iou(torch_mod=torch_mod, a=m_ref, b=m_cur)
        w = (m_ref * m_cur).detach() + 1e-6  # stabilize; grads mainly from masks/depth anyway
        dc = _torch_weighted_depth_corr01(torch_mod=torch_mod, da=d_ref, db=d_cur, w=w)

        # penalties
        ar = (torch_mod.sum(m_cur) + 1.0) / (torch_mod.sum(m_ref) + 1.0)
        area_pen = torch_mod.abs(torch_mod.log(torch_mod.clamp(ar, min=1e-6)))
        scale_pen = torch_mod.abs(torch_mod.log(torch_mod.clamp(scale / float(max(1e-6, s0)), min=1e-6)))
        bbox_pen = torch_mod.tensor(0.0, device=device, dtype=dtype)
        if np.isfinite(bbox_scale):
            bbox_pen = torch_mod.abs(torch_mod.log(torch_mod.tensor(float(max(1e-6, bbox_scale)), device=device, dtype=dtype)))

        view_pen = (torch_mod.tanh(p_yaw) ** 2 + torch_mod.tanh(p_pitch) ** 2 + torch_mod.tanh(p_roll) ** 2 + torch_mod.tanh(p_logd) ** 2)
        trans_pen = (torch_mod.tanh(p_tx) ** 2 + torch_mod.tanh(p_ty) ** 2 + torch_mod.tanh(p_logs) ** 2 + torch_mod.tanh(p_th) ** 2)

        sim = float(w_iou) * iou + float(w_dep) * dc
        sim = sim - float(pen_area) * area_pen - float(pen_scale) * scale_pen - float(pen_bbox) * bbox_pen - float(pen_view) * view_pen - float(pen_trans) * trans_pen

        loss = -sim
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        s_val = float(sim.detach().item()) if hasattr(sim, "detach") else float(sim)
        if np.isfinite(s_val) and s_val > best_sim:
            best_sim = float(s_val)
            best = {
                "ok": True,
                "backend": str(backend),
                "sim": float(max(0.0, min(1.0, s_val))),
                "iou": float(iou.detach().item()),
                "depth_corr": float(dc.detach().item()),
                "sem": float("nan"),
                "sem_used": 0,
                "yaw_deg": float(yaw.detach().item() * 180.0 / math.pi),
                "pitch_deg": float(pitch.detach().item() * 180.0 / math.pi),
                "roll_deg": float(roll.detach().item() * 180.0 / math.pi),
                "dist": float(dist.detach().item()),
                "scale": float(scale.detach().item()),
                "rot_deg": float(th.detach().item() * 180.0 / math.pi),
                "tx": float(tx.detach().item()),
                "ty": float(ty.detach().item()),
                "area_ratio": float(ar.detach().item()),
            }

    if best is None:
        return {"ok": False, "reason": "opt_failed"}
    best["iters"] = int(iters)
    best["lr"] = float(lr)
    return best


@dataclass
class ViewLibrary:
    root_dir: str
    image_wh: tuple[int, int]
    K: np.ndarray  # (3,3) float32
    mask_L: int
    masks_packed: np.ndarray  # (N,B) uint8；B=ceil(L/8)，bit-pack 的二值轮廓模板
    sums: np.ndarray  # (N,) int32
    means: np.ndarray  # (N,) float32
    vars: np.ndarray  # (N,) float32
    dirs: np.ndarray  # (N,3) float32 (camera position direction from center)
    dists: np.ndarray  # (N,) float32
    rolls: np.ndarray  # (N,) float32 (camera roll degrees)
    T_w2c: np.ndarray  # (N,4,4) float32
    contour_rgb_paths: list[str]  # absolute paths (len=N)
    ids: np.ndarray  # (N,) int32


def _save_view_library(
    *,
    lib_dir: str,
    meta: dict[str, Any],
    lib: ViewLibrary,
) -> None:
    _ensure_dir(lib_dir)
    meta_path = os.path.join(lib_dir, "view_library_meta.json")
    npz_path = os.path.join(lib_dir, "view_library_templates.npz")
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    try:
        rel = []
        for p in lib.contour_rgb_paths:
            try:
                rel.append(os.path.relpath(p, lib_dir))
            except Exception:
                rel.append(str(p))
        np.savez_compressed(
            npz_path,
            image_wh=np.asarray([int(lib.image_wh[0]), int(lib.image_wh[1])], dtype=np.int32),
            K=np.asarray(lib.K, dtype=np.float32).reshape(3, 3),
            mask_L=np.asarray(int(lib.mask_L), dtype=np.int32),
            mask_bitorder=np.asarray(str(_MASK_BITORDER), dtype=str),
            masks_packed=np.asarray(lib.masks_packed, dtype=np.uint8),
            sums=np.asarray(lib.sums, dtype=np.int32),
            means=np.asarray(lib.means, dtype=np.float32),
            vars=np.asarray(lib.vars, dtype=np.float32),
            dirs=np.asarray(lib.dirs, dtype=np.float32),
            dists=np.asarray(lib.dists, dtype=np.float32),
            rolls=np.asarray(lib.rolls, dtype=np.float32),
            T_w2c=np.asarray(lib.T_w2c, dtype=np.float32),
            contour_rgb_relpaths=np.asarray(rel, dtype=str),
        )
    except Exception:
        pass


def _load_view_library(lib_dir: str) -> ViewLibrary | None:
    npz_path = os.path.join(lib_dir, "view_library_templates.npz")
    if not os.path.isfile(npz_path):
        return None
    try:
        z = np.load(npz_path, allow_pickle=False)
        wh = np.asarray(z["image_wh"], dtype=np.int32).reshape(-1)
        image_wh = (int(wh[0]), int(wh[1]))
        K = np.asarray(z["K"], dtype=np.float32).reshape(3, 3)

        mask_L: int
        masks_packed: np.ndarray
        if "masks_packed" in getattr(z, "files", []):
            masks_packed = np.asarray(z["masks_packed"], dtype=np.uint8)
            if "mask_L" in getattr(z, "files", []):
                try:
                    mask_L = int(np.asarray(z["mask_L"], dtype=np.int32).reshape(-1)[0])
                except Exception:
                    mask_L = int(image_wh[0] * image_wh[1])
            else:
                mask_L = int(image_wh[0] * image_wh[1])
        elif "masks_flat" in getattr(z, "files", []):
            mf = np.asarray(z["masks_flat"], dtype=np.uint8)
            mask_L = int(mf.shape[1]) if mf.ndim == 2 else int(image_wh[0] * image_wh[1])
            mf2 = (mf > 0).astype(np.uint8) if mf.ndim == 2 else np.zeros((0, int(mask_L)), dtype=np.uint8)
            masks_packed = np.packbits(mf2, axis=1, bitorder=_MASK_BITORDER).astype(np.uint8)
        else:
            return None

        sums = np.asarray(z["sums"], dtype=np.int32).reshape(-1) if ("sums" in getattr(z, "files", [])) else np.zeros((int(masks_packed.shape[0]),), dtype=np.int32)
        if "means" in getattr(z, "files", []):
            means = np.asarray(z["means"], dtype=np.float32).reshape(-1)
        else:
            means = (sums.astype(np.float32) / float(max(1, int(mask_L)))).astype(np.float32)
        if "vars" in getattr(z, "files", []):
            vars_ = np.asarray(z["vars"], dtype=np.float32).reshape(-1)
        else:
            vars_ = (means * (1.0 - means)).astype(np.float32)
        dirs = np.asarray(z["dirs"], dtype=np.float32).reshape(-1, 3)
        dists = np.asarray(z["dists"], dtype=np.float32).reshape(-1)
        if "rolls" in getattr(z, "files", []):
            try:
                rolls = np.asarray(z["rolls"], dtype=np.float32).reshape(-1)
            except Exception:
                rolls = np.zeros_like(dists, dtype=np.float32)
        else:
            rolls = np.zeros_like(dists, dtype=np.float32)
        T_w2c = np.asarray(z["T_w2c"], dtype=np.float32).reshape(-1, 4, 4)
        rel = [str(s) for s in np.asarray(z["contour_rgb_relpaths"]).reshape(-1).tolist()]
        paths = [os.path.join(lib_dir, p) if (not os.path.isabs(p)) else p for p in rel]
        n = int(masks_packed.shape[0])
        ids = np.arange(n, dtype=np.int32)
        return ViewLibrary(
            root_dir=str(lib_dir),
            image_wh=image_wh,
            K=K,
            mask_L=int(mask_L),
            masks_packed=masks_packed,
            sums=sums,
            means=means,
            vars=vars_,
            dirs=dirs,
            dists=dists,
            rolls=rolls,
            T_w2c=T_w2c,
            contour_rgb_paths=paths,
            ids=ids,
        )
    except Exception:
        return None


def build_or_load_view_library(
    *,
    out_root: str,
    pts_w: np.ndarray,
    cfg: core.TrackConfig,
    n_views: int,
    n_dirs: int = 0,
    dist_mults: list[float],
    dist_jitter: float,
    roll_deg_list: list[float] | None = None,
    z_bias: float = 1.0,
    sampler: str = "combo",
    seed: int,
    force_rebuild: bool,
    save_depth_vis: bool = True,
    save_mask_png: bool = True,
    save_view_points: bool = True,
    view_points_n: int = 4096,
) -> tuple[ViewLibrary | None, dict[str, Any]]:
    """
    生成/加载“上半球多视点目标库”：
      - 每个视点输出 contour_rgb（用于后续匹配）+ 可选 depth 可视化/ mask
      - 同时在 npz 中保存模板的二值轮廓 mask（扁平化，便于向量化匹配）
    """
    lib_dir = _ensure_dir(os.path.join(out_root, "target_library"))
    meta_path = os.path.join(lib_dir, "view_library_meta.json")
    npz_path = os.path.join(lib_dir, "view_library_templates.npz")

    # try load
    if (not force_rebuild) and os.path.isfile(npz_path) and os.path.isfile(meta_path):
        lib = _load_view_library(lib_dir)
        meta = _safe_read_json(meta_path)
        if lib is not None and isinstance(meta, dict):
            try:
                print(f"[TargetLibrary] loaded cache: views={int(lib.ids.size)} image={int(lib.image_wh[0])}x{int(lib.image_wh[1])} dir={lib_dir}", flush=True)
            except Exception:
                pass
            return lib, meta

    pts = np.asarray(pts_w, dtype=np.float32).reshape(-1, 3)
    if pts.size == 0:
        return None, {"error": "empty_pts"}

    # image size aligned to cfg.semantic_image_size by default
    w, h = int(cfg.semantic_image_size[0]), int(cfg.semantic_image_size[1])
    image_wh = (w, h)
    K = _default_K_for_library(image_wh)

    # pointcloud stats
    pmin = np.nanmin(pts, axis=0)
    pmax = np.nanmax(pts, axis=0)
    center = 0.5 * (pmin + pmax)
    extent = (pmax - pmin)
    radius = float(0.5 * np.linalg.norm(extent))
    radius = float(max(1e-3, radius))

    sampler_l = (sampler or "combo").strip().lower()
    if sampler_l not in ("combo", "shell"):
        sampler_l = "combo"

    # dirs & distances & rolls（更大尺度 + 更多旋转角）
    n_views_req = int(max(1, int(n_views)))
    dist_mults2 = [float(v) for v in dist_mults if np.isfinite(float(v)) and float(v) > 0]
    if len(dist_mults2) == 0:
        dist_mults2 = [2.2, 2.8, 3.4]
    # shell sampler 时 dist_mults2 代表“半径层”，需要排序；combo sampler 则无所谓
    if sampler_l == "shell":
        dist_mults2 = sorted(set([float(v) for v in dist_mults2]))
    dist_mults2 = dist_mults2[:256]
    dmult = np.asarray(dist_mults2, dtype=np.float32).reshape(-1)  # shell multipliers or discrete dist list

    rolls_in = roll_deg_list if roll_deg_list is not None else [0.0]
    rolls2 = [float(v) for v in rolls_in if np.isfinite(float(v))]
    if len(rolls2) == 0:
        rolls2 = [0.0]
    rolls2 = rolls2[:32]
    rdeg = np.asarray(rolls2, dtype=np.float32).reshape(-1)

    rng = np.random.default_rng(int(seed))
    n_d = int(dmult.shape[0])
    n_r = int(rdeg.shape[0])

    if sampler_l == "shell":
        # 分层均匀：按半径层均分视点数；每个半径层在上半球表面均匀采样方向
        n_shells = int(max(1, n_d))
        base = int(n_views_req // n_shells)
        rem = int(n_views_req % n_shells)
        counts = [base + (1 if s < rem else 0) for s in range(n_shells)]
        dirs = np.zeros((int(n_views_req), 3), dtype=np.float32)
        rolls = np.zeros((int(n_views_req),), dtype=np.float32)
        dists = np.zeros((int(n_views_req),), dtype=np.float32)

        k = 0
        j = float(max(0.0, float(dist_jitter)))
        for s in range(n_shells):
            ns = int(counts[s])
            if ns <= 0:
                continue
            # 每层方向均匀采样；加一个确定性的 theta_offset 让不同层的 azimuth 纹理不完全重复
            th0 = 2.0 * math.pi * (float(s) / float(max(1, n_shells)))
            ds = _fibonacci_hemisphere_dirs(ns, z_bias=float(z_bias), theta_offset=float(th0)).astype(np.float32)
            dirs[k:k + ns] = ds
            # roll 均匀分配（循环），并对不同层做 shift，避免 roll 与方向固定绑定
            ridx = (np.arange(ns, dtype=np.int32) + int(s)) % int(max(1, n_r))
            rolls[k:k + ns] = rdeg[ridx].astype(np.float32)
            dm = float(dmult[int(min(s, n_d - 1))])
            dm_arr = np.full((ns,), float(dm), dtype=np.float32)
            if j > 1e-9:
                dm_arr = dm_arr * (1.0 + float(j) * rng.uniform(-1.0, 1.0, size=ns).astype(np.float32))
            dists[k:k + ns] = (dm_arr * float(radius)).astype(np.float32)
            k += ns
        n_views = int(n_views_req)
        n_dirs_eff = int(max(counts)) if counts else int(n_views_req)
    else:
        # combo：dir x dist x roll 的组合空间里随机抽样（覆盖更杂、更随机）
        n_dirs_eff = int(max(1, int(n_dirs))) if int(n_dirs) > 0 else int(n_views_req)
        dirs_base = _fibonacci_hemisphere_dirs(n_dirs_eff, z_bias=float(z_bias))
        total = int(n_dirs_eff) * int(n_d) * int(n_r)
        if total <= 0:
            return None, {"error": "invalid_view_params"}
        n_views_eff = int(min(int(n_views_req), int(total)))
        if n_views_eff < int(n_views_req):
            print(f"[TargetLibrary][Warning] requested n_views={n_views_req} but combos_total={total} -> using n_views={n_views_eff}")

        idx_all = np.arange(int(total), dtype=np.int64)
        rng.shuffle(idx_all)
        idx_pick = idx_all[: int(n_views_eff)]
        block = int(n_d) * int(n_r)
        dir_idx = (idx_pick // int(block)).astype(np.int32)
        rem2 = (idx_pick % int(block)).astype(np.int32)
        dist_idx = (rem2 // int(n_r)).astype(np.int32)
        roll_idx = (rem2 % int(n_r)).astype(np.int32)

        dirs = dirs_base[dir_idx].astype(np.float32)
        rolls = rdeg[roll_idx].astype(np.float32)
        dm_sel = dmult[dist_idx].astype(np.float32)
        j = float(max(0.0, float(dist_jitter)))
        if j > 1e-9:
            dm_sel = dm_sel * (1.0 + float(j) * rng.uniform(-1.0, 1.0, size=int(n_views_eff)).astype(np.float32))
        dists = (dm_sel.astype(np.float32) * float(radius)).astype(np.float32)
        n_views = int(n_views_eff)

    # output dirs
    out_renders = _ensure_dir(os.path.join(lib_dir, "renders"))
    out_cont = _ensure_dir(os.path.join(out_renders, "contour_rgb"))
    out_depth = _ensure_dir(os.path.join(out_renders, "depth_vis")) if save_depth_vis else out_renders
    out_mask = _ensure_dir(os.path.join(out_renders, "mask")) if save_mask_png else out_renders
    out_view_pts = _ensure_dir(os.path.join(lib_dir, "view_points_cam")) if save_view_points else ""

    # save world point-cloud once (作为“伪3D模型结果”的基准)
    try:
        _save_point_cloud_ply_ascii(
            os.path.join(lib_dir, "uvh_points_world.ply"),
            pts_xyz=np.asarray(pts, dtype=np.float32),
            colors_u8=_colorize_points_by_height(np.asarray(pts, dtype=np.float32), low_rgb=(40, 160, 255), high_rgb=(255, 220, 90)),
        )
    except Exception:
        pass

    # optional: fixed subset for per-view camera-space pointcloud export
    pts_sub = None
    vp_n = int(max(0, int(view_points_n)))
    if save_view_points and vp_n > 0:
        try:
            n_all = int(pts.shape[0])
            if n_all > vp_n:
                idx_sub = rng.choice(n_all, size=vp_n, replace=False)
                pts_sub = pts[idx_sub].astype(np.float32)
            else:
                pts_sub = pts.astype(np.float32)
        except Exception:
            pts_sub = None

    # 重要：64k 级别库必须避免 (N,L) 的 uint8 全展开（内存会爆）
    # 这里统一用 packbits 存储模板（二值轮廓线），并用 popcount 做向量化匹配。
    mask_L = int(w * h)
    n_bytes = int((int(mask_L) + 7) // 8)
    masks_packed = np.zeros((int(n_views), int(n_bytes)), dtype=np.uint8)
    sums2 = np.zeros((int(n_views),), dtype=np.int32)
    means2 = np.zeros((int(n_views),), dtype=np.float32)
    vars2 = np.zeros((int(n_views),), dtype=np.float32)
    Ts2 = np.zeros((int(n_views), 4, 4), dtype=np.float32)
    contour_paths: list[str] = [""] * int(n_views)

    L = int(mask_L)
    pad = max(4, len(str(max(0, int(n_views) - 1))))
    print(
        f"[TargetLibrary] sampler={sampler_l} views={n_views} (dirs={n_dirs_eff}, shells/dists={n_d}, rolls={n_r}, z_bias={float(z_bias):g}) "
        f"image={w}x{h} pts={int(pts.shape[0])} radius≈{radius:.2f}"
    )
    for i in range(n_views):
        d = float(dists[i])
        eye = center + dirs[i].astype(np.float32) * float(d)
        T_w2c = _look_at_T_w2c(eye_w=eye, target_w=center, roll_deg=float(rolls[i]))

        mask_u8, depth = _render_mask_depth_from_points(pts_w=pts, T_w2c=T_w2c, K=K, image_wh=image_wh)
        contour_rgb, filled = _mask_to_contour_rgb(mask_u8, image_wh, thickness=2)

        # save per-view camera-space 3D points（可理解为“每个 target 的 3D 模型结果”）
        if save_view_points and pts_sub is not None and out_view_pts:
            try:
                T = np.asarray(T_w2c, dtype=np.float32).reshape(4, 4)
                R = T[:3, :3].astype(np.float32)
                t = T[:3, 3].astype(np.float32)
                pc = (R @ pts_sub.T).T + t[None, :]
                _save_point_cloud_ply_ascii(
                    os.path.join(out_view_pts, f"pts_cam_{i:0{pad}d}.ply"),
                    pts_xyz=np.asarray(pc, dtype=np.float32),
                    colors_u8=_colorize_points_by_height(np.asarray(pc, dtype=np.float32), low_rgb=(180, 80, 255), high_rgb=(80, 255, 220)),
                )
            except Exception:
                pass

        # save images
        cpath = os.path.join(out_cont, f"contour_rgb_{i:0{pad}d}.png")
        try:
            cv2.imwrite(cpath, contour_rgb)
        except Exception:
            pass
        contour_paths[int(i)] = str(cpath)

        if save_mask_png:
            try:
                cv2.imwrite(os.path.join(out_mask, f"mask_{i:0{pad}d}.png"), filled)
            except Exception:
                pass

        if save_depth_vis:
            try:
                dep = np.asarray(depth, dtype=np.float32)
                m = np.isfinite(dep)
                if np.any(m):
                    v = dep[m]
                    d0 = float(np.nanmin(v))
                    d1 = float(np.nanmax(v))
                    den = (d1 - d0) if (d1 - d0) > 1e-6 else 1.0
                    dn = (dep - d0) / den
                    dn = np.clip(dn, 0.0, 1.0)
                    u8 = (255.0 * (1.0 - dn)).astype(np.uint8)
                    u8[~m] = 0
                else:
                    u8 = np.zeros((h, w), dtype=np.uint8)
                col = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
                cv2.imwrite(os.path.join(out_depth, f"depth_{i:0{pad}d}.png"), col)
            except Exception:
                pass

        # template mask for matching (binary contour line)
        mk = (np.asarray(contour_rgb[..., 0], dtype=np.uint8) > 0).astype(np.uint8).reshape(-1)
        if int(mk.size) != int(mask_L):
            # 保险：尺寸不一致时做截断/补零（理论上不应发生）
            if int(mk.size) > int(mask_L):
                mk = mk[: int(mask_L)]
            else:
                mk2 = np.zeros((int(mask_L),), dtype=np.uint8)
                mk2[: int(mk.size)] = mk
                mk = mk2
        s = int(np.sum(mk))
        mu = float(s) / float(max(1, L))
        var = float(mu * (1.0 - mu))
        try:
            masks_packed[int(i)] = np.packbits(mk.astype(np.uint8, copy=False), bitorder=_MASK_BITORDER).reshape(-1)[: int(n_bytes)]
        except Exception:
            # 极端兜底：直接清零
            masks_packed[int(i)] = 0
        sums2[int(i)] = int(s)
        means2[int(i)] = float(mu)
        vars2[int(i)] = float(var)
        Ts2[int(i)] = np.asarray(T_w2c, dtype=np.float32).reshape(4, 4)

        if (i + 1) % 128 == 0:
            print(f"[TargetLibrary] {i+1}/{n_views} done...", flush=True)

    # arrays already filled in-place (packed)

    lib = ViewLibrary(
        root_dir=str(lib_dir),
        image_wh=image_wh,
        K=K,
        mask_L=int(mask_L),
        masks_packed=masks_packed,
        sums=sums2,
        means=means2,
        vars=vars2,
        dirs=np.asarray(dirs, dtype=np.float32),
        dists=np.asarray(dists, dtype=np.float32),
        rolls=np.asarray(rolls, dtype=np.float32),
        T_w2c=Ts2,
        contour_rgb_paths=contour_paths,
        ids=np.arange(int(n_views), dtype=np.int32),
    )

    meta: dict[str, Any] = {
        "time": _now(),
        "sampler": str(sampler_l),
        "n_views": int(n_views),
        "n_dirs": int(n_dirs_eff),
        "n_dists": int(n_d),
        "n_rolls": int(n_r),
        "z_bias": float(z_bias),
        "image_wh": [int(w), int(h)],
        "seed": int(seed),
        "dist_mults": [float(v) for v in dist_mults2],
        "roll_deg_list": [float(v) for v in rolls2],
        "dist_jitter": float(dist_jitter),
        "K": lib.K.reshape(-1).tolist(),
        "save_view_points": int(bool(save_view_points)),
        "view_points_n": int(vp_n),
        "pc_stats": {
            "n_points": int(pts.shape[0]),
            "center": [float(center[0]), float(center[1]), float(center[2])],
            "extent": [float(extent[0]), float(extent[1]), float(extent[2])],
            "radius": float(radius),
        },
        "note": "View templates are produced by projecting uvh point-cloud (pseudo-3D heightfield) onto synthetic cameras on an upper-hemisphere.",
    }

    _save_view_library(lib_dir=lib_dir, meta=meta, lib=lib)
    return lib, meta


def match_contour_to_library(
    *,
    lib: ViewLibrary,
    contour_rgb: np.ndarray,
    prev_view_dir: np.ndarray | None,
    angle_gate_deg: float,
) -> dict[str, Any]:
    """
    对输入 contour_rgb，在目标库中找最佳匹配（向量化 fallback-sim）。
    返回：
      view_id, score, iou, corr, n_used, used_gate
    """
    mk = (np.asarray(contour_rgb[..., 0], dtype=np.uint8) > 0).astype(np.uint8).reshape(-1)
    L_in = int(mk.size)
    L = int(getattr(lib, "mask_L", L_in))
    if L <= 0:
        return {"view_id": -1, "score": 0.0, "iou": 0.0, "corr": 0.0, "n_used": 0, "used_gate": 0}
    if L_in != int(L):
        # 保险：尺寸不一致则截断/补零（理论上 lib.image_wh 应与 contour_rgb 一致）
        if L_in > int(L):
            mk = mk[: int(L)]
        else:
            mk2 = np.zeros((int(L),), dtype=np.uint8)
            mk2[: int(L_in)] = mk
            mk = mk2
    sB = int(np.sum(mk))
    if sB <= 2:
        return {"view_id": -1, "score": 0.0, "iou": 0.0, "corr": 0.0, "n_used": 0, "used_gate": 0}

    muB = float(sB) / float(max(1, L))
    varB = float(muB * (1.0 - muB))
    if varB <= 1e-12:
        varB = 0.0

    idx = None
    used_gate = 0
    n_used = int(lib.masks_packed.shape[0])
    if prev_view_dir is not None and np.isfinite(np.asarray(prev_view_dir)).all() and float(angle_gate_deg) > 1e-6:
        try:
            pv = np.asarray(prev_view_dir, dtype=np.float32).reshape(3)
            den = float(np.linalg.norm(pv))
            if den > 1e-9:
                pv = pv / den
                cos_thr = float(math.cos(float(angle_gate_deg) * math.pi / 180.0))
                dots = (np.asarray(lib.dirs, dtype=np.float32) @ pv.reshape(3, 1)).reshape(-1)
                sel = np.flatnonzero(dots >= cos_thr)
                if sel.size >= 32:
                    idx = sel.astype(np.int32)
                    used_gate = 1
                    n_used = int(sel.size)
        except Exception:
            idx = None

    if idx is None:
        A = lib.masks_packed
        sumsA = lib.sums
        muA = lib.means
        varA = lib.vars
        ids = lib.ids
    else:
        A = lib.masks_packed[idx]
        sumsA = lib.sums[idx]
        muA = lib.means[idx]
        varA = lib.vars[idx]
        ids = lib.ids[idx]

    # packbits + popcount：避免 (N,L) 展开造成巨大内存/耗时（64k 级别必须）
    mk_p = np.packbits(mk.astype(np.uint8, copy=False), bitorder=_MASK_BITORDER).reshape(-1)
    nb = int(A.shape[1])
    if int(mk_p.size) != int(nb):
        if int(mk_p.size) > int(nb):
            mk_p = mk_p[: int(nb)]
        else:
            mk2 = np.zeros((int(nb),), dtype=np.uint8)
            mk2[: int(mk_p.size)] = mk_p
            mk_p = mk2

    N = int(A.shape[0])
    block = int(max(256, _env_int("MIS_LIBRARY_MATCH_BLOCK", 4096)))
    inter_i = np.zeros((N,), dtype=np.int32)
    for s in range(0, N, block):
        e = int(min(N, s + block))
        andb = np.bitwise_and(A[s:e], mk_p)  # (B,nb)
        inter_i[s:e] = _POPCOUNT_LUT[andb].sum(axis=1).astype(np.int32)

    inter = inter_i.astype(np.float32)  # (N,)
    union = (sumsA.astype(np.float32) + float(sB) - inter)
    iou = inter / (union + 1e-6)

    # corr: cov / sqrt(varA*varB), cov = E[AB] - muA*muB, E[AB]=inter/L
    if varB <= 1e-12:
        corr01 = np.zeros_like(iou, dtype=np.float32)
        corr_raw = np.zeros_like(iou, dtype=np.float32)
    else:
        if int(muA.shape[0]) != int(sumsA.shape[0]):
            muA = (sumsA.astype(np.float32) / float(max(1, int(L)))).astype(np.float32)
        if int(varA.shape[0]) != int(sumsA.shape[0]):
            varA = (muA * (1.0 - muA)).astype(np.float32)
        Eab = inter / float(max(1, L))
        cov = Eab - (muA.astype(np.float32) * float(muB))
        den = np.sqrt(np.maximum(1e-12, varA.astype(np.float32) * float(varB)))
        corr_raw = cov / den
        corr_raw = np.clip(corr_raw, -1.0, 1.0)
        corr01 = 0.5 * (corr_raw + 1.0)

    sim = (0.55 * iou + 0.45 * corr01).astype(np.float32)
    j = int(np.argmax(sim)) if sim.size else -1
    best_id = int(ids[j]) if j >= 0 else -1
    return {
        "view_id": best_id,
        "score": float(sim[j]) if j >= 0 else 0.0,
        "iou": float(iou[j]) if j >= 0 else 0.0,
        "corr": float(corr01[j]) if j >= 0 else 0.0,
        "n_used": int(n_used),
        "used_gate": int(used_gate),
    }


def match_contour_to_library_topk(
    *,
    lib: ViewLibrary,
    contour_rgb: np.ndarray,
    prev_view_dir: np.ndarray | None,
    angle_gate_deg: float,
    topk: int = 3,
) -> dict[str, Any]:
    """
    与 match_contour_to_library 类似，但返回 top-k 视点（按 score 降序）。
    返回：
      view_ids(list[int]), scores(list[float]), ious(list[float]), corrs(list[float]),
      n_used, used_gate
    """
    k = int(max(1, int(topk)))
    mk = (np.asarray(contour_rgb[..., 0], dtype=np.uint8) > 0).astype(np.uint8).reshape(-1)
    L_in = int(mk.size)
    L = int(getattr(lib, "mask_L", L_in))
    if L <= 0:
        return {"view_ids": [], "scores": [], "ious": [], "corrs": [], "n_used": 0, "used_gate": 0}
    if L_in != int(L):
        if L_in > int(L):
            mk = mk[: int(L)]
        else:
            mk2 = np.zeros((int(L),), dtype=np.uint8)
            mk2[: int(L_in)] = mk
            mk = mk2
    sB = int(np.sum(mk))
    if sB <= 2:
        return {"view_ids": [], "scores": [], "ious": [], "corrs": [], "n_used": 0, "used_gate": 0}

    muB = float(sB) / float(max(1, L))
    varB = float(muB * (1.0 - muB))
    if varB <= 1e-12:
        varB = 0.0

    idx = None
    used_gate = 0
    n_used = int(lib.masks_packed.shape[0])
    if prev_view_dir is not None and np.isfinite(np.asarray(prev_view_dir)).all() and float(angle_gate_deg) > 1e-6:
        try:
            pv = np.asarray(prev_view_dir, dtype=np.float32).reshape(3)
            den = float(np.linalg.norm(pv))
            if den > 1e-9:
                pv = pv / den
                cos_thr = float(math.cos(float(angle_gate_deg) * math.pi / 180.0))
                dots = (np.asarray(lib.dirs, dtype=np.float32) @ pv.reshape(3, 1)).reshape(-1)
                sel = np.flatnonzero(dots >= cos_thr)
                if sel.size >= 32:
                    idx = sel.astype(np.int32)
                    used_gate = 1
                    n_used = int(sel.size)
        except Exception:
            idx = None

    if idx is None:
        A = lib.masks_packed
        sumsA = lib.sums
        muA = lib.means
        varA = lib.vars
        ids = lib.ids
    else:
        A = lib.masks_packed[idx]
        sumsA = lib.sums[idx]
        muA = lib.means[idx]
        varA = lib.vars[idx]
        ids = lib.ids[idx]

    mk_p = np.packbits(mk.astype(np.uint8, copy=False), bitorder=_MASK_BITORDER).reshape(-1)
    nb = int(A.shape[1])
    if int(mk_p.size) != int(nb):
        if int(mk_p.size) > int(nb):
            mk_p = mk_p[: int(nb)]
        else:
            mk2 = np.zeros((int(nb),), dtype=np.uint8)
            mk2[: int(mk_p.size)] = mk_p
            mk_p = mk2

    N = int(A.shape[0])
    if N <= 0:
        return {"view_ids": [], "scores": [], "ious": [], "corrs": [], "n_used": int(n_used), "used_gate": int(used_gate)}

    block = int(max(256, _env_int("MIS_LIBRARY_MATCH_BLOCK", 4096)))
    inter_i = np.zeros((N,), dtype=np.int32)
    for s in range(0, N, block):
        e = int(min(N, s + block))
        andb = np.bitwise_and(A[s:e], mk_p)
        inter_i[s:e] = _POPCOUNT_LUT[andb].sum(axis=1).astype(np.int32)

    inter = inter_i.astype(np.float32)
    union = (sumsA.astype(np.float32) + float(sB) - inter)
    iou = inter / (union + 1e-6)

    if varB <= 1e-12:
        corr01 = np.zeros_like(iou, dtype=np.float32)
    else:
        if int(muA.shape[0]) != int(sumsA.shape[0]):
            muA = (sumsA.astype(np.float32) / float(max(1, int(L)))).astype(np.float32)
        if int(varA.shape[0]) != int(sumsA.shape[0]):
            varA = (muA * (1.0 - muA)).astype(np.float32)
        Eab = inter / float(max(1, L))
        cov = Eab - (muA.astype(np.float32) * float(muB))
        den = np.sqrt(np.maximum(1e-12, varA.astype(np.float32) * float(varB)))
        corr_raw = cov / den
        corr_raw = np.clip(corr_raw, -1.0, 1.0)
        corr01 = 0.5 * (corr_raw + 1.0)

    sim = (0.55 * iou + 0.45 * corr01).astype(np.float32)
    k_eff = int(min(int(k), int(sim.size)))
    if k_eff <= 0:
        return {"view_ids": [], "scores": [], "ious": [], "corrs": [], "n_used": int(n_used), "used_gate": int(used_gate)}
    if int(sim.size) <= k_eff:
        ii = np.arange(int(sim.size), dtype=np.int32)
    else:
        ii = np.argpartition(-sim, kth=int(k_eff - 1))[: int(k_eff)].astype(np.int32)
    # sort by score desc
    ii = ii[np.argsort(sim[ii])[::-1]]
    v_ids = [int(x) for x in ids[ii].reshape(-1).tolist()]
    return {
        "view_ids": v_ids,
        "scores": [float(x) for x in sim[ii].reshape(-1).tolist()],
        "ious": [float(x) for x in iou[ii].reshape(-1).tolist()],
        "corrs": [float(x) for x in corr01[ii].reshape(-1).tolist()],
        "n_used": int(n_used),
        "used_gate": int(used_gate),
    }


def _ckpt_paths(out_root: str) -> dict[str, str]:
    return {
        "state": os.path.join(out_root, "checkpoint_state_track.json"),
        "track_json": os.path.join(out_root, "checkpoint_track_track.json"),
        "diag_json": os.path.join(out_root, "checkpoint_diag_track.json"),
    }


def _safe_read_json(path: str) -> Any:
    try:
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_write_json(path: str, obj: Any) -> None:
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        return


def _save_uvh_points_ply(
    path: str,
    *,
    pts_xyz: np.ndarray | None,
    color_rgb: tuple[int, int, int] | None = None,
    color_mode: str = "solid",
) -> str:
    try:
        if pts_xyz is None:
            return ""
        pts = np.asarray(pts_xyz, dtype=np.float32).reshape(-1, 3)
        if pts.size == 0:
            return ""
        cols = None
        mode = str(color_mode or "solid").strip().lower()
        if mode == "height":
            cols = _colorize_points_by_height(pts)
        elif color_rgb is not None:
            cols = _solid_point_colors(int(pts.shape[0]), tuple(color_rgb))
        _save_point_cloud_ply_ascii(path, pts_xyz=pts.astype(np.float32), colors_u8=cols)
        return path
    except Exception:
        return ""


def _save_uvh_pair_points_ply(
    path: str,
    *,
    pts_a: np.ndarray | None,
    color_a: tuple[int, int, int],
    pts_b: np.ndarray | None,
    color_b: tuple[int, int, int],
) -> str:
    try:
        arrs = []
        cols = []
        if pts_a is not None:
            a = np.asarray(pts_a, dtype=np.float32).reshape(-1, 3)
            if a.size > 0:
                arrs.append(a)
                cols.append(_solid_point_colors(int(a.shape[0]), color_a))
        if pts_b is not None:
            b = np.asarray(pts_b, dtype=np.float32).reshape(-1, 3)
            if b.size > 0:
                arrs.append(b)
                cols.append(_solid_point_colors(int(b.shape[0]), color_b))
        if not arrs:
            return ""
        pts = np.concatenate(arrs, axis=0).astype(np.float32, copy=False)
        col = np.concatenate(cols, axis=0).astype(np.uint8, copy=False)
        _save_point_cloud_ply_ascii(path, pts_xyz=pts, colors_u8=col)
        return path
    except Exception:
        return ""


def _save_uvh_point_match_overlay(
    path: str,
    *,
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    cur_pts_xyz: np.ndarray | None,
    projected_ref_pts_xyz: np.ndarray | None,
    max_draw: int = 2500,
) -> str:
    try:
        vis = np.asarray(frame_bgr, dtype=np.uint8).copy()
        if vis.ndim != 3 or vis.shape[2] != 3:
            return ""
        if isinstance(bbox, tuple) and len(bbox) == 4:
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(vis, (x, y), (x + max(1, w) - 1, y + max(1, h) - 1), (0, 255, 255), 2)

        def _draw_pts(pts_xyz: np.ndarray | None, color: tuple[int, int, int]) -> int:
            if pts_xyz is None:
                return 0
            pts = np.asarray(pts_xyz, dtype=np.float32).reshape(-1, 3)
            if pts.size == 0:
                return 0
            n = int(pts.shape[0])
            step = int(max(1, math.ceil(float(n) / float(max(1, int(max_draw))))))
            pts2 = pts[::step]
            H, W = vis.shape[:2]
            count = 0
            for p in pts2:
                u = int(round(float(p[0])))
                v = int(round(float(p[1])))
                if 0 <= u < W and 0 <= v < H:
                    cv2.circle(vis, (u, v), 1, color, -1, lineType=cv2.LINE_AA)
                    count += 1
            return int(count)

        n_cur = _draw_pts(cur_pts_xyz, (0, 255, 0))
        n_ref = _draw_pts(projected_ref_pts_xyz, (0, 64, 255))
        cv2.putText(
            vis,
            f"cur={n_cur} proj_ref={n_ref}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(path, vis)
        return path
    except Exception:
        return ""


def _save_uvh_render_debug_bundle(
    *,
    out_uvh_render: str,
    frame_idx: int,
    image_path: str,
    frame_bgr: np.ndarray,
    prev_bbox: tuple[int, int, int, int],
    best: dict[str, Any],
    ref_mode: str,
    ref_pts_xyz: np.ndarray | None,
) -> dict[str, str]:
    saved = {
        "meta_json": "",
        "ref_render_png": "",
        "cur_render_png": "",
        "render_compare_png": "",
        "ref_pts_ply": "",
        "cur_pts_ply": "",
        "projected_ref_pts_ply": "",
        "point_match_ply": "",
        "point_overlay_png": "",
    }
    try:
        if not out_uvh_render:
            return saved
        debug_stride = int(max(1, _env_int("MIS_UVH_RENDER_SAVE_DEBUG_STRIDE", 5)))
        if int(frame_idx) % int(debug_stride) != 0:
            return saved
        renders_dir = _ensure_dir(os.path.join(out_uvh_render, "renders"))
        points_dir = _ensure_dir(os.path.join(out_uvh_render, "points"))
        meta_dir = _ensure_dir(os.path.join(out_uvh_render, "meta"))
        matches_dir = _ensure_dir(os.path.join(out_uvh_render, "matches"))

        bbox = tuple(int(v) for v in best.get("bbox", prev_bbox))
        ref_render = best.get("uvh_render_prev_rgb", None)
        cur_render = best.get("uvh_render_cur_rgb", None)
        cur_pts = best.get("uvh_render_pts", None)
        proj_pts = best.get("uvh_projected_ref_pts", None)
        ref_color = (70, 140, 255)
        cur_color = (60, 220, 120)
        proj_color = (255, 110, 40)

        if isinstance(ref_render, np.ndarray):
            p = os.path.join(renders_dir, f"ref_render_{frame_idx:04d}.png")
            cv2.imwrite(p, np.asarray(ref_render, dtype=np.uint8))
            saved["ref_render_png"] = p
        if isinstance(cur_render, np.ndarray):
            p = os.path.join(renders_dir, f"cur_render_{frame_idx:04d}.png")
            cv2.imwrite(p, np.asarray(cur_render, dtype=np.uint8))
            saved["cur_render_png"] = p
        if isinstance(ref_render, np.ndarray) and isinstance(cur_render, np.ndarray):
            p = os.path.join(matches_dir, f"render_compare_{frame_idx:04d}.png")
            core._save_similarity_compare_image(
                p,
                prev_contour_rgb=np.asarray(ref_render, dtype=np.uint8),
                cur_contour_rgb=np.asarray(cur_render, dtype=np.uint8),
                similarity=float(best.get("uvh_render_sim", float("nan"))),
                title_left=f"ref(render,{str(ref_mode)})",
                title_right=f"cur(render,{str(best.get('uvh_decision_stage', ''))})",
            )
            saved["render_compare_png"] = p

        saved["ref_pts_ply"] = _save_uvh_points_ply(
            os.path.join(points_dir, f"ref_pts_{frame_idx:04d}.ply"),
            pts_xyz=ref_pts_xyz,
            color_rgb=ref_color,
        )
        saved["cur_pts_ply"] = _save_uvh_points_ply(
            os.path.join(points_dir, f"cur_pts_{frame_idx:04d}.ply"),
            pts_xyz=cur_pts,
            color_rgb=cur_color,
        )
        saved["projected_ref_pts_ply"] = _save_uvh_points_ply(
            os.path.join(points_dir, f"projected_ref_pts_{frame_idx:04d}.ply"),
            pts_xyz=proj_pts,
            color_rgb=proj_color,
        )
        saved["point_match_ply"] = _save_uvh_pair_points_ply(
            os.path.join(matches_dir, f"point_match_{frame_idx:04d}.ply"),
            pts_a=(np.asarray(cur_pts, dtype=np.float32) if isinstance(cur_pts, np.ndarray) else None),
            color_a=cur_color,
            pts_b=(np.asarray(proj_pts, dtype=np.float32) if isinstance(proj_pts, np.ndarray) else None),
            color_b=proj_color,
        )
        if isinstance(cur_pts, np.ndarray) or isinstance(proj_pts, np.ndarray):
            saved["point_overlay_png"] = _save_uvh_point_match_overlay(
                os.path.join(matches_dir, f"point_match_{frame_idx:04d}.png"),
                frame_bgr=frame_bgr,
                bbox=bbox,
                cur_pts_xyz=(np.asarray(cur_pts, dtype=np.float32) if isinstance(cur_pts, np.ndarray) else None),
                projected_ref_pts_xyz=(np.asarray(proj_pts, dtype=np.float32) if isinstance(proj_pts, np.ndarray) else None),
            )

        view_dir = best.get("uvh_view_dir", None)
        if isinstance(view_dir, np.ndarray):
            view_dir = np.asarray(view_dir, dtype=np.float32).reshape(-1).tolist()
        elif isinstance(view_dir, tuple):
            view_dir = list(view_dir)
        elif not isinstance(view_dir, list):
            view_dir = None

        meta = {
            "frame": int(frame_idx),
            "image": str(image_path),
            "ref_mode": str(ref_mode),
            "bbox": [int(v) for v in bbox],
            "prev_bbox": [int(v) for v in prev_bbox],
            "uvh_render_used": int(best.get("uvh_render_used", 0)),
            "uvh_decision_stage": str(best.get("uvh_decision_stage", "")),
            "uvh_render_sim": float(best.get("uvh_render_sim", float("nan"))),
            "uvh_decision_score": float(best.get("uvh_decision_score", float("nan"))),
            "uvh_mesh_sim": float(best.get("uvh_mesh_sim", float("nan"))),
            "uvh_mesh_iou": float(best.get("uvh_mesh_iou", float("nan"))),
            "uvh_mesh_depthcorr": float(best.get("uvh_mesh_depthcorr", float("nan"))),
            "uvh_sem_sim": float(best.get("uvh_sem_sim", float("nan"))),
            "uvh_sem_used": int(best.get("uvh_sem_used", 0)),
            "uvh_align_scale": float(best.get("uvh_align_scale", float("nan"))),
            "uvh_align_rot_deg": float(best.get("uvh_align_rot_deg", float("nan"))),
            "uvh_align_tx": float(best.get("uvh_align_tx", float("nan"))),
            "uvh_align_ty": float(best.get("uvh_align_ty", float("nan"))),
            "uvh_view_id": int(best.get("uvh_view_id", -1)),
            "uvh_view_score": float(best.get("uvh_view_score", float("nan"))),
            "uvh_view_dir": view_dir,
            "uvh_view_dist": float(best.get("uvh_view_dist", float("nan"))),
            "uvh_view_roll": float(best.get("uvh_view_roll", float("nan"))),
            "uvh_coverage_ratio": float(best.get("uvh_coverage_ratio", float("nan"))),
            "uvh_post_bbox_used": int(best.get("uvh_post_bbox_used", 0)),
            "uvh_post_bbox_source": str(best.get("uvh_post_bbox_source", "")),
            "uvh_rigid_used": int(best.get("uvh_rigid_used", 0)),
            "uvh_rigid_score": float(best.get("uvh_rigid_score", float("nan"))),
            "uvh_rf_attempted": int(best.get("uvh_rf_attempted", 0)),
            "uvh_rf_used": int(best.get("uvh_rf_used", 0)),
            "uvh_rf_best_loss": float(best.get("uvh_rf_best_loss", float("nan"))),
            "uvh_rf_steps": int(best.get("uvh_rf_steps", 0)),
            "uvh_rf_lr": float(best.get("uvh_rf_lr", float("nan"))),
            "uvh_local_bbox_used": int(best.get("uvh_local_bbox_used", 0)),
            "uvh_local_bbox_score": float(best.get("uvh_local_bbox_score", float("nan"))),
            "uvh_diff_used": int(best.get("uvh_diff_used", 0)),
            "uvh_diff_sim": float(best.get("uvh_diff_sim", float("nan"))),
            "ref_render_png": saved["ref_render_png"],
            "cur_render_png": saved["cur_render_png"],
            "render_compare_png": saved["render_compare_png"],
            "ref_pts_ply": saved["ref_pts_ply"],
            "cur_pts_ply": saved["cur_pts_ply"],
            "projected_ref_pts_ply": saved["projected_ref_pts_ply"],
            "point_match_ply": saved["point_match_ply"],
            "point_colors": {
                "ref_pts_rgb": list(ref_color),
                "cur_pts_rgb": list(cur_color),
                "projected_ref_pts_rgb": list(proj_color),
            },
            "point_overlay_png": saved["point_overlay_png"],
            "debug_save_stride": int(debug_stride),
        }
        meta_path = os.path.join(meta_dir, f"uvh_render_{frame_idx:04d}.json")
        _safe_write_json(meta_path, meta)
        saved["meta_json"] = meta_path
        return saved
    except Exception:
        return saved


def _save_checkpoint_track_only(
    *,
    out_root: str,
    cfg: core.TrackConfig,
    frames: list[str],
    last_frame: int,
    prev_bbox: tuple[int, int, int, int],
    prev_view_id: int | None = None,
    track_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, Any]],
    completed: bool = False,
) -> None:
    """断点续跑：只保存跟踪必要状态（不包含任何 RT/mesh）。"""
    try:
        paths = _ckpt_paths(out_root)
        last_frame = int(max(0, min(int(last_frame), len(frames) - 1)))
        state = {
            "version": 1,
            "time": _now(),
            "completed": bool(completed),
            "frames_dir": str(cfg.frames_dir),
            "proc_scale": float(cfg.proc_scale),
            "last_frame": int(last_frame),
            "prev_bbox": [int(x) for x in prev_bbox],
            "prev_view_id": int(prev_view_id) if prev_view_id is not None else -1,
        }
        _safe_write_json(paths["state"], state)
        _safe_write_json(paths["track_json"], track_rows)
        _safe_write_json(paths["diag_json"], diag_rows)

        # 同步一份便于查看的进度文件
        try:
            with open(os.path.join(out_root, "tracking_bboxes_and_scores.json"), "w", encoding="utf-8") as f:
                json.dump(track_rows, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    except Exception:
        return


def _try_load_checkpoint_track_only(
    *,
    out_root: str,
    cfg: core.TrackConfig,
    frames: list[str],
    out_mask: str,
    out_cont: str,
) -> Optional[dict[str, Any]]:
    """若 checkpoint 可用则恢复状态（不包含任何 RT/mesh）。"""
    # 与 main() 一致：未设置时视为不启用断点续跑，避免误读旧 checkpoint
    if not bool(getattr(cfg, "resume_enable", False)) or bool(getattr(cfg, "force_restart", False)):
        return None
    paths = _ckpt_paths(out_root)
    st = _safe_read_json(paths["state"])
    if not isinstance(st, dict):
        return None
    try:
        last_frame = int(st.get("last_frame", -1))
        completed = bool(st.get("completed", False))
        if completed:
            return None
        if str(st.get("frames_dir", "")).strip() and (str(st.get("frames_dir", "")).strip() != str(cfg.frames_dir).strip()):
            return None
        if last_frame < 0 or last_frame >= len(frames):
            return None
        ps = float(st.get("proc_scale", -1.0))
        if not np.isfinite(ps) or abs(ps - float(cfg.proc_scale)) > 1e-6:
            return None
        prev_bbox = st.get("prev_bbox", None)
        if not (isinstance(prev_bbox, list) and len(prev_bbox) == 4):
            return None
        prev_bbox_t = (int(prev_bbox[0]), int(prev_bbox[1]), int(prev_bbox[2]), int(prev_bbox[3]))
        try:
            prev_view_id = int(st.get("prev_view_id", -1))
        except Exception:
            prev_view_id = -1

        track_rows = _safe_read_json(paths["track_json"])
        diag_rows = _safe_read_json(paths["diag_json"])
        if not isinstance(track_rows, list):
            track_rows = []
        if not isinstance(diag_rows, list):
            diag_rows = []

        # 恢复上一帧 contour_rgb；若缺失则用 mask 重建（沿用 core 的工具函数）
        prev_cont = core._load_contour_rgb(os.path.join(out_cont, f"contour_rgb_{last_frame:04d}.png"))
        if prev_cont is None:
            mk = core._load_mask_u8(os.path.join(out_mask, f"mask_{last_frame:04d}.png"))
            if mk is not None:
                cnt = core.extract_largest_contour(mk)
                if cnt is not None:
                    cnt2 = core.resize_contour(cnt, (prev_bbox_t[2], prev_bbox_t[3]), cfg.semantic_image_size)
                    prev_cont = core.contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)
        if prev_cont is None:
            prev_cont = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)

        # 恢复 prev_mask_small
        mk = core._load_mask_u8(os.path.join(out_mask, f"mask_{last_frame:04d}.png"))
        if mk is None:
            mk = np.zeros((prev_bbox_t[3], prev_bbox_t[2]), dtype=np.uint8)
        prev_mask_small = _mask64_from_mask_u8(mk)

        return {
            "start_frame": int(last_frame + 1),
            "last_frame": int(last_frame),
            "prev_bbox": prev_bbox_t,
            "prev_view_id": int(prev_view_id),
            "prev_contour_rgb": prev_cont,
            "prev_mask_small": prev_mask_small,
            "track_rows": track_rows,
            "diag_rows": diag_rows,
        }
    except Exception:
        return None


# train_contour_similarity.py / ContourSimilarityNet 运行时状态
_CONTOUR_SIM_STATE: dict[str, Any] = {}


def _env_contour_sim_ckpt_default() -> str:
    return (
        os.environ.get(
            "MIS_CONTOUR_SIM_CKPT",
            r"E:\contour_sim_runs\chess_v1\contour_similarity_best.pt",
        )
        or ""
    ).strip().strip('"').strip("'")


def _contour_display_u8_to_chw01(contour_u8: np.ndarray, image_size: int) -> np.ndarray:
    """与 core.contour_to_rgb 一致的三通道灰度轮廓图 -> (1,1,S,S) float32 [0,1]。"""
    img = np.asarray(contour_u8, dtype=np.uint8)
    if img.ndim == 3:
        g = img[..., 0]
    else:
        g = img
    S = int(max(16, image_size))
    if int(g.shape[0]) != S or int(g.shape[1]) != S:
        g = cv2.resize(g, (S, S), interpolation=cv2.INTER_AREA)
    return (g.astype(np.float32) / 255.0).clip(0.0, 1.0).reshape(1, 1, S, S)


def _semantic_similarity_fallback_only(
    img1_rgb: np.ndarray,
    img2_rgb: np.ndarray,
    cfg: core.TrackConfig,
) -> float:
    """不加载 ContourSimilarityNet 时：仅用 core.fallback_similarity 做轮廓相似度。"""
    return float(core.fallback_similarity(img1_rgb, img2_rgb))


def _semantic_similarity_contour_net(
    img1_rgb: np.ndarray,
    img2_rgb: np.ndarray,
    cfg: core.TrackConfig,
) -> float:
    """ContourSimilarityNet 轮廓相似度（与 train_contour_similarity 输入域一致）。"""
    if not _CONTOUR_SIM_STATE:
        return float(core.fallback_similarity(img1_rgb, img2_rgb))
    try:
        import torch
        from contour_similarity_net import predict_similarity

        model = _CONTOUR_SIM_STATE["model"]
        device_t = _CONTOUR_SIM_STATE["device"]
        image_size = int(_CONTOUR_SIM_STATE["image_size"])
        t1 = torch.from_numpy(_contour_display_u8_to_chw01(img1_rgb, image_size)).to(
            device_t, dtype=torch.float32, non_blocking=True
        )
        t2 = torch.from_numpy(_contour_display_u8_to_chw01(img2_rgb, image_size)).to(
            device_t, dtype=torch.float32, non_blocking=True
        )
        with torch.no_grad():
            sim = predict_similarity(model, t1, t2)
        s0 = sim.reshape(-1)[0]
        v = float(s0.item())
        if not np.isfinite(v):
            return 0.0
        return float(max(0.0, min(1.0, v)))
    except Exception:
        return float(core.fallback_similarity(img1_rgb, img2_rgb))


def _load_contour_similarity_state(cfg: core.TrackConfig) -> tuple[bool, str]:
    """
    从 train_contour_similarity.py 保存的 checkpoint 加载 ContourSimilarityNet。
    环境变量：
      - MIS_CONTOUR_SIM_CKPT：权重路径（默认与 train 脚本 --out-dir 下 best.pt 一致）
      - MIS_CONTOUR_SIM_DISABLE=1：强制不走此路径
      - MIS_CONTOUR_SIM_DEVICE：cuda/cpu（默认同 core 设备选择）
    """
    global _CONTOUR_SIM_STATE
    _CONTOUR_SIM_STATE = {}
    if bool(_env_bool("MIS_CONTOUR_SIM_DISABLE", False)):
        return False, "MIS_CONTOUR_SIM_DISABLE=1"
    ckpt = _env_contour_sim_ckpt_default()
    if not ckpt or not os.path.isfile(ckpt):
        return False, f"contour_ckpt_missing:{ckpt}"
    if not bool(_env_bool("MIS_UVH_SEMANTIC_ENABLE", True)):
        return False, "MIS_UVH_SEMANTIC_ENABLE=0"
    try:
        import torch
        from contour_similarity_net import ContourSimilarityNet
    except Exception as e:
        return False, f"contour_sim_import:{e}"

    try:
        try:
            obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(ckpt, map_location="cpu")
    except Exception as e:
        return False, f"contour_ckpt_load:{e}"

    if isinstance(obj, dict) and "model" in obj:
        sd = obj["model"]
        conf = obj.get("config", {}) or {}
    else:
        sd = obj
        conf = {}
    if not isinstance(conf, dict):
        conf = {}

    image_size = int(conf.get("image_size", 128))
    latent_dim = int(conf.get("latent_dim", 128))
    base_ch = int(conf.get("base_ch", 32))

    dev_override = (os.environ.get("MIS_CONTOUR_SIM_DEVICE", "") or "").strip().lower()
    device_str = dev_override if dev_override else core._device_str(cfg)
    if str(device_str).startswith("cuda") and not torch.cuda.is_available():
        device_str = "cpu"
    device_t = torch.device(device_str)

    try:
        model = ContourSimilarityNet(
            image_size=int(image_size),
            latent_dim=int(latent_dim),
            base_ch=int(base_ch),
        )
        model.load_state_dict(sd, strict=True)
        model.to(device_t)
        model.eval()
    except Exception as e:
        _CONTOUR_SIM_STATE = {}
        return False, f"contour_model_build:{e}"

    _CONTOUR_SIM_STATE = {
        "model": model,
        "device": device_t,
        "image_size": int(image_size),
        "latent_dim": int(latent_dim),
        "base_ch": int(base_ch),
        "ckpt_path": str(ckpt),
    }
    return True, ""


def _init_semantic(cfg: core.TrackConfig) -> dict[str, Any]:
    """
    仅加载 train_contour_similarity.py 的 ContourSimilarityNet（MIS_CONTOUR_SIM_CKPT）。
    失败则 core.semantic_similarity 固定为轮廓回退相似度（不加载其他 .pth 语义模型）。
    """
    ok_c, err_c = _load_contour_similarity_state(cfg)
    if ok_c:
        core.semantic_similarity = _semantic_similarity_contour_net  # type: ignore[assignment]
        core.SEMANTIC_ENABLED = True
        print(
            f"[轮廓相似度] 已启用 ContourSimilarityNet "
            f"ckpt={_CONTOUR_SIM_STATE.get('ckpt_path')} "
            f"image_size={_CONTOUR_SIM_STATE.get('image_size')} device={_CONTOUR_SIM_STATE.get('device')}",
            flush=True,
        )

        warmup_ok = False
        warmup_err = None
        if bool(getattr(cfg, "semantic_warmup", True)):
            try:
                dummy = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
                _ = float(core.semantic_similarity(dummy, dummy, cfg))
                warmup_ok = True
            except Exception:
                warmup_ok = False
                warmup_err = "warmup_failed"

        return {
            "semantic_available": True,
            "semantic_enabled": True,
            "semantic_backend": "contour_similarity_net",
            "semantic_import_error": None,
            "contour_sim_ckpt": str(_CONTOUR_SIM_STATE.get("ckpt_path", "")),
            "contour_sim_image_size": int(_CONTOUR_SIM_STATE.get("image_size", 128)),
            "model_path": str(_CONTOUR_SIM_STATE.get("ckpt_path", "")),
            "model_exists": True,
            "device": str(_CONTOUR_SIM_STATE.get("device", "")),
            "semantic_warmup_ok": bool(warmup_ok),
            "semantic_warmup_error": warmup_err,
        }

    core.semantic_similarity = _semantic_similarity_fallback_only  # type: ignore[assignment]
    core.SEMANTIC_ENABLED = False
    ckpt_ref = _env_contour_sim_ckpt_default()

    if not bool(_env_bool("MIS_UVH_SEMANTIC_ENABLE", True)):
        return {
            "semantic_available": False,
            "semantic_enabled": False,
            "semantic_backend": "off",
            "semantic_import_error": None,
            "contour_sim_ckpt": "",
            "contour_sim_skip_reason": err_c,
            "model_path": str(ckpt_ref),
            "model_exists": bool(ckpt_ref and os.path.isfile(ckpt_ref)),
            "device": core._device_str(cfg),
            "semantic_warmup_ok": False,
            "semantic_warmup_error": "disabled_by_default",
        }

    warmup_ok = False
    warmup_err = None
    if bool(getattr(cfg, "semantic_warmup", True)):
        try:
            dummy = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
            _ = float(core.semantic_similarity(dummy, dummy, cfg))
            warmup_ok = True
        except Exception:
            warmup_ok = False
            warmup_err = "warmup_failed"

    return {
        "semantic_available": False,
        "semantic_enabled": False,
        "semantic_backend": "fallback_only",
        "semantic_import_error": None,
        "contour_sim_ckpt": "",
        "contour_sim_skip_reason": err_c,
        "model_path": str(ckpt_ref),
        "model_exists": bool(ckpt_ref and os.path.isfile(ckpt_ref)),
        "device": core._device_str(cfg),
        "semantic_warmup_ok": bool(warmup_ok),
        "semantic_warmup_error": warmup_err,
    }


def _select_initial_roi(first_bgr: np.ndarray, cfg: core.TrackConfig) -> tuple[int, int, int, int]:
    """尽量沿用 core.main 的 ROI 选框逻辑（但强制用于跟踪）。"""
    H, W = first_bgr.shape[:2]
    backend = (cfg.roi_backend or "auto").strip().lower()

    roi = None
    # A) init：环境变量框
    if backend == "init":
        roi = core._try_parse_bbox(cfg.init_bbox)
        if roi is None:
            raise RuntimeError("MIS_ROI_BACKEND=init 但 MIS_INIT_BBOX 无效/未设置")
        # 降分辨率坐标兜底
        try:
            x0, y0, w0, h0 = [int(v) for v in roi]
            if (x0 + w0 > W) or (y0 + h0 > H):
                s = float(max(1e-6, float(cfg.proc_scale)))
                roi = (int(round(x0 * s)), int(round(y0 * s)), int(round(w0 * s)), int(round(h0 * s)))
        except Exception:
            pass

    # B) opencv：强制 OpenCV GUI 选框
    if roi is None and backend == "opencv":
        if not bool(getattr(cfg, "vis_window", True)):
            raise RuntimeError("MIS_ROI_BACKEND=opencv 但 MIS_VIS_WINDOW=0，无法弹窗选框")
        cv2.namedWindow("MIS_ROI_Select", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MIS_ROI_Select", min(1200, W), min(900, H))
        roi = cv2.selectROI("MIS_ROI_Select", first_bgr, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("MIS_ROI_Select")

    # C) tk：强制 Tkinter 选框
    if roi is None and backend == "tk":
        roi = core._select_roi_tkinter(first_bgr, max_w=cfg.tk_max_w, max_h=cfg.tk_max_h)
        if roi is None:
            raise RuntimeError("Tkinter 选框取消/失败")

    # D) auto：优先 init_bbox -> OpenCV -> Tkinter -> 失败则报错（跟踪必须有 ROI）
    if roi is None and backend == "auto":
        if cfg.init_bbox:
            roi = core._try_parse_bbox(cfg.init_bbox)
            if roi is None:
                print(f"[警告] MIS_INIT_BBOX 格式无效: {cfg.init_bbox}，将尝试鼠标选框。")
                roi = None
        if roi is None and bool(getattr(cfg, "vis_window", True)):
            try:
                cv2.namedWindow("MIS_ROI_Select", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("MIS_ROI_Select", min(1200, W), min(900, H))
                roi = cv2.selectROI("MIS_ROI_Select", first_bgr, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("MIS_ROI_Select")
            except Exception as e:
                print(f"[警告] OpenCV HighGUI 不可用，将改用 Tkinter 选框: {e}")
                roi = core._select_roi_tkinter(first_bgr, max_w=cfg.tk_max_w, max_h=cfg.tk_max_h)

    if roi is None or (isinstance(roi, tuple) and len(roi) == 4 and int(roi[2]) <= 1):
        raise RuntimeError("ROI 选择无效/取消：跟踪流程需要有效 ROI。")

    x, y, w, h = [int(v) for v in roi]
    return core.clamp_bbox(x, y, w, h, W, H)


def _paint_target_mask_opencv(first_bgr: np.ndarray, cfg: core.TrackConfig) -> np.ndarray:
    """在首帧用鼠标涂抹方式标注像素级目标 mask（返回整帧 mask_u8: 0/255）。"""
    H, W = first_bgr.shape[:2]
    if not bool(getattr(cfg, "vis_window", True)):
        raise RuntimeError("MIS_VIS_WINDOW=0，无法进行鼠标标注 mask")
    win = "MIS_TargetMask_Paint"
    brush = int(max(1, _env_int("MIS_MASK_BRUSH", 10)))
    alpha = float(max(0.05, min(0.95, _env_float("MIS_MASK_OVERLAY_ALPHA", 0.35))))

    mask = np.zeros((H, W), dtype=np.uint8)
    drawing = False
    erasing = False
    last = None

    def _draw_to_mask(x: int, y: int) -> None:
        nonlocal last, mask, brush, erasing
        col = 0 if erasing else 255
        if last is None:
            last = (int(x), int(y))
            cv2.circle(mask, (int(x), int(y)), int(brush), int(col), thickness=-1)
            return
        x0, y0 = int(last[0]), int(last[1])
        x1, y1 = int(x), int(y)
        try:
            cv2.line(mask, (x0, y0), (x1, y1), int(col), thickness=int(max(1, brush * 2)))
            cv2.circle(mask, (x1, y1), int(brush), int(col), thickness=-1)
        except Exception:
            pass
        last = (x1, y1)

    def _on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        nonlocal drawing, erasing, last
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            erasing = False
            last = None
            _draw_to_mask(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            drawing = True
            erasing = True
            last = None
            _draw_to_mask(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            _draw_to_mask(x, y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            drawing = False
            erasing = False
            last = None

    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(1200, W), min(900, H))
        cv2.setMouseCallback(win, _on_mouse)
    except cv2.error as e:
        # OpenCV HighGUI 不可用（常见于 opencv-python-headless）
        raise RuntimeError(f"OpenCV HighGUI 不可用：{e}") from e

    print("[信息] 首帧像素级目标标注：左键涂抹=添加，右键涂抹=擦除，r=清空，[/]=调笔刷，Enter=确认，Esc/q=取消")
    while True:
        vis = np.asarray(first_bgr, dtype=np.uint8).copy()
        mk = mask
        if mk is not None and mk.size > 0:
            col = np.zeros_like(vis, dtype=np.uint8)
            col[..., 1] = mk  # green
            try:
                vis = cv2.addWeighted(vis, 1.0, col, float(alpha), 0.0)
            except Exception:
                pass
            try:
                ys, xs = np.nonzero(mk > 0)
                if xs.size > 0:
                    x0, x1 = int(xs.min()), int(xs.max())
                    y0, y1 = int(ys.min()), int(ys.max())
                    cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 2)
            except Exception:
                pass

        try:
            cv2.putText(vis, f"brush={brush}px  LMB=add  RMB=erase  r=reset  Enter=OK  Esc/q=cancel",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(vis, f"brush={brush}px  LMB=add  RMB=erase  r=reset  Enter=OK  Esc/q=cancel",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
        except Exception:
            pass

        cv2.imshow(win, vis)
        k = int(cv2.waitKey(20) & 0xFF)
        if k in (13, 10):  # Enter
            break
        if k in (27, ord("q")):  # Esc / q
            cv2.destroyWindow(win)
            raise RuntimeError("用户取消像素级 mask 标注")
        if k == ord("r"):
            mask[:] = 0
        if k in (ord("["), ord("-"), ord("_")):
            brush = int(max(1, int(brush) - 1))
        if k in (ord("]"), ord("+"), ord("=")):
            brush = int(min(128, int(brush) + 1))

    cv2.destroyWindow(win)
    mk = _mask_to_filled_u8(mask)
    if mk.shape != (H, W):
        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    if int(np.count_nonzero(mk > 0)) <= 10:
        raise RuntimeError("像素级 mask 标注为空/过小")
    return mk


def _paint_target_mask_tkinter(first_bgr: np.ndarray, cfg: core.TrackConfig) -> np.ndarray:
    """OpenCV HighGUI 不可用时，用 Tkinter 在首帧涂抹标注像素级 mask。"""
    try:
        import tkinter as tk
        import base64
    except Exception as e:
        raise RuntimeError(f"Tkinter 不可用，无法鼠标标注 mask：{e}") from e

    H, W = first_bgr.shape[:2]
    max_w = int(getattr(cfg, "tk_max_w", 1200))
    max_h = int(getattr(cfg, "tk_max_h", 900))
    s = float(min(1.0, float(max_w) / float(max(1, W)), float(max_h) / float(max(1, H))))
    dw = int(max(2, round(float(W) * s)))
    dh = int(max(2, round(float(H) * s)))

    base = np.asarray(first_bgr, dtype=np.uint8)
    base_disp = cv2.resize(base, (dw, dh), interpolation=cv2.INTER_AREA) if (abs(s - 1.0) > 1e-6) else base.copy()

    brush = int(max(1, _env_int("MIS_MASK_BRUSH", 10)))
    alpha = float(max(0.05, min(0.95, _env_float("MIS_MASK_OVERLAY_ALPHA", 0.35))))

    mask = np.zeros((H, W), dtype=np.uint8)
    drawing = False
    erasing = False
    last_xy: tuple[int, int] | None = None
    dirty = True
    cancelled = False
    confirmed = False

    def _bgr_to_photo(img_bgr: np.ndarray) -> "tk.PhotoImage":
        ok, buf = cv2.imencode(".png", np.asarray(img_bgr, dtype=np.uint8))
        if not ok:
            raise RuntimeError("imencode png failed")
        data = base64.b64encode(buf.tobytes()).decode("ascii")
        return tk.PhotoImage(data=data)

    def _apply_stroke(ix: int, iy: int) -> None:
        nonlocal last_xy, mask
        col = 0 if erasing else 255
        if last_xy is None:
            last_xy = (int(ix), int(iy))
            cv2.circle(mask, (int(ix), int(iy)), int(brush), int(col), thickness=-1)
            return
        x0, y0 = int(last_xy[0]), int(last_xy[1])
        x1, y1 = int(ix), int(iy)
        try:
            cv2.line(mask, (x0, y0), (x1, y1), int(col), thickness=int(max(1, brush * 2)))
            cv2.circle(mask, (x1, y1), int(brush), int(col), thickness=-1)
        except Exception:
            pass
        last_xy = (x1, y1)

    def _xy_to_img(x: int, y: int) -> tuple[int, int]:
        ix = int(round(float(x) / float(max(1e-6, s))))
        iy = int(round(float(y) / float(max(1e-6, s))))
        ix = int(max(0, min(W - 1, ix)))
        iy = int(max(0, min(H - 1, iy)))
        return ix, iy

    def on_l_down(e: Any) -> None:
        nonlocal drawing, erasing, last_xy, dirty
        drawing = True
        erasing = False
        last_xy = None
        ix, iy = _xy_to_img(int(e.x), int(e.y))
        _apply_stroke(ix, iy)
        dirty = True

    def on_r_down(e: Any) -> None:
        nonlocal drawing, erasing, last_xy, dirty
        drawing = True
        erasing = True
        last_xy = None
        ix, iy = _xy_to_img(int(e.x), int(e.y))
        _apply_stroke(ix, iy)
        dirty = True

    def on_move(e: Any) -> None:
        nonlocal dirty
        if not drawing:
            return
        ix, iy = _xy_to_img(int(e.x), int(e.y))
        _apply_stroke(ix, iy)
        dirty = True

    def on_up(e: Any) -> None:
        nonlocal drawing, erasing, last_xy
        drawing = False
        erasing = False
        last_xy = None

    def do_reset() -> None:
        nonlocal dirty, mask
        mask[:] = 0
        dirty = True

    def do_cancel() -> None:
        nonlocal cancelled
        cancelled = True
        root.quit()

    def do_ok() -> None:
        nonlocal confirmed
        if int(np.count_nonzero(mask > 0)) <= 10:
            return
        confirmed = True
        root.quit()

    def on_key(e: Any) -> None:
        nonlocal brush
        ks = str(getattr(e, "keysym", "") or "")
        if ks in ("Return", "KP_Enter"):
            do_ok()
        elif ks in ("Escape",):
            do_cancel()
        elif ks.lower() == "r":
            do_reset()
        elif ks in ("bracketleft", "minus", "underscore"):
            brush = int(max(1, int(brush) - 1))
        elif ks in ("bracketright", "equal", "plus"):
            brush = int(min(128, int(brush) + 1))

    root = tk.Tk()
    root.title("MIS Target Mask Paint (Tkinter)")
    root.resizable(False, False)

    info = tk.Label(
        root,
        text="左键拖动=添加  右键拖动=擦除  r=清空  [ ] 调笔刷  Enter=确认  Esc=取消",
        anchor="w",
    )
    info.pack(fill="x", padx=8, pady=6)

    canvas = tk.Canvas(root, width=dw, height=dh, highlightthickness=0)
    canvas.pack(padx=8, pady=8)

    btns = tk.Frame(root)
    btns.pack(fill="x", padx=8, pady=(0, 8))
    tk.Button(btns, text="清空(r)", command=do_reset).pack(side="left")
    tk.Button(btns, text="取消(Esc)", command=do_cancel).pack(side="right")
    tk.Button(btns, text="确认(Enter)", command=do_ok).pack(side="right", padx=(0, 8))

    photo0 = _bgr_to_photo(base_disp)
    img_item = canvas.create_image(0, 0, anchor="nw", image=photo0)
    canvas._photo_ref = photo0  # type: ignore[attr-defined]

    canvas.bind("<ButtonPress-1>", on_l_down)
    canvas.bind("<B1-Motion>", on_move)
    canvas.bind("<ButtonRelease-1>", on_up)
    canvas.bind("<ButtonPress-3>", on_r_down)
    canvas.bind("<B3-Motion>", on_move)
    canvas.bind("<ButtonRelease-3>", on_up)
    # 部分系统右键事件不稳定，提供 Shift+左键擦除
    canvas.bind("<Shift-ButtonPress-1>", on_r_down)
    canvas.bind("<Shift-B1-Motion>", on_move)
    canvas.bind("<Shift-ButtonRelease-1>", on_up)

    root.bind("<Key>", on_key)

    def refresh() -> None:
        nonlocal dirty
        if dirty:
            disp = base_disp.copy()
            mk_disp = cv2.resize(mask, (dw, dh), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            col = np.zeros_like(disp, dtype=np.uint8)
            col[..., 1] = (mk_disp > 0).astype(np.uint8) * 255
            try:
                disp = cv2.addWeighted(disp, 1.0, col, float(alpha), 0.0)
            except Exception:
                pass
            try:
                ys, xs = np.nonzero(mk_disp > 0)
                if xs.size > 0:
                    x0, x1 = int(xs.min()), int(xs.max())
                    y0, y1 = int(ys.min()), int(ys.max())
                    cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 255), 2)
            except Exception:
                pass
            try:
                cv2.putText(disp, f"brush={brush}px", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (20, 20, 20), 4, cv2.LINE_AA)
                cv2.putText(disp, f"brush={brush}px", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2, cv2.LINE_AA)
            except Exception:
                pass
            try:
                ph = _bgr_to_photo(disp)
                canvas.itemconfig(img_item, image=ph)
                canvas._photo_ref = ph  # type: ignore[attr-defined]
            except Exception:
                pass
            dirty = False
        root.after(33, refresh)

    refresh()
    canvas.focus_set()
    root.mainloop()
    try:
        root.destroy()
    except Exception:
        pass

    if cancelled or (not confirmed):
        raise RuntimeError("用户取消像素级 mask 标注")

    mk = _mask_to_filled_u8(mask)
    if mk.shape != (H, W):
        mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    if int(np.count_nonzero(mk > 0)) <= 10:
        raise RuntimeError("像素级 mask 标注为空/过小")
    return mk


def _select_initial_target_mask_and_bbox(first_bgr: np.ndarray, cfg: core.TrackConfig) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """
    首帧目标选择：
    - 默认：鼠标像素级标注 mask -> 自动生成包围框 bbox
    - 若提供 MIS_INIT_BBOX：可直接用 bbox（mask 退化为 bbox 填充矩形，便于无GUI跑）
    返回：(bbox, mask_full_u8)
    """
    H, W = first_bgr.shape[:2]
    # 1) 若用户提供 init_bbox，优先用它（便于复现实验/无GUI）
    if cfg.init_bbox:
        roi = core._try_parse_bbox(cfg.init_bbox)
        if roi is not None:
            x, y, w, h = [int(v) for v in roi]
            bbox = core.clamp_bbox(x, y, w, h, W, H)
            mk = np.zeros((H, W), dtype=np.uint8)
            x, y, w, h = bbox
            mk[y:y + h, x:x + w] = 255
            return bbox, mk

    # 2) 鼠标像素级标注（优先 OpenCV；若 HighGUI 不可用则回退 Tkinter）
    try:
        mk_full = _paint_target_mask_opencv(first_bgr, cfg)
    except Exception as e:
        print(f"[警告] OpenCV 像素标注不可用，将回退 Tkinter：{e}")
        try:
            mk_full = _paint_target_mask_tkinter(first_bgr, cfg)
        except Exception as e2:
            print(f"[警告] Tkinter 像素标注也不可用，将回退 bbox 选框：{e2}")
            bbox0 = _select_initial_roi(first_bgr, cfg)
            x, y, w, h = [int(v) for v in bbox0]
            mk = np.zeros((H, W), dtype=np.uint8)
            mk[y:y + h, x:x + w] = 255
            return bbox0, mk
    ys, xs = np.nonzero(np.asarray(mk_full, dtype=np.uint8) > 0)
    if xs.size <= 0:
        raise RuntimeError("像素级 mask 为空")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = core.clamp_bbox(x0, y0, int(x1 - x0 + 1), int(y1 - y0 + 1), W, H)
    return bbox, mk_full


def find_best_bbox_in_next_frame_with_library(
    *,
    frame_bgr: np.ndarray,
    prev_bbox: tuple[int, int, int, int],
    ref_mask_small: np.ndarray,
    prev_view_dir: np.ndarray | None,
    cfg: core.TrackConfig,
    lib: ViewLibrary,
    angle_gate_deg: float,
) -> dict[str, Any]:
    """
    基本沿用 core.find_best_bbox_in_next_frame 的“候选生成 -> 粗筛 -> 精排 -> 恢复”结构，
    但把精排里的“与上一帧轮廓做语义相似度”替换为“与多视点目标库做最佳匹配(向量化)”；
    仅依赖“首帧目标库（reference）”，不做帧间匹配/关键帧更新。

    返回 dict：
      bbox, score, semantic(=lib_score), quick, dx, dy, th_gray, mask_u8, contour_rgb, th_uint8,
      lib_view_id, lib_iou, lib_corr, lib_n_used, lib_used_gate
    """
    H, W = frame_bgr.shape[:2]
    px, py, pw, ph = prev_bbox
    cx0 = px + pw // 2
    cy0 = py + ph // 2

    step = int(max(1, cfg.search_step))
    rad = int(max(step, cfg.search_radius))
    # 为了减少“远处相似轮廓”导致的跳变/漂移：先在更小窗口内找，低分再扩大到全窗口
    rad_small = int(_env_int("MIS_LIB_RAD_SMALL", int(max(step, min(rad, max(step, rad // 2))))))  # 默认 rad/2
    rad_small = int(max(step, min(int(rad), int(rad_small))))
    rad_small_sem_min = float(_env_float("MIS_LIB_RAD_SMALL_SEM_MIN", 0.45))

    def _run(rad_now: int) -> dict[str, Any] | None:
        # 在“上一帧 bbox 扩大窗口”内先做一次 th_68（保证本帧候选的 th/mask 来自同一统计阈值）
        sx0 = int(max(0, px - int(rad_now)))
        sy0 = int(max(0, py - int(rad_now)))
        sx1 = int(min(W, px + pw + int(rad_now)))
        sy1 = int(min(H, py + ph + int(rad_now)))
        sw = int(max(0, sx1 - sx0))
        sh = int(max(0, sy1 - sy0))
        if sw < 4 or sh < 4:
            return None
        search_roi = frame_bgr[sy0:sy1, sx0:sx1]
        th_search, mask_search, th_u8_search = core.compute_th_gray_truncated(search_roi, cfg.th_percent, cfg)
        th_search = np.asarray(th_search, dtype=np.float32)
        mask_search = np.asarray(mask_search, dtype=np.uint8)
        if th_search.ndim != 2 or mask_search.ndim != 2:
            return None
        if th_search.shape[0] != sh or th_search.shape[1] != sw:
            th_search = cv2.resize(th_search, (sw, sh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        if mask_search.shape[0] != sh or mask_search.shape[1] != sw:
            mask_search = cv2.resize(mask_search, (sw, sh), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        search_ctx = {
            "sx0": int(sx0),
            "sy0": int(sy0),
            "sw": int(sw),
            "sh": int(sh),
            "th_gray": np.asarray(th_search, dtype=np.float32),
            "mask_u8": np.asarray(mask_search, dtype=np.uint8),
            "th_uint8": float(th_u8_search),
        }
        rebuild_cache: dict[tuple[int, int, int, int], dict[str, Any]] = {}

        cands_xy: dict[tuple[int, int], tuple[int, int, int, int, int, int]] = {}
        for dy in range(-rad_now, rad_now + 1, step):
            for dx in range(-rad_now, rad_now + 1, step):
                x0 = cx0 + dx - pw // 2
                y0 = cy0 + dy - ph // 2
                x, y, ww, hh = core.clamp_bbox(x0, y0, pw, ph, W, H)
                dx_eff = int((x + ww // 2) - cx0)
                dy_eff = int((y + hh // 2) - cy0)
                cands_xy[(int(x), int(y))] = (int(x), int(y), int(ww), int(hh), int(dx_eff), int(dy_eff))
        cands = list(cands_xy.values())

        scored_quick = []
        for (x, y, ww, hh, dx, dy) in cands:
            # 直接从 search_roi 的 mask 裁剪，保持阈值一致
            xi = int(x - sx0)
            yi = int(y - sy0)
            if xi < 0 or yi < 0 or (xi + int(ww) > sw) or (yi + int(hh) > sh):
                continue
            mk = mask_search[yi:yi + int(hh), xi:xi + int(ww)]
            ms = _mask64_from_mask_u8(mk)
            q = core.quick_mask_score(ref_mask_small, ms)
            dist = (abs(dx) + abs(dy)) / float(max(1, rad_now))
            q2 = float(q) - float(cfg.spatial_penalty) * float(dist)
            scored_quick.append((q2, q, x, y, ww, hh, dx, dy))
        scored_quick.sort(key=lambda t: t[0], reverse=True)
        top = scored_quick[: max(1, int(cfg.topk_semantic))]

        best = None
        best_score = -1e9
        for (q2, q, x, y, ww, hh, dx, dy) in top:
            xi = int(x - sx0)
            yi = int(y - sy0)
            if xi < 0 or yi < 0 or (xi + int(ww) > sw) or (yi + int(hh) > sh):
                continue
            th_gray = th_search[yi:yi + int(hh), xi:xi + int(ww)].astype(np.float32, copy=False)
            mask_u8 = mask_search[yi:yi + int(hh), xi:xi + int(ww)].astype(np.uint8, copy=False)
            th_uint8 = float(th_u8_search)

            cnt = _extract_contour_prefer_center(mask_u8)
            if cnt is None:
                cnt = core.extract_largest_contour(mask_u8)
            if cnt is None:
                contour_rgb = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
                info = {"view_id": -1, "iou": 0.0, "corr": 0.0, "n_used": 0, "used_gate": 0}
                sem = 0.0
            else:
                cnt2 = core.resize_contour(cnt, (ww, hh), cfg.semantic_image_size)
                contour_rgb = core.contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)
                mret = match_contour_to_library(
                    lib=lib,
                    contour_rgb=contour_rgb,
                    prev_view_dir=prev_view_dir,
                    angle_gate_deg=float(angle_gate_deg),
                )
                sem = float(mret.get("score", 0.0))
                # gate 兜底：若门控后得分很低，则补跑一次“全库匹配”，避免误 gate 漂移
                gate_retry_min = float(_env_float("MIS_LIBRARY_GATE_RETRY_MIN", 0.35))
                if int(mret.get("used_gate", 0)) == 1 and (np.isfinite(sem) and float(sem) < float(gate_retry_min)):
                    mret2 = match_contour_to_library(lib=lib, contour_rgb=contour_rgb, prev_view_dir=None, angle_gate_deg=0.0)
                    sem2 = float(mret2.get("score", 0.0))
                    if np.isfinite(sem2) and float(sem2) > float(sem):
                        mret = mret2
                        sem = float(sem2)
                info = {
                    "view_id": int(mret.get("view_id", -1)),
                    "iou": float(mret.get("iou", 0.0)),
                    "corr": float(mret.get("corr", 0.0)),
                    "n_used": int(mret.get("n_used", 0)),
                    "used_gate": int(mret.get("used_gate", 0)),
                }
                sem = 0.0 if (not np.isfinite(sem)) else float(max(0.0, min(1.0, sem)))

            score = float(0.85 * sem + 0.15 * max(0.0, q)) - float(cfg.spatial_penalty) * (
                (abs(dx) + abs(dy)) / float(max(1, rad_now))
            )
            if score > best_score:
                best_score = score
                vid = int(info["view_id"])
                if 0 <= vid < int(lib.ids.size):
                    try:
                        vdir = np.asarray(lib.dirs[int(vid)], dtype=np.float32).reshape(3)
                    except Exception:
                        vdir = None
                    try:
                        vdist = float(lib.dists[int(vid)])
                    except Exception:
                        vdist = float("nan")
                    try:
                        vroll = float(lib.rolls[int(vid)])
                    except Exception:
                        vroll = float("nan")
                    try:
                        vTw2c = np.asarray(lib.T_w2c[int(vid)], dtype=np.float32).reshape(-1).tolist()
                    except Exception:
                        vTw2c = None
                    try:
                        vpath = str(lib.contour_rgb_paths[int(vid)])
                    except Exception:
                        vpath = ""
                else:
                    vdir, vdist, vroll, vTw2c, vpath = None, float("nan"), float("nan"), None, ""
                best = {
                    "bbox": (int(x), int(y), int(ww), int(hh)),
                    "score": float(score),
                    "semantic": float(sem),
                    "quick": float(q),
                    "dx": int(dx),
                    "dy": int(dy),
                    "th_gray": th_gray,
                    "mask_u8": mask_u8,
                    "contour_rgb": contour_rgb,
                    "th_uint8": float(th_uint8),
                    "lib_view_id": int(info["view_id"]),
                    "lib_iou": float(info["iou"]),
                    "lib_corr": float(info["corr"]),
                    "lib_n_used": int(info["n_used"]),
                    "lib_used_gate": int(info["used_gate"]),
                    "lib_dir": (vdir.astype(float).reshape(-1).tolist() if isinstance(vdir, np.ndarray) else None),
                    "lib_dist": float(vdist),
                    "lib_roll": float(vroll),
                    "lib_T_w2c": vTw2c,
                    "lib_template_contour": vpath,
                }
        return best

    best = _run(rad_small)
    if int(rad_small) < int(rad):
        try:
            sem0 = float(best.get("semantic", 0.0)) if isinstance(best, dict) else 0.0
        except Exception:
            sem0 = 0.0
        if best is None or (not np.isfinite(sem0)) or float(sem0) < float(rad_small_sem_min):
            best_full = _run(rad)
            if best_full is not None and (best is None or float(best_full.get("score", -1e9)) > float(best.get("score", -1e9))):
                best = best_full
    if cfg.recover_enabled and (best is None or float(best.get("score", 0.0)) < float(cfg.recover_score)):
        rad2 = int(max(rad * 2, rad + step))
        best2 = _run(rad2)
        if best2 is not None and (best is None or float(best2.get("score", -1e9)) > float(best.get("score", -1e9))):
            best = best2

    if best is None:
        roi = core.crop(frame_bgr, prev_bbox)
        th_gray, mask_u8, th_uint8 = core.compute_th_gray_truncated(roi, cfg.th_percent, cfg)
        cnt = _extract_contour_prefer_center(mask_u8)
        if cnt is None:
            cnt = core.extract_largest_contour(mask_u8)
        if cnt is not None:
            cnt2 = core.resize_contour(cnt, (prev_bbox[2], prev_bbox[3]), cfg.semantic_image_size)
            contour_rgb = core.contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)
        else:
            contour_rgb = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
        best = {
            "bbox": prev_bbox,
            "score": 0.0,
            "semantic": 0.0,
            "quick": 0.0,
            "dx": 0,
            "dy": 0,
            "th_gray": th_gray,
            "mask_u8": mask_u8,
            "contour_rgb": contour_rgb,
            "th_uint8": float(th_uint8),
            "lib_view_id": -1,
            "lib_iou": 0.0,
            "lib_corr": 0.0,
            "lib_n_used": 0,
            "lib_used_gate": 0,
            "lib_dir": None,
            "lib_dist": float("nan"),
            "lib_roll": float("nan"),
            "lib_T_w2c": None,
            "lib_template_contour": "",
        }
    return best


def find_best_bbox_in_next_frame_uvh_render(
    *,
    frame_bgr: np.ndarray,
    prev_bbox: tuple[int, int, int, int],
    prev_mask_small: np.ndarray,
    prev_uvh_pts: np.ndarray,
    cfg: core.TrackConfig,
    view_lib: ViewLibrary | None = None,
    prev_view_id: int = -1,
    prev_view_dir: np.ndarray | None = None,
    angle_gate_deg: float = 0.0,
    ref_contour_rgb: np.ndarray | None = None,
    rigid_ref_uvh_pts: np.ndarray | None = None,
    diff_ref_verts: np.ndarray | None = None,
    diff_ref_faces: np.ndarray | None = None,
    renderfield_ref_ctx: dict[str, Any] | None = None,
    warm_prev_bbox: tuple[int, int, int, int] | None = None,
) -> dict[str, Any]:
    """
    伪3D(uvh)跟踪（不依赖 open3d）：
      - 每个候选 bbox 生成 uvh 点云（由 th_68 灰度截断图构造）
      - （可选）先做“多视角渲染检索”：候选轮廓在 view_lib 中取 top-k 视点
      - 再对 (scale, rot, tx, ty) 做迭代优化，使渲染 mask+depth 与 reference（首帧/上一帧）最相似
      - 相似度默认：filled-mask IoU + depth 相关性（更贴近“伪3D mesh”匹配）

    返回字段与 core.find_best_bbox_in_next_frame 对齐，并额外包含：
      uvh_render_used, uvh_render_sim, uvh_render_pts（仅用于主循环更新，不写入 json）
    """
    H, W = frame_bgr.shape[:2]
    px, py, pw, ph = prev_bbox
    cx0 = px + pw // 2
    cy0 = py + ph // 2

    step = int(max(1, cfg.search_step))
    rad = int(max(step, cfg.search_radius))
    rad_full = int(rad)
    
    # 优化：两阶段搜索（粗步长 + 细步长局部），默认开启
    two_stage_enable = bool(_env_bool("MIS_UVH_RENDER_TWO_STAGE_ENABLE", True))
    two_stage_coarse_step = int(_env_int("MIS_UVH_TWO_STAGE_COARSE_STEP", max(step, step * 2)))  # 粗步长默认2倍
    two_stage_fine_radius = int(_env_int("MIS_UVH_TWO_STAGE_FINE_RADIUS", max(step * 2, 16)))  # 细搜索半径
    
    if warm_prev_bbox is not None and bool(_env_bool("MIS_UVH_WARM_START_ENABLE", True)):
        try:
            px_w, py_w, pw_w, ph_w = [int(v) for v in warm_prev_bbox]
            cx_p = int(px + pw // 2)
            cy_p = int(py + ph // 2)
            cx_w = int(px_w + pw_w // 2)
            cy_w = int(py_w + ph_w // 2)
            mot = float(abs(cx_p - cx_w) + abs(cy_p - cy_w))
            pad = float(_env_float("MIS_UVH_WARM_PAD_PX", 28.0))
            rad_cap = int(max(float(step) * 2.0, mot + pad))
            rad = int(min(rad, rad_cap))
        except Exception:
            pass

    # 允许在“同一中心位置”下搜索不同 bbox 尺度（大尺度变化/zoom 时更稳）
    # 优化：减少默认尺度变体以提速（0.95,1.0,1.05）
    bbox_scales = _parse_float_list("MIS_UVH_RENDER_BBOX_SCALES", "0.95,1.0,1.05")
    bbox_scales2: list[float] = []
    for s in bbox_scales:
        try:
            sf = float(s)
            if np.isfinite(sf) and (0.25 <= sf <= 4.0):
                bbox_scales2.append(sf)
        except Exception:
            continue
    if not bbox_scales2:
        bbox_scales2 = [1.0]
    bbox_scales2 = sorted(set([float(s) for s in bbox_scales2]))[:5]  # 优化：从9降至5
    max_scale = float(max(bbox_scales2)) if bbox_scales2 else 1.0

    render_max_points = _env_int("MIS_UVH_RENDER_MAX_POINTS", 5000)  # 优化：从10000降至5000提速
    render_dist_mult = _env_float("MIS_UVH_RENDER_DIST_MULT", 3.0)
    rng = np.random.default_rng(0)

    prev_pts = np.asarray(prev_uvh_pts, dtype=np.float32).reshape(-1, 3)
    if int(prev_pts.shape[0]) < 200:
        # 退化：没有可靠的伪3D点云
        best2 = core.find_best_bbox_in_next_frame(frame_bgr, prev_bbox, np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), np.uint8), prev_mask_small, cfg)
        best2.update({
            "uvh_icp_used": 0,
            "uvh_icp_fitness": float("nan"),
            "uvh_icp_rmse": float("nan"),
            "uvh_render_used": 0,
            "uvh_render_sim": float("nan"),
            "uvh_render_pts": None,
            "uvh_render_prev_rgb": None,
            "uvh_render_cur_rgb": None,
            "uvh_align_scale": float("nan"),
            "uvh_align_rot_deg": float("nan"),
            "uvh_align_tx": float("nan"),
            "uvh_align_ty": float("nan"),
            "uvh_mesh_sim": float("nan"),
            "uvh_mesh_iou": float("nan"),
            "uvh_mesh_depthcorr": float("nan"),
            "uvh_sem_sim": float("nan"),
            "uvh_sem_used": 0,
            "uvh_area_ratio": float("nan"),
            "uvh_area_pen": float("nan"),
            "uvh_scale_pen": float("nan"),
            "uvh_bbox_scale": float("nan"),
            "uvh_bbox_scale_pen": float("nan"),
            "uvh_view_id": -1,
            "uvh_view_score": float("nan"),
            "uvh_view_used_gate": 0,
            "uvh_view_n_used": 0,
            "uvh_view_dir": None,
            "uvh_view_dist": float("nan"),
            "uvh_view_roll": float("nan"),
            "uvh_rf_attempted": 0,
            "uvh_rf_used": 0,
            "uvh_rf_best_loss": float("nan"),
            "uvh_rf_steps": 0,
            "uvh_rf_lr": float("nan"),
            "uvh_rf_bbox_used": 0,
            "uvh_rf_bbox_x": int(best2["bbox"][0]),
            "uvh_rf_bbox_y": int(best2["bbox"][1]),
            "uvh_rf_bbox_w": int(best2["bbox"][2]),
            "uvh_rf_bbox_h": int(best2["bbox"][3]),
            "uvh_rf_error": "",
        })
        return best2

    # 本帧整图只做一次 th_68；_eval_one 与各 rebuild 仅从全图裁剪（避免两阶段/recover 重复算子图 th_68）
    use_frame_th68 = bool(_env_bool("MIS_UVH_FRAME_TH68_ENABLE", True))
    th_frame_full: np.ndarray | None = None
    mask_frame_full: np.ndarray | None = None
    th_u8_frame_full = float("nan")
    if use_frame_th68:
        try:
            tf, mf, u8f = core.compute_th_gray_truncated(frame_bgr, cfg.th_percent, cfg)
            tf = np.asarray(tf, dtype=np.float32)
            mf = np.asarray(mf, dtype=np.uint8)
            if int(tf.shape[0]) != int(H) or int(tf.shape[1]) != int(W):
                tf = cv2.resize(tf, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            if int(mf.shape[0]) != int(H) or int(mf.shape[1]) != int(W):
                mf = cv2.resize(mf, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            th_frame_full, mask_frame_full = tf, mf
            th_u8_frame_full = float(u8f)
        except Exception:
            th_frame_full, mask_frame_full = None, None
            use_frame_th68 = False

    def _frame_th_rebuild_kw() -> dict[str, Any]:
        if use_frame_th68 and isinstance(th_frame_full, np.ndarray) and isinstance(mask_frame_full, np.ndarray):
            return {
                "frame_th_gray": th_frame_full,
                "frame_mask_u8": mask_frame_full,
                "frame_th_uint8": float(th_u8_frame_full),
            }
        return {"frame_th_gray": None, "frame_mask_u8": None, "frame_th_uint8": None}

    use_viewlib = bool(view_lib is not None and isinstance(view_lib, ViewLibrary))
    # reference 渲染缓存（按 view_id）
    ref_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    # 默认视角（无 view_lib 时使用“俯视”）
    if use_viewlib:
        image_wh = tuple(getattr(view_lib, "image_wh", (int(cfg.semantic_image_size[0]), int(cfg.semantic_image_size[1]))))
        cam_K = np.asarray(getattr(view_lib, "K", _default_K_for_library(image_wh)), dtype=np.float32).reshape(3, 3)
    else:
        image_wh = (int(cfg.semantic_image_size[0]), int(cfg.semantic_image_size[1]))
        cam_K = _default_K_for_library(image_wh)
    cam_T_w2c = None
    prev_render_rgb = None
    prev_filled_u8 = None
    depth_prev = None
    if not use_viewlib:
        pmin = np.nanmin(prev_pts, axis=0)
        pmax = np.nanmax(prev_pts, axis=0)
        center = 0.5 * (pmin + pmax)
        radius = float(0.5 * np.linalg.norm((pmax - pmin)))
        radius = float(max(1e-3, radius))
        dist = float(max(1e-3, float(render_dist_mult) * radius))
        eye = center + np.array([0.0, 0.0, 1.0], dtype=np.float32) * float(dist)
        cam_T_w2c = _look_at_T_w2c(eye_w=eye, target_w=center)
        mk_prev, depth_prev = _render_mask_depth_from_points(pts_w=prev_pts, T_w2c=cam_T_w2c, K=cam_K, image_wh=image_wh)
        prev_render_rgb, prev_filled_u8 = _mask_to_contour_rgb(mk_prev, image_wh, thickness=2)

    # ===== uvh-render：融合评分（mesh + 渲染轮廓语义）+ 约束惩罚（面积/尺度/bbox尺度）=====
    w_mesh = float(_env_float("MIS_UVH_SIM_W_MESH", 0.75))
    w_sem = float(_env_float("MIS_UVH_SIM_W_SEMANTIC", 0.25))
    w_mesh = float(max(0.0, w_mesh))
    w_sem = float(max(0.0, w_sem))
    wsum = float(w_mesh + w_sem)
    if wsum <= 1e-9:
        w_mesh, w_sem, wsum = 1.0, 0.0, 1.0
    w_mesh = float(w_mesh) / float(wsum)
    w_sem = float(w_sem) / float(wsum)

    pen_w_area = float(max(0.0, float(_env_float("MIS_UVH_PEN_W_AREA", 0.15))))
    pen_w_scale = float(max(0.0, float(_env_float("MIS_UVH_PEN_W_SCALE", 0.10))))
    pen_w_bbox = float(max(0.0, float(_env_float("MIS_UVH_PEN_W_BBOXSCALE", 0.0))))

    sem_enable = bool(_env_bool("MIS_UVH_SEMANTIC_ENABLE", True))
    sem_min_mesh = float(_env_float("MIS_UVH_SEMANTIC_MIN_MESH_SIM", 0.25))

    def _uvh_fuse_sim(
        *,
        mesh_sim: float,
        ref_contour_rgb: Optional[np.ndarray],
        cur_contour_rgb: Optional[np.ndarray],
        ref_filled_u8: Optional[np.ndarray],
        cur_filled_u8: Optional[np.ndarray],
        align_scale: float,
        bbox_scale: float,
    ) -> dict[str, Any]:
        ms = float(mesh_sim)
        if not np.isfinite(ms):
            ms = 0.0
        ms = float(max(0.0, min(1.0, ms)))

        sem_used = 0
        sem_sim = 0.0
        if sem_enable and (w_sem > 1e-9) and (ms >= float(sem_min_mesh)):
            if isinstance(ref_contour_rgb, np.ndarray) and isinstance(cur_contour_rgb, np.ndarray):
                try:
                    sem_sim = float(core.semantic_similarity(ref_contour_rgb, cur_contour_rgb, cfg))
                    if not np.isfinite(sem_sim):
                        sem_sim = 0.0
                    sem_sim = float(max(0.0, min(1.0, sem_sim)))
                    sem_used = 1
                except Exception:
                    sem_used = 0
                    sem_sim = 0.0

        area_ratio = float("nan")
        area_pen = float("nan")
        if isinstance(ref_filled_u8, np.ndarray) and isinstance(cur_filled_u8, np.ndarray):
            try:
                a_ref = float(int(np.count_nonzero(np.asarray(ref_filled_u8, dtype=np.uint8) > 0)))
                a_cur = float(int(np.count_nonzero(np.asarray(cur_filled_u8, dtype=np.uint8) > 0)))
                area_ratio = float((a_cur + 1.0) / (a_ref + 1.0))
                area_ratio = float(max(1e-6, area_ratio))
                area_pen = float(abs(math.log(area_ratio)))
            except Exception:
                area_ratio = float("nan")
                area_pen = float("nan")

        scale_pen = float("nan")
        try:
            if np.isfinite(float(align_scale)):
                scale_pen = float(abs(math.log(max(1e-6, float(align_scale)))))
        except Exception:
            scale_pen = float("nan")

        bbox_pen = float("nan")
        try:
            if np.isfinite(float(bbox_scale)):
                bbox_pen = float(abs(math.log(max(1e-6, float(bbox_scale)))))
        except Exception:
            bbox_pen = float("nan")

        sim_raw = float(w_mesh) * float(ms) + float(w_sem) * float(sem_sim)
        if np.isfinite(area_pen):
            sim_raw = float(sim_raw) - float(pen_w_area) * float(area_pen)
        if np.isfinite(scale_pen):
            sim_raw = float(sim_raw) - float(pen_w_scale) * float(scale_pen)
        if np.isfinite(bbox_pen):
            sim_raw = float(sim_raw) - float(pen_w_bbox) * float(bbox_pen)
        sim01 = 0.0 if (not np.isfinite(sim_raw)) else float(max(0.0, min(1.0, sim_raw)))
        return {
            "sim_total": float(sim01),
            "mesh_sim": float(ms),
            "sem_sim": float(sem_sim),
            "sem_used": int(sem_used),
            "area_ratio": float(area_ratio),
            "area_pen": float(area_pen),
            "scale_pen": float(scale_pen),
            "bbox_scale": float(bbox_scale),
            "bbox_pen": float(bbox_pen),
        }

    def _uvh_decision_score(
        *,
        sim_total: float,
        quick: float,
        bbox: tuple[int, int, int, int],
        dx: int,
        dy: int,
        rad_now: int,
        view_id: int = -1,
        view_dir: Any = None,
        rigid_score: float = float("nan"),
        local_score: float = float("nan"),
        coverage_ratio: float = float("nan"),
        area_ratio: float = float("nan"),
        area_pen: float = float("nan"),
        scale_pen: float = float("nan"),
        bbox_pen: float = float("nan"),
    ) -> dict[str, float]:
        """
        统一决策分数：
        - sim_total: 当前阶段主相似度（mesh/render/diff 后）
        - quick: 2D quick 粗筛分
        - rigid/local: 作为候选增强项
        - area/scale/bbox: 显式罚项
        - motion: 相对上一帧中心位移罚项
        """
        s_sim = 0.0 if (not np.isfinite(sim_total)) else float(max(0.0, min(1.0, sim_total)))
        s_quick = 0.0 if (not np.isfinite(quick)) else float(max(0.0, min(1.0, quick)))
        s_rigid = 0.0 if (not np.isfinite(rigid_score)) else float(max(0.0, min(1.0, rigid_score)))
        s_local = 0.0 if (not np.isfinite(local_score)) else float(max(0.0, min(1.0, local_score)))
        s_cov = 0.0 if (not np.isfinite(coverage_ratio)) else float(max(0.0, min(1.0, coverage_ratio)))

        motion = float((abs(int(dx)) + abs(int(dy))) / float(max(1, rad_now)))
        motion = float(max(0.0, motion))
        ar = float(area_ratio) if np.isfinite(area_ratio) else float("nan")
        ap = 0.0 if (not np.isfinite(area_pen)) else float(max(0.0, area_pen))
        sp = 0.0 if (not np.isfinite(scale_pen)) else float(max(0.0, scale_pen))
        bp = 0.0 if (not np.isfinite(bbox_pen)) else float(max(0.0, bbox_pen))

        # 时间连续性：上一帧 bbox IoU / 视点方向一致性
        temporal_iou = 0.0
        try:
            bx, by, bw, bh = [int(v) for v in bbox]
            px0, py0, pw0, ph0 = [int(v) for v in prev_bbox]
            ix0 = max(bx, px0)
            iy0 = max(by, py0)
            ix1 = min(bx + bw, px0 + pw0)
            iy1 = min(by + bh, py0 + ph0)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            union = max(1, bw * bh + pw0 * ph0 - inter)
            temporal_iou = float(inter) / float(union)
        except Exception:
            temporal_iou = 0.0

        temporal_view = 0.0
        try:
            if isinstance(prev_view_dir, np.ndarray):
                vd = np.asarray(view_dir, dtype=np.float32).reshape(-1)
                vp = np.asarray(prev_view_dir, dtype=np.float32).reshape(-1)
                if vd.size >= 3 and vp.size >= 3:
                    vd = vd[:3]
                    vp = vp[:3]
                    nd = float(np.linalg.norm(vd))
                    npv = float(np.linalg.norm(vp))
                    if nd > 1e-6 and npv > 1e-6:
                        temporal_view = float(np.dot(vd / nd, vp / npv))
                        temporal_view = float(max(0.0, min(1.0, 0.5 * (temporal_view + 1.0))))
            elif int(prev_view_id) >= 0 and int(view_id) >= 0:
                temporal_view = 1.0 if int(prev_view_id) == int(view_id) else 0.0
        except Exception:
            temporal_view = 0.0

        w_sim = float(max(0.0, _env_float("MIS_UVH_DECISION_W_SIM", 0.68)))
        w_quick = float(max(0.0, _env_float("MIS_UVH_DECISION_W_QUICK", 0.16)))
        w_rigid = float(max(0.0, _env_float("MIS_UVH_DECISION_W_RIGID", 0.10)))
        w_local = float(max(0.0, _env_float("MIS_UVH_DECISION_W_LOCAL", 0.06)))
        w_cov = float(max(0.0, _env_float("MIS_UVH_DECISION_W_COVERAGE", 0.16)))
        w_temp_iou = float(max(0.0, _env_float("MIS_UVH_DECISION_W_TEMP_IOU", 0.08)))
        w_temp_view = float(max(0.0, _env_float("MIS_UVH_DECISION_W_TEMP_VIEW", 0.04)))
        pen_motion = float(max(0.0, _env_float("MIS_UVH_DECISION_PEN_MOTION", float(cfg.spatial_penalty))))
        pen_area2 = float(max(0.0, _env_float("MIS_UVH_DECISION_PEN_AREA", 0.03)))
        pen_scale2 = float(max(0.0, _env_float("MIS_UVH_DECISION_PEN_SCALE", 0.02)))
        pen_bbox2 = float(max(0.0, _env_float("MIS_UVH_DECISION_PEN_BBOX", 0.02)))
        cov_soft_min = float(max(0.0, min(1.0, _env_float("MIS_UVH_DECISION_COVERAGE_SOFT_MIN", 0.92))))
        cov_under_pen = float(max(0.0, _env_float("MIS_UVH_DECISION_PEN_COVERAGE", 0.35)))
        cov_under_pen2 = float(max(0.0, _env_float("MIS_UVH_DECISION_PEN_COVERAGE_QUAD", 0.80)))
        cov_hard_min = float(max(0.0, min(cov_soft_min, _env_float("MIS_UVH_DECISION_COVERAGE_HARD_MIN", 0.82))))
        cov_hard_pen = float(max(0.0, _env_float("MIS_UVH_DECISION_PEN_COVERAGE_HARD", 0.55)))
        cov_bonus_w = float(max(0.0, _env_float("MIS_UVH_DECISION_W_COVERAGE_BONUS", 0.10)))
        cov_gamma = float(max(0.5, _env_float("MIS_UVH_DECISION_COVERAGE_GAMMA", 1.6)))
        area_under_min = float(max(0.0, _env_float("MIS_UVH_DECISION_AREA_UNDER_MIN", 0.96)))
        area_under_pen = float(max(0.0, _env_float("MIS_UVH_DECISION_PEN_AREA_UNDER", 0.18)))
        cov_gap = float(max(0.0, cov_soft_min - s_cov)) if np.isfinite(coverage_ratio) else 0.0
        cov_gap2 = float(cov_gap * cov_gap)
        cov_hard_gap = float(max(0.0, cov_hard_min - s_cov)) if np.isfinite(coverage_ratio) else 0.0
        cov_reward = float(s_cov ** cov_gamma) if np.isfinite(coverage_ratio) else 0.0
        cov_bonus = float(max(0.0, s_cov - cov_soft_min)) if np.isfinite(coverage_ratio) else 0.0
        area_under_gap = float(max(0.0, area_under_min - ar)) if np.isfinite(ar) else 0.0
        small_box_cov_pen = float(area_under_gap * cov_gap) if (np.isfinite(ar) and np.isfinite(coverage_ratio)) else 0.0

        score = (
            float(w_sim) * float(s_sim)
            + float(w_quick) * float(s_quick)
            + float(w_rigid) * float(s_rigid)
            + float(w_local) * float(s_local)
            + float(w_cov) * float(cov_reward)
            + float(cov_bonus_w) * float(cov_bonus)
            + float(w_temp_iou) * float(temporal_iou)
            + float(w_temp_view) * float(temporal_view)
            - float(pen_motion) * float(motion)
            - float(pen_area2) * float(ap)
            - float(pen_scale2) * float(sp)
            - float(pen_bbox2) * float(bp)
            - float(cov_under_pen) * float(cov_gap)
            - float(cov_under_pen2) * float(cov_gap2)
            - float(cov_hard_pen) * float(cov_hard_gap)
            - float(area_under_pen) * float(small_box_cov_pen)
        )
        return {
            "score": float(score),
            "sim": float(s_sim),
            "quick": float(s_quick),
            "rigid": float(s_rigid),
            "local": float(s_local),
            "coverage": float(s_cov),
            "coverage_reward": float(cov_reward),
            "coverage_bonus": float(cov_bonus),
            "motion_pen": float(motion),
            "temporal_iou": float(temporal_iou),
            "temporal_view": float(temporal_view),
            "area_ratio": float(ar) if np.isfinite(ar) else float("nan"),
            "area_pen": float(ap),
            "scale_pen": float(sp),
            "bbox_pen": float(bp),
            "coverage_gap": float(cov_gap),
            "coverage_gap_quad": float(cov_gap2),
            "coverage_hard_gap": float(cov_hard_gap),
            "small_box_cov_pen": float(small_box_cov_pen),
        }

    def _eval_one(
        rad_now: int,
        *,
        step_use: int | None = None,
        cx_anchor: int | None = None,
        cy_anchor: int | None = None,
    ) -> dict[str, Any] | None:
        step_loc = int(max(1, int(step_use) if step_use is not None else step))
        if cx_anchor is None or cy_anchor is None:
            cxa, cya = int(cx0), int(cy0)
            px_eff, py_eff = int(px), int(py)
        else:
            cxa, cya = int(cx_anchor), int(cy_anchor)
            px_eff = int(max(0, min(W - int(pw), int(cxa) - int(pw) // 2)))
            py_eff = int(max(0, min(H - int(ph), int(cya) - int(ph) // 2)))
        ex = int(max(0, round(0.5 * float(pw) * float(max_scale - 1.0))))
        ey = int(max(0, round(0.5 * float(ph) * float(max_scale - 1.0))))
        sx0 = int(max(0, px_eff - int(rad_now) - int(ex)))
        sy0 = int(max(0, py_eff - int(rad_now) - int(ey)))
        sx1 = int(min(W, px_eff + pw + int(rad_now) + int(ex)))
        sy1 = int(min(H, py_eff + ph + int(rad_now) + int(ey)))
        sw = int(max(0, sx1 - sx0))
        sh = int(max(0, sy1 - sy0))
        if sw < 4 or sh < 4:
            return None
        if (
            use_frame_th68
            and isinstance(th_frame_full, np.ndarray)
            and isinstance(mask_frame_full, np.ndarray)
            and int(th_frame_full.shape[0]) == int(H)
            and int(th_frame_full.shape[1]) == int(W)
        ):
            th_search = np.asarray(
                th_frame_full[int(sy0) : int(sy1), int(sx0) : int(sx1)],
                dtype=np.float32,
                order="C",
            )
            mask_search = np.asarray(
                mask_frame_full[int(sy0) : int(sy1), int(sx0) : int(sx1)],
                dtype=np.uint8,
                order="C",
            )
            th_u8_search = float(th_u8_frame_full)
        else:
            search_roi = frame_bgr[sy0:sy1, sx0:sx1]
            th_search, mask_search, th_u8_search = core.compute_th_gray_truncated(search_roi, cfg.th_percent, cfg)
            th_search = np.asarray(th_search, dtype=np.float32)
            mask_search = np.asarray(mask_search, dtype=np.uint8)
        if th_search.ndim != 2 or mask_search.ndim != 2:
            return None
        if th_search.shape[0] != sh or th_search.shape[1] != sw:
            th_search = cv2.resize(th_search, (sw, sh), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        if mask_search.shape[0] != sh or mask_search.shape[1] != sw:
            mask_search = cv2.resize(mask_search, (sw, sh), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
        search_ctx = {
            "sx0": int(sx0),
            "sy0": int(sy0),
            "sw": int(sw),
            "sh": int(sh),
            "th_gray": np.asarray(th_search, dtype=np.float32),
            "mask_u8": np.asarray(mask_search, dtype=np.uint8),
            "th_uint8": float(th_u8_search),
        }
        rebuild_cache: dict[tuple[int, int, int, int], dict[str, Any]] = {}
        pts_cur_cache: dict[tuple[int, int, int, int], np.ndarray] = {}

        cands_xy: dict[tuple[int, int, int, int], tuple[int, int, int, int, int, int]] = {}
        for dy in range(-rad_now, rad_now + 1, step_loc):
            for dx in range(-rad_now, rad_now + 1, step_loc):
                for sc in bbox_scales2:
                    ww0 = int(max(16, round(float(pw) * float(sc))))
                    hh0 = int(max(16, round(float(ph) * float(sc))))
                    x0 = cxa + int(dx) - ww0 // 2
                    y0 = cya + int(dy) - hh0 // 2
                    x, y, ww, hh = core.clamp_bbox(x0, y0, ww0, hh0, W, H)
                    dx_eff = int((x + ww // 2) - cx0)
                    dy_eff = int((y + hh // 2) - cy0)
                    cands_xy[(int(x), int(y), int(ww), int(hh))] = (int(x), int(y), int(ww), int(hh), int(dx_eff), int(dy_eff))

        scored_quick = []
        for (x, y, ww, hh, dx, dy) in cands_xy.values():
            # 直接从 search_roi 的 mask 裁剪，保持阈值一致
            xi = int(x - sx0)
            yi = int(y - sy0)
            if xi < 0 or yi < 0 or (xi + int(ww) > sw) or (yi + int(hh) > sh):
                continue
            mk = mask_search[yi:yi + int(hh), xi:xi + int(ww)]
            ms = _mask64_from_mask_u8(mk)
            q = core.quick_mask_score(prev_mask_small, ms)
            dist2 = (abs(dx) + abs(dy)) / float(max(1, rad_now))
            q2 = float(q) - float(cfg.spatial_penalty) * float(dist2)
            scored_quick.append((q2, q, x, y, ww, hh, dx, dy))
        scored_quick.sort(key=lambda t: t[0], reverse=True)
        top = scored_quick[: max(1, int(cfg.topk_semantic))]

        cand_list: list[dict[str, Any]] = []
        best = None
        best_score = -1e9
        
        # 核心优化：Early Exit 提前退出机制
        early_exit_threshold = float(_env_float("MIS_UVH_EARLY_EXIT_THRESHOLD", 0.75))  # 高分阈值
        max_consecutive_low = int(_env_int("MIS_UVH_MAX_CONSECUTIVE_LOW", 2))  # 允许连续低分次数
        consecutive_low_count = 0
        
        for cand_idx, (_q2, q, x, y, ww, hh, dx, dy) in enumerate(top):
            xi = int(x - sx0)
            yi = int(y - sy0)
            if xi < 0 or yi < 0 or (xi + int(ww) > sw) or (yi + int(hh) > sh):
                continue
            th_gray = th_search[yi:yi + int(hh), xi:xi + int(ww)].astype(np.float32, copy=False)
            mask_u8 = mask_search[yi:yi + int(hh), xi:xi + int(ww)].astype(np.uint8, copy=False)
            mask_u8 = _apply_reference_mask_guidance(
                th_gray=np.asarray(th_gray, dtype=np.float32),
                mask_u8=np.asarray(mask_u8, dtype=np.uint8),
                ref_mask_small=np.asarray(prev_mask_small, dtype=np.uint8),
            )
            th_uint8 = float(th_u8_search)

            cnt = _extract_contour_prefer_center(mask_u8)
            if cnt is None:
                cnt = core.extract_largest_contour(mask_u8)
            if cnt is None:
                contour_rgb = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
            else:
                cnt2 = core.resize_contour(cnt, (ww, hh), cfg.semantic_image_size)
                contour_rgb = core.contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)

            bb_pts_k = (int(x), int(y), int(ww), int(hh))
            if bb_pts_k in pts_cur_cache:
                pts_cur = pts_cur_cache[bb_pts_k]
            else:
                pts_cur = _uvh_points_from_th_roi(
                    th_gray_truncated=th_gray,
                    bbox=bb_pts_k,
                    z_scale=float(cfg.mesh_z_scale),
                    max_points=int(render_max_points),
                    rng=rng,
                    deterministic=True,
                )
                pts_cur_cache[bb_pts_k] = pts_cur
            mesh_sim = 0.0
            sem_sim = float("nan")
            sem_used = 0
            area_ratio = float("nan")
            area_pen = float("nan")
            scale_pen = float("nan")
            bbox_scale = float(math.sqrt((float(ww) * float(hh)) / float(max(1.0, float(pw) * float(ph) + 1e-6))))
            bbox_pen = float("nan")
            sim_total = 0.0
            cur_render_rgb = None
            prev_r = prev_render_rgb
            ref_filled_used = None
            align_scale = float("nan")
            align_rot_deg = float("nan")
            align_tx = float("nan")
            align_ty = float("nan")
            sim_iou = float("nan")
            sim_depth_corr = float("nan")
            uvh_view_id = -1
            uvh_view_score = float("nan")
            uvh_view_used_gate = 0
            uvh_view_n_used = 0
            uvh_view_dir = None
            uvh_view_dist = float("nan")
            uvh_view_roll = float("nan")
            if int(pts_cur.shape[0]) >= 200:
                if use_viewlib and view_lib is not None:
                    # 视点检索：先用候选 contour 在 view library 找 top-k
                    topk = int(max(1, _env_int("MIS_UVH_VIEWLIB_TOPK", 2)))
                    gate_retry_min = float(_env_float("MIS_UVH_VIEWLIB_GATE_RETRY_MIN", 0.30))
                    mret = match_contour_to_library_topk(
                        lib=view_lib,
                        contour_rgb=contour_rgb,
                        prev_view_dir=prev_view_dir,
                        angle_gate_deg=float(angle_gate_deg),
                        topk=int(topk),
                    )
                    v_ids = [int(v) for v in (mret.get("view_ids", []) or [])]
                    v_scores = [float(v) for v in (mret.get("scores", []) or [])]
                    uvh_view_used_gate = int(mret.get("used_gate", 0))
                    uvh_view_n_used = int(mret.get("n_used", 0))
                    if uvh_view_used_gate == 1 and (len(v_scores) <= 0 or (np.isfinite(v_scores[0]) and float(v_scores[0]) < float(gate_retry_min))):
                        mret2 = match_contour_to_library_topk(
                            lib=view_lib,
                            contour_rgb=contour_rgb,
                            prev_view_dir=None,
                            angle_gate_deg=0.0,
                            topk=int(topk),
                        )
                        v_ids2 = [int(v) for v in (mret2.get("view_ids", []) or [])]
                        v_scores2 = [float(v) for v in (mret2.get("scores", []) or [])]
                        if v_scores2 and ((not v_scores) or (np.isfinite(v_scores2[0]) and float(v_scores2[0]) > float(v_scores[0]))):
                            mret = mret2
                            v_ids = v_ids2
                            v_scores = v_scores2
                            uvh_view_used_gate = int(mret2.get("used_gate", 0))
                            uvh_view_n_used = int(mret2.get("n_used", 0))

                    if not v_ids:
                        v_ids = [-1]
                        v_scores = [float("nan")]

                    always_try_all = bool(_env_bool("MIS_UVH_VIEWLIB_TRY_ALL", False))
                    fallback_sim = float(_env_float("MIS_UVH_VIEWLIB_FALLBACK_SIM", 0.80))

                    best_ali = None
                    best_total = -1e9
                    best_vid = -1
                    best_vscore = float("nan")
                    best_ref_rgb = None
                    best_ref_filled = None
                    best_fused = None

                    for jv, vid in enumerate(v_ids):
                        if vid < 0 or vid >= int(view_lib.ids.size):
                            continue
                        if (jv > 0) and (not always_try_all) and (best_ali is not None) and np.isfinite(best_total) and float(best_total) >= float(fallback_sim):
                            break
                        if int(vid) not in ref_cache:
                            Tw2c = np.asarray(view_lib.T_w2c[int(vid)], dtype=np.float32)
                            mk_ref, dep_ref = _render_mask_depth_from_points(pts_w=prev_pts, T_w2c=Tw2c, K=cam_K, image_wh=image_wh)
                            ref_rgb, ref_filled = _mask_to_contour_rgb(mk_ref, image_wh, thickness=2)
                            ref_cache[int(vid)] = (ref_rgb, ref_filled, dep_ref)
                        ref_rgb, ref_filled, dep_ref = ref_cache[int(vid)]
                        Tw2c = np.asarray(view_lib.T_w2c[int(vid)], dtype=np.float32)
                        ali = _align_uvh_points_iterative(
                            pts_ref=prev_pts,
                            pts_cur=pts_cur,
                            cam_T_w2c=Tw2c,
                            cam_K=cam_K,
                            image_wh=image_wh,
                            ref_filled_u8=ref_filled,
                            ref_depth=dep_ref,
                        )
                        s0 = float(ali.get("sim", 0.0))
                        cur_rgb0 = ali.get("render_rgb", None)
                        cur_fill0 = ali.get("render_filled", None)
                        fused0 = _uvh_fuse_sim(
                            mesh_sim=float(s0),
                            ref_contour_rgb=ref_rgb,
                            cur_contour_rgb=(cur_rgb0 if isinstance(cur_rgb0, np.ndarray) else None),
                            ref_filled_u8=ref_filled,
                            cur_filled_u8=(cur_fill0 if isinstance(cur_fill0, np.ndarray) else None),
                            align_scale=float(ali.get("scale", float("nan"))),
                            bbox_scale=float(bbox_scale),
                        )
                        t0 = float(fused0.get("sim_total", 0.0))
                        if (best_ali is None) or (np.isfinite(t0) and float(t0) > float(best_total)):
                            best_ali = ali
                            best_total = float(t0)
                            best_vid = int(vid)
                            best_vscore = float(v_scores[jv]) if (jv < len(v_scores)) else float("nan")
                            best_ref_rgb = ref_rgb
                            best_ref_filled = ref_filled
                            best_fused = fused0

                    if best_ali is not None:
                        uvh_view_id = int(best_vid)
                        uvh_view_score = float(best_vscore)
                        try:
                            uvh_view_dir = np.asarray(view_lib.dirs[int(best_vid)], dtype=np.float32).reshape(-1).tolist()
                        except Exception:
                            uvh_view_dir = None
                        try:
                            uvh_view_dist = float(view_lib.dists[int(best_vid)])
                        except Exception:
                            uvh_view_dist = float("nan")
                        try:
                            uvh_view_roll = float(view_lib.rolls[int(best_vid)])
                        except Exception:
                            uvh_view_roll = float("nan")

                        mesh_sim = float(best_ali.get("sim", 0.0))
                        cur_render_rgb = best_ali.get("render_rgb", None)
                        align_scale = float(best_ali.get("scale", float("nan")))
                        align_rot_deg = float(best_ali.get("rot_deg", float("nan")))
                        align_tx = float(best_ali.get("tx", float("nan")))
                        align_ty = float(best_ali.get("ty", float("nan")))
                        sim_iou = float(best_ali.get("iou", float("nan")))
                        sim_depth_corr = float(best_ali.get("depth_corr", float("nan")))
                        prev_r = best_ref_rgb
                        ref_filled_used = best_ref_filled
                        if isinstance(best_fused, dict):
                            sim_total = float(best_fused.get("sim_total", 0.0))
                            sem_sim = float(best_fused.get("sem_sim", float("nan")))
                            sem_used = int(best_fused.get("sem_used", 0))
                            area_ratio = float(best_fused.get("area_ratio", float("nan")))
                            area_pen = float(best_fused.get("area_pen", float("nan")))
                            scale_pen = float(best_fused.get("scale_pen", float("nan")))
                            bbox_pen = float(best_fused.get("bbox_pen", float("nan")))
                    else:
                        prev_r = None
                else:
                    ali = _align_uvh_points_iterative(
                        pts_ref=prev_pts,
                        pts_cur=pts_cur,
                        cam_T_w2c=np.asarray(cam_T_w2c, dtype=np.float32),
                        cam_K=np.asarray(cam_K, dtype=np.float32),
                        image_wh=image_wh,
                        ref_filled_u8=np.asarray(prev_filled_u8, dtype=np.uint8),
                        ref_depth=np.asarray(depth_prev, dtype=np.float32),
                    )
                    mesh_sim = float(ali.get("sim", 0.0))
                    cur_render_rgb = ali.get("render_rgb", None)
                    cur_fill = ali.get("render_filled", None)
                    align_scale = float(ali.get("scale", float("nan")))
                    align_rot_deg = float(ali.get("rot_deg", float("nan")))
                    align_tx = float(ali.get("tx", float("nan")))
                    align_ty = float(ali.get("ty", float("nan")))
                    sim_iou = float(ali.get("iou", float("nan")))
                    sim_depth_corr = float(ali.get("depth_corr", float("nan")))
                    prev_r = prev_render_rgb
                    ref_filled_used = prev_filled_u8

                    fused = _uvh_fuse_sim(
                        mesh_sim=float(mesh_sim),
                        ref_contour_rgb=(prev_r if isinstance(prev_r, np.ndarray) else None),
                        cur_contour_rgb=(cur_render_rgb if isinstance(cur_render_rgb, np.ndarray) else None),
                        ref_filled_u8=(ref_filled_used if isinstance(ref_filled_used, np.ndarray) else None),
                        cur_filled_u8=(cur_fill if isinstance(cur_fill, np.ndarray) else None),
                        align_scale=float(align_scale),
                        bbox_scale=float(bbox_scale),
                    )
                    sim_total = float(fused.get("sim_total", 0.0))
                    sem_sim = float(fused.get("sem_sim", float("nan")))
                    sem_used = int(fused.get("sem_used", 0))
                    area_ratio = float(fused.get("area_ratio", float("nan")))
                    area_pen = float(fused.get("area_pen", float("nan")))
                    scale_pen = float(fused.get("scale_pen", float("nan")))
                    bbox_pen = float(fused.get("bbox_pen", float("nan")))
                mesh_sim = 0.0 if (not np.isfinite(mesh_sim)) else float(max(0.0, min(1.0, mesh_sim)))
                sim_total = 0.0 if (not np.isfinite(sim_total)) else float(max(0.0, min(1.0, sim_total)))

            dsc = _uvh_decision_score(
                sim_total=float(sim_total),
                quick=float(q),
                bbox=(int(x), int(y), int(ww), int(hh)),
                dx=int(dx),
                dy=int(dy),
                rad_now=int(rad_now),
                view_id=int(uvh_view_id),
                view_dir=uvh_view_dir,
                rigid_score=float("nan"),
                local_score=float("nan"),
                coverage_ratio=float("nan"),
                area_ratio=float(area_ratio),
                area_pen=float(area_pen),
                scale_pen=float(scale_pen),
                bbox_pen=float(bbox_pen),
            )
            score = float(dsc.get("score", 0.0))
            cand = {
                "bbox": (int(x), int(y), int(ww), int(hh)),
                "score": float(score),
                "uvh_decision_score": float(score),
                "semantic": float(sim_total),
                "quick": float(q),
                "dx": int(dx),
                "dy": int(dy),
                "th_gray": th_gray,
                "mask_u8": mask_u8,
                "contour_rgb": contour_rgb,
                "th_uint8": float(th_uint8),
                "uvh_icp_used": 0,
                "uvh_icp_fitness": float("nan"),
                "uvh_icp_rmse": float("nan"),
                "uvh_render_used": 1,
                "uvh_render_sim": float(sim_total),
                "uvh_render_prev_rgb": prev_r,
                "uvh_render_cur_rgb": cur_render_rgb,
                "uvh_render_pts": pts_cur,
                "uvh_decision_stage": "coarse",
                "uvh_coverage_ratio": float("nan"),
                "uvh_coverage_gap": float("nan"),
                "uvh_decision_motion_pen": float(dsc.get("motion_pen", float("nan"))),
                "uvh_decision_coverage": float(dsc.get("coverage", float("nan"))),
                "uvh_decision_coverage_gap": float(dsc.get("coverage_gap", float("nan"))),
                "uvh_decision_temporal_iou": float(dsc.get("temporal_iou", float("nan"))),
                "uvh_decision_temporal_view": float(dsc.get("temporal_view", float("nan"))),
                "uvh_align_scale": float(align_scale),
                "uvh_align_rot_deg": float(align_rot_deg),
                "uvh_align_tx": float(align_tx),
                "uvh_align_ty": float(align_ty),
                "uvh_mesh_sim": float(mesh_sim),
                "uvh_mesh_iou": float(sim_iou),
                "uvh_mesh_depthcorr": float(sim_depth_corr),
                "uvh_sem_sim": float(sem_sim) if np.isfinite(sem_sim) else float("nan"),
                "uvh_sem_used": int(sem_used),
                "uvh_area_ratio": float(area_ratio) if np.isfinite(area_ratio) else float("nan"),
                "uvh_area_pen": float(area_pen) if np.isfinite(area_pen) else float("nan"),
                "uvh_scale_pen": float(scale_pen) if np.isfinite(scale_pen) else float("nan"),
                "uvh_bbox_scale": float(bbox_scale) if np.isfinite(bbox_scale) else float("nan"),
                "uvh_bbox_scale_pen": float(bbox_pen) if np.isfinite(bbox_pen) else float("nan"),
                "uvh_view_id": int(uvh_view_id),
                "uvh_view_score": float(uvh_view_score),
                "uvh_view_used_gate": int(uvh_view_used_gate),
                "uvh_view_n_used": int(uvh_view_n_used),
                "uvh_view_dir": uvh_view_dir,
                "uvh_view_dist": float(uvh_view_dist),
                "uvh_view_roll": float(uvh_view_roll),
                # 刚体点云 coarse-to-fine refine（可选，Open3D）
                "uvh_rigid_used": 0,
                "uvh_rigid_method": "",
                "uvh_rigid_score": float("nan"),
                "uvh_rigid_fitness": float("nan"),
                "uvh_rigid_inlier_ratio": float("nan"),
                "uvh_rigid_rmse": float("nan"),
                "uvh_rigid_overlap": float("nan"),
                "uvh_rigid_normal_consistency": float("nan"),
                "uvh_rigid_corr_count": 0,
                "uvh_rigid_lowcorr_pen": float("nan"),
                "uvh_rigid_median_dist": float("nan"),
                "uvh_rigid_mean_dist": float("nan"),
                "uvh_rigid_voxel": float("nan"),
                "uvh_rigid_bbox_used": 0,
                "uvh_rigid_bbox_x": int(x),
                "uvh_rigid_bbox_y": int(y),
                "uvh_rigid_bbox_w": int(ww),
                "uvh_rigid_bbox_h": int(hh),
                "uvh_rf_attempted": 0,
                "uvh_rf_used": 0,
                "uvh_rf_best_loss": float("nan"),
                "uvh_rf_steps": 0,
                "uvh_rf_lr": float("nan"),
                "uvh_rf_bbox_used": 0,
                "uvh_rf_bbox_x": int(x),
                "uvh_rf_bbox_y": int(y),
                "uvh_rf_bbox_w": int(ww),
                "uvh_rf_bbox_h": int(hh),
                "uvh_rf_error": "",
                "uvh_local_bbox_used": 0,
                "uvh_local_bbox_score": float("nan"),
                "uvh_local_quick": float("nan"),
                "uvh_local_sem": float("nan"),
                "uvh_local_bbox_x": int(x),
                "uvh_local_bbox_y": int(y),
                "uvh_local_bbox_w": int(ww),
                "uvh_local_bbox_h": int(hh),
                # kaolin 可微渲染 refine（可选）
                "uvh_diff_used": 0,
                "uvh_diff_backend": "",
                "uvh_diff_iters": 0,
                "uvh_diff_lr": float("nan"),
                "uvh_diff_sim": float("nan"),
                "uvh_diff_iou": float("nan"),
                "uvh_diff_depthcorr": float("nan"),
                "uvh_diff_sem": float("nan"),
                "uvh_diff_sem_used": 0,
                "uvh_diff_yaw_deg": float("nan"),
                "uvh_diff_pitch_deg": float("nan"),
                "uvh_diff_roll_deg": float("nan"),
                "uvh_diff_dist": float("nan"),
                "uvh_diff_scale": float("nan"),
                "uvh_diff_rot_deg": float("nan"),
                "uvh_diff_tx": float("nan"),
                "uvh_diff_ty": float("nan"),
                "uvh_diff_area_ratio": float("nan"),
            }
            rebuild_cache[tuple(cand["bbox"])] = {
                "bbox": tuple(cand["bbox"]),
                "th_gray": np.asarray(th_gray, dtype=np.float32),
                "mask_u8": np.asarray(mask_u8, dtype=np.uint8),
                "th_uint8": float(th_uint8),
                "contour_rgb": np.asarray(contour_rgb, dtype=np.uint8),
                "uvh_render_pts": np.asarray(pts_cur, dtype=np.float32),
            }
            cand_list.append(cand)
            if float(score) > float(best_score):
                best_score = float(score)
                best = cand
                consecutive_low_count = 0  # 重置连续低分计数
                
                # Early Exit: 如果分数足够高且不是第一个候选，提前退出
                if cand_idx > 0 and best_score > early_exit_threshold:
                    # 已经找到高分候选，跳过剩余的评估
                    break
            else:
                consecutive_low_count += 1
                if consecutive_low_count >= max_consecutive_low:
                    # 连续多个候选都不如当前最佳，提前退出
                    break

        if best is None:
            return None

        # ===== 刚体点云 refine（FPFH/RANSAC + ICP）=====
        rf_global_enable = bool(
            _env_bool("MIS_UVH_RENDERFIELD_ENABLE", False)
            and _env_bool("MIS_UVH_RENDERFIELD_GLOBAL_MATCH_ENABLE", True)
            and isinstance(renderfield_ref_ctx, dict)
        )
        if rf_global_enable:
            try:
                rf_topk_global = int(max(1, _env_int("MIS_UVH_RENDERFIELD_TOPK", 2)))
                rf_blend_global = float(max(0.0, min(1.0, _env_float("MIS_UVH_RENDERFIELD_BLEND_W", 0.20))))
                rf_variants: list[dict[str, Any]] = []
                cand_sorted_rf = sorted(cand_list, key=lambda d: float(d.get("score", -1e9)), reverse=True)
                for rf_rank, cc in enumerate(cand_sorted_rf[: int(rf_topk_global)]):
                    cc["uvh_rf_attempted"] = 1
                    rf_match = _match_candidate_with_renderfield_reference(
                        renderfield_ref_ctx=renderfield_ref_ctx,
                        cand=cc,
                        cfg=cfg,
                        frame_wh=(W, H),
                    )
                    cc["uvh_rf_steps"] = int(rf_match.get("steps", 0) or 0)
                    cc["uvh_rf_best_loss"] = float(rf_match.get("best_loss", float("nan")))
                    cc["uvh_rf_error"] = str(rf_match.get("error", ""))
                    if not bool(rf_match.get("ok", False)):
                        continue
                    cand_rf = dict(cc)
                    cand_rf["uvh_rf_used"] = 1
                    cand_rf["uvh_decision_stage"] = "renderfield_global"
                    cand_rf["uvh_rf_pose_vec"] = (
                        np.asarray(rf_match.get("best_pose_vec", None), dtype=np.float32).reshape(-1).tolist()
                        if rf_match.get("best_pose_vec", None) is not None
                        else None
                    )
                    cand_rf["uvh_rf_delta_pose_vec"] = (
                        np.asarray(rf_match.get("delta_pose_vec", None), dtype=np.float32).reshape(-1).tolist()
                        if rf_match.get("delta_pose_vec", None) is not None
                        else None
                    )
                    cand_rf["uvh_rf_T_cur_ref"] = (
                        np.asarray(rf_match.get("T_cur_ref", None), dtype=np.float32).reshape(-1).tolist()
                        if rf_match.get("T_cur_ref", None) is not None
                        else None
                    )
                    cand_rf["uvh_rf_depth_loss"] = float(rf_match.get("depth_loss", float("nan")))
                    cand_rf["uvh_rf_normal_loss"] = float(rf_match.get("normal_loss", float("nan")))
                    cand_rf["uvh_rf_mask_iou"] = float(rf_match.get("mask_iou", float("nan")))
                    cand_rf["uvh_rf_valid_ratio"] = float(rf_match.get("valid_ratio", float("nan")))
                    rf_sim = float(max(0.0, min(1.0, rf_match.get("sim", 0.0))))
                    coarse_sim = float(cand_rf.get("uvh_render_sim", 0.0))
                    rf_mode = str(os.environ.get("MIS_UVH_RENDERFIELD_SCORE_MODE", "blend")).strip().lower()
                    if rf_mode == "blend":
                        fused_rf = float((1.0 - rf_blend_global) * coarse_sim + rf_blend_global * rf_sim)
                    else:
                        fused_rf = float(rf_sim)
                    cand_rf["uvh_rf_sim"] = float(rf_sim)
                    cand_rf["uvh_render_sim"] = float(max(0.0, min(1.0, fused_rf)))
                    cand_rf["semantic"] = float(cand_rf["uvh_render_sim"])
                    try:
                        q_rf = float(cand_rf.get("quick", 0.0))
                    except Exception:
                        q_rf = 0.0
                    try:
                        dx_rf = int(cand_rf.get("dx", 0))
                        dy_rf = int(cand_rf.get("dy", 0))
                    except Exception:
                        dx_rf, dy_rf = 0, 0
                    dsc_rf = _uvh_decision_score(
                        sim_total=float(cand_rf.get("uvh_render_sim", 0.0)),
                        quick=float(q_rf),
                        bbox=tuple(cand_rf.get("bbox", prev_bbox)),
                        dx=int(dx_rf),
                        dy=int(dy_rf),
                        rad_now=int(rad_now),
                        view_id=int(cand_rf.get("uvh_view_id", -1)),
                        view_dir=cand_rf.get("uvh_view_dir", None),
                        rigid_score=float("nan"),
                        local_score=float("nan"),
                        coverage_ratio=float(cand_rf.get("uvh_coverage_ratio", float("nan"))),
                        area_ratio=float(cand_rf.get("uvh_area_ratio", float("nan"))),
                        area_pen=float(cand_rf.get("uvh_area_pen", float("nan"))),
                        scale_pen=float(cand_rf.get("uvh_scale_pen", float("nan"))),
                        bbox_pen=float(cand_rf.get("uvh_bbox_scale_pen", float("nan"))),
                    )
                    cand_rf["score"] = float(dsc_rf.get("score", 0.0))
                    cand_rf["uvh_decision_score"] = float(cand_rf["score"])
                    cand_rf["uvh_coverage_gap"] = float(dsc_rf.get("coverage_gap", float("nan")))
                    cand_rf["uvh_decision_motion_pen"] = float(dsc_rf.get("motion_pen", float("nan")))
                    cand_rf["uvh_decision_coverage"] = float(dsc_rf.get("coverage", float("nan")))
                    cand_rf["uvh_decision_coverage_gap"] = float(dsc_rf.get("coverage_gap", float("nan")))
                    cand_rf["uvh_decision_temporal_iou"] = float(dsc_rf.get("temporal_iou", float("nan")))
                    cand_rf["uvh_decision_temporal_view"] = float(dsc_rf.get("temporal_view", float("nan")))
                    rf_variants.append(cand_rf)
                    if float(cand_rf.get("score", -1e9)) > float(best_score):
                        best_score = float(cand_rf["score"])
                        best = cand_rf
                if rf_variants:
                    cand_list.extend(rf_variants)
            except Exception as e:
                _log_once(
                    "uvh_rf_global_fail",
                    f"[UVH-RF][Skip] global match 失败: {e}",
                    cfg=cfg,
                    level=1,
                )

        rigid_enable = bool(_env_bool("MIS_UVH_RIGID_ENABLE", True))
        try:
            rigid_ref_pts_used = np.asarray(
                rigid_ref_uvh_pts if isinstance(rigid_ref_uvh_pts, np.ndarray) else prev_uvh_pts,
                dtype=np.float32,
            ).reshape(-1, 3)
        except Exception:
            rigid_ref_pts_used = np.zeros((0, 3), dtype=np.float32)
        if rigid_enable and int(rigid_ref_pts_used.shape[0]) >= 80 and (getattr(core, "o3d", None) is not None):
            try:
                topk_rigid = int(max(1, _env_int("MIS_UVH_RIGID_TOPK", 3)))
                min_coarse_rigid = float(_env_float("MIS_UVH_RIGID_MIN_COARSE_SIM", 0.40))
                rigid_blend = float(max(0.0, min(1.0, _env_float("MIS_UVH_RIGID_BLEND_W", 0.35))))
                local_blend = float(max(0.0, min(1.0, _env_float("MIS_UVH_LOCAL_BLEND_W", 0.25))))
                rf_topk = int(max(1, _env_int("MIS_UVH_RENDERFIELD_TOPK", topk_rigid)))
                rf_enable = bool(_env_bool("MIS_UVH_RENDERFIELD_RIGID_INIT_ENABLE", False))
                compare_all_stages = bool(_env_bool("MIS_UVH_COMPARE_ALL_STAGES", True))
                n_rigid_called = 0
                n_rigid_ok = 0
                rigid_variants: list[dict[str, Any]] = []
                cand_sorted_rigid = sorted(cand_list, key=lambda d: float(d.get("score", -1e9)), reverse=True)
                for rigid_rank, cc in enumerate(cand_sorted_rigid[: int(topk_rigid)]):
                    try:
                        coarse_sim = float(cc.get("uvh_render_sim", 0.0))
                    except Exception:
                        coarse_sim = 0.0
                    if float(coarse_sim) < float(min_coarse_rigid):
                        continue
                    try:
                        pts_cur_rigid = _uvh_points_from_th_mask_roi(
                            th_gray_truncated=np.asarray(cc.get("th_gray", None), dtype=np.float32),
                            mask_u8=np.asarray(cc.get("mask_u8", None), dtype=np.uint8),
                            bbox=tuple(cc.get("bbox", prev_bbox)),
                            z_scale=float(cfg.mesh_z_scale),
                            max_points=int(max(1000, _env_int("MIS_UVH_RIGID_MAX_POINTS", 60000))),
                            rng=np.random.default_rng(0),
                            fill_mask=True,
                            min_height=float(_env_float("MIS_UVH_RIGID_MIN_HEIGHT", 1.0)),
                            deterministic=True,
                        )
                    except Exception:
                        pts_cur_rigid = np.zeros((0, 3), dtype=np.float32)
                    if int(np.asarray(pts_cur_rigid).shape[0]) < 80:
                        continue
                    n_rigid_called += 1
                    res_rigid = _estimate_uvh_rigid_pose_from_points(
                        pts_ref=rigid_ref_pts_used,
                        pts_cur=pts_cur_rigid,
                        cfg=cfg,
                    )
                    if not bool(res_rigid.get("ok", False)):
                        continue
                    n_rigid_ok += 1
                    cc["uvh_rigid_used"] = 1
                    cc["uvh_rigid_method"] = str(res_rigid.get("method", ""))
                    cc["uvh_rigid_score"] = float(res_rigid.get("score", float("nan")))
                    cc["uvh_rigid_fitness"] = float(res_rigid.get("fitness", float("nan")))
                    cc["uvh_rigid_inlier_ratio"] = float(res_rigid.get("inlier_ratio", float("nan")))
                    cc["uvh_rigid_rmse"] = float(res_rigid.get("rmse", float("nan")))
                    cc["uvh_rigid_overlap"] = float(res_rigid.get("overlap", float("nan")))
                    cc["uvh_rigid_normal_consistency"] = float(res_rigid.get("normal_consistency", float("nan")))
                    cc["uvh_rigid_corr_count"] = int(res_rigid.get("corr_count", 0))
                    cc["uvh_rigid_lowcorr_pen"] = float(res_rigid.get("lowcorr_pen", float("nan")))
                    cc["uvh_rigid_median_dist"] = float(res_rigid.get("median_dist", float("nan")))
                    cc["uvh_rigid_mean_dist"] = float(res_rigid.get("mean_dist", float("nan")))
                    cc["uvh_rigid_voxel"] = float(res_rigid.get("voxel", float("nan")))
                    try:
                        T_ref_cur = np.asarray(res_rigid.get("T_ref_cur", np.eye(4)), dtype=np.float64).reshape(4, 4)
                        T_cur_ref = np.linalg.inv(T_ref_cur)
                        pts_ref_in_cur = _transform_uvh_points(
                            pts_xyz=rigid_ref_pts_used,
                            T_dst_src=T_cur_ref,
                        )
                        refined_bbox = _bbox_from_transformed_uvh_points(
                            pts_xyz=pts_ref_in_cur,
                            frame_wh=(W, H),
                            fallback_bbox=tuple(cc.get("bbox", prev_bbox)),
                        )
                    except Exception:
                        refined_bbox = None
                    if refined_bbox is not None:
                        cc["uvh_rigid_bbox_x"] = int(refined_bbox[0])
                        cc["uvh_rigid_bbox_y"] = int(refined_bbox[1])
                        cc["uvh_rigid_bbox_w"] = int(refined_bbox[2])
                        cc["uvh_rigid_bbox_h"] = int(refined_bbox[3])
                    rigid_score = float(res_rigid.get("score", float("nan")))
                    if np.isfinite(rigid_score):
                        fused_sim = float((1.0 - rigid_blend) * float(coarse_sim) + rigid_blend * float(rigid_score))
                        cand_rigid = dict(cc)
                        cand_rigid["uvh_render_sim"] = float(max(0.0, min(1.0, fused_sim)))
                        cand_rigid["semantic"] = float(cand_rigid["uvh_render_sim"])
                        cand_rigid["uvh_decision_stage"] = "rigid"
                        cand_rigid["uvh_projected_ref_pts"] = np.asarray(pts_ref_in_cur, dtype=np.float32)
                        rigid_cov = float("nan")
                        if refined_bbox is not None:
                            rebuilt = _rebuild_uvh_candidate_from_bbox(
                                frame_bgr=frame_bgr,
                                bbox=tuple(refined_bbox),
                                cfg=cfg,
                                ref_mask_small=np.asarray(prev_mask_small, dtype=np.uint8),
                                search_ctx=search_ctx,
                                cache=rebuild_cache,
                                **_frame_th_rebuild_kw(),
                            )
                            if isinstance(rebuilt, dict):
                                cand_rigid["bbox"] = tuple(rebuilt["bbox"])
                                cand_rigid["th_gray"] = rebuilt["th_gray"]
                                cand_rigid["mask_u8"] = rebuilt["mask_u8"]
                                cand_rigid["th_uint8"] = float(rebuilt["th_uint8"])
                                cand_rigid["contour_rgb"] = rebuilt["contour_rgb"]
                                cand_rigid["uvh_render_pts"] = rebuilt["uvh_render_pts"]
                                try:
                                    rx, ry, rw, rh = [int(v) for v in cand_rigid["bbox"]]
                                    cand_rigid["uvh_rigid_bbox_used"] = 1
                                    cand_rigid["dx"] = int((rx + rw // 2) - cx0)
                                    cand_rigid["dy"] = int((ry + rh // 2) - cy0)
                                    cand_rigid["uvh_bbox_scale"] = float(math.sqrt((float(rw) * float(rh)) / float(max(1.0, float(pw) * float(ph) + 1e-6))))
                                except Exception:
                                    pass
                        try:
                            rigid_cov = _bbox_point_coverage_ratio(
                                pts_xyz=pts_ref_in_cur,
                                bbox=tuple(cand_rigid.get("bbox", prev_bbox)),
                            )
                        except Exception:
                            rigid_cov = float("nan")
                        cand_rigid["uvh_coverage_ratio"] = float(rigid_cov) if np.isfinite(rigid_cov) else float("nan")
                        try:
                            q0 = float(cand_rigid.get("quick", 0.0))
                        except Exception:
                            q0 = 0.0
                        try:
                            dx0 = int(cand_rigid.get("dx", 0))
                            dy0 = int(cand_rigid.get("dy", 0))
                        except Exception:
                            dx0, dy0 = 0, 0
                        dsc_rigid = _uvh_decision_score(
                            sim_total=float(cand_rigid.get("uvh_render_sim", 0.0)),
                            quick=float(q0),
                            bbox=tuple(cand_rigid.get("bbox", prev_bbox)),
                            dx=int(dx0),
                            dy=int(dy0),
                            rad_now=int(rad_now),
                            view_id=int(cand_rigid.get("uvh_view_id", -1)),
                            view_dir=cand_rigid.get("uvh_view_dir", None),
                            rigid_score=float(cand_rigid.get("uvh_rigid_score", float("nan"))),
                            local_score=float("nan"),
                            coverage_ratio=float(cand_rigid.get("uvh_coverage_ratio", float("nan"))),
                            area_ratio=float(cand_rigid.get("uvh_area_ratio", float("nan"))),
                            area_pen=float(cand_rigid.get("uvh_area_pen", float("nan"))),
                            scale_pen=float(cand_rigid.get("uvh_scale_pen", float("nan"))),
                            bbox_pen=float(cand_rigid.get("uvh_bbox_scale_pen", float("nan"))),
                        )
                        cand_rigid["score"] = float(dsc_rigid.get("score", 0.0))
                        cand_rigid["uvh_decision_score"] = float(cand_rigid["score"])
                        cand_rigid["uvh_coverage_gap"] = float(dsc_rigid.get("coverage_gap", float("nan")))
                        cand_rigid["uvh_decision_motion_pen"] = float(dsc_rigid.get("motion_pen", float("nan")))
                        cand_rigid["uvh_decision_coverage"] = float(dsc_rigid.get("coverage", float("nan")))
                        cand_rigid["uvh_decision_coverage_gap"] = float(dsc_rigid.get("coverage_gap", float("nan")))
                        cand_rigid["uvh_decision_temporal_iou"] = float(dsc_rigid.get("temporal_iou", float("nan")))
                        cand_rigid["uvh_decision_temporal_view"] = float(dsc_rigid.get("temporal_view", float("nan")))
                        rigid_variants.append(cand_rigid)
                        rf_local_init = cand_rigid
                        if rf_enable and int(rigid_rank) < int(rf_topk):
                            cand_rigid["uvh_rf_attempted"] = 1
                            rf_result = _refine_pose_bbox_with_renderfield(
                                ref_points=rigid_ref_pts_used,
                                init_T_cur_ref=np.asarray(T_cur_ref, dtype=np.float32),
                                cand=cand_rigid,
                                frame_wh=(W, H),
                                cfg=cfg,
                            )
                            cand_rigid["uvh_rf_steps"] = int(rf_result.get("steps", 0) or 0)
                            cand_rigid["uvh_rf_lr"] = float(rf_result.get("lr", float("nan")))
                            cand_rigid["uvh_rf_best_loss"] = float(rf_result.get("best_loss", float("nan")))
                            cand_rigid["uvh_rf_error"] = str(rf_result.get("error", ""))
                            if bool(rf_result.get("ok", False)):
                                cand_rf = dict(cand_rigid)
                                cand_rf["uvh_rf_used"] = 1
                                cand_rf["uvh_decision_stage"] = "renderfield"
                                cand_rf["uvh_projected_ref_pts"] = np.asarray(
                                    rf_result.get("projected_ref_pts", pts_ref_in_cur),
                                    dtype=np.float32,
                                )
                                bbox_rf = rf_result.get("bbox", None)
                                if isinstance(bbox_rf, tuple) and len(bbox_rf) == 4:
                                    cand_rf["uvh_rf_bbox_used"] = 1
                                    cand_rf["uvh_rf_bbox_x"] = int(bbox_rf[0])
                                    cand_rf["uvh_rf_bbox_y"] = int(bbox_rf[1])
                                    cand_rf["uvh_rf_bbox_w"] = int(bbox_rf[2])
                                    cand_rf["uvh_rf_bbox_h"] = int(bbox_rf[3])
                                    rebuilt_rf = _rebuild_uvh_candidate_from_bbox(
                                        frame_bgr=frame_bgr,
                                        bbox=tuple(bbox_rf),
                                        cfg=cfg,
                                        ref_mask_small=np.asarray(prev_mask_small, dtype=np.uint8),
                                        search_ctx=search_ctx,
                                        cache=rebuild_cache,
                                        **_frame_th_rebuild_kw(),
                                    )
                                    if isinstance(rebuilt_rf, dict):
                                        cand_rf["bbox"] = tuple(rebuilt_rf["bbox"])
                                        cand_rf["th_gray"] = rebuilt_rf["th_gray"]
                                        cand_rf["mask_u8"] = rebuilt_rf["mask_u8"]
                                        cand_rf["th_uint8"] = float(rebuilt_rf["th_uint8"])
                                        cand_rf["contour_rgb"] = rebuilt_rf["contour_rgb"]
                                        cand_rf["uvh_render_pts"] = rebuilt_rf["uvh_render_pts"]
                                try:
                                    rfx, rfy, rfw, rfh = [int(v) for v in cand_rf.get("bbox", prev_bbox)]
                                    cand_rf["dx"] = int((rfx + rfw // 2) - cx0)
                                    cand_rf["dy"] = int((rfy + rfh // 2) - cy0)
                                    cand_rf["uvh_bbox_scale"] = float(
                                        math.sqrt((float(rfw) * float(rfh)) / float(max(1.0, float(pw) * float(ph) + 1e-6)))
                                    )
                                except Exception:
                                    pass
                                try:
                                    rf_cov = _bbox_point_coverage_ratio(
                                        pts_xyz=np.asarray(cand_rf.get("uvh_projected_ref_pts", None), dtype=np.float32),
                                        bbox=tuple(cand_rf.get("bbox", prev_bbox)),
                                    )
                                except Exception:
                                    rf_cov = float("nan")
                                cand_rf["uvh_coverage_ratio"] = float(rf_cov) if np.isfinite(rf_cov) else float("nan")
                                try:
                                    q_rf = float(cand_rf.get("quick", 0.0))
                                except Exception:
                                    q_rf = 0.0
                                try:
                                    dx_rf = int(cand_rf.get("dx", 0))
                                    dy_rf = int(cand_rf.get("dy", 0))
                                except Exception:
                                    dx_rf, dy_rf = 0, 0
                                dsc_rf = _uvh_decision_score(
                                    sim_total=float(cand_rf.get("uvh_render_sim", 0.0)),
                                    quick=float(q_rf),
                                    bbox=tuple(cand_rf.get("bbox", prev_bbox)),
                                    dx=int(dx_rf),
                                    dy=int(dy_rf),
                                    rad_now=int(rad_now),
                                    view_id=int(cand_rf.get("uvh_view_id", -1)),
                                    view_dir=cand_rf.get("uvh_view_dir", None),
                                    rigid_score=float(cand_rf.get("uvh_rigid_score", float("nan"))),
                                    local_score=float("nan"),
                                    coverage_ratio=float(cand_rf.get("uvh_coverage_ratio", float("nan"))),
                                    area_ratio=float(cand_rf.get("uvh_area_ratio", float("nan"))),
                                    area_pen=float(cand_rf.get("uvh_area_pen", float("nan"))),
                                    scale_pen=float(cand_rf.get("uvh_scale_pen", float("nan"))),
                                    bbox_pen=float(cand_rf.get("uvh_bbox_scale_pen", float("nan"))),
                                )
                                cand_rf["score"] = float(dsc_rf.get("score", 0.0))
                                cand_rf["uvh_decision_score"] = float(cand_rf["score"])
                                cand_rf["uvh_coverage_gap"] = float(dsc_rf.get("coverage_gap", float("nan")))
                                cand_rf["uvh_decision_motion_pen"] = float(dsc_rf.get("motion_pen", float("nan")))
                                cand_rf["uvh_decision_coverage"] = float(dsc_rf.get("coverage", float("nan")))
                                cand_rf["uvh_decision_coverage_gap"] = float(dsc_rf.get("coverage_gap", float("nan")))
                                cand_rf["uvh_decision_temporal_iou"] = float(dsc_rf.get("temporal_iou", float("nan")))
                                cand_rf["uvh_decision_temporal_view"] = float(dsc_rf.get("temporal_view", float("nan")))
                                rigid_variants.append(cand_rf)
                                rf_local_init = cand_rf

                        local_refined = _refine_bbox_by_local_mask_search(
                            frame_bgr=frame_bgr,
                            init_bbox=tuple(rf_local_init.get("bbox", cand_rigid.get("bbox", cc.get("bbox", prev_bbox)))),
                            ref_mask_small=np.asarray(prev_mask_small, dtype=np.uint8),
                            ref_contour_rgb=(np.asarray(ref_contour_rgb, dtype=np.uint8) if isinstance(ref_contour_rgb, np.ndarray) else None),
                            cfg=cfg,
                            search_ctx=search_ctx,
                            cache=rebuild_cache,
                            **_frame_th_rebuild_kw(),
                        )
                        if isinstance(local_refined, dict):
                            cand_local = dict(rf_local_init)
                            try:
                                lbx, lby, lbw, lbh = [int(v) for v in local_refined["bbox"]]
                                cand_local["uvh_local_bbox_x"] = int(lbx)
                                cand_local["uvh_local_bbox_y"] = int(lby)
                                cand_local["uvh_local_bbox_w"] = int(lbw)
                                cand_local["uvh_local_bbox_h"] = int(lbh)
                            except Exception:
                                pass
                            cand_local["bbox"] = tuple(local_refined["bbox"])
                            cand_local["th_gray"] = local_refined["th_gray"]
                            cand_local["mask_u8"] = local_refined["mask_u8"]
                            cand_local["th_uint8"] = float(local_refined["th_uint8"])
                            cand_local["contour_rgb"] = local_refined["contour_rgb"]
                            cand_local["uvh_render_pts"] = local_refined["uvh_render_pts"]
                            cand_local["uvh_local_bbox_used"] = 1
                            cand_local["uvh_local_bbox_score"] = float(local_refined.get("uvh_local_bbox_score", float("nan")))
                            cand_local["uvh_local_quick"] = float(local_refined.get("uvh_local_quick", float("nan")))
                            cand_local["uvh_local_sem"] = float(local_refined.get("uvh_local_sem", float("nan")))
                            cand_local["uvh_decision_stage"] = "local"
                            cand_local["uvh_projected_ref_pts"] = np.asarray(pts_ref_in_cur, dtype=np.float32)
                            try:
                                local_cov = _bbox_point_coverage_ratio(
                                    pts_xyz=pts_ref_in_cur,
                                    bbox=tuple(cand_local.get("bbox", prev_bbox)),
                                )
                            except Exception:
                                local_cov = float("nan")
                            cand_local["uvh_coverage_ratio"] = float(local_cov) if np.isfinite(local_cov) else float("nan")
                            local_score = float(local_refined.get("uvh_local_bbox_score", float("nan")))
                            if np.isfinite(local_score):
                                cand_local["uvh_render_sim"] = float(max(0.0, min(1.0, (1.0 - local_blend) * float(cand_rigid["uvh_render_sim"]) + local_blend * float(local_score))))
                                cand_local["semantic"] = float(cand_local["uvh_render_sim"])
                            try:
                                lx, ly, lw, lh = [int(v) for v in cand_local["bbox"]]
                                cand_local["dx"] = int((lx + lw // 2) - cx0)
                                cand_local["dy"] = int((ly + lh // 2) - cy0)
                                cand_local["uvh_bbox_scale"] = float(math.sqrt((float(lw) * float(lh)) / float(max(1.0, float(pw) * float(ph) + 1e-6))))
                            except Exception:
                                pass
                            try:
                                q1 = float(cand_local.get("quick", 0.0))
                            except Exception:
                                q1 = 0.0
                            try:
                                dx1 = int(cand_local.get("dx", 0))
                                dy1 = int(cand_local.get("dy", 0))
                            except Exception:
                                dx1, dy1 = 0, 0
                            dsc_local = _uvh_decision_score(
                                sim_total=float(cand_local.get("uvh_render_sim", 0.0)),
                                quick=float(q1),
                                bbox=tuple(cand_local.get("bbox", prev_bbox)),
                                dx=int(dx1),
                                dy=int(dy1),
                                rad_now=int(rad_now),
                                view_id=int(cand_local.get("uvh_view_id", -1)),
                                view_dir=cand_local.get("uvh_view_dir", None),
                                rigid_score=float(cand_local.get("uvh_rigid_score", float("nan"))),
                                local_score=float(cand_local.get("uvh_local_bbox_score", float("nan"))),
                                coverage_ratio=float(cand_local.get("uvh_coverage_ratio", float("nan"))),
                                area_ratio=float(cand_local.get("uvh_area_ratio", float("nan"))),
                                area_pen=float(cand_local.get("uvh_area_pen", float("nan"))),
                                scale_pen=float(cand_local.get("uvh_scale_pen", float("nan"))),
                                bbox_pen=float(cand_local.get("uvh_bbox_scale_pen", float("nan"))),
                            )
                            cand_local["score"] = float(dsc_local.get("score", 0.0))
                            cand_local["uvh_decision_score"] = float(cand_local["score"])
                            cand_local["uvh_coverage_gap"] = float(dsc_local.get("coverage_gap", float("nan")))
                            cand_local["uvh_decision_motion_pen"] = float(dsc_local.get("motion_pen", float("nan")))
                            cand_local["uvh_decision_coverage"] = float(dsc_local.get("coverage", float("nan")))
                            cand_local["uvh_decision_coverage_gap"] = float(dsc_local.get("coverage_gap", float("nan")))
                            cand_local["uvh_decision_temporal_iou"] = float(dsc_local.get("temporal_iou", float("nan")))
                            cand_local["uvh_decision_temporal_view"] = float(dsc_local.get("temporal_view", float("nan")))
                            rigid_variants.append(cand_local)
                if compare_all_stages and rigid_variants:
                    cand_list.extend(rigid_variants)
                best = max(cand_list, key=lambda d: float(d.get("score", -1e9)))
                best["uvh_rigid_called"] = int(n_rigid_called)
                best["uvh_rigid_ok"] = int(n_rigid_ok)
            except Exception:
                pass

        # ===== 可微分渲染 refine（Kaolin DIB-R + 反向优化视点）=====
        # 直接在代码里把默认 diff-render 设为启用（仍可用环境变量覆盖）
        diff_enable = bool(_env_bool("MIS_UVH_DIFFRENDER_ENABLE", False))
        if diff_enable and (diff_ref_verts is not None) and (diff_ref_faces is not None):
            try:
                topk_refine = int(max(1, _env_int("MIS_UVH_DIFF_TOPK", 2)))
                min_coarse = float(_env_float("MIS_UVH_DIFF_MIN_COARSE_SIM", 0.35))
                cand_sorted = sorted(cand_list, key=lambda d: float(d.get("score", -1e9)), reverse=True)
                n_mesh_small = 0
                n_called = 0
                n_ok = 0
                best_diff_sim = float("nan")
                best_diff_backend = ""
                for cc in cand_sorted[: int(topk_refine)]:
                    try:
                        if float(cc.get("uvh_render_sim", 0.0)) < float(min_coarse):
                            continue
                    except Exception:
                        continue

                    # build candidate mesh (heightfield) for differentiable rendering
                    try:
                        thm = np.asarray(cc.get("th_gray", None), dtype=np.float32)
                        mkm = np.asarray(cc.get("mask_u8", None), dtype=np.uint8)
                        if thm.ndim != 2 or mkm.ndim != 2 or thm.size <= 0 or mkm.size <= 0:
                            continue
                        if mkm.shape != thm.shape:
                            mkm = cv2.resize(mkm, (int(thm.shape[1]), int(thm.shape[0])), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                        # diff-render 需要有足够的面片；th_68 的“边缘高度场”在很多帧会非常稀疏，
                        # 这里用“填充 mask”把 ROI 内区域变成连通面片（高度仍用 th_gray，0 高度也允许存在面片）。
                        mkm_f = _mask_to_filled_u8(mkm)
                        if mkm_f.shape != thm.shape:
                            mkm_f = cv2.resize(mkm_f, (int(thm.shape[1]), int(thm.shape[0])), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                        thm2 = np.where(mkm_f > 0, np.maximum(thm, 1.0), 0.0).astype(np.float32)
                        v1, f1, _ = _make_heightfield_mesh_arrays(
                            th_gray_truncated=thm2,
                            bbox=tuple(cc.get("bbox", prev_bbox)),
                            z_scale=float(cfg.mesh_z_scale),
                        )
                    except Exception:
                        continue
                    if int(np.asarray(v1).shape[0]) < 50 or int(np.asarray(f1).shape[0]) < 50:
                        n_mesh_small += 1
                        continue

                    init_dir = None
                    try:
                        d = cc.get("uvh_view_dir", None)
                        if isinstance(d, list) and len(d) >= 3:
                            init_dir = np.asarray(d, dtype=np.float32).reshape(3)
                    except Exception:
                        init_dir = None

                    _log(
                        (
                            f"[UVH-Diff] start bbox={list(map(int, cc.get('bbox', prev_bbox)))} "
                            f"coarse_sim={float(cc.get('uvh_render_sim', float('nan'))):.3f} view_id={int(cc.get('uvh_view_id', -1))}"
                        ),
                        cfg=cfg,
                        level=2,
                    )
                    try:
                        sim_before = float(cc.get("uvh_render_sim", float("nan")))
                    except Exception:
                        sim_before = float("nan")
                    try:
                        score_before = float(cc.get("score", float("nan")))
                    except Exception:
                        score_before = float("nan")
                    try:
                        n_called += 1
                        res = _uvh_diffrefine_with_kaolin(
                            ref_verts_np=np.asarray(diff_ref_verts, dtype=np.float32),
                            ref_faces_np=np.asarray(diff_ref_faces, dtype=np.int32),
                            cur_verts_np=np.asarray(v1, dtype=np.float32),
                            cur_faces_np=np.asarray(f1, dtype=np.int32),
                            init_view_dir=init_dir,
                            init_view_roll_deg=float(cc.get("uvh_view_roll", 0.0)),
                            init_view_dist=float(cc.get("uvh_view_dist", float("nan"))),
                            init_scale=float(cc.get("uvh_align_scale", 1.0)),
                            init_rot_deg=float(cc.get("uvh_align_rot_deg", 0.0)),
                            init_tx=float(cc.get("uvh_align_tx", 0.0)),
                            init_ty=float(cc.get("uvh_align_ty", 0.0)),
                            bbox_scale=float(cc.get("uvh_bbox_scale", 1.0)),
                            cfg=cfg,
                        )
                    except Exception as e:
                        _log_once(
                            "uvh_diff_exception",
                            f"[UVH-Diff][Error] 可微渲染/优化异常：{e}",
                            cfg=cfg,
                            level=1,
                        )
                        continue
                    if bool(res.get("ok", False)):
                        n_ok += 1
                        _log(
                            (
                                f"[UVH-Diff] done sim={float(res.get('sim', float('nan'))):.3f} "
                                f"yaw={float(res.get('yaw_deg', float('nan'))):.1f} pitch={float(res.get('pitch_deg', float('nan'))):.1f} "
                                f"roll={float(res.get('roll_deg', float('nan'))):.1f} dist={float(res.get('dist', float('nan'))):.2f} "
                                f"backend={str(res.get('backend', ''))}"
                            ),
                            cfg=cfg,
                            level=2,
                        )
                        cc["uvh_diff_used"] = 1
                        cc["uvh_diff_backend"] = str(res.get("backend", ""))
                        cc["uvh_diff_iters"] = int(res.get("iters", 0))
                        cc["uvh_diff_lr"] = float(res.get("lr", float("nan")))
                        cc["uvh_diff_sim"] = float(res.get("sim", float("nan")))
                        cc["uvh_diff_iou"] = float(res.get("iou", float("nan")))
                        cc["uvh_diff_depthcorr"] = float(res.get("depth_corr", float("nan")))
                        cc["uvh_diff_sem"] = float(res.get("sem", float("nan")))
                        cc["uvh_diff_sem_used"] = int(res.get("sem_used", 0))
                        cc["uvh_diff_yaw_deg"] = float(res.get("yaw_deg", float("nan")))
                        cc["uvh_diff_pitch_deg"] = float(res.get("pitch_deg", float("nan")))
                        cc["uvh_diff_roll_deg"] = float(res.get("roll_deg", float("nan")))
                        cc["uvh_diff_dist"] = float(res.get("dist", float("nan")))
                        cc["uvh_diff_scale"] = float(res.get("scale", float("nan")))
                        cc["uvh_diff_rot_deg"] = float(res.get("rot_deg", float("nan")))
                        cc["uvh_diff_tx"] = float(res.get("tx", float("nan")))
                        cc["uvh_diff_ty"] = float(res.get("ty", float("nan")))
                        cc["uvh_diff_area_ratio"] = float(res.get("area_ratio", float("nan")))

                        s2 = float(res.get("sim", float("nan")))
                        if np.isfinite(s2):
                            s2 = float(max(0.0, min(1.0, s2)))
                            # 只在“diff 优化确实提升相似度”时，用其替换 coarse；否则保留 coarse 以免把候选分数拉低
                            accept_only_improve = bool(_env_bool("MIS_UVH_DIFF_ACCEPT_ONLY_IF_IMPROVE", True))
                            accept_eps = float(_env_float("MIS_UVH_DIFF_ACCEPT_EPS", 1e-4))
                            accept = True
                            if accept_only_improve and np.isfinite(sim_before):
                                accept = bool(float(s2) > float(sim_before) + float(accept_eps))
                            if accept:
                                cc["uvh_render_sim"] = float(s2)
                                cc["semantic"] = float(s2)
                            else:
                                # 恢复 coarse 分数
                                if np.isfinite(score_before):
                                    cc["score"] = float(score_before)
                            if (not np.isfinite(best_diff_sim)) or float(s2) > float(best_diff_sim):
                                best_diff_sim = float(s2)
                                best_diff_backend = str(res.get("backend", ""))

                        # recompute score with (maybe) refined sim
                        try:
                            q0 = float(cc.get("quick", 0.0))
                        except Exception:
                            q0 = 0.0
                        try:
                            dx0 = int(cc.get("dx", 0))
                            dy0 = int(cc.get("dy", 0))
                        except Exception:
                            dx0, dy0 = 0, 0
                        dsc_diff = _uvh_decision_score(
                            sim_total=float(cc.get("uvh_render_sim", 0.0)),
                            quick=float(q0),
                            bbox=tuple(cc.get("bbox", prev_bbox)),
                            dx=int(dx0),
                            dy=int(dy0),
                            rad_now=int(rad_now),
                            view_id=int(cc.get("uvh_view_id", -1)),
                            view_dir=cc.get("uvh_view_dir", None),
                            rigid_score=float(cc.get("uvh_rigid_score", float("nan"))),
                            local_score=float(cc.get("uvh_local_bbox_score", float("nan"))),
                            coverage_ratio=float(cc.get("uvh_coverage_ratio", float("nan"))),
                            area_ratio=float(cc.get("uvh_area_ratio", float("nan"))),
                            area_pen=float(cc.get("uvh_area_pen", float("nan"))),
                            scale_pen=float(cc.get("uvh_scale_pen", float("nan"))),
                            bbox_pen=float(cc.get("uvh_bbox_scale_pen", float("nan"))),
                        )
                        cc["score"] = float(dsc_diff.get("score", 0.0))
                        cc["uvh_decision_score"] = float(cc["score"])
                        cc["uvh_coverage_gap"] = float(dsc_diff.get("coverage_gap", float("nan")))
                        cc["uvh_decision_motion_pen"] = float(dsc_diff.get("motion_pen", float("nan")))
                        cc["uvh_decision_coverage"] = float(dsc_diff.get("coverage", float("nan")))
                        cc["uvh_decision_coverage_gap"] = float(dsc_diff.get("coverage_gap", float("nan")))
                        cc["uvh_decision_temporal_iou"] = float(dsc_diff.get("temporal_iou", float("nan")))
                        cc["uvh_decision_temporal_view"] = float(dsc_diff.get("temporal_view", float("nan")))
                    else:
                        try:
                            rr = str(res.get("reason", "") or "").strip()
                        except Exception:
                            rr = ""
                        if rr:
                            _log_once(
                                f"uvh_diff_skip_{rr}",
                                f"[UVH-Diff][Skip] reason={rr}（若已 activate kaolin 仍出现，可检查是否能 `import kaolin`/nvdiffrast）",
                                cfg=cfg,
                                level=1,
                            )

                best = max(cand_list, key=lambda d: float(d.get("score", -1e9)))
                # 把“本帧是否尝试/成功 diff refine”的计数写回 best，便于控制台/CSV确认
                try:
                    best["uvh_diff_called"] = int(n_called)
                    best["uvh_diff_ok"] = int(n_ok)
                except Exception:
                    pass
                if int(n_called) <= 0 and int(n_mesh_small) > 0:
                    _log_once(
                        "uvh_diff_all_mesh_small",
                        f"[UVH-Diff][Skip] 所有候选 mesh 过小（n_mesh_small={int(n_mesh_small)}），diff-render 无法执行",
                        cfg=cfg,
                        level=1,
                    )
                if int(n_called) > 0:
                    _log_once(
                        "uvh_diff_summary",
                        f"[UVH-Diff] refine called={int(n_called)} ok={int(n_ok)} best_sim={float(best_diff_sim) if np.isfinite(best_diff_sim) else float('nan'):.3f} backend={best_diff_backend}",
                        cfg=cfg,
                        level=1,
                    )
            except Exception:
                pass

        try:
            if isinstance(best, dict):
                best = _postprocess_bbox_for_full_coverage(
                    frame_bgr=frame_bgr,
                    cand=best,
                    ref_mask_small=np.asarray(prev_mask_small, dtype=np.uint8),
                    ref_contour_rgb=(np.asarray(ref_contour_rgb, dtype=np.uint8) if isinstance(ref_contour_rgb, np.ndarray) else None),
                    projected_ref_pts=best.get("uvh_projected_ref_pts", None),
                    cfg=cfg,
                    search_ctx=search_ctx,
                    cache=rebuild_cache,
                    **_frame_th_rebuild_kw(),
                )
        except Exception:
            pass
        return best

    two_stage = bool(_env_bool("MIS_UVH_RENDER_TWO_STAGE_ENABLE", True))
    best: dict[str, Any] | None = None
    if two_stage:
        coarse_frac = float(_env_float("MIS_UVH_RENDER_COARSE_RAD_FRAC", 0.72))
        coarse_step_mult = float(_env_float("MIS_UVH_RENDER_COARSE_STEP_MULT", 2.0))
        rad_c = int(max(step, min(int(rad), int(round(float(rad) * coarse_frac)))))
        step_c = int(max(step, int(round(float(step) * coarse_step_mult))))
        rad_c = int(max(step_c, rad_c))
        best_c = _eval_one(rad_c, step_use=step_c)
        best = best_c
        if best_c is not None:
            bb = best_c.get("bbox")
            if isinstance(bb, (tuple, list)) and len(bb) >= 4:
                bx, by, bw, bh = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
                cxf = bx + bw // 2
                cyf = by + bh // 2
                rad_f_cfg = int(_env_int("MIS_UVH_RENDER_FINE_RAD_PX", 0))
                if rad_f_cfg > 0:
                    rad_f = int(max(step, rad_f_cfg))
                else:
                    rad_f = int(max(step * 2, min(int(rad), int(rad) // 2 + int(step))))
                step_f_cfg = int(_env_int("MIS_UVH_RENDER_FINE_STEP_PX", 0))
                if step_f_cfg > 0:
                    step_f = int(max(1, step_f_cfg))
                else:
                    step_f = int(max(1, min(step, max(4, step // 2))))
                best_f = _eval_one(rad_f, step_use=step_f, cx_anchor=cxf, cy_anchor=cyf)
                if best_f is not None:
                    s_c = float(best_c.get("score", -1e9))
                    s_f = float(best_f.get("score", -1e9))
                    if s_f >= s_c:
                        best = best_f
        if best is None:
            best = _eval_one(int(rad))
    else:
        best = _eval_one(int(rad))

    if cfg.recover_enabled and (best is None or float(best.get("score", 0.0)) < float(cfg.recover_score)):
        rad2 = int(max(int(rad_full) * 2, int(rad_full) + int(step)))
        best2 = _eval_one(rad2)
        if best2 is not None and (best is None or float(best2.get("score", -1e9)) > float(best.get("score", -1e9))):
            best = best2

    if best is None:
        pbx, pby, pbw, pbh = [int(v) for v in prev_bbox]
        xa = int(max(0, min(W, pbx)))
        ya = int(max(0, min(H, pby)))
        xb = int(max(0, min(W, pbx + pbw)))
        yb = int(max(0, min(H, pby + pbh)))
        if (
            use_frame_th68
            and isinstance(th_frame_full, np.ndarray)
            and isinstance(mask_frame_full, np.ndarray)
            and int(th_frame_full.shape[0]) == int(H)
            and int(th_frame_full.shape[1]) == int(W)
            and (xb > xa)
            and (yb > ya)
        ):
            th_gray = np.asarray(th_frame_full[ya:yb, xa:xb], dtype=np.float32)
            mask_u8 = np.asarray(mask_frame_full[ya:yb, xa:xb], dtype=np.uint8)
            th_uint8 = float(th_u8_frame_full)
        else:
            roi = core.crop(frame_bgr, prev_bbox)
            th_gray, mask_u8, th_uint8 = core.compute_th_gray_truncated(roi, cfg.th_percent, cfg)
        cnt = _extract_contour_prefer_center(mask_u8)
        if cnt is None:
            cnt = core.extract_largest_contour(mask_u8)
        if cnt is not None:
            cnt2 = core.resize_contour(cnt, (prev_bbox[2], prev_bbox[3]), cfg.semantic_image_size)
            contour_rgb = core.contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)
        else:
            contour_rgb = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
        best = {
            "bbox": prev_bbox,
            "score": 0.0,
            "semantic": 0.0,
            "quick": 0.0,
            "dx": 0,
            "dy": 0,
            "th_gray": th_gray,
            "mask_u8": mask_u8,
            "contour_rgb": contour_rgb,
            "th_uint8": float(th_uint8),
            "uvh_icp_used": 0,
            "uvh_icp_fitness": float("nan"),
            "uvh_icp_rmse": float("nan"),
            "uvh_render_used": 0,
            "uvh_render_sim": float("nan"),
            "uvh_render_pts": None,
            "uvh_decision_stage": "fallback",
            "uvh_decision_score": 0.0,
            "uvh_decision_motion_pen": 0.0,
            "uvh_decision_temporal_iou": 0.0,
            "uvh_decision_temporal_view": 0.0,
            "uvh_render_prev_rgb": prev_render_rgb,
            "uvh_render_cur_rgb": None,
            "uvh_align_scale": float("nan"),
            "uvh_align_rot_deg": float("nan"),
            "uvh_align_tx": float("nan"),
            "uvh_align_ty": float("nan"),
            "uvh_mesh_sim": float("nan"),
            "uvh_mesh_iou": float("nan"),
            "uvh_mesh_depthcorr": float("nan"),
            "uvh_sem_sim": float("nan"),
            "uvh_sem_used": 0,
            "uvh_area_ratio": float("nan"),
            "uvh_area_pen": float("nan"),
            "uvh_scale_pen": float("nan"),
            "uvh_bbox_scale": float("nan"),
            "uvh_bbox_scale_pen": float("nan"),
            "uvh_view_id": -1,
            "uvh_view_score": float("nan"),
            "uvh_view_used_gate": 0,
            "uvh_view_n_used": 0,
            "uvh_view_dir": None,
            "uvh_view_dist": float("nan"),
            "uvh_view_roll": float("nan"),
            "uvh_rigid_used": 0,
            "uvh_rigid_method": "",
            "uvh_rigid_score": float("nan"),
            "uvh_rigid_fitness": float("nan"),
            "uvh_rigid_inlier_ratio": float("nan"),
            "uvh_rigid_rmse": float("nan"),
            "uvh_rigid_overlap": float("nan"),
            "uvh_rigid_normal_consistency": float("nan"),
            "uvh_rigid_median_dist": float("nan"),
            "uvh_rigid_mean_dist": float("nan"),
            "uvh_rigid_voxel": float("nan"),
            "uvh_rigid_bbox_used": 0,
            "uvh_rigid_bbox_x": int(prev_bbox[0]),
            "uvh_rigid_bbox_y": int(prev_bbox[1]),
            "uvh_rigid_bbox_w": int(prev_bbox[2]),
            "uvh_rigid_bbox_h": int(prev_bbox[3]),
            "uvh_rf_attempted": 0,
            "uvh_rf_used": 0,
            "uvh_rf_best_loss": float("nan"),
            "uvh_rf_steps": 0,
            "uvh_rf_lr": float("nan"),
            "uvh_rf_bbox_used": 0,
            "uvh_rf_bbox_x": int(prev_bbox[0]),
            "uvh_rf_bbox_y": int(prev_bbox[1]),
            "uvh_rf_bbox_w": int(prev_bbox[2]),
            "uvh_rf_bbox_h": int(prev_bbox[3]),
            "uvh_rf_error": "",
            "uvh_local_bbox_used": 0,
            "uvh_local_bbox_score": float("nan"),
            "uvh_local_quick": float("nan"),
            "uvh_local_sem": float("nan"),
            "uvh_local_bbox_x": int(prev_bbox[0]),
            "uvh_local_bbox_y": int(prev_bbox[1]),
            "uvh_local_bbox_w": int(prev_bbox[2]),
            "uvh_local_bbox_h": int(prev_bbox[3]),
        }
    return best


def find_best_bbox_in_next_frame_uvh_icp(
    *,
    frame_bgr: np.ndarray,
    prev_bbox: tuple[int, int, int, int],
    prev_mask_small: np.ndarray,
    prev_mesh: Any,
    cfg: core.TrackConfig,
) -> dict[str, Any]:
    """
    伪3D(uvh)跟踪：对候选 bbox 构建伪3D高度场 mesh，并用 ICP 与上一帧 mesh 配准。

    结构：候选生成 -> quick(IoU) 粗筛 -> topK 用 ICP 精排 -> (可选) 扩大范围恢复。
    返回字段与 core.find_best_bbox_in_next_frame 对齐，并额外包含：
      uvh_icp_used, uvh_icp_fitness, uvh_icp_rmse
    """
    H, W = frame_bgr.shape[:2]
    px, py, pw, ph = prev_bbox
    cx0 = px + pw // 2
    cy0 = py + ph // 2

    step = int(max(1, cfg.search_step))
    rad = int(max(step, cfg.search_radius))

    # 可选：把 ICP 对齐后的伪3D投影渲染成固定视角 contour_rgb，再用语义/回退相似度二次评分
    enable_render_sim = _env_bool("MIS_UVH_RENDER_SIM_ENABLE", True)
    render_max_points = _env_int("MIS_UVH_RENDER_MAX_POINTS", 5000)  # 优化：从10000降至5000提速
    render_dist_mult = _env_float("MIS_UVH_RENDER_DIST_MULT", 3.0)
    w_icp = _env_float("MIS_UVH_ICP_WEIGHT", 0.5)
    w_rnd = _env_float("MIS_UVH_RENDER_WEIGHT", 0.5)
    w_sum = float(abs(w_icp) + abs(w_rnd))
    if w_sum <= 1e-9:
        w_icp, w_rnd = 0.5, 0.5
    else:
        w_icp = float(w_icp) / w_sum
        w_rnd = float(w_rnd) / w_sum

    rng = np.random.default_rng(0)
    cam_K = None
    cam_T_w2c = None
    prev_render_rgb = None
    image_wh = (int(cfg.semantic_image_size[0]), int(cfg.semantic_image_size[1]))
    if enable_render_sim:
        try:
            pts_prev = _mesh_to_points(prev_mesh, max_points=int(render_max_points), rng=rng, deterministic=True)
            if int(pts_prev.shape[0]) < 200:
                raise RuntimeError("pts_prev too few")
            pmin = np.nanmin(pts_prev, axis=0)
            pmax = np.nanmax(pts_prev, axis=0)
            center = 0.5 * (pmin + pmax)
            radius = float(0.5 * np.linalg.norm((pmax - pmin)))
            radius = float(max(1e-3, radius))
            dist = float(max(1e-3, float(render_dist_mult) * radius))
            eye = center + np.array([0.0, 0.0, 1.0], dtype=np.float32) * float(dist)
            cam_T_w2c = _look_at_T_w2c(eye_w=eye, target_w=center)
            cam_K = _default_K_for_library(image_wh)
            mk_prev, _ = _render_mask_depth_from_points(pts_w=pts_prev, T_w2c=cam_T_w2c, K=cam_K, image_wh=image_wh)
            prev_render_rgb, _ = _mask_to_contour_rgb(mk_prev, image_wh, thickness=2)
        except Exception:
            enable_render_sim = False
            cam_K = None
            cam_T_w2c = None
            prev_render_rgb = None

    def _eval_one(rad_now: int) -> dict[str, Any] | None:
        # 生成候选 bbox
        cands_xy: dict[tuple[int, int], tuple[int, int, int, int, int, int]] = {}
        for dy in range(-rad_now, rad_now + 1, step):
            for dx in range(-rad_now, rad_now + 1, step):
                x0 = cx0 + dx - pw // 2
                y0 = cy0 + dy - ph // 2
                x, y, ww, hh = core.clamp_bbox(x0, y0, pw, ph, W, H)
                dx_eff = int((x + ww // 2) - cx0)
                dy_eff = int((y + hh // 2) - cy0)
                cands_xy[(int(x), int(y))] = (int(x), int(y), int(ww), int(hh), int(dx_eff), int(dy_eff))

        # quick 粗筛：仍用 2D mask IoU（加速）
        scored_quick = []
        for (x, y, ww, hh, dx, dy) in cands_xy.values():
            roi = frame_bgr[y:y + hh, x:x + ww]
            qs = float(max(0.1, min(1.0, cfg.quick_scale)))
            qw = int(max(16, round(ww * qs)))
            qh = int(max(16, round(hh * qs)))
            qmin = int(max(16, cfg.quick_min_size))
            if min(qw, qh) < qmin:
                s2 = float(qmin) / float(max(1, min(qw, qh)))
                qw = int(round(qw * s2))
                qh = int(round(qh * s2))
            roi_small = cv2.resize(roi, (qw, qh), interpolation=cv2.INTER_AREA)
            _, mask_small_u8, _ = core.compute_th_gray_truncated(roi_small, cfg.th_percent, cfg)
            ms = cv2.resize(mask_small_u8, (64, 64), interpolation=cv2.INTER_NEAREST)
            q = core.quick_mask_score(prev_mask_small, ms)
            dist = (abs(dx) + abs(dy)) / float(max(1, rad_now))
            q2 = float(q) - float(cfg.spatial_penalty) * float(dist)
            scored_quick.append((q2, q, x, y, ww, hh, dx, dy))
        scored_quick.sort(key=lambda t: t[0], reverse=True)
        top = scored_quick[: max(1, int(cfg.topk_semantic))]

        # ICP 精排：用 fitness 作为“semantic”分数（0~1），再按原融合公式评分
        best = None
        best_score = -1e9
        for (_q2, q, x, y, ww, hh, dx, dy) in top:
            roi = frame_bgr[y:y + hh, x:x + ww]
            th_gray, mask_u8, th_uint8 = core.compute_th_gray_truncated(roi, cfg.th_percent, cfg)
            cnt = core.extract_largest_contour(mask_u8)
            if cnt is None:
                contour_rgb = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
            else:
                cnt2 = core.resize_contour(cnt, (ww, hh), cfg.semantic_image_size)
                contour_rgb = core.contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)

            mesh_cur = core.make_heightfield_mesh_from_th_gray(th_gray, (int(x), int(y), int(ww), int(hh)), z_scale=float(cfg.mesh_z_scale))
            reg = core.estimate_T_prev_cur_icp(mesh_prev=prev_mesh, mesh_cur=mesh_cur, cfg=cfg) if mesh_cur is not None else None
            icp_fit = float(reg.fitness) if reg is not None else 0.0
            rmse = float(reg.rmse) if reg is not None else float("inf")

            render_used = 0
            render_sim = float("nan")
            render_prev_rgb = prev_render_rgb
            render_cur_rgb = None
            if enable_render_sim and (reg is not None) and (prev_render_rgb is not None) and (cam_K is not None) and (cam_T_w2c is not None):
                try:
                    pts_cur = _mesh_to_points(mesh_cur, max_points=int(render_max_points), rng=rng, deterministic=True) if mesh_cur is not None else np.zeros((0, 3), np.float32)
                    if int(pts_cur.shape[0]) >= 50:
                        T = np.asarray(reg.T, dtype=np.float64).reshape(4, 4)
                        R = T[:3, :3]
                        t = T[:3, 3]
                        pts_al = (R @ pts_cur.astype(np.float64).T).T + t[None, :]
                        mk_cur, _ = _render_mask_depth_from_points(pts_w=pts_al, T_w2c=cam_T_w2c, K=cam_K, image_wh=image_wh)
                        render_cur_rgb, _ = _mask_to_contour_rgb(mk_cur, image_wh, thickness=2)
                        render_sim = float(core.semantic_similarity(render_prev_rgb, render_cur_rgb, cfg))
                        render_used = 1
                except Exception:
                    render_used = 0
                    render_sim = float("nan")
                    render_cur_rgb = None

            sim_r = float(render_sim) if (render_used and np.isfinite(float(render_sim))) else 0.0
            sem = float(max(0.0, min(1.0, float(w_icp) * float(icp_fit) + float(w_rnd) * float(sim_r))))

            score = float(0.85 * sem + 0.15 * max(0.0, q)) - float(cfg.spatial_penalty) * (
                (abs(dx) + abs(dy)) / float(max(1, rad_now))
            )
            if score > best_score:
                best_score = score
                best = {
                    "bbox": (int(x), int(y), int(ww), int(hh)),
                    "score": float(score),
                    "semantic": float(sem),
                    "quick": float(q),
                    "dx": int(dx),
                    "dy": int(dy),
                    "th_gray": th_gray,
                    "mask_u8": mask_u8,
                    "contour_rgb": contour_rgb,
                    "th_uint8": float(th_uint8),
                    "uvh_icp_used": 1,
                    "uvh_icp_fitness": float(icp_fit),
                    "uvh_icp_rmse": float(rmse if np.isfinite(rmse) else float("nan")),
                    "uvh_render_used": int(render_used),
                    "uvh_render_sim": float(render_sim if np.isfinite(float(render_sim)) else float("nan")),
                    "uvh_render_prev_rgb": render_prev_rgb,
                    "uvh_render_cur_rgb": render_cur_rgb,
                }
        return best

    best = _eval_one(rad)
    if cfg.recover_enabled and (best is None or float(best.get("score", 0.0)) < float(cfg.recover_score)):
        rad2 = int(max(rad * 2, rad + step))
        best2 = _eval_one(rad2)
        if best2 is not None and (best is None or float(best2.get("score", -1e9)) > float(best.get("score", -1e9))):
            best = best2

    if best is None:
        roi = core.crop(frame_bgr, prev_bbox)
        th_gray, mask_u8, th_uint8 = core.compute_th_gray_truncated(roi, cfg.th_percent, cfg)
        cnt = core.extract_largest_contour(mask_u8)
        if cnt is not None:
            cnt2 = core.resize_contour(cnt, (prev_bbox[2], prev_bbox[3]), cfg.semantic_image_size)
            contour_rgb = core.contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)
        else:
            contour_rgb = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
        best = {
            "bbox": prev_bbox,
            "score": 0.0,
            "semantic": 0.0,
            "quick": 0.0,
            "dx": 0,
            "dy": 0,
            "th_gray": th_gray,
            "mask_u8": mask_u8,
            "contour_rgb": contour_rgb,
            "th_uint8": float(th_uint8),
            "uvh_icp_used": 0,
            "uvh_icp_fitness": float("nan"),
            "uvh_icp_rmse": float("nan"),
            "uvh_render_used": 0,
            "uvh_render_sim": float("nan"),
        }
    return best


def main() -> None:
    cfg = core.TrackConfig()

    # 核心优化：全局降分辨率（大幅提升速度）
    # 保守配置：75% 分辨率，速度提升约 2倍，精度损失 < 1.5%
    cfg.proc_scale = float(_env_float("MIS_PROC_SCALE", 0.75))  # 优化：从1.0降至0.75
    
    # 同步降低语义图像尺寸（保持比例）
    base_semantic_size = int(_env_int("MIS_BASE_SEMANTIC_SIZE", 112))  # 优化：从128降至112
    cfg.semantic_image_size = (base_semantic_size, base_semantic_size)

    # 断点续跑：默认关闭。MIS_RESUME_ENABLE=1 时启用；此时默认会读 checkpoint，
    # 若需在同目录强制从头跑可设 MIS_FORCE_RESTART=1。
    resume_on = bool(_env_bool("MIS_RESUME_ENABLE", False))
    cfg.resume_enable = bool(resume_on)
    if resume_on:
        cfg.force_restart = bool(_env_bool("MIS_FORCE_RESTART", False))
    else:
        cfg.force_restart = True

    # 输入帧目录：默认切到你当前实验的数据目录；
    # 可用环境变量 MIS_FRAMES_DIR 覆盖（为空则回退默认值）。
    default_frames_dir = r"E:\Surgical_Tracking_Datasets\Chess_Box_1_8\captured_photos3"
    cfg.frames_dir = (os.environ.get("MIS_FRAMES_DIR", default_frames_dir) or "").strip().strip('"').strip("'") or default_frames_dir

    # 输出目录：默认使用新的子目录，避免与带 RT 的脚本混在一起；
    # 可用环境变量 MIS_OUTPUT_DIR 覆盖（为空则回退默认值）。
    default_out = r"D:\MIS_Pose_Track_Re3D\Exp_Results_TrackOnly_3DMatch_captured_photos3"
    cfg.output_dir = (os.environ.get("MIS_OUTPUT_DIR", default_out) or "").strip() or default_out

    # 沿用 TrackConfig.model_path 记录轮廓网络权重路径（与 MIS_CONTOUR_SIM_CKPT 一致，供日志/元数据）
    cfg.model_path = _env_contour_sim_ckpt_default()

    # 默认恢复“鼠标像素级标记目标”的交互流程；
    # 若需要无人值守/复现实验，仍可通过 MIS_INIT_BBOX 显式覆盖。
    default_init_bbox = ""
    cfg.init_bbox = (os.environ.get("MIS_INIT_BBOX", default_init_bbox) or "").strip().strip('"').strip("'")

    # 最终验证配置：完整序列默认降低日志频率，避免控制台过度刷屏。
    cfg.print_every = int(_env_int("MIS_PRINT_EVERY", 25))

    # 跟踪搜索窗：以上一帧 bbox 为中心扩大 search_radius（proc_scale 后坐标系）。
    # MIS_SEARCH_STEP / MIS_TOPK_SEMANTIC 可用环境变量覆盖。
    # MIS_SAVE_DEBUG 默认 1：保存每帧带框图与 roi/mask/th/contour 等中间结果；设 0 可提速、减磁盘占用。
    cfg.save_debug_images = bool(_env_bool("MIS_SAVE_DEBUG", True))
    # 优化：默认更小的搜索窗口以提速（radius 56->32, step 20->24）
    cfg.search_radius = int(_env_int("MIS_SEARCH_RADIUS", 32))  # 优化：从56降至32
    cfg.search_step = int(_env_int("MIS_SEARCH_STEP", 24))  # 优化：从20增至24
    cfg.topk_semantic = int(_env_int("MIS_TOPK_SEMANTIC", 3))  # 优化：从4降至3

    # 诊断：是否额外运行“光流 -> uvh 匹配 -> RANSAC/IRLS”并保存中间结果
    enable_flow_uvh = bool(int(os.environ.get("MIS_ENABLE_FLOW_UVH", "0")))
    flow_uvh_mask_mode = (os.environ.get("MIS_FLOW_UVH_MASK_MODE", "bbox") or "").strip().lower()
    if flow_uvh_mask_mode not in ("bbox", "full"):
        flow_uvh_mask_mode = "bbox"

    # ===== 跟踪匹配模式 =====
    #  - lib        : 多视点目标库检索
    #  - uvh_icp    : 伪3D mesh + ICP（需 open3d）
    #  - uvh_render : 伪3D 点云 + 可选 viewlib + 迭代渲染匹配（代码默认，可用 MIS_TRACK_MATCH_MODE 覆盖）
    #  - 2d         : 2D 轮廓 + 语义/回退
    default_track_mode = "uvh_render"
    track_match_mode = (os.environ.get("MIS_TRACK_MATCH_MODE", default_track_mode) or default_track_mode).strip().lower()
    if track_match_mode not in ("lib", "uvh_icp", "uvh_render", "2d"):
        track_match_mode = str(default_track_mode)

    # 优化：多视点目标库（1024 渲染模板）用于后续匹配
    enable_view_lib = bool(track_match_mode == "lib")
    enable_uvh_icp_track = bool(track_match_mode == "uvh_icp")
    enable_uvh_render_track = bool(track_match_mode == "uvh_render")
    uvh_render_ref_mode = (os.environ.get("MIS_UVH_RENDER_REF_MODE", "first") or "").strip().lower()
    if uvh_render_ref_mode not in ("first", "prev"):
        uvh_render_ref_mode = "first"
    lib_n_views = _env_int("MIS_LIBRARY_NUM_VIEWS", 512)
    lib_pts_max = _env_int("MIS_LIBRARY_MAX_POINTS", 80000)
    lib_seed = _env_int("MIS_LIBRARY_SEED", 0)
    lib_force_rebuild = _env_bool("MIS_LIBRARY_FORCE_REBUILD", False)
    lib_sampler = (os.environ.get("MIS_LIBRARY_SAMPLER", "shell") or "").strip().lower()
    if lib_sampler not in ("combo", "shell"):
        lib_sampler = "shell"
    # “按球壳分层均匀采样”的半径范围（mult * r），例如 0.5r -> 5r
    lib_radius_min_mult = _env_float("MIS_LIBRARY_RADIUS_MIN_MULT", 0.5)
    lib_radius_max_mult = _env_float("MIS_LIBRARY_RADIUS_MAX_MULT", 5.0)
    lib_radius_levels = _env_int("MIS_LIBRARY_RADIUS_LEVELS", 64)  # >1 则用 range 生成 dist_mults；否则沿用 MIS_LIBRARY_DIST_MULTS
    lib_radius_spacing = (os.environ.get("MIS_LIBRARY_RADIUS_SPACING", "linear") or "").strip().lower()
    if lib_radius_spacing not in ("log", "linear"):
        lib_radius_spacing = "linear"
    lib_dist_mults = _parse_float_list("MIS_LIBRARY_DIST_MULTS", "2.2,2.8,3.4,4.2")
    if int(lib_radius_levels) > 1:
        try:
            a = float(max(1e-3, float(lib_radius_min_mult)))
            b = float(max(a * 1.0001, float(lib_radius_max_mult)))
            nlv = int(max(2, int(lib_radius_levels)))
            if lib_radius_spacing == "linear":
                vv = np.linspace(a, b, num=nlv, dtype=np.float64)
            else:
                vv = np.logspace(math.log10(a), math.log10(b), num=nlv, dtype=np.float64)
            lib_dist_mults = [float(x) for x in vv.tolist()]
        except Exception:
            pass
    lib_dist_jitter = _env_float("MIS_LIBRARY_DIST_JITTER", 0.06)
    lib_num_dirs = _env_int("MIS_LIBRARY_NUM_DIRS", 0)  # 0 表示 dirs=n_views；>0 表示 dirs 单独控制（会与 dist/roll 组合）
    lib_roll_deg_list = _parse_float_list("MIS_LIBRARY_ROLL_DEG_LIST", "0")
    lib_z_bias = _env_float("MIS_LIBRARY_Z_BIAS", 1.0)
    lib_angle_gate_deg = _env_float("MIS_LIBRARY_ANGLE_GATE_DEG", 60.0)  # 优化：从90降至60，更早过滤不匹配视点
    lib_save_depth_vis = _env_bool("MIS_LIBRARY_SAVE_DEPTH_VIS", True)
    lib_save_mask_png = _env_bool("MIS_LIBRARY_SAVE_MASK_PNG", False)
    lib_save_view_pts = _env_bool("MIS_LIBRARY_SAVE_VIEW_PTS", True)
    lib_view_pts_n = _env_int("MIS_LIBRARY_VIEW_PTS_N", 8096)
    lib_estimate_init_rt = _env_bool("MIS_LIBRARY_ESTIMATE_INIT_RT", True)
    lib_save_ref_mesh = _env_bool("MIS_LIBRARY_SAVE_REF_MESH", True)
    lib_save_ref_mesh_ply = _env_bool("MIS_LIBRARY_SAVE_REF_MESH_PLY", True)
    lib_save_ref_mesh_npz = _env_bool("MIS_LIBRARY_SAVE_REF_MESH_NPZ", True)

    # ===== uvh-render：可选使用“多视角 view library”做粗检索（思路3：检索 + 迭代优化）=====
    uvh_viewlib_enable = _env_bool("MIS_UVH_RENDER_VIEWLIB_ENABLE", False)
    uvh_viewlib_angle_gate_deg = _env_float("MIS_UVH_RENDER_VIEWLIB_ANGLE_GATE_DEG", float(lib_angle_gate_deg))

    # 跟踪脚本强制使用 ROI 模式（fullframe 会跳过 bbox 搜索，不符合“跟踪”）
    cfg.rt_mode = "roi"

    frames = core.list_frames(cfg.frames_dir)
    if cfg.max_frames and cfg.max_frames > 0:
        frames = frames[: int(cfg.max_frames)]
    if not frames:
        raise RuntimeError(f"未找到任何帧图像: {cfg.frames_dir}")

    # 输出目录结构（保留原脚本中与跟踪直接相关的部分）
    out_root = _ensure_dir(cfg.output_dir)
    out_frames = _ensure_dir(os.path.join(out_root, "frames_with_bbox"))
    out_roi = _ensure_dir(os.path.join(out_root, "roi"))
    out_mask = _ensure_dir(os.path.join(out_root, "mask"))
    out_th = _ensure_dir(os.path.join(out_root, "th_gray"))
    out_cont = _ensure_dir(os.path.join(out_root, "contour_rgb"))
    out_cmp = _ensure_dir(os.path.join(out_root, "contour_compare"))
    out_vis = _ensure_dir(os.path.join(out_root, "vis"))
    out_uvh_render = _ensure_dir(os.path.join(out_root, "uvh_render")) if (enable_uvh_icp_track or enable_uvh_render_track) else ""
    # 与 MIS_SAVE_DEBUG 联动：默认同开；仅需大图/ply 时可单独关 MIS_UVH_RENDER_SAVE_DEBUG=0
    _uvh_dbg_env = os.environ.get("MIS_UVH_RENDER_SAVE_DEBUG", "").strip()
    if not _uvh_dbg_env:
        save_uvh_render_debug = bool(
            (enable_uvh_icp_track or enable_uvh_render_track) and bool(out_uvh_render) and bool(cfg.save_debug_images)
        )
    else:
        save_uvh_render_debug = bool(
            (enable_uvh_icp_track or enable_uvh_render_track) and bool(out_uvh_render) and _env_bool("MIS_UVH_RENDER_SAVE_DEBUG", True)
        )
    uvh_render_debug_stride = int(max(1, _env_int("MIS_UVH_RENDER_SAVE_DEBUG_STRIDE", 10)))

    # 光流/uvh 伪3D匹配诊断输出目录（沿用原 RT 脚本的命名，便于对照）
    if enable_flow_uvh:
        out_pairs = _ensure_dir(os.path.join(out_root, "pair_results"))
        out_pairs_matches = _ensure_dir(os.path.join(out_pairs, "uvh_matches"))
        out_pairs_matches_vis = _ensure_dir(os.path.join(out_pairs_matches, "vis"))
        out_pairs_flow = _ensure_dir(os.path.join(out_pairs, "flow"))
        out_pairs_flow_hsv = _ensure_dir(os.path.join(out_pairs_flow, "hsv"))

    # 语义模块初始化（可用则启用，否则自动回退）
    sem_meta = _init_semantic(cfg)
    renderfield_pack = _get_renderfield_refiner(cfg=cfg) if enable_uvh_render_track else None

    _log(
        (
            f"[配置] frames={len(frames)} frames_dir={cfg.frames_dir} out_dir={cfg.output_dir} "
            f"proc_scale={float(cfg.proc_scale):g} search_radius={int(cfg.search_radius)} step={int(cfg.search_step)} "
            f"topk={int(cfg.topk_semantic)} print_every={int(cfg.print_every)} verbose={int(getattr(cfg, 'verbose', 1))} "
            f"mode={str(track_match_mode)} resume={int(bool(cfg.resume_enable))} force_restart={int(bool(cfg.force_restart))} "
            f"flow_uvh={int(enable_flow_uvh)} viewlib(lib_mode)={int(enable_view_lib)} "
            f"uvh_render_viewlib={int(bool(uvh_viewlib_enable))} uvh_diff={int(bool(_env_bool('MIS_UVH_DIFFRENDER_ENABLE', False)))} "
            f"uvh_debug_stride={int(uvh_render_debug_stride)}"
        ),
        cfg=cfg,
        level=1,
    )
    _log(
        (
            f"[语义] backend={sem_meta.get('semantic_backend', '')} "
            f"enabled={int(sem_meta.get('semantic_enabled', False))} available={int(sem_meta.get('semantic_available', False))} "
            f"device={sem_meta.get('device', '')} warmup_ok={int(sem_meta.get('semantic_warmup_ok', False))} "
            f"model_exists={int(sem_meta.get('model_exists', False))}"
        ),
        cfg=cfg,
        level=1,
    )

    meta = {
        "time": _now(),
        "task": "tracking_only_no_rt",
        "frames_dir": cfg.frames_dir,
        "output_dir": cfg.output_dir,
        "proc_scale": float(cfg.proc_scale),
        "th_percent": float(cfg.th_percent),
        "search_radius": int(cfg.search_radius),
        "search_step": int(cfg.search_step),
        "topk_semantic": int(cfg.topk_semantic),
        "spatial_penalty": float(cfg.spatial_penalty),
        "quick_scale": float(cfg.quick_scale),
        "quick_min_size": int(cfg.quick_min_size),
        "resume_enable": int(bool(getattr(cfg, "resume_enable", False))),
        "force_restart": int(bool(getattr(cfg, "force_restart", True))),
        "enable_flow_uvh": int(enable_flow_uvh),
        "flow_uvh_mask_mode": str(flow_uvh_mask_mode),
        "track_match_mode": str(track_match_mode),
        "enable_uvh_icp_track": int(enable_uvh_icp_track),
        "enable_uvh_render_track": int(enable_uvh_render_track),
        "uvh_render_ref_mode": str(uvh_render_ref_mode),
        "uvh_render_viewlib_enable": int(bool(uvh_viewlib_enable)),
        "uvh_render_viewlib_angle_gate_deg": float(uvh_viewlib_angle_gate_deg),
        "uvh_renderfield_enable": int(enable_uvh_render_track and bool(_env_bool("MIS_UVH_RENDERFIELD_ENABLE", False))),
        "uvh_renderfield_checkpoint": (
            str(renderfield_pack.get("checkpoint_path", ""))
            if isinstance(renderfield_pack, dict)
            else str(_RENDERFIELD_CACHE.get("checkpoint", ""))
        ),
        "uvh_renderfield_image_size": int(renderfield_pack.get("image_size", 0)) if isinstance(renderfield_pack, dict) else 0,
        "uvh_renderfield_device": str(renderfield_pack.get("device", "")) if isinstance(renderfield_pack, dict) else "",
        "enable_view_library": int(enable_view_lib),
        "library_num_views": int(lib_n_views),
        "library_max_points": int(lib_pts_max),
        "library_sampler": str(lib_sampler),
        "library_radius_min_mult": float(lib_radius_min_mult),
        "library_radius_max_mult": float(lib_radius_max_mult),
        "library_radius_levels": int(lib_radius_levels),
        "library_radius_spacing": str(lib_radius_spacing),
        "library_seed": int(lib_seed),
        "library_force_rebuild": int(bool(lib_force_rebuild)),
        "library_dist_mults": [float(v) for v in lib_dist_mults],
        "library_dist_jitter": float(lib_dist_jitter),
        "library_num_dirs": int(lib_num_dirs),
        "library_roll_deg_list": [float(v) for v in lib_roll_deg_list],
        "library_z_bias": float(lib_z_bias),
        "library_angle_gate_deg": float(lib_angle_gate_deg),
        "library_save_depth_vis": int(bool(lib_save_depth_vis)),
        "library_save_mask_png": int(bool(lib_save_mask_png)),
        "library_save_view_pts": int(bool(lib_save_view_pts)),
        "library_view_pts_n": int(lib_view_pts_n),
        "library_estimate_init_rt": int(bool(lib_estimate_init_rt)),
        **sem_meta,
        "note": (
            "Tracking-only pipeline (no RT export). "
            "If enable_view_library=1 (lib mode) or uvh_render_viewlib_enable=1 (uvh_render mode), the script builds/loads an upper-hemisphere multi-view template library (e.g., 1024/64k views) under target_library/. "
            "In lib mode it is used for fast 2D contour matching; in uvh_render mode it is used for coarse multi-view retrieval before iterative (s,theta,tx,ty) optimization. "
            "If enable_flow_uvh=1, this script also runs Farneback flow -> uvh matches -> RANSAC/IRLS rigid fitting as a diagnostic and saves intermediate results "
            "under pair_results/ (flow npz/hsv, uvh_matches npz/vis). "
            "The diagnostic flow/uvh outputs remain for verification/debug and are not used to pick the tracking bbox."
        ),
    }
    with open(os.path.join(out_root, "meta_track_only.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 读第一帧（与原脚本一致：可按 proc_scale 降分辨率处理）
    first = core._read_frame_bgr(frames[0], cfg)
    if first is None:
        raise RuntimeError(f"读取失败: {frames[0]}")

    H, W = first.shape[:2]
    print(f"[信息] 第一帧尺寸: {W}x{H}")
    if abs(float(cfg.proc_scale) - 1.0) > 1e-6:
        print(f"[信息] 已启用降分辨率处理: MIS_PROC_SCALE={cfg.proc_scale}")

    # 首帧目标：鼠标像素级标注 mask -> 自动包围框 bbox
    bbox, init_mask_full = _select_initial_target_mask_and_bbox(first, cfg)

    # 第一帧特征
    roi0 = core.crop(first, bbox)
    th0_raw, mask0_th68, th_u8_0 = core.compute_th_gray_truncated(roi0, cfg.th_percent, cfg)
    x0, y0, w0, h0 = [int(v) for v in bbox]
    try:
        mask0_user = np.asarray(init_mask_full[y0:y0 + h0, x0:x0 + w0], dtype=np.uint8)
    except Exception:
        mask0_user = np.zeros((int(h0), int(w0)), dtype=np.uint8)
    if mask0_user.ndim != 2 or mask0_user.shape[0] != int(h0) or mask0_user.shape[1] != int(w0):
        mask0_user = cv2.resize(mask0_user, (int(w0), int(h0)), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    mask0_user = (mask0_user > 0).astype(np.uint8) * 255
    # 伪3D高度场：仍使用 th_68 的灰度截断图（float32），但只保留“用户标注 mask”内的像素
    th0 = np.where(mask0_user > 0, np.asarray(th0_raw, dtype=np.float32), 0.0).astype(np.float32)
    # 跟踪目标 mask：使用像素级标注的 user mask（而不是 th_68 二值 mask）
    mask0 = mask0_user

    cnt0 = _extract_contour_prefer_center(mask0)
    if cnt0 is None:
        cnt0 = core.extract_largest_contour(mask0)
    if cnt0 is not None:
        cnt0s = core.resize_contour(cnt0, (bbox[2], bbox[3]), cfg.semantic_image_size)
        contour0_rgb = core.contour_to_rgb(cnt0s, cfg.semantic_image_size, thickness=2)
    else:
        contour0_rgb = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
    # quick 粗筛参考：固定使用首帧 reference（避免 prev_mask_small 逐帧更新导致漂移累积）
    # 注意：后续候选 quick 直接从“统一阈值的 search-window mask”裁剪并 resize 到 64x64，
    # 因此这里也用首帧 mask0 直接 resize，保证域一致。
    ref_mask_small = _mask64_from_mask_u8(mask0)
    prev_mask_small = ref_mask_small.copy()
    prev_roi_th = np.asarray(th0, dtype=np.float32)
    prev_roi_mask = np.asarray(mask0, dtype=np.uint8)

    # 保存第一帧中间结果
    if cfg.save_debug_images:
        cv2.imwrite(os.path.join(out_frames, "frame_0000.jpg"), core.draw_bbox(first, bbox, text="f=0"))
        cv2.imwrite(os.path.join(out_roi, "roi_0000.jpg"), roi0)
        cv2.imwrite(os.path.join(out_mask, "mask_0000.png"), mask0)
        # 同时保存一份 th_68 的二值 mask（便于对照）
        try:
            cv2.imwrite(os.path.join(out_mask, "mask_th68_0000.png"), np.asarray(mask0_th68, dtype=np.uint8))
        except Exception:
            pass
        cv2.imwrite(os.path.join(out_th, "th_gray_0000.png"), np.clip(th0, 0, 255).astype(np.uint8))
        cv2.imwrite(os.path.join(out_cont, "contour_rgb_0000.png"), contour0_rgb)
        cv2.imwrite(os.path.join(out_vis, "roi_overlay_0000.jpg"), core.make_overlay_mask(roi0, mask0))
        core._save_similarity_compare_image(
            os.path.join(out_cmp, "compare_0000.png"),
            prev_contour_rgb=contour0_rgb,
            cur_contour_rgb=contour0_rgb,
            similarity=1.0,
            title_left="prev(contour)",
            title_right="cur(contour)",
        )

    track_rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []

    track_rows.append(
        {
            "frame": 0,
            "image": frames[0],
            "bbox": list(map(int, bbox)),
            "th_uint8": float(th_u8_0),
            "score": 1.0,
            "semantic": 1.0,
            "quick": 1.0,
            "dx": 0,
            "dy": 0,
        }
    )

    # ===== 多视点目标库：基于首帧 ROI 的伪3D(uvh)点云，采样上半球多视点渲染模板 =====
    lib: ViewLibrary | None = None
    lib_meta: dict[str, Any] = {}
    prev_view_id: int = -1
    prev_view_dir: Optional[np.ndarray] = None
    ref_contour_rgb = contour0_rgb

    # ===== uvh-ICP：上一帧伪3D mesh（用于下一帧 ICP 匹配）=====
    prev_mesh = None
    if enable_uvh_icp_track:
        try:
            prev_mesh = core.make_heightfield_mesh_from_th_gray(th0, bbox, z_scale=float(cfg.mesh_z_scale))
            if prev_mesh is None:
                raise RuntimeError("prev_mesh=None (open3d 可能缺失或 mesh 太稀疏)")
        except Exception as e:
            print(f"[UVH-ICP][Warning] 伪3D mesh 初始化失败，将回退到 uvh_render 模式：{e}")
            enable_uvh_icp_track = False
            enable_uvh_render_track = True
            track_match_mode = "uvh_render"

    # ===== uvh-render：伪3D点云参考（first/prev）=====
    prev_uvh_pts: Optional[np.ndarray] = None
    uvh_rigid_ref_pts: Optional[np.ndarray] = None
    uvh_diff_ref_verts: Optional[np.ndarray] = None
    uvh_diff_ref_faces: Optional[np.ndarray] = None
    renderfield_ref_ctx: Optional[dict[str, Any]] = None
    if enable_uvh_render_track:
        try:
            mp = _env_int("MIS_UVH_RENDER_MAX_POINTS", 10000)
            rng0 = np.random.default_rng(0)
            prev_uvh_pts = _uvh_points_from_th_roi(
                th_gray_truncated=th0,
                bbox=bbox,
                z_scale=float(cfg.mesh_z_scale),
                max_points=int(mp),
                rng=rng0,
                deterministic=True,
            )
            if prev_uvh_pts is None or int(np.asarray(prev_uvh_pts).shape[0]) < 200:
                raise RuntimeError(f"prev_uvh_pts too few: {0 if prev_uvh_pts is None else int(np.asarray(prev_uvh_pts).shape[0])}")
            try:
                uvh_rigid_ref_pts = _uvh_points_from_th_mask_roi(
                    th_gray_truncated=th0,
                    mask_u8=mask0,
                    bbox=bbox,
                    z_scale=float(cfg.mesh_z_scale),
                    max_points=int(max(2000, _env_int("MIS_UVH_RIGID_MAX_POINTS", 30000))),
                    rng=np.random.default_rng(0),
                    fill_mask=True,
                    min_height=float(_env_float("MIS_UVH_RIGID_MIN_HEIGHT", 1.0)),
                    deterministic=True,
                )
                if uvh_rigid_ref_pts is None or int(np.asarray(uvh_rigid_ref_pts).shape[0]) < 200:
                    uvh_rigid_ref_pts = np.asarray(prev_uvh_pts, dtype=np.float32).copy()
            except Exception:
                uvh_rigid_ref_pts = np.asarray(prev_uvh_pts, dtype=np.float32).copy()
        except Exception as e:
            print(f"[UVH-Render][Warning] 伪3D点云初始化失败，将回退到 2D 模式：{e}")
            enable_uvh_render_track = False
            track_match_mode = "2d"
        # Kaolin 可微渲染：预先构建“首帧 reference mesh”（用于后续每帧视点连续优化）
        if enable_uvh_render_track and bool(_env_bool("MIS_UVH_DIFFRENDER_ENABLE", False)) and str(uvh_render_ref_mode) == "first":
            try:
                mk_ref = None
                try:
                    mk_ref = _mask_to_filled_u8(np.asarray(mask0, dtype=np.uint8))
                except Exception:
                    mk_ref = None
                th_ref = np.asarray(th0, dtype=np.float32)
                if mk_ref is not None and mk_ref.ndim == 2:
                    if mk_ref.shape != th_ref.shape:
                        mk_ref = cv2.resize(mk_ref, (int(th_ref.shape[1]), int(th_ref.shape[0])), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                    th_ref = np.where(mk_ref > 0, np.maximum(th_ref, 1.0), 0.0).astype(np.float32)
                vref, fref, _ = _make_heightfield_mesh_arrays(th_gray_truncated=th_ref, bbox=bbox, z_scale=float(cfg.mesh_z_scale))
                if int(np.asarray(vref).shape[0]) >= 50 and int(np.asarray(fref).shape[0]) >= 50:
                    uvh_diff_ref_verts = np.asarray(vref, dtype=np.float32)
                    uvh_diff_ref_faces = np.asarray(fref, dtype=np.int32)
                    _log_once(
                        "uvh_diff_ref_ready",
                        f"[UVH-Diff] reference mesh ready: verts={int(uvh_diff_ref_verts.shape[0])} faces={int(uvh_diff_ref_faces.shape[0])}",
                        cfg=cfg,
                        level=1,
                    )
                else:
                    _log_once(
                        "uvh_diff_ref_small",
                        f"[UVH-Diff][Skip] reference mesh too small: verts={int(np.asarray(vref).shape[0])} faces={int(np.asarray(fref).shape[0])} -> diff-render 将不执行",
                        cfg=cfg,
                        level=1,
                    )
            except Exception:
                uvh_diff_ref_verts, uvh_diff_ref_faces = None, None

    if enable_view_lib or (enable_uvh_render_track and bool(uvh_viewlib_enable)):
        try:
            rng = np.random.default_rng(int(lib_seed))
            pts0_w = _uvh_points_from_th_roi(
                th_gray_truncated=th0,
                bbox=bbox,
                z_scale=float(cfg.mesh_z_scale),
                max_points=int(lib_pts_max),
                rng=rng,
            )
            if int(pts0_w.shape[0]) < 200:
                raise RuntimeError(f"伪3D点云过少: {int(pts0_w.shape[0])}")

            # 保存首帧目标伪3D mesh（参考模型）
            if bool(lib_save_ref_mesh):
                try:
                    lib_dir0 = _ensure_dir(os.path.join(out_root, "target_library"))
                    mesh_dir = _ensure_dir(os.path.join(lib_dir0, "mesh"))
                    mk0 = None
                    try:
                        mk0 = _mask_to_filled_u8(np.asarray(mask0, dtype=np.uint8))
                    except Exception:
                        mk0 = None
                    thm0 = np.asarray(th0, dtype=np.float32)
                    if mk0 is not None and mk0.ndim == 2:
                        if mk0.shape != thm0.shape:
                            mk0 = cv2.resize(mk0, (int(thm0.shape[1]), int(thm0.shape[0])), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
                        thm0 = np.where(mk0 > 0, np.maximum(thm0, 1.0), 0.0).astype(np.float32)
                    v0, f0, c0 = _make_heightfield_mesh_arrays(th_gray_truncated=thm0, bbox=bbox, z_scale=float(cfg.mesh_z_scale))
                    if int(v0.shape[0]) > 0 and int(f0.shape[0]) > 0:
                        if bool(lib_save_ref_mesh_ply) or bool(lib_save_ref_mesh_npz):
                            _save_ply_ascii(os.path.join(mesh_dir, "ref_mesh_world.ply"), verts=v0, faces=f0, colors_u8=c0)
                except Exception:
                    pass

            lib, lib_meta = build_or_load_view_library(
                out_root=out_root,
                pts_w=pts0_w,
                cfg=cfg,
                n_views=int(lib_n_views),
                n_dirs=int(lib_num_dirs),
                dist_mults=list(lib_dist_mults),
                dist_jitter=float(lib_dist_jitter),
                roll_deg_list=list(lib_roll_deg_list),
                z_bias=float(lib_z_bias),
                sampler=str(lib_sampler),
                seed=int(lib_seed),
                force_rebuild=bool(lib_force_rebuild),
                save_depth_vis=bool(lib_save_depth_vis),
                save_mask_png=bool(lib_save_mask_png),
                save_view_points=bool(lib_save_view_pts),
                view_points_n=int(lib_view_pts_n),
            )
            if lib is None:
                raise RuntimeError("目标库生成失败(lib=None)")

            # 用 frame0 的轮廓，在目标库中找到“最像的视点”作为初始 view
            m0 = match_contour_to_library(lib=lib, contour_rgb=contour0_rgb, prev_view_dir=None, angle_gate_deg=0.0)
            prev_view_id = int(m0.get("view_id", -1))
            if 0 <= prev_view_id < int(lib.ids.size):
                prev_view_dir = np.asarray(lib.dirs[int(prev_view_id)], dtype=np.float32).reshape(3)
                try:
                    prev_view_dist = float(lib.dists[int(prev_view_id)])
                except Exception:
                    prev_view_dist = float("nan")
                try:
                    prev_view_roll = float(lib.rolls[int(prev_view_id)])
                except Exception:
                    prev_view_roll = float("nan")
                try:
                    ref0 = core._load_contour_rgb(lib.contour_rgb_paths[int(prev_view_id)])
                except Exception:
                    ref0 = None
                if ref0 is not None:
                    ref_contour_rgb = ref0

            if enable_uvh_render_track:
                try:
                    renderfield_ref_ctx = _build_renderfield_reference_context(
                        ref_points=(
                            np.asarray(uvh_rigid_ref_pts, dtype=np.float32)
                            if isinstance(uvh_rigid_ref_pts, np.ndarray)
                            else np.asarray(prev_uvh_pts, dtype=np.float32)
                        ),
                        view_lib=(lib if isinstance(lib, ViewLibrary) else None),
                        prev_view_id=int(prev_view_id),
                        cfg=cfg,
                    )
                    if isinstance(renderfield_ref_ctx, dict):
                        _log_once(
                            "uvh_rf_refctx_ready",
                            (
                                f"[UVH-RF] reference ready init_view={int(prev_view_id)} "
                                f"image={int(renderfield_ref_ctx.get('image_size', 0))}"
                            ),
                            cfg=cfg,
                            level=1,
                        )
                except Exception as e:
                    renderfield_ref_ctx = None
                    _log_once(
                        "uvh_rf_refctx_outer_fail",
                        f"[UVH-RF][Skip] reference 构建失败: {e}",
                        cfg=cfg,
                        level=1,
                    )

            # 仅在 lib 跟踪模式下写入 lib_* 字段（避免 uvh_render 使用 viewlib 时输出歧义）
            if enable_view_lib:
                track_rows[0].update(
                    {
                        "lib_view_id": int(prev_view_id),
                        "lib_score": float(m0.get("score", 0.0)),
                        "lib_iou": float(m0.get("iou", 0.0)),
                        "lib_corr": float(m0.get("corr", 0.0)),
                        "lib_n_used": int(m0.get("n_used", 0)),
                        "lib_used_gate": int(m0.get("used_gate", 0)),
                        "lib_dir": (prev_view_dir.astype(float).reshape(-1).tolist() if prev_view_dir is not None else None),
                        "lib_dist": (float(prev_view_dist) if "prev_view_dist" in locals() else float("nan")),
                        "lib_roll": (float(prev_view_roll) if "prev_view_roll" in locals() else float("nan")),
                        "lib_T_w2c": (np.asarray(lib.T_w2c[int(prev_view_id)], dtype=np.float32).reshape(-1).tolist() if (lib is not None and 0 <= int(prev_view_id) < int(lib.T_w2c.shape[0])) else None),
                        "lib_template_contour": (str(lib.contour_rgb_paths[int(prev_view_id)]) if (lib is not None and 0 <= int(prev_view_id) < len(lib.contour_rgb_paths)) else ""),
                    }
                )
                if cfg.save_debug_images:
                    try:
                        core._save_similarity_compare_image(
                            os.path.join(out_cmp, "compare_lib_0000.png"),
                            prev_contour_rgb=ref_contour_rgb,
                            cur_contour_rgb=contour0_rgb,
                            similarity=float(m0.get("score", 0.0)),
                            title_left=f"lib_view={int(prev_view_id)}",
                            title_right="frame0",
                        )
                    except Exception:
                        pass

            # 首两帧：光流+伪3D uvh 匹配估计 RT（保存到 target_library/，用于你后续渲染/校验）
            if bool(lib_estimate_init_rt) and len(frames) >= 2:
                try:
                    second = core._read_frame_bgr(frames[1], cfg)
                    if second is not None:
                        lib_dir = _ensure_dir(os.path.join(out_root, "target_library"))
                        init_dir = _ensure_dir(os.path.join(lib_dir, "init_pair_0000_0001"))

                        # 先用原始跟踪逻辑粗定位第二帧 bbox（更稳）
                        best01 = core.find_best_bbox_in_next_frame(second, bbox, contour0_rgb, prev_mask_small, cfg)
                        bbox1 = tuple(best01.get("bbox", bbox))

                        # prev：用 ROI 的 th/mask “贴回整帧坐标系”，限制采样只发生在目标区域
                        prev_th_full = np.zeros((H, W), dtype=np.float32)
                        prev_mask_full = np.zeros((H, W), dtype=np.uint8)
                        x0, y0, w0, h0 = [int(v) for v in bbox]
                        prev_th_full[y0:y0 + h0, x0:x0 + w0] = np.asarray(th0, dtype=np.float32)
                        prev_mask_full[y0:y0 + h0, x0:x0 + w0] = np.asarray(mask0, dtype=np.uint8)

                        # cur：整帧 th（允许目标略有位移）
                        cur_th_full, _, _ = core.compute_th_gray_truncated(second, cfg.th_percent, cfg)

                        cur_uvh, prev_uvh, _, flow_dbg = core._build_uvh_matches_from_flow(
                            prev_bgr=first,
                            cur_bgr=second,
                            prev_th=prev_th_full,
                            cur_th=cur_th_full,
                            prev_mask=prev_mask_full,
                            cfg=cfg,
                        )
                        T_flow = None
                        inl = None
                        rmse = float("nan")
                        if int(cur_uvh.shape[0]) >= 20:
                            T0, inl = core._ransac_rigid_T_xy(cur_uvh=cur_uvh, prev_uvh=prev_uvh, cfg=cfg)
                            if T0 is not None:
                                T_flow = core._irls_refine_T(T_init=T0, cur_uvh=cur_uvh, prev_uvh=prev_uvh, cfg=cfg, inlier_mask=inl)
                                try:
                                    Xh = (T_flow[:3, :3] @ cur_uvh.T).T + T_flow[:3, 3][None, :]
                                    e = np.linalg.norm((Xh[:, :2] - prev_uvh[:, :2]), axis=1)
                                    e_in = e[np.asarray(inl, dtype=bool)] if inl is not None else e
                                    rmse = float(np.median(e_in)) if e_in.size > 0 else float("nan")
                                except Exception:
                                    pass

                        # 保存中间结果
                        try:
                            np.savez_compressed(
                                os.path.join(init_dir, "uvh_matches_flow.npz"),
                                prev_uvh=np.asarray(prev_uvh, dtype=np.float32),
                                cur_uvh=np.asarray(cur_uvh, dtype=np.float32),
                                inliers=(np.asarray(inl, dtype=np.uint8) if inl is not None else None),
                                T_prev_cur=(np.asarray(T_flow, dtype=np.float64).reshape(4, 4) if T_flow is not None else None),
                                flow_rmse=float(rmse),
                                bbox0=np.asarray(bbox, dtype=np.int32),
                                bbox1=np.asarray(bbox1, dtype=np.int32),
                            )
                        except Exception:
                            pass
                        try:
                            flow = np.asarray(flow_dbg.get("flow", None))
                            if flow is not None and flow.ndim == 3 and flow.shape[2] == 2:
                                hsv_bgr = core._flow_to_hsv_bgr(flow, clip_mag=20.0)
                                cv2.imwrite(os.path.join(init_dir, "flow_hsv.png"), hsv_bgr)
                        except Exception:
                            pass
                        try:
                            if T_flow is not None:
                                core._save_flow_matches_vis(
                                    os.path.join(init_dir, "matches_vis.png"),
                                    prev_bgr=first,
                                    prev_uvh=prev_uvh,
                                    cur_uvh=cur_uvh,
                                    T_prev_cur=np.asarray(T_flow, dtype=np.float64).reshape(4, 4),
                                    inliers=inl,
                                    max_lines=800,
                                )
                        except Exception:
                            pass

                        # 可选：ICP（复用 Exp_Mehsh_Between_RT / core 的 mesh-ICP）
                        T_icp = None
                        icp_fit = float("nan")
                        icp_rmse = float("nan")
                        try:
                            mesh0 = core.make_heightfield_mesh_from_th_gray(th0, bbox, z_scale=float(cfg.mesh_z_scale))
                            th1 = best01.get("th_gray", None)
                            mesh1 = (
                                core.make_heightfield_mesh_from_th_gray(np.asarray(th1, dtype=np.float32), bbox1, z_scale=float(cfg.mesh_z_scale))
                                if isinstance(th1, np.ndarray) and th1.size > 0
                                else None
                            )
                            reg = core.estimate_T_prev_cur_icp(mesh_prev=mesh0, mesh_cur=mesh1, cfg=cfg) if (mesh0 is not None and mesh1 is not None) else None
                            if reg is not None:
                                T_icp = np.asarray(reg.T, dtype=np.float64).reshape(4, 4)
                                icp_fit = float(reg.fitness)
                                icp_rmse = float(reg.rmse)
                        except Exception:
                            pass

                        try:
                            with open(os.path.join(init_dir, "init_rt_summary.json"), "w", encoding="utf-8") as f:
                                json.dump(
                                    {
                                        "time": _now(),
                                        "bbox0": list(map(int, bbox)),
                                        "bbox1": list(map(int, bbox1)),
                                        "flow_n_matches": int(cur_uvh.shape[0]),
                                        "flow_n_inliers": int(np.count_nonzero(inl)) if inl is not None else 0,
                                        "flow_inlier_ratio": float(np.count_nonzero(inl)) / float(max(1, int(cur_uvh.shape[0]))) if inl is not None else float("nan"),
                                        "flow_rmse_median_px": float(rmse),
                                        "T_flow_prev_cur": (np.asarray(T_flow, dtype=np.float64).reshape(4, 4).tolist() if T_flow is not None else None),
                                        "icp_fitness": float(icp_fit),
                                        "icp_rmse": float(icp_rmse),
                                        "T_icp_prev_cur": (np.asarray(T_icp, dtype=np.float64).reshape(4, 4).tolist() if T_icp is not None else None),
                                        "note": "T_prev_cur maps cur(1) uvh points into prev(0) uvh coord. This is a shape-alignment RT on pseudo-3D (u,v,h), not guaranteed physical camera pose.",
                                    },
                                    f,
                                    ensure_ascii=False,
                                    indent=2,
                                )
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[TargetLibrary][InitRT][Warning] 首两帧 RT 估计失败（不影响跟踪/目标库）：{e}")

        except Exception as e:
            print(f"[TargetLibrary][Warning] 目标库生成/加载失败，将回退单模板跟踪：{e}")
            enable_view_lib = False
            lib = None
            prev_view_id = -1
            prev_view_dir = None
            ref_contour_rgb = contour0_rgb

    prev_bbox = bbox
    prev_contour_rgb = contour0_rgb

    # 光流/uvh：缓存上一帧整帧 th/mask（在整帧坐标系里进行 uvh 匹配）
    prev_full_bgr = first
    prev_full_th: Optional[np.ndarray] = None
    prev_full_mask: Optional[np.ndarray] = None
    if enable_flow_uvh:
        try:
            if str(flow_uvh_mask_mode) == "full":
                prev_full_th, prev_full_mask, _ = core.compute_th_gray_truncated(prev_full_bgr, cfg.th_percent, cfg)
            else:
                HH, WW = prev_full_bgr.shape[:2]
                prev_full_th = np.zeros((HH, WW), dtype=np.float32)
                prev_full_mask = np.zeros((HH, WW), dtype=np.uint8)
                x0, y0, w0, h0 = [int(v) for v in prev_bbox]
                prev_full_th[y0:y0 + h0, x0:x0 + w0] = np.asarray(prev_roi_th, dtype=np.float32)
                prev_full_mask[y0:y0 + h0, x0:x0 + w0] = np.asarray(prev_roi_mask, dtype=np.uint8)
        except Exception:
            prev_full_th, prev_full_mask = None, None

    # 断点续跑：若存在 checkpoint，则从断点开始
    start_frame = 1
    last_done = 0
    if cfg.resume_enable and (not cfg.force_restart):
        resume = _try_load_checkpoint_track_only(
            out_root=out_root,
            cfg=cfg,
            frames=frames,
            out_mask=out_mask,
            out_cont=out_cont,
        )
        if resume is not None:
            try:
                start_frame = int(resume.get("start_frame", 1))
                last_done = int(resume.get("last_frame", 0))
                prev_bbox = tuple(resume.get("prev_bbox", prev_bbox))
                try:
                    prev_view_id = int(resume.get("prev_view_id", prev_view_id))
                except Exception:
                    prev_view_id = int(prev_view_id)
                prev_contour_rgb = np.asarray(resume.get("prev_contour_rgb", prev_contour_rgb), dtype=np.uint8)
                prev_mask_small = np.asarray(resume.get("prev_mask_small", prev_mask_small), dtype=np.uint8)
                track_rows = list(resume.get("track_rows", track_rows))
                diag_rows = list(resume.get("diag_rows", diag_rows))
                print(f"[恢复] checkpoint: last_frame={last_done} -> start_frame={start_frame}")
                # 恢复 view-library 的 gating 状态
                if (lib is not None) and (enable_view_lib or (enable_uvh_render_track and bool(uvh_viewlib_enable))) and (0 <= int(prev_view_id) < int(lib.ids.size)):
                    prev_view_dir = np.asarray(lib.dirs[int(prev_view_id)], dtype=np.float32).reshape(3)
                    try:
                        rr = core._load_contour_rgb(lib.contour_rgb_paths[int(prev_view_id)])
                    except Exception:
                        rr = None
                    if rr is not None:
                        ref_contour_rgb = rr
                # 恢复 uvh-ICP：用 last_done 的 bbox 重建 prev_mesh
                if enable_uvh_icp_track:
                    try:
                        im_prev = core._read_frame_bgr(frames[int(last_done)], cfg)
                        if im_prev is not None:
                            roi_prev = core.crop(im_prev, prev_bbox)
                            th_prev, _, _ = core.compute_th_gray_truncated(roi_prev, cfg.th_percent, cfg)
                            prev_mesh = core.make_heightfield_mesh_from_th_gray(th_prev, prev_bbox, z_scale=float(cfg.mesh_z_scale))
                            if prev_mesh is None:
                                raise RuntimeError("prev_mesh=None")
                    except Exception as e:
                        print(f"[UVH-ICP][Warning] 恢复 prev_mesh 失败，将回退 uvh_render：{e}")
                        enable_uvh_icp_track = False
                        enable_uvh_render_track = True
                        track_match_mode = "uvh_render"
                # 恢复 uvh-render：用 last_done 的 bbox 重建 prev_uvh_pts
                if enable_uvh_render_track:
                    try:
                        mp = _env_int("MIS_UVH_RENDER_MAX_POINTS", 10000)
                        rng0 = np.random.default_rng(0)
                        if str(uvh_render_ref_mode) == "first":
                            # 用“首帧 bbox”作为全局目标参考（从 checkpoint 的 track_rows[0] 取）
                            bbox0_ref = None
                            try:
                                if isinstance(track_rows, list) and len(track_rows) > 0 and isinstance(track_rows[0], dict):
                                    b0 = track_rows[0].get("bbox", None)
                                    if isinstance(b0, list) and len(b0) == 4:
                                        bbox0_ref = (int(b0[0]), int(b0[1]), int(b0[2]), int(b0[3]))
                            except Exception:
                                bbox0_ref = None
                            if bbox0_ref is None:
                                bbox0_ref = tuple(prev_bbox)
                            im0 = core._read_frame_bgr(frames[0], cfg)
                            if im0 is not None:
                                roi0 = core.crop(im0, bbox0_ref)
                                th0_ref, mk0_ref, _ = core.compute_th_gray_truncated(roi0, cfg.th_percent, cfg)
                                # 优先使用首帧“像素级标注 mask”（mask_0000.png）恢复 reference；否则退回 th_68 mask
                                mk0_user = None
                                try:
                                    p0 = os.path.join(out_mask, "mask_0000.png")
                                    if os.path.exists(p0):
                                        mk0_user = core._load_mask_u8(p0)
                                except Exception:
                                    mk0_user = None
                                if mk0_user is not None and int(np.asarray(mk0_user).size) > 0:
                                    try:
                                        th0_ref = np.where(np.asarray(mk0_user, dtype=np.uint8) > 0, np.asarray(th0_ref, dtype=np.float32), 0.0).astype(np.float32)
                                    except Exception:
                                        pass
                                    ref_mask_small = _mask64_from_mask_u8(np.asarray(mk0_user, dtype=np.uint8))
                                else:
                                    ref_mask_small = _mask64_from_mask_u8(np.asarray(mk0_ref, dtype=np.uint8))
                                prev_uvh_pts = _uvh_points_from_th_roi(
                                    th_gray_truncated=th0_ref,
                                    bbox=bbox0_ref,
                                    z_scale=float(cfg.mesh_z_scale),
                                    max_points=int(mp),
                                    rng=rng0,
                                    deterministic=True,
                                )
                        else:
                            # prev：用 last_done 作为参考（便于连续跟踪）
                            im_prev = core._read_frame_bgr(frames[int(last_done)], cfg)
                            if im_prev is not None:
                                roi_prev = core.crop(im_prev, prev_bbox)
                                th_prev, _, _ = core.compute_th_gray_truncated(roi_prev, cfg.th_percent, cfg)
                                prev_uvh_pts = _uvh_points_from_th_roi(
                                    th_gray_truncated=th_prev,
                                    bbox=prev_bbox,
                                    z_scale=float(cfg.mesh_z_scale),
                                    max_points=int(mp),
                                    rng=rng0,
                                    deterministic=True,
                                )
                        if prev_uvh_pts is None or int(np.asarray(prev_uvh_pts).shape[0]) < 200:
                            raise RuntimeError("prev_uvh_pts too few")
                    except Exception as e:
                        print(f"[UVH-Render][Warning] 恢复 prev_uvh_pts 失败，将回退 2D：{e}")
                        enable_uvh_render_track = False
                        track_match_mode = "2d"
                # 恢复 flow/uvh 缓存：用 last_done 的整帧重算 th/mask
                if enable_flow_uvh:
                    try:
                        prev_full_bgr = core._read_frame_bgr(frames[int(last_done)], cfg)
                        if prev_full_bgr is None:
                            prev_full_bgr = first
                        if str(flow_uvh_mask_mode) == "full":
                            prev_full_th, prev_full_mask, _ = core.compute_th_gray_truncated(prev_full_bgr, cfg.th_percent, cfg)
                        else:
                            # 目标化：只在 prev_bbox 内生成 th/mask，再贴回整帧（避免把全图边缘都纳入 uvh 匹配）
                            roi_prev = core.crop(prev_full_bgr, prev_bbox)
                            th_prev_roi, mk_prev_roi, _ = core.compute_th_gray_truncated(roi_prev, cfg.th_percent, cfg)
                            prev_roi_th = np.asarray(th_prev_roi, dtype=np.float32)
                            prev_roi_mask = np.asarray(mk_prev_roi, dtype=np.uint8)
                            HH, WW = prev_full_bgr.shape[:2]
                            prev_full_th = np.zeros((HH, WW), dtype=np.float32)
                            prev_full_mask = np.zeros((HH, WW), dtype=np.uint8)
                            x0, y0, w0, h0 = [int(v) for v in prev_bbox]
                            prev_full_th[y0:y0 + h0, x0:x0 + w0] = np.asarray(prev_roi_th, dtype=np.float32)
                            prev_full_mask[y0:y0 + h0, x0:x0 + w0] = np.asarray(prev_roi_mask, dtype=np.uint8)
                    except Exception:
                        prev_full_bgr = first
                        prev_full_th, prev_full_mask = None, None
            except Exception as e:
                print(f"[恢复] 读取 checkpoint 失败，将从头继续: {e}")
                start_frame = 1
                last_done = 0

    if enable_uvh_render_track and renderfield_ref_ctx is None:
        try:
            renderfield_ref_ctx = _build_renderfield_reference_context(
                ref_points=(
                    np.asarray(uvh_rigid_ref_pts, dtype=np.float32)
                    if isinstance(uvh_rigid_ref_pts, np.ndarray)
                    else np.asarray(prev_uvh_pts, dtype=np.float32)
                ),
                view_lib=(lib if isinstance(lib, ViewLibrary) else None),
                prev_view_id=int(prev_view_id),
                cfg=cfg,
            )
        except Exception:
            renderfield_ref_ctx = None

    if enable_uvh_render_track and (prev_uvh_pts is not None) and int(np.asarray(prev_uvh_pts).shape[0]) >= 200:
        try:
            cam_dbg = _save_uvh_camera_init_debug(
                out_root=out_root,
                cfg=cfg,
                prev_uvh_pts=np.asarray(prev_uvh_pts, dtype=np.float32),
                bbox_ref=tuple(int(v) for v in bbox),
                lib=lib,
                uvh_viewlib_enable=bool(uvh_viewlib_enable),
                prev_view_id=int(prev_view_id),
                renderfield_ref_ctx=renderfield_ref_ctx if isinstance(renderfield_ref_ctx, dict) else None,
            )
            if cam_dbg:
                _log_once(
                    "camera_init_saved",
                    f"[CameraInit] 已写入相机初始化调试目录: {cam_dbg}（设 MIS_SAVE_CAMERA_INIT_DEBUG=0 可关闭）",
                    cfg=cfg,
                    level=1,
                )
        except Exception as e:
            _log_once("camera_init_save_fail", f"[CameraInit] 保存调试文件失败: {e}", cfg=cfg, level=1)

    # 主循环：只做跟踪（不做 RT/ICP/mesh）
    prev_prev_bbox: tuple[int, int, int, int] | None = None
    for i in range(int(start_frame), len(frames)):
        img_path = frames[i]
        frame = core._read_frame_bgr(img_path, cfg)
        if frame is None:
            print(f"[警告] 跳过读取失败帧: {img_path}")
            continue
        uvh_ref_pts_for_debug = None
        uvh_debug_paths: dict[str, str] = {}

        if (i % int(max(1, cfg.print_every))) == 0:
            _log(
                f"[进度] frame={i}/{len(frames)-1} img={os.path.basename(img_path)} mode={str(track_match_mode)} prev_bbox={list(map(int, prev_bbox))}",
                cfg=cfg,
                level=1,
            )

        if enable_uvh_icp_track and prev_mesh is not None:
            best = find_best_bbox_in_next_frame_uvh_icp(
                frame_bgr=frame,
                prev_bbox=prev_bbox,
                prev_mask_small=prev_mask_small,
                prev_mesh=prev_mesh,
                cfg=cfg,
            )
        elif enable_uvh_render_track and (prev_uvh_pts is not None):
            mask_small_for_uvh = ref_mask_small if str(uvh_render_ref_mode) == "first" else prev_mask_small
            try:
                uvh_ref_pts_for_debug = np.asarray(prev_uvh_pts, dtype=np.float32).copy()
            except Exception:
                uvh_ref_pts_for_debug = None
            best = find_best_bbox_in_next_frame_uvh_render(
                frame_bgr=frame,
                prev_bbox=prev_bbox,
                prev_mask_small=mask_small_for_uvh,
                prev_uvh_pts=np.asarray(prev_uvh_pts, dtype=np.float32),
                cfg=cfg,
                view_lib=(lib if (lib is not None and bool(uvh_viewlib_enable)) else None),
                prev_view_id=int(prev_view_id),
                prev_view_dir=prev_view_dir,
                angle_gate_deg=float(uvh_viewlib_angle_gate_deg),
                ref_contour_rgb=(
                    np.asarray(ref_contour_rgb, dtype=np.uint8)
                    if (str(uvh_render_ref_mode) == "first" and isinstance(ref_contour_rgb, np.ndarray))
                    else np.asarray(prev_contour_rgb, dtype=np.uint8)
                ),
                rigid_ref_uvh_pts=(
                    np.asarray(uvh_rigid_ref_pts, dtype=np.float32)
                    if (str(uvh_render_ref_mode) == "first" and isinstance(uvh_rigid_ref_pts, np.ndarray))
                    else np.asarray(prev_uvh_pts, dtype=np.float32)
                ),
                diff_ref_verts=uvh_diff_ref_verts,
                diff_ref_faces=uvh_diff_ref_faces,
                renderfield_ref_ctx=renderfield_ref_ctx,
                warm_prev_bbox=prev_prev_bbox,
            )
        elif enable_view_lib and lib is not None:
            best = find_best_bbox_in_next_frame_with_library(
                frame_bgr=frame,
                prev_bbox=prev_bbox,
                ref_mask_small=ref_mask_small,
                prev_view_dir=prev_view_dir,
                cfg=cfg,
                lib=lib,
                angle_gate_deg=float(lib_angle_gate_deg),
            )
        else:
            best = core.find_best_bbox_in_next_frame(frame, prev_bbox, prev_contour_rgb, prev_mask_small, cfg)
        bbox_i = best["bbox"]
        th_i = best["th_gray"]
        mask_i = best["mask_u8"]
        contour_i = best["contour_rgb"]

        # 控制台：打印每 N 帧的关键结果（可用 MIS_PRINT_EVERY/MIS_VERBOSE 控制频率与细节）
        if (i % int(max(1, cfg.print_every))) == 0:
            try:
                if enable_uvh_render_track:
                    _log(
                        (
                            f"[结果] frame={i} bbox={list(map(int, bbox_i))} score={float(best.get('score', 0.0)):.3f} "
                            f"dec={float(best.get('uvh_decision_score', float('nan'))):.3f} "
                            f"tiou={float(best.get('uvh_decision_temporal_iou', float('nan'))):.3f} "
                            f"tview={float(best.get('uvh_decision_temporal_view', float('nan'))):.3f} "
                            f"uvh_sim={float(best.get('uvh_render_sim', float('nan'))):.3f} "
                            f"mesh={float(best.get('uvh_mesh_sim', float('nan'))):.3f} "
                            f"sem={float(best.get('uvh_sem_sim', float('nan'))):.3f} "
                            f"rigid={float(best.get('uvh_rigid_score', float('nan'))):.3f} "
                            f"inlier={float(best.get('uvh_rigid_inlier_ratio', float('nan'))):.3f} "
                            f"nrm={float(best.get('uvh_rigid_normal_consistency', float('nan'))):.3f} "
                            f"cov={float(best.get('uvh_coverage_ratio', float('nan'))):.3f} "
                            f"post={int(best.get('uvh_post_bbox_used', 0))} "
                            f"corrN={int(best.get('uvh_rigid_corr_count', 0))} "
                            f"local={float(best.get('uvh_local_bbox_score', float('nan'))):.3f} "
                            f"stage={str(best.get('uvh_decision_stage', 'coarse'))} "
                            f"area_ratio={float(best.get('uvh_area_ratio', float('nan'))):.3f} "
                            f"bbox_scale={float(best.get('uvh_bbox_scale', float('nan'))):.3f} "
                            f"view={int(best.get('uvh_view_id', -1))} rf_used={int(best.get('uvh_rf_used', 0))} "
                            f"rf_loss={float(best.get('uvh_rf_best_loss', float('nan'))):.4f} diff_used={int(best.get('uvh_diff_used', 0))} "
                            f"diff_called={int(best.get('uvh_diff_called', 0))} diff_ok={int(best.get('uvh_diff_ok', 0))}"
                        ),
                        cfg=cfg,
                        level=1,
                    )
                    if int(best.get("uvh_diff_used", 0)) == 1:
                        _log(
                            (
                                f"[UVH-Diff] sim={float(best.get('uvh_diff_sim', float('nan'))):.3f} "
                                f"yaw={float(best.get('uvh_diff_yaw_deg', float('nan'))):.1f} "
                                f"pitch={float(best.get('uvh_diff_pitch_deg', float('nan'))):.1f} "
                                f"roll={float(best.get('uvh_diff_roll_deg', float('nan'))):.1f} "
                                f"dist={float(best.get('uvh_diff_dist', float('nan'))):.2f} backend={str(best.get('uvh_diff_backend', ''))}"
                            ),
                            cfg=cfg,
                            level=2,
                        )
                elif enable_view_lib and lib is not None:
                    _log(
                        (
                            f"[结果] frame={i} bbox={list(map(int, bbox_i))} score={float(best.get('score', 0.0)):.3f} "
                            f"lib_view={int(best.get('lib_view_id', -1))} lib_score={float(best.get('semantic', float('nan'))):.3f} "
                            f"quick={float(best.get('quick', float('nan'))):.3f}"
                        ),
                        cfg=cfg,
                        level=1,
                    )
                else:
                    _log(
                        (
                            f"[结果] frame={i} bbox={list(map(int, bbox_i))} score={float(best.get('score', 0.0)):.3f} "
                            f"semantic={float(best.get('semantic', float('nan'))):.3f} quick={float(best.get('quick', float('nan'))):.3f}"
                        ),
                        cfg=cfg,
                        level=1,
                    )
            except Exception:
                pass
        # lib 模式：粗筛锚定首帧 reference，避免 prev_mask_small 逐帧更新带来的漂移累积
        if enable_view_lib and lib is not None:
            prev_mask_small = np.asarray(ref_mask_small, dtype=np.uint8).copy()
        else:
            prev_mask_small = _mask64_from_mask_u8(mask_i)

        # uvh-ICP：更新上一帧 mesh（用于下一帧）
        if enable_uvh_icp_track:
            try:
                prev_mesh = core.make_heightfield_mesh_from_th_gray(th_i, bbox_i, z_scale=float(cfg.mesh_z_scale))
                if prev_mesh is None:
                    raise RuntimeError("prev_mesh=None")
            except Exception:
                # 不阻断：保留旧 prev_mesh
                pass

        # uvh-render：更新上一帧 uvh 点云（用于下一帧）
        if enable_uvh_render_track and str(uvh_render_ref_mode) == "prev":
            try:
                pts_new = best.get("uvh_render_pts", None)
                if isinstance(pts_new, np.ndarray) and int(pts_new.shape[0]) >= 200:
                    prev_uvh_pts = np.asarray(pts_new, dtype=np.float32)
            except Exception:
                pass

        # view-library：更新“上一视点方向”（用于下一帧的角度门控加速）
        if lib is not None and (enable_view_lib or (enable_uvh_render_track and bool(uvh_viewlib_enable))):
            try:
                if enable_view_lib:
                    vid = int(best.get("lib_view_id", -1))
                    vscore = float(best.get("semantic", float("nan")))
                    vmin = float(_env_float("MIS_LIB_UPDATE_VIEW_MIN", 0.55))
                else:
                    # uvh_render + viewlib：用 mesh 相似度更稳（coarse view_score 可能不够鲁棒）
                    vid = int(best.get("uvh_view_id", -1))
                    vscore = float(best.get("uvh_render_sim", float("nan")))
                    vmin = float(_env_float("MIS_UVH_VIEWLIB_UPDATE_MIN", 0.70))

                # 只在“分数足够高”时更新 gate，避免一次误匹配把后续模板门控带偏
                if (0 <= vid < int(lib.ids.size)) and (np.isfinite(vscore) and float(vscore) >= float(vmin)):
                    prev_view_id = int(vid)
                    prev_view_dir = np.asarray(lib.dirs[int(vid)], dtype=np.float32).reshape(3)
                    try:
                        rr = core._load_contour_rgb(lib.contour_rgb_paths[int(vid)])
                    except Exception:
                        rr = None
                    if rr is not None:
                        ref_contour_rgb = rr
            except Exception:
                pass

        # ===== 光流 -> uvh 匹配 -> RANSAC/IRLS（诊断/中间结果保存；不参与 bbox 选择）=====
        flow_info = {
            "flow_uvh_enabled": int(enable_flow_uvh),
            "flow_used": 0,
            "flow_n_matches": 0,
            "flow_n_inliers": 0,
            "flow_inlier_ratio": float("nan"),
            "flow_rmse": float("nan"),  # xy 平面（px）
            "flow_T_prev_cur": None,  # 4x4
        }
        if enable_flow_uvh:
            try:
                # 当前帧整帧 th/mask（或“只在 bbox 内贴回整帧”，取决于 flow_uvh_mask_mode）
                if str(flow_uvh_mask_mode) == "full":
                    cur_full_th, cur_full_mask, _ = core.compute_th_gray_truncated(frame, cfg.th_percent, cfg)
                else:
                    HH, WW = frame.shape[:2]
                    cur_full_th = np.zeros((HH, WW), dtype=np.float32)
                    cur_full_mask = np.zeros((HH, WW), dtype=np.uint8)
                    x0, y0, w0, h0 = [int(v) for v in bbox_i]
                    cur_full_th[y0:y0 + h0, x0:x0 + w0] = np.asarray(th_i, dtype=np.float32)
                    cur_full_mask[y0:y0 + h0, x0:x0 + w0] = np.asarray(mask_i, dtype=np.uint8)
                if prev_full_th is not None and prev_full_mask is not None:
                    cur_uvh, prev_uvh, _, flow_dbg = core._build_uvh_matches_from_flow(
                        prev_bgr=prev_full_bgr,
                        cur_bgr=frame,
                        prev_th=prev_full_th,
                        cur_th=cur_full_th,
                        prev_mask=prev_full_mask,
                        cfg=cfg,
                    )
                    flow_info["flow_used"] = 1
                    flow_info["flow_n_matches"] = int(cur_uvh.shape[0])

                    T_prev_cur = None
                    inl = None
                    if int(cur_uvh.shape[0]) >= 20:
                        T0, inl = core._ransac_rigid_T_xy(cur_uvh=cur_uvh, prev_uvh=prev_uvh, cfg=cfg)
                        if T0 is not None:
                            flow_info["flow_n_inliers"] = int(np.count_nonzero(inl)) if inl is not None else 0
                            flow_info["flow_inlier_ratio"] = float(flow_info["flow_n_inliers"]) / float(max(1, flow_info["flow_n_matches"]))
                            T_prev_cur = core._irls_refine_T(T_init=T0, cur_uvh=cur_uvh, prev_uvh=prev_uvh, cfg=cfg, inlier_mask=inl)
                            flow_info["flow_T_prev_cur"] = np.asarray(T_prev_cur, dtype=np.float64).reshape(4, 4)
                            # 计算 inlier xy 残差（用 median 作为鲁棒 rmse proxy，和原脚本一致）
                            try:
                                Xh = (T_prev_cur[:3, :3] @ cur_uvh.T).T + T_prev_cur[:3, 3][None, :]
                                e = np.linalg.norm((Xh[:, :2] - prev_uvh[:, :2]), axis=1)
                                if inl is not None:
                                    e_in = e[np.asarray(inl, dtype=bool)]
                                else:
                                    e_in = e
                                flow_info["flow_rmse"] = float(np.median(e_in)) if e_in.size > 0 else float("nan")
                            except Exception:
                                pass

                    # 保存 uvh matches（匹配结果）
                    try:
                        out_npz = os.path.join(out_pairs_matches, f"uvh_matches_prev_{i-1:04d}_cur_{i:04d}.npz")
                        np.savez_compressed(
                            out_npz,
                            prev_uvh=prev_uvh.astype(np.float32),
                            cur_uvh=cur_uvh.astype(np.float32),
                            inliers=(inl.astype(np.uint8) if inl is not None else None),
                            T_prev_cur=(flow_info["flow_T_prev_cur"] if flow_info["flow_T_prev_cur"] is not None else None),
                        )
                        flow_info["uvh_npz"] = out_npz
                    except Exception:
                        pass

                    # 保存光流原始/可视化（中间结果）
                    try:
                        if bool(getattr(cfg, "flow_save_raw", True)):
                            st = int(max(1, getattr(cfg, "flow_save_raw_stride", 2)))
                            flow = np.asarray(flow_dbg.get("flow", None))
                            if flow is not None and isinstance(flow, np.ndarray) and flow.ndim == 3 and flow.shape[2] == 2:
                                flow_ds = flow[::st, ::st, :].astype(np.float16)
                                out_f = os.path.join(out_pairs_flow, f"flow_prev_{i-1:04d}_cur_{i:04d}.npz")
                                np.savez_compressed(
                                    out_f,
                                    flow=flow_ds,
                                    stride=st,
                                    prev_xy=np.asarray(flow_dbg.get("prev_xy", np.zeros((0, 2), np.float32)), dtype=np.float32),
                                    cur_xy=np.asarray(flow_dbg.get("cur_xy", np.zeros((0, 2), np.float32)), dtype=np.float32),
                                )
                                flow_info["flow_npz"] = out_f
                    except Exception:
                        pass
                    try:
                        if bool(getattr(cfg, "flow_save_hsv", True)):
                            flow = np.asarray(flow_dbg.get("flow", None))
                            if flow is not None and isinstance(flow, np.ndarray) and flow.ndim == 3 and flow.shape[2] == 2:
                                hsv_bgr = core._flow_to_hsv_bgr(flow, clip_mag=20.0)
                                out_png = os.path.join(out_pairs_flow_hsv, f"flow_hsv_prev_{i-1:04d}_cur_{i:04d}.png")
                                cv2.imwrite(out_png, hsv_bgr)
                                flow_info["flow_hsv_png"] = out_png
                    except Exception:
                        pass
                    try:
                        if bool(getattr(cfg, "flow_save_vis", True)) and (flow_info["flow_T_prev_cur"] is not None):
                            out_png = os.path.join(out_pairs_matches_vis, f"matches_prev_{i-1:04d}_cur_{i:04d}.png")
                            core._save_flow_matches_vis(
                                out_png,
                                prev_bgr=prev_full_bgr,
                                prev_uvh=prev_uvh,
                                cur_uvh=cur_uvh,
                                T_prev_cur=np.asarray(flow_info["flow_T_prev_cur"], dtype=np.float64).reshape(4, 4),
                                inliers=inl,
                                max_lines=int(getattr(cfg, "flow_vis_max_lines", 800)),
                            )
                            flow_info["uvh_vis_png"] = out_png
                    except Exception:
                        pass

                    # 打印关键诊断信息（可确认“起作用”）
                    if (i % int(max(1, cfg.print_every))) == 0:
                        n_m = int(flow_info["flow_n_matches"])
                        n_in = int(flow_info["flow_n_inliers"])
                        ratio = float(n_in) / float(max(1, n_m))
                        rmse = flow_info["flow_rmse"]
                        print(f"[Flow/UVH] matches={n_m} inliers={n_in} ratio={ratio:.3f} rmse={rmse}")

                # 更新缓存（保证下一帧可用）
                prev_full_bgr = frame
                prev_full_th = cur_full_th
                prev_full_mask = cur_full_mask
            except Exception:
                # 异常不阻断跟踪：只跳过该帧对的 flow/uvh 保存
                try:
                    prev_full_bgr = frame
                except Exception:
                    pass

        if save_uvh_render_debug and (enable_uvh_icp_track or enable_uvh_render_track):
            try:
                uvh_debug_paths = _save_uvh_render_debug_bundle(
                    out_uvh_render=out_uvh_render,
                    frame_idx=int(i),
                    image_path=str(img_path),
                    frame_bgr=frame,
                    prev_bbox=tuple(prev_bbox),
                    best=best,
                    ref_mode=str(uvh_render_ref_mode),
                    ref_pts_xyz=(np.asarray(uvh_ref_pts_for_debug, dtype=np.float32) if isinstance(uvh_ref_pts_for_debug, np.ndarray) else None),
                )
            except Exception:
                uvh_debug_paths = {}

        # 保存跟踪与中间结果
        if cfg.save_debug_images:
            roi_i = core.crop(frame, bbox_i)
            if enable_uvh_icp_track or enable_uvh_render_track:
                icp_fit = float(best.get("uvh_icp_fitness", float("nan")))
                rnd_sim = float(best.get("uvh_render_sim", float("nan")))
                txt = f"f={i} score={best['score']:.3f} icp={icp_fit:.3f} rnd={rnd_sim:.3f} q={best['quick']:.3f}"
                # 优先用“渲染对比图”做可视化
                rprev = best.get("uvh_render_prev_rgb", None)
                rcur = best.get("uvh_render_cur_rgb", None)
                if isinstance(rprev, np.ndarray) and isinstance(rcur, np.ndarray):
                    ref_left = rprev
                    cmp_right = rcur
                    sim_show = rnd_sim if np.isfinite(rnd_sim) else float(best.get("semantic", 0.0))
                    title_left = "prev(render)"
                    title_right = "cur(render_aligned)"
                    try:
                        if out_uvh_render:
                            cv2.imwrite(os.path.join(out_uvh_render, f"prev_render_{i-1:04d}.png"), rprev)
                            cv2.imwrite(os.path.join(out_uvh_render, f"cur_render_{i:04d}.png"), rcur)
                    except Exception:
                        pass
                else:
                    ref_left = prev_contour_rgb
                    cmp_right = contour_i
                    sim_show = float(best.get("semantic", 0.0))
                    title_left = "prev(contour)"
                    title_right = "cur(contour)"
            elif enable_view_lib and lib is not None:
                vid = int(best.get("lib_view_id", -1))
                txt = f"f={i} score={best['score']:.3f} lib={vid} sem={best['semantic']:.3f} q={best['quick']:.3f}"
                ref_left = ref_contour_rgb
                cmp_right = contour_i
                sim_show = float(best.get("semantic", 0.0))
                title_left = f"lib(view={vid})"
                title_right = "cur(contour)"
            else:
                txt = f"f={i} score={best['score']:.3f} sem={best['semantic']:.3f} q={best['quick']:.3f}"
                ref_left = prev_contour_rgb
                cmp_right = contour_i
                sim_show = float(best.get("semantic", 0.0))
                title_left = "prev(contour)"
                title_right = "cur(contour)"
            cv2.imwrite(os.path.join(out_frames, f"frame_{i:04d}.jpg"), core.draw_bbox(frame, bbox_i, text=txt))
            cv2.imwrite(os.path.join(out_roi, f"roi_{i:04d}.jpg"), roi_i)
            cv2.imwrite(os.path.join(out_mask, f"mask_{i:04d}.png"), mask_i)
            cv2.imwrite(os.path.join(out_th, f"th_gray_{i:04d}.png"), np.clip(th_i, 0, 255).astype(np.uint8))
            cv2.imwrite(os.path.join(out_cont, f"contour_rgb_{i:04d}.png"), contour_i)
            cv2.imwrite(os.path.join(out_vis, f"roi_overlay_{i:04d}.jpg"), core.make_overlay_mask(roi_i, mask_i))
            core._save_similarity_compare_image(
                os.path.join(out_cmp, f"compare_{i:04d}.png"),
                prev_contour_rgb=ref_left,
                cur_contour_rgb=cmp_right,
                similarity=float(sim_show),
                title_left=title_left,
                title_right=title_right,
            )

        track_rows.append(
            {
                "frame": int(i),
                "image": img_path,
                "bbox": list(map(int, bbox_i)),
                "th_uint8": float(best.get("th_uint8", float("nan"))),
                "score": float(best["score"]),
                "semantic": float(best["semantic"]),
                "quick": float(best["quick"]),
                "dx": int(best["dx"]),
                "dy": int(best["dy"]),
                "uvh_icp_fitness": float(best.get("uvh_icp_fitness", float("nan"))) if enable_uvh_icp_track else float("nan"),
                "uvh_icp_rmse": float(best.get("uvh_icp_rmse", float("nan"))) if enable_uvh_icp_track else float("nan"),
                "uvh_render_sim": float(best.get("uvh_render_sim", float("nan"))) if (enable_uvh_icp_track or enable_uvh_render_track) else float("nan"),
                "uvh_render_used": int(best.get("uvh_render_used", 0)) if (enable_uvh_icp_track or enable_uvh_render_track) else 0,
                "uvh_render_meta_json": str(uvh_debug_paths.get("meta_json", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_ref_render_png": str(uvh_debug_paths.get("ref_render_png", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_cur_render_png": str(uvh_debug_paths.get("cur_render_png", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_compare_png": str(uvh_debug_paths.get("render_compare_png", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_ref_pts_ply": str(uvh_debug_paths.get("ref_pts_ply", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_cur_pts_ply": str(uvh_debug_paths.get("cur_pts_ply", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_projected_ref_pts_ply": str(uvh_debug_paths.get("projected_ref_pts_ply", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_point_match_ply": str(uvh_debug_paths.get("point_match_ply", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_point_overlay_png": str(uvh_debug_paths.get("point_overlay_png", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_decision_score": float(best.get("uvh_decision_score", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_decision_motion_pen": float(best.get("uvh_decision_motion_pen", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_decision_coverage": float(best.get("uvh_decision_coverage", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_decision_coverage_gap": float(best.get("uvh_decision_coverage_gap", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_decision_temporal_iou": float(best.get("uvh_decision_temporal_iou", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_decision_temporal_view": float(best.get("uvh_decision_temporal_view", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_align_scale": float(best.get("uvh_align_scale", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_decision_stage": str(best.get("uvh_decision_stage", "")) if (enable_uvh_render_track) else "",
                "uvh_coverage_ratio": float(best.get("uvh_coverage_ratio", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_coverage_gap": float(best.get("uvh_coverage_gap", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_align_rot_deg": float(best.get("uvh_align_rot_deg", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_align_tx": float(best.get("uvh_align_tx", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_align_ty": float(best.get("uvh_align_ty", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_mesh_sim": float(best.get("uvh_mesh_sim", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_mesh_iou": float(best.get("uvh_mesh_iou", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_mesh_depthcorr": float(best.get("uvh_mesh_depthcorr", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_sem_sim": float(best.get("uvh_sem_sim", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_sem_used": int(best.get("uvh_sem_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rigid_used": int(best.get("uvh_rigid_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rigid_method": str(best.get("uvh_rigid_method", "")) if (enable_uvh_render_track) else "",
                "uvh_rigid_score": float(best.get("uvh_rigid_score", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_fitness": float(best.get("uvh_rigid_fitness", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_inlier_ratio": float(best.get("uvh_rigid_inlier_ratio", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_rmse": float(best.get("uvh_rigid_rmse", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_overlap": float(best.get("uvh_rigid_overlap", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_normal_consistency": float(best.get("uvh_rigid_normal_consistency", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_corr_count": int(best.get("uvh_rigid_corr_count", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rigid_lowcorr_pen": float(best.get("uvh_rigid_lowcorr_pen", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_median_dist": float(best.get("uvh_rigid_median_dist", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_mean_dist": float(best.get("uvh_rigid_mean_dist", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_voxel": float(best.get("uvh_rigid_voxel", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_bbox_used": int(best.get("uvh_rigid_bbox_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rigid_bbox_x": int(best.get("uvh_rigid_bbox_x", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rigid_bbox_y": int(best.get("uvh_rigid_bbox_y", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rigid_bbox_w": int(best.get("uvh_rigid_bbox_w", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rigid_bbox_h": int(best.get("uvh_rigid_bbox_h", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rf_attempted": int(best.get("uvh_rf_attempted", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rf_used": int(best.get("uvh_rf_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rf_best_loss": float(best.get("uvh_rf_best_loss", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rf_steps": int(best.get("uvh_rf_steps", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rf_lr": float(best.get("uvh_rf_lr", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rf_bbox_used": int(best.get("uvh_rf_bbox_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rf_bbox_x": int(best.get("uvh_rf_bbox_x", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rf_bbox_y": int(best.get("uvh_rf_bbox_y", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rf_bbox_w": int(best.get("uvh_rf_bbox_w", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rf_bbox_h": int(best.get("uvh_rf_bbox_h", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rf_error": str(best.get("uvh_rf_error", "")) if (enable_uvh_render_track) else "",
                "uvh_local_bbox_used": int(best.get("uvh_local_bbox_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_local_bbox_score": float(best.get("uvh_local_bbox_score", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_local_quick": float(best.get("uvh_local_quick", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_local_sem": float(best.get("uvh_local_sem", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_local_bbox_x": int(best.get("uvh_local_bbox_x", -1)) if (enable_uvh_render_track) else -1,
                "uvh_local_bbox_y": int(best.get("uvh_local_bbox_y", -1)) if (enable_uvh_render_track) else -1,
                "uvh_local_bbox_w": int(best.get("uvh_local_bbox_w", -1)) if (enable_uvh_render_track) else -1,
                "uvh_local_bbox_h": int(best.get("uvh_local_bbox_h", -1)) if (enable_uvh_render_track) else -1,
                "uvh_post_bbox_used": int(best.get("uvh_post_bbox_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_post_bbox_score": float(best.get("uvh_post_bbox_score", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_post_bbox_source": str(best.get("uvh_post_bbox_source", "")) if (enable_uvh_render_track) else "",
                "uvh_post_bbox_touch": float(best.get("uvh_post_bbox_touch", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_post_bbox_area_ratio": float(best.get("uvh_post_bbox_area_ratio", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_area_ratio": float(best.get("uvh_area_ratio", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_area_pen": float(best.get("uvh_area_pen", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_scale_pen": float(best.get("uvh_scale_pen", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_bbox_scale": float(best.get("uvh_bbox_scale", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_bbox_scale_pen": float(best.get("uvh_bbox_scale_pen", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_view_id": int(best.get("uvh_view_id", -1)) if (enable_uvh_render_track) else -1,
                "uvh_view_score": float(best.get("uvh_view_score", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_view_used_gate": int(best.get("uvh_view_used_gate", 0)) if (enable_uvh_render_track) else 0,
                "uvh_view_n_used": int(best.get("uvh_view_n_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_view_dist": float(best.get("uvh_view_dist", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_view_roll": float(best.get("uvh_view_roll", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_view_dir": best.get("uvh_view_dir", None) if (enable_uvh_render_track) else None,
                "uvh_diff_used": int(best.get("uvh_diff_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_diff_sim": float(best.get("uvh_diff_sim", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_iou": float(best.get("uvh_diff_iou", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_depthcorr": float(best.get("uvh_diff_depthcorr", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_sem": float(best.get("uvh_diff_sem", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_sem_used": int(best.get("uvh_diff_sem_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_diff_backend": str(best.get("uvh_diff_backend", "")) if (enable_uvh_render_track) else "",
                "uvh_diff_iters": int(best.get("uvh_diff_iters", 0)) if (enable_uvh_render_track) else 0,
                "uvh_diff_lr": float(best.get("uvh_diff_lr", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_yaw_deg": float(best.get("uvh_diff_yaw_deg", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_pitch_deg": float(best.get("uvh_diff_pitch_deg", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_roll_deg": float(best.get("uvh_diff_roll_deg", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_dist": float(best.get("uvh_diff_dist", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_scale": float(best.get("uvh_diff_scale", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_rot_deg": float(best.get("uvh_diff_rot_deg", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_tx": float(best.get("uvh_diff_tx", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_ty": float(best.get("uvh_diff_ty", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_area_ratio": float(best.get("uvh_diff_area_ratio", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "lib_view_id": int(best.get("lib_view_id", -1)) if (enable_view_lib and lib is not None) else -1,
                "lib_score": float(best.get("semantic", float("nan"))) if (enable_view_lib and lib is not None) else float("nan"),
                "lib_iou": float(best.get("lib_iou", float("nan"))) if (enable_view_lib and lib is not None) else float("nan"),
                "lib_corr": float(best.get("lib_corr", float("nan"))) if (enable_view_lib and lib is not None) else float("nan"),
                "lib_n_used": int(best.get("lib_n_used", 0)) if (enable_view_lib and lib is not None) else 0,
                "lib_used_gate": int(best.get("lib_used_gate", 0)) if (enable_view_lib and lib is not None) else 0,
                "lib_dir": best.get("lib_dir", None) if (enable_view_lib and lib is not None) else None,
                "lib_dist": float(best.get("lib_dist", float("nan"))) if (enable_view_lib and lib is not None) else float("nan"),
                "lib_roll": float(best.get("lib_roll", float("nan"))) if (enable_view_lib and lib is not None) else float("nan"),
                "lib_template_contour": str(best.get("lib_template_contour", "")) if (enable_view_lib and lib is not None) else "",
            }
        )

        diag_rows.append(
            {
                "frame_idx": int(i),
                "img": str(img_path),
                "bbox_x": int(bbox_i[0]),
                "bbox_y": int(bbox_i[1]),
                "bbox_w": int(bbox_i[2]),
                "bbox_h": int(bbox_i[3]),
                "score": float(best["score"]),
                "semantic": float(best["semantic"]),
                "quick": float(best["quick"]),
                "dx": int(best["dx"]),
                "dy": int(best["dy"]),
                "th_uint8": float(best.get("th_uint8", float("nan"))),
                "track_match_mode": str(track_match_mode),
                "uvh_icp_used": int(best.get("uvh_icp_used", 0)) if enable_uvh_icp_track else 0,
                "uvh_icp_fitness": float(best.get("uvh_icp_fitness", float("nan"))) if enable_uvh_icp_track else float("nan"),
                "uvh_icp_rmse": float(best.get("uvh_icp_rmse", float("nan"))) if enable_uvh_icp_track else float("nan"),
                "uvh_render_used": int(best.get("uvh_render_used", 0)) if (enable_uvh_icp_track or enable_uvh_render_track) else 0,
                "uvh_render_sim": float(best.get("uvh_render_sim", float("nan"))) if (enable_uvh_icp_track or enable_uvh_render_track) else float("nan"),
                "uvh_render_meta_json": str(uvh_debug_paths.get("meta_json", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_ref_render_png": str(uvh_debug_paths.get("ref_render_png", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_cur_render_png": str(uvh_debug_paths.get("cur_render_png", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_compare_png": str(uvh_debug_paths.get("render_compare_png", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_ref_pts_ply": str(uvh_debug_paths.get("ref_pts_ply", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_cur_pts_ply": str(uvh_debug_paths.get("cur_pts_ply", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_projected_ref_pts_ply": str(uvh_debug_paths.get("projected_ref_pts_ply", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_point_match_ply": str(uvh_debug_paths.get("point_match_ply", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_render_point_overlay_png": str(uvh_debug_paths.get("point_overlay_png", "")) if (enable_uvh_icp_track or enable_uvh_render_track) else "",
                "uvh_decision_score": float(best.get("uvh_decision_score", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_decision_motion_pen": float(best.get("uvh_decision_motion_pen", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_decision_coverage": float(best.get("uvh_decision_coverage", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_decision_coverage_gap": float(best.get("uvh_decision_coverage_gap", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_decision_temporal_iou": float(best.get("uvh_decision_temporal_iou", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_decision_temporal_view": float(best.get("uvh_decision_temporal_view", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_decision_stage": str(best.get("uvh_decision_stage", "")) if (enable_uvh_render_track) else "",
                "uvh_coverage_ratio": float(best.get("uvh_coverage_ratio", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_coverage_gap": float(best.get("uvh_coverage_gap", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_align_scale": float(best.get("uvh_align_scale", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_align_rot_deg": float(best.get("uvh_align_rot_deg", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_align_tx": float(best.get("uvh_align_tx", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_align_ty": float(best.get("uvh_align_ty", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_mesh_sim": float(best.get("uvh_mesh_sim", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_mesh_iou": float(best.get("uvh_mesh_iou", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_mesh_depthcorr": float(best.get("uvh_mesh_depthcorr", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_sem_sim": float(best.get("uvh_sem_sim", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_sem_used": int(best.get("uvh_sem_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rigid_used": int(best.get("uvh_rigid_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rigid_method": str(best.get("uvh_rigid_method", "")) if (enable_uvh_render_track) else "",
                "uvh_rigid_score": float(best.get("uvh_rigid_score", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_fitness": float(best.get("uvh_rigid_fitness", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_inlier_ratio": float(best.get("uvh_rigid_inlier_ratio", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_rmse": float(best.get("uvh_rigid_rmse", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_overlap": float(best.get("uvh_rigid_overlap", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_normal_consistency": float(best.get("uvh_rigid_normal_consistency", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_corr_count": int(best.get("uvh_rigid_corr_count", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rigid_lowcorr_pen": float(best.get("uvh_rigid_lowcorr_pen", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_median_dist": float(best.get("uvh_rigid_median_dist", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_mean_dist": float(best.get("uvh_rigid_mean_dist", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_voxel": float(best.get("uvh_rigid_voxel", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rigid_bbox_used": int(best.get("uvh_rigid_bbox_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rigid_bbox_x": int(best.get("uvh_rigid_bbox_x", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rigid_bbox_y": int(best.get("uvh_rigid_bbox_y", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rigid_bbox_w": int(best.get("uvh_rigid_bbox_w", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rigid_bbox_h": int(best.get("uvh_rigid_bbox_h", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rf_attempted": int(best.get("uvh_rf_attempted", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rf_used": int(best.get("uvh_rf_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rf_best_loss": float(best.get("uvh_rf_best_loss", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rf_steps": int(best.get("uvh_rf_steps", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rf_lr": float(best.get("uvh_rf_lr", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_rf_bbox_used": int(best.get("uvh_rf_bbox_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_rf_bbox_x": int(best.get("uvh_rf_bbox_x", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rf_bbox_y": int(best.get("uvh_rf_bbox_y", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rf_bbox_w": int(best.get("uvh_rf_bbox_w", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rf_bbox_h": int(best.get("uvh_rf_bbox_h", -1)) if (enable_uvh_render_track) else -1,
                "uvh_rf_error": str(best.get("uvh_rf_error", "")) if (enable_uvh_render_track) else "",
                "uvh_local_bbox_used": int(best.get("uvh_local_bbox_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_local_bbox_score": float(best.get("uvh_local_bbox_score", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_local_quick": float(best.get("uvh_local_quick", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_local_sem": float(best.get("uvh_local_sem", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_local_bbox_x": int(best.get("uvh_local_bbox_x", -1)) if (enable_uvh_render_track) else -1,
                "uvh_local_bbox_y": int(best.get("uvh_local_bbox_y", -1)) if (enable_uvh_render_track) else -1,
                "uvh_local_bbox_w": int(best.get("uvh_local_bbox_w", -1)) if (enable_uvh_render_track) else -1,
                "uvh_local_bbox_h": int(best.get("uvh_local_bbox_h", -1)) if (enable_uvh_render_track) else -1,
                "uvh_post_bbox_used": int(best.get("uvh_post_bbox_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_post_bbox_score": float(best.get("uvh_post_bbox_score", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_post_bbox_source": str(best.get("uvh_post_bbox_source", "")) if (enable_uvh_render_track) else "",
                "uvh_post_bbox_touch": float(best.get("uvh_post_bbox_touch", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_post_bbox_area_ratio": float(best.get("uvh_post_bbox_area_ratio", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_area_ratio": float(best.get("uvh_area_ratio", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_area_pen": float(best.get("uvh_area_pen", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_scale_pen": float(best.get("uvh_scale_pen", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_bbox_scale": float(best.get("uvh_bbox_scale", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_bbox_scale_pen": float(best.get("uvh_bbox_scale_pen", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_viewlib_enabled": int(enable_uvh_render_track and bool(uvh_viewlib_enable) and (lib is not None)),
                "uvh_view_id": int(best.get("uvh_view_id", -1)) if (enable_uvh_render_track) else -1,
                "uvh_view_score": float(best.get("uvh_view_score", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_view_used_gate": int(best.get("uvh_view_used_gate", 0)) if (enable_uvh_render_track) else 0,
                "uvh_view_n_used": int(best.get("uvh_view_n_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_view_dist": float(best.get("uvh_view_dist", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_view_roll": float(best.get("uvh_view_roll", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_view_dir_x": (float(best.get("uvh_view_dir", [float('nan'), float('nan'), float('nan')])[0]) if isinstance(best.get("uvh_view_dir", None), list) and len(best.get("uvh_view_dir", [])) >= 3 else float("nan")) if (enable_uvh_render_track) else float("nan"),
                "uvh_view_dir_y": (float(best.get("uvh_view_dir", [float('nan'), float('nan'), float('nan')])[1]) if isinstance(best.get("uvh_view_dir", None), list) and len(best.get("uvh_view_dir", [])) >= 3 else float("nan")) if (enable_uvh_render_track) else float("nan"),
                "uvh_view_dir_z": (float(best.get("uvh_view_dir", [float('nan'), float('nan'), float('nan')])[2]) if isinstance(best.get("uvh_view_dir", None), list) and len(best.get("uvh_view_dir", [])) >= 3 else float("nan")) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_used": int(best.get("uvh_diff_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_diff_sim": float(best.get("uvh_diff_sim", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_iou": float(best.get("uvh_diff_iou", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_depthcorr": float(best.get("uvh_diff_depthcorr", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_sem": float(best.get("uvh_diff_sem", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_sem_used": int(best.get("uvh_diff_sem_used", 0)) if (enable_uvh_render_track) else 0,
                "uvh_diff_backend": str(best.get("uvh_diff_backend", "")) if (enable_uvh_render_track) else "",
                "uvh_diff_iters": int(best.get("uvh_diff_iters", 0)) if (enable_uvh_render_track) else 0,
                "uvh_diff_lr": float(best.get("uvh_diff_lr", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_yaw_deg": float(best.get("uvh_diff_yaw_deg", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_pitch_deg": float(best.get("uvh_diff_pitch_deg", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_roll_deg": float(best.get("uvh_diff_roll_deg", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_dist": float(best.get("uvh_diff_dist", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_scale": float(best.get("uvh_diff_scale", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_rot_deg": float(best.get("uvh_diff_rot_deg", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_tx": float(best.get("uvh_diff_tx", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_ty": float(best.get("uvh_diff_ty", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "uvh_diff_area_ratio": float(best.get("uvh_diff_area_ratio", float("nan"))) if (enable_uvh_render_track) else float("nan"),
                "lib_enabled": int(enable_view_lib and (lib is not None)),
                "lib_view_id": int(best.get("lib_view_id", -1)) if (enable_view_lib and lib is not None) else -1,
                "lib_score": float(best.get("semantic", float("nan"))) if (enable_view_lib and lib is not None) else float("nan"),
                "lib_iou": float(best.get("lib_iou", float("nan"))) if (enable_view_lib and lib is not None) else float("nan"),
                "lib_corr": float(best.get("lib_corr", float("nan"))) if (enable_view_lib and lib is not None) else float("nan"),
                "lib_n_used": int(best.get("lib_n_used", 0)) if (enable_view_lib and lib is not None) else 0,
                "lib_used_gate": int(best.get("lib_used_gate", 0)) if (enable_view_lib and lib is not None) else 0,
                "lib_dist": float(best.get("lib_dist", float("nan"))) if (enable_view_lib and lib is not None) else float("nan"),
                "lib_roll": float(best.get("lib_roll", float("nan"))) if (enable_view_lib and lib is not None) else float("nan"),
                "lib_dir_x": (float(best.get("lib_dir", [float('nan'), float('nan'), float('nan')])[0]) if isinstance(best.get("lib_dir", None), list) and len(best.get("lib_dir", [])) >= 3 else float("nan")) if (enable_view_lib and lib is not None) else float("nan"),
                "lib_dir_y": (float(best.get("lib_dir", [float('nan'), float('nan'), float('nan')])[1]) if isinstance(best.get("lib_dir", None), list) and len(best.get("lib_dir", [])) >= 3 else float("nan")) if (enable_view_lib and lib is not None) else float("nan"),
                "lib_dir_z": (float(best.get("lib_dir", [float('nan'), float('nan'), float('nan')])[2]) if isinstance(best.get("lib_dir", None), list) and len(best.get("lib_dir", [])) >= 3 else float("nan")) if (enable_view_lib and lib is not None) else float("nan"),
                "lib_template_contour": str(best.get("lib_template_contour", "")) if (enable_view_lib and lib is not None) else "",
                "flow_uvh_enabled": flow_info.get("flow_uvh_enabled", 0),
                "flow_used": flow_info.get("flow_used", 0),
                "flow_n_matches": flow_info.get("flow_n_matches", 0),
                "flow_n_inliers": flow_info.get("flow_n_inliers", 0),
                "flow_inlier_ratio": flow_info.get("flow_inlier_ratio", ""),
                "flow_rmse": flow_info.get("flow_rmse", ""),
                "flow_npz": flow_info.get("flow_npz", ""),
                "flow_hsv_png": flow_info.get("flow_hsv_png", ""),
                "uvh_npz": flow_info.get("uvh_npz", ""),
                "uvh_vis_png": flow_info.get("uvh_vis_png", ""),
            }
        )

        # 更新 prev（prev_prev_bbox 供 uvh_render 热启动缩小搜索窗）
        prev_prev_bbox = prev_bbox
        prev_bbox = bbox_i
        prev_contour_rgb = contour_i
        prev_roi_th = np.asarray(th_i, dtype=np.float32)
        prev_roi_mask = np.asarray(mask_i, dtype=np.uint8)

        # checkpoint
        last_done = int(i)
        if cfg.resume_enable and (not cfg.force_restart):
            every = int(max(1, getattr(cfg, "ckpt_every", 5)))
            if (last_done % every) == 0 or (last_done >= len(frames) - 1):
                _save_checkpoint_track_only(
                    out_root=out_root,
                    cfg=cfg,
                    frames=frames,
                    last_frame=last_done,
                    prev_bbox=prev_bbox,
                    prev_view_id=int(prev_view_id),
                    track_rows=track_rows,
                    diag_rows=diag_rows,
                    completed=False,
                )

    # 保存最终结果
    with open(os.path.join(out_root, "tracking_bboxes_and_scores.json"), "w", encoding="utf-8") as f:
        json.dump(track_rows, f, ensure_ascii=False, indent=2)

    try:
        diag_csv = os.path.join(out_root, "diag_track_only.csv")
        with open(diag_csv, "w", encoding="utf-8", newline="\n") as f:
            fieldnames = list(diag_rows[0].keys()) if diag_rows else [
                "frame_idx", "img", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score", "semantic", "quick", "dx", "dy", "th_uint8"
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in diag_rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
    except Exception:
        pass

    # 标记完成（避免下次继续续跑）
    if cfg.resume_enable and (not cfg.force_restart):
        try:
            _save_checkpoint_track_only(
                out_root=out_root,
                cfg=cfg,
                frames=frames,
                last_frame=int(max(0, min(len(frames) - 1, last_done))),
                prev_bbox=prev_bbox,
                prev_view_id=int(prev_view_id),
                track_rows=track_rows,
                diag_rows=diag_rows,
                completed=True,
            )
        except Exception:
            pass

    print(f"[完成] 输出目录: {out_root}")
    print(f"[完成] 跟踪结果: {os.path.join(out_root, 'tracking_bboxes_and_scores.json')}")
    print(f"[完成] 诊断CSV: {os.path.join(out_root, 'diag_track_only.csv')}")
    if core.SEMANTIC_ENABLED:
        print("[信息] 轮廓相似度已启用（ContourSimilarityNet，train_contour_similarity）。")
    else:
        if not bool(_env_bool("MIS_UVH_SEMANTIC_ENABLE", True)):
            print("[提示] MIS_UVH_SEMANTIC_ENABLE=0，轮廓神经网络未用于融合，core.semantic_similarity 为回退相似度。")
        elif str(sem_meta.get("semantic_backend", "")) == "fallback_only":
            ck = str(sem_meta.get("contour_sim_skip_reason", "") or "").strip()
            print(
                f"[提示] ContourSimilarityNet 未加载（{ck or '见 contour_sim_skip_reason'}），"
                f"将使用回退轮廓相似度。可检查 MIS_CONTOUR_SIM_CKPT={cfg.model_path!r}"
            )
        else:
            print("[提示] 轮廓神经网络未启用，将使用回退相似度。")


if __name__ == "__main__":
    main()

