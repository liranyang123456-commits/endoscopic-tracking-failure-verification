#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIS_C_Fe_Simi_Mesh_Match_Pose_Estimate_Track.py

按用户需求实现的流程（简化+工程可落地版）：
1) 读取帧序列 -> 第一帧鼠标框选 ROI 目标框；
   在 ROI 内计算 th_68（灰度截断图，不是二值）-> 生成轮廓 -> 送入预训练模型算“轮廓语义相似度”；
   之后每帧在上一帧 bbox 附近按步长滑动搜索，找到相似度最佳的 bbox（带时空先验惩罚与粗筛加速）。
2) 保存每帧 bbox、mask、th_68 灰度截断图；用 th_68 灰度截断图构建伪3D高度场 mesh（u,v,h）；
   计算帧间 mesh 的刚体配准 RT（ICP，返回 T_{t-1<-t} = cur->prev）。
3) 保存中间图像、mesh、RT、轨迹可视化到：D:\\MIS_Pose_Track_Re3D\\Exp_Results

说明：
- 语义相似度函数沿用参考脚本的 import 方式：
    from extract_semantic_features_Function import calculate_contour_similarity
  若该模块不存在，则自动回退到轻量相似度（mask IoU + 相关性）。
- mesh->RT 默认用 Open3D ICP；若 open3d 未安装，则跳过 RT 计算但仍保存跟踪结果。

推荐运行方式（PowerShell，直接复制即可）：

```powershell
$env:MIS_RT_MODE="fullframe"
$env:MIS_RT_ESTIMATOR="flow_ransac"
python d:/MIS_Pose_Track_Re3D/MIS_C_Fe_Simi_Mesh_Match_Pose_Estimate_Track.py
```

重要说明（关于“相机RT”的语义）：
- 本脚本输出的 `shape_rt_pairwise_4x4.txt / shape_rt_cumulative_4x4.txt` 是在伪3D高度场(uvh)上的形状对齐刚体变换。
- 在软组织形变/遮挡/光照变化存在时，它**不保证**等价真实物理相机位姿；但可作为相机相对运动的鲁棒代理，并可结合 diag.csv 的质量指标筛除坏帧。
"""

from __future__ import annotations

import os
import re
import json
import math
import time
import tempfile
import sys
import importlib
import traceback
import csv
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import cv2

# 可选依赖
try:
    import torch  # noqa: F401
except Exception:
    torch = None

try:
    import open3d as o3d  # noqa: F401
except Exception:
    o3d = None

# 语义相似度：延迟导入（避免“文件在但内部依赖缺失”被误判成未找到）
calculate_contour_similarity = None  # type: ignore
SEMANTIC_AVAILABLE = False
SEMANTIC_ENABLED = False
SEMANTIC_IMPORT_ERROR: str | None = None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _is_image_file(fn: str) -> bool:
    ext = os.path.splitext(fn)[1].lower()
    return ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _sort_key(path: str) -> tuple[int, str]:
    base = os.path.basename(path)
    m = re.findall(r"\d+", base)
    if m:
        try:
            return int(m[-1]), base
        except Exception:
            pass
    return 10**18, base


def list_frames(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"帧目录不存在: {folder}")
    files = [os.path.join(folder, f) for f in os.listdir(folder) if _is_image_file(f)]
    files.sort(key=_sort_key)
    return files


@dataclass
class TrackConfig:
    frames_dir: str = r"D:\Exp_MIS_ChessBox_Datas\MIS_Videso_1_6\extracted_frames6\extracted_frames"
    output_dir: str = r"D:\MIS_Pose_Track_Re3D\Exp_Results6"

    # 运行加速：对输入帧先做分辨率缩放再处理（强烈建议用于加速）
    # 例如 0.5 表示宽高各减半，像素数变为 1/4，th_68 / flow / mesh / ICP 通常会显著提速
    proc_scale: float = float(os.environ.get("MIS_PROC_SCALE", "0.5"))

    # 目标框匹配（时空连续性 + 滑动窗口）
    search_radius: int = int(os.environ.get("MIS_SEARCH_RADIUS", "56"))  # 像素
    search_step: int = int(os.environ.get("MIS_SEARCH_STEP", "20"))      # 像素（步长大则候选少）
    topk_semantic: int = int(os.environ.get("MIS_TOPK_SEMANTIC", "6"))   # 粗筛后进入精排的候选数
    spatial_penalty: float = float(os.environ.get("MIS_SPATIAL_PENALTY", "0.15"))
    quick_scale: float = float(os.environ.get("MIS_QUICK_SCALE", "0.5"))  # 粗筛时ROI缩放比例（提升速度）
    quick_min_size: int = int(os.environ.get("MIS_QUICK_MIN_SIZE", "96")) # 粗筛时ROI最小边长

    # th_68
    th_percent: float = float(os.environ.get("MIS_TH_PERCENT", "68.0"))
    bilateral_d: int = int(os.environ.get("MIS_BILATERAL_D", "7"))
    bilateral_sc: float = float(os.environ.get("MIS_BILATERAL_SC", "75"))
    bilateral_ss: float = float(os.environ.get("MIS_BILATERAL_SS", "75"))

    # 轮廓相似度模型
    model_path: str = os.environ.get("MIS_MODEL_PATH", r"E:\reloc3r\Autoencoder_129.pth")
    semantic_image_size: tuple[int, int] = (128, 128)
    device: str = os.environ.get("MIS_DEVICE", "cuda")  # cuda/cpu/auto
    semantic_warmup: bool = bool(int(os.environ.get("MIS_SEMANTIC_WARMUP", "1")))  # 启动时预热一次模型（避免首帧卡顿）
    # 语义模块代码可能依赖额外文件（如 Copy12.py），可用该变量添加额外搜索路径（;分隔）
    semantic_code_dirs: str = os.environ.get("MIS_SEMANTIC_CODE_DIRS", "").strip()

    # 后验评估：RT 对齐后渲染轮廓，再算相似度（用于诊断配准质量）
    postalign_eval: bool = bool(int(os.environ.get("MIS_POSTALIGN_EVAL", "1")))
    postalign_save_fullres: bool = bool(int(os.environ.get("MIS_POSTALIGN_SAVE_FULLRES", "0")))

    # mesh / RT（伪3D高度场）
    # 重要：z 的量纲和 x/y(像素)差很多，默认 1.0 容易“看起来像平面”，这里默认调大到 30 方便可视化
    mesh_z_scale: float = float(os.environ.get("MIS_MESH_Z_SCALE", "30.0"))
    icp_max_corr: float = float(os.environ.get("MIS_ICP_MAX_CORR", "10.0"))
    icp_max_iter: int = int(os.environ.get("MIS_ICP_MAX_ITER", "80"))
    voxel_size: float = float(os.environ.get("MIS_VOXEL_SIZE", "2.0"))
    sample_points: int = int(os.environ.get("MIS_SAMPLE_POINTS", "30000"))
    sample_poisson: bool = bool(int(os.environ.get("MIS_SAMPLE_POISSON", "1")))
    # 门控/恢复（参考 Exp_Mehsh_Between_RT.py 的核心思想）
    recovery_enable: bool = bool(int(os.environ.get("MIS_RECOVERY_ENABLE", "1")))
    recovery_fitness_min: float = float(os.environ.get("MIS_RECOVERY_FITNESS_MIN", "0.995"))
    recovery_rmse_max: float = float(os.environ.get("MIS_RECOVERY_RMSE_MAX", "4.0"))
    icp_multitrial: bool = bool(int(os.environ.get("MIS_ICP_MULTITRIAL", "1")))
    # 多次尝试的 max_corr/iter 放大倍率（逗号分隔）
    icp_corr_mults: str = os.environ.get("MIS_ICP_CORR_MULTS", "1.0,1.8,2.2").strip()
    icp_iter_mults: str = os.environ.get("MIS_ICP_ITER_MULTS", "1.0,1.5,2.0").strip()
    # keyframe 恢复
    keyframe_enable: bool = bool(int(os.environ.get("MIS_KEYFRAME_ENABLE", "1")))
    keyframe_interval: int = int(os.environ.get("MIS_KEYFRAME_INTERVAL", "10"))
    reject_on_fail: bool = bool(int(os.environ.get("MIS_REJECT_ON_FAIL", "0")))  # 恢复仍失败时是否将该帧pairwise置为I(避免漂移)

    # greedy recovery（参考 Exp_Mehsh_Between_RT.py 的 _greedy_candidates + 短ICP）
    greedy_enable: bool = bool(int(os.environ.get("MIS_GREEDY_ENABLE", "1")))
    greedy_icp_max_iter: int = int(os.environ.get("MIS_GREEDY_ICP_MAX_ITER", "48"))
    greedy_rot_deg_list: str = os.environ.get("MIS_GREEDY_ROT_DEG_LIST", "0,5,-5,10,-10,15,-15").strip()
    greedy_trans_list: str = os.environ.get("MIS_GREEDY_TRANS_LIST", "0,3,-3,6,-6,10,-10,15,-15").strip()
    greedy_max_cands: int = int(os.environ.get("MIS_GREEDY_MAX_CANDS", "64"))

    # RANSAC recovery（参考 Exp_Mehsh_Between_RT.py 的最后兜底思想；Open3D FPFH-RANSAC）
    ransac_enable: bool = bool(int(os.environ.get("MIS_RANSAC_ENABLE", "1")))
    ransac_only_on_jump: bool = bool(int(os.environ.get("MIS_RANSAC_ONLY_ON_JUMP", "1")))
    ransac_voxel: float = float(os.environ.get("MIS_RANSAC_VOXEL", "6.0"))
    ransac_max_corr_mult: float = float(os.environ.get("MIS_RANSAC_MAX_CORR_MULT", "2.5"))
    ransac_max_iter: int = int(os.environ.get("MIS_RANSAC_MAX_ITER", "50000"))
    ransac_confidence: float = float(os.environ.get("MIS_RANSAC_CONFIDENCE", "0.999"))

    # motion gate（简化版，用于决定是否优先 keyframe / 允许 RANSAC）
    gate_jump_enable: bool = bool(int(os.environ.get("MIS_GATE_JUMP_ENABLE", "1")))
    gate_center_shift_px: float = float(os.environ.get("MIS_GATE_CENTER_SHIFT_PX", "40.0"))
    gate_nz_ratio_mult: float = float(os.environ.get("MIS_GATE_NZ_RATIO_MULT", "2.2"))

    # ========== 非刚性辅助的“鲁棒刚体估计”(dense flow -> uvh matches -> RANSAC+SVD -> IRLS) ==========
    rt_estimator: str = os.environ.get("MIS_RT_ESTIMATOR", "flow_ransac").strip().lower()  # icp / flow_ransac
    flow_use_fullframe: bool = bool(int(os.environ.get("MIS_FLOW_USE_FULLFRAME", "1")))
    flow_grid_step: int = int(os.environ.get("MIS_FLOW_GRID_STEP", "6"))  # 采样点网格间隔（像素）
    flow_max_points: int = int(os.environ.get("MIS_FLOW_MAX_POINTS", "12000"))
    flow_min_h: float = float(os.environ.get("MIS_FLOW_MIN_H", "1e-6"))  # 采样 h 时的有效阈值
    flow_ransac_iters: int = int(os.environ.get("MIS_FLOW_RANSAC_ITERS", "1500"))
    flow_ransac_thr_xy: float = float(os.environ.get("MIS_FLOW_RANSAC_THR_XY", "6.0"))  # uv 平面残差阈值(px)
    flow_ransac_min_inliers: int = int(os.environ.get("MIS_FLOW_RANSAC_MIN_INLIERS", "300"))
    flow_irls_iters: int = int(os.environ.get("MIS_FLOW_IRLS_ITERS", "5"))
    flow_irls_c: float = float(os.environ.get("MIS_FLOW_IRLS_C", "8.0"))  # Tukey 截断尺度(px)
    flow_refine_icp: bool = bool(int(os.environ.get("MIS_FLOW_REFINE_ICP", "1")))
    flow_save_vis: bool = bool(int(os.environ.get("MIS_FLOW_SAVE_VIS", "1")))
    flow_vis_max_lines: int = int(os.environ.get("MIS_FLOW_VIS_MAX_LINES", "800"))
    flow_save_raw: bool = bool(int(os.environ.get("MIS_FLOW_SAVE_RAW", "1")))
    flow_save_raw_stride: int = int(os.environ.get("MIS_FLOW_SAVE_RAW_STRIDE", "2"))  # 保存原始flow时的降采样步长
    flow_save_hsv: bool = bool(int(os.environ.get("MIS_FLOW_SAVE_HSV", "1")))
    # flow_ransac 质量门控（与 ICP fitness/rmse 语义不同，单独给阈值）
    flow_inlier_ratio_min: float = float(os.environ.get("MIS_FLOW_INLIER_RATIO_MIN", "0.15"))
    flow_rmse_max: float = float(os.environ.get("MIS_FLOW_RMSE_MAX", "8.0"))  # px（xy 平面）
    # Farneback 光流参数（在内窥镜纹理上通常比稀疏点更稳，但仍会受光照/遮挡影响）
    fb_pyr_scale: float = float(os.environ.get("MIS_FB_PYR_SCALE", "0.5"))
    fb_levels: int = int(os.environ.get("MIS_FB_LEVELS", "3"))
    fb_winsize: int = int(os.environ.get("MIS_FB_WINSIZE", "21"))
    fb_iterations: int = int(os.environ.get("MIS_FB_ITERS", "3"))
    fb_poly_n: int = int(os.environ.get("MIS_FB_POLY_N", "7"))
    fb_poly_sigma: float = float(os.environ.get("MIS_FB_POLY_SIGMA", "1.5"))
    # RT估计模式：
    # - roi: 用ROI跟踪得到的 th_68 mesh 做帧间ICP
    # - fullframe: 直接用整帧前景(th_68) mesh 做帧间ICP（不依赖ROI跟踪）
    rt_mode: str = os.environ.get("MIS_RT_MODE", "fullframe").strip().lower()
    # 保存帧间匹配结果（参考 Exp_Mehsh_Between_RT.py）
    save_pair_results: bool = bool(int(os.environ.get("MIS_SAVE_PAIR_RESULTS", "1")))
    save_mesh_aligned_pair: bool = bool(int(os.environ.get("MIS_SAVE_MESH_ALIGNED_PAIR", "0")))
    # 将 shape-alignment 累计矩阵解释为“相机位姿”的方式（仅导出用，非物理保证）：
    # - camera：直接用 T_{0<-t} 作为 c2w（与 Exp_Mehsh_Between_RT.py 的默认一致）
    # - object：使用 inv(T_{0<-t}) 作为 c2w（相机不动、物体在动的对偶解释）
    pose_interpretation: str = os.environ.get("MIS_POSE_INTERPRETATION", "camera").strip().lower()

    # 运行控制
    max_frames: int = int(os.environ.get("MIS_MAX_FRAMES", "0"))  # 0 不限制
    vis_window: bool = bool(int(os.environ.get("MIS_VIS_WINDOW", "1")))
    roi_backend: str = os.environ.get("MIS_ROI_BACKEND", "auto").strip().lower()  # auto/opencv/tk/init/none
    tk_max_w: int = int(os.environ.get("MIS_TK_MAX_W", "1200"))
    tk_max_h: int = int(os.environ.get("MIS_TK_MAX_H", "900"))
    # 当 OpenCV 无GUI（cvNamedWindow/selectROI不可用）时，可用此参数指定初始ROI：
    # 格式: "x,y,w,h"（像素，基于第一帧坐标系）
    init_bbox: str = os.environ.get("MIS_INIT_BBOX", "").strip()
    save_mesh: bool = bool(int(os.environ.get("MIS_SAVE_MESH", "1")))
    save_debug_images: bool = bool(int(os.environ.get("MIS_SAVE_DEBUG", "0")))
    recover_enabled: bool = bool(int(os.environ.get("MIS_RECOVER_ENABLED", "1")))
    recover_score: float = float(os.environ.get("MIS_RECOVER_SCORE", "0.22"))  # 分数过低时扩大搜索

    # 控制台日志
    verbose: int = int(os.environ.get("MIS_VERBOSE", "1"))          # 0静默/1关键/2详细
    print_every: int = int(os.environ.get("MIS_PRINT_EVERY", "10"))  # 每N帧打印一次关键进度
    print_timing: bool = bool(int(os.environ.get("MIS_PRINT_TIMING", "1")))

    # ========== 断点续跑（默认开启）==========
    # 若输出目录下存在 checkpoint，则从 last_frame+1 继续执行；否则从 0 开始。
    resume_enable: bool = bool(int(os.environ.get("MIS_RESUME", "1")))
    ckpt_every: int = int(os.environ.get("MIS_CKPT_EVERY", "5"))  # 每 N 帧写一次 checkpoint（1 最稳但更慢）
    force_restart: bool = bool(int(os.environ.get("MIS_FORCE_RESTART", "0")))  # 忽略 checkpoint，强制从头跑

    # ========== 位姿可视化（Open3D，可选；给默认值避免 AttributeError）==========
    vis_open3d: bool = bool(int(os.environ.get("MIS_VIS_OPEN3D", "0")))  # 1:弹窗显示（某些环境会卡住）
    vis_frame_size: str = os.environ.get("MIS_VIS_FRAME_SIZE", "auto")    # "auto" 或 数值字符串
    vis_show_camera_frustum: bool = bool(int(os.environ.get("MIS_VIS_SHOW_FRUSTUM", "0")))
    vis_show_rt_frames: bool = bool(int(os.environ.get("MIS_VIS_SHOW_FRAMES", "0")))
    vis_rt_every: int = int(os.environ.get("MIS_VIS_RT_EVERY", "10"))


def _log(msg: str, *, cfg: Optional[TrackConfig] = None, level: int = 1) -> None:
    """轻量日志：level=1关键，level=2详细。"""
    try:
        v = 1 if cfg is None else int(getattr(cfg, "verbose", 1))
        if v >= int(level):
            print(msg, flush=True)
    except Exception:
        print(msg, flush=True)


def _ckpt_paths(out_root: str) -> dict[str, str]:
    return {
        "state": os.path.join(out_root, "checkpoint_state.json"),
        "rt_npz": os.path.join(out_root, "checkpoint_rt.npz"),
        "track_json": os.path.join(out_root, "checkpoint_track.json"),
        "diag_json": os.path.join(out_root, "checkpoint_diag.json"),
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
        # checkpoint 写失败不应中断主流程
        return


def _load_th_gray_u8(path: str) -> Optional[np.ndarray]:
    try:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        return im.astype(np.float32)
    except Exception:
        return None


def _load_mask_u8(path: str) -> Optional[np.ndarray]:
    try:
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        return im.astype(np.uint8)
    except Exception:
        return None


def _load_contour_rgb(path: str) -> Optional[np.ndarray]:
    try:
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is None:
            return None
        return im.astype(np.uint8)
    except Exception:
        return None


def _save_checkpoint(
    *,
    out_root: str,
    cfg: TrackConfig,
    frames: list[str],
    last_frame: int,
    prev_bbox: tuple[int, int, int, int],
    key_idx: int,
    pairwise_T: dict[int, np.ndarray],
    cumulative_T: dict[int, np.ndarray],
    track_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, Any]],
    completed: bool = False,
) -> None:
    """
    写断点续跑 checkpoint（尽量轻量，但保证可恢复关键状态）。
    - 不序列化 mesh/图像；恢复时从已保存的 th_gray/mask/contour_rgb 或直接重算。
    """
    try:
        paths = _ckpt_paths(out_root)
        last_frame = int(max(0, min(int(last_frame), len(frames) - 1)))
        # 保存 RT（0..last_frame）
        idxs = np.arange(0, last_frame + 1, dtype=np.int32)
        pw = np.stack([np.asarray(pairwise_T.get(int(i), np.eye(4)), dtype=np.float64).reshape(4, 4) for i in idxs], axis=0)
        cu = np.stack([np.asarray(cumulative_T.get(int(i), np.eye(4)), dtype=np.float64).reshape(4, 4) for i in idxs], axis=0)
        try:
            np.savez_compressed(paths["rt_npz"], frame_indices=idxs, pairwise_T=pw.astype(np.float64), cumulative_T=cu.astype(np.float64))
        except Exception:
            pass

        # 保存 track/diag（json）
        _safe_write_json(paths["track_json"], track_rows)
        _safe_write_json(paths["diag_json"], diag_rows)

        # 保存 state（轻量）
        state = {
            "version": 1,
            "time": _now(),
            "completed": bool(completed),
            "frames_dir": str(cfg.frames_dir),
            "proc_scale": float(cfg.proc_scale),
            "rt_mode": str(cfg.rt_mode),
            "rt_estimator": str(cfg.rt_estimator),
            "last_frame": int(last_frame),
            "prev_bbox": [int(x) for x in prev_bbox],
            "key_idx": int(key_idx),
        }
        _safe_write_json(paths["state"], state)

        # 同步一份“可读的中间输出”（便于用户看进度；写失败不影响）
        try:
            with open(os.path.join(out_root, "tracking_bboxes_and_scores.json"), "w", encoding="utf-8") as f:
                json.dump(track_rows, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        try:
            diag_csv = os.path.join(out_root, "diag_partial.csv")
            with open(diag_csv, "w", encoding="utf-8", newline="\n") as f:
                if diag_rows:
                    fieldnames = list(diag_rows[0].keys())
                else:
                    fieldnames = [
                        "frame_idx", "img", "pair_fitness", "pair_rmse", "method",
                        "rt_mode", "init_mode", "recovered", "key_idx",
                        "gate_jump", "gate_mode", "recovery_tag",
                        "nz_center_dxy", "nz_ratio_mult",
                        "rt_estimator", "flow_used", "flow_n_matches", "flow_n_inliers", "flow_rmse",
                    ]
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in diag_rows:
                    w.writerow({k: row.get(k, "") for k in fieldnames})
        except Exception:
            pass
    except Exception:
        return


def _try_load_checkpoint(
    *,
    out_root: str,
    cfg: TrackConfig,
    frames: list[str],
    out_mask: str,
    out_th: str,
    out_cont: str,
) -> Optional[dict[str, Any]]:
    """
    若 checkpoint 可用则返回恢复所需的状态；否则返回 None。
    """
    if not bool(getattr(cfg, "resume_enable", True)) or bool(getattr(cfg, "force_restart", False)):
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
        # frames_dir 不一致时不恢复（防止把旧序列的断点拿来续跑新序列）
        if str(st.get("frames_dir", "")).strip() and (str(st.get("frames_dir", "")).strip() != str(cfg.frames_dir).strip()):
            return None
        if last_frame < 0 or last_frame >= len(frames):
            return None
        # proc_scale 不一致时不自动恢复（bbox/尺寸都会错）
        ps = float(st.get("proc_scale", -1.0))
        if not np.isfinite(ps) or abs(ps - float(cfg.proc_scale)) > 1e-6:
            return None
        prev_bbox = st.get("prev_bbox", None)
        if not (isinstance(prev_bbox, list) and len(prev_bbox) == 4):
            return None
        prev_bbox_t = (int(prev_bbox[0]), int(prev_bbox[1]), int(prev_bbox[2]), int(prev_bbox[3]))
        key_idx = int(st.get("key_idx", 0))
        key_idx = int(max(0, min(key_idx, last_frame)))

        # 读取 RT
        pairwise_T: dict[int, np.ndarray] = {}
        cumulative_T: dict[int, np.ndarray] = {}
        if os.path.isfile(paths["rt_npz"]):
            try:
                z = np.load(paths["rt_npz"], allow_pickle=False)
                idxs = np.asarray(z["frame_indices"], dtype=np.int32).reshape(-1)
                pw = np.asarray(z["pairwise_T"], dtype=np.float64).reshape(-1, 4, 4)
                cu = np.asarray(z["cumulative_T"], dtype=np.float64).reshape(-1, 4, 4)
                for j, fi in enumerate(idxs.tolist()):
                    pairwise_T[int(fi)] = pw[j]
                    cumulative_T[int(fi)] = cu[j]
            except Exception:
                pairwise_T = {}
                cumulative_T = {}
        if (last_frame not in cumulative_T) or (last_frame not in pairwise_T):
            return None

        # track/diag
        track_rows = _safe_read_json(paths["track_json"])
        diag_rows = _safe_read_json(paths["diag_json"])
        if not isinstance(track_rows, list):
            track_rows = []
        if not isinstance(diag_rows, list):
            diag_rows = []

        # prev frame 的图像与特征（从已有输出恢复；缺失则重算）
        prev_full_bgr = _read_frame_bgr(frames[last_frame], cfg)
        if prev_full_bgr is None:
            return None

        # contour_rgb
        prev_cont = _load_contour_rgb(os.path.join(out_cont, f"contour_rgb_{last_frame:04d}.png"))
        if prev_cont is None:
            # 尝试用 mask 恢复 contour
            mk = _load_mask_u8(os.path.join(out_mask, f"mask_{last_frame:04d}.png"))
            if mk is not None:
                cnt = extract_largest_contour(mk)
                if cnt is not None:
                    cnt2 = resize_contour(cnt, (prev_bbox_t[2], prev_bbox_t[3]), cfg.semantic_image_size)
                    prev_cont = contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)
        if prev_cont is None:
            prev_cont = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)

        # prev_mask_small
        mk = _load_mask_u8(os.path.join(out_mask, f"mask_{last_frame:04d}.png"))
        if mk is None:
            # 重算 ROI mask
            roi = crop(prev_full_bgr, prev_bbox_t)
            _, mk, _ = compute_th_gray_truncated(roi, cfg.th_percent, cfg)
        prev_mask_small = cv2.resize(mk, (64, 64), interpolation=cv2.INTER_NEAREST)

        # mesh_prev / key_mesh（从 th_gray 恢复更快）
        mesh_prev = None
        key_mesh = None
        if o3d is not None:
            th_u8 = _load_th_gray_u8(os.path.join(out_th, f"th_gray_{last_frame:04d}.png"))
            if th_u8 is None:
                roi = crop(prev_full_bgr, prev_bbox_t)
                th_u8, _, _ = compute_th_gray_truncated(roi, cfg.th_percent, cfg)
            mesh_prev = make_heightfield_mesh_from_th_gray(th_u8, prev_bbox_t, z_scale=cfg.mesh_z_scale)
            # key
            if key_idx >= 0:
                key_bbox = prev_bbox_t  # 默认；若历史 bbox 不同则按保存 bbox 恢复
                try:
                    # 从 track_rows 找 key 的 bbox（如果有）
                    for row in reversed(track_rows):
                        if int(row.get("frame", -1)) == int(key_idx):
                            bb = row.get("bbox", None)
                            if isinstance(bb, list) and len(bb) == 4:
                                key_bbox = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
                            break
                except Exception:
                    pass
                key_bgr = _read_frame_bgr(frames[key_idx], cfg)
                if key_bgr is not None:
                    thk = _load_th_gray_u8(os.path.join(out_th, f"th_gray_{key_idx:04d}.png"))
                    if thk is None:
                        roi = crop(key_bgr, key_bbox)
                        thk, _, _ = compute_th_gray_truncated(roi, cfg.th_percent, cfg)
                    key_mesh = make_heightfield_mesh_from_th_gray(thk, key_bbox, z_scale=cfg.mesh_z_scale)

        return {
            "start_frame": int(last_frame + 1),
            "last_frame": int(last_frame),
            "prev_bbox": prev_bbox_t,
            "prev_contour_rgb": prev_cont,
            "prev_mask_small": prev_mask_small,
            "mesh_prev": mesh_prev,
            "key_idx": int(key_idx),
            "key_mesh": key_mesh,
            "pairwise_T": pairwise_T,
            "cumulative_T": cumulative_T,
            "track_rows": track_rows,
            "diag_rows": diag_rows,
            "prev_full_bgr": prev_full_bgr,
        }
    except Exception:
        return None

def _try_parse_bbox(s: str) -> Optional[tuple[int, int, int, int]]:
    try:
        t = str(s).strip()
        if not t:
            return None
        # 支持 "x,y,w,h" 或 "x y w h"
        parts = re.split(r"[,\s]+", t)
        parts = [p for p in parts if p]
        if len(parts) != 4:
            return None
        x, y, w, h = [int(float(p)) for p in parts]
        return int(x), int(y), int(w), int(h)
    except Exception:
        return None


@dataclass
class RegResult:
    T: np.ndarray
    fitness: float
    rmse: float
    method: str


def _parse_float_list_csv(s: str, default: list[float]) -> list[float]:
    try:
        t = str(s).strip()
        if not t:
            return list(default)
        out: list[float] = []
        for part in t.replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            out.append(float(part))
        return out if out else list(default)
    except Exception:
        return list(default)


def _need_recovery(fitness: float, rmse: float, cfg: TrackConfig) -> bool:
    try:
        if not np.isfinite(fitness) or not np.isfinite(rmse):
            return True
        return (float(fitness) < float(cfg.recovery_fitness_min)) or (float(rmse) > float(cfg.recovery_rmse_max))
    except Exception:
        return True


def _better_than(a: RegResult, b: RegResult) -> bool:
    """参考 Exp_Mehsh_Between_RT.py：优先 fitness，其次 rmse。"""
    if not np.isfinite(a.fitness):
        return False
    if not np.isfinite(b.fitness):
        return True
    if a.fitness > b.fitness + 1e-9:
        return True
    if b.fitness > a.fitness + 1e-9:
        return False
    return float(a.rmse) < float(b.rmse) - 1e-9


def _mesh_stats(mesh: Optional["o3d.geometry.TriangleMesh"], *, z_eps: float = 1e-6) -> dict[str, float]:
    """
    简化版 mesh stats（参考 Exp_Mehsh_Between_RT.py 的 gate 思想）：
    - nz_ratio：z>eps 的点比例
    - nz_center_x/y：非零点云中心（用于检测大跳变）
    """
    out = {
        "nz_ratio": float("nan"),
        "nz_center_x": float("nan"),
        "nz_center_y": float("nan"),
    }
    if o3d is None or mesh is None:
        return out
    try:
        v = np.asarray(mesh.vertices, dtype=np.float64)
        if v.ndim != 2 or v.shape[1] != 3 or v.shape[0] < 10:
            return out
        z = v[:, 2]
        m = np.isfinite(z) & (z > float(z_eps))
        nz = int(np.count_nonzero(m))
        out["nz_ratio"] = float(nz) / float(max(1, v.shape[0]))
        if nz > 0:
            vv = v[m]
            out["nz_center_x"] = float(np.mean(vv[:, 0]))
            out["nz_center_y"] = float(np.mean(vv[:, 1]))
        return out
    except Exception:
        return out


def _motion_gate(prev_stats: dict[str, float], cur_stats: dict[str, float], cfg: TrackConfig) -> dict[str, Any]:
    """
    简化 motion gate：
    - gate_jump：非零中心大位移 或 nz_ratio 大幅变化
    - gate_mode：small / medium / jump
    """
    out = {
        "gate_jump": 0,
        "gate_mode": "small",
        "nz_center_dxy": float("nan"),
        "nz_ratio_mult": float("nan"),
    }
    try:
        px, py = float(prev_stats.get("nz_center_x", float("nan"))), float(prev_stats.get("nz_center_y", float("nan")))
        cx, cy = float(cur_stats.get("nz_center_x", float("nan"))), float(cur_stats.get("nz_center_y", float("nan")))
        pr = float(prev_stats.get("nz_ratio", float("nan")))
        cr = float(cur_stats.get("nz_ratio", float("nan")))
        if np.isfinite(px) and np.isfinite(py) and np.isfinite(cx) and np.isfinite(cy):
            dxy = float(math.hypot(cx - px, cy - py))
        else:
            dxy = float("nan")
        if np.isfinite(pr) and np.isfinite(cr) and pr > 1e-9:
            rm = float(cr / pr)
        else:
            rm = float("nan")
        out["nz_center_dxy"] = dxy
        out["nz_ratio_mult"] = rm

        jump = False
        if cfg.gate_jump_enable:
            if np.isfinite(dxy) and dxy > float(cfg.gate_center_shift_px):
                jump = True
            if np.isfinite(rm) and (rm > float(cfg.gate_nz_ratio_mult) or rm < 1.0 / float(max(1e-6, cfg.gate_nz_ratio_mult))):
                jump = True
        out["gate_jump"] = 1 if jump else 0
        if jump:
            out["gate_mode"] = "jump"
        else:
            if np.isfinite(dxy) and dxy > 0.5 * float(cfg.gate_center_shift_px):
                out["gate_mode"] = "medium"
            else:
                out["gate_mode"] = "small"
        return out
    except Exception:
        return out


def _rot_axis_deg(axis: str, deg: float) -> np.ndarray:
    a = float(deg) * math.pi / 180.0
    c = float(math.cos(a))
    s = float(math.sin(a))
    ax = axis.lower().strip()
    if ax == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    elif ax == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    else:
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    return T


def _trans_axis(axis: str, t: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    ax = axis.lower().strip()
    if ax == "x":
        T[0, 3] = float(t)
    elif ax == "y":
        T[1, 3] = float(t)
    else:
        T[2, 3] = float(t)
    return T


def _greedy_candidates(init_T: np.ndarray, cfg: TrackConfig) -> list[tuple[str, np.ndarray]]:
    """
    参考 Exp_Mehsh_Between_RT.py 的 greedy candidates（简化版）：
    约定：T_candidate = dT @ init_T
    """
    rot_list = _parse_float_list_csv(cfg.greedy_rot_deg_list, [0.0, 5.0, -5.0, 10.0, -10.0])
    trans_list = _parse_float_list_csv(cfg.greedy_trans_list, [0.0, 3.0, -3.0, 6.0, -6.0])
    uniq: dict[bytes, tuple[str, np.ndarray]] = {}

    def _add(tag: str, T: np.ndarray) -> None:
        try:
            key = np.asarray(T, dtype=np.float64).round(6).tobytes()
            if key not in uniq:
                uniq[key] = (tag, np.asarray(T, dtype=np.float64))
        except Exception:
            pass

    init_T = np.asarray(init_T, dtype=np.float64).reshape(4, 4)
    _add("init", init_T)
    for a in rot_list:
        for ax in ("x", "y", "z"):
            dT = _rot_axis_deg(ax, a)
            _add(f"rot_{ax}_{a:+g}", dT @ init_T)
    for t in trans_list:
        for ax in ("x", "y", "z"):
            dT = _trans_axis(ax, t)
            _add(f"trans_{ax}_{t:+g}", dT @ init_T)

    out = list(uniq.values())
    out = out[: max(1, int(cfg.greedy_max_cands))]
    return out


def _preprocess_pcd_for_fpfh(pcd: "o3d.geometry.PointCloud", voxel: float) -> tuple["o3d.geometry.PointCloud", "o3d.pipelines.registration.Feature"]:
    p = pcd.voxel_down_sample(voxel_size=float(voxel))
    p.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(voxel * 2.0), max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        p,
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(voxel * 5.0), max_nn=100),
    )
    return p, f


def _ransac_then_icp(
    *,
    mesh_prev: Optional["o3d.geometry.TriangleMesh"],
    mesh_cur: Optional["o3d.geometry.TriangleMesh"],
    cfg: TrackConfig,
) -> Optional[RegResult]:
    """Open3D FPFH-RANSAC 粗配准 + ICP refine（作为兜底恢复）。"""
    if o3d is None or mesh_prev is None or mesh_cur is None:
        return None
    try:
        voxel = float(max(1e-6, cfg.ransac_voxel))
        max_corr = float(max(1e-6, voxel * float(cfg.ransac_max_corr_mult)))
        # 用较稀疏点云做特征
        p_prev = _mesh_to_pcd(mesh_prev, cfg)
        p_cur = _mesh_to_pcd(mesh_cur, cfg)
        if len(p_prev.points) < 300 or len(p_cur.points) < 300:
            return None
        src, fsrc = _preprocess_pcd_for_fpfh(p_cur, voxel)   # cur
        tgt, ftgt = _preprocess_pcd_for_fpfh(p_prev, voxel)  # prev
        if len(src.points) < 100 or len(tgt.points) < 100:
            return None
        ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src, tgt, fsrc, ftgt, True, max_corr,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(int(cfg.ransac_max_iter), float(cfg.ransac_confidence)),
        )
        T0 = np.asarray(ransac.transformation, dtype=np.float64).reshape(4, 4)
        # ICP refine
        rr = estimate_T_prev_cur_icp(mesh_prev, mesh_cur, cfg, init_T=T0, max_corr=float(cfg.icp_max_corr), max_iter=int(cfg.icp_max_iter))
        if rr is None:
            return None
        rr.method = "ransac+icp"
        return rr
    except Exception:
        return None


def _estimate_rigid_T_svd(src: np.ndarray, tgt: np.ndarray, weights: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """
    Umeyama/SVD 刚体拟合（无尺度）：
      src -> tgt
    src/tgt: (N,3)
    weights: (N,) 可选
    """
    try:
        X = np.asarray(src, dtype=np.float64).reshape(-1, 3)
        Y = np.asarray(tgt, dtype=np.float64).reshape(-1, 3)
        if X.shape[0] < 3:
            return None
        if weights is None:
            w = np.ones((X.shape[0],), dtype=np.float64)
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.shape[0] != X.shape[0]:
                return None
            w = np.clip(w, 0.0, np.inf)
        sw = float(np.sum(w))
        if sw <= 1e-12:
            return None
        w = w / sw
        muX = (X * w[:, None]).sum(axis=0)
        muY = (Y * w[:, None]).sum(axis=0)
        Xc = X - muX[None, :]
        Yc = Y - muY[None, :]
        # 加权协方差
        S = (Xc * w[:, None]).T @ Yc
        U, _, Vt = np.linalg.svd(S)
        R = Vt.T @ U.T
        # 处理反射
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = muY - R @ muX
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    except Exception:
        return None


def _ransac_rigid_T_xy(
    *,
    cur_uvh: np.ndarray,
    prev_uvh: np.ndarray,
    cfg: TrackConfig,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    RANSAC 估计刚体：cur_uvh -> prev_uvh（输出 T_{prev<-cur}）
    误差只在 xy 平面度量（更符合你“相机位姿主要来自uv运动”的直觉，同时避免 z 映射漂移拖偏）。
    返回 (T, inlier_mask)
    """
    X = np.asarray(cur_uvh, dtype=np.float64).reshape(-1, 3)
    Y = np.asarray(prev_uvh, dtype=np.float64).reshape(-1, 3)
    n = int(X.shape[0])
    if n < 6:
        return None, None
    thr = float(max(1.0, cfg.flow_ransac_thr_xy))
    iters = int(max(50, cfg.flow_ransac_iters))
    best_inl = None
    best_T = None
    best_cnt = -1
    rng = np.random.default_rng(0)
    for _ in range(iters):
        idx = rng.choice(n, size=6, replace=False)
        T = _estimate_rigid_T_svd(X[idx], Y[idx])
        if T is None:
            continue
        Xh = (T[:3, :3] @ X.T).T + T[:3, 3][None, :]
        e = np.linalg.norm((Xh[:, :2] - Y[:, :2]), axis=1)
        inl = e <= thr
        cnt = int(np.count_nonzero(inl))
        if cnt > best_cnt:
            best_cnt = cnt
            best_inl = inl
            best_T = T
    if best_T is None or best_inl is None:
        return None, None
    if best_cnt < int(cfg.flow_ransac_min_inliers):
        return best_T, best_inl
    # 用全部 inlier 再估一次
    T2 = _estimate_rigid_T_svd(X[best_inl], Y[best_inl])
    return (T2 if T2 is not None else best_T), best_inl


def _tukey_weights(res_xy: np.ndarray, c: float) -> np.ndarray:
    """Tukey biweight：用于 IRLS，把大残差当作形变/错配压到0权重。"""
    r = np.asarray(res_xy, dtype=np.float64).reshape(-1)
    c = float(max(1e-6, c))
    u = r / c
    w = np.zeros_like(u)
    m = np.abs(u) < 1.0
    w[m] = (1.0 - u[m] ** 2) ** 2
    return w


def _irls_refine_T(
    *,
    T_init: np.ndarray,
    cur_uvh: np.ndarray,
    prev_uvh: np.ndarray,
    cfg: TrackConfig,
    inlier_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """对 RANSAC 的 T 做 IRLS 细化（鲁棒拟合全局刚体，抑制形变/错配）。"""
    T = np.asarray(T_init, dtype=np.float64).reshape(4, 4).copy()
    X = np.asarray(cur_uvh, dtype=np.float64).reshape(-1, 3)
    Y = np.asarray(prev_uvh, dtype=np.float64).reshape(-1, 3)
    if inlier_mask is None:
        m = np.ones((X.shape[0],), dtype=bool)
    else:
        m = np.asarray(inlier_mask, dtype=bool).reshape(-1)
        if m.shape[0] != X.shape[0]:
            m = np.ones((X.shape[0],), dtype=bool)
    X0 = X[m]
    Y0 = Y[m]
    if X0.shape[0] < 6:
        return T
    for _ in range(int(max(1, cfg.flow_irls_iters))):
        Xh = (T[:3, :3] @ X0.T).T + T[:3, 3][None, :]
        r = np.linalg.norm((Xh[:, :2] - Y0[:, :2]), axis=1)
        w = _tukey_weights(r, float(cfg.flow_irls_c))
        if float(np.sum(w)) <= 1e-9:
            break
        Tn = _estimate_rigid_T_svd(X0, Y0, weights=w)
        if Tn is None:
            break
        T = Tn
    return T


def _build_uvh_matches_from_flow(
    *,
    prev_bgr: np.ndarray,
    cur_bgr: np.ndarray,
    prev_th: np.ndarray,
    cur_th: np.ndarray,
    prev_mask: np.ndarray,
    cfg: TrackConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    用稠密光流在 uv 平面建立对应，再采样 h 得到 uvh 对应：
      返回 (cur_uvh, prev_uvh, in_bounds_mask)
    坐标均在整帧坐标系（x=u, y=v）。
    """
    prev_g = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    cur_g = cv2.cvtColor(cur_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_g, cur_g, None,
        pyr_scale=float(cfg.fb_pyr_scale),
        levels=int(cfg.fb_levels),
        winsize=int(cfg.fb_winsize),
        iterations=int(cfg.fb_iterations),
        poly_n=int(cfg.fb_poly_n),
        poly_sigma=float(cfg.fb_poly_sigma),
        flags=0,
    )
    H, W = prev_g.shape[:2]
    step = int(max(2, cfg.flow_grid_step))
    # 在 prev_mask 内采样网格点
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    xs = xs.reshape(-1).astype(np.int32)
    ys = ys.reshape(-1).astype(np.int32)
    m = (prev_mask[ys, xs] > 0)
    xs = xs[m]
    ys = ys[m]
    if xs.size > int(cfg.flow_max_points):
        # 均匀下采样
        idx = np.linspace(0, xs.size - 1, int(cfg.flow_max_points)).astype(np.int32)
        xs = xs[idx]
        ys = ys[idx]
    # 光流预测到当前帧坐标
    fx = flow[ys, xs, 0]
    fy = flow[ys, xs, 1]
    x2 = xs.astype(np.float32) + fx.astype(np.float32)
    y2 = ys.astype(np.float32) + fy.astype(np.float32)
    inb = (x2 >= 0) & (x2 <= W - 1) & (y2 >= 0) & (y2 <= H - 1)
    xs0 = xs[inb].astype(np.float32)
    ys0 = ys[inb].astype(np.float32)
    x2 = x2[inb]
    y2 = y2[inb]
    if xs0.size < 10:
        return np.zeros((0, 3), np.float64), np.zeros((0, 3), np.float64), inb, {"flow": flow, "prev_xy": np.zeros((0, 2), np.float32), "cur_xy": np.zeros((0, 2), np.float32)}

    # 采样 h：prev 用最近邻；cur 用双线性（更平滑）
    hp = prev_th[ys0.astype(np.int32), xs0.astype(np.int32)].astype(np.float32).reshape(-1)
    # cv2.remap 需要 2D map；这里把 (N,) 变成 (N,1)，再 flatten 回 (N,)
    mapx = x2.astype(np.float32).reshape(-1, 1)
    mapy = y2.astype(np.float32).reshape(-1, 1)
    hc = cv2.remap(
        cur_th.astype(np.float32),
        mapx,
        mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).reshape(-1)
    # 有效高度过滤
    okh = ((hp > float(cfg.flow_min_h)) & (hc > float(cfg.flow_min_h))).reshape(-1)
    xs0 = xs0[okh]; ys0 = ys0[okh]; x2 = x2[okh]; y2 = y2[okh]; hp = hp[okh]; hc = hc[okh]
    if xs0.size < 10:
        return np.zeros((0, 3), np.float64), np.zeros((0, 3), np.float64), inb, {"flow": flow, "prev_xy": np.zeros((0, 2), np.float32), "cur_xy": np.zeros((0, 2), np.float32)}
    # uvh（注意 z 用 th_gray 的强度映射，与 mesh 一致）
    zprev = (hp / 255.0) * float(cfg.mesh_z_scale)
    zcur = (hc / 255.0) * float(cfg.mesh_z_scale)
    prev_uvh = np.stack([xs0, ys0, zprev], axis=1).astype(np.float64)
    cur_uvh = np.stack([x2, y2, zcur], axis=1).astype(np.float64)
    dbg = {
        "flow": flow,
        "prev_xy": np.stack([xs0, ys0], axis=1).astype(np.float32),
        "cur_xy": np.stack([x2, y2], axis=1).astype(np.float32),
    }
    return cur_uvh, prev_uvh, inb, dbg


def _save_flow_matches_vis(
    out_png: str,
    *,
    prev_bgr: np.ndarray,
    prev_uvh: np.ndarray,
    cur_uvh: np.ndarray,
    T_prev_cur: np.ndarray,
    inliers: Optional[np.ndarray],
    max_lines: int = 800,
) -> None:
    """
    在 prev 图上画匹配可视化：
    - prev 点：绿
    - cur 点经 T_prev<-cur 变换到 prev 后：红
    - 可选画连线（只画部分，避免太密）
    """
    try:
        im = prev_bgr.copy()
        P = np.asarray(prev_uvh, dtype=np.float32).reshape(-1, 3)
        C = np.asarray(cur_uvh, dtype=np.float32).reshape(-1, 3)
        T = np.asarray(T_prev_cur, dtype=np.float64).reshape(4, 4)
        # transform cur -> prev
        Cp = (T[:3, :3] @ C.T).T + T[:3, 3][None, :]
        pxy = P[:, :2]
        cxy = Cp[:, :2].astype(np.float32)
        n = int(P.shape[0])
        if inliers is not None:
            m = np.asarray(inliers, dtype=bool).reshape(-1)
            if m.shape[0] == n:
                pxy = pxy[m]
                cxy = cxy[m]
                n = int(pxy.shape[0])
        if n <= 0:
            return
        step = int(max(1, n // int(max(50, max_lines))))
        pxy = pxy[::step]
        cxy = cxy[::step]
        for (x0, y0), (x1, y1) in zip(pxy, cxy):
            cv2.circle(im, (int(round(x0)), int(round(y0))), 2, (0, 255, 0), -1)
            cv2.circle(im, (int(round(x1)), int(round(y1))), 2, (0, 0, 255), -1)
            cv2.line(im, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), (255, 255, 0), 1)
        cv2.imwrite(out_png, im)
    except Exception:
        return


def _flow_to_hsv_bgr(flow: np.ndarray, *, clip_mag: float = 20.0) -> np.ndarray:
    """把光流转成 HSV 颜色图（BGR 输出）。"""
    f = np.asarray(flow, dtype=np.float32)
    if f.ndim != 3 or f.shape[2] != 2:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    fx = f[..., 0]
    fy = f[..., 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
    # 角度映射到 [0,180)
    h = (ang / 2.0).astype(np.uint8)
    # 幅值归一化
    cm = float(max(1e-6, clip_mag))
    v = np.clip((mag / cm) * 255.0, 0, 255).astype(np.uint8)
    s = np.full_like(v, 255, dtype=np.uint8)
    hsv = np.stack([h, s, v], axis=-1)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def _save_pair_overlay_pcd(
    *,
    out_path: str,
    pcd_prev: "o3d.geometry.PointCloud",
    pcd_cur_aligned: "o3d.geometry.PointCloud",
) -> None:
    """
    参考 Exp_Mehsh_Between_RT.py：
    保存“上一帧 vs 当前帧(对齐后)”的叠加点云：
      - prev: 绿色
      - cur_aligned: 红色
    """
    if o3d is None:
        return
    p0 = o3d.geometry.PointCloud(pcd_prev)
    p1 = o3d.geometry.PointCloud(pcd_cur_aligned)
    try:
        n0 = len(p0.points)
        n1 = len(p1.points)
        if n0 > 0:
            p0.colors = o3d.utility.Vector3dVector(np.tile(np.array([[0.0, 1.0, 0.0]], dtype=np.float64), (n0, 1)))
        if n1 > 0:
            p1.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float64), (n1, 1)))
    except Exception:
        pass
    out = p0 + p1
    o3d.io.write_point_cloud(out_path, out, write_ascii=False, compressed=False)


def _select_roi_tkinter(frame_bgr: np.ndarray, *, max_w: int = 1200, max_h: int = 900) -> Optional[tuple[int, int, int, int]]:
    """
    在 OpenCV HighGUI 不可用时的鼠标选框兜底实现（Tkinter）。
    返回 (x,y,w,h)（基于原始 frame_bgr 坐标系），取消则返回 None。
    """
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        return None

    H, W = frame_bgr.shape[:2]
    if H <= 1 or W <= 1:
        return None

    # 缩放显示，避免超出屏幕
    max_w = int(max(200, max_w))
    max_h = int(max(200, max_h))
    s = min(float(max_w) / float(W), float(max_h) / float(H), 1.0)
    disp_w = int(round(W * s))
    disp_h = int(round(H * s))
    disp = cv2.resize(frame_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA) if s < 0.999 else frame_bgr

    # Tkinter 的 PhotoImage 可直接从 png 文件加载；这里用临时文件避免依赖 Pillow
    tmp = None
    try:
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        ok, buf = cv2.imencode(".png", rgb)
        if not ok:
            return None
        fd, tmp = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())

        root = tk.Tk()
        root.title("MIS ROI Select (Tkinter) - 拖拽鼠标画框，松开确认，Esc取消")

        # 简单说明
        try:
            messagebox.showinfo("ROI选择", "拖拽鼠标画框，松开确认。\n按 Esc 取消。")
        except Exception:
            pass

        img = tk.PhotoImage(file=tmp)
        canvas = tk.Canvas(root, width=disp_w, height=disp_h)
        canvas.pack()
        canvas_img = canvas.create_image(0, 0, anchor="nw", image=img)

        state: dict[str, Any] = {
            "x0": None, "y0": None, "x1": None, "y1": None,
            "rect": None, "done": False, "cancel": False,
        }

        def on_down(ev):
            state["x0"], state["y0"] = int(ev.x), int(ev.y)
            state["x1"], state["y1"] = int(ev.x), int(ev.y)
            if state["rect"] is not None:
                canvas.delete(state["rect"])
            state["rect"] = canvas.create_rectangle(state["x0"], state["y0"], state["x1"], state["y1"], outline="lime", width=2)

        def on_move(ev):
            if state["x0"] is None:
                return
            state["x1"], state["y1"] = int(ev.x), int(ev.y)
            if state["rect"] is not None:
                canvas.coords(state["rect"], state["x0"], state["y0"], state["x1"], state["y1"])

        def on_up(ev):
            if state["x0"] is None:
                return
            state["x1"], state["y1"] = int(ev.x), int(ev.y)
            state["done"] = True
            root.quit()

        def on_esc(ev=None):
            state["cancel"] = True
            state["done"] = True
            root.quit()

        canvas.bind("<ButtonPress-1>", on_down)
        canvas.bind("<B1-Motion>", on_move)
        canvas.bind("<ButtonRelease-1>", on_up)
        root.bind("<Escape>", on_esc)

        root.mainloop()
        try:
            root.destroy()
        except Exception:
            pass

        if state["cancel"]:
            return None
        if state["x0"] is None or state["x1"] is None:
            return None

        x0 = int(max(0, min(disp_w - 1, state["x0"])))
        y0 = int(max(0, min(disp_h - 1, state["y0"])))
        x1 = int(max(0, min(disp_w - 1, state["x1"])))
        y1 = int(max(0, min(disp_h - 1, state["y1"])))
        x = int(min(x0, x1))
        y = int(min(y0, y1))
        w = int(abs(x1 - x0))
        h = int(abs(y1 - y0))
        if w <= 1 or h <= 1:
            return None
        # 映射回原图坐标
        if s <= 1e-8:
            return None
        ox = int(round(x / s))
        oy = int(round(y / s))
        ow = int(round(w / s))
        oh = int(round(h / s))
        return (ox, oy, ow, oh)
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def _maybe_add_sys_path(p: str) -> None:
    try:
        pp = str(p)
        if pp and os.path.isdir(pp) and (pp not in sys.path):
            sys.path.insert(0, pp)
    except Exception:
        pass


def _resolve_model_path(cfg: TrackConfig) -> str:
    """
    解决“默认模型路径盘符不一致”导致的误报。
    - 若 cfg.model_path 已存在：直接返回
    - 若用户通过 MIS_MODEL_PATH 指定但不存在：保持原样（让用户修）
    - 若未指定或默认值不存在：在常见候选路径中自动选择存在的那个
    """
    try:
        cur = str(cfg.model_path)
    except Exception:
        cur = ""
    if cur and os.path.isfile(cur):
        return cur

    # 如果用户显式设置了 MIS_MODEL_PATH，但文件不存在，则不自动改（避免惊喜）
    if os.environ.get("MIS_MODEL_PATH", "").strip():
        return cur

    candidates = [
        r"E:\reloc3r\Autoencoder_129.pth",
        r"D:\reloc3r\Autoencoder_129.pth",
        r"d:\reloc3r\Autoencoder_129.pth",
        os.path.join(os.getcwd(), "Autoencoder_129.pth"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            cfg.model_path = p
            return p
    return cur


def _init_semantic_module(cfg: TrackConfig) -> tuple[bool, str | None]:
    """
    尝试导入 extract_semantic_features_Function.calculate_contour_similarity。
    返回 (ok, error_str)。
    - 额外处理：如果报 Copy12 找不到，会把 model_path 所在目录（如 E:\\reloc3r）加入 sys.path 后重试。
    """
    global calculate_contour_similarity, SEMANTIC_AVAILABLE, SEMANTIC_IMPORT_ERROR
    SEMANTIC_AVAILABLE = False
    SEMANTIC_IMPORT_ERROR = None
    calculate_contour_similarity = None  # type: ignore

    # 1) 先加入一些常用搜索路径
    _maybe_add_sys_path(os.getcwd())
    _maybe_add_sys_path(os.path.dirname(os.path.abspath(__file__)))
    # 用户额外指定的目录
    if cfg.semantic_code_dirs:
        for part in cfg.semantic_code_dirs.replace(",", ";").split(";"):
            _maybe_add_sys_path(part.strip())
    # 模型所在目录（很多人把 Copy12.py 和权重放一起）
    try:
        _maybe_add_sys_path(os.path.dirname(str(cfg.model_path)))
    except Exception:
        pass

    def _try_import() -> None:
        # NOTE: 必须写回全局变量；否则会因 Python 作用域规则把赋值落到局部变量上，
        # 进而导致 SEMANTIC_AVAILABLE=True 但 calculate_contour_similarity 仍为 None。
        global calculate_contour_similarity
        mod = importlib.import_module("extract_semantic_features_Function")
        fn = getattr(mod, "calculate_contour_similarity", None)
        if fn is None:
            raise AttributeError("extract_semantic_features_Function 中没有 calculate_contour_similarity")
        calculate_contour_similarity = fn  # type: ignore

    try:
        _try_import()
        SEMANTIC_AVAILABLE = True
        return True, None
    except Exception as e1:
        # 若是 Copy12 缺失，尝试把模型目录/用户指定目录加进 sys.path 后再重试（上面已加过，这里保留结构清晰）
        err1 = "".join(traceback.format_exception_only(type(e1), e1)).strip()
        # 再试一次（有些情况下第一次 import 失败后路径刚加入，需要二次 import）
        try:
            if "No module named 'Copy12'" in err1 or "No module named 'Copy12'" in str(e1):
                _maybe_add_sys_path(os.path.dirname(str(cfg.model_path)))
            _try_import()
            SEMANTIC_AVAILABLE = True
            return True, None
        except Exception as e2:
            SEMANTIC_IMPORT_ERROR = "".join(traceback.format_exception_only(type(e2), e2)).strip()
            return False, SEMANTIC_IMPORT_ERROR


def _device_str(cfg: TrackConfig) -> str:
    d = (cfg.device or "auto").strip().lower()
    if d == "auto":
        if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if d in ("cuda", "cpu"):
        if d == "cuda" and (torch is None or (hasattr(torch, "cuda") and not torch.cuda.is_available())):
            return "cpu"
        return d
    return "cpu"


def clamp_bbox(x: int, y: int, w: int, h: int, W: int, H: int) -> tuple[int, int, int, int]:
    w = max(2, int(w))
    h = max(2, int(h))
    x = int(max(0, min(int(x), W - w)))
    y = int(max(0, min(int(y), H - h)))
    return x, y, w, h


def _resize_for_processing(img_bgr: np.ndarray, cfg: TrackConfig) -> np.ndarray:
    """按 cfg.proc_scale 缩放图像，用于整体提速。"""
    s = float(getattr(cfg, "proc_scale", 1.0))
    if not np.isfinite(s) or s <= 0:
        s = 1.0
    if abs(s - 1.0) < 1e-6:
        return img_bgr
    H, W = img_bgr.shape[:2]
    nw = int(max(2, round(W * s)))
    nh = int(max(2, round(H * s)))
    return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def _read_frame_bgr(path: str, cfg: TrackConfig) -> Optional[np.ndarray]:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    return _resize_for_processing(im, cfg)


def crop(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return frame_bgr[y:y + h, x:x + w].copy()


def compute_th_gray_truncated(roi_bgr: np.ndarray, percent: float, cfg: TrackConfig) -> tuple[np.ndarray, np.ndarray, float]:
    """
    返回：
    - th_gray_truncated: float32, 0~255，灰度截断图（不是二值）
    - mask_u8: uint8, 0/255（th_68 二值阈值图，用作 mask）
    - th_uint8: 阈值（0~255）
    """
    # 关键：双边滤波的 sigmaColor/sigmaSpace 是按“像素强度尺度”工作的。
    # 参考代码通常在 uint8(0~255) 上做双边，因此这里也先用 uint8 做 bilateral，
    # 避免在 0~1 浮点上使用 sigmaColor=75 导致滤波几乎退化的问题。
    gray_u8 = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    f_u8 = cv2.bilateralFilter(gray_u8, cfg.bilateral_d, cfg.bilateral_sc, cfg.bilateral_ss)
    f = f_u8.astype(np.float32)
    # Sobel 梯度幅值
    dx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(dx, dy)
    mmax = float(np.max(mag)) if mag.size else 0.0
    if mmax <= 1e-8:
        grad_u8 = np.zeros_like(gray_u8, dtype=np.uint8)
        th_u8 = 0.0
    else:
        mag_n = np.clip(mag / mmax, 0.0, 1.0)
        grad_u8 = np.clip(mag_n * 255.0, 0, 255).astype(np.uint8)
        th_u8 = float(np.percentile(grad_u8.reshape(-1), float(percent)))
    th_uint8 = float(max(0.0, min(255.0, th_u8)))
    # th_68 二值图（严格意义上的阈值结果）：作为 mask 使用
    # 轻微形态学闭运算，增强连通性（便于轮廓稳定）
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = np.where(grad_u8 > th_uint8, 255, 0).astype(np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k, iterations=1)
    # th_68 灰度截断图：用“形态学后的 mask”对梯度做截断（避免 mask 与 th_gray 不一致导致 mesh 高度稀疏/断裂）
    th_gray_truncated = np.where(mask_u8 > 0, grad_u8, 0).astype(np.float32)
    return th_gray_truncated, mask_u8, th_uint8


def _save_similarity_compare_image(
    out_path: str,
    *,
    prev_contour_rgb: np.ndarray,
    cur_contour_rgb: np.ndarray,
    similarity: float,
    title_left: str = "prev",
    title_right: str = "cur",
) -> None:
    """
    保存一张对比图，把“转换相似度/欧氏距离(由相似度反推)”写在图上，方便直接肉眼看到数值。
    similarity 按 extract_semantic_features_Function 的定义：sim = 1/(1+dist)
    """
    try:
        sim = float(similarity)
        sim = 0.0 if (not np.isfinite(sim)) else float(max(0.0, min(1.0, sim)))
        # 当 sim≈0 时，反推 d=1/sim-1 会爆到极大值（例如 1e8），对诊断图没有意义；
        # 这里把“极小相似度”显示为 inf，避免出现 99999999 这类误导数字。
        if sim <= 1e-6:
            dist = float("inf")
        else:
            dist = (1.0 / sim) - 1.0
    except Exception:
        sim = 0.0
        dist = float("inf")

    a = np.asarray(prev_contour_rgb, dtype=np.uint8)
    b = np.asarray(cur_contour_rgb, dtype=np.uint8)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if b.ndim == 2:
        b = np.stack([b, b, b], axis=-1)
    # 我们内部 contour_rgb 是“灰度三通道”，这里当作 BGR 直接写即可
    h = max(a.shape[0], b.shape[0])
    w = max(a.shape[1], b.shape[1])
    a2 = cv2.resize(a, (w, h), interpolation=cv2.INTER_NEAREST) if (a.shape[0] != h or a.shape[1] != w) else a
    b2 = cv2.resize(b, (w, h), interpolation=cv2.INTER_NEAREST) if (b.shape[0] != h or b.shape[1] != w) else b
    canvas = np.hstack([a2, b2]).copy()

    # 文本背景条
    bar_h = 70
    out = np.zeros((h + bar_h, canvas.shape[1], 3), dtype=np.uint8)
    out[bar_h:, :] = canvas
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, bar_h - 1), (20, 20, 20), -1)

    cv2.putText(out, f"{title_left}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
    cv2.putText(out, f"{title_right}", (w + 10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
    if np.isfinite(dist):
        dist_txt = f"{dist:.4f}"
    else:
        dist_txt = "inf"
    cv2.putText(out, f"distance={dist_txt}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 240, 180), 2, cv2.LINE_AA)
    # 对齐你提到的文案：转换相似度
    cv2.putText(out, f"converted_similarity={sim:.4f}", (300, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 240, 180), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, out)


def extract_largest_contour(mask_u8: np.ndarray) -> Optional[np.ndarray]:
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    if cv2.contourArea(cnts[0]) <= 5.0:
        return None
    return cnts[0]


def contour_to_rgb(contour: Optional[np.ndarray], size: tuple[int, int], thickness: int = 2) -> np.ndarray:
    """
    将轮廓栅格化为 RGB 图（用于语义模型输入）。
    """
    w, h = int(size[0]), int(size[1])
    canvas = np.zeros((h, w), dtype=np.uint8)
    if contour is not None and len(contour) >= 3:
        # contour 坐标一般在 ROI 内；这里假定调用方已把它缩放到 canvas 参考系
        cv2.drawContours(canvas, [contour], -1, 255, thickness=thickness)
    rgb = np.stack([canvas, canvas, canvas], axis=-1)
    return rgb


def resize_contour(contour: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]) -> np.ndarray:
    """
    将 ROI 内轮廓从 src_wh 缩放到 dst_wh。
    src_wh=(w,h), dst_wh=(w,h)
    """
    sw, sh = float(src_wh[0]), float(src_wh[1])
    dw, dh = float(dst_wh[0]), float(dst_wh[1])
    sx = (dw - 1.0) / max(1.0, (sw - 1.0))
    sy = (dh - 1.0) / max(1.0, (sh - 1.0))
    c = contour.astype(np.float32).copy()
    c[:, 0, 0] = c[:, 0, 0] * sx
    c[:, 0, 1] = c[:, 0, 1] * sy
    c[:, 0, 0] = np.clip(c[:, 0, 0], 0, dw - 1.0)
    c[:, 0, 1] = np.clip(c[:, 0, 1], 0, dh - 1.0)
    return c.astype(np.int32)


def quick_mask_score(prev_mask_u8: np.ndarray, cur_mask_u8: np.ndarray) -> float:
    """
    轻量粗筛分数（0~1）：mask IoU（在低分辨率下计算，抗噪/更快）。
    """
    a = (prev_mask_u8 > 0).astype(np.uint8)
    b = (cur_mask_u8 > 0).astype(np.uint8)
    inter = int(np.sum((a & b) > 0))
    union = int(np.sum((a | b) > 0))
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def fallback_similarity(img1_rgb: np.ndarray, img2_rgb: np.ndarray) -> float:
    """
    当语义模型不可用时的回退相似度（0~1）：
    - 用 mask 的相关性/IoU 做近似。
    """
    a = (img1_rgb[..., 0] > 0).astype(np.float32)
    b = (img2_rgb[..., 0] > 0).astype(np.float32)
    # 相关系数 + IoU 组合（更稳）
    aa = a.reshape(-1)
    bb = b.reshape(-1)
    am = float(np.mean(aa))
    bm = float(np.mean(bb))
    av = float(np.mean((aa - am) ** 2))
    bv = float(np.mean((bb - bm) ** 2))
    if av <= 1e-8 or bv <= 1e-8:
        corr = 0.0
    else:
        corr = float(np.mean((aa - am) * (bb - bm)) / math.sqrt(av * bv))
        corr = max(-1.0, min(1.0, corr))
        corr = 0.5 * (corr + 1.0)
    inter = float(np.sum((a > 0) & (b > 0)))
    union = float(np.sum((a > 0) | (b > 0))) + 1e-8
    iou = inter / union
    return float(0.55 * iou + 0.45 * corr)


def semantic_similarity(img1_rgb: np.ndarray, img2_rgb: np.ndarray, cfg: TrackConfig) -> float:
    """
    与参考脚本一致的调用风格；若不可用则回退。
    """
    # 这里不重复做文件存在检查（避免每候选调用一次 os.path.isfile）；
    # 是否启用在 main() 启动阶段一次性决定（SEMANTIC_ENABLED）。
    if SEMANTIC_ENABLED and calculate_contour_similarity is not None:
        try:
            dev = _device_str(cfg)
            s = float(
                calculate_contour_similarity(
                    contour1=img1_rgb,
                    contour2=img2_rgb,
                    model_path=str(cfg.model_path),
                    image_size=tuple(cfg.semantic_image_size),
                    device=dev,
                )
            )
            if not np.isfinite(s):
                return 0.0
            return float(max(0.0, min(1.0, s)))
        except Exception:
            return fallback_similarity(img1_rgb, img2_rgb)
    return fallback_similarity(img1_rgb, img2_rgb)


def _contour_rgb_from_full_mask(mask_u8: np.ndarray, full_wh: tuple[int, int], cfg: TrackConfig) -> np.ndarray:
    """
    将“整帧坐标系”的 mask 提取最大轮廓，并缩放栅格化为语义网络输入尺寸。
    full_wh=(W,H) 为 mask 所在坐标系的宽高（通常是处理后分辨率）。
    """
    try:
        W, H = int(full_wh[0]), int(full_wh[1])
        cnt = extract_largest_contour(mask_u8)
        if cnt is None:
            return np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
        cnt2 = resize_contour(cnt, (W, H), cfg.semantic_image_size)
        return contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)
    except Exception:
        return np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)


def _render_aligned_contour_rgb(
    *,
    cur_mask_u8: np.ndarray,
    cur_th_gray: np.ndarray,
    T_prev_cur: np.ndarray,
    full_wh: tuple[int, int],
    cfg: TrackConfig,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    用 T_prev<-cur 把“当前帧整帧轮廓(uv+h)”对齐到上一帧坐标系，然后栅格化成 contour_rgb。
    返回 (aligned_contour_rgb, aligned_canvas_u8_or_None)。
    - aligned_canvas_u8 是 full-res 的二值渲染结果（可选保存，用于排查）。
    """
    W, H = int(full_wh[0]), int(full_wh[1])
    out_blank = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
    try:
        cnt = extract_largest_contour(cur_mask_u8)
        if cnt is None or len(cnt) < 3:
            return out_blank, None

        # contour 点 (u,v) 在整帧坐标系里
        uv = cnt.reshape(-1, 2).astype(np.float32)
        u = np.clip(np.round(uv[:, 0]).astype(np.int32), 0, W - 1)
        v = np.clip(np.round(uv[:, 1]).astype(np.int32), 0, H - 1)

        th = np.asarray(cur_th_gray, dtype=np.float32)
        if th.ndim != 2:
            return out_blank, None
        if th.shape[0] != H or th.shape[1] != W:
            th = cv2.resize(th, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        h = (th[v, u] / 255.0) * float(cfg.mesh_z_scale)

        pts = np.stack([u.astype(np.float32), v.astype(np.float32), h.astype(np.float32)], axis=1)
        ones = np.ones((pts.shape[0], 1), dtype=np.float32)
        pts4 = np.concatenate([pts, ones], axis=1)  # Nx4
        Tp = np.asarray(T_prev_cur, dtype=np.float64).reshape(4, 4)
        pts4_al = (Tp @ pts4.T).T
        ua = np.round(pts4_al[:, 0]).astype(np.int32)
        va = np.round(pts4_al[:, 1]).astype(np.int32)

        valid = (ua >= 0) & (ua < W) & (va >= 0) & (va < H)
        if int(np.sum(valid)) < 10:
            return out_blank, None
        ua = ua[valid]
        va = va[valid]

        cnt_al = np.stack([ua, va], axis=1).reshape(-1, 1, 2).astype(np.int32)
        canvas = np.zeros((H, W), dtype=np.uint8)
        # 直接把对齐后的轮廓画出来（等价于“渲染轮廓”）
        cv2.drawContours(canvas, [cnt_al], -1, 255, thickness=2)
        # 轻微闭运算让线更连续
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, k, iterations=1)

        cnt2 = extract_largest_contour(canvas)
        if cnt2 is None:
            return out_blank, canvas
        cnt2s = resize_contour(cnt2, (W, H), cfg.semantic_image_size)
        rgb = contour_to_rgb(cnt2s, cfg.semantic_image_size, thickness=2)
        return rgb, canvas
    except Exception:
        return out_blank, None


def make_heightfield_mesh_from_th_gray(
    th_gray_truncated: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    z_scale: float,
) -> Optional["o3d.geometry.TriangleMesh"]:
    """
    用 th_68 灰度截断图构建伪3D高度场 mesh（在整帧坐标系里：x=u+bx, y=v+by, z=th/255*z_scale）。
    - 只对 mask>0 的像素创建顶点，并用规则网格三角化（背景像素会导致断裂，所以用 mask 过滤三角形）。
    """
    if o3d is None:
        return None
    bx, by, w, h = bbox
    img = np.asarray(th_gray_truncated, dtype=np.float32)
    if img.ndim != 2:
        return None
    hh, ww = img.shape[:2]
    if hh != h or ww != w:
        # 保险：按 bbox 尺寸一致化
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        hh, ww = img.shape[:2]

    m = img > 0
    if int(np.sum(m)) < 50:
        return None

    # 为每个像素分配 vertex id（背景=-1）
    vid = -np.ones((hh, ww), dtype=np.int32)
    idx = np.flatnonzero(m.reshape(-1))
    vid.reshape(-1)[idx] = np.arange(len(idx), dtype=np.int32)

    ys, xs = np.nonzero(m)
    z = (img[ys, xs] / 255.0) * float(z_scale)
    vx = xs.astype(np.float32) + float(bx)
    vy = ys.astype(np.float32) + float(by)
    verts = np.stack([vx, vy, z.astype(np.float32)], axis=1)

    # 规则网格三角化：对每个 2x2 cell，如果四角都在 mask 内则连成两个三角形
    tris = []
    for y in range(hh - 1):
        vrow = vid[y]
        vrow2 = vid[y + 1]
        for x in range(ww - 1):
            v00 = int(vrow[x])
            v10 = int(vrow[x + 1])
            v01 = int(vrow2[x])
            v11 = int(vrow2[x + 1])
            if v00 < 0 or v10 < 0 or v01 < 0 or v11 < 0:
                continue
            tris.append([v00, v10, v11])
            tris.append([v00, v11, v01])

    if len(tris) < 50:
        return None

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(tris, dtype=np.int32))
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    try:
        mesh.compute_vertex_normals()
    except Exception:
        pass
    # 用高度上色（便于 debug / colored icp）
    try:
        v = np.asarray(mesh.vertices, dtype=np.float64)
        z0 = v[:, 2]
        zmin = float(np.min(z0))
        zmax = float(np.max(z0))
        den = (zmax - zmin) if (zmax - zmin) > 1e-9 else 1.0
        zn = (z0 - zmin) / den
        col = np.stack([zn, 1.0 - zn, 0.2 + 0.0 * zn], axis=1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(col.astype(np.float64))
    except Exception:
        pass
    return mesh


def _mesh_to_pcd(mesh: "o3d.geometry.TriangleMesh", cfg: TrackConfig) -> "o3d.geometry.PointCloud":
    n = int(max(0, int(cfg.sample_points)))
    if n <= 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        if mesh.has_vertex_normals():
            pcd.normals = mesh.vertex_normals
    else:
        # 参考 Exp_Mehsh_Between_RT.py：优先 poisson disk 采样（更均匀），失败则回退 uniform
        try:
            if bool(cfg.sample_poisson):
                pcd = mesh.sample_points_poisson_disk(number_of_points=int(n), init_factor=5)
            else:
                pcd = mesh.sample_points_uniformly(number_of_points=int(n))
        except Exception:
            pcd = mesh.sample_points_uniformly(number_of_points=int(n))
    if float(cfg.voxel_size) > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(cfg.voxel_size))
    # normals for point-to-plane
    try:
        rad = float(max(2.0, float(cfg.voxel_size) * 2.0))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=30))
    except Exception:
        pass
    return pcd


def estimate_T_prev_cur_icp(
    mesh_prev: Optional["o3d.geometry.TriangleMesh"],
    mesh_cur: Optional["o3d.geometry.TriangleMesh"],
    cfg: TrackConfig,
    *,
    init_T: Optional[np.ndarray] = None,
    max_corr: Optional[float] = None,
    max_iter: Optional[int] = None,
    method: str = "point_to_plane",
) -> Optional[RegResult]:
    """
    返回 T_{prev<-cur}（4x4）：把 cur 的点变换到 prev 坐标系。
    """
    if o3d is None or mesh_prev is None or mesh_cur is None:
        return None
    try:
        src = _mesh_to_pcd(mesh_cur, cfg)
        tgt = _mesh_to_pcd(mesh_prev, cfg)
        if len(src.points) < 300 or len(tgt.points) < 300:
            return None

        method_l = (method or "point_to_plane").strip().lower()
        if method_l == "point_to_plane":
            est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
        else:
            est = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        mc = float(cfg.icp_max_corr if max_corr is None else max_corr)
        itn = int(cfg.icp_max_iter if max_iter is None else max_iter)
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max(5, itn)))
        T0 = np.eye(4, dtype=np.float64) if init_T is None else np.asarray(init_T, dtype=np.float64).reshape(4, 4)
        icp = o3d.pipelines.registration.registration_icp(
            src, tgt, float(max(1e-6, mc)), T0, est, criteria
        )
        T = np.asarray(icp.transformation, dtype=np.float64).reshape(4, 4)
        if not np.isfinite(T).all():
            return None
        fit = float(getattr(icp, "fitness", 0.0))
        rmse = float(getattr(icp, "inlier_rmse", float("inf")))
        return RegResult(T=T, fitness=fit, rmse=rmse, method=f"icp_{method_l}")
    except Exception:
        return None


def estimate_reg_with_trials(
    *,
    mesh_prev: Optional["o3d.geometry.TriangleMesh"],
    mesh_cur: Optional["o3d.geometry.TriangleMesh"],
    cfg: TrackConfig,
    init_T: Optional[np.ndarray],
) -> tuple[RegResult, str]:
    """
    多次尝试 ICP，选择更优结果（参考 Exp_Mehsh_Between_RT.py 的“try better”策略）。
    返回 (best_reg, init_mode_tag)。
    """
    # 基础一次（tracking icp）
    base = estimate_T_prev_cur_icp(mesh_prev, mesh_cur, cfg, init_T=init_T)
    if base is None:
        base = RegResult(T=np.eye(4, dtype=np.float64), fitness=0.0, rmse=float("inf"), method="icp_fail")
    best = base
    init_mode = "tracking_init"

    if not cfg.icp_multitrial:
        return best, init_mode

    corr_mults = _parse_float_list_csv(cfg.icp_corr_mults, [1.0, 1.8, 2.2])
    iter_mults = _parse_float_list_csv(cfg.icp_iter_mults, [1.0, 1.5, 2.0])
    # 尝试更宽松的 max_corr / 更大迭代次数
    for cm in corr_mults:
        for im in iter_mults:
            mc = float(cfg.icp_max_corr) * float(max(0.2, cm))
            itn = int(max(10, round(float(cfg.icp_max_iter) * float(max(0.2, im)))))
            rr = estimate_T_prev_cur_icp(mesh_prev, mesh_cur, cfg, init_T=init_T, max_corr=mc, max_iter=itn)
            if rr is not None and _better_than(rr, best):
                best = rr
                init_mode = f"tracking_init_try(cm={cm:.2f},im={im:.2f})"

    # 如果 point-to-plane 很差，再试 point-to-point 作为兜底（在形变/法向不稳时有时更好）
    if _need_recovery(best.fitness, best.rmse, cfg):
        rr2 = estimate_T_prev_cur_icp(mesh_prev, mesh_cur, cfg, init_T=init_T, method="point_to_point")
        if rr2 is not None and _better_than(rr2, best):
            best = rr2
            init_mode = "tracking_init_p2p"

    return best, init_mode


def draw_bbox(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int], color=(0, 255, 0), text: str | None = None) -> np.ndarray:
    x, y, w, h = bbox
    out = frame_bgr.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
    if text:
        cv2.putText(out, str(text), (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def make_overlay_mask(roi_bgr: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8)
    col = roi_bgr.copy()
    tint = np.zeros_like(col)
    tint[..., 1] = 180
    out = cv2.addWeighted(col, 0.75, tint, 0.25, 0.0)
    out[m == 0] = col[m == 0]
    return out


def find_best_bbox_in_next_frame(
    frame_bgr: np.ndarray,
    prev_bbox: tuple[int, int, int, int],
    prev_contour_rgb: np.ndarray,
    prev_mask_small: np.ndarray,
    cfg: TrackConfig,
) -> dict[str, Any]:
    """
    返回 dict：
      bbox, score, semantic, quick, dx, dy, th_gray, mask_u8, contour_rgb, th_uint8
    """
    H, W = frame_bgr.shape[:2]
    px, py, pw, ph = prev_bbox
    cx0 = px + pw // 2
    cy0 = py + ph // 2

    step = int(max(1, cfg.search_step))
    rad = int(max(step, cfg.search_radius))

    # 生成候选 bbox（保持 w/h 不变），并在边界 clamp 后去重（避免重复候选浪费算力）
    cands_xy: dict[tuple[int, int], tuple[int, int, int, int, int, int]] = {}
    for dy in range(-rad, rad + 1, step):
        for dx in range(-rad, rad + 1, step):
            x0 = cx0 + dx - pw // 2
            y0 = cy0 + dy - ph // 2
            x, y, ww, hh = clamp_bbox(x0, y0, pw, ph, W, H)
            # clamp 后的真实偏移（用于空间惩罚）
            dx_eff = int((x + ww // 2) - cx0)
            dy_eff = int((y + hh // 2) - cy0)
            key = (int(x), int(y))
            cands_xy[key] = (int(x), int(y), int(ww), int(hh), int(dx_eff), int(dy_eff))
    cands = list(cands_xy.values())

    # 粗筛：在“缩小ROI”上计算 th_68/mask，提高速度
    scored_quick = []
    for (x, y, ww, hh, dx, dy) in cands:
        roi = frame_bgr[y:y + hh, x:x + ww]
        # 粗筛ROI缩放
        qs = float(max(0.1, min(1.0, cfg.quick_scale)))
        qw = int(max(16, round(ww * qs)))
        qh = int(max(16, round(hh * qs)))
        qmin = int(max(16, cfg.quick_min_size))
        # 保证至少有一边达到 quick_min_size（避免ROI过小导致mask全噪）
        if min(qw, qh) < qmin:
            s2 = float(qmin) / float(max(1, min(qw, qh)))
            qw = int(round(qw * s2))
            qh = int(round(qh * s2))
        roi_small = cv2.resize(roi, (qw, qh), interpolation=cv2.INTER_AREA)
        _, mask_small_u8, _ = compute_th_gray_truncated(roi_small, cfg.th_percent, cfg)
        # 统一到 64x64 做 IoU（与 prev_mask_small 对齐）
        ms = cv2.resize(mask_small_u8, (64, 64), interpolation=cv2.INTER_NEAREST)
        q = quick_mask_score(prev_mask_small, ms)
        # 空间惩罚（越远越差）
        dist = (abs(dx) + abs(dy)) / float(max(1, rad))
        q2 = float(q) - float(cfg.spatial_penalty) * float(dist)
        scored_quick.append((q2, q, x, y, ww, hh, dx, dy))

    scored_quick.sort(key=lambda t: t[0], reverse=True)
    top = scored_quick[: max(1, int(cfg.topk_semantic))]

    best = None
    best_score = -1e9
    # 精排：对 top-K 重新用“全分辨率ROI”计算 th_68/mask/轮廓/语义相似度
    for (q2, q, x, y, ww, hh, dx, dy) in top:
        roi = frame_bgr[y:y + hh, x:x + ww]
        th_gray, mask_u8, th_uint8 = compute_th_gray_truncated(roi, cfg.th_percent, cfg)
        cnt = extract_largest_contour(mask_u8)
        if cnt is None:
            sem = 0.0
            contour_rgb = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
        else:
            cnt2 = resize_contour(cnt, (ww, hh), cfg.semantic_image_size)
            contour_rgb = contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)
            sem = semantic_similarity(prev_contour_rgb, contour_rgb, cfg)
        # 融合：语义为主，quick 为辅 + 空间先验
        score = float(0.85 * sem + 0.15 * max(0.0, q)) - float(cfg.spatial_penalty) * ((abs(dx) + abs(dy)) / float(max(1, rad)))
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
            }

    # 恢复：若分数很低，扩大搜索范围再试一次（保持默认开）
    if cfg.recover_enabled and (best is None or float(best.get("score", 0.0)) < float(cfg.recover_score)):
        rad2 = int(max(rad * 2, rad + step))
        # 递归式重跑会更复杂；这里用一次性扩大候选集合，仍走“粗筛+精排”
        cands_xy2: dict[tuple[int, int], tuple[int, int, int, int, int, int]] = {}
        for dy in range(-rad2, rad2 + 1, step):
            for dx in range(-rad2, rad2 + 1, step):
                x0 = cx0 + dx - pw // 2
                y0 = cy0 + dy - ph // 2
                x, y, ww, hh = clamp_bbox(x0, y0, pw, ph, W, H)
                dx_eff = int((x + ww // 2) - cx0)
                dy_eff = int((y + hh // 2) - cy0)
                cands_xy2[(int(x), int(y))] = (int(x), int(y), int(ww), int(hh), int(dx_eff), int(dy_eff))
        scored_quick2 = []
        for (x, y, ww, hh, dx, dy) in cands_xy2.values():
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
            _, mask_small_u8, _ = compute_th_gray_truncated(roi_small, cfg.th_percent, cfg)
            ms = cv2.resize(mask_small_u8, (64, 64), interpolation=cv2.INTER_NEAREST)
            q = quick_mask_score(prev_mask_small, ms)
            dist = (abs(dx) + abs(dy)) / float(max(1, rad2))
            q2 = float(q) - float(cfg.spatial_penalty) * float(dist)
            scored_quick2.append((q2, q, x, y, ww, hh, dx, dy))
        scored_quick2.sort(key=lambda t: t[0], reverse=True)
        top2 = scored_quick2[: max(1, int(cfg.topk_semantic))]
        for (q2, q, x, y, ww, hh, dx, dy) in top2:
            roi = frame_bgr[y:y + hh, x:x + ww]
            th_gray, mask_u8, th_uint8 = compute_th_gray_truncated(roi, cfg.th_percent, cfg)
            cnt = extract_largest_contour(mask_u8)
            if cnt is None:
                sem = 0.0
                contour_rgb = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
            else:
                cnt2 = resize_contour(cnt, (ww, hh), cfg.semantic_image_size)
                contour_rgb = contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)
                sem = semantic_similarity(prev_contour_rgb, contour_rgb, cfg)
            score = float(0.85 * sem + 0.15 * max(0.0, q)) - float(cfg.spatial_penalty) * (
                (abs(dx) + abs(dy)) / float(max(1, rad2))
            )
            if best is None or score > float(best.get("score", -1e9)):
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
                }

    if best is None:
        # 极端兜底：直接用 prev_bbox
        roi = crop(frame_bgr, prev_bbox)
        th_gray, mask_u8, th_uint8 = compute_th_gray_truncated(roi, cfg.th_percent, cfg)
        cnt = extract_largest_contour(mask_u8)
        if cnt is not None:
            cnt2 = resize_contour(cnt, (prev_bbox[2], prev_bbox[3]), cfg.semantic_image_size)
            contour_rgb = contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)
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
        }
    return best


def save_rt_txt(path: str, pairwise: dict[int, np.ndarray], title: str) -> None:
    lines = [f"# {title}", f"# time={_now()}", ""]
    for k in sorted(pairwise.keys()):
        T = np.asarray(pairwise[k], dtype=np.float64).reshape(4, 4)
        lines.append(f"[frame {int(k)}]")
        for r in range(4):
            lines.append(" ".join([f"{T[r, c]: .8f}" for c in range(4)]))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_camera_poses_and_centers(
    *,
    out_dir: str,
    cumulative_T: dict[int, np.ndarray],
    pose_interp: str,
) -> None:
    """
    参考 Exp_Mehsh_Between_RT.py：
    从累计 shape RT (T_{0<-t}) 导出“相机位姿/相机中心”文本。

    注意：这是基于伪3D高度场(uvh)的形状对齐变换推导出来的“伪相机位姿”，并不保证等价真实相机位姿。
    """
    try:
        pose_interp = (pose_interp or "camera").strip().lower()
        if pose_interp not in ("camera", "object"):
            pose_interp = "camera"
        keys = sorted([int(k) for k in cumulative_T.keys()])
        if len(keys) == 0:
            return

        cam_pose_path = os.path.join(out_dir, "camera_poses_c2w_4x4.txt")
        cam_center_path = os.path.join(out_dir, "camera_centers.txt")
        pseudo_pose_path = os.path.join(out_dir, "pseudo_camera_poses_c2w_4x4.txt")
        pseudo_center_path = os.path.join(out_dir, "pseudo_camera_centers.txt")

        def _write_pose_and_centers(pose_path: str, center_path: str, header_prefix: str) -> None:
            with open(pose_path, "w", encoding="utf-8") as f:
                f.write(f"# {header_prefix} camera pose (c2w)\n")
                f.write("# WARNING: derived from shape-alignment RT in (u,v,h). NOT a physical camera pose.\n")
                f.write(f"# POSE_INTERPRETATION={pose_interp}\n")
                f.write("# coordinate_system=uvh_heightfield\n\n")
                for fi in keys:
                    T0_raw = np.asarray(cumulative_T[int(fi)], dtype=np.float64).reshape(4, 4)
                    # Exp_Mehsh_Between_RT.py：object 模式取逆
                    T0 = np.linalg.inv(T0_raw) if pose_interp == "object" else T0_raw
                    f.write(f"frame_idx={int(fi)}\n")
                    for r in range(4):
                        f.write(" ".join([f"{T0[r, c]:.10f}" for c in range(4)]) + "\n")
                    f.write("\n")
            with open(center_path, "w", encoding="utf-8") as f:
                f.write(f"# {header_prefix} frame_idx cx cy cz\n")
                f.write("# WARNING: pseudo centers from uvh shape-alignment.\n")
                for fi in keys:
                    T0_raw = np.asarray(cumulative_T[int(fi)], dtype=np.float64).reshape(4, 4)
                    T0 = np.linalg.inv(T0_raw) if pose_interp == "object" else T0_raw
                    c = T0[:3, 3].reshape(3)
                    f.write(f"{int(fi)} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")

        # 为兼容旧习惯，提供 LEGACY + PSEUDO 两套文件名（两者内容一致，前缀不同）
        _write_pose_and_centers(cam_pose_path, cam_center_path, "LEGACY")
        _write_pose_and_centers(pseudo_pose_path, pseudo_center_path, "PSEUDO")
    except Exception:
        # 不阻断主流程
        return


def save_pose_visualization_outputs(
    *,
    out_dir: str,
    cumulative_T: dict[int, np.ndarray],
    pose_interp: str,
    cfg: TrackConfig,
) -> None:
    """
    参考 Exp_Mehsh_Between_RT.py：保存/展示位姿可视化结果。

    输出到：<out_dir>\\rt_vis\\
      - trajectory_lines.ply
      - camera_frustums.ply（可选）
    并可选弹 Open3D 窗口展示。
    """
    if o3d is None:
        return
    try:
        pose_interp = (pose_interp or "camera").strip().lower()
        if pose_interp not in ("camera", "object"):
            pose_interp = "camera"
        keys = sorted([int(k) for k in cumulative_T.keys()])
        if len(keys) == 0:
            return

        Ts = []
        centers = []
        for fi in keys:
            T0_raw = np.asarray(cumulative_T[int(fi)], dtype=np.float64).reshape(4, 4)
            T0 = np.linalg.inv(T0_raw) if pose_interp == "object" else T0_raw
            Ts.append(T0)
            centers.append(T0[:3, 3].reshape(3))
        C = np.asarray(centers, dtype=np.float64)

        vis_dir = _ensure_dir(os.path.join(out_dir, "rt_vis"))

        # frame size
        fs = cfg.vis_frame_size
        if fs == "auto":
            diag = float(np.linalg.norm(np.max(C, axis=0) - np.min(C, axis=0))) if C.shape[0] >= 2 else 50.0
            frame_size = float(max(1.0, diag * 0.03))
        else:
            try:
                frame_size = float(fs)
            except Exception:
                frame_size = 30.0

        # trajectory line set
        traj = _trajectory_lineset(C, color_rgb=(0.2, 0.9, 1.0))
        try:
            o3d.io.write_line_set(os.path.join(vis_dir, "trajectory_lines.ply"), traj, write_ascii=False, compressed=False)
        except Exception:
            pass

        geoms: list[Any] = [traj]

        # frustums
        if cfg.vis_show_camera_frustum:
            frustums = []
            every = int(max(1, cfg.vis_rt_every))
            for idx, T0 in enumerate(Ts):
                if (idx % every) != 0:
                    continue
                fr = _make_camera_frustum_lineset(scale=frame_size * 1.2)
                fr.transform(np.asarray(T0, dtype=np.float64))
                frustums.append(fr)
            if frustums:
                fr_all = _merge_linesets(frustums)
                try:
                    o3d.io.write_line_set(os.path.join(vis_dir, "camera_frustums.ply"), fr_all, write_ascii=False, compressed=False)
                except Exception:
                    pass
                geoms.append(fr_all)

        # coordinate frames
        if cfg.vis_show_rt_frames:
            every = int(max(1, cfg.vis_rt_every))
            for idx, T0 in enumerate(Ts):
                if (idx % every) != 0:
                    continue
                try:
                    cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(frame_size), origin=[0, 0, 0])
                    cf.transform(np.asarray(T0, dtype=np.float64))
                    geoms.append(cf)
                except Exception:
                    pass

        if cfg.vis_open3d:
            try:
                o3d.visualization.draw_geometries(geoms, window_name="Pose Visualization (uvh shape-alignment)")
            except Exception:
                pass
    except Exception:
        return


def _make_camera_frustum_lineset(
    *,
    scale: float,
    color_rgb: tuple[float, float, float] = (1.0, 0.6, 0.1),
) -> "o3d.geometry.LineSet":
    s = float(max(1e-6, scale))
    z = 1.0 * s
    a = 0.6 * s
    pts = np.array([[0.0, 0.0, 0.0], [-a, -a, z], [a, -a, z], [a, a, z], [-a, a, z]], dtype=np.float64)
    lines = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]], dtype=np.int32)
    ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(pts), lines=o3d.utility.Vector2iVector(lines))
    try:
        col = np.tile(np.array([color_rgb], dtype=np.float64), (lines.shape[0], 1))
        ls.colors = o3d.utility.Vector3dVector(col)
    except Exception:
        pass
    return ls


def _merge_linesets(linesets: list["o3d.geometry.LineSet"]) -> "o3d.geometry.LineSet":
    pts_all = []
    lines_all = []
    cols_all = []
    v_off = 0
    for ls in linesets:
        p = np.asarray(ls.points, dtype=np.float64)
        l = np.asarray(ls.lines, dtype=np.int32)
        c = np.asarray(ls.colors, dtype=np.float64) if ls.has_colors() else None
        if p.size == 0 or l.size == 0:
            continue
        pts_all.append(p)
        lines_all.append(l + v_off)
        if c is not None and c.shape[0] == l.shape[0]:
            cols_all.append(c)
        v_off += p.shape[0]
    out = o3d.geometry.LineSet()
    if pts_all:
        out.points = o3d.utility.Vector3dVector(np.vstack(pts_all))
        out.lines = o3d.utility.Vector2iVector(np.vstack(lines_all))
        if cols_all:
            out.colors = o3d.utility.Vector3dVector(np.vstack(cols_all))
    return out


def _trajectory_lineset(centers: np.ndarray, *, color_rgb=(0.2, 0.9, 1.0)) -> "o3d.geometry.LineSet":
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    n = centers.shape[0]
    ls = o3d.geometry.LineSet()
    if n < 2:
        ls.points = o3d.utility.Vector3dVector(centers)
        ls.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        return ls
    lines = np.stack([np.arange(0, n - 1, dtype=np.int32), np.arange(1, n, dtype=np.int32)], axis=1)
    ls.points = o3d.utility.Vector3dVector(centers)
    ls.lines = o3d.utility.Vector2iVector(lines)
    try:
        cols = np.tile(np.array([color_rgb], dtype=np.float64), (lines.shape[0], 1))
        ls.colors = o3d.utility.Vector3dVector(cols)
    except Exception:
        pass
    return ls

def plot_trajectory(output_png: str, cum_T: dict[int, np.ndarray]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:
        return
    keys = sorted(cum_T.keys())
    if len(keys) < 2:
        return
    pts = []
    for k in keys:
        T = np.asarray(cum_T[k], dtype=np.float64).reshape(4, 4)
        # T_{0<-t} 的逆可近似视为 frame t 在 frame0 中的位姿（用于画轨迹更直观）
        Ti = np.linalg.inv(T)
        pts.append(Ti[:3, 3])
    P = np.asarray(pts, dtype=np.float64)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(P[:, 0], P[:, 1], P[:, 2], "-o", linewidth=1.5, markersize=3)
    ax.set_title("Pose/Shape Trajectory (inverse cumulative translation)")
    ax.set_xlabel("X (pixel)")
    ax.set_ylabel("Y (pixel)")
    ax.set_zlabel("Z (height)")
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main() -> None:
    cfg = TrackConfig()
    t0_all = time.time()
    # 输出目录强制固定（按你要求）：不受环境变量影响
    cfg.output_dir = r"D:\MIS_Pose_Track_Re3D\Exp_Results"
    # 自动探测默认模型路径（D盘/E盘切换时常见）
    _resolve_model_path(cfg)
    frames = list_frames(cfg.frames_dir)
    if cfg.max_frames and cfg.max_frames > 0:
        frames = frames[: int(cfg.max_frames)]
    if not frames:
        raise RuntimeError(f"未找到任何帧图像: {cfg.frames_dir}")

    _log(f"[配置] MIS_RT_MODE={cfg.rt_mode}  MIS_RT_ESTIMATOR={cfg.rt_estimator}", cfg=cfg, level=1)
    _log(f"[配置] frames_dir={cfg.frames_dir}", cfg=cfg, level=1)
    _log(f"[配置] output_dir={cfg.output_dir}", cfg=cfg, level=1)
    _log(f"[信息] 共找到帧数: {len(frames)}", cfg=cfg, level=1)

    out_root = _ensure_dir(cfg.output_dir)
    out_frames = _ensure_dir(os.path.join(out_root, "frames_with_bbox"))
    out_roi = _ensure_dir(os.path.join(out_root, "roi"))
    out_mask = _ensure_dir(os.path.join(out_root, "mask"))
    out_th = _ensure_dir(os.path.join(out_root, "th_gray"))
    out_cont = _ensure_dir(os.path.join(out_root, "contour_rgb"))
    out_cmp = _ensure_dir(os.path.join(out_root, "contour_compare"))
    out_mesh = _ensure_dir(os.path.join(out_root, "mesh"))
    out_rt = _ensure_dir(os.path.join(out_root, "rt"))
    out_vis = _ensure_dir(os.path.join(out_root, "vis"))
    out_pairs = _ensure_dir(os.path.join(out_root, "pair_results"))
    out_pairs_pcd = _ensure_dir(os.path.join(out_pairs, "overlay_pcd"))
    out_pairs_mesh = _ensure_dir(os.path.join(out_pairs, "aligned_mesh"))
    out_pairs_matches = _ensure_dir(os.path.join(out_pairs, "uvh_matches"))
    out_pairs_matches_vis = _ensure_dir(os.path.join(out_pairs_matches, "vis"))
    out_pairs_flow = _ensure_dir(os.path.join(out_pairs, "flow"))
    out_pairs_flow_hsv = _ensure_dir(os.path.join(out_pairs_flow, "hsv"))
    out_pairs_post = _ensure_dir(os.path.join(out_pairs, "postalign_eval"))
    out_pairs_post_cont = _ensure_dir(os.path.join(out_pairs_post, "aligned_contour_rgb"))
    out_pairs_post_cmp = _ensure_dir(os.path.join(out_pairs_post, "compare"))
    out_pairs_post_full = _ensure_dir(os.path.join(out_pairs_post, "aligned_fullmask"))

    global SEMANTIC_ENABLED, SEMANTIC_AVAILABLE
    # 启动时初始化语义模块（给出更明确的错误原因）
    ok_sem, sem_err = _init_semantic_module(cfg)
    model_exists = bool(os.path.isfile(str(cfg.model_path)))
    semantic_enabled = bool(ok_sem and model_exists and (calculate_contour_similarity is not None))
    SEMANTIC_ENABLED = bool(semantic_enabled)
    _log(f"[信息] semantic_enabled={int(semantic_enabled)}  model_exists={int(model_exists)}", cfg=cfg, level=1)

    # 预热语义模型（只做一次）：避免第一次计算相似度时初始化/compile 导致“突然卡一下”
    semantic_warmup_ok = False
    semantic_warmup_err = None
    if SEMANTIC_ENABLED and cfg.semantic_warmup:
        try:
            dummy = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
            _ = float(
                calculate_contour_similarity(
                    contour1=dummy,
                    contour2=dummy,
                    model_path=str(cfg.model_path),
                    image_size=tuple(cfg.semantic_image_size),
                    device=_device_str(cfg),
                )
            )
            semantic_warmup_ok = True
        except Exception:
            # 预热失败不影响主流程：只记录错误，不强制关闭；
            # 后续每次调用 semantic_similarity 内部仍会 try/except 自动回退。
            semantic_warmup_ok = False
            semantic_warmup_err = traceback.format_exc(limit=2)
    meta = {
        "time": _now(),
        "frames_dir": cfg.frames_dir,
        "output_dir": cfg.output_dir,
        "proc_scale": float(cfg.proc_scale),
        "semantic_available": bool(SEMANTIC_AVAILABLE),
        "semantic_enabled": bool(semantic_enabled),
        "semantic_warmup_ok": bool(semantic_warmup_ok),
        "semantic_warmup_error": semantic_warmup_err,
        "semantic_import_error": sem_err,
        "model_path": cfg.model_path,
        "model_exists": bool(model_exists),
        "device": _device_str(cfg),
        "th_percent": cfg.th_percent,
        "search_radius": cfg.search_radius,
        "search_step": cfg.search_step,
        "topk_semantic": cfg.topk_semantic,
        "spatial_penalty": cfg.spatial_penalty,
        "quick_scale": cfg.quick_scale,
        "quick_min_size": cfg.quick_min_size,
        "open3d_available": bool(o3d is not None),
        "rt_mode": cfg.rt_mode,
        "pose_interpretation": cfg.pose_interpretation,
        "save_pair_results": int(cfg.save_pair_results),
        "save_mesh_aligned_pair": int(cfg.save_mesh_aligned_pair),
        "sample_points": int(cfg.sample_points),
        "sample_poisson": int(cfg.sample_poisson),
        "note": "RT is estimated by ICP on pseudo-3D heightfield mesh (u,v,h), not a physical camera pose.",
    }
    with open(os.path.join(out_root, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 读第一帧（可选：先按 proc_scale 缩放，用于提速）
    first = _read_frame_bgr(frames[0], cfg)
    if first is None:
        raise RuntimeError(f"读取失败: {frames[0]}")
    H, W = first.shape[:2]
    _log(f"[信息] 第一帧尺寸: {W}x{H}", cfg=cfg, level=1)
    if abs(float(cfg.proc_scale) - 1.0) > 1e-6:
        _log(f"[信息] 已启用降分辨率处理: MIS_PROC_SCALE={cfg.proc_scale}", cfg=cfg, level=1)

    # 选择 ROI（仅在 rt_mode=roi 时需要；fullframe 则用整帧）
    roi = None
    if cfg.rt_mode == "fullframe":
        roi = (0, 0, W, H)
    backend = (cfg.roi_backend or "auto").strip().lower()

    # A) init：强制用环境变量框
    if backend == "init":
        roi = _try_parse_bbox(cfg.init_bbox)
        if roi is None:
            raise RuntimeError("MIS_ROI_BACKEND=init 但 MIS_INIT_BBOX 无效/未设置")
        # 若用户提供的是“原始分辨率坐标”，在降分辨率后可能越界：这里做一次自适应缩放兜底
        try:
            x0, y0, w0, h0 = [int(v) for v in roi]
            if (x0 + w0 > W) or (y0 + h0 > H):
                s = float(max(1e-6, float(cfg.proc_scale)))
                roi = (int(round(x0 * s)), int(round(y0 * s)), int(round(w0 * s)), int(round(h0 * s)))
        except Exception:
            pass

    # B) opencv：强制用 OpenCV GUI
    if roi is None and backend == "opencv":
        if not cfg.vis_window:
            raise RuntimeError("MIS_ROI_BACKEND=opencv 但 MIS_VIS_WINDOW=0，无法弹窗选框")
        cv2.namedWindow("MIS_ROI_Select", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("MIS_ROI_Select", min(1200, W), min(900, H))
        roi = cv2.selectROI("MIS_ROI_Select", first, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("MIS_ROI_Select")

    # C) tk：强制用 Tkinter GUI
    if roi is None and backend == "tk":
        roi = _select_roi_tkinter(first, max_w=cfg.tk_max_w, max_h=cfg.tk_max_h)
        if roi is None:
            raise RuntimeError("Tkinter 选框取消/失败")

    # D) auto：按优先级自动选择
    if roi is None and backend == "auto":
        # 1) 如果用户提供 init_bbox，先用它（可复现实验）
        if cfg.init_bbox:
            roi = _try_parse_bbox(cfg.init_bbox)
            if roi is None:
                print(f"[警告] MIS_INIT_BBOX 格式无效: {cfg.init_bbox}，将尝试鼠标选框/整帧兜底")
                roi = None
            else:
                # 降分辨率时，若 bbox 越界则尝试按比例缩放
                try:
                    x0, y0, w0, h0 = [int(v) for v in roi]
                    if (x0 + w0 > W) or (y0 + h0 > H):
                        s = float(max(1e-6, float(cfg.proc_scale)))
                        roi = (int(round(x0 * s)), int(round(y0 * s)), int(round(w0 * s)), int(round(h0 * s)))
                except Exception:
                    pass
        # 2) 尝试 OpenCV GUI
        if roi is None and cfg.vis_window:
            try:
                cv2.namedWindow("MIS_ROI_Select", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("MIS_ROI_Select", min(1200, W), min(900, H))
                roi = cv2.selectROI("MIS_ROI_Select", first, fromCenter=False, showCrosshair=True)
                cv2.destroyWindow("MIS_ROI_Select")
            except Exception as e:
                # 3) OpenCV HighGUI 不可用则回退 Tkinter
                print(f"[警告] OpenCV HighGUI 不可用，将改用 Tkinter 选框: {e}")
                roi = _select_roi_tkinter(first, max_w=cfg.tk_max_w, max_h=cfg.tk_max_h)
        # 4) 若都失败，整帧兜底
        if roi is None:
            roi = (0, 0, W, H)

    # E) none：不选框，直接整帧
    if roi is None and backend in ("none", "full", "all"):
        roi = (0, 0, W, H)

    # 未知后端兜底
    if roi is None:
        roi = (0, 0, W, H)
    x, y, w, h = [int(v) for v in roi]
    if w <= 1 or h <= 1:
        raise RuntimeError("ROI 选择无效（w/h 太小）")
    bbox = clamp_bbox(x, y, w, h, W, H)

    # 第一帧特征（ROI 或 fullframe）
    roi0 = crop(first, bbox)
    th0, mask0, th_u8_0 = compute_th_gray_truncated(roi0, cfg.th_percent, cfg)
    cnt0 = extract_largest_contour(mask0)
    if cnt0 is not None:
        cnt0s = resize_contour(cnt0, (bbox[2], bbox[3]), cfg.semantic_image_size)
        contour0_rgb = contour_to_rgb(cnt0s, cfg.semantic_image_size, thickness=2)
    else:
        contour0_rgb = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)

    prev_mask_small = cv2.resize(mask0, (64, 64), interpolation=cv2.INTER_NEAREST)

    # 保存第一帧中间结果
    if cfg.save_debug_images:
        cv2.imwrite(os.path.join(out_frames, f"frame_0000.jpg"), draw_bbox(first, bbox, text="f=0"))
        cv2.imwrite(os.path.join(out_roi, f"roi_0000.jpg"), roi0)
        cv2.imwrite(os.path.join(out_mask, f"mask_0000.png"), mask0)
        cv2.imwrite(os.path.join(out_th, f"th_gray_0000.png"), np.clip(th0, 0, 255).astype(np.uint8))
        cv2.imwrite(os.path.join(out_cont, f"contour_rgb_0000.png"), contour0_rgb)
        cv2.imwrite(os.path.join(out_vis, f"roi_overlay_0000.jpg"), make_overlay_mask(roi0, mask0))
        # 第一帧自身对比图（sim=1）
        _save_similarity_compare_image(
            os.path.join(out_cmp, "compare_0000.png"),
            prev_contour_rgb=contour0_rgb,
            cur_contour_rgb=contour0_rgb,
            similarity=1.0,
            title_left="prev(contour)",
            title_right="cur(contour)",
        )

    # mesh / RT 初始化（bbox 已经决定了是 ROI 还是 fullframe）
    mesh_prev = make_heightfield_mesh_from_th_gray(th0, bbox, z_scale=cfg.mesh_z_scale)
    if cfg.save_mesh and o3d is not None and mesh_prev is not None:
        try:
            o3d.io.write_triangle_mesh(os.path.join(out_mesh, f"mesh_0000.ply"), mesh_prev, write_vertex_colors=True)
        except Exception:
            pass

    track_rows = []
    diag_rows: list[dict[str, Any]] = []
    pairwise_T: dict[int, np.ndarray] = {0: np.eye(4, dtype=np.float64)}
    cumulative_T: dict[int, np.ndarray] = {0: np.eye(4, dtype=np.float64)}

    track_rows.append({
        "frame": 0,
        "image": frames[0],
        "bbox": list(map(int, bbox)),
        "th_uint8": float(th_u8_0),
        "score": 1.0,
        "semantic": 1.0,
        "quick": 1.0,
        "dx": 0,
        "dy": 0,
    })

    prev_bbox = bbox
    prev_contour_rgb = contour0_rgb
    prev_mesh_bbox = bbox
    # keyframe（用于恢复）
    key_idx = 0
    key_mesh = mesh_prev

    # 整帧缓存（用于 flow_ransac 以及 postalign_eval；避免每步重复读上一帧/重复算 th_68）
    prev_full_bgr = first
    prev_full_th = None
    prev_full_mask = None
    if cfg.rt_estimator == "flow_ransac" or cfg.postalign_eval:
        try:
            prev_full_th, prev_full_mask, _ = compute_th_gray_truncated(prev_full_bgr, cfg.th_percent, cfg)
        except Exception:
            prev_full_th, prev_full_mask = None, None

    # ========== 断点续跑（默认开启）==========
    # 注：为保持改动最小，这里仍会执行 frame0 初始化，但若检测到 checkpoint，会在进入主循环前覆盖状态并从断点继续。
    start_frame = 1
    last_done = 0
    if cfg.resume_enable and (not cfg.force_restart):
        resume = _try_load_checkpoint(
            out_root=out_root,
            cfg=cfg,
            frames=frames,
            out_mask=out_mask,
            out_th=out_th,
            out_cont=out_cont,
        )
        if resume is not None:
            try:
                start_frame = int(resume.get("start_frame", 1))
                last_done = int(resume.get("last_frame", 0))
                prev_bbox = tuple(resume.get("prev_bbox", prev_bbox))
                prev_contour_rgb = np.asarray(resume.get("prev_contour_rgb", prev_contour_rgb), dtype=np.uint8)
                prev_mask_small = np.asarray(resume.get("prev_mask_small", prev_mask_small), dtype=np.uint8)
                mesh_prev = resume.get("mesh_prev", mesh_prev)
                key_idx = int(resume.get("key_idx", key_idx))
                key_mesh = resume.get("key_mesh", key_mesh)
                pairwise_T = {int(k): np.asarray(v, dtype=np.float64).reshape(4, 4) for k, v in resume.get("pairwise_T", pairwise_T).items()}
                cumulative_T = {int(k): np.asarray(v, dtype=np.float64).reshape(4, 4) for k, v in resume.get("cumulative_T", cumulative_T).items()}
                track_rows = list(resume.get("track_rows", track_rows))
                diag_rows = list(resume.get("diag_rows", diag_rows))
                prev_full_bgr = resume.get("prev_full_bgr", prev_full_bgr)
                if cfg.rt_estimator == "flow_ransac" and prev_full_bgr is not None:
                    try:
                        prev_full_th, prev_full_mask, _ = compute_th_gray_truncated(prev_full_bgr, cfg.th_percent, cfg)
                    except Exception:
                        prev_full_th, prev_full_mask = None, None
                _log(f"[恢复] checkpoint: last_frame={last_done} -> start_frame={start_frame}", cfg=cfg, level=1)
            except Exception as e:
                _log(f"[恢复] 读取 checkpoint 失败，将从头继续: {e}", cfg=cfg, level=1)
                start_frame = 1
                last_done = 0

    # 主循环
    for i in range(int(start_frame), len(frames)):
        t_frame0 = time.time()
        img_path = frames[i]
        frame = _read_frame_bgr(img_path, cfg)
        if frame is None:
            print(f"[警告] 跳过读取失败帧: {img_path}")
            continue
        H, W = frame.shape[:2]
        if (i % int(max(1, cfg.print_every))) == 0:
            _log(f"[进度] frame={i}/{len(frames)-1} img={os.path.basename(img_path)}", cfg=cfg, level=1)

        if cfg.rt_mode == "fullframe":
            bbox_i = (0, 0, W, H)
            roi_full = frame
            t_th = time.time()
            th_i, mask_i, th_u8_i = compute_th_gray_truncated(roi_full, cfg.th_percent, cfg)
            if cfg.print_timing and (i % int(max(1, cfg.print_every))) == 0:
                _log(f"[耗时] frame{i} th_68: {(time.time()-t_th)*1000:.1f} ms", cfg=cfg, level=2)
            # fullframe 模式下不做轮廓语义匹配，仍生成一个 contour_rgb 方便可视化/兼容
            cnt = extract_largest_contour(mask_i)
            if cnt is not None:
                cnt2 = resize_contour(cnt, (bbox_i[2], bbox_i[3]), cfg.semantic_image_size)
                contour_i = contour_to_rgb(cnt2, cfg.semantic_image_size, thickness=2)
            else:
                contour_i = np.zeros((cfg.semantic_image_size[1], cfg.semantic_image_size[0], 3), dtype=np.uint8)
            # fullframe 模式下也计算“轮廓相似度”用于诊断图，避免一直显示为 0 导致 distance 爆表
            try:
                sem_full = float(semantic_similarity(prev_contour_rgb, contour_i, cfg))
            except Exception:
                sem_full = 0.0
            best = {
                "bbox": bbox_i,
                "score": float(sem_full),
                "semantic": float(sem_full),
                "quick": 0.0,
                "dx": 0,
                "dy": 0,
                "th_gray": th_i,
                "mask_u8": mask_i,
                "contour_rgb": contour_i,
                "th_uint8": float(th_u8_i),
            }
        else:
            best = find_best_bbox_in_next_frame(frame, prev_bbox, prev_contour_rgb, prev_mask_small, cfg)
            bbox_i = best["bbox"]
            th_i = best["th_gray"]
            mask_i = best["mask_u8"]
            contour_i = best["contour_rgb"]
            prev_mask_small = cv2.resize(mask_i, (64, 64), interpolation=cv2.INTER_NEAREST)

        # --- 后验评估准备：取“整帧”的 th/mask（在 uvh 全局坐标下评估 RT 对齐质量） ---
        prev_full_th_ref = prev_full_th
        prev_full_mask_ref = prev_full_mask
        cur_full_th = None
        cur_full_mask = None
        if cfg.postalign_eval:
            # fullframe 模式下 th_i/mask_i 就是整帧；ROI 模式则这里补算一次整帧（用于 postalign 评估）
            if cfg.rt_mode == "fullframe":
                cur_full_th = th_i
                cur_full_mask = mask_i
            else:
                try:
                    cur_full_th, cur_full_mask, _ = compute_th_gray_truncated(frame, cfg.th_percent, cfg)
                except Exception:
                    cur_full_th, cur_full_mask = None, None

        # 保存跟踪与中间结果
        roi_i = crop(frame, bbox_i)
        if cfg.save_debug_images:
            # 把关键相似度数值写到图上：融合分数/语义转换相似度/粗筛IoU
            txt = f"f={i} score={best['score']:.3f} sem={best['semantic']:.3f} q={best['quick']:.3f}"
            cv2.imwrite(os.path.join(out_frames, f"frame_{i:04d}.jpg"), draw_bbox(frame, bbox_i, text=txt))
            cv2.imwrite(os.path.join(out_roi, f"roi_{i:04d}.jpg"), roi_i)
            cv2.imwrite(os.path.join(out_mask, f"mask_{i:04d}.png"), mask_i)
            cv2.imwrite(os.path.join(out_th, f"th_gray_{i:04d}.png"), np.clip(th_i, 0, 255).astype(np.uint8))
            cv2.imwrite(os.path.join(out_cont, f"contour_rgb_{i:04d}.png"), contour_i)
            cv2.imwrite(os.path.join(out_vis, f"roi_overlay_{i:04d}.jpg"), make_overlay_mask(roi_i, mask_i))
            # 轮廓对比图：把“转换相似度”直接写进去（距离由 sim 反推）
            _save_similarity_compare_image(
                os.path.join(out_cmp, f"compare_{i:04d}.png"),
                prev_contour_rgb=prev_contour_rgb,
                cur_contour_rgb=contour_i,
                similarity=float(best.get("semantic", 0.0)),
                title_left="prev(contour)",
                title_right="cur(contour)",
            )

        # mesh + RT
        t_mesh = time.time()
        mesh_cur = make_heightfield_mesh_from_th_gray(th_i, bbox_i, z_scale=cfg.mesh_z_scale)
        if cfg.print_timing and (i % int(max(1, cfg.print_every))) == 0:
            _log(f"[耗时] frame{i} mesh: {(time.time()-t_mesh)*1000:.1f} ms", cfg=cfg, level=2)
        if cfg.save_mesh and o3d is not None and mesh_cur is not None:
            try:
                o3d.io.write_triangle_mesh(os.path.join(out_mesh, f"mesh_{i:04d}.ply"), mesh_cur, write_vertex_colors=True)
            except Exception:
                pass
        # ========== RT 估计：两条路线 ==========
        # (1) icp：沿用当前 mesh->pcd 的 ICP/recovery 体系
        # (2) flow_ransac：非刚性辅助（稠密光流建立 uv 对应 -> 采样 h 得到 uvh 点对 -> RANSAC+SVD 得到全局 Tg）
        init_T = np.asarray(pairwise_T.get(i - 1, np.eye(4, dtype=np.float64)), dtype=np.float64).reshape(4, 4)
        flow_info = {"n_matches": 0, "n_inliers": 0, "flow_used": 0, "flow_rmse": float("nan")}
        use_flow = (cfg.rt_estimator == "flow_ransac")
        if use_flow:
            # 为了坐标一致性，强制用整帧构建 th/mask（否则 ROI 平移会污染 flow 对应）
            if not cfg.flow_use_fullframe:
                # 若用户强行关闭 fullframe，则回退 icp
                use_flow = False
            else:
                prev_img = prev_full_bgr
                cur_img = frame
                if prev_img is not None and cur_img is not None and prev_full_th is not None and prev_full_mask is not None:
                    # 当前帧整帧 th/mask
                    th_cur, mask_cur, _ = compute_th_gray_truncated(cur_img, cfg.th_percent, cfg)
                    cur_uvh, prev_uvh, _, flow_dbg = _build_uvh_matches_from_flow(
                        prev_bgr=prev_img,
                        cur_bgr=cur_img,
                        prev_th=prev_full_th,
                        cur_th=th_cur,
                        prev_mask=prev_full_mask,
                        cfg=cfg,
                    )
                    flow_info["n_matches"] = int(cur_uvh.shape[0])
                    flow_info["flow_used"] = 1
                    if cur_uvh.shape[0] >= 20:
                        T0, inl = _ransac_rigid_T_xy(cur_uvh=cur_uvh, prev_uvh=prev_uvh, cfg=cfg)
                        if T0 is not None:
                            flow_info["n_inliers"] = int(np.count_nonzero(inl)) if inl is not None else 0
                            # IRLS 细化（鲁棒拟合全局刚体，压制形变）
                            T_irls = _irls_refine_T(T_init=T0, cur_uvh=cur_uvh, prev_uvh=prev_uvh, cfg=cfg, inlier_mask=inl)
                            # 用 IRLS 后的残差给一个“flow 质量指标”，方便门控与diag
                            try:
                                Xh = (T_irls[:3, :3] @ cur_uvh.T).T + T_irls[:3, 3][None, :]
                                e = np.linalg.norm((Xh[:, :2] - prev_uvh[:, :2]), axis=1)
                                if inl is not None:
                                    e_in = e[np.asarray(inl, dtype=bool)]
                                else:
                                    e_in = e
                                flow_info["flow_rmse"] = float(np.median(e_in)) if e_in.size > 0 else float("nan")
                            except Exception:
                                pass
                            # 可选：以 Tg 为初值再做一次 ICP refine（仍输出刚体，但更贴合点云）
                            if cfg.flow_refine_icp and (mesh_prev is not None) and (mesh_cur is not None):
                                rr_icp = estimate_T_prev_cur_icp(mesh_prev, mesh_cur, cfg, init_T=T_irls)
                                if rr_icp is not None:
                                    reg = rr_icp
                                    reg.method = "flow_ransac+irls+icp"
                                    init_mode = "flow_ransac"
                                else:
                                    fit0 = float(flow_info["n_inliers"]) / float(max(1, flow_info["n_matches"]))
                                    reg = RegResult(T=T_irls, fitness=fit0, rmse=float(flow_info["flow_rmse"]), method="flow_ransac+irls")
                                    init_mode = "flow_ransac"
                            else:
                                fit0 = float(flow_info["n_inliers"]) / float(max(1, flow_info["n_matches"]))
                                reg = RegResult(T=T_irls, fitness=fit0, rmse=float(flow_info["flow_rmse"]), method="flow_ransac+irls")
                                init_mode = "flow_ransac"

                            # 保存点对（类似 SIFT matches）
                            try:
                                out_npz = os.path.join(out_pairs_matches, f"uvh_matches_prev_{i-1:04d}_cur_{i:04d}.npz")
                                np.savez_compressed(
                                    out_npz,
                                    prev_uvh=prev_uvh.astype(np.float32),
                                    cur_uvh=cur_uvh.astype(np.float32),
                                    inliers=(inl.astype(np.uint8) if inl is not None else None),
                                )
                            except Exception:
                                pass
                            # 保存光流与采样点（中间结果）
                            if cfg.flow_save_raw:
                                try:
                                    st = int(max(1, cfg.flow_save_raw_stride))
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
                                except Exception:
                                    pass
                            if cfg.flow_save_hsv:
                                try:
                                    flow = np.asarray(flow_dbg.get("flow", None))
                                    if flow is not None and isinstance(flow, np.ndarray) and flow.ndim == 3 and flow.shape[2] == 2:
                                        hsv_bgr = _flow_to_hsv_bgr(flow, clip_mag=20.0)
                                        out_png = os.path.join(out_pairs_flow_hsv, f"flow_hsv_prev_{i-1:04d}_cur_{i:04d}.png")
                                        cv2.imwrite(out_png, hsv_bgr)
                                except Exception:
                                    pass
                            # 保存可视化
                            if cfg.flow_save_vis:
                                try:
                                    out_png = os.path.join(out_pairs_matches_vis, f"matches_prev_{i-1:04d}_cur_{i:04d}.png")
                                    _save_flow_matches_vis(
                                        out_png,
                                        prev_bgr=prev_img,
                                        prev_uvh=prev_uvh,
                                        cur_uvh=cur_uvh,
                                        T_prev_cur=reg.T,
                                        inliers=inl,
                                        max_lines=int(cfg.flow_vis_max_lines),
                                    )
                                except Exception:
                                    pass
                        else:
                            reg, init_mode = estimate_reg_with_trials(mesh_prev=mesh_prev, mesh_cur=mesh_cur, cfg=cfg, init_T=init_T)
                    else:
                        reg, init_mode = estimate_reg_with_trials(mesh_prev=mesh_prev, mesh_cur=mesh_cur, cfg=cfg, init_T=init_T)
                else:
                    reg, init_mode = estimate_reg_with_trials(mesh_prev=mesh_prev, mesh_cur=mesh_cur, cfg=cfg, init_T=init_T)
                # 更新 flow 缓存（即使 th_cur 失败也更新上一帧图像，避免卡死在旧帧）
                prev_full_bgr = cur_img
                if "th_cur" in locals() and "mask_cur" in locals():
                    try:
                        prev_full_th = th_cur
                        prev_full_mask = mask_cur
                    except Exception:
                        pass
        if cfg.rt_estimator != "flow_ransac" or (not use_flow):
            reg, init_mode = estimate_reg_with_trials(mesh_prev=mesh_prev, mesh_cur=mesh_cur, cfg=cfg, init_T=init_T)

        if (i % int(max(1, cfg.print_every))) == 0:
            if int(flow_info.get("flow_used", 0)) == 1:
                n_m = int(flow_info.get("n_matches", 0))
                n_in = int(flow_info.get("n_inliers", 0))
                ratio = float(n_in) / float(max(1, n_m))
                _log(f"[Flow] matches={n_m} inliers={n_in} ratio={ratio:.3f} rmse={flow_info.get('flow_rmse', float('nan'))}", cfg=cfg, level=1)
            _log(f"[RT] method={reg.method} fit={reg.fitness:.3f} rmse={reg.rmse:.3f} init={init_mode}", cfg=cfg, level=1)

        recovered = 0
        used_key = ""
        gate = _motion_gate(_mesh_stats(mesh_prev), _mesh_stats(mesh_cur), cfg)
        gate_jump = int(gate.get("gate_jump", 0)) == 1
        gate_mode = str(gate.get("gate_mode", "small"))
        recovery_tag = ""

        def _is_bad() -> bool:
            # flow_ransac（未ICP refine）时，fitness/rmse 的语义是“内点比例/xy残差”，不能用 ICP 的 0.995 门槛
            if int(flow_info.get("flow_used", 0)) == 1 and ("flow_ransac" in str(getattr(reg, "method", ""))) and ("icp" not in str(getattr(reg, "method", ""))):
                n_m = int(flow_info.get("n_matches", 0))
                n_i = int(flow_info.get("n_inliers", 0))
                ratio = float(n_i) / float(max(1, n_m))
                rmse = float(flow_info.get("flow_rmse", float("nan")))
                if (ratio < float(cfg.flow_inlier_ratio_min)):
                    return True
                if np.isfinite(rmse) and (rmse > float(cfg.flow_rmse_max)):
                    return True
                return False
            return _need_recovery(float(reg.fitness), float(reg.rmse), cfg)
        # (A) 若检测到跳变（或对齐很差），优先 keyframe 恢复（参考 Exp_Mehsh_Between_RT.py）
        if cfg.recovery_enable and _is_bad() and cfg.keyframe_enable and gate_jump:
            if (key_mesh is not None) and (mesh_cur is not None) and (int(key_idx) in cumulative_T) and (int(i - 1) in cumulative_T):
                reg_key, key_init_mode = estimate_reg_with_trials(mesh_prev=key_mesh, mesh_cur=mesh_cur, cfg=cfg, init_T=None)
                try:
                    T0_prev = np.asarray(cumulative_T[int(i - 1)], dtype=np.float64).reshape(4, 4)
                    T0_key = np.asarray(cumulative_T[int(key_idx)], dtype=np.float64).reshape(4, 4)
                    # T_prev<-key = inv(T0<-prev) @ (T0<-key)
                    T_prev_key = np.linalg.inv(T0_prev) @ T0_key
                    # T_prev<-cur = T_prev<-key @ T_key<-cur
                    T_prev_cur = np.asarray(T_prev_key, dtype=np.float64) @ np.asarray(reg_key.T, dtype=np.float64)
                    reg = RegResult(T=T_prev_cur, fitness=float(reg_key.fitness), rmse=float(reg_key.rmse), method=f"keyframe:{reg_key.method}")
                    init_mode = f"keyframe(key={int(key_idx)}):{key_init_mode}"
                    recovered = 1
                    used_key = str(int(key_idx))
                    recovery_tag = "keyframe"
                except Exception:
                    pass

        # (B) greedy recovery（小/中位移，且还很差）
        if cfg.recovery_enable and cfg.greedy_enable and (not gate_jump) and _is_bad():
            best_reg = reg
            best_tag = "tracking"
            for tag, Tcand in _greedy_candidates(init_T, cfg):
                rr = estimate_T_prev_cur_icp(mesh_prev, mesh_cur, cfg, init_T=Tcand, max_iter=int(cfg.greedy_icp_max_iter))
                if rr is not None and _better_than(rr, best_reg):
                    best_reg = rr
                    best_tag = f"greedy:{tag}"
            # accept if improved or meets thresholds
            if _better_than(best_reg, reg) or (not _need_recovery(best_reg.fitness, best_reg.rmse, cfg)):
                reg = best_reg
                init_mode = best_tag
                recovered = 1
                recovery_tag = (recovery_tag + "+greedy").strip("+")

        # (C) RANSAC recovery（跳变/或极差时兜底）
        if cfg.recovery_enable and cfg.ransac_enable and _is_bad():
            allow = (not cfg.ransac_only_on_jump) or gate_jump
            if allow:
                rr = _ransac_then_icp(mesh_prev=mesh_prev, mesh_cur=mesh_cur, cfg=cfg)
                if rr is not None and (_better_than(rr, reg) or (not _need_recovery(rr.fitness, rr.rmse, cfg))):
                    reg = rr
                    init_mode = "ransac+icp"
                    recovered = 1
                    recovery_tag = (recovery_tag + "+ransac").strip("+")

        # 若仍然很差，可选“拒绝更新”避免累计漂移
        if cfg.recovery_enable and _is_bad() and cfg.reject_on_fail:
            reg = RegResult(T=np.eye(4, dtype=np.float64), fitness=float(reg.fitness), rmse=float(reg.rmse), method=f"{reg.method}+rejected")
            init_mode = f"{init_mode}+rejected"
            recovery_tag = (recovery_tag + "+rejected").strip("+")

        pairwise_T[i] = np.asarray(reg.T, dtype=np.float64).reshape(4, 4)
        cumulative_T[i] = np.asarray(cumulative_T[i - 1], dtype=np.float64).reshape(4, 4) @ pairwise_T[i]
        if (i % int(max(1, cfg.print_every))) == 0:
            _log(f"[Gate] jump={int(gate_jump)} mode={gate_mode} rec={recovered} tag={recovery_tag}", cfg=cfg, level=1)
            if cfg.print_timing:
                _log(f"[耗时] frame{i} total: {(time.time()-t_frame0)*1000:.1f} ms", cfg=cfg, level=2)

        # 保存帧间匹配结果（叠加点云/可选对齐mesh）+ diag 行（参考 Exp_Mehsh_Between_RT.py）
        if cfg.save_pair_results and (o3d is not None) and (mesh_prev is not None) and (mesh_cur is not None):
            try:
                pcd_prev = _mesh_to_pcd(mesh_prev, cfg)
                pcd_cur = _mesh_to_pcd(mesh_cur, cfg)
                pcd_cur_aligned = o3d.geometry.PointCloud(pcd_cur)
                pcd_cur_aligned.transform(pairwise_T[i])
                out_pcd = os.path.join(out_pairs_pcd, f"pair_overlay_prev_{i-1:04d}_cur_{i:04d}.ply")
                _save_pair_overlay_pcd(out_path=out_pcd, pcd_prev=pcd_prev, pcd_cur_aligned=pcd_cur_aligned)
            except Exception:
                pass
            if cfg.save_mesh_aligned_pair:
                try:
                    m_al = o3d.geometry.TriangleMesh(mesh_cur)
                    m_al.transform(pairwise_T[i])
                    out_ply = os.path.join(out_pairs_mesh, f"mesh_cur_aligned_prev_{i-1:04d}_cur_{i:04d}.ply")
                    o3d.io.write_triangle_mesh(out_ply, m_al, write_vertex_colors=True)
                except Exception:
                    pass

        # ====== 后验渲染评估：RT 对齐轮廓 -> 语义相似度（诊断配准质量）======
        postalign_sim = float("nan")
        if cfg.postalign_eval and prev_full_th_ref is not None and prev_full_mask_ref is not None and cur_full_th is not None and cur_full_mask is not None:
            try:
                prev_full_rgb = _contour_rgb_from_full_mask(prev_full_mask_ref, (W, H), cfg)
                cur_al_rgb, cur_al_full = _render_aligned_contour_rgb(
                    cur_mask_u8=cur_full_mask,
                    cur_th_gray=cur_full_th,
                    T_prev_cur=pairwise_T[i],
                    full_wh=(W, H),
                    cfg=cfg,
                )
                postalign_sim = float(semantic_similarity(prev_full_rgb, cur_al_rgb, cfg))
                if cfg.save_pair_results:
                    cv2.imwrite(os.path.join(out_pairs_post_cont, f"aligned_contour_rgb_prev_{i-1:04d}_cur_{i:04d}.png"), cur_al_rgb)
                    _save_similarity_compare_image(
                        os.path.join(out_pairs_post_cmp, f"compare_prev_{i-1:04d}_cur_{i:04d}.png"),
                        prev_contour_rgb=prev_full_rgb,
                        cur_contour_rgb=cur_al_rgb,
                        similarity=postalign_sim,
                        title_left="prev(full contour)",
                        title_right="cur(aligned by RT)",
                    )
                    if cfg.postalign_save_fullres and cur_al_full is not None:
                        cv2.imwrite(os.path.join(out_pairs_post_full, f"aligned_fullmask_prev_{i-1:04d}_cur_{i:04d}.png"), cur_al_full)
            except Exception:
                postalign_sim = float("nan")

        diag_rows.append({
            "frame_idx": int(i),
            "img": str(img_path),
            "pair_fitness": float(reg.fitness),
            "pair_rmse": float(reg.rmse),
            "method": str(reg.method),
            "rt_mode": str(cfg.rt_mode),
            "init_mode": str(init_mode),
            "recovered": int(recovered),
            "key_idx": used_key,
            "gate_jump": int(gate_jump),
            "gate_mode": gate_mode,
            "recovery_tag": recovery_tag,
            "nz_center_dxy": gate.get("nz_center_dxy", ""),
            "nz_ratio_mult": gate.get("nz_ratio_mult", ""),
            "rt_estimator": str(cfg.rt_estimator),
            "flow_used": int(flow_info.get("flow_used", 0)),
            "flow_n_matches": int(flow_info.get("n_matches", 0)),
            "flow_n_inliers": int(flow_info.get("n_inliers", 0)),
            "flow_rmse": flow_info.get("flow_rmse", ""),
            "postalign_sim": (postalign_sim if np.isfinite(postalign_sim) else ""),
        })

        # 更新 postalign 缓存（用于下一帧的 prev_full_* 参考）
        if cfg.postalign_eval:
            try:
                prev_full_bgr = frame
                if cur_full_th is not None and cur_full_mask is not None:
                    prev_full_th = cur_full_th
                    prev_full_mask = cur_full_mask
            except Exception:
                pass

        # 记录
        track_rows.append({
            "frame": int(i),
            "image": img_path,
            "bbox": list(map(int, bbox_i)),
            "th_uint8": float(best.get("th_uint8", float("nan"))),
            "score": float(best["score"]),
            "semantic": float(best["semantic"]),
            "quick": float(best["quick"]),
            "dx": int(best["dx"]),
            "dy": int(best["dy"]),
        })

        # 更新 prev
        prev_bbox = bbox_i
        prev_contour_rgb = contour_i
        mesh_prev = mesh_cur
        prev_mesh_bbox = bbox_i

        # 更新 keyframe：只在“本帧配准质量不过门控”时才更新
        if cfg.keyframe_enable and int(cfg.keyframe_interval) > 0:
            if (not _is_bad()) and (i % int(cfg.keyframe_interval) == 0):
                key_idx = int(i)
                key_mesh = mesh_cur

        # 断点续跑：周期性写 checkpoint（默认每 5 帧）
        last_done = int(i)
        if cfg.resume_enable and (not cfg.force_restart):
            every = int(max(1, getattr(cfg, "ckpt_every", 5)))
            if (last_done % every) == 0 or (last_done >= len(frames) - 1):
                _save_checkpoint(
                    out_root=out_root,
                    cfg=cfg,
                    frames=frames,
                    last_frame=last_done,
                    prev_bbox=prev_bbox,
                    key_idx=key_idx,
                    pairwise_T=pairwise_T,
                    cumulative_T=cumulative_T,
                    track_rows=track_rows,
                    diag_rows=diag_rows,
                    completed=False,
                )

    # 保存跟踪结果
    with open(os.path.join(out_root, "tracking_bboxes_and_scores.json"), "w", encoding="utf-8") as f:
        json.dump(track_rows, f, ensure_ascii=False, indent=2)

    save_rt_txt(os.path.join(out_rt, "shape_rt_pairwise_4x4.txt"), pairwise_T, "Shape RT: T_{t-1<-t} (cur->prev) in (u,v,h)")
    save_rt_txt(os.path.join(out_rt, "shape_rt_cumulative_4x4.txt"), cumulative_T, "Shape RT: T_{0<-t} (cur->frame0) in (u,v,h)")

    plot_trajectory(os.path.join(out_rt, "trajectory.png"), cumulative_T)

    # 参考 Exp_Mehsh_Between_RT.py：导出“伪相机位姿/相机中心”
    save_camera_poses_and_centers(
        out_dir=out_root,
        cumulative_T=cumulative_T,
        pose_interp=cfg.pose_interpretation,
    )

    # 参考 Exp_Mehsh_Between_RT.py：保存/展示位姿可视化（Open3D）
    save_pose_visualization_outputs(
        out_dir=out_root,
        cumulative_T=cumulative_T,
        pose_interp=cfg.pose_interpretation,
        cfg=cfg,
    )

    # 保存 diag.csv（参考 Exp_Mehsh_Between_RT.py 的核心字段）
    try:
        diag_csv = os.path.join(out_root, "diag.csv")
        with open(diag_csv, "w", encoding="utf-8", newline="\n") as f:
            fieldnames = [
                "frame_idx", "img", "pair_fitness", "pair_rmse", "method",
                "rt_mode", "init_mode", "recovered", "key_idx",
                "gate_jump", "gate_mode", "recovery_tag",
                "nz_center_dxy", "nz_ratio_mult",
                "rt_estimator", "flow_used", "flow_n_matches", "flow_n_inliers", "flow_rmse",
                "postalign_sim",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in diag_rows:
                w.writerow({k: row.get(k, "") for k in fieldnames})
    except Exception:
        pass

    # 断点续跑：标记完成（下次启动不会再从 checkpoint 续跑）
    if cfg.resume_enable and (not cfg.force_restart):
        try:
            _save_checkpoint(
                out_root=out_root,
                cfg=cfg,
                frames=frames,
                last_frame=int(max(0, min(len(frames) - 1, last_done))),
                prev_bbox=prev_bbox,
                key_idx=key_idx,
                pairwise_T=pairwise_T,
                cumulative_T=cumulative_T,
                track_rows=track_rows,
                diag_rows=diag_rows,
                completed=True,
            )
        except Exception:
            pass

    print(f"[完成] 输出目录: {out_root}")
    print(f"[完成] 跟踪结果: {os.path.join(out_root, 'tracking_bboxes_and_scores.json')}")
    print(f"[完成] RT结果: {os.path.join(out_rt, 'shape_rt_pairwise_4x4.txt')}")
    print(f"[完成] 轨迹图: {os.path.join(out_rt, 'trajectory.png')}")
    # 语义相似度状态提示（避免把“内部依赖缺失”误报成“文件未找到”）
    if SEMANTIC_ENABLED:
        print("[信息] 语义相似度已启用（转换相似度）。")
    else:
        if SEMANTIC_IMPORT_ERROR:
            print(f"[提示] 语义模块导入失败，将使用回退相似度: {SEMANTIC_IMPORT_ERROR}")
            print("[提示] 请确认 `extract_semantic_features_Function.py` 的依赖（如 `Copy12.py`）在同目录，或设置 `MIS_SEMANTIC_CODE_DIRS` 指向依赖所在目录。")
        elif not model_exists:
            print(f"[提示] 语义模型文件不存在，将使用回退相似度: {cfg.model_path}")
        else:
            print("[提示] 语义相似度未启用（可能缺少依赖/torch 未安装/被禁用），将使用回退相似度。")
    if o3d is None:
        print("[提示] open3d 未安装，已跳过 mesh/RT（仍保存 bbox/mask/th_gray 等）。")
    if cfg.print_timing:
        _log(f"[总耗时] {(time.time()-t0_all):.2f} s", cfg=cfg, level=1)


if __name__ == "__main__":
    main()

