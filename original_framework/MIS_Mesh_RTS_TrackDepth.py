#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIS_Mesh_RTS_Track.py

基于伪3D Mesh 的跟踪（固定视点神经渲染 + MeshSimilarityNet；Autoencoder_129 可选）

核心思路：
  1. 首帧 target：用 PointRT RenderField 在「正前规范位姿」下渲染；该位姿即固定相机视点（尺度由规范
     化半径锁定）。
  2. 无旋转目标库。粗匹配参考的 depth/normal 轮廓与 AE pack 回退：均来自首帧 ROI 的 **th_68 截断灰度**
     派生的伪深度图 + 梯度幅值图（与候选窗同源定义），**不再依赖首帧神经渲染 pack**。
  3. 跟踪帧：以上一帧 bbox 为中心按 MIS_SEARCH_EXPAND（默认 64）四边扩展搜索区，必要时若融合分低于
     MIS_MATCH_RETRY_SIM 再用 MIS_MATCH_EXPAND_LARGE（默认 128）重扫；**搜索区内对当前帧 BGR 只做一次 th_68**，
     伪 3D、RGB/伪 depth/伪 normal 轮廓均在同一套 th 与派生图上 **按候选框坐标裁剪**，**不对候选再做神经渲染**。
     滑动窗（步长 MIS_LIB_TRACK_STRIDE）内：三路轮廓与 Autoencoder_129 比相似度，按 MIS_MATCH_W_RGB /
     MIS_MATCH_W_DEPTH / MIS_MATCH_W_NORMAL 加权（语义不可用时回退 MIS_MATCH_W_PACK + MIS_MATCH_W_CONTOUR）；
     argmax 后，再对 target 与裁剪伪 3D 在规范空间做点-点 ICP（MIS_ICP_*）。
  4. EMA 平滑与位移/面积门控仍可用于抑制漂移。
  5. lib_sim 较高时可用 MIS_LIB_RELAX_SHIFT_MIN_SIM / MIS_LIB_RELAX_SHIFT_MULT 放宽 center_shift；
     lib_sim 较低时可用 MIS_LIB_ADAPT_SIM_T1/T2 与 MIS_LIB_ADAPT_EXPAND_ADD1/ADD2 自动加大搜索并重扫。

默认用 MeshSimilarityNet（MIS_MESH_SIMI_*）做滑动窗相似度；可选 MIS_AE_ENABLE=1 + MIS_MODEL_PATH 启用
Autoencoder_129。RenderField 仍用于视距自动取景（MIS_LIB_VIEW_AUTO）与可选调试图
（render_debug / lib_match_render）；**滑动粗匹配不调用每候选 render**。

  视距与分辨率（目标库正前渲染 / AE 打包，与 RenderField 多视角验证的 MIS_RENDER_VIEW_DIST 可独立）：
  - MIS_LIB_VIEW_DIST：显式正前视距（规范空间 Z 平移，越小相机越近）；0 表示用 MIS_RENDER_VIEW_DIST*MIS_LIB_VIEW_DIST_MULT。
  - MIS_LIB_VIEW_AUTO=1：按前景占比自动微调视距（见 MIS_LIB_AUTO_*）。
  - MIS_AE_PACK_W/H：打包进 Autoencoder 的 BGR 分辨率（默认 128）。
  - MIS_LIB_SAVE_RENDERS_MIN_SIDE：导出 PNG 短边下限（放大弱分辨率渲染便于检查）。
  - MIS_LIB_EXPORT_VIEW_RECORDS=1：建库时在 target_rotation_lib/view_records/ 下保存每视点 3D 点云
    （npy/ply）、法线 RGB 预览、AE 打包图，以及首帧 reference_target_roi_bgr.png（见 MIS_LIB_EXPORT_*）。
  - MIS_LIB_RTS_TOPK=8：神经分 Top-8 后，在「统一 th_68 伪3D」上按候选框裁剪，与参考 mesh 做 optimize_rts 再定窗；
    MIS_LIB_MESH_SCOPE=search（默认）或 frame 指定统一建点范围。
  - MIS_LIB_RTS_MIN_MARGIN>0：仅当 Top-K 内最优 RTS 分 ≥ 神经胜出窗的 RTS 分 + 该边际时才改窗（默认 0=保持原行为）；
    非法 MIS_LIB_MESH_SCOPE 会告警并回退为 search。

  MeshSimilarityNet（可选，与 train_mesh_similarity_chess 权重一致）：
  - MIS_MESH_SIMI_ENABLE=1（默认）：无 Autoencoder 时直接用每窗伪3D 与 target 的 MeshSimilarity 打分选窗；
    有 AE 时仍在 AE 融合分 Top-K 内做点云相似度加权重排，再 ICP。
  - MIS_MESH_SIMI_CKPT：权重路径（默认 mesh_sim_runs/chess_v1/mesh_similarity_best.pt）。
  - MIS_MESH_SIMI_TOPK：参与重排的候选数；MIS_MESH_SIMI_W：相似度权重（combo=(1-w)*AE + w*mesh）。
  - 在线热身：默认 MIS_MESH_SIMI_TRAIN_EPOCHS=500、每 epoch MIS_MESH_SIMI_TRAIN_SPA 对样本、MIS_MESH_SIMI_TRAIN_BATCH
    小批量反传；并从整帧伪3D 随机取与 target 框低 IoU 的补丁作负样本（MIS_MESH_WARMUP_NEG_*）。
    另：归一化 target 仅旋转、无平移（默认 rot_mode=z：u-v 平面单角度；MIS_MESH_WARMUP_ROT_MODE=full 为 SO(3)），GT=1。
    MIS_MESH_SIMI_TRAIN_STEPS>0 时改为「仅按步数」模式并忽略 EPOCHS。
  - MIS_MESH_PATCH_METRIC：空=MeshSimilarity 点云/网格；depth=伪深度 patch 皮尔逊相关（与 uvh 的 h 一致，推荐）；normal=Sobel 梯度 patch 相关。非空时在线热身标签与滑动窗打分均用该度量（可与 MIS_MESH_SIMI_MESH_INPUT 并存，优先 patch）。
  - MIS_MESH_PATCH_GRID：patch 对齐后栅格边长（默认 64）。
  - MIS_MESH_WARMUP_EXPORT=1：将热身样本对保存为 ply_pairs/*.ply + warmup_manifest.json（见 README_查看说明.txt）；
    MIS_MESH_WARMUP_EXPORT_MAX 限制条数；MIS_MESH_WARMUP_EXPORT_DIR 可指定目录。
    权重默认写到 MIS_OUTPUT_DIR/mesh_simi_online_warmup.pt。MIS_MESH_SIMI_WARMUP_CKPT 可自定义路径。

运行（PowerShell）：
  python d:/MIS_Pose_Track_Re3D/MIS_Mesh_RTS_Track.py

GT 评估（可选）：设置 MIS_GT_MASK_DIR 为 Chess_Box_Mask 下与帧目录同名的子目录
  （或含 summary.csv 的完整路径），跑完后在 output_dir 生成 eval_vs_gt_summary.json。
库-候选渲染对比（可选）：MIS_LIB_MATCH_VIZ=1，结果在 output_dir/lib_match_render/。
"""

from __future__ import annotations

import os
import json
import time
import math
import warnings
import csv
from dataclasses import dataclass
from typing import Any

import numpy as np
import cv2
import torch
import torch.nn as nn

import MIS_C_Fe_Simi_Mesh_Match_Pose_Estimate_Track as core


# ═══════════════════════════════════════════════════════════
#  配置
# ═══════════════════════════════════════════════════════════

def _ei(name: str, default: int) -> int:
    try:
        v = os.environ.get(name, "").strip()
        return int(v) if v else int(default)
    except Exception:
        return int(default)


def _ef(name: str, default: float) -> float:
    try:
        v = os.environ.get(name, "").strip()
        return float(v) if v else float(default)
    except Exception:
        return float(default)


@dataclass
class RTSConfig:
    # 数据
    frames_dir: str = (
        os.environ.get("MIS_FRAMES_DIR", "").strip()
        or r"E:\Surgical_Tracking_Datasets\Chess_Box_1_8\captured_photos3"
    )
    output_dir: str = (
        os.environ.get("MIS_OUTPUT_DIR", "").strip()
        or r"D:\MIS_Pose_Track_Re3D\Exp_Results_RTS_Track"
    )

    proc_scale: float = _ef("MIS_PROC_SCALE", 0.5)
    search_expand: int = _ei("MIS_SEARCH_EXPAND", 64)

    # th_68 参数（复用 core.TrackConfig 的默认值）
    th_percent: float = _ef("MIS_TH_PERCENT", 68.0)
    bilateral_d: int = _ei("MIS_BILATERAL_D", 7)
    bilateral_sc: float = _ef("MIS_BILATERAL_SC", 75.0)
    bilateral_ss: float = _ef("MIS_BILATERAL_SS", 75.0)
    mesh_z_scale: float = _ef("MIS_MESH_Z_SCALE", 30.0)
    max_points: int = _ei("MIS_MAX_POINTS", 60000)

    # RTS 优化
    grid_size: int = _ei("MIS_GRID_SIZE", 96)
    rts_iters: int = _ei("MIS_RTS_ITERS", 10)
    rts_rot_range_deg: float = _ef("MIS_RTS_ROT_RANGE", 25.0)
    rts_rot_steps: int = _ei("MIS_RTS_ROT_STEPS", 12)
    rts_scale_min: float = _ef("MIS_RTS_SCALE_MIN", 0.65)
    rts_scale_max: float = _ef("MIS_RTS_SCALE_MAX", 1.55)
    rts_scale_steps: int = _ei("MIS_RTS_SCALE_STEPS", 10)
    # 平移搜索半径（像素，单边）；建议 ≥ MIS_SEARCH_EXPAND 量级
    rts_trans_range_px: float = _ef("MIS_RTS_TRANS_RANGE", 72.0)
    rts_trans_steps: int = _ei("MIS_RTS_TRANS_STEPS", 10)
    rts_shrink: float = _ef("MIS_RTS_SHRINK", 0.55)

    # 匹配权重
    w_iou: float = _ef("MIS_W_IOU", 0.35)
    w_depth_corr: float = _ef("MIS_W_DEPTH_CORR", 0.40)
    w_depth_l1: float = _ef("MIS_W_DEPTH_L1", 0.15)
    w_prior: float = _ef("MIS_W_PRIOR", 0.10)

    # 参考模式: "first" 始终用首帧 | "prev" 用上一帧
    ref_mode: str = os.environ.get("MIS_REF_MODE", "first").strip().lower()

    # bbox 后处理
    bbox_pad: int = _ei("MIS_BBOX_PAD", 2)
    bbox_percentile_lo: float = _ef("MIS_BBOX_PCTL_LO", 1.0)
    bbox_percentile_hi: float = _ef("MIS_BBOX_PCTL_HI", 99.0)

    # ── 反漂移参数 ──
    # 匹配得分低于此值视为失败，回退到上帧 bbox
    score_reject_thr: float = _ef("MIS_SCORE_REJECT_THR", 0.15)
    # bbox 中心帧间最大允许位移（像素），超过则拒绝
    max_center_shift_px: float = _ef("MIS_MAX_CENTER_SHIFT", 56)
    # bbox 面积变化比允许范围 [lo, hi]，超出则拒绝
    area_ratio_lo: float = _ef("MIS_AREA_RATIO_LO", 0.50)
    area_ratio_hi: float = _ef("MIS_AREA_RATIO_HI", 1.85)
    # bbox 指数移动平均平滑系数（0=不平滑, 1=完全用旧值）
    bbox_ema_alpha: float = _ef("MIS_BBOX_EMA_ALPHA", 0.64)
    # 连续失败 N 帧后搜索区域翻倍
    fail_expand_after: int = _ei("MIS_FAIL_EXPAND_AFTER", 3)
    # 连续失败上限，超过后停止扩大（防止搜索范围爆炸）
    fail_expand_max: int = _ei("MIS_FAIL_EXPAND_MAX", 8)

    # ── 神经渲染场验证 ──
    render_model_path: str = (
        os.environ.get("MIS_RENDER_MODEL", "").strip()
        or r"D:\MIS_Pose_Track_Re3D\uvh_point_rt_runs\captured_photos2_v1\point_rt_renderfield_best.pt"
    )
    render_enabled: bool = bool(_ei("MIS_RENDER_ENABLED", 1))
    render_n_points: int = _ei("MIS_RENDER_N_POINTS", 4096)
    render_view_dist: float = _ef("MIS_RENDER_VIEW_DIST", 3.0)
    render_reject_thr: float = _ef("MIS_RENDER_REJECT_THR", 0.28)
    render_device: str = os.environ.get("MIS_RENDER_DEVICE", "cuda").strip()
    # 仅用「初始化正前方位姿」渲染/打分（与训练单视角一致，更快）
    render_init_pose_only: bool = bool(_ei("MIS_RENDER_INIT_POSE_ONLY", 0))
    # 滑动窗口：在搜索区内密集采样窗口，用渲染相似度找最佳位置
    render_sw_enable: bool = bool(_ei("MIS_RENDER_SW", 1))
    render_sw_stride: int = _ei("MIS_RENDER_SW_STRIDE", 24)
    render_sw_batch: int = _ei("MIS_RENDER_SW_BATCH", 12)
    render_sw_max_windows: int = _ei("MIS_RENDER_SW_MAX_WIN", 220)
    render_sw_min_pts: int = _ei("MIS_RENDER_SW_MIN_PTS", 80)
    # 局部最佳分数 ≥ 此值才用滑动窗结果覆盖 RTS bbox
    render_sw_replace_thr: float = _ef("MIS_RENDER_SW_REPLACE_THR", 0.40)
    # 与上一帧 bbox 的最小 IoU，低于则不用滑动窗覆盖（防跳到相似背景）；连续失败时可自动放宽
    render_sw_min_iou_prev: float = _ef("MIS_RENDER_SW_MIN_IOU", 0.14)
    # AE 可用且 ae_sw_scan≥0 时，胜出窗轮廓相似度低于此则仍不采用滑动窗覆盖
    render_replace_min_ae_sw: float = _ef("MIS_RENDER_REPLACE_MIN_AE", 0.22)
    # 局部最大分数 < 此值 → 在全图范围再做一遍稀疏滑动窗（默认关闭，全图易误锁背景）
    render_full_poor_thr: float = _ef("MIS_RENDER_FULL_POOR_THR", 0.26)
    render_full_enable: bool = bool(_ei("MIS_RENDER_FULL_ENABLE", 0))
    render_full_stride: int = _ei("MIS_RENDER_FULL_STRIDE", 56)
    render_full_max_windows: int = _ei("MIS_RENDER_FULL_MAX_WIN", 360)
    # 全图胜出时需在 replace 阈值上额外高出的分数（抑制弱匹配大跳）
    render_full_score_margin: float = _ef("MIS_RENDER_FULL_SCORE_MARGIN", 0.09)
    # 滑动窗扫描时仅用正前方单视角（快）；最终验证仍可用多视角
    render_sw_front_only: bool = bool(_ei("MIS_RENDER_SW_FRONT_ONLY", 1))
    # 仅当本帧实际采用了滑动窗 bbox 时才放宽位移；全图模式上限更低
    render_relax_shift_thr: float = _ef("MIS_RF_RELAX_SHIFT_THR", 0.52)
    render_relax_shift_mult: float = _ef("MIS_RF_RELAX_SHIFT_MULT", 1.48)
    render_relax_cap_full: float = _ef("MIS_RF_RELAX_CAP_FULL", 1.12)
    # AE 开启时：放宽位移还需 ae_sw_scan ≥ 此值（轮廓一致时才允许大跳）
    render_relax_min_ae_sw: float = _ef("MIS_RF_RELAX_MIN_AE", 0.30)

    # ── Autoencoder_129 轮廓语义约束（extract_semantic_features_Function.calculate_contour_similarity）──
    # 输出为归一化相似度 sim = 1/(1+dist)，范围约 [0,1]，与 core.semantic_similarity 一致
    ae_enable: bool = bool(_ei("MIS_AE_ENABLE", 0))
    ae_model_path: str = os.environ.get("MIS_MODEL_PATH", "").strip()
    ae_device: str = os.environ.get("MIS_DEVICE", "cuda").strip()
    ae_warmup: bool = bool(_ei("MIS_AE_WARMUP", 1))
    # 最终帧：低于此归一化相似度则否决（与 render_reject 并列）
    ae_reject_thr: float = _ef("MIS_AE_REJECT_THR", 0.24)
    # 滑动窗：对 render Top-K 候选用 AE 重排，融合分 = (1-w)*render + w*ae（略降权减轻误重排）
    ae_sw_weight: float = _ef("MIS_AE_SW_WEIGHT", 0.22)
    ae_sw_topk: int = _ei("MIS_AE_SW_TOPK", 10)

    # ── 旋转目标库 + AE 滑动窗跟踪 ──
    # 目标库规模：默认 4096(Y)×4(X倾角)=16384；设 MIS_LIB_N_ROT 时忽略下列三项，仅 Y 轴 n 等分
    lib_n_rot_y: int = _ei("MIS_LIB_N_ROT_Y", 4096)
    lib_n_rot_tilt_rings: int = _ei("MIS_LIB_N_ROT_TILT", 4)
    lib_tilt_span_deg: float = _ef("MIS_LIB_TILT_SPAN", 12.0)
    lib_render_batch: int = _ei("MIS_LIB_RENDER_BATCH", 32)
    lib_ae_batch: int = _ei("MIS_LIB_AE_BATCH", 32)
    lib_track_stride: int = _ei("MIS_LIB_TRACK_STRIDE", 16)
    lib_track_max_windows: int = _ei("MIS_LIB_TRACK_MAX_WIN", 480)
    lib_min_pts: int = _ei("MIS_LIB_MIN_PTS", 80)
    # >0：神经分取 Top-K 后，对 K 个候选用「整搜索区/整帧统一 th_68 伪3D → 按框裁剪」与 ref 做 optimize_rts 再定框；0=关闭
    lib_rts_topk: int = _ei("MIS_LIB_RTS_TOPK", 0)
    # 统一伪3D 的 th_68 范围：search=与滑动窗同一 search_bbox；frame=整幅当前帧（大分辨率时较慢）
    lib_mesh_unify_scope: str = os.environ.get("MIS_LIB_MESH_SCOPE", "search").strip().lower()
    # >0：RTS 改窗需优于「神经 argmax 窗」在统一伪3D 上的 optimize_rts 分至少该边际；0=Top-K 内 RTS 最高即改窗
    lib_rts_min_margin: float = _ef("MIS_LIB_RTS_MIN_MARGIN", 0.0)

    # ── 无旋转目标库：滑动窗 粗匹配（三路轮廓 AE 或回退 pack+轮廓）+ ICP ──
    match_expand_large: int = _ei("MIS_MATCH_EXPAND_LARGE", 128)
    match_retry_sim_thr: float = _ef("MIS_MATCH_RETRY_SIM", 0.48)
    # 主路径：RGB ROI、渲染 depth、渲染 normal 各自 th_68→轮廓→Autoencoder 相似度加权
    match_w_rgb_contour: float = _ef("MIS_MATCH_W_RGB", 0.34)
    match_w_depth_contour: float = _ef("MIS_MATCH_W_DEPTH", 0.33)
    match_w_normal_contour: float = _ef("MIS_MATCH_W_NORMAL", 0.33)
    # 轮廓语义未就绪时回退：pack latent + 单路 ROI 轮廓
    match_w_contour: float = _ef("MIS_MATCH_W_CONTOUR", 0.38)
    match_w_pack: float = _ef("MIS_MATCH_W_PACK", 0.62)
    icp_max_iters: int = _ei("MIS_ICP_ITERS", 30)
    icp_max_points: int = _ei("MIS_ICP_MAX_PTS", 3500)
    icp_rel_tol: float = _ef("MIS_ICP_TOL", 1e-5)

    # ── MeshSimilarityNet：候选窗伪3D vs target 伪3D 相似度（train_mesh_similarity_chess）──
    mesh_simi_enable: bool = bool(_ei("MIS_MESH_SIMI_ENABLE", 1))
    mesh_simi_ckpt: str = (
        os.environ.get("MIS_MESH_SIMI_CKPT", "").strip()
        or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "mesh_sim_runs",
            "chess_v1",
            "mesh_similarity_best.pt",
        )
    )
    mesh_simi_device: str = os.environ.get("MIS_MESH_SIMI_DEVICE", "cuda").strip()
    mesh_simi_n_points: int = _ei("MIS_MESH_SIMI_N_PTS", 1024)
    # 1：MeshSimilarity 从 th 高度场三角网格表面采样得 (u,v,h) 再归一化进网；0：沿用像素级伪 3D 点云
    mesh_simi_use_heightfield_mesh: bool = bool(_ei("MIS_MESH_SIMI_MESH_INPUT", 1))
    # 推理时表面采样池 = max(N_PTS * MULT, 256)；MULT 越大越稳越慢（原实现等效≈4 且每窗重复建参考网格）
    mesh_simi_mesh_infer_pool_mult: int = _ei("MIS_MESH_SIMI_MESH_POOL_MULT", 2)
    # 空：滑动窗用 MeshSimilarity 点云/网格；depth：伪深度 patch 相关（与 uvh 的 h 一致）；normal：Sobel 梯度幅值 patch 相关
    mesh_simi_patch_metric: str = os.environ.get("MIS_MESH_PATCH_METRIC", "").strip().lower()
    mesh_simi_patch_grid: int = _ei("MIS_MESH_PATCH_GRID", 64)
    mesh_simi_topk_rerank: int = _ei("MIS_MESH_SIMI_TOPK", 16)
    mesh_simi_fuse_weight: float = _ef("MIS_MESH_SIMI_W", 0.35)
    # 热身：默认按 epoch（500）× 每 epoch 样本对数；STEPS>0 时仅走步数模式并忽略 EPOCHS
    mesh_simi_train_epochs: int = _ei("MIS_MESH_SIMI_TRAIN_EPOCHS", 500)
    mesh_simi_train_samples_per_epoch: int = _ei("MIS_MESH_SIMI_TRAIN_SPA", 48)
    mesh_simi_train_batch: int = _ei("MIS_MESH_SIMI_TRAIN_BATCH", 12)
    mesh_simi_train_steps: int = _ei("MIS_MESH_SIMI_TRAIN_STEPS", 0)
    mesh_simi_train_lr: float = _ef("MIS_MESH_SIMI_TRAIN_LR", 8e-4)
    # 非空：热身权重保存路径；空且需要热身时用 output_dir/mesh_simi_online_warmup.pt
    mesh_simi_warmup_ckpt: str = os.environ.get("MIS_MESH_SIMI_WARMUP_CKPT", "").strip()
    # 在线热身：从首帧整图伪3D 随机取与 target 框低 IoU 的补丁作负样本，GT 相似度 ∈ [LO,HI]（默认 ≤0.1）
    mesh_simi_warmup_neg_enable: bool = bool(_ei("MIS_MESH_WARMUP_NEG", 1))
    mesh_simi_warmup_neg_ratio: float = _ef("MIS_MESH_WARMUP_NEG_RATIO", 0.45)
    mesh_simi_warmup_neg_label_lo: float = _ef("MIS_MESH_WARMUP_NEG_LO", 0.02)
    mesh_simi_warmup_neg_label_hi: float = _ef("MIS_MESH_WARMUP_NEG_HI", 0.10)
    mesh_simi_warmup_neg_iou_max: float = _ef("MIS_MESH_WARMUP_NEG_IOU", 0.10)
    mesh_simi_warmup_neg_retries: int = _ei("MIS_MESH_WARMUP_NEG_TRY", 80)
    mesh_simi_warmup_neg_min_pts: int = _ei("MIS_MESH_WARMUP_NEG_MINPTS", 24)
    # 同 target：归一化后仅旋转、无平移（旋转后再去质心）。默认 z=绕 h 轴即 u-v 平面内单角度；full=随机 SO(3)
    mesh_simi_warmup_rot_enable: bool = bool(_ei("MIS_MESH_WARMUP_ROT", 1))
    mesh_simi_warmup_rot_ratio: float = _ef("MIS_MESH_WARMUP_ROT_RATIO", 0.55)
    mesh_simi_warmup_rot_label: float = _ef("MIS_MESH_WARMUP_ROT_GT", 1.0)
    mesh_simi_warmup_rot_mode: str = os.environ.get(
        "MIS_MESH_WARMUP_ROT_MODE", "z"
    ).strip().lower()
    # 导出热身时实际喂给网络的 (ref, query, label) 子集，便于检查；0=关闭
    mesh_simi_warmup_export: bool = bool(_ei("MIS_MESH_WARMUP_EXPORT", 1))
    mesh_simi_warmup_export_max: int = _ei("MIS_MESH_WARMUP_EXPORT_MAX", 8192)
    mesh_simi_warmup_export_dir: str = os.environ.get(
        "MIS_MESH_WARMUP_EXPORT_DIR", ""
    ).strip()

    lib_rebuild: bool = bool(_ei("MIS_LIB_REBUILD", 0))
    # 融合分阈值（AE + pack + 轮廓）；pack 项已改为 ROI：RGB→th_68→轮廓→AE（见 lib_fuse_w_pack），不再用双渲染 pack_r4
    lib_sim_reject_thr: float = _ef("MIS_LIB_SIM_REJECT", 0.32)
    # AE 特征距离：cosine=行 L2 归一化后 0.5*(1+cos) 再按原权重融合；euclid=原 1/(1+L2)
    lib_ae_metric: str = os.environ.get("MIS_LIB_AE_METRIC", "cosine").strip().lower()
    lib_fuse_w_ae: float = _ef("MIS_LIB_FUSE_W_AE", 0.52)
    lib_fuse_w_pack: float = _ef("MIS_LIB_FUSE_W_PACK", 0.30)
    lib_fuse_w_contour: float = _ef("MIS_LIB_FUSE_W_CONTOUR", 0.18)
    # 库内 top1-top2 组合 AE 分间隔低于此则拒绝（0=关闭）
    lib_ae_margin_min: float = _ef("MIS_LIB_AE_MARGIN_MIN", 0.042)
    # lib_sim ≥ 此值时，放宽 center_shift 门限（乘 lib_relax_shift_mult），避免「高分匹配却被位移拒绝」
    lib_relax_shift_min_sim: float = _ef("MIS_LIB_RELAX_SHIFT_MIN_SIM", 0.68)
    lib_relax_shift_mult: float = _ef("MIS_LIB_RELAX_SHIFT_MULT", 1.85)
    # 次级：sim 略低时仍适度放宽（设 MIN_SIM2=999 可关闭）
    lib_relax_shift_min_sim_2: float = _ef("MIS_LIB_RELAX_SHIFT_MIN_SIM2", 0.62)
    lib_relax_shift_mult_2: float = _ef("MIS_LIB_RELAX_SHIFT_MULT2", 1.38)
    # 自适应扩大搜索：首遍后若 best_sim 低于阈值，用更大 expand 再扫（0=关闭该档）
    lib_adapt_sim_t1: float = _ef("MIS_LIB_ADAPT_SIM_T1", 0.55)
    lib_adapt_expand_add1: int = _ei("MIS_LIB_ADAPT_EXPAND_ADD1", 64)
    lib_adapt_sim_t2: float = _ef("MIS_LIB_ADAPT_SIM_T2", 0.48)
    lib_adapt_expand_add2: int = _ei("MIS_LIB_ADAPT_EXPAND_ADD2", 64)

    # 构建库时是否把固定视点渲染的 depth/normal/silhouette 存盘（仅 need_build 时写入）
    lib_save_renders: bool = bool(_ei("MIS_LIB_SAVE_RENDERS", 1))
    # 每隔多少个角度保存一组图（1=全存，角度多时会极大占用磁盘）
    lib_save_renders_stride: int = _ei("MIS_LIB_SAVE_RENDERS_STRIDE", 1)
    # >0 时优先：在 [0,N) 上均匀最多保存这么多组（忽略 stride）
    lib_save_renders_max: int = _ei("MIS_LIB_SAVE_RENDERS_MAX", 0)
    # 导出 PNG 最长边上限，0 表示用渲染原始分辨率
    lib_save_renders_max_side: int = _ei("MIS_LIB_SAVE_RENDERS_MAX_SIDE", 0)
    # 导出 PNG 时若原生分辨率较小，将短边至少放大到此像素（0=不强制；与 max_side 可同时用）
    lib_save_renders_min_side: int = _ei("MIS_LIB_SAVE_RENDERS_MIN_SIDE", 0)

    # 建库时导出「每视点 3D 点云 + 该视点神经渲染 RGB + AE 打包图」（不经过轮廓提取；仅 need_build 写入）
    lib_export_view_records: bool = bool(_ei("MIS_LIB_EXPORT_VIEW_RECORDS", 0))
    lib_export_view_stride: int = _ei("MIS_LIB_EXPORT_VIEW_STRIDE", 1)
    lib_export_view_max: int = _ei("MIS_LIB_EXPORT_VIEW_MAX", 0)
    lib_export_mesh_npy: bool = bool(_ei("MIS_LIB_EXPORT_MESH_NPY", 1))
    lib_export_mesh_ply: bool = bool(_ei("MIS_LIB_EXPORT_MESH_PLY", 1))
    lib_export_render_rgb: bool = bool(_ei("MIS_LIB_EXPORT_RENDER_RGB", 1))
    lib_export_ae_pack_png: bool = bool(_ei("MIS_LIB_EXPORT_AE_PACK_PNG", 1))
    # 导出 RGB/打包图最长边上限（0=不缩放）；点云为完整分辨率
    lib_export_rgb_max_side: int = _ei("MIS_LIB_EXPORT_RGB_MAX_SIDE", 512)

    # ── 目标库正前渲染视距（仅影响 render_batch_front：建库 + 跟踪候选；不影响 RenderField 多视角验证）──
    # MIS_LIB_VIEW_DIST>0 时优先；否则 视距 = MIS_RENDER_VIEW_DIST * MIS_LIB_VIEW_DIST_MULT
    lib_view_dist: float = _ef("MIS_LIB_VIEW_DIST", 0.0)
    lib_view_dist_mult: float = _ef("MIS_LIB_VIEW_DIST_MULT", 1.0)
    # 自动取景：根据首帧规范点云在正前渲染下前景占比，迭代调整视距（使目标在画面中比例适中）
    lib_view_auto: bool = bool(_ei("MIS_LIB_VIEW_AUTO", 0))
    lib_view_auto_cover_lo: float = _ef("MIS_LIB_AUTO_COVER_LO", 0.10)
    lib_view_auto_cover_hi: float = _ef("MIS_LIB_AUTO_COVER_HI", 0.42)
    lib_view_auto_d_min: float = _ef("MIS_LIB_AUTO_D_MIN", 1.12)
    lib_view_auto_d_max: float = _ef("MIS_LIB_AUTO_D_MAX", 6.5)
    lib_view_auto_step: float = _ef("MIS_LIB_AUTO_STEP", 0.12)
    lib_view_auto_max_iter: int = _ei("MIS_LIB_AUTO_MAX_ITER", 28)

    # AE 输入打包分辨率（宽×高，与 Autoencoder_129 训练尺寸不一致时仍可试，但可能影响相似度分布）
    ae_pack_w: int = _ei("MIS_AE_PACK_W", 128)
    ae_pack_h: int = _ei("MIS_AE_PACK_H", 128)
    # 每帧保存「库最佳角度渲染 vs 当前帧最佳窗渲染」对比图（额外 2×前向，较慢）
    lib_match_viz: bool = bool(_ei("MIS_LIB_MATCH_VIZ", 0))
    lib_match_viz_stride: int = _ei("MIS_LIB_MATCH_VIZ_STRIDE", 1)
    # 拒绝帧也强制存对比图（便于分析 center_shift 等）
    lib_match_viz_on_reject: bool = bool(_ei("MIS_LIB_MATCH_VIZ_ON_REJECT", 1))
    lib_match_viz_max_side: int = _ei("MIS_LIB_MATCH_VIZ_MAX_SIDE", 360)

    # 显示/保存
    save_debug: bool = True
    # mesh_match 四联大图：1=每帧；N>1 每 N 帧；0=不保存该面板（省大量 OpenCV 绘图）
    mesh_match_save_stride: int = _ei("MIS_MESH_MATCH_STRIDE", 1)
    verbose: int = _ei("MIS_VERBOSE", 1)
    print_every: int = _ei("MIS_PRINT_EVERY", 10)
    max_frames: int = _ei("MIS_MAX_FRAMES", 0)
    init_bbox: str = os.environ.get("MIS_INIT_BBOX", "").strip()
    vis_window: bool = True


def _clamp_ae_pack_dim(v: int, lo: int = 64, hi: int = 512) -> int:
    return int(max(lo, min(hi, int(v))))


def base_lib_render_view_dist(cfg: RTSConfig) -> float:
    """库/候选正前渲染使用的基准视距（规范空间下相机 Z 平移分量，越小越近）。"""
    if float(cfg.lib_view_dist) > 1e-9:
        return float(cfg.lib_view_dist)
    return float(cfg.render_view_dist) * float(cfg.lib_view_dist_mult)


def resolve_lib_front_distance_auto(
    rv: RenderFieldValidator,
    canon_probe: np.ndarray,
    cfg: RTSConfig,
) -> float:
    """
    在 base_lib_render_view_dist 基础上可选自动调整：前景占比过低则拉近（减小 d），过高则拉远。
    canon_probe 为已规范化的首帧点云（与建库时 canon0 一致）。
    """
    d = float(base_lib_render_view_dist(cfg))
    if not cfg.lib_view_auto or not rv.enabled:
        return d
    lo = float(cfg.lib_view_auto_cover_lo)
    hi = float(cfg.lib_view_auto_cover_hi)
    d_min = float(cfg.lib_view_auto_d_min)
    d_max = float(cfg.lib_view_auto_d_max)
    step = float(cfg.lib_view_auto_step)
    n_iter = max(1, int(cfg.lib_view_auto_max_iter))

    def _cover_at(dist: float) -> float:
        lst = rv.render_batch_front([canon_probe], batch_size=1, view_dist=float(dist))
        r = lst[0] if lst else None
        if r is None:
            return -1.0
        rr = np.asarray(r, dtype=np.float32)
        if rr.ndim != 3 or rr.shape[0] < 1:
            return -1.0
        sil = rr[0] > 1e-6
        return float(np.mean(sil.astype(np.float32)))

    last_cov = -1.0
    for _ in range(n_iter):
        cov = _cover_at(d)
        last_cov = cov
        if cov < 0:
            break
        if lo <= cov <= hi:
            break
        if cov < lo:
            d = max(d_min, d - step)
        else:
            d = min(d_max, d + step)
    _log(
        f"[目标库] 视距自动取景: d={d:.3f} (base={base_lib_render_view_dist(cfg):.3f}, "
        f"最后前景占比≈{last_cov:.3f}, 目标区间 [{lo:.2f},{hi:.2f}])",
        cfg,
    )
    return float(d)


def _make_core_cfg(cfg: RTSConfig) -> core.TrackConfig:
    """用 RTSConfig 构造 core.TrackConfig 供复用工具函数"""
    c = core.TrackConfig()
    c.frames_dir = cfg.frames_dir
    c.output_dir = cfg.output_dir
    c.proc_scale = cfg.proc_scale
    c.th_percent = cfg.th_percent
    c.bilateral_d = cfg.bilateral_d
    c.bilateral_sc = cfg.bilateral_sc
    c.bilateral_ss = cfg.bilateral_ss
    c.mesh_z_scale = cfg.mesh_z_scale
    c.init_bbox = cfg.init_bbox
    c.vis_window = cfg.vis_window
    c.save_debug_images = cfg.save_debug
    c.verbose = cfg.verbose
    c.print_every = cfg.print_every
    c.max_frames = cfg.max_frames
    c.semantic_image_size = (
        _clamp_ae_pack_dim(cfg.ae_pack_w),
        _clamp_ae_pack_dim(cfg.ae_pack_h),
    )
    c.model_path = str(cfg.ae_model_path).strip()
    c.device = str(cfg.ae_device).strip()
    c.semantic_warmup = bool(cfg.ae_warmup)
    return c


def _mesh_warmup_should_run(cfg: RTSConfig) -> bool:
    return int(cfg.mesh_simi_train_steps) > 0 or int(cfg.mesh_simi_train_epochs) > 0


def _mesh_warmup_save_path(cfg: RTSConfig) -> str:
    """在线热身权重保存路径；未启用热身时返回空串。"""
    if not _mesh_warmup_should_run(cfg):
        return ""
    custom = str(cfg.mesh_simi_warmup_ckpt).strip()
    if custom:
        return os.path.abspath(custom)
    od = str(cfg.output_dir).strip()
    root = os.path.abspath(od) if od else os.getcwd()
    return os.path.join(root, "mesh_simi_online_warmup.pt")


def _write_point_cloud_ply_xyz(path: str, pts: np.ndarray) -> None:
    """将 (N,3) 点云写入 binary_little_endian PLY（x,y,z float32），便于 MeshLab / CloudCompare 查看。"""
    p = np.asarray(pts, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"expected (N,3) points, got {p.shape}")
    n = int(p.shape[0])
    hdr = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    d = os.path.dirname(path)
    if d:
        _ensure_dir(d)
    with open(path, "wb") as f:
        f.write(hdr.encode("ascii"))
        f.write(np.ascontiguousarray(p).tobytes())


def build_uvh_heightfield_mesh(
    th_gray: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    z_scale: float,
    *,
    min_th: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 ROI 内 th 图视为高度场：每个前景像素为一顶点，四邻域成四边形剖分为两个三角形。
    顶点坐标为整帧像素系 [u,v,h]，h = (th/255)*z_scale。
    """
    bx, by, w, h = (int(v) for v in bbox_xywh)
    img = np.asarray(th_gray, dtype=np.float32)
    if img.ndim != 2:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    active = img > float(min_th)
    if int(np.count_nonzero(active)) < 3:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    index = np.full((h, w), -1, dtype=np.int32)
    ys, xs = np.nonzero(active)
    n_v = int(xs.size)
    index[ys, xs] = np.arange(n_v, dtype=np.int32)
    u = xs.astype(np.float32) + float(bx)
    v = ys.astype(np.float32) + float(by)
    z = (img[ys, xs] / 255.0) * float(z_scale)
    vertices = np.stack([u, v, z.astype(np.float32)], axis=1).astype(np.float32)
    faces_l: list[list[int]] = []
    for yy in range(h - 1):
        for xx in range(w - 1):
            i00 = int(index[yy, xx])
            i10 = int(index[yy, xx + 1])
            i01 = int(index[yy + 1, xx])
            i11 = int(index[yy + 1, xx + 1])
            if i00 < 0 or i10 < 0 or i01 < 0 or i11 < 0:
                continue
            faces_l.append([i00, i10, i01])
            faces_l.append([i10, i11, i01])
    if not faces_l:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32)
    faces = np.asarray(faces_l, dtype=np.int32)
    return vertices, faces


def sample_points_from_triangle_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """按三角形面积比例在网格表面均匀采样 n 个 3D 点。"""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if v.shape[0] < 3 or f.shape[0] < 1 or n <= 0:
        return np.zeros((0, 3), np.float32)
    v0 = v[f[:, 0]]
    v1 = v[f[:, 1]]
    v2 = v[f[:, 2]]
    cr = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cr, axis=1)
    areas = np.maximum(areas, 0.0)
    s = float(np.sum(areas))
    n = int(n)
    if s < 1e-24:
        pick = rng.integers(0, int(v.shape[0]), size=n, endpoint=False)
        return v[pick].astype(np.float32)
    probs = (areas / s).astype(np.float64)
    probs /= np.sum(probs)
    tri_idx = rng.choice(int(f.shape[0]), size=n, p=probs)
    r1 = rng.random(n).astype(np.float64)
    r2 = rng.random(n).astype(np.float64)
    sr1 = np.sqrt(r1)
    w0 = 1.0 - sr1
    w1 = sr1 * (1.0 - r2)
    w2 = sr1 * r2
    p0 = v0[tri_idx]
    p1 = v1[tri_idx]
    p2 = v2[tri_idx]
    pts = (
        w0[:, None] * p0 + w1[:, None] * p1 + w2[:, None] * p2
    ).astype(np.float32)
    return pts


def mesh_surface_points_uvh(
    th_gray: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    z_scale: float,
    n_samples: int,
    rng: np.random.Generator,
    *,
    min_th: float = 1e-6,
) -> np.ndarray:
    """由 th ROI + 全局 bbox 建高度场三角网格，再表面采样得到 (N,3) uvh。"""
    verts, faces = build_uvh_heightfield_mesh(
        th_gray, bbox_xywh, z_scale, min_th=min_th
    )
    if int(faces.shape[0]) < 1 or int(verts.shape[0]) < 3:
        return np.zeros((0, 3), np.float32)
    return sample_points_from_triangle_mesh(
        verts, faces, int(n_samples), rng
    )


def pseudo_depth_u8_from_th(th_f: np.ndarray) -> np.ndarray:
    """th_68 截断灰度 float → 前景内 min-max 归一化到 uint8，作伪深度图（与 uvh 的 h 同源）。"""
    t = np.asarray(th_f, dtype=np.float32)
    u8 = np.zeros(t.shape, dtype=np.uint8)
    m = t > 1e-6
    if not np.any(m):
        return u8
    vals = t[m]
    lo, hi = float(np.min(vals)), float(np.max(vals))
    rng = max(hi - lo, 1e-6)
    u8[m] = np.clip(((t[m] - lo) / rng) * 255.0, 0, 255).astype(np.uint8)
    return u8


def pseudo_normal_gray_u8_from_th(th_f: np.ndarray) -> np.ndarray:
    """对 th 浮点图一次 Sobel 梯度幅值 → 归一化 uint8，作伪法线/边缘强度图。"""
    f = np.asarray(th_f, dtype=np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    u8 = np.zeros(f.shape, dtype=np.uint8)
    m = mag > 1e-8
    if not np.any(m):
        return u8
    lo = float(np.min(mag[m]))
    hi = float(np.max(mag[m]))
    rng = max(hi - lo, 1e-6)
    u8[m] = np.clip(((mag[m] - lo) / rng) * 255.0, 0, 255).astype(np.uint8)
    return u8


def th_pair_patch_similarity(
    th_ref: np.ndarray,
    th_query: np.ndarray,
    *,
    mode: str,
    grid: int,
) -> float:
    """
    两幅 ROI 的 th_68 图：对齐到 ref 尺寸后，在伪深度或伪梯度图上算带掩膜皮尔逊相关，映射到 [0,1]。
    mode: depth | normal
    """
    m = (mode or "depth").strip().lower()
    a = np.asarray(th_ref, dtype=np.float32)
    b = np.asarray(th_query, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        return 0.0
    ha, wa = int(a.shape[0]), int(a.shape[1])
    hb, wb = int(b.shape[0]), int(b.shape[1])
    if ha < 2 or wa < 2:
        return 0.0
    if (ha, wa) != (hb, wb):
        b = cv2.resize(b, (wa, ha), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    if m == "normal":
        da = pseudo_normal_gray_u8_from_th(a).astype(np.float64) / 255.0
        db = pseudo_normal_gray_u8_from_th(b).astype(np.float64) / 255.0
    else:
        da = pseudo_depth_u8_from_th(a).astype(np.float64) / 255.0
        db = pseudo_depth_u8_from_th(b).astype(np.float64) / 255.0
    g = max(8, int(grid))
    da = cv2.resize(da, (g, g), interpolation=cv2.INTER_AREA)
    db = cv2.resize(db, (g, g), interpolation=cv2.INTER_AREA)
    mask = (da > 1e-6) & (db > 1e-6)
    if int(np.count_nonzero(mask)) < 8:
        return 0.0
    va = da[mask] - float(np.mean(da[mask]))
    vb = db[mask] - float(np.mean(db[mask]))
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    corr = float(np.dot(va, vb) / (na * nb))
    return float(np.clip(0.5 * (corr + 1.0), 0.0, 1.0))


def _warmup_occlude_th(th: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    t = np.asarray(th, dtype=np.float32).copy()
    if frac <= 1e-6:
        return t
    h, w = int(t.shape[0]), int(t.shape[1])
    drop = rng.random((h, w)) < float(np.clip(frac, 0.0, 0.95))
    t[drop] = 0.0
    return t


def _warmup_rotate_th_2d(th: np.ndarray, deg: float) -> np.ndarray:
    t = np.asarray(th, dtype=np.float32)
    h, w = int(t.shape[0]), int(t.shape[1])
    if h < 2 or w < 2:
        return t.copy()
    M = cv2.getRotationMatrix2D((0.5 * (w - 1), 0.5 * (h - 1)), float(deg), 1.0)
    return cv2.warpAffine(
        t,
        M,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)


def _warmup_sample_negative_th_resized(
    th_full: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    frame_wh: tuple[int, int],
    out_hw: tuple[int, int],
    *,
    iou_max: float,
    rng: np.random.Generator,
    max_tries: int,
) -> np.ndarray | None:
    """与负样本窗同分布，返回与 ref 同尺寸的 th 裁剪（双线性缩放到 out_hw）。"""
    W, H = int(frame_wh[0]), int(frame_wh[1])
    oh, ow = int(out_hw[0]), int(out_hw[1])
    th2 = np.asarray(th_full, dtype=np.float32)
    if th2.ndim != 2 or th2.shape[0] != H or th2.shape[1] != W:
        return None
    _, _, tw, th = (int(v) for v in target_bbox)
    tpl_w = max(4, min(tw, W))
    tpl_h = max(4, min(th, H))
    if tpl_w > W or tpl_h > H or oh < 2 or ow < 2:
        return None
    iou_m = float(iou_max)
    for _ in range(max(1, int(max_tries))):
        xmax = max(0, W - tpl_w)
        ymax = max(0, H - tpl_h)
        x = int(rng.integers(0, xmax + 1)) if xmax > 0 else 0
        y = int(rng.integers(0, ymax + 1)) if ymax > 0 else 0
        cand = (x, y, tpl_w, tpl_h)
        if bbox_iou_xywh(cand, target_bbox) > iou_m:
            continue
        crop = np.ascontiguousarray(th2[y : y + tpl_h, x : x + tpl_w])
        if int(np.count_nonzero(crop > 1e-6)) < 8:
            continue
        return cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_NEAREST).astype(
            np.float32
        )
    return None


def mesh_similarity_quick_warmup(
    ref_pts: np.ndarray,
    cfg: RTSConfig,
    *,
    frame_uvh_pts: np.ndarray | None = None,
    target_bbox: tuple[int, int, int, int] | None = None,
    frame_wh: tuple[int, int] | None = None,
    ref_th_roi: np.ndarray | None = None,
    frame_th_full: np.ndarray | None = None,
) -> tuple[str, str]:
    """
    首帧 target 热身：① 归一化后仅旋转（默认绕 z/h，单角度；可 full）→ 去质心 → 与 target 成对，GT=1；
    ② 随机遮挡正样本；③ 可选整帧低 IoU 负样本（GT 低）。
    MIS_MESH_SIMI_MESH_INPUT=1 且传入 ref_th_roi / frame_th_full 时，target 与负样本 query 由 th 高度场三角网格表面采样得到。
    默认写入 output_dir/mesh_simi_online_warmup.pt，不覆盖 MIS_MESH_SIMI_CKPT。
    返回 (ckpt_path, export_dir)；export_dir 为导出 PLY/manifest 的根目录，未导出或失败时为空串。
    """
    if not bool(cfg.mesh_simi_enable):
        _log("[MeshSimi] MIS_MESH_SIMI_ENABLE=0，跳过在线热身。", cfg)
        return "", ""
    use_step_mode = int(cfg.mesh_simi_train_steps) > 0
    n_epochs_cfg = max(0, int(cfg.mesh_simi_train_epochs))
    n_step_only = max(0, int(cfg.mesh_simi_train_steps))
    if not use_step_mode and n_epochs_cfg <= 0:
        _log(
            "[MeshSimi] 在线热身已关闭（MIS_MESH_SIMI_TRAIN_EPOCHS=0 且 MIS_MESH_SIMI_TRAIN_STEPS=0），"
            "将仅用预训练/已有权重。",
            cfg,
        )
        return "", ""
    out_ckpt = _mesh_warmup_save_path(cfg)
    if not out_ckpt:
        return "", ""
    base_ckpt = str(cfg.mesh_simi_ckpt).strip()
    try:
        import train_mesh_similarity_chess as tmc
        from mesh_similarity_model import MeshSimilarityNet
    except Exception as e:
        _log(f"[MeshSimi] 热身跳过（导入失败）: {e}", cfg)
        return "", ""
    dev = torch.device(
        cfg.mesh_simi_device
        if torch.cuda.is_available() and str(cfg.mesh_simi_device).startswith("cuda")
        else "cpu"
    )
    use_hf_mesh = bool(cfg.mesh_simi_use_heightfield_mesh)
    z_s = float(cfg.mesh_z_scale)
    pool_n = max(int(cfg.mesh_simi_n_points) * 8, 4096)
    ref0 = np.asarray(ref_pts, dtype=np.float32)
    if (
        use_hf_mesh
        and ref_th_roi is not None
        and target_bbox is not None
        and int(np.asarray(ref_th_roi).size) > 0
    ):
        rth = np.asarray(ref_th_roi, dtype=np.float32)
        ref0 = mesh_surface_points_uvh(
            rth, target_bbox, z_s, pool_n, np.random.default_rng(12345)
        )
        if int(ref0.shape[0]) < 32:
            _log(
                f"[MeshSimi] 高度场网格采样点过少 {int(ref0.shape[0])}，回退伪3D点云热身",
                cfg,
            )
            ref0 = np.asarray(ref_pts, dtype=np.float32)
    ref_n = tmc.normalize_point_cloud(ref0)
    if ref_n.shape[0] < 32:
        _log(f"[MeshSimi] 热身跳过：点过少 {ref_n.shape[0]}", cfg)
        return "", ""
    n_pts = int(cfg.mesh_simi_n_points)
    custom_ex = str(cfg.mesh_simi_warmup_export_dir).strip()
    export_dir_base = (
        os.path.abspath(custom_ex)
        if custom_ex
        else os.path.join(
            os.path.dirname(os.path.abspath(out_ckpt)), "mesh_simi_warmup_export"
        )
    )
    export_on = (
        bool(cfg.mesh_simi_warmup_export)
        and int(cfg.mesh_simi_warmup_export_max) > 0
    )
    export_max = max(0, int(cfg.mesh_simi_warmup_export_max))
    ex_ref: list[np.ndarray] = []
    ex_q: list[np.ndarray] = []
    ex_lab: list[float] = []
    ex_tag: list[int] = []

    def _export_push(
        r_arr: np.ndarray, q_arr: np.ndarray, lab: float, tag: int
    ) -> None:
        if not export_on or len(ex_ref) >= export_max:
            return
        ex_ref.append(np.asarray(r_arr, dtype=np.float32).copy())
        ex_q.append(np.asarray(q_arr, dtype=np.float32).copy())
        ex_lab.append(float(lab))
        ex_tag.append(int(tag))

    if export_on:
        _log(
            f"[MeshSimi] 训练数据导出: 至多 {export_max} 对 → {export_dir_base}",
            cfg,
        )

    model = MeshSimilarityNet(in_dim=3, embed_dim=256, head_hidden=256).to(dev)
    if base_ckpt and os.path.isfile(base_ckpt):
        try:
            try:
                ck0 = torch.load(base_ckpt, map_location=dev, weights_only=False)
            except TypeError:
                ck0 = torch.load(base_ckpt, map_location=dev)
            model.load_state_dict(ck0["model"], strict=True)
            _log(f"[MeshSimi] 在线热身：从预训练初始化 {base_ckpt}", cfg)
        except Exception as e:
            _log(f"[MeshSimi] 预训练加载失败，从零热身: {e}", cfg)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.mesh_simi_train_lr), weight_decay=1e-4
    )
    loss_fn = nn.SmoothL1Loss(beta=0.05)
    rng = np.random.default_rng(12345)
    model.train()
    occ_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    batch = max(1, int(cfg.mesh_simi_train_batch))
    spa = max(1, int(cfg.mesh_simi_train_samples_per_epoch))
    n_ref = int(ref_n.shape[0])
    neg_lo = float(
        np.clip(float(cfg.mesh_simi_warmup_neg_label_lo), 0.0, 0.49))
    neg_hi = float(
        np.clip(float(cfg.mesh_simi_warmup_neg_label_hi), neg_lo, 1.0))
    neg_ratio = float(
        np.clip(float(cfg.mesh_simi_warmup_neg_ratio), 0.0, 0.95))
    fp_uvh = (
        np.asarray(frame_uvh_pts, dtype=np.float32)
        if frame_uvh_pts is not None
        else np.zeros((0, 3), np.float32)
    )
    n_frame = int(fp_uvh.shape[0])
    neg_mesh_ok = (
        use_hf_mesh
        and frame_th_full is not None
        and target_bbox is not None
        and frame_wh is not None
        and int(np.asarray(frame_th_full).size) > 0
    )
    neg_on = (
        bool(cfg.mesh_simi_warmup_neg_enable)
        and neg_ratio > 1e-6
        and target_bbox is not None
        and frame_wh is not None
        and (
            (
                neg_mesh_ok
                and int(np.asarray(frame_th_full).ndim) == 2
            )
            or (
                n_frame >= int(cfg.mesh_simi_warmup_neg_min_pts) * 4
            )
        )
    )
    neg_fallbacks = 0
    rot_on = bool(cfg.mesh_simi_warmup_rot_enable)
    rot_ratio = float(
        np.clip(float(cfg.mesh_simi_warmup_rot_ratio), 0.0, 1.0))
    rot_gt = float(
        np.clip(float(cfg.mesh_simi_warmup_rot_label), 0.0, 1.0))
    rot_mode = str(cfg.mesh_simi_warmup_rot_mode).strip().lower()
    patch_metric = str(cfg.mesh_simi_patch_metric or "").strip().lower()
    patch_lab = (
        patch_metric in ("depth", "normal")
        and ref_th_roi is not None
        and int(np.asarray(ref_th_roi).size) > 0
    )
    ref_th0 = (
        np.asarray(ref_th_roi, dtype=np.float32)
        if patch_lab
        else np.zeros((0, 0), np.float32)
    )
    pgrid = max(8, int(cfg.mesh_simi_patch_grid))

    def run_batch(bs: int) -> tuple[float, float]:
        """单步反传：负样本 / 旋转等变正样本(GT≈1) / 遮挡正样本。"""
        nonlocal neg_fallbacks
        rows_r: list[np.ndarray] = []
        rows_q: list[np.ndarray] = []
        labels: list[float] = []
        for _ in range(bs):
            if neg_on and float(rng.random()) < neg_ratio:
                neg_raw = None
                if neg_mesh_ok:
                    neg_raw = _mesh_warmup_sample_negative_mesh(
                        np.asarray(frame_th_full, dtype=np.float32),
                        target_bbox,  # type: ignore[arg-type]
                        frame_wh,  # type: ignore[arg-type]
                        z_scale=z_s,
                        n_surface_samples=pool_n,
                        iou_max=float(cfg.mesh_simi_warmup_neg_iou_max),
                        min_pts=int(cfg.mesh_simi_warmup_neg_min_pts),
                        rng=rng,
                        max_tries=int(cfg.mesh_simi_warmup_neg_retries),
                    )
                if neg_raw is None:
                    neg_raw = _mesh_warmup_sample_negative_uvh(
                        fp_uvh,
                        target_bbox,  # type: ignore[arg-type]
                        frame_wh,  # type: ignore[arg-type]
                        iou_max=float(cfg.mesh_simi_warmup_neg_iou_max),
                        min_pts=int(cfg.mesh_simi_warmup_neg_min_pts),
                        rng=rng,
                        max_tries=int(cfg.mesh_simi_warmup_neg_retries),
                    )
                if neg_raw is not None:
                    qn = tmc.normalize_point_cloud(
                        np.asarray(neg_raw, dtype=np.float32))
                    q = tmc.resample_points(qn, n_pts, rng)
                    r = tmc.resample_points(ref_n, n_pts, rng)
                    if (
                        patch_lab
                        and frame_th_full is not None
                        and target_bbox is not None
                        and frame_wh is not None
                    ):
                        th_neg = _warmup_sample_negative_th_resized(
                            np.asarray(frame_th_full, dtype=np.float32),
                            target_bbox,
                            frame_wh,
                            (int(ref_th0.shape[0]), int(ref_th0.shape[1])),
                            iou_max=float(cfg.mesh_simi_warmup_neg_iou_max),
                            rng=rng,
                            max_tries=int(cfg.mesh_simi_warmup_neg_retries),
                        )
                        if th_neg is not None:
                            lab = float(
                                th_pair_patch_similarity(
                                    ref_th0,
                                    th_neg,
                                    mode=patch_metric,
                                    grid=pgrid,
                                )
                            )
                        else:
                            lab = float(rng.uniform(neg_lo, neg_hi))
                    else:
                        lab = float(rng.uniform(neg_lo, neg_hi))
                    rows_r.append(r)
                    rows_q.append(q)
                    labels.append(lab)
                    _export_push(r, q, lab, 0)
                    continue
                neg_fallbacks += 1
            if rot_on and float(rng.random()) < rot_ratio:
                Rm = _warmup_random_rotation_matrix(rng, rot_mode)
                ref_rot = (ref_n @ Rm.T).astype(np.float32)
                ref_rot = ref_rot - np.mean(
                    ref_rot, axis=0, keepdims=True, dtype=np.float64
                ).astype(np.float32)
                r = tmc.resample_points(ref_n, n_pts, rng)
                q = tmc.resample_points(ref_rot, n_pts, rng)
                if patch_lab:
                    deg = float(rng.uniform(0.0, 360.0))
                    tw = _warmup_rotate_th_2d(ref_th0, deg)
                    lab = float(
                        th_pair_patch_similarity(
                            ref_th0, tw, mode=patch_metric, grid=pgrid
                        )
                    )
                else:
                    lab = rot_gt
                rows_r.append(r)
                rows_q.append(q)
                labels.append(lab)
                _export_push(r, q, lab, 1)
                continue
            occ = float(rng.choice(occ_levels))
            q_sub = tmc.random_occlude_points(ref_n, occ, rng)
            q = tmc.resample_points(q_sub, n_pts, rng)
            r = tmc.resample_points(ref_n, n_pts, rng)
            if patch_lab:
                tocc = _warmup_occlude_th(ref_th0, occ, rng)
                lab = float(
                    th_pair_patch_similarity(
                        ref_th0, tocc, mode=patch_metric, grid=pgrid
                    )
                )
            else:
                lab = max(0.0, min(1.0, 1.0 - occ))
            rows_r.append(r)
            rows_q.append(q)
            labels.append(lab)
            _export_push(r, q, lab, 2)
        t_r = torch.from_numpy(np.stack(rows_r, axis=0)).float().to(dev)
        t_q = torch.from_numpy(np.stack(rows_q, axis=0)).float().to(dev)
        y = torch.tensor(labels, dtype=torch.float32, device=dev)
        opt.zero_grad(set_to_none=True)
        pred = model(t_r, t_q)
        loss = loss_fn(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        lab_mean = float(np.mean(labels))
        return float(loss.detach().cpu()), lab_mean

    ckpt_epoch_field = -1
    if use_step_mode:
        total_opt_steps = n_step_only
        total_pair_samples = total_opt_steps * batch
        log_every = max(1, total_opt_steps // 10)
        _log(
            f"[MeshSimi] 数据集统计: target{'网格面采样' if use_hf_mesh and ref_th_roi is not None else '点云'} N={n_ref} "
            f"整帧{'th全图' if neg_mesh_ok else '伪3D'} N={n_frame if not neg_mesh_ok else int(np.asarray(frame_th_full).size)} "
            f"每对点数={n_pts} "
            f"遮挡档={len(occ_levels)} | 负样本={'开' if neg_on else '关'}"
            + (
                f" 比例≈{neg_ratio:.2f} GT∈[{neg_lo:.3f},{neg_hi:.3f}] IoU≤{cfg.mesh_simi_warmup_neg_iou_max}"
                if neg_on
                else ""
            )
            + (
                f" | 旋转正样本={'开' if rot_on else '关'}"
                + (
                    f" 比例≈{rot_ratio:.2f} GT={rot_gt:.3f} rot_mode={rot_mode}"
                    if rot_on
                    else ""
                )
            )
            + (
                f" | patch标签={patch_metric}(grid={pgrid})"
                if patch_lab
                else ""
            )
            + f" | 模式=按步数 STEPS={total_opt_steps} BATCH={batch} "
            f"→ 总样本对≈{total_pair_samples} 总反传步={total_opt_steps}",
            cfg,
        )
        _log(
            f"[MeshSimi] 在线热身开始 lr={cfg.mesh_simi_train_lr} device={dev} → {out_ckpt}",
            cfg,
        )
        for step_i in range(total_opt_steps):
            loss_v, lab_m = run_batch(batch)
            if step_i == 0 or (step_i + 1) % log_every == 0 or step_i + 1 == total_opt_steps:
                _log(
                    f"[MeshSimi] 步 {step_i + 1}/{total_opt_steps}  loss={loss_v:.5f}  "
                    f"label均值≈{lab_m:.2f}",
                    cfg,
                )
        ckpt_epoch_field = -1
    else:
        n_ep = max(1, n_epochs_cfg)
        steps_per_epoch = max(1, (spa + batch - 1) // batch)
        total_opt_steps = n_ep * steps_per_epoch
        total_pair_samples = n_ep * spa
        log_epoch_every = max(1, n_ep // 20)
        _log(
            f"[MeshSimi] 数据集统计: target{'网格面采样' if use_hf_mesh and ref_th_roi is not None else '点云'} N={n_ref} "
            f"整帧{'th全图' if neg_mesh_ok else '伪3D'} N={n_frame if not neg_mesh_ok else int(np.asarray(frame_th_full).size)} "
            f"每对点数={n_pts} "
            f"遮挡档={len(occ_levels)} | 负样本={'开' if neg_on else '关'}"
            + (
                f" 比例≈{neg_ratio:.2f} GT∈[{neg_lo:.3f},{neg_hi:.3f}] IoU≤{cfg.mesh_simi_warmup_neg_iou_max}"
                if neg_on
                else ""
            )
            + (
                f" | 旋转正样本={'开' if rot_on else '关'}"
                + (
                    f" 比例≈{rot_ratio:.2f} GT={rot_gt:.3f} rot_mode={rot_mode}"
                    if rot_on
                    else ""
                )
            )
            + (
                f" | patch标签={patch_metric}(grid={pgrid})"
                if patch_lab
                else ""
            )
            + f" | 模式=epoch EPOCHS={n_ep} SPA={spa} BATCH={batch} "
            f"每epoch反传≈{steps_per_epoch} → 总样本对={total_pair_samples} 总反传≈{total_opt_steps}",
            cfg,
        )
        _log(
            f"[MeshSimi] 在线热身开始 lr={cfg.mesh_simi_train_lr} device={dev} → {out_ckpt}",
            cfg,
        )
        global_step = 0
        for ep in range(n_ep):
            elosses: list[float] = []
            remaining = spa
            while remaining > 0:
                bs = min(batch, remaining)
                loss_v, _ = run_batch(bs)
                elosses.append(loss_v)
                remaining -= bs
                global_step += 1
            mean_ep = float(np.mean(elosses)) if elosses else 0.0
            if ep == 0 or (ep + 1) % log_epoch_every == 0 or ep + 1 == n_ep:
                _log(
                    f"[MeshSimi] epoch {ep + 1}/{n_ep}  mean_loss={mean_ep:.5f}  "
                    f"本epoch样本对={spa} 累计反传步={global_step}",
                    cfg,
                )
        ckpt_epoch_field = int(n_ep)
    export_saved = ""
    if export_on and len(ex_ref) > 0:
        try:
            os.makedirs(export_dir_base, exist_ok=True)
            pairs_subdir = os.path.join(export_dir_base, "ply_pairs")
            os.makedirs(pairs_subdir, exist_ok=True)
            n_pairs = len(ex_ref)
            manifest_pairs: list[dict[str, Any]] = []
            for i in range(n_pairs):
                bn = f"pair_{i:05d}"
                pref = os.path.join(pairs_subdir, f"{bn}_ref.ply")
                pq = os.path.join(pairs_subdir, f"{bn}_query.ply")
                _write_point_cloud_ply_xyz(pref, ex_ref[i])
                _write_point_cloud_ply_xyz(pq, ex_q[i])
                manifest_pairs.append(
                    {
                        "index": i,
                        "ref_ply": f"ply_pairs/{bn}_ref.ply",
                        "query_ply": f"ply_pairs/{bn}_query.ply",
                        "label": float(ex_lab[i]),
                        "tag": int(ex_tag[i]),
                    }
                )
            meta_js = {
                "format": "ply_binary_little_endian",
                "ply_subdir": "ply_pairs",
                "tag_map": {
                    "0": "spatial_negative",
                    "1": "rotation_positive",
                    "2": "occlusion_positive",
                },
                "n_saved": int(n_pairs),
                "n_pts_per_cloud": int(n_pts),
                "export_cap": int(export_max),
                "may_be_truncated": bool(n_pairs >= export_max),
                "pairs": manifest_pairs,
            }
            man_path = os.path.join(export_dir_base, "warmup_manifest.json")
            with open(man_path, "w", encoding="utf-8") as fj:
                json.dump(meta_js, fj, ensure_ascii=False, indent=2)
            read_me = os.path.join(export_dir_base, "README_查看说明.txt")
            with open(read_me, "w", encoding="utf-8") as fr:
                fr.write(
                    "MeshSimilarity 在线热身导出（与训练中送入网络的点云一致）\n\n"
                    "目录 ply_pairs/ 下每条样本两个文件：\n"
                    "  pair_XXXXX_ref.ply   — 与 forward(pts_a) 一致\n"
                    "  pair_XXXXX_query.ply — 与 forward(pts_b) 一致\n"
                    "PLY 为 binary_little_endian，float32 x,y,z；可用 MeshLab、CloudCompare、Open3D 打开。\n\n"
                    "warmup_manifest.json：每条含 index, ref_ply, query_ply, label(相似度GT), tag\n"
                    "  tag: 0=整帧空间负样本 1=同target旋转正 2=遮挡正样本\n\n"
                    "Open3D 示例:\n"
                    "  import open3d as o3d\n"
                    "  a = o3d.io.read_point_cloud('ply_pairs/pair_00000_ref.ply')\n"
                    "  b = o3d.io.read_point_cloud('ply_pairs/pair_00000_query.ply')\n\n"
                    f"若训练对总数超过 MIS_MESH_WARMUP_EXPORT_MAX(={export_max})，"
                    "此处仅为按生成顺序截取的前缀子集。\n"
                )
            export_saved = os.path.abspath(export_dir_base)
            _log(
                f"[MeshSimi] 热身数据集已保存: {pairs_subdir}/  共 {n_pairs} 对 PLY + {man_path}",
                cfg,
            )
        except Exception as e:
            _log(f"[MeshSimi] 导出热身数据集失败(忽略): {e}", cfg)
    ck_dir = os.path.dirname(os.path.abspath(out_ckpt))
    if ck_dir:
        os.makedirs(ck_dir, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": int(ckpt_epoch_field),
            "config": {
                "note": "MIS_Mesh_RTS_Track online warmup from first target",
                "n_ref_points": n_ref,
                "n_pts_per_cloud": n_pts,
                "occ_levels_n": len(occ_levels),
                "train_batch": batch,
                "samples_per_epoch": spa if not use_step_mode else None,
                "train_epochs": n_epochs_cfg if not use_step_mode else None,
                "train_steps": n_step_only if use_step_mode else None,
                "warmup_neg_enabled": bool(neg_on),
                "warmup_neg_ratio": float(neg_ratio),
                "warmup_neg_label_range": [float(neg_lo), float(neg_hi)],
                "warmup_neg_iou_max": float(cfg.mesh_simi_warmup_neg_iou_max),
                "warmup_frame_pts_n": int(n_frame),
                "warmup_rot_enabled": bool(rot_on),
                "warmup_rot_ratio": float(rot_ratio),
                "warmup_rot_gt": float(rot_gt),
                "warmup_rot_mode": str(rot_mode),
                "warmup_export_dir": export_saved or None,
                "warmup_export_n_pairs": int(len(ex_ref)),
                "warmup_export_cap": int(export_max) if export_on else 0,
                "warmup_export_format": "ply",
                "mesh_simi_input_source": (
                    "heightfield_mesh" if use_hf_mesh and ref_th_roi is not None else "pixel_cloud"
                ),
                "warmup_patch_metric": patch_metric if patch_lab else "",
                "warmup_patch_grid": int(pgrid) if patch_lab else 0,
            },
        },
        out_ckpt,
    )
    if neg_fallbacks > 0:
        _log(
            f"[MeshSimi] 负样本采样有 {neg_fallbacks} 次未找到低IoU窗，已回退为正样本（遮挡）",
            cfg,
        )
    _log(
        f"[MeshSimi] 在线热身完成 总样本对≈{total_pair_samples} "
        f"总反传步={global_step if not use_step_mode else total_opt_steps} → {out_ckpt}",
        cfg,
    )
    return os.path.abspath(out_ckpt), export_saved


class MeshSimilarityScorer:
    """加载 MeshSimilarityNet，对两组 uvh 点云（与训练同预处理）输出相似度 ∈(0,1)。"""

    def __init__(self, cfg: RTSConfig, primary_ckpt: str | None = None):
        self.cfg = cfg
        self.enabled = False
        self.model: nn.Module | None = None
        self.device = torch.device(
            cfg.mesh_simi_device
            if torch.cuda.is_available() and str(cfg.mesh_simi_device).startswith("cuda")
            else "cpu"
        )
        if not bool(cfg.mesh_simi_enable):
            return
        pc = str(primary_ckpt).strip() if primary_ckpt else ""
        ckpt = pc if pc and os.path.isfile(pc) else str(cfg.mesh_simi_ckpt).strip()
        if not ckpt or not os.path.isfile(ckpt):
            _log(
                f"[MeshSimi] 未启用：找不到权重 primary={pc or '(无)'} ckpt={cfg.mesh_simi_ckpt}",
                cfg,
            )
            return
        try:
            from mesh_similarity_model import MeshSimilarityNet
        except Exception as e:
            _log(f"[MeshSimi] 导入模型失败: {e}", cfg)
            return
        try:
            self.model = MeshSimilarityNet(
                in_dim=3, embed_dim=256, head_hidden=256
            ).to(self.device)
            try:
                ck = torch.load(ckpt, map_location=self.device, weights_only=False)
            except TypeError:
                ck = torch.load(ckpt, map_location=self.device)
            self.model.load_state_dict(ck["model"], strict=True)
            self.model.eval()
            self.enabled = True
            _log(f"[MeshSimi] 已加载: {ckpt} device={self.device}", cfg)
        except Exception as e:
            _log(f"[MeshSimi] 加载失败: {e}", cfg)
            self.model = None

    @torch.no_grad()
    def _score_world_pair(
        self,
        ref_xyz: np.ndarray,
        cand_xyz: np.ndarray,
        rng: np.random.Generator,
    ) -> float:
        if not self.enabled or self.model is None:
            return 0.0
        try:
            import train_mesh_similarity_chess as tmc
        except Exception:
            return 0.0
        r = tmc.normalize_point_cloud(np.asarray(ref_xyz, dtype=np.float32))
        q = tmc.normalize_point_cloud(np.asarray(cand_xyz, dtype=np.float32))
        if r.shape[0] < 8 or q.shape[0] < 8:
            return 0.0
        n = int(self.cfg.mesh_simi_n_points)
        r = tmc.resample_points(r, n, rng)
        q = tmc.resample_points(q, n, rng)
        t_r = torch.from_numpy(r).float().unsqueeze(0).to(self.device)
        t_q = torch.from_numpy(q).float().unsqueeze(0).to(self.device)
        s = self.model(t_r, t_q).squeeze().detach().float().cpu().item()
        return float(max(0.0, min(1.0, s)))

    @torch.no_grad()
    def score_pair(self, ref_xyz: np.ndarray, cand_xyz: np.ndarray) -> float:
        rng = np.random.default_rng(0)
        return self._score_world_pair(ref_xyz, cand_xyz, rng)

    def _mesh_infer_surface_pool_n(self) -> int:
        m = max(1, int(self.cfg.mesh_simi_mesh_infer_pool_mult))
        return max(int(self.cfg.mesh_simi_n_points) * m, 256)

    @torch.no_grad()
    def score_pair_th_roi(
        self,
        th_ref: np.ndarray,
        bbox_ref: tuple[int, int, int, int],
        th_cand: np.ndarray,
        bbox_cand: tuple[int, int, int, int],
        rng: np.random.Generator,
    ) -> float:
        """与 score_pair 相同预处理，但输入为 th ROI + 全局 bbox，先表面采样再进网。"""
        if not self.enabled or self.model is None:
            return 0.0
        pool = self._mesh_infer_surface_pool_n()
        z = float(self.cfg.mesh_z_scale)
        pa = mesh_surface_points_uvh(th_ref, bbox_ref, z, pool, rng)
        pb = mesh_surface_points_uvh(th_cand, bbox_cand, z, pool, rng)
        return self._score_world_pair(pa, pb, rng)

    @torch.no_grad()
    def score_cand_th_vs_ref_surface(
        self,
        ref_surface_world: np.ndarray,
        th_cand: np.ndarray,
        bbox_cand: tuple[int, int, int, int],
        rng: np.random.Generator,
    ) -> float:
        """候选窗只做一次表面采样；参考侧点云由调用方对 ref 预采样后传入（避免每窗重复建参考网格）。"""
        if not self.enabled or self.model is None:
            return 0.0
        if int(np.asarray(ref_surface_world).shape[0]) < 8:
            return 0.0
        pool = self._mesh_infer_surface_pool_n()
        z = float(self.cfg.mesh_z_scale)
        pb = mesh_surface_points_uvh(th_cand, bbox_cand, z, pool, rng)
        return self._score_world_pair(
            np.asarray(ref_surface_world, dtype=np.float32), pb, rng
        )


# ═══════════════════════════════════════════════════════════
#  工具
# ═══════════════════════════════════════════════════════════

def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def _log(msg: str, cfg: RTSConfig | None = None, level: int = 1):
    v = 1 if cfg is None else cfg.verbose
    if v >= level:
        print(msg, flush=True)


def contour_rgb_from_roi_mask(
    mask_u8: np.ndarray,
    bbox_wh: tuple[int, int],
    semantic_sz: tuple[int, int],
) -> np.ndarray:
    """从 ROI 内 th_68 mask 提取最大轮廓 → 128×128 三通道线图（供 Autoencoder_129）。"""
    mk = np.asarray(mask_u8, dtype=np.uint8)
    cnt = core.extract_largest_contour(mk)
    if cnt is None:
        return np.zeros((semantic_sz[1], semantic_sz[0], 3), dtype=np.uint8)
    bw, bh = int(bbox_wh[0]), int(bbox_wh[1])
    cnt2 = core.resize_contour(cnt, (bw, bh), semantic_sz)
    return core.contour_to_rgb(cnt2, semantic_sz, thickness=2)


def contour_rgb_after_th68_on_roi(
    roi_bgr: np.ndarray,
    cfg: RTSConfig,
    ccfg: core.TrackConfig,
    semantic_sz: tuple[int, int],
) -> np.ndarray:
    """ROI BGR（或 H×W×3）→ th_68 → 最大轮廓 → AE 用 RGB 线图。"""
    bgr = np.asarray(roi_bgr, dtype=np.uint8)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    elif bgr.ndim == 3 and bgr.shape[2] == 1:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    _th, mask, _ = core.compute_th_gray_truncated(bgr, cfg.th_percent, ccfg)
    h, w = int(mask.shape[0]), int(mask.shape[1])
    return contour_rgb_from_roi_mask(mask, (w, h), semantic_sz)


def crop_xywh_from_search_patch(
    patch_hw: np.ndarray,
    search_bbox: tuple[int, int, int, int],
    win_xywh: tuple[int, int, int, int],
) -> np.ndarray:
    """
    patch_hw 与 search_bbox 对齐（高 sh × 宽 sw），按 win_xywh 在整帧中的位置裁剪子块。
    滑动窗均在 search_bbox 内时，子块尺寸与 win 的 w×h 一致。
    """
    sx, sy = int(search_bbox[0]), int(search_bbox[1])
    bx, by, bw, bh = (int(v) for v in win_xywh)
    lx, ly = bx - sx, by - sy
    return np.ascontiguousarray(patch_hw[ly : ly + bh, lx : lx + bw])


def contour_rgb_from_pseudo_gray_mask(
    gray_u8: np.ndarray,
    semantic_sz: tuple[int, int],
) -> np.ndarray:
    """伪 depth / 梯度图 uint8 → 非零二值 mask → 轮廓线图（不再做第二次 th_68）。"""
    g = np.asarray(gray_u8, dtype=np.uint8)
    mk = (g > 0).astype(np.uint8) * 255
    h, w = int(mk.shape[0]), int(mk.shape[1])
    return contour_rgb_from_roi_mask(mk, (w, h), semantic_sz)


def ref_contours_from_th_roi(
    th_roi_float: np.ndarray,
    semantic_sz: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """首帧/参考 ROI 上已得到的 th_68 截断灰度 → 与候选同定义的伪 depth / 伪 normal 轮廓。"""
    du8 = pseudo_depth_u8_from_th(th_roi_float)
    nu8 = pseudo_normal_gray_u8_from_th(th_roi_float)
    return (
        contour_rgb_from_pseudo_gray_mask(du8, semantic_sz),
        contour_rgb_from_pseudo_gray_mask(nu8, semantic_sz),
    )


def pack_pseudo_dn_bgr(
    d_u8: np.ndarray,
    n_u8: np.ndarray,
    out_wh: tuple[int, int],
) -> np.ndarray:
    """伪深度 + 伪梯度 → BGR pack（B=depth, G=R=grad），供 AE pack 回退路径。"""
    ow, oh = int(out_wh[0]), int(out_wh[1])
    bd = cv2.resize(np.asarray(d_u8, dtype=np.uint8), (ow, oh), interpolation=cv2.INTER_AREA)
    bn = cv2.resize(np.asarray(n_u8, dtype=np.uint8), (ow, oh), interpolation=cv2.INTER_AREA)
    return cv2.merge([bd, bn, bn])


def setup_autoencoder_constraint(ccfg: core.TrackConfig, cfg: RTSConfig) -> bool:
    """
    初始化 Autoencoder_129 轮廓模块，并设置 core.SEMANTIC_ENABLED。
    相似度为 calculate_contour_similarity 的归一化值 sim=1/(1+dist)，与 core.semantic_similarity 一致。
    """
    if not cfg.ae_enable:
        core.SEMANTIC_ENABLED = False
        return False
    ok_sem, sem_err = core._init_semantic_module(ccfg)
    model_exists = os.path.isfile(str(ccfg.model_path))
    fn = getattr(core, "calculate_contour_similarity", None)
    fn_ok = callable(fn)
    core.SEMANTIC_ENABLED = bool(ok_sem and model_exists and fn_ok)
    if cfg.verbose >= 1:
        _log(
            f"[AE] SEMANTIC_ENABLED={int(core.SEMANTIC_ENABLED)} "
            f"model_exists={int(model_exists)} path={ccfg.model_path}",
            cfg,
        )
        if sem_err and not ok_sem:
            _log(f"[AE] 模块导入: {str(sem_err)[:240]}", cfg)
    if core.SEMANTIC_ENABLED and cfg.ae_warmup and fn_ok:
        try:
            z = np.zeros(
                (ccfg.semantic_image_size[1], ccfg.semantic_image_size[0], 3),
                dtype=np.uint8,
            )
            fn(
                contour1=z,
                contour2=z,
                model_path=str(ccfg.model_path),
                image_size=tuple(ccfg.semantic_image_size),
                device=core._device_str(ccfg),
            )
        except Exception as e:
            _log(f"[AE] 预热失败(忽略): {e}", cfg)
    return bool(core.SEMANTIC_ENABLED)


# ═══════════════════════════════════════════════════════════
#  伪3D 点云构建
# ═══════════════════════════════════════════════════════════

def build_uvh_points(
    th_gray: np.ndarray,
    bbox: tuple[int, int, int, int],
    z_scale: float,
    max_points: int = 60000,
) -> np.ndarray:
    """
    从 th_68 灰度截断图构建 (N,3) 伪3D点云 [u, v, h]。
    坐标在整帧像素坐标系中。
    """
    bx, by, w, h = (int(v) for v in bbox)
    img = np.asarray(th_gray, dtype=np.float32)
    if img.ndim != 2:
        return np.zeros((0, 3), np.float32)
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)

    ys, xs = np.nonzero(img > 0)
    if xs.size < 10:
        return np.zeros((0, 3), np.float32)

    z = (img[ys, xs] / 255.0) * float(z_scale)
    u = xs.astype(np.float32) + float(bx)
    v = ys.astype(np.float32) + float(by)
    pts = np.stack([u, v, z.astype(np.float32)], axis=1)

    if pts.shape[0] > max_points:
        idx = np.linspace(0, pts.shape[0] - 1, max_points, dtype=np.int64)
        pts = pts[idx]
    return pts.astype(np.float32)


def crop_uvh_points_in_bbox(
    pts: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    """
    从整帧坐标系下的伪 3D 点云中，按轴对齐矩形框裁剪子集（半开区间 [bx,bx+bw)×[by,by+bh)）。
    用于：先在 search 区或整帧上统一 th_68 建点云，再按候选滑动窗取局部 mesh，避免逐窗 th_68 边界不一致。
    """
    bx, by, bw, bh = (int(v) for v in bbox)
    p = np.asarray(pts, dtype=np.float32)
    if p.shape[0] == 0:
        return p
    u, v = p[:, 0], p[:, 1]
    m = (
        (u >= float(bx))
        & (u < float(bx + bw))
        & (v >= float(by))
        & (v < float(by + bh))
    )
    return p[m].copy()


def build_unified_frame_uvh_points(
    frame_bgr: np.ndarray,
    *,
    scope: str,
    search_bbox: tuple[int, int, int, int] | None,
    th_percent: float,
    ccfg: core.TrackConfig,
    mesh_z_scale: float,
    max_points: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """
    在当前帧上统一做 th_68 后构建伪 3D 点云（整帧坐标）。

    Returns
    -------
    pts : (N,3)
    bounds : (u_min,v_min,u_max,v_max) 供 optimize_rts 栅格化边界
    """
    scope = str(scope or "search").strip().lower()
    if scope not in ("frame", "search"):
        warnings.warn(
            f"MIS_LIB_MESH_SCOPE 无效值 {scope!r}，已回退为 'search'（有效: frame, search）。",
            UserWarning,
            stacklevel=2,
        )
        scope = "search"
    H, W = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    if scope == "frame":
        th, _, _ = core.compute_th_gray_truncated(frame_bgr, th_percent, ccfg)
        bb = (0, 0, W, H)
        pts = build_uvh_points(th, bb, mesh_z_scale, max_points)
        bnds = (0.0, 0.0, float(W), float(H))
        return pts, bnds
    if search_bbox is None:
        return np.zeros((0, 3), np.float32), (0.0, 0.0, 1.0, 1.0)
    sx, sy, sw, sh = (int(v) for v in search_bbox)
    roi = frame_bgr[sy : sy + sh, sx : sx + sw]
    th, _, _ = core.compute_th_gray_truncated(roi, th_percent, ccfg)
    pts = build_uvh_points(th, search_bbox, mesh_z_scale, max_points)
    bnds = compute_search_bounds(search_bbox)
    return pts, bnds


# ═══════════════════════════════════════════════════════════
#  栅格化 & 深度图比较
# ═══════════════════════════════════════════════════════════

def rasterize_to_depth_grid(
    pts: np.ndarray,
    bounds: tuple[float, float, float, float],
    grid_size: int = 96,
) -> tuple[np.ndarray, np.ndarray]:
    """
    将 (N,3) uvh 点云栅格化到 (grid_size, grid_size) 深度图。

    Parameters
    ----------
    pts : (N,3) float32 — [u, v, h]
    bounds : (u_min, v_min, u_max, v_max)
    grid_size : 网格分辨率

    Returns
    -------
    depth : (G, G) float32 — 深度值（h），无效区域为 0
    mask  : (G, G) bool    — 有效像素
    """
    gs = int(grid_size)
    u_min, v_min, u_max, v_max = (float(x) for x in bounds)
    u_rng = max(u_max - u_min, 1e-6)
    v_rng = max(v_max - v_min, 1e-6)

    p = np.asarray(pts, dtype=np.float32)
    if p.shape[0] < 3:
        return np.zeros((gs, gs), np.float32), np.zeros((gs, gs), bool)

    gi = np.clip(((p[:, 0] - u_min) / u_rng * (gs - 1)), 0, gs - 1).astype(np.intp)
    gj = np.clip(((p[:, 1] - v_min) / v_rng * (gs - 1)), 0, gs - 1).astype(np.intp)

    h_sum = np.zeros((gs, gs), np.float64)
    h_cnt = np.zeros((gs, gs), np.float64)
    np.add.at(h_sum, (gj, gi), p[:, 2].astype(np.float64))
    np.add.at(h_cnt, (gj, gi), 1.0)

    mask = h_cnt > 0
    depth = np.zeros((gs, gs), np.float32)
    depth[mask] = (h_sum[mask] / h_cnt[mask]).astype(np.float32)
    return depth, mask


def depth_grid_similarity(
    depth_a: np.ndarray,
    mask_a: np.ndarray,
    depth_b: np.ndarray,
    mask_b: np.ndarray,
) -> dict[str, float]:
    """
    比较两张深度栅格图的相似度。

    Returns
    -------
    dict: iou, depth_corr (0~1), depth_l1_score (0~1)
    """
    inter = mask_a & mask_b
    union = mask_a | mask_b
    n_inter = int(np.count_nonzero(inter))
    n_union = int(np.count_nonzero(union))

    if n_inter < 5 or n_union < 5:
        return {"iou": 0.0, "depth_corr": 0.0, "depth_l1_score": 0.0}

    iou = float(n_inter) / float(n_union)

    da = depth_a[inter].astype(np.float64)
    db = depth_b[inter].astype(np.float64)

    # 深度相关性
    da_m = np.mean(da)
    db_m = np.mean(db)
    da0 = da - da_m
    db0 = db - db_m
    va = float(np.mean(da0 * da0))
    vb = float(np.mean(db0 * db0))
    if va < 1e-10 or vb < 1e-10:
        corr = 0.0
    else:
        corr = float(np.clip(np.mean(da0 * db0) / math.sqrt(va * vb + 1e-12), -1.0, 1.0))
    corr01 = 0.5 * (corr + 1.0)

    # 深度 L1
    l1 = float(np.mean(np.abs(da - db)))
    h_range = float(max(1e-6, np.max(da) - np.min(da), np.max(db) - np.min(db)))
    l1_norm = l1 / h_range
    l1_score = float(math.exp(-l1_norm * 3.0))

    return {"iou": iou, "depth_corr": corr01, "depth_l1_score": l1_score}


# ═══════════════════════════════════════════════════════════
#  RTS 变换
# ═══════════════════════════════════════════════════════════

def apply_rts(
    pts: np.ndarray,
    center: np.ndarray,
    rot_deg: float,
    scale: float,
    tx: float,
    ty: float,
) -> np.ndarray:
    """
    以 center 为旋转/缩放中心，对点云施加 RTS 变换。
      R : uv 平面内绕 z 轴旋转（高度 h 跟随缩放但不旋转 z 分量）
      T : (tx, ty) 在 uv 平面平移
      S : 统一缩放
    """
    p = np.asarray(pts, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)

    pc = p - c[None, :]
    pc = pc * float(scale)

    theta = float(rot_deg) * math.pi / 180.0
    ct, st = math.cos(theta), math.sin(theta)
    R = np.array([[ct, -st, 0.0],
                   [st,  ct, 0.0],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    pc = (R @ pc.T).T

    pc += c[None, :]
    pc[:, 0] += float(tx)
    pc[:, 1] += float(ty)
    return pc.astype(np.float32)


# ═══════════════════════════════════════════════════════════
#  RTS 块坐标下降（深度栅格 + 可选恒等先验）
# ═══════════════════════════════════════════════════════════

def optimize_rts(
    ref_pts: np.ndarray,
    search_pts: np.ndarray,
    search_bounds: tuple[float, float, float, float],
    cfg: RTSConfig,
) -> dict[str, Any]:
    """
    以参考 mesh 重心为旋转/缩放中心，在 (R, S, tx, ty) 上做块坐标下降：
    每轮依次在四个参数轴上各做一维网格搜索，再按 MIS_RTS_SHRINK 收缩搜索窗。

    目标：变换后参考 mesh 与搜索 mesh 在同一 search_bounds 下的深度栅格
    相似度最大；相似度为 IoU、深度相关、深度 L1 的加权和，可选 MIS_W_PRIOR
    惩罚偏离恒等变换。
    """
    ref = np.asarray(ref_pts, dtype=np.float32)
    search = np.asarray(search_pts, dtype=np.float32)

    fail = {"ok": False, "score": 0.0, "rot_deg": 0.0, "scale": 1.0,
            "tx": 0.0, "ty": 0.0, "iou": 0.0, "depth_corr": 0.0,
            "depth_l1_score": 0.0, "inlier_count": 0, "tmpl_confidence": 0.0,
            "warp_confidence": 0.0, "rmse": 999.0, "method": "none"}

    if ref.shape[0] < 50 or search.shape[0] < 50:
        return fail

    gs = int(cfg.grid_size)
    c_ref = np.mean(ref, axis=0).astype(np.float64)
    r_max = float(cfg.rts_rot_range_deg)
    smin = float(cfg.rts_scale_min)
    smax = float(cfg.rts_scale_max)
    t_max = float(cfg.rts_trans_range_px)
    span_s = max(smax - smin, 1e-6)
    span_1 = max(max(smax - 1.0, 1.0 - smin), 1e-6)

    w_io = float(cfg.w_iou)
    w_dc = float(cfg.w_depth_corr)
    w_dl = float(cfg.w_depth_l1)
    w_pr = float(cfg.w_prior)

    def eval_score(
        rot_deg: float, scale: float, tx: float, ty: float,
    ) -> tuple[float, dict[str, float]]:
        pts_t = apply_rts(ref, c_ref, rot_deg, scale, tx, ty)
        d_ref_t, m_r = rasterize_to_depth_grid(pts_t, search_bounds, gs)
        d_s, m_s = rasterize_to_depth_grid(search, search_bounds, gs)
        sim = depth_grid_similarity(d_ref_t, m_r, d_s, m_s)
        geo = w_io * sim["iou"] + w_dc * sim["depth_corr"] + w_dl * sim["depth_l1_score"]
        if w_pr > 1e-12:
            rr = abs(rot_deg) / max(r_max, 1e-6)
            ss = abs(scale - 1.0) / span_1
            tt = math.hypot(tx, ty) / max(t_max, 1e-6)
            prior = math.exp(-(rr * rr + ss * ss + tt * tt))
            wsum = w_io + w_dc + w_dl + w_pr
            score = (geo + w_pr * prior) / wsum
        else:
            wsum = w_io + w_dc + w_dl
            score = geo / wsum if wsum > 1e-12 else 0.0
        return float(score), sim

    br, bs, btx, bty = 0.0, 1.0, 0.0, 0.0
    best_score, _ = eval_score(br, bs, btx, bty)
    best_score = float(best_score)

    n_rot = max(2, int(cfg.rts_rot_steps))
    n_sc = max(2, int(cfg.rts_scale_steps))
    n_tr = max(2, int(cfg.rts_trans_steps))
    shrink = max(float(cfg.rts_shrink), 0.05)

    rot_w = 2.0 * r_max
    scale_w = span_s
    trans_w = 2.0 * t_max

    for _ in range(int(cfg.rts_iters)):
        lo = max(-r_max, br - 0.5 * rot_w)
        hi = min(r_max, br + 0.5 * rot_w)
        if hi > lo + 1e-9:
            for r in np.linspace(lo, hi, n_rot):
                sc, _ = eval_score(float(r), bs, btx, bty)
                if sc > best_score:
                    best_score, br = float(sc), float(r)

        lo_s = max(smin, bs - 0.5 * scale_w)
        hi_s = min(smax, bs + 0.5 * scale_w)
        if hi_s > lo_s + 1e-9:
            for s in np.linspace(lo_s, hi_s, n_sc):
                sc, _ = eval_score(br, float(s), btx, bty)
                if sc > best_score:
                    best_score, bs = float(sc), float(s)

        lo_tx = max(-t_max, btx - 0.5 * trans_w)
        hi_tx = min(t_max, btx + 0.5 * trans_w)
        if hi_tx > lo_tx + 1e-9:
            for txv in np.linspace(lo_tx, hi_tx, n_tr):
                sc, _ = eval_score(br, bs, float(txv), bty)
                if sc > best_score:
                    best_score, btx = float(sc), float(txv)

        lo_ty = max(-t_max, bty - 0.5 * trans_w)
        hi_ty = min(t_max, bty + 0.5 * trans_w)
        if hi_ty > lo_ty + 1e-9:
            for tyv in np.linspace(lo_ty, hi_ty, n_tr):
                sc, _ = eval_score(br, bs, btx, float(tyv))
                if sc > best_score:
                    best_score, bty = float(sc), float(tyv)

        rot_w = max(rot_w * shrink, r_max * 0.02)
        scale_w = max(scale_w * shrink, span_s * 0.02)
        trans_w = max(trans_w * shrink, t_max * 0.02)

    best_score, best_sim = eval_score(br, bs, btx, bty)
    pts_t = apply_rts(ref, c_ref, br, bs, btx, bty)
    d_ref_t, m_r = rasterize_to_depth_grid(pts_t, search_bounds, gs)
    d_s, m_s = rasterize_to_depth_grid(search, search_bounds, gs)
    inter = m_r & m_s
    if np.any(inter):
        diff = d_ref_t[inter].astype(np.float64) - d_s[inter].astype(np.float64)
        rmse = float(np.sqrt(np.mean(diff * diff)))
        inlier_count = int(np.count_nonzero(inter))
    else:
        rmse = 999.0
        inlier_count = 0

    return {
        "ok": True,
        "score": float(best_score),
        "rot_deg": float(br),
        "scale": float(bs),
        "tx": float(btx),
        "ty": float(bty),
        "iou": float(best_sim["iou"]),
        "depth_corr": float(best_sim["depth_corr"]),
        "depth_l1_score": float(best_sim["depth_l1_score"]),
        "ref_centroid": c_ref.tolist(),
        "tmpl_confidence": 0.0,
        "warp_confidence": 0.0,
        "tmpl_scale": 1.0,
        "rmse": rmse,
        "inlier_count": inlier_count,
        "method": "rts_coord_desc",
    }


# ═══════════════════════════════════════════════════════════
#  bbox 反投影 / 搜索区域
# ═══════════════════════════════════════════════════════════

def rts_to_bbox(
    ref_pts: np.ndarray,
    center: np.ndarray,
    rot_deg: float,
    scale: float,
    tx: float,
    ty: float,
    frame_wh: tuple[int, int],
    pad: int = 2,
    pctl_lo: float = 1.0,
    pctl_hi: float = 99.0,
) -> tuple[int, int, int, int] | None:
    """
    将 RTS 变换后的参考点投影回 2D，得到紧缩 bbox。
    使用百分位裁剪而非全局 min/max，避免离群点撑大边框。
    """
    pts = apply_rts(ref_pts, center, rot_deg, scale, tx, ty)
    if pts.shape[0] < 10:
        return None

    xs = pts[:, 0].astype(np.float64)
    ys = pts[:, 1].astype(np.float64)

    x0 = float(np.percentile(xs, pctl_lo)) - pad
    y0 = float(np.percentile(ys, pctl_lo)) - pad
    x1 = float(np.percentile(xs, pctl_hi)) + pad
    y1 = float(np.percentile(ys, pctl_hi)) + pad

    W, H = frame_wh
    return core.clamp_bbox(
        int(round(x0)), int(round(y0)),
        max(4, int(round(x1 - x0))), max(4, int(round(y1 - y0))),
        W, H,
    )


def compute_search_bbox(
    prev_bbox: tuple[int, int, int, int],
    expand: int,
    frame_wh: tuple[int, int],
) -> tuple[int, int, int, int]:
    """以上一帧 bbox 为中心，四周各扩展 expand 像素"""
    x, y, w, h = (int(v) for v in prev_bbox)
    W, H = frame_wh
    sx = max(0, x - expand)
    sy = max(0, y - expand)
    sx2 = min(W, x + w + expand)
    sy2 = min(H, y + h + expand)
    return core.clamp_bbox(sx, sy, sx2 - sx, sy2 - sy, W, H)


def validate_bbox_shift(
    new_bbox: tuple[int, int, int, int],
    prev_bbox: tuple[int, int, int, int],
    cfg: RTSConfig,
    max_shift_scale: float = 1.0,
) -> tuple[bool, str]:
    """
    检查新 bbox 是否相对于上一帧产生了不合理的漂移。

    max_shift_scale: 乘在 max_center_shift_px 上（渲染置信高时可 >1 放宽）。

    Returns: (accepted, reason)
    """
    nx, ny, nw, nh = (float(v) for v in new_bbox)
    px, py, pw, ph = (float(v) for v in prev_bbox)

    nc = np.array([nx + nw / 2, ny + nh / 2])
    pc = np.array([px + pw / 2, py + ph / 2])
    shift = float(np.linalg.norm(nc - pc))
    lim = float(cfg.max_center_shift_px) * max(0.5, float(max_shift_scale))
    if shift > lim:
        return False, f"center_shift={shift:.1f}>{lim:.0f}"

    new_area = nw * nh
    prev_area = pw * ph
    if prev_area < 1:
        return True, "ok"
    ratio = new_area / prev_area
    if ratio < cfg.area_ratio_lo or ratio > cfg.area_ratio_hi:
        return False, f"area_ratio={ratio:.2f} out [{cfg.area_ratio_lo},{cfg.area_ratio_hi}]"

    return True, "ok"


def bbox_iou_xywh(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """xywh 格式 bbox 的 IoU，无交集时为 0。"""
    ax, ay, aw, ah = (int(v) for v in a)
    bx, by, bw, bh = (int(v) for v in b)
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = float(iw * ih)
    ua = float(max(0, aw) * max(0, ah) + max(0, bw) * max(0, bh)) - inter
    return inter / ua if ua > 1e-6 else 0.0


def _mesh_warmup_sample_negative_uvh(
    frame_pts: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    frame_wh: tuple[int, int],
    *,
    iou_max: float,
    min_pts: int,
    rng: np.random.Generator,
    max_tries: int,
) -> np.ndarray | None:
    """
    从整帧伪 3D 点云中随机取与 target_bbox 空间重叠很小（IoU≤iou_max）的等尺寸窗，
    裁剪点云作为与 target 不相关的 query（用于热身负样本）。
    """
    W, H = int(frame_wh[0]), int(frame_wh[1])
    _, _, tw, th = (int(v) for v in target_bbox)
    tpl_w = max(4, min(tw, W))
    tpl_h = max(4, min(th, H))
    if tpl_w > W or tpl_h > H:
        return None
    fp = np.asarray(frame_pts, dtype=np.float32)
    if fp.shape[0] < int(min_pts):
        return None
    iou_m = float(iou_max)
    for _ in range(max(1, int(max_tries))):
        xmax = max(0, W - tpl_w)
        ymax = max(0, H - tpl_h)
        x = int(rng.integers(0, xmax + 1)) if xmax > 0 else 0
        y = int(rng.integers(0, ymax + 1)) if ymax > 0 else 0
        cand = (x, y, tpl_w, tpl_h)
        if bbox_iou_xywh(cand, target_bbox) > iou_m:
            continue
        sub = crop_uvh_points_in_bbox(fp, cand)
        if int(sub.shape[0]) < int(min_pts):
            continue
        return sub
    return None


def _mesh_warmup_sample_negative_mesh(
    th_full: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    frame_wh: tuple[int, int],
    *,
    z_scale: float,
    n_surface_samples: int,
    iou_max: float,
    min_pts: int,
    rng: np.random.Generator,
    max_tries: int,
) -> np.ndarray | None:
    """
    在全帧 th 图上随机取与 target 等尺寸、IoU 低的窗，对该窗建高度场网格并表面采样作负样本 query。
    """
    W, H = int(frame_wh[0]), int(frame_wh[1])
    th2 = np.asarray(th_full, dtype=np.float32)
    if th2.ndim != 2 or th2.shape[0] != H or th2.shape[1] != W:
        return None
    _, _, tw, th = (int(v) for v in target_bbox)
    tpl_w = max(4, min(tw, W))
    tpl_h = max(4, min(th, H))
    if tpl_w > W or tpl_h > H:
        return None
    iou_m = float(iou_max)
    for _ in range(max(1, int(max_tries))):
        xmax = max(0, W - tpl_w)
        ymax = max(0, H - tpl_h)
        x = int(rng.integers(0, xmax + 1)) if xmax > 0 else 0
        y = int(rng.integers(0, ymax + 1)) if ymax > 0 else 0
        cand = (x, y, tpl_w, tpl_h)
        if bbox_iou_xywh(cand, target_bbox) > iou_m:
            continue
        th_crop = np.ascontiguousarray(th2[y : y + tpl_h, x : x + tpl_w])
        pts = mesh_surface_points_uvh(
            th_crop,
            cand,
            float(z_scale),
            int(n_surface_samples),
            rng,
        )
        if int(pts.shape[0]) < int(min_pts):
            continue
        return pts
    return None


def _warmup_random_rotation_matrix(
    rng: np.random.Generator,
    mode: str,
) -> np.ndarray:
    """
    热身用旋转：默认仅改变角度、无平移（与 normalize 后零质心配合）。
    mode: z|h|uv|yaw → 绕第 3 轴（u-v 平面内转 θ）；y|pitch、x|roll；full|so3|3 → Haar SO(3)。
    """
    m = (mode or "z").strip().lower()
    if m in ("full", "so3", "3", "xyz"):
        return _random_rotation_matrix_so3(rng)
    theta = float(rng.uniform(0.0, 2.0 * math.pi))
    c, s = float(np.cos(theta)), float(np.sin(theta))
    if m in ("z", "h", "uv", "yaw", ""):
        return np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    if m in ("y", "pitch"):
        return np.array(
            [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)
    if m in ("x", "roll"):
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float32)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _random_rotation_matrix_so3(rng: np.random.Generator) -> np.ndarray:
    """Haar 均匀随机旋转矩阵 (3,3) float32（full 模式）。"""
    u1, u2, u3 = rng.random(3).astype(np.float64)
    q1 = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    q2 = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    q3 = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    q0 = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    qq0, qq1, qq2, qq3 = q0 * q0, q1 * q1, q2 * q2, q3 * q3
    R = np.array(
        [
            [
                1.0 - 2.0 * (qq2 + qq3),
                2.0 * (q1 * q2 - q0 * q3),
                2.0 * (q1 * q3 + q0 * q2),
            ],
            [
                2.0 * (q1 * q2 + q0 * q3),
                1.0 - 2.0 * (qq1 + qq3),
                2.0 * (q2 * q3 - q0 * q1),
            ],
            [
                2.0 * (q1 * q3 - q0 * q2),
                2.0 * (q2 * q3 + q0 * q1),
                1.0 - 2.0 * (qq1 + qq2),
            ],
        ],
        dtype=np.float32,
    )
    return R


def ema_smooth_bbox(
    new_bbox: tuple[int, int, int, int],
    prev_bbox: tuple[int, int, int, int],
    alpha: float,
) -> tuple[int, int, int, int]:
    """
    对 bbox 做指数移动平均平滑，减少帧间抖动。
    alpha=0 → 完全用新值, alpha=1 → 完全用旧值。
    """
    if alpha <= 0.0:
        return new_bbox
    nx, ny, nw, nh = (float(v) for v in new_bbox)
    px, py, pw, ph = (float(v) for v in prev_bbox)
    a = float(np.clip(alpha, 0, 0.9))
    sx = a * px + (1 - a) * nx
    sy = a * py + (1 - a) * ny
    sw = a * pw + (1 - a) * nw
    sh = a * ph + (1 - a) * nh
    return (int(round(sx)), int(round(sy)), max(4, int(round(sw))), max(4, int(round(sh))))


def compute_search_bounds(search_bbox: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    """搜索区域在全局坐标系中的 (u_min, v_min, u_max, v_max)"""
    sx, sy, sw, sh = (int(v) for v in search_bbox)
    return (float(sx), float(sy), float(sx + sw), float(sy + sh))


# ═══════════════════════════════════════════════════════════
#  可视化
# ═══════════════════════════════════════════════════════════

def save_rts_debug_image(
    path: str,
    frame_bgr: np.ndarray,
    ref_pts: np.ndarray,
    center: np.ndarray,
    rts: dict[str, Any],
    search_bbox: tuple[int, int, int, int],
    result_bbox: tuple[int, int, int, int],
):
    """保存带 RTS 匹配结果的可视化图"""
    vis = frame_bgr.copy()
    sx, sy, sw, sh = search_bbox

    cv2.rectangle(vis, (sx, sy), (sx + sw - 1, sy + sh - 1), (255, 180, 0), 1)

    bx, by, bw, bh = result_bbox
    cv2.rectangle(vis, (bx, by), (bx + bw - 1, by + bh - 1), (0, 255, 0), 2)

    pts_t = apply_rts(
        ref_pts, center,
        rts.get("rot_deg", 0), rts.get("scale", 1),
        rts.get("tx", 0), rts.get("ty", 0),
    )
    H, W = vis.shape[:2]
    step = max(1, pts_t.shape[0] // 1500)
    for p in pts_t[::step]:
        u, v = int(round(float(p[0]))), int(round(float(p[1])))
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(vis, (u, v), 1, (0, 64, 255), -1, cv2.LINE_AA)

    txt = (f"R={rts.get('rot_deg', 0):.1f} S={rts.get('scale', 1):.3f} "
           f"T=({rts.get('tx', 0):.1f},{rts.get('ty', 0):.1f}) "
           f"sc={rts.get('score', 0):.3f} IoU={rts.get('iou', 0):.3f}")
    cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (10, 10, 10), 3, cv2.LINE_AA)
    cv2.putText(vis, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (220, 255, 220), 1, cv2.LINE_AA)

    cv2.imwrite(path, vis)


def _depth_to_u8(d: np.ndarray, m: np.ndarray) -> np.ndarray:
    if not np.any(m):
        return np.zeros_like(d, dtype=np.uint8)
    vals = d[m]
    lo, hi = float(np.min(vals)), float(np.max(vals))
    rng = max(hi - lo, 1e-6)
    n = np.clip((d - lo) / rng, 0, 1)
    u8 = (n * 255).astype(np.uint8)
    u8[~m] = 0
    return u8


def save_depth_grid_compare(
    path: str,
    depth_ref: np.ndarray,
    mask_ref: np.ndarray,
    depth_search: np.ndarray,
    mask_search: np.ndarray,
    score: float,
):
    """保存参考/搜索深度栅格的对比图"""
    r8 = _depth_to_u8(depth_ref, mask_ref)
    s8 = _depth_to_u8(depth_search, mask_search)
    cr = cv2.applyColorMap(r8, cv2.COLORMAP_TURBO)
    cs = cv2.applyColorMap(s8, cv2.COLORMAP_TURBO)
    cr[~mask_ref] = 0
    cs[~mask_search] = 0

    h, w = cr.shape[:2]
    gap = np.full((h, 4, 3), 200, dtype=np.uint8)
    row = np.concatenate([cr, gap, cs], axis=1)

    cv2.putText(row, f"Ref (transformed)", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(row, f"Search | score={score:.3f}", (w + 8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(path, row)


def _pts_to_scatter_img(
    pts: np.ndarray,
    bounds: tuple[float, float, float, float],
    img_size: int = 256,
    color: tuple[int, int, int] = (0, 200, 255),
    bg: int = 30,
) -> np.ndarray:
    """将 (N,3) uvh 点云绘制到 img_size×img_size 的散点图（BGR），颜色按高度渐变"""
    sz = int(img_size)
    canvas = np.full((sz, sz, 3), bg, dtype=np.uint8)
    if pts.shape[0] < 3:
        return canvas
    u_min, v_min, u_max, v_max = (float(x) for x in bounds)
    u_rng = max(u_max - u_min, 1e-6)
    v_rng = max(v_max - v_min, 1e-6)

    px = ((pts[:, 0] - u_min) / u_rng * (sz - 1)).astype(np.int32)
    py = ((pts[:, 1] - v_min) / v_rng * (sz - 1)).astype(np.int32)

    h_vals = pts[:, 2].astype(np.float32)
    h_lo, h_hi = float(np.min(h_vals)), float(np.max(h_vals))
    h_rng = max(h_hi - h_lo, 1e-6)
    h_norm = np.clip((h_vals - h_lo) / h_rng, 0, 1)

    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for k in range(256):
        lut[k, 0] = k
    lut_color = cv2.applyColorMap(lut, cv2.COLORMAP_TURBO).reshape(256, 3)

    step = max(1, pts.shape[0] // 8000)
    for j in range(0, pts.shape[0], step):
        x, y = int(np.clip(px[j], 0, sz - 1)), int(np.clip(py[j], 0, sz - 1))
        ci = int(np.clip(h_norm[j] * 255, 0, 255))
        c = tuple(int(v) for v in lut_color[ci])
        cv2.circle(canvas, (x, y), 1, c, -1, cv2.LINE_AA)
    return canvas


def save_mesh_match_process(
    path: str,
    frame_bgr: np.ndarray,
    ref_pts: np.ndarray,
    search_pts: np.ndarray,
    ref_center: np.ndarray,
    rts: dict[str, Any],
    search_bbox: tuple[int, int, int, int],
    result_bbox: tuple[int, int, int, int],
    grid_size: int = 96,
    panel_size: int = 256,
):
    """
    保存完整的匹配中间过程可视化（4×2 面板）：
      上排: [搜索区域原图] [target mesh] [search mesh] [叠加对比]
      下排: [target 深度栅格] [search 深度栅格] [RTS变换后 target 深度] [差异热力图]
    """
    ps = int(panel_size)
    sx, sy, sw, sh = (int(v) for v in search_bbox)
    search_bounds = (float(sx), float(sy), float(sx + sw), float(sy + sh))

    # --- 上排 Panel 1: 搜索区域原图 + bbox ---
    H, W = frame_bgr.shape[:2]
    crop_y0 = max(0, sy)
    crop_y1 = min(H, sy + sh)
    crop_x0 = max(0, sx)
    crop_x1 = min(W, sx + sw)
    search_crop = frame_bgr[crop_y0:crop_y1, crop_x0:crop_x1].copy()
    p1 = cv2.resize(search_crop, (ps, ps), interpolation=cv2.INTER_AREA)
    # 在搜索区域图上标出结果 bbox
    bx, by, bw, bh = (int(v) for v in result_bbox)
    rx0 = int((bx - sx) / max(sw, 1) * ps)
    ry0 = int((by - sy) / max(sh, 1) * ps)
    rx1 = int((bx + bw - sx) / max(sw, 1) * ps)
    ry1 = int((by + bh - sy) / max(sh, 1) * ps)
    cv2.rectangle(p1, (rx0, ry0), (rx1, ry1), (0, 255, 0), 2)
    cv2.putText(p1, "Search ROI + bbox", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    # --- 上排 Panel 2: Target (ref) mesh 散点图 ---
    p2 = _pts_to_scatter_img(ref_pts, search_bounds, ps)
    cv2.putText(p2, f"Target mesh ({ref_pts.shape[0]}pts)", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 255, 200), 1)

    # --- 上排 Panel 3: Search mesh 散点图 ---
    p3 = _pts_to_scatter_img(search_pts, search_bounds, ps)
    cv2.putText(p3, f"Search mesh ({search_pts.shape[0]}pts)", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 255), 1)

    # --- 上排 Panel 4: 叠加对比 (RTS 变换后 target → 绿色, search → 红色) ---
    pts_t = apply_rts(
        ref_pts, ref_center,
        rts.get("rot_deg", 0), rts.get("scale", 1),
        rts.get("tx", 0), rts.get("ty", 0),
    )
    p4 = np.full((ps, ps, 3), 30, dtype=np.uint8)
    u_min, v_min, u_max, v_max = search_bounds
    u_rng = max(u_max - u_min, 1e-6)
    v_rng = max(v_max - v_min, 1e-6)
    step_s = max(1, search_pts.shape[0] // 6000)
    for j in range(0, search_pts.shape[0], step_s):
        px = int(np.clip((search_pts[j, 0] - u_min) / u_rng * (ps - 1), 0, ps - 1))
        py = int(np.clip((search_pts[j, 1] - v_min) / v_rng * (ps - 1), 0, ps - 1))
        cv2.circle(p4, (px, py), 1, (80, 80, 220), -1, cv2.LINE_AA)
    step_t = max(1, pts_t.shape[0] // 6000)
    for j in range(0, pts_t.shape[0], step_t):
        px = int(np.clip((pts_t[j, 0] - u_min) / u_rng * (ps - 1), 0, ps - 1))
        py = int(np.clip((pts_t[j, 1] - v_min) / v_rng * (ps - 1), 0, ps - 1))
        cv2.circle(p4, (px, py), 1, (80, 255, 80), -1, cv2.LINE_AA)
    cv2.putText(p4, "Overlay: green=ref red=search", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # --- 下排 Panel 5: Target 深度栅格（原始，未变换） ---
    ref_bounds_u = (float(np.min(ref_pts[:, 0])), float(np.min(ref_pts[:, 1])),
                    float(np.max(ref_pts[:, 0])), float(np.max(ref_pts[:, 1])))
    d_ref_orig, m_ref_orig = rasterize_to_depth_grid(ref_pts, ref_bounds_u, grid_size)
    r8 = _depth_to_u8(d_ref_orig, m_ref_orig)
    p5 = cv2.applyColorMap(r8, cv2.COLORMAP_TURBO)
    p5[~m_ref_orig] = 30
    p5 = cv2.resize(p5, (ps, ps), interpolation=cv2.INTER_NEAREST)
    cv2.putText(p5, "Target depth (orig)", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

    # --- 下排 Panel 6: Search 深度栅格 ---
    d_search, m_search = rasterize_to_depth_grid(search_pts, search_bounds, grid_size)
    s8 = _depth_to_u8(d_search, m_search)
    p6 = cv2.applyColorMap(s8, cv2.COLORMAP_TURBO)
    p6[~m_search] = 30
    p6 = cv2.resize(p6, (ps, ps), interpolation=cv2.INTER_NEAREST)
    cv2.putText(p6, "Search depth", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

    # --- 下排 Panel 7: RTS 变换后 target 深度栅格 ---
    d_ref_t, m_ref_t = rasterize_to_depth_grid(pts_t, search_bounds, grid_size)
    t8 = _depth_to_u8(d_ref_t, m_ref_t)
    p7 = cv2.applyColorMap(t8, cv2.COLORMAP_TURBO)
    p7[~m_ref_t] = 30
    p7 = cv2.resize(p7, (ps, ps), interpolation=cv2.INTER_NEAREST)
    cv2.putText(p7, "Target depth (RTS)", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)

    # --- 下排 Panel 8: 差异热力图 ---
    inter = m_ref_t & m_search
    diff = np.zeros_like(d_search, dtype=np.float32)
    if np.any(inter):
        diff[inter] = np.abs(d_ref_t[inter] - d_search[inter])
        d_max = float(np.max(diff[inter]))
        if d_max > 1e-6:
            diff = diff / d_max
    diff_u8 = (np.clip(diff, 0, 1) * 255).astype(np.uint8)
    p8 = cv2.applyColorMap(diff_u8, cv2.COLORMAP_HOT)
    p8[~inter] = 30
    p8 = cv2.resize(p8, (ps, ps), interpolation=cv2.INTER_NEAREST)
    score = rts.get("score", 0)
    iou = rts.get("iou", 0)
    corr = rts.get("depth_corr", 0)
    cv2.putText(p8, f"Diff sc={score:.3f} IoU={iou:.2f}", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # --- 拼接 4×2 ---
    gap = 3
    g_col = np.full((ps, gap, 3), 180, dtype=np.uint8)
    g_row = np.full((gap, ps * 4 + gap * 3, 3), 180, dtype=np.uint8)
    top = np.concatenate([p1, g_col, p2, g_col, p3, g_col, p4], axis=1)
    bot = np.concatenate([p5, g_col, p6, g_col, p7, g_col, p8], axis=1)
    canvas = np.concatenate([top, g_row, bot], axis=0)

    # 底部状态栏
    bar_h = 28
    bar = np.full((bar_h, canvas.shape[1], 3), 40, dtype=np.uint8)
    info = (f"R={rts.get('rot_deg', 0):.2f}deg  S={rts.get('scale', 1):.3f}  "
            f"T=({rts.get('tx', 0):.1f},{rts.get('ty', 0):.1f})  "
            f"score={score:.4f}  IoU={iou:.3f}  corr={corr:.3f}  "
            f"ref={ref_pts.shape[0]}pts  search={search_pts.shape[0]}pts")
    cv2.putText(bar, info, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (200, 230, 200), 1, cv2.LINE_AA)
    canvas = np.concatenate([canvas, bar], axis=0)

    cv2.imwrite(path, canvas)


# ═══════════════════════════════════════════════════════════
#  神经渲染场验证器（Render Field Validator）
# ═══════════════════════════════════════════════════════════

class RenderFieldValidator:
    """
    利用训练好的 PointRT Render Field 模型验证跟踪匹配质量。

    思路：
      1. 首帧：将参考点云归一化到规范空间，以模型重心为中心、
         固定距离的正前方位姿渲染 depth / normal / silhouette
      2. 每帧：将候选区域点云同样归一化并渲染
      3. 比较两者相似度 → 相似度低说明形状不匹配 → 大概率漂移
    """

    def __init__(self, cfg: RTSConfig):
        self.enabled = False
        self.n_points = int(cfg.render_n_points)
        self.view_dist = float(cfg.render_view_dist)
        self.point_in_dim = 3
        self.model = None
        self.device_t = None
        self.ref_renders: list = []
        self.canonical_poses: list = []
        self._torch = None

        self.lib_front_dist = float(cfg.render_view_dist)
        if not cfg.render_enabled:
            _log("[RenderField] 已禁用 (render_enabled=False)", cfg)
            return
        ckpt_path = cfg.render_model_path
        if not ckpt_path or not os.path.isfile(ckpt_path):
            _log(f"[RenderField] 未找到模型: {ckpt_path}", cfg)
            return

        try:
            import torch
            self._torch = torch
            import uvh_point_rt_renderfield as upr

            pack = upr.load_point_rt_renderfield_checkpoint(
                ckpt_path, device=cfg.render_device)
            self.model = pack["model"]
            self.device_t = pack["device"]
            ckpt = pack.get("checkpoint", {})
            mcfg = ckpt.get("model_config", {}) if isinstance(ckpt, dict) else {}
            self.point_in_dim = int(mcfg.get("point_in_dim", 3) or 3)

            self.canonical_poses = self._build_canonical_poses()
            npose = len(self.canonical_poses)
            if cfg.render_init_pose_only:
                self.final_view_indices = [0]
            else:
                self.final_view_indices = list(range(npose))
            if cfg.render_sw_front_only:
                self.sw_view_indices = [0]
            else:
                self.sw_view_indices = list(self.final_view_indices)
            self.enabled = True
            self.lib_front_dist = float(base_lib_render_view_dist(cfg))
            _log(
                f"[RenderField] 模型加载成功: point_in_dim={self.point_in_dim}, "
                f"poses={npose}, final_views={self.final_view_indices}, "
                f"sw_views={self.sw_view_indices}, "
                f"dist={self.view_dist:.1f}, lib_front_dist={self.lib_front_dist:.3f}, "
                f"device={self.device_t}", cfg)
        except Exception as e:
            _log(f"[RenderField] 加载失败(将禁用): {e}", cfg)

    # ── 规范位姿生成 ──

    def _build_canonical_poses(self) -> list[np.ndarray]:
        """
        构建 3 个规范相机位姿（正前方 + 俯视偏移 + 侧偏移），
        保证从不同角度看到模型全貌。

        pose_vec = [rot6d(6), translation(3)]  共 9 维。
        T_w2c 约定：pts_cam = R @ pts_world + t
        """
        d = self.view_dist
        poses: list[np.ndarray] = []

        # View 1: 正前方 — camera 在 (0, 0, −d)，沿 +Z 看模型
        #   R = I, t = (0, 0, d)
        poses.append(np.array(
            [1, 0, 0,  0, 1, 0,  0, 0, d], dtype=np.float32))

        # View 2: 俯视偏移 20° — 绕 X 轴旋转 −20°
        ax = np.radians(20.0)
        cx_, sx_ = float(np.cos(ax)), float(np.sin(ax))
        Rx = np.array([[1, 0, 0], [0, cx_, sx_], [0, -sx_, cx_]],
                       dtype=np.float64)
        cam2 = np.array([0, -d * np.sin(ax), -d * np.cos(ax)], dtype=np.float64)
        t2 = (-Rx @ cam2).astype(np.float32)
        Rx32 = Rx.astype(np.float32)
        poses.append(np.concatenate(
            [Rx32[:, 0], Rx32[:, 1], t2], axis=0).astype(np.float32))

        # View 3: 侧偏移 20° — 绕 Y 轴旋转 20°
        ay = np.radians(20.0)
        cy_, sy_ = float(np.cos(ay)), float(np.sin(ay))
        Ry = np.array([[cy_, 0, -sy_], [0, 1, 0], [sy_, 0, cy_]],
                       dtype=np.float64)
        cam3 = np.array([d * np.sin(ay), 0, -d * np.cos(ay)], dtype=np.float64)
        t3 = (-Ry @ cam3).astype(np.float32)
        Ry32 = Ry.astype(np.float32)
        poses.append(np.concatenate(
            [Ry32[:, 0], Ry32[:, 1], t3], axis=0).astype(np.float32))

        return poses

    def set_lib_front_dist(self, d: float) -> None:
        """正前单视角渲染视距（规范空间 Z 平移）；建库/跟踪候选与此一致。"""
        self.lib_front_dist = float(max(0.35, min(32.0, float(d))))

    # ── 点云预处理 ──

    def _normalize_to_canon(self, pts_uvh: np.ndarray) -> np.ndarray:
        """将 uvh 点云归一化到规范空间（中心在原点，半径≈1）"""
        p = np.asarray(pts_uvh, dtype=np.float32)
        pmin = np.min(p, axis=0)
        pmax = np.max(p, axis=0)
        center = (0.5 * (pmin + pmax)).astype(np.float32)
        radius = float(max(1e-6, 0.5 * np.linalg.norm(pmax - pmin)))
        return ((p - center) / radius).astype(np.float32)

    def _resample(self, pts: np.ndarray) -> np.ndarray:
        """采样/填充到固定点数"""
        n = pts.shape[0]
        if n == self.n_points:
            return pts.copy()
        if n > self.n_points:
            idx = np.linspace(0, n - 1, self.n_points, dtype=np.int64)
            return pts[idx].copy()
        reps = self.n_points // max(1, n) + 1
        return np.tile(pts, (reps, 1))[:self.n_points].copy()

    def _pts_to_tensor(self, pts_canon: np.ndarray):
        """转为模型输入张量 (1, N, in_dim)"""
        torch = self._torch
        p = self._resample(pts_canon)
        if self.point_in_dim == 6 and p.shape[1] == 3:
            normals = np.zeros_like(p)
            normals[:, 2] = 1.0
            p = np.concatenate([p, normals], axis=1)
        return torch.from_numpy(p).float().unsqueeze(0)

    # ── 渲染 ──

    def _render_all_views(self, pts_canon: np.ndarray) -> list:
        """从所有规范位姿渲染，返回 list of (4,H,W) tensor"""
        torch = self._torch
        pts_t = self._pts_to_tensor(pts_canon).to(self.device_t)
        renders = []
        with torch.no_grad():
            for pose_np in self.canonical_poses:
                pose_t = torch.from_numpy(pose_np).float().unsqueeze(0).to(self.device_t)
                out = self.model(pts_t, pose_t)
                renders.append(out["render"][0].detach().cpu())
        return renders

    # ── 公开 API ──

    def set_reference(self, ref_pts_uvh: np.ndarray, cfg: RTSConfig | None = None):
        """用首帧参考点云生成参考渲染"""
        if not self.enabled:
            return
        if ref_pts_uvh.shape[0] < 100:
            _log("[RenderField] 参考点云过少，禁用验证", cfg)
            self.enabled = False
            return
        canon = self._normalize_to_canon(ref_pts_uvh)
        self.ref_renders = self._render_all_views(canon)
        _log(f"[RenderField] 参考渲染完成: {len(self.ref_renders)} 视图, "
             f"points={ref_pts_uvh.shape[0]}", cfg)

    def score_candidates_batch(
        self,
        pts_list: list[np.ndarray],
        *,
        view_indices: list[int] | None = None,
        batch_size: int = 12,
    ) -> np.ndarray:
        """
        批量评估多个候选点云与参考的相似度（用于滑动窗口）。
        返回 (M,) float32，与 pts_list 等长。
        """
        torch = self._torch
        if not self.enabled or not self.ref_renders or len(pts_list) == 0:
            return np.ones(len(pts_list), dtype=np.float32)
        vis = view_indices if view_indices is not None else self.sw_view_indices
        acc = np.zeros(len(pts_list), dtype=np.float64)
        for vi in vis:
            ref_r = self.ref_renders[vi]
            pose_np = self.canonical_poses[vi]
            pose_row = torch.from_numpy(pose_np).float().unsqueeze(0)

            for s0 in range(0, len(pts_list), int(batch_size)):
                chunk = pts_list[s0:s0 + int(batch_size)]
                mats: list[np.ndarray] = []
                valid: list[bool] = []
                for p in chunk:
                    if p.shape[0] < 10:
                        mats.append(np.zeros((self.n_points, 3), np.float32))
                        valid.append(False)
                    else:
                        mats.append(self._resample(self._normalize_to_canon(p)))
                        valid.append(True)
                arr = np.stack(mats, axis=0)
                if self.point_in_dim == 6:
                    z = np.zeros_like(arr)
                    z[:, :, 2] = 1.0
                    arr = np.concatenate([arr, z], axis=2)
                pts_t = torch.from_numpy(arr).float().to(self.device_t)
                B = int(pts_t.shape[0])
                pose_t = pose_row.expand(B, -1).to(self.device_t)
                with torch.no_grad():
                    out = self.model(pts_t, pose_t)
                pred = out["render"].detach().cpu()
                for j in range(B):
                    if not valid[j]:
                        acc[s0 + j] += 0.0
                    else:
                        acc[s0 + j] += self._compare_renders(ref_r, pred[j])
        acc /= max(1, len(vis))
        return acc.astype(np.float32)

    def score_candidate(self, cand_pts_uvh: np.ndarray) -> float:
        """评估候选点云与参考的形状相似度 (0~1, 1=完全一致)"""
        if not self.enabled or not self.ref_renders:
            return 1.0
        if cand_pts_uvh.shape[0] < 100:
            return 0.0
        canon = self._normalize_to_canon(cand_pts_uvh)
        cand_renders = self._render_all_views(canon)

        scores = []
        for vi in self.final_view_indices:
            scores.append(self._compare_renders(self.ref_renders[vi], cand_renders[vi]))
        return float(np.mean(scores))

    def _compare_renders(self, ref, cand) -> float:
        """比较两张 (4, H, W) 渲染图的相似度"""
        torch = self._torch

        mask_r = (ref[0:1] > 1e-6).float()
        mask_c = (cand[0:1] > 1e-6).float()

        # 轮廓 IoU
        inter = (mask_r * mask_c).sum()
        union = ((mask_r + mask_c) > 0.5).float().sum()
        iou = float(inter / torch.clamp(union, min=1.0))

        overlap = mask_r * mask_c
        n_ov = float(overlap.sum())
        if n_ov < 10:
            return iou * 0.3

        ov_flat = overlap[0] > 0.5

        # 深度相关性
        dr = ref[0][ov_flat]
        dc = cand[0][ov_flat]
        dr_m, dc_m = dr.mean(), dc.mean()
        dr_s, dc_s = dr.std(), dc.std()
        if float(dr_s) < 1e-6 or float(dc_s) < 1e-6:
            depth_corr = 0.5
        else:
            depth_corr = float(((dr - dr_m) * (dc - dc_m)).mean() / (dr_s * dc_s + 1e-8))
            depth_corr = max(0.0, min(1.0, (depth_corr + 1.0) * 0.5))

        # 法向相似度
        nr = ref[1:4] * 2.0 - 1.0
        nc = cand[1:4] * 2.0 - 1.0
        cos_sim = (nr * nc).sum(dim=0)
        normal_sim = float((cos_sim[ov_flat]).mean())
        normal_sim = max(0.0, min(1.0, (normal_sim + 1.0) * 0.5))

        return 0.30 * iou + 0.40 * depth_corr + 0.30 * normal_sim

    def render_batch_front(
        self,
        canon_list: list[np.ndarray],
        batch_size: int,
        view_dist: float | None = None,
    ) -> list[np.ndarray | None]:
        """
        多个规范点云在正前位姿下批量渲染；返回各 (4,H,W) float32 或 None。
        view_dist：规范空间相机 Z 平移（与 _build_canonical_poses 首视角一致），
        None 时用 self.lib_front_dist（建库与候选共用，可与 RenderField 多视角 dist 不同）。
        """
        torch = self._torch
        if not self.enabled or not canon_list:
            return [None] * len(canon_list)
        d_use = float(self.lib_front_dist) if view_dist is None else float(view_dist)
        pose_np = np.array(
            [1, 0, 0, 0, 1, 0, 0, 0, d_use], dtype=np.float32)
        pose_row = torch.from_numpy(pose_np).float().unsqueeze(0)
        res: list[np.ndarray | None] = [None] * len(canon_list)
        bs = max(1, int(batch_size))
        for s0 in range(0, len(canon_list), bs):
            chunk = canon_list[s0 : s0 + bs]
            mats: list[np.ndarray] = []
            valid: list[bool] = []
            for cp in chunk:
                cp = np.asarray(cp, dtype=np.float32)
                if cp.shape[0] < 10:
                    mats.append(np.zeros((self.n_points, 3), np.float32))
                    valid.append(False)
                else:
                    mats.append(self._resample(cp))
                    valid.append(True)
            arr = np.stack(mats, axis=0)
            if self.point_in_dim == 6:
                z = np.zeros_like(arr)
                z[:, :, 2] = 1.0
                arr = np.concatenate([arr, z], axis=2)
            pts_t = torch.from_numpy(arr).float().to(self.device_t)
            B = int(pts_t.shape[0])
            pose_t = pose_row.expand(B, -1).to(self.device_t)
            with torch.no_grad():
                out = self.model(pts_t, pose_t)
            pred = out["render"].detach().cpu().numpy()
            for j in range(B):
                if valid[j]:
                    res[s0 + j] = pred[j].astype(np.float32)
        return res

    # ── 调试可视化 ──

    def save_render_debug(
        self, path: str, ref_pts: np.ndarray, cand_pts: np.ndarray, score: float,
    ):
        """保存参考/候选渲染对比图"""
        if not self.enabled or not self.ref_renders:
            return
        torch = self._torch
        canon_cand = self._normalize_to_canon(cand_pts)
        cand_renders = self._render_all_views(canon_cand)

        panels = []
        for vi, (rr, cr) in enumerate(zip(self.ref_renders, cand_renders)):
            ref_d = (rr[0].numpy() * 255).clip(0, 255).astype(np.uint8)
            cand_d = (cr[0].numpy() * 255).clip(0, 255).astype(np.uint8)
            ref_cm = cv2.applyColorMap(ref_d, cv2.COLORMAP_TURBO)
            cand_cm = cv2.applyColorMap(cand_d, cv2.COLORMAP_TURBO)
            ref_cm[ref_d < 2] = 30
            cand_cm[cand_d < 2] = 30
            gap = np.full((ref_cm.shape[0], 3, 3), 180, dtype=np.uint8)
            panel = np.concatenate([ref_cm, gap, cand_cm], axis=1)
            cv2.putText(panel, f"v{vi} ref", (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            cv2.putText(panel, f"v{vi} cand", (ref_cm.shape[1] + 6, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            panels.append(panel)

        row_gap = np.full((3, panels[0].shape[1], 3), 180, dtype=np.uint8)
        canvas = panels[0]
        for p in panels[1:]:
            canvas = np.concatenate([canvas, row_gap, p], axis=0)

        bar = np.full((22, canvas.shape[1], 3), 40, dtype=np.uint8)
        cv2.putText(bar, f"RenderField score={score:.4f}", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 230, 200), 1)
        canvas = np.concatenate([canvas, bar], axis=0)
        cv2.imwrite(path, canvas)


def build_sliding_bbox_grid(
    region_xywh: tuple[int, int, int, int],
    win_w: int,
    win_h: int,
    stride: int,
    frame_wh: tuple[int, int],
    max_windows: int,
    priority_boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """
    在矩形搜索区内生成滑动窗 bbox 列表（去重），priority_boxes 插到前面优先打分。
    """
    W, H = frame_wh
    rx, ry, rw, rh = (int(v) for v in region_xywh)
    win_w = max(4, min(int(win_w), rw))
    win_h = max(4, min(int(win_h), rh))
    st = max(1, int(stride))

    out: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for pb in priority_boxes:
        bb = core.clamp_bbox(int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3]), W, H)
        if bb not in seen and bb[2] >= 4 and bb[3] >= 4:
            seen.add(bb)
            out.append(bb)

    y0 = ry
    y_end = ry + rh - win_h
    x_end = rx + rw - win_w
    if y_end < y0:
        y0 = ry
        y_end = ry
    if x_end < rx:
        x_end = rx

    yy = y0
    while yy <= y_end:
        xx = rx
        while xx <= x_end:
            bb = core.clamp_bbox(xx, yy, win_w, win_h, W, H)
            if bb not in seen and bb[2] >= 4 and bb[3] >= 4:
                seen.add(bb)
                out.append(bb)
            xx += st
        yy += st

    if not out:
        bb = core.clamp_bbox(rx, ry, min(win_w, rw), min(win_h, rh), W, H)
        if bb[2] >= 4 and bb[3] >= 4:
            out.append(bb)

    if len(out) > int(max_windows):
        idx = np.linspace(0, len(out) - 1, int(max_windows), dtype=np.int64)
        out = [out[int(k)] for k in idx]
    return out


def render_field_sliding_refine(
    *,
    frame_bgr: np.ndarray,
    region_xywh: tuple[int, int, int, int],
    template_wh: tuple[int, int],
    priority_boxes: list[tuple[int, int, int, int]],
    stride: int,
    max_windows: int,
    cfg: RTSConfig,
    ccfg: core.TrackConfig,
    render_validator: RenderFieldValidator,
    frame_wh: tuple[int, int],
    ref_contour_rgb: np.ndarray | None = None,
) -> tuple[tuple[int, int, int, int], float, int, float]:
    """
    在给定区域内滑动窗口，批量神经渲染打分；可选对 Top-K 用 Autoencoder_129 轮廓相似度融合重排。

    返回 (最佳bbox, 融合分, 候选数, 胜出窗的 AE 归一化相似度；-1 表示未算 AE)。
    融合分 = (1-ae_sw_weight)*render + ae_sw_weight*AE（仅对 Top-K 更新，其余保持 render 分）。
    """
    if not render_validator.enabled:
        pb = priority_boxes[0] if priority_boxes else (0, 0, 4, 4)
        return pb, 0.0, 0, -1.0

    boxes = build_sliding_bbox_grid(
        region_xywh, template_wh[0], template_wh[1], stride, frame_wh,
        max_windows, priority_boxes,
    )
    pts_list: list[np.ndarray] = []
    masks_list: list[np.ndarray] = []
    for bb in boxes:
        bx, by, bw, bh = bb
        roi = frame_bgr[by:by + bh, bx:bx + bw]
        th_roi, mask_roi, _ = core.compute_th_gray_truncated(roi, cfg.th_percent, ccfg)
        masks_list.append(mask_roi)
        pts_list.append(
            build_uvh_points(th_roi, bb, cfg.mesh_z_scale, cfg.max_points))

    scores = render_validator.score_candidates_batch(
        pts_list, batch_size=cfg.render_sw_batch)
    min_pts = int(cfg.render_sw_min_pts)
    for _j, _p in enumerate(pts_list):
        if int(_p.shape[0]) < min_pts:
            scores[_j] = -1.0
    if float(np.max(scores)) < 0.0:
        fb = priority_boxes[0] if priority_boxes else boxes[0]
        return fb, -1.0, len(boxes), -1.0

    fused = scores.astype(np.float64).copy()
    ae_at_best = -1.0
    use_ae = (
        bool(cfg.ae_enable)
        and bool(getattr(core, "SEMANTIC_ENABLED", False))
        and ref_contour_rgb is not None
        and int(ref_contour_rgb.size) > 0
    )
    if use_ae:
        w_ae = float(np.clip(cfg.ae_sw_weight, 0.0, 1.0))
        topk = max(1, min(int(cfg.ae_sw_topk), len(boxes)))
        order = np.argsort(-fused)
        sem_sz = tuple(ccfg.semantic_image_size)
        for _k in range(topk):
            j = int(order[_k])
            if fused[j] < 0.0:
                continue
            bb = boxes[j]
            cur_rgb = contour_rgb_from_roi_mask(
                masks_list[j], (bb[2], bb[3]), sem_sz)
            ae_j = float(core.semantic_similarity(ref_contour_rgb, cur_rgb, ccfg))
            fused[j] = (1.0 - w_ae) * float(scores[j]) + w_ae * ae_j

    bi = int(np.argmax(fused))
    if use_ae:
        bbb = boxes[bi]
        cur_best = contour_rgb_from_roi_mask(
            masks_list[bi], (bbb[2], bbb[3]), tuple(ccfg.semantic_image_size))
        ae_at_best = float(core.semantic_similarity(ref_contour_rgb, cur_best, ccfg))

    return boxes[bi], float(fused[bi]), len(boxes), ae_at_best


# ═══════════════════════════════════════════════════════════
#  固定视点 + 旋转目标库 + AE 滑动窗
# ═══════════════════════════════════════════════════════════

def rotate_y_canon(pts: np.ndarray, deg: float) -> np.ndarray:
    """规范系下绕 +Y 轴旋转（右手系），用于目标库方位角采样。"""
    p = np.asarray(pts, dtype=np.float32)
    t = math.radians(float(deg))
    c, s = math.cos(t), math.sin(t)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    xp = c * x + s * z
    zp = -s * x + c * z
    return np.stack([xp, y, zp], axis=1).astype(np.float32)


def rotate_x_canon(pts: np.ndarray, deg: float) -> np.ndarray:
    """规范系下绕 +X 轴旋转（右手系），用于目标库倾角环。"""
    p = np.asarray(pts, dtype=np.float32)
    t = math.radians(float(deg))
    c, s = math.cos(t), math.sin(t)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    yp = c * y - s * z
    zp = s * y + c * z
    return np.stack([x, yp, zp], axis=1).astype(np.float32)


def resolve_lib_rot_sampling(cfg: RTSConfig) -> tuple[int, int, float]:
    """
    返回 (n_y, n_tilt_rings, tilt_span_deg)。
    若环境变量 MIS_LIB_N_ROT 有值：仅 Y 轴均匀 n_y 份，n_tilt=1、span=0。
    否则：Y 轴 n_y 等分 [0,360)，另 n_tilt 档 X 倾角均匀覆盖 [-span, span]。
    """
    if os.environ.get("MIS_LIB_N_ROT", "").strip():
        return max(2, _ei("MIS_LIB_N_ROT", 8092)), 1, 0.0
    return (
        max(2, int(cfg.lib_n_rot_y)),
        max(1, int(cfg.lib_n_rot_tilt_rings)),
        float(cfg.lib_tilt_span_deg),
    )


def build_lib_y_tilt_grids(
    n_y: int, n_tilt: int, tilt_span: float,
) -> tuple[np.ndarray, np.ndarray]:
    """均匀 Y 角 + 均匀 X 倾角环；展平长度 n_y * n_tilt（tilt 外圈、Y 内圈）。"""
    ay = np.linspace(0.0, 360.0, int(n_y), endpoint=False, dtype=np.float64)
    if int(n_tilt) <= 1:
        tx = np.zeros(int(n_y), dtype=np.float64)
        return ay, tx
    tx1d = np.linspace(-float(tilt_span), float(tilt_span), int(n_tilt), dtype=np.float64)
    angles_y = np.tile(ay, int(n_tilt))
    tilt_x = np.repeat(tx1d, int(n_y))
    return angles_y, tilt_x


def canon_pose_for_lib_entry(
    canon0: np.ndarray, yaw_deg: float, tilt_x_deg: float,
) -> np.ndarray:
    """与建库一致：先绕 X 倾角，再绕 Y 方位。"""
    return rotate_y_canon(rotate_x_canon(canon0, float(tilt_x_deg)), float(yaw_deg))


def render_tensor_to_bgr_pack(
    r4: np.ndarray,
    out_wh: tuple[int, int],
) -> np.ndarray:
    """(4,H,W) depth+normal+… → (out_h,out_w,3) uint8 BGR，供 Autoencoder 预处理。"""
    r4 = np.asarray(r4, dtype=np.float32)
    d = r4[0]
    sil = d > 1e-6
    d_vis = np.zeros_like(d, dtype=np.float32)
    if np.any(sil):
        dv = d[sil]
        lo, hi = float(np.min(dv)), float(np.max(dv))
        rng = max(hi - lo, 1e-6)
        d_vis[sil] = (d[sil] - lo) / rng
    n0 = np.clip(r4[1], 0.0, 1.0)
    n1 = np.clip(r4[2], 0.0, 1.0)
    bgr = np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)
    bgr[..., 0] = (np.clip(d_vis * 255.0, 0, 255)).astype(np.uint8)
    bgr[..., 1] = (np.clip(n0 * 255.0, 0, 255)).astype(np.uint8)
    bgr[..., 2] = (np.clip(n1 * 255.0, 0, 255)).astype(np.uint8)
    ow, oh = int(out_wh[0]), int(out_wh[1])
    return cv2.resize(bgr, (ow, oh), interpolation=cv2.INTER_AREA)


def r4_depth_to_u8_vis(d: np.ndarray) -> np.ndarray:
    """(H,W) depth 通道 → 与 pack 可视化一致的 0–255 uint8（背景 0）。"""
    d = np.asarray(d, dtype=np.float32)
    sil = d > 1e-6
    out = np.zeros(d.shape[:2], dtype=np.uint8)
    if np.any(sil):
        dv = d[sil]
        lo, hi = float(np.min(dv)), float(np.max(dv))
        rng = max(hi - lo, 1e-6)
        out[sil] = (np.clip((d[sil] - lo) / rng, 0, 1) * 255.0).astype(np.uint8)
    return out


def r4_normal_to_gray_u8(r4: np.ndarray) -> np.ndarray:
    """(4,H,W) 渲染张量 → 法线三通道均值灰度 uint8，无前景处为 0。"""
    r4 = np.asarray(r4, dtype=np.float32)
    if r4.ndim != 3 or r4.shape[0] < 4:
        return np.zeros((1, 1), dtype=np.uint8)
    sil = r4[0] > 1e-6
    n0 = np.clip(r4[1], 0.0, 1.0)
    n1 = np.clip(r4[2], 0.0, 1.0)
    n2 = np.clip(r4[3], 0.0, 1.0)
    g = ((n0 + n1 + n2) / 3.0 * 255.0).astype(np.uint8)
    g = np.where(sil, g, 0).astype(np.uint8)
    return g


def ref_contours_from_render_r4(
    r4: np.ndarray,
    cfg: RTSConfig,
    ccfg: core.TrackConfig,
    semantic_sz: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """(4,H,W) 神经渲染 → depth/normal 轮廓。主跟踪已改用 ref_contours_from_th_roi；本函数供导出/对比复用。"""
    r4 = np.asarray(r4, dtype=np.float32)
    d_u8 = r4_depth_to_u8_vis(r4[0])
    d_bgr = cv2.cvtColor(d_u8, cv2.COLOR_GRAY2BGR)
    n_u8 = r4_normal_to_gray_u8(r4)
    n_bgr = cv2.cvtColor(n_u8, cv2.COLOR_GRAY2BGR)
    return (
        contour_rgb_after_th68_on_roi(d_bgr, cfg, ccfg, semantic_sz),
        contour_rgb_after_th68_on_roi(n_bgr, cfg, ccfg, semantic_sz),
    )


def pack_r4_similarity(r_a: np.ndarray, r_b: np.ndarray) -> float:
    """
    两帧 (4,H,W) 渲染张量（depth + normal + …）的轻量相似度，与 depth_grid_similarity 思路一致：
    前景 IoU + 交集上深度相关 + 归一化 L1，加权到 [0,1]。
    （滑动库跟踪中 pack 分项已改为 ROI→th_68→轮廓→AE，本函数保留供其它脚本复用。）
    """
    ra = np.asarray(r_a, dtype=np.float32)
    rb = np.asarray(r_b, dtype=np.float32)
    if ra.ndim != 3 or rb.ndim != 3 or ra.shape[0] < 4 or rb.shape[0] < 4:
        return 0.0
    ha, wa = int(ra.shape[1]), int(ra.shape[2])
    hb, wb = int(rb.shape[1]), int(rb.shape[2])
    if (ha, wa) != (hb, wb):
        rb = np.stack(
            [
                cv2.resize(rb[i], (wa, ha), interpolation=cv2.INTER_LINEAR).astype(np.float32)
                for i in range(4)
            ],
            axis=0,
        )
    da, db = ra[0], rb[0]
    ma = da > 1e-6
    mb = db > 1e-6
    inter = ma & mb
    union = ma | mb
    ni, nu = int(np.count_nonzero(inter)), int(np.count_nonzero(union))
    if ni < 5 or nu < 5:
        return 0.0
    iou = float(ni) / float(nu)

    daa = da[inter].astype(np.float64)
    dbb = db[inter].astype(np.float64)
    da_m = np.mean(daa)
    db_m = np.mean(dbb)
    da0 = daa - da_m
    db0 = dbb - db_m
    va = float(np.mean(da0 * da0))
    vb = float(np.mean(db0 * db0))
    if va < 1e-10 or vb < 1e-10:
        corr = 0.0
    else:
        corr = float(np.clip(np.mean(da0 * db0) / math.sqrt(va * vb + 1e-12), -1.0, 1.0))
    corr01 = 0.5 * (corr + 1.0)

    l1 = float(np.mean(np.abs(daa - dbb)))
    h_range = float(max(1e-6, np.max(daa) - np.min(daa), np.max(dbb) - np.min(dbb)))
    l1_norm = l1 / h_range
    l1_score = float(math.exp(-l1_norm * 3.0))

    return float(0.40 * iou + 0.40 * corr01 + 0.20 * l1_score)


def _lib_render_save_indices(n: int, stride: int, cap: int) -> list[int]:
    """决定要落盘的旋转库角度下标。"""
    if n <= 0:
        return []
    if int(cap) > 0:
        k = min(int(cap), n)
        return np.unique(np.linspace(0, n - 1, k, dtype=np.int64)).tolist()
    st = max(1, int(stride))
    return [i for i in range(n) if i % st == 0]


def save_rotation_render_channels(
    r4: np.ndarray,
    stem: str,
    *,
    max_side: int = 0,
    min_side: int = 0,
) -> None:
    """
    将 (4,H,W) 渲染张量存为三张 uint8 PNG：depth（伪彩）、normal（RGB）、silhouette（二值）。
    stem 为无后缀路径前缀，例如 .../rot_00012_y30.00deg
    min_side>0：先保证短边至少 min_side（放大，双三次）；再应用 max_side 上限（缩小）。
    """
    r4 = np.asarray(r4, dtype=np.float32)
    if r4.ndim != 3 or r4.shape[0] < 4:
        return
    d = r4[0]
    sil = d > 1e-6
    h, w = int(d.shape[0]), int(d.shape[1])

    def _maybe_resize(img: np.ndarray) -> np.ndarray:
        nh, nw = int(img.shape[0]), int(img.shape[1])
        if int(min_side) > 0:
            ms0 = min(nh, nw)
            if ms0 > 0 and ms0 < int(min_side):
                sc = float(min_side) / float(ms0)
                nw = max(1, int(round(nw * sc)))
                nh = max(1, int(round(nh * sc)))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
                nh, nw = int(img.shape[0]), int(img.shape[1])
        if int(max_side) > 0:
            mh = max(nh, nw)
            if mh > int(max_side):
                scale = float(max_side) / float(mh)
                nw = max(1, int(round(nw * scale)))
                nh = max(1, int(round(nh * scale)))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        return img

    # depth → turbo
    d_vis = np.zeros_like(d, dtype=np.float32)
    if np.any(sil):
        dv = d[sil]
        lo, hi = float(np.min(dv)), float(np.max(dv))
        rng = max(hi - lo, 1e-6)
        d_vis[sil] = (d[sil] - lo) / rng
    d_u8 = (np.clip(d_vis, 0, 1) * 255).astype(np.uint8)
    d_bgr = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
    d_bgr[~sil] = 0
    cv2.imwrite(f"{stem}_depth.png", _maybe_resize(d_bgr))

    # normal ch1..3 → BGR（模型输出为 [0,1]，与 _compare_renders 中 *2-1 前的存储一致）
    n_bgr = np.stack(
        [
            (np.clip(r4[1], 0, 1) * 255).astype(np.uint8),
            (np.clip(r4[2], 0, 1) * 255).astype(np.uint8),
            (np.clip(r4[3], 0, 1) * 255).astype(np.uint8),
        ],
        axis=-1,
    )
    n_bgr[~sil] = 0
    cv2.imwrite(f"{stem}_normal.png", _maybe_resize(n_bgr))

    sil_u8 = (sil.astype(np.uint8) * 255)
    cv2.imwrite(f"{stem}_silhouette.png", _maybe_resize(sil_u8))


def _resize_u8_hw3_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    if int(max_side) <= 0 or img.size == 0:
        return img
    h, w = int(img.shape[0]), int(img.shape[1])
    mh = max(h, w)
    if mh <= int(max_side):
        return img
    sc = float(max_side) / float(mh)
    nw = max(1, int(round(w * sc)))
    nh = max(1, int(round(h * sc)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def render_r4_to_normal_rgb_u8(r4: np.ndarray) -> np.ndarray:
    """(4,H,W) 渲染张量 → (H,W,3) uint8 RGB，背景黑。"""
    r4 = np.asarray(r4, dtype=np.float32)
    if r4.ndim != 3 or r4.shape[0] < 4:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    sil = r4[0] > 1e-6
    h, w = int(r4.shape[1]), int(r4.shape[2])
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        ch = (np.clip(r4[1 + c], 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb[:, :, c] = np.where(sil, ch, 0)
    return rgb


def write_point_cloud_ply_ascii(
    path: str,
    xyz: np.ndarray,
    *,
    rgb_u8: np.ndarray | None = None,
) -> None:
    """将 (N,3) 点云写入 ASCII PLY；rgb_u8 可选 (N,3) uint8。"""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] < 1:
        return
    n = int(xyz.shape[0])
    cols = None
    if rgb_u8 is not None:
        t = np.asarray(rgb_u8, dtype=np.uint8)
        if t.shape == (n, 3):
            cols = t
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if cols is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if cols is None:
            for i in range(n):
                f.write(f"{xyz[i, 0]} {xyz[i, 1]} {xyz[i, 2]}\n")
        else:
            for i in range(n):
                f.write(
                    f"{xyz[i, 0]} {xyz[i, 1]} {xyz[i, 2]} "
                    f"{int(cols[i, 0])} {int(cols[i, 1])} {int(cols[i, 2])}\n"
                )


def _export_one_library_view_record(
    vr_dir: str,
    idx: int,
    y_deg: float,
    tilt_x_deg: float,
    pts_canon: np.ndarray,
    r4: np.ndarray | None,
    bgr_pack: np.ndarray,
    cfg: RTSConfig,
) -> dict[str, Any]:
    """单视点：3D 点云（npy/ply）+ 神经渲染法线 RGB + AE 打包 BGR；返回 manifest 行。"""
    base = f"view_{idx:05d}_y{y_deg:.2f}_tx{tilt_x_deg:.2f}"
    stem = os.path.join(vr_dir, base)
    p = np.asarray(pts_canon, dtype=np.float32)
    rec: dict[str, Any] = {
        "idx": int(idx),
        "y_deg": float(y_deg),
        "tilt_x_deg": float(tilt_x_deg),
        "n_pts": int(p.shape[0]),
        "coord_space": "canon_radius_norm_rotated_y_then_x",
    }
    ms = int(cfg.lib_export_rgb_max_side)
    if cfg.lib_export_mesh_npy and p.shape[0] > 0:
        np.save(f"{stem}_mesh.npy", p)
        rec["mesh_npy"] = f"{base}_mesh.npy"
    if cfg.lib_export_mesh_ply and p.shape[0] > 0:
        z = p[:, 2]
        lo, hi = float(np.min(z)), float(np.max(z))
        zn = (z.astype(np.float64) - lo) / max(hi - lo, 1e-6)
        cc = (np.stack([zn, zn, 1.0 - zn], axis=1) * 255.0).clip(0, 255).astype(np.uint8)
        write_point_cloud_ply_ascii(f"{stem}_mesh.ply", p.astype(np.float64), rgb_u8=cc)
        rec["mesh_ply"] = f"{base}_mesh.ply"
    if cfg.lib_export_render_rgb and r4 is not None:
        rgb = render_r4_to_normal_rgb_u8(r4)
        rgb = _resize_u8_hw3_max_side(rgb, ms)
        cv2.imwrite(
            f"{stem}_render_normal_rgb.png",
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
        )
        rec["render_normal_rgb_png"] = f"{base}_render_normal_rgb.png"
    if cfg.lib_export_ae_pack_png and bgr_pack is not None and bgr_pack.size > 0:
        ap = np.asarray(bgr_pack, dtype=np.uint8)
        ap = _resize_u8_hw3_max_side(ap, ms)
        cv2.imwrite(f"{stem}_ae_input_pack_bgr.png", ap)
        rec["ae_input_pack_bgr_png"] = f"{base}_ae_input_pack_bgr.png"
    return rec


def _r4_to_depth_normal_sil_strip(r4: np.ndarray, gap_px: int = 4) -> np.ndarray:
    """(4,H,W) → 横向拼接 [depth turbo | normal BGR | silhouette]，BGR uint8。"""
    r4 = np.asarray(r4, dtype=np.float32)
    if r4.ndim != 3 or r4.shape[0] < 4:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    d = r4[0]
    sil = d > 1e-6
    d_vis = np.zeros_like(d, dtype=np.float32)
    if np.any(sil):
        dv = d[sil]
        lo, hi = float(np.min(dv)), float(np.max(dv))
        rng = max(hi - lo, 1e-6)
        d_vis[sil] = (d[sil] - lo) / rng
    d_u8 = (np.clip(d_vis, 0, 1) * 255).astype(np.uint8)
    d_bgr = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)
    d_bgr[~sil] = 0
    n_bgr = np.stack(
        [
            (np.clip(r4[1], 0, 1) * 255).astype(np.uint8),
            (np.clip(r4[2], 0, 1) * 255).astype(np.uint8),
            (np.clip(r4[3], 0, 1) * 255).astype(np.uint8),
        ],
        axis=-1,
    )
    n_bgr[~sil] = 0
    sil_3 = cv2.cvtColor((sil.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
    h = max(d_bgr.shape[0], n_bgr.shape[0], sil_3.shape[0])
    w = max(d_bgr.shape[1], n_bgr.shape[1], sil_3.shape[1])

    def _sz(im: np.ndarray) -> np.ndarray:
        if im.shape[0] != h or im.shape[1] != w:
            return cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        return im

    d_bgr, n_bgr, sil_3 = _sz(d_bgr), _sz(n_bgr), _sz(sil_3)
    g = np.full((h, gap_px, 3), 200, dtype=np.uint8)
    return np.concatenate([d_bgr, g, n_bgr, g, sil_3], axis=1)


def save_lib_cand_render_compare_panel(
    rv: RenderFieldValidator,
    ref_pts_uvh: np.ndarray,
    cand_pts_uvh: np.ndarray,
    lib_angle_deg: float,
    out_path: str,
    *,
    title_top: str,
    title_bot: str,
    max_side: int = 360,
    lib_tilt_x_deg: float = 0.0,
) -> bool:
    """
    上排：target 点云在库最佳姿态（X 倾角 + Y 方位）下固定视点渲染；下排：当前帧最佳候选窗同视点。
    """
    if not rv.enabled or ref_pts_uvh.shape[0] < 10 or cand_pts_uvh.shape[0] < 10:
        return False
    c0 = rv._normalize_to_canon(ref_pts_uvh)
    r0 = canon_pose_for_lib_entry(c0, float(lib_angle_deg), float(lib_tilt_x_deg))
    c1 = rv._normalize_to_canon(cand_pts_uvh)
    rens = rv.render_batch_front([r0, c1], batch_size=2)
    if rens[0] is None or rens[1] is None:
        return False
    row0 = _r4_to_depth_normal_sil_strip(np.asarray(rens[0], dtype=np.float32))
    row1 = _r4_to_depth_normal_sil_strip(np.asarray(rens[1], dtype=np.float32))
    w = max(row0.shape[1], row1.shape[1])
    tb = 24

    def _bar(text: str, width: int) -> np.ndarray:
        b = np.full((tb, width, 3), 45, dtype=np.uint8)
        cv2.putText(
            b, text[:120], (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (230, 230, 230), 1, cv2.LINE_AA,
        )
        return b

    if row0.shape[1] != w:
        row0 = cv2.resize(row0, (w, row0.shape[0]), interpolation=cv2.INTER_AREA)
    if row1.shape[1] != w:
        row1 = cv2.resize(row1, (w, row1.shape[0]), interpolation=cv2.INTER_AREA)
    top = np.vstack([_bar(title_top, w), row0])
    bot = np.vstack([_bar(title_bot, w), row1])
    gap = np.full((6, w, 3), 200, dtype=np.uint8)
    canvas = np.vstack([top, gap, bot])
    if max_side > 0:
        mh = max(canvas.shape[0], canvas.shape[1])
        if mh > int(max_side):
            sc = float(max_side) / float(mh)
            canvas = cv2.resize(
                canvas,
                (int(round(canvas.shape[1] * sc)), int(round(canvas.shape[0] * sc))),
                interpolation=cv2.INTER_AREA,
            )
    _ensure_dir(os.path.dirname(out_path) or ".")
    cv2.imwrite(out_path, canvas)
    return True


def _ae_init_modelmanager(ccfg: core.TrackConfig) -> Any:
    import extract_semantic_features_Function as esf
    mm = esf.ModelManager()
    dev = str(ccfg.device).strip() or None
    mm.initialize(
        model_path=str(ccfg.model_path),
        image_size=tuple(ccfg.semantic_image_size),
        device=dev,
    )
    mm.get_feature_extractor()
    return mm


def ae_encode_bgr_batch(
    mm: Any,
    bgr_list: list[np.ndarray],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    import torch
    import torch.nn.functional as F

    fe = mm.get_feature_extractor()
    device = mm.get_device()
    lat_chunks: list[np.ndarray] = []
    bn_chunks: list[np.ndarray] = []
    sw_chunks: list[np.ndarray] = []
    bs = max(1, int(batch_size))
    for s0 in range(0, len(bgr_list), bs):
        ch = bgr_list[s0 : s0 + bs]
        tlist = [mm.preprocess_image(im) for im in ch]
        x = torch.cat(tlist, dim=0).to(device)
        with torch.no_grad():
            feat = fe.extract_features(x)
        lat_chunks.append(feat["latent"].cpu().numpy().astype(np.float64))
        if "swin_bottleneck" in feat:
            b = feat["swin_bottleneck"]
            bn_chunks.append(
                F.adaptive_avg_pool2d(b, (1, 1)).squeeze(-1).squeeze(-1).cpu().numpy().astype(
                    np.float64))
        if "swin4" in feat:
            s = feat["swin4"]
            sw_chunks.append(
                F.adaptive_avg_pool2d(s, (1, 1)).squeeze(-1).squeeze(-1).cpu().numpy().astype(
                    np.float64))
    lat = np.concatenate(lat_chunks, axis=0)
    bn = np.concatenate(bn_chunks, axis=0) if bn_chunks else None
    sw = np.concatenate(sw_chunks, axis=0) if sw_chunks else None
    return lat, bn, sw


def _row_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def max_combined_ae_sim(
    c_lat: np.ndarray,
    c_bn: np.ndarray | None,
    c_sw: np.ndarray | None,
    lib_lat: np.ndarray,
    lib_bn: np.ndarray | None,
    lib_sw: np.ndarray | None,
    *,
    metric: str = "cosine",
) -> tuple[float, int, float, float]:
    """
    单候选 vs 库全体；加权方式与 calculate_contour_similarity 一致。

    metric:
      - cosine: 各尺度向量行 L2 归一化，得分 0.5*(1+cos)，再 0.4/0.35/0.25 加权。
      - euclid: 原 1/(1+L2) 与各尺度相同加权。

    返回
    ----
    best_sim, best_lib_idx, second_best_sim, margin (best - second)
    """
    m = str(metric or "cosine").strip().lower()
    lib_lat = np.asarray(lib_lat, dtype=np.float64)
    c_lat = np.asarray(c_lat, dtype=np.float64).reshape(-1)

    if m == "euclid":
        d0 = np.linalg.norm(lib_lat - c_lat.reshape(1, -1), axis=1)
        s0 = 1.0 / (1.0 + d0)
    else:
        nl = _row_normalize(lib_lat)
        nc = c_lat / max(float(np.linalg.norm(c_lat)), 1e-12)
        s0 = 0.5 * (1.0 + (nl @ nc))

    if lib_bn is not None and c_bn is not None and lib_sw is not None and c_sw is not None:
        lib_bn = np.asarray(lib_bn, dtype=np.float64)
        lib_sw = np.asarray(lib_sw, dtype=np.float64)
        c_bn = np.asarray(c_bn, dtype=np.float64).reshape(-1)
        c_sw = np.asarray(c_sw, dtype=np.float64).reshape(-1)
        if m == "euclid":
            d1 = np.linalg.norm(lib_bn - c_bn.reshape(1, -1), axis=1)
            d2 = np.linalg.norm(lib_sw - c_sw.reshape(1, -1), axis=1)
            s1 = 1.0 / (1.0 + d1)
            s2 = 1.0 / (1.0 + d2)
        else:
            nb = _row_normalize(lib_bn)
            nsw = _row_normalize(lib_sw)
            cb = c_bn / max(float(np.linalg.norm(c_bn)), 1e-12)
            csw = c_sw / max(float(np.linalg.norm(c_sw)), 1e-12)
            s1 = 0.5 * (1.0 + (nb @ cb))
            s2 = 0.5 * (1.0 + (nsw @ csw))
        comb = 0.4 * s0 + 0.35 * s1 + 0.25 * s2
    else:
        comb = s0

    comb = np.asarray(comb, dtype=np.float64).reshape(-1)
    bi = int(np.argmax(comb))
    best = float(comb[bi])
    if comb.size <= 1:
        return best, bi, best, 1.0
    c2 = comb.copy()
    c2[bi] = -np.inf
    second = float(np.max(c2))
    margin = best - second
    return best, bi, second, margin



def ae_pack_similarity_vs_ref(
    c_lat: np.ndarray,
    c_bn: np.ndarray | None,
    c_sw: np.ndarray | None,
    r_lat: np.ndarray,
    r_bn: np.ndarray | None,
    r_sw: np.ndarray | None,
    *,
    metric: str = "cosine",
) -> float:
    """单候选 pack 特征与参考（首帧渲染打包）的相似度，尺度加权与 max_combined_ae_sim 一致。"""
    m = str(metric or "cosine").strip().lower()
    c_lat = np.asarray(c_lat, dtype=np.float64).reshape(-1)
    r_lat = np.asarray(r_lat, dtype=np.float64).reshape(-1)
    if m == "euclid":
        s0 = 1.0 / (1.0 + float(np.linalg.norm(r_lat - c_lat)))
    else:
        s0 = 0.5 * (
            1.0
            + float(
                np.dot(
                    r_lat / max(float(np.linalg.norm(r_lat)), 1e-12),
                    c_lat / max(float(np.linalg.norm(c_lat)), 1e-12),
                )
            )
        )
    if r_bn is not None and c_bn is not None and r_sw is not None and c_sw is not None:
        r_bn = np.asarray(r_bn, dtype=np.float64).reshape(-1)
        r_sw = np.asarray(r_sw, dtype=np.float64).reshape(-1)
        c_bn = np.asarray(c_bn, dtype=np.float64).reshape(-1)
        c_sw = np.asarray(c_sw, dtype=np.float64).reshape(-1)
        if m == "euclid":
            s1 = 1.0 / (1.0 + float(np.linalg.norm(r_bn - c_bn)))
            s2 = 1.0 / (1.0 + float(np.linalg.norm(r_sw - c_sw)))
        else:
            s1 = 0.5 * (
                1.0
                + float(
                    np.dot(
                        r_bn / max(float(np.linalg.norm(r_bn)), 1e-12),
                        c_bn / max(float(np.linalg.norm(c_bn)), 1e-12),
                    )
                )
            )
            s2 = 0.5 * (
                1.0
                + float(
                    np.dot(
                        r_sw / max(float(np.linalg.norm(r_sw)), 1e-12),
                        c_sw / max(float(np.linalg.norm(c_sw)), 1e-12),
                    )
                )
            )
        return float(0.4 * s0 + 0.35 * s1 + 0.25 * s2)
    return float(s0)


def _icp_kabsch_step(
    src: np.ndarray,
    tgt_matched: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """行向量点集：src (N,3), tgt_matched (N,3)；返回 R(3,3), t(3,) 使 src @ R.T + t ≈ tgt。"""
    s = np.asarray(src, dtype=np.float64)
    t = np.asarray(tgt_matched, dtype=np.float64)
    mu_s = s.mean(axis=0)
    mu_t = t.mean(axis=0)
    p = s - mu_s
    q = t - mu_t
    h = p.T @ q
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    tt = mu_t - (r @ mu_s)
    return r.astype(np.float64), tt.astype(np.float64)


def icp_align_canonical(
    ref_tgt_canon: np.ndarray,
    src_canon: np.ndarray,
    cfg: RTSConfig,
) -> dict[str, Any]:
    """
    规范空间下点-点 ICP：将 src 对齐到 ref（两者均为已归一化坐标）。
    返回 R_acc(3,3), t_acc(3,), rmse, ok。
    """
    z3 = np.zeros(3, dtype=np.float64)
    fail: dict[str, Any] = {
        "ok": False,
        "R": np.eye(3, dtype=np.float64),
        "t": z3,
        "rmse": 999.0,
        "rot_deg_z": 0.0,
    }
    tgt = np.asarray(ref_tgt_canon, dtype=np.float64)
    src0 = np.asarray(src_canon, dtype=np.float64)
    if tgt.shape[0] < 30 or src0.shape[0] < 30:
        return fail

    n_max = max(200, int(cfg.icp_max_points))
    if tgt.shape[0] > n_max:
        idx = np.linspace(0, tgt.shape[0] - 1, n_max, dtype=np.int64)
        tgt = tgt[idx].copy()
    if src0.shape[0] > n_max:
        idx = np.linspace(0, src0.shape[0] - 1, n_max, dtype=np.int64)
        src0 = src0[idx].copy()

    try:
        from scipy.spatial import cKDTree
    except Exception:
        cKDTree = None

    src = src0.copy()
    r_acc = np.eye(3, dtype=np.float64)
    t_acc = np.zeros(3, dtype=np.float64)
    nit = max(3, int(cfg.icp_max_iters))
    tol = float(max(1e-9, float(cfg.icp_rel_tol)))

    for _ in range(nit):
        if cKDTree is not None:
            tree = cKDTree(tgt)
            _, idx = tree.query(src, k=1)
        else:
            d2 = np.sum((src[:, None, :] - tgt[None, :, :]) ** 2, axis=2)
            idx = np.argmin(d2, axis=1)
        mt = tgt[idx]
        r_step, t_step = _icp_kabsch_step(src, mt)
        prev = src.copy()
        src = (src @ r_step.T) + t_step
        r_acc = r_step @ r_acc
        t_acc = (r_step @ t_acc) + t_step
        delta = float(np.mean((src - prev) ** 2))
        if delta < tol:
            break

    if cKDTree is not None:
        tree = cKDTree(tgt)
        _, idx = tree.query(src, k=1)
    else:
        d2 = np.sum((src[:, None, :] - tgt[None, :, :]) ** 2, axis=2)
        idx = np.argmin(d2, axis=1)
    mt = tgt[idx]
    err = np.linalg.norm(src - mt, axis=1)
    rmse = float(np.sqrt(np.mean(err * err))) if err.size else 999.0
    rz = float(math.degrees(math.atan2(float(r_acc[1, 0]), float(r_acc[0, 0]))))
    return {
        "ok": True,
        "R": r_acc,
        "t": t_acc,
        "rmse": rmse,
        "rot_deg_z": rz,
    }


def _slide_match_bbox_pack_contour_icp_once(
    *,
    frame_bgr: np.ndarray,
    prev_bbox: tuple[int, int, int, int],
    first_bbox: tuple[int, int, int, int],
    frame_wh: tuple[int, int],
    rv: RenderFieldValidator,
    cfg: RTSConfig,
    ccfg: core.TrackConfig,
    mm: Any,
    cur_expand: int,
    ref_pts_uvh: np.ndarray,
    ref_contour_rgb: np.ndarray | None,
    ref_depth_contour_rgb: np.ndarray | None,
    ref_normal_contour_rgb: np.ndarray | None,
    ref_lat: np.ndarray | None,
    ref_bn: np.ndarray | None,
    ref_sw: np.ndarray | None,
    mesh_scorer: MeshSimilarityScorer | None = None,
    ref_th_roi: np.ndarray | None = None,
    ref_bbox_xywh: tuple[int, int, int, int] | None = None,
) -> tuple[
    tuple[int, int, int, int] | None,
    float,
    int,
    int,
    list[tuple[int, int, int, int]],
    dict[str, float],
]:
    search_bbox = compute_search_bbox(prev_bbox, cur_expand, frame_wh)
    tpl_w, tpl_h = int(first_bbox[2]), int(first_bbox[3])
    boxes = build_sliding_bbox_grid(
        search_bbox,
        tpl_w,
        tpl_h,
        int(cfg.lib_track_stride),
        frame_wh,
        int(cfg.lib_track_max_windows),
        [prev_bbox],
    )
    empty_ex: dict[str, float] = {
        "ae_raw": 0.0,
        "ae_margin": 0.0,
        "ae_second": 0.0,
        "pack_sim": 0.0,
        "contour_sim": 0.0,
        "fused": 0.0,
        "rts_refine_score": -1.0,
        "rts_refine_used": 0.0,
        "rts_refine_base_score": -1.0,
        "icp_rmse": 999.0,
        "icp_ok": 0.0,
        "icp_rot_deg_z": 0.0,
        "icp_tx": 0.0,
        "icp_ty": 0.0,
        "icp_tz": 0.0,
        "match_sim_rgb": 0.0,
        "match_sim_depth": 0.0,
        "match_sim_normal": 0.0,
        "mesh_simi": 0.0,
        "mesh_simi_ae_fused": 0.0,
    }
    if not boxes or not rv.enabled:
        return None, 0.0, -1, 0, [], empty_ex

    sx, sy, sw, sh = (int(v) for v in search_bbox)
    search_roi_bgr = frame_bgr[sy : sy + sh, sx : sx + sw]
    th_search_roi, mask_search_roi, _ = core.compute_th_gray_truncated(
        search_roi_bgr, float(cfg.th_percent), ccfg)
    pts_full = build_uvh_points(
        th_search_roi,
        search_bbox,
        float(cfg.mesh_z_scale),
        int(cfg.max_points),
    )
    if int(pts_full.shape[0]) < 50:
        return None, 0.0, -1, len(boxes), boxes, empty_ex

    use_mesh_simi_surface = (
        bool(cfg.mesh_simi_use_heightfield_mesh)
        and mesh_scorer is not None
        and mesh_scorer.enabled
        and ref_th_roi is not None
        and ref_bbox_xywh is not None
        and int(np.asarray(ref_th_roi).size) > 0
    )
    rng_mesh = np.random.default_rng(42)
    ref_surface_world: np.ndarray | None = None
    if use_mesh_simi_surface:
        pool_ref = max(
            int(cfg.mesh_simi_n_points)
            * max(1, int(cfg.mesh_simi_mesh_infer_pool_mult)),
            256,
        )
        ref_surface_world = mesh_surface_points_uvh(
            ref_th_roi,
            ref_bbox_xywh,  # type: ignore[arg-type]
            float(cfg.mesh_z_scale),
            pool_ref,
            rng_mesh,
        )
        if int(ref_surface_world.shape[0]) < 8:
            ref_surface_world = None

    patch_m = str(cfg.mesh_simi_patch_metric or "").strip().lower()
    use_th_patch_sim = (
        patch_m in ("depth", "normal")
        and ref_th_roi is not None
        and int(np.asarray(ref_th_roi).size) > 0
    )
    pgrid_tr = max(8, int(cfg.mesh_simi_patch_grid))

    depth_u8_search = pseudo_depth_u8_from_th(th_search_roi)
    normal_gray_search = pseudo_normal_gray_u8_from_th(th_search_roi)

    valid_idx: list[int] = []
    sem_sz = tuple(ccfg.semantic_image_size)
    met = str(cfg.lib_ae_metric or "cosine").strip().lower()

    use_triple = (
        bool(getattr(core, "SEMANTIC_ENABLED", False))
        and ref_contour_rgb is not None
        and int(getattr(ref_contour_rgb, "size", 0) or 0) > 0
        and ref_depth_contour_rgb is not None
        and int(ref_depth_contour_rgb.size) > 0
        and ref_normal_contour_rgb is not None
        and int(ref_normal_contour_rgb.size) > 0
    )

    wr = float(cfg.match_w_rgb_contour)
    wd = float(cfg.match_w_depth_contour)
    wn = float(cfg.match_w_normal_contour)
    wsum3 = wr + wd + wn
    if wsum3 < 1e-9:
        wsum3 = 1.0
        wr = wd = wn = 1.0 / 3.0

    wp = float(cfg.match_w_pack)
    wc = float(cfg.match_w_contour)
    wsum = wp + wc
    if wsum < 1e-9:
        wsum = 1.0
        wp, wc = 0.5, 0.5

    r_lat0 = (
        ref_lat[0]
        if ref_lat is not None and int(getattr(ref_lat, "size", 0) or 0) > 0
        else None
    )
    r_bn0 = ref_bn[0] if ref_bn is not None else None
    r_sw0 = ref_sw[0] if ref_sw is not None else None

    for bi, bb in enumerate(boxes):
        mask_pre = crop_xywh_from_search_patch(mask_search_roi, search_bbox, bb)
        if int(np.count_nonzero(mask_pre)) < int(cfg.lib_min_pts):
            continue
        valid_idx.append(bi)

    if not valid_idx:
        return None, 0.0, -1, len(boxes), boxes, empty_ex

    wh = (int(ccfg.semantic_image_size[0]), int(ccfg.semantic_image_size[1]))

    fused_scores: list[float] = []
    pack_sims: list[float] = []
    contour_sims: list[float] = []
    rgb_sims: list[float] = []
    depth_sims: list[float] = []
    normal_sims: list[float] = []
    mesh_primary_scoring = False

    if use_triple:
        ref_rgb = ref_contour_rgb
        ref_d = ref_depth_contour_rgb
        ref_n = ref_normal_contour_rgb
        for j in range(len(valid_idx)):
            bi = valid_idx[j]
            bb = boxes[bi]
            bx, by, bw, bh = bb

            mask_roi = crop_xywh_from_search_patch(mask_search_roi, search_bbox, bb)
            cur_rgb = contour_rgb_from_roi_mask(mask_roi, (bw, bh), sem_sz)
            sim_rgb = float(core.semantic_similarity(ref_rgb, cur_rgb, ccfg))

            d8 = crop_xywh_from_search_patch(depth_u8_search, search_bbox, bb)
            n8 = crop_xywh_from_search_patch(normal_gray_search, search_bbox, bb)
            c_depth = contour_rgb_from_pseudo_gray_mask(d8, sem_sz)
            c_nr = contour_rgb_from_pseudo_gray_mask(n8, sem_sz)
            sim_d = float(core.semantic_similarity(ref_d, c_depth, ccfg))
            sim_n = float(core.semantic_similarity(ref_n, c_nr, ccfg))

            fused = (wr * sim_rgb + wd * sim_d + wn * sim_n) / wsum3
            fused_scores.append(float(fused))
            rgb_sims.append(sim_rgb)
            depth_sims.append(sim_d)
            normal_sims.append(sim_n)
            pack_sims.append(float(0.5 * (sim_d + sim_n)))
            contour_sims.append(sim_rgb)
    else:
        use_mesh_primary = (
            (
                (mesh_scorer is not None and mesh_scorer.enabled)
                or use_th_patch_sim
            )
            and (r_lat0 is None or mm is None)
        )
        if use_mesh_primary:
            mesh_primary_scoring = True
            for j in range(len(valid_idx)):
                bi = valid_idx[j]
                bb = boxes[bi]
                if use_th_patch_sim:
                    th_c = crop_xywh_from_search_patch(
                        th_search_roi, search_bbox, bb)
                    ms = float(
                        th_pair_patch_similarity(
                            ref_th_roi,
                            th_c,
                            mode=patch_m,
                            grid=pgrid_tr,
                        )
                    )
                elif use_mesh_simi_surface and ref_surface_world is not None:
                    th_c = crop_xywh_from_search_patch(
                        th_search_roi, search_bbox, bb)
                    ms = float(
                        mesh_scorer.score_cand_th_vs_ref_surface(
                            ref_surface_world,
                            th_c,
                            bb,
                            rng_mesh,
                        )
                    )
                else:
                    cand_pts_i = crop_uvh_points_in_bbox(pts_full, bb)
                    if int(cand_pts_i.shape[0]) < 8:
                        ms = 0.0
                    else:
                        ms = float(mesh_scorer.score_pair(ref_pts_uvh, cand_pts_i))
                fused_scores.append(float(ms))
                pack_sims.append(float(ms))
                contour_sims.append(0.0)
                rgb_sims.append(0.0)
                depth_sims.append(0.0)
                normal_sims.append(0.0)
        elif r_lat0 is None or mm is None:
            return None, 0.0, -1, len(boxes), boxes, empty_ex
        else:
            bgrs = [
                pack_pseudo_dn_bgr(
                    crop_xywh_from_search_patch(
                        depth_u8_search, search_bbox, boxes[valid_idx[j]]),
                    crop_xywh_from_search_patch(
                        normal_gray_search, search_bbox, boxes[valid_idx[j]]),
                    wh,
                )
                for j in range(len(valid_idx))
            ]
            c_lat, c_bn, c_sw = ae_encode_bgr_batch(mm, bgrs, int(cfg.lib_ae_batch))

            for j in range(len(valid_idx)):
                bi = valid_idx[j]
                bb = boxes[bi]
                bx, by, bw, bh = bb
                mask_roi = crop_xywh_from_search_patch(mask_search_roi, search_bbox, bb)

                pk = ae_pack_similarity_vs_ref(
                    c_lat[j],
                    c_bn[j] if c_bn is not None else None,
                    c_sw[j] if c_sw is not None else None,
                    r_lat0,
                    r_bn0,
                    r_sw0,
                    metric=met,
                )
                pack_sims.append(float(pk))

                if ref_contour_rgb is not None and core.SEMANTIC_ENABLED:
                    cur_rgb = contour_rgb_from_roi_mask(mask_roi, (bw, bh), sem_sz)
                    cs = float(core.semantic_similarity(ref_contour_rgb, cur_rgb, ccfg))
                else:
                    cs = 0.0
                contour_sims.append(float(cs))
                fused_scores.append(float((wp * pk + wc * cs) / wsum))
                rgb_sims.append(cs)
                depth_sims.append(0.0)
                normal_sims.append(0.0)

    order = np.argsort(-np.asarray(fused_scores, dtype=np.float64))
    best_j = int(order[0])
    mesh_simi_score = 0.0
    topk_m = int(getattr(cfg, "mesh_simi_topk_rerank", 0))
    can_mesh_rerank = (
        mesh_scorer is not None
        and mesh_scorer.enabled
        or use_th_patch_sim
    )
    if (
        not mesh_primary_scoring
        and can_mesh_rerank
        and topk_m > 0
        and len(order) > 0
    ):
        wmesh = max(0.0, min(1.0, float(cfg.mesh_simi_fuse_weight)))
        k = int(min(topk_m, len(order)))
        combo_list: list[tuple[float, int, float]] = []
        for t in range(k):
            jpos = int(order[t])
            bb_i = boxes[valid_idx[jpos]]
            if use_th_patch_sim:
                th_c = crop_xywh_from_search_patch(
                    th_search_roi, search_bbox, bb_i)
                ms = float(
                    th_pair_patch_similarity(
                        ref_th_roi,
                        th_c,
                        mode=patch_m,
                        grid=pgrid_tr,
                    )
                )
            elif use_mesh_simi_surface and ref_surface_world is not None:
                th_c = crop_xywh_from_search_patch(
                    th_search_roi, search_bbox, bb_i)
                ms = float(
                    mesh_scorer.score_cand_th_vs_ref_surface(
                        ref_surface_world,
                        th_c,
                        bb_i,
                        rng_mesh,
                    )
                )
            elif mesh_scorer is not None and mesh_scorer.enabled:
                cand_pts_i = crop_uvh_points_in_bbox(pts_full, bb_i)
                if int(cand_pts_i.shape[0]) < 8:
                    ms = 0.0
                else:
                    ms = float(mesh_scorer.score_pair(ref_pts_uvh, cand_pts_i))
            else:
                ms = 0.0
            fused_i = float(fused_scores[jpos])
            combo = (1.0 - wmesh) * fused_i + wmesh * ms
            combo_list.append((combo, jpos, ms))
        combo_list.sort(key=lambda x: -x[0])
        combo_score, best_j, mesh_simi_score = combo_list[0]
        second_c = combo_list[1][0] if len(combo_list) > 1 else float(combo_score)
        margin = float(combo_score - second_c)
    else:
        combo_score = float(fused_scores[best_j])
        second_f = float(fused_scores[int(order[1])]) if len(order) > 1 else combo_score
        margin = float(combo_score - second_f)

    # 仅 1 个有效候选时 margin=0 会误触「模糊拒绝」，视为不模糊
    if len(fused_scores) < 2:
        margin = 1.0

    if mesh_primary_scoring:
        mesh_simi_score = float(fused_scores[best_j])

    ae_fused_at_pick = float(fused_scores[best_j])
    out_bb = boxes[valid_idx[best_j]]
    ref_c = rv._normalize_to_canon(np.asarray(ref_pts_uvh, dtype=np.float32))
    cand_pts = crop_uvh_points_in_bbox(pts_full, out_bb)
    cand_c = rv._normalize_to_canon(cand_pts)
    icp_d = icp_align_canonical(ref_c, cand_c, cfg)

    tt = np.asarray(icp_d.get("t", np.zeros(3)), dtype=np.float64).reshape(-1)
    ae_raw_val = (
        float(ae_fused_at_pick)
        if use_triple
        else float(pack_sims[best_j])
    )
    second_for_ex = (
        float(fused_scores[int(order[1])]) if len(order) > 1 else float(ae_fused_at_pick)
    )
    extra: dict[str, float] = {
        "ae_raw": ae_raw_val,
        "ae_margin": float(margin),
        "ae_second": float(second_for_ex),
        "pack_sim": float(pack_sims[best_j]),
        "contour_sim": float(contour_sims[best_j]),
        "fused": float(combo_score),
        "rts_refine_score": -1.0,
        "rts_refine_used": 0.0,
        "rts_refine_base_score": -1.0,
        "icp_rmse": float(icp_d.get("rmse", 999.0)),
        "icp_ok": 1.0 if icp_d.get("ok") else 0.0,
        "icp_rot_deg_z": float(icp_d.get("rot_deg_z", 0.0)),
        "icp_tx": float(tt[0]) if tt.size > 0 else 0.0,
        "icp_ty": float(tt[1]) if tt.size > 1 else 0.0,
        "icp_tz": float(tt[2]) if tt.size > 2 else 0.0,
        "match_sim_rgb": float(rgb_sims[best_j]),
        "match_sim_depth": float(depth_sims[best_j]),
        "match_sim_normal": float(normal_sims[best_j]),
        "mesh_simi": float(mesh_simi_score),
        "mesh_simi_ae_fused": float(ae_fused_at_pick),
    }
    return out_bb, float(combo_score), -1, len(boxes), boxes, extra


def slide_match_bbox_pack_contour_icp_adaptive(
    *,
    frame_bgr: np.ndarray,
    prev_bbox: tuple[int, int, int, int],
    first_bbox: tuple[int, int, int, int],
    frame_wh: tuple[int, int],
    rv: RenderFieldValidator,
    cfg: RTSConfig,
    ccfg: core.TrackConfig,
    mm: Any,
    base_expand: int,
    ref_pts_uvh: np.ndarray,
    ref_contour_rgb: np.ndarray | None,
    ref_depth_contour_rgb: np.ndarray | None,
    ref_normal_contour_rgb: np.ndarray | None,
    ref_lat: np.ndarray | None,
    ref_bn: np.ndarray | None,
    ref_sw: np.ndarray | None,
    mesh_scorer: MeshSimilarityScorer | None = None,
    ref_th_roi: np.ndarray | None = None,
    ref_bbox_xywh: tuple[int, int, int, int] | None = None,
) -> tuple[
    tuple[int, int, int, int] | None,
    float,
    int,
    int,
    list[tuple[int, int, int, int]],
    int,
    str,
    dict[str, float],
]:
    """
    扩大搜索区 → 当前帧搜索区内统一伪3D → 滑动窗；
    主路径：每窗 RGB ROI、渲染 depth、渲染 normal 各自 th_68→轮廓→AE 相似度加权；
    回退：pack latent + ROI 轮廓。最佳窗后在规范空间对 target / 候选裁剪点云做 ICP。
    若融合分低于 match_retry_sim_thr，则用 match_expand_large 再扫一遍（默认 128 像素边扩展）。
    """
    bb, sim, li, nw, boxes, ex = _slide_match_bbox_pack_contour_icp_once(
        frame_bgr=frame_bgr,
        prev_bbox=prev_bbox,
        first_bbox=first_bbox,
        frame_wh=frame_wh,
        rv=rv,
        cfg=cfg,
        ccfg=ccfg,
        mm=mm,
        cur_expand=int(base_expand),
        ref_pts_uvh=ref_pts_uvh,
        ref_contour_rgb=ref_contour_rgb,
        ref_depth_contour_rgb=ref_depth_contour_rgb,
        ref_normal_contour_rgb=ref_normal_contour_rgb,
        ref_lat=ref_lat,
        ref_bn=ref_bn,
        ref_sw=ref_sw,
        mesh_scorer=mesh_scorer,
        ref_th_roi=ref_th_roi,
        ref_bbox_xywh=ref_bbox_xywh,
    )
    expand_used = int(base_expand)
    tag = ""
    if bb is None:
        sim = -1.0

    ex2 = int(cfg.match_expand_large)
    if ex2 > int(base_expand) and sim + 1e-9 < float(cfg.match_retry_sim_thr):
        b1, s1, l1, n1, bx1, ex1 = _slide_match_bbox_pack_contour_icp_once(
            frame_bgr=frame_bgr,
            prev_bbox=prev_bbox,
            first_bbox=first_bbox,
            frame_wh=frame_wh,
            rv=rv,
            cfg=cfg,
            ccfg=ccfg,
            mm=mm,
            cur_expand=int(ex2),
            ref_pts_uvh=ref_pts_uvh,
            ref_contour_rgb=ref_contour_rgb,
            ref_depth_contour_rgb=ref_depth_contour_rgb,
            ref_normal_contour_rgb=ref_normal_contour_rgb,
            ref_lat=ref_lat,
            ref_bn=ref_bn,
            ref_sw=ref_sw,
            mesh_scorer=mesh_scorer,
            ref_th_roi=ref_th_roi,
            ref_bbox_xywh=ref_bbox_xywh,
        )
        if b1 is not None and s1 > sim:
            bb, sim, li, nw, boxes, ex = b1, s1, l1, n1, bx1, ex1
            expand_used = int(ex2)
            tag = f"+exp{ex2}"

    return bb, float(sim), int(li), int(nw), boxes, expand_used, tag, ex
# ═══════════════════════════════════════════════════════════
#  ROI 选取
# ═══════════════════════════════════════════════════════════

def select_initial_roi(first_bgr: np.ndarray, cfg: RTSConfig) -> tuple[int, int, int, int]:
    H, W = first_bgr.shape[:2]

    if cfg.init_bbox:
        roi = core._try_parse_bbox(cfg.init_bbox)
        if roi is not None:
            x, y, w, h = (int(v) for v in roi)
            return core.clamp_bbox(x, y, w, h, W, H)

    if cfg.vis_window:
        try:
            win = "RTS_Track_SelectROI"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, min(1200, W), min(900, H))
            roi = cv2.selectROI(win, first_bgr, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow(win)
            if roi is not None and int(roi[2]) > 1:
                return core.clamp_bbox(int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]), W, H)
        except Exception:
            pass

    try:
        roi = core._select_roi_tkinter(first_bgr, max_w=1200, max_h=900)
        if roi is not None:
            return core.clamp_bbox(int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]), W, H)
    except Exception:
        pass

    raise RuntimeError("ROI 选择失败：请通过 MIS_INIT_BBOX 环境变量指定初始框")


# ═══════════════════════════════════════════════════════════
#  主循环
# ═══════════════════════════════════════════════════════════

def main():
    cfg = RTSConfig()
    ccfg = _make_core_cfg(cfg)

    frames = core.list_frames(cfg.frames_dir)
    if cfg.max_frames > 0:
        frames = frames[:cfg.max_frames]
    if not frames:
        raise RuntimeError(f"未找到帧图像: {cfg.frames_dir}")

    # 输出目录
    out_root = _ensure_dir(cfg.output_dir)
    out_frames = _ensure_dir(os.path.join(out_root, "frames_with_bbox"))
    out_roi = _ensure_dir(os.path.join(out_root, "roi"))
    out_mask = _ensure_dir(os.path.join(out_root, "mask"))
    out_th = _ensure_dir(os.path.join(out_root, "th_gray"))
    out_debug = _ensure_dir(os.path.join(out_root, "rts_debug"))
    out_depth = _ensure_dir(os.path.join(out_root, "depth_compare"))
    out_match = _ensure_dir(os.path.join(out_root, "mesh_match"))

    # 读取第一帧
    first = core._read_frame_bgr(frames[0], ccfg)
    if first is None:
        raise RuntimeError(f"无法读取首帧: {frames[0]}")
    H, W = first.shape[:2]
    frame_wh = (W, H)
    _log(f"[信息] 首帧尺寸: {W}x{H}  帧数: {len(frames)}", cfg)

    # 选取初始 ROI
    bbox = select_initial_roi(first, cfg)
    _log(f"[信息] 初始 ROI: {list(bbox)}", cfg)

    # 首帧特征
    roi0 = core.crop(first, bbox)
    th0, mask0, _ = core.compute_th_gray_truncated(roi0, cfg.th_percent, ccfg)

    ae_constraint_ok = setup_autoencoder_constraint(ccfg, cfg)
    ref_contour_rgb: np.ndarray | None = None
    if ae_constraint_ok:
        ref_contour_rgb = contour_rgb_from_roi_mask(
            mask0, (bbox[2], bbox[3]), tuple(ccfg.semantic_image_size))

    # 构建参考 mesh
    ref_pts = build_uvh_points(th0, bbox, cfg.mesh_z_scale, cfg.max_points)
    if ref_pts.shape[0] < 50:
        raise RuntimeError(f"首帧参考点云过少: {ref_pts.shape[0]}")
    ref_center = np.mean(ref_pts, axis=0).astype(np.float64)
    _log(f"[信息] 参考 mesh: {ref_pts.shape[0]} 点, 重心={ref_center[:2].tolist()}", cfg)

    th_full, _, _ = core.compute_th_gray_truncated(first, cfg.th_percent, ccfg)
    frame_uvh_pts = build_uvh_points(
        th_full, (0, 0, W, H), cfg.mesh_z_scale, cfg.max_points)
    _log(
        f"[信息] 整帧伪3D（热身负样本池）: {int(frame_uvh_pts.shape[0])} 点",
        cfg,
    )
    warm_ckpt_written, warm_export_dir = mesh_similarity_quick_warmup(
        ref_pts,
        cfg,
        frame_uvh_pts=frame_uvh_pts,
        target_bbox=bbox,
        frame_wh=(W, H),
        ref_th_roi=th0 if bool(cfg.mesh_simi_use_heightfield_mesh) else None,
        frame_th_full=th_full if bool(cfg.mesh_simi_use_heightfield_mesh) else None,
    )
    mesh_scorer = MeshSimilarityScorer(
        cfg, primary_ckpt=(warm_ckpt_written or None))

    # 保存首帧
    if cfg.save_debug:
        cv2.imwrite(os.path.join(out_frames, "frame_0000.jpg"),
                     core.draw_bbox(first, bbox, text="f=0 init"))
        cv2.imwrite(os.path.join(out_roi, "roi_0000.jpg"), roi0)
        cv2.imwrite(os.path.join(out_mask, "mask_0000.png"), mask0)
        cv2.imwrite(os.path.join(out_th, "th_gray_0000.png"),
                     np.clip(th0, 0, 255).astype(np.uint8))
        # 首帧参考 mesh 面板（target = search, RTS 为零变换）
        rts_init = {"ok": True, "score": 1.0, "rot_deg": 0.0, "scale": 1.0,
                    "tx": 0.0, "ty": 0.0, "iou": 1.0, "depth_corr": 1.0,
                    "depth_l1_score": 1.0}
        save_mesh_match_process(
            os.path.join(out_match, "match_0000.jpg"),
            first, ref_pts, ref_pts, ref_center, rts_init,
            bbox, bbox, cfg.grid_size,
        )

    # 跟踪结果
    track_rows: list[dict[str, Any]] = []
    track_rows.append({
        "frame": 0, "image": frames[0], "bbox": list(map(int, bbox)),
        "score": 1.0, "rot_deg": 0.0, "scale": 1.0, "tx": 0.0, "ty": 0.0,
        "iou": 1.0, "depth_corr": 1.0, "depth_l1_score": 1.0,
        "render_sw_mode": "init", "render_sw_scan_score": 1.0, "render_score": 1.0,
        "ae_score": 1.0, "ae_sw_scan": 1.0,
        "lib_fused_score": 1.0,
        "lib_ae_raw": 1.0,
        "lib_pack_sim": 1.0,
        "lib_contour_sim": 1.0,
        "lib_rts_refine_score": -1.0,
        "lib_rts_refine_used": 0,
        "lib_rts_refine_base_score": -1.0,
        "icp_rmse": 0.0,
        "icp_ok": 1,
        "icp_rot_deg_z": 0.0,
        "icp_tx": 0.0,
        "icp_ty": 0.0,
        "icp_tz": 0.0,
        "lib_ae_margin": 1.0,
        "lib_ae_second": 1.0,
        "lib_ae_metric": str(cfg.lib_ae_metric),
        "lib_sliding_n": 0,
        "lib_match_idx": 0,
        "lib_match_angle_deg": 0.0,
        "lib_match_tilt_x_deg": 0.0,
        "lib_expand_used": int(cfg.search_expand),
        "lib_adapt_tag": "",
        "lib_shift_scale": 1.0,
        "raw_match_bbox": list(map(int, bbox)),
        "lib_match_sim_rgb": 1.0,
        "lib_match_sim_depth": 1.0,
        "lib_match_sim_normal": 1.0,
        "mesh_simi": 1.0,
        "mesh_simi_ae_fused": 1.0,
    })

    prev_bbox = bbox
    first_ref_pts = ref_pts.copy()
    first_ref_center = ref_center.copy()
    first_ref_th = th0.copy()
    first_ref_mask = mask0.copy()
    first_ref_bbox = bbox

    ref_th_img = th0
    ref_mask_img = mask0
    ref_bbox_for_tmpl = bbox

    consecutive_fail_count = 0

    # 神经渲染正前 pack + AE（无旋转目标库）
    render_validator = RenderFieldValidator(cfg)
    render_validator.set_reference(ref_pts, cfg)
    if not render_validator.enabled:
        raise RuntimeError(
            "需要 RenderField：请设置 MIS_RENDER_ENABLED=1 与有效 MIS_RENDER_MODEL 权重路径。"
        )
    out_render = _ensure_dir(os.path.join(out_root, "render_debug")) if cfg.save_debug else ""
    out_lib_match = _ensure_dir(os.path.join(out_root, "lib_match_render")) if cfg.lib_match_viz else ""

    ae_mm = _ae_init_modelmanager(ccfg) if ae_constraint_ok else None
    patch_on = str(cfg.mesh_simi_patch_metric or "").strip().lower() in (
        "depth",
        "normal",
    )
    if not ae_constraint_ok and not mesh_scorer.enabled and not patch_on:
        raise RuntimeError(
            "滑动窗匹配需要其一：MIS_MESH_SIMI_ENABLE=1 且 mesh_similarity 权重可读；"
            "或 MIS_AE_ENABLE=1 且 MIS_MODEL_PATH 指向 Autoencoder 权重并成功加载；"
            "或设置 MIS_MESH_PATCH_METRIC=depth 或 normal（伪深度/梯度 patch 相关度）。"
        )

    wh_ae = (int(ccfg.semantic_image_size[0]), int(ccfg.semantic_image_size[1]))
    canon0 = render_validator._normalize_to_canon(ref_pts)
    d_use = resolve_lib_front_distance_auto(render_validator, canon0, cfg)
    render_validator.set_lib_front_dist(d_use)

    sem_sz_t = tuple(ccfg.semantic_image_size)
    ref_depth_contour_rgb, ref_normal_contour_rgb = ref_contours_from_th_roi(
        th0, sem_sz_t)
    d0 = pseudo_depth_u8_from_th(th0)
    n0 = pseudo_normal_gray_u8_from_th(th0)
    ref_bgr_pack = pack_pseudo_dn_bgr(d0, n0, wh_ae)
    ref_lat: np.ndarray | None
    ref_bn: np.ndarray | None
    ref_sw: np.ndarray | None
    if bool(getattr(core, "SEMANTIC_ENABLED", False)):
        ref_lat = None
        ref_bn = None
        ref_sw = None
    elif ae_constraint_ok and ae_mm is not None:
        ref_lat, ref_bn, ref_sw = ae_encode_bgr_batch(
            ae_mm, [ref_bgr_pack], int(cfg.lib_ae_batch))
    else:
        ref_lat = None
        ref_bn = None
        ref_sw = None

    # 保存元信息
    meta = {
        "time": _now(),
        "task": "mesh_slide_pack_icp_tracking",
        "frames_dir": cfg.frames_dir,
        "output_dir": cfg.output_dir,
        "proc_scale": cfg.proc_scale,
        "search_expand": cfg.search_expand,
        "th_percent": cfg.th_percent,
        "mesh_z_scale": cfg.mesh_z_scale,
        "grid_size": cfg.grid_size,
        "rts_iters": cfg.rts_iters,
        "rts_rot_range_deg": cfg.rts_rot_range_deg,
        "ref_mode": cfg.ref_mode,
        "score_reject_thr": cfg.score_reject_thr,
        "max_center_shift_px": cfg.max_center_shift_px,
        "area_ratio_range": [cfg.area_ratio_lo, cfg.area_ratio_hi],
        "bbox_ema_alpha": cfg.bbox_ema_alpha,
        "mesh_match_save_stride": int(cfg.mesh_match_save_stride),
        "render_enabled": render_validator.enabled,
        "render_model_path": cfg.render_model_path if render_validator.enabled else "",
        "render_view_dist": cfg.render_view_dist,
        "lib_render_view_dist": float(
            getattr(render_validator, "lib_front_dist", base_lib_render_view_dist(cfg))),
        "lib_view_dist_cfg": float(cfg.lib_view_dist),
        "lib_view_dist_mult": float(cfg.lib_view_dist_mult),
        "lib_view_auto": bool(cfg.lib_view_auto),
        "ae_semantic_pack_wh": list(ccfg.semantic_image_size),
        "render_reject_thr": cfg.render_reject_thr,
        "render_n_views": len(render_validator.canonical_poses),
        "render_sw_enable": cfg.render_sw_enable,
        "render_sw_stride": cfg.render_sw_stride,
        "render_sw_replace_thr": cfg.render_sw_replace_thr,
        "render_sw_min_iou_prev": cfg.render_sw_min_iou_prev,
        "render_replace_min_ae_sw": cfg.render_replace_min_ae_sw,
        "render_full_poor_thr": cfg.render_full_poor_thr,
        "render_full_stride": cfg.render_full_stride,
        "render_full_score_margin": cfg.render_full_score_margin,
        "render_relax_cap_full": cfg.render_relax_cap_full,
        "render_relax_min_ae_sw": cfg.render_relax_min_ae_sw,
        "ae_enable": cfg.ae_enable,
        "ae_constraint_ok": bool(ae_constraint_ok),
        "ae_model_path": str(ccfg.model_path),
        "ae_reject_thr": cfg.ae_reject_thr,
        "ae_sw_weight": cfg.ae_sw_weight,
        "ae_sw_topk": cfg.ae_sw_topk,
        "match_expand_large": int(cfg.match_expand_large),
        "match_retry_sim_thr": float(cfg.match_retry_sim_thr),
        "match_w_rgb_contour": float(cfg.match_w_rgb_contour),
        "match_w_depth_contour": float(cfg.match_w_depth_contour),
        "match_w_normal_contour": float(cfg.match_w_normal_contour),
        "match_w_contour": float(cfg.match_w_contour),
        "match_w_pack": float(cfg.match_w_pack),
        "icp_max_iters": int(cfg.icp_max_iters),
        "icp_max_points": int(cfg.icp_max_points),
        "icp_rel_tol": float(cfg.icp_rel_tol),
        "lib_track_stride": cfg.lib_track_stride,
        "lib_track_max_windows": cfg.lib_track_max_windows,
        "lib_min_pts": cfg.lib_min_pts,
        "lib_rts_topk": int(cfg.lib_rts_topk),
        "lib_mesh_unify_scope": str(cfg.lib_mesh_unify_scope),
        "lib_rts_min_margin": float(cfg.lib_rts_min_margin),
        "lib_sim_reject_thr": cfg.lib_sim_reject_thr,
        "lib_ae_metric": str(cfg.lib_ae_metric),
        "lib_fuse_w_ae": cfg.lib_fuse_w_ae,
        "lib_fuse_w_pack": cfg.lib_fuse_w_pack,
        "lib_fuse_w_contour": cfg.lib_fuse_w_contour,
        "lib_ae_margin_min": cfg.lib_ae_margin_min,
        "lib_relax_shift_min_sim": cfg.lib_relax_shift_min_sim,
        "lib_relax_shift_mult": cfg.lib_relax_shift_mult,
        "lib_adapt_sim_t1": cfg.lib_adapt_sim_t1,
        "lib_adapt_expand_add1": cfg.lib_adapt_expand_add1,
        "lib_adapt_sim_t2": cfg.lib_adapt_sim_t2,
        "lib_adapt_expand_add2": cfg.lib_adapt_expand_add2,
        "lib_relax_shift_min_sim_2": cfg.lib_relax_shift_min_sim_2,
        "lib_relax_shift_mult_2": cfg.lib_relax_shift_mult_2,
        "lib_match_viz": bool(cfg.lib_match_viz),
        "lib_save_renders": bool(cfg.lib_save_renders),
        "lib_save_renders_stride": cfg.lib_save_renders_stride,
        "lib_save_renders_max": cfg.lib_save_renders_max,
        "lib_save_renders_max_side": cfg.lib_save_renders_max_side,
        "lib_save_renders_min_side": cfg.lib_save_renders_min_side,
        "lib_export_view_records": bool(cfg.lib_export_view_records),
        "lib_export_view_stride": cfg.lib_export_view_stride,
        "lib_export_view_max": cfg.lib_export_view_max,
        "lib_export_rgb_max_side": cfg.lib_export_rgb_max_side,
        "n_frames": len(frames),
        "first_bbox": list(map(int, bbox)),
        "first_ref_n_pts": int(ref_pts.shape[0]),
        "mesh_simi_enable": bool(cfg.mesh_simi_enable),
        "mesh_simi_ckpt": str(cfg.mesh_simi_ckpt),
        "mesh_simi_topk_rerank": int(cfg.mesh_simi_topk_rerank),
        "mesh_simi_fuse_weight": float(cfg.mesh_simi_fuse_weight),
        "mesh_simi_train_epochs": int(cfg.mesh_simi_train_epochs),
        "mesh_simi_train_samples_per_epoch": int(cfg.mesh_simi_train_samples_per_epoch),
        "mesh_simi_train_batch": int(cfg.mesh_simi_train_batch),
        "mesh_simi_train_steps": int(cfg.mesh_simi_train_steps),
        "mesh_simi_n_points": int(cfg.mesh_simi_n_points),
        "mesh_simi_warmup_written": warm_ckpt_written or "",
        "mesh_simi_warmup_cfg": str(cfg.mesh_simi_warmup_ckpt).strip(),
        "mesh_simi_warmup_neg_enable": bool(cfg.mesh_simi_warmup_neg_enable),
        "mesh_simi_warmup_neg_ratio": float(cfg.mesh_simi_warmup_neg_ratio),
        "mesh_simi_warmup_neg_label_lo": float(cfg.mesh_simi_warmup_neg_label_lo),
        "mesh_simi_warmup_neg_label_hi": float(cfg.mesh_simi_warmup_neg_label_hi),
        "mesh_simi_warmup_neg_iou_max": float(cfg.mesh_simi_warmup_neg_iou_max),
        "mesh_simi_warmup_rot_enable": bool(cfg.mesh_simi_warmup_rot_enable),
        "mesh_simi_warmup_rot_ratio": float(cfg.mesh_simi_warmup_rot_ratio),
        "mesh_simi_warmup_rot_label": float(cfg.mesh_simi_warmup_rot_label),
        "mesh_simi_warmup_rot_mode": str(cfg.mesh_simi_warmup_rot_mode).strip(),
        "mesh_simi_warmup_export": bool(cfg.mesh_simi_warmup_export),
        "mesh_simi_warmup_export_max": int(cfg.mesh_simi_warmup_export_max),
        "mesh_simi_warmup_dataset_dir": warm_export_dir or "",
        "mesh_simi_patch_metric": str(cfg.mesh_simi_patch_metric).strip(),
        "mesh_simi_patch_grid": int(cfg.mesh_simi_patch_grid),
    }
    with open(os.path.join(out_root, "meta_rts_track.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ===================== 主循环 =====================
    t_start = time.time()

    for i in range(1, len(frames)):
        img_path = frames[i]
        frame = core._read_frame_bgr(img_path, ccfg)
        if frame is None:
            _log(f"[警告] 跳过读取失败帧: {img_path}", cfg)
            continue

        # 1. 基础搜索边距（连续失败时自适应扩大）
        cur_expand = cfg.search_expand
        if consecutive_fail_count >= cfg.fail_expand_after:
            extra = min(consecutive_fail_count - cfg.fail_expand_after + 1,
                        cfg.fail_expand_max - cfg.fail_expand_after)
            cur_expand = int(cfg.search_expand * (1.0 + 0.5 * max(0, extra)))

        # 2. 参考 mesh（可视化 / ref_mode prev）
        if cfg.ref_mode == "first":
            cur_ref_pts = first_ref_pts
            cur_ref_center = first_ref_center
            cur_ref_th = first_ref_th
            cur_ref_mask = first_ref_mask
            cur_ref_bbox = first_ref_bbox
        else:
            cur_ref_pts = ref_pts
            cur_ref_center = ref_center
            cur_ref_th = ref_th_img
            cur_ref_mask = ref_mask_img
            cur_ref_bbox = ref_bbox_for_tmpl

        # 3. prev 模式：参考 ROI 的 th_68 截断图更新伪 depth/normal 轮廓与 pack（与当前帧搜索区同源，不再每帧神经渲染）
        if cfg.ref_mode == "prev":
            wh_ae2 = (
                int(ccfg.semantic_image_size[0]),
                int(ccfg.semantic_image_size[1]),
            )
            ref_depth_contour_rgb, ref_normal_contour_rgb = ref_contours_from_th_roi(
                cur_ref_th, sem_sz_t)
            d_prev = pseudo_depth_u8_from_th(cur_ref_th)
            n_prev = pseudo_normal_gray_u8_from_th(cur_ref_th)
            if bool(getattr(core, "SEMANTIC_ENABLED", False)):
                ref_lat = None
                ref_bn = None
                ref_sw = None
            elif ae_constraint_ok and ae_mm is not None:
                ref_bgr_pack = pack_pseudo_dn_bgr(d_prev, n_prev, wh_ae2)
                ref_lat, ref_bn, ref_sw = ae_encode_bgr_batch(
                    ae_mm, [ref_bgr_pack], int(cfg.lib_ae_batch))
            else:
                ref_lat = None
                ref_bn = None
                ref_sw = None

        if (
            cfg.ref_mode == "prev"
            and ref_contour_rgb is not None
            and cur_ref_mask is not None
        ):
            ref_contour_for_match = contour_rgb_from_roi_mask(
                cur_ref_mask,
                (int(cur_ref_bbox[2]), int(cur_ref_bbox[3])),
                sem_sz_t,
            )
        else:
            ref_contour_for_match = ref_contour_rgb

        # 滑动窗：搜索区一次 th_68 → 伪 depth/梯度图裁剪；三路轮廓 AE 加权 → 最佳框 → ICP（候选不再神经渲染）
        (
            raw_bbox,
            best_sim,
            best_lib_idx,
            n_sliding,
            _all_boxes,
            expand_used,
            lib_adapt_tag,
            lib_score_ex,
        ) = slide_match_bbox_pack_contour_icp_adaptive(
            frame_bgr=frame,
            prev_bbox=prev_bbox,
            first_bbox=first_ref_bbox,
            frame_wh=frame_wh,
            rv=render_validator,
            cfg=cfg,
            ccfg=ccfg,
            mm=ae_mm,
            base_expand=int(cur_expand),
            ref_pts_uvh=cur_ref_pts,
            ref_contour_rgb=ref_contour_for_match,
            ref_depth_contour_rgb=ref_depth_contour_rgb,
            ref_normal_contour_rgb=ref_normal_contour_rgb,
            ref_lat=ref_lat,
            ref_bn=ref_bn,
            ref_sw=ref_sw,
            mesh_scorer=mesh_scorer,
            ref_th_roi=cur_ref_th if bool(cfg.mesh_simi_use_heightfield_mesh) else None,
            ref_bbox_xywh=(
                cur_ref_bbox if bool(cfg.mesh_simi_use_heightfield_mesh) else None
            ),
        )

        # 4. 与最终 expand 一致的搜索区 mesh（调试图）
        search_bbox = compute_search_bbox(prev_bbox, expand_used, frame_wh)
        search_bounds = compute_search_bounds(search_bbox)
        sx, sy, sw, sh = search_bbox
        search_roi = frame[sy:sy + sh, sx:sx + sw]
        th_search, mask_search, _ = core.compute_th_gray_truncated(search_roi, cfg.th_percent, ccfg)
        search_pts = build_uvh_points(th_search, search_bbox, cfg.mesh_z_scale, cfg.max_points)
        match_angle = float(lib_score_ex.get("icp_rot_deg_z", 0.0))
        match_tilt = 0.0
        cand_pts_viz = np.zeros((0, 3), np.float32)
        if raw_bbox is not None:
            rbx, rby, rbw, rbh = (int(v) for v in raw_bbox)
            rroi = frame[rby : rby + rbh, rbx : rbx + rbw]
            th_r, _, _ = core.compute_th_gray_truncated(rroi, cfg.th_percent, ccfg)
            cand_pts_viz = build_uvh_points(th_r, raw_bbox, cfg.mesh_z_scale, cfg.max_points)

        ae_raw = float(lib_score_ex.get("ae_raw", best_sim))
        ae_margin = float(lib_score_ex.get("ae_margin", 0.0))
        pack_sim = float(lib_score_ex.get("pack_sim", 0.0))
        contour_sim = float(lib_score_ex.get("contour_sim", 0.0))
        rts = {
            "ok": raw_bbox is not None,
            "score": float(best_sim),
            "rot_deg": match_angle,
            "scale": 1.0,
            "tx": float(lib_score_ex.get("icp_tx", 0.0)),
            "ty": float(lib_score_ex.get("icp_ty", 0.0)),
            "iou": 0.0,
            "depth_corr": 0.0,
            "depth_l1_score": 0.0,
            "rmse": float(lib_score_ex.get("icp_rmse", 999.0)),
            "inlier_count": 0,
            "tmpl_confidence": 0.0,
            "warp_confidence": 0.0,
            "method": (
                "slide_pack_icp"
                + (f" adapt{lib_adapt_tag}" if lib_adapt_tag else "")
            ),
        }
        rts_vis = {
            **rts,
            "rot_deg": 0.0,
            "scale": 1.0,
            "tx": 0.0,
            "ty": 0.0,
        }

        match_accepted = raw_bbox is not None and float(best_sim) >= float(
            cfg.lib_sim_reject_thr)
        reject_reason = "lib_fail"
        if raw_bbox is None:
            reject_reason = "no_valid_candidate"
        elif not match_accepted:
            reject_reason = f"lib_fused_low={best_sim:.4f}<{cfg.lib_sim_reject_thr}"
        elif float(cfg.lib_ae_margin_min) > 1e-12 and ae_margin + 1e-12 < float(
            cfg.lib_ae_margin_min
        ):
            match_accepted = False
            reject_reason = (
                f"ae_margin={ae_margin:.4f}<{cfg.lib_ae_margin_min} "
                f"(ambiguous_lib_match)"
            )

        shift_scale = 1.0
        if raw_bbox is not None:
            bs = float(best_sim)
            if bs >= float(cfg.lib_relax_shift_min_sim):
                shift_scale = max(shift_scale, float(cfg.lib_relax_shift_mult))
            elif bs >= float(cfg.lib_relax_shift_min_sim_2):
                shift_scale = max(shift_scale, float(cfg.lib_relax_shift_mult_2))

        if match_accepted and raw_bbox is not None:
            ok, reason = validate_bbox_shift(raw_bbox, prev_bbox, cfg, shift_scale)
            if not ok:
                match_accepted = False
                reject_reason = reason

        if match_accepted and raw_bbox is not None:
            new_bbox = ema_smooth_bbox(raw_bbox, prev_bbox, cfg.bbox_ema_alpha)
            new_bbox = core.clamp_bbox(new_bbox[0], new_bbox[1],
                                       new_bbox[2], new_bbox[3], *frame_wh)
            consecutive_fail_count = 0
        else:
            new_bbox = prev_bbox
            consecutive_fail_count += 1
            if cfg.verbose >= 1 and consecutive_fail_count <= 5:
                _log(f"[拒绝] f={i} {reject_reason} → 回退上帧bbox, 连续失败={consecutive_fail_count}", cfg)

        if (
            out_lib_match
            and render_validator.enabled
            and raw_bbox is not None
            and int(cand_pts_viz.shape[0]) >= max(20, int(cfg.lib_min_pts) // 2)
        ):
            st = max(1, int(cfg.lib_match_viz_stride))
            if (i % st == 0) or (cfg.lib_match_viz_on_reject and not match_accepted):
                ang = float(match_angle)
                txm = 0.0
                vp = os.path.join(
                    out_lib_match,
                    f"f{i:04d}_a{int(match_accepted)}_fus{best_sim:.3f}_ae{ae_raw:.3f}_icpz{ang:.1f}.png",
                )
                save_lib_cand_render_compare_panel(
                    render_validator,
                    first_ref_pts,
                    cand_pts_viz,
                    ang,
                    vp,
                    lib_tilt_x_deg=txm,
                    title_top=(
                        f"Target vs cand (ICP rot_z≈{ang:.1f}deg, fixed front view)"
                    ),
                    title_bot=(
                        f"Best cand raw_bbox={list(map(int, raw_bbox))} "
                        f"fused={best_sim:.3f} ae={ae_raw:.3f} pack={pack_sim:.3f} "
                        f"cont={contour_sim:.3f} margin={ae_margin:.3f} "
                        f"acc={int(match_accepted)} "
                        f"{reject_reason if not match_accepted else 'ok'}"
                    )[:220],
                    max_side=int(cfg.lib_match_viz_max_side),
                )

        # 5. 用新 bbox 提取当前帧特征
        nx, ny, nw, nh = new_bbox
        cur_roi = frame[ny:ny + nh, nx:nx + nw]
        th_cur, mask_cur, _ = core.compute_th_gray_truncated(cur_roi, cfg.th_percent, ccfg)
        cur_pts = build_uvh_points(th_cur, new_bbox, cfg.mesh_z_scale, cfg.max_points)

        rf_score = float(best_sim) if raw_bbox is not None else 0.0
        ae_score_final = float(ae_raw) if raw_bbox is not None else -1.0
        sw_mode = "slide_pack_icp" + (lib_adapt_tag or "")
        sw_best_score = float(best_sim)
        ae_sw_scan = float(ae_raw) if raw_bbox is not None else -1.0

        # 6. 更新状态
        if cfg.ref_mode == "prev" and cur_pts.shape[0] >= 50:
            ref_pts = cur_pts
            ref_center = np.mean(ref_pts, axis=0).astype(np.float64)
            ref_th_img = th_cur
            ref_mask_img = mask_cur
            ref_bbox_for_tmpl = new_bbox

        prev_bbox = new_bbox

        # 日志
        if i % cfg.print_every == 0:
            elapsed = time.time() - t_start
            fps = float(i) / max(0.01, elapsed)
            status = "OK" if match_accepted else f"REJECT({reject_reason})"
            _log(
                f"[进度] f={i}/{len(frames)-1} bbox={list(map(int, new_bbox))} "
                f"{status} fused={rts.get('score', 0):.3f} "
                f"icp_z={rts.get('rot_deg', 0):.1f}° icp_rmse={rts.get('rmse', 0):.4f} "
                f"win_n={n_sliding} "
                f"[{rts.get('method', '?')}] best={rf_score:.3f} "
                f"rf_sw={sw_best_score:.3f}[{sw_mode}] exp={expand_used} "
                f"{'' if ae_score_final < 0.0 else f'ae={ae_score_final:.3f} '} "
                f"mrg={ae_margin:.3f} "
                f"fail_cnt={consecutive_fail_count} fps={fps:.1f}",
                cfg,
            )

        # 保存结果
        track_rows.append({
            "frame": i,
            "image": img_path,
            "bbox": list(map(int, new_bbox)),
            "accepted": match_accepted,
            "reject_reason": "" if match_accepted else reject_reason,
            "consecutive_fail": consecutive_fail_count,
            "score": float(rts.get("score", 0)),
            "rot_deg": float(rts.get("rot_deg", 0)),
            "scale": float(rts.get("scale", 1)),
            "tx": float(rts.get("tx", 0)),
            "ty": float(rts.get("ty", 0)),
            "iou": float(rts.get("iou", 0)),
            "depth_corr": float(rts.get("depth_corr", 0)),
            "depth_l1_score": float(rts.get("depth_l1_score", 0)),
            "search_bbox": list(map(int, search_bbox)),
            "n_ref_pts": int(cur_ref_pts.shape[0]),
            "n_search_pts": int(search_pts.shape[0]),
            "rmse": float(rts.get("rmse", 999)),
            "inlier_count": int(rts.get("inlier_count", 0)),
            "tmpl_confidence": float(rts.get("tmpl_confidence", 0)),
            "warp_confidence": float(rts.get("warp_confidence", 0)),
            "method": str(rts.get("method", "none")),
            "render_score": float(rf_score),
            "render_sw_mode": sw_mode if render_validator.enabled else "off",
            "render_sw_scan_score": float(sw_best_score),
            "ae_score": float(ae_score_final),
            "ae_sw_scan": float(ae_sw_scan),
            "lib_fused_score": float(best_sim),
            "lib_ae_raw": float(ae_raw),
            "lib_pack_sim": float(pack_sim),
            "lib_contour_sim": float(contour_sim),
            "lib_rts_refine_score": float(lib_score_ex.get("rts_refine_score", -1.0)),
            "lib_rts_refine_used": int(lib_score_ex.get("rts_refine_used", 0)),
            "lib_rts_refine_base_score": float(
                lib_score_ex.get("rts_refine_base_score", -1.0)),
            "lib_ae_margin": float(ae_margin),
            "lib_ae_second": float(lib_score_ex.get("ae_second", 0.0)),
            "lib_ae_metric": str(cfg.lib_ae_metric),
            "lib_sliding_n": int(n_sliding),
            "lib_match_idx": -1,
            "lib_match_angle_deg": float(match_angle),
            "lib_match_tilt_x_deg": float(match_tilt),
            "icp_rmse": float(lib_score_ex.get("icp_rmse", 999.0)),
            "icp_ok": int(lib_score_ex.get("icp_ok", 0)),
            "icp_rot_deg_z": float(lib_score_ex.get("icp_rot_deg_z", 0.0)),
            "icp_tx": float(lib_score_ex.get("icp_tx", 0.0)),
            "icp_ty": float(lib_score_ex.get("icp_ty", 0.0)),
            "icp_tz": float(lib_score_ex.get("icp_tz", 0.0)),
            "lib_expand_used": int(expand_used),
            "lib_adapt_tag": lib_adapt_tag,
            "lib_shift_scale": float(shift_scale),
            "raw_match_bbox": list(map(int, raw_bbox)) if raw_bbox is not None else [],
            "lib_match_sim_rgb": float(lib_score_ex.get("match_sim_rgb", 0.0)),
            "lib_match_sim_depth": float(lib_score_ex.get("match_sim_depth", 0.0)),
            "lib_match_sim_normal": float(lib_score_ex.get("match_sim_normal", 0.0)),
            "mesh_simi": float(lib_score_ex.get("mesh_simi", 0.0)),
            "mesh_simi_ae_fused": float(
                lib_score_ex.get("mesh_simi_ae_fused", 0.0)),
        })

        # 保存调试图
        if cfg.save_debug:
            # 带 bbox 的帧
            txt = (f"f={i} fus={rts.get('score', 0):.3f} "
                   f"icpz={rts.get('rot_deg', 0):.0f}°")
            cv2.imwrite(
                os.path.join(out_frames, f"frame_{i:04d}.jpg"),
                core.draw_bbox(frame, new_bbox, text=txt),
            )
            cv2.imwrite(os.path.join(out_roi, f"roi_{i:04d}.jpg"), cur_roi)
            cv2.imwrite(os.path.join(out_mask, f"mask_{i:04d}.png"), mask_cur)
            cv2.imwrite(os.path.join(out_th, f"th_gray_{i:04d}.png"),
                         np.clip(th_cur, 0, 255).astype(np.uint8))

            debug_stride = max(1, _ei("MIS_RTS_DEBUG_STRIDE", 5))
            if i % debug_stride == 0:
                save_rts_debug_image(
                    os.path.join(out_debug, f"rts_{i:04d}.jpg"),
                    frame, cur_ref_pts, cur_ref_center, rts_vis,
                    search_bbox, new_bbox,
                )
                # 深度图对比（库匹配无语义 RTS，用恒等变换展示 ref vs 搜索区 mesh）
                pts_t = apply_rts(
                    cur_ref_pts, cur_ref_center,
                    rts_vis.get("rot_deg", 0), rts_vis.get("scale", 1),
                    rts_vis.get("tx", 0), rts_vis.get("ty", 0),
                )
                d_ref_t, m_ref_t = rasterize_to_depth_grid(pts_t, search_bounds, cfg.grid_size)
                d_search_grid, m_search_grid = rasterize_to_depth_grid(
                    search_pts, search_bounds, cfg.grid_size)
                save_depth_grid_compare(
                    os.path.join(out_depth, f"depth_{i:04d}.png"),
                    d_ref_t, m_ref_t, d_search_grid, m_search_grid,
                    rts.get("score", 0),
                )

            # 完整匹配过程面板（大图）；MIS_MESH_MATCH_STRIDE>1 可降频，=0 跳过
            ms = int(cfg.mesh_match_save_stride)
            if ms > 0 and (i % ms == 0):
                save_mesh_match_process(
                    os.path.join(out_match, f"match_{i:04d}.jpg"),
                    frame, cur_ref_pts, search_pts,
                    cur_ref_center, rts_vis,
                    search_bbox, new_bbox,
                    cfg.grid_size,
                )

            # 渲染场对比（每 debug_stride 帧）
            if (render_validator.enabled and out_render
                    and i % debug_stride == 0 and cur_pts.shape[0] >= 100):
                render_validator.save_render_debug(
                    os.path.join(out_render, f"render_{i:04d}.png"),
                    ref_pts, cur_pts, rf_score,
                )

    # ===================== 保存最终结果 =====================
    elapsed_total = time.time() - t_start

    # JSON
    results_json = os.path.join(out_root, "tracking_results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(track_rows, f, ensure_ascii=False, indent=2)

    # CSV
    diag_csv = os.path.join(out_root, "diag_rts.csv")
    if track_rows:
        with open(diag_csv, "w", encoding="utf-8", newline="\n") as f:
            keys_set: set[str] = set()
            for _row in track_rows:
                keys_set.update(_row.keys())
            keys = sorted(keys_set)
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in track_rows:
                w.writerow({k: row.get(k, "") for k in keys})

    _log(f"\n[完成] 耗时: {elapsed_total:.1f}s  帧数: {len(track_rows)}", cfg)
    _log(f"[完成] 结果: {results_json}", cfg)
    _log(f"[完成] 诊断: {diag_csv}", cfg)
    _log(f"[完成] 输出: {out_root}", cfg)

    gt_dir_env = os.environ.get("MIS_GT_MASK_DIR", "").strip()
    if gt_dir_env:
        try:
            from eval_mesh_rts_track_vs_gt import evaluate_and_save

            ev_path = evaluate_and_save(
                results_json_path=results_json,
                frames_dir=cfg.frames_dir,
                gt_mask_dir_user=gt_dir_env,
                out_dir=out_root,
            )
            _log(f"[GT评估] 已写入: {ev_path}", cfg)
        except Exception as ex:
            _log(f"[GT评估] 失败(可稍后手动运行 eval_mesh_rts_track_vs_gt.py): {ex}", cfg)


if __name__ == "__main__":
    main()
