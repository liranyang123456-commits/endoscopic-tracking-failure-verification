#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chess Box 数据集：用首帧 GT mask 构建 target 伪 3D 点云（uvh），训练 MeshSimilarityNet。

- 默认 **动态数据集**：每个 step 随机成对（不预展开），`--steps-per-epoch` 控制每轮更新次数，
  配合 `set_epoch` 使各 epoch 组合不同；等效无限扩充样本，适合数据量小时训练。
- `--frames-per-seq 0`：使用该序列 **全部** GT 成功帧；>0 则均匀抽 K 帧。
- 负样本：从 **其它序列任意帧** 点云随机抽取（不仅首帧）。
- 几何增强：uv 平面小旋转、整体缩放、高斯噪声（归一化空间内）。
- **在线/持续学习**：
  - **脚本内循环**：`--online-loop` 每训完一轮（`--epochs`）后休眠 `--online-sleep-sec`，再 **重新扫盘** 加载序列，并从脚本顶部 `_MESH_SIM_OUT_DIR` 下的 `mesh_similarity_best.pt` 续训（第 2 轮起）；第 1 轮仍只用 `--resume`（若未指定则从头训）。
  - **外部调度**：也可用计划任务 / cron 周期性执行同一命令并带 `--resume .../mesh_similarity_best.pt` + `--finetune-lr 2e-4`。
- **加快训练**：`--geom-on-gpu`、`--batch-size`（显存允许）、略减 `--steps-per-epoch`；已默认 AMP、`num_workers`、`pin_memory`。
- **多数据根**：`--extra-data-root` 可重复指定其它 Chess 盒目录；序列键为 `数据集文件夹名__captured_photos*`。
  若多套数据里 `captured_photos*` 重名，GT 请使用 `gt_root/<数据集文件夹名>/<captured_photos*>/summary.csv`，否则仍可用扁平 `gt_root/<captured_photos*>/`（单套数据时与旧版一致）。
- **正样本（同一伪 3D mesh）**：从同序列选一帧归一化点云 `base`，对 **query 支路** 施加随机 **SE(3)**（Haar SO(3)+有界平移）得到「同 mesh、不同位姿」；**无遮挡时标签=1**。`--random-ref-prob` 控制 `base` 来自随机帧还是首帧缓存。
- **遮挡**：以 `--posit-occ-prob` 对变换后的 query 随机丢点；**相似度标签** 默认 **1−遮挡率**（`--label-is-occlusion` 时为遮挡率）。
- **训练/验证**：默认按 **序列名** 划分（`--val-fraction`，默认 0.2）；验证集 **关闭几何增广**，每 epoch 用 `--val-steps` 动态采样。
  `mesh_similarity_best.pt` 以 **验证 loss 最低** 为准（仅 1 条序列或 `--val-fraction 0` 时无验证，则按训练 loss）。
- **配对可视化**：路径在脚本顶部常量 `_MESH_SIM_OUT_DIR` / `_PAIR_VIZ_*` 配置；训练开始前导出 **ref/query** PNG；`_PAIR_VIZ_ENABLED=False` 可关。需 **matplotlib**。
- **早停**：`--early-stop-patience N`（N>0）：监控指标与保存 best 一致（有 val 用 val_loss，否则 train_loss）连续 **N** 个 epoch 相对历史最优下降不足 `--early-stop-min-delta` 则提前结束。
- **速度**：瓶颈多在「每 epoch 迭代次数 × Python/CUDA 启动」而非 FLOPs。CUDA 上默认 **混合精度 AMP**、`pin_memory`+`non_blocking`、**num_workers>0** 预取；显存够时增大 `--batch-size`、或减少 `--steps-per-epoch` 用更少步但更大 batch 逼近同等吞吐。
- **CPU 很满、显存不满**：几何增广可在 GPU 上做（`--geom-on-gpu`），DataLoader 只负责采样/遮挡，显著降低 CPU 占用；同时可看启动时 `[显存]` 提示加大 batch。

## 模型与推理（MeshSimilarityNet，见 mesh_similarity_model.py）

- **结构**：两路共享权重的 PointNet 编码器（Conv1d+BN+ReLU → max pool）→ 256 维嵌入；
  将 e1、e2、|e1-e2| 拼成 768 维，经 MLP + Sigmoid 得到标量。
- **损失**：每 batch `SmoothL1Loss(pred, label)`（beta=0.05），epoch 打印的 **train_loss / val_loss** 即各 batch 损失算术平均（mean_loss）。
  标签为 **标量回归**：同 mesh 经 SE(3) 对齐语义下 **无遮挡→1**；有遮挡→**1−遮挡**（或 `--label-is-occlusion`）；异 mesh 负样本 **∼U(0, neg_max_sim)**（默认 neg_max_sim=0.1，上界开区间）。
  缓存阶段仍为 **ROI 内归一化 uvh**；正对分支在归一化坐标内显式 **R,t**，再配合 uv 平面增广等减轻与真实相机位姿的分布差。
- **输入**：`pts_a`, `pts_b` 均为 **float32**，形状 **(B, N, 3)**，坐标需与训练一致：
  先由 ROI+阈值得到 uvh 点云，再按本脚本的 `normalize_point_cloud`（去质心、按最大半径缩放到单位球）；
  推理时常用固定 **N=训练时的 n_points**（默认 1024），对点云做随机或均匀重采样即可。
- **输出**：`sim = model(pts_a, pts_b)` 形状 **(B,)**，取值约 **(0,1)**，表示两团点云与同一目标的相似程度
  （训练标签：同 mesh+SE(3) → 1 或 1−遮挡；异序列 → 低分 < neg_max_sim）。
- **加载权重**：`ckpt = torch.load(".../mesh_similarity_best.pt", map_location=device)`，`model.load_state_dict(ckpt["model"])`；
  构造须与 `train_config.json` 一致（`embed_dim` / `head_dropout` / `l2_normalize_embed` 等）。

运行示例：
  python train_mesh_similarity_chess.py --epochs 100 --geom-on-gpu --batch-size 64 --device cuda
  python train_mesh_similarity_chess.py --online-loop --epochs 5 --finetune-lr 2e-4 --online-sleep-sec 300
  python train_mesh_similarity_chess.py --val-fraction 0 --resume mesh_sim_runs/live/mesh_similarity_best.pt --finetune-lr 2e-4 --epochs 20
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 仓库根目录
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# 训练输出根目录与配对图导出（只改这里，无需命令行传路径）
# ---------------------------------------------------------------------------
_MESH_SIM_OUT_DIR = Path(r"D:\MIS_Pose_Track_Re3D\mesh_sim_runs\chess_v1")
_PAIR_VIZ_ENABLED = True
_PAIR_VIZ_SUBDIR = "pair_viz"
_PAIR_VIZ_COUNT = 48
_PAIR_VIZ_DPI = 120

import compare_chessboard_trackers as cct
import MIS_C_Fe_Simi_Mesh_Match_Pose_Estimate_Track as core
try:
    import MIS_Mesh_RTS_Track as mrt
except ImportError:
    import MIS_Mesh_RTS_TrackDepth as mrt

from mesh_similarity_model import MeshSimilarityNet


def _sorted_seq_dirs(data_root: Path) -> list[Path]:
    if not data_root.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name.lower().startswith("captured_photos"):
            out.append(p)
    return out


def split_seq_names_for_val(
    all_names: list[str],
    val_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """按序列名划分训练/验证（序列级划分，避免同序列泄漏）。"""
    names = sorted(all_names)
    n = len(names)
    vf = float(np.clip(val_fraction, 0.0, 0.95))
    if n <= 1 or vf <= 0.0:
        return names, []
    rng = np.random.default_rng(int(seed) + 902_383)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * vf)))
    n_val = min(n_val, n - 1)
    val_ix = perm[:n_val]
    train_ix = perm[n_val:]
    val_names = sorted([names[i] for i in val_ix.tolist()])
    train_names = sorted([names[i] for i in train_ix.tolist()])
    return train_names, val_names


def resolve_gt_directory(gt_root: Path, bundle_root: Path, seq_name: str) -> Path | None:
    """
    多数据根时避免 captured_photos* 重名：优先 gt_root/<数据集文件夹名>/<seq_name>/summary.csv，
    否则回退 gt_root/<seq_name>/（与旧版单根布局兼容）。
    """
    nested = gt_root / bundle_root.name / seq_name
    flat = gt_root / seq_name
    if (nested / "summary.csv").is_file():
        return nested
    if (flat / "summary.csv").is_file():
        return flat
    return None


def _pick_frame_indices(n_total: int, k: int, seed: int) -> list[int]:
    if n_total <= 0 or k <= 0:
        return []
    k = min(k, n_total)
    rng = np.random.default_rng(int(seed))
    if k >= n_total:
        return list(range(n_total))
    # 均匀锚点 + 少量随机，保证覆盖
    base = np.linspace(0, n_total - 1, k, dtype=np.int64)
    idx = sorted(set(int(x) for x in base.tolist()))
    while len(idx) < k:
        idx.append(int(rng.integers(0, n_total)))
        idx = sorted(set(idx))
    return idx[:k]


def framegt_to_uvh_points(
    gt: cct.FrameGT,
    *,
    th_percent: float,
    mesh_z_scale: float,
    max_points: int,
    ccfg: core.TrackConfig,
) -> np.ndarray:
    bgr = cv2.imread(str(gt.image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return np.zeros((0, 3), np.float32)
    x, y, w, h = (int(v) for v in gt.bbox)
    H, W = bgr.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(4, min(w, W - x))
    h = max(4, min(h, H - y))
    roi = bgr[y : y + h, x : x + w]
    th, _mask, _ = core.compute_th_gray_truncated(roi, float(th_percent), ccfg)
    bbox = (x, y, w, h)
    pts = mrt.build_uvh_points(th, bbox, float(mesh_z_scale), int(max_points))
    return np.asarray(pts, dtype=np.float32)


def random_rotation_so3_numpy(rng: np.random.Generator) -> np.ndarray:
    """Haar 均匀随机旋转矩阵 (3,3) float32；行向量点 p' = p @ R.T。"""
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


def apply_so3_points_rows(pts: np.ndarray, R: np.ndarray) -> np.ndarray:
    """pts (N,3) 行向量，R (3,3) 正交；返回 pts @ R.T。"""
    p = np.asarray(pts, dtype=np.float32)
    r = np.asarray(R, dtype=np.float32)
    if p.ndim != 2 or p.shape[1] != 3 or r.shape != (3, 3):
        return p
    return (p @ r.T).astype(np.float32)


def apply_se3_points_rows(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """行向量点云：p' = p @ R.T + t；R (3,3)，t 可广播为 (3,)。"""
    p = np.asarray(pts, dtype=np.float32)
    r = np.asarray(R, dtype=np.float32)
    tt = np.asarray(t, dtype=np.float32).reshape(1, 3)
    if p.ndim != 2 or p.shape[1] != 3 or r.shape != (3, 3):
        return p
    return (p @ r.T + tt).astype(np.float32)


def normalize_point_cloud(pts: np.ndarray) -> np.ndarray:
    """零均值 + 最大半径归一，便于 PointNet。"""
    p = np.asarray(pts, dtype=np.float32)
    if p.shape[0] < 8:
        return np.zeros((0, 3), np.float32)
    c = np.mean(p, axis=0)
    p0 = p - c[None, :]
    r = float(np.max(np.linalg.norm(p0, axis=1))) + 1e-6
    return (p0 / r).astype(np.float32)


def resample_points(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    p = np.asarray(pts, dtype=np.float32)
    if p.shape[0] == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if p.shape[0] >= n:
        idx = rng.choice(p.shape[0], size=n, replace=False)
        return p[idx].copy()
    idx = rng.integers(0, p.shape[0], size=n)
    return p[idx].copy()


def random_occlude_points(
    pts: np.ndarray,
    occlusion_ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """随机丢弃 occlusion_ratio 比例的点（不放回），再返回子集。"""
    p = np.asarray(pts, dtype=np.float32)
    if p.shape[0] < 4:
        return p.copy()
    oc = float(np.clip(occlusion_ratio, 0.0, 0.95))
    n_keep = max(4, int(round(p.shape[0] * (1.0 - oc))))
    n_keep = min(n_keep, p.shape[0])
    idx = rng.choice(p.shape[0], size=n_keep, replace=False)
    return p[idx].copy()


def geom_augment_normalized(
    pts: np.ndarray,
    rng: np.random.Generator,
    *,
    rot_deg_max: float,
    scale_min: float,
    scale_max: float,
    noise_std: float,
    apply_prob: float,
) -> np.ndarray:
    """在已归一化点云上：绕 Z 轴旋转（对应图像 u,v）、均匀缩放、加性噪声。"""
    p = np.asarray(pts, dtype=np.float32).copy()
    if p.shape[0] < 4 or rng.random() > float(apply_prob):
        return p
    th = math.radians(float(rng.uniform(-rot_deg_max, rot_deg_max)))
    c, s = float(np.cos(th)), float(np.sin(th))
    xy = p[:, :2].copy()
    p[:, 0] = c * xy[:, 0] - s * xy[:, 1]
    p[:, 1] = s * xy[:, 0] + c * xy[:, 1]
    sc = float(rng.uniform(scale_min, scale_max))
    p *= sc
    if noise_std > 1e-9:
        p += rng.normal(0.0, noise_std, size=p.shape).astype(np.float32)
    return p


def geom_augment_batch_torch(
    pts: torch.Tensor,
    *,
    rot_deg_max: float,
    scale_min: float,
    scale_max: float,
    noise_std: float,
    apply_prob: float,
    generator: torch.Generator,
) -> torch.Tensor:
    """(B,N,3) 与 `geom_augment_normalized` 一致的批量增广；应在 GPU 上调用以减轻 CPU。"""
    if pts.dim() != 3 or pts.shape[-1] != 3:
        raise ValueError(f"expected (B,N,3), got {tuple(pts.shape)}")
    B = int(pts.shape[0])
    if B < 1:
        return pts
    device = pts.device
    dtype = pts.dtype
    ap = float(np.clip(apply_prob, 0.0, 1.0))
    apply = torch.rand(B, device=device, dtype=torch.float32, generator=generator) <= ap
    if not bool(apply.any().item()):
        return pts
    p = pts.clone()
    deg2rad = math.pi / 180.0
    th = (torch.rand(B, device=device, dtype=torch.float32, generator=generator) * 2.0 - 1.0) * (
        float(rot_deg_max) * deg2rad
    )
    c = torch.cos(th).unsqueeze(1)
    s = torch.sin(th).unsqueeze(1)
    x = p[..., 0]
    y = p[..., 1]
    xn = c * x - s * y
    yn = s * x + c * y
    p = torch.stack([xn, yn, p[..., 2]], dim=-1)
    u = torch.rand(B, 1, 1, device=device, dtype=dtype, generator=generator)
    sc = u * (float(scale_max) - float(scale_min)) + float(scale_min)
    p = p * sc
    if float(noise_std) > 1e-9:
        noise = torch.randn(
            B, p.shape[1], 3, device=device, dtype=dtype, generator=generator
        )
        p = p + float(noise_std) * noise
    m = apply.to(device=device, dtype=dtype).unsqueeze(1).unsqueeze(2)
    return torch.where(m > 0.5, p, pts)


def _worker_init_fn(worker_id: int) -> None:
    """多进程 DataLoader 下各 worker 独立随机种子。"""
    np.random.seed(int(torch.initial_seed() % 2**32) + worker_id * 9973)
    random.seed(int(torch.initial_seed() % 2**32) + worker_id * 7919)


def build_seq_clouds_cache(
    *,
    data_roots: list[Path],
    gt_root: Path,
    frames_per_seq: int,
    th_percent: float,
    mesh_z_scale: float,
    max_build_points: int,
    ccfg: core.TrackConfig,
    seed: int,
) -> dict[str, dict]:
    """加载所有数据根下的序列：唯一键为「数据集文件夹名__captured_photos*」；ref + 每帧归一化点云。"""
    seq_clouds: dict[str, dict] = {}

    for data_root in data_roots:
        if not data_root.is_dir():
            print(f"[跳过] 数据根不存在: {data_root}", flush=True)
            continue
        seqs = _sorted_seq_dirs(data_root)
        if not seqs:
            print(f"[跳过] 未找到 captured_photos*: {data_root}", flush=True)
            continue

        for seq_dir in seqs:
            seq_name = seq_dir.name
            name = f"{data_root.name}__{seq_name}"
            gt_dir = resolve_gt_directory(gt_root, data_root, seq_name)
            if gt_dir is None:
                print(f"[跳过] 无 GT summary: {gt_root / data_root.name / seq_name} 或 {gt_root / seq_name}", flush=True)
                continue
            try:
                gt_list = cct._load_gt_from_mask_dir(seq_dir, gt_dir)
            except Exception as e:
                print(f"[跳过] 加载 GT 失败 {name}: {e}", flush=True)
                continue
            if len(gt_list) < 1:
                continue

            gt_list = sorted(gt_list, key=lambda g: int(g.frame_idx))
            ref_gt = gt_list[0]
            ref_pts = framegt_to_uvh_points(
                ref_gt,
                th_percent=float(th_percent),
                mesh_z_scale=float(mesh_z_scale),
                max_points=int(max_build_points),
                ccfg=ccfg,
            )
            ref_pts = normalize_point_cloud(ref_pts)
            if ref_pts.shape[0] < 32:
                print(f"[跳过] 首帧点过少: {name} n={ref_pts.shape[0]}", flush=True)
                continue

            n_tot = len(gt_list)
            if int(frames_per_seq) <= 0 or int(frames_per_seq) >= n_tot:
                idxs = list(range(n_tot))
            else:
                idxs = _pick_frame_indices(
                    n_tot, int(frames_per_seq), int(seed) + abs(hash(name)) % 1_000_000
                )

            frame_clouds: list[np.ndarray] = []
            for ii in idxs:
                g = gt_list[ii]
                q = framegt_to_uvh_points(
                    g,
                    th_percent=float(th_percent),
                    mesh_z_scale=float(mesh_z_scale),
                    max_points=int(max_build_points),
                    ccfg=ccfg,
                )
                q = normalize_point_cloud(q)
                if q.shape[0] >= 32:
                    frame_clouds.append(q)

            if not frame_clouds:
                print(f"[跳过] 无有效帧点云: {name}", flush=True)
                continue

            seq_clouds[name] = {"ref": ref_pts, "frames": frame_clouds}

    if not seq_clouds:
        raise RuntimeError("没有可用序列（检查路径与 GT）")
    return seq_clouds


class ChessMeshSimilarityDataset(Dataset):
    """
    动态模式（默认）：__len__ = steps_per_epoch，每次 __getitem__ 随机成对，epoch 间 set_epoch 换种子。
    静态模式（--prebuild-samples）：预生成固定列表，行为接近旧版。

    采样语义：正样本为同一归一化点云经 SE(3)（+ 可选遮挡）得到 ref/query；负样本为异序列配对，标签 <0.1。
    """

    def __init__(
        self,
        *,
        seq_clouds: dict[str, dict] | None = None,
        data_roots: list[Path] | None = None,
        gt_root: Path | None = None,
        frames_per_seq: int = 0,
        n_points: int = 1024,
        th_percent: float = 68.0,
        mesh_z_scale: float = 30.0,
        max_build_points: int = 12000,
        occlusion_levels: list[float] | None = None,
        neg_max_sim: float = 0.1,
        neg_prob: float = 0.38,
        label_is_occlusion: bool = False,
        seed: int = 42,
        dynamic: bool = True,
        steps_per_epoch: int = 8192,
        prebuild_multiplier: int = 48,
        geom_rot_deg_max: float = 12.0,
        geom_scale_min: float = 0.88,
        geom_scale_max: float = 1.12,
        geom_noise_std: float = 0.012,
        geom_aug_prob: float = 0.9,
        random_ref_prob: float = 0.28,
        posit_rt_trans_max: float = 0.35,
        posit_occ_prob: float = 0.58,
        posit_so3_prob: float = 0.0,
        allowed_names: list[str] | None = None,
        enable_geom_aug: bool = True,
        split_tag: str = "",
        rng_offset: int = 0,
    ):
        super().__init__()
        if occlusion_levels is None:
            occlusion_levels = [0.0, 0.1, 0.2, 0.3]
        self.n_points = int(n_points)
        self.max_build_points = int(max_build_points)
        self.neg_max_sim = float(neg_max_sim)
        self.neg_prob = float(neg_prob)
        self.label_is_occlusion = bool(label_is_occlusion)
        self.occlusion_levels = [float(x) for x in occlusion_levels]
        self.rng_seed = int(seed)
        self.dynamic = bool(dynamic)
        self.steps_per_epoch = max(32, int(steps_per_epoch))
        self.prebuild_multiplier = max(1, int(prebuild_multiplier))
        self.geom_rot_deg_max = float(geom_rot_deg_max)
        self.geom_scale_min = float(geom_scale_min)
        self.geom_scale_max = float(geom_scale_max)
        self.geom_noise_std = float(geom_noise_std)
        self.geom_aug_prob = float(geom_aug_prob)
        self.random_ref_prob = float(np.clip(random_ref_prob, 0.0, 1.0))
        self.posit_rt_trans_max = float(max(0.0, posit_rt_trans_max))
        self.posit_occ_prob = float(np.clip(posit_occ_prob, 0.0, 1.0))
        self.posit_so3_prob = float(np.clip(posit_so3_prob, 0.0, 1.0))
        self.enable_geom_aug = bool(enable_geom_aug)
        self.rng_offset = int(rng_offset)
        self._epoch = 0
        self._split_tag = str(split_tag).strip()

        if seq_clouds is not None:
            base = dict(seq_clouds)
        else:
            if not data_roots or gt_root is None:
                raise ValueError("需提供 seq_clouds，或同时提供 data_roots 与 gt_root")
            ccfg = core.TrackConfig()
            ccfg.proc_scale = 1.0
            ccfg.verbose = 0
            base = build_seq_clouds_cache(
                data_roots=list(data_roots),
                gt_root=gt_root,
                frames_per_seq=int(frames_per_seq),
                th_percent=float(th_percent),
                mesh_z_scale=float(mesh_z_scale),
                max_build_points=int(max_build_points),
                ccfg=ccfg,
                seed=int(seed),
            )

        if allowed_names is not None:
            self.seq_clouds = {k: base[k] for k in allowed_names if k in base}
            missing = [k for k in allowed_names if k not in base]
            if missing:
                raise KeyError(f"allowed_names 中有未知序列: {missing[:5]}...")
        else:
            self.seq_clouds = base

        self.names = list(self.seq_clouds.keys())
        if not self.names:
            tag = f"[{self._split_tag}] " if self._split_tag else ""
            raise RuntimeError(f"{tag}划分后无可用序列")

        n_fr = sum(len(self.seq_clouds[n]["frames"]) for n in self.names)

        self.samples: list[dict] = []
        if not self.dynamic:
            rng = np.random.default_rng(seed)
            n_pairs = max(
                512,
                n_fr * self.prebuild_multiplier,
                len(self.names) * 64,
            )
            for _ in range(n_pairs):
                rng_loc = np.random.default_rng(rng.integers(0, 2**31 - 1))
                self.samples.append(self._sample_pair_np(rng_loc))

        tag = f"[{self._split_tag}] " if self._split_tag else ""
        aug_s = "增广开" if self.enable_geom_aug else "增广关"
        extra_s = (
            f" 正对=SE(3)+平移±{self.posit_rt_trans_max:.2f} occ_p={self.posit_occ_prob:.2f}"
            + (f" +extra_SO3={self.posit_so3_prob:.2f}" if self.posit_so3_prob > 1e-6 else "")
        )
        mode_s = (
            f"动态 每epoch迭代步数={self.steps_per_epoch}"
            if self.dynamic
            else f"静态 预生成样本对={len(self.samples)}"
        )
        print(
            f"{tag}[数据集] 序列数={len(self.names)} 帧点云数={n_fr} | "
            f"每条样本: ref点={self.n_points} query点={self.n_points} (每对共{2 * self.n_points}点) | "
            f"ROI建网格缓存≤{self.max_build_points}点/帧 | "
            f"{aug_s}{extra_s} | {mode_s}",
            flush=True,
        )

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def _rng_for_index(self, idx: int) -> np.random.Generator:
        wi = torch.utils.data.get_worker_info()
        wid = int(wi.id) if wi is not None else 0
        return np.random.default_rng(
            self.rng_seed
            + self.rng_offset
            + self._epoch * 1_000_003
            + int(idx) * 1_000_033
            + wid * 17_389
        )

    def _aug(self, pts: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.enable_geom_aug:
            return np.asarray(pts, dtype=np.float32).copy()
        return geom_augment_normalized(
            pts,
            rng,
            rot_deg_max=self.geom_rot_deg_max,
            scale_min=self.geom_scale_min,
            scale_max=self.geom_scale_max,
            noise_std=self.geom_noise_std,
            apply_prob=self.geom_aug_prob,
        )

    def _sample_pair_np(self, rng: np.random.Generator) -> dict:
        name = str(rng.choice(self.names))
        ref = self.seq_clouds[name]["ref"]
        frames = self.seq_clouds[name]["frames"]

        if rng.random() < self.neg_prob and len(self.names) > 1:
            n2 = str(rng.choice([n for n in self.names if n != name]))
            pool = self.seq_clouds[n2]["frames"]
            base_neg = pool[int(rng.integers(0, len(pool)))]
            q = resample_points(base_neg, self.n_points, rng)
            r = resample_points(ref, self.n_points, rng)
            r = self._aug(r, rng)
            q = self._aug(q, rng)
            neg_hi = float(max(1e-6, min(self.neg_max_sim, 0.1)))
            lab = float(rng.uniform(0.0, neg_hi))
        else:
            # 同一底层 mesh：单帧归一化点云 base；ref=重采样(base)，query=SE(3)(base) 后可选遮挡再重采样
            if self.random_ref_prob > 0.0 and rng.random() < self.random_ref_prob:
                base = frames[int(rng.integers(0, len(frames)))]
            else:
                base = ref

            Rm = random_rotation_so3_numpy(rng)
            tm = rng.uniform(
                -self.posit_rt_trans_max,
                self.posit_rt_trans_max,
                size=3,
            ).astype(np.float32)
            transformed = apply_se3_points_rows(base, Rm, tm)

            occ = 0.0
            if self.posit_occ_prob > 1e-9 and rng.random() < self.posit_occ_prob:
                occ = float(rng.choice(self.occlusion_levels))
                transformed = random_occlude_points(transformed, occ, rng)

            r = resample_points(base, self.n_points, rng)
            q = resample_points(transformed, self.n_points, rng)
            r = self._aug(r, rng)
            q = self._aug(q, rng)
            if self.posit_so3_prob > 1e-9 and rng.random() < self.posit_so3_prob:
                R2 = random_rotation_so3_numpy(rng)
                q = apply_so3_points_rows(q, R2)

            if self.label_is_occlusion:
                lab = float(occ)
            else:
                lab = float(max(0.0, min(1.0, 1.0 - occ)))

        return {
            "ref": r.astype(np.float32),
            "query": q.astype(np.float32),
            "label": float(lab),
        }

    def __len__(self) -> int:
        if self.dynamic:
            return self.steps_per_epoch
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if self.dynamic:
            rng = self._rng_for_index(idx)
            s = self._sample_pair_np(rng)
        else:
            s = self.samples[int(idx) % len(self.samples)]
        return {
            "ref": torch.from_numpy(s["ref"]),
            "query": torch.from_numpy(s["query"]),
            "label": torch.tensor(s["label"], dtype=torch.float32),
        }


def collate_fn(batch: list[dict]) -> dict:
    ref = torch.stack([b["ref"] for b in batch], dim=0)
    query = torch.stack([b["query"] for b in batch], dim=0)
    label = torch.stack([b["label"] for b in batch], dim=0)
    return {"ref": ref, "query": query, "label": label}


def save_mesh_similarity_pair_viz_images(
    ds: ChessMeshSimilarityDataset,
    out_dir: Path,
    *,
    count: int,
    dpi: int,
    split_tag: str,
    start_idx: int = 0,
) -> int:
    """
    将数据集 ref/query 点云（uvh）画成 3D 散点图并保存 PNG。
    返回成功写入的文件数。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — 注册 3d 投影
    except ImportError as e:
        print(f"[跳过] 保存训练对可视化需要 matplotlib: {e}", flush=True)
        return 0

    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    n_save = max(0, int(count))
    if n_save <= 0:
        return 0

    ds_len = len(ds)
    if n_save > ds_len:
        print(
            f"[可视化][{split_tag}] 请求保存 {n_save} 张但 len(dataset)={ds_len}，将只保存 {ds_len} 张",
            flush=True,
        )

    ds.set_epoch(0)
    saved = 0
    first_fp: Path | None = None
    tag = str(split_tag).strip() or "data"
    for k in range(n_save):
        idx = int(start_idx) + k
        if idx < 0 or idx >= ds_len:
            break
        try:
            item = ds[idx]
            ref = item["ref"].detach().cpu().numpy()
            qry = item["query"].detach().cpu().numpy()
            lab = float(item["label"].detach().cpu().item())
        except Exception as e:
            print(f"[可视化][{tag}] idx={idx} 取样本失败: {e}", flush=True)
            continue

        if ref.size == 0 or qry.size == 0 or ref.shape[-1] != 3 or qry.shape[-1] != 3:
            print(f"[可视化][{tag}] idx={idx} 跳过：空点云或维度异常 ref{ref.shape} qry{qry.shape}", flush=True)
            continue

        try:
            fig, axes = plt.subplots(
                1,
                2,
                figsize=(11.0, 5.2),
                subplot_kw={"projection": "3d"},
            )
            ax0, ax1 = axes[0], axes[1]
            ax0.scatter(ref[:, 0], ref[:, 1], ref[:, 2], s=2, c=ref[:, 2], cmap="viridis", alpha=0.55)
            ax0.set_title(f"ref ({tag})")
            ax0.set_xlabel("u")
            ax0.set_ylabel("v")
            ax0.set_zlabel("h")

            ax1.scatter(qry[:, 0], qry[:, 1], qry[:, 2], s=2, c=qry[:, 2], cmap="plasma", alpha=0.55)
            ax1.set_title(f"query  label={lab:.4f}")
            ax1.set_xlabel("u")
            ax1.set_ylabel("v")
            ax1.set_zlabel("h")

            for ax in (ax0, ax1):
                ax.set_box_aspect((1, 1, 1))

            fig.suptitle(f"{tag}  pair idx={idx}  n_pts={ref.shape[0]}", fontsize=11)
            try:
                fig.tight_layout()
            except Exception:
                fig.subplots_adjust(top=0.88)

            lab_fn = f"{lab:.4f}".replace(".", "p")
            fp = out_dir / f"{tag}_idx{idx:06d}_lab{lab_fn}.png"
            fig.savefig(fp, dpi=int(dpi), bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[可视化][{tag}] idx={idx} 绘图/保存失败: {e}", flush=True)
            try:
                plt.close("all")
            except Exception:
                pass
            continue

        if fp.is_file() and fp.stat().st_size > 0:
            saved += 1
            if first_fp is None:
                first_fp = fp
        else:
            print(f"[可视化][{tag}] idx={idx} 文件未生成或为空: {fp}", flush=True)

    abs_msg = f"[可视化][{tag}] 已写入 {saved} 张 → 目录绝对路径:\n  {out_dir}"
    if first_fp is not None:
        abs_msg += f"\n  示例文件: {first_fp}"
    print(abs_msg, flush=True)
    if saved == 0 and n_save > 0:
        print(
            f"[可视化][{tag}] 未生成任何 PNG：请确认已 `pip install matplotlib`，"
            f"且数据集 __getitem__ 正常（len={ds_len}）",
            flush=True,
        )
    return saved


def _resolve_online_resume_path(args: argparse.Namespace, out_dir: Path, round_idx: int) -> str:
    explicit = str(args.resume or "").strip()
    best = out_dir / "mesh_similarity_best.pt"
    if round_idx > 0 and best.is_file():
        return str(best.resolve())
    return explicit


def run_mesh_similarity_training(args: argparse.Namespace, *, round_idx: int = 0) -> None:
    """单次训练：扫盘建缓存、划分、训练、保存。在线模式下每轮重新执行本函数。"""
    if round_idx > 0:
        print(f"\n[在线] 第 {round_idx + 1} 轮：重新扫描磁盘并加载 best 权重续训\n", flush=True)

    data_root = Path(args.data_root).resolve()
    extra_roots = [Path(p).resolve() for p in (args.extra_data_root or []) if str(p).strip()]
    data_roots = [data_root] + [p for p in extra_roots if p != data_root]
    gt_root = Path(args.gt_root)
    out_dir = _MESH_SIM_OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[输出] 权重与配置目录（代码常量 _MESH_SIM_OUT_DIR）: {out_dir}", flush=True)

    occ_levels = [float(x.strip()) for x in str(args.occlusion_levels).split(",") if x.strip()]
    if not occ_levels:
        occ_levels = [0.0, 0.1, 0.2, 0.3]
    
    ccfg_build = core.TrackConfig()
    ccfg_build.proc_scale = 1.0
    ccfg_build.verbose = 0
    full_seq_clouds = build_seq_clouds_cache(
        data_roots=data_roots,
        gt_root=gt_root,
        frames_per_seq=int(args.frames_per_seq),
        th_percent=float(args.th_percent),
        mesh_z_scale=float(args.mesh_z_scale),
        max_build_points=int(args.max_build_points),
        ccfg=ccfg_build,
        seed=int(args.seed),
    )
    all_seq_names = sorted(full_seq_clouds.keys())
    train_names, val_names = split_seq_names_for_val(
        all_seq_names, float(args.val_fraction), int(args.seed)
    )
    n_tr_fr = sum(len(full_seq_clouds[k]["frames"]) for k in train_names)
    n_va_fr = sum(len(full_seq_clouds[k]["frames"]) for k in val_names) if val_names else 0
    n_all_seq = len(all_seq_names)
    n_tr_seq = len(train_names)
    n_va_seq = len(val_names)
    print(
        f"[划分] 全库序列={n_all_seq} | 训练序列={n_tr_seq} 验证序列={n_va_seq} | "
        f"训练帧点云={n_tr_fr} 验证帧点云={n_va_fr} | "
        f"帧点云合计={n_tr_fr + n_va_fr}",
        flush=True,
    )
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    geom_on_gpu = bool(args.geom_on_gpu) and device.type == "cuda"
    if bool(args.geom_on_gpu) and device.type != "cuda":
        print("[警告] --geom-on-gpu 需要 CUDA，已回退为 CPU 几何增广", flush=True)

    ds_common = dict(
        seq_clouds=full_seq_clouds,
        n_points=int(args.n_points),
        max_build_points=int(args.max_build_points),
        occlusion_levels=occ_levels,
        neg_max_sim=float(args.neg_max_sim),
        neg_prob=float(args.neg_prob),
        label_is_occlusion=bool(args.label_is_occlusion),
        seed=int(args.seed),
        dynamic=not bool(args.prebuild_samples),
        prebuild_multiplier=int(args.prebuild_multiplier),
        geom_rot_deg_max=float(args.geom_rot_deg_max),
        geom_scale_min=float(args.geom_scale_min),
        geom_scale_max=float(args.geom_scale_max),
        geom_noise_std=float(args.geom_noise_std),
        geom_aug_prob=float(args.geom_aug_prob),
        random_ref_prob=float(args.random_ref_prob),
        posit_rt_trans_max=float(args.posit_rt_trans_max),
        posit_occ_prob=float(args.posit_occ_prob),
        posit_so3_prob=float(args.posit_so3_prob),
    )
    
    ds_train = ChessMeshSimilarityDataset(
        **ds_common,
        allowed_names=train_names,
        steps_per_epoch=int(args.steps_per_epoch),
        enable_geom_aug=not geom_on_gpu,
        split_tag="train",
        rng_offset=0,
    )
    
    ds_val: ChessMeshSimilarityDataset | None = None
    if val_names:
        ds_val = ChessMeshSimilarityDataset(
            **ds_common,
            allowed_names=val_names,
            steps_per_epoch=max(32, int(args.val_steps)),
            enable_geom_aug=False,
            split_tag="val",
            rng_offset=123_457,
        )

    bs = max(1, int(args.batch_size))
    n_pts = int(args.n_points)

    def _loader_epoch_stats(ds: ChessMeshSimilarityDataset) -> tuple[int, int, int]:
        """返回 (dataset长度即步数, 每epoch batch数 drop_last, 每epoch 样本对数)。"""
        n_step = len(ds)
        n_batch = n_step // bs
        n_pairs = n_batch * bs
        return n_step, n_batch, n_pairs

    tr_step, tr_b, tr_pairs = _loader_epoch_stats(ds_train)
    if ds_val is not None:
        va_step, va_b, va_pairs = _loader_epoch_stats(ds_val)
        print(
            f"[数据集规模] 训练集 DataLoader: len={tr_step} batch={bs} → 每epoch {tr_b} 个batch "
            f"≈ {tr_pairs} 对/epoch | 每对点云 {2 * n_pts} 点 (ref+query 各{n_pts})",
            flush=True,
        )
        print(
            f"[数据集规模] 验证集 DataLoader: len={va_step} batch={bs} → 每epoch {va_b} batch "
            f"≈ {va_pairs} 对/epoch | 每对点云 {2 * n_pts} 点",
            flush=True,
        )
    else:
        print(
            f"[数据集规模] 训练集 DataLoader: len={tr_step} batch={bs} → 每epoch {tr_b} batch "
            f"≈ {tr_pairs} 对/epoch | 每对点云 {2 * n_pts} 点 | 无验证集",
            flush=True,
        )

    use_amp = device.type == "cuda" and not bool(args.no_amp)
    if use_amp and bool(args.amp_bf16) and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    elif use_amp:
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    
    pin_mem = device.type == "cuda"
    nb = bool(pin_mem)
    nw = int(args.num_workers)
    prefetch = max(2, int(args.prefetch_factor)) if nw > 0 else None
    
    def _make_loader(ds: ChessMeshSimilarityDataset, *, shuffle: bool) -> DataLoader:
        kw: dict = dict(
            batch_size=int(args.batch_size),
            shuffle=shuffle,
            num_workers=nw,
            collate_fn=collate_fn,
            drop_last=True,
            pin_memory=pin_mem,
            worker_init_fn=_worker_init_fn if nw > 0 else None,
            persistent_workers=bool(nw > 0),
        )
        if prefetch is not None:
            kw["prefetch_factor"] = prefetch
        return DataLoader(ds, **kw)
    
    loader = _make_loader(ds_train, shuffle=True)
    val_loader = _make_loader(ds_val, shuffle=False) if ds_val is not None else None

    pair_viz_root_resolved: str | None = None
    if _PAIR_VIZ_ENABLED:
        vroot = out_dir / _PAIR_VIZ_SUBDIR
        if int(round_idx) > 0:
            vroot = vroot / f"round_{int(round_idx):04d}"
        pair_viz_root_resolved = str(vroot.resolve())
        nc = max(1, int(_PAIR_VIZ_COUNT))
        dpi_v = max(48, int(_PAIR_VIZ_DPI))
        print(
            f"[可视化] 开始导出配对图（matplotlib）根目录: {pair_viz_root_resolved}  "
            f"每划分各 {nc} 张",
            flush=True,
        )
        save_mesh_similarity_pair_viz_images(
            ds_train, vroot / "train", count=nc, dpi=dpi_v, split_tag="train"
        )
        if ds_val is not None:
            save_mesh_similarity_pair_viz_images(
                ds_val, vroot / "val", count=nc, dpi=dpi_v, split_tag="val"
            )

    print(
        f"[加速] device={device} amp={use_amp} amp_dtype={amp_dtype} "
        f"workers={nw} pin_memory={pin_mem} prefetch={prefetch} geom_on_gpu={geom_on_gpu}",
        flush=True,
    )
    if device.type == "cuda" and hasattr(torch.cuda, "mem_get_info"):
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            print(
                f"[显存] 当前设备空闲≈{free_b / 2**30:.2f} GiB / 总计≈{total_b / 2**30:.2f} GiB — "
                f"若空闲较多可增大 --batch-size（或略减 --num-workers）以拉高 GPU 利用率",
                flush=True,
            )
        except Exception:
            pass
    
    l2_embed = bool(args.l2_embed)
    model = MeshSimilarityNet(
        in_dim=3,
        embed_dim=int(args.embed_dim),
        head_hidden=int(args.head_hidden),
        encoder_width=int(args.encoder_width),
        l2_normalize_embed=l2_embed,
        head_dropout=float(args.head_dropout),
    ).to(device)
    resume_path = _resolve_online_resume_path(args, out_dir, round_idx)
    if resume_path:
        try:
            ck = torch.load(resume_path, map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"], strict=True)
        print(f"[恢复] 已加载: {resume_path}", flush=True)
    if bool(args.torch_compile) and device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")  # type: ignore[assignment]
            print("[加速] torch.compile(mode=reduce-overhead) 已启用（首轮可能较慢）", flush=True)
        except Exception as e:
            print(f"[加速] torch.compile 未启用: {e}", flush=True)
    lr_use = float(args.lr)
    if resume_path and float(args.finetune_lr) > 0:
        lr_use = float(args.finetune_lr)
    elif resume_path:
        lr_use = float(args.lr) * 0.25
        print(f"[恢复] 使用较小 lr={lr_use}（可指定 --finetune-lr）", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_use, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(args.epochs)))
    loss_fn = nn.SmoothL1Loss(beta=0.05)
    scaler: torch.amp.GradScaler | None
    if device.type == "cuda" and use_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None
    
    def _amp_autocast():
        if device.type == "cuda" and use_amp:
            return torch.amp.autocast("cuda", dtype=amp_dtype)
        return contextlib.nullcontext()

    early_patience = max(0, int(getattr(args, "early_stop_patience", 0)))
    early_min_delta = float(max(0.0, getattr(args, "early_stop_min_delta", 1e-5)))
    
    cfg_dump = {
        "data_roots": [str(p) for p in data_roots],
        "data_root": str(data_root),
        "gt_root": str(gt_root),
        "frames_per_seq": int(args.frames_per_seq),
        "dynamic": not bool(args.prebuild_samples),
        "steps_per_epoch": int(args.steps_per_epoch),
        "val_fraction": float(args.val_fraction),
        "val_steps": int(args.val_steps),
        "train_seq_names": train_names,
        "val_seq_names": val_names,
        "prebuild_multiplier": int(args.prebuild_multiplier),
        "n_points": int(args.n_points),
        "th_percent": float(args.th_percent),
        "mesh_z_scale": float(args.mesh_z_scale),
        "occlusion_levels": occ_levels,
        "label_is_occlusion": bool(args.label_is_occlusion),
        "neg_max_sim": float(args.neg_max_sim),
        "neg_prob": float(args.neg_prob),
        "geom": {
            "rot_deg_max": float(args.geom_rot_deg_max),
            "scale": [float(args.geom_scale_min), float(args.geom_scale_max)],
            "noise_std": float(args.geom_noise_std),
            "aug_prob": float(args.geom_aug_prob),
        },
        "random_ref_prob": float(args.random_ref_prob),
        "posit_rt_trans_max": float(args.posit_rt_trans_max),
        "posit_occ_prob": float(args.posit_occ_prob),
        "posit_so3_prob": float(args.posit_so3_prob),
        "model": {
            "embed_dim": int(args.embed_dim),
            "encoder_width": int(args.encoder_width),
            "head_hidden": int(args.head_hidden),
            "head_dropout": float(args.head_dropout),
            "l2_normalize_embed": l2_embed,
        },
        "resume": resume_path or None,
        "lr": lr_use,
        "speed": {
            "amp": use_amp,
            "amp_dtype": str(amp_dtype),
            "num_workers": nw,
            "prefetch_factor": prefetch,
            "pin_memory": pin_mem,
            "torch_compile_requested": bool(args.torch_compile),
            "geom_on_gpu": geom_on_gpu,
        },
        "online": {
            "loop_enabled": bool(getattr(args, "online_loop", False)),
            "sleep_sec": float(getattr(args, "online_sleep_sec", 0.0)),
            "session_round": int(round_idx),
        },
        "pair_viz": {
            "root": pair_viz_root_resolved,
            "count": int(_PAIR_VIZ_COUNT) if pair_viz_root_resolved else 0,
            "dpi": int(_PAIR_VIZ_DPI) if pair_viz_root_resolved else 0,
            "enabled": bool(_PAIR_VIZ_ENABLED),
        },
        "early_stop": {
            "patience": early_patience,
            "min_delta": early_min_delta,
        },
        "note": "序列级 train/val；--online-loop 扫盘续训；--extra-data-root 合并多盒数据",
    }
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, ensure_ascii=False, indent=2)
    
    best_metric = float("inf")
    best_tag = "train" if val_loader is None else "val"
    stagn_epochs = 0
    for epoch in range(int(args.epochs)):
        ds_train.set_epoch(epoch)
        train_geom_gen: torch.Generator | None = None
        if geom_on_gpu:
            train_geom_gen = torch.Generator(device=device)
            train_geom_gen.manual_seed(int(args.seed) + epoch * 1_009_661 + 17)
        model.train()
        losses = []
        for batch in loader:
            ref = batch["ref"].to(device, non_blocking=nb)
            q = batch["query"].to(device, non_blocking=nb)
            y = batch["label"].to(device, non_blocking=nb)
            if geom_on_gpu and train_geom_gen is not None:
                gkw = dict(
                    rot_deg_max=float(args.geom_rot_deg_max),
                    scale_min=float(args.geom_scale_min),
                    scale_max=float(args.geom_scale_max),
                    noise_std=float(args.geom_noise_std),
                    apply_prob=float(args.geom_aug_prob),
                    generator=train_geom_gen,
                )
                ref = geom_augment_batch_torch(ref, **gkw)
                q = geom_augment_batch_torch(q, **gkw)
            opt.zero_grad(set_to_none=True)
            with _amp_autocast():
                pred = model(ref, q)
                loss = loss_fn(pred, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
            losses.append(float(loss.item()))
        sched.step()
        mean_ls = float(np.mean(losses)) if losses else 0.0
    
        mean_va = float("nan")
        if val_loader is not None and ds_val is not None:
            ds_val.set_epoch(epoch)
            model.eval()
            vlosses: list[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    ref = batch["ref"].to(device, non_blocking=nb)
                    q = batch["query"].to(device, non_blocking=nb)
                    y = batch["label"].to(device, non_blocking=nb)
                    with _amp_autocast():
                        pred = model(ref, q)
                        vlosses.append(float(loss_fn(pred, y).item()))
            mean_va = float(np.mean(vlosses)) if vlosses else float("nan")
    
        lr_now = sched.get_last_lr()[0]
        if val_loader is not None and not math.isnan(mean_va):
            print(
                f"epoch {epoch+1}/{args.epochs}  train_loss={mean_ls:.5f}  val_loss={mean_va:.5f}  lr={lr_now:.6f}",
                flush=True,
            )
            metric_for_best = mean_va
        else:
            print(
                f"epoch {epoch+1}/{args.epochs}  train_loss={mean_ls:.5f}  lr={lr_now:.6f}",
                flush=True,
            )
            metric_for_best = mean_ls
    
        old_best = best_metric
        if metric_for_best < best_metric:
            best_metric = metric_for_best
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "config": cfg_dump,
                "best_metric": best_metric,
                "best_metric_on": best_tag,
            }
            torch.save(ckpt, out_dir / "mesh_similarity_best.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "config": cfg_dump,
                "train_loss": mean_ls,
                "val_loss": mean_va,
            },
            out_dir / "mesh_similarity_last.pt",
        )

        if early_patience > 0:
            improved = not math.isfinite(old_best) or (metric_for_best < old_best - early_min_delta)
            if improved:
                stagn_epochs = 0
            else:
                stagn_epochs += 1
            if stagn_epochs >= early_patience:
                metric_name = (
                    "val_loss"
                    if val_loader is not None and not math.isnan(mean_va)
                    else "train_loss"
                )
                print(
                    f"[早停] 连续 {early_patience} 个 epoch 的 {metric_name} "
                    f"相对历史最优下降不足 {early_min_delta} "
                    f"（本轮={metric_for_best:.6f}，训练前历史最优={old_best:.6f}）",
                    flush=True,
                )
                break
    
    print(
        f"[完成] best_{best_tag}_loss≈{best_metric:.5f}  权重: {out_dir / 'mesh_similarity_best.pt'}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Chess GT 伪 3D 点云相似度网络训练")
    ap.add_argument(
        "--data-root",
        type=str,
        default=os.environ.get("CHESS_DATA_ROOT", r"E:\Surgical_Tracking_Datasets\Chess_Box_1_8"),
    )
    ap.add_argument(
        "--gt-root",
        type=str,
        default=os.environ.get("CHESS_GT_ROOT", r"E:\Surgical_Tracking_Datasets\Chess_Box_Mask"),
    )
    ap.add_argument(
        "--extra-data-root",
        action="append",
        default=[],
        metavar="PATH",
        help="附加图像数据根（可多次），目录结构同 --data-root；合并后负样本与跨序列多样性增加",
    )
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--finetune-lr",
        type=float,
        default=0.0,
        help=">0 且 --resume 时用此学习率；0 表示与 --lr 相同",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--resume",
        type=str,
        default="",
        help="加载 mesh_similarity_*.pt 的 state_dict，用于增量/在线续训",
    )
    ap.add_argument(
        "--frames-per-seq",
        type=int,
        default=0,
        help="每序列采样帧数；0=使用 GT 中全部成功帧",
    )
    ap.add_argument(
        "--steps-per-epoch",
        type=int,
        default=20480,
        help="训练集动态采样：每 epoch 迭代步数（越大单轮采样越多）",
    )
    ap.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="验证集占比（按序列个数划分）；0=不划分，全部用于训练",
    )
    ap.add_argument(
        "--val-steps",
        type=int,
        default=6144,
        help="每 epoch 验证集动态采样步数（仅 val-fraction>0 且序列>=2 时生效）",
    )
    ap.add_argument(
        "--prebuild-samples",
        action="store_true",
        help="关闭动态采样，改为预生成固定列表（省 CPU 随机，多样性较低）",
    )
    ap.add_argument(
        "--prebuild-multiplier",
        type=int,
        default=48,
        help="静态模式：预生成对数 ≈ 总帧数×此系数",
    )
    ap.add_argument("--n-points", type=int, default=1024)
    ap.add_argument("--th-percent", type=float, default=68.0)
    ap.add_argument("--mesh-z-scale", type=float, default=30.0)
    ap.add_argument("--max-build-points", type=int, default=12000)
    ap.add_argument(
        "--neg-max-sim",
        type=float,
        default=0.1,
        help="负样本标签 ~ U(0, min(此值,0.1))，保证异 mesh 相似度 <0.1",
    )
    ap.add_argument("--neg-prob", type=float, default=0.38, help="每条合成样本中采负例的概率")
    ap.add_argument(
        "--occlusion-levels",
        type=str,
        default="0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.4",
        help="随机遮挡比例列表（对查询点云丢点）",
    )
    ap.add_argument(
        "--label-is-occlusion",
        action="store_true",
        help="若设：标签=遮挡率（0.1→0.1）；默认：标签=1-遮挡率（0.1→0.9）",
    )
    ap.add_argument("--geom-rot-deg-max", type=float, default=12.0, help="uv 平面最大旋转角")
    ap.add_argument("--geom-scale-min", type=float, default=0.88)
    ap.add_argument("--geom-scale-max", type=float, default=1.12)
    ap.add_argument("--geom-noise-std", type=float, default=0.012)
    ap.add_argument("--geom-aug-prob", type=float, default=0.9)
    ap.add_argument(
        "--random-ref-prob",
        type=float,
        default=0.28,
        help="正样本底层 mesh 点云：从同序列随机帧取 base 的概率，否则用首帧缓存 ref",
    )
    ap.add_argument(
        "--posit-rt-trans-max",
        type=float,
        default=0.35,
        help="正对分支：归一化坐标下随机平移分量绝对值上界（与 Haar SO(3) 组成 SE(3)）",
    )
    ap.add_argument(
        "--posit-occ-prob",
        type=float,
        default=0.58,
        help="正对分支：对 SE(3) 后的 query 再随机丢点（遮挡）的概率；标签默认 1−遮挡率",
    )
    ap.add_argument(
        "--posit-so3-prob",
        type=float,
        default=0.0,
        help=(
            "正对分支：几何增广后再对 query 额外施加 Haar SO(3) 的概率（SE(3) 已含旋转，通常保持 0）"
        ),
    )
    ap.add_argument("--embed-dim", type=int, default=256, help="PointNet 全局嵌入维度（改后需重训，与旧 ckpt 不兼容）")
    ap.add_argument("--encoder-width", type=int, default=128, help="Conv1d 中间通道宽")
    ap.add_argument("--head-hidden", type=int, default=256, help="相似度 MLP 隐藏层宽度")
    ap.add_argument(
        "--head-dropout",
        type=float,
        default=0.0,
        help="进入 MLP 前对拼接特征做 dropout（如 0.08~0.15，抑过拟合；不改 state_dict 键名）",
    )
    ap.add_argument(
        "--l2-embed",
        action="store_true",
        help="编码向量 L2 归一化后再进相似度头（新训练推荐；与未开此选项的旧权重混用时需行为一致）",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader 进程数；Windows 若死锁/报错可试 0",
    )
    ap.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="num_workers>0 时每 worker 预取批次数",
    )
    ap.add_argument(
        "--no-amp",
        action="store_true",
        help="关闭 CUDA 混合精度（默认在 GPU 上开启 AMP 以加速）",
    )
    ap.add_argument(
        "--amp-bf16",
        action="store_true",
        help="AMP 使用 bfloat16（需 GPU 支持；否则回退 float16）",
    )
    ap.add_argument(
        "--torch-compile",
        action="store_true",
        help="尝试 torch.compile(model)（PyTorch2+；Windows 上可能不支持或首次极慢）",
    )
    ap.add_argument(
        "--geom-on-gpu",
        action="store_true",
        help="训练时几何增广在 GPU 上做（CPU 只做重采样/遮挡）；需 CUDA，可明显降 CPU 占用",
    )
    ap.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help=">0：有 val 时监控 val_loss，否则 train_loss；连续 N 个 epoch 较历史最优改进 < min-delta 则提前结束；0=关闭",
    )
    ap.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=1e-5,
        help="早停：相对上一轮「历史最优」至少下降这么多才重置耐心计数",
    )
    ap.add_argument(
        "--online-loop",
        action="store_true",
        help="在线训练：每轮训完后休眠，再扫盘并从 _MESH_SIM_OUT_DIR/mesh_similarity_best.pt 续训",
    )
    ap.add_argument(
        "--online-sleep-sec",
        type=float,
        default=120.0,
        help="--online-loop 下每轮结束后的休眠秒数（等新数据写入后再扫盘）",
    )
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    _MESH_SIM_OUT_DIR.resolve().mkdir(parents=True, exist_ok=True)
    if not bool(args.online_loop):
        run_mesh_similarity_training(args, round_idx=0)
        return
    r = 0
    while True:
        try:
            run_mesh_similarity_training(args, round_idx=r)
        except KeyboardInterrupt:
            print("[在线] 已中断并退出循环。", flush=True)
            break
        sec = float(args.online_sleep_sec)
        if sec > 0:
            print(f"[在线] 休眠 {sec:.0f}s… (Ctrl+C 退出)", flush=True)
            time.sleep(sec)
        r += 1


if __name__ == "__main__":
    main()
