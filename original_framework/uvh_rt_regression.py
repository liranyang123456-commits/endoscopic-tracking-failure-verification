from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import MIS_C_F_SIMI_Mesh_Match_Track_3DMatch as mis3d

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

_COMPARE_CHESS_MOD: Any = None


def _get_compare_chessboard() -> Any:
    """延迟导入，避免仅加载 RenderField 时强依赖 MIS_Mesh_RTS_Track 模块名。"""
    global _COMPARE_CHESS_MOD
    if _COMPARE_CHESS_MOD is None:
        import compare_chessboard_trackers as compare

        _COMPARE_CHESS_MOD = compare
    return _COMPARE_CHESS_MOD


@dataclass
class SequenceCache:
    name: str
    split: str
    npz_path: str
    n_samples: int
    frames_dir: str
    gt_dir: str


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    name = str(device or "cuda").strip().lower()
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请求使用 CUDA，但当前环境不可用")
        return torch.device("cuda")
    if name in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _auto_num_workers(requested: int, device: torch.device, fallback: int) -> int:
    if int(requested) > 0:
        return int(requested)
    if sys.platform.startswith("win"):
        return 0
    if device.type == "cuda":
        return int(max(2, fallback))
    return 0


def _normalize_root_path(root: Path) -> Path:
    if root.exists():
        return root
    name = root.name
    parent = root.parent
    alt_names = []
    if name.startswith("captured_"):
        alt_names.append(name.replace("captured_", "capture_", 1))
    if name.startswith("capture_"):
        alt_names.append(name.replace("capture_", "captured_", 1))
    for alt in alt_names:
        cand = parent / alt
        if cand.exists():
            return cand
    return root


def _candidate_dirs(root: Path) -> list[Path]:
    dirs: list[Path] = []
    if root.exists() and root.is_dir():
        dirs.append(root)
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir():
                dirs.append(child)
    return dirs


def _resolve_image_dir(root: Path) -> Path | None:
    root = _normalize_root_path(root)
    best_dir = None
    best_n = 0
    for d in _candidate_dirs(root):
        try:
            n = len([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        except Exception:
            n = 0
        if n > best_n:
            best_n = n
            best_dir = d
    return best_dir if best_n > 0 else None


def _resolve_gt_mask_dir(root: Path) -> Path | None:
    root = _normalize_root_path(root)
    if root.exists() and root.is_dir() and (root / "summary.csv").exists() and (root / "mask").is_dir():
        return root
    return None


def _discover_dataset_names(image_root: Path, gt_root: Path) -> list[str]:
    image_names = {p.name for p in image_root.iterdir() if p.is_dir()}
    gt_names = {p.name for p in gt_root.iterdir() if p.is_dir()}
    return sorted(image_names & gt_names, key=lambda s: s.lower())


def _build_cfg(image_size: int) -> mis3d.core.TrackConfig:
    cfg = mis3d.core.TrackConfig()
    cfg.proc_scale = 0.5
    cfg.search_radius = 128
    cfg.search_step = 16
    cfg.topk_semantic = 8
    cfg.verbose = 0
    cfg.print_every = 1000000
    cfg.print_timing = False
    cfg.save_debug_images = False
    cfg.save_mesh = False
    cfg.resume_enable = False
    cfg.force_restart = True
    cfg.vis_window = False
    cfg.init_bbox = ""
    cfg.semantic_image_size = (int(image_size), int(image_size))
    return cfg


def _extract_mask_roi(mask_full: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = [int(v) for v in bbox]
    mk = np.asarray(mask_full, dtype=np.uint8)
    if mk.ndim != 2:
        return np.zeros((h, w), dtype=np.uint8)
    roi = mk[max(0, y):max(0, y) + max(0, h), max(0, x):max(0, x) + max(0, w)]
    if roi.shape != (h, w):
        out = np.zeros((h, w), dtype=np.uint8)
        hh = min(h, roi.shape[0])
        ww = min(w, roi.shape[1])
        out[:hh, :ww] = roi[:hh, :ww]
        roi = out
    return (roi > 0).astype(np.uint8) * 255


def _expand_bbox(bbox: tuple[int, int, int, int], margin_px: int, shape_hw: tuple[int, int]) -> tuple[int, int, int, int]:
    x, y, w, h = [int(v) for v in bbox]
    m = int(max(0, margin_px))
    H, W = int(shape_hw[0]), int(shape_hw[1])
    x0 = max(0, x - m)
    y0 = max(0, y - m)
    x1 = min(W, x + w + m)
    y1 = min(H, y + h + m)
    return mis3d.core.clamp_bbox(x0, y0, max(1, x1 - x0), max(1, y1 - y0), W, H)


def _parse_expand_values(text: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(text, (list, tuple)):
        vals = [int(v) for v in text if int(v) >= 0]
        return vals if vals else [32, 64]
    raw = str(text or "").strip()
    if not raw:
        return [32, 64]
    vals: list[int] = []
    for part in raw.replace(";", ",").split(","):
        part = str(part).strip()
        if not part:
            continue
        try:
            vals.append(max(0, int(part)))
        except Exception:
            pass
    vals = sorted(set(vals))
    return vals if vals else [32, 64]


def _select_reference_indices(n_frames: int, n_refs: int) -> list[int]:
    n = int(max(1, n_frames))
    k = int(max(1, min(n, n_refs)))
    if k >= n:
        return list(range(n))
    idx = np.linspace(0, n - 1, num=k, dtype=np.int32).tolist()
    out: list[int] = []
    seen: set[int] = set()
    for v in idx:
        i = int(v)
        if i not in seen:
            out.append(i)
            seen.add(i)
    return out


def _normalize_depth(depth: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    dep = np.asarray(depth, dtype=np.float32)
    mk = np.asarray(mask_u8, dtype=np.uint8) > 0
    valid = mk & np.isfinite(dep)
    out = np.zeros_like(dep, dtype=np.float32)
    if int(np.count_nonzero(valid)) <= 0:
        return out
    vals = dep[valid]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if abs(vmax - vmin) <= 1e-6:
        out[valid] = 1.0
        return out
    out[valid] = ((vals - vmin) / (vmax - vmin)).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _depth_to_normal_map(depth: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    dep = np.asarray(depth, dtype=np.float32)
    mk = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
    out = np.zeros((int(dep.shape[0]), int(dep.shape[1]), 3), dtype=np.float32)
    valid = (mk > 0) & np.isfinite(dep)
    if int(np.count_nonzero(valid)) <= 0:
        return out
    dep0 = np.where(valid, dep, 0.0).astype(np.float32)
    dep_blur = cv2.GaussianBlur(dep0, (3, 3), sigmaX=0.0)
    dzdx = cv2.Sobel(dep_blur, cv2.CV_32F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(dep_blur, cv2.CV_32F, 0, 1, ksize=3)
    nx = -dzdx
    ny = -dzdy
    nz = np.ones_like(dep_blur, dtype=np.float32)
    nrm = np.stack([nx, ny, nz], axis=2).astype(np.float32)
    den = np.linalg.norm(nrm, axis=2, keepdims=True)
    den = np.maximum(den, 1e-6)
    nrm = nrm / den
    out[valid] = nrm[valid]
    return out.astype(np.float32)


def _build_render_input(mask_u8: np.ndarray, depth: np.ndarray, image_wh: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mk = np.asarray(mask_u8, dtype=np.uint8)
    dep = np.asarray(depth, dtype=np.float32)
    contour_rgb, filled = mis3d._mask_to_contour_rgb(mk, image_wh, thickness=2)
    depth_norm = _normalize_depth(dep, filled)
    normal = _depth_to_normal_map(dep, filled)
    normal_01 = np.clip((normal + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)
    inp = np.concatenate(
        [
            depth_norm[None, ...],
            np.transpose(normal_01, (2, 0, 1)),
        ],
        axis=0,
    ).astype(np.float32)
    return inp, np.asarray(contour_rgb, dtype=np.uint8), normal.astype(np.float32)


def _matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    mat = np.asarray(R, dtype=np.float32).reshape(3, 3)
    # rot6d should serialize the first two rotation columns column-wise:
    # [R[:,0], R[:,1]], matching _rot6d_to_matrix() which reads 0:3 and 3:6.
    return np.concatenate([mat[:, 0], mat[:, 1]], axis=0).astype(np.float32)


def _rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    proj = torch.sum(b1 * a2, dim=1, keepdim=True)
    b2 = F.normalize(a2 - proj * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)


def _transform_to_pose_vec(T: np.ndarray) -> np.ndarray:
    mat = np.asarray(T, dtype=np.float32).reshape(4, 4)
    rot6d = _matrix_to_rot6d(mat[:3, :3])
    t = np.asarray(mat[:3, 3], dtype=np.float32).reshape(3)
    return np.concatenate([rot6d, t], axis=0).astype(np.float32)


def _pose_vec_to_transform_tensor(pose_vec: torch.Tensor) -> torch.Tensor:
    R = _rot6d_to_matrix(pose_vec[:, :6])
    t = pose_vec[:, 6:9]
    B = int(pose_vec.shape[0])
    T = torch.eye(4, device=pose_vec.device, dtype=pose_vec.dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def _rotation_geodesic_deg(pred_R: torch.Tensor, gt_R: torch.Tensor) -> torch.Tensor:
    r = torch.matmul(pred_R.transpose(1, 2), gt_R)
    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    cosv = torch.clamp((trace - 1.0) * 0.5, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.rad2deg(torch.acos(cosv))


def _render_loss(pred_render: torch.Tensor, gt_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_depth = gt_input[:, 0:1]
    gt_normal = torch.clamp(gt_input[:, 1:4] * 2.0 - 1.0, -1.0, 1.0)
    pred_depth = pred_render[:, 0:1]
    pred_normal = torch.clamp(pred_render[:, 1:4] * 2.0 - 1.0, -1.0, 1.0)
    depth_loss = F.smooth_l1_loss(pred_depth, gt_depth)
    ncos = torch.sum(F.normalize(pred_normal, dim=1) * F.normalize(gt_normal, dim=1), dim=1, keepdim=True)
    normal_loss = (1.0 - torch.clamp(ncos, -1.0, 1.0)).mean()
    return depth_loss + normal_loss, depth_loss, normal_loss


def _prepare_reference_points(
    *,
    frame_bgr: np.ndarray,
    mask_full: np.ndarray,
    bbox: tuple[int, int, int, int],
    expand_px: int,
    cfg: mis3d.core.TrackConfig,
    seed: int,
) -> dict[str, Any] | None:
    bbox_exp = _expand_bbox(bbox, int(expand_px), frame_bgr.shape[:2])
    roi = mis3d.core.crop(frame_bgr, bbox_exp)
    if roi is None or roi.ndim != 3 or int(roi.shape[0]) < 4 or int(roi.shape[1]) < 4:
        return None
    th_gray, _mask0, _ = mis3d.core.compute_th_gray_truncated(roi, cfg.th_percent, cfg)
    th_gray = np.asarray(th_gray, dtype=np.float32)
    mask_roi = _extract_mask_roi(np.asarray(mask_full, dtype=np.uint8), bbox_exp)
    if mask_roi.shape != th_gray.shape:
        mask_roi = cv2.resize(mask_roi, (int(th_gray.shape[1]), int(th_gray.shape[0])), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    mask_roi = mis3d._mask_to_filled_u8(mask_roi)
    pts = mis3d._uvh_points_from_th_mask_roi(
        th_gray_truncated=np.asarray(th_gray, dtype=np.float32),
        mask_u8=np.asarray(mask_roi, dtype=np.uint8),
        bbox=tuple(bbox_exp),
        z_scale=float(cfg.mesh_z_scale),
        max_points=20000,
        rng=np.random.default_rng(int(seed)),
        fill_mask=True,
        min_height=1.0,
        deterministic=True,
    )
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if int(pts.shape[0]) < 200:
        return None
    pmin = np.nanmin(pts, axis=0)
    pmax = np.nanmax(pts, axis=0)
    center = 0.5 * (pmin + pmax)
    radius = float(max(1e-6, 0.5 * np.linalg.norm(pmax - pmin)))
    pts_canon = ((pts - center[None, :]) / float(radius)).astype(np.float32)
    return {
        "bbox_exp": tuple(int(v) for v in bbox_exp),
        "pts_world": pts,
        "pts_canon": pts_canon,
        "center": center.astype(np.float32),
        "radius": float(radius),
    }


def build_rt_regression_cache(
    *,
    image_root: str,
    gt_root: str,
    out_root: str,
    dataset_name: str = "",
    val_ratio: float = 0.27,
    seed: int = 42,
    image_size: int = 128,
    n_views: int = 2048,
    radius_levels: int = 64,
    radius_min_mult: float = 0.5,
    radius_max_mult: float = 5.0,
    n_reference_frames: int = 12,
    bbox_expand_values: str | list[int] | tuple[int, ...] = "32,64",
    force_rebuild: bool = False,
) -> str:
    _set_seed(int(seed))
    image_root_p = Path(image_root)
    gt_root_p = Path(gt_root)
    out_root_p = Path(out_root)
    seq_out_root = out_root_p / "rt_sequence_cache"
    seq_out_root.mkdir(parents=True, exist_ok=True)

    names = _discover_dataset_names(image_root_p, gt_root_p)
    dataset_name_s = str(dataset_name or "").strip()
    if dataset_name_s:
        names = [n for n in names if str(n) == dataset_name_s]
        if not names:
            raise RuntimeError(f"未找到指定数据集: {dataset_name_s}")
    rng = random.Random(int(seed))
    single_sequence_mode = bool(len(names) == 1)
    if single_sequence_mode:
        val_names: set[str] = set()
    else:
        shuffled = list(names)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(float(len(shuffled)) * float(val_ratio))))
        val_names = set(shuffled[:n_val])
    cfg = _build_cfg(int(image_size))
    expand_values = _parse_expand_values(bbox_expand_values)
    dist_mults = np.linspace(float(radius_min_mult), float(radius_max_mult), num=int(max(2, radius_levels)), dtype=np.float32).tolist()

    sequences: list[SequenceCache] = []
    for name in names:
        split = "val" if name in val_names else "train"
        frames_dir = _resolve_image_dir(image_root_p / name)
        gt_dir = _resolve_gt_mask_dir(gt_root_p / name)
        if frames_dir is None or gt_dir is None:
            continue
        seq_dir = seq_out_root / name
        seq_dir.mkdir(parents=True, exist_ok=True)
        seq_npz = seq_dir / "rt_regression_sequence.npz"
        train_npz = seq_dir / "rt_regression_train.npz"
        val_npz = seq_dir / "rt_regression_val.npz"
        if single_sequence_mode and (not bool(force_rebuild)) and train_npz.exists() and val_npz.exists():
            try:
                train_data = np.load(str(train_npz), allow_pickle=False)
                val_data = np.load(str(val_npz), allow_pickle=False)
                n_train = int(np.asarray(train_data["inputs"]).shape[0])
                n_val_final = int(np.asarray(val_data["inputs"]).shape[0])
                if n_train > 0 and n_val_final > 0:
                    sequences.append(
                        SequenceCache(
                            name=f"{name}_train",
                            split="train",
                            npz_path=str(train_npz),
                            n_samples=n_train,
                            frames_dir=str(frames_dir),
                            gt_dir=str(gt_dir),
                        )
                    )
                    sequences.append(
                        SequenceCache(
                            name=f"{name}_val",
                            split="val",
                            npz_path=str(val_npz),
                            n_samples=n_val_final,
                            frames_dir=str(frames_dir),
                            gt_dir=str(gt_dir),
                        )
                    )
                    print(f"[UVH-RT] reuse single-sequence cache: {name} train={n_train} val={n_val_final}", flush=True)
                    continue
            except Exception:
                pass
        if (not bool(force_rebuild)) and seq_npz.exists():
            try:
                data = np.load(str(seq_npz), allow_pickle=False)
                n_samples = int(np.asarray(data["inputs"]).shape[0])
                if n_samples > 0:
                    sequences.append(
                        SequenceCache(
                            name=name,
                            split=split,
                            npz_path=str(seq_npz),
                            n_samples=n_samples,
                            frames_dir=str(frames_dir),
                            gt_dir=str(gt_dir),
                        )
                    )
                    print(f"[UVH-RT] reuse sequence cache: {name} samples={n_samples}", flush=True)
                    continue
            except Exception:
                pass

        gt_frames = _get_compare_chessboard()._load_gt_from_mask_dir(
            frames_dir, gt_dir
        )
        if len(gt_frames) < 1:
            continue
        ref_indices = _select_reference_indices(len(gt_frames), int(max(1, n_reference_frames)))
        inputs: list[np.ndarray] = []
        pose_vecs: list[np.ndarray] = []
        T_list: list[np.ndarray] = []
        dirs: list[np.ndarray] = []
        dists: list[float] = []
        rolls: list[float] = []
        ref_frame_indices: list[int] = []
        expand_px_list: list[int] = []
        view_indices: list[int] = []
        pose_table_arr: np.ndarray | None = None

        for ref_pick_idx in ref_indices:
            ref_gt = gt_frames[int(ref_pick_idx)]
            ref_bgr = cv2.imread(str(ref_gt.image_path), cv2.IMREAD_COLOR)
            if ref_bgr is None:
                continue
            for expand_px in expand_values:
                ref_pack = _prepare_reference_points(
                    frame_bgr=ref_bgr,
                    mask_full=np.asarray(ref_gt.mask, dtype=np.uint8),
                    bbox=tuple(ref_gt.bbox),
                    expand_px=int(expand_px),
                    cfg=cfg,
                    seed=int(seed) + int(ref_gt.frame_idx) + int(expand_px),
                )
                if not isinstance(ref_pack, dict):
                    continue
                ref_root = seq_dir / f"ref_{int(ref_gt.frame_idx):04d}_exp{int(expand_px)}"
                lib, _meta = mis3d.build_or_load_view_library(
                    out_root=str(ref_root),
                    pts_w=np.asarray(ref_pack["pts_canon"], dtype=np.float32),
                    cfg=cfg,
                    n_views=int(n_views),
                    n_dirs=0,
                    dist_mults=list(dist_mults),
                    dist_jitter=0.0,
                    roll_deg_list=[0.0],
                    z_bias=1.0,
                    sampler="shell",
                    seed=int(seed),
                    force_rebuild=bool(force_rebuild),
                    save_depth_vis=False,
                    save_mask_png=False,
                    save_view_points=False,
                    view_points_n=0,
                )
                if lib is None:
                    continue
                if pose_table_arr is None:
                    pose_table_arr = np.asarray([_transform_to_pose_vec(np.asarray(t, dtype=np.float32).reshape(4, 4)) for t in np.asarray(lib.T_w2c, dtype=np.float32)], dtype=np.float32)
                for i in range(int(np.asarray(lib.T_w2c).shape[0])):
                    T = np.asarray(lib.T_w2c[i], dtype=np.float32).reshape(4, 4)
                    mask_u8, depth = mis3d._render_mask_depth_from_points(
                        pts_w=np.asarray(ref_pack["pts_canon"], dtype=np.float32),
                        T_w2c=T,
                        K=np.asarray(lib.K, dtype=np.float32),
                        image_wh=tuple(lib.image_wh),
                    )
                    if int(np.count_nonzero(mask_u8)) <= 20:
                        continue
                    inp, _contour_rgb, _normal = _build_render_input(mask_u8, depth, tuple(lib.image_wh))
                    inputs.append(np.asarray(inp, dtype=np.float32))
                    pose_vecs.append(_transform_to_pose_vec(T))
                    T_list.append(T.astype(np.float32))
                    dirs.append(np.asarray(lib.dirs[i], dtype=np.float32).reshape(3))
                    dists.append(float(np.asarray(lib.dists[i]).reshape(-1)[0]))
                    rolls.append(float(np.asarray(lib.rolls[i]).reshape(-1)[0]) if np.asarray(lib.rolls).size > 0 else 0.0)
                    ref_frame_indices.append(int(ref_gt.frame_idx))
                    expand_px_list.append(int(expand_px))
                    view_indices.append(int(i))

        if not inputs:
            continue

        inputs_arr = np.asarray(inputs, dtype=np.float32)
        pose_vecs_arr = np.asarray(pose_vecs, dtype=np.float32)
        T_arr = np.asarray(T_list, dtype=np.float32)
        dirs_arr = np.asarray(dirs, dtype=np.float32)
        dists_arr = np.asarray(dists, dtype=np.float32)
        rolls_arr = np.asarray(rolls, dtype=np.float32)
        ref_frame_indices_arr = np.asarray(ref_frame_indices, dtype=np.int32)
        expand_px_arr = np.asarray(expand_px_list, dtype=np.int32)
        view_indices_arr = np.asarray(view_indices, dtype=np.int32)
        if pose_table_arr is None:
            pose_table_arr = np.zeros((0, 9), dtype=np.float32)

        def _write_npz(npz_path_local: Path, idx_sel: np.ndarray) -> int:
            np.savez_compressed(
                npz_path_local,
                inputs=np.asarray(inputs_arr[idx_sel], dtype=np.float32),
                pose_vecs=np.asarray(pose_vecs_arr[idx_sel], dtype=np.float32),
                T_w2c=np.asarray(T_arr[idx_sel], dtype=np.float32),
                dirs=np.asarray(dirs_arr[idx_sel], dtype=np.float32),
                dists=np.asarray(dists_arr[idx_sel], dtype=np.float32),
                rolls=np.asarray(rolls_arr[idx_sel], dtype=np.float32),
                ref_frame_indices=np.asarray(ref_frame_indices_arr[idx_sel], dtype=np.int32),
                expand_px=np.asarray(expand_px_arr[idx_sel], dtype=np.int32),
                view_indices=np.asarray(view_indices_arr[idx_sel], dtype=np.int32),
                pose_table=np.asarray(pose_table_arr, dtype=np.float32),
            )
            return int(idx_sel.size)

        if single_sequence_mode:
            idx_all = np.arange(int(inputs_arr.shape[0]), dtype=np.int32)
            rng_np = np.random.default_rng(int(seed))
            rng_np.shuffle(idx_all)
            n_val_local = max(1, int(round(float(idx_all.size) * float(val_ratio))))
            n_val_local = min(max(1, n_val_local), max(1, int(idx_all.size) - 1))
            val_idx = np.sort(idx_all[:n_val_local]).astype(np.int32)
            train_idx = np.sort(idx_all[n_val_local:]).astype(np.int32)
            train_npz = seq_dir / "rt_regression_train.npz"
            val_npz = seq_dir / "rt_regression_val.npz"
            n_train = _write_npz(train_npz, train_idx)
            n_val_final = _write_npz(val_npz, val_idx)
            sequences.append(
                SequenceCache(
                    name=f"{name}_train",
                    split="train",
                    npz_path=str(train_npz),
                    n_samples=int(n_train),
                    frames_dir=str(frames_dir),
                    gt_dir=str(gt_dir),
                )
            )
            sequences.append(
                SequenceCache(
                    name=f"{name}_val",
                    split="val",
                    npz_path=str(val_npz),
                    n_samples=int(n_val_final),
                    frames_dir=str(frames_dir),
                    gt_dir=str(gt_dir),
                )
            )
            print(
                f"[UVH-RT] built single-sequence cache: {name} refs={len(ref_indices)} expand={expand_values} "
                f"train={n_train} val={n_val_final}",
                flush=True,
            )
        else:
            _write_npz(seq_npz, np.arange(int(inputs_arr.shape[0]), dtype=np.int32))
            sequences.append(
                SequenceCache(
                    name=name,
                    split=split,
                    npz_path=str(seq_npz),
                    n_samples=int(len(inputs)),
                    frames_dir=str(frames_dir),
                    gt_dir=str(gt_dir),
                )
            )
            print(
                f"[UVH-RT] built sequence cache: {name} refs={len(ref_indices)} expand={expand_values} samples={len(inputs)}",
                flush=True,
            )

    manifest = {
        "image_root": str(image_root_p),
        "gt_root": str(gt_root_p),
        "dataset_name": dataset_name_s,
        "seed": int(seed),
        "image_size": int(image_size),
        "n_views": int(n_views),
        "radius_levels": int(radius_levels),
        "radius_min_mult": float(radius_min_mult),
        "radius_max_mult": float(radius_max_mult),
        "n_reference_frames": int(n_reference_frames),
        "bbox_expand_values": [int(v) for v in expand_values],
        "sequences": [seq.__dict__ for seq in sequences],
    }
    manifest_path = out_root_p / "uvh_rt_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(manifest_path)


def _load_pose_table_from_sequence_npz(npz_path: str) -> np.ndarray:
    seq_dir = Path(npz_path).parent
    ref_dirs = sorted([p for p in seq_dir.iterdir() if p.is_dir() and p.name.startswith("ref_")], key=lambda p: p.name.lower())
    for ref_dir in ref_dirs:
        lib_npz = ref_dir / "target_library" / "view_library_templates.npz"
        if not lib_npz.exists():
            continue
        try:
            z = np.load(str(lib_npz), allow_pickle=False)
            T_w2c = np.asarray(z["T_w2c"], dtype=np.float32).reshape(-1, 4, 4)
            return np.asarray([_transform_to_pose_vec(t) for t in T_w2c], dtype=np.float32)
        except Exception:
            continue
    return np.zeros((0, 9), dtype=np.float32)


def _infer_view_indices_from_pose_vecs(pose_vecs: np.ndarray, pose_table: np.ndarray) -> np.ndarray:
    pv = np.asarray(pose_vecs, dtype=np.float32).reshape(-1, 9)
    pt = np.asarray(pose_table, dtype=np.float32).reshape(-1, 9)
    if int(pt.shape[0]) <= 0:
        return np.zeros((int(pv.shape[0]),), dtype=np.int32)
    lut: dict[tuple[float, ...], int] = {}
    for i in range(int(pt.shape[0])):
        key = tuple(np.round(pt[i], 5).astype(np.float32).tolist())
        lut[key] = int(i)
    out = np.zeros((int(pv.shape[0]),), dtype=np.int32)
    for i in range(int(pv.shape[0])):
        key = tuple(np.round(pv[i], 5).astype(np.float32).tolist())
        if key in lut:
            out[i] = int(lut[key])
            continue
        dif = np.mean(np.abs(pt - pv[i:i + 1]), axis=1)
        out[i] = int(np.argmin(dif))
    return out.astype(np.int32)


class UVHRTDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        max_samples_per_sequence: int = 0,
        seed: int = 42,
        repeat_factor: int = 1,
        enable_warp: bool = False,
    ):
        self.manifest_path = str(manifest_path)
        self.split = str(split)
        self.max_samples_per_sequence = int(max_samples_per_sequence)
        self.repeat_factor = int(max(1, repeat_factor))
        self.enable_warp = bool(enable_warp)
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.sequences: list[dict[str, Any]] = []
        self.samples: list[tuple[int, int]] = []
        self.n_views = int(self.manifest.get("n_views", 0))
        self.image_size = int(self.manifest.get("image_size", 128))
        self.pose_table: np.ndarray | None = None
        rng = np.random.default_rng(int(seed))
        for item in self.manifest.get("sequences", []):
            if str(item.get("split", "")) != self.split:
                continue
            data = np.load(str(item["npz_path"]), allow_pickle=False)
            pose_table = np.asarray(data["pose_table"], dtype=np.float32) if ("pose_table" in getattr(data, "files", [])) else _load_pose_table_from_sequence_npz(str(item["npz_path"]))
            view_indices = np.asarray(data["view_indices"], dtype=np.int32) if ("view_indices" in getattr(data, "files", [])) else _infer_view_indices_from_pose_vecs(np.asarray(data["pose_vecs"], dtype=np.float32), pose_table)
            seq = {
                "name": str(item["name"]),
                "inputs": data["inputs"],
                "pose_vecs": np.asarray(data["pose_vecs"], dtype=np.float32),
                "T_w2c": np.asarray(data["T_w2c"], dtype=np.float32),
                "dirs": np.asarray(data["dirs"], dtype=np.float32),
                "dists": np.asarray(data["dists"], dtype=np.float32),
                "rolls": np.asarray(data["rolls"], dtype=np.float32),
                "ref_frame_indices": np.asarray(data["ref_frame_indices"], dtype=np.int32),
                "expand_px": np.asarray(data["expand_px"], dtype=np.int32),
                "view_indices": np.asarray(view_indices, dtype=np.int32),
                "pose_table": np.asarray(pose_table, dtype=np.float32),
            }
            if self.pose_table is None and int(pose_table.shape[0]) > 0:
                self.pose_table = np.asarray(pose_table, dtype=np.float32)
            seq_idx = len(self.sequences)
            self.sequences.append(seq)
            sample_indices = list(range(int(seq["inputs"].shape[0])))
            cap = int(self.max_samples_per_sequence)
            if cap > 0 and len(sample_indices) > cap:
                sample_indices = sorted(rng.choice(sample_indices, size=cap, replace=False).astype(np.int32).tolist())
            for _ in range(self.repeat_factor):
                for sample_idx in sample_indices:
                    self.samples.append((seq_idx, sample_idx))

    def __len__(self) -> int:
        return int(len(self.samples))

    def _warp_input(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32).copy()
        if not self.enable_warp:
            return arr
        if random.random() < 0.80:
            h, w = int(arr.shape[1]), int(arr.shape[2])
            jitter = float(max(2.0, min(h, w) * 0.06))
            src = np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            dst = src + np.random.uniform(-jitter, jitter, size=(4, 2)).astype(np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            warped = np.zeros_like(arr, dtype=np.float32)
            for c in range(int(arr.shape[0])):
                warped[c] = cv2.warpPerspective(
                    arr[c],
                    M,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0.0,
                ).astype(np.float32)
            arr = warped
        if random.random() < 0.30:
            arr += np.random.normal(loc=0.0, scale=0.01, size=arr.shape).astype(np.float32)
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        seq_idx, sample_idx = self.samples[int(index)]
        seq = self.sequences[int(seq_idx)]
        x = np.asarray(seq["inputs"][sample_idx], dtype=np.float32)
        if int(x.ndim) == 3 and int(x.shape[0]) == 5:
            x = np.asarray(x[[0, 2, 3, 4], ...], dtype=np.float32)
        x = self._warp_input(x)
        return {
            "input": torch.from_numpy(x),
            "pose_vec": torch.from_numpy(np.asarray(seq["pose_vecs"][sample_idx], dtype=np.float32)),
            "T_w2c": torch.from_numpy(np.asarray(seq["T_w2c"][sample_idx], dtype=np.float32)),
            "view_index": int(seq["view_indices"][sample_idx]),
            "seq_idx": int(seq_idx),
        }


class RTConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        latent_dim: int = 128,
        base_channels: int = 24,
        depth: int = 4,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        cin = int(in_channels)
        last_c = int(base_channels)
        for i in range(int(max(2, depth))):
            cout = int(base_channels * (2 ** i))
            cout = int(min(512, max(16, cout)))
            groups = max(1, min(16, cout // 8))
            layers.append(nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1))
            layers.append(nn.GroupNorm(groups, cout))
            layers.append(nn.SiLU(inplace=True))
            cin = cout
            last_c = cout
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(int(last_c), int(latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        y = torch.flatten(y, 1)
        y = self.fc(y)
        return F.normalize(y, dim=1)


class RenderDecoder(nn.Module):
    def __init__(self, latent_dim: int = 128, out_channels: int = 4, image_size: int = 128):
        super().__init__()
        self.image_size = int(image_size)
        self.seed_hw = 8
        self.seed_c = int(max(32, latent_dim))
        self.fc = nn.Sequential(
            nn.Linear(int(latent_dim), int(self.seed_c * self.seed_hw * self.seed_hw)),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(self.seed_c, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(inplace=True)),
            nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(128, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.SiLU(inplace=True)),
            nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(96, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(inplace=True)),
            nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), nn.Conv2d(64, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.SiLU(inplace=True)),
        ])
        self.out = nn.Conv2d(48, int(out_channels), kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.fc(z)
        y = y.view(int(z.shape[0]), int(self.seed_c), int(self.seed_hw), int(self.seed_hw))
        for blk in self.blocks:
            y = blk(y)
        if int(y.shape[-1]) != int(self.image_size):
            y = F.interpolate(y, size=(int(self.image_size), int(self.image_size)), mode="bilinear", align_corners=False)
        y = self.out(y)
        depth = torch.sigmoid(y[:, 0:1])
        normal = torch.tanh(y[:, 1:4])
        normal = F.normalize(normal, dim=1)
        normal01 = (normal + 1.0) * 0.5
        return torch.cat([depth, normal01], dim=1)


class UVHRTRegressor(nn.Module):
    def __init__(
        self,
        n_views: int,
        pose_table: torch.Tensor | np.ndarray,
        image_size: int = 128,
        latent_dim: int = 128,
        encoder_base_channels: int = 24,
        encoder_depth: int = 4,
        head_hidden_dim: int = 256,
        head_depth: int = 3,
    ):
        super().__init__()
        self.encoder = RTConvEncoder(
            in_channels=4,
            latent_dim=int(latent_dim),
            base_channels=int(encoder_base_channels),
            depth=int(encoder_depth),
        )
        pose_table_t = torch.as_tensor(np.asarray(pose_table, dtype=np.float32) if isinstance(pose_table, np.ndarray) else pose_table, dtype=torch.float32)
        self.register_buffer("pose_table", pose_table_t.reshape(-1, 9), persistent=True)
        self.n_views = int(n_views)
        self.target_token = nn.Parameter(torch.zeros((1, int(latent_dim)), dtype=torch.float32))
        self.coarse_head = nn.Linear(int(latent_dim), int(n_views))
        layers: list[nn.Module] = []
        in_dim = int(latent_dim + 9)
        hid = int(head_hidden_dim)
        for _ in range(int(max(1, head_depth - 1))):
            layers.append(nn.Linear(in_dim, hid))
            layers.append(nn.SiLU(inplace=True))
            layers.append(nn.Dropout(p=0.10))
            in_dim = hid
        layers.append(nn.Linear(in_dim, 9))
        self.pose_head = nn.Sequential(*layers)
        self.render_pose = nn.Sequential(
            nn.Linear(9 + int(latent_dim), int(head_hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(head_hidden_dim), int(latent_dim)),
            nn.SiLU(inplace=True),
        )
        self.render_decoder = RenderDecoder(latent_dim=int(latent_dim), out_channels=4, image_size=int(image_size))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encode(x)
        logits = self.coarse_head(feat)
        probs = torch.softmax(logits, dim=1)
        pose_anchor = torch.matmul(probs, self.pose_table)
        pose_delta = self.pose_head(torch.cat([feat, pose_anchor], dim=1))
        pred_pose = pose_anchor + 0.25 * pose_delta
        token = self.target_token.expand(int(feat.shape[0]), -1)
        render_latent = self.render_pose(torch.cat([pred_pose, feat + token], dim=1))
        render_pred = self.render_decoder(render_latent)
        return {
            "feat": feat,
            "logits": logits,
            "pose_anchor": pose_anchor,
            "pose_delta": pose_delta,
            "pose": pred_pose,
            "render_pred": render_pred,
        }


def _infer_model_config_from_state_dict(state: dict[str, torch.Tensor]) -> dict[str, int]:
    cfg: dict[str, int] = {}
    try:
        cfg["latent_dim"] = int(state["encoder.fc.bias"].shape[0])
    except Exception:
        cfg["latent_dim"] = 128
    try:
        cfg["encoder_base_channels"] = int(state["encoder.net.0.bias"].shape[0])
    except Exception:
        cfg["encoder_base_channels"] = 24
    try:
        conv_keys = [k for k, v in state.items() if k.startswith("encoder.net.") and int(v.ndim) == 4]
        cfg["encoder_depth"] = int(len(conv_keys))
    except Exception:
        cfg["encoder_depth"] = 4
    try:
        linear_keys = [k for k, v in state.items() if k.startswith("pose_head.") and int(v.ndim) == 2]
        cfg["head_depth"] = int(max(1, len(linear_keys)))
        cfg["head_hidden_dim"] = int(state[linear_keys[0]].shape[0]) if len(linear_keys) > 1 else 256
    except Exception:
        cfg["head_depth"] = 3
        cfg["head_hidden_dim"] = 256
    return cfg


def load_rt_regressor_checkpoint(
    *,
    checkpoint_path: str,
    n_views: int,
    pose_table: torch.Tensor | np.ndarray,
    image_size: int = 128,
    device: str | torch.device = "cuda",
    latent_dim: int = 128,
    encoder_base_channels: int = 24,
    encoder_depth: int = 4,
    head_hidden_dim: int = 256,
    head_depth: int = 3,
) -> tuple[UVHRTRegressor, dict[str, Any]]:
    dev = _resolve_device(device)
    ckpt = torch.load(str(checkpoint_path), map_location=dev)
    model_cfg = dict(ckpt.get("model_config", {})) if isinstance(ckpt, dict) else {}
    if not model_cfg and isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt)
        if isinstance(state, dict):
            model_cfg = _infer_model_config_from_state_dict(state)
    model = UVHRTRegressor(
        n_views=int(n_views),
        pose_table=pose_table,
        image_size=int(image_size),
        latent_dim=int(model_cfg.get("latent_dim", latent_dim)),
        encoder_base_channels=int(model_cfg.get("encoder_base_channels", encoder_base_channels)),
        encoder_depth=int(model_cfg.get("encoder_depth", encoder_depth)),
        head_hidden_dim=int(model_cfg.get("head_hidden_dim", head_hidden_dim)),
        head_depth=int(model_cfg.get("head_depth", head_depth)),
    ).to(dev)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, (ckpt if isinstance(ckpt, dict) else {"model_config": model_cfg})


def _run_one_epoch(
    *,
    model: UVHRTRegressor,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    coarse_loss_weight: float,
    rot_loss_weight: float,
    trans_loss_weight: float,
    render_loss_weight: float,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_coarse = 0.0
    total_rot = 0.0
    total_trans = 0.0
    total_render = 0.0
    total_rot_deg = 0.0
    n_items = 0

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        gt_T = batch["T_w2c"].to(device, non_blocking=True)
        view_index = batch["view_index"].to(device, non_blocking=True)
        gt_R = gt_T[:, :3, :3]
        gt_t = gt_T[:, :3, 3]
        with torch.amp.autocast(device_type=device.type, enabled=bool(use_amp and device.type == "cuda")):
            out = model(x)
            pred_pose = out["pose"]
            pred_R = _rot6d_to_matrix(pred_pose[:, :6])
            pred_t = pred_pose[:, 6:9]
            coarse_loss = F.cross_entropy(out["logits"], view_index)
            rot_deg = _rotation_geodesic_deg(pred_R, gt_R)
            rot_loss = torch.mean(torch.deg2rad(rot_deg))
            trans_loss = F.smooth_l1_loss(pred_t, gt_t)
            render_loss, _depth_loss, _normal_loss = _render_loss(out["render_pred"], x)
            loss = (
                float(coarse_loss_weight) * coarse_loss
                + float(rot_loss_weight) * rot_loss
                + float(trans_loss_weight) * trans_loss
                + float(render_loss_weight) * render_loss
            )
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and bool(use_amp and device.type == "cuda"):
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        bs = int(x.shape[0])
        total_loss += float(loss.detach().cpu()) * bs
        total_coarse += float(coarse_loss.detach().cpu()) * bs
        total_rot += float(rot_loss.detach().cpu()) * bs
        total_trans += float(trans_loss.detach().cpu()) * bs
        total_render += float(render_loss.detach().cpu()) * bs
        total_rot_deg += float(rot_deg.mean().detach().cpu()) * bs
        n_items += bs

    denom = float(max(1, n_items))
    return {
        "loss": float(total_loss / denom),
        "coarse_loss": float(total_coarse / denom),
        "rot_loss": float(total_rot / denom),
        "trans_loss": float(total_trans / denom),
        "render_loss": float(total_render / denom),
        "mean_rot_err_deg": float(total_rot_deg / denom),
    }


@torch.no_grad()
def evaluate_rt_regressor(
    *,
    model: UVHRTRegressor,
    dataset: UVHRTDataset,
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    use_amp: bool = False,
) -> dict[str, float]:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(max(0, num_workers)),
        persistent_workers=bool(int(max(0, num_workers)) > 0),
        pin_memory=bool(pin_memory),
    )
    model.eval()
    rot_errs: list[float] = []
    trans_errs: list[float] = []
    coarse_hits = 0
    render_losses: list[float] = []
    acc_5_005 = 0
    acc_10_01 = 0
    n = 0
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        gt_T = batch["T_w2c"].to(device, non_blocking=True)
        view_index = batch["view_index"].to(device, non_blocking=True)
        gt_R = gt_T[:, :3, :3]
        gt_t = gt_T[:, :3, 3]
        with torch.amp.autocast(device_type=device.type, enabled=bool(use_amp and device.type == "cuda")):
            out = model(x)
            pred_pose = out["pose"]
            pred_R = _rot6d_to_matrix(pred_pose[:, :6])
            pred_t = pred_pose[:, 6:9]
            render_loss, _depth_loss, _normal_loss = _render_loss(out["render_pred"], x)
        pred_view = torch.argmax(out["logits"], dim=1)
        coarse_hits += int(torch.sum((pred_view == view_index).int()).detach().cpu().item())
        rot_deg = _rotation_geodesic_deg(pred_R, gt_R).detach().cpu().numpy().astype(np.float32)
        trans = torch.linalg.norm(pred_t - gt_t, dim=1).detach().cpu().numpy().astype(np.float32)
        render_losses.extend([float(render_loss.detach().cpu().item())] * int(x.shape[0]))
        rot_errs.extend([float(v) for v in rot_deg.tolist()])
        trans_errs.extend([float(v) for v in trans.tolist()])
        acc_5_005 += int(np.sum((rot_deg <= 5.0) & (trans <= 0.05)))
        acc_10_01 += int(np.sum((rot_deg <= 10.0) & (trans <= 0.10)))
        n += int(x.shape[0])

    def _mean(xs: list[float]) -> float:
        vals = [float(x) for x in xs if np.isfinite(float(x))]
        return float(np.mean(vals)) if vals else float("nan")

    def _median(xs: list[float]) -> float:
        vals = [float(x) for x in xs if np.isfinite(float(x))]
        return float(np.median(vals)) if vals else float("nan")

    return {
        "n_samples": int(n),
        "coarse_top1": float(coarse_hits / max(1, n)),
        "mean_render_loss": _mean(render_losses),
        "mean_rot_err_deg": _mean(rot_errs),
        "median_rot_err_deg": _median(rot_errs),
        "mean_trans_err": _mean(trans_errs),
        "median_trans_err": _median(trans_errs),
        "acc_5deg_005": float(acc_5_005 / max(1, n)),
        "acc_10deg_01": float(acc_10_01 / max(1, n)),
    }


@torch.no_grad()
def build_feature_bank(
    *,
    model: UVHRTRegressor,
    dataset: UVHRTDataset,
    device: torch.device,
    max_items: int = 8192,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=0)
    feats: list[np.ndarray] = []
    pose_vecs: list[np.ndarray] = []
    n_seen = 0
    model.eval()
    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        feat = model.encode(x).detach().cpu().numpy().astype(np.float32)
        feats.append(feat)
        pose_vecs.append(np.asarray(batch["pose_vec"].cpu().numpy(), dtype=np.float32))
        n_seen += int(feat.shape[0])
        if n_seen >= int(max_items):
            break
    if not feats:
        return {"features": np.zeros((0, 1), dtype=np.float32), "pose_vecs": np.zeros((0, 9), dtype=np.float32)}
    f = np.concatenate(feats, axis=0)[: int(max_items)].astype(np.float32)
    p = np.concatenate(pose_vecs, axis=0)[: int(max_items)].astype(np.float32)
    return {"features": f, "pose_vecs": p}


@torch.no_grad()
def predict_rt_with_confidence(
    *,
    model: UVHRTRegressor,
    input_tensor: np.ndarray | torch.Tensor,
    device: str | torch.device = "cuda",
    feature_bank: np.ndarray | None = None,
) -> dict[str, Any]:
    dev = _resolve_device(device)
    model = model.to(dev)
    model.eval()
    if isinstance(input_tensor, np.ndarray):
        x = torch.from_numpy(np.asarray(input_tensor, dtype=np.float32))
    else:
        x = input_tensor.detach().float().cpu()
    if x.ndim == 3:
        x = x.unsqueeze(0)
    x = x.to(dev, non_blocking=True)
    feat = model.encode(x)
    out = model(x)
    pose = out["pose"]
    T = _pose_vec_to_transform_tensor(pose).detach().cpu().numpy()[0].astype(np.float32)
    conf = float("nan")
    if isinstance(feature_bank, np.ndarray) and int(feature_bank.size) > 0:
        bank = torch.from_numpy(np.asarray(feature_bank, dtype=np.float32)).to(dev)
        bank = F.normalize(bank, dim=1)
        sim = torch.matmul(F.normalize(feat, dim=1), bank.transpose(0, 1))
        conf = float(torch.max(sim).detach().cpu().item())
    return {
        "T_w2c": T,
        "pose_vec": pose.detach().cpu().numpy()[0].astype(np.float32),
        "coarse_view_index": int(torch.argmax(out["logits"], dim=1).detach().cpu().item()),
        "in_distribution_score": conf,
    }


def train_rt_regressor(
    *,
    manifest_path: str,
    out_dir: str,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    coarse_loss_weight: float = 1.0,
    rot_loss_weight: float = 1.0,
    trans_loss_weight: float = 1.0,
    render_loss_weight: float = 0.5,
    seed: int = 42,
    max_train_samples_per_sequence: int = 0,
    train_repeat_factor: int = 1,
    enable_train_warp: bool = True,
    latent_dim: int = 128,
    encoder_base_channels: int = 24,
    encoder_depth: int = 4,
    head_hidden_dim: int = 256,
    head_depth: int = 3,
    init_checkpoint: str = "",
    eval_interval: int = 5,
    early_stop_patience: int = 40,
    early_stop_min_delta: float = 1e-4,
    train_num_workers: int = 0,
    val_num_workers: int = 0,
    device: str = "cuda",
    use_amp: bool = True,
) -> dict[str, Any]:
    _set_seed(int(seed))
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    device_t = _resolve_device(device)
    if device_t.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp_enabled = bool(use_amp and device_t.type == "cuda")

    init_checkpoint_path = str(init_checkpoint or "").strip()
    ckpt_for_init = None
    if init_checkpoint_path and Path(init_checkpoint_path).exists():
        ckpt_for_init = torch.load(init_checkpoint_path, map_location=device_t)
        if isinstance(ckpt_for_init, dict):
            model_cfg = dict(ckpt_for_init.get("model_config", {}))
            if not model_cfg:
                state_for_cfg = ckpt_for_init.get("model", ckpt_for_init)
                if isinstance(state_for_cfg, dict):
                    model_cfg = _infer_model_config_from_state_dict(state_for_cfg)
            latent_dim = int(model_cfg.get("latent_dim", latent_dim))
            encoder_base_channels = int(model_cfg.get("encoder_base_channels", encoder_base_channels))
            encoder_depth = int(model_cfg.get("encoder_depth", encoder_depth))
            head_hidden_dim = int(model_cfg.get("head_hidden_dim", head_hidden_dim))
            head_depth = int(model_cfg.get("head_depth", head_depth))

    train_set = UVHRTDataset(
        manifest_path=manifest_path,
        split="train",
        max_samples_per_sequence=int(max_train_samples_per_sequence),
        seed=int(seed),
        repeat_factor=int(max(1, train_repeat_factor)),
        enable_warp=bool(enable_train_warp),
    )
    val_set = UVHRTDataset(
        manifest_path=manifest_path,
        split="val",
        max_samples_per_sequence=0,
        seed=int(seed),
        repeat_factor=1,
        enable_warp=False,
    )
    if len(train_set) <= 0 or len(val_set) <= 0:
        raise RuntimeError("训练集或验证集为空，无法训练 UVH RT regressor")
    pose_table = train_set.pose_table if isinstance(train_set.pose_table, np.ndarray) else None
    if not isinstance(pose_table, np.ndarray) or int(pose_table.shape[0]) <= 0:
        raise RuntimeError("pose_table 为空，无法训练 coarse pose classifier")

    train_workers = _auto_num_workers(int(train_num_workers), device_t, fallback=4)
    val_workers = _auto_num_workers(int(val_num_workers), device_t, fallback=2)
    pin_memory = bool(device_t.type == "cuda")
    train_loader = DataLoader(
        train_set,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(train_workers),
        persistent_workers=bool(train_workers > 0),
        pin_memory=pin_memory,
    )
    model = UVHRTRegressor(
        n_views=int(train_set.n_views),
        pose_table=np.asarray(pose_table, dtype=np.float32),
        image_size=int(train_set.image_size),
        latent_dim=int(latent_dim),
        encoder_base_channels=int(encoder_base_channels),
        encoder_depth=int(encoder_depth),
        head_hidden_dim=int(head_hidden_dim),
        head_depth=int(head_depth),
    ).to(device_t)
    if init_checkpoint_path and Path(init_checkpoint_path).exists():
        ckpt = ckpt_for_init if ckpt_for_init is not None else torch.load(init_checkpoint_path, map_location=device_t)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=True)
        print(f"[UVH-RT] loaded init checkpoint: {init_checkpoint_path}", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    model_params = int(sum(int(p.numel()) for p in model.parameters()))
    print(
        f"[UVH-RT] params={model_params} latent={int(latent_dim)} enc_base={int(encoder_base_channels)} "
        f"enc_depth={int(encoder_depth)} head_hidden={int(head_hidden_dim)} head_depth={int(head_depth)} "
        f"warp={int(bool(enable_train_warp))} n_views={int(train_set.n_views)} device={device_t.type} amp={int(amp_enabled)} "
        f"train_workers={int(train_workers)} val_workers={int(val_workers)}",
        flush=True,
    )

    best_state = None
    best_metric = float("inf")
    epochs_since_improve = 0
    eval_every = int(max(1, eval_interval))
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _run_one_epoch(
            model=model,
            loader=train_loader,
            device=device_t,
            optimizer=optimizer,
            coarse_loss_weight=float(coarse_loss_weight),
            rot_loss_weight=float(rot_loss_weight),
            trans_loss_weight=float(trans_loss_weight),
            render_loss_weight=float(render_loss_weight),
            use_amp=amp_enabled,
            scaler=scaler,
        )
        do_eval = bool(epoch == 1 or epoch == int(epochs) or (epoch % int(eval_every) == 0))
        val_metrics = None
        if do_eval:
            val_metrics = evaluate_rt_regressor(
                model=model,
                dataset=val_set,
                device=device_t,
                batch_size=int(batch_size),
                num_workers=int(val_workers),
                pin_memory=pin_memory,
                use_amp=amp_enabled,
            )
        row = {"epoch": int(epoch), "train": train_metrics, "val": val_metrics}
        history.append(row)
        if isinstance(val_metrics, dict):
            print(
                f"[UVH-RT] epoch={epoch}/{epochs} train_loss={train_metrics['loss']:.4f} "
                f"train_coarse={train_metrics['coarse_loss']:.4f} "
                f"train_rot_deg={train_metrics['mean_rot_err_deg']:.2f} "
                f"val_coarse={val_metrics['coarse_top1']:.4f} "
                f"val_rot_deg={val_metrics['mean_rot_err_deg']:.2f} "
                f"val_trans={val_metrics['mean_trans_err']:.4f} "
                f"val_render={val_metrics['mean_render_loss']:.4f} val_acc10={val_metrics['acc_10deg_01']:.4f}",
                flush=True,
            )
            score = (
                50.0 * (1.0 - float(val_metrics.get("coarse_top1", 0.0)))
                + float(val_metrics.get("mean_rot_err_deg", float("inf")))
                + 10.0 * float(val_metrics.get("mean_trans_err", float("inf")))
                + 5.0 * float(val_metrics.get("mean_render_loss", float("inf")))
            )
            if float(score) + float(max(0.0, early_stop_min_delta)) < float(best_metric):
                best_metric = score
                epochs_since_improve = 0
                best_state = {
                    "model": model.state_dict(),
                    "epoch": int(epoch),
                    "val_metrics": val_metrics,
                    "manifest_path": str(manifest_path),
                    "model_config": {
                        "latent_dim": int(latent_dim),
                        "encoder_base_channels": int(encoder_base_channels),
                        "encoder_depth": int(encoder_depth),
                        "head_hidden_dim": int(head_hidden_dim),
                        "head_depth": int(head_depth),
                    },
                }
            else:
                epochs_since_improve += int(eval_every)
        else:
            print(
                f"[UVH-RT] epoch={epoch}/{epochs} train_loss={train_metrics['loss']:.4f} "
                f"train_coarse={train_metrics['coarse_loss']:.4f} train_rot_deg={train_metrics['mean_rot_err_deg']:.2f}",
                flush=True,
            )
        latest_summary = {
            "last_epoch": int(epoch),
            "best_metric": float(best_metric),
            "epochs_since_improve": int(epochs_since_improve),
            "latest_train": train_metrics,
            "latest_val": val_metrics,
        }
        (out_dir_p / "train_summary_latest.json").write_text(json.dumps(latest_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if int(early_stop_patience) > 0 and int(epochs_since_improve) >= int(early_stop_patience):
            print(f"[UVH-RT] early stop at epoch={epoch} patience={early_stop_patience}", flush=True)
            break

    if best_state is None:
        final_eval = evaluate_rt_regressor(
            model=model,
            dataset=val_set,
            device=device_t,
            batch_size=int(batch_size),
            num_workers=int(val_workers),
            pin_memory=pin_memory,
            use_amp=amp_enabled,
        )
        best_state = {
            "model": model.state_dict(),
            "epoch": int(max(1, len(history))),
            "val_metrics": final_eval,
            "manifest_path": str(manifest_path),
            "model_config": {
                "latent_dim": int(latent_dim),
                "encoder_base_channels": int(encoder_base_channels),
                "encoder_depth": int(encoder_depth),
                "head_hidden_dim": int(head_hidden_dim),
                "head_depth": int(head_depth),
            },
        }

    bank = build_feature_bank(model=model, dataset=train_set, device=device_t, max_items=8192, batch_size=int(batch_size))
    bank_path = out_dir_p / "train_feature_bank.npz"
    np.savez_compressed(bank_path, features=np.asarray(bank["features"], dtype=np.float32), pose_vecs=np.asarray(bank["pose_vecs"], dtype=np.float32))

    ckpt_path = out_dir_p / "uvh_rt_regressor_best.pt"
    best_state["feature_bank_path"] = str(bank_path)
    torch.save(best_state, str(ckpt_path))
    final_val = evaluate_rt_regressor(
        model=model,
        dataset=val_set,
        device=device_t,
        batch_size=int(batch_size),
        num_workers=int(val_workers),
        pin_memory=pin_memory,
        use_amp=amp_enabled,
    )
    summary = {
        "checkpoint": str(ckpt_path),
        "feature_bank": str(bank_path),
        "best_epoch": int(best_state["epoch"]),
        "best_val": best_state["val_metrics"],
        "final_val": final_val,
        "history": history,
        "train_size": int(len(train_set)),
        "val_size": int(len(val_set)),
        "n_train_sequences": int(len(train_set.sequences)),
        "n_val_sequences": int(len(val_set.sequences)),
        "device": str(device_t),
        "enable_train_warp": int(bool(enable_train_warp)),
        "model_params": int(model_params),
        "latent_dim": int(latent_dim),
        "encoder_base_channels": int(encoder_base_channels),
        "encoder_depth": int(encoder_depth),
        "head_hidden_dim": int(head_hidden_dim),
        "head_depth": int(head_depth),
        "coarse_loss_weight": float(coarse_loss_weight),
        "init_checkpoint": init_checkpoint_path,
        "render_loss_weight": float(render_loss_weight),
        "eval_interval": int(eval_every),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "train_num_workers": int(train_workers),
        "val_num_workers": int(val_workers),
        "amp_enabled": bool(amp_enabled),
        "manifest_path": str(manifest_path),
        "out_dir": str(out_dir_p),
    }
    (out_dir_p / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
