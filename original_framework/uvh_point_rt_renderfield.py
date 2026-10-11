from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class RenderFieldSequence:
    name: str
    split: str
    npz_path: str
    n_samples: int
    n_pointclouds: int
    frames_dir: str
    gt_dir: str


def _warp_points_np(
    pts: np.ndarray,
    *,
    seed: int,
    scale_std: float = 0.04,
    bend_amp: float = 0.03,
    noise_std: float = 0.005,
) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if int(arr.shape[0]) <= 0:
        return arr.astype(np.float32, copy=True)
    rng = np.random.default_rng(int(seed))
    center = np.mean(arr, axis=0, dtype=np.float32)
    centered = arr - center[None, :]
    radius = float(np.max(np.linalg.norm(centered, axis=1))) if int(centered.shape[0]) > 0 else 1.0
    radius = max(radius, 1e-6)
    normed = centered / radius
    scale = 1.0 + rng.normal(0.0, float(max(0.0, scale_std)), size=(3,)).astype(np.float32)
    phase = rng.uniform(-np.pi, np.pi, size=(3,)).astype(np.float32)
    freq = rng.uniform(0.8, 2.0, size=(3,)).astype(np.float32)
    warped = normed.copy()
    warped[:, 0] = warped[:, 0] * scale[0] + float(bend_amp) * np.sin(freq[0] * warped[:, 1] + phase[0])
    warped[:, 1] = warped[:, 1] * scale[1] + float(bend_amp) * np.sin(freq[1] * warped[:, 2] + phase[1])
    warped[:, 2] = warped[:, 2] * scale[2] + float(bend_amp) * np.sin(freq[2] * warped[:, 0] + phase[2])
    if float(noise_std) > 0.0:
        warped += rng.normal(0.0, float(noise_std), size=warped.shape).astype(np.float32)
    return (warped * radius + center[None, :]).astype(np.float32, copy=False)


_MIS3D_MODULE = None
_COMPARE_MODULE = None
_URR_MODULE = None


# ---------------------------------------------------------------------------
# Mesh-with-normals helpers
# ---------------------------------------------------------------------------

def _compute_mesh_vertex_normals_np(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex normals by averaging adjacent face normals (area-weighted).

    Parameters
    ----------
    verts : (V, 3) float array
    faces : (F, 3) int array

    Returns
    -------
    normals : (V, 3) float32, unit-length per vertex
    """
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    nv = int(v.shape[0])
    nf = int(f.shape[0])
    if nf < 1 or nv < 3:
        return np.zeros((nv, 3), dtype=np.float32)
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    fn = np.cross(e1, e2)  # (F, 3) area-weighted face normals
    normals = np.zeros((nv, 3), dtype=np.float64)
    for i in range(3):
        np.add.at(normals, f[:, i], fn)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return (normals / norms).astype(np.float32)


def _sample_mesh_with_normals(
    verts: np.ndarray,
    faces: np.ndarray,
    vert_normals: np.ndarray,
    n_points: int,
    seed: int,
) -> np.ndarray:
    """Area-weighted surface sampling with interpolated normals.

    Returns
    -------
    pts6 : (n_points, 6) float32  —  columns 0:3 = xyz, 3:6 = unit normal
    """
    v = np.asarray(verts, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int32)
    n = np.asarray(vert_normals, dtype=np.float32)
    n_pts = int(n_points)
    rng = np.random.default_rng(int(seed))
    nf = int(f.shape[0])
    if nf < 1 or int(v.shape[0]) < 3:
        return np.zeros((n_pts, 6), dtype=np.float32)
    # Face-area weights
    e1 = (v[f[:, 1]] - v[f[:, 0]]).astype(np.float64)
    e2 = (v[f[:, 2]] - v[f[:, 0]]).astype(np.float64)
    area = np.linalg.norm(np.cross(e1, e2), axis=1) * 0.5
    area = np.maximum(area, 1e-12)
    prob = (area / area.sum()).astype(np.float64)
    face_idx = rng.choice(nf, size=n_pts, p=prob)
    # Uniform barycentric coordinates
    r1 = rng.random(n_pts).astype(np.float32)
    r2 = rng.random(n_pts).astype(np.float32)
    sqrt_r1 = np.sqrt(r1)
    b0 = (1.0 - sqrt_r1)[:, None]
    b1 = (sqrt_r1 * (1.0 - r2))[:, None]
    b2 = (sqrt_r1 * r2)[:, None]
    fi = f[face_idx]  # (n_pts, 3)
    pts = (b0 * v[fi[:, 0]] + b1 * v[fi[:, 1]] + b2 * v[fi[:, 2]]).astype(np.float32)
    nrm = (b0 * n[fi[:, 0]] + b1 * n[fi[:, 1]] + b2 * n[fi[:, 2]]).astype(np.float32)
    norms = np.linalg.norm(nrm, axis=1, keepdims=True).astype(np.float32)
    norms = np.where(norms < 1e-7, 1.0, norms)
    nrm = nrm / norms
    return np.concatenate([pts, nrm], axis=1).astype(np.float32)


def _build_heightfield_mesh_canon(
    *,
    frame_bgr: np.ndarray,
    mask_full: np.ndarray,
    bbox: tuple,
    expand_px: int,
    cfg: Any,
    seed: int,
    min_height: float = 1.0,
) -> "dict[str, Any] | None":
    """Build a canonical heightfield mesh with per-vertex normals.

    Returns a dict with keys
        verts_canon  : (V, 3) float32  canonical mesh vertices
        faces        : (F, 3) int32    triangle faces
        vert_normals : (V, 3) float32  unit per-vertex normals
        pts_canon    : (V, 3) float32  same as verts_canon (for compat)
        center       : (3,)  float32
        radius       : float
        bbox_exp     : (x, y, w, h)  int
    or None on failure.
    """
    import cv2 as _cv2
    mis3d = _get_mis3d()
    urr = _get_urr()

    bbox_exp = urr._expand_bbox(tuple(bbox), int(expand_px), frame_bgr.shape[:2])
    roi = mis3d.core.crop(frame_bgr, bbox_exp)
    if roi is None or roi.ndim != 3 or int(roi.shape[0]) < 4 or int(roi.shape[1]) < 4:
        return None
    th_gray, _mask0, _ = mis3d.core.compute_th_gray_truncated(roi, cfg.th_percent, cfg)
    th_gray = np.asarray(th_gray, dtype=np.float32)

    mask_roi = urr._extract_mask_roi(np.asarray(mask_full, dtype=np.uint8), bbox_exp)
    if mask_roi.shape != th_gray.shape:
        mask_roi = _cv2.resize(
            mask_roi, (int(th_gray.shape[1]), int(th_gray.shape[0])),
            interpolation=_cv2.INTER_NEAREST,
        ).astype(np.uint8)
    mask_roi = mis3d._mask_to_filled_u8(mask_roi)
    if mask_roi.shape != th_gray.shape:
        mask_roi = _cv2.resize(
            mask_roi, (int(th_gray.shape[1]), int(th_gray.shape[0])),
            interpolation=_cv2.INTER_NEAREST,
        ).astype(np.uint8)

    # Apply mask: give masked-in pixels a minimum height so they appear in mesh
    th_masked = np.where(mask_roi > 0, np.maximum(th_gray, float(min_height)), 0.0).astype(np.float32)

    verts, faces, _ = mis3d._make_heightfield_mesh_arrays(
        th_gray_truncated=th_masked,
        bbox=tuple(bbox_exp),
        z_scale=float(cfg.mesh_z_scale),
    )
    if int(verts.shape[0]) < 200 or int(faces.shape[0]) < 100:
        return None

    pmin = np.nanmin(verts, axis=0)
    pmax = np.nanmax(verts, axis=0)
    center = (0.5 * (pmin + pmax)).astype(np.float32)
    radius = float(max(1e-6, 0.5 * np.linalg.norm(pmax - pmin)))
    verts_canon = ((verts - center[None, :]) / radius).astype(np.float32)
    vert_normals = _compute_mesh_vertex_normals_np(verts_canon, faces)
    return {
        "verts_canon": verts_canon,
        "faces": np.asarray(faces, dtype=np.int32),
        "vert_normals": vert_normals,
        "pts_canon": verts_canon,  # compatibility alias
        "center": center,
        "radius": radius,
        "bbox_exp": tuple(int(v) for v in bbox_exp),
    }


def _get_mis3d():
    global _MIS3D_MODULE
    if _MIS3D_MODULE is None:
        import MIS_C_F_SIMI_Mesh_Match_Track_3DMatch as mis3d
        _MIS3D_MODULE = mis3d
    return _MIS3D_MODULE


def _get_compare():
    global _COMPARE_MODULE
    if _COMPARE_MODULE is None:
        import compare_chessboard_trackers as compare
        _COMPARE_MODULE = compare
    return _COMPARE_MODULE


def _get_urr():
    global _URR_MODULE
    if _URR_MODULE is None:
        import uvh_rt_regression as urr
        _URR_MODULE = urr
    return _URR_MODULE


class FourierPoseEncoder(nn.Module):
    def __init__(self, in_dim: int = 9, num_bands: int = 6):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_bands = int(max(1, num_bands))
        self.out_dim = int(self.in_dim * (1 + 2 * self.num_bands))
        freq = 2.0 ** torch.arange(self.num_bands, dtype=torch.float32)
        self.register_buffer("freq_bands", freq, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [x]
        for f in self.freq_bands:
            xf = x * float(np.pi) * float(f)
            outs.append(torch.sin(xf))
            outs.append(torch.cos(xf))
        return torch.cat(outs, dim=1)


class PointNetEncoder(nn.Module):
    def __init__(self, in_dim: int = 3, latent_dim: int = 128, hidden_dim: int = 128, local_dim: int = 48):
        super().__init__()
        self.local_mlp_in = nn.Linear(int(in_dim), int(hidden_dim))
        self.local_norm1 = nn.LayerNorm(int(hidden_dim))
        self.local_mlp_mid = nn.Linear(int(hidden_dim), int(hidden_dim))
        self.local_norm2 = nn.LayerNorm(int(hidden_dim))
        self.local_mlp_out = nn.Linear(int(hidden_dim), int(local_dim))
        self.local_norm3 = nn.LayerNorm(int(local_dim))
        self.global_mlp = nn.Sequential(
            nn.Linear(int(local_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(hidden_dim), int(latent_dim)),
        )

    def forward(self, pts: torch.Tensor) -> dict[str, torch.Tensor]:
        h = F.silu(self.local_norm1(self.local_mlp_in(pts)))
        h = h + F.silu(self.local_norm2(self.local_mlp_mid(h)))
        local_feat = F.silu(self.local_norm3(self.local_mlp_out(h)))
        global_feat = self.global_mlp(local_feat)
        global_feat = torch.max(global_feat, dim=1).values
        return {
            "local": local_feat,
            "global": F.normalize(global_feat, dim=1),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        groups = 8 if int(out_ch) >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(int(in_ch), int(out_ch), kernel_size=3, stride=int(stride), padding=1),
            nn.GroupNorm(groups, int(out_ch)),
            nn.SiLU(inplace=True),
            nn.Conv2d(int(out_ch), int(out_ch), kernel_size=3, padding=1),
            nn.GroupNorm(groups, int(out_ch)),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = ConvBlock(int(in_ch) + int(skip_ch), int(out_ch), stride=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def _rot6d_to_matrix_torch(rot_6d: torch.Tensor) -> torch.Tensor:
    a1 = rot_6d[:, 0:3]
    a2 = rot_6d[:, 3:6]
    b1 = F.normalize(a1, dim=1, eps=1e-6)
    b2 = a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)


class PointRTPoseRenderField(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 96,
        point_latent_dim: int = 128,
        point_hidden_dim: int = 128,
        pose_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
        fusion_depth: int = 3,
        point_in_dim: int = 3,
    ):
        super().__init__()
        self.image_size = int(image_size)
        self.point_in_dim = int(point_in_dim)
        self.local_dim = int(max(24, min(64, point_hidden_dim // 2 if point_hidden_dim > 32 else point_hidden_dim)))
        self.point_encoder = PointNetEncoder(
            in_dim=int(point_in_dim),
            latent_dim=int(point_latent_dim),
            hidden_dim=int(point_hidden_dim),
            local_dim=int(self.local_dim),
        )
        self.pose_encoder = FourierPoseEncoder(in_dim=9, num_bands=6)
        pose_in = int(self.pose_encoder.out_dim)
        self.pose_proj = nn.Sequential(
            nn.Linear(pose_in, int(pose_hidden_dim)),
            nn.SiLU(inplace=True),
            nn.Linear(int(pose_hidden_dim), int(point_latent_dim)),
            nn.SiLU(inplace=True),
        )
        fusion_layers: list[nn.Module] = []
        in_dim = int(point_latent_dim * 2)
        hid = int(fusion_hidden_dim)
        for _ in range(int(max(1, fusion_depth - 1))):
            fusion_layers.append(nn.Linear(in_dim, hid))
            fusion_layers.append(nn.SiLU(inplace=True))
            in_dim = hid
        fusion_layers.append(nn.Linear(in_dim, int(point_latent_dim)))
        fusion_layers.append(nn.SiLU(inplace=True))
        self.fusion = nn.Sequential(*fusion_layers)
        proj_ch = int(max(32, point_latent_dim // 2))
        ctx_ch = int(max(16, point_latent_dim // 4))
        self.project_feat = nn.Sequential(
            nn.Linear(int(self.local_dim + 4), int(proj_ch)),
            nn.SiLU(inplace=True),
            nn.Linear(int(proj_ch), int(proj_ch)),
            nn.SiLU(inplace=True),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(int(point_latent_dim), int(ctx_ch)),
            nn.SiLU(inplace=True),
        )
        self.focal = nn.Parameter(torch.tensor(float(self.image_size) * 0.85, dtype=torch.float32))
        self.visibility_gain = nn.Parameter(torch.tensor(4.0, dtype=torch.float32))
        self.occlusion_beta = nn.Parameter(torch.tensor(8.0, dtype=torch.float32))
        self.enc1 = ConvBlock(int(proj_ch + 3 + ctx_ch), 64, stride=1)
        self.enc2 = ConvBlock(64, 96, stride=2)
        self.enc3 = ConvBlock(96, 128, stride=2)
        self.up2 = UpBlock(128, 96, 96)
        self.up1 = UpBlock(96, 64, 64)
        self.head = nn.Conv2d(64, 5, kernel_size=1)

    def _project_points(
        self,
        points: torch.Tensor,
        pose_vec: torch.Tensor,
        point_local: torch.Tensor,
        global_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, n_pts, _ = points.shape
        device = points.device
        img = int(self.image_size)
        hw = int(img * img)
        rot = _rot6d_to_matrix_torch(pose_vec[:, 0:6])
        trans = pose_vec[:, 6:9]
        pts_cam = torch.matmul(points, rot.transpose(1, 2)) + trans.unsqueeze(1)
        z = pts_cam[:, :, 2:3]
        z_safe = torch.clamp(z, min=1e-4)
        fx = torch.clamp(self.focal, min=float(img) * 0.25, max=float(img) * 2.5)
        cx = float(img - 1) * 0.5
        cy = float(img - 1) * 0.5
        u = fx * (pts_cam[:, :, 0:1] / z_safe) + cx
        v = fx * (pts_cam[:, :, 1:2] / z_safe) + cy
        inv_z = 1.0 / z_safe
        proj_input = torch.cat([point_local, pts_cam, inv_z], dim=2)
        proj_feat = self.project_feat(proj_input)
        proj_feat32 = proj_feat.float()
        inv_z_flat = inv_z.squeeze(2).float()
        inv_z_norm = inv_z_flat / torch.clamp(torch.amax(inv_z_flat, dim=1, keepdim=True), min=1e-6)
        exp_arg = torch.clamp(self.visibility_gain.float(), min=0.5, max=6.0) * torch.clamp(inv_z_norm, min=0.0, max=1.0)
        depth_boost = torch.exp(torch.clamp(exp_arg, min=0.0, max=6.0))
        base_weight = torch.clamp(inv_z_flat * depth_boost, min=0.0, max=1e3)
        valid = (z.squeeze(2) > 1e-4) & torch.isfinite(u.squeeze(2)) & torch.isfinite(v.squeeze(2))
        x0 = torch.floor(u.squeeze(2))
        y0 = torch.floor(v.squeeze(2))
        x1 = x0 + 1.0
        y1 = y0 + 1.0
        wx1 = u.squeeze(2) - x0
        wy1 = v.squeeze(2) - y0
        wx0 = 1.0 - wx1
        wy0 = 1.0 - wy1
        batch_offset = (torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1) * int(hw)).expand(bsz, n_pts)
        corners_wf = [
            (x0, y0, wx0 * wy0),
            (x1, y0, wx1 * wy0),
            (x0, y1, wx0 * wy1),
            (x1, y1, wx1 * wy1),
        ]

        # ----------------------------------------------------------------
        # Pass 1: per-pixel minimum depth (detached) for occlusion gating.
        # Using detached z so this path does not block pose_param gradients.
        # ----------------------------------------------------------------
        z_safe_d = z_safe.detach()
        z_min_acc = torch.full((int(bsz * hw), 1), float("inf"), device=device, dtype=torch.float32)
        for (xf, yf, wf) in corners_wf:
            xi = xf.long()
            yi = yf.long()
            ok = valid & (xi >= 0) & (xi < img) & (yi >= 0) & (yi < img) & (wf > 0.0)
            if not bool(torch.any(ok)):
                continue
            idx_ok = ((yi * img + xi).long() + batch_offset).reshape(-1, 1)[ok.reshape(-1)]
            src_z = z_safe_d.reshape(-1, 1)[ok.reshape(-1)]
            if hasattr(z_min_acc, "scatter_reduce_"):
                z_min_acc.scatter_reduce_(0, idx_ok, src_z, reduce="amin", include_self=True)
        z_min_acc = z_min_acc.detach()

        # ----------------------------------------------------------------
        # Pass 2: collect all per-corner contributions into lists, then
        # perform a SINGLE non-inplace scatter_add for each accumulator.
        # This avoids inplace modification of gradient-bearing tensors and
        # allows loss.backward() to flow gradients through pose_vec.
        # ----------------------------------------------------------------
        feat_dim = int(proj_feat.shape[2])
        occ_beta = torch.clamp(self.occlusion_beta.float(), min=1.0, max=16.0)
        all_idx: list[torch.Tensor] = []
        all_feat: list[torch.Tensor] = []
        all_w: list[torch.Tensor] = []
        all_z: list[torch.Tensor] = []
        for (xf, yf, wf) in corners_wf:
            xi = xf.long()
            yi = yf.long()
            ok = valid & (xi >= 0) & (xi < img) & (yi >= 0) & (yi < img) & (wf > 0.0)
            if not bool(torch.any(ok)):
                continue
            ok_flat = ok.reshape(-1)
            idx_ok = ((yi * img + xi).long() + batch_offset).reshape(-1)[ok_flat]
            pixel_min_z = z_min_acc.view(-1)[idx_ok]
            src_z_ok = z_safe.reshape(-1)[ok_flat].float()
            rel_depth = (src_z_ok - pixel_min_z) / torch.clamp(pixel_min_z, min=1e-4)
            visibility = torch.exp(-occ_beta * torch.clamp(rel_depth, min=0.0, max=8.0))
            point_w = wf.reshape(-1)[ok_flat].float() * base_weight.reshape(-1)[ok_flat].float() * visibility
            point_w = torch.nan_to_num(point_w, nan=0.0, posinf=0.0, neginf=0.0)
            all_idx.append(idx_ok)
            all_feat.append(proj_feat32.reshape(-1, feat_dim)[ok_flat] * point_w.unsqueeze(1))
            all_w.append(point_w)
            all_z.append(src_z_ok * point_w)

        if all_idx:
            cat_idx = torch.cat(all_idx, dim=0)           # (M,)
            cat_feat = torch.cat(all_feat, dim=0)          # (M, feat_dim)
            cat_w = torch.cat(all_w, dim=0).unsqueeze(1)   # (M, 1)
            cat_z = torch.cat(all_z, dim=0).unsqueeze(1)   # (M, 1)
            # torch.zeros(...).scatter_add(...) returns a NEW tensor – no inplace op
            # on any gradient-bearing tensor, so autograd can backprop through pose_vec.
            feat_acc = torch.zeros(bsz * hw, feat_dim, device=device, dtype=torch.float32).scatter_add(
                0, cat_idx.unsqueeze(1).expand(-1, feat_dim), cat_feat
            )
            weight_acc = torch.zeros(bsz * hw, 1, device=device, dtype=torch.float32).scatter_add(
                0, cat_idx.unsqueeze(1), cat_w
            )
            z_acc = torch.zeros(bsz * hw, 1, device=device, dtype=torch.float32).scatter_add(
                0, cat_idx.unsqueeze(1), cat_z
            )
        else:
            feat_acc = torch.zeros(bsz * hw, feat_dim, device=device, dtype=torch.float32)
            weight_acc = torch.zeros(bsz * hw, 1, device=device, dtype=torch.float32)
            z_acc = torch.zeros(bsz * hw, 1, device=device, dtype=torch.float32)
        feat_map = feat_acc / torch.clamp(weight_acc, min=1e-6)
        feat_map = feat_map.view(int(bsz), img, img, int(proj_feat.shape[2])).permute(0, 3, 1, 2).contiguous()
        occ_map = (1.0 - torch.exp(-weight_acc)).view(int(bsz), img, img, 1).permute(0, 3, 1, 2).contiguous()
        depth_hint = (z_acc / torch.clamp(weight_acc, min=1e-6)).view(int(bsz), img, img, 1).permute(0, 3, 1, 2).contiguous()
        z_min_map = z_min_acc.view(int(bsz), img, img, 1).permute(0, 3, 1, 2).contiguous()
        z_min_map = torch.where(torch.isfinite(z_min_map), z_min_map, torch.zeros_like(z_min_map))
        depth_hint = depth_hint / torch.clamp(torch.amax(depth_hint, dim=(2, 3), keepdim=True), min=1e-6)
        z_min_hint = z_min_map / torch.clamp(torch.amax(z_min_map, dim=(2, 3), keepdim=True), min=1e-6)
        feat_map = torch.nan_to_num(feat_map, nan=0.0, posinf=0.0, neginf=0.0)
        occ_map = torch.nan_to_num(occ_map, nan=0.0, posinf=1.0, neginf=0.0)
        depth_hint = torch.nan_to_num(depth_hint, nan=0.0, posinf=1.0, neginf=0.0)
        z_min_hint = torch.nan_to_num(z_min_hint, nan=0.0, posinf=1.0, neginf=0.0)
        feat_dtype = proj_feat.dtype
        feat_map = feat_map.to(dtype=feat_dtype)
        occ_map = occ_map.to(dtype=feat_dtype)
        depth_hint = depth_hint.to(dtype=feat_dtype)
        z_min_hint = z_min_hint.to(dtype=feat_dtype)
        ctx_map = global_context[:, :, None, None].expand(-1, -1, img, img).to(dtype=feat_dtype)
        spatial = torch.cat([feat_map, occ_map, depth_hint, z_min_hint, ctx_map], dim=1)
        return spatial, occ_map, depth_hint, z_min_hint

    def forward(self, points: torch.Tensor, pose_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        # points: (B, N, 3) or (B, N, 6).  Only xyz is used for projection;
        # the full feature vector (xyz or xyz+normal) goes to the encoder.
        points_xyz = points[:, :, :3].contiguous()
        point_pack = self.point_encoder(points)
        shape_latent = point_pack["global"]
        point_local = point_pack["local"]
        pose_latent = self.pose_proj(self.pose_encoder(pose_vec))
        fused = self.fusion(torch.cat([shape_latent, pose_latent], dim=1))
        fused = F.normalize(fused, dim=1, eps=1e-6)
        spatial, occ_map, depth_hint, z_min_hint = self._project_points(points_xyz, pose_vec, point_local, self.context_proj(fused))
        x1 = self.enc1(spatial)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        y2 = self.up2(x3, x2)
        y1 = self.up1(y2, x1)
        raw = self.head(y1)
        mask_logits = raw[:, 4:5] + 2.0 * (occ_map - 0.5)
        mask_prob = torch.sigmoid(mask_logits)
        depth = torch.sigmoid(raw[:, 0:1]) * mask_prob
        normal = F.normalize(torch.tanh(raw[:, 1:4]), dim=1, eps=1e-6)
        normal01 = ((normal + 1.0) * 0.5) * mask_prob
        render = torch.cat([depth, normal01], dim=1)
        render = torch.nan_to_num(render, nan=0.0, posinf=1.0, neginf=0.0)
        return {
            "shape_latent": shape_latent,
            "pose_latent": pose_latent,
            "fused_latent": fused,
            "occ_map": occ_map,
            "depth_hint": depth_hint,
            "z_min_hint": z_min_hint,
            "mask_logits": mask_logits,
            "mask_prob": mask_prob,
            "normal": normal,
            "render": render,
        }


def _sample_points(pts: np.ndarray, n_points: int, seed: int) -> np.ndarray:
    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
    if int(arr.shape[0]) <= 0:
        return np.zeros((int(n_points), 3), dtype=np.float32)
    rng = np.random.default_rng(int(seed))
    if int(arr.shape[0]) >= int(n_points):
        idx = rng.choice(int(arr.shape[0]), size=int(n_points), replace=False)
    else:
        idx = rng.choice(int(arr.shape[0]), size=int(n_points), replace=True)
    return np.asarray(arr[idx], dtype=np.float32)


def build_point_rt_renderfield_cache(
    *,
    image_root: str,
    gt_root: str,
    out_root: str,
    dataset_name: str = "",
    val_ratio: float = 0.27,
    test_ratio: float = 0.15,
    seed: int = 42,
    image_size: int = 96,
    n_views: int = 256,
    radius_levels: int = 32,
    radius_min_mult: float = 1.5,
    radius_max_mult: float = 5.0,
    n_reference_frames: int = 8,
    bbox_expand_values: str | list[int] | tuple[int, ...] = "32,64",
    n_points: int = 4096,
    first_frame_only: bool = True,
    n_warp_augs: int = 0,
    warp_scale_std: float = 0.04,
    warp_bend_amp: float = 0.03,
    warp_noise_std: float = 0.005,
    force_rebuild: bool = False,
    use_mesh_normals: bool = True,
) -> str:
    urr = _get_urr()
    mis3d = _get_mis3d()
    compare = _get_compare()
    urr._set_seed(int(seed))
    image_root_p = Path(image_root)
    gt_root_p = Path(gt_root)
    out_root_p = Path(out_root)
    seq_out_root = out_root_p / "point_rt_renderfield_cache"
    seq_out_root.mkdir(parents=True, exist_ok=True)

    names = urr._discover_dataset_names(image_root_p, gt_root_p)
    dataset_name_s = str(dataset_name or "").strip()
    if dataset_name_s:
        names = [n for n in names if str(n) == dataset_name_s]
        if not names:
            raise RuntimeError(f"未找到指定数据集: {dataset_name_s}")
    rng = np.random.default_rng(int(seed))
    single_sequence_mode = bool(len(names) == 1)
    cfg = urr._build_cfg(int(image_size))
    expand_values = urr._parse_expand_values(bbox_expand_values)
    dist_mults = np.linspace(float(radius_min_mult), float(radius_max_mult), num=int(max(2, radius_levels)), dtype=np.float32).tolist()
    if single_sequence_mode:
        val_names: set[str] = set()
        test_names: set[str] = set()
    else:
        shuffled = list(names)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(float(len(shuffled)) * float(val_ratio))))
        n_test = max(1, int(round(float(len(shuffled)) * float(test_ratio))))
        n_test = min(max(1, n_test), max(1, len(shuffled) - n_val - 1)) if len(shuffled) >= 3 else 0
        test_names = set(shuffled[:n_test]) if n_test > 0 else set()
        val_names = set(shuffled[n_test:n_test + n_val])

    sequences: list[RenderFieldSequence] = []
    for name in names:
        split = "test" if name in test_names else ("val" if name in val_names else "train")
        print(f"[PointRTField] prepare sequence={name} split={split}", flush=True)
        frames_dir = urr._resolve_image_dir(image_root_p / name)
        gt_dir = urr._resolve_gt_mask_dir(gt_root_p / name)
        if frames_dir is None or gt_dir is None:
            print(f"[PointRTField] skip sequence={name} reason=missing_dirs", flush=True)
            continue
        seq_dir = seq_out_root / name
        seq_dir.mkdir(parents=True, exist_ok=True)
        seq_npz = seq_dir / "renderfield_sequence.npz"
        train_npz = seq_dir / "renderfield_train.npz"
        val_npz = seq_dir / "renderfield_val.npz"
        test_npz = seq_dir / "renderfield_test.npz"
        expected_point_dim = 6 if bool(use_mesh_normals) else 3
        if single_sequence_mode and (not bool(force_rebuild)) and train_npz.exists() and val_npz.exists() and test_npz.exists():
            train_data = np.load(str(train_npz), allow_pickle=False)
            val_data = np.load(str(val_npz), allow_pickle=False)
            test_data = np.load(str(test_npz), allow_pickle=False)
            n_train = int(np.asarray(train_data["renders"]).shape[0])
            n_val_final = int(np.asarray(val_data["renders"]).shape[0])
            n_test_final = int(np.asarray(test_data["renders"]).shape[0])
            n_pc = int(np.asarray(train_data["point_clouds"]).shape[0]) if ("point_clouds" in getattr(train_data, "files", [])) else 0
            existing_dim = int(np.asarray(train_data["point_clouds"]).shape[2]) if ("point_clouds" in getattr(train_data, "files", []) and np.asarray(train_data["point_clouds"]).ndim == 3) else 3
            if n_train > 0 and n_val_final > 0 and n_test_final > 0 and existing_dim == expected_point_dim:
                sequences.append(RenderFieldSequence(name=f"{name}_train", split="train", npz_path=str(train_npz), n_samples=n_train, n_pointclouds=n_pc, frames_dir=str(frames_dir), gt_dir=str(gt_dir)))
                sequences.append(RenderFieldSequence(name=f"{name}_val", split="val", npz_path=str(val_npz), n_samples=n_val_final, n_pointclouds=n_pc, frames_dir=str(frames_dir), gt_dir=str(gt_dir)))
                sequences.append(RenderFieldSequence(name=f"{name}_test", split="test", npz_path=str(test_npz), n_samples=n_test_final, n_pointclouds=n_pc, frames_dir=str(frames_dir), gt_dir=str(gt_dir)))
                continue
            elif existing_dim != expected_point_dim:
                print(f"[PointRTField] sequence={name} point_dim mismatch (cached={existing_dim} expected={expected_point_dim}), rebuilding", flush=True)
        if (not bool(force_rebuild)) and seq_npz.exists():
            data = np.load(str(seq_npz), allow_pickle=False)
            n_samples = int(np.asarray(data["renders"]).shape[0])
            n_pc = int(np.asarray(data["point_clouds"]).shape[0]) if ("point_clouds" in getattr(data, "files", [])) else 0
            existing_dim = int(np.asarray(data["point_clouds"]).shape[2]) if ("point_clouds" in getattr(data, "files", []) and np.asarray(data["point_clouds"]).ndim == 3) else 3
            if n_samples > 0 and existing_dim == expected_point_dim:
                sequences.append(RenderFieldSequence(name=name, split=split, npz_path=str(seq_npz), n_samples=n_samples, n_pointclouds=n_pc, frames_dir=str(frames_dir), gt_dir=str(gt_dir)))
                continue
            elif existing_dim != expected_point_dim:
                print(f"[PointRTField] sequence={name} point_dim mismatch (cached={existing_dim} expected={expected_point_dim}), rebuilding", flush=True)

        gt_frames = compare._load_gt_from_mask_dir(frames_dir, gt_dir)
        if len(gt_frames) < 1:
            print(f"[PointRTField] skip sequence={name} reason=no_gt_frames", flush=True)
            continue
        ref_indices = [0] if bool(first_frame_only) else urr._select_reference_indices(len(gt_frames), int(max(1, n_reference_frames)))
        print(f"[PointRTField] sequence={name} gt_frames={len(gt_frames)} refs={len(ref_indices)}", flush=True)
        point_clouds: list[np.ndarray] = []
        renders: list[np.ndarray] = []
        pose_vecs: list[np.ndarray] = []
        pc_indices: list[int] = []
        ref_frame_indices: list[int] = []
        expand_px_list: list[int] = []
        warp_variant_list: list[int] = []

        for ref_pick_idx in ref_indices:
            ref_gt = gt_frames[int(ref_pick_idx)]
            print(f"[PointRTField] sequence={name} ref_frame={int(ref_gt.frame_idx)}", flush=True)
            ref_bgr = mis3d.cv2.imread(str(ref_gt.image_path), mis3d.cv2.IMREAD_COLOR) if hasattr(mis3d, "cv2") else None
            if ref_bgr is None:
                import cv2
                ref_bgr = cv2.imread(str(ref_gt.image_path), cv2.IMREAD_COLOR)
            if ref_bgr is None:
                continue
            for expand_px in expand_values:
                print(f"[PointRTField] sequence={name} ref_frame={int(ref_gt.frame_idx)} expand={int(expand_px)}", flush=True)
                # ------------------------------------------------------------------
                # Build reference point representation
                # When use_mesh_normals=True  -> heightfield mesh → (N,6) xyz+normal
                # Otherwise (legacy)          -> uvh scatter    → (N,3) xyz only
                # ------------------------------------------------------------------
                mesh_pack: "dict[str, Any] | None" = None
                if bool(use_mesh_normals):
                    mesh_pack = _build_heightfield_mesh_canon(
                        frame_bgr=ref_bgr,
                        mask_full=np.asarray(ref_gt.mask, dtype=np.uint8),
                        bbox=tuple(ref_gt.bbox),
                        expand_px=int(expand_px),
                        cfg=cfg,
                        seed=int(seed) + int(ref_gt.frame_idx) + int(expand_px),
                    )
                    if mesh_pack is None:
                        print(f"[PointRTField] sequence={name} ref_frame={int(ref_gt.frame_idx)} expand={int(expand_px)} mesh=failed, fallback to pointcloud", flush=True)

                # Fall back to legacy if mesh build failed or not requested
                ref_pack = urr._prepare_reference_points(
                    frame_bgr=ref_bgr,
                    mask_full=np.asarray(ref_gt.mask, dtype=np.uint8),
                    bbox=tuple(ref_gt.bbox),
                    expand_px=int(expand_px),
                    cfg=cfg,
                    seed=int(seed) + int(ref_gt.frame_idx) + int(expand_px),
                )
                if not isinstance(ref_pack, dict):
                    continue
                # 视角库 / 深度渲染用的稠密 xyz：须与下面存入 NPZ 的 mesh 采样同源，否则 warp 后
                # 「渲染目标」与「网络输入点云」几何不一致（旧逻辑曾 warp pts_canon 却从 warped mesh 采样）。
                if bool(use_mesh_normals) and mesh_pack is not None:
                    geom_for_views = np.asarray(mesh_pack["verts_canon"], dtype=np.float32)
                else:
                    geom_for_views = np.asarray(ref_pack["pts_canon"], dtype=np.float32)

                # Build warp variants（与 mesh 分支 verts_w 使用相同 seed 公式）
                warp_variants_xyz: list[np.ndarray] = [geom_for_views]
                for warp_idx in range(int(max(0, n_warp_augs))):
                    warp_seed = int(seed) + int(ref_gt.frame_idx) * 100 + int(expand_px) * 10 + int(warp_idx) + 1
                    warp_variants_xyz.append(
                        _warp_points_np(
                            geom_for_views,
                            seed=warp_seed,
                            scale_std=float(warp_scale_std),
                            bend_amp=float(warp_bend_amp),
                            noise_std=float(warp_noise_std),
                        )
                    )
                for warp_variant_idx, pts_variant in enumerate(warp_variants_xyz):
                    pc_idx = len(point_clouds)
                    # ----------------------------------------------------------
                    # Store point cloud: (N,6) when use_mesh_normals, else (N,3)
                    # ----------------------------------------------------------
                    if bool(use_mesh_normals) and mesh_pack is not None:
                        if warp_variant_idx == 0:
                            # Base: sample from the canonical mesh with normals
                            pts6 = _sample_mesh_with_normals(
                                mesh_pack["verts_canon"],
                                mesh_pack["faces"],
                                mesh_pack["vert_normals"],
                                int(n_points),
                                int(seed) + pc_idx,
                            )
                        else:
                            # Warp: deform mesh vertices, recompute normals, resample
                            verts_w = _warp_points_np(
                                mesh_pack["verts_canon"],
                                seed=int(seed) + int(ref_gt.frame_idx) * 100 + int(expand_px) * 10 + int(warp_variant_idx - 1) + 1,
                                scale_std=float(warp_scale_std),
                                bend_amp=float(warp_bend_amp),
                                noise_std=float(warp_noise_std),
                            )
                            vnormals_w = _compute_mesh_vertex_normals_np(verts_w, mesh_pack["faces"])
                            pts6 = _sample_mesh_with_normals(
                                verts_w,
                                mesh_pack["faces"],
                                vnormals_w,
                                int(n_points),
                                int(seed) + pc_idx,
                            )
                        point_clouds.append(pts6.astype(np.float32))
                    else:
                        point_clouds.append(_sample_points(np.asarray(pts_variant, dtype=np.float32), n_points=int(n_points), seed=int(seed) + pc_idx))
                    ref_root = seq_dir / f"ref_{int(ref_gt.frame_idx):04d}_exp{int(expand_px)}_warp{int(warp_variant_idx)}"
                    lib, _meta = mis3d.build_or_load_view_library(
                        out_root=str(ref_root),
                        pts_w=np.asarray(pts_variant, dtype=np.float32),
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
                        print(
                            f"[PointRTField] sequence={name} ref_frame={int(ref_gt.frame_idx)} expand={int(expand_px)} warp={int(warp_variant_idx)} lib=none",
                            flush=True,
                        )
                        continue
                    print(
                        f"[PointRTField] sequence={name} ref_frame={int(ref_gt.frame_idx)} expand={int(expand_px)} warp={int(warp_variant_idx)} views={int(np.asarray(lib.T_w2c).shape[0])}",
                        flush=True,
                    )
                    for i in range(int(np.asarray(lib.T_w2c).shape[0])):
                        T = np.asarray(lib.T_w2c[i], dtype=np.float32).reshape(4, 4)
                        mask_u8, depth = mis3d._render_mask_depth_from_points(
                            pts_w=np.asarray(pts_variant, dtype=np.float32),
                            T_w2c=T,
                            K=np.asarray(lib.K, dtype=np.float32),
                            image_wh=tuple(lib.image_wh),
                        )
                        if int(np.count_nonzero(mask_u8)) <= 20:
                            continue
                        render, _contour_rgb, _normal = urr._build_render_input(mask_u8, depth, tuple(lib.image_wh))
                        renders.append(np.asarray(render, dtype=np.float32))
                        pose_vecs.append(urr._transform_to_pose_vec(T))
                        pc_indices.append(int(pc_idx))
                        ref_frame_indices.append(int(ref_gt.frame_idx))
                        expand_px_list.append(int(expand_px))
                        warp_variant_list.append(int(warp_variant_idx))

        if not renders:
            continue

        renders_arr = np.asarray(renders, dtype=np.float32)
        pose_vecs_arr = np.asarray(pose_vecs, dtype=np.float32)
        point_clouds_arr = np.asarray(point_clouds, dtype=np.float32)
        pc_indices_arr = np.asarray(pc_indices, dtype=np.int32)
        ref_frame_arr = np.asarray(ref_frame_indices, dtype=np.int32)
        expand_arr = np.asarray(expand_px_list, dtype=np.int32)
        warp_arr = np.asarray(warp_variant_list, dtype=np.int32)

        def _write_npz(path_local: Path, idx_sel: np.ndarray) -> int:
            np.savez_compressed(
                path_local,
                renders=np.asarray(renders_arr[idx_sel], dtype=np.float32),
                pose_vecs=np.asarray(pose_vecs_arr[idx_sel], dtype=np.float32),
                point_cloud_indices=np.asarray(pc_indices_arr[idx_sel], dtype=np.int32),
                point_clouds=np.asarray(point_clouds_arr, dtype=np.float32),
                ref_frame_indices=np.asarray(ref_frame_arr[idx_sel], dtype=np.int32),
                expand_px=np.asarray(expand_arr[idx_sel], dtype=np.int32),
                warp_variant=np.asarray(warp_arr[idx_sel], dtype=np.int32),
            )
            return int(idx_sel.size)

        if single_sequence_mode:
            idx_all = np.arange(int(renders_arr.shape[0]), dtype=np.int32)
            rng_local = np.random.default_rng(int(seed))
            rng_local.shuffle(idx_all)
            total_n = int(idx_all.size)
            n_test_local = max(1, int(round(float(total_n) * float(test_ratio))))
            n_val_local = max(1, int(round(float(total_n) * float(val_ratio))))
            n_test_local = min(max(1, n_test_local), max(1, total_n - 2))
            n_val_local = min(max(1, n_val_local), max(1, total_n - n_test_local - 1))
            if (n_test_local + n_val_local) >= total_n:
                n_val_local = max(1, total_n - n_test_local - 1)
            test_idx = np.sort(idx_all[:n_test_local]).astype(np.int32)
            val_idx = np.sort(idx_all[n_test_local:n_test_local + n_val_local]).astype(np.int32)
            train_idx = np.sort(idx_all[n_test_local + n_val_local:]).astype(np.int32)
            n_train = _write_npz(train_npz, train_idx)
            n_val_final = _write_npz(val_npz, val_idx)
            n_test_final = _write_npz(test_npz, test_idx)
            sequences.append(RenderFieldSequence(name=f"{name}_train", split="train", npz_path=str(train_npz), n_samples=n_train, n_pointclouds=int(point_clouds_arr.shape[0]), frames_dir=str(frames_dir), gt_dir=str(gt_dir)))
            sequences.append(RenderFieldSequence(name=f"{name}_val", split="val", npz_path=str(val_npz), n_samples=n_val_final, n_pointclouds=int(point_clouds_arr.shape[0]), frames_dir=str(frames_dir), gt_dir=str(gt_dir)))
            sequences.append(RenderFieldSequence(name=f"{name}_test", split="test", npz_path=str(test_npz), n_samples=n_test_final, n_pointclouds=int(point_clouds_arr.shape[0]), frames_dir=str(frames_dir), gt_dir=str(gt_dir)))
            print(f"[PointRTField] built single-sequence cache: {name} train={n_train} val={n_val_final} test={n_test_final} pcs={int(point_clouds_arr.shape[0])}", flush=True)
        else:
            _write_npz(seq_npz, np.arange(int(renders_arr.shape[0]), dtype=np.int32))
            sequences.append(RenderFieldSequence(name=name, split=split, npz_path=str(seq_npz), n_samples=int(renders_arr.shape[0]), n_pointclouds=int(point_clouds_arr.shape[0]), frames_dir=str(frames_dir), gt_dir=str(gt_dir)))
            print(f"[PointRTField] built sequence cache: {name} samples={int(renders_arr.shape[0])} pcs={int(point_clouds_arr.shape[0])}", flush=True)

    manifest = {
        "image_root": str(image_root_p),
        "gt_root": str(gt_root_p),
        "dataset_name": dataset_name_s,
        "used_datasets": [str(n) for n in names],
        "n_datasets": int(len(names)),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "seed": int(seed),
        "image_size": int(image_size),
        "n_views": int(n_views),
        "radius_levels": int(radius_levels),
        "radius_min_mult": float(radius_min_mult),
        "radius_max_mult": float(radius_max_mult),
        "n_reference_frames": int(n_reference_frames),
        "first_frame_only": bool(first_frame_only),
        "bbox_expand_values": [int(v) for v in expand_values],
        "n_points": int(n_points),
        "n_warp_augs": int(n_warp_augs),
        "warp_scale_std": float(warp_scale_std),
        "warp_bend_amp": float(warp_bend_amp),
        "warp_noise_std": float(warp_noise_std),
        "use_mesh_normals": bool(use_mesh_normals),
        "point_in_dim": 6 if bool(use_mesh_normals) else 3,
        "sequences": [seq.__dict__ for seq in sequences],
    }
    manifest_path = out_root_p / "point_rt_renderfield_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(manifest_path)


class PointRTRenderFieldDataset(Dataset):
    def __init__(self, manifest_path: str, split: str, seed: int = 42):
        self.manifest_path = str(manifest_path)
        self.split = str(split)
        self.seed = int(seed)
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        self.sequences: list[dict[str, Any]] = []
        self.samples: list[tuple[int, int]] = []
        point_in_dim_expected = int(self.manifest.get("point_in_dim", 3) or 3)
        for item in self.manifest.get("sequences", []):
            if str(item.get("split", "")) != self.split:
                continue
            data = np.load(str(item["npz_path"]), allow_pickle=False)
            renders = np.asarray(data["renders"], dtype=np.float32)
            if int(renders.ndim) == 4 and int(renders.shape[1]) == 5:
                renders = np.asarray(renders[:, [0, 2, 3, 4], :, :], dtype=np.float32)
            pcs = np.asarray(data["point_clouds"], dtype=np.float32)
            # Validate cached point_cloud dimensions vs manifest's point_in_dim
            if pcs.ndim == 3 and int(pcs.shape[2]) != point_in_dim_expected:
                raise RuntimeError(
                    f"缓存 NPZ 维度不一致: sequence={item.get('name','')} "
                    f"point_clouds.shape={pcs.shape} (dim={pcs.shape[2]}) "
                    f"但 manifest point_in_dim={point_in_dim_expected}。\n"
                    "缓存已过时，请重新运行并加上 --force-rebuild-cache 1"
                )
            seq = {
                "name": str(item["name"]),
                "renders": renders,
                "pose_vecs": np.asarray(data["pose_vecs"], dtype=np.float32),
                "point_cloud_indices": np.asarray(data["point_cloud_indices"], dtype=np.int32),
                "point_clouds": pcs,
                "warp_variant": np.asarray(data["warp_variant"], dtype=np.int32) if ("warp_variant" in getattr(data, "files", [])) else np.zeros((int(renders.shape[0]),), dtype=np.int32),
            }
            seq_idx = len(self.sequences)
            self.sequences.append(seq)
            for sample_idx in range(int(seq["renders"].shape[0])):
                self.samples.append((seq_idx, sample_idx))

    def __len__(self) -> int:
        return int(len(self.samples))

    def __getitem__(self, index: int) -> dict[str, Any]:
        seq_idx, sample_idx = self.samples[int(index)]
        seq = self.sequences[int(seq_idx)]
        pc_idx = int(seq["point_cloud_indices"][sample_idx])
        return {
            "points": torch.from_numpy(np.asarray(seq["point_clouds"][pc_idx], dtype=np.float32)),
            "pose_vec": torch.from_numpy(np.asarray(seq["pose_vecs"][sample_idx], dtype=np.float32)),
            "render_gt": torch.from_numpy(np.asarray(seq["renders"][sample_idx], dtype=np.float32)),
            "seq_idx": int(seq_idx),
            "warp_variant": int(seq["warp_variant"][sample_idx]),
        }


def summarize_renderfield_manifest(manifest_path: str) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    sequences = manifest.get("sequences", [])
    summary: dict[str, Any] = {
        "dataset_name": manifest.get("dataset_name", ""),
        "used_datasets": list(manifest.get("used_datasets", [])),
        "n_datasets": int(manifest.get("n_datasets", 0) or 0),
        "image_size": int(manifest.get("image_size", 0) or 0),
        "n_views": int(manifest.get("n_views", 0) or 0),
        "n_reference_frames": int(manifest.get("n_reference_frames", 0) or 0),
        "first_frame_only": bool(manifest.get("first_frame_only", False)),
        "val_ratio": float(manifest.get("val_ratio", 0.0) or 0.0),
        "test_ratio": float(manifest.get("test_ratio", 0.0) or 0.0),
        "bbox_expand_values": list(manifest.get("bbox_expand_values", [])),
        "n_points": int(manifest.get("n_points", 0) or 0),
        "n_warp_augs": int(manifest.get("n_warp_augs", 0) or 0),
        "warp_scale_std": float(manifest.get("warp_scale_std", 0.0) or 0.0),
        "warp_bend_amp": float(manifest.get("warp_bend_amp", 0.0) or 0.0),
        "warp_noise_std": float(manifest.get("warp_noise_std", 0.0) or 0.0),
        "use_mesh_normals": bool(manifest.get("use_mesh_normals", False)),
        "point_in_dim": int(manifest.get("point_in_dim", 3) or 3),
        "train_sequences": 0,
        "val_sequences": 0,
        "test_sequences": 0,
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
        "train_pointclouds": 0,
        "val_pointclouds": 0,
        "test_pointclouds": 0,
    }
    for item in sequences:
        split = str(item.get("split", ""))
        if split == "train":
            summary["train_sequences"] += 1
            summary["train_samples"] += int(item.get("n_samples", 0) or 0)
            summary["train_pointclouds"] += int(item.get("n_pointclouds", 0) or 0)
        elif split == "val":
            summary["val_sequences"] += 1
            summary["val_samples"] += int(item.get("n_samples", 0) or 0)
            summary["val_pointclouds"] += int(item.get("n_pointclouds", 0) or 0)
        elif split == "test":
            summary["test_sequences"] += 1
            summary["test_samples"] += int(item.get("n_samples", 0) or 0)
            summary["test_pointclouds"] += int(item.get("n_pointclouds", 0) or 0)
    return summary


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sum(x * mask) / torch.clamp(torch.sum(mask), min=1.0)


def _tensor_img_to_u8(x: torch.Tensor, kind: str) -> np.ndarray:
    arr = np.nan_to_num(x.detach().cpu().float().numpy(), nan=0.0, posinf=1.0, neginf=0.0)
    if kind == "depth":
        img = np.clip(arr[0], 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
        return np.repeat(img[:, :, None], 3, axis=2)
    if kind == "mask":
        img = np.clip(arr[0], 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
        return np.repeat(img[:, :, None], 3, axis=2)
    if kind == "normal":
        img = np.clip(np.transpose(arr, (1, 2, 0)), 0.0, 1.0)
        return (img * 255.0).astype(np.uint8)
    raise ValueError(f"未知图像类型: {kind}")


@torch.no_grad()
def _save_render_previews(
    *,
    model: PointRTPoseRenderField,
    dataset: PointRTRenderFieldDataset,
    device: torch.device,
    out_dir: Path,
    epoch: int,
    max_items: int = 4,
) -> None:
    import cv2

    if len(dataset) <= 0 or int(max_items) <= 0:
        return
    preview_dir = out_dir / "render_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    n_items = min(int(max_items), len(dataset))
    for idx in range(n_items):
        sample = dataset[idx]
        pts = sample["points"].unsqueeze(0).to(device)
        pose = sample["pose_vec"].unsqueeze(0).to(device)
        gt = sample["render_gt"]
        out = model(pts, pose)
        pred = out["render"][0].detach().cpu()
        pred_mask = out["mask_prob"][0].detach().cpu()
        occ_map = out["occ_map"][0].detach().cpu()
        depth_hint = out["depth_hint"][0].detach().cpu()
        z_min_hint = out["z_min_hint"][0].detach().cpu()
        gt_depth = gt[0:1]
        gt_normal = gt[1:4]
        gt_mask = (gt_depth > 1e-6).float()
        pred_depth_img = _tensor_img_to_u8(pred[0:1], "depth")
        gt_depth_img = _tensor_img_to_u8(gt_depth, "depth")
        pred_normal_img = _tensor_img_to_u8(pred[1:4], "normal")
        gt_normal_img = _tensor_img_to_u8(gt_normal, "normal")
        pred_mask_img = _tensor_img_to_u8(pred_mask, "mask")
        gt_mask_img = _tensor_img_to_u8(gt_mask, "mask")
        occ_img = _tensor_img_to_u8(occ_map, "mask")
        depth_hint_img = _tensor_img_to_u8(depth_hint, "depth")
        z_min_hint_img = _tensor_img_to_u8(z_min_hint, "depth")
        row1 = np.concatenate([gt_depth_img, pred_depth_img, gt_mask_img], axis=1)
        row2 = np.concatenate([gt_normal_img, pred_normal_img, pred_mask_img], axis=1)
        row3 = np.concatenate([occ_img, depth_hint_img, z_min_hint_img], axis=1)
        canvas = np.concatenate([row1, row2, row3], axis=0)
        cv2.putText(canvas, f"epoch={int(epoch)} sample={int(idx)}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        save_path = preview_dir / f"epoch_{int(epoch):04d}_sample_{int(idx):02d}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _ssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Compute 1 - SSIM as a loss (higher = worse). Expects (B,C,H,W) inputs."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window_size // 2
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=pad)
    sigma_pred_sq = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_pred * mu_pred
    sigma_target_sq = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_target * mu_target
    sigma_cross = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_pred * mu_target
    ssim_map = ((2.0 * mu_pred * mu_target + c1) * (2.0 * sigma_cross + c2)) / (
        (mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred_sq + sigma_target_sq + c2) + 1e-8
    )
    return 1.0 - ssim_map.mean()


def _render_loss(
    pred_out: dict[str, torch.Tensor],
    gt_render: torch.Tensor,
    *,
    depth_w: float = 1.0,
    normal_w: float = 1.0,
    mask_w: float = 0.25,
    ssim_w: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_depth = gt_render[:, 0:1]
    gt_normal = torch.clamp(gt_render[:, 1:4] * 2.0 - 1.0, -1.0, 1.0)
    gt_mask = (gt_depth > 1e-6).float()
    pred_render = torch.nan_to_num(pred_out["render"], nan=0.0, posinf=1.0, neginf=0.0)
    pred_depth = pred_render[:, 0:1]
    pred_normal = F.normalize(torch.nan_to_num(pred_out["normal"], nan=0.0, posinf=0.0, neginf=0.0), dim=1, eps=1e-6)
    pred_mask_logits = torch.nan_to_num(pred_out["mask_logits"], nan=0.0, posinf=20.0, neginf=-20.0)
    depth_map = F.smooth_l1_loss(pred_depth, gt_depth, reduction="none")
    depth_loss = _masked_mean(depth_map, gt_mask)
    ncos = torch.sum(F.normalize(pred_normal, dim=1) * F.normalize(gt_normal, dim=1), dim=1, keepdim=True)
    normal_map = 1.0 - torch.clamp(ncos, -1.0, 1.0)
    normal_loss = _masked_mean(normal_map, gt_mask)
    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_mask)
    total = float(depth_w) * depth_loss + float(normal_w) * normal_loss + float(mask_w) * mask_loss
    if ssim_w > 0.0:
        gt_depth_masked = gt_depth * gt_mask
        pred_depth_masked = pred_depth * gt_mask
        gt_norm01 = ((gt_normal + 1.0) * 0.5) * gt_mask
        pred_norm01 = ((pred_normal + 1.0) * 0.5) * gt_mask
        ssim_depth = _ssim_loss(pred_depth_masked, gt_depth_masked)
        ssim_normal = _ssim_loss(pred_norm01, gt_norm01)
        total = total + float(ssim_w) * (ssim_depth + ssim_normal)
    return total, depth_loss, normal_loss, mask_loss


@torch.no_grad()
def predict_render_from_pose(
    *,
    model: PointRTPoseRenderField,
    points: torch.Tensor,
    pose_vec: torch.Tensor,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    urr = _get_urr()
    device_t = device if isinstance(device, torch.device) else urr._resolve_device(str(device))
    model.eval()
    pts = points.to(device_t, non_blocking=True)
    pose = pose_vec.to(device_t, non_blocking=True)
    if pts.ndim == 2:
        pts = pts.unsqueeze(0)
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)
    out = model(pts, pose)
    return out["render"].detach().cpu()


def optimize_pose_from_render_observation(
    *,
    model: PointRTPoseRenderField,
    points: torch.Tensor,
    render_obs: torch.Tensor,
    init_pose_vec: torch.Tensor,
    steps: int = 120,
    lr: float = 1e-2,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    urr = _get_urr()
    device_t = device if isinstance(device, torch.device) else urr._resolve_device(str(device))
    model.eval()
    pts = points.to(device_t)
    obs = render_obs.to(device_t)
    pose = init_pose_vec.to(device_t)
    if pts.ndim == 2:
        pts = pts.unsqueeze(0)
    if obs.ndim == 3:
        obs = obs.unsqueeze(0)
    if pose.ndim == 1:
        pose = pose.unsqueeze(0)
    pose_param = nn.Parameter(pose.clone())
    optimizer = torch.optim.Adam([pose_param], lr=float(lr))
    best_loss = float("inf")
    best_pose = pose.detach().clone()
    history: list[float] = []
    for _ in range(int(max(1, steps))):
        optimizer.zero_grad(set_to_none=True)
        out = model(pts, pose_param)
        loss, _depth_loss, _normal_loss, _mask_loss = _render_loss(out, obs)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        history.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_pose = pose_param.detach().clone()
    return {
        "best_pose_vec": best_pose.detach().cpu(),
        "best_loss": float(best_loss),
        "history": history,
    }


def load_point_rt_renderfield_checkpoint(
    checkpoint_path: str,
    *,
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    urr = _get_urr()
    device_t = device if isinstance(device, torch.device) else urr._resolve_device(str(device))
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    model_cfg = dict(ckpt.get("model_config", {})) if isinstance(ckpt, dict) else {}
    model = PointRTPoseRenderField(
        image_size=int(model_cfg.get("image_size", 96) or 96),
        point_latent_dim=int(model_cfg.get("point_latent_dim", 128) or 128),
        point_hidden_dim=int(model_cfg.get("point_hidden_dim", 128) or 128),
        pose_hidden_dim=int(model_cfg.get("pose_hidden_dim", 128) or 128),
        fusion_hidden_dim=int(model_cfg.get("fusion_hidden_dim", 256) or 256),
        fusion_depth=int(model_cfg.get("fusion_depth", 3) or 3),
        point_in_dim=int(model_cfg.get("point_in_dim", 3) or 3),
    ).to(device_t)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return {
        "model": model,
        "checkpoint": ckpt,
        "manifest_path": str(ckpt.get("manifest_path", "")) if isinstance(ckpt, dict) else "",
        "device": device_t,
    }


def _axis_angle_to_matrix_np(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_v = np.asarray(axis, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(axis_v))
    if norm <= 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z = (axis_v / norm).astype(np.float32).tolist()
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    C = 1.0 - c
    return np.asarray(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float32,
    )


def _perturb_pose_vec_np(
    pose_vec: np.ndarray,
    *,
    rng: np.random.Generator,
    rot_noise_deg: float,
    trans_noise_std: float,
) -> np.ndarray:
    urr = _get_urr()
    pose_t = torch.from_numpy(np.asarray(pose_vec, dtype=np.float32)).unsqueeze(0)
    T = urr._pose_vec_to_transform_tensor(pose_t).detach().cpu().numpy()[0].astype(np.float32)
    R = np.asarray(T[:3, :3], dtype=np.float32)
    t = np.asarray(T[:3, 3], dtype=np.float32)
    axis = rng.normal(0.0, 1.0, size=(3,)).astype(np.float32)
    angle = np.deg2rad(float(rng.normal(0.0, float(max(0.0, rot_noise_deg)))))
    dR = _axis_angle_to_matrix_np(axis, float(angle))
    dt = rng.normal(0.0, float(max(0.0, trans_noise_std)), size=(3,)).astype(np.float32)
    T_new = np.eye(4, dtype=np.float32)
    T_new[:3, :3] = np.asarray(dR @ R, dtype=np.float32)
    T_new[:3, 3] = np.asarray(t + dt, dtype=np.float32)
    return urr._transform_to_pose_vec(T_new)


def _pose_metrics(
    pred_pose: torch.Tensor,
    gt_pose: torch.Tensor,
) -> dict[str, float]:
    urr = _get_urr()
    pred_R = urr._rot6d_to_matrix(pred_pose[:, :6])
    gt_R = urr._rot6d_to_matrix(gt_pose[:, :6])
    rot_deg = float(urr._rotation_geodesic_deg(pred_R, gt_R).detach().cpu().item())
    trans_err = float(torch.linalg.norm(pred_pose[:, 6:9] - gt_pose[:, 6:9], dim=1).detach().cpu().item())
    return {"rot_deg": rot_deg, "trans_err": trans_err}


def evaluate_pose_recovery_from_renderfield(
    *,
    checkpoint_path: str,
    manifest_path: str = "",
    split: str = "val",
    max_items: int = 128,
    seed: int = 42,
    rot_noise_deg: float = 20.0,
    trans_noise_std: float = 0.08,
    optimize_steps: int = 120,
    optimize_lr: float = 1e-2,
    device: str = "cuda",
    out_json: str = "",
    progress_every: int = 1,
) -> dict[str, Any]:
    urr = _get_urr()
    pack = load_point_rt_renderfield_checkpoint(checkpoint_path, device=device)
    model = pack["model"]
    device_t = pack["device"]
    manifest_final = str(manifest_path or pack.get("manifest_path", "")).strip()
    if not manifest_final:
        raise RuntimeError("缺少 manifest_path，无法进行 pose recovery 评估")
    dataset = PointRTRenderFieldDataset(manifest_path=manifest_final, split=str(split), seed=int(seed))
    if len(dataset) <= 0:
        raise RuntimeError(f"split={split} 数据集为空，无法评估 pose recovery")
    rng = np.random.default_rng(int(seed))
    n_eval = min(int(max_items), len(dataset))
    rows: list[dict[str, Any]] = []
    init_rot_list: list[float] = []
    init_trans_list: list[float] = []
    final_rot_list: list[float] = []
    final_trans_list: list[float] = []
    init_render_list: list[float] = []
    final_render_list: list[float] = []
    for idx in range(n_eval):
        sample = dataset[idx]
        pts = sample["points"].unsqueeze(0).to(device_t)
        gt_pose = sample["pose_vec"].unsqueeze(0).to(device_t)
        obs = sample["render_gt"].unsqueeze(0).to(device_t)
        init_pose_np = _perturb_pose_vec_np(
            np.asarray(sample["pose_vec"].numpy(), dtype=np.float32),
            rng=rng,
            rot_noise_deg=float(rot_noise_deg),
            trans_noise_std=float(trans_noise_std),
        )
        init_pose = torch.from_numpy(np.asarray(init_pose_np, dtype=np.float32)).unsqueeze(0).to(device_t)
        with torch.no_grad():
            init_out = model(pts, init_pose)
            init_render_loss, _init_depth_loss, _init_normal_loss, _init_mask_loss = _render_loss(init_out, obs)
        opt_result = optimize_pose_from_render_observation(
            model=model,
            points=pts,
            render_obs=obs,
            init_pose_vec=init_pose,
            steps=int(optimize_steps),
            lr=float(optimize_lr),
            device=device_t,
        )
        best_pose = opt_result["best_pose_vec"].to(device_t)
        with torch.no_grad():
            final_out = model(pts, best_pose)
            final_render_loss, _final_depth_loss, _final_normal_loss, _final_mask_loss = _render_loss(final_out, obs)
        init_metrics = _pose_metrics(init_pose, gt_pose)
        final_metrics = _pose_metrics(best_pose, gt_pose)
        init_rot_list.append(float(init_metrics["rot_deg"]))
        init_trans_list.append(float(init_metrics["trans_err"]))
        final_rot_list.append(float(final_metrics["rot_deg"]))
        final_trans_list.append(float(final_metrics["trans_err"]))
        init_render_list.append(float(init_render_loss.detach().cpu().item()))
        final_render_list.append(float(final_render_loss.detach().cpu().item()))
        rows.append(
            {
                "index": int(idx),
                "warp_variant": int(sample.get("warp_variant", 0)),
                "init_rot_deg": float(init_metrics["rot_deg"]),
                "init_trans_err": float(init_metrics["trans_err"]),
                "final_rot_deg": float(final_metrics["rot_deg"]),
                "final_trans_err": float(final_metrics["trans_err"]),
                "init_render_loss": float(init_render_loss.detach().cpu().item()),
                "final_render_loss": float(final_render_loss.detach().cpu().item()),
            }
        )
        if int(progress_every) > 0 and ((idx + 1) % int(progress_every) == 0 or (idx + 1) == int(n_eval)):
            print(
                f"[PointRTField][PoseEval] {idx + 1}/{n_eval} "
                f"init_rot={init_metrics['rot_deg']:.3f} final_rot={final_metrics['rot_deg']:.3f} "
                f"init_trans={init_metrics['trans_err']:.4f} final_trans={final_metrics['trans_err']:.4f} "
                f"init_render={float(init_render_loss.detach().cpu().item()):.4f} final_render={float(final_render_loss.detach().cpu().item()):.4f}",
                flush=True,
            )
            if str(out_json or "").strip():
                partial = {
                    "checkpoint_path": str(checkpoint_path),
                    "manifest_path": str(manifest_final),
                    "split": str(split),
                    "n_eval_target": int(n_eval),
                    "done_items": int(idx + 1),
                    "details": rows,
                }
                Path(out_json).write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")

    def _mean(v: list[float]) -> float:
        return float(np.mean(np.asarray(v, dtype=np.float32))) if v else float("nan")

    def _median(v: list[float]) -> float:
        return float(np.median(np.asarray(v, dtype=np.float32))) if v else float("nan")

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_final),
        "split": str(split),
        "n_eval": int(n_eval),
        "rot_noise_deg": float(rot_noise_deg),
        "trans_noise_std": float(trans_noise_std),
        "optimize_steps": int(optimize_steps),
        "optimize_lr": float(optimize_lr),
        "progress_every": int(progress_every),
        "init_rot_deg_mean": _mean(init_rot_list),
        "init_rot_deg_median": _median(init_rot_list),
        "init_trans_err_mean": _mean(init_trans_list),
        "init_trans_err_median": _median(init_trans_list),
        "final_rot_deg_mean": _mean(final_rot_list),
        "final_rot_deg_median": _median(final_rot_list),
        "final_trans_err_mean": _mean(final_trans_list),
        "final_trans_err_median": _median(final_trans_list),
        "init_render_loss_mean": _mean(init_render_list),
        "final_render_loss_mean": _mean(final_render_list),
        "rot_success_5deg": float(np.mean((np.asarray(final_rot_list, dtype=np.float32) <= 5.0).astype(np.float32))) if final_rot_list else float("nan"),
        "rot_success_10deg": float(np.mean((np.asarray(final_rot_list, dtype=np.float32) <= 10.0).astype(np.float32))) if final_rot_list else float("nan"),
        "trans_success_005": float(np.mean((np.asarray(final_trans_list, dtype=np.float32) <= 0.05).astype(np.float32))) if final_trans_list else float("nan"),
        "trans_success_010": float(np.mean((np.asarray(final_trans_list, dtype=np.float32) <= 0.10).astype(np.float32))) if final_trans_list else float("nan"),
        "details": rows,
    }
    if str(out_json or "").strip():
        Path(out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


@torch.no_grad()
def _online_point_augment(
    pts: torch.Tensor,
    jitter_std: float = 0.003,
    dropout_rate: float = 0.05,
) -> torch.Tensor:
    """Online augmentation: coordinate jitter + random point zeroing."""
    if pts.ndim != 3:
        return pts
    bsz, n_pts, dim = pts.shape
    noise = torch.randn(bsz, n_pts, min(3, dim), device=pts.device, dtype=pts.dtype) * jitter_std
    pts = pts.clone()
    pts[:, :, :3] = pts[:, :, :3] + noise
    if dropout_rate > 0.0:
        keep = (torch.rand(bsz, n_pts, 1, device=pts.device) > dropout_rate).to(pts.dtype)
        pts = pts * keep
    return pts


def _run_one_epoch(
    *,
    model: PointRTPoseRenderField,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    use_amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    depth_w: float = 1.0,
    normal_w: float = 1.0,
    mask_w: float = 0.25,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    total_depth = 0.0
    total_normal = 0.0
    total_mask = 0.0
    n_items = 0
    n_skipped = 0
    for batch in loader:
        pts = batch["points"].to(device, non_blocking=True)
        pose = batch["pose_vec"].to(device, non_blocking=True)
        gt = batch["render_gt"].to(device, non_blocking=True)
        if train_mode:
            pts = _online_point_augment(pts)
        with torch.amp.autocast(device_type=device.type, enabled=bool(use_amp and device.type == "cuda")):
            out = model(pts, pose)
            loss, depth_loss, normal_loss, mask_loss = _render_loss(out, gt, depth_w=depth_w, normal_w=normal_w, mask_w=mask_w)
        if not bool(torch.isfinite(loss).all()):
            if train_mode and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            n_skipped += int(pts.shape[0])
            continue
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and bool(use_amp and device.type == "cuda"):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        bs = int(pts.shape[0])
        total_loss += float(loss.detach().cpu()) * bs
        total_depth += float(depth_loss.detach().cpu()) * bs
        total_normal += float(normal_loss.detach().cpu()) * bs
        total_mask += float(mask_loss.detach().cpu()) * bs
        n_items += bs
    denom = float(max(1, n_items))
    return {
        "loss": float(total_loss / denom),
        "depth_loss": float(total_depth / denom),
        "normal_loss": float(total_normal / denom),
        "mask_loss": float(total_mask / denom),
        "skipped_items": int(n_skipped),
    }


@torch.no_grad()
def evaluate_point_rt_renderfield(
    *,
    model: PointRTPoseRenderField,
    dataset: PointRTRenderFieldDataset,
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool = False,
    use_amp: bool = False,
    depth_w: float = 5.0,
    normal_w: float = 1.0,
    mask_w: float = 0.5,
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
    total_loss = 0.0
    total_depth = 0.0
    total_normal = 0.0
    total_mask = 0.0
    n_items = 0
    for batch in loader:
        pts = batch["points"].to(device, non_blocking=True)
        pose = batch["pose_vec"].to(device, non_blocking=True)
        gt = batch["render_gt"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=bool(use_amp and device.type == "cuda")):
            out = model(pts, pose)
            loss, depth_loss, normal_loss, mask_loss = _render_loss(out, gt, depth_w=depth_w, normal_w=normal_w, mask_w=mask_w)
        bs = int(pts.shape[0])
        total_loss += float(loss.detach().cpu()) * bs
        total_depth += float(depth_loss.detach().cpu()) * bs
        total_normal += float(normal_loss.detach().cpu()) * bs
        total_mask += float(mask_loss.detach().cpu()) * bs
        n_items += bs
    denom = float(max(1, n_items))
    return {
        "n_samples": int(n_items),
        "loss": float(total_loss / denom),
        "depth_loss": float(total_depth / denom),
        "normal_loss": float(total_normal / denom),
        "mask_loss": float(total_mask / denom),
    }


def train_point_rt_renderfield(
    *,
    manifest_path: str,
    out_dir: str,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    seed: int = 42,
    point_latent_dim: int = 128,
    point_hidden_dim: int = 128,
    pose_hidden_dim: int = 128,
    fusion_hidden_dim: int = 256,
    fusion_depth: int = 3,
    eval_interval: int = 5,
    early_stop_patience: int = 40,
    early_stop_min_delta: float = 1e-4,
    early_stop_patience_evals: int = 8,
    train_num_workers: int = 0,
    val_num_workers: int = 0,
    device: str = "cuda",
    use_amp: bool = True,
    preview_count: int = 4,
    resume_checkpoint: str = "",
    lr_scheduler: str = "cosine",
    lr_warmup_epochs: int = 5,
    depth_w: float = 5.0,
    normal_w: float = 1.0,
    mask_w: float = 0.5,
) -> dict[str, Any]:
    urr = _get_urr()
    urr._set_seed(int(seed))
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    device_t = urr._resolve_device(device)
    if device_t.type == "cuda":
        torch.backends.cudnn.benchmark = True
    amp_enabled = bool(use_amp and device_t.type == "cuda")

    train_set = PointRTRenderFieldDataset(manifest_path=manifest_path, split="train", seed=int(seed))
    val_set = PointRTRenderFieldDataset(manifest_path=manifest_path, split="val", seed=int(seed))
    if len(train_set) <= 0 or len(val_set) <= 0:
        raise RuntimeError("训练集或验证集为空，无法训练 point-RT render field")
    dataset_summary = summarize_renderfield_manifest(manifest_path)
    (out_dir_p / "dataset_summary.json").write_text(json.dumps(dataset_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[PointRTField] dataset_summary train_samples={dataset_summary['train_samples']} val_samples={dataset_summary['val_samples']} "
        f"train_pointclouds={dataset_summary['train_pointclouds']} val_pointclouds={dataset_summary['val_pointclouds']} "
        f"refs={dataset_summary['n_reference_frames']} expands={dataset_summary['bbox_expand_values']} "
        f"warp_augs={dataset_summary['n_warp_augs']}",
        flush=True,
    )

    train_workers = urr._auto_num_workers(int(train_num_workers), device_t, fallback=2)
    val_workers = urr._auto_num_workers(int(val_num_workers), device_t, fallback=1)
    pin_memory = bool(device_t.type == "cuda")
    train_loader = DataLoader(
        train_set,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=int(train_workers),
        persistent_workers=bool(train_workers > 0),
        pin_memory=pin_memory,
    )

    model = PointRTPoseRenderField(
        image_size=int(train_set.manifest.get("image_size", 96)),
        point_latent_dim=int(point_latent_dim),
        point_hidden_dim=int(point_hidden_dim),
        pose_hidden_dim=int(pose_hidden_dim),
        fusion_hidden_dim=int(fusion_hidden_dim),
        fusion_depth=int(fusion_depth),
        point_in_dim=int(train_set.manifest.get("point_in_dim", 3) or 3),
    ).to(device_t)
    resume_path = str(resume_checkpoint or "").strip()
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(
            f"[PointRTField] resume checkpoint={resume_path} missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    # LR scheduler: linear warmup + cosine annealing
    import math as _math
    _lr_scheduler_name = str(lr_scheduler or "none").strip().lower()
    _warmup_ep = int(max(0, lr_warmup_epochs))
    def _lr_lambda(epoch_0idx: int) -> float:
        ep = epoch_0idx + 1
        if ep <= _warmup_ep and _warmup_ep > 0:
            return float(ep) / float(_warmup_ep)
        if _lr_scheduler_name == "cosine":
            progress = float(ep - _warmup_ep) / float(max(1, epochs - _warmup_ep))
            return float(max(0.05, 0.5 * (1.0 + _math.cos(_math.pi * progress))))
        return 1.0
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda) if _lr_scheduler_name != "none" else None

    # EMA model for smoother evaluation
    import copy as _copy
    ema_model = _copy.deepcopy(model)
    ema_model.eval()
    ema_decay = 0.999

    @torch.no_grad()
    def _update_ema() -> None:
        for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(ema_decay).add_(p_model.data, alpha=1.0 - ema_decay)
        for b_ema, b_model in zip(ema_model.buffers(), model.buffers()):
            b_ema.data.copy_(b_model.data)

    model_params = int(sum(int(p.numel()) for p in model.parameters()))
    print(
        f"[PointRTField] params={model_params} point_latent={int(point_latent_dim)} point_hidden={int(point_hidden_dim)} "
        f"pose_hidden={int(pose_hidden_dim)} fusion_hidden={int(fusion_hidden_dim)} fusion_depth={int(fusion_depth)} "
        f"device={device_t.type} amp={int(amp_enabled)} train_workers={int(train_workers)} val_workers={int(val_workers)} "
        f"lr_scheduler={_lr_scheduler_name} warmup={_warmup_ep} depth_w={depth_w:.1f} normal_w={normal_w:.1f} mask_w={mask_w:.2f}",
        flush=True,
    )

    best_state = None
    best_metric = float("inf")
    epochs_since_improve = 0
    no_improve_evals = 0
    eval_every = int(max(1, eval_interval))
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        train_metrics = _run_one_epoch(
            model=model,
            loader=train_loader,
            device=device_t,
            optimizer=optimizer,
            use_amp=amp_enabled,
            scaler=scaler,
            depth_w=float(depth_w),
            normal_w=float(normal_w),
            mask_w=float(mask_w),
        )
        if lr_sched is not None:
            lr_sched.step()
        _update_ema()
        current_lr = float(optimizer.param_groups[0]["lr"])
        do_eval = bool(epoch == 1 or epoch == int(epochs) or (epoch % int(eval_every) == 0))
        val_metrics = None
        if do_eval:
            val_metrics = evaluate_point_rt_renderfield(
                model=ema_model,
                dataset=val_set,
                device=device_t,
                batch_size=int(batch_size),
                num_workers=int(val_workers),
                pin_memory=pin_memory,
                use_amp=amp_enabled,
                depth_w=float(depth_w),
                normal_w=float(normal_w),
                mask_w=float(mask_w),
            )
            _save_render_previews(
                model=ema_model,
                dataset=val_set,
                device=device_t,
                out_dir=out_dir_p,
                epoch=int(epoch),
                max_items=int(preview_count),
            )
        row = {"epoch": int(epoch), "train": train_metrics, "val": val_metrics}
        history.append(row)
        if isinstance(val_metrics, dict):
            print(
                f"[PointRTField] epoch={epoch}/{epochs} lr={current_lr:.2e} train_loss={train_metrics['loss']:.4f} "
                f"train_depth={train_metrics['depth_loss']:.4f} train_normal={train_metrics['normal_loss']:.4f} train_mask={train_metrics['mask_loss']:.4f} train_skipped={train_metrics['skipped_items']} "
                f"val_loss={val_metrics['loss']:.4f} val_depth={val_metrics['depth_loss']:.4f} val_normal={val_metrics['normal_loss']:.4f} val_mask={val_metrics['mask_loss']:.4f}",
                flush=True,
            )
            score = float(val_metrics.get("loss", float("inf")))
            if float(score) + float(max(0.0, early_stop_min_delta)) < float(best_metric):
                best_metric = score
                epochs_since_improve = 0
                no_improve_evals = 0
                best_state = {
                    "model": ema_model.state_dict(),
                    "epoch": int(epoch),
                    "val_metrics": val_metrics,
                    "manifest_path": str(manifest_path),
                    "model_config": {
                        "image_size": int(train_set.manifest.get("image_size", 96)),
                        "point_latent_dim": int(point_latent_dim),
                        "point_hidden_dim": int(point_hidden_dim),
                        "pose_hidden_dim": int(pose_hidden_dim),
                        "fusion_hidden_dim": int(fusion_hidden_dim),
                        "fusion_depth": int(fusion_depth),
                        "point_in_dim": int(train_set.manifest.get("point_in_dim", 3) or 3),
                    },
                }
                torch.save(best_state, str(out_dir_p / "point_rt_renderfield_best.pt"))
            else:
                epochs_since_improve += int(eval_every)
                no_improve_evals += 1
        else:
            print(
                f"[PointRTField] epoch={epoch}/{epochs} lr={current_lr:.2e} train_loss={train_metrics['loss']:.4f} "
                f"train_depth={train_metrics['depth_loss']:.4f} train_normal={train_metrics['normal_loss']:.4f} train_mask={train_metrics['mask_loss']:.4f} train_skipped={train_metrics['skipped_items']}",
                flush=True,
            )
        latest_summary = {
            "last_epoch": int(epoch),
            "best_metric": float(best_metric),
            "epochs_since_improve": int(epochs_since_improve),
            "no_improve_evals": int(no_improve_evals),
            "latest_train": train_metrics,
            "latest_val": val_metrics,
        }
        (out_dir_p / "train_summary_latest.json").write_text(json.dumps(latest_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        stop_by_eval = int(early_stop_patience_evals) > 0 and int(no_improve_evals) >= int(early_stop_patience_evals)
        stop_by_epoch = int(early_stop_patience) > 0 and int(epochs_since_improve) >= int(early_stop_patience)
        if stop_by_eval or stop_by_epoch:
            print(
                f"[PointRTField] early stop at epoch={epoch} "
                f"no_improve_evals={int(no_improve_evals)}/{int(early_stop_patience_evals)} "
                f"epochs_since_improve={int(epochs_since_improve)}/{int(early_stop_patience)}",
                flush=True,
            )
            break

    if best_state is None:
        final_eval = evaluate_point_rt_renderfield(
            model=model,
            dataset=val_set,
            device=device_t,
            batch_size=int(batch_size),
            num_workers=int(val_workers),
            pin_memory=pin_memory,
            use_amp=amp_enabled,
        )
        best_state = {
            "model": ema_model.state_dict(),
            "epoch": int(max(1, len(history))),
            "val_metrics": final_eval,
            "manifest_path": str(manifest_path),
            "model_config": {
                "image_size": int(train_set.manifest.get("image_size", 96)),
                "point_latent_dim": int(point_latent_dim),
                "point_hidden_dim": int(point_hidden_dim),
                "pose_hidden_dim": int(pose_hidden_dim),
                "fusion_hidden_dim": int(fusion_hidden_dim),
                "fusion_depth": int(fusion_depth),
                "point_in_dim": int(train_set.manifest.get("point_in_dim", 3) or 3),
            },
        }

    ckpt_path = out_dir_p / "point_rt_renderfield_best.pt"
    torch.save(best_state, str(ckpt_path))
    final_val = evaluate_point_rt_renderfield(
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
        "best_epoch": int(best_state["epoch"]),
        "best_val": best_state["val_metrics"],
        "final_val": final_val,
        "history": history,
        "train_size": int(len(train_set)),
        "val_size": int(len(val_set)),
        "n_train_sequences": int(len(train_set.sequences)),
        "n_val_sequences": int(len(val_set.sequences)),
        "device": str(device_t),
        "model_params": int(model_params),
        "point_latent_dim": int(point_latent_dim),
        "point_hidden_dim": int(point_hidden_dim),
        "pose_hidden_dim": int(pose_hidden_dim),
        "fusion_hidden_dim": int(fusion_hidden_dim),
        "fusion_depth": int(fusion_depth),
        "visibility_gain": float(model.visibility_gain.detach().cpu()),
        "occlusion_beta": float(model.occlusion_beta.detach().cpu()),
        "eval_interval": int(eval_every),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_patience_evals": int(early_stop_patience_evals),
        "early_stop_min_delta": float(early_stop_min_delta),
        "train_num_workers": int(train_workers),
        "val_num_workers": int(val_workers),
        "amp_enabled": bool(amp_enabled),
        "preview_count": int(preview_count),
        "resume_checkpoint": str(resume_path),
        "dataset_summary": dataset_summary,
        "manifest_path": str(manifest_path),
        "out_dir": str(out_dir_p),
    }
    (out_dir_p / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
