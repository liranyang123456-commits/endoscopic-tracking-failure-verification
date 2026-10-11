#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练轻量级轮廓相似度网络（与 train_uvh_point_rt_renderfield.py 同源数据：Chess_Box_1_8 + Chess_Box_Mask）。

正样本：同一 GT mask 轮廓栅格 + 随机仿射/透视/薄形变；可选随机遮挡，目标相似度随可见比例下降。
负样本：不同序列或不同帧的轮廓，目标相似度为低值。

默认：较大 samples_per_epoch、较多 epoch，适合离线充分训练。

TrackVes（手术血管轮廓）请改用同目录脚本：
  - train_contour_sim_trackves.py（MSE / 与 Chess 相同配对方式）
  - train_contour_sim_trackves_triplet.py（三元组损失，强调跨序列实例区分）
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import compare_chessboard_trackers as cct
import uvh_rt_regression as urr
from contour_similarity_net import ContourSimilarityNet


def _rasterize_contour(
    mask_roi: np.ndarray,
    out_size: int,
    thickness: int | None = None,
) -> np.ndarray:
    """ROI 二值 mask → 最大外轮廓线画在 [0,1] 方阵上。"""
    m = (np.asarray(mask_roi, dtype=np.uint8) > 0).astype(np.uint8) * 255
    S = int(out_size)
    if int(np.count_nonzero(m)) < 8:
        return np.zeros((S, S), dtype=np.float32)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((S, S), dtype=np.float32)
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 4.0:
        return np.zeros((S, S), dtype=np.float32)
    x, y, w, h = cv2.boundingRect(c)
    if w < 2 or h < 2:
        return np.zeros((S, S), dtype=np.float32)
    crop = m[y : y + h, x : x + w]
    sc = min((S - 4) / max(w, 1), (S - 4) / max(h, 1))
    nw, nh = max(2, int(round(w * sc))), max(2, int(round(h * sc)))
    small = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((S, S), dtype=np.uint8)
    ox = (S - nw) // 2
    oy = (S - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = small
    cnts2, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts2:
        return np.zeros((S, S), dtype=np.float32)
    c2 = max(cnts2, key=cv2.contourArea)
    out = np.zeros((S, S), dtype=np.uint8)
    th = int(thickness) if thickness is not None else max(1, S // 64)
    cv2.drawContours(out, [c2], -1, 255, thickness=th)
    return (out.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _aug_affine(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w * 0.5, h * 0.5)
    ang = float(rng.uniform(-28.0, 28.0))
    s = float(rng.uniform(0.82, 1.18))
    M = cv2.getRotationMatrix2D(center, ang, s)
    M[0, 2] += float(rng.uniform(-0.12, 0.12)) * w
    M[1, 2] += float(rng.uniform(-0.12, 0.12)) * h
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)


def _aug_perspective(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    j = float(rng.uniform(0.03, 0.10))
    dst = src + rng.uniform(-j, j, size=src.shape).astype(np.float32) * np.array(
        [w, h], dtype=np.float32
    )
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        img,
        H,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)


def _aug_elastic_light(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    sigma = float(rng.uniform(2.5, 5.0))
    alpha = float(rng.uniform(6.0, 14.0))
    dx = cv2.GaussianBlur(
        rng.standard_normal((h, w)).astype(np.float32) * alpha,
        ksize=(0, 0),
        sigmaX=sigma,
    )
    dy = cv2.GaussianBlur(
        rng.standard_normal((h, w)).astype(np.float32) * alpha,
        ksize=(0, 0),
        sigmaX=sigma,
    )
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    mapx = (xx + dx).astype(np.float32)
    mapy = (yy + dy).astype(np.float32)
    return cv2.remap(
        img,
        mapx,
        mapy,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.float32)


def _foreground_ratio(img: np.ndarray) -> float:
    return float(np.mean(img > 0.04))


def _random_occlude(
    img: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """随机矩形遮挡；返回 (遮挡后, 相对原前景的遮挡比例)。"""
    ys, xs = np.where(img > 0.04)
    if xs.size < 6:
        return img, 0.0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    out = img.copy()
    fg0 = max(1, int(np.count_nonzero(img > 0.04)))
    for _ in range(int(rng.integers(1, 4))):
        rw = int(max(2, bw * float(rng.uniform(0.12, 0.55))))
        rh = int(max(2, bh * float(rng.uniform(0.12, 0.55))))
        cx = int(rng.integers(x0, x1 + 1))
        cy = int(rng.integers(y0, y1 + 1))
        xA = max(0, cx - rw // 2)
        yA = max(0, cy - rh // 2)
        xB = min(img.shape[1], xA + rw)
        yB = min(img.shape[0], yA + rh)
        out[yA:yB, xA:xB] = 0.0
    removed = fg0 - int(np.count_nonzero(out > 0.04))
    occ = max(0.0, min(1.0, removed / float(fg0)))
    return out.astype(np.float32), occ


def _crop_mask(gt: Any) -> np.ndarray:
    x, y, w, h = (int(v) for v in gt.bbox)
    mk = np.asarray(gt.mask, dtype=np.uint8)
    H, W = mk.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(1, min(w, W - x))
    h = max(1, min(h, H - y))
    return mk[y : y + h, x : x + w]


def _split_names(
    names: list[str], val_ratio: float, test_ratio: float, seed: int
) -> tuple[set[str], set[str], set[str]]:
    rng = random.Random(int(seed))
    shuffled = list(names)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(round(n * float(test_ratio)))) if n >= 3 else 0
    n_test = min(n_test, max(0, n - 2))
    n_val = max(1, int(round(n * float(val_ratio)))) if n >= 3 else 0
    n_val = min(n_val, max(0, n - n_test - 1))
    test_s = set(shuffled[:n_test])
    val_s = set(shuffled[n_test : n_test + n_val])
    train_s = set(shuffled[n_test + n_val :])
    if not train_s and n:
        train_s = set(shuffled)
        val_s, test_s = set(), set()
    return train_s, val_s, test_s


class ContourSimilarityDataset(Dataset):
    """每个 index 在线采样一对轮廓与回归目标相似度。"""

    def __init__(
        self,
        entries: list[tuple[str, Any]],
        *,
        image_size: int,
        length: int,
        seed: int,
        positive_prob: float = 0.58,
        occlusion_prob: float = 0.42,
        split: str = "train",
    ) -> None:
        self.entries = entries
        self.S = int(image_size)
        self.length = int(length)
        self.base_seed = int(seed)
        self.positive_prob = float(positive_prob)
        self.occlusion_prob = float(occlusion_prob)
        self.split = split
        self.epoch = 0
        by_ds: dict[str, list[Any]] = {}
        for ds, g in entries:
            by_ds.setdefault(ds, []).append(g)
        self.by_ds = by_ds
        self.ds_names = sorted(by_ds.keys())

    def set_epoch(self, e: int) -> None:
        self.epoch = int(e)

    def __len__(self) -> int:
        return self.length

    def _pick_negative_pair(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if len(self.entries) < 2:
            return self._pick_positive_pair(rng)
        if len(self.ds_names) >= 2 and rng.random() < 0.72:
            a, b = rng.choice(np.array(self.ds_names), size=2, replace=False)
            ga = rng.choice(self.by_ds[str(a)])
            gb = rng.choice(self.by_ds[str(b)])
        else:
            ga = rng.choice(self.entries)
            gb = ga
            for _ in range(64):
                gb = rng.choice(self.entries)
                if (ga[0], str(ga[1].image_path)) != (gb[0], str(gb[1].image_path)):
                    break
        ra = _rasterize_contour(_crop_mask(ga[1]), self.S)
        rb = _rasterize_contour(_crop_mask(gb[1]), self.S)
        tgt = float(rng.uniform(0.04, 0.16))
        return ra, rb, tgt

    def _pick_positive_pair(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, float]:
        ds, g = self.entries[int(rng.integers(0, len(self.entries)))]
        base = _rasterize_contour(_crop_mask(g), self.S)
        if float(np.max(base)) < 1e-3:
            z = np.zeros((self.S, self.S), dtype=np.float32)
            return z, z, 0.0
        a = base.copy()
        b = base.copy()
        b = _aug_affine(b, rng)
        if rng.random() < 0.55:
            b = _aug_perspective(b, rng)
        if rng.random() < 0.35:
            b = _aug_elastic_light(b, rng)
        if rng.random() < self.occlusion_prob:
            b, occ = _random_occlude(b, rng)
            visible = max(0.0, 1.0 - occ)
            tgt = 0.08 + 0.92 * (visible**0.85)
        else:
            tgt = float(rng.uniform(0.94, 1.0))
        if _foreground_ratio(b) < 1e-3:
            return a, base, 1.0
        return a, b, tgt

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(
            self.base_seed + self.epoch * 1_000_003 + int(idx) * 1_000_007
        )
        if len(self.entries) < 2:
            ra = np.zeros((self.S, self.S), dtype=np.float32)
            return {
                "x1": torch.from_numpy(ra)[None, ...],
                "x2": torch.from_numpy(ra)[None, ...],
                "target": torch.tensor(0.0, dtype=torch.float32),
            }
        if rng.random() < self.positive_prob:
            x1, x2, tgt = self._pick_positive_pair(rng)
        else:
            try:
                x1, x2, tgt = self._pick_negative_pair(rng)
            except Exception:
                x1, x2, tgt = self._pick_positive_pair(rng)
        t = torch.tensor(float(np.clip(tgt, 0.0, 1.0)), dtype=torch.float32)
        return {
            "x1": torch.from_numpy(x1.astype(np.float32))[None, ...],
            "x2": torch.from_numpy(x2.astype(np.float32))[None, ...],
            "target": t,
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "x1": torch.stack([b["x1"] for b in batch], dim=0),
        "x2": torch.stack([b["x2"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
    }


@torch.no_grad()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    se = 0.0
    n = 0
    for batch in loader:
        x1 = batch["x1"].to(device, non_blocking=True)
        x2 = batch["x2"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)
        ac = (
            torch.amp.autocast("cuda", enabled=True)
            if use_amp
            else nullcontext()
        )
        with ac:
            pred, _, _ = model(x1, x2)
            loss = F.mse_loss(pred, y)
        bs = int(x1.shape[0])
        se += float(loss.item()) * bs
        n += bs
    return {"mse": se / max(1, n)}


def _apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    p = str(getattr(args, "experiment_profile", "custom") or "custom").strip().lower()
    if p in ("", "custom"):
        return args
    if p == "paper":
        args.epochs = max(int(args.epochs), 220)
        args.samples_per_epoch = max(int(args.samples_per_epoch), 32768)
        args.val_batches = max(int(args.val_batches), 160)
        args.eval_interval = min(int(args.eval_interval), 2)
        return args
    if p == "debug":
        args.epochs = 4
        args.samples_per_epoch = 512
        args.val_batches = 4
        args.batch_size = min(int(args.batch_size), 16)
        return args
    raise ValueError(f"未知 --experiment-profile: {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train contour similarity net (Chess Box datasets)")
    ap.add_argument(
        "--experiment-profile",
        default="custom",
        choices=["custom", "paper", "debug"],
        help="paper=更长训练+更大每轮采样；debug=冒烟",
    )
    ap.add_argument("--image-root", default=r"E:\Surgical_Tracking_Datasets\Chess_Box_1_8")
    ap.add_argument("--gt-root", default=r"E:\Surgical_Tracking_Datasets\Chess_Box_Mask")
    ap.add_argument("--dataset-name", default="", help="仅训练该子目录名；空=全部配对序列")
    ap.add_argument("--out-dir", default=r"E:\contour_sim_runs\chess_v1")
    ap.add_argument("--val-ratio", type=float, default=0.22)
    ap.add_argument("--test-ratio", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--base-ch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--samples-per-epoch", type=int, default=24576)
    ap.add_argument("--val-batches", type=int, default=120)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--eval-interval", type=int, default=2)
    args = ap.parse_args()
    args = _apply_profile(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    image_root = Path(args.image_root)
    gt_root = Path(args.gt_root)
    all_names = urr._discover_dataset_names(image_root, gt_root)
    if str(args.dataset_name).strip():
        all_names = [n for n in all_names if n == str(args.dataset_name).strip()]
    train_n, val_n, test_n = _split_names(
        all_names, float(args.val_ratio), float(args.test_ratio), int(args.seed)
    )

    def collect(split_names: set[str]) -> list[tuple[str, Any]]:
        acc: list[tuple[str, Any]] = []
        for n in sorted(split_names):
            fd = urr._resolve_image_dir(image_root / n)
            gd = urr._resolve_gt_mask_dir(gt_root / n)
            if fd is None or gd is None:
                continue
            try:
                gts = cct._load_gt_from_mask_dir(fd, gd)
            except Exception:
                continue
            for g in gts:
                acc.append((n, g))
        return acc

    train_entries = collect(train_n)
    val_entries = collect(val_n)
    if not val_entries and train_entries:
        val_entries = train_entries[: max(64, len(train_entries) // 8)]
    if not train_entries:
        raise RuntimeError(
            "未加载到任何 GT 帧：请检查 --image-root / --gt-root 与 summary.csv+mask 结构。"
        )

    print(
        json.dumps(
            {
                "n_names_total": len(all_names),
                "train_names": sorted(train_n),
                "val_names": sorted(val_n),
                "test_names": sorted(test_n),
                "n_train_frames": len(train_entries),
                "n_val_frames": len(val_entries),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )

    train_ds = ContourSimilarityDataset(
        train_entries,
        image_size=int(args.image_size),
        length=int(args.samples_per_epoch),
        seed=int(args.seed) + 11,
        split="train",
    )
    val_len = int(max(512, min(len(val_entries) * 24, int(args.val_batches) * int(args.batch_size))))
    val_ds = ContourSimilarityDataset(
        val_entries,
        image_size=int(args.image_size),
        length=val_len,
        seed=int(args.seed) + 99,
        split="val",
    )

    nw = int(max(0, args.num_workers))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=nw,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=max(0, min(1, nw)),
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )

    device = torch.device(
        args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu"
    )
    use_amp = bool(int(args.amp)) and device.type == "cuda"
    model = ContourSimilarityNet(
        image_size=int(args.image_size),
        latent_dim=int(args.latent_dim),
        base_ch=int(args.base_ch),
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, int(args.epochs)), eta_min=float(args.lr) * 0.02
    )
    scaler: torch.amp.GradScaler | None
    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    best_val = math.inf
    best_path = out_dir / "contour_similarity_best.pt"
    last_path = out_dir / "contour_similarity_last.pt"
    t0 = time.time()

    for epoch in range(int(args.epochs)):
        train_ds.set_epoch(epoch)
        val_ds.set_epoch(epoch + 10_000)
        model.train()
        running = 0.0
        nb = 0
        for batch in train_loader:
            x1 = batch["x1"].to(device, non_blocking=True)
            x2 = batch["x2"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            ac = (
                torch.amp.autocast("cuda", enabled=True)
                if use_amp
                else nullcontext()
            )
            with ac:
                pred, _, _ = model(x1, x2)
                loss = F.mse_loss(pred, y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            running += float(loss.item())
            nb += 1
        sched.step()

        train_mse = running / max(1, nb)
        if (epoch + 1) % int(max(1, args.eval_interval)) == 0 or epoch == 0:
            vm = _validate(model, val_loader, device, use_amp)
            val_mse = float(vm["mse"])
            if val_mse < best_val:
                best_val = val_mse
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "val_mse": val_mse,
                        "config": vars(args),
                    },
                    str(best_path),
                )
            print(
                f"[epoch {epoch+1}/{args.epochs}] train_mse={train_mse:.6f} val_mse={val_mse:.6f} "
                f"best_val={best_val:.6f} lr={sched.get_last_lr()[0]:.2e}",
                flush=True,
            )
        else:
            print(
                f"[epoch {epoch+1}/{args.epochs}] train_mse={train_mse:.6f} lr={sched.get_last_lr()[0]:.2e}",
                flush=True,
            )

        torch.save(
            {"epoch": epoch, "model": model.state_dict(), "config": vars(args)},
            str(last_path),
        )

    dt = time.time() - t0
    summary = {
        "best_val_mse": float(best_val),
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "seconds": float(dt),
        "n_train_frames": len(train_entries),
    }
    (out_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
