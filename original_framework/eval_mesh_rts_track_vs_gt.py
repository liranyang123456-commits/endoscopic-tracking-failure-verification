#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 MIS_Mesh_RTS_Track 输出的 tracking_results.json 与 Chess_Box_Mask 目录下的 GT 对齐评估。

GT 目录需含 summary.csv（与 compare_chessboard_trackers._load_gt_from_mask_dir 一致）。
若 MIS_GT_MASK_DIR 指向序列根（含 captured_photos3/summary.csv），会自动用 frames_dir 的文件夹名拼接。

用法：
  python eval_mesh_rts_track_vs_gt.py ^
    --results D:\\...\\tracking_results.json ^
    --frames-dir E:\\...\\Chess_Box_1_8\\captured_photos3 ^
    --gt-mask-dir E:\\...\\Chess_Box_Mask\\captured_photos3

或在跑跟踪前设置环境变量 MIS_GT_MASK_DIR，主程序结束时会自动调用本模块。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

import compare_chessboard_trackers as cct


def resolve_gt_mask_dir(frames_dir: Path, gt_user: Path) -> Path:
    gt_user = gt_user.expanduser()
    if (gt_user / "summary.csv").is_file():
        return gt_user.resolve()
    seq = Path(frames_dir).name
    cand = gt_user / seq
    if (cand / "summary.csv").is_file():
        return cand.resolve()
    raise FileNotFoundError(
        f"找不到 GT summary.csv：已尝试 {gt_user} 与 {cand}"
    )


def evaluate_and_save(
    *,
    results_json_path: str,
    frames_dir: str,
    gt_mask_dir_user: str,
    out_dir: str,
    precision_thresh_px: float = 20.0,
    success_iou_thresh: float = 0.5,
) -> str:
    fd = Path(frames_dir)
    gt_dir = resolve_gt_mask_dir(fd, Path(gt_mask_dir_user))
    gt_frames = cct._load_gt_from_mask_dir(fd, gt_dir)
    if not gt_frames:
        raise RuntimeError(f"未加载到 GT 帧: frames={fd} gt={gt_dir}")

    gt_by_stem: dict[str, cct.FrameGT] = {}
    for g in gt_frames:
        gt_by_stem[Path(g.image_path).name.lower()] = g

    with open(results_json_path, "r", encoding="utf-8") as f:
        track_rows: list[dict] = json.load(f)

    per_rows: list[dict] = []
    bbox_ious: list[float] = []
    mask_ious: list[float] = []
    center_errs: list[float] = []
    prec_hits = 0
    succ_hits = 0
    n_matched = 0

    bbox_ious_acc: list[float] = []
    mask_ious_acc: list[float] = []
    center_errs_acc: list[float] = []
    n_accepted = 0

    for row in track_rows:
        img_path = Path(str(row.get("image", "")))
        stem = img_path.name.lower()
        gt = gt_by_stem.get(stem)
        if gt is None:
            continue

        h, w = int(gt.mask.shape[0]), int(gt.mask.shape[1])
        bb = row.get("bbox", None)
        if not bb or len(bb) < 4:
            continue
        pred_bbox = cct._clip_bbox(tuple(int(x) for x in bb[:4]), (h, w))
        pred_mask = cct._bbox_to_mask((h, w), pred_bbox)

        bb_iou = float(cct._bbox_iou(gt.bbox, pred_bbox))
        mk_iou = float(cct._mask_iou(gt.mask, pred_mask))
        c_err = float(cct._center_error(gt.bbox, pred_bbox))

        bbox_ious.append(bb_iou)
        mask_ious.append(mk_iou)
        center_errs.append(c_err)
        prec_hits += int(c_err <= precision_thresh_px)
        succ_hits += int(bb_iou >= success_iou_thresh)
        n_matched += 1

        acc = bool(row.get("accepted", True))
        if acc:
            bbox_ious_acc.append(bb_iou)
            mask_ious_acc.append(mk_iou)
            center_errs_acc.append(c_err)
            n_accepted += 1

        per_rows.append(
            {
                "frame": int(row.get("frame", -1)),
                "image_stem": img_path.stem,
                "accepted": int(acc),
                "bbox_iou": bb_iou,
                "mask_iou": mk_iou,
                "center_error_px": c_err,
                "precision_hit": int(c_err <= precision_thresh_px),
                "success_hit": int(bb_iou >= success_iou_thresh),
                "lib_sim": float(row.get("score", 0) or 0),
                "reject_reason": str(row.get("reject_reason", "") or ""),
            }
        )

    def _mean(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    summary = {
        "results_json": str(results_json_path),
        "frames_dir": str(fd),
        "gt_mask_dir": str(gt_dir),
        "n_track_rows": len(track_rows),
        "n_gt_matched_frames": n_matched,
        "n_tracker_accepted_in_matched": n_accepted,
        "precision_thresh_px": float(precision_thresh_px),
        "success_iou_thresh": float(success_iou_thresh),
        "bbox_iou_mean": _mean(bbox_ious),
        "mask_iou_mean": _mean(mask_ious),
        "center_error_mean_px": _mean(center_errs),
        "precision_rate": float(prec_hits) / float(max(1, n_matched)),
        "success_rate": float(succ_hits) / float(max(1, n_matched)),
        "bbox_iou_mean_accepted_only": _mean(bbox_ious_acc),
        "mask_iou_mean_accepted_only": _mean(mask_ious_acc),
        "center_error_mean_px_accepted_only": _mean(center_errs_acc),
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summ_path = out / "eval_vs_gt_summary.json"
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = out / "eval_vs_gt_per_frame.csv"
    if per_rows:
        keys = sorted({k for r in per_rows for k in r})
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in per_rows:
                w.writerow({k: r.get(k, "") for k in keys})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return str(summ_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="MIS_Mesh_RTS_Track vs Chess_Box_Mask GT 评估")
    ap.add_argument("--results", default="", help="tracking_results.json")
    ap.add_argument("--frames-dir", default=os.environ.get("MIS_FRAMES_DIR", "").strip())
    ap.add_argument("--gt-mask-dir", default=os.environ.get("MIS_GT_MASK_DIR", "").strip())
    ap.add_argument("--out-dir", default="", help="默认与 results 同目录")
    ap.add_argument("--precision-thresh", type=float, default=20.0)
    ap.add_argument("--success-iou", type=float, default=0.5)
    args = ap.parse_args()

    rp = args.results.strip()
    if not rp:
        raise SystemExit("请指定 --results 或设置环境变量指向 tracking_results.json")
    fd = args.frames_dir.strip()
    gt = args.gt_mask_dir.strip()
    if not fd or not gt:
        raise SystemExit("请指定 --frames-dir / --gt-mask-dir 或对应 MIS_* 环境变量")

    out_dir = args.out_dir.strip() or str(Path(rp).parent)
    evaluate_and_save(
        results_json_path=rp,
        frames_dir=fd,
        gt_mask_dir_user=gt,
        out_dir=out_dir,
        precision_thresh_px=float(args.precision_thresh),
        success_iou_thresh=float(args.success_iou),
    )


if __name__ == "__main__":
    main()
