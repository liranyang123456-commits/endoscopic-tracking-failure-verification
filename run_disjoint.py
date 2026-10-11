from __future__ import annotations

import argparse
import importlib
import hashlib
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from protocol import (
    HERE, PROJECT, GroundTruth, align_predictions, canonical_chess,
    canonical_trackves, sha256, validate_split, write_csv, write_json,
)
from run_official_trackers import FrameReader


def groups(records: list[GroundTruth], domain: str) -> dict[str, list[GroundTruth]]:
    result = defaultdict(list)
    for record in records:
        result[f"{domain}:{record.sequence}"].append(record)
    return dict(result)


def build_cache(path: Path, per_sequence: int = 64) -> None:
    sys.path.insert(0, str(PROJECT))
    from train_contour_similarity import _rasterize_contour

    sequence_records = groups(canonical_chess(), "chess") | groups(
        canonical_trackves(), "trackves"
    )
    masks, names, frame_keys = [], [], []
    for name, records in sorted(sequence_records.items()):
        indices = np.unique(np.linspace(0, len(records) - 1, min(per_sequence, len(records))).astype(int))
        reader = FrameReader()
        try:
            for index in indices:
                record = records[int(index)]
                image = reader.read(record)
                x, y, w, h = (int(round(value)) for value in record.box)
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(image.shape[1], x + w), min(image.shape[0], y + h)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"Invalid training ROI: {name} {record.frame}")
                gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
                binary = np.where(gray > 68, 255, 0).astype(np.uint8)
                raster = _rasterize_contour(binary, 128)
                masks.append(np.round(raster * 255).astype(np.uint8))
                names.append(name)
                frame_keys.append(f"{name}:{record.unit}:{record.frame}")
        finally:
            reader.close()
        print(f"CACHE {name}: {len(indices)} image-derived patches", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, masks=np.stack(masks), sequence=np.asarray(names),
        frame_key=np.asarray(frame_keys),
    )
    write_json(path.with_suffix(".manifest.json"), {
        "sequence_count": len(sequence_records), "patch_count": len(masks),
        "patches_per_sequence_limit": per_sequence,
        "training_input": "fixed-gray-cutoff image raster inside annotated ROI boxes",
        "human_pixel_masks_used_as_training_targets": False,
        "roi_box_supervision": True, "cache_sha256": sha256(path),
        "cached_heldout_patches_are_never_used_in_fitting": True,
    })


class ContourPairs(Dataset):
    def __init__(self, masks: np.ndarray, length: int, seed: int):
        if len(masks) < 2:
            raise ValueError("Contour fitting requires at least two training patches")
        self.masks, self.length, self.seed, self.epoch = masks, length, seed, 0

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):
        rng = np.random.default_rng(self.seed + self.epoch * 1000003 + index)
        first = int(rng.integers(len(self.masks)))
        a = self.masks[first].astype(np.float32) / 255
        if rng.random() < 0.58:
            angle, scale = float(rng.uniform(-25, 25)), float(rng.uniform(0.85, 1.15))
            transform = cv2.getRotationMatrix2D((63.5, 63.5), angle, scale)
            transform[:, 2] += rng.uniform(-8, 8, size=2)
            b = cv2.warpAffine(a, transform, (128, 128), flags=cv2.INTER_LINEAR)
            target = 1.0
            if rng.random() < 0.42:
                width, height = int(rng.integers(12, 40)), int(rng.integers(12, 40))
                x, y = int(rng.integers(0, 128 - width)), int(rng.integers(0, 128 - height))
                b[y:y + height, x:x + width] = 0
                target = 1.0 - width * height / (128 * 128)
        else:
            second = int(rng.integers(len(self.masks) - 1))
            second += int(second >= first)
            b = self.masks[second].astype(np.float32) / 255
            target = 0.1
        return (
            torch.from_numpy(a[None].copy()), torch.from_numpy(b[None].copy()),
            torch.tensor(target, dtype=torch.float32),
        )


def split_for(names: list[str], test: str) -> tuple[list[str], list[str]]:
    target_domain = test.split(":")[0]
    same = sorted(name for name in names if name.startswith(target_domain + ":"))
    other = sorted(name for name in names if not name.startswith(target_domain + ":"))
    validation = [same[(same.index(test) + 1) % len(same)], other[same.index(test) % len(other)]]
    train = sorted(set(names) - set(validation) - {test})
    validate_split(train, validation, [test])
    return train, validation


@torch.no_grad()
def validation_loss(model, loader, device) -> float:
    model.eval()
    total = count = 0
    for a, b, target in loader:
        prediction = model(a.to(device), b.to(device))[0]
        total += float(torch.nn.functional.mse_loss(prediction, target.to(device), reduction="sum"))
        count += len(target)
    if count == 0:
        raise ValueError("Empty validation data")
    return total / count


def train(cache: Path, output: Path, epochs: int, samples: int, selected: str = "") -> None:
    sys.path.insert(0, str(PROJECT))
    from contour_similarity_net import ContourSimilarityNet

    data = np.load(cache, allow_pickle=False)
    names = sorted(np.unique(data["sequence"]).tolist())
    if len(names) != 20:
        raise ValueError("Expected all 9 TrackVes and 11 chess sequence groups")
    target_names = [selected] if selected else names
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for number, test in enumerate(target_names, 1):
        if test not in names:
            raise ValueError(f"Unknown held-out sequence: {test}")
        seed = 20261011 + names.index(test)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        training, validation = split_for(names, test)
        train_mask = np.isin(data["sequence"], training)
        val_mask = np.isin(data["sequence"], validation)
        if np.any(train_mask & (data["sequence"] == test)) or np.any(val_mask & (data["sequence"] == test)):
            raise ValueError("Held-out sequence entered contour fitting or selection")
        train_data = ContourPairs(data["masks"][train_mask], samples, seed)
        val_data = ContourPairs(data["masks"][val_mask], 512, seed + 100)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=64, shuffle=False, num_workers=0)
        model = ContourSimilarityNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        best_loss, best_epoch, best_state = float("inf"), -1, None
        history = []
        for epoch in range(epochs):
            model.train()
            train_data.epoch = epoch
            total = count = 0
            for a, b, target in train_loader:
                prediction = model(a.to(device), b.to(device))[0]
                loss = torch.nn.functional.mse_loss(prediction, target.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite contour training loss: {test}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(target)
                count += len(target)
            scheduler.step()
            score = validation_loss(model, val_loader, device)
            history.append({"epoch": epoch + 1, "train_mse": total / count, "validation_mse": score})
            if score < best_loss:
                best_loss, best_epoch = score, epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError(f"No selected contour checkpoint for {test}")
        name = test.replace(":", "_")
        checkpoint = output / "contour_folds" / f"{name}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "test_sequences": [test], "validation_sequences": validation,
            "training_sequences": training, "initialization": "random; no legacy checkpoint",
            "all_learned_components_test_excluded": True,
            "epochs": epochs, "sampled_pairs_per_epoch": samples, "batch_size": 64,
            "learning_rate": 3e-4, "optimizer": "AdamW", "seed": seed,
            "best_validation_epoch": best_epoch, "best_validation_mse": best_loss,
            "training_targets": "image-derived fixed-gray rasters and synthetic pair similarity",
            "pixel_annotation_supervision": False, "roi_box_supervision": True,
            "cache_sha256": sha256(cache),
        }
        torch.save({"model": best_state, "metadata": metadata}, checkpoint)
        write_json(checkpoint.with_suffix(".manifest.json"), {
            **metadata, "checkpoint_sha256": sha256(checkpoint), "history": history,
        })
        print(f"TRAIN [{number}/{len(target_names)}] {test}, best epoch={best_epoch}, "
              f"validation MSE={best_loss:.6f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def infer_sequence(test: str, output: Path) -> None:
    checkpoint = output / "contour_folds" / f"{test.replace(':', '_')}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    os.environ.update({
        "MIS_CONTOUR_SIM_CKPT": str(checkpoint),
        "MIS_CONTOUR_SIM_CONF_CKPT": str(checkpoint),
        "MIS_CHESS_ONLINE_SIMI": "0", "MIS_FUSION_HEAD_NPZ": "",
        "MIS_SILENT": "1", "MIS_SILENT_IMPORT": "1",
    })
    sys.path.insert(0, str(PROJECT))
    frozen = importlib.import_module("2026-06-03-ours_scheme_c_nano方法")
    frozen.apply_method_defaults("v3_chess")
    domain, sequence = test.split(":", 1)
    gt = canonical_chess() if domain == "chess" else canonical_trackves()
    gt = [record for record in gt if record.sequence == sequence]
    units = defaultdict(list)
    for record in gt:
        units[record.unit].append(record)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    validate_split(metadata["training_sequences"], metadata["validation_sequences"], [test])
    predictions, input_hashes = {}, {}
    for unit, records in sorted(units.items()):
        reader = FrameReader()
        try:
            image = reader.read(records[0])
            input_hashes[str(records[0].key)] = hashlib.sha256(image.tobytes()).hexdigest()
            tracker = frozen.create_tracker(image, tuple(int(v) for v in records[0].box))
            if tracker.contour_sim_model is None:
                raise RuntimeError("A fallback replaced the required fold-specific contour network")
            current_state = tracker.contour_sim_model.state_dict()
            if not all(torch.equal(current_state[key].cpu(), tensor) for key, tensor in payload["model"].items()):
                raise RuntimeError("Inference did not load the held-out-fold checkpoint exactly")
            if hasattr(tracker, "_defm") and Path(tracker._defm._CONTOUR_SIM_CONF_CKPT) != checkpoint:
                raise RuntimeError("Confidence model inherited a legacy checkpoint")
            predictions[records[0].key] = records[0].box
            for record in records[1:]:
                image = reader.read(record)
                input_hashes[str(record.key)] = hashlib.sha256(image.tobytes()).hexdigest()
                result = tracker.update(image)
                values = tuple(float(value) for value in result["bbox"])
                if len(values) != 4 or not all(np.isfinite(values)):
                    raise ValueError(f"Invalid fold tracker prediction: {test} {record.frame}")
                predictions[record.key] = values if values[2] > 0 and values[3] > 0 else None
        finally:
            reader.close()
        print(f"INFER {test}/{unit}: {len(records)} GT frames", flush=True)
    write_csv(output / "disjoint" / domain / f"{sequence}.csv", align_predictions(gt, predictions))
    write_json(output / "disjoint" / domain / f"{sequence}.manifest.json", {
        "checkpoint_sha256": sha256(checkpoint), **metadata,
        "fold_checkpoint_exactly_verified_at_inference": True,
        "confidence_checkpoint": str(checkpoint),
        "external_fitted_fusion_head": None,
        "offline_autoencoder_loaded": False,
        "online_test_sequence_fine_tuning": False,
        "protocol": "shared first-GT initialization; GT-frame updates",
        "input_frame_sha256": input_hashes,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("cache", "train", "infer", "all"))
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples-per-epoch", type=int, default=1024)
    parser.add_argument("--sequence", default="")
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output = args.output.resolve()
    cache = args.cache.resolve() if args.cache else args.output / "image_derived_contour_patches.npz"
    if args.stage in ("cache", "all"):
        build_cache(cache)
    if args.stage in ("train", "all"):
        train(cache, args.output, args.epochs, args.samples_per_epoch, args.sequence)
    if args.stage == "infer":
        if not args.sequence:
            parser.error("--sequence is required for isolated inference")
        infer_sequence(args.sequence, args.output)
    if args.stage == "all":
        names = sorted(np.unique(np.load(cache, allow_pickle=False)["sequence"]).tolist())
        for sequence in names:
            subprocess.run([
                sys.executable, str(Path(__file__).resolve()), "infer",
                "--output", str(args.output), "--sequence", sequence,
            ], check=True, cwd=HERE)


if __name__ == "__main__":
    main()
