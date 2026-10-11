from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from protocol import (
    HERE, LEARNED, PROJECT, align_predictions, box_iou,
    canonical_trackves, sha256, validate_split, write_csv, write_json,
)
from run_official_trackers import FrameReader

for directory in (PROJECT, LEARNED):
    sys.path.insert(0, str(directory))
from global_redetection_gate_oof import _features
from global_redetection_probe import OrbRedetector, template_candidate
from global_redetection_visual_gate_oof import VisualQualityGate, _best_route, _predict
from global_gate_online_tracker import GlobalGateOnlineTracker
from visual_feature_exporter import _crop, _responses
from trackves_eval_all_methods import NanoTrackTracker


def build(output: Path) -> None:
    records = canonical_trackves()
    by_sequence = defaultdict(list)
    for record in records:
        by_sequence[record.sequence].append(record)
    maps, scalars, targets, sequence_names, frame_numbers = [], [], [], [], []
    input_hashes = {}
    for sequence, gt in sorted(by_sequence.items()):
        reader = FrameReader()
        try:
            first = reader.read(gt[0])
            nano = NanoTrackTracker(first, tuple(int(x) for x in gt[0].box))
            template = GlobalGateOnlineTracker._raw_crop(first, np.asarray(gt[0].box))
            import cv2
            template_map = cv2.resize(template, (24, 24), interpolation=cv2.INTER_AREA)
            orb = OrbRedetector(template)
            for record in gt:
                if record.is_initialization:
                    input_hashes[str(record.key)] = hashlib.sha256(first.tobytes()).hexdigest()
                    continue
                image = reader.read(record)
                input_hashes[str(record.key)] = hashlib.sha256(image.tobytes()).hexdigest()
                nano_result = nano.update(image)
                nano_box = np.asarray(nano_result["bbox"], dtype=np.float32)
                template_box, template_score = template_candidate(
                    image, template, (0.65, 0.8, 1.0, 1.2, 1.4)
                )
                orb_box, orb_score = orb.detect(image)
                boxes = np.stack([nano_box, template_box, orb_box]).astype(np.float32)
                scores = np.asarray([[1.0, template_score, orb_score]], dtype=np.float32)
                scalar = _features(
                    boxes[None], scores,
                    np.asarray([nano_result["score"]], dtype=np.float32),
                    np.asarray([record.frame], dtype=np.float32),
                )[0]
                response = np.stack([_responses(_crop(image, box), template_map) for box in boxes])
                target = np.asarray([box_iou(record.box, tuple(box)) for box in boxes], dtype=np.float32)
                maps.append(response)
                scalars.append(scalar)
                targets.append(target)
                sequence_names.append(sequence)
                frame_numbers.append(record.frame)
        finally:
            reader.close()
        print(f"GATE CACHE {sequence}: {len(gt) - 1} scored frames", flush=True)
    path = output / "gate_training_inputs.npz"
    np.savez_compressed(
        path, maps=np.stack(maps), scalar=np.stack(scalars), targets=np.stack(targets),
        sequence=np.asarray(sequence_names), frame_idx=np.asarray(frame_numbers),
    )
    write_json(path.with_suffix(".manifest.json"), {
        "input_sha256": sha256(path), "input_frame_sha256": input_hashes,
        "learned_feature_extractor": None,
        "frozen_foundation_model": "official OpenCV NanoTrack",
        "ContourSimilarityNet_used": False, "offline_autoencoder_used": False,
        "ground_truth_in_inference_features": False,
        "GT_role": "first-box initialization and supervised training targets only",
    })


def train(output: Path, epochs: int = 18) -> None:
    path = output / "gate_training_inputs.npz"
    data = np.load(path, allow_pickle=False)
    maps = data["maps"]
    scalar_raw = data["scalar"]
    targets = data["targets"]
    sequence = data["sequence"].astype(str)
    names = sorted(np.unique(sequence))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifests = []
    for fold, test_name in enumerate(names):
        seed = 20261011 + fold
        np.random.seed(seed)
        torch.manual_seed(seed)
        validation_name = names[(fold + 1) % len(names)]
        training_names = [n for n in names if n not in (test_name, validation_name)]
        validate_split(training_names, [validation_name], [test_name])
        training_idx = np.flatnonzero(np.isin(sequence, training_names))
        validation_idx = np.flatnonzero(sequence == validation_name)
        mean, std = scalar_raw[training_idx].mean(axis=0), scalar_raw[training_idx].std(axis=0)
        std[std < 1e-6] = 1
        scalar = (scalar_raw - mean) / std
        weights = np.asarray([
            1.0 / np.sum(sequence[training_idx] == name)
            for name in sequence[training_idx]
        ], dtype=np.float32)
        dataset = TensorDataset(
            torch.from_numpy(maps[training_idx]).float() / 255,
            torch.from_numpy(scalar[training_idx]).float(),
            torch.from_numpy(targets[training_idx]).float(),
        )
        sampler = WeightedRandomSampler(torch.from_numpy(weights), len(training_idx), replacement=True)
        loader = DataLoader(dataset, batch_size=256, sampler=sampler, num_workers=0)
        model = VisualQualityGate(scalar.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
        best_score, best_state, best_threshold, best_epoch = -1.0, None, 0.0, 0
        for epoch in range(epochs):
            model.train()
            for response, features, target in loader:
                prediction = model(response.to(device), features.to(device))
                loss = torch.nn.functional.smooth_l1_loss(prediction, target.to(device), beta=0.1)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite gate loss: {test_name}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            quality = _predict(model, maps, scalar, validation_idx, device)
            score, threshold, _ = _best_route(quality, targets[validation_idx])
            if score > best_score:
                best_score, best_threshold, best_epoch = score, threshold, epoch + 1
                best_state = {key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError(f"No validation-selected gate checkpoint: {test_name}")
        checkpoint = output / "gate_folds" / f"{test_name}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "training_sequences": training_names, "validation_sequences": [validation_name],
            "test_sequences": [test_name], "seed": seed,
            "best_validation_epoch": best_epoch, "epochs": epochs,
            "scalar_normalization_fit_on": training_names,
            "checkpoint_and_threshold_selected_on": [validation_name],
            "learned_components": ["VisualQualityGate"],
            "ContourSimilarityNet_used": False, "offline_autoencoder_used": False,
            "all_active_learned_components_test_excluded": True,
            "training_input_sha256": sha256(path),
        }
        torch.save({
            "state_dict": best_state, "scalar_mean": mean.astype(np.float32),
            "scalar_std": std.astype(np.float32), "threshold": best_threshold,
            "scalar_dim": int(scalar.shape[1]), "metadata": metadata,
        }, checkpoint)
        write_json(checkpoint.with_suffix(".manifest.json"), {
            **metadata, "checkpoint_sha256": sha256(checkpoint),
        })
        manifests.append(metadata)
        print(f"GATE TRAIN {test_name}: epoch={best_epoch}, margin={best_threshold:.2f}", flush=True)
    write_json(output / "gate_split_manifest.json", manifests)


def infer(output: Path) -> None:
    by_sequence = defaultdict(list)
    for record in canonical_trackves():
        by_sequence[record.sequence].append(record)
    predictions, input_hashes, raw_nano_predictions, route_rows = {}, {}, {}, []
    for sequence, records in sorted(by_sequence.items()):
        checkpoint = output / "gate_folds" / f"{sequence}.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        validate_split(metadata["training_sequences"], metadata["validation_sequences"], [sequence])
        reader = FrameReader()
        try:
            image = reader.read(records[0])
            tracker = GlobalGateOnlineTracker(
                image, tuple(int(x) for x in records[0].box), checkpoint,
                frame_indices=[r.frame for r in records],
            )
            predictions[records[0].key] = records[0].box
            raw_nano_predictions[records[0].key] = records[0].box
            input_hashes[str(records[0].key)] = hashlib.sha256(image.tobytes()).hexdigest()
            for record in records[1:]:
                image = reader.read(record)
                input_hashes[str(record.key)] = hashlib.sha256(image.tobytes()).hexdigest()
                predictions[record.key] = tuple(float(v) for v in tracker.update(image)["bbox"])
                raw_nano_predictions[record.key] = tuple(float(v) for v in tracker.nano.last_bbox)
                route_rows.append({
                    "sequence": sequence, "frame_idx": record.frame,
                    "selected_candidate": ("nano", "template", "orb")[tracker.last_route],
                })
        finally:
            reader.close()
        print(f"GATE INFER {sequence}: {len(records) - 1} scored frames", flush=True)
    gt = [record for records in by_sequence.values() for record in records]
    write_csv(output / "official" / "trackves" / "learnedgate.csv", align_predictions(gt, predictions))
    write_csv(
        output / "official" / "trackves" / "gate_ungated_nano.csv",
        align_predictions(gt, raw_nano_predictions),
    )
    write_csv(output / "gate_route_audit.csv", route_rows)
    write_json(output / "official" / "trackves" / "learnedgate.manifest.json", {
        "input_frame_sha256": input_hashes, "active_learning": "sequence-specific VisualQualityGate",
        "ContourSimilarityNet_used": False, "all_active_learning_sequence_disjoint": True,
        "first_gt_frame_is_initialization_not_scored": True,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("build", "train", "infer", "all"))
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--epochs", type=int, default=18)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.stage in ("build", "all"):
        build(args.output)
    if args.stage in ("train", "all"):
        train(args.output, args.epochs)
    if args.stage in ("infer", "all"):
        infer(args.output)


if __name__ == "__main__":
    main()
