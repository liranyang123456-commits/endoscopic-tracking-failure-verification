"""Sequence-LOSO CNN quality gate for global re-detection candidates."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from candidate_exporter import box_iou
from global_redetection_gate_oof import _features, _iou_vector


class VisualQualityGate(nn.Module):
    def __init__(self, scalar_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(32 + scalar_dim + 3, 48), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(48, 1), nn.Sigmoid(),
        )

    def forward(self, maps: torch.Tensor, scalar: torch.Tensor) -> torch.Tensor:
        batch, candidates = maps.shape[:2]
        embedding = self.encoder(maps.flatten(0, 1)).reshape(batch, candidates, -1)
        expanded = scalar[:, None].expand(-1, candidates, -1)
        identity = torch.eye(candidates, device=maps.device)[None].expand(batch, -1, -1)
        return self.head(torch.cat([embedding, expanded, identity], dim=2)).squeeze(-1)


@torch.no_grad()
def _predict(
    model: nn.Module, maps: np.ndarray, scalar: np.ndarray,
    indices: np.ndarray, device: torch.device,
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(indices), 512):
        batch = indices[start:start + 512]
        output.append(model(
            torch.from_numpy(maps[batch]).float().to(device) / 255.0,
            torch.from_numpy(scalar[batch]).float().to(device),
        ).cpu().numpy())
    return np.concatenate(output)


def _best_route(
    quality: np.ndarray, actual: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    best_score, best_threshold, best_choice = -1.0, 0.0, np.zeros(len(actual), int)
    winner = np.argmax(quality, axis=1)
    gain = quality[np.arange(len(winner)), winner] - quality[:, 0]
    for threshold in np.linspace(0.0, 0.4, 41):
        choice = np.where(gain > threshold, winner, 0)
        score = float(actual[np.arange(len(choice)), choice].mean())
        if score > best_score:
            best_score, best_threshold, best_choice = score, float(threshold), choice
    return best_score, best_threshold, best_choice


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("artifacts/candidate_dataset.npz"))
    parser.add_argument("--redetection", type=Path, default=Path("artifacts/global_redetection_all.npz"))
    parser.add_argument("--visual", type=Path, default=Path("artifacts/global_redetection_visual.npz"))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--output", type=Path, default=Path("artifacts/global_visual_gate_oof.json"))
    parser.add_argument(
        "--fold-dir", type=Path,
        default=Path("artifacts/global_visual_gate_folds"),
    )
    args = parser.parse_args()
    random.seed(20260718)
    np.random.seed(20260718)
    torch.manual_seed(20260718)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.dataset, allow_pickle=False)
    redetection = np.load(args.redetection, allow_pickle=False)
    visual = np.load(args.visual, allow_pickle=False)
    indices = redetection["indices"].astype(int)
    boxes = redetection["boxes"].astype(np.float32)
    candidate_scores = redetection["scores"].astype(np.float32)
    maps = visual["maps"]
    valid = visual["valid"].astype(bool)
    sequence = data["sequence"][indices].astype(str)
    frame_idx = data["frame_idx"][indices].astype(np.float32)
    gt = data["gt"][indices]
    nano_score = data["features"][indices, 1, 9].astype(np.float32)
    scalar_raw = _features(boxes, candidate_scores, nano_score, frame_idx)
    targets = np.stack([
        _iou_vector(boxes[:, candidate], gt) for candidate in range(3)
    ], axis=1).astype(np.float32)
    names = sorted(np.unique(sequence))
    selected = np.zeros(len(indices), dtype=int)
    predicted = np.zeros_like(targets)
    fold_rows = []
    for fold, test_name in enumerate(names):
        val_name = names[(fold + 1) % len(names)]
        test_idx = np.flatnonzero((sequence == test_name) & valid)
        val_idx = np.flatnonzero((sequence == val_name) & valid)
        train_idx = np.flatnonzero(
            (sequence != test_name) & (sequence != val_name) & valid
        )
        mean = scalar_raw[train_idx].mean(axis=0)
        std = scalar_raw[train_idx].std(axis=0)
        std[std < 1e-6] = 1.0
        scalar = (scalar_raw - mean) / std
        weights = np.zeros(len(train_idx), dtype=np.float32)
        for name in np.unique(sequence[train_idx]):
            mask = sequence[train_idx] == name
            weights[mask] = 1.0 / max(int(mask.sum()), 1)
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights), len(train_idx), replacement=True
        )
        dataset = TensorDataset(
            torch.from_numpy(maps[train_idx]).float() / 255.0,
            torch.from_numpy(scalar[train_idx]).float(),
            torch.from_numpy(targets[train_idx]).float(),
        )
        loader = DataLoader(dataset, batch_size=256, sampler=sampler)
        model = VisualQualityGate(scalar.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
        best_score, best_state, best_threshold = -1.0, None, 0.0
        for _ in range(args.epochs):
            model.train()
            for batch_maps, batch_scalar, batch_target in loader:
                prediction = model(batch_maps.to(device), batch_scalar.to(device))
                loss = nn.functional.smooth_l1_loss(
                    prediction, batch_target.to(device), beta=0.1
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val_quality = _predict(model, maps, scalar, val_idx, device)
            score, threshold, _ = _best_route(val_quality, targets[val_idx])
            if score > best_score:
                best_score, best_threshold = score, threshold
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
        assert best_state is not None
        model.load_state_dict(best_state)
        args.fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": best_state, "scalar_mean": mean.astype(np.float32),
            "scalar_std": std.astype(np.float32), "threshold": best_threshold,
            "test_sequence": test_name, "validation_sequence": val_name,
            "scalar_dim": int(scalar.shape[1]), "parameters": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
        }, args.fold_dir / f"{test_name}.pt")
        test_quality = _predict(model, maps, scalar, test_idx, device)
        winner = np.argmax(test_quality, axis=1)
        gain = test_quality[np.arange(len(winner)), winner] - test_quality[:, 0]
        choice = np.where(gain > best_threshold, winner, 0)
        selected[test_idx], predicted[test_idx] = choice, test_quality
        chosen = targets[test_idx][np.arange(len(choice)), choice]
        fold_rows.append({
            "sequence": test_name, "validation_sequence": val_name,
            "threshold": best_threshold, "nano_iou": float(targets[test_idx, 0].mean()),
            "gated_iou": float(chosen.mean()),
            "template_rate": float(np.mean(choice == 1)),
            "orb_rate": float(np.mean(choice == 2)),
        })
        print(json.dumps(fold_rows[-1]), flush=True)
    report = {
        "protocol": "strict sequence LOSO; validation-selected epoch and margin",
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "macro_nano_iou": float(np.mean([row["nano_iou"] for row in fold_rows])),
        "macro_gated_iou": float(np.mean([row["gated_iou"] for row in fold_rows])),
        "sequences": fold_rows,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output.with_suffix(".npz"), selected=selected,
        predicted_quality=predicted, indices=indices,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
