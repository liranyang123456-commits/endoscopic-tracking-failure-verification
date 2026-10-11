"""Dataset loading, grouped dual-domain OOF splits, and leakage checks."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Fold:
    fold_id: str
    target_domain: str
    test_sequence: str
    val_sequences: tuple[tuple[str, str], ...]
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class CandidateDataset(Dataset):
    def __init__(self, arrays: dict[str, np.ndarray], indices: np.ndarray):
        self.arrays = arrays
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        j = self.indices[i]
        return {
            "features": torch.from_numpy(self.arrays["features"][j]).float(),
            "boxes": torch.from_numpy(self.arrays["boxes"][j]).float(),
            "gt": torch.from_numpy(self.arrays["gt"][j]).float(),
            "index": torch.tensor(j, dtype=torch.long),
        }


class VisualCandidateDataset(CandidateDataset):
    def __init__(
        self, arrays: dict[str, np.ndarray], visual_maps: np.ndarray,
        visual_valid: np.ndarray, indices: np.ndarray,
    ):
        super().__init__(arrays, indices)
        self.visual_maps = visual_maps
        self.visual_valid = visual_valid

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        item = super().__getitem__(i)
        j = self.indices[i]
        item["maps"] = torch.from_numpy(self.visual_maps[j]).float().div_(255.0)
        item["visual_valid"] = torch.tensor(
            float(self.visual_valid[j]), dtype=torch.float32
        )
        return item


def sequence_balanced_weights(
    arrays: dict[str, np.ndarray], indices: np.ndarray
) -> np.ndarray:
    """Give each domain and each sequence equal expected sampling mass."""
    domains = np.asarray(arrays["domain"][indices]).astype(str)
    sequences = np.asarray(arrays["sequence"][indices]).astype(str)
    weights = np.zeros(len(indices), dtype=np.float64)
    unique_domains = sorted(set(domains))
    for domain in unique_domains:
        domain_mask = domains == domain
        domain_sequences = sorted(set(sequences[domain_mask]))
        for sequence in domain_sequences:
            mask = domain_mask & (sequences == sequence)
            weights[mask] = 1.0 / (
                len(unique_domains) * len(domain_sequences) * max(mask.sum(), 1)
            )
    return weights / weights.sum()


def load_arrays(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def _mask(arrays: dict[str, np.ndarray], pairs: set[tuple[str, str]]) -> np.ndarray:
    return np.asarray([
        (str(d), str(s)) in pairs
        for d, s in zip(arrays["domain"], arrays["sequence"])
    ])


def build_dual_oof_folds(arrays: dict[str, np.ndarray]) -> list[Fold]:
    domains = {
        "trackves": sorted(set(map(str, arrays["sequence"][arrays["domain"] == "trackves"]))),
        "chess": sorted(set(map(str, arrays["sequence"][arrays["domain"] == "chess"]))),
    }
    folds: list[Fold] = []
    for target_domain in ("trackves", "chess"):
        other_domain = "chess" if target_domain == "trackves" else "trackves"
        target_seqs, other_seqs = domains[target_domain], domains[other_domain]
        for i, test_seq in enumerate(target_seqs):
            val_target = target_seqs[(i + 1) % len(target_seqs)]
            val_other = other_seqs[i % len(other_seqs)]
            test_pairs = {(target_domain, test_seq)}
            val_pairs = {(target_domain, val_target), (other_domain, val_other)}
            test_mask = _mask(arrays, test_pairs)
            val_mask = _mask(arrays, val_pairs)
            train_mask = ~(test_mask | val_mask)
            fold = Fold(
                fold_id=f"{target_domain}_{test_seq}",
                target_domain=target_domain,
                test_sequence=test_seq,
                val_sequences=tuple(sorted(val_pairs)),
                train_idx=np.flatnonzero(train_mask),
                val_idx=np.flatnonzero(val_mask),
                test_idx=np.flatnonzero(test_mask),
            )
            assert not set(fold.train_idx) & set(fold.val_idx)
            assert not set(fold.train_idx) & set(fold.test_idx)
            assert not set(fold.val_idx) & set(fold.test_idx)
            assert len(fold.test_idx) > 0 and len(fold.val_idx) > 0
            folds.append(fold)
    return folds


def fit_feature_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = features.reshape(-1, features.shape[-1])
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    # Source flags/domain/availability remain semantically binary.
    binary_cols = [4, 5, 6, 7, 20, 21]
    mean[binary_cols] = 0.0
    std[binary_cols] = 1.0
    return mean, std


def apply_standardizer(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
