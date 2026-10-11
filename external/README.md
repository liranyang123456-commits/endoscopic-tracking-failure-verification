# Third-party source trees (clone at the pinned commits)

| Repository | Commit | Local folder |
|---|---|---|
| https://github.com/botaoye/OSTrack | `33b5e12586216b7fd0e95d255bd01ba44cbec759` | `external/OSTrack` |
| https://github.com/researchmm/Stark | `162e1d37a38f92251cb380e90ccdb59608f2df3c` | `external/STARK` |
| https://github.com/JonathonLuiten/TrackEval | `12c8791b303e0a0b50f753af204249e622d0281a` | `external/TrackEval` |

The only local modifications made during the study were compatibility shims for current PyTorch/NumPy
(`torch._six` aliases, the torchvision URL-loader import, and `np.float` in TrackEval's HOTA). No architecture,
trained head, scoring rule or HOTA equation was changed. `run_official_trackers.py` records the commit and the
checkpoint hash in every run manifest.
