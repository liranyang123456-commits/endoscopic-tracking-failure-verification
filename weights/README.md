# Model weights (not redistributed)

Place the following files in this folder. The SHA-256 values are those recorded in the run manifests under
`results/external_verification/*/*.manifest.json` and `results/official/*/*.manifest.json`.

| File | Source | SHA-256 |
|---|---|---|
| `ostrack_vitb256_ce.pth.tar` | OSTrack author model folder https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy (file ID `1dySfPrSk-knAEo0lTXqZ1kMk3BCvVofV`, `vitb_256_mae_ce_32x4_ep300`) | `8e01d6251569ac84dbb53fdd062cd938551be2d889582f789aa3070196f42f41` |
| `stark_st50.pth.tar` | STARK author model folder https://drive.google.com/drive/folders/1fSgll53ZnVKeUn22W37Nijk-b9LGhMdN (file ID `1sV_idlYLyxeCIO2o4AQDvUFO5b-X-E4w`, `STARK-ST50`) | `23c8a677d9c2163e96e5225c1648ac4803e43232aef7c79044a367bde4a86b95` |
| `nanotrack_backbone_sim.onnx` | https://github.com/HonglinChu/SiamTrackers (NanoTrack V2 ONNX, as used by OpenCV `TrackerNano`) | `530bdd0cd00f19afab79a863e71ba71e3312395a5dc9151af675082bdaaa2fc4` |
| `nanotrack_head_sim.onnx` | same | `0d8c0637be849f092cc7236cae02e55c8b9455ebe37ba50601d6115db4247cd9` |

`run_official_trackers.py` converts the author archives once into `*.state.pt` files containing only the
network tensors and loads them with `strict=True`.
