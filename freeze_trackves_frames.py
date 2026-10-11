from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import cv2

from protocol import HERE, canonical_trackves, sha256, write_json


def main() -> None:
    root = HERE / "results" / "frozen_trackves_frames"
    groups = defaultdict(list)
    for record in canonical_trackves():
        groups[record.sequence].append(record)
    hashes = {}
    for sequence, records in sorted(groups.items()):
        video = Path(records[0].image_path)
        capture = cv2.VideoCapture(
            str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        if not capture.isOpened():
            raise OSError(f"Cannot open single-thread decoder: {video}")
        requested = {record.frame: record for record in records}
        destination = root / sequence
        destination.mkdir(parents=True, exist_ok=True)
        decoded = 0
        try:
            for frame in range(max(requested) + 1):
                ok, image = capture.read()
                if not ok:
                    raise OSError(f"Cannot decode {sequence} frame {frame}; GT frame cannot be removed")
                if frame not in requested:
                    continue
                path = destination / f"{frame:06d}.png"
                if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
                    raise OSError(f"Cannot persist lossless frozen frame: {path}")
                restored = cv2.imread(str(path))
                original_hash = hashlib.sha256(image.tobytes()).hexdigest()
                if original_hash != hashlib.sha256(restored.tobytes()).hexdigest():
                    raise ValueError(f"Lossless frame round trip mismatch: {path}")
                hashes[str(requested[frame].key)] = original_hash
                decoded += 1
        finally:
            capture.release()
        if decoded != len(records):
            raise ValueError(f"GT-frame decoding coverage mismatch: {sequence}")
        print(f"FROZEN {sequence}: {decoded}/{len(records)} annotated frames", flush=True)
    write_json(root / "manifest.json", {
        "opencv_version": cv2.__version__, "decoder_threads": 1,
        "decode_strategy": "one sequential read pass; no random seeks or grab-only skips",
        "frame_count": len(hashes), "input_frame_sha256": hashes,
        "all_annotated_frames_retained": True,
        "lossless_round_trip_every_frame_verified": True,
        "videos": {sequence: sha256(Path(records[0].image_path)) for sequence, records in groups.items()},
    })


if __name__ == "__main__":
    main()
