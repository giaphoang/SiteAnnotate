from __future__ import annotations

import shlex
import statistics
import subprocess
import tempfile
from pathlib import Path

from .io import write_jsonl
from .models import FrameRecord, SequenceManifest
from .rellis import resolve_frame_path


def _timestamp_durations(frames: list[FrameRecord], nominal_fps: float) -> list[float]:
    timestamps = [frame.camera_timestamp_ns for frame in frames]
    durations: list[float] = []
    for index in range(len(frames)):
        if index + 1 < len(frames) and timestamps[index] is not None and timestamps[index + 1] is not None:
            delta = (timestamps[index + 1] - timestamps[index]) / 1_000_000_000
            durations.append(max(0.001, delta))
        else:
            durations.append(1.0 / nominal_fps)
    return durations


def _nominal_fps(frames: list[FrameRecord], default: float = 10.0) -> float:
    deltas = [
        (right.camera_timestamp_ns - left.camera_timestamp_ns) / 1_000_000_000
        for left, right in zip(frames, frames[1:])
        if left.camera_timestamp_ns is not None
        and right.camera_timestamp_ns is not None
        and right.camera_timestamp_ns > left.camera_timestamp_ns
    ]
    if not deltas:
        return default
    return 1.0 / statistics.median(deltas)


def _prepare_overlay(source: Path, destination: Path, text: str) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("timestamp overlays require Pillow; install the runtime extra") from exc
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 900), 36), fill=(0, 0, 0))
    draw.text((8, 8), text, fill=(255, 255, 255))
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def build_timestamped_video(
    manifest: SequenceManifest,
    frames: list[FrameRecord],
    output_video: Path,
    mapping_path: Path,
    *,
    ffmpeg: str = "ffmpeg",
    overlay_timestamps: bool = True,
    nominal_fps: float | None = None,
) -> dict[str, object]:
    if not frames:
        raise ValueError("cannot encode an empty frame sequence")
    nominal_fps = nominal_fps or _nominal_fps(frames)
    durations = _timestamp_durations(frames, nominal_fps)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="siteannotate-video-") as temporary:
        temporary_path = Path(temporary)
        concat_path = temporary_path / "frames.txt"
        concat_lines: list[str] = []
        mapping: list[dict[str, object]] = []
        for index, (frame, duration) in enumerate(zip(frames, durations)):
            source = resolve_frame_path(manifest, frame.camera_path)
            if not source.exists():
                raise FileNotFoundError(source)
            image_path = source
            if overlay_timestamps:
                image_path = temporary_path / f"{index:08d}.png"
                timestamp = frame.camera_timestamp_ns if frame.camera_timestamp_ns is not None else "unknown"
                _prepare_overlay(source, image_path, f"{frame.sample_id}  timestamp_ns={timestamp}")
            concat_lines.append(f"file {shlex.quote(str(image_path))}\n")
            concat_lines.append(f"duration {duration:.9f}\n")
            mapping.append(
                {
                    "encoded_frame_index": index,
                    "frame_key": frame.frame_key,
                    "sample_id": frame.sample_id,
                    "camera_frame_index": frame.camera_frame_index,
                    "lidar_scan_index": frame.lidar_scan_index,
                    "camera_path": frame.camera_path,
                    "lidar_path": frame.lidar_path,
                    "camera_timestamp_ns": frame.camera_timestamp_ns,
                    "lidar_timestamp_ns": frame.lidar_timestamp_ns,
                    "duration_s": duration,
                }
            )
        concat_path.write_text("".join(concat_lines), encoding="utf-8")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-vsync",
            "vfr",
            "-pix_fmt",
            "yuv420p",
            str(output_video),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(f"FFmpeg failed: {completed.stderr.strip()}")
    write_jsonl(mapping_path, mapping)
    return {"video": str(output_video), "mapping": str(mapping_path), "frame_count": len(frames), "fps": nominal_fps}
