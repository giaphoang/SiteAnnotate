from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from .io import read_json, read_jsonl, write_json, write_jsonl


def vendor_root() -> Path:
    return Path(__file__).resolve().parents[2] / "vendor" / "paidf-curation-and-retrieval"


def require_paidf() -> Path:
    root = vendor_root()
    if not (root / "Makefile").exists():
        raise RuntimeError("PAIDF source is not initialized; run make bootstrap-sources")
    return root


def run_make(target: str, variables: dict[str, str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    root = require_paidf()
    command = ["make", target]
    command.extend(f"{key}={value}" for key, value in variables.items())
    environment = os.environ.copy()
    completed = subprocess.run(command, cwd=root, env=environment, text=True, capture_output=True)
    if check and completed.returncode:
        raise RuntimeError(f"PAIDF make {target} failed:\n{completed.stdout}\n{completed.stderr}")
    return completed


def _time_value(record: dict[str, Any], names: Iterable[str]) -> float | None:
    for name in names:
        value = record.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _dict_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _dict_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _dict_records(child)


def _frames_for_interval(mapping: list[dict[str, Any]], start_s: float, end_s: float) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    cursor = 0.0
    for item in mapping:
        duration = max(0.001, float(item.get("duration_s", 0.0)))
        next_cursor = cursor + duration
        if cursor < end_s and next_cursor > start_s:
            frames.append(item)
        cursor = next_cursor
    return frames


def write_curation_provenance(run_dir: Path, config_file: Path) -> Path:
    sequence = read_json(run_dir / "sequence.json")
    mapping_path = run_dir / "videos" / f"{sequence['sequence_id']}.frames.jsonl"
    if not mapping_path.exists():
        raise FileNotFoundError(f"timestamp mapping is missing; encode the run first: {mapping_path}")
    output = run_dir / "curation" / "provenance.json"
    write_json(
        output,
        {
            "schema_version": 1,
            "sequence_id": sequence["sequence_id"],
            "source_video": f"videos/{sequence['sequence_id']}.mp4",
            "source_frame_mapping": f"videos/{sequence['sequence_id']}.frames.jsonl",
            "paidf_config": str(config_file),
            "data_dir": str((run_dir / "curation-input").resolve()),
            "timestamp_policy": sequence["timestamp_policy"],
        },
    )
    return output


def write_curation_links(run_dir: Path) -> Path:
    """Materialize PAIDF clip windows as frame-to-LiDAR provenance rows."""
    sequence = read_json(run_dir / "sequence.json")
    mapping_path = run_dir / "videos" / f"{sequence['sequence_id']}.frames.jsonl"
    mapping = list(read_jsonl(mapping_path))
    output_root = run_dir / "curation-input" / "output"
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()
    if output_root.exists():
        for metadata_path in sorted(output_root.rglob("*.json")):
            try:
                payload = read_json(metadata_path)
            except ValueError:
                continue
            for record in _dict_records(payload):
                start = _time_value(record, ("start_time", "clip_start_time", "start"))
                end = _time_value(record, ("end_time", "clip_end_time", "end"))
                if start is None or end is None or end <= start:
                    continue
                clip_id = str(record.get("clip_id") or record.get("id") or metadata_path.stem)
                identity = (clip_id, start, end)
                if identity in seen:
                    continue
                seen.add(identity)
                frames = _frames_for_interval(mapping, start, end)
                rows.append(
                    {
                        "sample_id": clip_id,
                        "clip_id": clip_id,
                        "source_metadata": str(metadata_path.relative_to(run_dir)),
                        "start_time_s": start,
                        "end_time_s": end,
                        "frame_keys": [item.get("frame_key") for item in frames],
                        "camera_timestamps_ns": [item.get("camera_timestamp_ns") for item in frames],
                        "lidar_timestamps_ns": [item.get("lidar_timestamp_ns") for item in frames],
                        "lidar_scan_indices": [item.get("lidar_scan_index") for item in frames],
                    }
                )
    if not rows:
        rows.append(
            {
                "sample_id": f"{sequence['sequence_id']}-source-video",
                "clip_id": f"{sequence['sequence_id']}-source-video",
                "source_metadata": None,
                "start_time_s": 0.0,
                "end_time_s": sum(float(item.get("duration_s", 0.0)) for item in mapping),
                "frame_keys": [item.get("frame_key") for item in mapping],
                "camera_timestamps_ns": [item.get("camera_timestamp_ns") for item in mapping],
                "lidar_timestamps_ns": [item.get("lidar_timestamp_ns") for item in mapping],
                "lidar_scan_indices": [item.get("lidar_scan_index") for item in mapping],
            }
        )
    output = run_dir / "curation" / "frame_links.jsonl"
    write_jsonl(output, rows)
    return output


def run_curation(run_dir: Path, models_dir: Path, ffmpeg_dir: Path, config_file: Path | None = None) -> subprocess.CompletedProcess[str]:
    config_file = config_file or (Path(__file__).resolve().parents[2] / "cookbook" / "rellis-offroad" / "split-minimal.yaml")
    run_dir = run_dir.resolve()
    models_dir = models_dir.resolve()
    ffmpeg_dir = ffmpeg_dir.resolve()
    config_file = config_file.resolve()
    data_dir = run_dir / "curation-input"
    write_curation_provenance(run_dir, config_file)
    completed = run_make(
        "run-pipeline",
        {
            "CONFIG_FILE": str(config_file),
            "MODELS_DIR": str(models_dir),
            "DATA_DIR": str(data_dir),
            "FFMPEG_DIR": str(ffmpeg_dir),
        },
    )
    write_curation_links(run_dir)
    return completed


def run_retrieval(run_dir: Path, config_file: Path | None = None) -> subprocess.CompletedProcess[str]:
    config_file = config_file or (Path(__file__).resolve().parents[2] / "cookbook" / "nearest-neighbor-mining" / "tmm.yaml")
    run_dir = run_dir.resolve()
    config_file = config_file.resolve()
    return run_make(
        "run-data-mining-select",
        {"DATA_DIR": str(run_dir / "retrieval"), "TMM_CONFIG_FILE": str(config_file)},
    )
