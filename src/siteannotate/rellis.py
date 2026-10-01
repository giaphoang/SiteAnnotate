from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .io import read_jsonl, relative_path, write_json, write_jsonl
from .models import FrameRecord, SequenceManifest
from .timestamps import assert_monotonic, timestamp_quality

_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class SourceFile:
    path: Path
    index: int
    stem: str


def _numeric_index(path: Path, fallback: int) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else fallback


def sorted_source_files(directory: Path, suffixes: Iterable[str]) -> list[SourceFile]:
    suffix_set = {suffix.lower() for suffix in suffixes}
    paths = [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in suffix_set]
    paths.sort(key=lambda path: (_numeric_index(path, 0), path.name))
    return [SourceFile(path=path, index=_numeric_index(path, position), stem=path.stem) for position, path in enumerate(paths)]


def _timestamp_to_ns(token: str) -> int:
    value = float(token)
    magnitude = abs(value)
    if magnitude < 1e11:
        return int(round(value * 1_000_000_000))
    if magnitude < 1e14:
        return int(round(value * 1_000_000))
    if magnitude < 1e17:
        return int(round(value * 1_000))
    return int(round(value))


def read_pose_timestamps(path: Path) -> list[int]:
    """Read a scalar timestamp file without treating pose matrices as times.

    RELLIS ``poses.txt`` contains one pose per scan.  Its rows commonly have
    12 or 16 values, so this helper deliberately accepts only scalar rows (or
    ``index,timestamp`` rows) and returns an empty list for pose matrices.
    """
    if not path.exists():
        return []
    timestamps: list[int] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values = _NUMBER.findall(line)
        if len(values) == 1:
            timestamps.append(_timestamp_to_ns(values[0]))
        elif len(values) == 2:
            timestamps.append(_timestamp_to_ns(values[-1]))
    return timestamps


def _timestamp_file(sequence_dir: Path, names: Iterable[str]) -> tuple[list[int], str | None]:
    for name in names:
        path = sequence_dir / name
        values = read_pose_timestamps(path)
        if values:
            return values, path.name
    return [], None


def read_sensor_timestamps(sequence_dir: Path, sensor: str) -> tuple[list[int], str | None]:
    """Find optional recorded timestamps shipped beside an extracted sequence."""
    if sensor == "camera":
        names = (
            "camera_timestamps.txt",
            "image_timestamps.txt",
            "pylon_camera_node_timestamps.txt",
            "camera_timestamps.csv",
        )
    elif sensor == "lidar":
        names = (
            "lidar_timestamps.txt",
            "scan_timestamps.txt",
            "os1_cloud_node_timestamps.txt",
            "lidar_timestamps.csv",
        )
    else:
        raise ValueError(f"unsupported timestamp sensor: {sensor}")
    return _timestamp_file(sequence_dir, names)


def _matrix_from_values(values: list[float], size: int) -> list[float]:
    values = values[:]
    if size == 3 and len(values) >= 12:
        projection = values[-12:]
        return [
            projection[0], projection[1], projection[2],
            projection[4], projection[5], projection[6],
            projection[8], projection[9], projection[10],
        ]
    if len(values) >= size * size:
        return values[-size * size :]
    if size == 4 and len(values) >= 12:
        transform = values[-12:]
        return [
            transform[0], transform[1], transform[2], transform[3],
            transform[4], transform[5], transform[6], transform[7],
            transform[8], transform[9], transform[10], transform[11],
            0.0, 0.0, 0.0, 1.0,
        ]
    if size == 3 and len(values) >= 9:
        return values[-9:]
    return []


def parse_calibration(path: Path) -> dict[str, Any]:
    """Parse common KITTI/RELLIS calibration text without assuming one layout."""
    result: dict[str, Any] = {"status": "missing", "raw_file": path.as_posix()}
    if not path.exists():
        result["status"] = "missing"
        result["intrinsic"] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        result["extrinsic"] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        return result

    keyed: dict[str, list[float]] = {}
    all_values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        values = [float(value) for value in _NUMBER.findall(line)]
        if not values:
            continue
        all_values.extend(values)
        key = line.split(":", 1)[0].strip().lower() if ":" in line else ""
        if key:
            keyed[key] = values

    intrinsic: list[float] = []
    extrinsic: list[float] = []
    for key, values in keyed.items():
        if not intrinsic and (key.startswith("p") or "intr" in key or "camera" in key):
            intrinsic = _matrix_from_values(values, 3)
        if not extrinsic and ("tr" in key or "extr" in key or "velo" in key or "lidar" in key):
            extrinsic = _matrix_from_values(values, 4)
    if not intrinsic:
        intrinsic = _matrix_from_values(all_values, 3) if len(all_values) in {9, 12} else []
    if not extrinsic:
        extrinsic = _matrix_from_values(all_values, 4) if len(all_values) in {12, 16} else []
    if len(intrinsic) != 9:
        intrinsic = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    if len(extrinsic) != 16:
        extrinsic = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    result.update({"status": "parsed", "intrinsic": intrinsic, "extrinsic": extrinsic, "keys": sorted(keyed)})
    return result


def load_ontology(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _matching_lidar(camera: SourceFile, lidars: list[SourceFile]) -> SourceFile:
    exact = next((item for item in lidars if item.stem == camera.stem), None)
    if exact is not None:
        return exact
    return min(lidars, key=lambda item: (abs(item.index - camera.index), item.index))


def _matching_label(lidar: SourceFile, labels: list[SourceFile]) -> Optional[SourceFile]:
    return next((item for item in labels if item.stem == lidar.stem), None)


def ingest_extracted(
    dataset_root: Path,
    sequence_id: str,
    run_dir: Path,
    ontology_path: Path = Path("configs/rellis_ontology.json"),
) -> SequenceManifest:
    sequence_dir = dataset_root / sequence_id
    if not sequence_dir.exists():
        raise FileNotFoundError(f"RELLIS sequence directory does not exist: {sequence_dir}")
    camera_dir = sequence_dir / "pylon_camera_node"
    lidar_dir = sequence_dir / "os1_cloud_node_kitti_bin"
    label_dir = sequence_dir / "os1_cloud_node_semantickitti_label_id"
    if not camera_dir.exists() or not lidar_dir.exists():
        raise FileNotFoundError(
            f"expected RELLIS camera/LiDAR directories below {sequence_dir}; "
            "required pylon_camera_node and os1_cloud_node_kitti_bin"
        )

    cameras = sorted_source_files(camera_dir, {".png", ".jpg", ".jpeg"})
    lidars = sorted_source_files(lidar_dir, {".bin"})
    labels = sorted_source_files(label_dir, {".label"}) if label_dir.exists() else []
    if not cameras or not lidars:
        raise ValueError("RELLIS sequence contains no camera images or LiDAR scans")
    camera_timestamps, camera_timestamp_file = read_sensor_timestamps(sequence_dir, "camera")
    lidar_timestamps, lidar_timestamp_file = read_sensor_timestamps(sequence_dir, "lidar")
    shared_timestamps, shared_timestamp_file = _timestamp_file(
        sequence_dir, ("timestamps.txt", "timestamps.csv", "times.txt")
    )
    if not camera_timestamps:
        camera_timestamps = shared_timestamps
        camera_timestamp_file = shared_timestamp_file
    if not lidar_timestamps:
        lidar_timestamps = shared_timestamps
        lidar_timestamp_file = shared_timestamp_file
    calibration = parse_calibration(sequence_dir / "calib.txt")
    calibration["sequence_id"] = sequence_id
    ontology_file = ontology_path if ontology_path.is_absolute() else Path(__file__).resolve().parents[2] / ontology_path
    ontology = load_ontology(ontology_file)

    run_dir.mkdir(parents=True, exist_ok=True)
    frames: list[FrameRecord] = []
    for position, camera in enumerate(cameras):
        lidar = _matching_lidar(camera, lidars)
        label = _matching_label(lidar, labels)
        camera_timestamp = camera_timestamps[position] if position < len(camera_timestamps) else None
        lidar_position = lidars.index(lidar)
        lidar_timestamp = lidar_timestamps[lidar_position] if lidar_position < len(lidar_timestamps) else None
        if camera_timestamp is not None and lidar_timestamp is not None:
            delta = lidar_timestamp - camera_timestamp
            source = "record"
        elif camera_timestamp is not None or lidar_timestamp is not None:
            delta = None
            source = "record"
        else:
            delta = None
            source = "index-only"
        timestamp_quality(source)
        frame_key = f"{position:06d}"
        frames.append(
            FrameRecord(
                sample_id=f"rellis-{sequence_id}-{frame_key}",
                sequence_id=sequence_id,
                frame_key=frame_key,
                camera_path=relative_path(camera.path, dataset_root),
                lidar_path=relative_path(lidar.path, dataset_root),
                semantic_path=relative_path(label.path, dataset_root) if label else None,
                camera_timestamp_ns=camera_timestamp,
                lidar_timestamp_ns=lidar_timestamp,
                camera_frame_index=position,
                lidar_scan_index=lidar_position,
                sync_delta_ns=delta,
                timestamp_source=source,
                calibration_id=f"rellis-{sequence_id}",
                metadata={"camera_stem": camera.stem, "lidar_stem": lidar.stem},
            )
        )

    for timestamps in (camera_timestamps, lidar_timestamps):
        if timestamps:
            assert_monotonic(timestamps, allow_equal=True)
    frames_path = run_dir / "frames.jsonl"
    write_jsonl(frames_path, (frame.to_dict() for frame in frames))
    manifest = SequenceManifest(
        schema_version=1,
        dataset="RELLIS-3D",
        sequence_id=sequence_id,
        source={
            "input_type": "extracted",
            "dataset_root": str(dataset_root.resolve()),
            "sequence_dir": str(sequence_dir.resolve()),
            "camera_sensor": "pylon_camera_node",
            "lidar_sensor": "os1_cloud_node",
            "timestamp_files": {
                "camera": camera_timestamp_file,
                "lidar": lidar_timestamp_file,
            },
        },
        frame_count=len(frames),
        frames_path="frames.jsonl",
        calibration=calibration,
        ontology_path=str(ontology_path),
        timestamp_policy={
            "authoritative": ["header", "record"],
            "derived": ["index-only"],
            "max_sync_delta_ns": 50_000_000,
        },
        ontology=ontology,
    )
    write_json(run_dir / "sequence.json", manifest.to_dict())
    return manifest


def resolve_frame_path(manifest: SequenceManifest, frame_path: str) -> Path:
    root = Path(manifest.source["dataset_root"])
    return root / frame_path


def load_run_manifest(run_dir: Path) -> tuple[SequenceManifest, list[FrameRecord]]:
    import json

    manifest = SequenceManifest.from_dict(json.loads((run_dir / "sequence.json").read_text(encoding="utf-8")))
    frames = [FrameRecord.from_dict(item) for item in read_jsonl(run_dir / manifest.frames_path)]
    return manifest, frames
