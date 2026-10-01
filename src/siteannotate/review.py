from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import read_json, read_jsonl, sha256_file, write_json
from .models import FrameRecord, SequenceManifest
from .pointcloud import convert_bin_to_pcd, read_kitti_bin, write_semantickitti_labels
from .rellis import load_ontology, resolve_frame_path


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def _calibration_payload(manifest: SequenceManifest) -> dict[str, Any]:
    calibration = manifest.calibration
    return {
        "intrinsic": calibration.get("intrinsic", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        "extrinsic": calibration.get(
            "extrinsic", [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ),
    }


def _prediction_for(prediction_dir: Path | None, frame_key: str) -> Path | None:
    if prediction_dir is None or not prediction_dir.exists():
        return None
    direct = prediction_dir / f"{frame_key}.label"
    if direct.exists():
        return direct
    return next(iter(sorted(prediction_dir.rglob(f"{frame_key}.label"))), None)


def stage_sustech(
    run_dir: Path,
    output_dir: Path | None = None,
    prediction_dir: Path | None = None,
) -> Path:
    sequence = SequenceManifest.from_dict(read_json(run_dir / "sequence.json"))
    frames = [FrameRecord.from_dict(item) for item in read_jsonl(run_dir / sequence.frames_path)]
    review_root = output_dir or (run_dir / "review" / "sustech")
    scene_root = review_root / "data" / sequence.sequence_id
    for folder in ("camera/front", "lidar", "label", "semantic", "calib/camera"):
        (scene_root / folder).mkdir(parents=True, exist_ok=True)

    index: list[dict[str, Any]] = []
    for frame in frames:
        camera_source = resolve_frame_path(sequence, frame.camera_path)
        lidar_source = resolve_frame_path(sequence, frame.lidar_path)
        label_source = resolve_frame_path(sequence, frame.semantic_path) if frame.semantic_path else None
        prediction_source = _prediction_for(prediction_dir, frame.frame_key)
        camera_destination = scene_root / "camera" / "front" / f"{frame.frame_key}{camera_source.suffix.lower()}"
        pcd_destination = scene_root / "lidar" / f"{frame.frame_key}.pcd"
        semantic_destination = scene_root / "semantic" / f"{frame.frame_key}.label"
        box_destination = scene_root / "label" / f"{frame.frame_key}.json"
        _link_or_copy(camera_source, camera_destination)
        if not pcd_destination.exists():
            convert_bin_to_pcd(lidar_source, pcd_destination)
        semantic_source = prediction_source or label_source
        if semantic_source and semantic_source.exists():
            # Semantic review edits must never write through into the downloaded
            # RELLIS source tree, so labels are always copied rather than linked.
            shutil.copy2(semantic_source, semantic_destination)
        else:
            write_semantickitti_labels(semantic_destination, [0] * len(read_kitti_bin(lidar_source)))
        if not box_destination.exists():
            write_json(box_destination, [])
        index.append(
            {
                "frame_key": frame.frame_key,
                "sample_id": frame.sample_id,
                "camera_timestamp_ns": frame.camera_timestamp_ns,
                "lidar_timestamp_ns": frame.lidar_timestamp_ns,
                "source_camera": frame.camera_path,
                "source_lidar": frame.lidar_path,
                "semantic_source": str(semantic_source) if semantic_source else None,
                "semantic_path": str(semantic_destination.relative_to(review_root)),
                "box_path": str(box_destination.relative_to(review_root)),
            }
        )
        update_review_status(
            run_dir,
            frame.frame_key,
            input_hash=sha256_file(camera_destination),
            annotation_hash=sha256_file(semantic_destination),
        )

    write_json(scene_root / "calib" / "camera" / "front.json", _calibration_payload(sequence))
    write_json(scene_root / "desc.json", {"sequence_id": sequence.sequence_id, "dataset": "RELLIS-3D"})
    write_json(
        review_root / "review_index.json",
        {
            "sequence_id": sequence.sequence_id,
            "ontology_path": sequence.ontology_path,
            "ontology": sequence.ontology or load_ontology(
                Path(sequence.ontology_path)
                if Path(sequence.ontology_path).is_absolute()
                else Path(__file__).resolve().parents[2] / sequence.ontology_path
            ),
            "frames": index,
        },
    )
    write_json(run_dir / "review" / "current.json", {"root": str(review_root.resolve())})
    return review_root


def update_review_status(run_dir: Path, frame_key: str, **updates: Any) -> Path:
    review_dir = run_dir / "review" / "status"
    review_dir.mkdir(parents=True, exist_ok=True)
    path = review_dir / f"{frame_key}.json"
    value = read_json(path) if path.exists() else {
        "frame_key": frame_key,
        "caption_status": "pending",
        "box_status": "pending",
        "semantic_status": "pending",
    }
    value.update(updates)
    if {"caption_status", "box_status", "semantic_status", "reviewer", "notes"}.intersection(updates):
        value["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(path, value)
    return path


def load_review_statuses(run_dir: Path) -> dict[str, dict[str, Any]]:
    status_dir = run_dir / "review" / "status"
    if not status_dir.exists():
        return {}
    return {path.stem: read_json(path) for path in sorted(status_dir.glob("*.json"))}


def serve_sustech(review_root: Path, host: str = "127.0.0.1", port: int = 8081) -> None:
    """Launch the pinned SUSTechPOINTS UI against a staged review directory.

    SUSTechPOINTS resolves its data and templates relative to its process
    directory. A temporary runtime overlay supplies those paths without
    modifying the vendor checkout or the source dataset.
    """
    vendor = Path(__file__).resolve().parents[2] / "vendor" / "SUSTechPOINTS"
    main = vendor / "main.py"
    server_config = vendor / "server.conf"
    data_root = review_root / "data"
    if not main.exists() or not server_config.exists():
        raise RuntimeError("SUSTechPOINTS source is not initialized; run make bootstrap-sources")
    if not data_root.exists():
        raise FileNotFoundError(f"staged SUSTechPOINTS data directory does not exist: {data_root}")

    runtime = Path(tempfile.mkdtemp(prefix="siteannotate-sustech-", dir=str(review_root)))
    try:
        for source in vendor.iterdir():
            if source.name in {".git", "data", "server.conf", "temp", "__pycache__"}:
                continue
            destination = runtime / source.name
            destination.symlink_to(source, target_is_directory=source.is_dir())
        (runtime / "data").symlink_to(data_root.resolve(), target_is_directory=True)
        (runtime / "temp").mkdir()
        config = server_config.read_text(encoding="utf-8")
        config = re.sub(r"server\.socket_host\s*=.*", f'server.socket_host = "{host}"', config)
        config = re.sub(r"server\.socket_port\s*=.*", f"server.socket_port = {port}", config)
        (runtime / "server.conf").write_text(config, encoding="utf-8")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(runtime), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        subprocess.run([sys.executable, str(runtime / "main.py")], cwd=runtime, env=environment, check=True)
    finally:
        shutil.rmtree(runtime, ignore_errors=True)
