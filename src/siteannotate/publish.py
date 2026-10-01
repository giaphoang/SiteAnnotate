from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import Any

from .io import read_json, read_jsonl, relative_path, sha256_file, write_json, write_jsonl
from .models import FrameRecord, PublicationManifest, SequenceManifest
from .pointcloud import read_kitti_bin, read_semantickitti_labels
from .rellis import load_ontology, resolve_frame_path
from .review import load_review_statuses
from .semantic import semantic_ids
from .timestamps import assert_monotonic


def _find_ontology(run_dir: Path, manifest: SequenceManifest) -> dict[str, Any]:
    if manifest.ontology:
        return manifest.ontology
    path = Path(manifest.ontology_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return load_ontology(path)


def _review_root(run_dir: Path) -> Path:
    pointer = run_dir / "review" / "current.json"
    if pointer.exists():
        return Path(read_json(pointer)["root"])
    return run_dir / "review" / "sustech"


def _review_scene(run_dir: Path, sequence: SequenceManifest) -> Path:
    return _review_root(run_dir) / "data" / sequence.sequence_id


def _semantic_source(run_dir: Path, sequence: SequenceManifest, frame: FrameRecord) -> Path | None:
    reviewed = _review_scene(run_dir, sequence) / "semantic" / f"{frame.frame_key}.label"
    if reviewed.exists():
        return reviewed
    if frame.semantic_path:
        source = resolve_frame_path(sequence, frame.semantic_path)
        if source.exists():
            return source
    return None


def _box_source(run_dir: Path, sequence: SequenceManifest, frame: FrameRecord) -> Path | None:
    source = _review_scene(run_dir, sequence) / "label" / f"{frame.frame_key}.json"
    return source if source.exists() else None


def validate_run(run_dir: Path, *, require_review: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    sequence = SequenceManifest.from_dict(read_json(run_dir / "sequence.json"))
    frames = [FrameRecord.from_dict(item) for item in read_jsonl(run_dir / sequence.frames_path)]
    ontology = _find_ontology(run_dir, sequence)
    class_ids = set(ontology["class_ids"])
    if not frames:
        errors.append("sequence contains no frames")
    if len({frame.frame_key for frame in frames}) != len(frames):
        errors.append("frame keys are not unique")
    camera_times = [frame.camera_timestamp_ns for frame in frames if frame.camera_timestamp_ns is not None]
    lidar_times = [frame.lidar_timestamp_ns for frame in frames if frame.lidar_timestamp_ns is not None]
    try:
        assert_monotonic(camera_times, allow_equal=True)
        assert_monotonic(lidar_times, allow_equal=True)
    except ValueError as exc:
        errors.append(str(exc))

    for frame in frames:
        camera_path = resolve_frame_path(sequence, frame.camera_path)
        lidar_path = resolve_frame_path(sequence, frame.lidar_path)
        if not camera_path.exists():
            errors.append(f"missing camera file: {camera_path}")
        if not lidar_path.exists():
            errors.append(f"missing LiDAR file: {lidar_path}")
            continue
        point_count = len(read_kitti_bin(lidar_path))
        semantic_path = _semantic_source(run_dir, sequence, frame)
        if semantic_path is None:
            errors.append(f"missing semantic label for {frame.frame_key}")
        else:
            labels = semantic_ids(read_semantickitti_labels(semantic_path))
            if len(labels) != point_count:
                errors.append(f"semantic point count mismatch for {frame.frame_key}")
            invalid = sorted(set(labels) - class_ids)
            if invalid:
                errors.append(f"invalid semantic IDs for {frame.frame_key}: {invalid}")
        if frame.sync_delta_ns is not None and abs(frame.sync_delta_ns) > sequence.timestamp_policy["max_sync_delta_ns"]:
            errors.append(f"sync delta exceeds policy for {frame.frame_key}")

    caption_files = list((run_dir / "captions").glob("*.json"))
    if not caption_files:
        if require_review:
            errors.append("no caption record found")
        else:
            warnings.append("no caption record found")
    else:
        caption = read_json(caption_files[0])
        if caption.get("parse_status") != "parsed":
            errors.append("caption response was not valid structured JSON")

    statuses = load_review_statuses(run_dir)
    if require_review:
        if len(statuses) != len(frames):
            errors.append("every frame must have a review status before publication")
        for frame in frames:
            status = statuses.get(frame.frame_key, {})
            for field in ("caption_status", "box_status", "semantic_status"):
                if status.get(field) != "approved":
                    errors.append(f"{frame.frame_key}: {field} is not approved")

    report = {
        "schema_version": 1,
        "run": str(run_dir),
        "sequence_id": sequence.sequence_id,
        "frame_count": len(frames),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors,
    }
    write_json(run_dir / "validation" / "report.json", report)
    return report


def _copy_file(source: Path, destination: Path, root: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return {
        "path": relative_path(destination, root),
        "sha256": sha256_file(destination),
        "bytes": destination.stat().st_size,
    }


def publish_run(run_dir: Path, version: str = "v1") -> Path:
    report = validate_run(run_dir, require_review=True)
    if not report["valid"]:
        raise ValueError("publication validation failed; inspect validation/report.json")
    sequence = SequenceManifest.from_dict(read_json(run_dir / "sequence.json"))
    frames = [FrameRecord.from_dict(item) for item in read_jsonl(run_dir / sequence.frames_path)]
    publication = run_dir / "publication" / version
    archive = run_dir / "publication" / f"siteannotate-{version}.tar"
    if publication.exists():
        raise FileExistsError(f"publication already exists: {publication}; choose a new version")
    if archive.exists():
        raise FileExistsError(f"publication archive already exists: {archive}; choose a new version")
    publication.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    for frame in frames:
        source_lidar = resolve_frame_path(sequence, frame.lidar_path)
        lidar_target = publication / "frames" / f"{frame.frame_key}.lidar.bin"
        files.append(_copy_file(source_lidar, lidar_target, publication))
        source_label = _semantic_source(run_dir, sequence, frame)
        if source_label is not None:
            label_target = publication / "frames" / f"{frame.frame_key}.semantic.label"
            files.append(_copy_file(source_label, label_target, publication))
        camera = resolve_frame_path(sequence, frame.camera_path)
        camera_target = publication / "frames" / f"{frame.frame_key}{camera.suffix.lower()}"
        files.append(_copy_file(camera, camera_target, publication))
        box_source = _box_source(run_dir, sequence, frame)
        if box_source is not None:
            box_target = publication / "frames" / f"{frame.frame_key}.boxes.json"
            files.append(_copy_file(box_source, box_target, publication))

    video_files = list((run_dir / "videos").glob("*.mp4"))
    if video_files:
        files.append(_copy_file(video_files[0], publication / "clips" / video_files[0].name, publication))
    caption_files = list((run_dir / "captions").glob("*.json"))
    if caption_files:
        files.append(_copy_file(caption_files[0], publication / "clips" / caption_files[0].name, publication))

    sequence_mapping = run_dir / "videos" / f"{sequence.sequence_id}.frames.jsonl"
    if sequence_mapping.exists():
        files.append(_copy_file(sequence_mapping, publication / "curation" / "source_frame_mapping.jsonl", publication))
    for relative in ("curation/provenance.json", "curation/frame_links.jsonl"):
        source = run_dir / relative
        if source.exists():
            files.append(_copy_file(source, publication / relative, publication))

    rows = []
    for frame in frames:
        semantic = _semantic_source(run_dir, sequence, frame)
        boxes = _box_source(run_dir, sequence, frame)
        rows.append(
            {
                "sample_id": frame.sample_id,
                "sequence_id": frame.sequence_id,
                "frame_key": frame.frame_key,
                "camera_timestamp_ns": frame.camera_timestamp_ns,
                "lidar_timestamp_ns": frame.lidar_timestamp_ns,
                "sync_delta_ns": frame.sync_delta_ns,
                "timestamp_source": frame.timestamp_source,
                "camera": f"frames/{frame.frame_key}{Path(frame.camera_path).suffix.lower()}",
                "lidar": f"frames/{frame.frame_key}.lidar.bin",
                "semantic": f"frames/{frame.frame_key}.semantic.label" if semantic else None,
                "boxes": f"frames/{frame.frame_key}.boxes.json" if boxes else None,
            }
        )
    write_jsonl(publication / "manifest.jsonl", rows)
    write_json(publication / "sequence.json", sequence.to_dict())
    write_json(publication / "validation.json", report)
    sources_lock = read_json(Path(__file__).resolve().parents[2] / "sources.lock.json")
    write_json(publication / "sources.lock.json", sources_lock)
    publication_manifest = PublicationManifest(
        schema_version=1,
        dataset_name="SiteAnnotate-RELLIS-3D",
        dataset_version=version,
        run_id=run_dir.name,
        source_manifest="sequence.json",
        upstream_sources=sources_lock.get("sources", {}),
        model_revisions={
            "caption": read_json(caption_files[0]).get("model", {}) if caption_files else {},
            "semantic": "RELLIS-3D reviewed labels or SalsaNext output",
            "boxes": "SUSTechPOINTS-compatible human review",
        },
        clips=[{"manifest": "manifest.jsonl", "count": len(rows)}],
        files=files,
        validation_report="validation.json",
    )
    write_json(publication / "publication.json", publication_manifest.to_dict())
    with tarfile.open(archive, "w") as handle:
        handle.add(publication, arcname=publication.name)
    return publication
