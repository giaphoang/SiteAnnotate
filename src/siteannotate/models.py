from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class FrameRecord:
    sample_id: str
    sequence_id: str
    frame_key: str
    camera_path: str
    lidar_path: str
    semantic_path: Optional[str] = None
    camera_timestamp_ns: Optional[int] = None
    lidar_timestamp_ns: Optional[int] = None
    camera_frame_index: Optional[int] = None
    lidar_scan_index: Optional[int] = None
    sync_delta_ns: Optional[int] = None
    timestamp_source: str = "unknown"
    calibration_id: str = "default"
    pose_path: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FrameRecord":
        return cls(**value)


@dataclass
class SequenceManifest:
    schema_version: int
    dataset: str
    sequence_id: str
    source: dict[str, Any]
    frame_count: int
    frames_path: str
    calibration: dict[str, Any]
    ontology_path: str
    timestamp_policy: dict[str, Any]
    status: str = "ingested"
    ontology: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SequenceManifest":
        return cls(**value)


@dataclass
class ClipRecord:
    clip_id: str
    sequence_id: str
    source_video: str
    source_frame_start: int
    source_frame_end: int
    frame_keys: list[str]
    timestamps_ns: list[int]
    curator_metadata_path: Optional[str] = None
    caption_path: Optional[str] = None
    embedding_path: Optional[str] = None
    review_path: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CaptionRecord:
    schema_version: int
    clip_id: str
    raw_text: str
    scene: dict[str, Any]
    events: list[dict[str, Any]]
    model: dict[str, Any]
    parse_status: str
    frame_mapping: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReviewRecord:
    frame_key: str
    caption_status: str = "pending"
    box_status: str = "pending"
    semantic_status: str = "pending"
    reviewer: Optional[str] = None
    reviewed_at: Optional[str] = None
    notes: str = ""
    input_hash: Optional[str] = None
    annotation_hash: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PublicationManifest:
    schema_version: int
    dataset_name: str
    dataset_version: str
    run_id: str
    source_manifest: str
    upstream_sources: dict[str, Any]
    model_revisions: dict[str, Any]
    clips: list[dict[str, Any]]
    files: list[dict[str, Any]]
    validation_report: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
