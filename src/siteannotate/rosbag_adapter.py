from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

from .io import relative_path, write_json, write_jsonl
from .models import FrameRecord, SequenceManifest
from .timestamps import TimedItem, nearest_synchronize


def _stamp_ns(message: Any, fallback: Any) -> int:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        stamp = fallback
    if hasattr(stamp, "to_nsec"):
        return int(stamp.to_nsec())
    if hasattr(stamp, "secs"):
        return int(stamp.secs) * 1_000_000_000 + int(getattr(stamp, "nsecs", 0))
    return int(float(stamp) * 1_000_000_000)


def ingest_rosbag(
    bag_path: Path,
    run_dir: Path,
    sequence_id: str,
    *,
    camera_topic: str = "/pylon_camera_node/image_raw",
    lidar_topic: str = "/os1_cloud_node/points",
    max_delta_ns: int = 50_000_000,
) -> SequenceManifest:
    """Extract the synchronized RELLIS ROS1 camera/LiDAR pair.

    ROS1 dependencies are intentionally optional because the rest of the
    integration and its tests run without a ROS installation.
    """
    try:
        import cv2
        import rosbag
        from cv_bridge import CvBridge
        from sensor_msgs import point_cloud2
    except ImportError as exc:  # pragma: no cover - ROS-only path
        raise RuntimeError("ROS bag ingestion requires rosbag, cv_bridge, sensor_msgs, and OpenCV") from exc

    extraction_root = run_dir / "rosbag-data"
    sequence_root = extraction_root / sequence_id
    camera_root = sequence_root / "pylon_camera_node"
    lidar_root = sequence_root / "os1_cloud_node_kitti_bin"
    camera_root.mkdir(parents=True, exist_ok=True)
    lidar_root.mkdir(parents=True, exist_ok=True)
    camera_items: list[TimedItem] = []
    lidar_items: list[TimedItem] = []
    bridge = CvBridge()
    camera_info: dict[str, Any] | None = None

    with rosbag.Bag(str(bag_path), "r") as bag:
        for topic, message, bag_time in bag.read_messages(topics=[camera_topic, lidar_topic, "/pylon_camera_node/camera_info"]):
            timestamp_ns = _stamp_ns(message, bag_time)
            if topic == camera_topic:
                frame_index = len(camera_items)
                image = bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                image_path = camera_root / f"{frame_index:06d}.png"
                if not cv2.imwrite(str(image_path), image):
                    raise RuntimeError(f"failed to write extracted image: {image_path}")
                camera_items.append(TimedItem(frame_index, timestamp_ns, image_path))
            elif topic == lidar_topic:
                scan_index = len(lidar_items)
                points = []
                for point in point_cloud2.read_points(message, field_names=None, skip_nans=True):
                    x, y, z = float(point[0]), float(point[1]), float(point[2])
                    intensity = float(point[3]) if len(point) > 3 else 0.0
                    points.append(struct.pack("<ffff", x, y, z, intensity))
                lidar_path = lidar_root / f"{scan_index:06d}.bin"
                lidar_path.write_bytes(b"".join(points))
                lidar_items.append(TimedItem(scan_index, timestamp_ns, lidar_path))
            else:
                camera_info = {"K": list(getattr(message, "K", [])), "D": list(getattr(message, "D", []))}

    pairs = nearest_synchronize(camera_items, lidar_items, max_delta_ns=max_delta_ns)
    if not pairs:
        raise ValueError("ROS bag has no camera/LiDAR pairs within the synchronization tolerance")
    frames: list[FrameRecord] = []
    for frame_index, pair in enumerate(pairs):
        frames.append(
            FrameRecord(
                sample_id=f"rellis-{sequence_id}-{frame_index:06d}",
                sequence_id=sequence_id,
                frame_key=f"{frame_index:06d}",
                camera_path=relative_path(pair.camera.payload, extraction_root),
                lidar_path=relative_path(pair.lidar.payload, extraction_root),
                camera_timestamp_ns=pair.camera.timestamp_ns,
                lidar_timestamp_ns=pair.lidar.timestamp_ns,
                camera_frame_index=pair.camera.index,
                lidar_scan_index=pair.lidar.index,
                sync_delta_ns=pair.delta_ns,
                timestamp_source="header",
                calibration_id=f"rosbag-{sequence_id}",
            )
        )
    write_jsonl(run_dir / "frames.jsonl", (frame.to_dict() for frame in frames))
    calibration = {
        "status": "camera-info-only",
        "intrinsic": camera_info.get("K", []) if camera_info else [],
        "extrinsic": [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    }
    manifest = SequenceManifest(
        schema_version=1,
        dataset="RELLIS-3D",
        sequence_id=sequence_id,
        source={"input_type": "rosbag", "bag_path": str(bag_path.resolve()), "dataset_root": str(extraction_root.resolve())},
        frame_count=len(frames),
        frames_path="frames.jsonl",
        calibration=calibration,
        ontology_path="configs/rellis_ontology.json",
        timestamp_policy={"authoritative": ["header", "record"], "max_sync_delta_ns": max_delta_ns},
    )
    write_json(run_dir / "sequence.json", manifest.to_dict())
    return manifest

