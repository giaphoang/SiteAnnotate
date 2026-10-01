from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .caption import caption_run
from .io import read_json, read_jsonl
from .models import FrameRecord, SequenceManifest
from .paidf import run_curation, run_retrieval
from .publish import publish_run, validate_run
from .rellis import ingest_extracted
from .review import serve_sustech, stage_sustech, update_review_status
from .rosbag_adapter import ingest_rosbag
from .pointcloud import read_kitti_bin
from .semantic import import_ground_truth, run_salsanext
from .semantic_editor import serve_semantic_editor
from .video import build_timestamped_video


def _run(path: str) -> Path:
    return Path(path)


def _sequence(run_dir: Path) -> tuple[SequenceManifest, list[FrameRecord]]:
    manifest = SequenceManifest.from_dict(read_json(run_dir / "sequence.json"))
    frames = [FrameRecord.from_dict(item) for item in read_jsonl(run_dir / manifest.frames_path)]
    return manifest, frames


def _encode(run_dir: Path, overlay: bool = True) -> Path:
    manifest, frames = _sequence(run_dir)
    output = run_dir / "videos" / f"{manifest.sequence_id}.mp4"
    mapping = run_dir / "videos" / f"{manifest.sequence_id}.frames.jsonl"
    build_timestamped_video(manifest, frames, output, mapping, overlay_timestamps=overlay)
    return output


def _stage_ground_truth(run_dir: Path) -> int:
    manifest, frames = _sequence(run_dir)
    destination = run_dir / "predictions" / "ground-truth"
    destination.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        if not frame.semantic_path:
            raise ValueError(f"frame {frame.frame_key} has no RELLIS semantic label")
        source = Path(manifest.source["dataset_root"]) / frame.semantic_path
        point_count = len(read_kitti_bin(Path(manifest.source["dataset_root"]) / frame.lidar_path))
        import_ground_truth(source, destination / f"{frame.frame_key}.label", point_count)
    return len(frames)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="siteannotate")
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("ingest-extracted", help="normalize an extracted RELLIS sequence")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--sequence", required=True)
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--ontology", type=Path, default=Path("configs/rellis_ontology.json"))

    command = sub.add_parser("ingest-rosbag", help="extract a synchronized RELLIS ROS1 bag")
    command.add_argument("--bag", type=Path, required=True)
    command.add_argument("--sequence", required=True)
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--camera-topic", default="/pylon_camera_node/image_raw")
    command.add_argument("--lidar-topic", default="/os1_cloud_node/points")

    command = sub.add_parser("encode", help="create a timestamp-preserving camera video")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--no-overlay", action="store_true")

    command = sub.add_parser("curate", help="invoke PAIDF Curator")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--models-dir", type=Path, required=True)
    command.add_argument("--ffmpeg-dir", type=Path, required=True)
    command.add_argument("--config", type=Path)

    command = sub.add_parser("caption", help="caption the run video using Cosmos Reason")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--prompt", type=Path, default=Path("prompts/rellis_temporal_caption.yaml"))
    command.add_argument("--base-url")
    command.add_argument("--model")

    command = sub.add_parser("predict-semantics", help="run SalsaNext or stage RELLIS ground truth")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--weights-dir", type=Path)
    command.add_argument("--rellis-root", type=Path)

    command = sub.add_parser("stage-review", help="stage SUSTechPOINTS review files")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--output", type=Path)
    command.add_argument("--predictions-dir", type=Path)
    command.add_argument("--serve", action="store_true")
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8090)

    command = sub.add_parser("semantic-editor", help="serve the semantic label editor")
    command.add_argument("--review-root", type=Path, required=True)
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8090)

    command = sub.add_parser("box-review", help="launch the pinned SUSTechPOINTS box review UI")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--host", default="127.0.0.1")
    command.add_argument("--port", type=int, default=8081)

    command = sub.add_parser("update-review", help="record an approval decision")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--frame", required=True)
    command.add_argument("--caption", choices=["pending", "approved", "rejected"])
    command.add_argument("--boxes", choices=["pending", "approved", "rejected"])
    command.add_argument("--semantic", choices=["pending", "approved", "rejected"])
    command.add_argument("--reviewer")
    command.add_argument("--notes", default="")

    command = sub.add_parser("validate", help="run publication validation")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--allow-pending", action="store_true")

    command = sub.add_parser("publish", help="publish a validated run")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--version", default="v1")

    command = sub.add_parser("retrieve", help="invoke PAIDF Data Mining")
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--config", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "ingest-extracted":
            manifest = ingest_extracted(args.root, args.sequence, args.run, args.ontology)
            print(f"ingested {manifest.frame_count} frames into {args.run}")
        elif args.command == "ingest-rosbag":
            manifest = ingest_rosbag(args.bag, args.run, args.sequence, camera_topic=args.camera_topic, lidar_topic=args.lidar_topic)
            print(f"extracted {manifest.frame_count} synchronized frames into {args.run}")
        elif args.command == "encode":
            print(_encode(args.run, overlay=not args.no_overlay))
        elif args.command == "curate":
            manifest, _ = _sequence(args.run)
            video = args.run / "videos" / f"{manifest.sequence_id}.mp4"
            if not video.exists():
                _encode(args.run)
            input_dir = args.run / "curation-input" / "videos"
            input_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video, input_dir / video.name)
            result = run_curation(args.run, args.models_dir, args.ffmpeg_dir, args.config)
            print(result.stdout)
        elif args.command == "caption":
            record = caption_run(args.run, args.prompt, base_url=args.base_url, model=args.model)
            print(f"caption status: {record.parse_status}")
        elif args.command == "predict-semantics":
            if args.weights_dir:
                result = run_salsanext(args.run, args.weights_dir, args.rellis_root)
                print(result.stdout)
            else:
                print(f"staged { _stage_ground_truth(args.run) } RELLIS ground-truth label files")
        elif args.command == "stage-review":
            review_root = stage_sustech(args.run, args.output, args.predictions_dir)
            print(review_root)
            if args.serve:
                serve_semantic_editor(review_root, args.host, args.port)
        elif args.command == "semantic-editor":
            serve_semantic_editor(args.review_root, args.host, args.port)
        elif args.command == "box-review":
            pointer = args.run / "review" / "current.json"
            review_root = Path(read_json(pointer)["root"]) if pointer.exists() else args.run / "review" / "sustech"
            serve_sustech(review_root, args.host, args.port)
        elif args.command == "update-review":
            updates = {}
            if args.caption:
                updates["caption_status"] = args.caption
            if args.boxes:
                updates["box_status"] = args.boxes
            if args.semantic:
                updates["semantic_status"] = args.semantic
            if args.reviewer:
                updates["reviewer"] = args.reviewer
            if args.notes:
                updates["notes"] = args.notes
            print(update_review_status(args.run, args.frame, **updates))
        elif args.command == "validate":
            report = validate_run(args.run, require_review=not args.allow_pending)
            print(json_dump(report))
            if not report["valid"]:
                raise SystemExit(1)
        elif args.command == "publish":
            print(publish_run(args.run, args.version))
        elif args.command == "retrieve":
            result = run_retrieval(args.run, args.config)
            print(result.stdout)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"siteannotate: error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def json_dump(value: object) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True)
