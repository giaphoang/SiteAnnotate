from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from siteannotate.caption import _extract_json, _map_event_frames, load_prompt
from siteannotate.io import read_json, read_jsonl, write_json, write_jsonl
from siteannotate.pointcloud import read_semantickitti_labels
from siteannotate.paidf import write_curation_links
from siteannotate.publish import publish_run, validate_run
from siteannotate.rellis import ingest_extracted, parse_calibration, read_pose_timestamps
from siteannotate.review import stage_sustech, update_review_status
from siteannotate.semantic_editor import SemanticEditor
from siteannotate.timestamps import TimedItem, nearest_synchronize


ROOT = Path(__file__).resolve().parents[1]


class SiteAnnotateIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="siteannotate-test-")
        self.root = Path(self.temporary.name)
        self.dataset_root = self.root / "RELLIS-3D"
        self.sequence_id = "00000"
        self.sequence_root = self.dataset_root / self.sequence_id
        for folder in (
            "pylon_camera_node",
            "os1_cloud_node_kitti_bin",
            "os1_cloud_node_semantickitti_label_id",
        ):
            (self.sequence_root / folder).mkdir(parents=True)

        points = [
            (1.0, 2.0, 3.0, 0.1),
            (4.0, 5.0, 6.0, 0.2),
            (7.0, 8.0, 9.0, 0.3),
        ]
        for index in range(2):
            key = f"{index:06d}"
            (self.sequence_root / "pylon_camera_node" / f"{key}.png").write_bytes(
                f"camera-{index}".encode()
            )
            (self.sequence_root / "os1_cloud_node_kitti_bin" / f"{key}.bin").write_bytes(
                b"".join(struct.pack("<ffff", *point) for point in points)
            )
            (self.sequence_root / "os1_cloud_node_semantickitti_label_id" / f"{key}.label").write_bytes(
                b"".join(struct.pack("<I", label) for label in (1, 23, 34))
            )

        # This is deliberately pose-shaped, not timestamp-shaped. Ingestion
        # must not interpret its first coordinate as a sensor timestamp.
        (self.sequence_root / "poses.txt").write_text(
            "1 0 0 0 0 1 0 0 0 0 1 0\n" * 2,
            encoding="utf-8",
        )
        (self.sequence_root / "camera_timestamps.txt").write_text("1.000\n1.100\n", encoding="utf-8")
        (self.sequence_root / "lidar_timestamps.txt").write_text("1.010\n1.110\n", encoding="utf-8")
        (self.sequence_root / "calib.txt").write_text(
            "P0: 100 0 50 0 0 100 40 0 0 0 1 0\n"
            "Tr: 1 0 0 1 0 1 0 2 0 0 1 3\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def ingest(self) -> Path:
        run_dir = self.root / "run"
        ingest_extracted(
            self.dataset_root,
            self.sequence_id,
            run_dir,
            ROOT / "configs" / "rellis_ontology.json",
        )
        return run_dir

    def test_ingest_preserves_recorded_timestamps_and_calibration(self) -> None:
        run_dir = self.ingest()
        sequence = read_json(run_dir / "sequence.json")
        frames = list(read_jsonl(run_dir / "frames.jsonl"))

        self.assertEqual(sequence["frame_count"], 2)
        self.assertIn(34, sequence["ontology"]["class_ids"])
        self.assertEqual(frames[0]["camera_timestamp_ns"], 1_000_000_000)
        self.assertEqual(frames[0]["lidar_timestamp_ns"], 1_010_000_000)
        self.assertEqual(frames[0]["sync_delta_ns"], 10_000_000)
        self.assertEqual(frames[0]["timestamp_source"], "record")
        self.assertEqual(sequence["calibration"]["intrinsic"], [100.0, 0.0, 50.0, 0.0, 100.0, 40.0, 0.0, 0.0, 1.0])
        self.assertEqual(sequence["calibration"]["extrinsic"][3], 1.0)
        self.assertEqual(read_pose_timestamps(self.sequence_root / "poses.txt"), [])

    def test_stage_edit_validate_and_publish(self) -> None:
        run_dir = self.ingest()
        review_root = stage_sustech(run_dir)
        scene = review_root / "data" / self.sequence_id
        self.assertTrue((scene / "lidar" / "000000.pcd").exists())
        self.assertFalse((scene / "semantic" / "000000.label").is_symlink())

        editor = SemanticEditor(review_root)
        payload = editor._frame_payload("000000")
        result = editor._save_updates(
            "000000",
            {
                "expected_sha256": payload["sha256"],
                "updates": [{"index": 1, "label": 19}],
            },
        )
        self.assertEqual(result["updated"], 1)
        self.assertEqual(read_semantickitti_labels(scene / "semantic" / "000000.label"), [1, 19, 34])
        self.assertEqual(
            read_semantickitti_labels(self.sequence_root / "os1_cloud_node_semantickitti_label_id" / "000000.label"),
            [1, 23, 34],
        )

        write_jsonl(
            run_dir / "videos" / "00000.frames.jsonl",
            [
                {
                    "frame_key": "000000",
                    "camera_timestamp_ns": 1_000_000_000,
                    "lidar_timestamp_ns": 1_010_000_000,
                    "lidar_scan_index": 0,
                    "duration_s": 0.15,
                },
                {
                    "frame_key": "000001",
                    "camera_timestamp_ns": 1_100_000_000,
                    "lidar_timestamp_ns": 1_110_000_000,
                    "lidar_scan_index": 1,
                    "duration_s": 0.15,
                },
            ],
        )
        write_json(
            run_dir / "curation-input" / "output" / "split" / "clip-001.json",
            {"clip_id": "clip-001", "start_time": 0.1, "end_time": 0.25},
        )
        links = list(read_jsonl(write_curation_links(run_dir)))
        self.assertEqual(links[0]["frame_keys"], ["000000", "000001"])
        self.assertEqual(links[0]["lidar_scan_indices"], [0, 1])

        write_json(
            run_dir / "captions" / "00000.json",
            {"parse_status": "parsed", "model": {"id": "test-cosmos-reason"}},
        )
        for frame in ("000000", "000001"):
            update_review_status(
                run_dir,
                frame,
                caption_status="approved",
                box_status="approved",
                semantic_status="approved",
                reviewer="offline-test",
            )

        report = validate_run(run_dir)
        self.assertTrue(report["valid"], report)
        publication = publish_run(run_dir, "test-v1")
        self.assertTrue((publication / "publication.json").exists())
        self.assertTrue((run_dir / "publication" / "siteannotate-test-v1.tar").exists())
        self.assertTrue((publication / "curation" / "frame_links.jsonl").exists())
        manifest_rows = list(read_jsonl(publication / "manifest.jsonl"))
        self.assertEqual(manifest_rows[0]["lidar"], "frames/000000.lidar.bin")
        published_labels = publication / "frames" / "000000.semantic.label"
        self.assertEqual(read_semantickitti_labels(published_labels), [1, 19, 34])
        publication_manifest = read_json(publication / "publication.json")
        self.assertEqual(publication_manifest["files"][0]["path"], "frames/000000.lidar.bin")

    def test_timestamp_pairing_and_caption_mapping(self) -> None:
        cameras = [TimedItem(0, 100, "camera-0"), TimedItem(1, 210, "camera-1")]
        lidars = [TimedItem(0, 95, "lidar-0"), TimedItem(1, 205, "lidar-1")]
        pairs = nearest_synchronize(cameras, lidars, max_delta_ns=10)
        self.assertEqual([(pair.camera.index, pair.lidar.index) for pair in pairs], [(0, 0), (1, 1)])

        parsed, status = _extract_json("```json\n{\"scene\": {}, \"events\": [{\"start\": 0.1, \"end\": 0.2}]}\n```")
        self.assertEqual(status, "parsed")
        mapping = [
            {"frame_key": "000000", "camera_timestamp_ns": 100, "duration_s": 0.15},
            {"frame_key": "000001", "camera_timestamp_ns": 200, "duration_s": 0.15},
        ]
        events = _map_event_frames(parsed["events"], mapping)
        self.assertEqual(events[0]["frame_start"], "000000")
        self.assertEqual(events[0]["frame_end"], "000001")
        prompt = load_prompt(ROOT / "prompts" / "rellis_temporal_caption.yaml")
        self.assertIn("You analyze off-road", prompt)
        self.assertIn("Describe the notable terrain", prompt)

    def test_calibration_defaults_when_file_is_missing(self) -> None:
        parsed = parse_calibration(self.root / "does-not-exist.txt")
        self.assertEqual(parsed["status"], "missing")
        self.assertEqual(len(parsed["intrinsic"]), 9)
        self.assertEqual(len(parsed["extrinsic"]), 16)


if __name__ == "__main__":
    unittest.main()
