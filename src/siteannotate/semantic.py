from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from .io import read_json, sha256_file
from .pointcloud import read_semantickitti_labels, write_semantickitti_labels


def semantic_ids(labels: list[int]) -> list[int]:
    return [value & 0xFFFF for value in labels]


def validate_label_file(label_path: Path, point_count: int, class_ids: set[int]) -> dict[str, Any]:
    labels = semantic_ids(read_semantickitti_labels(label_path))
    invalid = sorted({value for value in labels if value not in class_ids})
    return {
        "path": str(label_path),
        "point_count": point_count,
        "label_count": len(labels),
        "count_matches": len(labels) == point_count,
        "invalid_class_ids": invalid,
        "valid": len(labels) == point_count and not invalid,
    }


def import_ground_truth(source_label: Path, destination: Path, point_count: int) -> dict[str, Any]:
    labels = semantic_ids(read_semantickitti_labels(source_label))
    if len(labels) != point_count:
        raise ValueError(f"semantic label/point count mismatch for {source_label}: {len(labels)} != {point_count}")
    write_semantickitti_labels(destination, labels)
    return {"source": str(source_label), "destination": str(destination), "sha256": sha256_file(destination)}


def run_salsanext(run_dir: Path, weights_dir: Path, rellis_root: Path | None = None) -> subprocess.CompletedProcess[str]:
    repo = Path(__file__).resolve().parents[2] / "vendor" / "RELLIS-3D" / "benchmarks" / "SalsaNext"
    infer = repo / "train" / "tasks" / "semantic" / "infer2.py"
    if not infer.exists():
        raise RuntimeError("RELLIS SalsaNext source is not initialized; run make bootstrap-sources")
    dataset = rellis_root or Path(json_source_root(run_dir))
    command = [
        "python3",
        str(infer),
        "-d",
        str(dataset),
        "-l",
        str(run_dir / "predictions" / "salsanext"),
        "-s",
        "test",
        "-m",
        str(weights_dir),
    ]
    return subprocess.run(command, text=True, capture_output=True, check=True)


def json_source_root(run_dir: Path) -> str:
    sequence = read_json(run_dir / "sequence.json")
    return str(sequence["source"]["dataset_root"])
