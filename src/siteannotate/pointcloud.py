from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable, Iterator

Point = tuple[float, float, float, float]


def read_kitti_bin(path: Path) -> list[Point]:
    data = path.read_bytes()
    if len(data) % 16:
        raise ValueError(f"invalid KITTI point cloud byte length: {path}")
    return list(struct.iter_unpack("<ffff", data))


def read_semantickitti_labels(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"invalid SemanticKITTI label byte length: {path}")
    return [value[0] for value in struct.iter_unpack("<I", data)]


def write_semantickitti_labels(path: Path, labels: Iterable[int]) -> None:
    values = list(labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(struct.pack("<I", int(value)) for value in values))


def write_ascii_pcd(path: Path, points: Iterable[Point]) -> int:
    values = list(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("# .PCD v0.7 - Point Cloud Data file format\n")
        handle.write("VERSION 0.7\n")
        handle.write("FIELDS x y z intensity\n")
        handle.write("SIZE 4 4 4 4\n")
        handle.write("TYPE F F F F\n")
        handle.write("COUNT 1 1 1 1\n")
        handle.write(f"WIDTH {len(values)}\n")
        handle.write("HEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n")
        handle.write(f"POINTS {len(values)}\nDATA ascii\n")
        for x, y, z, intensity in values:
            handle.write(f"{x:.7g} {y:.7g} {z:.7g} {intensity:.7g}\n")
    return len(values)


def convert_bin_to_pcd(source: Path, destination: Path) -> int:
    return write_ascii_pcd(destination, read_kitti_bin(source))


def read_ascii_pcd(path: Path, max_points: int | None = None) -> list[Point]:
    lines = path.read_text(encoding="ascii", errors="replace").splitlines()
    data_index = next((index for index, line in enumerate(lines) if line.strip().lower() == "data ascii"), None)
    if data_index is None:
        raise ValueError(f"only ASCII PCD is supported by the lightweight editor: {path}")
    points: list[Point] = []
    for line in lines[data_index + 1 :]:
        fields = line.split()
        if len(fields) < 3:
            continue
        intensity = float(fields[3]) if len(fields) > 3 else 0.0
        points.append((float(fields[0]), float(fields[1]), float(fields[2]), intensity))
        if max_points is not None and len(points) >= max_points:
            break
    return points


def downsample_indices(count: int, max_points: int) -> list[int]:
    if count <= max_points:
        return list(range(count))
    step = max(1, count // max_points)
    return list(range(0, count, step))[:max_points]

