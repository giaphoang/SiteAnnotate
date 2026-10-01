from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TimedItem:
    index: int
    timestamp_ns: int
    payload: object


@dataclass(frozen=True)
class SynchronizedPair:
    camera: TimedItem
    lidar: TimedItem
    delta_ns: int


def nearest_synchronize(
    cameras: Sequence[TimedItem],
    lidars: Sequence[TimedItem],
    max_delta_ns: int = 50_000_000,
) -> list[SynchronizedPair]:
    """Pair each camera frame with the nearest unused LiDAR scan.

    Inputs must be timestamp ordered. The function is deterministic and never
    reuses a LiDAR item, which makes the resulting frame manifest auditable.
    """
    pairs: list[SynchronizedPair] = []
    lidar_cursor = 0
    used: set[int] = set()
    for camera in cameras:
        while lidar_cursor + 1 < len(lidars) and lidars[lidar_cursor + 1].timestamp_ns <= camera.timestamp_ns:
            lidar_cursor += 1
        candidates = [lidar_cursor]
        if lidar_cursor + 1 < len(lidars):
            candidates.append(lidar_cursor + 1)
        candidates = [index for index in candidates if index not in used]
        if not candidates:
            continue
        chosen_index = min(candidates, key=lambda index: abs(lidars[index].timestamp_ns - camera.timestamp_ns))
        lidar = lidars[chosen_index]
        delta = lidar.timestamp_ns - camera.timestamp_ns
        if abs(delta) <= max_delta_ns:
            used.add(chosen_index)
            pairs.append(SynchronizedPair(camera=camera, lidar=lidar, delta_ns=delta))
    return pairs


def assert_monotonic(values: Iterable[int], *, allow_equal: bool = False) -> None:
    previous: int | None = None
    for value in values:
        if previous is not None and (value < previous or (value == previous and not allow_equal)):
            raise ValueError("timestamps are not strictly monotonic")
        previous = value


def timestamp_quality(source: str) -> str:
    allowed = {"header", "record", "pose-derived", "index-only", "unknown"}
    if source not in allowed:
        raise ValueError(f"unsupported timestamp source: {source}")
    return source

