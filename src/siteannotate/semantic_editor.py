from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .io import read_json, sha256_file
from .pointcloud import downsample_indices, read_ascii_pcd, read_semantickitti_labels, write_semantickitti_labels


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _html() -> bytes:
    return (Path(__file__).resolve().parent / "static" / "semantic_editor.html").read_bytes()


def _ontology(review_root: Path) -> dict[str, Any]:
    index = read_json(review_root / "review_index.json")
    if "ontology" in index:
        return index["ontology"]
    path = Path(index.get("ontology_path", "configs/rellis_ontology.json"))
    if not path.is_absolute():
        path = _repo_root() / path
    return read_json(path)


class SemanticEditor:
    def __init__(self, review_root: Path, host: str = "127.0.0.1", port: int = 8090):
        self.review_root = review_root
        self.host = host
        self.port = port

    def _scene_root(self) -> Path:
        index = read_json(self.review_root / "review_index.json")
        return self.review_root / "data" / index["sequence_id"]

    def _frame_payload(self, frame_key: str) -> dict[str, Any]:
        scene = self._scene_root()
        pcd = scene / "lidar" / f"{frame_key}.pcd"
        label_path = scene / "semantic" / f"{frame_key}.label"
        if not pcd.exists():
            raise FileNotFoundError(frame_key)
        all_points = read_ascii_pcd(pcd)
        display_indices = downsample_indices(len(all_points), 30_000)
        labels = read_semantickitti_labels(label_path) if label_path.exists() and label_path.stat().st_size else [0] * len(all_points)
        labels = [value & 0xFFFF for value in labels]
        if len(labels) != len(all_points):
            raise ValueError(f"label count mismatch for {frame_key}")
        return {
            "frame_key": frame_key,
            "points": [all_points[index] for index in display_indices],
            "display_indices": display_indices,
            "labels": [labels[index] for index in display_indices],
            "sha256": sha256_file(label_path) if label_path.exists() else None,
        }

    def _save_updates(self, frame_key: str, payload: dict[str, Any]) -> dict[str, Any]:
        path = self._scene_root() / "semantic" / f"{frame_key}.label"
        if not path.exists():
            raise FileNotFoundError(frame_key)
        labels = read_semantickitti_labels(path)
        expected = payload.get("expected_sha256")
        if expected and expected != sha256_file(path):
            raise ValueError("label file changed since it was loaded; reload before saving")
        updates = payload.get("updates", [])
        class_ids = set(_ontology(self.review_root)["class_ids"])
        for update in updates:
            index, label = int(update["index"]), int(update["label"])
            if index < 0 or index >= len(labels):
                raise ValueError(f"point index out of range: {index}")
            if label not in class_ids:
                raise ValueError(f"unknown RELLIS semantic class ID: {label}")
            labels[index] = label
        write_semantickitti_labels(path, labels)
        with (self.review_root / "semantic-audit.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"frame_key": frame_key, "updates": updates}) + "\n")
        return {"frame_key": frame_key, "updated": len(updates), "sha256": sha256_file(path)}

    def serve_forever(self) -> None:
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def _json(self, value: Any, status: int = 200) -> None:
                body = json.dumps(value).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                return json.loads(self.rfile.read(length).decode("utf-8")) if length else {}

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/":
                    body = _html()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if path == "/api/frames":
                    index = read_json(parent.review_root / "review_index.json")
                    ontology = _ontology(parent.review_root)
                    classes = [
                        {
                            "id": int(key),
                            "name": value,
                            "color": ontology.get("colors", {}).get(key, [255, 255, 255]),
                        }
                        for key, value in ontology["class_names"].items()
                    ]
                    self._json({"frames": index["frames"], "classes": classes})
                    return
                prefix = "/api/frame/"
                if path.startswith(prefix) and not path.endswith("/label"):
                    try:
                        self._json(parent._frame_payload(unquote(path[len(prefix) :])))
                    except (FileNotFoundError, ValueError) as exc:
                        self._json({"error": str(exc)}, HTTPStatus.NOT_FOUND)
                    return
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802
                prefix = "/api/frame/"
                if not self.path.startswith(prefix) or not self.path.endswith("/label"):
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                    return
                frame_key = unquote(self.path[len(prefix) : -len("/label")])
                try:
                    self._json(parent._save_updates(frame_key, self._body()))
                except (FileNotFoundError, ValueError) as exc:
                    self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

            def log_message(self, *_: Any) -> None:
                return

        print(f"Semantic editor: http://{self.host}:{self.port}/")
        ThreadingHTTPServer((self.host, self.port), Handler).serve_forever()


def serve_semantic_editor(review_root: Path, host: str = "127.0.0.1", port: int = 8090) -> None:
    SemanticEditor(review_root, host, port).serve_forever()
