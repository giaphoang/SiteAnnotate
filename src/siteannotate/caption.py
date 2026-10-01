from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_json
from .models import CaptionRecord


def _endpoint_url(base_url: str, suffix: str) -> str:
    return base_url.rstrip("/") + "/" + suffix.lstrip("/")


class OpenAICompatibleCaptionClient:
    def __init__(self, base_url: str, api_key: str = "EMPTY", model: str = "cosmos-reason1-7b", timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(_endpoint_url(self.base_url, path), data=body, method="GET" if body is None else "POST")
        request.add_header("Accept", "application/json")
        if body is not None:
            request.add_header("Content-Type", "application/json")
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"caption endpoint returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"caption endpoint unavailable: {exc.reason}") from exc

    def health(self) -> dict[str, Any]:
        return self._request("v1/models" if not self.base_url.endswith("/v1") else "models")

    def caption_video(self, video_path: Path, prompt: str, max_tokens: int = 2048) -> str:
        encoded = base64.b64encode(video_path.read_bytes()).decode("ascii")
        content = [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{encoded}"}},
        ]
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        response = self._request("v1/chat/completions" if not self.base_url.endswith("/v1") else "chat/completions", payload)
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"caption endpoint response lacks choices[0].message.content: {response}") from exc


def _prompt_block(text: str, name: str) -> str:
    match = re.search(
        rf"^{name}:\s*\|\s*\n(?P<body>(?:^[ \t]+.*\n?)+)",
        text,
        re.MULTILINE,
    )
    if not match:
        return ""
    return "\n".join(
        line[2:] if line.startswith("  ") else line for line in match.group("body").splitlines()
    ).strip()


def load_prompt(path: Path) -> str:
    """Read the system and user blocks from the repository's small YAML prompt format."""
    text = path.read_text(encoding="utf-8")
    system = _prompt_block(text, "system_prompt")
    user = _prompt_block(text, "user_prompt")
    if not user:
        raise ValueError(f"prompt file has no user_prompt block: {path}")
    return "\n\n".join(part for part in (system, user) if part)


def _extract_json(text: str) -> tuple[dict[str, Any], str]:
    candidate = text.strip()
    if "```" in candidate:
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(candidate)
        if isinstance(value, dict):
            return value, "parsed"
    except json.JSONDecodeError:
        pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(candidate[start : end + 1])
            if isinstance(value, dict):
                return value, "parsed"
        except json.JSONDecodeError:
            pass
    return {"scene": {"summary": candidate}, "events": []}, "invalid-json"


def _map_event_frames(events: list[dict[str, Any]], mapping: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not mapping:
        return events
    starts: list[float] = []
    current = 0.0
    for item in mapping:
        starts.append(current)
        current += float(item.get("duration_s", 0.0))

    def frame_at(seconds: float) -> dict[str, Any]:
        chosen = 0
        for index, start in enumerate(starts):
            if start <= seconds:
                chosen = index
            else:
                break
        return mapping[chosen]

    output: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        item = dict(event)
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        item["frame_start"] = frame_at(start).get("frame_key")
        item["frame_end"] = frame_at(end).get("frame_key")
        item["start_timestamp_ns"] = frame_at(start).get("camera_timestamp_ns")
        item["end_timestamp_ns"] = frame_at(end).get("camera_timestamp_ns")
        output.append(item)
    return output


def caption_run(run_dir: Path, prompt_path: Path, *, base_url: str | None = None, model: str | None = None) -> CaptionRecord:
    sequence = json.loads((run_dir / "sequence.json").read_text(encoding="utf-8"))
    sequence_id = sequence["sequence_id"]
    video_path = run_dir / "videos" / f"{sequence_id}.mp4"
    mapping_path = run_dir / "videos" / f"{sequence_id}.frames.jsonl"
    if not video_path.exists() or not mapping_path.exists():
        raise FileNotFoundError("run has no timestamped video; run the encode stage first")
    base_url = base_url or os.environ.get("COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL")
    if not base_url:
        raise RuntimeError("set COSMOS_CURATOR_OPENAI_CAPTION_BASE_URL or pass --base-url")
    client = OpenAICompatibleCaptionClient(
        base_url,
        api_key=os.environ.get("COSMOS_CURATOR_OPENAI_CAPTION_API_KEY", "EMPTY"),
        model=model or os.environ.get("COSMOS_REASON_MODEL", "nvidia/Cosmos-Reason1-7B"),
    )
    raw_text = client.caption_video(video_path, load_prompt(prompt_path))
    parsed, parse_status = _extract_json(raw_text)
    mapping = list(read_jsonl(mapping_path))
    events = _map_event_frames(parsed.get("events", []), mapping)
    record = CaptionRecord(
        schema_version=1,
        clip_id=f"{sequence_id}-full",
        raw_text=raw_text,
        scene=parsed.get("scene", {}) if isinstance(parsed.get("scene", {}), dict) else {},
        events=events,
        model={"id": client.model, "endpoint": base_url},
        parse_status=parse_status,
        frame_mapping=mapping,
    )
    output = run_dir / "captions" / f"{sequence_id}.json"
    write_json(output, record.to_dict())
    return record
