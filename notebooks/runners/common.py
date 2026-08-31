"""Shared helpers for experiment runners."""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOC = ROOT / "EXPERIMENT_RESULTS.md"


def result_dir(name: str) -> Path:
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")


def append_unit_section(unit: str, payload: dict) -> None:
    """Append a detail section and update the summary table row for `unit`."""
    status = payload.get("status", "UNKNOWN")
    duration = payload.get("duration_sec")
    duration_s = f"{duration:.1f}s" if isinstance(duration, (int, float)) else ""
    metrics = payload.get("metrics_summary", "")
    artifacts = payload.get("artifacts", "")

    text = DOC.read_text() if DOC.exists() else ""
    lines = text.splitlines()
    new_lines = []
    for line in lines:
        if line.startswith("|") and not line.startswith("|---") and not line.startswith("| Unit"):
            cols = [c.strip() for c in line.split("|")[1:-1]]
            if cols and (cols[0] == unit or cols[0].startswith(f"{unit} ")):
                if len(cols) >= 5:
                    cols[1] = status
                    cols[2] = duration_s
                    cols[3] = str(metrics)[:80]
                    cols[4] = str(artifacts)
                    line = "| " + " | ".join(cols) + " |"
        new_lines.append(line)
    text = "\n".join(new_lines)

    section = [
        "",
        f"### Unit {unit}",
        "",
        f"- **Status**: {status}",
        f"- **Started**: {payload.get('started')}",
        f"- **Ended**: {payload.get('ended')}",
        f"- **Duration (s)**: {duration}",
        f"- **Metrics**: {payload.get('metrics')}",
        f"- **Artifacts**: {artifacts}",
    ]
    if payload.get("error"):
        section.append(f"- **Error**: `{payload['error']}`")
    if payload.get("notes"):
        section.append(f"- **Notes**: {payload['notes']}")
    section.append("")

    # Replace existing section if present
    marker = f"### Unit {unit}"
    if marker in text:
        before, _, rest = text.partition(marker)
        # drop until next ### or end
        idx = rest.find("\n### ")
        if idx >= 0:
            rest = rest[idx:]
            text = before.rstrip() + "\n" + "\n".join(section) + rest
        else:
            text = before.rstrip() + "\n" + "\n".join(section)
    else:
        text = text.rstrip() + "\n" + "\n".join(section)

    DOC.write_text(text if text.endswith("\n") else text + "\n")


class UnitRun:
    def __init__(self, unit: str, out_name: str):
        self.unit = unit
        self.out = result_dir(out_name)
        self.started = None
        self.t0 = None

    def __enter__(self):
        self.started = now_iso()
        self.t0 = time.time()
        return self

    def success(self, metrics: dict, notes: str = ""):
        ended = now_iso()
        duration = time.time() - self.t0
        payload = {
            "status": "PASS",
            "started": self.started,
            "ended": ended,
            "duration_sec": duration,
            "metrics": metrics,
            "metrics_summary": ", ".join(f"{k}={v}" for k, v in metrics.items()),
            "artifacts": str(self.out),
            "notes": notes,
        }
        write_json(self.out / "result.json", payload)
        append_unit_section(self.unit, payload)
        return payload

    def fail(self, exc: BaseException):
        ended = now_iso()
        duration = time.time() - self.t0
        err = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        (self.out / "error.txt").write_text(tb)
        payload = {
            "status": "FAIL",
            "started": self.started,
            "ended": ended,
            "duration_sec": duration,
            "metrics": {},
            "metrics_summary": "failed",
            "artifacts": str(self.out),
            "error": err,
            "notes": "see error.txt",
        }
        write_json(self.out / "result.json", payload)
        append_unit_section(self.unit, payload)
        return payload

    def __exit__(self, exc_type, exc, tb):
        return False
