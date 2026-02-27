"""Global workflow registry mapping workflow_id to repo metadata.

Stores entries in ``~/.swe-af/workflows.json`` with file-level locking
so concurrent processes can safely read/write the registry.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _registry_path() -> Path:
    """Return the path to the registry file, creating the directory if needed."""
    dir_path = Path.home() / ".swe-af"
    dir_path.mkdir(mode=0o700, exist_ok=True)
    return dir_path / "workflows.json"


def _read_registry(path: Path) -> dict:
    """Read the registry file under a shared lock. Returns empty dict if missing."""
    if not path.exists():
        return {}
    fd = os.open(str(path), os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        try:
            with open(fd, "r", closefd=False) as f:
                return json.load(f)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _write_registry(path: Path, data: dict) -> None:
    """Write the registry file under an exclusive lock."""
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            with open(fd, "w", closefd=False) as f:
                json.dump(data, f, indent=2, default=str)
                f.truncate()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def register_workflow(
    workflow_id: str,
    repo_path: str,
    artifacts_dir: str,
    goal: str,
) -> None:
    """Add a new workflow entry with status='planning' and timestamps."""
    path = _registry_path()
    # Read-then-write under exclusive lock
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            with open(fd, "r+", closefd=False) as f:
                try:
                    data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    data = {}
                now = _now_iso()
                data[workflow_id] = {
                    "workflow_id": workflow_id,
                    "repo_path": repo_path,
                    "artifacts_dir": artifacts_dir,
                    "goal": goal,
                    "status": "planning",
                    "created_at": now,
                    "updated_at": now,
                }
                f.seek(0)
                json.dump(data, f, indent=2, default=str)
                f.truncate()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def update_workflow(workflow_id: str, **kwargs: str) -> None:
    """Update fields on an existing workflow entry.

    Accepts any subset of: status, repo_path, artifacts_dir, goal.
    Always sets ``updated_at`` to the current time.
    """
    allowed = {"status", "repo_path", "artifacts_dir", "goal"}
    path = _registry_path()
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            with open(fd, "r+", closefd=False) as f:
                try:
                    data = json.load(f)
                except (json.JSONDecodeError, ValueError):
                    data = {}
                if workflow_id not in data:
                    return
                for key, value in kwargs.items():
                    if key in allowed:
                        data[workflow_id][key] = value
                data[workflow_id]["updated_at"] = _now_iso()
                f.seek(0)
                json.dump(data, f, indent=2, default=str)
                f.truncate()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def lookup_workflow(workflow_id: str) -> dict | None:
    """Return the entry dict for a workflow_id, or None if not found."""
    path = _registry_path()
    data = _read_registry(path)
    return data.get(workflow_id)


def list_workflows() -> list[dict]:
    """Return all workflow entries as a list."""
    path = _registry_path()
    data = _read_registry(path)
    return list(data.values())
