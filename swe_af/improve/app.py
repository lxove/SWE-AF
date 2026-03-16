"""swe_af.improve.app — Continuous improvement Agent entry point.

Exposes:
  - app: Agent instance with node_id='swe-improve'
  - improve: end-to-end improve loop reasoner
  - main: entry point for swe-improve console script
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone

from agentfield import Agent

from swe_af.execution.envelope import unwrap_call_result as _unwrap
from swe_af.improve import improve_router
from swe_af.improve.schemas import (
    ImproveConfig,
    ImproveResult,
    ImprovementArea,
    ImprovementState,
    RunRecord,
)

NODE_ID = os.getenv("NODE_ID", "swe-improve")

app = Agent(
    node_id=NODE_ID,
    version="1.0.0",
    description="Continuous recursive improvement agent",
    agentfield_server=os.getenv("AGENTFIELD_SERVER", "http://localhost:8080"),
    api_key=os.getenv("AGENTFIELD_API_KEY"),
)

app.include_router(improve_router)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------


def _state_path(repo_path: str) -> str:
    """Return path to state file."""
    return os.path.join(repo_path, ".swe-af", "improvements.json")


def _load_state(repo_path: str) -> ImprovementState:
    """Load state from disk, or return empty state."""
    path = _state_path(repo_path)
    if not os.path.exists(path):
        return ImprovementState(repo_path=repo_path)
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return ImprovementState(**data)
    except Exception:
        # Corrupted state — start fresh
        return ImprovementState(repo_path=repo_path)


def _save_state(repo_path: str, state: ImprovementState) -> None:
    """Save state to disk atomically."""
    state_dir = os.path.join(repo_path, ".swe-af")
    os.makedirs(state_dir, exist_ok=True)
    path = _state_path(repo_path)

    # Atomic write via temp file + rename
    fd, tmp_path = tempfile.mkstemp(dir=state_dir, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state.model_dump(), f, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def _ensure_gitignore(repo_path: str) -> None:
    """Ensure .swe-af/ is in .gitignore."""
    gitignore_path = os.path.join(repo_path, ".gitignore")
    entry = ".swe-af/"

    if os.path.exists(gitignore_path):
        with open(gitignore_path, "r") as f:
            content = f.read()
        if entry in content or ".swe-af" in content:
            return
        with open(gitignore_path, "a") as f:
            if not content.endswith("\n"):
                f.write("\n")
            f.write(f"{entry}\n")
    else:
        with open(gitignore_path, "w") as f:
            f.write(f"{entry}\n")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _pick_next_improvement(
    state: ImprovementState,
    categories: list[str] | None,
) -> ImprovementArea | None:
    """Pick highest-priority pending improvement, optionally filtered by category."""
    candidates = [
        imp for imp in state.improvements
        if imp.status == "pending"
        and (categories is None or imp.category in categories)
    ]
    if not candidates:
        return None
    # Sort by priority (ascending = higher priority first)
    candidates.sort(key=lambda x: x.priority)
    return candidates[0]


def _within_budget(start_time: float, cfg: ImproveConfig) -> bool:
    """Check if we're still within time budget."""
    return time.time() - start_time < cfg.max_time_seconds


@app.reasoner()
async def improve(
    repo_path: str,
    config: dict | None = None,
) -> dict:
    """Continuous recursive improvement loop.

    10-step loop:
    1. Load state from .swe-af/improvements.json
    2. Pick highest-priority pending improvement
    3. Validate it's still relevant
    4. If no pending improvements, trigger a scan
    5. Plan the improvement
    6. Execute the improvement
    7. Verify tests pass
    8. Commit with improve: prefix
    9. Record new findings
    10. Loop until budget exhausted

    Args:
        repo_path: Absolute path to the repository.
        config: ImproveConfig as dict.

    Returns:
        ImproveResult.model_dump()
    """
    cfg = ImproveConfig(**(config or {}))

    # Validate repo_path
    if not repo_path or not os.path.isdir(repo_path):
        return ImproveResult(
            improvements_completed=[],
            improvements_found=[],
            improvements_skipped=[],
            improvements_failed=[],
            budget_remaining_seconds=cfg.max_time_seconds,
            stopped_reason="error",
            summary=f"Invalid repo_path: {repo_path}",
            run_record=RunRecord(
                started_at=datetime.now(timezone.utc).isoformat(),
                stopped_reason="error",
            ),
        ).model_dump()

    # Ensure .gitignore excludes .swe-af/
    _ensure_gitignore(repo_path)

    # Initialize run tracking
    start_time = time.time()
    run_started = datetime.now(timezone.utc).isoformat()
    completed: list[ImprovementArea] = []
    found: list[ImprovementArea] = []
    skipped: list[ImprovementArea] = []
    failed: list[ImprovementArea] = []

    app.note(
        f"Improve loop starting: repo={repo_path}, max_time={cfg.max_time_seconds}s",
        tags=["improve", "loop", "start"],
    )

    # Load state
    state = _load_state(repo_path)

    improvements_this_run = 0
    stopped_reason = "no_more_improvements"

    while _within_budget(start_time, cfg):
        # Check max_improvements limit
        if cfg.max_improvements > 0 and improvements_this_run >= cfg.max_improvements:
            stopped_reason = "max_improvements_reached"
            break

        # Pick next improvement
        improvement = _pick_next_improvement(state, cfg.categories)

        # If no pending improvements, trigger a scan
        if improvement is None:
            app.note("No pending improvements — triggering scan", tags=["improve", "scan"])

            raw_scan = await app.call(
                f"{NODE_ID}.scan_for_improvements",
                repo_path=repo_path,
                config=cfg.model_dump(),
                existing_improvements=[imp.model_dump() for imp in state.improvements],
            )
            scan_result = _unwrap(raw_scan, "scan_for_improvements")

            new_areas = scan_result.get("new_areas", [])
            for area_dict in new_areas:
                area = ImprovementArea(**area_dict)
                state.improvements.append(area)
                found.append(area)

            state.last_scan_at = datetime.now(timezone.utc).isoformat()
            _save_state(repo_path, state)

            # Re-pick after scan
            improvement = _pick_next_improvement(state, cfg.categories)
            if improvement is None:
                stopped_reason = "no_more_improvements"
                break

        # Mark as in_progress
        improvement.status = "in_progress"
        _save_state(repo_path, state)

        # Validate improvement
        app.note(f"Validating: {improvement.id}", tags=["improve", "validate"])
        raw_validate = await app.call(
            f"{NODE_ID}.validate_improvement",
            improvement_area=improvement.model_dump(),
            repo_path=repo_path,
            config=cfg.model_dump(),
        )
        validate_result = _unwrap(raw_validate, "validate_improvement")

        if not validate_result.get("is_valid", True):
            # Mark as stale and continue
            improvement.status = "stale"
            improvement.notes = validate_result.get("reason", "Marked stale by validator")
            skipped.append(improvement)
            _save_state(repo_path, state)
            app.note(f"Skipped stale: {improvement.id}", tags=["improve", "stale"])
            continue

        # Calculate per-improvement timeout
        remaining = cfg.max_time_seconds - (time.time() - start_time)
        timeout = int(min(remaining, 300))
        if timeout <= 0:
            stopped_reason = "budget_exhausted"
            improvement.status = "pending"  # Revert to pending
            _save_state(repo_path, state)
            break

        # Execute improvement
        app.note(f"Executing: {improvement.id} (timeout={timeout}s)", tags=["improve", "execute"])
        raw_exec = await app.call(
            f"{NODE_ID}.execute_improvement",
            improvement_area=improvement.model_dump(),
            repo_path=repo_path,
            timeout_seconds=timeout,
            config=cfg.model_dump(),
        )
        exec_result = _unwrap(raw_exec, "execute_improvement")

        if exec_result.get("success", False):
            # Mark completed
            improvement.status = "completed"
            improvement.completed_by_run = datetime.now(timezone.utc).isoformat()
            improvement.commit_sha = exec_result.get("commit_sha")
            completed.append(improvement)
            improvements_this_run += 1

            # Append new findings
            new_findings = exec_result.get("new_findings", [])
            for finding_dict in new_findings:
                finding = ImprovementArea(**finding_dict)
                finding.found_by_run = datetime.now(timezone.utc).isoformat()
                finding.status = "pending"
                state.improvements.append(finding)
                found.append(finding)

            app.note(
                f"Completed: {improvement.id} -> {improvement.commit_sha}",
                tags=["improve", "complete"],
            )
        else:
            # Mark failed
            improvement.status = "failed"
            improvement.notes = exec_result.get("error", "Execution failed")
            failed.append(improvement)
            app.note(
                f"Failed: {improvement.id}: {improvement.notes}",
                tags=["improve", "failed"],
            )

        _save_state(repo_path, state)

        # Check budget again
        if not _within_budget(start_time, cfg):
            stopped_reason = "budget_exhausted"
            break

    # Final budget check
    if _within_budget(start_time, cfg) and stopped_reason == "no_more_improvements":
        pass  # Already set correctly
    elif not _within_budget(start_time, cfg):
        stopped_reason = "budget_exhausted"

    # Record run
    budget_used = time.time() - start_time
    run_record = RunRecord(
        started_at=run_started,
        ended_at=datetime.now(timezone.utc).isoformat(),
        improvements_found=len(found),
        improvements_completed=len(completed),
        improvements_skipped=len(skipped),
        budget_used_seconds=budget_used,
        stopped_reason=stopped_reason,
    )
    state.runs.append(run_record)
    _save_state(repo_path, state)

    summary = (
        f"Completed {len(completed)}, found {len(found)}, "
        f"skipped {len(skipped)}, failed {len(failed)} improvements. "
        f"Stopped: {stopped_reason}"
    )

    app.note(f"Improve loop finished: {summary}", tags=["improve", "loop", "done"])

    return ImproveResult(
        improvements_completed=completed,
        improvements_found=found,
        improvements_skipped=skipped,
        improvements_failed=failed,
        budget_remaining_seconds=max(0, cfg.max_time_seconds - budget_used),
        stopped_reason=stopped_reason,
        summary=summary,
        run_record=run_record,
    ).model_dump()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for swe-improve console script."""
    app.run(port=int(os.getenv("PORT", "8005")), host="0.0.0.0")


if __name__ == "__main__":
    main()
