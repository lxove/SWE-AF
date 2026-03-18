"""swe_af.improve.app — Continuous improvement Agent entry point.

Exposes:
  - app: Agent instance with node_id='swe-improve'
  - improve: end-to-end improve loop reasoner
  - main: entry point for swe-improve console script
"""

from __future__ import annotations

import json
import os
import random
import string
import subprocess
import tempfile
import time
from datetime import datetime, timezone

from agentfield import Agent

from swe_af.execution.envelope import unwrap_call_result as _unwrap
from swe_af.execution.schemas import _derive_repo_name as _repo_name_from_url
from swe_af.improve import improve_router
from swe_af.improve.metrics_util import UsageAccumulator
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


def _setup_work_branch(repo_path: str, cfg: ImproveConfig) -> str | None:
    """Create a work branch from the configured base branch and push it to origin.

    1. Verify ``cfg.branch`` exists locally — fail early if not.
    2. Create ``improve/<YYYYMMDD>_<random6>`` from it.
    3. Push the (empty) branch to origin immediately so the remote tracks it.

    Returns the work branch name on success, or ``None`` on failure (with a
    note logged via ``app.note``).
    """
    base = cfg.branch or "main"

    # Verify base branch exists
    check = subprocess.run(
        ["git", "rev-parse", "--verify", base],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        # Maybe it exists as origin/<base> but not locally — try to create it
        fetch = subprocess.run(
            ["git", "fetch", "origin", base],
            cwd=repo_path,
            capture_output=True,
            text=True,
        )
        if fetch.returncode != 0:
            app.note(
                f"Base branch '{base}' not found locally or on remote",
                tags=["improve", "branch", "error"],
            )
            return None

    # Generate work branch name: improve/<YYYYMMDD>_<random6>
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    work_branch = f"improve/{date_str}_{suffix}"

    # Create and switch to work branch
    result = subprocess.run(
        ["git", "checkout", "-b", work_branch, base],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        app.note(
            f"Failed to create work branch '{work_branch}': {result.stderr.strip()}",
            tags=["improve", "branch", "error"],
        )
        return None

    app.note(
        f"Created work branch {work_branch} from {base}",
        tags=["improve", "branch", "create"],
    )

    # Push the empty branch to origin immediately
    push = subprocess.run(
        ["git", "push", "-u", "origin", work_branch],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if push.returncode != 0:
        app.note(
            f"Failed to push empty work branch (non-fatal): {push.stderr.strip()}",
            tags=["improve", "branch", "push-error"],
        )
        # Non-fatal — we'll try again after commits
    else:
        app.note(
            f"Pushed empty work branch {work_branch} to origin",
            tags=["improve", "branch", "push"],
        )

    return work_branch


@app.reasoner()
async def improve(
    repo_path: str = "",
    repo_url: str = "",
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
        repo_path: Absolute path to the repository (optional if repo_url given).
        repo_url: Git URL to clone if repo_path is empty.
        config: ImproveConfig as dict.

    Returns:
        ImproveResult.model_dump()
    """
    cfg = ImproveConfig(**(config or {}))

    # Allow repo_url from direct parameter (overrides config)
    effective_repo_url = repo_url or cfg.repo_url

    # Auto-derive repo_path from repo_url when not specified
    if effective_repo_url and not repo_path:
        repo_path = f"/workspaces/{_repo_name_from_url(effective_repo_url)}"

    # Validate that we have a repo_path
    if not repo_path:
        return ImproveResult(
            improvements_completed=[],
            improvements_found=[],
            improvements_skipped=[],
            improvements_failed=[],
            budget_remaining_seconds=cfg.max_time_seconds,
            stopped_reason="error",
            summary="Either repo_path or repo_url must be provided",
            run_record=RunRecord(
                started_at=datetime.now(timezone.utc).isoformat(),
                stopped_reason="error",
            ),
        ).model_dump()

    # Clone if repo_url is set and target doesn't exist yet
    git_dir = os.path.join(repo_path, ".git")
    if effective_repo_url and not os.path.exists(git_dir):
        app.note(f"Cloning {effective_repo_url} → {repo_path}", tags=["improve", "clone"])
        os.makedirs(repo_path, exist_ok=True)
        clone_cmd = ["git", "clone", effective_repo_url, repo_path]
        clone_branch = cfg.branch or ""
        if clone_branch:
            clone_cmd += ["--branch", clone_branch]
        clone_result = subprocess.run(
            clone_cmd,
            capture_output=True,
            text=True,
        )
        if clone_result.returncode != 0:
            err = clone_result.stderr.strip()
            return ImproveResult(
                improvements_completed=[],
                improvements_found=[],
                improvements_skipped=[],
                improvements_failed=[],
                budget_remaining_seconds=cfg.max_time_seconds,
                stopped_reason="error",
                summary=f"git clone failed (exit {clone_result.returncode}): {err}",
                run_record=RunRecord(
                    started_at=datetime.now(timezone.utc).isoformat(),
                    stopped_reason="error",
                ),
            ).model_dump()

    # Validate repo_path exists
    if not os.path.isdir(repo_path):
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

    # Create work branch from the configured base branch
    work_branch = _setup_work_branch(repo_path, cfg)
    if work_branch is None:
        return ImproveResult(
            improvements_completed=[],
            improvements_found=[],
            improvements_skipped=[],
            improvements_failed=[],
            budget_remaining_seconds=cfg.max_time_seconds,
            stopped_reason="error",
            summary=f"Base branch '{cfg.branch}' not found",
            run_record=RunRecord(
                started_at=datetime.now(timezone.utc).isoformat(),
                stopped_reason="error",
            ),
        ).model_dump()

    # Initialize run tracking
    start_time = time.time()
    run_started = datetime.now(timezone.utc).isoformat()
    completed: list[ImprovementArea] = []
    found: list[ImprovementArea] = []
    skipped: list[ImprovementArea] = []
    failed: list[ImprovementArea] = []
    usage_acc = UsageAccumulator()

    app.note(
        f"Improve loop starting: repo={repo_path}, branch={work_branch}, max_time={cfg.max_time_seconds}s",
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
            usage_acc.add(scan_result.pop("_metrics", None))

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
        usage_acc.add(validate_result.pop("_metrics", None))

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
        usage_acc.add(exec_result.pop("_metrics", None))

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

            # Push immediately after each commit so work is never stuck locally
            _push_branch(repo_path)
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

    # Build usage summary
    usage_summary = usage_acc.to_summary()

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
        usage=usage_summary,
    )
    state.runs.append(run_record)
    _save_state(repo_path, state)

    usage_line = usage_acc.format_summary_line()
    summary = (
        f"Completed {len(completed)}, found {len(found)}, "
        f"skipped {len(skipped)}, failed {len(failed)} improvements. "
        f"Stopped: {stopped_reason}"
    )
    if usage_line:
        summary += f"\nToken usage: {usage_line}"

    app.note(f"Improve loop finished: {summary}", tags=["improve", "loop", "done"])

    # Branch was already pushed after each commit; do a final push for safety
    pr_url = ""
    if completed:
        _push_branch(repo_path)
        # Create draft PR on the already-pushed branch
        if cfg.enable_github_pr:
            pr_url = await _maybe_create_pr(repo_path, cfg, work_branch, completed, summary)

    return ImproveResult(
        improvements_completed=completed,
        improvements_found=found,
        improvements_skipped=skipped,
        improvements_failed=failed,
        budget_remaining_seconds=max(0, cfg.max_time_seconds - budget_used),
        stopped_reason=stopped_reason,
        summary=summary,
        run_record=run_record,
        usage=usage_summary,
        remote_branch=work_branch,
        pr_url=pr_url,
    ).model_dump()


# ---------------------------------------------------------------------------
# Push & PR creation
# ---------------------------------------------------------------------------


def _git(repo_path: str, *args: str) -> str:
    """Run a git command and return stdout, or empty string on failure."""
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _push_branch(repo_path: str) -> str:
    """Push current branch to origin. Returns the branch name or empty string on failure."""
    remote_url = _git(repo_path, "remote", "get-url", "origin")
    if not remote_url:
        app.note("No remote found — skipping push", tags=["improve", "push", "skip"])
        return ""

    current_branch = _git(repo_path, "rev-parse", "--abbrev-ref", "HEAD")
    if not current_branch or current_branch == "HEAD":
        app.note("Detached HEAD — skipping push", tags=["improve", "push", "skip"])
        return ""

    app.note(
        f"Pushing branch {current_branch} to origin",
        tags=["improve", "push", "start"],
    )

    result = subprocess.run(
        ["git", "push", "-u", "origin", current_branch],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        app.note(
            f"Push failed (non-fatal): {result.stderr.strip()}",
            tags=["improve", "push", "error"],
        )
        return ""

    app.note(
        f"Pushed branch {current_branch} to origin",
        tags=["improve", "push", "complete"],
    )
    return current_branch


async def _maybe_create_pr(
    repo_path: str,
    cfg: ImproveConfig,
    current_branch: str,
    completed: list[ImprovementArea],
    summary: str,
) -> str:
    """Create a draft PR for the already-pushed branch. Returns PR URL or empty string."""
    # Determine base branch for PR — defaults to cfg.branch (the base we branched from)
    base_branch = cfg.github_pr_base or cfg.branch or "main"

    # Build a goal string from completed improvements
    if len(completed) == 1:
        goal = f"improve: {completed[0].title}"
    else:
        goal = f"improve: {len(completed)} improvements"

    completed_issues = [
        {"issue_name": imp.id, "result_summary": imp.title}
        for imp in completed
    ]

    app.note(
        f"Creating draft PR: {current_branch} → {base_branch}",
        tags=["improve", "pr", "start"],
    )

    try:
        raw_pr = await app.call(
            f"{NODE_ID}.run_github_pr",
            repo_path=repo_path,
            integration_branch=current_branch,
            base_branch=base_branch,
            goal=goal,
            build_summary=summary,
            completed_issues=completed_issues,
            accumulated_debt=[],
            artifacts_dir="",
            model="sonnet",
            permission_mode=cfg.permission_mode,
            ai_provider="claude" if cfg.runtime == "claude_code" else "opencode",
        )
        pr_result = _unwrap(raw_pr, "run_github_pr")
        pr_url = pr_result.get("pr_url", "")
        if pr_url:
            app.note(f"Draft PR created: {pr_url}", tags=["improve", "pr", "complete"])
        return pr_url
    except Exception as e:
        app.note(f"PR creation failed (non-fatal): {e}", tags=["improve", "pr", "error"])
        return ""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for swe-improve console script."""
    app.run(port=int(os.getenv("PORT", "8005")), host="0.0.0.0")


if __name__ == "__main__":
    main()
