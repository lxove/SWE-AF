"""Tests for the resume(workflow_id) entry point.

Covers:
- Resume with execution checkpoint only → calls execute with resume=True
- Resume with plan checkpoint only → calls plan then execute
- Resume with both checkpoints → uses execution checkpoint (takes precedence)
- Resume with unknown workflow_id → raises RuntimeError
- Resume with no checkpoints → raises RuntimeError

Uses mocks for app.call, lookup_workflow, and filesystem checks.
Follows existing test patterns: _original_func to bypass @app.reasoner() decorator,
AsyncMock for app.call.
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import AsyncMock, patch

import pytest


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _get_resume_fn():
    """Import and return the unwrapped resume function."""
    import swe_af.app as _app_module

    return getattr(_app_module.resume, "_original_func", _app_module.resume)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_registry():
    """Patch lookup_workflow to return controlled data."""
    with patch("swe_af.workflow_registry.lookup_workflow") as mock:
        yield mock


@pytest.fixture
def mock_app_call():
    """Patch swe_af.app.app.call with an AsyncMock."""
    mock_call = AsyncMock()
    with patch("swe_af.app.app.call", mock_call):
        yield mock_call


@pytest.fixture
def mock_app_note():
    """Patch swe_af.app.app.note to silence observability output."""
    with patch("swe_af.app.app.note"):
        yield


@pytest.fixture
def mock_update_workflow():
    """Patch update_workflow to prevent file I/O."""
    with patch("swe_af.app.update_workflow"):
        yield


# ---------------------------------------------------------------------------
# Helper: create checkpoint files on disk
# ---------------------------------------------------------------------------


def _write_exec_checkpoint(base: str, data: dict | None = None) -> str:
    """Write an execution checkpoint file and return its path."""
    exec_dir = os.path.join(base, "execution")
    os.makedirs(exec_dir, exist_ok=True)
    path = os.path.join(exec_dir, "checkpoint.json")
    content = data or {
        "all_issues": [{"name": "issue-1"}],
        "levels": [["issue-1"]],
        "artifacts_dir": base,
        "original_plan_summary": "test plan",
    }
    with open(path, "w") as f:
        json.dump(content, f)
    return path


def _write_plan_checkpoint(base: str, data: dict | None = None) -> str:
    """Write a plan checkpoint file and return its path."""
    plan_dir = os.path.join(base, "plan")
    os.makedirs(plan_dir, exist_ok=True)
    path = os.path.join(plan_dir, "checkpoint.json")
    content = data or {
        "workflow_id": "test-wf",
        "goal": "Build a test app",
        "repo_path": "/tmp/test-repo",
        "artifacts_dir": ".artifacts",
        "phase": "pm",
        "workspace_manifest": None,
    }
    with open(path, "w") as f:
        json.dump(content, f)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResumeUnknownWorkflow:
    """Tests for resume with unknown or missing workflow_id."""

    @pytest.mark.asyncio
    async def test_unknown_workflow_id_raises(self, mock_app_call, mock_app_note):
        """resume raises RuntimeError for unknown workflow_id."""
        resume_fn = _get_resume_fn()

        with patch("swe_af.app.lookup_workflow", return_value=None):
            with pytest.raises(RuntimeError, match="not found in registry"):
                await resume_fn(workflow_id="nonexistent")


class TestResumeNoCheckpoints:
    """Tests for resume when no checkpoints exist on disk."""

    @pytest.mark.asyncio
    async def test_no_checkpoints_raises(self, tmp_path, mock_app_call, mock_app_note):
        """resume raises RuntimeError when no checkpoint files exist."""
        resume_fn = _get_resume_fn()
        repo_path = str(tmp_path / "repo")
        os.makedirs(repo_path, exist_ok=True)

        entry = {
            "workflow_id": "wf-no-ckpt",
            "repo_path": repo_path,
            "artifacts_dir": ".artifacts",
            "goal": "test",
        }

        with patch("swe_af.app.lookup_workflow", return_value=entry):
            with pytest.raises(RuntimeError, match="No checkpoints found"):
                await resume_fn(workflow_id="wf-no-ckpt")


class TestResumeExecutionCheckpoint:
    """Tests for resume with execution checkpoint present."""

    @pytest.mark.asyncio
    async def test_exec_checkpoint_calls_execute(
        self, tmp_path, mock_app_call, mock_app_note
    ):
        """resume with execution checkpoint calls execute with resume=True."""
        resume_fn = _get_resume_fn()
        repo_path = str(tmp_path / "repo")
        os.makedirs(repo_path, exist_ok=True)
        base = os.path.join(repo_path, ".artifacts")

        _write_exec_checkpoint(base)

        entry = {
            "workflow_id": "wf-exec",
            "repo_path": repo_path,
            "artifacts_dir": ".artifacts",
            "goal": "test",
        }

        mock_app_call.return_value = {"dag_result": "success"}

        with patch("swe_af.app.lookup_workflow", return_value=entry):
            await resume_fn(workflow_id="wf-exec")

        # Should have called execute (the only app.call)
        assert mock_app_call.call_count == 1
        call_args = mock_app_call.call_args
        # First positional arg contains the node target for execute
        assert "execute" in call_args[0][0]
        # resume=True must be passed
        assert call_args[1]["resume"] is True

    @pytest.mark.asyncio
    async def test_exec_checkpoint_takes_precedence_over_plan(
        self, tmp_path, mock_app_call, mock_app_note
    ):
        """When both checkpoints exist, execution checkpoint takes precedence."""
        resume_fn = _get_resume_fn()
        repo_path = str(tmp_path / "repo")
        os.makedirs(repo_path, exist_ok=True)
        base = os.path.join(repo_path, ".artifacts")

        _write_exec_checkpoint(base)
        _write_plan_checkpoint(base)

        entry = {
            "workflow_id": "wf-both",
            "repo_path": repo_path,
            "artifacts_dir": ".artifacts",
            "goal": "test",
        }

        mock_app_call.return_value = {"dag_result": "success"}

        with patch("swe_af.app.lookup_workflow", return_value=entry):
            await resume_fn(workflow_id="wf-both")

        # Only one call (execute), not two (plan then execute)
        assert mock_app_call.call_count == 1
        call_args = mock_app_call.call_args
        assert "execute" in call_args[0][0]
        assert call_args[1]["resume"] is True

    @pytest.mark.asyncio
    async def test_exec_checkpoint_plan_result_shape(
        self, tmp_path, mock_app_call, mock_app_note
    ):
        """Execution resume constructs plan_result from checkpoint data."""
        resume_fn = _get_resume_fn()
        repo_path = str(tmp_path / "repo")
        os.makedirs(repo_path, exist_ok=True)
        base = os.path.join(repo_path, ".artifacts")

        _write_exec_checkpoint(
            base,
            {
                "all_issues": [{"name": "i1"}, {"name": "i2"}],
                "levels": [["i1"], ["i2"]],
                "artifacts_dir": base,
                "original_plan_summary": "my plan",
            },
        )

        entry = {
            "workflow_id": "wf-shape",
            "repo_path": repo_path,
            "artifacts_dir": ".artifacts",
            "goal": "test",
        }

        mock_app_call.return_value = {}

        with patch("swe_af.app.lookup_workflow", return_value=entry):
            await resume_fn(workflow_id="wf-shape")

        call_args = mock_app_call.call_args
        plan_result = call_args[1]["plan_result"]
        assert plan_result["issues"] == [{"name": "i1"}, {"name": "i2"}]
        assert plan_result["levels"] == [["i1"], ["i2"]]
        assert plan_result["rationale"] == "my plan"


class TestResumePlanCheckpoint:
    """Tests for resume with only plan checkpoint present."""

    @pytest.mark.asyncio
    async def test_plan_checkpoint_calls_plan_then_execute(
        self, tmp_path, mock_app_call, mock_app_note, mock_update_workflow
    ):
        """resume with plan checkpoint only calls plan, then execute."""
        resume_fn = _get_resume_fn()
        repo_path = str(tmp_path / "repo")
        os.makedirs(repo_path, exist_ok=True)
        base = os.path.join(repo_path, ".artifacts")

        _write_plan_checkpoint(base, {
            "workflow_id": "wf-plan",
            "goal": "Build something",
            "repo_path": repo_path,
            "artifacts_dir": ".artifacts",
            "phase": "pm",
            "workspace_manifest": None,
        })

        entry = {
            "workflow_id": "wf-plan",
            "repo_path": repo_path,
            "artifacts_dir": ".artifacts",
            "goal": "Build something",
        }

        # First call returns plan result (wrapped in envelope-free dict),
        # second call returns execute result
        plan_result = {
            "prd": {},
            "architecture": {},
            "review": {},
            "issues": [{"name": "i1"}],
            "levels": [["i1"]],
            "file_conflicts": [],
            "artifacts_dir": ".artifacts",
            "rationale": "test",
        }
        exec_result = {"dag_result": "success"}
        mock_app_call.side_effect = [plan_result, exec_result]

        with patch("swe_af.app.lookup_workflow", return_value=entry):
            await resume_fn(workflow_id="wf-plan")

        # Two calls: first plan, then execute
        assert mock_app_call.call_count == 2

        # First call is plan
        plan_call = mock_app_call.call_args_list[0]
        assert "plan" in plan_call[0][0]
        assert plan_call[1]["workflow_id"] == "wf-plan"

        # Second call is execute
        exec_call = mock_app_call.call_args_list[1]
        assert "execute" in exec_call[0][0]

    @pytest.mark.asyncio
    async def test_plan_resume_passes_goal_from_checkpoint(
        self, tmp_path, mock_app_call, mock_app_note, mock_update_workflow
    ):
        """Plan resume uses goal from checkpoint data."""
        resume_fn = _get_resume_fn()
        repo_path = str(tmp_path / "repo")
        os.makedirs(repo_path, exist_ok=True)
        base = os.path.join(repo_path, ".artifacts")

        _write_plan_checkpoint(base, {
            "workflow_id": "wf-goal",
            "goal": "Checkpoint goal",
            "repo_path": repo_path,
            "artifacts_dir": ".artifacts",
            "phase": "pm",
            "workspace_manifest": {"repos": ["main"]},
        })

        entry = {
            "workflow_id": "wf-goal",
            "repo_path": repo_path,
            "artifacts_dir": ".artifacts",
            "goal": "Registry goal",
        }

        plan_result = {
            "prd": {},
            "architecture": {},
            "review": {},
            "issues": [],
            "levels": [],
            "file_conflicts": [],
            "artifacts_dir": ".artifacts",
            "rationale": "",
        }
        mock_app_call.side_effect = [plan_result, {}]

        with patch("swe_af.app.lookup_workflow", return_value=entry):
            await resume_fn(workflow_id="wf-goal")

        plan_call = mock_app_call.call_args_list[0]
        # Goal comes from checkpoint (falls back to entry if missing)
        assert plan_call[1]["goal"] == "Checkpoint goal"
