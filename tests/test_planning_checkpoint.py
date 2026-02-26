"""Tests for planning-phase checkpoint/resume support.

Covers:
- PlanningCheckpoint round-trip serialization
- _save/_load helpers
- _phase_done ordering logic
- goal_hash validation
- plan() resume skipping completed phases
- Partial issue-writer resume
- resume_build() with planning checkpoint fallback
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers (reused from test_planner_pipeline.py)
# ---------------------------------------------------------------------------


def _make_prd_dict() -> dict[str, Any]:
    return {
        "validated_description": "A test goal.",
        "acceptance_criteria": ["AC-1: does something"],
        "must_have": ["feature-x"],
        "nice_to_have": [],
        "out_of_scope": [],
        "assumptions": [],
        "risks": [],
    }


def _make_architecture_dict() -> dict[str, Any]:
    return {
        "summary": "Simple architecture.",
        "components": [
            {
                "name": "component-a",
                "responsibility": "Does A",
                "touches_files": ["a.py"],
                "depends_on": [],
            }
        ],
        "interfaces": ["interface-1"],
        "decisions": [
            {"decision": "Use Python", "rationale": "It is available."}
        ],
        "file_changes_overview": "Only a.py is changed.",
    }


def _make_review_approved_dict() -> dict[str, Any]:
    return {
        "approved": True,
        "feedback": "Looks good.",
        "scope_issues": [],
        "complexity_assessment": "appropriate",
        "summary": "Architecture approved.",
    }


def _make_sprint_result_dict(issue_name: str = "my-issue") -> dict[str, Any]:
    return {
        "issues": [
            {
                "name": issue_name,
                "title": "My Issue",
                "description": "Do the thing.",
                "acceptance_criteria": ["AC-1"],
                "depends_on": [],
                "provides": ["thing"],
                "estimated_complexity": "small",
                "files_to_create": ["thing.py"],
                "files_to_modify": [],
                "testing_strategy": "pytest",
                "sequence_number": None,
                "guidance": None,
            }
        ],
        "rationale": "This is the rationale.",
    }


def _make_issue_writer_result_dict() -> dict[str, Any]:
    return {"success": True, "path": "/tmp/test-issue.md"}


# ---------------------------------------------------------------------------
# Shared plan() invocation helper
# ---------------------------------------------------------------------------


async def _call_plan(tmp_path: str, **kwargs) -> dict:
    """Invoke plan() directly via _original_func to bypass the decorator."""
    import swe_af.app as _app_module

    real_fn = getattr(_app_module.plan, "_original_func", _app_module.plan)

    defaults = {
        "goal": "Build a test app",
        "repo_path": tmp_path,
        "artifacts_dir": ".artifacts",
        "additional_context": "",
        "max_review_iterations": 2,
        "pm_model": "sonnet",
        "architect_model": "sonnet",
        "tech_lead_model": "sonnet",
        "sprint_planner_model": "sonnet",
        "issue_writer_model": "sonnet",
        "permission_mode": "",
        "ai_provider": "claude",
        "resume_from_checkpoint": False,
    }
    defaults.update(kwargs)
    return await real_fn(**defaults)


async def _call_resume_build(tmp_path: str, **kwargs) -> dict:
    """Invoke resume_build() directly via _original_func."""
    import swe_af.app as _app_module

    real_fn = getattr(_app_module.resume_build, "_original_func", _app_module.resume_build)

    defaults = {
        "repo_path": tmp_path,
        "repo_url": "",
        "artifacts_dir": ".artifacts",
        "config": None,
        "git_config": None,
        "goal": "",
        "additional_context": "",
    }
    defaults.update(kwargs)
    return await real_fn(**defaults)


# ---------------------------------------------------------------------------
# Unit tests: PlanningCheckpoint round-trip
# ---------------------------------------------------------------------------


class TestPlanningCheckpointRoundTrip:
    def test_serialize_and_reconstruct(self):
        from swe_af.reasoners.schemas import PlanningCheckpoint, PlanningPhase

        ckpt = PlanningCheckpoint(
            completed_phase=PlanningPhase.ARCHITECT,
            prd=_make_prd_dict(),
            architecture=_make_architecture_dict(),
            review=None,
            artifacts_dir="/tmp/test",
            goal_hash="abc123",
            written_issue_names=["issue-1"],
        )
        dumped = ckpt.model_dump()
        raw = json.dumps(dumped, default=str)
        restored = PlanningCheckpoint(**json.loads(raw))

        assert restored.completed_phase == PlanningPhase.ARCHITECT
        assert restored.prd == _make_prd_dict()
        assert restored.architecture == _make_architecture_dict()
        assert restored.review is None
        assert restored.artifacts_dir == "/tmp/test"
        assert restored.goal_hash == "abc123"
        assert restored.written_issue_names == ["issue-1"]

    def test_all_phases_serializable(self):
        from swe_af.reasoners.schemas import PlanningCheckpoint, PlanningPhase

        for phase in PlanningPhase:
            ckpt = PlanningCheckpoint(completed_phase=phase, artifacts_dir="/tmp")
            raw = json.dumps(ckpt.model_dump(), default=str)
            restored = PlanningCheckpoint(**json.loads(raw))
            assert restored.completed_phase == phase


# ---------------------------------------------------------------------------
# Unit tests: _save/_load helpers
# ---------------------------------------------------------------------------


class TestSaveLoadHelpers:
    def test_save_and_load(self, tmp_path):
        from swe_af.reasoners.pipeline import (
            _load_planning_checkpoint,
            _save_planning_checkpoint,
        )
        from swe_af.reasoners.schemas import PlanningCheckpoint, PlanningPhase

        artifacts_dir = str(tmp_path / "artifacts")
        os.makedirs(os.path.join(artifacts_dir, "plan"), exist_ok=True)

        ckpt = PlanningCheckpoint(
            completed_phase=PlanningPhase.PM,
            prd=_make_prd_dict(),
            artifacts_dir=artifacts_dir,
            goal_hash="test_hash",
        )
        _save_planning_checkpoint(ckpt)

        # Verify file exists
        expected_path = os.path.join(artifacts_dir, "plan", "planning_checkpoint.json")
        assert os.path.exists(expected_path)

        # Load and verify
        loaded = _load_planning_checkpoint(artifacts_dir)
        assert loaded is not None
        assert loaded.completed_phase == PlanningPhase.PM
        assert loaded.prd == _make_prd_dict()
        assert loaded.goal_hash == "test_hash"

    def test_load_missing_returns_none(self, tmp_path):
        from swe_af.reasoners.pipeline import _load_planning_checkpoint

        loaded = _load_planning_checkpoint(str(tmp_path / "nonexistent"))
        assert loaded is None

    def test_checkpoint_path(self, tmp_path):
        from swe_af.reasoners.pipeline import _planning_checkpoint_path

        path = _planning_checkpoint_path(str(tmp_path))
        assert path == os.path.join(str(tmp_path), "plan", "planning_checkpoint.json")


# ---------------------------------------------------------------------------
# Unit tests: _phase_done
# ---------------------------------------------------------------------------


class TestPhaseDone:
    def test_none_checkpoint_always_false(self):
        from swe_af.app import _phase_done
        from swe_af.reasoners.schemas import PlanningPhase

        for phase in PlanningPhase:
            assert _phase_done(None, phase) is False

    def test_exact_phase_match(self):
        from swe_af.app import _phase_done
        from swe_af.reasoners.schemas import PlanningCheckpoint, PlanningPhase

        ckpt = PlanningCheckpoint(
            completed_phase=PlanningPhase.ARCHITECT,
            artifacts_dir="/tmp",
        )
        assert _phase_done(ckpt, PlanningPhase.ARCHITECT) is True

    def test_earlier_phases_done(self):
        from swe_af.app import _phase_done
        from swe_af.reasoners.schemas import PlanningCheckpoint, PlanningPhase

        ckpt = PlanningCheckpoint(
            completed_phase=PlanningPhase.SPRINT_PLANNER,
            artifacts_dir="/tmp",
        )
        assert _phase_done(ckpt, PlanningPhase.PM) is True
        assert _phase_done(ckpt, PlanningPhase.ARCHITECT) is True
        assert _phase_done(ckpt, PlanningPhase.TECH_LEAD_REVIEW) is True
        assert _phase_done(ckpt, PlanningPhase.SPRINT_PLANNER) is True

    def test_later_phases_not_done(self):
        from swe_af.app import _phase_done
        from swe_af.reasoners.schemas import PlanningCheckpoint, PlanningPhase

        ckpt = PlanningCheckpoint(
            completed_phase=PlanningPhase.ARCHITECT,
            artifacts_dir="/tmp",
        )
        assert _phase_done(ckpt, PlanningPhase.TECH_LEAD_REVIEW) is False
        assert _phase_done(ckpt, PlanningPhase.SPRINT_PLANNER) is False
        assert _phase_done(ckpt, PlanningPhase.ISSUE_WRITERS) is False
        assert _phase_done(ckpt, PlanningPhase.COMPLETE) is False

    def test_complete_covers_all(self):
        from swe_af.app import _phase_done
        from swe_af.reasoners.schemas import PlanningCheckpoint, PlanningPhase

        ckpt = PlanningCheckpoint(
            completed_phase=PlanningPhase.COMPLETE,
            artifacts_dir="/tmp",
        )
        for phase in PlanningPhase:
            assert _phase_done(ckpt, phase) is True


# ---------------------------------------------------------------------------
# Unit tests: goal_hash
# ---------------------------------------------------------------------------


class TestGoalHash:
    def test_matching_hash(self):
        from swe_af.app import _goal_hash

        goal = "Build a test app"
        h1 = _goal_hash(goal)
        h2 = _goal_hash(goal)
        assert h1 == h2
        assert len(h1) == 16

    def test_different_goals_different_hash(self):
        from swe_af.app import _goal_hash

        h1 = _goal_hash("Build app A")
        h2 = _goal_hash("Build app B")
        assert h1 != h2


# ---------------------------------------------------------------------------
# Integration: plan() resume skips completed phases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_resume_skips_completed_phases(mock_agent_ai, tmp_path):
    """Create a checkpoint at ARCHITECT phase, resume plan().

    PM and Architect should be skipped. Tech Lead, Sprint Planner, and
    Issue Writers should still be called.
    """
    from swe_af.reasoners.pipeline import _save_planning_checkpoint
    from swe_af.reasoners.schemas import PlanningCheckpoint, PlanningPhase
    from swe_af.app import _goal_hash

    prd = _make_prd_dict()
    arch = _make_architecture_dict()
    review = _make_review_approved_dict()
    sprint = _make_sprint_result_dict()
    issue_writer = _make_issue_writer_result_dict()

    # Create a checkpoint at ARCHITECT phase
    base = os.path.join(str(tmp_path), ".artifacts")
    os.makedirs(os.path.join(base, "plan"), exist_ok=True)

    ckpt = PlanningCheckpoint(
        completed_phase=PlanningPhase.ARCHITECT,
        prd=prd,
        architecture=arch,
        artifacts_dir=base,
        goal_hash=_goal_hash("Build a test app"),
    )
    _save_planning_checkpoint(ckpt)

    # Only Tech Lead, Sprint Planner, and Issue Writer should be called
    # (PM and Architect are skipped)
    mock_agent_ai.side_effect = [review, sprint, issue_writer]

    result = await _call_plan(str(tmp_path), resume_from_checkpoint=True)

    assert isinstance(result, dict)
    assert result["prd"] == prd
    assert result["architecture"] == arch
    assert result["review"] == review
    assert "levels" in result

    # Verify only 3 calls were made (tech_lead, sprint_planner, issue_writer)
    assert mock_agent_ai.call_count == 3

    # First call should be tech_lead (not PM or architect)
    first_call_target = mock_agent_ai.call_args_list[0][0][0]
    assert "run_tech_lead" in first_call_target


@pytest.mark.asyncio
async def test_plan_resume_skips_through_tech_lead(mock_agent_ai, tmp_path):
    """Create a checkpoint at TECH_LEAD_REVIEW. Sprint Planner and beyond should run."""
    from swe_af.reasoners.pipeline import _save_planning_checkpoint
    from swe_af.reasoners.schemas import PlanningCheckpoint, PlanningPhase
    from swe_af.app import _goal_hash

    prd = _make_prd_dict()
    arch = _make_architecture_dict()
    review = _make_review_approved_dict()
    sprint = _make_sprint_result_dict()
    issue_writer = _make_issue_writer_result_dict()

    base = os.path.join(str(tmp_path), ".artifacts")
    os.makedirs(os.path.join(base, "plan"), exist_ok=True)

    ckpt = PlanningCheckpoint(
        completed_phase=PlanningPhase.TECH_LEAD_REVIEW,
        prd=prd,
        architecture=arch,
        review=review,
        artifacts_dir=base,
        goal_hash=_goal_hash("Build a test app"),
    )
    _save_planning_checkpoint(ckpt)

    # Only Sprint Planner and Issue Writer should be called
    mock_agent_ai.side_effect = [sprint, issue_writer]

    result = await _call_plan(str(tmp_path), resume_from_checkpoint=True)

    assert isinstance(result, dict)
    assert mock_agent_ai.call_count == 2

    first_call_target = mock_agent_ai.call_args_list[0][0][0]
    assert "run_sprint_planner" in first_call_target


# ---------------------------------------------------------------------------
# Integration: partial issue-writer resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_resume_partial_issue_writers(mock_agent_ai, tmp_path):
    """Checkpoint at LEVEL_COMPUTATION with 1/2 issues written.

    Only the missing issue should be dispatched to issue_writer.
    """
    from swe_af.reasoners.pipeline import _save_planning_checkpoint
    from swe_af.reasoners.schemas import PlanningCheckpoint, PlanningPhase
    from swe_af.app import _goal_hash

    prd = _make_prd_dict()
    arch = _make_architecture_dict()
    review = _make_review_approved_dict()

    issues = [
        {
            "name": "issue-alpha",
            "title": "Issue Alpha",
            "description": "Alpha task.",
            "acceptance_criteria": ["AC-A"],
            "depends_on": [],
            "provides": [],
            "estimated_complexity": "small",
            "files_to_create": ["alpha.py"],
            "files_to_modify": [],
            "testing_strategy": "",
            "sequence_number": 1,
            "guidance": None,
        },
        {
            "name": "issue-beta",
            "title": "Issue Beta",
            "description": "Beta task.",
            "acceptance_criteria": ["AC-B"],
            "depends_on": [],
            "provides": [],
            "estimated_complexity": "small",
            "files_to_create": ["beta.py"],
            "files_to_modify": [],
            "testing_strategy": "",
            "sequence_number": 2,
            "guidance": None,
        },
    ]

    base = os.path.join(str(tmp_path), ".artifacts")
    os.makedirs(os.path.join(base, "plan"), exist_ok=True)

    ckpt = PlanningCheckpoint(
        completed_phase=PlanningPhase.LEVEL_COMPUTATION,
        prd=prd,
        architecture=arch,
        review=review,
        issues=issues,
        levels=[["issue-alpha", "issue-beta"]],
        file_conflicts=[],
        rationale="Two parallel issues.",
        written_issue_names=["issue-alpha"],  # alpha already written
        artifacts_dir=base,
        goal_hash=_goal_hash("Build a test app"),
    )
    _save_planning_checkpoint(ckpt)

    # Only one issue writer call for issue-beta
    issue_writer = _make_issue_writer_result_dict()
    mock_agent_ai.side_effect = [issue_writer]

    result = await _call_plan(str(tmp_path), resume_from_checkpoint=True)

    assert isinstance(result, dict)
    # Only 1 issue writer call (for issue-beta, since issue-alpha was already written)
    assert mock_agent_ai.call_count == 1

    call_kwargs = mock_agent_ai.call_args_list[0][1]
    assert call_kwargs["issue"]["name"] == "issue-beta"


# ---------------------------------------------------------------------------
# Integration: resume_build() with planning checkpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_build_with_planning_checkpoint(mock_agent_ai, tmp_path):
    """No execution checkpoint, but planning checkpoint exists.

    resume_build() should call plan(resume_from_checkpoint=True) then execute().
    """
    from swe_af.reasoners.pipeline import _save_planning_checkpoint
    from swe_af.reasoners.schemas import PlanningCheckpoint, PlanningPhase
    from swe_af.app import _goal_hash

    prd = _make_prd_dict()
    arch = _make_architecture_dict()
    review = _make_review_approved_dict()

    base = os.path.join(str(tmp_path), ".artifacts")
    os.makedirs(os.path.join(base, "plan"), exist_ok=True)

    ckpt = PlanningCheckpoint(
        completed_phase=PlanningPhase.TECH_LEAD_REVIEW,
        prd=prd,
        architecture=arch,
        review=review,
        artifacts_dir=base,
        goal_hash=_goal_hash("Build a test app"),
    )
    _save_planning_checkpoint(ckpt)

    # resume_build calls app.call twice: once for plan(), once for execute()
    plan_result = {
        "prd": prd,
        "architecture": arch,
        "review": review,
        "issues": [],
        "levels": [],
        "file_conflicts": [],
        "artifacts_dir": base,
        "rationale": "test",
    }
    execute_result = {"success": True, "completed_issues": []}

    mock_agent_ai.side_effect = [plan_result, execute_result]

    result = await _call_resume_build(
        str(tmp_path),
        goal="Build a test app",
    )

    assert mock_agent_ai.call_count == 2

    # First call should be plan with resume_from_checkpoint=True
    first_call = mock_agent_ai.call_args_list[0]
    assert "plan" in first_call[0][0]
    assert first_call[1].get("resume_from_checkpoint") is True

    # Second call should be execute
    second_call = mock_agent_ai.call_args_list[1]
    assert "execute" in second_call[0][0]


@pytest.mark.asyncio
async def test_resume_build_execution_checkpoint_takes_priority(mock_agent_ai, tmp_path):
    """When both execution and planning checkpoints exist, execution wins."""
    from swe_af.reasoners.pipeline import _save_planning_checkpoint
    from swe_af.reasoners.schemas import PlanningCheckpoint, PlanningPhase
    from swe_af.app import _goal_hash

    base = os.path.join(str(tmp_path), ".artifacts")
    os.makedirs(os.path.join(base, "plan"), exist_ok=True)
    os.makedirs(os.path.join(base, "execution"), exist_ok=True)

    # Create planning checkpoint
    ckpt = PlanningCheckpoint(
        completed_phase=PlanningPhase.PM,
        prd=_make_prd_dict(),
        artifacts_dir=base,
        goal_hash=_goal_hash("Build a test app"),
    )
    _save_planning_checkpoint(ckpt)

    # Create execution checkpoint
    exec_ckpt = {
        "all_issues": [{"name": "issue-1"}],
        "levels": [["issue-1"]],
        "artifacts_dir": base,
        "original_plan_summary": "test",
    }
    exec_path = os.path.join(base, "execution", "checkpoint.json")
    with open(exec_path, "w") as f:
        json.dump(exec_ckpt, f)

    execute_result = {"success": True}
    mock_agent_ai.side_effect = [execute_result]

    result = await _call_resume_build(str(tmp_path))

    # Should call execute (not plan), since execution checkpoint takes priority
    assert mock_agent_ai.call_count == 1
    first_call = mock_agent_ai.call_args_list[0]
    assert "execute" in first_call[0][0]
    assert first_call[1].get("resume") is True


@pytest.mark.asyncio
async def test_resume_build_no_checkpoint_raises(mock_agent_ai, tmp_path):
    """When no checkpoint exists, resume_build() raises RuntimeError."""
    with pytest.raises(RuntimeError, match="No checkpoint found"):
        await _call_resume_build(str(tmp_path))


@pytest.mark.asyncio
async def test_resume_build_planning_checkpoint_requires_goal(mock_agent_ai, tmp_path):
    """When planning checkpoint exists but goal is empty, raises ValueError."""
    from swe_af.reasoners.pipeline import _save_planning_checkpoint
    from swe_af.reasoners.schemas import PlanningCheckpoint, PlanningPhase

    base = os.path.join(str(tmp_path), ".artifacts")
    os.makedirs(os.path.join(base, "plan"), exist_ok=True)

    ckpt = PlanningCheckpoint(
        completed_phase=PlanningPhase.PM,
        prd=_make_prd_dict(),
        artifacts_dir=base,
    )
    _save_planning_checkpoint(ckpt)

    with pytest.raises(ValueError, match="goal.*required"):
        await _call_resume_build(str(tmp_path), goal="")


# ---------------------------------------------------------------------------
# Integration: plan() without resume still works (no regression)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_without_resume_saves_checkpoints(mock_agent_ai, tmp_path):
    """A normal plan() run should save checkpoints along the way."""
    from swe_af.reasoners.pipeline import _load_planning_checkpoint
    from swe_af.reasoners.schemas import PlanningPhase

    prd = _make_prd_dict()
    arch = _make_architecture_dict()
    review = _make_review_approved_dict()
    sprint = _make_sprint_result_dict()
    issue_writer = _make_issue_writer_result_dict()

    mock_agent_ai.side_effect = [prd, arch, review, sprint, issue_writer]

    result = await _call_plan(str(tmp_path))

    assert isinstance(result, dict)

    # Verify final checkpoint was saved with COMPLETE phase
    base = os.path.join(str(tmp_path), ".artifacts")
    loaded = _load_planning_checkpoint(base)
    assert loaded is not None
    assert loaded.completed_phase == PlanningPhase.COMPLETE
    assert loaded.prd == prd
    assert loaded.architecture == arch
