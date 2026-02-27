"""Tests for PlanCheckpoint model and save/load helpers.

Covers:
- PlanCheckpoint round-trip (save → load produces equal model_dump)
- _load_plan_checkpoint returns None for nonexistent file
- PlanCheckpoint with minimal fields (only required ones)
- architecture_revisions and review_iterations accumulate correctly
"""

from __future__ import annotations

import json
import os

import pytest

from swe_af.reasoners.schemas import PlanCheckpoint


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Return a temporary artifacts directory for checkpoint tests."""
    return str(tmp_path / "artifacts")


def _make_full_checkpoint(artifacts_dir: str) -> PlanCheckpoint:
    """Create a PlanCheckpoint with all fields populated."""
    return PlanCheckpoint(
        workflow_id="abc123",
        goal="Build a REST API",
        repo_path="/tmp/test-repo",
        artifacts_dir=artifacts_dir,
        config={"key": "value"},
        workspace_manifest={"repos": ["main"]},
        phase="sprint_planner",
        prd={"validated_description": "A REST API", "must_have": ["endpoints"]},
        architecture={"summary": "Layered arch", "components": []},
        review={"approved": True, "feedback": "Looks good"},
        architecture_revisions=[
            {"summary": "v1", "components": []},
            {"summary": "v2", "components": []},
        ],
        review_iterations=[
            {"approved": False, "feedback": "Needs work"},
            {"approved": True, "feedback": "Looks good"},
        ],
        review_loop_iteration=2,
        review_loop_sub_phase="tech_lead_done",
        sprint_plan={"issues": [{"name": "issue-1"}]},
        levels=[["issue-1"]],
        issue_writer_progress=["issue-1"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPlanCheckpointModel:
    """Tests for the PlanCheckpoint Pydantic model itself."""

    def test_minimal_fields(self):
        """PlanCheckpoint can be constructed with only required fields."""
        cp = PlanCheckpoint(
            workflow_id="wf1",
            goal="test goal",
            repo_path="/tmp/repo",
            artifacts_dir="/tmp/artifacts",
        )
        assert cp.workflow_id == "wf1"
        assert cp.goal == "test goal"
        assert cp.repo_path == "/tmp/repo"
        assert cp.artifacts_dir == "/tmp/artifacts"
        # Defaults
        assert cp.phase == ""
        assert cp.prd is None
        assert cp.architecture is None
        assert cp.review is None
        assert cp.config is None
        assert cp.workspace_manifest is None
        assert cp.architecture_revisions == []
        assert cp.review_iterations == []
        assert cp.review_loop_iteration == 0
        assert cp.review_loop_sub_phase == ""
        assert cp.sprint_plan is None
        assert cp.levels is None
        assert cp.issue_writer_progress == []

    def test_all_fields_populated(self, checkpoint_dir):
        """PlanCheckpoint with all fields populated retains values."""
        cp = _make_full_checkpoint(checkpoint_dir)
        assert cp.workflow_id == "abc123"
        assert cp.phase == "sprint_planner"
        assert len(cp.architecture_revisions) == 2
        assert len(cp.review_iterations) == 2
        assert cp.review_loop_iteration == 2
        assert cp.review_loop_sub_phase == "tech_lead_done"
        assert cp.levels == [["issue-1"]]

    def test_architecture_revisions_accumulate(self):
        """architecture_revisions list grows as revisions are appended."""
        cp = PlanCheckpoint(
            workflow_id="wf1",
            goal="test",
            repo_path="/tmp",
            artifacts_dir="/tmp/art",
        )
        assert cp.architecture_revisions == []
        cp.architecture_revisions.append({"summary": "v1"})
        cp.architecture_revisions.append({"summary": "v2"})
        cp.architecture_revisions.append({"summary": "v3"})
        assert len(cp.architecture_revisions) == 3
        assert cp.architecture_revisions[2]["summary"] == "v3"

    def test_review_iterations_accumulate(self):
        """review_iterations list grows as reviews are appended."""
        cp = PlanCheckpoint(
            workflow_id="wf1",
            goal="test",
            repo_path="/tmp",
            artifacts_dir="/tmp/art",
        )
        cp.review_iterations.append({"approved": False, "feedback": "fix it"})
        cp.review_iterations.append({"approved": True, "feedback": "ok"})
        assert len(cp.review_iterations) == 2
        assert cp.review_iterations[0]["approved"] is False
        assert cp.review_iterations[1]["approved"] is True


class TestPlanCheckpointSaveLoad:
    """Tests for _save_plan_checkpoint and _load_plan_checkpoint."""

    def test_round_trip(self, checkpoint_dir):
        """Save then load produces equal model_dump output."""
        from unittest.mock import patch

        from swe_af.reasoners.pipeline import (
            _load_plan_checkpoint,
            _save_plan_checkpoint,
        )

        original = _make_full_checkpoint(checkpoint_dir)

        with patch("swe_af.reasoners.pipeline.update_workflow"):
            _save_plan_checkpoint(original)

        loaded = _load_plan_checkpoint(checkpoint_dir)
        assert loaded is not None
        assert loaded.model_dump() == original.model_dump()

    def test_load_nonexistent_returns_none(self, checkpoint_dir):
        """_load_plan_checkpoint returns None when checkpoint file doesn't exist."""
        from swe_af.reasoners.pipeline import _load_plan_checkpoint

        result = _load_plan_checkpoint(checkpoint_dir)
        assert result is None

    def test_save_creates_directory(self, tmp_path):
        """_save_plan_checkpoint creates the plan directory if it doesn't exist."""
        from unittest.mock import patch

        from swe_af.reasoners.pipeline import _save_plan_checkpoint

        artifacts_dir = str(tmp_path / "deep" / "nested" / "artifacts")
        cp = PlanCheckpoint(
            workflow_id="wf1",
            goal="test",
            repo_path="/tmp",
            artifacts_dir=artifacts_dir,
        )

        with patch("swe_af.reasoners.pipeline.update_workflow"):
            _save_plan_checkpoint(cp)

        path = os.path.join(artifacts_dir, "plan", "checkpoint.json")
        assert os.path.exists(path)

    def test_save_is_valid_json(self, checkpoint_dir):
        """Saved checkpoint file contains valid JSON."""
        from unittest.mock import patch

        from swe_af.reasoners.pipeline import _save_plan_checkpoint

        cp = _make_full_checkpoint(checkpoint_dir)

        with patch("swe_af.reasoners.pipeline.update_workflow"):
            _save_plan_checkpoint(cp)

        path = os.path.join(checkpoint_dir, "plan", "checkpoint.json")
        with open(path) as f:
            data = json.load(f)
        assert data["workflow_id"] == "abc123"
        assert data["phase"] == "sprint_planner"

    def test_round_trip_minimal(self, checkpoint_dir):
        """Round-trip with minimal fields works correctly."""
        from unittest.mock import patch

        from swe_af.reasoners.pipeline import (
            _load_plan_checkpoint,
            _save_plan_checkpoint,
        )

        original = PlanCheckpoint(
            workflow_id="minimal",
            goal="simple test",
            repo_path="/tmp/repo",
            artifacts_dir=checkpoint_dir,
        )

        with patch("swe_af.reasoners.pipeline.update_workflow"):
            _save_plan_checkpoint(original)

        loaded = _load_plan_checkpoint(checkpoint_dir)
        assert loaded is not None
        assert loaded.model_dump() == original.model_dump()

    def test_overwrite_existing_checkpoint(self, checkpoint_dir):
        """Saving a checkpoint overwrites the previous one."""
        from unittest.mock import patch

        from swe_af.reasoners.pipeline import (
            _load_plan_checkpoint,
            _save_plan_checkpoint,
        )

        cp1 = PlanCheckpoint(
            workflow_id="wf1",
            goal="first",
            repo_path="/tmp",
            artifacts_dir=checkpoint_dir,
            phase="pm",
        )
        cp2 = PlanCheckpoint(
            workflow_id="wf1",
            goal="first",
            repo_path="/tmp",
            artifacts_dir=checkpoint_dir,
            phase="tech_lead",
        )

        with patch("swe_af.reasoners.pipeline.update_workflow"):
            _save_plan_checkpoint(cp1)
            _save_plan_checkpoint(cp2)

        loaded = _load_plan_checkpoint(checkpoint_dir)
        assert loaded is not None
        assert loaded.phase == "tech_lead"
