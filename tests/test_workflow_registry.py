"""Tests for the WorkflowRegistry module.

Covers:
- register_workflow creates entry with correct fields
- lookup_workflow returns entry for existing ID, None for unknown
- update_workflow modifies fields and updates timestamp
- list_workflows returns all entries
- Uses tmp_path to override the registry file location
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from swe_af.workflow_registry import (
    list_workflows,
    lookup_workflow,
    register_workflow,
    update_workflow,
)


# ---------------------------------------------------------------------------
# Fixture: redirect registry to tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _use_tmp_registry(tmp_path):
    """Redirect the registry file to a temporary directory."""
    registry_file = tmp_path / "workflows.json"
    with patch(
        "swe_af.workflow_registry._registry_path", return_value=registry_file
    ):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRegisterWorkflow:
    """Tests for register_workflow."""

    def test_creates_entry_with_correct_fields(self):
        """register_workflow creates an entry with all expected fields."""
        register_workflow("wf1", "/tmp/repo", ".artifacts", "Build an API")
        entry = lookup_workflow("wf1")
        assert entry is not None
        assert entry["build_id"] == "wf1"
        assert entry["repo_path"] == "/tmp/repo"
        assert entry["artifacts_dir"] == ".artifacts"
        assert entry["goal"] == "Build an API"
        assert entry["status"] == "planning"
        assert "created_at" in entry
        assert "updated_at" in entry

    def test_status_defaults_to_planning(self):
        """New workflows start with status='planning'."""
        register_workflow("wf2", "/tmp/repo", ".art", "test")
        entry = lookup_workflow("wf2")
        assert entry is not None
        assert entry["status"] == "planning"

    def test_timestamps_are_set(self):
        """created_at and updated_at are set on registration."""
        register_workflow("wf3", "/tmp", ".art", "test")
        entry = lookup_workflow("wf3")
        assert entry is not None
        assert entry["created_at"] == entry["updated_at"]
        # Timestamp should be an ISO format string
        assert "T" in entry["created_at"]

    def test_multiple_registrations(self):
        """Multiple workflows can be registered independently."""
        register_workflow("a", "/tmp/a", ".a", "goal a")
        register_workflow("b", "/tmp/b", ".b", "goal b")
        assert lookup_workflow("a") is not None
        assert lookup_workflow("b") is not None
        assert lookup_workflow("a")["goal"] == "goal a"  # type: ignore[index]
        assert lookup_workflow("b")["goal"] == "goal b"  # type: ignore[index]


class TestLookupWorkflow:
    """Tests for lookup_workflow."""

    def test_returns_entry_for_existing_id(self):
        """lookup_workflow returns the entry dict for a known ID."""
        register_workflow("exists", "/tmp", ".art", "test")
        result = lookup_workflow("exists")
        assert result is not None
        assert result["build_id"] == "exists"

    def test_returns_none_for_unknown_id(self):
        """lookup_workflow returns None for an unknown ID."""
        result = lookup_workflow("nonexistent")
        assert result is None

    def test_returns_none_on_empty_registry(self):
        """lookup_workflow returns None when registry has no entries."""
        result = lookup_workflow("anything")
        assert result is None


class TestUpdateWorkflow:
    """Tests for update_workflow."""

    def test_updates_status(self):
        """update_workflow changes the status field."""
        register_workflow("upd1", "/tmp", ".art", "test")
        update_workflow("upd1", status="executing")
        entry = lookup_workflow("upd1")
        assert entry is not None
        assert entry["status"] == "executing"

    def test_updates_multiple_fields(self):
        """update_workflow can change multiple fields at once."""
        register_workflow("upd2", "/tmp", ".art", "old goal")
        update_workflow("upd2", status="completed", goal="new goal")
        entry = lookup_workflow("upd2")
        assert entry is not None
        assert entry["status"] == "completed"
        assert entry["goal"] == "new goal"

    def test_updates_updated_at_timestamp(self):
        """update_workflow always refreshes updated_at."""
        register_workflow("upd3", "/tmp", ".art", "test")
        original = lookup_workflow("upd3")
        assert original is not None
        original_updated = original["updated_at"]

        time.sleep(0.01)  # Ensure time difference
        update_workflow("upd3", status="executing")

        updated = lookup_workflow("upd3")
        assert updated is not None
        assert updated["updated_at"] >= original_updated

    def test_ignores_disallowed_fields(self):
        """update_workflow silently ignores fields not in the allowed set."""
        register_workflow("upd4", "/tmp", ".art", "test")
        update_workflow("upd4", status="executing", bad_field="ignored")
        entry = lookup_workflow("upd4")
        assert entry is not None
        assert "bad_field" not in entry

    def test_noop_for_unknown_id(self):
        """update_workflow does nothing if build_id doesn't exist."""
        # Ensure registry is initialized with at least one entry
        register_workflow("existing", "/tmp", ".art", "test")
        update_workflow("ghost", status="failed")
        assert lookup_workflow("ghost") is None

    def test_preserves_created_at(self):
        """update_workflow does not change created_at."""
        register_workflow("upd5", "/tmp", ".art", "test")
        original = lookup_workflow("upd5")
        assert original is not None
        created = original["created_at"]

        update_workflow("upd5", status="completed")
        updated = lookup_workflow("upd5")
        assert updated is not None
        assert updated["created_at"] == created


class TestListWorkflows:
    """Tests for list_workflows."""

    def test_returns_all_entries(self):
        """list_workflows returns all registered workflows."""
        register_workflow("l1", "/tmp/1", ".a1", "goal 1")
        register_workflow("l2", "/tmp/2", ".a2", "goal 2")
        register_workflow("l3", "/tmp/3", ".a3", "goal 3")
        result = list_workflows()
        assert len(result) == 3
        ids = {e["build_id"] for e in result}
        assert ids == {"l1", "l2", "l3"}

    def test_returns_empty_list_when_no_entries(self):
        """list_workflows returns empty list when registry is empty."""
        result = list_workflows()
        assert result == []

    def test_entries_are_dicts(self):
        """Each entry from list_workflows is a dict with expected keys."""
        register_workflow("ld1", "/tmp", ".art", "test")
        result = list_workflows()
        assert len(result) == 1
        entry = result[0]
        assert isinstance(entry, dict)
        assert "build_id" in entry
        assert "status" in entry
        assert "created_at" in entry
