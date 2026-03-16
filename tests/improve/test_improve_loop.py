"""Integration tests for the improve loop orchestrator.

Tests cover the full improve() loop with mocked app.call to simulate scanner,
validator, and executor reasoners. Verifies state persistence, budget handling,
category filtering, and all loop exit conditions.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from swe_af.improve.app import (
    _ensure_gitignore,
    _load_state,
    _pick_next_improvement,
    _save_state,
    improve,
)
from swe_af.improve.schemas import (
    ImproveConfig,
    ImproveResult,
    ImprovementArea,
    ImprovementState,
)


@pytest.fixture
def temp_repo(tmp_path):
    """Create a temporary repository directory."""
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()
    return str(repo_path)


@pytest.fixture
def sample_improvement():
    """Return a sample ImprovementArea."""
    return ImprovementArea(
        id="test-improvement-1",
        category="test-coverage",
        title="Add missing tests",
        description="Add tests for user authentication",
        files=["auth.py"],
        priority=3,
        status="pending",
        found_by_run=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# State Persistence Tests
# ---------------------------------------------------------------------------


def test_load_state_empty(temp_repo):
    """Test loading state when no file exists."""
    state = _load_state(temp_repo)
    assert state.repo_path == temp_repo
    assert state.improvements == []
    assert state.last_scan_at is None
    assert state.runs == []


def test_save_and_load_state(temp_repo, sample_improvement):
    """Test saving and loading state with improvements."""
    state = ImprovementState(
        repo_path=temp_repo,
        improvements=[sample_improvement],
        last_scan_at=datetime.now(timezone.utc).isoformat(),
    )
    _save_state(temp_repo, state)

    loaded_state = _load_state(temp_repo)
    assert loaded_state.repo_path == temp_repo
    assert len(loaded_state.improvements) == 1
    assert loaded_state.improvements[0].id == "test-improvement-1"
    assert loaded_state.last_scan_at is not None


def test_save_state_atomic(temp_repo):
    """Test that state saving uses atomic writes with temp file."""
    state = ImprovementState(repo_path=temp_repo)
    _save_state(temp_repo, state)

    state_path = os.path.join(temp_repo, ".swe-af", "improvements.json")
    assert os.path.exists(state_path)

    # Verify no temp files left behind
    state_dir = os.path.join(temp_repo, ".swe-af")
    files = os.listdir(state_dir)
    assert not any(f.endswith(".tmp") for f in files)


def test_load_state_corrupted(temp_repo):
    """Test loading state with corrupted JSON file."""
    state_dir = os.path.join(temp_repo, ".swe-af")
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, "improvements.json")

    # Write invalid JSON
    with open(state_path, "w") as f:
        f.write("{invalid json")

    # Should return empty state without raising
    state = _load_state(temp_repo)
    assert state.repo_path == temp_repo
    assert state.improvements == []


# ---------------------------------------------------------------------------
# Gitignore Tests
# ---------------------------------------------------------------------------


def test_ensure_gitignore_creates_file(temp_repo):
    """Test creating .gitignore with .swe-af/ entry."""
    _ensure_gitignore(temp_repo)

    gitignore_path = os.path.join(temp_repo, ".gitignore")
    assert os.path.exists(gitignore_path)

    with open(gitignore_path, "r") as f:
        content = f.read()
    assert ".swe-af/" in content


def test_ensure_gitignore_appends_to_existing(temp_repo):
    """Test appending to existing .gitignore."""
    gitignore_path = os.path.join(temp_repo, ".gitignore")
    with open(gitignore_path, "w") as f:
        f.write("*.pyc\n__pycache__/\n")

    _ensure_gitignore(temp_repo)

    with open(gitignore_path, "r") as f:
        content = f.read()
    assert "*.pyc" in content
    assert ".swe-af/" in content


def test_ensure_gitignore_no_duplicate(temp_repo):
    """Test that .swe-af/ is not added twice."""
    gitignore_path = os.path.join(temp_repo, ".gitignore")
    with open(gitignore_path, "w") as f:
        f.write(".swe-af/\n")

    _ensure_gitignore(temp_repo)

    with open(gitignore_path, "r") as f:
        content = f.read()
    assert content.count(".swe-af") == 1


# ---------------------------------------------------------------------------
# Pick Next Improvement Tests
# ---------------------------------------------------------------------------


def test_pick_next_improvement_priority_order(temp_repo):
    """Test that improvements are picked in priority order."""
    imp1 = ImprovementArea(
        id="imp-1",
        category="test-coverage",
        title="Test 1",
        description="Desc 1",
        files=["a.py"],
        priority=5,
        status="pending",
        found_by_run="2024-01-01T00:00:00Z",
    )
    imp2 = ImprovementArea(
        id="imp-2",
        category="code-quality",
        title="Test 2",
        description="Desc 2",
        files=["b.py"],
        priority=2,
        status="pending",
        found_by_run="2024-01-01T00:00:00Z",
    )
    imp3 = ImprovementArea(
        id="imp-3",
        category="error-handling",
        title="Test 3",
        description="Desc 3",
        files=["c.py"],
        priority=8,
        status="pending",
        found_by_run="2024-01-01T00:00:00Z",
    )

    state = ImprovementState(
        repo_path=temp_repo,
        improvements=[imp1, imp2, imp3],
    )

    next_imp = _pick_next_improvement(state, None)
    assert next_imp is not None
    assert next_imp.id == "imp-2"  # Priority 2 is highest


def test_pick_next_improvement_category_filter(temp_repo):
    """Test filtering improvements by category."""
    imp1 = ImprovementArea(
        id="imp-1",
        category="test-coverage",
        title="Test 1",
        description="Desc 1",
        files=["a.py"],
        priority=1,
        status="pending",
        found_by_run="2024-01-01T00:00:00Z",
    )
    imp2 = ImprovementArea(
        id="imp-2",
        category="code-quality",
        title="Test 2",
        description="Desc 2",
        files=["b.py"],
        priority=2,
        status="pending",
        found_by_run="2024-01-01T00:00:00Z",
    )

    state = ImprovementState(
        repo_path=temp_repo,
        improvements=[imp1, imp2],
    )

    next_imp = _pick_next_improvement(state, ["code-quality"])
    assert next_imp is not None
    assert next_imp.id == "imp-2"


def test_pick_next_improvement_no_pending(temp_repo):
    """Test when no pending improvements exist."""
    imp1 = ImprovementArea(
        id="imp-1",
        category="test-coverage",
        title="Test 1",
        description="Desc 1",
        files=["a.py"],
        priority=1,
        status="completed",
        found_by_run="2024-01-01T00:00:00Z",
    )

    state = ImprovementState(
        repo_path=temp_repo,
        improvements=[imp1],
    )

    next_imp = _pick_next_improvement(state, None)
    assert next_imp is None


# ---------------------------------------------------------------------------
# Integration Tests with Mocked app.call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_improve_first_run_triggers_scan(temp_repo):
    """Test that first run with no state triggers scanner."""
    from swe_af.improve import app as improve_app_module

    mock_scan_result = {
        "new_areas": [
            {
                "id": "found-1",
                "category": "test-coverage",
                "title": "Add tests",
                "description": "Add tests for auth",
                "files": ["auth.py"],
                "priority": 3,
                "status": "pending",
                "found_by_run": datetime.now(timezone.utc).isoformat(),
            }
        ],
        "scan_depth_used": "normal",
        "summary": "Found 1 improvement",
        "files_analyzed": 5,
    }

    mock_validate_result = {
        "is_valid": True,
        "reason": "All files exist",
        "file_changes_detected": [],
    }

    mock_exec_result = {
        "success": True,
        "commit_sha": "abc123",
        "commit_message": "improve: add tests",
        "files_changed": ["tests/test_auth.py"],
        "new_findings": [],
        "error": "",
    }

    # Mock app.call at the module level
    with patch.object(improve_app_module.app, 'call', new=AsyncMock()) as mock_call:
        with patch.object(improve_app_module.app, 'note', new=MagicMock()):
            # Set up call responses in order
            mock_call.side_effect = [
                mock_scan_result,  # scan_for_improvements
                mock_validate_result,  # validate_improvement
                mock_exec_result,  # execute_improvement
            ]

            result = await improve(temp_repo, {"max_improvements": 1})

    result_obj = ImproveResult(**result)
    assert len(result_obj.improvements_found) == 1
    assert result_obj.improvements_found[0].id == "found-1"
    assert len(result_obj.improvements_completed) == 1
    assert result_obj.stopped_reason == "max_improvements_reached"


@pytest.mark.asyncio
async def test_improve_loads_existing_state(temp_repo, sample_improvement):
    """Test that improve loads existing state and picks pending improvement."""
    from swe_af.improve import app as improve_app_module

    # Create state with pending improvement
    state = ImprovementState(
        repo_path=temp_repo,
        improvements=[sample_improvement],
    )
    _save_state(temp_repo, state)

    mock_validate_result = {
        "is_valid": True,
        "reason": "Valid",
        "file_changes_detected": [],
    }

    mock_exec_result = {
        "success": True,
        "commit_sha": "def456",
        "commit_message": "improve: add tests",
        "files_changed": ["tests/test_auth.py"],
        "new_findings": [],
        "error": "",
    }

    with patch.object(improve_app_module.app, 'call', new=AsyncMock()) as mock_call:
        with patch.object(improve_app_module.app, 'note', new=MagicMock()):
            mock_call.side_effect = [
                mock_validate_result,  # validate_improvement
                mock_exec_result,  # execute_improvement
            ]

            result = await improve(temp_repo, {"max_improvements": 1})

    result_obj = ImproveResult(**result)
    assert len(result_obj.improvements_completed) == 1
    assert result_obj.improvements_completed[0].id == "test-improvement-1"
    assert result_obj.stopped_reason == "max_improvements_reached"


@pytest.mark.asyncio
async def test_improve_stale_detection(temp_repo):
    """Test that validator marking improvement stale causes it to be skipped."""
    from swe_af.improve import app as improve_app_module

    imp = ImprovementArea(
        id="stale-imp",
        category="test-coverage",
        title="Stale test",
        description="This will be marked stale",
        files=["deleted.py"],
        priority=1,
        status="pending",
        found_by_run=datetime.now(timezone.utc).isoformat(),
    )

    state = ImprovementState(repo_path=temp_repo, improvements=[imp])
    _save_state(temp_repo, state)

    mock_validate_result = {
        "is_valid": False,
        "reason": "File deleted",
        "file_changes_detected": ["deleted.py"],
    }

    with patch.object(improve_app_module.app, 'call', new=AsyncMock(return_value=mock_validate_result)):
        with patch.object(improve_app_module.app, 'note', new=MagicMock()):
            result = await improve(temp_repo, {"max_improvements": 1})

    result_obj = ImproveResult(**result)
    assert len(result_obj.improvements_skipped) == 1
    assert result_obj.improvements_skipped[0].id == "stale-imp"
    assert result_obj.improvements_skipped[0].status == "stale"
    assert len(result_obj.improvements_completed) == 0


@pytest.mark.asyncio
async def test_improve_budget_exhaustion(temp_repo):
    """Test that improve stops when budget is exhausted."""
    from swe_af.improve import app as improve_app_module

    # Create multiple pending improvements
    improvements = [
        ImprovementArea(
            id=f"imp-{i}",
            category="test-coverage",
            title=f"Test {i}",
            description=f"Desc {i}",
            files=[f"file{i}.py"],
            priority=i,
            status="pending",
            found_by_run=datetime.now(timezone.utc).isoformat(),
        )
        for i in range(5)
    ]

    state = ImprovementState(repo_path=temp_repo, improvements=improvements)
    _save_state(temp_repo, state)

    mock_validate_result = {
        "is_valid": True,
        "reason": "Valid",
        "file_changes_detected": [],
    }

    # Mock executor that takes 1 second per call
    async def slow_exec(*args, **kwargs):
        await AsyncMock()()  # Small async operation
        time.sleep(1)  # Simulate work
        return {
            "success": True,
            "commit_sha": "abc123",
            "commit_message": "improve: test",
            "files_changed": ["test.py"],
            "new_findings": [],
            "error": "",
        }

    call_count = [0]

    async def call_handler(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] % 2 == 1:  # Odd calls are validate
            return mock_validate_result
        else:  # Even calls are execute
            return await slow_exec(*args, **kwargs)

    with patch.object(improve_app_module.app, 'call', new=AsyncMock(side_effect=call_handler)):
        with patch.object(improve_app_module.app, 'note', new=MagicMock()):
            # Set very short budget (3 seconds)
            result = await improve(temp_repo, {"max_time_seconds": 3, "max_improvements": 0})

    result_obj = ImproveResult(**result)
    # Should complete some but not all due to budget
    assert len(result_obj.improvements_completed) < 5
    assert result_obj.stopped_reason == "budget_exhausted"
    assert result_obj.budget_remaining_seconds >= 0


@pytest.mark.asyncio
async def test_improve_max_improvements_limit(temp_repo):
    """Test that improve stops after max_improvements reached."""
    from swe_af.improve import app as improve_app_module

    # Create multiple pending improvements
    improvements = [
        ImprovementArea(
            id=f"imp-{i}",
            category="test-coverage",
            title=f"Test {i}",
            description=f"Desc {i}",
            files=[f"file{i}.py"],
            priority=i,
            status="pending",
            found_by_run=datetime.now(timezone.utc).isoformat(),
        )
        for i in range(5)
    ]

    state = ImprovementState(repo_path=temp_repo, improvements=improvements)
    _save_state(temp_repo, state)

    mock_validate_result = {
        "is_valid": True,
        "reason": "Valid",
        "file_changes_detected": [],
    }

    mock_exec_result = {
        "success": True,
        "commit_sha": "abc123",
        "commit_message": "improve: test",
        "files_changed": ["test.py"],
        "new_findings": [],
        "error": "",
    }

    # Alternate between validate and execute results
    results = []
    for _ in range(10):  # More than we need
        results.append(mock_validate_result)
        results.append(mock_exec_result)

    with patch.object(improve_app_module.app, 'call', new=AsyncMock(side_effect=results)):
        with patch.object(improve_app_module.app, 'note', new=MagicMock()):
            result = await improve(temp_repo, {"max_improvements": 2, "max_time_seconds": 300})

    result_obj = ImproveResult(**result)
    assert len(result_obj.improvements_completed) == 2
    assert result_obj.stopped_reason == "max_improvements_reached"


@pytest.mark.asyncio
async def test_improve_category_filter(temp_repo):
    """Test that only matching categories are picked."""
    from swe_af.improve import app as improve_app_module

    improvements = [
        ImprovementArea(
            id="imp-coverage",
            category="test-coverage",
            title="Test coverage",
            description="Add tests",
            files=["test.py"],
            priority=1,
            status="pending",
            found_by_run=datetime.now(timezone.utc).isoformat(),
        ),
        ImprovementArea(
            id="imp-quality",
            category="code-quality",
            title="Code quality",
            description="Refactor",
            files=["code.py"],
            priority=2,
            status="pending",
            found_by_run=datetime.now(timezone.utc).isoformat(),
        ),
    ]

    state = ImprovementState(repo_path=temp_repo, improvements=improvements)
    _save_state(temp_repo, state)

    mock_validate_result = {
        "is_valid": True,
        "reason": "Valid",
        "file_changes_detected": [],
    }

    mock_exec_result = {
        "success": True,
        "commit_sha": "abc123",
        "commit_message": "improve: code quality",
        "files_changed": ["code.py"],
        "new_findings": [],
        "error": "",
    }

    with patch.object(improve_app_module.app, 'call', new=AsyncMock(side_effect=[mock_validate_result, mock_exec_result])):
        with patch.object(improve_app_module.app, 'note', new=MagicMock()):
            result = await improve(
                temp_repo,
                {"categories": ["code-quality"], "max_improvements": 1}
            )

    result_obj = ImproveResult(**result)
    assert len(result_obj.improvements_completed) == 1
    assert result_obj.improvements_completed[0].id == "imp-quality"


@pytest.mark.asyncio
async def test_improve_new_findings_appended(temp_repo, sample_improvement):
    """Test that new findings from executor are appended to state."""
    from swe_af.improve import app as improve_app_module

    state = ImprovementState(repo_path=temp_repo, improvements=[sample_improvement])
    _save_state(temp_repo, state)

    mock_validate_result = {
        "is_valid": True,
        "reason": "Valid",
        "file_changes_detected": [],
    }

    mock_exec_result = {
        "success": True,
        "commit_sha": "abc123",
        "commit_message": "improve: add tests",
        "files_changed": ["test.py"],
        "new_findings": [
            {
                "id": "new-finding-1",
                "category": "error-handling",
                "title": "Add error handling",
                "description": "Found missing error handling",
                "files": ["error.py"],
                "priority": 4,
                "status": "pending",
                "found_by_run": datetime.now(timezone.utc).isoformat(),
            }
        ],
        "error": "",
    }

    with patch.object(improve_app_module.app, 'call', new=AsyncMock(side_effect=[mock_validate_result, mock_exec_result])):
        with patch.object(improve_app_module.app, 'note', new=MagicMock()):
            result = await improve(temp_repo, {"max_improvements": 1})

    result_obj = ImproveResult(**result)
    assert len(result_obj.improvements_found) == 1
    assert result_obj.improvements_found[0].id == "new-finding-1"
    assert result_obj.improvements_found[0].status == "pending"

    # Verify state was saved with new finding
    loaded_state = _load_state(temp_repo)
    assert len(loaded_state.improvements) == 2


@pytest.mark.asyncio
async def test_improve_invalid_repo_path():
    """Test improve with invalid repo_path returns error."""
    result = await improve("/nonexistent/path", {})

    result_obj = ImproveResult(**result)
    assert result_obj.stopped_reason == "error"
    assert "Invalid repo_path" in result_obj.summary
    assert len(result_obj.improvements_completed) == 0


@pytest.mark.asyncio
async def test_improve_timeout_calculation(temp_repo):
    """Test that per-improvement timeout is capped at 300 seconds."""
    from swe_af.improve import app as improve_app_module

    imp = ImprovementArea(
        id="test-imp",
        category="test-coverage",
        title="Test",
        description="Test",
        files=["test.py"],
        priority=1,
        status="pending",
        found_by_run=datetime.now(timezone.utc).isoformat(),
    )

    state = ImprovementState(repo_path=temp_repo, improvements=[imp])
    _save_state(temp_repo, state)

    mock_validate_result = {
        "is_valid": True,
        "reason": "Valid",
        "file_changes_detected": [],
    }

    mock_exec_result = {
        "success": True,
        "commit_sha": "abc123",
        "commit_message": "improve: test",
        "files_changed": ["test.py"],
        "new_findings": [],
        "error": "",
    }

    timeout_used = None

    async def capture_timeout(*args, **kwargs):
        nonlocal timeout_used
        # Extract timeout from call arguments
        timeout_used = kwargs.get("timeout_seconds")
        return mock_exec_result

    call_count = [0]

    async def call_handler(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:  # validate
            return mock_validate_result
        else:  # execute
            return await capture_timeout(*args, **kwargs)

    with patch.object(improve_app_module.app, 'call', new=AsyncMock(side_effect=call_handler)):
        with patch.object(improve_app_module.app, 'note', new=MagicMock()):
            # Set large budget (1000 seconds)
            result = await improve(temp_repo, {"max_time_seconds": 1000, "max_improvements": 1})

    # Verify timeout was capped at 300
    assert timeout_used is not None
    assert timeout_used <= 300


@pytest.mark.asyncio
async def test_improve_run_record_saved(temp_repo, sample_improvement):
    """Test that RunRecord is appended to state.runs."""
    from swe_af.improve import app as improve_app_module

    state = ImprovementState(repo_path=temp_repo, improvements=[sample_improvement])
    _save_state(temp_repo, state)

    mock_validate_result = {
        "is_valid": True,
        "reason": "Valid",
        "file_changes_detected": [],
    }

    mock_exec_result = {
        "success": True,
        "commit_sha": "abc123",
        "commit_message": "improve: test",
        "files_changed": ["test.py"],
        "new_findings": [],
        "error": "",
    }

    with patch.object(improve_app_module.app, 'call', new=AsyncMock(side_effect=[mock_validate_result, mock_exec_result])):
        with patch.object(improve_app_module.app, 'note', new=MagicMock()):
            result = await improve(temp_repo, {"max_improvements": 1})

    # Verify run record in result
    result_obj = ImproveResult(**result)
    assert result_obj.run_record.improvements_completed == 1
    assert result_obj.run_record.stopped_reason == "max_improvements_reached"

    # Verify run record saved to state
    loaded_state = _load_state(temp_repo)
    assert len(loaded_state.runs) == 1
    assert loaded_state.runs[0].improvements_completed == 1
    assert loaded_state.runs[0].stopped_reason == "max_improvements_reached"
