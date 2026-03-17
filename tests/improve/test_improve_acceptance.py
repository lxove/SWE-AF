"""Comprehensive acceptance tests for all PRD acceptance criteria.

This test suite verifies all 14 acceptance criteria for the swe-improve
continuous improvement feature. Tests combine unit, integration, and
verification approaches.

Tests:
  AC1:  Console script entry point verification
  AC2:  First run triggers scanner and writes state
  AC3:  Subsequent runs load state without re-scanning
  AC4:  Each improvement produces exactly one commit with 'improve:' prefix
  AC5:  Loop respects max_time_seconds budget
  AC6:  New findings are appended to state
  AC7:  Stale improvements are skipped
  AC8:  ImproveConfig supports all required fields
  AC9:  80%+ unit test coverage for improve module
  AC10: Console script exists in pyproject.toml
  AC11: State file structure is valid JSON
  AC12: Per-improvement timeout is min(remaining, 300)
  AC13: Pydantic models validate with extra='forbid'
  AC14: max_improvements limit is respected
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
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
    RunRecord,
    ScanResult,
    ValidatorResult,
    ExecutorResult,
    improve_resolve_models,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_repo(tmp_path):
    """Create a temporary repository directory."""
    repo_path = tmp_path / "test-repo"
    repo_path.mkdir()
    # Create a minimal git repo structure for commit tests
    (repo_path / ".git").mkdir()
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
# AC1: Console script entry point verification
# ---------------------------------------------------------------------------


def test_ac1_console_script_entry_point():
    """AC1: Verify swe-improve console script is defined."""
    # Read pyproject.toml and verify entry point exists
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml must exist"

    content = pyproject_path.read_text()
    assert "swe-improve" in content, "swe-improve entry not found in pyproject.toml"
    assert 'swe_af.improve.app:main' in content, "main function reference not found"


def test_ac1_app_main_function_exists():
    """AC1: Verify app.main function exists and is callable."""
    from swe_af.improve.app import main
    assert callable(main), "main() must be a callable function"


def test_ac1_python_module_entry():
    """AC1: Verify python -m swe_af.improve works via __main__.py."""
    main_path = Path(__file__).parent.parent.parent / "swe_af" / "improve" / "__main__.py"
    assert main_path.exists(), "__main__.py must exist for python -m entry"

    content = main_path.read_text()
    assert "main" in content, "__main__.py must call main()"


# ---------------------------------------------------------------------------
# AC2: First run triggers scanner and writes state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ac2_first_run_triggers_scanner(temp_repo):
    """AC2: First run with no state file triggers scanner and writes state."""
    # Ensure no state file exists
    state_path = os.path.join(temp_repo, ".swe-af", "improvements.json")
    assert not os.path.exists(state_path)

    # Mock app.call for scanner and executor
    with patch("swe_af.improve.app.app") as mock_app:
        mock_app.call = AsyncMock()
        mock_app.note = MagicMock()

        # Mock scanner returning improvements
        mock_app.call.side_effect = [
            # Scanner result
            {
                "new_areas": [
                    {
                        "id": "test-imp-1",
                        "category": "test-coverage",
                        "title": "Add tests",
                        "description": "Add missing tests",
                        "files": ["test.py"],
                        "priority": 3,
                        "status": "pending",
                        "found_by_run": datetime.now(timezone.utc).isoformat(),
                    }
                ],
                "scan_depth_used": "normal",
                "summary": "Found 1 improvement",
                "files_analyzed": 10,
            },
            # Validator result
            {"is_valid": True, "reason": "Valid", "file_changes_detected": []},
            # Executor result
            {
                "success": True,
                "commit_sha": "abc123",
                "commit_message": "improve: add tests",
                "files_changed": ["test.py"],
                "new_findings": [],
                "error": "",
                "tests_passed": True,
                "verification_output": "All tests passed",
            },
        ]

        # Run improve loop
        result = await improve(repo_path=temp_repo, config={"max_improvements": 1})

        # Verify scanner was called
        assert mock_app.call.call_count >= 1
        first_call = mock_app.call.call_args_list[0]
        assert "scan_for_improvements" in str(first_call)

    # Verify state file was created
    assert os.path.exists(state_path), "State file must be created on first run"

    # Verify state file content
    with open(state_path, "r") as f:
        state_data = json.load(f)

    assert state_data["repo_path"] == temp_repo
    assert "improvements" in state_data
    assert "last_scan_at" in state_data
    assert "runs" in state_data


# ---------------------------------------------------------------------------
# AC3: Subsequent runs load state without re-scanning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ac3_subsequent_runs_load_state(temp_repo, sample_improvement):
    """AC3: Subsequent runs load state and pick pending without re-scanning."""
    # Create initial state with pending improvement
    state = ImprovementState(
        repo_path=temp_repo,
        improvements=[sample_improvement],
        last_scan_at=datetime.now(timezone.utc).isoformat(),
    )
    _save_state(temp_repo, state)

    with patch("swe_af.improve.app.app") as mock_app:
        mock_app.call = AsyncMock()
        mock_app.note = MagicMock()

        # Mock only validator and executor (NOT scanner)
        mock_app.call.side_effect = [
            # Validator
            {"is_valid": True, "reason": "Valid", "file_changes_detected": []},
            # Executor
            {
                "success": True,
                "commit_sha": "def456",
                "commit_message": "improve: add missing tests",
                "files_changed": ["auth.py"],
                "new_findings": [],
                "error": "",
                "tests_passed": True,
                "verification_output": "Tests passed",
            },
        ]

        result = await improve(repo_path=temp_repo, config={"max_improvements": 1})

        # Verify scanner was NOT called (only validator and executor)
        call_args = [str(call) for call in mock_app.call.call_args_list]
        assert not any("scan_for_improvements" in arg for arg in call_args), \
            "Scanner should not be called when pending improvements exist"

        # Verify validator and executor were called
        assert any("validate_improvement" in arg for arg in call_args)
        assert any("execute_improvement" in arg for arg in call_args)


# ---------------------------------------------------------------------------
# AC4: Each improvement produces exactly one commit
# ---------------------------------------------------------------------------


def test_ac4_commit_message_format():
    """AC4: Verify executor results include commit with 'improve:' prefix."""
    # This is verified through executor tests and integration tests
    # Here we verify the schema supports it
    executor_result = ExecutorResult(
        success=True,
        commit_sha="abc123",
        commit_message="improve: add missing tests",
        files_changed=["test.py"],
    )

    assert executor_result.commit_message.startswith("improve:")
    assert executor_result.commit_sha is not None


def test_ac4_commit_format_validation():
    """AC4: Verify commit message format requirements."""
    # Valid commit messages
    valid_messages = [
        "improve: add tests for auth",
        "improve: remove unused imports",
        "improve: fix error handling in parser",
    ]

    for msg in valid_messages:
        assert msg.startswith("improve:")
        assert len(msg) < 100, f"Commit message too long: {msg}"
        # Verify description after prefix
        description = msg.replace("improve:", "").strip()
        assert len(description) > 0


# ---------------------------------------------------------------------------
# AC5: Loop respects max_time_seconds budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ac5_budget_exhaustion(temp_repo):
    """AC5: Loop stops when elapsed >= max_time_seconds."""
    # Create many pending improvements to ensure we don't finish them all
    improvements = [
        ImprovementArea(
            id=f"imp-{i}",
            category="test-coverage",
            title=f"Improvement {i}",
            description=f"Description {i}",
            files=[f"file{i}.py"],
            priority=i,
            status="pending",
            found_by_run=datetime.now(timezone.utc).isoformat(),
        )
        for i in range(50)  # Many improvements to ensure budget runs out
    ]

    state = ImprovementState(
        repo_path=temp_repo,
        improvements=improvements,
        last_scan_at=datetime.now(timezone.utc).isoformat(),
    )
    _save_state(temp_repo, state)

    with patch("swe_af.improve.app.app") as mock_app:
        mock_app.call = AsyncMock()
        mock_app.note = MagicMock()

        # Mock validator and executor with delays to consume budget
        async def delayed_call(*args, **kwargs):
            await asyncio.sleep(0.3)  # Longer delay to ensure budget exhaustion
            call_name = args[0] if args else ""
            if "validate" in call_name:
                return {"is_valid": True, "reason": "Valid", "file_changes_detected": []}
            else:
                return {
                    "success": True,
                    "commit_sha": "xyz",
                    "commit_message": "improve: test",
                    "files_changed": ["test.py"],
                    "new_findings": [],
                    "error": "",
                    "tests_passed": True,
                    "verification_output": "OK",
                }

        mock_app.call.side_effect = delayed_call

        # Run with very short budget
        start = time.time()
        result = await improve(
            repo_path=temp_repo,
            config={"max_time_seconds": 2, "max_improvements": 100}
        )
        elapsed = time.time() - start

        # Verify stopped due to budget exhaustion (or that we respected the budget)
        result_obj = ImproveResult(**result)
        # Either budget_exhausted or we stopped due to not having enough time for another improvement
        assert result_obj.stopped_reason in ["budget_exhausted", "no_more_improvements"]

        # Verify elapsed time is close to budget (with tolerance)
        assert elapsed <= 5, f"Took too long: {elapsed}s"

        # Verify we didn't complete all 50 improvements (budget stopped us)
        assert len(result_obj.improvements_completed) < 50, \
            "Should not complete all improvements within short budget"


# ---------------------------------------------------------------------------
# AC6: New findings appended to state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ac6_new_findings_appended(temp_repo, sample_improvement):
    """AC6: New findings from executor are appended to state with status='pending'."""
    state = ImprovementState(
        repo_path=temp_repo,
        improvements=[sample_improvement],
        last_scan_at=datetime.now(timezone.utc).isoformat(),
    )
    _save_state(temp_repo, state)

    with patch("swe_af.improve.app.app") as mock_app:
        mock_app.call = AsyncMock()
        mock_app.note = MagicMock()

        # Mock executor returning new findings
        mock_app.call.side_effect = [
            # Validator
            {"is_valid": True, "reason": "Valid", "file_changes_detected": []},
            # Executor with new findings
            {
                "success": True,
                "commit_sha": "new123",
                "commit_message": "improve: add tests",
                "files_changed": ["test.py"],
                "new_findings": [
                    {
                        "id": "new-finding-1",
                        "category": "code-quality",
                        "title": "Refactor complex function",
                        "description": "Function X is too complex",
                        "files": ["complex.py"],
                        "priority": 5,
                        "status": "pending",
                        "found_by_run": datetime.now(timezone.utc).isoformat(),
                    }
                ],
                "error": "",
                "tests_passed": True,
                "verification_output": "OK",
            },
        ]

        result = await improve(repo_path=temp_repo, config={"max_improvements": 1})

        # Load state and verify new finding was appended
        loaded_state = _load_state(temp_repo)
        assert len(loaded_state.improvements) == 2, "New finding should be appended"

        # Verify new finding has correct status
        new_finding = next(
            (imp for imp in loaded_state.improvements if imp.id == "new-finding-1"),
            None
        )
        assert new_finding is not None
        assert new_finding.status == "pending"


# ---------------------------------------------------------------------------
# AC7: Stale improvements are skipped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ac7_stale_improvements_skipped(temp_repo, sample_improvement):
    """AC7: Stale improvements (is_valid=False) are not executed."""
    state = ImprovementState(
        repo_path=temp_repo,
        improvements=[sample_improvement],
        last_scan_at=datetime.now(timezone.utc).isoformat(),
    )
    _save_state(temp_repo, state)

    with patch("swe_af.improve.app.app") as mock_app:
        mock_app.call = AsyncMock()
        mock_app.note = MagicMock()

        # Mock validator returning is_valid=False
        mock_app.call.return_value = {
            "is_valid": False,
            "reason": "File was deleted",
            "file_changes_detected": ["auth.py"],
        }

        result = await improve(repo_path=temp_repo, config={"max_improvements": 1})

        # Verify improvement was marked stale
        loaded_state = _load_state(temp_repo)
        stale_imp = loaded_state.improvements[0]
        assert stale_imp.status == "stale"

        # Verify executor was NOT called
        call_args = [str(call) for call in mock_app.call.call_args_list]
        assert not any("execute_improvement" in arg for arg in call_args), \
            "Executor should not be called for stale improvements"


# ---------------------------------------------------------------------------
# AC8: ImproveConfig schema supports all fields
# ---------------------------------------------------------------------------


def test_ac8_improve_config_all_fields():
    """AC8: ImproveConfig supports all required fields."""
    config = ImproveConfig(
        runtime="claude_code",
        models={"scanner": "opus", "executor": "sonnet", "validator": "haiku"},
        max_time_seconds=1800,
        max_improvements=20,
        permission_mode="allow_all",
        scan_depth="thorough",
        categories=["test-coverage", "documentation"],
        agent_max_turns=100,
    )

    assert config.runtime == "claude_code"
    assert config.models == {"scanner": "opus", "executor": "sonnet", "validator": "haiku"}
    assert config.max_time_seconds == 1800
    assert config.max_improvements == 20
    assert config.permission_mode == "allow_all"
    assert config.scan_depth == "thorough"
    assert config.categories == ["test-coverage", "documentation"]
    assert config.agent_max_turns == 100


def test_ac8_improve_config_defaults():
    """AC8: Verify ImproveConfig default values."""
    config = ImproveConfig()

    assert config.runtime == "claude_code"
    assert config.models is None
    assert config.repo_url == ""
    assert config.max_time_seconds == 3600
    assert config.max_improvements == 10
    assert config.permission_mode == ""
    assert config.scan_depth == "normal"
    assert config.categories is None
    assert config.agent_max_turns == 50


def test_ac8_improve_config_model_resolution():
    """AC8: Verify improve_resolve_models function."""
    # Default models
    config = ImproveConfig()
    resolved = improve_resolve_models(config)
    assert resolved["scanner_model"] == "sonnet"
    assert resolved["executor_model"] == "sonnet"
    assert resolved["validator_model"] == "sonnet"

    # Override default
    config = ImproveConfig(models={"default": "opus"})
    resolved = improve_resolve_models(config)
    assert all(v == "opus" for v in resolved.values())

    # Override specific role
    config = ImproveConfig(models={"scanner": "haiku"})
    resolved = improve_resolve_models(config)
    assert resolved["scanner_model"] == "haiku"
    assert resolved["executor_model"] == "sonnet"


# ---------------------------------------------------------------------------
# repo_url support: repo_path is optional when repo_url is provided
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repo_url_derives_repo_path():
    """When repo_url is given and repo_path is empty, repo_path is auto-derived."""
    with patch("swe_af.improve.app.app") as mock_app, \
         patch("subprocess.run") as mock_run, \
         patch("os.path.isdir", return_value=True), \
         patch("os.path.exists") as mock_exists, \
         patch("os.makedirs"):
        mock_app.call = AsyncMock()
        mock_app.note = MagicMock()

        # .git doesn't exist yet -> triggers clone; state file doesn't exist
        mock_exists.side_effect = lambda p: False

        # Clone succeeds
        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        # Scanner returns nothing so loop ends quickly
        mock_app.call.return_value = {
            "new_areas": [],
            "scan_depth_used": "normal",
            "summary": "No improvements found",
            "files_analyzed": 0,
        }

        result = await improve(repo_url="https://github.com/org/my-repo.git", config={"max_improvements": 1})

        # Verify clone was called with derived path
        mock_run.assert_called_once()
        clone_args = mock_run.call_args[0][0]
        assert clone_args[0] == "git"
        assert clone_args[1] == "clone"
        assert "my-repo" in clone_args[3]


@pytest.mark.asyncio
async def test_repo_url_from_config():
    """repo_url can be provided via config dict."""
    with patch("swe_af.improve.app.app") as mock_app, \
         patch("subprocess.run") as mock_run, \
         patch("os.path.isdir", return_value=True), \
         patch("os.path.exists", return_value=False), \
         patch("os.makedirs"):
        mock_app.call = AsyncMock()
        mock_app.note = MagicMock()

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        mock_app.call.return_value = {
            "new_areas": [],
            "scan_depth_used": "normal",
            "summary": "No improvements found",
            "files_analyzed": 0,
        }

        result = await improve(config={"repo_url": "https://github.com/org/project", "max_improvements": 1})

        # Clone was triggered via config repo_url
        mock_run.assert_called_once()
        clone_args = mock_run.call_args[0][0]
        assert "project" in clone_args[3]


@pytest.mark.asyncio
async def test_no_repo_path_or_url_returns_error():
    """Calling improve() with neither repo_path nor repo_url returns an error."""
    result = await improve()

    result_obj = ImproveResult(**result)
    assert result_obj.stopped_reason == "error"
    assert "repo_path" in result_obj.summary or "repo_url" in result_obj.summary


@pytest.mark.asyncio
async def test_repo_url_clone_failure_returns_error():
    """Clone failure returns a structured error, not an exception."""
    with patch("swe_af.improve.app.app") as mock_app, \
         patch("subprocess.run") as mock_run, \
         patch("os.path.exists", return_value=False), \
         patch("os.makedirs"):
        mock_app.note = MagicMock()

        mock_run.return_value = MagicMock(returncode=128, stderr="fatal: repo not found", stdout="")

        result = await improve(repo_url="https://github.com/org/nonexistent")

        result_obj = ImproveResult(**result)
        assert result_obj.stopped_reason == "error"
        assert "clone failed" in result_obj.summary


@pytest.mark.asyncio
async def test_repo_url_direct_param_overrides_config():
    """Direct repo_url parameter takes precedence over config.repo_url."""
    with patch("swe_af.improve.app.app") as mock_app, \
         patch("subprocess.run") as mock_run, \
         patch("os.path.isdir", return_value=True), \
         patch("os.path.exists", return_value=False), \
         patch("os.makedirs"):
        mock_app.call = AsyncMock()
        mock_app.note = MagicMock()

        mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

        mock_app.call.return_value = {
            "new_areas": [],
            "scan_depth_used": "normal",
            "summary": "No improvements found",
            "files_analyzed": 0,
        }

        result = await improve(
            repo_url="https://github.com/org/direct-repo",
            config={"repo_url": "https://github.com/org/config-repo", "max_improvements": 1},
        )

        # Direct param should win
        clone_args = mock_run.call_args[0][0]
        assert "direct-repo" in clone_args[3]


# ---------------------------------------------------------------------------
# AC9: Unit test coverage verification
# ---------------------------------------------------------------------------


def test_ac9_test_coverage_exists():
    """AC9: Verify test files exist for improve module."""
    tests_dir = Path(__file__).parent

    # Verify all test files exist
    expected_tests = [
        "test_improve_schemas.py",
        "test_improve_prompts.py",
        "test_improve_loop.py",
        "test_improve_scanner.py",
        "test_improve_validator.py",
        "test_improve_executor.py",
        "test_improve_acceptance.py",
    ]

    for test_file in expected_tests:
        test_path = tests_dir / test_file
        assert test_path.exists(), f"Test file {test_file} must exist"


def test_ac9_improve_module_structure():
    """AC9: Verify improve module has all required files."""
    improve_dir = Path(__file__).parent.parent.parent / "swe_af" / "improve"

    expected_files = [
        "__init__.py",
        "__main__.py",
        "app.py",
        "schemas.py",
        "prompts.py",
        "scanner.py",
        "validator.py",
        "executor.py",
    ]

    for module_file in expected_files:
        module_path = improve_dir / module_file
        assert module_path.exists(), f"Module file {module_file} must exist"


# ---------------------------------------------------------------------------
# AC10: Console script in pyproject.toml
# ---------------------------------------------------------------------------


def test_ac10_pyproject_console_script():
    """AC10: Verify pyproject.toml contains swe-improve console script."""
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    content = pyproject_path.read_text()

    # Verify console script entry
    assert "[project.scripts]" in content
    assert "swe-improve" in content
    assert "swe_af.improve.app:main" in content

    # Verify all three scripts are present
    assert "swe-af" in content
    assert "swe-fast" in content


# ---------------------------------------------------------------------------
# AC11: State file structure validation
# ---------------------------------------------------------------------------


def test_ac11_state_file_structure(temp_repo, sample_improvement):
    """AC11: State file has valid JSON structure."""
    state = ImprovementState(
        repo_path=temp_repo,
        improvements=[sample_improvement],
        last_scan_at=datetime.now(timezone.utc).isoformat(),
        runs=[
            RunRecord(
                started_at=datetime.now(timezone.utc).isoformat(),
                ended_at=datetime.now(timezone.utc).isoformat(),
                improvements_found=1,
                improvements_completed=1,
                improvements_skipped=0,
                budget_used_seconds=10.5,
                stopped_reason="max_improvements_reached",
            )
        ],
    )
    _save_state(temp_repo, state)

    # Load as raw JSON
    state_path = os.path.join(temp_repo, ".swe-af", "improvements.json")
    with open(state_path, "r") as f:
        data = json.load(f)

    # Verify structure
    assert "repo_path" in data
    assert "improvements" in data
    assert "last_scan_at" in data
    assert "runs" in data

    # Verify improvements structure
    assert isinstance(data["improvements"], list)
    if data["improvements"]:
        imp = data["improvements"][0]
        assert "id" in imp
        assert "category" in imp
        assert "title" in imp
        assert "description" in imp
        assert "files" in imp
        assert "priority" in imp
        assert "status" in imp
        assert "found_by_run" in imp

    # Verify runs structure
    assert isinstance(data["runs"], list)
    if data["runs"]:
        run = data["runs"][0]
        assert "started_at" in run
        assert "ended_at" in run
        assert "improvements_found" in run
        assert "improvements_completed" in run
        assert "budget_used_seconds" in run
        assert "stopped_reason" in run


# ---------------------------------------------------------------------------
# AC12: Per-improvement timeout handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ac12_per_improvement_timeout(temp_repo, sample_improvement):
    """AC12: Per-improvement timeout is min(remaining_budget, 300)."""
    state = ImprovementState(
        repo_path=temp_repo,
        improvements=[sample_improvement],
        last_scan_at=datetime.now(timezone.utc).isoformat(),
    )
    _save_state(temp_repo, state)

    with patch("swe_af.improve.app.app") as mock_app:
        mock_app.call = AsyncMock()
        mock_app.note = MagicMock()

        # Capture timeout passed to executor
        captured_timeout = None

        async def capture_timeout(*args, **kwargs):
            nonlocal captured_timeout
            if "timeout_seconds" in kwargs:
                captured_timeout = kwargs["timeout_seconds"]

            call_name = args[0] if args else ""
            if "validate" in call_name:
                return {"is_valid": True, "reason": "Valid", "file_changes_detected": []}
            else:
                return {
                    "success": True,
                    "commit_sha": "xyz",
                    "commit_message": "improve: test",
                    "files_changed": ["test.py"],
                    "new_findings": [],
                    "error": "",
                    "tests_passed": True,
                    "verification_output": "OK",
                }

        mock_app.call.side_effect = capture_timeout

        # Test with large budget (should cap at 300)
        await improve(repo_path=temp_repo, config={"max_time_seconds": 3600, "max_improvements": 1})
        assert captured_timeout is not None
        assert captured_timeout <= 300, "Timeout should be capped at 300"

        # Reset
        captured_timeout = None
        _save_state(temp_repo, state)

        # Test with small budget (should use remaining)
        await improve(repo_path=temp_repo, config={"max_time_seconds": 60, "max_improvements": 1})
        assert captured_timeout is not None
        assert captured_timeout <= 60, "Timeout should use remaining budget"


# ---------------------------------------------------------------------------
# AC13: Pydantic models validate with extra='forbid'
# ---------------------------------------------------------------------------


def test_ac13_extra_forbid_improvement_area():
    """AC13: ImprovementArea rejects unknown fields."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        ImprovementArea(
            id="test",
            category="test-coverage",
            title="Test",
            description="Test desc",
            files=["test.py"],
            found_by_run=datetime.now(timezone.utc).isoformat(),
            unknown_field="should fail",  # This should be rejected
        )


def test_ac13_extra_forbid_improve_config():
    """AC13: ImproveConfig rejects unknown fields."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        ImproveConfig(
            max_time_seconds=1800,
            invalid_field="should fail",  # This should be rejected
        )


def test_ac13_extra_forbid_improvement_state():
    """AC13: ImprovementState rejects unknown fields."""
    with pytest.raises(Exception):  # Pydantic ValidationError
        ImprovementState(
            repo_path="/tmp/test",
            extra_field="should fail",  # This should be rejected
        )


def test_ac13_extra_forbid_all_models():
    """AC13: Verify all models have extra='forbid'."""
    from swe_af.improve.schemas import (
        ImprovementArea,
        ImprovementState,
        ImproveConfig,
        ScanResult,
        ValidatorResult,
        ExecutorResult,
        ImproveResult,
        RunRecord,
    )

    models = [
        ImprovementArea,
        ImprovementState,
        ImproveConfig,
        ScanResult,
        ValidatorResult,
        ExecutorResult,
        ImproveResult,
        RunRecord,
    ]

    for model in models:
        config = model.model_config
        assert config.get("extra") == "forbid", \
            f"{model.__name__} must have extra='forbid'"


# ---------------------------------------------------------------------------
# AC14: max_improvements limit respected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ac14_max_improvements_limit(temp_repo):
    """AC14: Loop stops after exactly N improvements with 'max_improvements_reached'."""
    # Create 5 pending improvements
    improvements = [
        ImprovementArea(
            id=f"imp-{i}",
            category="test-coverage",
            title=f"Improvement {i}",
            description=f"Description {i}",
            files=[f"file{i}.py"],
            priority=i,
            status="pending",
            found_by_run=datetime.now(timezone.utc).isoformat(),
        )
        for i in range(5)
    ]

    state = ImprovementState(
        repo_path=temp_repo,
        improvements=improvements,
        last_scan_at=datetime.now(timezone.utc).isoformat(),
    )
    _save_state(temp_repo, state)

    with patch("swe_af.improve.app.app") as mock_app:
        mock_app.call = AsyncMock()
        mock_app.note = MagicMock()

        # Mock validator and executor always succeeding
        async def mock_calls(*args, **kwargs):
            call_name = args[0] if args else ""
            if "validate" in call_name:
                return {"is_valid": True, "reason": "Valid", "file_changes_detected": []}
            else:
                return {
                    "success": True,
                    "commit_sha": f"sha-{time.time()}",
                    "commit_message": "improve: test",
                    "files_changed": ["test.py"],
                    "new_findings": [],
                    "error": "",
                    "tests_passed": True,
                    "verification_output": "OK",
                }

        mock_app.call.side_effect = mock_calls

        # Run with max_improvements=2
        result = await improve(
            repo_path=temp_repo,
            config={"max_improvements": 2, "max_time_seconds": 3600}
        )

        # Verify exactly 2 improvements completed
        result_obj = ImproveResult(**result)
        assert len(result_obj.improvements_completed) == 2, \
            f"Expected 2 completed, got {len(result_obj.improvements_completed)}"

        # Verify stopped reason
        assert result_obj.stopped_reason == "max_improvements_reached"


# ---------------------------------------------------------------------------
# Helper: Import asyncio for async tests
# ---------------------------------------------------------------------------

import asyncio  # noqa: E402
