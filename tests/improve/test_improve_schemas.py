"""Unit tests for swe_af.improve.schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from swe_af.improve.schemas import (
    ExecutorResult,
    ImproveConfig,
    ImproveResult,
    ImprovementArea,
    ImprovementCategory,
    ImprovementState,
    RunRecord,
    ScanResult,
    ValidatorResult,
    improve_resolve_models,
)


# ---------------------------------------------------------------------------
# ImprovementArea Tests
# ---------------------------------------------------------------------------


def test_improvement_area_required_fields():
    """Test that ImprovementArea requires id, category, title, description, files, found_by_run."""
    area = ImprovementArea(
        id="test-improvement",
        category="test-coverage",
        title="Add missing tests",
        description="Add tests for user authentication",
        files=["auth.py"],
        found_by_run="2024-01-01T00:00:00Z",
    )
    assert area.id == "test-improvement"
    assert area.category == "test-coverage"
    assert area.title == "Add missing tests"
    assert area.description == "Add tests for user authentication"
    assert area.files == ["auth.py"]
    assert area.found_by_run == "2024-01-01T00:00:00Z"


def test_improvement_area_defaults():
    """Test ImprovementArea default values."""
    area = ImprovementArea(
        id="test",
        category="code-quality",
        title="Test",
        description="Test desc",
        files=["test.py"],
        found_by_run="2024-01-01T00:00:00Z",
    )
    assert area.priority == 5
    assert area.status == "pending"
    assert area.completed_by_run is None
    assert area.commit_sha is None
    assert area.notes == ""


def test_improvement_area_status_values():
    """Test ImprovementArea status accepts only valid values."""
    valid_statuses = ["pending", "in_progress", "completed", "stale", "skipped", "failed"]

    for status in valid_statuses:
        area = ImprovementArea(
            id="test",
            category="test-coverage",
            title="Test",
            description="Test",
            files=["test.py"],
            found_by_run="2024-01-01T00:00:00Z",
            status=status,
        )
        assert area.status == status

    # Invalid status should fail
    with pytest.raises(ValidationError):
        ImprovementArea(
            id="test",
            category="test-coverage",
            title="Test",
            description="Test",
            files=["test.py"],
            found_by_run="2024-01-01T00:00:00Z",
            status="invalid_status",
        )


def test_improvement_area_category_values():
    """Test ImprovementArea category accepts only valid values."""
    valid_categories = [
        "test-coverage",
        "code-quality",
        "error-handling",
        "consistency",
        "dead-code",
        "performance",
        "documentation",
    ]

    for category in valid_categories:
        area = ImprovementArea(
            id="test",
            category=category,
            title="Test",
            description="Test",
            files=["test.py"],
            found_by_run="2024-01-01T00:00:00Z",
        )
        assert area.category == category

    # Invalid category should fail
    with pytest.raises(ValidationError):
        ImprovementArea(
            id="test",
            category="invalid-category",
            title="Test",
            description="Test",
            files=["test.py"],
            found_by_run="2024-01-01T00:00:00Z",
        )


def test_improvement_area_extra_forbid():
    """Test ImprovementArea rejects unknown fields."""
    with pytest.raises(ValidationError) as exc_info:
        ImprovementArea(
            id="test",
            category="test-coverage",
            title="Test",
            description="Test",
            files=["test.py"],
            found_by_run="2024-01-01T00:00:00Z",
            unknown_field="should fail",
        )
    assert "extra" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# RunRecord Tests
# ---------------------------------------------------------------------------


def test_run_record_required_fields():
    """Test RunRecord requires started_at."""
    record = RunRecord(started_at="2024-01-01T00:00:00Z")
    assert record.started_at == "2024-01-01T00:00:00Z"


def test_run_record_defaults():
    """Test RunRecord default values."""
    record = RunRecord(started_at="2024-01-01T00:00:00Z")
    assert record.ended_at is None
    assert record.improvements_found == 0
    assert record.improvements_completed == 0
    assert record.improvements_skipped == 0
    assert record.budget_used_seconds == 0.0
    assert record.stopped_reason == ""


def test_run_record_integer_fields():
    """Test RunRecord tracks improvements as integers."""
    record = RunRecord(
        started_at="2024-01-01T00:00:00Z",
        improvements_found=5,
        improvements_completed=3,
        improvements_skipped=2,
    )
    assert isinstance(record.improvements_found, int)
    assert isinstance(record.improvements_completed, int)
    assert isinstance(record.improvements_skipped, int)
    assert record.improvements_found == 5
    assert record.improvements_completed == 3
    assert record.improvements_skipped == 2


def test_run_record_extra_forbid():
    """Test RunRecord rejects unknown fields."""
    with pytest.raises(ValidationError) as exc_info:
        RunRecord(
            started_at="2024-01-01T00:00:00Z",
            unknown_field="should fail",
        )
    assert "extra" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# ImprovementState Tests
# ---------------------------------------------------------------------------


def test_improvement_state_required_fields():
    """Test ImprovementState requires repo_path."""
    state = ImprovementState(repo_path="/path/to/repo")
    assert state.repo_path == "/path/to/repo"


def test_improvement_state_defaults():
    """Test ImprovementState default values."""
    state = ImprovementState(repo_path="/path/to/repo")
    assert state.improvements == []
    assert state.last_scan_at is None
    assert state.runs == []


def test_improvement_state_with_improvements():
    """Test ImprovementState with improvements list."""
    area = ImprovementArea(
        id="test",
        category="test-coverage",
        title="Test",
        description="Test",
        files=["test.py"],
        found_by_run="2024-01-01T00:00:00Z",
    )
    state = ImprovementState(
        repo_path="/path/to/repo",
        improvements=[area],
    )
    assert len(state.improvements) == 1
    assert state.improvements[0].id == "test"


def test_improvement_state_extra_forbid():
    """Test ImprovementState rejects unknown fields."""
    with pytest.raises(ValidationError) as exc_info:
        ImprovementState(
            repo_path="/path/to/repo",
            unknown_field="should fail",
        )
    assert "extra" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# ImproveConfig Tests
# ---------------------------------------------------------------------------


def test_improve_config_defaults():
    """Test ImproveConfig default values match spec."""
    config = ImproveConfig()
    assert config.runtime == "claude_code"
    assert config.models is None
    assert config.max_time_seconds == 3600
    assert config.max_improvements == 10
    assert config.permission_mode == ""
    assert config.scan_depth == "normal"
    assert config.categories is None
    assert config.agent_max_turns == 50


def test_improve_config_scan_depth_values():
    """Test ImproveConfig scan_depth accepts only valid values."""
    valid_depths = ["quick", "normal", "thorough"]

    for depth in valid_depths:
        config = ImproveConfig(scan_depth=depth)
        assert config.scan_depth == depth

    # Invalid depth should fail
    with pytest.raises(ValidationError):
        ImproveConfig(scan_depth="invalid")


def test_improve_config_runtime_values():
    """Test ImproveConfig runtime accepts only valid values."""
    config1 = ImproveConfig(runtime="claude_code")
    assert config1.runtime == "claude_code"

    config2 = ImproveConfig(runtime="open_code")
    assert config2.runtime == "open_code"

    # Invalid runtime should fail
    with pytest.raises(ValidationError):
        ImproveConfig(runtime="invalid_runtime")


def test_improve_config_categories_filter():
    """Test ImproveConfig categories filter."""
    config = ImproveConfig(categories=["test-coverage", "code-quality"])
    assert config.categories == ["test-coverage", "code-quality"]


def test_improve_config_extra_forbid():
    """Test ImproveConfig rejects unknown fields."""
    with pytest.raises(ValidationError) as exc_info:
        ImproveConfig(unknown_field="should fail")
    assert "extra" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# ScanResult Tests
# ---------------------------------------------------------------------------


def test_scan_result_required_fields():
    """Test ScanResult requires new_areas, scan_depth_used, summary."""
    result = ScanResult(
        new_areas=[],
        scan_depth_used="normal",
        summary="No improvements found",
    )
    assert result.new_areas == []
    assert result.scan_depth_used == "normal"
    assert result.summary == "No improvements found"


def test_scan_result_defaults():
    """Test ScanResult default values."""
    result = ScanResult(
        new_areas=[],
        scan_depth_used="quick",
        summary="Test",
    )
    assert result.files_analyzed == 0


def test_scan_result_extra_forbid():
    """Test ScanResult rejects unknown fields."""
    with pytest.raises(ValidationError) as exc_info:
        ScanResult(
            new_areas=[],
            scan_depth_used="normal",
            summary="Test",
            unknown_field="should fail",
        )
    assert "extra" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# ValidatorResult Tests
# ---------------------------------------------------------------------------


def test_validator_result_required_fields():
    """Test ValidatorResult requires is_valid and reason."""
    result = ValidatorResult(
        is_valid=True,
        reason="Improvement is still valid",
    )
    assert result.is_valid is True
    assert result.reason == "Improvement is still valid"


def test_validator_result_defaults():
    """Test ValidatorResult default values."""
    result = ValidatorResult(
        is_valid=False,
        reason="Files were deleted",
    )
    assert result.file_changes_detected == []


def test_validator_result_extra_forbid():
    """Test ValidatorResult rejects unknown fields."""
    with pytest.raises(ValidationError) as exc_info:
        ValidatorResult(
            is_valid=True,
            reason="Test",
            unknown_field="should fail",
        )
    assert "extra" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# ExecutorResult Tests
# ---------------------------------------------------------------------------


def test_executor_result_required_fields():
    """Test ExecutorResult requires success."""
    result = ExecutorResult(success=True)
    assert result.success is True


def test_executor_result_defaults():
    """Test ExecutorResult default values."""
    result = ExecutorResult(success=False)
    assert result.commit_sha is None
    assert result.commit_message == ""
    assert result.files_changed == []
    assert result.new_findings == []
    assert result.error == ""
    assert result.tests_passed is True
    assert result.verification_output == ""


def test_executor_result_extra_forbid():
    """Test ExecutorResult rejects unknown fields."""
    with pytest.raises(ValidationError) as exc_info:
        ExecutorResult(
            success=True,
            unknown_field="should fail",
        )
    assert "extra" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# ImproveResult Tests
# ---------------------------------------------------------------------------


def test_improve_result_required_fields():
    """Test ImproveResult requires all list fields, budget, stopped_reason, summary, run_record."""
    record = RunRecord(started_at="2024-01-01T00:00:00Z")
    result = ImproveResult(
        improvements_completed=[],
        improvements_found=[],
        improvements_skipped=[],
        improvements_failed=[],
        budget_remaining_seconds=100.0,
        stopped_reason="no_more_improvements",
        summary="Test completed",
        run_record=record,
    )
    assert result.improvements_completed == []
    assert result.improvements_found == []
    assert result.improvements_skipped == []
    assert result.improvements_failed == []
    assert result.budget_remaining_seconds == 100.0
    assert result.stopped_reason == "no_more_improvements"
    assert result.summary == "Test completed"
    assert result.run_record == record


def test_improve_result_stopped_reason_values():
    """Test ImproveResult stopped_reason accepts only valid values."""
    record = RunRecord(started_at="2024-01-01T00:00:00Z")
    valid_reasons = [
        "budget_exhausted",
        "max_improvements_reached",
        "no_more_improvements",
        "error",
    ]

    for reason in valid_reasons:
        result = ImproveResult(
            improvements_completed=[],
            improvements_found=[],
            improvements_skipped=[],
            improvements_failed=[],
            budget_remaining_seconds=0.0,
            stopped_reason=reason,
            summary="Test",
            run_record=record,
        )
        assert result.stopped_reason == reason

    # Invalid reason should fail
    with pytest.raises(ValidationError):
        ImproveResult(
            improvements_completed=[],
            improvements_found=[],
            improvements_skipped=[],
            improvements_failed=[],
            budget_remaining_seconds=0.0,
            stopped_reason="invalid_reason",
            summary="Test",
            run_record=record,
        )


def test_improve_result_extra_forbid():
    """Test ImproveResult rejects unknown fields."""
    record = RunRecord(started_at="2024-01-01T00:00:00Z")
    with pytest.raises(ValidationError) as exc_info:
        ImproveResult(
            improvements_completed=[],
            improvements_found=[],
            improvements_skipped=[],
            improvements_failed=[],
            budget_remaining_seconds=0.0,
            stopped_reason="no_more_improvements",
            summary="Test",
            run_record=record,
            unknown_field="should fail",
        )
    assert "extra" in str(exc_info.value).lower() or "unexpected" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# improve_resolve_models Tests
# ---------------------------------------------------------------------------


def test_improve_resolve_models_claude_code_default():
    """Test default model resolution for claude_code runtime."""
    config = ImproveConfig(runtime="claude_code")
    resolved = improve_resolve_models(config)

    assert resolved["scanner_model"] == "sonnet"
    assert resolved["executor_model"] == "sonnet"
    assert resolved["validator_model"] == "sonnet"


def test_improve_resolve_models_open_code_default():
    """Test default model resolution for open_code runtime."""
    config = ImproveConfig(runtime="open_code")
    resolved = improve_resolve_models(config)

    assert resolved["scanner_model"] == "qwen/qwen-2.5-coder-32b-instruct"
    assert resolved["executor_model"] == "qwen/qwen-2.5-coder-32b-instruct"
    assert resolved["validator_model"] == "qwen/qwen-2.5-coder-32b-instruct"


def test_improve_resolve_models_with_default_override():
    """Test models dict with 'default' key overrides all roles."""
    config = ImproveConfig(
        runtime="claude_code",
        models={"default": "custom-model"},
    )
    resolved = improve_resolve_models(config)

    assert resolved["scanner_model"] == "custom-model"
    assert resolved["executor_model"] == "custom-model"
    assert resolved["validator_model"] == "custom-model"


def test_improve_resolve_models_with_role_specific_override():
    """Test models dict with role-specific keys."""
    config = ImproveConfig(
        runtime="claude_code",
        models={
            "scanner": "scanner-model",
            "executor": "executor-model",
            "validator": "validator-model",
        },
    )
    resolved = improve_resolve_models(config)

    assert resolved["scanner_model"] == "scanner-model"
    assert resolved["executor_model"] == "executor-model"
    assert resolved["validator_model"] == "validator-model"


def test_improve_resolve_models_default_then_role_override():
    """Test resolution order: default is applied first, then role-specific overrides."""
    config = ImproveConfig(
        runtime="claude_code",
        models={
            "default": "base-model",
            "executor": "special-executor",
        },
    )
    resolved = improve_resolve_models(config)

    assert resolved["scanner_model"] == "base-model"
    assert resolved["executor_model"] == "special-executor"
    assert resolved["validator_model"] == "base-model"


def test_improve_resolve_models_unknown_key_raises():
    """Test that unknown keys in models dict raise ValueError."""
    config = ImproveConfig(
        runtime="claude_code",
        models={"unknown_role": "some-model"},
    )

    with pytest.raises(ValueError) as exc_info:
        improve_resolve_models(config)

    assert "Unknown role key" in str(exc_info.value)
    assert "unknown_role" in str(exc_info.value)


def test_improve_resolve_models_valid_keys():
    """Test that valid keys include 'default' and the three role names."""
    config = ImproveConfig(
        runtime="claude_code",
        models={
            "default": "default-model",
            "scanner": "scanner-model",
            "executor": "executor-model",
            "validator": "validator-model",
        },
    )
    # Should not raise
    resolved = improve_resolve_models(config)
    assert resolved["scanner_model"] == "scanner-model"
    assert resolved["executor_model"] == "executor-model"
    assert resolved["validator_model"] == "validator-model"


# ---------------------------------------------------------------------------
# Datetime/ISO 8601 Tests
# ---------------------------------------------------------------------------


def test_datetime_field_serialization():
    """Test that datetime fields serialize/deserialize correctly."""
    record = RunRecord(
        started_at="2024-01-01T12:30:45.123Z",
        ended_at="2024-01-01T13:45:30Z",
    )

    # Test serialization
    data = record.model_dump()
    assert data["started_at"] == "2024-01-01T12:30:45.123Z"
    assert data["ended_at"] == "2024-01-01T13:45:30Z"

    # Test deserialization
    record2 = RunRecord(**data)
    assert record2.started_at == "2024-01-01T12:30:45.123Z"
    assert record2.ended_at == "2024-01-01T13:45:30Z"


def test_improvement_area_datetime_fields():
    """Test ImprovementArea datetime field round-trip."""
    area = ImprovementArea(
        id="test",
        category="test-coverage",
        title="Test",
        description="Test",
        files=["test.py"],
        found_by_run="2024-01-01T10:00:00Z",
        completed_by_run="2024-01-01T11:00:00Z",
    )

    data = area.model_dump()
    assert data["found_by_run"] == "2024-01-01T10:00:00Z"
    assert data["completed_by_run"] == "2024-01-01T11:00:00Z"

    area2 = ImprovementArea(**data)
    assert area2.found_by_run == "2024-01-01T10:00:00Z"
    assert area2.completed_by_run == "2024-01-01T11:00:00Z"


# ---------------------------------------------------------------------------
# Additional Coverage Tests
# ---------------------------------------------------------------------------


def test_scan_result_scan_depth_used_values():
    """Test ScanResult scan_depth_used accepts only valid enum values."""
    valid_depths = ["quick", "normal", "thorough"]

    for depth in valid_depths:
        result = ScanResult(
            new_areas=[],
            scan_depth_used=depth,
            summary="Test",
        )
        assert result.scan_depth_used == depth

    # Invalid depth should fail
    with pytest.raises(ValidationError):
        ScanResult(
            new_areas=[],
            scan_depth_used="invalid_depth",
            summary="Test",
        )


def test_improve_config_models_dict_validation():
    """Test ImproveConfig models dict accepts valid role keys."""
    config = ImproveConfig(
        models={
            "scanner": "custom-scanner",
            "executor": "custom-executor",
            "validator": "custom-validator",
        }
    )
    assert config.models == {
        "scanner": "custom-scanner",
        "executor": "custom-executor",
        "validator": "custom-validator",
    }


def test_run_record_budget_used_seconds_as_float():
    """Test RunRecord budget_used_seconds is a float."""
    record = RunRecord(
        started_at="2024-01-01T00:00:00Z",
        budget_used_seconds=123.456,
    )
    assert isinstance(record.budget_used_seconds, float)
    assert record.budget_used_seconds == 123.456


@pytest.mark.parametrize(
    "status",
    ["pending", "in_progress", "completed", "stale", "skipped", "failed"],
)
def test_improvement_area_status_parametrized(status):
    """Parametrized test for ImprovementArea status values."""
    area = ImprovementArea(
        id="test",
        category="test-coverage",
        title="Test",
        description="Test",
        files=["test.py"],
        found_by_run="2024-01-01T00:00:00Z",
        status=status,
    )
    assert area.status == status


@pytest.mark.parametrize(
    "category",
    [
        "test-coverage",
        "code-quality",
        "error-handling",
        "consistency",
        "dead-code",
        "performance",
        "documentation",
    ],
)
def test_improvement_area_category_parametrized(category):
    """Parametrized test for ImprovementArea category values."""
    area = ImprovementArea(
        id="test",
        category=category,
        title="Test",
        description="Test",
        files=["test.py"],
        found_by_run="2024-01-01T00:00:00Z",
    )
    assert area.category == category


@pytest.mark.parametrize("scan_depth", ["quick", "normal", "thorough"])
def test_improve_config_scan_depth_parametrized(scan_depth):
    """Parametrized test for ImproveConfig scan_depth values."""
    config = ImproveConfig(scan_depth=scan_depth)
    assert config.scan_depth == scan_depth


@pytest.mark.parametrize(
    "stopped_reason",
    ["budget_exhausted", "max_improvements_reached", "no_more_improvements", "error"],
)
def test_improve_result_stopped_reason_parametrized(stopped_reason):
    """Parametrized test for ImproveResult stopped_reason values."""
    record = RunRecord(started_at="2024-01-01T00:00:00Z")
    result = ImproveResult(
        improvements_completed=[],
        improvements_found=[],
        improvements_skipped=[],
        improvements_failed=[],
        budget_remaining_seconds=0.0,
        stopped_reason=stopped_reason,
        summary="Test",
        run_record=record,
    )
    assert result.stopped_reason == stopped_reason


# ---------------------------------------------------------------------------
# JSON Serialization/Deserialization Tests
# ---------------------------------------------------------------------------


def test_improvement_area_json_roundtrip():
    """Test ImprovementArea serialization and deserialization."""
    area = ImprovementArea(
        id="test-improvement",
        category="test-coverage",
        title="Add missing tests",
        description="Add tests for user authentication",
        files=["auth.py", "utils.py"],
        priority=3,
        status="in_progress",
        found_by_run="2024-01-01T00:00:00Z",
        completed_by_run="2024-01-01T12:00:00Z",
        commit_sha="abc123",
        notes="Some notes",
    )

    # Serialize
    data = area.model_dump()
    assert data["id"] == "test-improvement"
    assert data["category"] == "test-coverage"
    assert data["priority"] == 3
    assert data["status"] == "in_progress"
    assert data["commit_sha"] == "abc123"

    # Deserialize
    area2 = ImprovementArea(**data)
    assert area2.id == area.id
    assert area2.category == area.category
    assert area2.title == area.title
    assert area2.description == area.description
    assert area2.files == area.files
    assert area2.priority == area.priority
    assert area2.status == area.status
    assert area2.notes == area.notes


def test_improvement_state_json_roundtrip():
    """Test ImprovementState serialization and deserialization."""
    area = ImprovementArea(
        id="test",
        category="test-coverage",
        title="Test",
        description="Test",
        files=["test.py"],
        found_by_run="2024-01-01T00:00:00Z",
    )
    record = RunRecord(
        started_at="2024-01-01T00:00:00Z",
        ended_at="2024-01-01T01:00:00Z",
        improvements_found=5,
        improvements_completed=3,
    )
    state = ImprovementState(
        repo_path="/path/to/repo",
        improvements=[area],
        last_scan_at="2024-01-01T00:00:00Z",
        runs=[record],
    )

    # Serialize
    data = state.model_dump()
    assert data["repo_path"] == "/path/to/repo"
    assert len(data["improvements"]) == 1
    assert len(data["runs"]) == 1

    # Deserialize
    state2 = ImprovementState(**data)
    assert state2.repo_path == state.repo_path
    assert len(state2.improvements) == 1
    assert state2.improvements[0].id == "test"
    assert len(state2.runs) == 1


def test_improve_config_json_roundtrip():
    """Test ImproveConfig serialization and deserialization."""
    config = ImproveConfig(
        runtime="open_code",
        models={"default": "custom-model"},
        max_time_seconds=7200,
        max_improvements=20,
        permission_mode="auto",
        scan_depth="thorough",
        categories=["test-coverage", "code-quality"],
        agent_max_turns=100,
    )

    # Serialize
    data = config.model_dump()
    assert data["runtime"] == "open_code"
    assert data["models"] == {"default": "custom-model"}
    assert data["max_time_seconds"] == 7200

    # Deserialize
    config2 = ImproveConfig(**data)
    assert config2.runtime == config.runtime
    assert config2.models == config.models
    assert config2.max_time_seconds == config.max_time_seconds
    assert config2.max_improvements == config.max_improvements
    assert config2.permission_mode == config.permission_mode
    assert config2.scan_depth == config.scan_depth
    assert config2.categories == config.categories
    assert config2.agent_max_turns == config.agent_max_turns


def test_scan_result_json_roundtrip():
    """Test ScanResult serialization and deserialization."""
    area = ImprovementArea(
        id="test",
        category="test-coverage",
        title="Test",
        description="Test",
        files=["test.py"],
        found_by_run="2024-01-01T00:00:00Z",
    )
    result = ScanResult(
        new_areas=[area],
        scan_depth_used="thorough",
        summary="Found 1 improvement",
        files_analyzed=42,
    )

    # Serialize
    data = result.model_dump()
    assert len(data["new_areas"]) == 1
    assert data["scan_depth_used"] == "thorough"
    assert data["files_analyzed"] == 42

    # Deserialize
    result2 = ScanResult(**data)
    assert len(result2.new_areas) == 1
    assert result2.new_areas[0].id == "test"
    assert result2.scan_depth_used == result.scan_depth_used
    assert result2.summary == result.summary
    assert result2.files_analyzed == result.files_analyzed


def test_validator_result_json_roundtrip():
    """Test ValidatorResult serialization and deserialization."""
    result = ValidatorResult(
        is_valid=True,
        reason="All files still exist",
        file_changes_detected=["file1.py", "file2.py"],
    )

    # Serialize
    data = result.model_dump()
    assert data["is_valid"] is True
    assert data["reason"] == "All files still exist"
    assert data["file_changes_detected"] == ["file1.py", "file2.py"]

    # Deserialize
    result2 = ValidatorResult(**data)
    assert result2.is_valid == result.is_valid
    assert result2.reason == result.reason
    assert result2.file_changes_detected == result.file_changes_detected


def test_executor_result_json_roundtrip():
    """Test ExecutorResult serialization and deserialization."""
    area = ImprovementArea(
        id="new-finding",
        category="code-quality",
        title="New Issue",
        description="Found during execution",
        files=["main.py"],
        found_by_run="2024-01-01T00:00:00Z",
    )
    result = ExecutorResult(
        success=True,
        commit_sha="def456",
        commit_message="Fixed the issue",
        files_changed=["auth.py", "utils.py"],
        new_findings=[area],
        error="",
        tests_passed=True,
        verification_output="All tests passed",
    )

    # Serialize
    data = result.model_dump()
    assert data["success"] is True
    assert data["commit_sha"] == "def456"
    assert len(data["new_findings"]) == 1

    # Deserialize
    result2 = ExecutorResult(**data)
    assert result2.success == result.success
    assert result2.commit_sha == result.commit_sha
    assert result2.commit_message == result.commit_message
    assert result2.files_changed == result.files_changed
    assert len(result2.new_findings) == 1
    assert result2.new_findings[0].id == "new-finding"
    assert result2.tests_passed == result.tests_passed
    assert result2.verification_output == result.verification_output


def test_improve_result_json_roundtrip():
    """Test ImproveResult serialization and deserialization."""
    area1 = ImprovementArea(
        id="completed",
        category="test-coverage",
        title="Test",
        description="Test",
        files=["test.py"],
        found_by_run="2024-01-01T00:00:00Z",
    )
    area2 = ImprovementArea(
        id="found",
        category="code-quality",
        title="Test",
        description="Test",
        files=["main.py"],
        found_by_run="2024-01-01T00:00:00Z",
    )
    record = RunRecord(
        started_at="2024-01-01T00:00:00Z",
        ended_at="2024-01-01T01:00:00Z",
        improvements_found=1,
        improvements_completed=1,
    )
    result = ImproveResult(
        improvements_completed=[area1],
        improvements_found=[area2],
        improvements_skipped=[],
        improvements_failed=[],
        budget_remaining_seconds=500.5,
        stopped_reason="budget_exhausted",
        summary="Completed 1 improvement",
        run_record=record,
    )

    # Serialize
    data = result.model_dump()
    assert len(data["improvements_completed"]) == 1
    assert len(data["improvements_found"]) == 1
    assert data["stopped_reason"] == "budget_exhausted"

    # Deserialize
    result2 = ImproveResult(**data)
    assert len(result2.improvements_completed) == 1
    assert result2.improvements_completed[0].id == "completed"
    assert len(result2.improvements_found) == 1
    assert result2.improvements_found[0].id == "found"
    assert result2.budget_remaining_seconds == result.budget_remaining_seconds
    assert result2.stopped_reason == result.stopped_reason
    assert result2.summary == result.summary


def test_run_record_json_roundtrip():
    """Test RunRecord serialization and deserialization."""
    record = RunRecord(
        started_at="2024-01-01T00:00:00Z",
        ended_at="2024-01-01T01:00:00Z",
        improvements_found=10,
        improvements_completed=8,
        improvements_skipped=1,
        budget_used_seconds=3456.789,
        stopped_reason="budget_exhausted",
    )

    # Serialize
    data = record.model_dump()
    assert data["started_at"] == "2024-01-01T00:00:00Z"
    assert data["improvements_found"] == 10
    assert data["budget_used_seconds"] == 3456.789

    # Deserialize
    record2 = RunRecord(**data)
    assert record2.started_at == record.started_at
    assert record2.ended_at == record.ended_at
    assert record2.improvements_found == record.improvements_found
    assert record2.improvements_completed == record.improvements_completed
    assert record2.improvements_skipped == record.improvements_skipped
    assert record2.budget_used_seconds == record.budget_used_seconds
    assert record2.stopped_reason == record.stopped_reason
