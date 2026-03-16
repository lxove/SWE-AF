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
