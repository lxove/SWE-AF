"""Unit tests for swe_af.improve.validator module.

Tests cover all acceptance criteria:
- AC1: validate_improvement is decorated with @improve_router.reasoner()
- AC2: Function signature: improvement_area: dict, repo_path: str, config: dict -> dict
- AC3: Accepts ImproveConfig dict, resolves to validator_model via improve_resolve_models
- AC4: Creates AgentAI instance with correct provider, model, cwd, allowed_tools (READ, GLOB, GREP only)
- AC5: AgentAI max_turns hardcoded to 10
- AC6: Calls ai.run with VALIDATOR_SYSTEM_PROMPT and ValidatorResult output_schema
- AC7: On success: returns ValidatorResult.model_dump() as dict
- AC8: On failure: returns ValidatorResult(is_valid=True, reason='Validator failed...') fallback
- AC9: Uses improve_router.note() for instrumentation (start, complete, error tags with improvement ID)
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentfield import AgentRouter
from swe_af.improve.schemas import ValidatorResult


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _registered_names(router: AgentRouter) -> set[str]:
    """Get set of registered reasoner function names from a router."""
    return {r["func"].__name__ for r in router.reasoners}


def _run(coro):
    """Run an async coroutine synchronously for tests."""
    return asyncio.run(coro)


def _make_mock_response(parsed: ValidatorResult | None) -> MagicMock:
    """Create a mock AgentResponse with the given parsed result."""
    response = MagicMock()
    response.parsed = parsed
    return response


# ---------------------------------------------------------------------------
# AC1: validate_improvement is registered on improve_router
# ---------------------------------------------------------------------------


class TestValidateImprovementRegistered:
    """Test that validate_improvement is decorated with @improve_router.reasoner()."""

    def test_validate_improvement_is_registered_on_improve_router(self) -> None:
        """validate_improvement should be registered on improve_router."""
        from swe_af.improve.validator import validate_improvement  # noqa: F401
        from swe_af.improve import improve_router

        names = _registered_names(improve_router)
        assert "validate_improvement" in names, (
            f"validate_improvement not registered on improve_router. Found: {names}"
        )


# ---------------------------------------------------------------------------
# AC2: Function signature verification
# ---------------------------------------------------------------------------


class TestFunctionSignature:
    """Test that validate_improvement has the correct function signature."""

    def test_function_signature_matches_spec(self) -> None:
        """validate_improvement should accept improvement_area, repo_path, config."""
        from swe_af.improve.validator import validate_improvement

        sig = inspect.signature(validate_improvement)
        params = list(sig.parameters.keys())
        assert params == ["improvement_area", "repo_path", "config"], (
            f"Expected params ['improvement_area', 'repo_path', 'config'], got {params}"
        )

    def test_function_is_async(self) -> None:
        """validate_improvement should be an async function."""
        from swe_af.improve.validator import validate_improvement

        assert asyncio.iscoroutinefunction(validate_improvement), (
            "validate_improvement must be an async function"
        )


# ---------------------------------------------------------------------------
# AC3: ImproveConfig parsing and model resolution
# ---------------------------------------------------------------------------


class TestConfigParsingAndModelResolution:
    """Test that validate_improvement parses ImproveConfig and resolves validator_model."""

    def test_validator_model_resolved_from_config(self) -> None:
        """validate_improvement should resolve validator_model via improve_resolve_models."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Add missing tests",
            "description": "Add tests for auth module",
            "files": ["auth.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
            "models": {"validator": "haiku"},
        }

        result_validator = ValidatorResult(
            is_valid=True,
            reason="All files exist and issue is present",
            file_changes_detected=[],
        )
        mock_response = _make_mock_response(result_validator)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            result = _run(validate_improvement(
                improvement_area=improvement,
                repo_path="/test/repo",
                config=config,
            ))

            # Verify AgentAIConfig was called with resolved model
            config_arg = MockAgentAI.call_args[0][0]
            assert config_arg.model == "haiku", (
                f"Expected model='haiku', got {config_arg.model}"
            )

    def test_default_model_used_when_no_validator_override(self) -> None:
        """When no validator model specified, should use runtime default."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
        }

        result_validator = ValidatorResult(
            is_valid=True,
            reason="Valid",
            file_changes_detected=[],
        )
        mock_response = _make_mock_response(result_validator)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            _run(validate_improvement(
                improvement_area=improvement,
                repo_path="/test/repo",
                config=config,
            ))

            # Verify default model (sonnet for claude_code)
            config_arg = MockAgentAI.call_args[0][0]
            assert config_arg.model == "sonnet"


# ---------------------------------------------------------------------------
# AC4: AgentAI initialization with correct parameters
# ---------------------------------------------------------------------------


class TestAgentAIInitialization:
    """Test that AgentAI is created with correct provider, model, cwd, and allowed_tools."""

    def test_agent_ai_created_with_correct_parameters(self) -> None:
        """AgentAI should be initialized with provider, model, cwd from config."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
            "models": {"validator": "opus"},
        }

        result_validator = ValidatorResult(
            is_valid=True,
            reason="Valid",
            file_changes_detected=[],
        )
        mock_response = _make_mock_response(result_validator)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            _run(validate_improvement(
                improvement_area=improvement,
                repo_path="/test/repo",
                config=config,
            ))

            # Verify AgentAIConfig (claude_code runtime maps to "claude" provider)
            config_arg = MockAgentAI.call_args[0][0]
            assert config_arg.provider == "claude"
            assert config_arg.model == "opus"
            assert config_arg.cwd == "/test/repo"

    def test_agent_ai_provider_mapping_open_code(self) -> None:
        """open_code runtime should map to 'opencode' provider."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "open_code",
            "models": {"validator": "qwen/qwen-2.5-coder-32b-instruct"},
        }

        result_validator = ValidatorResult(
            is_valid=True,
            reason="Valid",
            file_changes_detected=[],
        )
        mock_response = _make_mock_response(result_validator)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            _run(validate_improvement(
                improvement_area=improvement,
                repo_path="/test/repo",
                config=config,
            ))

            # Verify open_code runtime maps to "opencode" provider
            config_arg = MockAgentAI.call_args[0][0]
            assert config_arg.provider == "opencode"

    def test_agent_ai_allowed_tools_read_glob_grep_only(self) -> None:
        """AgentAI should be initialized with READ, GLOB, GREP tools only (no BASH/WRITE)."""
        from swe_af.improve.validator import validate_improvement
        from swe_af.agent_ai.types import Tool

        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
        }

        result_validator = ValidatorResult(
            is_valid=True,
            reason="Valid",
            file_changes_detected=[],
        )
        mock_response = _make_mock_response(result_validator)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            _run(validate_improvement(
                improvement_area=improvement,
                repo_path="/test/repo",
                config=config,
            ))

            # Verify allowed_tools
            config_arg = MockAgentAI.call_args[0][0]
            assert config_arg.allowed_tools == [Tool.READ, Tool.GLOB, Tool.GREP], (
                f"Expected READ/GLOB/GREP only, got {config_arg.allowed_tools}"
            )
            # Ensure BASH and WRITE are NOT in allowed_tools
            assert Tool.BASH not in config_arg.allowed_tools
            assert Tool.WRITE not in config_arg.allowed_tools


# ---------------------------------------------------------------------------
# AC5: max_turns hardcoded to 10
# ---------------------------------------------------------------------------


class TestMaxTurnsConstraint:
    """Test that AgentAI max_turns is hardcoded to 10."""

    def test_max_turns_is_ten(self) -> None:
        """AgentAI should have max_turns=10 for quick validation."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
        }

        result_validator = ValidatorResult(
            is_valid=True,
            reason="Valid",
            file_changes_detected=[],
        )
        mock_response = _make_mock_response(result_validator)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            _run(validate_improvement(
                improvement_area=improvement,
                repo_path="/test/repo",
                config=config,
            ))

            # Verify max_turns
            config_arg = MockAgentAI.call_args[0][0]
            assert config_arg.max_turns == 10, (
                f"Expected max_turns=10, got {config_arg.max_turns}"
            )


# ---------------------------------------------------------------------------
# AC6: Calls ai.run with correct prompts and schema
# ---------------------------------------------------------------------------


class TestAgentAIRunCall:
    """Test that ai.run is called with VALIDATOR_SYSTEM_PROMPT and ValidatorResult schema."""

    def test_ai_run_called_with_validator_system_prompt(self) -> None:
        """ai.run should be called with VALIDATOR_SYSTEM_PROMPT."""
        from swe_af.improve.validator import validate_improvement
        from swe_af.improve.prompts import VALIDATOR_SYSTEM_PROMPT

        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
        }

        result_validator = ValidatorResult(
            is_valid=True,
            reason="Valid",
            file_changes_detected=[],
        )
        mock_response = _make_mock_response(result_validator)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            _run(validate_improvement(
                improvement_area=improvement,
                repo_path="/test/repo",
                config=config,
            ))

            # Verify ai.run was called
            instance.run.assert_called_once()
            call_kwargs = instance.run.call_args[1]
            assert call_kwargs["system_prompt"] == VALIDATOR_SYSTEM_PROMPT

    def test_ai_run_called_with_validator_result_schema(self) -> None:
        """ai.run should be called with ValidatorResult as output_schema."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
        }

        result_validator = ValidatorResult(
            is_valid=True,
            reason="Valid",
            file_changes_detected=[],
        )
        mock_response = _make_mock_response(result_validator)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            _run(validate_improvement(
                improvement_area=improvement,
                repo_path="/test/repo",
                config=config,
            ))

            # Verify output_schema
            call_kwargs = instance.run.call_args[1]
            assert call_kwargs["output_schema"].__name__ == "ValidatorResult"

    def test_task_prompt_built_with_improvement_and_repo_path(self) -> None:
        """Task prompt should be built with improvement and repo_path."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement-123",
            "category": "test-coverage",
            "title": "Add missing tests",
            "description": "Add tests for auth module",
            "files": ["auth.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
        }

        result_validator = ValidatorResult(
            is_valid=True,
            reason="Valid",
            file_changes_detected=[],
        )
        mock_response = _make_mock_response(result_validator)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            _run(validate_improvement(
                improvement_area=improvement,
                repo_path="/test/repo",
                config=config,
            ))

            # Verify task prompt contains improvement ID and repo path
            call_args = instance.run.call_args[0]
            task_prompt = call_args[0]
            assert "test-improvement-123" in task_prompt
            assert "/test/repo" in task_prompt


# ---------------------------------------------------------------------------
# AC7: Success path returns ValidatorResult.model_dump()
# ---------------------------------------------------------------------------


class TestSuccessPath:
    """Test that successful validation returns ValidatorResult.model_dump() as dict."""

    def test_success_returns_validator_result_dict(self) -> None:
        """On success, should return ValidatorResult.model_dump() as dict."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
        }

        result_validator = ValidatorResult(
            is_valid=True,
            reason="All files exist and issue is still present",
            file_changes_detected=["test.py"],
        )
        mock_response = _make_mock_response(result_validator)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            result = _run(validate_improvement(
                improvement_area=improvement,
                repo_path="/test/repo",
                config=config,
            ))

            # Verify result is a dict with expected fields
            assert isinstance(result, dict)
            assert result["is_valid"] is True
            assert result["reason"] == "All files exist and issue is still present"
            assert result["file_changes_detected"] == ["test.py"]

    def test_success_invalid_result_returns_correctly(self) -> None:
        """Should correctly return is_valid=False when improvement is stale."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
        }

        result_validator = ValidatorResult(
            is_valid=False,
            reason="File was deleted",
            file_changes_detected=["test.py"],
        )
        mock_response = _make_mock_response(result_validator)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            result = _run(validate_improvement(
                improvement_area=improvement,
                repo_path="/test/repo",
                config=config,
            ))

            assert result["is_valid"] is False
            assert result["reason"] == "File was deleted"


# ---------------------------------------------------------------------------
# AC8: Failure path returns is_valid=True fallback
# ---------------------------------------------------------------------------


class TestFailurePath:
    """Test that failures return ValidatorResult with is_valid=True fallback."""

    def test_agent_exception_returns_is_valid_true_fallback(self) -> None:
        """When AgentAI.run() raises exception, should return is_valid=True fallback."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
        }

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(side_effect=RuntimeError("LLM connection error"))
            MockAgentAI.return_value = instance

            result = _run(validate_improvement(
                improvement_area=improvement,
                repo_path="/test/repo",
                config=config,
            ))

            # Verify fallback result
            assert isinstance(result, dict)
            assert result["is_valid"] is True
            assert "Validator failed" in result["reason"]
            assert result["file_changes_detected"] == []

    def test_parsed_none_returns_is_valid_true_fallback(self) -> None:
        """When response.parsed is None, should return is_valid=True fallback."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
        }

        mock_response = _make_mock_response(None)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            result = _run(validate_improvement(
                improvement_area=improvement,
                repo_path="/test/repo",
                config=config,
            ))

            assert result["is_valid"] is True
            assert "Validator failed" in result["reason"]

    def test_config_parsing_error_returns_is_valid_true_fallback(self) -> None:
        """When config parsing fails, should return is_valid=True fallback."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        # Invalid config (extra='forbid' will reject this)
        config = {
            "runtime": "claude_code",
            "invalid_field": "should_fail",
        }

        result = _run(validate_improvement(
            improvement_area=improvement,
            repo_path="/test/repo",
            config=config,
        ))

        assert result["is_valid"] is True
        assert "Validator failed" in result["reason"]


# ---------------------------------------------------------------------------
# AC9: improve_router.note() instrumentation
# ---------------------------------------------------------------------------


class TestInstrumentation:
    """Test that improve_router.note() is called with correct tags and improvement ID."""

    def test_note_called_with_start_tag_and_improvement_id(self) -> None:
        """Should call improve_router.note() with 'start' tag and improvement ID."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement-xyz",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
        }

        result_validator = ValidatorResult(
            is_valid=True,
            reason="Valid",
            file_changes_detected=[],
        )
        mock_response = _make_mock_response(result_validator)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            with patch("swe_af.improve.validator._note") as mock_note:
                instance = MagicMock()
                instance.run = AsyncMock(return_value=mock_response)
                MockAgentAI.return_value = instance

                _run(validate_improvement(
                    improvement_area=improvement,
                    repo_path="/test/repo",
                    config=config,
                ))

                # Verify note called with start tag and improvement ID
                calls = [call for call in mock_note.call_args_list]
                assert len(calls) >= 2  # At least start and complete

                # Check start call
                start_call = calls[0]
                assert "test-improvement-xyz" in start_call[0][0]
                assert "validator" in start_call[1]["tags"]
                assert "start" in start_call[1]["tags"]
                assert "improvement_id:test-improvement-xyz" in start_call[1]["tags"]

    def test_note_called_with_complete_tag_on_success(self) -> None:
        """Should call improve_router.note() with 'complete' tag on success."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement-123",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
        }

        result_validator = ValidatorResult(
            is_valid=True,
            reason="Valid",
            file_changes_detected=[],
        )
        mock_response = _make_mock_response(result_validator)

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            with patch("swe_af.improve.validator._note") as mock_note:
                instance = MagicMock()
                instance.run = AsyncMock(return_value=mock_response)
                MockAgentAI.return_value = instance

                _run(validate_improvement(
                    improvement_area=improvement,
                    repo_path="/test/repo",
                    config=config,
                ))

                # Verify complete call
                calls = [call for call in mock_note.call_args_list]
                complete_call = calls[-1]
                assert "test-improvement-123" in complete_call[0][0]
                assert "validator" in complete_call[1]["tags"]
                assert "complete" in complete_call[1]["tags"]
                assert "improvement_id:test-improvement-123" in complete_call[1]["tags"]

    def test_note_called_with_error_tag_on_failure(self) -> None:
        """Should call improve_router.note() with 'error' tag on failure."""
        from swe_af.improve.validator import validate_improvement

        improvement = {
            "id": "test-improvement-error",
            "category": "test-coverage",
            "title": "Test",
            "description": "Test",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        config = {
            "runtime": "claude_code",
        }

        with patch("swe_af.improve.validator.AgentAI") as MockAgentAI:
            with patch("swe_af.improve.validator._note") as mock_note:
                instance = MagicMock()
                instance.run = AsyncMock(side_effect=RuntimeError("Error"))
                MockAgentAI.return_value = instance

                _run(validate_improvement(
                    improvement_area=improvement,
                    repo_path="/test/repo",
                    config=config,
                ))

                # Verify error call
                calls = [call for call in mock_note.call_args_list]
                error_call = calls[-1]
                assert "test-improvement-error" in error_call[0][0]
                assert "validator" in error_call[1]["tags"]
                assert "error" in error_call[1]["tags"]
                assert "improvement_id:test-improvement-error" in error_call[1]["tags"]
