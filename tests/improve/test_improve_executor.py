"""Unit tests for swe_af.improve.executor module.

Tests cover all acceptance criteria:
- AC1: execute_improvement is decorated with @improve_router.reasoner()
- AC2: Function signature: improvement_area: dict, repo_path: str, timeout_seconds: int, config: dict -> dict
- AC3: Accepts ImproveConfig dict, resolves to executor_model via improve_resolve_models
- AC4: Creates AgentAI instance with correct provider, model, cwd, allowed_tools (READ, WRITE, EDIT, BASH, GLOB, GREP)
- AC5: AgentAI max_turns respects config.agent_max_turns
- AC6: Wraps ai.run() call with asyncio.wait_for(coro, timeout=timeout_seconds)
- AC7: On asyncio.TimeoutError: returns ExecutorResult(success=False, error='Timed out after Xs')
- AC8: On success: returns ExecutorResult.model_dump() including commit_sha and new_findings list
- AC9: On other errors: returns ExecutorResult(success=False, error=str(e))
- AC10: Uses improve_router.note() for instrumentation (start, complete, timeout, error tags)
"""

from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentfield import AgentRouter
from swe_af.improve.schemas import ExecutorResult, ImprovementArea


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _registered_names(router: AgentRouter) -> set[str]:
    """Get set of registered reasoner function names from a router."""
    return {r["func"].__name__ for r in router.reasoners}


def _run(coro):
    """Run an async coroutine synchronously for tests."""
    return asyncio.run(coro)


def _make_mock_response(parsed: ExecutorResult | None) -> MagicMock:
    """Create a mock AgentResponse with the given parsed result."""
    response = MagicMock()
    response.parsed = parsed
    return response


# ---------------------------------------------------------------------------
# AC1: execute_improvement is registered on improve_router
# ---------------------------------------------------------------------------


class TestExecuteImprovementRegistered:
    """Test that execute_improvement is decorated with @improve_router.reasoner()."""

    def test_execute_improvement_is_registered_on_improve_router(self) -> None:
        """execute_improvement should be registered on improve_router."""
        from swe_af.improve.executor import execute_improvement  # noqa: F401
        from swe_af.improve import improve_router

        names = _registered_names(improve_router)
        assert "execute_improvement" in names, (
            f"execute_improvement not registered on improve_router. Found: {names}"
        )


# ---------------------------------------------------------------------------
# AC2: Function signature verification
# ---------------------------------------------------------------------------


class TestFunctionSignature:
    """Test that execute_improvement has the correct function signature."""

    def test_function_signature_matches_spec(self) -> None:
        """execute_improvement should accept improvement_area, repo_path, timeout_seconds, config."""
        from swe_af.improve.executor import execute_improvement

        sig = inspect.signature(execute_improvement)
        params = list(sig.parameters.keys())
        assert params == ["improvement_area", "repo_path", "timeout_seconds", "config"], (
            f"Expected params ['improvement_area', 'repo_path', 'timeout_seconds', 'config'], got {params}"
        )

    def test_function_is_async(self) -> None:
        """execute_improvement should be an async function."""
        from swe_af.improve.executor import execute_improvement

        assert asyncio.iscoroutinefunction(execute_improvement), (
            "execute_improvement must be an async function"
        )


# ---------------------------------------------------------------------------
# AC3: ImproveConfig parsing and model resolution
# ---------------------------------------------------------------------------


class TestConfigParsingAndModelResolution:
    """Test that execute_improvement correctly parses config and resolves models."""

    @patch("swe_af.improve.executor.AgentAI")
    def test_uses_improve_resolve_models(self, mock_agent_ai_class: MagicMock) -> None:
        """execute_improvement should use improve_resolve_models to get executor_model."""
        from swe_af.improve.executor import execute_improvement

        # Setup mock
        mock_ai_instance = MagicMock()
        mock_ai_instance.run = AsyncMock(
            return_value=_make_mock_response(
                ExecutorResult(success=True, commit_sha="abc123")
            )
        )
        mock_agent_ai_class.return_value = mock_ai_instance

        # Call execute_improvement with a specific runtime and models config
        config = {
            "runtime": "claude_code",
            "models": {"executor": "opus"},
            "agent_max_turns": 30,
        }
        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test Improvement",
            "description": "Test description",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }

        _run(execute_improvement(improvement, "/fake/repo", 120, config))

        # Verify AgentAI was instantiated with executor_model='opus'
        assert mock_agent_ai_class.call_count == 1
        agent_config = mock_agent_ai_class.call_args[1]["config"]
        assert agent_config.model == "opus", f"Expected model='opus', got {agent_config.model}"


# ---------------------------------------------------------------------------
# AC4: AgentAI instance creation with correct tools
# ---------------------------------------------------------------------------


class TestAgentAICreation:
    """Test that execute_improvement creates AgentAI with correct parameters."""

    @patch("swe_af.improve.executor.AgentAI")
    def test_creates_agent_ai_with_correct_provider_and_tools(self, mock_agent_ai_class: MagicMock) -> None:
        """execute_improvement should create AgentAI with provider, model, cwd, and 6 tools."""
        from swe_af.improve.executor import execute_improvement
        from swe_af.agent_ai.types import Tool

        # Setup mock
        mock_ai_instance = MagicMock()
        mock_ai_instance.run = AsyncMock(
            return_value=_make_mock_response(
                ExecutorResult(success=True, commit_sha="abc123")
            )
        )
        mock_agent_ai_class.return_value = mock_ai_instance

        config = {"runtime": "claude_code", "agent_max_turns": 50}
        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test Improvement",
            "description": "Test description",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }

        _run(execute_improvement(improvement, "/repo/path", 300, config))

        # Verify AgentAI instantiation
        assert mock_agent_ai_class.call_count == 1
        agent_config = mock_agent_ai_class.call_args[1]["config"]

        # Check provider mapping (claude_code -> claude)
        assert agent_config.provider == "claude", f"Expected provider='claude', got {agent_config.provider}"

        # Check cwd
        assert agent_config.cwd == "/repo/path", f"Expected cwd='/repo/path', got {agent_config.cwd}"

        # Check allowed_tools (should have READ, WRITE, EDIT, BASH, GLOB, GREP)
        expected_tools = {Tool.READ, Tool.WRITE, Tool.EDIT, Tool.BASH, Tool.GLOB, Tool.GREP}
        actual_tools = set(agent_config.allowed_tools)
        assert actual_tools == expected_tools, (
            f"Expected tools {expected_tools}, got {actual_tools}"
        )

    @patch("swe_af.improve.executor.AgentAI")
    def test_provider_mapping_open_code(self, mock_agent_ai_class: MagicMock) -> None:
        """execute_improvement should map 'open_code' runtime to 'opencode' provider."""
        from swe_af.improve.executor import execute_improvement

        # Setup mock
        mock_ai_instance = MagicMock()
        mock_ai_instance.run = AsyncMock(
            return_value=_make_mock_response(
                ExecutorResult(success=True, commit_sha="abc123")
            )
        )
        mock_agent_ai_class.return_value = mock_ai_instance

        config = {"runtime": "open_code", "agent_max_turns": 50}
        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test Improvement",
            "description": "Test description",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }

        _run(execute_improvement(improvement, "/repo/path", 300, config))

        # Verify provider
        agent_config = mock_agent_ai_class.call_args[1]["config"]
        assert agent_config.provider == "opencode", f"Expected provider='opencode', got {agent_config.provider}"


# ---------------------------------------------------------------------------
# AC5: AgentAI max_turns respects config.agent_max_turns
# ---------------------------------------------------------------------------


class TestMaxTurnsConfiguration:
    """Test that execute_improvement respects config.agent_max_turns."""

    @patch("swe_af.improve.executor.AgentAI")
    def test_respects_agent_max_turns(self, mock_agent_ai_class: MagicMock) -> None:
        """execute_improvement should set AgentAI max_turns from config.agent_max_turns."""
        from swe_af.improve.executor import execute_improvement

        # Setup mock
        mock_ai_instance = MagicMock()
        mock_ai_instance.run = AsyncMock(
            return_value=_make_mock_response(
                ExecutorResult(success=True, commit_sha="abc123")
            )
        )
        mock_agent_ai_class.return_value = mock_ai_instance

        config = {"runtime": "claude_code", "agent_max_turns": 42}
        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test Improvement",
            "description": "Test description",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }

        _run(execute_improvement(improvement, "/repo/path", 300, config))

        # Verify max_turns
        agent_config = mock_agent_ai_class.call_args[1]["config"]
        assert agent_config.max_turns == 42, f"Expected max_turns=42, got {agent_config.max_turns}"


# ---------------------------------------------------------------------------
# AC6: ai.run() wrapped with asyncio.wait_for
# ---------------------------------------------------------------------------


class TestTimeoutWrapping:
    """Test that execute_improvement wraps ai.run() with asyncio.wait_for."""

    @patch("swe_af.improve.executor.AgentAI")
    @patch("swe_af.improve.executor.asyncio.wait_for")
    def test_wraps_ai_run_with_asyncio_wait_for(
        self, mock_wait_for: AsyncMock, mock_agent_ai_class: MagicMock
    ) -> None:
        """execute_improvement should wrap ai.run() with asyncio.wait_for(timeout=timeout_seconds)."""
        from swe_af.improve.executor import execute_improvement

        # Setup mock
        mock_ai_instance = MagicMock()
        mock_ai_instance.run = AsyncMock()
        mock_agent_ai_class.return_value = mock_ai_instance

        mock_wait_for.return_value = _make_mock_response(
            ExecutorResult(success=True, commit_sha="abc123")
        )

        config = {"runtime": "claude_code", "agent_max_turns": 50}
        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test Improvement",
            "description": "Test description",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        timeout_seconds = 180

        _run(execute_improvement(improvement, "/repo/path", timeout_seconds, config))

        # Verify asyncio.wait_for was called with timeout
        assert mock_wait_for.call_count == 1
        _, kwargs = mock_wait_for.call_args
        assert kwargs.get("timeout") == timeout_seconds, (
            f"Expected timeout={timeout_seconds}, got {kwargs.get('timeout')}"
        )


# ---------------------------------------------------------------------------
# AC7: Timeout handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    """Test that execute_improvement handles asyncio.TimeoutError correctly."""

    @patch("swe_af.improve.executor.AgentAI")
    def test_returns_timeout_error_on_timeout(self, mock_agent_ai_class: MagicMock) -> None:
        """execute_improvement should return ExecutorResult with timeout error on asyncio.TimeoutError."""
        from swe_af.improve.executor import execute_improvement

        # Setup mock to raise TimeoutError
        mock_ai_instance = MagicMock()
        mock_ai_instance.run = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_agent_ai_class.return_value = mock_ai_instance

        config = {"runtime": "claude_code", "agent_max_turns": 50}
        improvement = {
            "id": "timeout-test",
            "category": "test-coverage",
            "title": "Timeout Test",
            "description": "Test timeout handling",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        timeout_seconds = 120

        result = _run(execute_improvement(improvement, "/repo/path", timeout_seconds, config))

        # Verify result structure
        assert result["success"] is False, "Expected success=False on timeout"
        assert "Timed out after 120s" in result["error"], (
            f"Expected timeout error message, got {result['error']}"
        )

    def test_logs_timeout_with_correct_tags(self) -> None:
        """execute_improvement should call _note() with 'timeout' tag on timeout."""
        from swe_af.improve.executor import execute_improvement

        config = {"runtime": "claude_code", "agent_max_turns": 50}
        improvement = {
            "id": "timeout-test",
            "category": "test-coverage",
            "title": "Timeout Test",
            "description": "Test timeout handling",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }

        with patch("swe_af.improve.executor.AgentAI") as mock_agent_ai_class:
            with patch("swe_af.improve.executor._note") as mock_note:
                # Setup mock to raise TimeoutError
                mock_ai_instance = MagicMock()
                mock_ai_instance.run = AsyncMock(side_effect=asyncio.TimeoutError())
                mock_agent_ai_class.return_value = mock_ai_instance

                _run(execute_improvement(improvement, "/repo/path", 120, config))

                # Verify _note was called with timeout tag
                note_calls = [call for call in mock_note.call_args_list]
                timeout_calls = [
                    call for call in note_calls
                    if call[1].get("tags") and "improve_executor" in call[1]["tags"] and "timeout" in call[1]["tags"]
                ]
                assert len(timeout_calls) > 0, "Expected _note() to be called with 'timeout' tag"


# ---------------------------------------------------------------------------
# AC8: Success path with commit_sha and new_findings
# ---------------------------------------------------------------------------


class TestSuccessPath:
    """Test that execute_improvement returns ExecutorResult.model_dump() on success."""

    @patch("swe_af.improve.executor.AgentAI")
    def test_returns_executor_result_on_success(self, mock_agent_ai_class: MagicMock) -> None:
        """execute_improvement should return ExecutorResult.model_dump() on success."""
        from swe_af.improve.executor import execute_improvement

        # Setup mock with successful result
        new_finding = ImprovementArea(
            id="new-finding",
            category="code-quality",
            title="New Finding",
            description="Found during execution",
            files=["new.py"],
            found_by_run="2024-01-01T00:00:00Z",
        )
        expected_result = ExecutorResult(
            success=True,
            commit_sha="abc123def456",
            commit_message="improve: add missing tests",
            files_changed=["test.py", "test_module.py"],
            new_findings=[new_finding],
            tests_passed=True,
        )

        mock_ai_instance = MagicMock()
        mock_ai_instance.run = AsyncMock(
            return_value=_make_mock_response(expected_result)
        )
        mock_agent_ai_class.return_value = mock_ai_instance

        config = {"runtime": "claude_code", "agent_max_turns": 50}
        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test Improvement",
            "description": "Test description",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }

        result = _run(execute_improvement(improvement, "/repo/path", 300, config))

        # Verify result structure
        assert result["success"] is True, "Expected success=True"
        assert result["commit_sha"] == "abc123def456", f"Expected commit_sha, got {result.get('commit_sha')}"
        assert result["commit_message"] == "improve: add missing tests"
        assert result["files_changed"] == ["test.py", "test_module.py"]
        assert len(result["new_findings"]) == 1, f"Expected 1 new_finding, got {len(result['new_findings'])}"
        assert result["new_findings"][0]["id"] == "new-finding"
        assert result["tests_passed"] is True

    @patch("swe_af.improve.executor.AgentAI")
    def test_passes_system_prompt_and_output_schema(self, mock_agent_ai_class: MagicMock) -> None:
        """execute_improvement should pass EXECUTOR_SYSTEM_PROMPT and ExecutorResult to ai.run()."""
        from swe_af.improve.executor import execute_improvement

        # Setup mock
        mock_ai_instance = MagicMock()
        mock_ai_instance.run = AsyncMock(
            return_value=_make_mock_response(
                ExecutorResult(success=True, commit_sha="abc123")
            )
        )
        mock_agent_ai_class.return_value = mock_ai_instance

        config = {"runtime": "claude_code", "agent_max_turns": 50}
        improvement = {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Test Improvement",
            "description": "Test description",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }

        _run(execute_improvement(improvement, "/repo/path", 300, config))

        # Verify ai.run was called with correct parameters
        assert mock_ai_instance.run.call_count == 1
        call_args = mock_ai_instance.run.call_args

        # Check that system_prompt is passed
        assert "system_prompt" in call_args[1], "Expected system_prompt kwarg"
        from swe_af.improve.prompts import EXECUTOR_SYSTEM_PROMPT
        assert call_args[1]["system_prompt"] == EXECUTOR_SYSTEM_PROMPT

        # Check that output_schema is ExecutorResult
        assert "output_schema" in call_args[1], "Expected output_schema kwarg"
        from swe_af.improve.schemas import ExecutorResult as ExpectedSchema
        assert call_args[1]["output_schema"] is ExpectedSchema


# ---------------------------------------------------------------------------
# AC9: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test that execute_improvement handles errors correctly."""

    @patch("swe_af.improve.executor.AgentAI")
    def test_returns_error_result_on_exception(self, mock_agent_ai_class: MagicMock) -> None:
        """execute_improvement should return ExecutorResult(success=False, error=str(e)) on exception."""
        from swe_af.improve.executor import execute_improvement

        # Setup mock to raise exception
        mock_ai_instance = MagicMock()
        mock_ai_instance.run = AsyncMock(side_effect=ValueError("Test error"))
        mock_agent_ai_class.return_value = mock_ai_instance

        config = {"runtime": "claude_code", "agent_max_turns": 50}
        improvement = {
            "id": "error-test",
            "category": "test-coverage",
            "title": "Error Test",
            "description": "Test error handling",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }

        result = _run(execute_improvement(improvement, "/repo/path", 300, config))

        # Verify result structure
        assert result["success"] is False, "Expected success=False on error"
        assert "Test error" in result["error"], f"Expected error message, got {result['error']}"

    @patch("swe_af.improve.executor.AgentAI")
    def test_returns_error_when_parsed_is_none(self, mock_agent_ai_class: MagicMock) -> None:
        """execute_improvement should return error when response.parsed is None."""
        from swe_af.improve.executor import execute_improvement

        # Setup mock with parsed=None
        mock_ai_instance = MagicMock()
        mock_ai_instance.run = AsyncMock(
            return_value=_make_mock_response(None)
        )
        mock_agent_ai_class.return_value = mock_ai_instance

        config = {"runtime": "claude_code", "agent_max_turns": 50}
        improvement = {
            "id": "error-test",
            "category": "test-coverage",
            "title": "Error Test",
            "description": "Test error handling",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }

        result = _run(execute_improvement(improvement, "/repo/path", 300, config))

        # Verify result structure
        assert result["success"] is False, "Expected success=False when parsed is None"
        assert "unparseable" in result["error"].lower(), f"Expected unparseable error, got {result['error']}"


# ---------------------------------------------------------------------------
# AC10: Instrumentation with improve_router.note()
# ---------------------------------------------------------------------------


class TestInstrumentation:
    """Test that execute_improvement uses improve_router.note() correctly."""

    def test_logs_start_and_complete_tags(self) -> None:
        """execute_improvement should call _note() with start and complete tags."""
        from swe_af.improve.executor import execute_improvement

        config = {"runtime": "claude_code", "agent_max_turns": 50}
        improvement = {
            "id": "instrumentation-test",
            "category": "test-coverage",
            "title": "Instrumentation Test",
            "description": "Test instrumentation",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }

        with patch("swe_af.improve.executor.AgentAI") as mock_agent_ai_class:
            with patch("swe_af.improve.executor._note") as mock_note:
                # Setup mock
                mock_ai_instance = MagicMock()
                mock_ai_instance.run = AsyncMock(
                    return_value=_make_mock_response(
                        ExecutorResult(success=True, commit_sha="abc123")
                    )
                )
                mock_agent_ai_class.return_value = mock_ai_instance

                _run(execute_improvement(improvement, "/repo/path", 300, config))

                # Verify _note was called
                note_calls = [call for call in mock_note.call_args_list]
                assert len(note_calls) >= 2, "Expected at least 2 calls to _note()"

                # Check for start tag
                start_calls = [
                    call for call in note_calls
                    if call[1].get("tags") and "start" in call[1]["tags"]
                ]
                assert len(start_calls) > 0, "Expected _note() to be called with 'start' tag"

                # Check for complete tag
                complete_calls = [
                    call for call in note_calls
                    if call[1].get("tags") and "complete" in call[1]["tags"]
                ]
                assert len(complete_calls) > 0, "Expected _note() to be called with 'complete' tag"

    def test_logs_error_tag_on_failure(self) -> None:
        """execute_improvement should call _note() with error tag on failure."""
        from swe_af.improve.executor import execute_improvement

        config = {"runtime": "claude_code", "agent_max_turns": 50}
        improvement = {
            "id": "error-test",
            "category": "test-coverage",
            "title": "Error Test",
            "description": "Test error handling",
            "files": ["test.py"],
            "found_by_run": "2024-01-01T00:00:00Z",
        }

        with patch("swe_af.improve.executor.AgentAI") as mock_agent_ai_class:
            with patch("swe_af.improve.executor._note") as mock_note:
                # Setup mock to raise exception
                mock_ai_instance = MagicMock()
                mock_ai_instance.run = AsyncMock(side_effect=RuntimeError("Test error"))
                mock_agent_ai_class.return_value = mock_ai_instance

                _run(execute_improvement(improvement, "/repo/path", 300, config))

                # Verify _note was called with error tag
                note_calls = [call for call in mock_note.call_args_list]
                error_calls = [
                    call for call in note_calls
                    if call[1].get("tags") and "error" in call[1]["tags"]
                ]
                assert len(error_calls) > 0, "Expected _note() to be called with 'error' tag"


# ---------------------------------------------------------------------------
# Integration test: executor_task_prompt building
# ---------------------------------------------------------------------------


class TestTaskPromptBuilding:
    """Test that execute_improvement builds the task prompt correctly."""

    @patch("swe_af.improve.executor.AgentAI")
    def test_builds_task_prompt_with_all_params(self, mock_agent_ai_class: MagicMock) -> None:
        """execute_improvement should build task prompt with improvement, repo_path, timeout_seconds."""
        from swe_af.improve.executor import execute_improvement

        # Setup mock
        mock_ai_instance = MagicMock()
        mock_ai_instance.run = AsyncMock(
            return_value=_make_mock_response(
                ExecutorResult(success=True, commit_sha="abc123")
            )
        )
        mock_agent_ai_class.return_value = mock_ai_instance

        config = {"runtime": "claude_code", "agent_max_turns": 50}
        improvement = {
            "id": "prompt-test",
            "category": "test-coverage",
            "title": "Prompt Test",
            "description": "Test prompt building",
            "files": ["test.py"],
            "priority": 3,
            "notes": "Some notes",
            "found_by_run": "2024-01-01T00:00:00Z",
        }
        repo_path = "/test/repo"
        timeout_seconds = 180

        _run(execute_improvement(improvement, repo_path, timeout_seconds, config))

        # Verify ai.run was called and extract the task prompt
        assert mock_ai_instance.run.call_count == 1
        call_args = mock_ai_instance.run.call_args
        task_prompt = call_args[0][0]  # First positional arg

        # Verify task prompt contains expected elements
        assert "prompt-test" in task_prompt, "Expected improvement ID in task prompt"
        assert "test-coverage" in task_prompt, "Expected category in task prompt"
        assert "Prompt Test" in task_prompt, "Expected title in task prompt"
        assert "Test prompt building" in task_prompt, "Expected description in task prompt"
        assert "test.py" in task_prompt, "Expected files in task prompt"
        assert "/test/repo" in task_prompt, "Expected repo_path in task prompt"
        assert "180" in task_prompt, "Expected timeout_seconds in task prompt"
