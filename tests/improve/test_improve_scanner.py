"""Tests for swe_af.improve.scanner — scan_for_improvements reasoner.

Covers:
- AC1: scan_for_improvements is decorated with @improve_router.reasoner()
- AC2: Function signature matches: repo_path, config, existing_improvements -> dict
- AC3: Accepts ImproveConfig dict, resolves to scanner_model via improve_resolve_models
- AC4: Creates AgentAI instance with correct provider, model, cwd, allowed_tools
- AC5: AgentAI max_turns respects config.agent_max_turns
- AC6: Calls ai.run with SCANNER_SYSTEM_PROMPT and ScanResult output_schema
- AC7: On success stamps found_by_run (ISO datetime) and status='pending' on all new_areas
- AC8: On failure returns empty ScanResult via _fallback_scan_result
- AC9: Uses improve_router.note() for instrumentation
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentfield import AgentRouter
from swe_af.agent_ai import Tool
from swe_af.improve.schemas import ImprovementArea, ScanResult


def _registered_names(router: AgentRouter) -> set[str]:
    """Get registered reasoner names from a router."""
    return {r["func"].__name__ for r in router.reasoners}


def _run(coro):
    """Run an async coroutine synchronously for tests."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# AC1: scan_for_improvements is registered on improve_router
# ---------------------------------------------------------------------------


class TestScanForImprovementsRegistered:
    def test_scan_for_improvements_registered_on_improve_router(self) -> None:
        """Test that scan_for_improvements is registered on improve_router."""
        import swe_af.improve.scanner  # noqa: F401
        from swe_af.improve import improve_router

        names = _registered_names(improve_router)
        assert "scan_for_improvements" in names, (
            f"scan_for_improvements not registered on improve_router. Found: {names}"
        )


# ---------------------------------------------------------------------------
# AC2: Function signature verification
# ---------------------------------------------------------------------------


class TestFunctionSignature:
    def test_scan_for_improvements_has_correct_signature(self) -> None:
        """Test that scan_for_improvements has the correct signature."""
        import inspect

        from swe_af.improve.scanner import scan_for_improvements

        sig = inspect.signature(scan_for_improvements)
        params = list(sig.parameters.keys())

        assert params == ["repo_path", "config", "existing_improvements"], (
            f"Expected parameters [repo_path, config, existing_improvements], got {params}"
        )

        # Check types (annotations may be strings in some Python versions)
        repo_path_ann = sig.parameters["repo_path"].annotation
        assert repo_path_ann == str or repo_path_ann == "str"
        config_ann = sig.parameters["config"].annotation
        assert config_ann == dict or config_ann == "dict"
        existing_ann = sig.parameters["existing_improvements"].annotation
        assert existing_ann == list[dict] or existing_ann == "list[dict]"
        return_ann = sig.return_annotation
        assert return_ann == dict or return_ann == "dict"


# ---------------------------------------------------------------------------
# AC3: scanner_task_prompt is built correctly
# ---------------------------------------------------------------------------


class TestScannerTaskPromptBuilding:
    def test_scanner_task_prompt_receives_correct_arguments(self) -> None:
        """Test that scanner_task_prompt is called with correct arguments."""
        from swe_af.improve.scanner import scan_for_improvements

        config = {
            "runtime": "claude_code",
            "scan_depth": "thorough",
            "categories": ["test-coverage", "code-quality"],
            "agent_max_turns": 50,
        }
        existing = [{"id": "existing-1"}, {"id": "existing-2"}]

        mock_response = MagicMock()
        mock_response.parsed = ScanResult(
            new_areas=[],
            scan_depth_used="thorough",
            summary="Test",
            files_analyzed=0,
        )

        with patch("swe_af.improve.scanner.scanner_task_prompt") as mock_task_prompt:
            with patch("swe_af.improve.scanner.AgentAI") as MockAgentAI:
                mock_task_prompt.return_value = "Test prompt"
                instance = MagicMock()
                instance.run = AsyncMock(return_value=mock_response)
                MockAgentAI.return_value = instance

                _run(scan_for_improvements(
                    repo_path="/test/repo",
                    config=config,
                    existing_improvements=existing,
                ))

            mock_task_prompt.assert_called_once_with(
                repo_path="/test/repo",
                scan_depth="thorough",
                existing_improvements=existing,
                categories=["test-coverage", "code-quality"],
            )


# ---------------------------------------------------------------------------
# AC4 & AC5: AgentAI initialization with correct parameters
# ---------------------------------------------------------------------------


class TestAgentAIInitialization:
    def test_agent_ai_initialized_with_correct_model_from_resolver(self) -> None:
        """Test that AgentAI is initialized with scanner_model from improve_resolve_models."""
        from swe_af.improve.scanner import scan_for_improvements

        config = {
            "runtime": "claude_code",
            "models": {"scanner": "opus"},
            "agent_max_turns": 50,
        }

        mock_response = MagicMock()
        mock_response.parsed = ScanResult(
            new_areas=[],
            scan_depth_used="normal",
            summary="Test",
            files_analyzed=0,
        )

        with patch("swe_af.improve.scanner.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            _run(scan_for_improvements(
                repo_path="/test/repo",
                config=config,
                existing_improvements=[],
            ))

            # Verify AgentAI was called with correct config
            call_args = MockAgentAI.call_args
            agent_config = call_args[0][0]

            assert agent_config.model == "opus", f"Expected model='opus', got {agent_config.model}"
            assert agent_config.provider == "claude"
            assert agent_config.cwd == "/test/repo"
            assert agent_config.max_turns == 50

    def test_agent_ai_tools_include_read_glob_grep_bash(self) -> None:
        """Test that AgentAI is initialized with READ, GLOB, GREP, BASH tools."""
        from swe_af.improve.scanner import scan_for_improvements

        config = {"runtime": "claude_code", "agent_max_turns": 50}

        mock_response = MagicMock()
        mock_response.parsed = ScanResult(
            new_areas=[],
            scan_depth_used="normal",
            summary="Test",
            files_analyzed=0,
        )

        with patch("swe_af.improve.scanner.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            _run(scan_for_improvements(
                repo_path="/test/repo",
                config=config,
                existing_improvements=[],
            ))

            call_args = MockAgentAI.call_args
            agent_config = call_args[0][0]

            expected_tools = [Tool.READ, Tool.GLOB, Tool.GREP, Tool.BASH]
            assert agent_config.allowed_tools == expected_tools, (
                f"Expected tools {expected_tools}, got {agent_config.allowed_tools}"
            )

    def test_agent_ai_max_turns_respects_config(self) -> None:
        """Test that AgentAI max_turns respects config.agent_max_turns."""
        from swe_af.improve.scanner import scan_for_improvements

        config = {"runtime": "claude_code", "agent_max_turns": 25}

        mock_response = MagicMock()
        mock_response.parsed = ScanResult(
            new_areas=[],
            scan_depth_used="normal",
            summary="Test",
            files_analyzed=0,
        )

        with patch("swe_af.improve.scanner.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            _run(scan_for_improvements(
                repo_path="/test/repo",
                config=config,
                existing_improvements=[],
            ))

            call_args = MockAgentAI.call_args
            agent_config = call_args[0][0]

            assert agent_config.max_turns == 25


# ---------------------------------------------------------------------------
# AC6: ai.run called with correct parameters
# ---------------------------------------------------------------------------


class TestAgentAIRunCall:
    def test_ai_run_called_with_scanner_system_prompt_and_scan_result_schema(self) -> None:
        """Test that ai.run is called with SCANNER_SYSTEM_PROMPT and ScanResult output_schema."""
        from swe_af.improve.prompts import SCANNER_SYSTEM_PROMPT
        from swe_af.improve.scanner import scan_for_improvements

        config = {"runtime": "claude_code", "agent_max_turns": 50}

        mock_response = MagicMock()
        mock_response.parsed = ScanResult(
            new_areas=[],
            scan_depth_used="normal",
            summary="Test",
            files_analyzed=0,
        )

        with patch("swe_af.improve.scanner.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            _run(scan_for_improvements(
                repo_path="/test/repo",
                config=config,
                existing_improvements=[],
            ))

            # Verify ai.run was called with correct parameters
            instance.run.assert_called_once()
            call_kwargs = instance.run.call_args[1]

            assert call_kwargs["system_prompt"] == SCANNER_SYSTEM_PROMPT
            assert call_kwargs["output_schema"] == ScanResult


# ---------------------------------------------------------------------------
# AC7: Success path with timestamp stamping
# ---------------------------------------------------------------------------


class TestSuccessPathTimestampStamping:
    def test_on_success_new_areas_stamped_with_found_by_run_and_status_pending(self) -> None:
        """Test that on success, new_areas are stamped with found_by_run and status='pending'."""
        from swe_af.improve.scanner import scan_for_improvements

        config = {"runtime": "claude_code", "agent_max_turns": 50}

        # Create mock areas without found_by_run or with different status
        area1 = ImprovementArea(
            id="test-1",
            category="test-coverage",
            title="Add tests",
            description="Need tests",
            files=["test.py"],
            found_by_run="",
            status="completed",  # Should be overwritten
        )
        area2 = ImprovementArea(
            id="test-2",
            category="code-quality",
            title="Improve code",
            description="Improve quality",
            files=["code.py"],
            found_by_run="",
            status="in_progress",  # Should be overwritten
        )

        mock_scan = ScanResult(
            new_areas=[area1, area2],
            scan_depth_used="normal",
            summary="Found 2 improvements",
            files_analyzed=10,
        )
        mock_response = MagicMock()
        mock_response.parsed = mock_scan

        with patch("swe_af.improve.scanner.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            before_call = datetime.now(timezone.utc)
            result = _run(scan_for_improvements(
                repo_path="/test/repo",
                config=config,
                existing_improvements=[],
            ))
            after_call = datetime.now(timezone.utc)

        assert isinstance(result, dict)
        assert "new_areas" in result
        assert len(result["new_areas"]) == 2

        # Check that found_by_run is an ISO timestamp
        for area_dict in result["new_areas"]:
            assert area_dict["found_by_run"], "found_by_run should not be empty"
            # Parse the timestamp to ensure it's valid ISO format
            ts = datetime.fromisoformat(area_dict["found_by_run"])
            # Verify it's recent (between before and after the call)
            assert before_call <= ts <= after_call, (
                f"Timestamp {ts} not in expected range [{before_call}, {after_call}]"
            )

            # Check that status is 'pending'
            assert area_dict["status"] == "pending", (
                f"Expected status='pending', got {area_dict['status']}"
            )


# ---------------------------------------------------------------------------
# AC8: Failure path returns empty ScanResult
# ---------------------------------------------------------------------------


class TestFailurePathFallback:
    def test_on_agent_ai_exception_returns_empty_scan_result(self) -> None:
        """Test that on AgentAI.run() failure, returns empty ScanResult without raising."""
        from swe_af.improve.scanner import scan_for_improvements

        config = {"runtime": "claude_code", "agent_max_turns": 50}

        with patch("swe_af.improve.scanner.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(side_effect=RuntimeError("LLM connection error"))
            MockAgentAI.return_value = instance

            result = _run(scan_for_improvements(
                repo_path="/test/repo",
                config=config,
                existing_improvements=[],
            ))

        # Should return empty ScanResult
        assert isinstance(result, dict)
        assert result["new_areas"] == []
        assert result["files_analyzed"] == 0
        assert "Scanner agent failed" in result["summary"] or "failed" in result["summary"].lower()

    def test_on_parsed_none_returns_empty_scan_result(self) -> None:
        """Test that when response.parsed is None, returns empty ScanResult."""
        from swe_af.improve.scanner import scan_for_improvements

        config = {"runtime": "claude_code", "agent_max_turns": 50}

        mock_response = MagicMock()
        mock_response.parsed = None

        with patch("swe_af.improve.scanner.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            result = _run(scan_for_improvements(
                repo_path="/test/repo",
                config=config,
                existing_improvements=[],
            ))

        # Should return empty ScanResult
        assert isinstance(result, dict)
        assert result["new_areas"] == []
        assert result["files_analyzed"] == 0


# ---------------------------------------------------------------------------
# AC9: improve_router.note() instrumentation
# ---------------------------------------------------------------------------


class TestRouterNoteInstrumentation:
    def test_improve_router_note_called_with_correct_tags(self) -> None:
        """Test that improve_router.note() is called with start, complete, error tags."""
        from swe_af.improve.scanner import scan_for_improvements

        config = {"runtime": "claude_code", "agent_max_turns": 50}

        mock_response = MagicMock()
        mock_response.parsed = ScanResult(
            new_areas=[],
            scan_depth_used="normal",
            summary="Test",
            files_analyzed=0,
        )

        with patch("swe_af.improve.scanner.AgentAI") as MockAgentAI:
            with patch("swe_af.improve.scanner._note") as mock_note:
                instance = MagicMock()
                instance.run = AsyncMock(return_value=mock_response)
                MockAgentAI.return_value = instance

                _run(scan_for_improvements(
                    repo_path="/test/repo",
                    config=config,
                    existing_improvements=[],
                ))

                # Verify _note() was called
                assert mock_note.call_count >= 2, "Expected at least start and complete calls"

                # Check for 'start' tag (tags is a keyword argument)
                start_calls = [
                    call for call in mock_note.call_args_list
                    if "tags" in call[1] and "scanner" in call[1]["tags"] and "start" in call[1]["tags"]
                ]
                assert len(start_calls) >= 1, "Expected at least one call with 'scanner' and 'start' tags"

                # Check for 'complete' tag
                complete_calls = [
                    call for call in mock_note.call_args_list
                    if "tags" in call[1] and "scanner" in call[1]["tags"] and "complete" in call[1]["tags"]
                ]
                assert len(complete_calls) >= 1, "Expected at least one call with 'scanner' and 'complete' tags"

    def test_error_tag_used_on_failure(self) -> None:
        """Test that 'error' tag is used when AgentAI.run() fails."""
        from swe_af.improve.scanner import scan_for_improvements

        config = {"runtime": "claude_code", "agent_max_turns": 50}

        with patch("swe_af.improve.scanner.AgentAI") as MockAgentAI:
            with patch("swe_af.improve.scanner._note") as mock_note:
                instance = MagicMock()
                instance.run = AsyncMock(side_effect=RuntimeError("Error"))
                MockAgentAI.return_value = instance

                _run(scan_for_improvements(
                    repo_path="/test/repo",
                    config=config,
                    existing_improvements=[],
                ))

                # Check for 'error' tag (tags is a keyword argument)
                error_calls = [
                    call for call in mock_note.call_args_list
                    if "tags" in call[1] and "scanner" in call[1]["tags"] and "error" in call[1]["tags"]
                ]
                assert len(error_calls) >= 1, "Expected at least one call with 'scanner' and 'error' tags"


# ---------------------------------------------------------------------------
# Edge cases and integration tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_returned_dict_is_valid_scan_result(self) -> None:
        """Test that the returned dict can be parsed back into ScanResult."""
        from swe_af.improve.scanner import scan_for_improvements

        config = {"runtime": "claude_code", "agent_max_turns": 50}

        mock_scan = ScanResult(
            new_areas=[],
            scan_depth_used="normal",
            summary="Test scan",
            files_analyzed=5,
        )
        mock_response = MagicMock()
        mock_response.parsed = mock_scan

        with patch("swe_af.improve.scanner.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            result = _run(scan_for_improvements(
                repo_path="/test/repo",
                config=config,
                existing_improvements=[],
            ))

        # Should be able to parse back
        parsed = ScanResult.model_validate(result)
        assert parsed.scan_depth_used == "normal"
        assert parsed.summary == "Test scan"
        assert parsed.files_analyzed == 5

    def test_open_code_runtime_uses_correct_provider(self) -> None:
        """Test that open_code runtime maps to correct provider."""
        from swe_af.improve.scanner import scan_for_improvements

        config = {
            "runtime": "open_code",
            "agent_max_turns": 50,
        }

        mock_response = MagicMock()
        mock_response.parsed = ScanResult(
            new_areas=[],
            scan_depth_used="normal",
            summary="Test",
            files_analyzed=0,
        )

        with patch("swe_af.improve.scanner.AgentAI") as MockAgentAI:
            instance = MagicMock()
            instance.run = AsyncMock(return_value=mock_response)
            MockAgentAI.return_value = instance

            _run(scan_for_improvements(
                repo_path="/test/repo",
                config=config,
                existing_improvements=[],
            ))

            call_args = MockAgentAI.call_args
            agent_config = call_args[0][0]

            assert agent_config.provider == "open_code", (
                f"Expected provider='open_code', got {agent_config.provider}"
            )
