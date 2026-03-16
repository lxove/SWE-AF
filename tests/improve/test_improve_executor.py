"""Unit tests for swe_af.improve.executor module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from swe_af.improve.schemas import ExecutorResult


class TestExecuteImprovement:
    """Tests for execute_improvement reasoner."""

    @pytest.fixture
    def improvement_area(self):
        """Sample improvement area for testing."""
        return {
            "id": "test-improvement",
            "category": "test-coverage",
            "title": "Add missing tests",
            "description": "Add unit tests for authentication module",
            "files": ["src/auth.py"],
            "priority": 5,
            "status": "pending",
            "found_by_run": "2024-01-01T00:00:00Z",
        }

    @pytest.fixture
    def config(self):
        """Sample ImproveConfig for testing."""
        return {
            "runtime": "claude_code",
            "models": None,
            "max_time_seconds": 3600,
            "max_improvements": 10,
            "permission_mode": "",
            "scan_depth": "normal",
            "categories": None,
            "agent_max_turns": 50,
        }

    @pytest.fixture
    def temp_repo_dir(self, tmp_path):
        """Create a temporary repository directory for testing."""
        repo_dir = tmp_path / "test_repo"
        repo_dir.mkdir()
        return str(repo_dir)

    @pytest.mark.asyncio
    async def test_executor_task_prompt_built_correctly(self, improvement_area, config, temp_repo_dir):
        """Test that executor_task_prompt is built with correct parameters."""
        from swe_af.improve.executor import execute_improvement

        with patch("swe_af.agent_ai.client.AgentAI") as mock_agent_class:
            mock_ai = AsyncMock()
            mock_agent_class.return_value = mock_ai

            # Mock successful response
            mock_response = MagicMock()
            mock_response.parsed = None
            mock_response.result = '{"success": true, "commit_sha": "abc123"}'
            mock_ai.run.return_value = mock_response

            with patch("swe_af.improve.prompts.executor_task_prompt") as mock_prompt_builder:
                mock_prompt_builder.return_value = "test prompt"

                await execute_improvement(
                    improvement_area,
                    temp_repo_dir,
                    300,
                    config,
                )

                # Verify executor_task_prompt was called with correct args
                mock_prompt_builder.assert_called_once_with(
                    improvement=improvement_area,
                    repo_path=temp_repo_dir,
                    timeout_seconds=300,
                )

    @pytest.mark.asyncio
    async def test_agent_ai_initialized_with_all_six_tools(self, improvement_area, config, temp_repo_dir):
        """Test that AgentAI is initialized with READ, WRITE, EDIT, BASH, GLOB, GREP."""
        from swe_af.improve.executor import execute_improvement

        with patch("swe_af.agent_ai.client.AgentAI") as mock_agent_class:
            mock_ai = AsyncMock()
            mock_agent_class.return_value = mock_ai

            # Mock successful response
            mock_response = MagicMock()
            mock_response.parsed = None
            mock_response.result = '{"success": true, "commit_sha": "abc123"}'
            mock_ai.run.return_value = mock_response

            await execute_improvement(
                improvement_area,
                "/test/repo",
                300,
                config,
            )

            # Verify AgentAI was initialized
            assert mock_agent_class.called
            agent_config = mock_agent_class.call_args[1]["config"]

            # Verify all 6 tools are present
            from swe_af.agent_ai.types import Tool
            expected_tools = [
                Tool.READ,
                Tool.WRITE,
                Tool.EDIT,
                Tool.BASH,
                Tool.GLOB,
                Tool.GREP,
            ]
            assert agent_config.allowed_tools == expected_tools

    @pytest.mark.asyncio
    async def test_agent_ai_max_turns_respects_config(self, improvement_area, temp_repo_dir):
        """Test that AgentAI max_turns respects config.agent_max_turns."""
        from swe_af.improve.executor import execute_improvement

        config = {
            "runtime": "claude_code",
            "models": None,
            "max_time_seconds": 3600,
            "max_improvements": 10,
            "permission_mode": "",
            "scan_depth": "normal",
            "categories": None,
            "agent_max_turns": 75,  # Custom value
        }

        with patch("swe_af.agent_ai.client.AgentAI") as mock_agent_class:
            mock_ai = AsyncMock()
            mock_agent_class.return_value = mock_ai

            # Mock successful response
            mock_response = MagicMock()
            mock_response.parsed = None
            mock_response.result = '{"success": true, "commit_sha": "abc123"}'
            mock_ai.run.return_value = mock_response

            await execute_improvement(
                improvement_area,
                "/test/repo",
                300,
                config,
            )

            # Verify max_turns is set correctly
            agent_config = mock_agent_class.call_args[1]["config"]
            assert agent_config.max_turns == 75

    @pytest.mark.asyncio
    async def test_ai_run_wrapped_with_asyncio_wait_for(self, improvement_area, config, temp_repo_dir):
        """Test that ai.run() is wrapped with asyncio.wait_for with timeout_seconds."""
        from swe_af.improve.executor import execute_improvement

        with patch("swe_af.agent_ai.client.AgentAI") as mock_agent_class:
            mock_ai = AsyncMock()
            mock_agent_class.return_value = mock_ai

            # Mock successful response
            mock_response = MagicMock()
            mock_response.parsed = None
            mock_response.result = '{"success": true, "commit_sha": "abc123"}'

            # Track the call to ai.run
            async def mock_run(*args, **kwargs):
                return mock_response

            mock_ai.run = AsyncMock(side_effect=mock_run)

            # Spy on asyncio.wait_for
            original_wait_for = asyncio.wait_for
            wait_for_called = []

            async def spy_wait_for(coro, timeout):
                wait_for_called.append({"timeout": timeout})
                return await original_wait_for(coro, timeout=timeout)

            with patch("asyncio.wait_for", side_effect=spy_wait_for):
                await execute_improvement(
                    improvement_area,
                    temp_repo_dir,
                    300,
                    config,
                )

                # Verify asyncio.wait_for was called with correct timeout
                assert len(wait_for_called) == 1
                assert wait_for_called[0]["timeout"] == 300

    @pytest.mark.asyncio
    async def test_timeout_returns_executor_result_with_error(self, improvement_area, config, temp_repo_dir):
        """Test that asyncio.TimeoutError returns ExecutorResult(success=False, error='Timed out...')."""
        from swe_af.improve.executor import execute_improvement

        with patch("swe_af.agent_ai.client.AgentAI") as mock_agent_class:
            mock_ai = AsyncMock()
            mock_agent_class.return_value = mock_ai

            # Make ai.run timeout
            async def timeout_run(*args, **kwargs):
                await asyncio.sleep(10)  # Never completes

            mock_ai.run = AsyncMock(side_effect=timeout_run)

            # Patch asyncio.wait_for to raise timeout immediately
            async def immediate_timeout(*args, **kwargs):
                raise asyncio.TimeoutError()

            with patch("asyncio.wait_for", side_effect=immediate_timeout):
                result = await execute_improvement(
                    improvement_area,
                    temp_repo_dir,
                    300,
                    config,
                )

                # Verify result indicates timeout
                assert result["success"] is False
                assert "Timed out after 300s" in result["error"]
                assert result["commit_sha"] is None

    @pytest.mark.asyncio
    async def test_success_path_returns_commit_sha_and_new_findings(self, improvement_area, config, temp_repo_dir):
        """Test that success path returns ExecutorResult with commit_sha and new_findings list."""
        from swe_af.improve.executor import execute_improvement

        with patch("swe_af.agent_ai.client.AgentAI") as mock_agent_class:
            mock_ai = AsyncMock()
            mock_agent_class.return_value = mock_ai

            # Mock successful response with new_findings
            executor_result = ExecutorResult(
                success=True,
                commit_sha="abc123def456",
                commit_message="improve: add missing tests",
                files_changed=["src/auth.py", "tests/test_auth.py"],
                new_findings=[
                    {
                        "id": "new-improvement",
                        "category": "documentation",
                        "title": "Add docstrings",
                        "description": "Add missing docstrings to auth module",
                        "files": ["src/auth.py"],
                        "priority": 7,
                        "status": "pending",
                        "found_by_run": "2024-01-01T00:00:00Z",
                    }
                ],
                tests_passed=True,
                verification_output="All tests passed",
            )

            mock_response = MagicMock()
            mock_response.parsed = None
            mock_response.result = executor_result.model_dump_json()
            mock_ai.run.return_value = mock_response

            result = await execute_improvement(
                improvement_area,
                "/test/repo",
                300,
                config,
            )

            # Verify success response
            assert result["success"] is True
            assert result["commit_sha"] == "abc123def456"
            assert result["commit_message"] == "improve: add missing tests"
            assert len(result["new_findings"]) == 1
            assert result["new_findings"][0]["id"] == "new-improvement"
            assert result["tests_passed"] is True

    @pytest.mark.asyncio
    async def test_failure_path_returns_executor_result_with_error(self, improvement_area, config, temp_repo_dir):
        """Test that general exceptions return ExecutorResult(success=False, error=str(e))."""
        from swe_af.improve.executor import execute_improvement

        with patch("swe_af.agent_ai.client.AgentAI") as mock_agent_class:
            mock_ai = AsyncMock()
            mock_agent_class.return_value = mock_ai

            # Make ai.run raise a general exception
            mock_ai.run.side_effect = RuntimeError("Test error")

            result = await execute_improvement(
                improvement_area,
                "/test/repo",
                300,
                config,
            )

            # Verify error handling
            assert result["success"] is False
            assert "Test error" in result["error"]
            assert result["commit_sha"] is None

    @pytest.mark.asyncio
    async def test_executor_resolves_model_via_improve_resolve_models(self, improvement_area, temp_repo_dir):
        """Test that executor uses improve_resolve_models to get executor_model."""
        from swe_af.improve.executor import execute_improvement

        config = {
            "runtime": "claude_code",
            "models": {"executor": "opus"},
            "max_time_seconds": 3600,
            "max_improvements": 10,
            "permission_mode": "",
            "scan_depth": "normal",
            "categories": None,
            "agent_max_turns": 50,
        }

        with patch("swe_af.agent_ai.client.AgentAI") as mock_agent_class:
            mock_ai = AsyncMock()
            mock_agent_class.return_value = mock_ai

            # Mock successful response
            mock_response = MagicMock()
            mock_response.parsed = None
            mock_response.result = '{"success": true, "commit_sha": "abc123"}'
            mock_ai.run.return_value = mock_response

            await execute_improvement(
                improvement_area,
                "/test/repo",
                300,
                config,
            )

            # Verify model was resolved correctly
            agent_config = mock_agent_class.call_args[1]["config"]
            assert agent_config.model == "opus"

    @pytest.mark.asyncio
    async def test_provider_set_based_on_runtime(self, improvement_area, temp_repo_dir):
        """Test that provider is set to 'opencode' for open_code runtime."""
        from swe_af.improve.executor import execute_improvement

        config = {
            "runtime": "open_code",
            "models": None,
            "max_time_seconds": 3600,
            "max_improvements": 10,
            "permission_mode": "",
            "scan_depth": "normal",
            "categories": None,
            "agent_max_turns": 50,
        }

        with patch("swe_af.agent_ai.client.AgentAI") as mock_agent_class:
            mock_ai = AsyncMock()
            mock_agent_class.return_value = mock_ai

            # Mock successful response
            mock_response = MagicMock()
            mock_response.parsed = None
            mock_response.result = '{"success": true, "commit_sha": "abc123"}'
            mock_ai.run.return_value = mock_response

            await execute_improvement(
                improvement_area,
                "/test/repo",
                300,
                config,
            )

            # Verify provider was set correctly
            agent_config = mock_agent_class.call_args[1]["config"]
            assert agent_config.provider == "opencode"

    @pytest.mark.asyncio
    async def test_new_findings_returned_as_list_of_dicts(self, improvement_area, config, temp_repo_dir):
        """Test that new_findings are returned as a list of dicts (ImprovementArea)."""
        from swe_af.improve.executor import execute_improvement

        with patch("swe_af.agent_ai.client.AgentAI") as mock_agent_class:
            mock_ai = AsyncMock()
            mock_agent_class.return_value = mock_ai

            # Mock successful response with multiple new findings
            executor_result = ExecutorResult(
                success=True,
                commit_sha="abc123",
                new_findings=[
                    {
                        "id": "finding-1",
                        "category": "test-coverage",
                        "title": "Add tests for module A",
                        "description": "Missing tests",
                        "files": ["a.py"],
                        "priority": 5,
                        "status": "pending",
                        "found_by_run": "2024-01-01T00:00:00Z",
                    },
                    {
                        "id": "finding-2",
                        "category": "documentation",
                        "title": "Add docstrings",
                        "description": "Missing docstrings",
                        "files": ["b.py"],
                        "priority": 6,
                        "status": "pending",
                        "found_by_run": "2024-01-01T00:00:00Z",
                    },
                ],
            )

            mock_response = MagicMock()
            mock_response.parsed = None
            mock_response.result = executor_result.model_dump_json()
            mock_ai.run.return_value = mock_response

            result = await execute_improvement(
                improvement_area,
                "/test/repo",
                300,
                config,
            )

            # Verify new_findings is a list of dicts
            assert isinstance(result["new_findings"], list)
            assert len(result["new_findings"]) == 2
            assert all(isinstance(f, dict) for f in result["new_findings"])
            assert result["new_findings"][0]["id"] == "finding-1"
            assert result["new_findings"][1]["id"] == "finding-2"
