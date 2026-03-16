"""swe_af.improve.executor — execute_improvement reasoner with timeout handling."""

from __future__ import annotations

import asyncio
import logging

from swe_af.agent_ai.client import AgentAI, AgentAIConfig
from swe_af.agent_ai.types import Tool
from swe_af.improve import improve_router
from swe_af.improve.prompts import EXECUTOR_SYSTEM_PROMPT, executor_task_prompt
from swe_af.improve.schemas import ExecutorResult, ImproveConfig, improve_resolve_models

logger = logging.getLogger(__name__)


def _note(msg: str, tags: list[str] | None = None) -> None:
    """Log a message via improve_router.note() when attached, else fall back to logger."""
    try:
        improve_router.note(msg, tags=tags or [])
    except RuntimeError:
        logger.debug("[executor] %s (tags=%s)", msg, tags)


@improve_router.reasoner()
async def execute_improvement(
    improvement_area: dict,
    repo_path: str,
    timeout_seconds: int,
    config: dict,
) -> dict:
    """Execute a single improvement with timeout handling.

    Args:
        improvement_area: ImprovementArea as dict.
        repo_path: Absolute path to the repository.
        timeout_seconds: Maximum time in seconds for this improvement.
        config: ImproveConfig as dict.

    Returns:
        ExecutorResult.model_dump() with execution results.
    """
    _note(
        f"Executor: starting improvement {improvement_area.get('id', 'unknown')}",
        tags=["improve_executor", "start"],
    )

    # Parse config and resolve model
    improve_config = ImproveConfig.model_validate(config)
    resolved_models = improve_resolve_models(improve_config)
    executor_model = resolved_models["executor_model"]

    # Determine provider from runtime
    provider = "claude" if improve_config.runtime == "claude_code" else "opencode"

    # Create AgentAI instance with all required tools
    allowed_tools = [
        Tool.READ,
        Tool.WRITE,
        Tool.EDIT,
        Tool.BASH,
        Tool.GLOB,
        Tool.GREP,
    ]

    agent_config = AgentAIConfig(
        provider=provider,
        model=executor_model,
        cwd=repo_path,
        allowed_tools=allowed_tools,
        max_turns=improve_config.agent_max_turns,
        permission_mode=improve_config.permission_mode or None,
    )

    ai = AgentAI(config=agent_config)

    # Build task prompt
    task_prompt = executor_task_prompt(
        improvement=improvement_area,
        repo_path=repo_path,
        timeout_seconds=timeout_seconds,
    )

    try:
        # Wrap ai.run() with asyncio.wait_for for timeout handling
        response = await asyncio.wait_for(
            ai.run(
                task_prompt,
                system_prompt=EXECUTOR_SYSTEM_PROMPT,
                output_schema=ExecutorResult,
            ),
            timeout=timeout_seconds,
        )

        # Parse the result - expecting ExecutorResult JSON
        if response.parsed is not None:
            result: ExecutorResult = response.parsed
            _note(
                f"Executor: completed improvement {improvement_area.get('id', 'unknown')}, "
                f"success={result.success}",
                tags=["improve_executor", "complete"],
            )
            return result.model_dump()

        # If parsed is None, treat as error
        _note(
            f"Executor: improvement {improvement_area.get('id', 'unknown')} "
            "returned unparseable response",
            tags=["improve_executor", "error"],
        )
        return ExecutorResult(
            success=False,
            error="Agent returned unparseable response",
        ).model_dump()

    except asyncio.TimeoutError:
        _note(
            f"Executor: improvement {improvement_area.get('id', 'unknown')} "
            f"timed out after {timeout_seconds}s",
            tags=["improve_executor", "timeout"],
        )
        return ExecutorResult(
            success=False,
            error=f"Timed out after {timeout_seconds}s",
        ).model_dump()

    except Exception as e:
        _note(
            f"Executor: improvement {improvement_area.get('id', 'unknown')} "
            f"failed: {e}",
            tags=["improve_executor", "error"],
        )
        return ExecutorResult(
            success=False,
            error=str(e),
        ).model_dump()
