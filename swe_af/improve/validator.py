"""swe_af.improve.validator — validate_improvement reasoner.

Validates whether a previously-identified improvement is still relevant and
actionable by examining affected files and checking if the described issue
still exists.
"""

from __future__ import annotations

import logging

from swe_af.agent_ai import AgentAI, AgentAIConfig
from swe_af.agent_ai.types import Tool
from swe_af.improve import improve_router
from swe_af.improve.prompts import VALIDATOR_SYSTEM_PROMPT, validator_task_prompt
from swe_af.improve.schemas import ImproveConfig, ValidatorResult, improve_resolve_models

logger = logging.getLogger(__name__)


def _note(msg: str, tags: list[str] | None = None) -> None:
    """Log a message via improve_router.note() when attached, else fall back to logger."""
    try:
        improve_router.note(msg, tags=tags or [])
    except RuntimeError:
        logger.debug("[validator] %s (tags=%s)", msg, tags)


def _fallback_result(error_msg: str) -> ValidatorResult:
    """Return a fallback ValidatorResult that assumes validity on error."""
    return ValidatorResult(
        is_valid=True,
        reason=f"Validator failed: {error_msg}. Assuming improvement is still valid.",
        file_changes_detected=[],
    )


@improve_router.reasoner()
async def validate_improvement(
    improvement_area: dict,
    repo_path: str,
    config: dict,
) -> dict:
    """Validate whether an improvement is still relevant and actionable.

    Uses an AgentAI instance with READ/GLOB/GREP tools (no write access) to
    examine the affected files and determine if the described issue still
    exists. On success, returns a ValidatorResult. On failure, returns a
    fallback result with is_valid=True (assume valid to avoid blocking).

    Args:
        improvement_area: ImprovementArea as dict containing id, category,
            title, description, files, etc.
        repo_path: Absolute path to the repository on disk.
        config: ImproveConfig as dict with runtime, models, etc.

    Returns:
        A dict produced by ValidatorResult.model_dump() with is_valid, reason,
        and file_changes_detected fields.
    """
    improvement_id = improvement_area.get("id", "unknown")
    _note(
        f"validate_improvement: starting validation for {improvement_id!r}",
        tags=["validator", "start", f"improvement_id:{improvement_id}"],
    )

    # Parse config and resolve model
    try:
        improve_config = ImproveConfig(**config)
        resolved_models = improve_resolve_models(improve_config)
        validator_model = resolved_models["validator_model"]
    except Exception as e:
        logger.exception("validate_improvement: failed to parse config or resolve model")
        _note(
            f"validate_improvement: config parsing failed for {improvement_id!r}",
            tags=["validator", "error", f"improvement_id:{improvement_id}"],
        )
        return _fallback_result(f"Config parsing error: {e}").model_dump()

    # Build task prompt
    task_prompt = validator_task_prompt(
        improvement=improvement_area,
        repo_path=repo_path,
    )

    # Create AgentAI with read-only tools
    # Map runtime to provider: claude_code -> claude, open_code -> opencode
    ai_provider = "claude" if improve_config.runtime == "claude_code" else "opencode"
    ai = AgentAI(
        AgentAIConfig(
            provider=ai_provider,
            model=validator_model,
            cwd=repo_path,
            max_turns=10,  # Validation should be quick
            allowed_tools=[Tool.READ, Tool.GLOB, Tool.GREP],  # Read-only
            permission_mode=improve_config.permission_mode or None,
        )
    )

    # Run validation
    try:
        response = await ai.run(
            task_prompt,
            system_prompt=VALIDATOR_SYSTEM_PROMPT,
            output_schema=ValidatorResult,
        )
    except Exception as e:
        logger.exception("validate_improvement: AgentAI.run() raised an exception")
        _note(
            f"validate_improvement: agent failed for {improvement_id!r}",
            tags=["validator", "error", f"improvement_id:{improvement_id}"],
        )
        return _fallback_result(f"Agent error: {e}").model_dump()

    # Parse response
    if response.parsed is None:
        _note(
            f"validate_improvement: parsed response is None for {improvement_id!r}",
            tags=["validator", "error", f"improvement_id:{improvement_id}"],
        )
        return _fallback_result("Agent returned unparseable response").model_dump()

    result: ValidatorResult = response.parsed
    _note(
        f"validate_improvement: completed for {improvement_id!r}, is_valid={result.is_valid}",
        tags=["validator", "complete", f"improvement_id:{improvement_id}"],
    )
    return result.model_dump()
