"""swe_af.improve.scanner — scan_for_improvements reasoner.

Scans a repository to identify actionable improvement opportunities using
an AgentAI instance with appropriate tools and prompts.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from swe_af.agent_ai import AgentAI, AgentAIConfig, Tool
from swe_af.improve import improve_router
from swe_af.improve.prompts import SCANNER_SYSTEM_PROMPT, scanner_task_prompt
from swe_af.improve.schemas import ImproveConfig, ScanResult, improve_resolve_models

logger = logging.getLogger(__name__)


def _note(msg: str, tags: list[str] | None = None) -> None:
    """Log a message via improve_router.note() when attached, else fall back to logger."""
    try:
        improve_router.note(msg, tags=tags or [])
    except RuntimeError:
        logger.debug("[scanner] %s (tags=%s)", msg, tags)


def _fallback_scan_result() -> ScanResult:
    """Return an empty ScanResult when the LLM call fails."""
    return ScanResult(
        new_areas=[],
        scan_depth_used="normal",
        summary="Scanner agent failed to produce results",
        files_analyzed=0,
    )


@improve_router.reasoner()
async def scan_for_improvements(
    repo_path: str,
    config: dict,
    existing_improvements: list[dict],
) -> dict:
    """Scan repository for improvement opportunities.

    Args:
        repo_path: Absolute path to the repository.
        config: ImproveConfig as dict.
        existing_improvements: Already-known improvements (to avoid duplicates).

    Returns:
        ScanResult.model_dump() with discovered improvement areas.
    """
    _note(
        f"scan_for_improvements: starting scan for repo_path={repo_path!r}",
        tags=["scanner", "start"],
    )

    # Parse config and resolve model
    improve_config = ImproveConfig.model_validate(config)
    models = improve_resolve_models(improve_config)
    scanner_model = models["scanner_model"]

    # Build task prompt
    task_prompt = scanner_task_prompt(
        repo_path=repo_path,
        scan_depth=improve_config.scan_depth,
        existing_improvements=existing_improvements,
        categories=improve_config.categories,
    )

    # Create AgentAI instance with proper configuration
    ai = AgentAI(
        AgentAIConfig(
            provider=improve_config.runtime if improve_config.runtime != "claude_code" else "claude",
            model=scanner_model,
            cwd=repo_path,
            max_turns=improve_config.agent_max_turns,
            allowed_tools=[Tool.READ, Tool.GLOB, Tool.GREP, Tool.BASH],
            permission_mode=improve_config.permission_mode or None,
        )
    )

    try:
        response = await ai.run(
            task_prompt,
            system_prompt=SCANNER_SYSTEM_PROMPT,
            output_schema=ScanResult,
        )
    except Exception:
        logger.exception("scan_for_improvements: AgentAI.run() raised an exception")
        _note(
            "scan_for_improvements: LLM call failed; returning empty scan result",
            tags=["scanner", "error"],
        )
        return _fallback_scan_result().model_dump()

    if response.parsed is None:
        _note(
            "scan_for_improvements: parsed response is None; returning empty scan result",
            tags=["scanner", "error"],
        )
        return _fallback_scan_result().model_dump()

    scan_result: ScanResult = response.parsed

    # Stamp found_by_run (ISO datetime) and status='pending' on all new_areas
    timestamp = datetime.now(timezone.utc).isoformat()
    for area in scan_result.new_areas:
        area.found_by_run = timestamp
        area.status = "pending"

    _note(
        f"scan_for_improvements: found {len(scan_result.new_areas)} improvement(s)",
        tags=["scanner", "complete"],
    )

    return scan_result.model_dump()
