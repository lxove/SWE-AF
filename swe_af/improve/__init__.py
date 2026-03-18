"""swe_af.improve — Continuous recursive improvement node.

Exports
-------
improve_router : AgentRouter
    Router tagged 'swe-improve' with scanner, validator, executor reasoners.

Intentionally does NOT import swe_af.reasoners.pipeline to avoid loading
planning agents into this process.
"""

from __future__ import annotations

from agentfield import AgentRouter

improve_router = AgentRouter(tags=["swe-improve"])

# Import submodules to register reasoners on improve_router
from . import scanner   # noqa: E402, F401 — registers scan_for_improvements
from . import validator # noqa: E402, F401 — registers validate_improvement
from . import executor  # noqa: E402, F401 — registers execute_improvement


# ---------------------------------------------------------------------------
# Thin wrapper: run_github_pr (delegates to execution_agents)
# ---------------------------------------------------------------------------


@improve_router.reasoner()
async def run_github_pr(
    repo_path: str,
    integration_branch: str,
    base_branch: str,
    goal: str,
    build_summary: str = "",
    completed_issues: list[dict] | None = None,
    accumulated_debt: list[dict] | None = None,
    artifacts_dir: str = "",
    model: str = "sonnet",
    permission_mode: str = "",
    ai_provider: str = "claude",
) -> dict:
    """Thin wrapper around execution_agents.run_github_pr."""
    import swe_af.reasoners.execution_agents as _ea  # noqa: PLC0415
    return await _ea.run_github_pr(
        repo_path=repo_path, integration_branch=integration_branch,
        base_branch=base_branch, goal=goal, build_summary=build_summary,
        completed_issues=completed_issues, accumulated_debt=accumulated_debt,
        artifacts_dir=artifacts_dir, model=model,
        permission_mode=permission_mode, ai_provider=ai_provider,
    )


__all__ = ["improve_router"]
