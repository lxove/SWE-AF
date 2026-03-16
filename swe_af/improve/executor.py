"""swe_af.improve.executor — execute_improvement reasoner.

This module will be implemented in issue improve-executor.
For now, it's a stub that registers a placeholder reasoner.
"""

from __future__ import annotations

from swe_af.improve import improve_router


@improve_router.reasoner()
async def execute_improvement(
    improvement: dict,
    repo_path: str,
    config: dict,
) -> dict:
    """Execute a single improvement (stub implementation).

    Args:
        improvement: ImprovementArea as dict.
        repo_path: Absolute path to the repository.
        config: ImproveConfig as dict.

    Returns:
        ExecutorResult.model_dump() with execution results.
    """
    # Stub implementation - will be replaced in improve-executor issue
    return {
        "success": False,
        "commit_sha": None,
        "commit_message": "",
        "files_changed": [],
        "new_findings": [],
        "error": "Executor not yet implemented",
        "tests_passed": False,
        "verification_output": "",
    }
