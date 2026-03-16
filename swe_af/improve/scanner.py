"""swe_af.improve.scanner — scan_for_improvements reasoner.

This module will be implemented in issue improve-scanner.
For now, it's a stub that registers a placeholder reasoner.
"""

from __future__ import annotations

from swe_af.improve import improve_router


@improve_router.reasoner()
async def scan_for_improvements(
    repo_path: str,
    config: dict,
    existing_improvements: list[dict],
) -> dict:
    """Scan repository for improvement opportunities (stub implementation).

    Args:
        repo_path: Absolute path to the repository.
        config: ImproveConfig as dict.
        existing_improvements: Already-known improvements (to avoid duplicates).

    Returns:
        ScanResult.model_dump() with discovered improvement areas.
    """
    # Stub implementation - will be replaced in improve-scanner issue
    return {
        "new_areas": [],
        "scan_depth_used": "normal",
        "summary": "Scanner not yet implemented",
        "files_analyzed": 0,
    }
