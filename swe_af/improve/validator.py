"""swe_af.improve.validator — validate_improvement reasoner.

This module will be implemented in issue improve-validator.
For now, it's a stub that registers a placeholder reasoner.
"""

from __future__ import annotations

from swe_af.improve import improve_router


@improve_router.reasoner()
async def validate_improvement(
    improvement: dict,
    repo_path: str,
) -> dict:
    """Validate whether an improvement is still relevant (stub implementation).

    Args:
        improvement: ImprovementArea as dict.
        repo_path: Absolute path to the repository.

    Returns:
        ValidatorResult.model_dump() with validation assessment.
    """
    # Stub implementation - will be replaced in improve-validator issue
    return {
        "is_valid": True,
        "reason": "Validator not yet implemented",
        "file_changes_detected": [],
    }
