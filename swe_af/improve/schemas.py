"""Pydantic schemas for the swe-improve continuous improvement module."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# Improvement Categories
# ---------------------------------------------------------------------------

ImprovementCategory = Literal[
    "test-coverage",
    "code-quality",
    "error-handling",
    "consistency",
    "dead-code",
    "performance",
    "documentation",
]

# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------


class ImprovementArea(BaseModel):
    """A single actionable improvement in the repository."""

    model_config = ConfigDict(extra="forbid")

    id: str                              # unique slug, e.g. "missing-test-dag-utils"
    category: ImprovementCategory        # one of the 7 categories
    title: str                           # short human-readable title (<=60 chars)
    description: str                     # what needs improving and why
    files: list[str]                     # affected file paths (1-3 files typically)
    priority: int = 5                    # 1 (highest) to 10 (lowest)
    status: Literal[
        "pending",
        "in_progress",
        "completed",
        "stale",
        "skipped",
        "failed",
    ] = "pending"
    found_by_run: str                    # ISO timestamp when discovered
    completed_by_run: str | None = None  # ISO timestamp when completed
    commit_sha: str | None = None        # git commit hash if completed
    notes: str = ""                      # additional context (file hashes, etc.)


class RunRecord(BaseModel):
    """Metadata for a single improve loop invocation."""

    model_config = ConfigDict(extra="forbid")

    started_at: str                      # ISO timestamp
    ended_at: str | None = None          # ISO timestamp
    improvements_found: int = 0          # count discovered this run
    improvements_completed: int = 0      # count completed this run
    improvements_skipped: int = 0        # count marked stale/skipped
    budget_used_seconds: float = 0.0     # wall-clock time consumed
    stopped_reason: str = ""             # why loop terminated


class ImprovementState(BaseModel):
    """Root state persisted to .swe-af/improvements.json."""

    model_config = ConfigDict(extra="forbid")

    repo_path: str                       # absolute path to repo
    improvements: list[ImprovementArea] = []
    last_scan_at: str | None = None      # ISO timestamp of last scan
    runs: list[RunRecord] = []           # history of run invocations


class ImproveConfig(BaseModel):
    """Configuration for a single improve loop run."""

    model_config = ConfigDict(extra="forbid")

    runtime: Literal["claude_code", "open_code"] = "claude_code"
    models: dict[str, str] | None = None  # keys: "default", "scanner", "executor", "validator"
    repo_url: str = ""                    # git URL; used to clone if repo_path is empty
    max_time_seconds: int = 3600          # overall budget (default 1 hour)
    max_improvements: int = 10            # max improvements to complete (0 = unlimited)
    permission_mode: str = ""             # forwarded to agents
    scan_depth: Literal["quick", "normal", "thorough"] = "normal"
    categories: list[ImprovementCategory] | None = None  # filter to specific categories
    agent_max_turns: int = 50             # max reasoning turns per agent


class ScanResult(BaseModel):
    """Result of scanning the repository for improvements."""

    model_config = ConfigDict(extra="forbid")

    new_areas: list[ImprovementArea]     # discovered improvement areas
    scan_depth_used: Literal["quick", "normal", "thorough"]
    summary: str                          # human-readable summary
    files_analyzed: int = 0               # count of files examined


class ValidatorResult(BaseModel):
    """Result of validating whether an improvement is still relevant."""

    model_config = ConfigDict(extra="forbid")

    is_valid: bool                        # True if improvement should proceed
    reason: str                           # explanation of decision
    file_changes_detected: list[str] = [] # files that changed since discovery


class ExecutorResult(BaseModel):
    """Result of executing a single improvement."""

    model_config = ConfigDict(extra="forbid")

    success: bool                         # True if improvement was committed
    commit_sha: str | None = None         # git commit hash if successful
    commit_message: str = ""              # full commit message used
    files_changed: list[str] = []         # files modified by improvement
    new_findings: list[ImprovementArea] = []  # new improvements discovered
    error: str = ""                       # error message if failed
    tests_passed: bool = True             # whether verification passed
    verification_output: str = ""         # test/lint output summary


class ImproveResult(BaseModel):
    """Top-level result of an improve loop run."""

    model_config = ConfigDict(extra="forbid")

    improvements_completed: list[ImprovementArea]  # completed this run
    improvements_found: list[ImprovementArea]      # discovered this run
    improvements_skipped: list[ImprovementArea]    # marked stale/skipped
    improvements_failed: list[ImprovementArea]     # attempted but failed
    budget_remaining_seconds: float
    stopped_reason: Literal[
        "budget_exhausted",
        "max_improvements_reached",
        "no_more_improvements",
        "error",
    ]
    summary: str                           # human-readable summary
    run_record: RunRecord                  # this run's metadata


# ---------------------------------------------------------------------------
# Model Resolution
# ---------------------------------------------------------------------------

# Role keys for model resolution
_IMPROVE_ROLES: tuple[str, ...] = ("scanner_model", "executor_model", "validator_model")

_ROLE_KEY_MAP: dict[str, str] = {
    "scanner": "scanner_model",
    "executor": "executor_model",
    "validator": "validator_model",
}

_RUNTIME_DEFAULTS: dict[str, str] = {
    "claude_code": "sonnet",
    "open_code": "qwen/qwen-2.5-coder-32b-instruct",
}


def improve_resolve_models(config: ImproveConfig) -> dict[str, str]:
    """Resolve model strings for scanner, executor, validator roles.

    Resolution order (last wins):
      1. Runtime default (sonnet for claude_code, qwen for open_code)
      2. models["default"] — overrides all roles
      3. models["<role>"] — overrides a specific role

    Returns:
        Dict with keys: scanner_model, executor_model, validator_model

    Raises:
        ValueError: If models dict contains unknown keys.
    """
    runtime_default = _RUNTIME_DEFAULTS[config.runtime]
    resolved: dict[str, str] = {role: runtime_default for role in _IMPROVE_ROLES}

    if config.models:
        valid_keys = {"default"} | set(_ROLE_KEY_MAP.keys())
        for key in config.models:
            if key not in valid_keys:
                raise ValueError(
                    f"Unknown role key {key!r} in models dict. "
                    f"Valid keys are: {sorted(valid_keys)}"
                )

        if "default" in config.models:
            for role in _IMPROVE_ROLES:
                resolved[role] = config.models["default"]

        for role_key, resolved_key in _ROLE_KEY_MAP.items():
            if role_key in config.models:
                resolved[resolved_key] = config.models[role_key]

    return resolved
