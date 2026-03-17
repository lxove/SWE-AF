# Goal: Add repo_url Clone Support to swe-improve

The `swe_af/improve/` module currently only accepts `repo_path` (a local directory). On Railway, users send `repo_url` (a GitHub URL) via the control plane. The planner (`swe_af/app.py`) handles this with inline clone logic (lines 210-279), but `swe-improve` cannot. Extract the clone/reset logic into a shared utility and wire it into both `build()` and `improve()`.

## What It Does

1. **Shared clone utility**: A new `swe_af/git_utils.py` module that extracts the clone-or-reset logic from `swe_af/app.py:210-279` into reusable functions.
2. **improve() accepts repo_url**: The `improve` reasoner in `swe_af/improve/app.py` gains `repo_url` and `branch` parameters. When `repo_url` is provided, it clones/resets the repo before entering the improvement loop.
3. **build() refactored**: The inline clone block in `swe_af/app.py` is replaced with a call to the shared utility. No behavior change.

## Architecture

### Files to Create

- `swe_af/git_utils.py` — Shared clone/reset utility extracted from `swe_af/app.py`
- `tests/test_git_utils.py` — Unit tests for the shared clone utility

### Files to Modify

- `swe_af/improve/app.py` — Add `repo_url` and `branch` params to `improve()`, call shared clone utility before state loading
- `swe_af/app.py` — Replace inline clone logic (lines 210-279) with call to shared `git_utils`

### Shared Utility (`swe_af/git_utils.py`)

```python
import os
import shutil
import subprocess
from swe_af.execution.schemas import _derive_repo_name


def resolve_repo_path(repo_url: str, repo_path: str = "") -> str:
    """Derive repo_path from repo_url if not explicitly provided.

    If repo_path is already set, returns it as-is.
    Otherwise derives /workspaces/<repo-name> from the URL.

    Args:
        repo_url: GitHub clone URL.
        repo_path: Explicit local path (takes precedence).

    Returns:
        Resolved absolute path for the repository.
    """
    if repo_path:
        return repo_path
    if not repo_url:
        raise ValueError("Either repo_url or repo_path must be provided")
    name = _derive_repo_name(repo_url)
    if not name:
        raise ValueError(f"Cannot derive repo name from URL: {repo_url}")
    return f"/workspaces/{name}"


def ensure_repo_cloned(
    repo_url: str,
    repo_path: str,
    branch: str = "",
    default_branch: str = "main",
    logger=None,
) -> str:
    """Clone or reset a repository to a clean state.

    Behavior:
    - If no .git dir exists at repo_path: clone repo_url into repo_path
      (optionally checking out `branch` if provided).
    - If .git dir exists: fetch origin, then hard-reset to origin/{default_branch}.
    - If reset fails: nuke repo_path and re-clone as last resort.
    - If repo_url is empty: just ensure repo_path directory exists (no-op for local repos).

    Args:
        repo_url: Git clone URL. If empty, only ensures repo_path exists.
        repo_path: Local path for the repository.
        branch: Optional branch to clone (--branch flag). Only used on fresh clone.
        default_branch: Branch to reset to when repo already exists. Defaults to "main".
        logger: Optional callable(msg, tags) for logging (e.g. app.note).

    Returns:
        The repo_path (unchanged, for convenience).
    """

    def _log(msg: str, tags: list[str] | None = None):
        if logger:
            logger(msg, tags=tags or [])

    if not repo_url:
        # No URL — just ensure directory exists for local-path usage
        os.makedirs(repo_path, exist_ok=True)
        return repo_path

    git_dir = os.path.join(repo_path, ".git")

    if not os.path.exists(git_dir):
        # Fresh clone
        _log(f"Cloning {repo_url} → {repo_path}", ["clone"])
        os.makedirs(repo_path, exist_ok=True)
        cmd = ["git", "clone"]
        if branch:
            cmd += ["--branch", branch]
        cmd += [repo_url, repo_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = result.stderr.strip()
            _log(f"Clone failed (exit {result.returncode}): {err}", ["clone", "error"])
            raise RuntimeError(f"git clone failed (exit {result.returncode}): {err}")
    else:
        # Repo exists — reset to clean state
        _log(
            f"Repo already exists at {repo_path} — resetting to origin/{default_branch}",
            ["clone", "reset"],
        )

        # Remove stale worktrees
        worktrees_dir = os.path.join(repo_path, ".worktrees")
        if os.path.isdir(worktrees_dir):
            shutil.rmtree(worktrees_dir, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=repo_path, capture_output=True, text=True,
        )

        # Fetch latest remote state
        fetch = subprocess.run(
            ["git", "fetch", "origin"],
            cwd=repo_path, capture_output=True, text=True,
        )
        if fetch.returncode != 0:
            _log(f"git fetch failed: {fetch.stderr.strip()}", ["clone", "error"])

        # Force-checkout default branch
        subprocess.run(
            ["git", "checkout", "-f", default_branch],
            cwd=repo_path, capture_output=True, text=True,
        )
        reset = subprocess.run(
            ["git", "reset", "--hard", f"origin/{default_branch}"],
            cwd=repo_path, capture_output=True, text=True,
        )
        if reset.returncode != 0:
            # Hard reset failed — nuke and re-clone
            _log(
                f"Reset to origin/{default_branch} failed — re-cloning",
                ["clone", "reclone"],
            )
            shutil.rmtree(repo_path, ignore_errors=True)
            os.makedirs(repo_path, exist_ok=True)
            clone_result = subprocess.run(
                ["git", "clone", repo_url, repo_path],
                capture_output=True, text=True,
            )
            if clone_result.returncode != 0:
                err = clone_result.stderr.strip()
                raise RuntimeError(f"git re-clone failed: {err}")

    return repo_path
```

### Changes to `swe_af/improve/app.py`

Update the `improve` reasoner signature and add clone logic before state loading:

```python
@app.reasoner()
async def improve(
    repo_path: str = "",
    repo_url: str = "",
    branch: str = "",
    config: dict | None = None,
) -> dict:
    """Continuous recursive improvement loop."""
    cfg = ImproveConfig(**(config or {}))

    # Resolve repo_path from repo_url if needed
    from swe_af.git_utils import resolve_repo_path, ensure_repo_cloned

    if repo_url:
        repo_path = resolve_repo_path(repo_url, repo_path)
        ensure_repo_cloned(
            repo_url=repo_url,
            repo_path=repo_path,
            branch=branch,
            logger=lambda msg, tags=None: app.note(msg, tags=["improve"] + (tags or [])),
        )

    # Validate repo_path
    if not repo_path or not os.path.isdir(repo_path):
        return ImproveResult(
            # ... error response (same as current)
        ).model_dump()

    # ... rest of loop unchanged
```

Key points:
- `repo_path` default changes from required to `""` (empty string)
- `repo_url` and `branch` are new optional params
- Clone happens before the `os.path.isdir(repo_path)` validation check
- If only `repo_path` is given, behavior is identical to current (backward compat)

### Changes to `swe_af/app.py`

Replace the inline clone block (lines 210-279) with:

```python
from swe_af.git_utils import ensure_repo_cloned

# Clone if repo_url is set, or ensure repo_path exists
ensure_repo_cloned(
    repo_url=cfg.repo_url,
    repo_path=repo_path,
    default_branch=cfg.github_pr_base or "main",
    logger=lambda msg, tags=None: app.note(msg, tags=["build"] + (tags or [])),
)
```

This replaces lines 210-279 entirely. The `resolve_repo_path` call is not needed here because `swe_af/app.py` already resolves `repo_path` from `cfg.repo_url` on line 200 using `_repo_name_from_url`.

### Unit Tests (`tests/test_git_utils.py`)

Test the following scenarios:

1. **`resolve_repo_path`**:
   - Returns `repo_path` unchanged when provided
   - Derives `/workspaces/<name>` from HTTPS URL
   - Derives `/workspaces/<name>` from SSH URL
   - Raises `ValueError` when neither `repo_url` nor `repo_path` given
   - Raises `ValueError` when URL yields empty name

2. **`ensure_repo_cloned` — fresh clone**:
   - Calls `git clone` when no `.git` dir exists
   - Passes `--branch` flag when `branch` is provided
   - Raises `RuntimeError` on clone failure

3. **`ensure_repo_cloned` — existing repo reset**:
   - Fetches, checks out, and resets when `.git` exists
   - Removes stale `.worktrees` directory
   - Falls back to nuke + re-clone when reset fails

4. **`ensure_repo_cloned` — no URL (local path)**:
   - Creates directory if needed, does not run any git commands
   - Returns repo_path as-is

5. **Logger integration**:
   - Calls logger with appropriate messages and tags
   - Works fine when logger is None

Use `unittest.mock.patch` to mock `subprocess.run` and `os.path.exists` / `os.path.isdir`. Use `tmp_path` (pytest fixture) for filesystem tests where needed.

## Acceptance Criteria

1. `swe-improve.improve` accepts `repo_url` and clones the repo before running the improvement loop
2. `repo_path` still works standalone without `repo_url` (backward compatible, no behavior change)
3. Re-runs with the same `repo_url` reset the repo to a clean state (fetch + hard reset to default branch)
4. `swe-planner.build` in `swe_af/app.py` uses the same shared `git_utils` utility (no behavior change)
5. Unit tests cover `resolve_repo_path` and `ensure_repo_cloned` (fresh clone, reset, nuke+reclone, local-only, logger)

## Important Context for the Agents

- The shared utility lives in `swe_af/git_utils.py` — a new top-level module, not inside `improve/` or `execution/`.
- `_derive_repo_name` is imported from `swe_af/execution/schemas.py` — it already exists and is tested.
- The `branch` parameter is only used on fresh clones (`--branch` flag). On reset, the repo resets to `origin/{default_branch}`.
- `default_branch` in `improve()` defaults to `"main"`. The `build()` function uses `cfg.github_pr_base or "main"`.
- Do NOT change the `improve` reasoner's loop logic — only add the clone preamble before state loading.
- The logger pattern uses `app.note(msg, tags=[...])` — pass a lambda wrapper so `git_utils` stays decoupled from the Agent instance.
- Keep the `swe_af/app.py` refactor minimal: replace the inline block with one function call, keep surrounding code unchanged.
