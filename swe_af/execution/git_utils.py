"""Lightweight git helper utilities shared across entry points."""

from __future__ import annotations

import subprocess


def detect_remote_default_branch(repo_path: str) -> str:
    """Detect the default branch of the remote 'origin' for a local repo.

    Tries ``git symbolic-ref refs/remotes/origin/HEAD`` first (fast, local),
    then falls back to ``git remote show origin`` (network call).

    Returns the branch name (e.g. ``"main"``, ``"master"``, ``"develop"``),
    or an empty string if detection fails.
    """
    # Fast path: local symbolic ref (works if repo was cloned normally)
    try:
        r = subprocess.run(
            ["git", "-C", repo_path, "symbolic-ref", "refs/remotes/origin/HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            # Output looks like "refs/remotes/origin/main"
            return r.stdout.strip().split("/")[-1]
    except Exception:
        pass

    # Slow path: ask the remote directly
    try:
        r = subprocess.run(
            ["git", "-C", repo_path, "remote", "show", "origin"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if r.returncode == 0:
            for line in r.stdout.splitlines():
                line = line.strip()
                if line.startswith("HEAD branch:"):
                    branch = line.split(":", 1)[1].strip()
                    if branch and branch != "(unknown)":
                        return branch
    except Exception:
        pass

    return ""
