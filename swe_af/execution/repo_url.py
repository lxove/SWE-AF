"""Repository URL parsing, normalization, and platform detection.

Provides a unified utility for handling git repository URLs across the
SWE-AF codebase, replacing ad-hoc regex parsing scattered across modules.

Usage::

    from swe_af.execution.repo_url import parse_repo_url, normalize_repo_url

    parsed = parse_repo_url("https://github.com/org/my-project.git")
    parsed.repo_name   # "my-project"
    parsed.owner       # "org"
    parsed.platform    # "github"

    # Shorthand expansion
    parsed = parse_repo_url("org/my-project")
    parsed.url         # "https://github.com/org/my-project"
    parsed.platform    # "github"

    # Normalization
    normalize_repo_url("https://GitHub.com/org/repo.git/")
    # "https://github.com/org/repo"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class GitPlatform(str, Enum):
    """Known git hosting platforms."""

    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    AZURE_DEVOPS = "azure_devops"
    GITEA = "gitea"
    UNKNOWN = "unknown"


# Mapping from host patterns to platform identifiers.
# Order matters: first match wins.
_HOST_PLATFORM_MAP: list[tuple[str, GitPlatform]] = [
    ("github.com", GitPlatform.GITHUB),
    ("github.", GitPlatform.GITHUB),       # github.example.com (GHE)
    ("gitlab.com", GitPlatform.GITLAB),
    ("gitlab.", GitPlatform.GITLAB),       # self-hosted GitLab
    ("bitbucket.org", GitPlatform.BITBUCKET),
    ("bitbucket.", GitPlatform.BITBUCKET),
    ("dev.azure.com", GitPlatform.AZURE_DEVOPS),
    ("visualstudio.com", GitPlatform.AZURE_DEVOPS),
    ("gitea.", GitPlatform.GITEA),
    ("forgejo.", GitPlatform.GITEA),
    ("codeberg.org", GitPlatform.GITEA),
]

# Regex for owner/repo shorthand (e.g. "org/my-project")
_SHORTHAND_RE = re.compile(
    r"^(?P<owner>[a-zA-Z0-9._-]+)/(?P<repo>[a-zA-Z0-9._-]+)$"
)

# Regex for SSH URLs: git@host:owner/repo.git
_SSH_RE = re.compile(
    r"^(?P<user>[a-zA-Z0-9._-]+)@(?P<host>[^:]+):(?P<path>.+)$"
)

# Regex for HTTPS/HTTP URLs
_HTTPS_RE = re.compile(
    r"^(?P<proto>https?://)(?P<host>[^/]+)/(?P<path>.+)$"
)


@dataclass(frozen=True)
class ParsedRepoURL:
    """Structured representation of a parsed git repository URL."""

    url: str
    """The (possibly expanded/normalized) full URL."""

    host: str
    """The hostname, e.g. 'github.com'."""

    owner: str
    """The repository owner or organization."""

    repo_name: str
    """The repository name (without .git suffix)."""

    platform: GitPlatform
    """Detected git hosting platform."""

    protocol: str
    """'https', 'http', or 'ssh'."""


def _detect_platform(host: str) -> GitPlatform:
    """Detect the git platform from a hostname."""
    host_lower = host.lower()
    for pattern, platform in _HOST_PLATFORM_MAP:
        if host_lower == pattern or host_lower.startswith(pattern) or host_lower.endswith("." + pattern):
            return platform
    return GitPlatform.UNKNOWN


def _strip_git_suffix(name: str) -> str:
    """Remove trailing .git from a name."""
    if name.endswith(".git"):
        return name[:-4]
    return name


def expand_shorthand(shorthand: str, default_host: str = "github.com") -> str:
    """Expand ``owner/repo`` shorthand to a full HTTPS URL.

    Args:
        shorthand: A string like ``"org/my-project"``.
        default_host: The host to use for expansion. Defaults to ``github.com``.

    Returns:
        A full HTTPS URL, e.g. ``"https://github.com/org/my-project"``.

    Raises:
        ValueError: If the string is not a valid ``owner/repo`` shorthand.
    """
    match = _SHORTHAND_RE.match(shorthand.strip())
    if not match:
        raise ValueError(f"Not a valid owner/repo shorthand: {shorthand!r}")
    owner = match.group("owner")
    repo = match.group("repo")
    return f"https://{default_host}/{owner}/{repo}"


def is_shorthand(url: str) -> bool:
    """Return True if the string looks like an ``owner/repo`` shorthand."""
    return bool(_SHORTHAND_RE.match(url.strip()))


def parse_repo_url(url: str) -> ParsedRepoURL:
    """Parse a git repository URL into its components.

    Supports:
    - HTTPS URLs: ``https://github.com/org/repo.git``
    - HTTP URLs: ``http://gitlab.example.com/org/repo``
    - SSH URLs: ``git@github.com:org/repo.git``
    - Azure DevOps: ``https://dev.azure.com/org/project/_git/repo``
    - Shorthand: ``org/repo`` (expanded to GitHub HTTPS)

    Args:
        url: The repository URL or shorthand string.

    Returns:
        A :class:`ParsedRepoURL` with parsed components.

    Raises:
        ValueError: If the URL cannot be parsed.
    """
    if not url or not url.strip():
        raise ValueError("Repository URL cannot be empty")

    url = url.strip()

    # Check for owner/repo shorthand first
    if is_shorthand(url):
        full_url = expand_shorthand(url)
        return parse_repo_url(full_url)

    # Try SSH format: git@host:path
    ssh_match = _SSH_RE.match(url)
    if ssh_match:
        host = ssh_match.group("host")
        path = _strip_git_suffix(ssh_match.group("path").rstrip("/"))
        parts = path.split("/")
        if len(parts) < 2:
            raise ValueError(f"Cannot extract owner/repo from SSH URL: {url!r}")
        owner = "/".join(parts[:-1])
        repo_name = parts[-1]
        platform = _detect_platform(host)
        normalized = f"{ssh_match.group('user')}@{host.lower()}:{owner}/{repo_name}"
        return ParsedRepoURL(
            url=normalized,
            host=host.lower(),
            owner=owner,
            repo_name=repo_name,
            platform=platform,
            protocol="ssh",
        )

    # Try HTTPS/HTTP format
    https_match = _HTTPS_RE.match(url)
    if https_match:
        proto = https_match.group("proto")  # "https://" or "http://"
        host = https_match.group("host")
        path = _strip_git_suffix(https_match.group("path").rstrip("/"))

        # Azure DevOps special handling: dev.azure.com/org/project/_git/repo
        if "dev.azure.com" in host.lower() or "visualstudio.com" in host.lower():
            parts = path.split("/")
            # Pattern: org/project/_git/repo
            if len(parts) >= 4 and parts[-2] == "_git":
                owner = parts[0]
                repo_name = parts[-1]
            elif len(parts) >= 2:
                owner = parts[0]
                repo_name = parts[-1]
            else:
                raise ValueError(f"Cannot parse Azure DevOps URL: {url!r}")
        else:
            parts = path.split("/")
            if len(parts) < 2:
                raise ValueError(f"Cannot extract owner/repo from URL: {url!r}")
            owner = "/".join(parts[:-1])
            repo_name = parts[-1]

        protocol = "https" if proto.startswith("https") else "http"
        platform = _detect_platform(host)
        normalized = f"{proto.lower()}{host.lower()}/{path}"
        return ParsedRepoURL(
            url=normalized,
            host=host.lower(),
            owner=owner,
            repo_name=repo_name,
            platform=platform,
            protocol=protocol,
        )

    raise ValueError(
        f"Cannot parse repository URL: {url!r}. "
        f"Expected HTTPS, HTTP, SSH (git@), or owner/repo shorthand."
    )


def normalize_repo_url(url: str) -> str:
    """Normalize a repository URL to a canonical form.

    - Lowercase hostname
    - Remove trailing ``.git``
    - Remove trailing slashes
    - Expand shorthand to full URL

    Args:
        url: A repository URL or shorthand.

    Returns:
        The normalized URL string.

    Raises:
        ValueError: If the URL cannot be parsed.
    """
    parsed = parse_repo_url(url)
    return parsed.url


def derive_repo_name(url: str) -> str:
    """Extract the repository name from a git URL.

    This is the unified replacement for ``_derive_repo_name()`` in schemas.py
    and ``_repo_name_from_url()`` in fast/app.py.

    Args:
        url: A git repository URL or shorthand.

    Returns:
        The repository name (e.g. ``"my-project"``).
        Returns ``""`` if the URL is empty.
    """
    if not url or not url.strip():
        return ""
    try:
        parsed = parse_repo_url(url)
        return parsed.repo_name
    except ValueError:
        # Fallback: strip .git and take last path component
        stripped = re.sub(r"\.git$", "", url.rstrip("/"))
        name = re.split(r"[/:]", stripped)[-1]
        return name
