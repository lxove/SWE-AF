# PRD: Improve Repository URL Support

## Problem Statement

The current repo URL handling in SWE-AF has several limitations:

1. **Limited URL format support** — Only validates `http://`, `https://`, and `git@` prefixes; no support for `owner/repo` shorthand or platform-specific URL patterns.
2. **No platform detection** — The system cannot identify whether a URL points to GitHub, GitLab, Bitbucket, Azure DevOps, or a self-hosted instance.
3. **Duplicate URL parsing logic** — `_derive_repo_name()` in `swe_af/execution/schemas.py` and `_repo_name_from_url()` in `swe_af/fast/app.py` implement the same logic differently.
4. **No URL normalization** — Trailing slashes, `.git` suffixes, and protocol variations are not normalized consistently.
5. **GitHub-centric assumptions** — PR creation and git init prompts assume GitHub without detecting the actual platform.

## Goals

- Support `owner/repo` shorthand (expanded to GitHub HTTPS by default)
- Detect git hosting platform from URLs (GitHub, GitLab, Bitbucket, Azure DevOps, Gitea/Forgejo, self-hosted)
- Normalize URLs to a canonical form for deduplication and comparison
- Unify repo name extraction into a single shared utility
- Maintain full backward compatibility with existing configurations

## Non-Goals

- Implementing platform-specific PR creation for non-GitHub hosts (future work)
- SSH key management or credential handling
- URL reachability validation (network checks)

## Acceptance Criteria

1. `owner/repo` shorthand is accepted by `RepoSpec` and expanded to `https://github.com/owner/repo`
2. `parse_repo_url()` returns a structured `ParsedRepoURL` with host, owner, repo, platform fields
3. Platform detection correctly identifies: github.com, gitlab.com, bitbucket.org, dev.azure.com, and custom hosts
4. `normalize_repo_url()` produces canonical URLs (lowercase host, no trailing `.git`, no trailing slash)
5. `_derive_repo_name()` is replaced by the unified `parse_repo_url().repo_name` across all modules
6. `fast/app.py` uses the shared utility instead of its own regex
7. All existing tests continue to pass
8. New tests cover: shorthand expansion, platform detection, normalization, edge cases (empty, invalid)
9. Azure DevOps `dev.azure.com/org/project/_git/repo` URLs are parsed correctly

## Implementation Plan

### 1. New module: `swe_af/execution/repo_url.py`
- `ParsedRepoURL` dataclass with fields: `url`, `host`, `owner`, `repo_name`, `platform`, `protocol`
- `parse_repo_url(url: str) -> ParsedRepoURL` — main parser
- `normalize_repo_url(url: str) -> str` — canonical form
- `expand_shorthand(shorthand: str) -> str` — `owner/repo` → full URL
- Platform enum: `github`, `gitlab`, `bitbucket`, `azure_devops`, `gitea`, `unknown`

### 2. Update `swe_af/execution/schemas.py`
- Import and use `parse_repo_url` in `_derive_repo_name()`
- Update `RepoSpec._validate_repo_url()` to accept `owner/repo` shorthand
- Auto-expand shorthand in the validator

### 3. Update `swe_af/fast/app.py`
- Replace `_repo_name_from_url()` with import from `repo_url` module

### 4. Tests: `tests/test_repo_url.py`
- Comprehensive test coverage for all URL formats and edge cases
