"""Tests for swe_af.execution.repo_url — URL parsing, normalization, and platform detection."""

from __future__ import annotations

import pytest

from swe_af.execution.repo_url import (
    GitPlatform,
    ParsedRepoURL,
    derive_repo_name,
    expand_shorthand,
    is_shorthand,
    normalize_repo_url,
    parse_repo_url,
)


# ---------------------------------------------------------------------------
# is_shorthand
# ---------------------------------------------------------------------------


class TestIsShorthand:
    def test_simple_owner_repo(self) -> None:
        assert is_shorthand("org/repo") is True

    def test_owner_with_dots(self) -> None:
        assert is_shorthand("my.org/my.repo") is True

    def test_owner_with_hyphens(self) -> None:
        assert is_shorthand("my-org/my-repo") is True

    def test_owner_with_underscores(self) -> None:
        assert is_shorthand("my_org/my_repo") is True

    def test_https_url_not_shorthand(self) -> None:
        assert is_shorthand("https://github.com/org/repo") is False

    def test_ssh_url_not_shorthand(self) -> None:
        assert is_shorthand("git@github.com:org/repo") is False

    def test_three_parts_not_shorthand(self) -> None:
        assert is_shorthand("a/b/c") is False

    def test_empty_not_shorthand(self) -> None:
        assert is_shorthand("") is False

    def test_single_word_not_shorthand(self) -> None:
        assert is_shorthand("repo") is False


# ---------------------------------------------------------------------------
# expand_shorthand
# ---------------------------------------------------------------------------


class TestExpandShorthand:
    def test_default_github(self) -> None:
        assert expand_shorthand("org/repo") == "https://github.com/org/repo"

    def test_custom_host(self) -> None:
        assert expand_shorthand("org/repo", "gitlab.com") == "https://gitlab.com/org/repo"

    def test_strips_whitespace(self) -> None:
        assert expand_shorthand("  org/repo  ") == "https://github.com/org/repo"

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            expand_shorthand("not-valid")

    def test_url_raises(self) -> None:
        with pytest.raises(ValueError):
            expand_shorthand("https://github.com/org/repo")


# ---------------------------------------------------------------------------
# parse_repo_url — HTTPS
# ---------------------------------------------------------------------------


class TestParseHTTPS:
    def test_github_https_with_git(self) -> None:
        parsed = parse_repo_url("https://github.com/org/my-project.git")
        assert parsed.host == "github.com"
        assert parsed.owner == "org"
        assert parsed.repo_name == "my-project"
        assert parsed.platform == GitPlatform.GITHUB
        assert parsed.protocol == "https"
        assert ".git" not in parsed.url

    def test_github_https_without_git(self) -> None:
        parsed = parse_repo_url("https://github.com/org/repo")
        assert parsed.repo_name == "repo"
        assert parsed.owner == "org"
        assert parsed.platform == GitPlatform.GITHUB

    def test_github_https_trailing_slash(self) -> None:
        parsed = parse_repo_url("https://github.com/org/repo/")
        assert parsed.repo_name == "repo"

    def test_gitlab_https(self) -> None:
        parsed = parse_repo_url("https://gitlab.com/org/repo.git")
        assert parsed.platform == GitPlatform.GITLAB
        assert parsed.repo_name == "repo"

    def test_bitbucket_https(self) -> None:
        parsed = parse_repo_url("https://bitbucket.org/org/repo")
        assert parsed.platform == GitPlatform.BITBUCKET
        assert parsed.repo_name == "repo"

    def test_http_url(self) -> None:
        parsed = parse_repo_url("http://github.example.com/org/repo.git")
        assert parsed.protocol == "http"
        assert parsed.repo_name == "repo"
        assert parsed.platform == GitPlatform.GITHUB  # github.* pattern

    def test_self_hosted_gitlab(self) -> None:
        parsed = parse_repo_url("https://gitlab.mycompany.com/team/project")
        assert parsed.platform == GitPlatform.GITLAB
        assert parsed.repo_name == "project"

    def test_unknown_host(self) -> None:
        parsed = parse_repo_url("https://mygit.example.com/org/repo")
        assert parsed.platform == GitPlatform.UNKNOWN
        assert parsed.repo_name == "repo"

    def test_host_lowercased(self) -> None:
        parsed = parse_repo_url("https://GitHub.COM/Org/Repo")
        assert parsed.host == "github.com"

    def test_codeberg(self) -> None:
        parsed = parse_repo_url("https://codeberg.org/user/project")
        assert parsed.platform == GitPlatform.GITEA
        assert parsed.repo_name == "project"


# ---------------------------------------------------------------------------
# parse_repo_url — SSH
# ---------------------------------------------------------------------------


class TestParseSSH:
    def test_github_ssh(self) -> None:
        parsed = parse_repo_url("git@github.com:org/repo.git")
        assert parsed.host == "github.com"
        assert parsed.owner == "org"
        assert parsed.repo_name == "repo"
        assert parsed.platform == GitPlatform.GITHUB
        assert parsed.protocol == "ssh"

    def test_gitlab_ssh(self) -> None:
        parsed = parse_repo_url("git@gitlab.com:org/repo.git")
        assert parsed.platform == GitPlatform.GITLAB
        assert parsed.repo_name == "repo"

    def test_bitbucket_ssh(self) -> None:
        parsed = parse_repo_url("git@bitbucket.org:org/repo.git")
        assert parsed.platform == GitPlatform.BITBUCKET

    def test_ssh_without_git_suffix(self) -> None:
        parsed = parse_repo_url("git@github.com:org/repo")
        assert parsed.repo_name == "repo"

    def test_ssh_nested_path(self) -> None:
        parsed = parse_repo_url("git@gitlab.com:group/subgroup/repo.git")
        assert parsed.repo_name == "repo"
        assert parsed.owner == "group/subgroup"


# ---------------------------------------------------------------------------
# parse_repo_url — Azure DevOps
# ---------------------------------------------------------------------------


class TestParseAzureDevOps:
    def test_azure_devops_standard(self) -> None:
        parsed = parse_repo_url("https://dev.azure.com/org/project/_git/repo")
        assert parsed.platform == GitPlatform.AZURE_DEVOPS
        assert parsed.owner == "org"
        assert parsed.repo_name == "repo"

    def test_azure_devops_with_git_suffix(self) -> None:
        parsed = parse_repo_url("https://dev.azure.com/org/project/_git/repo.git")
        assert parsed.repo_name == "repo"
        assert parsed.platform == GitPlatform.AZURE_DEVOPS

    def test_visualstudio_com(self) -> None:
        parsed = parse_repo_url("https://org.visualstudio.com/project/_git/repo")
        assert parsed.platform == GitPlatform.AZURE_DEVOPS
        assert parsed.repo_name == "repo"


# ---------------------------------------------------------------------------
# parse_repo_url — Shorthand
# ---------------------------------------------------------------------------


class TestParseShorthand:
    def test_owner_repo_expanded(self) -> None:
        parsed = parse_repo_url("org/my-project")
        assert parsed.url == "https://github.com/org/my-project"
        assert parsed.platform == GitPlatform.GITHUB
        assert parsed.repo_name == "my-project"
        assert parsed.owner == "org"

    def test_shorthand_preserves_names(self) -> None:
        parsed = parse_repo_url("my-org/my-repo")
        assert parsed.owner == "my-org"
        assert parsed.repo_name == "my-repo"


# ---------------------------------------------------------------------------
# parse_repo_url — Error cases
# ---------------------------------------------------------------------------


class TestParseErrors:
    def test_empty_string(self) -> None:
        with pytest.raises(ValueError):
            parse_repo_url("")

    def test_whitespace_only(self) -> None:
        with pytest.raises(ValueError):
            parse_repo_url("   ")

    def test_bare_word(self) -> None:
        with pytest.raises(ValueError):
            parse_repo_url("not-a-valid-url")

    def test_missing_repo_path(self) -> None:
        with pytest.raises(ValueError):
            parse_repo_url("https://github.com/only-one-part")


# ---------------------------------------------------------------------------
# normalize_repo_url
# ---------------------------------------------------------------------------


class TestNormalizeRepoURL:
    def test_strips_dot_git(self) -> None:
        result = normalize_repo_url("https://github.com/org/repo.git")
        assert result == "https://github.com/org/repo"

    def test_strips_trailing_slash(self) -> None:
        result = normalize_repo_url("https://github.com/org/repo/")
        assert result == "https://github.com/org/repo"

    def test_lowercases_host(self) -> None:
        result = normalize_repo_url("https://GitHub.COM/org/repo")
        assert result == "https://github.com/org/repo"

    def test_shorthand_expanded(self) -> None:
        result = normalize_repo_url("org/repo")
        assert result == "https://github.com/org/repo"

    def test_ssh_normalized(self) -> None:
        result = normalize_repo_url("git@github.com:org/repo.git")
        assert result == "git@github.com:org/repo"

    def test_idempotent(self) -> None:
        url = "https://github.com/org/repo"
        assert normalize_repo_url(url) == url


# ---------------------------------------------------------------------------
# derive_repo_name
# ---------------------------------------------------------------------------


class TestDeriveRepoName:
    def test_https_with_dot_git(self) -> None:
        assert derive_repo_name("https://github.com/org/my-project.git") == "my-project"

    def test_https_without_dot_git(self) -> None:
        assert derive_repo_name("https://github.com/org/repo") == "repo"

    def test_ssh_url(self) -> None:
        assert derive_repo_name("git@github.com:org/repo.git") == "repo"

    def test_empty_string(self) -> None:
        assert derive_repo_name("") == ""

    def test_shorthand(self) -> None:
        assert derive_repo_name("org/my-project") == "my-project"

    def test_azure_devops(self) -> None:
        assert derive_repo_name("https://dev.azure.com/org/project/_git/myrepo") == "myrepo"

    def test_trailing_slash(self) -> None:
        assert derive_repo_name("https://github.com/org/repo/") == "repo"

    def test_gitlab(self) -> None:
        assert derive_repo_name("https://gitlab.com/group/subgroup/repo.git") == "repo"


# ---------------------------------------------------------------------------
# Integration: RepoSpec accepts shorthand
# ---------------------------------------------------------------------------


class TestRepoSpecShorthand:
    def test_shorthand_expanded_in_repospec(self) -> None:
        from swe_af.execution.schemas import RepoSpec
        r = RepoSpec(repo_url="org/my-project", role="primary")
        assert r.repo_url == "https://github.com/org/my-project"

    def test_shorthand_in_build_config(self) -> None:
        from swe_af.execution.schemas import BuildConfig
        cfg = BuildConfig(repo_url="myorg/myrepo")
        assert cfg.repo_url == "https://github.com/myorg/myrepo"
        assert len(cfg.repos) == 1
        assert cfg.repos[0].repo_url == "https://github.com/myorg/myrepo"

    def test_existing_https_still_works(self) -> None:
        from swe_af.execution.schemas import RepoSpec
        r = RepoSpec(repo_url="https://github.com/org/repo.git", role="primary")
        assert r.repo_url == "https://github.com/org/repo.git"

    def test_existing_ssh_still_works(self) -> None:
        from swe_af.execution.schemas import RepoSpec
        r = RepoSpec(repo_url="git@github.com:org/repo.git", role="primary")
        assert r.repo_url == "git@github.com:org/repo.git"

    def test_invalid_url_still_raises(self) -> None:
        from swe_af.execution.schemas import RepoSpec
        from pydantic import ValidationError
        with pytest.raises((ValidationError, ValueError)):
            RepoSpec(repo_url="not-a-valid-url", role="primary")
