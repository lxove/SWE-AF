"""Unit tests for swe_af.improve.prompts module."""

from __future__ import annotations

import pytest

from swe_af.improve.prompts import (
    EXECUTOR_SYSTEM_PROMPT,
    SCANNER_SYSTEM_PROMPT,
    VALIDATOR_SYSTEM_PROMPT,
    executor_task_prompt,
    scanner_task_prompt,
    validator_task_prompt,
)


class TestScannerSystemPrompt:
    """Tests for SCANNER_SYSTEM_PROMPT."""

    def test_contains_all_seven_categories(self):
        """SCANNER_SYSTEM_PROMPT should list all 7 improvement categories."""
        categories = [
            "test-coverage",
            "code-quality",
            "error-handling",
            "consistency",
            "dead-code",
            "performance",
            "documentation",
        ]
        for category in categories:
            assert category in SCANNER_SYSTEM_PROMPT, f"Missing category: {category}"

    def test_contains_what_to_avoid_section(self):
        """SCANNER_SYSTEM_PROMPT should include 'What to Avoid' guidance."""
        assert "What to Avoid" in SCANNER_SYSTEM_PROMPT
        assert "architectural rewrites" in SCANNER_SYSTEM_PROMPT.lower()
        assert "new features" in SCANNER_SYSTEM_PROMPT.lower()
        assert "style-only" in SCANNER_SYSTEM_PROMPT.lower()

    def test_references_scan_result_schema(self):
        """SCANNER_SYSTEM_PROMPT should reference ScanResult schema."""
        assert "ScanResult" in SCANNER_SYSTEM_PROMPT

    def test_provides_category_examples(self):
        """SCANNER_SYSTEM_PROMPT should provide examples for categories."""
        assert "Example:" in SCANNER_SYSTEM_PROMPT
        # Should have multiple examples
        assert SCANNER_SYSTEM_PROMPT.count("Example:") >= 7


class TestScannerTaskPrompt:
    """Tests for scanner_task_prompt function."""

    def test_output_contains_repo_path(self):
        """scanner_task_prompt output should include repo_path."""
        prompt = scanner_task_prompt(
            repo_path="/test/repo",
            scan_depth="normal",
            existing_improvements=[],
        )
        assert "/test/repo" in prompt

    def test_output_contains_scan_depth(self):
        """scanner_task_prompt output should include scan_depth value."""
        prompt = scanner_task_prompt(
            repo_path="/test/repo",
            scan_depth="thorough",
            existing_improvements=[],
        )
        assert "thorough" in prompt

    def test_output_contains_scan_depth_explanation(self):
        """scanner_task_prompt output should explain scan_depth levels."""
        prompt = scanner_task_prompt(
            repo_path="/test/repo",
            scan_depth="normal",
            existing_improvements=[],
        )
        assert "quick:" in prompt
        assert "normal:" in prompt
        assert "thorough:" in prompt

    def test_output_contains_existing_improvements(self):
        """scanner_task_prompt output should list existing improvement IDs."""
        existing = [
            {"id": "test-coverage-auth"},
            {"id": "dead-code-utils"},
        ]
        prompt = scanner_task_prompt(
            repo_path="/test/repo",
            scan_depth="normal",
            existing_improvements=existing,
        )
        assert "test-coverage-auth" in prompt
        assert "dead-code-utils" in prompt
        assert "Already-Known Improvements" in prompt

    def test_output_contains_category_filter(self):
        """scanner_task_prompt output should include category filter when provided."""
        prompt = scanner_task_prompt(
            repo_path="/test/repo",
            scan_depth="normal",
            existing_improvements=[],
            categories=["test-coverage", "documentation"],
        )
        assert "Category Filter" in prompt
        assert "test-coverage" in prompt
        assert "documentation" in prompt

    def test_output_without_category_filter(self):
        """scanner_task_prompt should work without category filter."""
        prompt = scanner_task_prompt(
            repo_path="/test/repo",
            scan_depth="normal",
            existing_improvements=[],
            categories=None,
        )
        # Should not have category filter section
        assert "Category Filter" not in prompt

    def test_output_is_non_empty(self):
        """scanner_task_prompt should return non-empty string."""
        prompt = scanner_task_prompt(
            repo_path="/test/repo",
            scan_depth="normal",
            existing_improvements=[],
        )
        assert len(prompt) > 0
        assert prompt.strip() != ""

    def test_references_scan_result_schema(self):
        """scanner_task_prompt should reference ScanResult schema."""
        prompt = scanner_task_prompt(
            repo_path="/test/repo",
            scan_depth="normal",
            existing_improvements=[],
        )
        assert "ScanResult" in prompt

    def test_truncates_long_existing_list(self):
        """scanner_task_prompt should truncate very long existing improvements list."""
        existing = [{"id": f"improvement-{i}"} for i in range(30)]
        prompt = scanner_task_prompt(
            repo_path="/test/repo",
            scan_depth="normal",
            existing_improvements=existing,
        )
        # Should show ellipsis for truncation
        assert "..." in prompt


class TestExecutorSystemPrompt:
    """Tests for EXECUTOR_SYSTEM_PROMPT."""

    def test_contains_commit_format(self):
        """EXECUTOR_SYSTEM_PROMPT should specify 'improve:' commit format."""
        assert "improve:" in EXECUTOR_SYSTEM_PROMPT

    def test_contains_commit_constraints(self):
        """EXECUTOR_SYSTEM_PROMPT should document commit message constraints."""
        assert "50 characters" in EXECUTOR_SYSTEM_PROMPT
        assert "lowercase" in EXECUTOR_SYSTEM_PROMPT.lower()
        assert "present tense" in EXECUTOR_SYSTEM_PROMPT.lower()
        assert "no period" in EXECUTOR_SYSTEM_PROMPT.lower()

    def test_provides_commit_examples(self):
        """EXECUTOR_SYSTEM_PROMPT should provide commit message examples."""
        assert "improve: add missing tests" in EXECUTOR_SYSTEM_PROMPT
        assert "improve: remove unused" in EXECUTOR_SYSTEM_PROMPT
        assert "improve: fix error" in EXECUTOR_SYSTEM_PROMPT

    def test_references_executor_result_schema(self):
        """EXECUTOR_SYSTEM_PROMPT should reference ExecutorResult schema."""
        assert "ExecutorResult" in EXECUTOR_SYSTEM_PROMPT


class TestExecutorTaskPrompt:
    """Tests for executor_task_prompt function."""

    def test_output_contains_improvement_details(self):
        """executor_task_prompt output should include improvement ID, category, title."""
        improvement = {
            "id": "test-auth-missing",
            "category": "test-coverage",
            "title": "Add tests for authentication",
            "priority": 2,
            "description": "Missing unit tests for auth module",
            "files": ["auth.py", "test_auth.py"],
        }
        prompt = executor_task_prompt(
            improvement=improvement,
            repo_path="/test/repo",
            timeout_seconds=300,
        )
        assert "test-auth-missing" in prompt
        assert "test-coverage" in prompt
        assert "Add tests for authentication" in prompt
        assert "2" in prompt  # priority

    def test_output_contains_repo_path(self):
        """executor_task_prompt output should include repo_path."""
        improvement = {"id": "test", "title": "Test"}
        prompt = executor_task_prompt(
            improvement=improvement,
            repo_path="/my/repo/path",
            timeout_seconds=300,
        )
        assert "/my/repo/path" in prompt

    def test_output_contains_timeout(self):
        """executor_task_prompt output should include timeout_seconds."""
        improvement = {"id": "test", "title": "Test"}
        prompt = executor_task_prompt(
            improvement=improvement,
            repo_path="/test/repo",
            timeout_seconds=180,
        )
        assert "180" in prompt

    def test_output_contains_description(self):
        """executor_task_prompt output should include improvement description."""
        improvement = {
            "id": "test",
            "title": "Test",
            "description": "This is a detailed description of the issue",
        }
        prompt = executor_task_prompt(
            improvement=improvement,
            repo_path="/test/repo",
            timeout_seconds=300,
        )
        assert "This is a detailed description of the issue" in prompt

    def test_output_contains_files_list(self):
        """executor_task_prompt output should list affected files."""
        improvement = {
            "id": "test",
            "title": "Test",
            "files": ["module_a.py", "module_b.py", "test_module.py"],
        }
        prompt = executor_task_prompt(
            improvement=improvement,
            repo_path="/test/repo",
            timeout_seconds=300,
        )
        assert "module_a.py" in prompt
        assert "module_b.py" in prompt
        assert "test_module.py" in prompt

    def test_output_is_non_empty(self):
        """executor_task_prompt should return non-empty string."""
        improvement = {"id": "test", "title": "Test"}
        prompt = executor_task_prompt(
            improvement=improvement,
            repo_path="/test/repo",
            timeout_seconds=300,
        )
        assert len(prompt) > 0
        assert prompt.strip() != ""

    def test_references_executor_result_schema(self):
        """executor_task_prompt should reference ExecutorResult schema."""
        improvement = {"id": "test", "title": "Test"}
        prompt = executor_task_prompt(
            improvement=improvement,
            repo_path="/test/repo",
            timeout_seconds=300,
        )
        assert "ExecutorResult" in prompt

    def test_handles_missing_optional_fields(self):
        """executor_task_prompt should handle missing optional fields gracefully."""
        improvement = {
            "id": "test-minimal",
            "title": "Minimal Test",
        }
        prompt = executor_task_prompt(
            improvement=improvement,
            repo_path="/test/repo",
            timeout_seconds=300,
        )
        # Should not crash and should still produce a valid prompt
        assert "test-minimal" in prompt
        assert "Minimal Test" in prompt


class TestValidatorSystemPrompt:
    """Tests for VALIDATOR_SYSTEM_PROMPT."""

    def test_contains_staleness_criteria(self):
        """VALIDATOR_SYSTEM_PROMPT should define staleness criteria."""
        assert "stale" in VALIDATOR_SYSTEM_PROMPT.lower()
        assert "deleted" in VALIDATOR_SYSTEM_PROMPT.lower()
        assert "30%" in VALIDATOR_SYSTEM_PROMPT
        assert "fixed" in VALIDATOR_SYSTEM_PROMPT.lower()

    def test_describes_file_deletion_criterion(self):
        """VALIDATOR_SYSTEM_PROMPT should mention file deletion as staleness criterion."""
        assert "deleted" in VALIDATOR_SYSTEM_PROMPT.lower() or "moved" in VALIDATOR_SYSTEM_PROMPT.lower()

    def test_describes_issue_fixed_criterion(self):
        """VALIDATOR_SYSTEM_PROMPT should mention issue already fixed as criterion."""
        assert "already fixed" in VALIDATOR_SYSTEM_PROMPT.lower()

    def test_describes_line_diff_criterion(self):
        """VALIDATOR_SYSTEM_PROMPT should mention >30% line diff as criterion."""
        assert "30%" in VALIDATOR_SYSTEM_PROMPT

    def test_describes_description_mismatch_criterion(self):
        """VALIDATOR_SYSTEM_PROMPT should mention description mismatch as criterion."""
        assert "description" in VALIDATOR_SYSTEM_PROMPT.lower()
        assert "no longer matches" in VALIDATOR_SYSTEM_PROMPT.lower() or "mismatch" in VALIDATOR_SYSTEM_PROMPT.lower()

    def test_references_validator_result_schema(self):
        """VALIDATOR_SYSTEM_PROMPT should reference ValidatorResult schema."""
        assert "ValidatorResult" in VALIDATOR_SYSTEM_PROMPT


class TestValidatorTaskPrompt:
    """Tests for validator_task_prompt function."""

    def test_output_contains_improvement_details(self):
        """validator_task_prompt output should include improvement ID, category, title."""
        improvement = {
            "id": "stale-check-test",
            "category": "code-quality",
            "title": "Refactor complex function",
            "description": "Function has cyclomatic complexity of 15",
            "files": ["complex_module.py"],
            "found_by_run": "2025-03-15T10:00:00Z",
        }
        prompt = validator_task_prompt(
            improvement=improvement,
            repo_path="/test/repo",
        )
        assert "stale-check-test" in prompt
        assert "code-quality" in prompt
        assert "Refactor complex function" in prompt
        assert "2025-03-15T10:00:00Z" in prompt

    def test_output_contains_files_to_check(self):
        """validator_task_prompt output should list files to check."""
        improvement = {
            "id": "test",
            "title": "Test",
            "files": ["file_a.py", "file_b.py"],
        }
        prompt = validator_task_prompt(
            improvement=improvement,
            repo_path="/test/repo",
        )
        assert "file_a.py" in prompt
        assert "file_b.py" in prompt

    def test_output_contains_repo_path(self):
        """validator_task_prompt output should include repo_path."""
        improvement = {"id": "test", "title": "Test"}
        prompt = validator_task_prompt(
            improvement=improvement,
            repo_path="/validator/repo",
        )
        assert "/validator/repo" in prompt

    def test_output_is_non_empty(self):
        """validator_task_prompt should return non-empty string."""
        improvement = {"id": "test", "title": "Test"}
        prompt = validator_task_prompt(
            improvement=improvement,
            repo_path="/test/repo",
        )
        assert len(prompt) > 0
        assert prompt.strip() != ""

    def test_references_validator_result_schema(self):
        """validator_task_prompt should reference ValidatorResult schema."""
        improvement = {"id": "test", "title": "Test"}
        prompt = validator_task_prompt(
            improvement=improvement,
            repo_path="/test/repo",
        )
        assert "ValidatorResult" in prompt

    def test_handles_missing_optional_fields(self):
        """validator_task_prompt should handle missing optional fields gracefully."""
        improvement = {
            "id": "minimal-validator-test",
            "title": "Minimal",
        }
        prompt = validator_task_prompt(
            improvement=improvement,
            repo_path="/test/repo",
        )
        # Should not crash
        assert "minimal-validator-test" in prompt


class TestPromptKeywords:
    """Tests to verify all prompts contain required keywords."""

    def test_scanner_contains_categories_keyword(self):
        """Scanner prompts should mention 'categories'."""
        assert "categories" in SCANNER_SYSTEM_PROMPT.lower()
        prompt = scanner_task_prompt(
            repo_path="/test",
            scan_depth="normal",
            existing_improvements=[],
        )
        assert "categories" in prompt.lower()

    def test_executor_contains_improve_keyword(self):
        """Executor prompts should mention 'improve:' commit format."""
        assert "improve:" in EXECUTOR_SYSTEM_PROMPT
        prompt = executor_task_prompt(
            improvement={"id": "test", "title": "test"},
            repo_path="/test",
            timeout_seconds=100,
        )
        assert "improve:" in prompt

    def test_all_system_prompts_non_empty(self):
        """All system prompts should be non-empty."""
        assert len(SCANNER_SYSTEM_PROMPT) > 0
        assert len(EXECUTOR_SYSTEM_PROMPT) > 0
        assert len(VALIDATOR_SYSTEM_PROMPT) > 0

    def test_all_schema_references_present(self):
        """All prompts should reference their respective schema names."""
        assert "ScanResult" in SCANNER_SYSTEM_PROMPT
        assert "ExecutorResult" in EXECUTOR_SYSTEM_PROMPT
        assert "ValidatorResult" in VALIDATOR_SYSTEM_PROMPT
