"""Unit tests for swe_af.improve module initialization."""

from __future__ import annotations

import sys

import pytest
from agentfield import AgentRouter


def test_improve_router_exists():
    """Test that improve_router is an AgentRouter instance."""
    from swe_af.improve import improve_router

    assert isinstance(improve_router, AgentRouter)


def test_improve_router_has_correct_tags():
    """Test that improve_router has the 'swe-improve' tag."""
    from swe_af.improve import improve_router

    assert "swe-improve" in improve_router.tags


def test_improve_router_in_all():
    """Test that improve_router is exported in __all__."""
    import swe_af.improve

    assert "improve_router" in swe_af.improve.__all__


def test_no_planning_agent_imports():
    """Test that importing swe_af.improve does not import planning agents."""
    # Clear any previous imports
    modules_to_clear = [k for k in sys.modules if k.startswith("swe_af.improve")]
    for mod in modules_to_clear:
        del sys.modules[mod]

    # Import the module
    import swe_af.improve

    # Verify planning agent modules are not loaded
    assert "swe_af.reasoners.pipeline" not in sys.modules


def test_reasoners_registered_after_import():
    """Test that scanner, validator, executor reasoners are registered after import."""
    from swe_af.improve import improve_router

    # The reasoners should be registered after the lazy imports
    # The actual reasoner registration happens in scanner.py, validator.py, executor.py
    assert hasattr(improve_router, "reasoners")
    assert isinstance(improve_router.reasoners, list)

    # Verify that three reasoners are registered (scanner, validator, executor)
    assert len(improve_router.reasoners) == 3

    # Verify the reasoner function names
    reasoner_names = {r["func"].__name__ for r in improve_router.reasoners}
    assert "scan_for_improvements" in reasoner_names
    assert "validate_improvement" in reasoner_names
    assert "execute_improvement" in reasoner_names


def test_module_can_be_imported_multiple_times():
    """Test that the module can be imported multiple times without errors."""
    from swe_af.improve import improve_router as router1

    from swe_af.improve import improve_router as router2

    # Should be the same instance
    assert router1 is router2


def test_import_from_swe_af_improve():
    """Test that improve_router can be imported directly from swe_af.improve."""
    from swe_af.improve import improve_router

    assert improve_router is not None
    assert isinstance(improve_router, AgentRouter)
    assert "swe-improve" in improve_router.tags
