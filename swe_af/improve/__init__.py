"""swe_af.improve — Continuous recursive improvement node.

Exports
-------
improve_router : AgentRouter
    Router tagged 'swe-improve' with scanner, validator, executor reasoners.

Intentionally does NOT import swe_af.reasoners.pipeline to avoid loading
planning agents into this process.
"""

from __future__ import annotations

from agentfield import AgentRouter

improve_router = AgentRouter(tags=["swe-improve"])

# Import submodules to register reasoners on improve_router
from . import scanner   # noqa: E402, F401 — registers scan_for_improvements
from . import validator # noqa: E402, F401 — registers validate_improvement
from . import executor  # noqa: E402, F401 — registers execute_improvement

__all__ = ["improve_router"]
