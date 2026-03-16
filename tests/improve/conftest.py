"""Shared fixtures for swe_af.improve tests.

Problem: importing improve-based modules can cause state pollution across tests
when reasoners are registered on improve_router. Similar to tests/fast/conftest.py,
tests that inspect improve_router.reasoners need a fresh router instance before
each test.

Fix: a function-scoped autouse fixture that reloads swe_af.improve and its
reasoner sub-modules before every test, producing a fresh improve_router whose
reasoners still carry their original names. The swe_af.improve.app module (if
it exists) is NOT reloaded because reloading it would call include_router again
and immediately re-mangle the fresh router.
"""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture(autouse=True)
def _reset_improve_router() -> None:  # type: ignore[return]
    """Reload swe_af.improve to get fresh router state for each test.

    This fixture prevents cross-test pollution by:
    1. Saving the current state of swe_af.improve.* modules (except app)
    2. Removing them from sys.modules
    3. Re-importing swe_af.improve to create a fresh improve_router
    4. Re-importing scanner, validator, executor to re-register reasoners
    5. Restoring the saved modules after the test completes

    This is a no-op until swe_af.improve.app has been imported for the first
    time. After that it becomes necessary to restore the clean state that
    tests expect.
    """
    # Only reload when swe_af.improve.app is already cached (i.e. include_router
    # has already been called and may have mangled improve_router.reasoners).
    if "swe_af.improve.app" not in sys.modules:
        yield
        return

    # Save and evict all swe_af.improve.* sub-modules EXCEPT app itself.
    sub_keys = [
        k for k in list(sys.modules)
        if k.startswith("swe_af.improve") and k != "swe_af.improve.app"
    ]
    saved = {k: sys.modules.pop(k) for k in sub_keys}

    try:
        # Re-import swe_af.improve — this recreates improve_router and re-registers
        # all the @improve_router.reasoner() wrappers with original func references.
        importlib.import_module("swe_af.improve")
        # Re-import sub-modules that register reasoners on the fresh improve_router.
        for mod in (
            "swe_af.improve.scanner",
            "swe_af.improve.validator",
            "swe_af.improve.executor",
        ):
            try:
                importlib.import_module(mod)
            except ImportError:
                pass

        yield
    finally:
        # After the test, evict the freshly-loaded sub-modules so that the
        # next test gets a clean reload too.
        for k in list(sys.modules):
            if k.startswith("swe_af.improve") and k != "swe_af.improve.app":
                sys.modules.pop(k, None)
        # Restore the saved copies so the process stays consistent.
        sys.modules.update(saved)
