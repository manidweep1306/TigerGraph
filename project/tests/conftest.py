"""
Conftest — sets LOG_DIR to a temp directory for tests so no prod logs are polluted.
"""
import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture(autouse=True)
def use_temp_logs(tmp_path, monkeypatch):
    """Redirect all log writes to tmp_path during tests."""
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    # Also patch the LOG_DIR in modules that use it at module level
    import backend.core.ledger as ledger_mod
    import backend.core.voi_scorer as voi_mod
    import backend.core.synthesis as synth_mod
    monkeypatch.setattr(ledger_mod, "LOG_DIR", tmp_path)
    monkeypatch.setattr(voi_mod, "LOG_DIR", tmp_path)
    monkeypatch.setattr(synth_mod, "LOG_DIR", tmp_path)
    yield
