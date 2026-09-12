"""Shared fixtures: every test runs against a throwaway database."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from it_support import db
from it_support.config import PROJECT_ROOT, Settings, get_settings
from it_support.tools import knowledge


@pytest.fixture(autouse=True)
def temporary_database(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Point the whole application at a fresh database seeded from sample data."""
    seed_source = PROJECT_ROOT / "data" / "seed"
    seed_target = tmp_path / "seed"
    shutil.copytree(seed_source, seed_target)

    settings = Settings(
        db_path=tmp_path / "test.db",
        seed_dir=seed_target,
        checkpoint_db_path=tmp_path / "checkpoints.db",
    )

    get_settings.cache_clear()
    monkeypatch.setattr("it_support.config.get_settings", lambda: settings)
    monkeypatch.setattr("it_support.db.get_settings", lambda: settings)
    monkeypatch.setattr("it_support.tools.knowledge.get_settings", lambda: settings)
    monkeypatch.setattr("it_support.tools.tickets.get_settings", lambda: settings)
    monkeypatch.setattr("it_support.graph.get_settings", lambda: settings)

    db.reset_initialisation_cache()
    knowledge.clear_index_cache()
    db.initialise(settings)

    yield settings

    db.reset_initialisation_cache()
    knowledge.clear_index_cache()
    get_settings.cache_clear()
