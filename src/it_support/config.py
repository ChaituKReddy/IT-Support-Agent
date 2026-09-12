"""Application settings, loaded from environment variables or a .env file.

Every tunable value in the project is defined here so that no module needs a
hard-coded path, model name, or threshold.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration for the assistant."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ------------------------------------------------------
    llm_provider: Literal["ollama", "anthropic"] = "ollama"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b-mlx"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5"

    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: int = 120

    # --- Storage -----------------------------------------------------------
    db_path: Path = Path("data/it_support.db")
    seed_dir: Path = Path("data/seed")
    checkpoint_db_path: Path = Path("data/checkpoints.db")

    # --- Behaviour ---------------------------------------------------------
    kb_search_top_k: int = Field(default=3, ge=1, le=10)
    duplicate_ticket_window_days: int = Field(default=7, ge=0)
    max_agent_steps: int = Field(default=6, ge=1)
    log_level: str = "INFO"

    def resolve(self, value: Path) -> Path:
        """Turn a possibly relative configured path into an absolute one."""
        return value if value.is_absolute() else PROJECT_ROOT / value

    @property
    def database_file(self) -> Path:
        return self.resolve(self.db_path)

    @property
    def seed_directory(self) -> Path:
        return self.resolve(self.seed_dir)

    @property
    def checkpoint_file(self) -> Path:
        return self.resolve(self.checkpoint_db_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
