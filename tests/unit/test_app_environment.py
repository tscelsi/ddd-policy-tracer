"""Tests for application environment-file loading."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ddd_policy_tracer.app import load_app_environment


def test_load_app_environment_reads_file_without_overriding_process_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load missing settings from .env while preserving explicit process configuration."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OLLAMA_HOST=http://dotenv.example\nOLLAMA_MODEL=dotenv-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OLLAMA_HOST", "http://process.example")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)

    load_app_environment(env_file)

    assert os.environ["OLLAMA_HOST"] == "http://process.example"
    assert os.environ["OLLAMA_MODEL"] == "dotenv-model"
