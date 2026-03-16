from __future__ import annotations

from pathlib import Path

from ai_export_builder.config import Settings


def test_dotenv_overrides_process_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-proj-localQUA\n", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-globalOLD")

    settings = Settings(_env_file=str(env_file))

    assert settings.openai_api_key == "sk-proj-localQUA"


def test_process_environment_used_when_dotenv_missing(monkeypatch):
    missing_env_file = Path("does-not-exist.env")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-globalONLY")

    settings = Settings(_env_file=str(missing_env_file))

    assert settings.openai_api_key == "sk-proj-globalONLY"