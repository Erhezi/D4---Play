"""Application settings loaded from .env via pydantic-settings."""

from __future__ import annotations

import json
from pathlib import Path
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


# Look for .env in ai_export_builder/ first, then fall back to project root
_PKG_ENV = Path(__file__).resolve().parent / ".env"
_ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
_ENV_FILE = str(_PKG_ENV) if _PKG_ENV.exists() else str(_ROOT_ENV)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Prefer the repo-local .env over inherited machine/user environment vars.
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-5-mini"
    openai_base_url: str = ""  # leave blank for default api.openai.com
    openai_ca_bundle: str = ""  # optional PEM file for corporate/intercepting proxies
    openai_use_system_cert_store: bool = True

    # SQL Server
    db_driver: str = "{ODBC Driver 17 for SQL Server}"
    db_server: str = ""
    db_database: str = ""
    db_username: str = ""
    db_password: str = ""
    db_timeout: int = 30

    # App
    daily_request_limit: int = 10
    max_export_rows: int = 100_000
    fiscal_year_start_month: int = 1

    # Hardcoded test user (MVP)
    test_user_id: str = "test_user"
    test_user_name: str = "Test User"
    test_user_facilities: str = '["ALL"]'

    @property
    def user_facilities(self) -> list[str]:
        return json.loads(self.test_user_facilities)

    @property
    def connection_string(self) -> str:
        base = (
            f"DRIVER={self.db_driver};"
            f"SERVER={self.db_server};"
            f"DATABASE={self.db_database};"
            f"TIMEOUT={self.db_timeout};"
        )
        if self.db_username:
            base += f"UID={self.db_username};PWD={self.db_password};"
        else:
            base += "Trusted_Connection=yes;"
        return base


settings = Settings()
