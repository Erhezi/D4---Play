"""Helpers for constructing an OpenAI HTTP client with explicit TLS settings."""

from __future__ import annotations

import ssl
from pathlib import Path

import httpx

from ai_export_builder.config import settings


def _resolve_ca_bundle_path() -> str | None:
    """Return the configured CA bundle path, validating that it exists."""
    if not settings.openai_ca_bundle:
        return None

    ca_bundle = Path(settings.openai_ca_bundle).expanduser()
    if not ca_bundle.is_file():
        raise FileNotFoundError(
            "OPENAI_CA_BUNDLE does not point to a readable file: "
            f"{ca_bundle}"
        )
    return str(ca_bundle)


def build_openai_verify() -> ssl.SSLContext | str | bool:
    """Build the TLS verify setting used by HTTPX for OpenAI calls."""
    ca_bundle = _resolve_ca_bundle_path()

    if settings.openai_use_system_cert_store:
        context = ssl.create_default_context()
        if ca_bundle:
            context.load_verify_locations(cafile=ca_bundle)
        return context

    if ca_bundle:
        return ca_bundle

    return True


def build_openai_http_client() -> httpx.Client:
    """Create a short-lived HTTPX client for OpenAI API calls."""
    return httpx.Client(
        verify=build_openai_verify(),
        timeout=httpx.Timeout(60.0, connect=15.0),
    )