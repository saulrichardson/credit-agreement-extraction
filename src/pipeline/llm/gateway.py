from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


DEFAULT_GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:8000")


class GatewayUnavailable(RuntimeError):
    """Raised when the local agent-gateway client cannot be imported."""


def _prefer_pinned_agent_gateway_on_sys_path() -> Path:
    """Prefer the repo's pinned agent-gateway submodule over any globally-installed package.

    Motivation:
    - Avoid silent drift between a globally installed `gateway` package and the repo's submodule.
    - Keep the gateway/client behavior stable for run-scoped artifacts.
    """

    repo_root = Path(__file__).resolve().parents[3]
    root = repo_root / "agent-gateway" / "src"
    if root.exists() and str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _ensure_gateway_client_sync() -> Any:
    """Return the sync helper complete_response_sync from agent-gateway."""

    root = _prefer_pinned_agent_gateway_on_sys_path()
    try:
        from gateway.client import complete_response_sync  # type: ignore

        return complete_response_sync
    except ModuleNotFoundError as exc:  # pragma: no cover
        if root.exists():
            raise GatewayUnavailable(
                "agent-gateway present but import failed; ensure dependencies are installed"
            ) from exc
        raise GatewayUnavailable(
            "agent-gateway submodule not available; run `git submodule update --init --recursive`"
        ) from exc


def _ensure_gateway_client_async() -> Any:
    """Return the async GatewayAgentClient from agent-gateway."""

    root = _prefer_pinned_agent_gateway_on_sys_path()
    try:
        from gateway.client import GatewayAgentClient  # type: ignore

        return GatewayAgentClient
    except ModuleNotFoundError as exc:  # pragma: no cover
        if root.exists():
            raise GatewayUnavailable(
                "agent-gateway present but import failed; ensure dependencies are installed"
            ) from exc
        raise GatewayUnavailable(
            "agent-gateway submodule not available; run `git submodule update --init --recursive`"
        ) from exc

