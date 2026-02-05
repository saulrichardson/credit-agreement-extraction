from __future__ import annotations

from .cli.common import resolve_run_config as _resolve_run_config
from .cli.main import cli, main


def resolve_run_config(run_id: str, base_dir: str, workers: int, bandwidth: int):
    """Backward-compatible wrapper around `pipeline.cli.common.resolve_run_config`."""

    return _resolve_run_config(run_id, base_dir, workers=workers, bandwidth=bandwidth)


__all__ = ["cli", "main", "resolve_run_config"]


if __name__ == "__main__":
    main()
