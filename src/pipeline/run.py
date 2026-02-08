from __future__ import annotations

from pipeline.cli.common import resolve_run_config as _resolve_run_config
from pipeline.cli.main import cli, main


def resolve_run_config(run_id: str, base_dir: str, workers: int, bandwidth: int):
    """Resolve run configuration for CLI and scripts."""

    return _resolve_run_config(run_id, base_dir, workers=workers, bandwidth=bandwidth)


__all__ = ["cli", "main", "resolve_run_config"]


if __name__ == "__main__":
    main()
