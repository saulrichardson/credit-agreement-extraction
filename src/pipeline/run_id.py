from __future__ import annotations

import re
from typing import Optional

import click

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_LEN = 80


def validate_run_id(_ctx: Optional[click.Context], _param: click.Parameter, value: str) -> str:
    """Click callback to enforce safe run_id values.

    Rules:
    - required, non-empty
    - only letters, numbers, dot, underscore, hyphen
    - length <= 80
    """

    if not value or not isinstance(value, str):
        raise click.BadParameter("run_id is required.")
    if len(value) > _MAX_LEN:
        raise click.BadParameter(f"run_id too long (max {_MAX_LEN} characters).")
    if not _RUN_ID_RE.match(value):
        raise click.BadParameter("run_id may contain only letters, numbers, '.', '_', and '-'.")
    return value
