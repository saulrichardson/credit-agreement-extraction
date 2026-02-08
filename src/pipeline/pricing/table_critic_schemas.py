from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TableCritiqueIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    message: str
    source_refs: list[str] = Field(default_factory=list)


class TableCritique(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_valid: bool
    issues: list[TableCritiqueIssue] = Field(default_factory=list)
    fix_instructions: str | None = None

    @property
    def has_blockers(self) -> bool:
        return not self.is_valid or bool(self.issues)

