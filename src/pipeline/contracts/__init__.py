"""LLM output contracts (schemas + semantic validators).

Goal: every LLM stage has an explicit, versioned contract that is enforced at runtime
via strict JSON parsing + Pydantic validation + semantic checks.

We intentionally do not keep backwards compatibility here; contracts are the source of truth.
"""

