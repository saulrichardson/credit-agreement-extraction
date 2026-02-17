from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from pipeline.contracts.common import StructuredValidationContext
from pipeline.contracts.covenant_simple_v2 import empty_payload as covenant_empty_payload
from pipeline.contracts.covenant_simple_v2 import retry_prompt as covenant_retry_prompt
from pipeline.contracts.covenant_simple_v2 import validate as covenant_validate
from pipeline.contracts.pricing_structured_v2 import empty_payload as pricing_empty_payload
from pipeline.contracts.pricing_structured_v2 import retry_prompt as pricing_retry_prompt
from pipeline.contracts.pricing_structured_v2 import validate as pricing_validate


StructuredContractName = Literal["pricing_structured_v2", "covenant_simple_v2"]


ValidateFn = Callable[[object, StructuredValidationContext], dict[str, Any]]
RetryPromptFn = Callable[[str, int, str, str], str]
EmptyPayloadFn = Callable[[str], dict[str, Any]]


@dataclass(frozen=True)
class StructuredContract:
    name: StructuredContractName
    allowed_root_types: tuple[type, ...]
    validate: ValidateFn
    retry_prompt: RetryPromptFn
    empty_payload: EmptyPayloadFn


def get_structured_contract(name: StructuredContractName) -> StructuredContract:
    if name == "pricing_structured_v2":
        return StructuredContract(
            name=name,
            allowed_root_types=(dict,),
            validate=lambda payload, ctx: pricing_validate(payload, ctx=ctx),
            retry_prompt=pricing_retry_prompt,
            empty_payload=lambda reason: pricing_empty_payload(reason=reason),
        )
    if name == "covenant_simple_v2":
        return StructuredContract(
            name=name,
            allowed_root_types=(dict,),
            validate=lambda payload, ctx: covenant_validate(payload, ctx=ctx),
            retry_prompt=covenant_retry_prompt,
            empty_payload=lambda reason: covenant_empty_payload(reason=reason),
        )
    raise KeyError(f"unknown structured contract {name!r}")

