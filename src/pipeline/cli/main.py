from __future__ import annotations

import click

from ..contract_ir_v0_2_cli import contractir_v0_2_cli
from .all_v2 import all_v2
from .contract_pricing import contract_pricing, contract_pricing_flow, contract_pricing_v3
from .definitions import (
    blocking_terms_compiler_v1,
    compustat_overlay_v1,
    definitions_compiler_v1,
    definitions_v2,
)
from .facility import agreement_metadata, analysis_export, facility_fundamentals
from .ingest import ingest, normalize
from .indexing_v2 import index_v2, retrieve_v2, structured_v2
from .toc import toc_v1


@click.group()
def cli() -> None:
    """Run-scoped EX-10 processing pipeline."""


# Separate namespace for the imposed-structure pricing-kernel approach (ContractIR v0.2).
cli.add_command(contractir_v0_2_cli)

# Ingest / normalize
cli.add_command(ingest)
cli.add_command(normalize)

# Indexing/retrieval/structured (v2)
cli.add_command(index_v2)
cli.add_command(retrieve_v2)
cli.add_command(structured_v2)

# Pricing compilation
cli.add_command(contract_pricing)
cli.add_command(contract_pricing_v3)
cli.add_command(contract_pricing_flow)

# Definitions + term compiler + overlay
cli.add_command(definitions_v2)
cli.add_command(definitions_compiler_v1)
cli.add_command(blocking_terms_compiler_v1)
cli.add_command(compustat_overlay_v1)

# Facility & agreement metadata + analysis export
cli.add_command(facility_fundamentals)
cli.add_command(agreement_metadata)
cli.add_command(analysis_export)

# TOC and full pipeline
cli.add_command(toc_v1)
cli.add_command(all_v2)


def main() -> None:
    cli()
