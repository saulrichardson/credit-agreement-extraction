from __future__ import annotations

import click

from .all_v2 import all_v2_full
from .definitions import (
    blocking_terms_compiler_v1,
    compustat_overlay_eval_v1,
    compustat_overlay_v1,
    definitions_compiler_v1,
)
from .facility import agreement_metadata, analysis_export_v2, facility_fundamentals
from .ingest import ingest, normalize
from .indexing_v2 import index_v2, retrieve_v2, structured_v2


@click.group()
def cli() -> None:
    """Run-scoped EX-10 processing pipeline."""


# Ingest / normalize
cli.add_command(ingest)
cli.add_command(normalize)

# Indexing/retrieval/structured (v2)
cli.add_command(index_v2)
cli.add_command(retrieve_v2)
cli.add_command(structured_v2)

# Definitions + term compiler + overlay
cli.add_command(definitions_compiler_v1)
cli.add_command(blocking_terms_compiler_v1)
cli.add_command(compustat_overlay_v1)
cli.add_command(compustat_overlay_eval_v1)

# Facility & agreement metadata + analysis export
cli.add_command(facility_fundamentals)
cli.add_command(agreement_metadata)
cli.add_command(analysis_export_v2)

# Locked full pipeline
cli.add_command(all_v2_full)


def main() -> None:
    cli()
