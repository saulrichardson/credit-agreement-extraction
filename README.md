credit-agreement-extraction
===========================

This package bundles the utilities we need to extract, catalogue, and analyse
segment-level filings from the SEC EDGAR tar archives.  It is intended to live
next to (but independent from) the larger covenant extraction repository so
that we can eventually promote it into its own project or submodule.

Key features
------------

* A light-weight `EdgarFiling` object that wraps an individual segment and
  exposes the HTML, plain-text, and table representations we need for
  downstream regex and LLM workflows.
* Manifest generation that walks the `.tar.gz` archives and records segment
  metadata (file name, segment number, document type, table presence, etc.).
* Extraction helpers that materialise the raw HTML for selected segments while
  keeping metadata in sync.
* Built-in (and customizable) filtering for binary-only exhibits (e.g.
  GRAPHIC/PDF attachments) so downstream notebooks only receive text-based
  segments, with skip reasons captured in the extraction metadata. Provide your
  own filter hook when you need to layer additional policies.
* Canonical `segment_id`/`segment_digest` identifiers for every manifest,
  extraction, and normalization row, making it easy to de-duplicate, resume, or
  route segments through downstream agents and LLM workflows.

The API is intentionally simple so we can embed it inside larger pipelines or
call it from notebooks.  Use the `edgar-pipeline` CLI (or the functions in
`edgar_filing_pipeline.workflow`) for end-to-end examples.

Quick start (Poetry)
--------------------

```bash
cd credit-agreement-extraction

# install dependencies and create .venv/
make install

# optional: activate the Poetry shell
make shell

# Build a manifest for every .nc document in a tarball directory
make manifest TAR_ROOT=/path/to/daily_filings/2003/QTR2 OUTPUT=manifest_2003_q2.parquet

# Extract the HTML for a filtered subset of segments
make extract MANIFEST=manifest_2003_q2.parquet TAR_ROOT=/path/to/daily_filings OUTPUT_DIR=./html_segments

# Normalize the surviving segments into text + Markdown tables
make normalize MANIFEST=data/manifest_19960103.parquet TAR_ROOT=data/daily_filings/1996/QTR1 OUTPUT=data/processed/v1/1996_Q1.parquet
```

You can also run the entire workflow in one go:

```bash
make pipeline TAR_ROOT=/path/to/daily_filings MANIFEST_OUT=manifest.parquet EXTRACT_DIR=./html_segments NORMALIZED_OUT=./normalized.parquet
```

Need a plain `requirements.txt` for remote clusters? Run `poetry export --without-hashes -f requirements.txt --output requirements.txt` (requires the `poetry-plugin-export` we ship) before copying files around.

Legacy helpers in `scripts/` continue to exist for backwards compatibility, but
the CLI (and the Python APIs it wraps) are the supported way to run the core
pipeline steps.

Once extracted, wrap each file with `EdgarFiling` to access plain text,
Markdown-, or DataFrame-rendered tables suitable for LLM prompts.

Custom filtering
----------------

The extractor now accepts a pluggable filter hook so you can apply arbitrary
policies (blacklists, prior-run lookups, etc.) in addition to the default binary
checks:

```python
from pathlib import Path

import pandas as pd
from edgar_filing_pipeline import extract_from_manifest, default_filter


def skip_short_docs(header, html, context):
    """Example composite filter."""
    # Chain the legacy binary detection first.
    reason = default_filter(header, html, context)
    if reason:
        return reason
    if len(html) < 1024:
        return "policy:too_short"
    manifest_row = context["manifest_row"]
    if manifest_row.get("doc_type") == "EX-10.K":
        return "policy:blacklisted_doc_type"
    return None


def enrich_context(manifest_row):
    # Optional hook: look up previous metadata, attach to the filter context.
    prior = {"seen_before": manifest_row["segment_no"] < 5}
    return {"previous_metadata": prior}


manifest = pd.read_parquet("data/manifest.parquet")
records = extract_from_manifest(
    manifest,
    tar_root=Path("data/daily_filings"),
    output_dir=Path("data/extracted"),
    filter_fn=skip_short_docs,
    filter_context_builder=enrich_context,
)
```

The filter always receives the SGML header, raw HTML, and a context dict that at
minimum includes the manifest row under `manifest_row`. Return `None` to keep a
segment or a string reason (e.g. `policy:blacklist`) to drop it while storing
the explanation alongside the extraction metadata.


Segment identifiers & orchestration
-----------------------------------

Every manifest/extraction/normalization row now carries a deterministic
`segment_id` (`tarfile::member::segment_no`) plus a SHA-256 `segment_digest`.
Use these IDs to join metadata across runs, de-duplicate work, or hand segments
to downstream agents confidently. Helper functions live in
`edgar_filing_pipeline.identifiers`.

For programmatic workflows, call the orchestration helpers in
`edgar_filing_pipeline.workflow` instead of shelling out to individual scripts:

```python
from pathlib import Path
from edgar_filing_pipeline.workflow import (
    build_manifest_to_path,
    extract_segments_to_dir,
    normalize_from_manifest,
    read_manifest,
)

tar_root = Path("data/daily_filings/1996/QTR1")
manifest_path = Path("data/manifests/1996_q1.parquet")
extract_dir = Path("data/extracted/1996_q1")
normalized_path = Path("data/processed/1996_q1.parquet")

manifest_df = build_manifest_to_path(
    tar_root=tar_root,
    output_path=manifest_path,
)

records = extract_segments_to_dir(
    manifest=manifest_df,
    tar_root=tar_root,
    output_dir=extract_dir,
)

normalized_df = normalize_from_manifest(
    manifest=manifest_df,
    tar_root=tar_root,
)
normalized_df.to_parquet(normalized_path, index=False)
```

The `run_pipeline` helper wraps all three steps (manifest, extraction,
normalization) when you want a single call that returns file paths +
per-segment extraction metadata, ready for downstream LLM batching.


HPC usage (NYU Greene or similar)
---------------------------------

The repository includes a helper that bootstraps a virtual environment with
the pinned requirements.  This keeps the dependency footprint small and avoids
exhausting your `$HOME` quota on shared clusters.

```bash
module load python/3.10            # adjust to whatever module exists
git clone git@github.com:saulrichardson/credit-agreement-extraction.git
cd credit-agreement-extraction

./scripts/bootstrap_venv.sh $SCRATCH/venvs/edgar-filing
source $SCRATCH/venvs/edgar-filing/bin/activate
```

From there you can run the scripts exactly as in the quick start section.  If
you prefer containers (Singularity/Apptainer), copy the install stanza from
`bootstrap_venv.sh` into your definition file.

For reproducible jobs, capture the fully resolved dependency set once the
environment is active:

```bash
poetry run pip freeze > requirements.lock.txt
```

Ship that lock file with your batch scripts and install via
`pip install -r requirements.lock.txt` inside the container or virtualenv.

Container build (optional)
--------------------------

To bundle the project and its pinned dependencies into a portable Apptainer/Singularity
image:

```bash
cd credit-agreement-extraction
./apptainer/build_sif.sh
```

The script binds the repository into the build context and produces
`edgar-filing.sif` using the definition file in `apptainer/edgar_filing.def`.
On NYU Greene you can run this inside an interactive compute session (see the
cluster notes above); the script automatically uses `$SCRATCH` for the build
cache if available.

Once built, run commands inside the image with e.g.

```bash
apptainer exec edgar-filing.sif python -m edgar_filing_pipeline.scripts.build_manifest ...
```

and add `--nv` if you ever need GPU access.

### Optional: Singularity overlay workflow (NYU Greene)

If the cluster does not permit `singularity/apptainer build --fakeroot`, use a
base Python image together with a writable overlay to keep dependencies in a
single file.  The following snippet is what we ran on Greene:

```bash
# once per project
SCRATCH=/scratch/$USER
cd $SCRATCH/repos
REPO=$SCRATCH/repos/credit-agreement-extraction
cd $REPO
singularity pull python311.sif docker://python:3.11-slim
singularity overlay create --size 4096 edgar-filing.ext3

# bootstrap a virtualenv inside the overlay
singularity exec --overlay edgar-filing.ext3 $REPO/python311.sif python -m venv /ext3/venv
singularity exec --overlay edgar-filing.ext3 --bind "$REPO:/workspace" \
  $REPO/python311.sif /ext3/venv/bin/pip install -r /workspace/requirements.txt
singularity exec --overlay edgar-filing.ext3 --bind "$REPO:/workspace" \
  $REPO/python311.sif /ext3/venv/bin/pip install /workspace
```

Use the environment by calling the interpreter in `/ext3/venv/bin/python`:

```bash
singularity exec --overlay edgar-filing.ext3 \
  --bind "$REPO:/workspace" \
  $REPO/python311.sif /ext3/venv/bin/python \
  -m edgar_filing_pipeline.scripts.build_manifest ...
```

Both `edgar-filing.ext3` and `python311.sif` are single files, so you stay well
within Greene's inode quotas.  Adjust the overlay size if you need more space.

Alternatively, run the helper script to perform all three steps in one go:

```bash
SCRATCH=/scratch/$USER
./scripts/setup_overlay.sh $SCRATCH/repos/credit-agreement-extraction
```



### Generate plain text with Markdown tables

If you already have extracted HTML segments and simply want `.txt` files that
retain the tables as Markdown (for LLM prompts, for example), run:

```bash
python scripts/export_markdown_text.py \
    --input-dir /Users/saul/projects/dan-covenants/clean/edgar-covenants/isdebt_contracts/2003_high_confidence_html \
    --output-dir /Users/saul/projects/dan-covenants/clean/edgar-covenants/isdebt_contracts/2003_high_confidence_markdown
```

This leaves the original HTML files untouched and writes Markdown-enhanced text
files to the destination directory.


### Batch LLM reasoning over agreements

To analyse a folder of agreement `.txt` files with `gpt-5-mini` (reasoning mode
set to “medium”), use the batch runner:

```bash
python scripts/run_reasoning_batch.py \
    --input-dir /path/to/agreements \
    --system-prompt-file /path/to/system_prompt.txt \
    --output-dir ./llm_responses \
    --workers 6 \
    --skip-existing
```

Key notes:

- The user prompt template is optional; when omitted, the script defaults to
  wrapping the agreement between `<<BEGIN DOC>>` and `<<END DOC>>`. If you
  provide a template, include `{{document}}` where the full agreement should be
  injected; otherwise the document text is appended after the prompt.
- Use `--system-prompt-file` when you want to send a separate system message;
  the agreement is always delivered in the user message.
- Responses are written one-per-agreement (same basename) in the chosen output
  directory.
- Set `OPENAI_API_KEY` in your environment or pass `--api-key`.
- The runner uses concurrent workers with exponential backoff and jitter to
  minimise failures from transient API issues. Use `--max-retries`/
  `--initial-backoff` to tune the retry policy as needed.

Gateway for agents
------------------

Run `gateway-server` to expose a FastAPI service that orchestrates segment access
and OpenAI Responses API calls on behalf of downstream agents:

```bash
gateway-server     --tar-root /path/to/daily_filings     --manifest data/manifest.parquet     --host 127.0.0.1 --port 8080
```

POST `/jobs` with the segment you want plus model/prompt instructions and the
gateway will normalize the segment, build the prompt (respecting table/metadata
flags), call the OpenAI Responses API, and store the result. Poll `/jobs/{id}` to
retrieve the response text, raw payload, and the document snapshot that was
sent to the model. This keeps API keys, rate limiting, retries, and pipeline
policy in one place while letting any agent (or workflow runner) make a single
HTTP request per task.
