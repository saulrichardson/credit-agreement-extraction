edgar-filing-pipeline
=====================

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

The API is intentionally simple so we can embed it inside larger pipelines or
call it from notebooks.  See `scripts/build_manifest.py` and
`scripts/extract_segments.py` for end-to-end examples.

Quick start (pip/virtualenv)
---------------------------

```bash
cd edgar-filing-pipeline

# optional but recommended: create an isolated environment
python3 -m venv .venv
source .venv/bin/activate

# install pinned runtime dependencies
pip install -r requirements.txt

# install the package in editable mode for local development
pip install -e .

# Build a manifest for every .nc document in a tarball directory
python scripts/build_manifest.py \
    --tar-root /path/to/daily_filings/2003/QTR2 \
    --output manifest_2003_q2.parquet

# Extract the HTML for a filtered subset of segments
python scripts/extract_segments.py \
    --manifest manifest_2003_q2.parquet \
    --tar-root /path/to/daily_filings \
    --output-dir ./html_segments
```

Once extracted, wrap each file with `EdgarFiling` to access plain text,
Markdown-, or DataFrame-rendered tables suitable for LLM prompts.


HPC usage (NYU Greene or similar)
---------------------------------

The repository includes a helper that bootstraps a virtual environment with
the pinned requirements.  This keeps the dependency footprint small and avoids
exhausting your `$HOME` quota on shared clusters.

```bash
module load python/3.10            # adjust to whatever module exists
git clone git@github.com:saulrichardson/edgar-filing-pipeline.git
cd edgar-filing-pipeline

./scripts/bootstrap_venv.sh $SCRATCH/venvs/edgar-filing
source $SCRATCH/venvs/edgar-filing/bin/activate
```

From there you can run the scripts exactly as in the quick start section.  If
you prefer containers (Singularity/Apptainer), copy the install stanza from
`bootstrap_venv.sh` into your definition file.

For reproducible jobs, capture the fully resolved dependency set once the
environment is active:

```bash
pip freeze > requirements.lock.txt
```

Ship that lock file with your batch scripts and install via
`pip install -r requirements.lock.txt` inside the container or virtualenv.

Container build (optional)
--------------------------

To bundle the project and its pinned dependencies into a portable Apptainer/Singularity
image:

```bash
cd edgar-filing-pipeline
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
REPO=$SCRATCH/repos/edgar-filing-pipeline
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
