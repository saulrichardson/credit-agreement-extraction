Anchoring & Reverse Highlighting Status
=======================================

Overview
--------

The new anchoring pipeline (`CanonicalDocumentBuilder`, `anchors.tsv`, `prompt_view.txt`, `ReverseHighlighter`) has been exercised on three representative EDGAR segments. This document tracks current robustness, known gaps, and next steps so we have a single place to coordinate follow-ups.

Validation Summary
------------------

* `data/segments_19960103/0000013386-96-000003_seg1.html` (narrative prospectus)
  * ✅ `canonical.txt`: normalized paragraphs with preserved content.
  * ✅ `anchors.tsv`: 936 rows covering paragraphs (`p`), sentences (`s`), headings (`h`); `prev_id` / `next_id` link in reading order.
  * ✅ `prompt_view.txt`: inline sentence ids (`⟦s000001⟧ …`) suitable for LLM prompts.
  * ✅ `highlights_sample.json`: reverse highlighter returns the first five sentence spans with correct offsets (`start=0, end=217`, etc.).

* `data/segments_19960103/0000003521-96-000001_seg2.html` (Financial Data Schedule)
  * ✅ FDS parsing path creates `t01r##c01` anchors with `row_header` populated (`ARTICLE`, `NAME`, …).
  * ✅ Serialized rows appear in canonical text; prompt view now renders as `⟦t01r05c01⟧ NAME: …`, surfacing the header/value pair instead of the raw `Table 1 Field …` prefix.
  * ✅ Reverse highlighter sample spans align with canonical offsets.

* `data/segments_19960103/0000705200-96-000001_seg2.html` (HTML table with `<TR>/<TD>`)
  * ✅ Structured-table path (`_process_structured_table`) generates cell anchors, including rowspan/colspan handling.
  * ✅ `anchors.tsv` entries contain `attributes_json` with row/col coordinates and headers.
  * ✅ Prompt view shows the expected `⟦t01r##c##⟧` lines.

Status
------

* Anchor generation and prompt view tooling are functioning against early EDGAR HTML, including SGML-based FDS tables and structured `<TR>/<TD>` tables.
* Layout tables that only carry prose/checkbox glyphs are now flattened into sentence anchors (no more `Row Header: ■` noise); true data tables render as Markdown grids; and obvious machine payloads (XML/XBRL/uuencode) are skipped before prompting.
* Single-character filler lines (just `.`/`*`/`•`) are dropped when building sentence anchors so prompts stay compact.
* Reverse highlighter currently returns deterministic anchor spans (no LLM refinement yet); validation confirms offsets and text align.
* Fail-fast behaviour: missing text or unsupported table formats raise exceptions instead of silently dropping content.

Open Questions / Concerns
-------------------------

* Later-period filings may include nested tables, merged headers, or non-standard HTML that `_process_structured_table` does not yet cover.
* ASCII-style grid detection beyond FDS patterns remains untested.
* We have not evaluated documents with embedded exhibits/PDF text; multi-source anchoring will require additional metadata.
* Reverse highlighter still uses raw anchor ranges (no LLM minimization yet).

2025-XX Sampling Notes
----------------------

Spot checks on newly mirrored tarballs (1996→2024, `analysis_samples/`) surfaced concrete cases to guide the open items:

* **FDS header pairing still imperfect** — e.g. `analysis_samples/sample_2000_struct_nested_seg1.html` produces `⟦t01r02c01⟧ Delaware | Delaware: Delaware` because the first row holds values while the second row carries the captions. We need heuristics to detect “label row below value row” so table cells emit `Value (header)` correctly.
* **Checkbox matrices** — filings such as `analysis_samples/sample_2015_struct_nested_seg1.html` and `analysis_samples/sample_2024_struct_nested_seg1.html` contain tables of checkboxes (`¨`, `■`). The current prompt view duplicates the glyph across row/column headers (`⟦t02r01c01⟧ ¨ | ¨: ¨`). We should suppress placeholder glyphs and surface the descriptive text only.
* **Deeply nested HTML tables** — 2015/2020/2023/2024 samples include multi-level tables with inline styling. Row/column ancestry is captured, but header trails are noisy. We likely need to derive header text from `<th>` elements or styled caption rows instead of assuming “first row = header”.
* **XBRL/structured XML blobs** — `analysis_samples/sample_2023_ascii_seg5.html` is pure XBRL. Anchoring yields a single enormous sentence (`⟦s000001⟧ … link:presentationLink …`). These segments should probably be flagged as non-text (skip or compress) before reaching the LLM.

Next iteration should incorporate these examples into automated tests once the parsing rules are refined.

Next Steps
----------

1. Collect and anchor a broader sample (e.g., 2000s filings with modern HTML) to stress rowspan/colspan logic and identify missing table heuristics.
2. Add unit/integration tests for canonicalization, sentence segmentation, and table parsing using fixtures from the current sample set.
3. Integrate `ReverseHighlighter` with the LLM-assisted minimal-span finder once prompt patterns are finalized.
4. Extend `anchors.tsv` schema with cross-reference anchors (`x`), footnotes (`f`), and definition linkage when we begin definition extraction.

Artifacts
---------

Generated bundles are under `output/anchors/`:

* `0000013386-96-000003_seg1/`
* `0000003521-96-000001_seg2/`
* `0000705200-96-000001_seg2/`
* `0000889812-96-000005_seg1/`

Each directory contains `canonical.txt`, `anchors.tsv`, `prompt_view.txt`, and `highlights_sample.json`.
