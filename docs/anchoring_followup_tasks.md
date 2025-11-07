Anchoring Follow-up Tasks
=========================

Context
-------
Prompt views now render table anchors as `⟦tXXrYYcZZ⟧ HEADER: value`, reducing the noisy “Table Field …” pattern for Financial Data Schedules. Canonical text and anchor metadata remain unchanged. The reverse-highlighter previews reflect the same spans.

Pending Decisions / Next Steps
------------------------------
1. Confirm whether the table prompt view should also surface column headers (currently we join `row_header` and `col_header` with `|` only when both exist).
2. Evaluate a richer layout for structured `<tr>/<td>` tables (e.g., emit Markdown or HTML grids backed by the captured row/column header trails). Use the 2015/2020/2024 samples in `analysis_samples/` to verify checkbox tables and multi-row headers render cleanly.
3. Once table formatting is finalized, integrate the prompt-view generator into the LLM pipeline so Stage 1/2 use the inline IDs by default.
4. Expand validation to newer filings with complex HTML tables, ASCII grids, and mixed content; capture findings in `docs/anchoring_status.md`. The current sample set highlights FDS (1996/2000), checkbox grids (2015/2024), and XBRL blobs (2023).
5. Add automated tests covering canonicalization, sentence segmentation, FDS parsing, structured-table parsing, and prompt view output.
