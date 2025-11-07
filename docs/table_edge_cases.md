Table Edge Cases — Sample Artifacts
===================================

Purpose
-------
Provide concrete examples of how today’s anchoring/prompt logic behaves on richly formatted tables so we can evaluate header-extraction strategies without modifying code yet. Each sample lives in `analysis_samples/…` (raw HTML plus helper dumps) with corresponding anchor bundles under `output/anchors/`.

Artefact Index
--------------

### 1. 2000 Form 8-K (two-row headers)
- Raw HTML: `analysis_samples/sample_2000_struct_nested_seg1.html`
- Markup with table boundaries: `analysis_samples/sample_2000_struct_nested_markup.html`
- Row dump: `analysis_samples/sample_2000_struct_nested_seg1_table_rows.txt`
- Anchored output: `output/anchors/sample_2000_struct_nested/prompt_view.txt`, `anchors.tsv`, `highlight_preview.html`

Issue illustrated: value rows precede caption rows, yielding prompt lines like `⟦t01r02c01⟧ Delaware | Delaware: Delaware`. Good testbed for a structural header/value pairing algorithm.

### 2. 2015 Form 8-K (checkbox grid + captions)
- Raw HTML: `analysis_samples/sample_2015_struct_nested_seg1.html`
- Markup: `analysis_samples/sample_2015_struct_nested_markup.html`
- Row dump: `analysis_samples/sample_2015_struct_nested_seg1_table_rows.txt`
- Anchored output: `output/anchors/sample_2015_struct_nested/…`

Highlights:
- Rows 2–3 form a caption/value pair with parentheses.
- Later tables contain “¨” checkboxes whose descriptions live in neighbouring cells — prompt view currently shows `⟦t02r01c01⟧ ¨ | ¨: ¨`.

### 3. 2020 Pricing Supplement (multi-row headings + fine-grained styling)
- Raw HTML: `analysis_samples/sample_2020_struct_nested_seg1.html`
- Markup: `analysis_samples/sample_2020_struct_nested_markup.html`
- Row dump: `analysis_samples/sample_2020_struct_nested_seg1_table_rows.txt`
- Anchored output: `output/anchors/sample_2020_struct_nested/…`

Notes: The first table mixes date text, registration statement captions, and nested rows. Good candidate for testing header trail extraction (derive “January … | Registration Statement …”).

### 4. 2024 Pricing Supplement (checkbox tables with detailed prose)
- Raw HTML: `analysis_samples/sample_2024_struct_nested_seg1.html`
- Markup: `analysis_samples/sample_2024_struct_nested_markup.html`
- Row dump: `analysis_samples/sample_2024_struct_nested_seg1_table_rows.txt`
- Anchored output: `output/anchors/sample_2024_struct_nested/…`

Issue: prompt view emits repeated `■` glyphs, while the descriptive text lives in the adjacent column. Use this to validate glyph filtering + header consolidation.

### 5. 2023 XBRL snippet
- Raw XML: `analysis_samples/sample_2023_ascii_seg5.html`
- Anchored output: `output/anchors/sample_2023_ascii/prompt_view.txt`

Shows how an XBRL schema currently becomes one giant sentence. Decide whether to skip or compress.

How to Use
----------
1. Open the markup (`*_markup.html`) to visualise where tables and nested tables sit.
2. Review the row dumps (`*_table_rows.txt`) to see row ordering, captions, and placeholder glyphs without rendering.
3. Compare against the current prompt view / highlight preview to spot mismatches.
4. Prototype header-detection/glyph-filtering logic against these files before touching production code.
