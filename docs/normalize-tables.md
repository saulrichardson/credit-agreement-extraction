# Table preservation in normalization

Issue observed  
Some EDGAR filings embed pricing grids as ASCII tables inside `<TABLE>` tags but without `<tr>/<td>` cells (e.g., using `<CAPTION>`, `<S>`, `<C>` markers). Our normalizer previously dropped any table that lacked rows, which caused loss of Applicable Margin grids (example: accession 0000950117-96-000184, item `_4`).

Fix implemented  
- In `src/pipeline/normalize.py`, we now fall back to preserving the raw table text when no `<tr>/<td>` rows are found, wrapping it in `[[TABLE]] ... [[/TABLE]]` so it survives later processing.  
- Truly empty tables are still dropped.

Guidance / future work  
- Prefer to parse EDGAR ASCII tables into Markdown when feasible, but never drop the content.  
- If other edge formats appear (e.g., fixed-width grids), consider a secondary parser before falling back to raw preservation.  
- When debugging missing pricing, check the ingested HTML; normalization should no longer be the point of loss.  
