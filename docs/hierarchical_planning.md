# Hierarchical Facility Planning (LLM-Only)

Context: chunk-level scoring/extraction sometimes misattributes pricing/covenant text because agreements mix global clauses with facility-specific ones. We want a zero-heuristic approach that still keeps token counts manageable.

## Proposed Workflow

1. **Coarse segments** – Run the existing `plan-document` (or an adapted prompt) over the full prompt view to get contiguous anchor ranges for major sections. This call can use a higher-end model once per agreement.
2. **Facility-focused drills** – For every segment tagged as facilities/pricing/covenants, run a second LLM prompt on just that subrange (plus a halo) asking for:
   - facility names & aliases
   - anchor ranges per facility
   - applicability notes (single facility vs. multi-facility vs. global)
   - evidence anchors
3. **Chunk labelling** – When chunking, feed each chunk (with a halo) plus the facility map back to the model and ask it to label which facilities it governs. Store these labels with the chunk metadata.
4. **Topic scoring/extraction** – Downstream prompts (pricing, covenants, metadata) consume the chunk labels and include the facility info in the question, so the model reasons with scoped context instead of guessing.

## Token Strategy

- Only the coarse planner sees the entire document.
- Drill-down prompts operate on section slices, keeping window sizes manageable.
- Facility/chunk labelling and subsequent scoring reuse the stored metadata, so we avoid resending large context repeatedly.
- When a chunk is mostly a table, extend the halo to include nearby prose so the LLM can infer applicability without heuristics.

## Next Steps

- Draft specialized planner prompts (facility map + applicability schema).
- Script the multi-stage run so artifacts live alongside the canonical bundle (e.g., `facility_plan.json`, `chunk_facility_labels.json`).
- Once stable, integrate into the chunk-scoring loop so every topic score references the same facility hierarchy.
