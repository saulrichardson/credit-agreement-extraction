Semantic planner artifacts (all plain text):

- canonical.txt — normalized document text.
- prompt_view.txt — canonical text with inline anchor IDs.
- canonical_with_anchors.txt — same as prompt view, duplicated for convenience.
- semantic_plan.txt — JSON content from the LLM partition (renamed with .txt extension).
- semantic_scores.txt — JSON scoring output for each semantic segment.
- semantic_segments_all_scores.txt — full document text grouped by segment, each with score + rationale.
- semantic_segments_score_1.txt — only the score=1 segments (pricing-critical according to the scorer).
- semantic_segments_score_0.5.txt — only the score=0.5 segments (supporting context).

Recreate via:
poetry run edgar-pipeline plan-document --canonical-dir /tmp/pricing_grid_sample --output /tmp/pricing_grid_sample/planner_semantic_index.json
OPENAI_API_KEY=… poetry run edgar-pipeline score-plan --plan /tmp/pricing_grid_sample/planner_semantic_index.json --output /tmp/pricing_grid_sample/planner_semantic_index_scores.json
cp /tmp/pricing_grid_sample/planner_semantic_index.json docs/artifacts/semantic/semantic_plan.txt
cp /tmp/pricing_grid_sample/planner_semantic_index_scores.json docs/artifacts/semantic/semantic_scores.txt
