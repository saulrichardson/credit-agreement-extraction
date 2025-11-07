Chunking + scoring artifacts (plain text):

- chunk_plan.txt — JSON chunk map (40-sentence windows, stride 20).
- chunk_scores.txt — JSON with LLM scores (0/0.5/1) answering the pricing question.
- prompt_with_chunks_all_scores.txt — human-readable chunk slices with headers showing range + score.
- chunks_score_1.txt — only the score=1 chunks (pricing core).
- chunks_score_0.5.txt — only the score=0.5 chunks (support context).

Recreate via:
poetry run edgar-pipeline chunk-document --canonical-dir /tmp/pricing_grid_sample --chunk-size 40 --stride 20 --output /tmp/pricing_grid_sample/chunks_40_20.json
OPENAI_API_KEY=… poetry run edgar-pipeline score-plan --plan /tmp/pricing_grid_sample/chunks_40_20.json --output /tmp/pricing_grid_sample/chunks_40_20_scores.json
