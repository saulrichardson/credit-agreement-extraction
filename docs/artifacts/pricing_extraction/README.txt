Pricing extraction outputs produced by scripts/run_pricing_extraction.py.
Each JSON/TXT pair corresponds to a different selector:
  - semantic_pricing.* from semantic-plan score=1 segments
  - chunk_pricing.* from chunk score=1 slices
  - anchors_pricing.* from the LLM anchor screener selections
- semantic_structured.* etc. contain outputs from prompts/extraction_structured.txt
