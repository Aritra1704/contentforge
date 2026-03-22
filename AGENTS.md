# Role
Text generation engine.

## Responsibilities
- Generate greeting messages
- Provide multiple variations
- Support tone/theme
- Rank and shortlist candidates with reasons

## Problems
- duplicate outputs
- incomplete sentences
- poor ranking

## Rules
- output must be complete sentences
- no duplicates
- rank based on quality + relevance
- support shared request fields such as `app_id`, `content_type`, and `creative_brief`

## Output Format
- clean shortlist
- ready for selection (no post-processing required)
- include recommendation/ranking metadata for downstream UI and orchestration
