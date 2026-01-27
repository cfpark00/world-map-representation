# Session Log: ICML Title and Abstract Revision

**Date**: 2026-01-27 04:50
**Summary**: Revised ICML paper title and abstract to emphasize convergence/divergence findings; added editorial ick points and key speculation about why distance task is divergent.

## Context

- ICLR 2026 paper was rejected
- Pivoting to ICML 2026 submission (deadline: Jan 28 AoE)
- Senior researcher advice: focus on divergent task finding, frame the mystery as a contribution

## Tasks Completed

### Title Revision
- Changed from: "Origins and Roles of World Representations in Neural Networks"
- Changed to: "Convergent World Representations and Divergent Tasks"
- Rationale: Old title was vague; new title captures core tension between convergence (multi-task) and divergence (distance task)

### Abstract Rewrite
- Restructured to lead with convergence/divergence findings
- Key changes:
  - "conditions governing their geometry and their roles in downstream adaptability"
  - Framework clearly separating world, data generation process and model
  - Explicit mention of "divergent" tasks that harm new entity integration
  - New punchline: "training on multiple relational tasks reliably produces convergent world representations, but some lurking divergent tasks can catastrophically harm new entity integration via fine-tuning"

### Style Fixes
- Removed all Oxford commas throughout paper (user preference)

### Editorial Ick Points Added
Comments marked with `%##ICKPOINT##` for future revision:
1. "This work" paragraph doesn't mention divergent tasks - needs alignment with new title
2. Contribution bullet 2 buries divergence story
3. Discussion is dense - needs tightening, emphasize forward/backward modularity
4. Conclusion doesn't land the punchline - should echo title framing

### Key Speculation Block
Added `%## KEY SPECULATION: WHY IS DISTANCE DIVERGENT? ##` comment block in Discussion:
- Hypothesis: Divergence is property of task-architecture pairing (gradient geometry), not learned weights
- This explains why single-task CKA predicts fine-tuning failure even after joint multi-task pretraining
- Added pointers from Limitations and FT Result 1 sections

### Documentation
- Updated AUTHOR_INSTRUCTIONS.md with ICML Peer Review Ethics and desk rejection criteria

## Files Modified

- `paper_icml/main.tex` - Title, abstract, ick points, speculation comments
- `paper_icml/AUTHOR_INSTRUCTIONS.md` - Added peer review ethics section

## Key Decisions

1. **No hypothesis for why distance is divergent**: Framed as empirical observation + speculation, left mechanistic understanding to future work
2. **Gradient geometry hypothesis**: Divergence may be a property of task-architecture pairing, not the learned weights - the gradient signal from certain tasks inherently routes updates through pathways that bypass shared representations

## Open Questions / Next Steps

1. Address ick points in intro/contributions to align with new title
2. Tighten Discussion section
3. Revise Conclusion to land the convergent/divergent punchline
4. Final proofread and submission prep
