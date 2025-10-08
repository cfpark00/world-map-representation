# Paper Writing Session - Sections 4 and 5 of ICLR 2026 Paper

**Date:** 2025-09-25
**Time:** 03:29
**Focus:** Writing and refining results sections for "Origins and Roles of World Representations in Neural Networks"

## Summary

Completed significant writing for the ICLR 2026 paper on world representations in neural networks. Focused on Section 4 (Formation of World Representations) and began Section 5 (Fine-tuning Effects).

## Major Accomplishments

### Section 4: Formation of World Representations

1. **Result 1 - Refined existing content**:
   - Polished description of world representation emergence during training
   - Softened claim about representations being "not surprising" to more nuanced language
   - Added connection to grokking literature

2. **Result 2 - Wrote complete content**:
   - Explained how different tasks create different representational geometries
   - Added details about thread-like vs 2D manifold structures
   - Incorporated the "linear probing only surfaces what we look for" insight
   - Noted crossing task failure in single-task settings

3. **Result 3 - Wrote complete content**:
   - Presented evidence for Multitask Scaling Hypothesis
   - Described pairwise training experiments (all 7C2 combinations)
   - Highlighted CKA saturation at ~0.8 for layers 4-6
   - Added auxiliary finding about crossing task succeeding in multi-task (implicit curriculum)

### Section 5: Fine-tuning Effects

1. **Result 1 - Wrote complete content**:
   - Introduced Atlantis experiments framework
   - Explained surprising finding that single-task CKA predicts multi-task fine-tuning
   - Proposed mechanism for why this occurs
   - Described generalization matrix showing task-specific transfer patterns

2. **Setup Results 2 & 3**:
   - Outlined structure for non-linear combination effects
   - Prepared framework for elicitation vs representation problems

### Figure Management

- Added proper figure for 7-task model at end of Section 4
- Fixed figure references and captions throughout
- Clarified that panel (c) shows training curves, not just performance

### Documentation Updates

- Fully updated `/docs/paper_writing/onboarding.md` with:
  - Current paper status (Sections 1-4 complete, 5 in progress)
  - Detailed summary of all results written
  - Writing style guidelines and terminology
  - Common pitfalls to avoid
  - Remaining work needed

## Key Writing Decisions

1. **Tone adjustments**: Softened strong claims with "To the best of our knowledge"
2. **Terminology choices**: Used "generalization matrix" instead of "kernel"
3. **Structure**: Kept auxiliary findings (crossing task) after main results to maintain flow
4. **Citations**: Added proper references to grokking, gradient plateaus, fractured representations

## Technical Details Clarified

- CKA values: Single-task models < 0.3, multi-task > 0.7
- R² values: Geometric tasks > 0.8, classification tasks < 0.4
- Crossing task: Fails completely in isolation (0.0 in CKA)
- 7-task model: Trained on all tasks simultaneously for fine-tuning base
- Atlantis: 100 synthetic cities at (-35, 35) for OOD testing

## Issues Addressed

- Fixed typos and grammatical errors throughout
- Resolved inconsistent figure referencing
- Clarified ambiguous technical descriptions
- Improved flow between results sections

## Next Steps (for future sessions)

1. Complete Section 5 Result 2 (non-linear fine-tuning effects)
2. Complete Section 5 Result 3 (elicitation efficiency)
3. Update Discussion section with specific quantitative results
4. Add missing figures for Section 5
5. Polish abstract once all results are complete
6. Trim to meet 8-page conference limit

## Files Modified

- `/paper/iclr2026_conference.tex` - Main paper file with new content
- `/docs/paper_writing/onboarding.md` - Complete update for new writers

## Notes

The paper is shaping up well with a clear narrative arc: from how representations form (Section 4) to how they affect adaptation (Section 5). The surprising finding about single-task CKA predicting multi-task fine-tuning behavior is particularly interesting and well-positioned as a key contribution.