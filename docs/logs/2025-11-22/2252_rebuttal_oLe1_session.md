# Rebuttal oLe1 Response Session

## Summary
Continued ICLR 2026 rebuttal work, focusing on polishing the oLe1 reviewer response. Rewrote informal draft into professional response with proper structure and citations.

## Work Completed

### Reviewer oLe1 Response (Complete)
Rewrote all weaknesses and questions:

- **W1 (Introduction Meanders)**: Described actual revisions made:
  - Removed LLM debate paragraph
  - Removed speculative questions about disentanglement, learning algorithms
  - Created direct path: open questions → controlled setup → framework → contributions
  - Moved literature to dedicated Related Work section

- **On Surprisingness**: Acknowledged multi-task convergence aligns with PRH expectations, but emphasized:
  - Controlled setup enables precise follow-up questions
  - Highlighted unexpected "divergent tasks" finding (Section 4)
  - Added "we didn't expect it at all"

- **Formatting**: Fixed missing reference

- **Q1 (Regularization/Training Factors)**:
  - Explained weight decay/LR variations explored
  - Asked clarifying question about "speeding up alignment"
  - Noted representations stabilize after sharp loss drop (App. Fig. 16)
  - Added citations [1,2] for gradient plateau literature

- **Q2 (Model Size)**:
  - Clarified we don't know pretraining interaction with scale
  - Reported 2× wider model result: divergent pattern persists (App. Fig. 20)
  - Acknowledged systematic scale study as future work

Added references:
- [1] Pezeshki et al. (2021): https://arxiv.org/abs/2011.09468
- [2] Shah et al. (2020): https://arxiv.org/abs/2006.07710

### Bug Fixes
- Fixed typo "We the original" → "We agree the original"
- Fixed typo "create use hidden" → "use hidden"
- Fixed typo "Howver" → "However"
- Fixed typo "atleast" → "at least"
- Fixed typo "confirmwhetehr" → "confirm whether"

### Figure Reference Corrections
- App. Fig. 19 = Pretraining with Atlantis experiment
- App. Fig. 20 = Wider model experiment
- Updated all references throughout rebuttal to use correct figure numbers

### Citation Memo Added
- Added memo in taAU Q2 response flagging incorrect citation:
  - Text says "Entezari et al., 2022, Linear Mode Connectivity"
  - Arxiv link is actually Keller Jordan paper
  - Needs verification/correction

## Files Modified
- `/n/home12/cfpark00/datadir/WM_1/rebuttal/REBUTTAL.txt`

## Current State
- PuHx: Ready
- 8dPS: Ready
- oLe1: Ready
- taAU: Still needs work (has [RESPONSE] placeholder, citation issues)

## Next Steps
- Polish taAU response
- Verify/fix the Entezari citation issue
- Add loss plateau citations to taAU Q3
