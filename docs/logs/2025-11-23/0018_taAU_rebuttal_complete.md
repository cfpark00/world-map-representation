# taAU Rebuttal Response - Complete Polish

## Summary
Completed the full taAU reviewer response for the ICLR 2026 rebuttal. This reviewer gave a score of 2 (lowest among reviewers), so the response required careful attention to address all concerns thoroughly while maintaining a respectful tone.

## Work Completed

### W1: PRH Claims (Major Revision)
- Created detailed table showing all 7 locations where PRH claims were softened
- Reframed as "Multitask Scaling Hypothesis" evidence, not full PRH
- Added pushback: no existing testable setup for PRH despite interest
- Noted all 20 LLMs in original PRH paper are transformers

### W2: Related Works (Comprehensive Literature Survey)
Added survey of 7 papers showing gap in literature:
- Kim et al. (2024) - ICL plateaus, no representation geometry
- Aghajanyan et al. (2021) - implicit representation, no internal study
- Kumar et al. (2022) - fine-tuning features, no world representations
- Maurer et al. (2016) - theoretical only
- Yang & Hospedales (2017) - survey on architectures, not convergence
- Michaud et al. (2023) - single task family with different parameters
- Zhang et al. (2025) - behavioral study, not representations

Quoted the new Related Work section added to main text.

### W3: Surprisingness
- Acknowledged Section 3 may be unsurprising (including to us)
- Cited Gurnee & Tegmark, Marks & Tegmark, Bricken et al., Templeton et al.
- Defended Section 4 as genuinely surprising (divergent tasks)
- Ended with respectful invitation for dialogue

### W4: Weak CKA Correlation
- Agreed it's weak, noted multi-seed improved it somewhat
- Pivoted to stronger finding: divergent tasks
- Added quantification (linear probe reconstruction error)

### W5: Limited Tasks
- Defended 7 tasks (C(7,2)=21, C(7,3)=35 combinations)
- Key contributions: framework + divergent task discovery

### Minor Points
- Bullet list of all figure fixes

### Q1: Tokenization
- Clear example with `<bos>` and `<eos>` tokens
- Action: Expanded App. B

### Q2: Cross-Seed Consistency
- Rephrased distinctly from 8dPS response
- Listed 4 robust findings with figure references
- Action: 3-seed pretraining, 4-seed fine-tuning

### Q3: Crossing Task Failure
- Added "from our experience training transformers on synthetic data"
- Cited Shalev-Shwartz et al. (2017) [15] and Bachmann et al. (2024) [16]
- Action: Added loss plateau discussion to Related Work

### Q4: Linear Decoding
- Direct answer with figure reference

### Summary
- Thanked for framing/calibration help
- Politely asked for score increase

### References
Added 16 citations total with URLs

## Tone Adjustments
- Removed "extensive" from "despite extensive familiarity"
- Removed "in fact" from Q2 opening
- Ensured no condescending language

## Files Modified
- `/n/home12/cfpark00/datadir/WM_1/rebuttal/REBUTTAL.txt` (taAU section complete)

## Current State
- All 4 reviewer responses: Complete
- General section: Complete
- Rebuttal ready for final review and submission

## Next Steps
- Final read-through of entire rebuttal
- Submit to ICLR
