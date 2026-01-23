# Rebuttal Writing Session

## Summary
Extended session working on ICLR 2026 rebuttal responses for reviewers PuHx and 8dPS. Rewrote informal draft responses into polished, professional rebuttals with proper citations and figure references.

## Work Completed

### Reviewer PuHx Response (Complete)
Rewrote all weaknesses and questions:

- **W1 (Small Figures)**: Acknowledged, described figure merging and resizing
- **W2 (Missing Citations/Legend)**: Fixed, expanded related works
- **W3 (Quantitative Measure for Ill-Integration)**: Added linear probe reconstruction error metric (~5× higher for distance task), referenced Fig. 6d
- **W4 (Figure 8a Hard to Interpret)**: Transposed matrix, moved raw to appendix, added Fig. 6c
- **Q1 (Real-World LLMs)**: Two-angle response (pretraining vs fine-tuning), added citations to reversal curse [1] and Lampinen [2]
- **Q2 (Interventions to Increase CKA)**: Clarified correlational finding, not causal
- **Q3 (Isolating Non-Overlapping Pairs)**: Explained no good matrix permutation exists, added annotation

Added references:
- [1] Berglund et al. (2024) - Reversal Curse
- [2] Lampinen et al. (2025) - Knowledge updating

### Reviewer 8dPS Response (Complete)
Rewrote all weaknesses and questions:

- **W1 (Single Seed)**: Major rewrite emphasizing:
  - We never claimed visual similarity = high CKA (quoted paper)
  - Same-task-different-seed CKA can be low, especially for distance
  - All core findings replicate across 3 seeds (63 pretraining models)
  - CKA-to-generalization correlation improved with averaging (R²: 0.126 → 0.188)
  - Bonus finding: multi-task reduces cross-seed variance

- **W2 (City ID Assignment)**: Explained character-level tokenization rationale (citing Bachmann [1]), soft prompts analogy [7], random ID assignment, leading-zero exclusion explained

- **W3 (Clustered Atlantis)**: A priori reasoning + pretraining-with-Atlantis experiment, mentioned scattered Atlantis experiment in progress

- **W4 (2D Planar World)**: Explained analysis clarity rationale, cited Engels & Tegmark [2] and Csordás et al. [3] for nonlinear readout challenges

- **W5 (Limited Explanation of Divergent Tasks)**: Honest answer that we don't fully understand why, focused on demonstrating phenomenon

- **Q1 (Crossing + Distance)**: Crossing succeeds with any task (loss plateau literature [4,5,6]), distance is divergent in representation not training

- **Q2 (Linear Probe Accuracies)**: Pointed to App. Fig. 8 for training dynamics, Fig. 6b/6d for Atlantis integration

- **Q3 (Normalized Improvement Definition)**: Added full formulas for log-ratio and linear normalization

Added references:
- [1] Bachmann et al. (2024) - Pitfalls of next-token prediction
- [2] Engels & Tegmark (2024) - Nonlinear representations
- [3] Csordás et al. (2024) - Onion representations
- [4] Hoffmann et al. (2023) - Eureka moments
- [5] Gopalani et al. (2025) - Loss plateaus
- [6] Kim et al. (2024) - Task diversity shortens plateau
- [7] Lester et al. (2021) - Soft prompts

### Remaining Work (for tomorrow)
- **oLe1**: Needs rewrite (has `[RESPONSE]` and `###` placeholders)
- **taAU**: Needs rewrite (still in draft form)

## Key Decisions Made
1. Strong defensive stance on single-seed criticism - we never claimed visual = CKA
2. Honest about not knowing why distance is divergent
3. Mentioned scattered Atlantis experiment as in-progress
4. Added "beg for score" at end of summaries
5. Expressed genuine thanks for multi-seed suggestion (it really improved the paper)

## Files Modified
- `/n/home12/cfpark00/datadir/WM_1/rebuttal/REBUTTAL.txt`

## Current State
- PuHx: Ready to submit
- 8dPS: Ready to submit
- oLe1: Needs work tomorrow
- taAU: Needs work tomorrow
