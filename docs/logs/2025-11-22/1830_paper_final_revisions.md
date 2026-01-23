# Paper Final Revisions - ICLR 2026 Submission

## Summary
Final editing pass on the paper, focusing on toning down claims, fixing stylistic issues (em-dashes), rewriting conclusion, and addressing reviewer concerns about 2D geometry choice.

## Changes Made

### 1. Title Change
- Changed from: "World Representations in Neural Networks: A Controlled Study of Formation and Adaptation"
- Changed to: "Origins and Roles of World Representations in Neural Networks"

### 2. Result 3 Title (Pretraining Section)
- Removed "Evidence for the Platonic Hypothesis" from title
- Now just: "Result 3: Task diversity aligns representations."

### 3. PRH Claims Toned Down
- "directly connects to" → "partially connects to" the Platonic Representation Hypothesis
- "ideal testbed" → "potential testbed"

### 4. Forward/Backward Modularity Discussion
- "often fractured" → "can be fractured"
- Removed speculative sentences about computational graphs and ICL
- Kept core message: nice forward-pass representations don't guarantee nice adaptation

### 5. Limitations Section Rewritten
- New opening: "We study world representation formation and adaptation in a controlled synthetic setting with small-scale models."
- Emphasizes non-trivial phenomenology found
- Acknowledges difficulty generalizing to large-scale models

### 6. Conclusion Completely Rewritten
New structure:
1. Framework emphasis (World-Data-Model separation)
2. Key property: consistent world updates enabling fine-tuning experiments
3. Multi-task convergence finding (partial evidence for Multitask Scaling Hypothesis)
4. Main finding: convergence doesn't guarantee adaptation; divergent tasks harm integration
5. Punchline: forward/backward modularity distinction

### 7. Em-Dash Cleanup
Reduced from 7 to 3 in main text. Kept only purposeful ones:
- Result 1: `training---representations are essentially static`
- Discussion: `world coordinates---yet these world models can be fractured`
- Conclusion: `modularity---clean, structured representations do not necessarily adapt cleanly`

### 8. Geometric Deep Learning Appendix (Extended Related Works)
Addressed reviewer concern about 2D planar world:
- Added: "one might ask: why not a sphere, torus, or other manifold?"
- Explained: "not our focus" - studying adaptation in arbitrarily chosen geometry
- Key insight: "a change in world geometry can be absorbed into the task definition (e.g., geodesic vs. Euclidean distance)"
- Practical reason: planar allows clean linear probing

### 9. BibTeX Fix
- Removed duplicate entry for `zweiger2025selfadaptinglanguagemodels`

### 10. Paper Zipped
- Created `paper_submission.zip` for submission

## Files Modified
- `paper/iclr2026_conference.tex` - Multiple sections
- `paper/iclr2026_conference.bib` - Removed duplicate

## Final State
- Paper compiles successfully
- 31 pages
- All major reviewer concerns addressed
- Claims appropriately calibrated
- Ready for submission
