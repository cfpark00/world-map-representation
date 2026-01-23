# Paper Editing Session - ICLR 2026 Submission

## Summary
Extensive editing session for `iclr2026_conference.tex` focusing on Section 3 (Pretraining) and Section 4 (Fine-tuning).

## Key Changes

### Section 3: World Representations Converge Under Multi-Task Learning
- Shortened section title (removed "Task-dependent")
- Result 3 CKA paragraph: clarified experimental setup (7 selected pairs for 3 seeds, not all 21)
- Fixed flow: "CKA is high → might expect high if tasks shared → but even disjoint pairs show high alignment"
- Added note about crossing task excluded in Fig 3d caption
- Changed "controlled evidence" → "partial evidence" for Multitask Scaling Hypothesis
- Added footnote about needing cross-architecture convergence for full PRH test
- Auxiliary finding about crossing task: clarified it works when paired with ANY other task
- Added open question about why multi-task training drives linearly surfaced representations

### Section 4: Divergent Tasks Harm Entity Integration via Fine-Tuning
- New section title (was "Fine-tuning's Representational Shift Predicts Downstream Generalization")
- Replaced all "Atlantis" with `\texttt{Atlantis}` throughout paper for consistency
- Result 1: Simplified fine-tuning method description, removed footnote, added appendix reference
- Added figure caption improvements (result2-1: explained rows/columns, panel b explanation)
- Defined terminology with bold: **Divergent tasks**, **Hidden spaces**
- Formatted hypothesis in quote block with italic (matching Platonic Hypothesis style)
- Result 2: Introduced "best-teacher model" terminology for the heuristic expectation
- Fixed equation notation (D_j instead of j)
- Updated figure result2-2 caption with new panel descriptions (a-d)
- Changed "red vertical bands" → "red horizontal bands" (models are rows now)
- Result 3: Connected explicitly to Question 2 from hypothesis
- Rewrote Result 3 to clearly explain two exemplar models and linear probe analysis
- Added link to interactive 3D visualizations

### Citations
- Added kumar2022finetuningdistortpretrainedfeatures for gradient descent citation
- Added mccloskey1989catastrophic for catastrophic forgetting (replaced Kirkpatrick_2017)

### Other
- SGD → "gradient descent" throughout
- Various figure reference fixes (1-1, 1-2, result2-1, result2-2)
- Added appendix labels: app:world, app:data
- Fixed arrows in captions (→ to $\rightarrow$)

## Files Modified
- `iclr2026_conference.tex` - main paper
- `iclr2026_conference.bib` - added citations

## Current State
Paper compiles successfully at 24 pages. Still has undefined reference warnings for appendix figures that need to be created (fig:app_reprs, fig:app_cka_pt1, fig:app_training, app:finetuning, app:cka, app:reprs).
