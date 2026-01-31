# Session Log: ICML Appendix Content Restoration

**Date**: 2026-01-27 11:10

## Summary
Restored missing appendix content from ICLR version to ICML paper, merged duplicate Extended Related Work sections, and fixed various compilation issues.

## Tasks Completed

### Appendix Content Restoration
- **Normalized Improvement equations**: Added both error-based and accuracy-based NI formulas (lines 433-441 from ICLR)
- **Representation Extraction**: Added full explanation with colorbox token example, "Omitting cities with leading zeros" paragraph
- **Linear Probing & PCA**: Added Linear Probing, PCA, and Reconstruction Error paragraphs
- **CKA**: Added full CKA formula and computation details with citation to kornblithcka

### Section Explanations Added
- **Qualitative Representations**: Added 2 paragraphs describing task-specific patterns (thread-like, manifold, clusters)
- **Training Dynamics**: Added 2 paragraphs explaining figure panels and key observations
- **Representation Dynamics**: Added full paragraph about early representation stabilization with mircea2025 citation
- **Single-Task CKA Across Layers**: Added observations about layer progression, distance divergence, crossing failure
- **Two-Task CKA**: Added explanation of alignment increase and inter-seed variance reduction
- **CKA vs Task Count (Per-Seed)**: Added seed pooling explanation
- **Aggregated CKA Trends**: Added non-overlapping pairs methodology and seed variability findings
- **CKA vs Generalization (Annotated)**: Updated caption with task abbreviation legend
- **Additional Fine-Tuning**: Added intro text and improved captions
- **Pretraining with Atlantis**: Added full explanation of OOD integration failure being optimization dynamics
- **Wider Model**: Added capacity experiment explanation with distance-containing combination findings

### Extended Related Work Merge
- Merged two duplicate Extended Related Work sections (lines 374 and 683)
- Final section has 7 topics:
  1. Internal Representations (merged + new citations)
  2. Fine-tuning (merged + new citations)
  3. Multi-task Learning
  4. Synthetic Data
  5. Dynamics of Representations (from ICLR)
  6. Geometric Deep Learning (from ICLR)
  7. Loss Plateaus (from ICLR)
- Fixed duplicate `\label{app:related}` issue

### Compilation Fixes
- Removed duplicate bib entries I accidentally added (olah2017feature, pearce2025tree, li2025tracing, higgins2016beta, arditi2024refusal, lee2024mechanistic)
- All citations already existed in bib file

## Files Modified
- `paper_icml/main.tex` - All appendix content additions and merges
- `paper_icml/main.bib` - Removed duplicate entries

## Key Decisions
1. All citations from ICLR Extended Related Work were verified to exist in main.bib before adding
2. Merged Extended Related Work by keeping first section structure (bold headers) and adding 3 new topics from second section
3. Kept all detailed figure captions from ICLR version

## Compilation
- Paper compiles successfully at 41 pages
- No undefined citation warnings
- No duplicate label warnings

## Git
- Committed and pushed: `0c1576e` "Restore missing appendix content from ICLR, merge Extended Related Work sections"

## Next Steps
- Final read-through before submission
- Check 8-page main body limit compliance
- Submit before January 28 AoE deadline
