# Appendix Writing and Revision Session - ICLR 2026 Submission

## Summary
Extensive session focused on writing and improving the appendix sections of `iclr2026_conference.tex`, grounding descriptions in actual code implementations.

## Key Changes

### Appendix C: Experimental Details

#### World Section
- Fixed city count clarification: "5,075 real-world cities plus 100 synthetic Atlantis cities (5,175 total)"
- Added discussion of flat 2D manifold choice vs spherical globe:
  - Early experiments used spherical coordinates
  - No canonical geometry for nonlinear models
  - Planar enables clean linear probing
  - Cited Engels et al. and Csordas et al. on nonlinear feature extraction challenges
  - Cited Bronstein et al. for geometric deep learning context

#### Data Generation Process
- Moved Atlantis identical-outputs note to Tasks paragraph
- Added Dataset Sizes paragraph with full fine-tuning composition:
  - 100k rows target task with Atlantis
  - 20k rows from pretraining (catastrophic forgetting prevention)
  - 256 rows per task without Atlantis (elicit multi-task performance)
- Added Atlantis-in-pretraining baseline description

#### Model and Training
- Expanded Tokenization paragraph:
  - Justified character-level tokenization choice
  - Cited Bachmann et al. (2025) on pitfalls of next-token prediction
  - Explained why synthetic-friendly tokenization was avoided
- Added City ID Assignment paragraph:
  - Random assignment from [0, 9999]
  - No geographic information leakage

### Appendix D: Analysis Methods

#### Representation Extraction
- Added layer extraction info: residual stream after transformer blocks 3, 4, 5, 6
- Layer 5 is default unless otherwise specified
- Added `<bos>` token to input sequence example
- Rewrote "Omitting cities with leading zeros" paragraph:
  - Explained why 0 is special (never appears as leading digit in numerical outputs)
  - Model encodes feature distinguishing identifiers from numbers

#### Linear Probing & PCA
- Grounded in actual code (`analyze_representations_higher.py`):
  - Train/test split: 3250/1250 cities (from config)
  - OLS without regularization (method: linear)
  - Separate probes for x and y coordinates
- Fixed PCA description: color by geographic region (not longitude/hue)

#### Reconstruction Error
- Added: measured as absolute Euclidean distance

#### CKA
- Rewrote based on actual implementation (`cka_v2/compute_cka.py`, `analyze_cka_pair.py`):
  - Added formula: CKA = <K,L>_F / (||K||_F ||L||_F)
  - Linear kernel matrices K = XX^T, L = YY^T with centering
  - City filter excludes Atlantis and IDs starting with zeros
- Verified all CKA scripts use the same underlying computation

### Package/Compilation Fixes
- Added `\usepackage{booktabs}` for table formatting
- Defined `\degree` command (gensymb not available)
- Fixed xcolor package clash
- Removed duplicate Bachmann citation from bib file

## Files Modified
- `iclr2026_conference.tex` - appendix sections
- `iclr2026_conference.bib` - removed duplicate citation

## Code Files Reviewed (for grounding descriptions)
- `src/analysis/analyze_representations_higher.py` - linear probing implementation
- `src/analysis/cka_v2/compute_cka.py` - CKA computation
- `src/scripts/analyze_cka_pair.py` - CKA analysis script
- `configs/analysis_representation_higher/seed1/pt1-4_seed1/compass_firstcity_last_and_trans_l5.yaml` - probe config
- `configs/revision/exp4/cka_cross_seed_first3/*/layer5.yaml` - CKA config

## Current State
Paper compiles successfully. Appendix sections A-D are now well-written with code-grounded descriptions.
