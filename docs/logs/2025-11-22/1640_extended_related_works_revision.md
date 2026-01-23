# Extended Related Works Revision Session - ICLR 2026 Submission

## Summary
Major cleanup and revision of Extended Related Works section in appendix. Removed duplicate citations already in main Related Works, reorganized paragraphs, added new citations, and wrote proper prose for all subsections.

## Key Changes

### Extended Related Works Cleanup
- Removed all citations already present in main Related Works (Sec. 5)
- Added "See Sec. X for main related works" references where appropriate

### Interpretability & Internal Representations Paragraph
- Removed duplicates: li2022emergent, gurnee2023language, nanda2023emergent, rosenblatt1958, rumelhart1986, marks2024, towardsmonosemanticity, templeton2024
- Kept unique citations: hubel1962receptive (neuroscience), fukushima1980neocognitron, bengio2014representationlearningreviewnew
- Added: vafa2025foundationmodelfoundusing (replaces vafa2024 which was in main)
- Added: olah2017feature (feature visualization)
- Added concepts/features/abstractions sentence with pearce2025tree, higgins2016beta, lee2025geometry, arditi2024refusal

### Fine-tuning Paragraph
- Removed duplicates from main text
- Kept unique directions: parameter efficiency (hu2021lora, lester2021), zeroth-order (malladi2024), weight composition (ilharco2023), representation adaptation (wu2024reft)
- Kept: out-of-context reasoning (treutlein2024), off-target effects (betley2025)
- Added: zhao2025echochamber, zweiger2025selfadapting

### Dynamics of Representations Paragraph (NEW prose)
- Wrote full paragraph about studying representation evolution
- In-context learning: park2024icl, shai2025transformersrepresentbeliefstate, demircan2024sparseautoencoders
- Fine-tuning: casademunt2025steering, minder2025overcomingsparsity
- SAE temporal limitations: lubana2025priorsintime
- VLM unused representations: fu2025hiddenplainsightvlms

### Geometric Deep Learning Paragraph (NEW prose)
- Cited foundational works: bronstein2021, cohen2016groupequivariant, weiler2021generale2equivariant
- Explicitly stated: we don't study geometric inductive biases
- Our approach: geometry emergent from training, not imposed

### Loss Plateaus Paragraph (NEW prose)
- Connected to our crossing task failure (escapes initial plateau but fails overall)
- Transformer-specific studies: hoffmann2024eureka, gopalani2025, singh2024needsrightinductionhead
- General optimization: shah2020simplicity, pezeshki2021gradient, bachmann2025pitfalls
- Most related: kim2025taskdiversity (multi-task shortens plateaus - similar to our crossing finding)

### Removed Sections
- "Other" paragraph (fu2025 moved to Dynamics)
- "Extended Background" paragraph (content either moved or redundant with main text)

## New Citations Added to Bib
- shai2025transformersrepresentbeliefstate
- demircan2024sparseautoencodersrevealtemporal
- lubana2025priorstimemissinginductive
- vafa2025foundationmodelfoundusing
- cohen2016groupequivariantconvolutionalnetworks
- weiler2021generale2equivariantsteerablecnns
- olah2017feature
- singh2024needsrightinductionhead

## Bug Fixes
- Removed duplicate singh2024needsrightinductionhead entry from bib (was at line 81 and 5154)

## Files Modified
- `iclr2026_conference.tex` - Extended Related Works section (lines 604-622)
- `iclr2026_conference.bib` - Added 8 new citations, removed 1 duplicate

## Compilation
- Paper compiles successfully (30 pages)
- No undefined citations or references
- Only float specifier warnings (harmless)

## Current State
Extended Related Works is now clean, well-organized, with proper prose and no duplicates from main text.
