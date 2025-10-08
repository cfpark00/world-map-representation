# Paper Structure Setup and Writer Onboarding
**Date**: 2025-09-21
**Time**: 14:57
**Focus**: Setting up ICLR 2026 paper structure and creating onboarding documentation

## Summary
Set up complete paper structure for ICLR 2026 submission, established appendix organization, and created comprehensive onboarding documentation for new paper writers.

## Key Accomplishments

### 1. Paper Setup and Structure
- Extracted and compiled ICLR 2026 LaTeX template
- Updated author information (Core Francisco Park, Harvard/CBS-NTT)
- Created custom `\todo{}` command for red bracketed notes

### 2. Paper Content Organization

#### Title and Sections
- **New Title**: "The Origins and Roles of World Representations in Neural Networks"
- **Section 3 Renamed**: "Setup: A Model System of Worlds" → "Experimental Framework"

#### Introduction Restructuring
- Changed format to use `\paragraph{This work.}` style
- Converted contributions to bulleted list with tighter spacing
- Removed unsubstantiated backward propagation claim
- Replaced with task-dependent representation quality finding

#### Related Works
- Reordered to: Pretraining & Fine-tuning, Interpretability, Internal representations
- Converted all paragraph titles to sentence case

#### Discussion
- Reorganized order: Origins → Off-target effects → Continual models → Limitations
- Added todo note about adaptation being key for generalization

### 3. Appendix Structure
Created 5-section appendix with placeholder content:
- **A. World Setup** - Geographic data and Atlantis construction
- **B. Tasks and Datasets** - 11 task definitions and sampling strategies
- **C. Model and Training** - Architecture and hyperparameters
- **D. Analysis Methods** - Linear probing, PCA, gradient analysis
- **E. Code and Data Availability** - Simple availability statement

Added `\clearpage` before appendix and "APPENDIX" header for clear separation.

### 4. Bibliography Updates
- Added Vafa et al. 2024 citation on evaluating world models
- Fixed bibliography compilation issues

### 5. Writer Onboarding Documentation
Created comprehensive `/docs/paper_writing/onboarding.md` covering:
- Research questions and objectives
- Key findings and experimental approach
- Important code files and data locations
- Writing guidelines and terminology
- Current paper structure
- Open questions and tasks for new writers

## Key Changes to Paper Narrative

### Removed Claims
- Backward propagation of representations (not conclusively established)

### Added/Emphasized Claims
- Task-dependent representation quality (distance/area vs classification tasks)
- Data diversity promoting better factorization
- Off-target effects of fine-tuning correlating with OOD performance

## Technical Details
- Paper currently 4 pages main + 2 pages appendix
- All sections have placeholder text with clear TODO markers
- Itemize spacing customized using enumitem package
- Appendix starts on new page (page 5)

## Files Modified
- `/paper/iclr2026/iclr2026_conference.tex` - Main paper
- `/paper/iclr2026/iclr2026_conference.bib` - Bibliography
- `/docs/paper_writing/onboarding.md` - New onboarding document

## Next Steps
- Fill in actual results for Sections 4 and 5
- Add specific citations to Related Works
- Create key figures for representation evolution
- Determine which tasks produce best representations
- Add subsections to appendix sections

## Notes
- User prefers concise, direct writing style
- Emphasis on controllability of framework
- Paper should acknowledge limitations upfront
- Focus on practical implications for training robust models