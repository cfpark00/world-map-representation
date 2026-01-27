# Session Log: ICML Paper Setup

**Date**: 2026-01-23 04:24
**Summary**: Set up ICML 2026 paper submission by porting content from ICLR paper and resolving merge conflicts with Overleaf.

## Tasks Completed

### Paper Setup
- Cloned Overleaf repo for ICML paper (`paper_icml/`)
- Copied entire content from ICLR paper (`paper/iclr2026_conference.tex`) to ICML format
- Copied bibliography file (`main.bib`)
- Created `figures/` directory and copied all 21 figures

### LaTeX Fixes
- Installed texlive packages: `texlive-latex-extra`, `texlive-fonts-recommended`, `texlive-bibtex-extra`, `biber`
- Added missing packages: `enumitem`, `xcolor`, `wrapfig`
- Defined `\degree` command
- Fixed BibTeX accent issue: `raventós` → `raventos` in citation key
- Removed 27 duplicate bibliography entries

### Figure Formatting
- Made figures 1, 2, 3 two-column (`figure*`)
- Converted wrapfigures to single-column `figure`
- Made figure 6 and all appendix figures two-column
- Moved figure 6 earlier in source (after equation) to improve float placement

### Text Cleanup
- Replaced all em-dashes (`—`) with commas
- Shortened `Improvement` to `Imp` in overflow equation, added definition in text

### Overleaf Sync
- Resolved merge conflict where figure 6 was duplicated
- Successfully pushed all changes to Overleaf

### Submission Prep
- Created `AUTHOR_INSTRUCTIONS.md` (gitignored) with ICML submission guidelines
- Paper compiles at 21 pages (main body ~8 pages + refs + appendix)
- Added Impact Statement section
- Confirmed no page adjustment needed for now

## Files Modified/Created
- `paper_icml/main.tex` - Main ICML paper (ported from ICLR)
- `paper_icml/main.bib` - Bibliography
- `paper_icml/figures/` - All figures (21 files)
- `paper_icml/.gitignore` - Added AUTHOR_INSTRUCTIONS.md
- `paper_icml/AUTHOR_INSTRUCTIONS.md` - ICML submission guidelines

## Key Decisions
- Used `[t]` placement for `figure*` instead of `[h]` (which is ignored for two-column floats)
- ICML page limit: 8 pages main body, unlimited references and appendix
- No acknowledgements for anonymous submission

## Open Questions / Next Steps
- Paper is ready for submission as-is
- May need page adjustments later if content changes
