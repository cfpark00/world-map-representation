# SciForDL Workshop Paper Assembly

**Date**: 2026-02-04 02:20

## Summary

Split the arXiv paper (arxiv:2602.00533) into two 4-page workshop papers for the ICLR 2026 SciForDL workshop (submission deadline: Feb 4 AoE). Assembled both papers with figures, captions, abstracts, titles, and full shared appendix. Moved final papers into `paper/workshops/` to share the Overleaf git repo.

## Tasks Completed

- Evaluated paper fit for SciForDL workshop (strong match: controlled experiments, empirical phenomena, scientific method)
- Created `workshop_papers/` directory (later removed) with workshop template and call-for-papers README
- Downloaded and unzipped SciForDL 2026 style template
- Downloaded arxiv paper 2602.00533 to `resources/park_2026/paper/` (1,431 lines markdown, 22 figures)
- Analyzed figure descriptions and determined figure distribution:
  - Paper 1 (pretraining): Fig 1, 2, 3, 4
  - Paper 2 (finetuning): Fig 1, 4, 5, 6
- Assembled both papers:
  - Copied figures (main text + all appendix figures) to both
  - Copied full bibliography
  - Copied full appendix verbatim from original paper
  - Placed main text figures with original captions
  - Left all main text sections as TODO placeholders (no written text)
- Set titles:
  - Paper 1: "Multi-Task Pretraining Drives Representational Convergence in Transformers"
  - Paper 2: "Divergent Tasks Harm Representational Integration of New Entities via Fine-Tuning"
- Wrote abstracts for both papers
- Added author info (Core Francisco Park) to both; verified auto-anonymization works
- Compiled both papers with pdflatex + bibtex (21 pages each, all references resolved)
- Moved papers to `paper/workshops/` to use paper/'s Overleaf git:
  - `paper/workshops/scidl_1/scidl_pretraining.tex`
  - `paper/workshops/scidl_2/scidl_finetuning.tex`
  - `paper/workshops/scidl_template/`
- Renamed tex files from `iclr2026_conference.tex` to avoid Overleaf conflicts with root paper
- Committed and pushed to Overleaf
- Cleaned up: removed `workshop_papers/` directory, removed macOS junk and build artifacts

## Files Created/Modified

### New files
- `paper/workshops/scidl_1/` - Pretraining workshop paper (tex, bib, figures, style files)
- `paper/workshops/scidl_2/` - Finetuning workshop paper (tex, bib, figures, style files)
- `paper/workshops/scidl_template/` - Original unmodified workshop template
- `resources/park_2026/paper/paper.md` - Converted arxiv paper with AI figure descriptions
- `workshop_papers/iclr2026_scidl/README.md` - Workshop call for papers (later removed with directory)

### Modified files
- `.gitignore` - Added `workshop_papers/` (now unnecessary since directory was removed)

## Key Decisions

- Split into 2 papers rather than 1: pretraining convergence story vs finetuning divergence story
- Fig 4 (7-task model) appears in both papers: as climax in Paper 1, as setup in Paper 2
- No appendix figures selected for main text; full shared appendix in both
- Papers live in `paper/workshops/` to share the Overleaf git repo (paper/ is a separate nested git)
- Tex files renamed to `scidl_pretraining.tex` / `scidl_finetuning.tex` to avoid Overleaf filename conflicts

## Open Questions / Next Steps

- Write main text for both papers (currently all sections are TODO)
- 4-page limit for main text; unlimited appendix
- May need to trim appendix to only relevant sections per paper
- Submission deadline: Feb 4, 2026 AoE
