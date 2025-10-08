# Paper Writing Onboarding: World Representations Research

## Quick Start
**Paper Title:** Origins and Roles of World Representations in Neural Networks
**Target Venue:** ICLR 2026
**Paper Location:** `/paper/iclr2026_conference.tex`
**Bibliography:** `/paper/iclr2026_conference.bib`
**Status:** Paper is ~75% complete. Sections 1-4 fully written with figures. Section 5 partially written with incomplete results.

**Last Updated:** September 25, 2024 (Session 2)

---

## Current Paper Structure & Status

### ✅ Complete Sections
1. **Abstract**: Placeholder Lorem Ipsum for length calibration (needs real abstract)
2. **Introduction**: Condensed overview of framework and contributions
3. **Related Works**: 3 paragraphs covering fine-tuning, interpretability, synthetic data
4. **Section 3: Experimental Framework**: Geographic setup, 7 tasks, Atlantis experiments
5. **Section 4: Task-dependent world representations converge under multi-task learning**
   - Result 1: World representations emerge through autoregressive training ✅
   - Result 2: Data generation process controls world representation geometry ✅
   - Result 3: Task diversity aligns representations: Evidence for the Platonic Hypothesis ✅

### ⚠️ In Progress: Section 5 (Fine-tuning)
- **Result 1** ✅: Single-task pretraining CKA predicts fine-tuning generalization
  - Includes new hypothesis about "naturally diverging tasks" and "hidden spaces"
  - Figure 7 (result2-1): Generalization matrix and CKA correlation scatter plot
- **Result 2** 📝: Fine-tuning effects combine non-linearly [TO BE WRITTEN]
- **Result 3** 📝: Elicitation efficiency depends on representations [TO BE WRITTEN]

### 📝 Needs Work
- **Discussion**: Currently has placeholders, needs quantitative results added
- **Abstract**: Replace Lorem Ipsum with real abstract (~150 words)
- **Appendix references**: Some sections reference appendices that may need updating

---

## Figures in Paper

### Main Text Figures
1. **Figure 1** (fig1.png): Overview of World-Data-Model framework
2. **Figure 2** (setup.png): Seven geometric tasks visualization
3. **Figure 3** (result1-1.png): Training dynamics for angle task - shows probing R² vs task accuracy
4. **Figure 4** (result1-2.png): Task-dependent representation geometries (PCA) + CKA matrices
5. **Figure 5** (result1-3.png): Multi-task convergence results - CKA between models with varying task counts
6. **Figure 6** (7taskmodel.png): 7-task model PCA, probes, and training curves
7. **Figure 7** (result2-1.png): Fine-tuning generalization matrix (left) and CKA correlation scatter (right)

### Appendix Figures
- **Figure 8** (cities_map.png): Geographic distribution of 5,175 cities [Caption TODO]

**Note**: All figure captions have been shortened for conciseness. Figures use standard `\begin{figure}` environments (wrapfig package not available on system).

---

## Core Research Story

### The Framework
We decouple **world** (5,175 city coordinates) from **data generation** (7 geometric tasks) to study how different tasks shape neural representations.

### Key Findings (What's Actually in the Paper)

#### Pretraining Stage (Section 4)
1. **World representations emerge** through standard autoregressive training, with coordinate decodability appearing before task accuracy
2. **Different tasks → different geometries**: Distance creates thread-like structures, angle creates 2D manifolds, crossing fails entirely
3. **Multi-task drives convergence**: Training on multiple tasks aligns representations (CKA ~0.8), providing first controlled evidence for Platonic Hypothesis

#### Fine-tuning Stage (Section 5)
1. **Single-task CKA predicts generalization**: Surprisingly, representations from single-task pretraining predict how well Atlantis generalizes across tasks after multi-task fine-tuning
2. **New conceptual framework**: "Naturally diverging tasks" (low CKA when trained alone) and "hidden spaces" (representations not surfaced by PCA/probing)

### The Atlantis Experiment
- 100 synthetic cities added to Atlantic Ocean
- Fine-tuning protocol: 100k examples with Atlantis + 256 elicitation examples
- Tests whether models can consistently update world representations
- Result: Task-dependent generalization patterns correlate with single-task CKA

---

## Writing Guidelines (CRITICAL)

### Style Rules
1. **Be concise** - Paper targets 8-page limit, we're currently at ~10 pages
2. **Use "we"** not passive voice
3. **Soften claims** with "to the best of our knowledge"
4. **Figure references**: Always use `Fig.~\ref{}`
5. **Task names**: Use `\texttt{distance}`, `\texttt{crossing}`, etc.
6. **No emojis** unless explicitly requested
7. **No excessive comments** in text

### Critical Terminology
- **World-Data-Model Framework**: Our three-component separation
- **Naturally diverging tasks**: Tasks with low CKA when trained in isolation
- **Hidden spaces**: Representation spaces not surfaced by PCA or probing
- **Representational alignment**: Similarity measured by CKA
- **Elicitation vs. representational problems**: Two failure modes in fine-tuning

### Recent Changes (Sept 25, 2024)

#### Session 1
1. Updated all section titles for clarity:
   - Section 4: "Task-dependent world representations converge under multi-task learning"
   - Result titles now more active and specific
2. Shortened all figure captions by ~50% for space
3. Condensed "This work" paragraph from 200→70 words
4. Fixed Multitask Scaling Hypothesis quote spacing
5. Added Figure 7 (result2-1) for fine-tuning results
6. Updated fine-tuning methodology description (100k + 256 examples)
7. Introduced "naturally diverging tasks" hypothesis

#### Session 2 (Latest)
1. Further refined Section 4 title after several iterations
2. Updated all three Result paragraph titles to be more concise and active:
   - Result 1: "World representations emerge through autoregressive training"
   - Result 2: "Data generation process controls world representation geometry"
   - Result 3: "Task diversity aligns representations: Evidence for the Platonic Hypothesis"
3. Added comprehensive figure captions for:
   - Figure 3 (result1-1): Training dynamics showing probing vs task accuracy
   - Figure 4 (result1-2): PCA projections and CKA matrices
   - Figure 5 (result1-3): Multi-task convergence results
   - Figure 7 (result2-1): Fine-tuning generalization matrix and CKA correlation
4. Shortened captions for Figures 4, 5, and 6 for space constraints
5. Condensed "This work" paragraph to exactly 70 words
6. Added placeholder Lorem Ipsum abstract for length calibration
7. Fixed wrapfig package issue by reverting to standard figure environments
8. Rewrote hypothesis paragraph introducing "naturally diverging tasks" concept
9. Corrected fine-tuning methodology to accurately describe two-stage approach

---

## Immediate Tasks for New Writers

### High Priority
1. **Write Section 5 Result 2**: Fine-tuning effects combine non-linearly
   - Need to analyze how multiple task fine-tuning interacts
   - Show that effects aren't simply additive

2. **Write Section 5 Result 3**: Elicitation efficiency analysis
   - Distinguish elicitation (quick fix) vs representation (fundamental) problems
   - Provide empirical evidence for the distinction

3. **Write real abstract** (~150 words):
   - Framework introduction
   - Key findings (task-dependent geometry, convergence, fine-tuning prediction)
   - Implications for field

### Medium Priority
4. **Update Discussion section**: Add specific quantitative results
5. **Complete Appendix Figure 8 caption**: Geographic distribution description
6. **Review for length**: Need to cut ~2 pages

### Optional Enhancements
7. Consider additional ablations for reviewer concerns
8. Strengthen connection to broader literature
9. Add limitations discussion if space permits

---

## Technical Details

### Data Format
- Input: `dist(c_0865,c_4879)=769`
- Character-level tokenization: `d i s t ( c _ 0 8 6 5 , c _ 4 8 7 9 ) = 7 6 9`
- 98 ASCII vocabulary, no special tokens

### Model Architecture
- Decoder-only Transformer
- 6 layers, 128 hidden dim, 4 heads
- ~2.5M parameters
- Standard autoregressive training (no loss masking)

### Training Details
- **Pretraining**: 1M examples per task, 15 epochs
- **Fine-tuning**: 100k Atlantis examples + 256 elicitation, 20 epochs
- **Batch sizes**: 128 (pretraining), 64 (fine-tuning)
- **Learning rate**: 3e-4 (pretraining), 1e-5 (fine-tuning)

### Analysis Pipeline
1. Train models with different task combinations
2. Probe representations with Ridge regression for (x,y) coordinates
3. Compute CKA between models/layers
4. Visualize with PCA projections
5. Test Atlantis generalization

---

## Key Files and Directories

```
/paper/
├── iclr2026_conference.tex    # Main paper
├── iclr2026_conference.bib    # Bibliography
├── figures/
│   ├── fig1.png               # Overview figure
│   ├── setup.png              # 7 tasks visualization
│   ├── result1-1.png          # Training dynamics
│   ├── result1-2.png          # Task geometries
│   ├── result1-3.png          # Multi-task convergence
│   ├── 7taskmodel.png         # 7-task model
│   ├── result2-1.png          # Fine-tuning matrix
│   └── cities_map.png         # Geographic distribution

/configs/
├── analysis_*/                # Various analysis configs
├── eval/                      # Evaluation configs
├── data_generation/           # Data creation configs
└── training/                  # Training configs
```

---

## Common LaTeX Issues

### Compilation
```bash
cd /paper
pdflatex iclr2026_conference.tex
bibtex iclr2026_conference
pdflatex iclr2026_conference.tex
pdflatex iclr2026_conference.tex
```

### Package Notes
- **No wrapfig**: System doesn't have wrapfig.sty, use standard figures only
- **Citations**: All references in .bib file, use \citep{} for citations
- **Math mode**: Remember $ $ for math symbols like R²
- **Figure placement**: Use [t] or [h] for placement, avoid [H] (requires float package)
- **Quote spacing**: Avoid blank lines before/after \begin{quote} blocks

---

## Review Preparation Checklist

- [ ] Abstract written (not Lorem Ipsum)
- [ ] All TODOs removed from text
- [ ] All figure captions complete
- [ ] Paper under 8 pages (currently ~10)
- [ ] All claims have citations or "to our knowledge"
- [ ] Limitations section written
- [ ] Reproducibility statement verified
- [ ] Anonymous (no author names for review)

---

## Quick Onboarding Checklist

### Day 1: Understanding
1. Read the current paper (`/paper/iclr2026_conference.tex`)
2. Focus on Sections 3-4 to understand the framework and main results
3. Review all 7 figures to understand visual story
4. Read Section 5.1 to see what's written about fine-tuning

### Day 2: Contributing
1. Pick either Result 2 or Result 3 in Section 5 to write
2. Look at relevant configs in `/configs/` for the experiments
3. Check what analysis has been done (ask team for notebooks/results)
4. Write 1-2 paragraphs following style of Result 1

### Day 3: Polishing
1. Write the abstract
2. Help trim paper to 8 pages
3. Update Discussion with quantitative results
4. Final proofread for consistency

---

## Contact & Resources

- Paper draft: `/paper/iclr2026_conference.tex`
- Interactive visualizations: [OSF Link](https://osf.io/4huk3/?view_only=8ddee09b18ed43b0a302b96f6bfecd50)
- Previous logs: `/docs/logs/`

Remember: The goal is controlled, systematic evidence about how neural networks form and use world representations. Every claim should tie back to this central narrative.