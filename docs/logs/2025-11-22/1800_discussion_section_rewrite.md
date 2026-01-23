# Discussion Section Rewrite - ICLR 2026 Submission

## Summary
Major restructuring and rewriting of the Discussion section. Consolidated from 5 paragraphs to 4, added extensive new citations, and refined the narrative around continual learning, representation dynamics, and forward/backward modularity.

## New Discussion Structure

### 1. Continual Learning and World Models
**Completely rewritten** with new framing:
- Opens with motivation: fundamental properties of DNNs as building blocks toward general intelligence
- World models must represent AND adapt consistently when world changes
- "Cascading updates across different computational tasks"
- "Robustly adaptable internal representations are a prerequisite for general intelligence"
- ICL vs fine-tuning gap (Brown 2020, Lampinen 2025, Park 2025 new news)
- Recent approaches: transformer augmentations (Generative Adapter, Text-to-LoRA, SEAL) and stateful architectures (LSTM, Linear Transformers, Titans, Gated Delta)
- Ends with: controlled setting to examine transformer representations under gradient descent, future direction comparing architectures

### 2. Dynamics of Representations
**Completely rewritten** with literature survey:
- Long-standing topic (Rosenblatt, Rumelhart)
- Most work on fixed networks or pretraining formation
- Growing interest in test-time/adaptation changes
- ICL representations (Park 2024, Shai 2025 belief states)
- Inference-time adaptation (Bigelow 2025, Lubana 2025)
- Fine-tuning representations (Wang 2025 steering analogy, Casademunt 2025, Minder 2025)
- Our contribution: updatable world with consistent expected changes, Atlantis as test case

### 3. Forward and Backward Modularity
**Refined** with key insight:
- Forward modularity ≠ backward modularity
- Multi-task training produces clean representations, but "fractured and partial" for adaptation
- Gradient descent routes updates through pathways bypassing shared manifold
- Computational graph structure tells little about learning signal propagation
- ICL sidesteps by operating purely in forward pass
- Open question: architectural innovations to align backward with forward modularity

### 4. Limitations
**Restructured** (benefits first):
- Our study enables holistic world→data→model pipeline analysis
- Non-trivial phenomenology despite simplifications
- Scale limitations acknowledged
- **NEW**: "findings are largely correlational---we do not yet understand the mechanisms"
- **NEW**: PRH claims are partial (single architecture/modality, no *true* multimodality or cross-architecture convergence)

## Removed Section
- **Continual world models** paragraph - key vocabulary merged into paragraphs 1 and 3

## New Citations Added to Bib (18 total)
### Stateful architectures
- hochreiter1997long (LSTM)
- schlag2021lineartransformerssecretlyfast (Linear Transformers as Fast Weight Programmers)
- behrouz2024titanslearningmemorizetest (Titans)
- yang2025gateddeltanetworksimproving (Gated Delta Networks)

### Transformer augmentations
- chen2024generativeadaptercontextualizinglanguage (Generative Adapter)
- charakorn2025texttolorainstanttransformeradaption (Text-to-LoRA)
- zweiger2025selfadaptinglanguagemodels (SEAL)

### Representation dynamics
- wang2025simplemechanisticexplanationsoutofcontext (FT as steering)
- bigelow2025beliefdynamicsrevealdual (Belief dynamics ICL/steering)

### Critical learning / plasticity
- achille2019criticallearningperiodsdeep (Critical Learning Periods)
- dohare2024maintainingplasticitydeepcontinual (Maintaining Plasticity)

## Key Vocabulary Preserved from Old "Continual World Models"
- "more than surface statistics"
- "genuine world models"
- "fractured and partial"
- "adapt consistently when the world changes"
- "cascading updates across different computational tasks"
- "robustly adaptable internal representations are a prerequisite"

## Also Added
- Citation to Brown 2020 for in-context learning
- \textit{true} multimodality (italicized per user request)
- updatable (corrected from updateable)

## Files Modified
- `iclr2026_conference.tex` - Discussion section (lines 251-276)
- `iclr2026_conference.bib` - Added 11 new citations

## Current State
Discussion section is now 4 well-structured paragraphs with extensive citations and clear narrative flow. Ready for compilation and final review.
