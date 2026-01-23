# Paper Polish and Anonymization Check

## Summary
Final polish pass on ICLR 2026 paper focusing on citation style, terminology consistency, and anonymization verification.

## Changes Made

### 1. Citation Style Fixes (`\cite` → `\citet`)
Fixed 5 instances where citations were used as sentence subjects but used `\cite` instead of `\citet`:
- Line 91: `\cite{jain2023mechanistically}` → `\citet{jain2023mechanistically}`
- Line 91: `\cite{nishi2024representation}` → `\citet{nishi2024representation}`
- Line 637: `\cite{lubana2025priorstimemissinginductive}` → `\citet{lubana2025priorstimemissinginductive}`
- Line 637: `\cite{fu2025hiddenplainsightvlms}` → `\citet{fu2025hiddenplainsightvlms}`
- Line 643: `\cite{kim2025taskdiversityshortensicl}` → `\citet{kim2025taskdiversityshortensicl}`

### 2. Section Title Fix
- "Related Works" → "Related Work" (singular is standard academic convention)
- Also fixed in appendix "Extended Related Works" → "Extended Related Work"
- Fixed reference text "extended related works" → "extended related work"

### 3. Task Naming Consistency
- Fixed `\texttt{Triangle area}` → `\texttt{triangle area}` (lowercase in prose)
- Kept `\texttt{triarea}` in technical appendix tables where actual data format is shown

### 4. Self-Citation Reduction (8 → 4)
Removed redundant self-citations to reduce from 8 to 4 total:
- `park2024iclrincontextlearningrepresentations`: 4 → 1 (kept in Discussion line 273 where most relevant)
- `park2025textitnewnewssystem2finetuning`: 2 → 1 (kept in Discussion line 271)
- Remaining: `park2024competition` (1), `park2024emergencehiddencapabilitiesexploring` (1)

### 5. Discussion Section Label Fix
- `\label{app:discussion}` → `\label{sec:discussion}` (was incorrectly labeled as appendix)

### 6. Footnote Wording
- "We believe \textit{linear} decodability..." → "We regard \textit{linear} decodability as..."
- Sounds more confident/assertive

### 7. Quantitative Claim Precision
- "roughly 90% of training" → "first ${\sim}$15% of training" (actually measured)
- Fixed in both main text (line 125) and appendix (line 500)

### 8. Hedging Removal
- Removed "It is unclear if this is generally true." from line 206
- The preceding "though this relationship is noisy" already hedges sufficiently

### 9. Citation Spacing Fix
- `GeoNames\cite{...}` → `GeoNames~\citep{...}` (proper spacing)

### 10. Figure Reference Added
- Added `(Fig.~\ref{fig:7taskmodel})` reference to 7-task model discussion

## Anonymization Verification
- Confirmed `\iclrfinalcopy` is commented out - author info hidden in PDF
- PDF shows "Anonymous authors" and "Paper under double-blind review"
- All self-citations use third person (no "our prior work", "we previously showed", etc.)
- No identifying info in comments or file paths

## Files Modified
- `paper/iclr2026_conference.tex`

## Final State
- Paper recompiled successfully (30 pages)
- Created new `paper.zip` with all changes
- Ready for anonymous submission
