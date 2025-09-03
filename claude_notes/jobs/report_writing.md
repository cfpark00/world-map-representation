# Report Writing Guidelines

## Overview
Reports in this project are research blog posts written in MDX format for publication. They should communicate research findings clearly while maintaining scientific rigor.

## Prerequisites
Before writing any report, read `/n/home12/cfpark00/WM_1/claude_notes/docs/mdx-blog/README.md` for MDX blog format specifications and technical requirements.

## Writing Style
- **Tone**: Excited and engaging without overhyping. Use precise language that conveys enthusiasm through clarity of insight rather than superlatives
- **Avoid hype words**: Skip "revolutionary", "breakthrough", "game-changing". Instead use "demonstrates", "reveals", "suggests"
- **Be scientifically honest**: Present limitations alongside achievements
- **Target audience**: Technical readers who appreciate depth but also want clear takeaways

## Report Structure

### 1. TL;DR Section (Required)
- Place at the very top of the report
- 2-3 sentences maximum
- Summarize the key finding and its significance
- Include the most important quantitative result

### 2. Hero Visual (Required)
- Immediately after TL;DR
- Choose the most compelling figure/GIF/interactive element
- This should instantly communicate the core result visually
- Prefer animated GIFs showing evolution/progression when available
- Include a descriptive caption explaining what the reader is seeing

### 3. Main Content Flow
- **Introduction**: Set up the problem and why it matters
- **Experiment Design**: What you did and why
- **Results**: What you found with supporting visualizations
- **Analysis**: What the results mean
- **Implications**: Why this matters for the field
- **Future Work**: Natural next steps (brief)
- **Conclusion**: Concise wrap-up

### 4. Technical Appendix (Required)
- Use foldable `<details>` HTML element
- Dump all technical specifications here:
  - Full model architecture configs
  - Training hyperparameters
  - Dataset generation details
  - Evaluation metrics formulas
  - Computational resources used
  - Code snippets for key algorithms
- This keeps the main text accessible while preserving reproducibility

## Preparation Workflow

### 1. Comprehensive File Review
Before writing, read ALL relevant files:
- Training scripts (`src/training/`)
- Configuration files (`configs/`)
- Data generation scripts (`src/data_processing/`)
- Analysis scripts (`src/analysis/`)
- Experimental outputs (`outputs/experiments/*/`)
- Any relevant notebooks (`notebooks/`)

### 2. Asset Collection
- Copy figures/GIFs from experimental result directories to report directory
- Rename files descriptively (e.g., `training_dynamics.png` not `summary.png`)
- If existing plots are inadequate, create new visualizations:
  ```bash
  cd reports/your-report-name/
  python generate_figures.py  # Small focused scripts for custom plots
  ```

### 3. Content Development
- Start with outline based on experimental narrative
- Write sections maintaining logical flow
- Add visualizations at natural break points
- Ensure each section flows into the next

## Quality Checklist

### Before Finishing
1. **Validate all numbers**: Cross-check metrics against experimental outputs
2. **Test all image paths**: Ensure images load correctly
3. **Verify code snippets**: Any included code should be runnable
4. **Check citations**: Link to relevant papers/resources
5. **Flow check**: Read through ensuring smooth transitions
6. **Technical accuracy**: Verify all technical claims against actual results
7. **Run MDX validator**: Validate the MDX syntax and structure:
   ```bash
   cd /n/home12/cfpark00/WM_1/claude_notes/docs/mdx-blog/validator-internals
   node validate.js /absolute/path/to/your/report/index.mdx
   ```
   The validator checks MDX compilation, import statements, and syntax.

### MDX-Specific Validation
- Ensure proper import statements for all images
- Check that interactive elements render correctly
- Validate that the appendix folds/unfolds properly
- Test any embedded code highlighting

## Common Patterns

### Figure Integration
```mdx
import figureName from './figure_name.png'

<div className="flex justify-center my-6">
  <figure className="w-full max-w-3xl">
    <Image src={figureName} alt="Clear description" />
    <figcaption className="text-center text-sm text-muted-foreground mt-2">
      <strong>Figure N:</strong> Detailed caption explaining what's shown
    </figcaption>
  </figure>
</div>
```

### Foldable Appendix
```mdx
<details className="not-prose my-4 rounded-lg border border-gray-200 dark:border-gray-700 p-3">
  <summary className="cursor-pointer font-semibold">Appendix: Technical Details</summary>
  <div className="mt-3 text-sm text-gray-600 dark:text-gray-400">
    [Technical content here]
  </div>
</details>
```

## Tips for Effective Reports

1. **Lead with results**: Don't bury the lede - show impact immediately
2. **Use progressive disclosure**: Simple story first, details in appendix
3. **Visualize everything**: A good figure is worth 1000 words
4. **Be quantitative**: Include specific numbers, not just trends
5. **Acknowledge limitations**: This builds trust and suggests future work
6. **Connect to bigger picture**: Why should readers care?
7. **Consult with the human**: Get input on high-level scope, length, and flow of the blog post
8. **Ask when torn between decisions**: Which figures to show, what claims to make, etc. - consult the human when uncertain

## Example Reports
See `reports/emergent-geographic-representations/` for a well-structured research blog post that follows these guidelines.