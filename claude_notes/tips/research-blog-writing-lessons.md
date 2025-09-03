# Lessons Learned: Writing Research Reports/Blog Posts

*Based on creating the WM_1 city coordinates introduction post*

## What the Guide Didn't Cover (But We Learned)

### 1. Framing Within Broader Research Programs

**The Challenge**: The blog guide assumed standalone posts, but research work often needs to be positioned within larger programs.

**What We Learned**:
- Start with the **big research question** first, then zoom into your specific contribution
- Use phrases like "As a testbed for this broader research program..." to position your work
- Explain why your specific approach advances the larger goals
- Connect your setup/debugging to future research potential

**Example Structure**:
```markdown
## The Broader Vision: [Big Research Question]
[Explain the field-wide challenge]

## Why [Your Approach] Could Be Perfect
[Connect your specific work to the broader goals]

## What We Built: [Your Specific Contribution]
[Detail your approach/system]
```

### 2. Turning Debugging Stories Into Research Insights

**The Challenge**: Debugging feels like "just fixing bugs" but can reveal important insights for the field.

**What We Learned**:
- Frame debugging discoveries as **methodological contributions**
- Connect each "failure" to a broader principle
- Use debugging to show your systematic approach
- Turn embarrassing mistakes into transferable lessons

**Example Framing**:
- ❌ "We had a bug in our batch size"
- ✅ "This revealed a critical insight about gradient update frequency in small-scale representation research"

### 3. Creating Figures That Advance the Narrative

**The Guide Covered**: How to format and caption figures
**What It Missed**: Which figures actually serve the story

**What We Learned**:
- **Timeline figures** work great for debugging narratives
- **Comparison figures** (before/after) make problems concrete
- **Density/distribution plots** help readers understand your data choice
- **Task example figures** clarify complex setups quickly

**Figure Selection Strategy**:
1. What's the hardest thing for readers to visualize? (Make a figure)
2. What's your key insight? (Show the before/after)
3. What makes your approach special? (Visualize the unique aspects)

### 4. Balancing Technical Detail with Accessibility

**The Challenge**: Research reports need enough detail for reproducibility but still need to engage readers.

**What We Learned**:
- Put **configuration details in dedicated sections** (not scattered throughout)
- Use **concrete examples** instead of abstract descriptions
- **Code snippets** should illustrate concepts, not show implementation
- Save detailed methodology for dedicated sections

**Good Balance Example**:
```markdown
## The Setup: Three Simple Tasks
[Visual examples of input/output]

## Technical Configuration
[Detailed parameters for reproduction]
```

### 5. Connecting Setup Posts to Future Work

**The Guide Covered**: How to end posts
**What It Missed**: How to tease future work without spoiling it

**What We Learned**:
- **List specific research questions** you're now equipped to answer
- **Categorize future experiments** (e.g., "For Representation Studies:", "For Controlled Experiments:")
- **Hint at surprising results** without revealing them
- Use phrases like "The answers are already looking fascinating—but that's a story for the next posts"

### 6. Writing for Multiple Audiences Simultaneously

**The Challenge**: Research reports need to serve both specialists and curious generalists.

**What We Learned**:
- **TL;DR should serve specialists** (they want the bottom line quickly)
- **Introduction should hook generalists** (start with the interesting question)
- **Technical sections should enable reproduction** (specialists need this)
- **Conclusions should inspire future work** (both audiences want this)

### 7. When to Create Custom Figures vs Use Existing Ones

**What We Discovered**:
- **Timeline figures** almost always need to be custom (your specific story)
- **Comparison figures** are worth creating even if data is simple
- **Task examples** benefit from clean, custom illustrations
- **Data distributions** can often reuse existing analysis plots

**Time Investment Guide**:
- Spend time on figures that **uniquely support your narrative**
- Reuse/adapt figures that show **standard analyses**
- Create custom figures for **your specific insights/discoveries**

## Practical Workflow That Worked

### 1. Content Creation Order
1. **Write the narrative first** (without worrying about figures)
2. **Identify what's hard to visualize** while writing
3. **Create figures to support specific points** in the narrative
4. **Integrate figures into the flow** (not as afterthoughts)

### 2. Figure Creation Strategy
```python
# Create a single script that generates all figures
# This makes it easy to:
# - Maintain consistent styling
# - Regenerate if data changes  
# - Ensure all figures work together
```

### 3. Iterative Refinement
- Write narrative → identify confusing parts → create supporting figures
- Test the "header-only" read (do headers tell the story?)
- Check: Does each figure advance understanding?

## Red Flags We Learned to Avoid

### Content Red Flags
- **Generic introductions** ("In this post we will explore...")
- **Burying the lede** (taking too long to get to the interesting part)
- **Too much technical detail upfront** (save configuration for later)
- **Figures that repeat text** (figures should add new information)

### Structural Red Flags  
- **Headers that don't flow** (read only headers - do they tell a story?)
- **Figures without clear purpose** (each should advance the narrative)
- **Missing connections to broader significance** (why should anyone care?)

## What Made This Post Work

1. **Clear research positioning**: Readers understand how this fits into bigger questions
2. **Debugging as systematic methodology**: Failures become transferable insights  
3. **Visual support for key concepts**: Complex ideas get concrete illustrations
4. **Future research potential**: Clear path from current work to exciting questions
5. **Honest about challenges**: Acknowledging difficulties builds credibility

## Templates We'd Use Again

### For Setup/Introduction Posts:
```markdown
## The Broader Vision: [Research Program Goal]
## Why [Approach] Could Be Perfect: [Your Specific Contribution]  
## The [Challenge/Debugging] Journey: [Systematic Problem-Solving]
## What We Built: [Technical Setup]
## What's Next: The Perfect Testbed for [Research Questions]
```

### For Figure Integration:
```markdown
[Context paragraph]

<div className="flex justify-center my-6">
  <figure className="w-full max-w-3xl">
    <Image src={figure} alt="descriptive alt text" />
    <figcaption>
      <strong>Figure N:</strong> What this shows and why it matters
    </figcaption>
  </figure>
</div>

[How this connects to the next point]
```

### For Technical Dropdowns:
```markdown
<details className="not-prose my-4 rounded-lg border border-gray-200 dark:border-gray-700 p-3">
  <summary className="cursor-pointer font-semibold text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100">Dropdown Title</summary>
  <div className="mt-3 text-sm text-gray-600 dark:text-gray-400">
    [Content here]
  </div>
</details>
```

**Use this exact styling for all dropdowns** - it provides proper visual hierarchy, dark mode support, and consistent hover states. Don't use simplified versions like `className="cursor-pointer font-semibold"` as they lack the visual polish.

## Bottom Line

Research blog posts are different from general blog posts because they need to:
1. **Position work within research programs** (not just standalone insights)
2. **Turn methodology into transferable lessons** (debugging becomes systematic knowledge)
3. **Set up future work** (this post enables future discoveries)
4. **Serve multiple audiences** (specialists and curious generalists)

The blog writing guide was excellent for the basics, but research posts need this additional layer of strategic positioning and future-oriented thinking.