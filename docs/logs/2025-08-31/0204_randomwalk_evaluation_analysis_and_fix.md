# Random Walk Evaluation Analysis and Critical Bug Fix

**Date**: 2025-08-31
**Time**: 02:04
**Focus**: Analyzing random walk checkpoint performance and fixing evaluation bug

## Summary
Analyzed the random walk model checkpoint to evaluate proper transition ratios and discovered/fixed a critical bug in the evaluation logic that was making validation metrics meaningless.

## Tasks Completed

### 1. Random Walk Checkpoint Analysis
Created comprehensive analysis script in `analysis/randomwalk_checkpoint/analyze_transitions.py` to evaluate the quality of generated random walks.

**Key findings:**
- **Sampling approach** (temp=1.0, top_p=0.95): 4.87% proper transitions
- **Greedy decoding with random starts**: 23.39% proper transitions (~5x better!)
- Greedy decoding produces more consistent walk lengths and better validity

**Implementation details:**
- Initially implemented one-by-one generation (very slow)
- Optimized to use batched generation (64 samples at once)
- Added support for both sampling and greedy decoding modes
- Used left padding for varying-length prompts in greedy mode

### 2. Critical Evaluation Bug Discovery and Fix

**The Bug:**
The random walk evaluation in `src/utils.py` was using greedy decoding with identical prompts (`<bos>walk_200=`) for all validation samples. This meant:
- All 128 validation samples generated the **exact same walk**
- Validation metrics were meaningless (same walk evaluated 128 times)
- No diversity in evaluation, making it impossible to assess model generalization

**The Fix:**
Modified `evaluate_with_generation()` in `src/utils.py` to:
1. Detect when evaluating random walk tasks
2. Sample a random starting city for each validation item
3. Append the city to the prompt (e.g., `<bos>walk_200=c_1823,`)
4. Use left padding for the varying-length prompts

**Verification:**
- Created test script to verify the fix works
- Confirmed different walks are now generated (std > 0)
- Validation now properly tests model's ability to generate walks from diverse starting points

## Key Code Changes

### Files Created:
- `analysis/randomwalk_checkpoint/analyze_transitions.py` - Main analysis script
- `analysis/randomwalk_checkpoint/test_eval_fix.py` - Test script for evaluation fix
- `analysis/randomwalk_checkpoint/generated_walks_sampling.txt` - Sample outputs
- `analysis/randomwalk_checkpoint/generated_walks_greedy.txt` - Greedy outputs
- `analysis/randomwalk_checkpoint/analysis_results_*.json` - Detailed metrics

### Files Modified:
- `src/utils.py` - Fixed random walk evaluation logic (lines 425-436)

## Technical Insights

### Batch Generation with Diverse Prompts
The key insight was maintaining batch efficiency while ensuring diversity:
- Each item in batch gets a different starting city
- Use left padding to handle varying prompt lengths
- Model generates different walks in parallel
- Greedy decoding becomes useful when prompts vary

### Evaluation Design Lesson
For tasks where the prompt is identical across validation set:
- Greedy decoding will produce identical outputs
- Must introduce variation in prompts for meaningful evaluation
- Random starting points are essential for path/sequence generation tasks

## Metrics Summary
- Random walk model achieves ~23% valid transitions with greedy decoding
- Distance threshold: 200km between consecutive cities
- Model has learned some geographic constraints but room for improvement

## Next Steps/Recommendations
1. Consider training with more diverse starting cities
2. Maybe implement curriculum learning (start with easier constraints)
3. Add explicit position encodings for geographic tasks
4. Consider augmenting training data with more random walks

## Commands Reference
```bash
# Run analysis with sampling
python analysis/randomwalk_checkpoint/analyze_transitions.py

# Run analysis with greedy decoding
python analysis/randomwalk_checkpoint/analyze_transitions.py --greedy --num-samples 1000

# Test evaluation fix
python analysis/randomwalk_checkpoint/test_eval_fix.py
```