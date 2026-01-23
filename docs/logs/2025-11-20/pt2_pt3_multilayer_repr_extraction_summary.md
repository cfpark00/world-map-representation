# PT2/PT3 Multi-Layer Representation Extraction Infrastructure

**Date**: 2025-11-20
**Purpose**: Extract representations for layers 3,4,5,6 for all 63 PT experiments (PT1-X + PT2 + PT3 across 3 seeds)

## Current Status

### Original Experiments (Seed 42) - ✓ COMPLETE
- **PT2 (8 variants)**: All have 4 layers (3,4,5,6) extracted
- **PT3 (8 variants)**: All have 4 layers (3,4,5,6) extracted
- Total: 16 experiments × 4 layers = 64 representations ✓

### Seed Experiments - ⚠ INCOMPLETE
**PT2 Seed Experiments (14 total):**
- PT2-1 to PT2-7, seed1: ⚠ Only layer 5 (needs 3,4,6)
- PT2-1 to PT2-7, seed2: ✗ No representations (needs 3,4,5,6)

**PT3 Seed Experiments (14 total):**
- PT3-1 to PT3-7, seed1: ✗ No representations (needs 3,4,5,6)
- PT3-1 to PT3-7, seed2: ✗ No representations (needs 3,4,5,6)

## Total Representations Needed

**For complete 63-model, 4-layer coverage:**
- Original PT2/PT3 (seed 42): ✓ 16 × 4 = 64 (complete)
- PT2 seed1: 7 × 3 layers (3,4,6) = 21 needed
- PT2 seed2: 7 × 4 layers (3,4,5,6) = 28 needed
- PT3 seed1: 7 × 4 layers (3,4,5,6) = 28 needed
- PT3 seed2: 7 × 4 layers (3,4,5,6) = 28 needed

**Total new representations to extract: 105**

Note: PT1-X (7 tasks × 3 seeds = 21 experiments) already have 4-layer representations from Exp4.

## Created Infrastructure

### Configs (105 total)
**PT2**: 49 configs
- Location: `configs/revision/exp2/pt2_seed/extract_representations_multilayer/`
- Coverage:
  - Seed1: 21 configs (7 variants × 3 layers: 3,4,6)
  - Seed2: 28 configs (7 variants × 4 layers: 3,4,5,6)

**PT3**: 56 configs
- Location: `configs/revision/exp2/pt3_seed/extract_representations_multilayer/`
- Coverage:
  - Seed1: 28 configs (7 variants × 4 layers: 3,4,5,6)
  - Seed2: 28 configs (7 variants × 4 layers: 3,4,5,6)

### Scripts (11 total)
**PT2 Scripts (5):**
- `extract_pt2_layer3.sh` - Extract layer 3 for all PT2 seeds
- `extract_pt2_layer4.sh` - Extract layer 4 for all PT2 seeds
- `extract_pt2_layer5.sh` - Extract layer 5 for PT2 seed2 only
- `extract_pt2_layer6.sh` - Extract layer 6 for all PT2 seeds
- `extract_pt2_all_layers.sh` - Master script

**PT3 Scripts (5):**
- `extract_pt3_layer3.sh` - Extract layer 3 for all PT3 seeds
- `extract_pt3_layer4.sh` - Extract layer 4 for all PT3 seeds
- `extract_pt3_layer5.sh` - Extract layer 5 for all PT3 seeds
- `extract_pt3_layer6.sh` - Extract layer 6 for all PT3 seeds
- `extract_pt3_all_layers.sh` - Master script

**Combined Master (1):**
- `extract_pt2_pt3_all_multilayer.sh` - Extract all PT2+PT3 layers

## Usage

### Extract Everything
```bash
bash scripts/revision/exp2/extract_pt2_pt3_all_multilayer.sh
```

### Extract by Component
```bash
# PT2 only (all layers)
bash scripts/revision/exp2/pt2_seed/extract_representations_multilayer/extract_pt2_all_layers.sh

# PT3 only (all layers)
bash scripts/revision/exp2/pt3_seed/extract_representations_multilayer/extract_pt3_all_layers.sh
```

### Extract by Layer
```bash
# PT2 layer 3 only
bash scripts/revision/exp2/pt2_seed/extract_representations_multilayer/extract_pt2_layer3.sh

# PT3 layer 5 only
bash scripts/revision/exp2/pt3_seed/extract_representations_multilayer/extract_pt3_layer5.sh
```

## Output Locations

All representations will be saved to:
```
data/experiments/revision/exp2/
├── pt2-{1-7}_seed{1,2}/
│   └── analysis_higher/
│       └── {task}_firstcity_last_and_trans_l{3,4,5,6}/
│           └── representations/
└── pt3-{1-7}_seed{1,2}/
    └── analysis_higher/
        └── {task}_firstcity_last_and_trans_l{3,4,5,6}/
            └── representations/
```

## Task Mappings

**PT2 (first task from each pair):**
- pt2-1: distance, pt2-2: angle, pt2-3: inside, pt2-4: crossing
- pt2-5: trianglearea, pt2-6: compass, pt2-7: perimeter

**PT3 (first task from each triple):**
- pt3-1: distance, pt3-2: compass, pt3-3: crossing, pt3-4: angle
- pt3-5: perimeter, pt3-6: trianglearea, pt3-7: inside

## Complete 63-Model Coverage

After running these extractions, you will have:

**PT1-X (21 models, already complete):**
- 7 tasks × 3 seeds (orig, seed1, seed2) × 4 layers = 84 representations ✓

**PT2 (22 trained models):**
- 8 variants × seed42 × 4 layers = 32 representations ✓ (existing)
- 7 variants × seed1 × 4 layers = 28 representations (21 existing + 7 missing layer 3,4,6 → all will be complete)
- 7 variants × seed2 × 4 layers = 28 representations (all new)
- Total: 88 representations

**PT3 (22 trained models):**
- 8 variants × seed42 × 4 layers = 32 representations ✓ (existing)
- 7 variants × seed1 × 4 layers = 28 representations (all new)
- 7 variants × seed2 × 4 layers = 28 representations (all new)
- Total: 88 representations

**Grand Total: 260 representations (84 PT1-X + 88 PT2 + 88 PT3)**

This covers all 65 trained experiments (21 PT1-X + 22 PT2 + 22 PT3) across 4 layers each.

## Next Steps

1. **Run extraction**: Execute `extract_pt2_pt3_all_multilayer.sh`
2. **Verify completion**: Check that all 105 new representations exist
3. **Generate CKA configs**: Create CKA comparison configs for non-overlapping pairs
4. **Run CKA analysis**: Compute CKA for all relevant pairs
5. **Create CKA trends plot**: Visualize PT1 vs PT2 vs PT3 similarity trends

## Generation Scripts

- `src/scripts/generate_pt2_pt3_multilayer_repr_configs.py` - Config generator
- `src/scripts/generate_pt2_pt3_multilayer_run_scripts.py` - Script generator

Both are rerunnable and idempotent.
