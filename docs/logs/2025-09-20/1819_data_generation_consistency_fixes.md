# Data Generation Consistency Fixes
**Date:** 2025-09-20
**Time:** 18:19

## Summary
Fixed inconsistencies across all data generation tasks to ensure uniform behavior for Atlantis city inclusion and position shuffling.

## Tasks Analyzed
1. **Distance** - Euclidean distance between 2 cities
2. **Compass** - Direction from one city to another (8 directions)
3. **Angle** - Angle at center city formed by 3 cities
4. **Crossing** - Whether two line segments intersect (4 cities)
5. **TriangleArea** - Area of triangle formed by 3 cities
6. **Perimeter** - Perimeter of polygon (2-10 cities)
7. **Inside** - Whether point is inside convex hull (4+ cities)

## Key Issues Identified

### 1. Inconsistent `must_include` Strategy
- **Distance/Compass**: Used complex 10%/90% ratio for Atlantis-Atlantis vs Atlantis-World pairs
- **Angle**: Had 70% center-Atlantis probability with complex 60/30/10 distribution
- **TriangleArea**: Had 60/30/10 ratio for 1/2/3 Atlantis cities
- **Crossing**: Simple implementation - just ensured 1 of 4 was Atlantis
- **Inside**: Forced Atlantis into hull vertices only

### 2. Missing or Inconsistent Shuffling
- **Distance**: Had 50% random swapping
- **Compass**: No shuffling initially
- **Angle**: No shuffling initially
- **Crossing**: No shuffling initially
- **TriangleArea**: No shuffling at all
- **Perimeter**: Only shuffled for must_include strategy
- **Inside**: Only shuffled hull vertices

## Fixes Applied

### Simplified `must_include` Strategy
Modified all tasks to use simple rule: **exactly one city must be from Atlantis group**
- Removed complex ratio calculations
- Random selection of which position gets Atlantis city
- Consistent behavior across all tasks

### Files Modified:
1. `/src/data_processing/data_utils.py`
   - Simplified `generate_must_include_pairs()` function
   - Removed 10/90 inter/cross ratio

2. `/src/data_processing/create_angle_dataset.py`
   - Simplified `generate_must_include_triples()`
   - Added random shuffling of all 3 cities

3. `/src/data_processing/create_trianglearea_dataset.py`
   - Simplified `generate_must_include_triples()` to match angle
   - Added random shuffling of all 3 cities

4. `/src/data_processing/create_crossing_dataset.py`
   - Added complete random shuffling of all 4 cities
   - Added recalculation after shuffling

5. `/src/data_processing/create_compass_dataset.py`
   - Added 50% random swapping to match distance

6. `/src/data_processing/create_inside_dataset.py`
   - Simplified to allow Atlantis anywhere (test point or hull)
   - Complete shuffling before role assignment

## Final Behavior

### All Tasks Now Have:
1. **Simple Atlantis inclusion**: At least one city must be from Atlantis group
2. **Random positioning**: Shuffling ensures no positional bias
3. **Proper calculation order**: All answers calculated after shuffling

### Shuffling Patterns:
- **2-city tasks** (Distance, Compass): 50% random swap
- **3-city tasks** (Angle, TriangleArea): Complete random shuffle
- **4-city tasks** (Crossing): Complete random shuffle with recalculation
- **Variable tasks** (Perimeter, Inside): Shuffle after selection

## Testing Notes
- All calculations properly occur after shuffling
- "No Atlantis" configs work correctly via `within_groups` strategy
- Group filtering happens before selection, ensuring proper exclusion

## Impact
These changes ensure:
1. Consistent Atlantis distribution across all tasks
2. No positional biases that models could exploit
3. Simpler, more maintainable code
4. Fair cross-task comparison for research