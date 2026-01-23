# World Map Without Atlantis Figure

## Summary
Created a version of the colorful world map figure (`app_world.png`) that excludes Atlantis cities.

## Task
User requested a version of the revision world map figure without Atlantis, matching the style of `cities_basic.png` (no legend, no X/Y axis labels).

## Files Created
- `src/scripts/plot_cities_world_no_atlantis.py` - New script based on `plot_cities_basic.py` but filters out Atlantis
- `configs/revision/additional_plots/world_no_atlantis.yaml` - Config file
- `scripts/revision/additional_plots/plot_world_no_atlantis.sh` - Runner script

## Output
- `data/experiments/revision/additional_plots/world_no_atlantis/figures/world_no_atlantis.png`

## Key Difference from Original
The script filters out Atlantis cities with:
```python
df = df[df['region'] != 'Atlantis']
```

This removes the 100 Atlantis cities from the 5,175 total, leaving 5,075 world cities.

## Run Command
```bash
bash scripts/revision/additional_plots/plot_world_no_atlantis.sh
```
