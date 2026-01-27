#!/usr/bin/env python3
"""
Generate CSV of raw evaluation metrics for all models × all tasks.
Includes original (seed0=42) and seed1, seed2, seed3.
"""

import json
from pathlib import Path


EXP_BASE = Path("/data/experiments")
REVISION_BASE = EXP_BASE / "revision/exp1"
OUTPUT_DIR = REVISION_BASE / "tables"

TASKS = ["distance", "trianglearea", "angle", "compass", "inside", "perimeter", "crossing"]

# seed0 = original (seed 42), seed1/2/3 = revision seeds
SEEDS = ["seed0", "seed1", "seed2", "seed3"]


def get_model_dirs():
    """Get all model directories for all seeds (4 seeds × 29 models = 116 models)."""
    models = []

    for seed in SEEDS:
        if seed == "seed0":
            # Original models (seed 42)
            base_dir = EXP_BASE / "pt1"
            if base_dir.exists():
                models.append((seed, "base", base_dir))
            for i in range(1, 8):
                ftwb1_dir = EXP_BASE / f"pt1_ftwb1-{i}"
                if ftwb1_dir.exists():
                    models.append((seed, f"ftwb1-{i}", ftwb1_dir))
            for i in range(1, 22):
                ftwb2_dir = EXP_BASE / f"pt1_ftwb2-{i}"
                if ftwb2_dir.exists():
                    models.append((seed, f"ftwb2-{i}", ftwb2_dir))
        else:
            # Revision seeds (seed1, seed2, seed3)
            base_dir = REVISION_BASE / f"pt1_{seed}"
            if base_dir.exists():
                models.append((seed, "base", base_dir))
            for i in range(1, 8):
                ftwb1_dir = REVISION_BASE / f"pt1_{seed}_ftwb1-{i}"
                if ftwb1_dir.exists():
                    models.append((seed, f"ftwb1-{i}", ftwb1_dir))
            for i in range(1, 22):
                ftwb2_dir = REVISION_BASE / f"pt1_{seed}_ftwb2-{i}"
                if ftwb2_dir.exists():
                    models.append((seed, f"ftwb2-{i}", ftwb2_dir))

    return models


def load_eval_metrics(model_dir, task, atlantis=False):
    """Load evaluation metrics for a specific task (last checkpoint)."""
    task_name = f"atlantis_{task}" if atlantis else task
    eval_path = model_dir / "evals" / task_name / "eval_data" / "evaluation_results.json"

    if not eval_path.exists():
        return None

    with open(eval_path, 'r') as f:
        data = json.load(f)

    # Get the last checkpoint by sorting checkpoint numbers
    checkpoints = list(data.keys())
    # Sort by checkpoint number (extract number from "checkpoint-XXXX")
    checkpoints.sort(key=lambda x: int(x.split('-')[1]))
    last_checkpoint = checkpoints[-1]
    metrics = data[last_checkpoint]

    # Extract mean metric
    metric_key = f"eval_{task}_metric_mean"
    if metric_key in metrics:
        return metrics[metric_key]
    return None


def generate_latex_table(models, output_path):
    """Generate a LaTeX table: 29 models (rows) × 14 task columns (7 ID + 7 OOD)."""

    lines = []
    lines.append("% Raw Evaluation Metrics Table (Original models)")
    lines.append("% Copy everything below into your LaTeX document")
    lines.append("")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Raw evaluation metrics (mean error) for original models. ID = in-distribution, OOD = out-of-distribution (Atlantis).}")
    lines.append("\\label{tab:raw_metrics}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{l" + "rr" * len(TASKS) + "}")
    lines.append("\\toprule")

    # Header row 1: Task names spanning ID/OOD columns
    header1 = "Model"
    for task in TASKS:
        header1 += f" & \\multicolumn{{2}}{{c}}{{{task.capitalize()}}}"
    header1 += " \\\\"
    lines.append(header1)

    # Header row 2: ID/OOD for each task
    header2 = ""
    for _ in TASKS:
        header2 += " & ID & OOD"
    header2 += " \\\\"
    lines.append(header2)
    lines.append("\\midrule")

    # Data rows
    for model_name, model_dir in models:
        row = model_name
        for task in TASKS:
            id_metric = load_eval_metrics(model_dir, task, atlantis=False)
            ood_metric = load_eval_metrics(model_dir, task, atlantis=True)

            id_str = f"{id_metric:.2g}" if id_metric is not None else "-"
            ood_str = f"{ood_metric:.2g}" if ood_metric is not None else "-"
            row += f" & {id_str} & {ood_str}"

        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    return len(models)


def generate_csv(models, output_path):
    """Generate a CSV file with all metrics."""

    lines = []
    # Header with note about seed0
    lines.append("# seed0 corresponds to original seed=42")
    header = "seed,model"
    for task in TASKS:
        header += f",{task}_id,{task}_ood"
    lines.append(header)

    for seed, model_name, model_dir in models:
        row = f"{seed},{model_name}"
        for task in TASKS:
            id_metric = load_eval_metrics(model_dir, task, atlantis=False)
            ood_metric = load_eval_metrics(model_dir, task, atlantis=True)

            id_str = f"{id_metric:.2g}" if id_metric is not None else ""
            ood_str = f"{ood_metric:.2g}" if ood_metric is not None else ""
            row += f",{id_str},{ood_str}"
        lines.append(row)

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    print("=" * 60)
    print("Generating Raw Metrics CSV (all seeds)")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = get_model_dirs()
    print(f"\nFound {len(models)} models (4 seeds × 29 models)")

    # Generate CSV
    csv_path = OUTPUT_DIR / "raw_metrics_all_seeds.csv"
    generate_csv(models, csv_path)
    print(f"\nGenerated CSV: {csv_path}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
