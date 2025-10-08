import numpy as np
import torch
import yaml
import argparse
from pathlib import Path
import json
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def compute_twonn_dimension(representations, k=20):
    """
    Compute TwoNN intrinsic dimension estimate.
    Based on "Maximum likelihood estimation of intrinsic dimension" (Levina & Bickel, 2004)
    """
    n_samples = representations.shape[0]

    # Find k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(representations)
    distances, indices = nbrs.kneighbors(representations)

    # Remove self (first neighbor)
    distances = distances[:, 1:]

    # Compute ratio of distances to 2nd and 1st nearest neighbors
    r1 = distances[:, 0]
    r2 = distances[:, 1]

    # Avoid division by zero
    valid = r1 > 0
    mu = np.mean(np.log(r2[valid] / r1[valid]))

    # TwoNN dimension estimate
    d_twonn = 1 / mu if mu > 0 else float('inf')

    return d_twonn

def compute_participation_ratio(representations):
    """
    Compute participation ratio (effective dimensionality).
    PR = (sum(lambda_i))^2 / sum(lambda_i^2)
    """
    # Center the data
    centered = representations - representations.mean(axis=0)

    # Compute covariance eigenvalues
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]  # Keep positive eigenvalues

    # Participation ratio
    pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

    return pr

def compute_local_participation_energy(representations, n_neighbors=50):
    """
    Compute local participation energy.
    Measures how distributed representations are in local neighborhoods.
    """
    n_samples, n_dims = representations.shape

    # Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='auto').fit(representations)
    distances, indices = nbrs.kneighbors(representations)

    local_energies = []

    for i in range(n_samples):
        # Get local neighborhood (excluding self)
        local_idx = indices[i, 1:]
        local_points = representations[local_idx]

        # Center around current point
        centered = local_points - representations[i]

        # Compute local covariance
        if len(centered) > 1:
            local_cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(local_cov)
            eigenvalues = eigenvalues[eigenvalues > 0]

            if len(eigenvalues) > 0:
                # Local participation ratio
                local_pr = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
                local_energies.append(local_pr)

    return np.mean(local_energies) if local_energies else 0

def load_representations(exp_dir, layer=5):
    """Load layer representations from experiment directory."""
    # First try the new format
    rep_dir = Path(exp_dir) / "representations" / f"layer_{layer}"
    rep_file = rep_dir / "representations.npy"
    label_file = rep_dir / "labels.npy"

    if rep_file.exists():
        reps = np.load(rep_file)
        labels = np.load(label_file) if label_file.exists() else None
        return reps, labels

    # Fallback to old format
    analysis_dir = Path(exp_dir) / "analysis_higher"
    if analysis_dir.exists():
        rep_files = list(analysis_dir.glob(f"*_l{layer}"))
        if rep_files:
            rep_dir = rep_files[0]
            rep_file = rep_dir / "representations.npy"
            label_file = rep_dir / "labels.npy"
            if rep_file.exists():
                reps = np.load(rep_file)
                labels = np.load(label_file) if label_file.exists() else None
                return reps, labels

    return None, None

def analyze_experiment(exp_name, exp_dir, layer=5, sample_size=5000):
    """Analyze dimensionality for a single experiment."""
    print(f"\nAnalyzing {exp_name}...")

    # Load representations
    reps, labels = load_representations(exp_dir, layer)

    if reps is None:
        print(f"  No layer {layer} representations found")
        return None

    print(f"  Loaded representations: {reps.shape}")

    # Sample if too large
    if reps.shape[0] > sample_size:
        idx = np.random.choice(reps.shape[0], sample_size, replace=False)
        reps = reps[idx]

    # Compute metrics
    results = {
        'experiment': exp_name,
        'n_samples': reps.shape[0],
        'n_dims': reps.shape[1],
        'twonn_dimension': compute_twonn_dimension(reps),
        'participation_ratio': compute_participation_ratio(reps),
        'local_participation_energy': compute_local_participation_energy(reps)
    }

    print(f"  TwoNN dimension: {results['twonn_dimension']:.2f}")
    print(f"  Participation ratio: {results['participation_ratio']:.2f}")
    print(f"  Local participation energy: {results['local_participation_energy']:.2f}")

    return results

def main(config_path, overwrite=False, debug=False):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize output directory
    output_dir = Path(config['output_dir'])
    if output_dir.exists() and not overwrite:
        print(f"Output directory {output_dir} exists. Use --overwrite to continue.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Create subdirectories
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'results').mkdir(exist_ok=True)

    # Analyze experiments
    results = []

    # PT1 experiments (single task, 42 epochs)
    print("\n=== Analyzing PT1 (single-task) experiments ===")
    for i in range(1, 8):
        exp_name = f"pt1-{i}"
        exp_dir = Path(f"data/experiments/{exp_name}")
        if exp_dir.exists():
            result = analyze_experiment(exp_name, exp_dir,
                                       layer=config.get('layer', 5),
                                       sample_size=config.get('sample_size', 5000))
            if result:
                result['group'] = 'pt1'
                result['task_type'] = 'single'
                results.append(result)

    # PT2 experiments (multi-task, 21 epochs)
    print("\n=== Analyzing PT2 (multi-task) experiments ===")
    for i in range(1, 9):
        exp_name = f"pt2-{i}"
        exp_dir = Path(f"data/experiments/{exp_name}")
        if exp_dir.exists():
            result = analyze_experiment(exp_name, exp_dir,
                                       layer=config.get('layer', 5),
                                       sample_size=config.get('sample_size', 5000))
            if result:
                result['group'] = 'pt2'
                result['task_type'] = 'multi'
                results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / 'results' / 'dimensionality_metrics.csv', index=False)

    with open(output_dir / 'results' / 'dimensionality_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Create visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # TwoNN dimension
    ax = axes[0]
    pt1_twonn = [r['twonn_dimension'] for r in results if r['group'] == 'pt1']
    pt2_twonn = [r['twonn_dimension'] for r in results if r['group'] == 'pt2']

    positions = [1, 2]
    bp1 = ax.boxplot([pt1_twonn, pt2_twonn], positions=positions, widths=0.6,
                      patch_artist=True, labels=['PT1 (single)', 'PT2 (multi)'])

    for patch, color in zip(bp1['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)

    ax.set_ylabel('TwoNN Dimension')
    ax.set_title('Intrinsic Dimensionality (TwoNN)')
    ax.grid(True, alpha=0.3)

    # Participation Ratio
    ax = axes[1]
    pt1_pr = [r['participation_ratio'] for r in results if r['group'] == 'pt1']
    pt2_pr = [r['participation_ratio'] for r in results if r['group'] == 'pt2']

    bp2 = ax.boxplot([pt1_pr, pt2_pr], positions=positions, widths=0.6,
                      patch_artist=True, labels=['PT1 (single)', 'PT2 (multi)'])

    for patch, color in zip(bp2['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)

    ax.set_ylabel('Participation Ratio')
    ax.set_title('Effective Dimensionality (PR)')
    ax.grid(True, alpha=0.3)

    # Local Participation Energy
    ax = axes[2]
    pt1_lpe = [r['local_participation_energy'] for r in results if r['group'] == 'pt1']
    pt2_lpe = [r['local_participation_energy'] for r in results if r['group'] == 'pt2']

    bp3 = ax.boxplot([pt1_lpe, pt2_lpe], positions=positions, widths=0.6,
                      patch_artist=True, labels=['PT1 (single)', 'PT2 (multi)'])

    for patch, color in zip(bp3['boxes'], ['lightblue', 'lightcoral']):
        patch.set_facecolor(color)

    ax.set_ylabel('Local Participation Energy')
    ax.set_title('Local Dimensionality (LPE)')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Manifold Dimensionality: Single-task vs Multi-task Models', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'dimensionality_comparison.png', dpi=150, bbox_inches='tight')

    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("\nPT1 (Single-task):")
    print(f"  TwoNN: {np.mean(pt1_twonn):.2f} ± {np.std(pt1_twonn):.2f}")
    print(f"  PR: {np.mean(pt1_pr):.2f} ± {np.std(pt1_pr):.2f}")
    print(f"  LPE: {np.mean(pt1_lpe):.2f} ± {np.std(pt1_lpe):.2f}")

    print("\nPT2 (Multi-task):")
    print(f"  TwoNN: {np.mean(pt2_twonn):.2f} ± {np.std(pt2_twonn):.2f}")
    print(f"  PR: {np.mean(pt2_pr):.2f} ± {np.std(pt2_pr):.2f}")
    print(f"  LPE: {np.mean(pt2_lpe):.2f} ± {np.std(pt2_lpe):.2f}")

    # Statistical test
    from scipy import stats

    print("\n=== Statistical Tests (Mann-Whitney U) ===")
    u_twonn, p_twonn = stats.mannwhitneyu(pt1_twonn, pt2_twonn, alternative='two-sided')
    print(f"TwoNN: U={u_twonn:.2f}, p={p_twonn:.4f}")

    u_pr, p_pr = stats.mannwhitneyu(pt1_pr, pt2_pr, alternative='two-sided')
    print(f"PR: U={u_pr:.2f}, p={p_pr:.4f}")

    u_lpe, p_lpe = stats.mannwhitneyu(pt1_lpe, pt2_lpe, alternative='two-sided')
    print(f"LPE: U={u_lpe:.2f}, p={p_lpe:.4f}")

    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    import pandas as pd  # Import here to avoid issues if not installed

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)