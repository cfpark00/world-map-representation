#!/usr/bin/env python3
"""
Quick test of dimensionality hypothesis by extracting representations from a few samples.
"""

import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, Qwen2ForCausalLM
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def get_sample_prompts():
    """Generate a few sample prompts for testing."""
    prompts = []
    # Distance prompts
    for i in range(100):
        prompts.append(f"<bos> d i s t ( c _ {i:04d} , c _ {(i+50)%1000:04d} ) =")

    # Crossing prompts
    for i in range(100):
        prompts.append(f"<bos> c r o s s ( c _ {i:04d} , c _ {(i+50)%1000:04d} , c _ {(i+100)%1000:04d} , c _ {(i+150)%1000:04d} ) =")

    return prompts

def extract_layer_representations(model, tokenizer, prompts, layer=5):
    """Extract representations from a specific layer."""
    model.eval()
    device = next(model.parameters()).device

    representations = []

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            outputs = model(**inputs, output_hidden_states=True)

            # Get hidden states from specified layer
            hidden_states = outputs.hidden_states[layer + 1]  # +1 for embeddings

            # Average over sequence length
            rep = hidden_states.mean(dim=1).squeeze().cpu().numpy()
            representations.append(rep)

    return np.array(representations)

def compute_intrinsic_dim(X, k=10):
    """Estimate intrinsic dimensionality using TwoNN."""
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    distances = distances[:, 1:]  # Remove self

    r1 = distances[:, 0]
    r2 = distances[:, 1]

    valid = r1 > 0
    if valid.sum() > 0:
        mu = np.mean(np.log(r2[valid] / r1[valid]))
        return 1 / mu if mu > 0 else float('inf')
    return float('inf')

def compute_participation_ratio(X):
    """Compute participation ratio."""
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) > 0:
        return (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return 0

def main():
    print("Quick Dimensionality Test for PT1 vs PT2 Models")
    print("=" * 50)

    # Generate test prompts
    prompts = get_sample_prompts()
    print(f"Generated {len(prompts)} test prompts")

    results = []

    # Test PT1 models (single-task)
    print("\n=== Testing PT1 Models (Single-task) ===")
    pt1_dims = []
    pt1_prs = []

    for exp_idx in [1, 2, 3]:  # Test a few pt1 models
        exp_name = f"pt1-{exp_idx}"
        exp_dir = Path(f"data/experiments/{exp_name}")

        if not exp_dir.exists():
            continue

        # Find last checkpoint in checkpoints subdirectory
        checkpoints_dir = exp_dir / "checkpoints"
        if not checkpoints_dir.exists():
            print(f"  No checkpoints directory found")
            continue

        checkpoints = sorted(checkpoints_dir.glob("checkpoint-*"))
        if not checkpoints:
            print(f"  No checkpoints found")
            continue

        checkpoint = checkpoints[-1]
        print(f"\n{exp_name}: Loading {checkpoint.name}")

        try:
            # Load model and tokenizer
            model = Qwen2ForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16)
            # Try to find tokenizer in the checkpoint or checkpoints dir
            if (checkpoint / "tokenizer_config.json").exists():
                tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            elif (checkpoints_dir / "tokenizer").exists():
                tokenizer = AutoTokenizer.from_pretrained(checkpoints_dir / "tokenizer")
            else:
                # Use default tokenizer
                tokenizer = AutoTokenizer.from_pretrained("data/tokenizers/default_tokenizer")

            if torch.cuda.is_available():
                model = model.cuda()

            # Extract representations
            reps = extract_layer_representations(model, tokenizer, prompts, layer=5)

            # Compute metrics
            dim = compute_intrinsic_dim(reps)
            pr = compute_participation_ratio(reps)

            print(f"  Intrinsic dim: {dim:.2f}")
            print(f"  Participation ratio: {pr:.2f}")

            pt1_dims.append(dim)
            pt1_prs.append(pr)

            results.append({
                'model': exp_name,
                'group': 'pt1',
                'intrinsic_dim': dim,
                'participation_ratio': pr
            })

            # Clean up memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Test PT2 models (multi-task)
    print("\n=== Testing PT2 Models (Multi-task) ===")
    pt2_dims = []
    pt2_prs = []

    for exp_idx in [1, 2, 3]:  # Test a few pt2 models
        exp_name = f"pt2-{exp_idx}"
        exp_dir = Path(f"data/experiments/{exp_name}")

        if not exp_dir.exists():
            continue

        # Find last checkpoint in checkpoints subdirectory
        checkpoints_dir = exp_dir / "checkpoints"
        if not checkpoints_dir.exists():
            print(f"  No checkpoints directory found")
            continue

        checkpoints = sorted(checkpoints_dir.glob("checkpoint-*"))
        if not checkpoints:
            print(f"  No checkpoints found")
            continue

        checkpoint = checkpoints[-1]
        print(f"\n{exp_name}: Loading {checkpoint.name}")

        try:
            # Load model and tokenizer
            model = Qwen2ForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16)
            # Try to find tokenizer in the checkpoint or checkpoints dir
            if (checkpoint / "tokenizer_config.json").exists():
                tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            elif (checkpoints_dir / "tokenizer").exists():
                tokenizer = AutoTokenizer.from_pretrained(checkpoints_dir / "tokenizer")
            else:
                # Use default tokenizer
                tokenizer = AutoTokenizer.from_pretrained("data/tokenizers/default_tokenizer")

            if torch.cuda.is_available():
                model = model.cuda()

            # Extract representations
            reps = extract_layer_representations(model, tokenizer, prompts, layer=5)

            # Compute metrics
            dim = compute_intrinsic_dim(reps)
            pr = compute_participation_ratio(reps)

            print(f"  Intrinsic dim: {dim:.2f}")
            print(f"  Participation ratio: {pr:.2f}")

            pt2_dims.append(dim)
            pt2_prs.append(pr)

            results.append({
                'model': exp_name,
                'group': 'pt2',
                'intrinsic_dim': dim,
                'participation_ratio': pr
            })

            # Clean up memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if pt1_dims:
        print(f"\nPT1 (Single-task):")
        print(f"  Avg intrinsic dim: {np.mean(pt1_dims):.2f} ± {np.std(pt1_dims):.2f}")
        print(f"  Avg participation ratio: {np.mean(pt1_prs):.2f} ± {np.std(pt1_prs):.2f}")

    if pt2_dims:
        print(f"\nPT2 (Multi-task):")
        print(f"  Avg intrinsic dim: {np.mean(pt2_dims):.2f} ± {np.std(pt2_dims):.2f}")
        print(f"  Avg participation ratio: {np.mean(pt2_prs):.2f} ± {np.std(pt2_prs):.2f}")

    if pt1_dims and pt2_dims:
        print(f"\nDifference:")
        print(f"  Intrinsic dim reduction: {np.mean(pt1_dims) - np.mean(pt2_dims):.2f}")
        print(f"  PR reduction: {np.mean(pt1_prs) - np.mean(pt2_prs):.2f}")

    # Simple visualization
    if results:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # Intrinsic dimension
        pt1_vals = [r['intrinsic_dim'] for r in results if r['group'] == 'pt1']
        pt2_vals = [r['intrinsic_dim'] for r in results if r['group'] == 'pt2']

        if pt1_vals and pt2_vals:
            ax1.bar(['PT1\n(single)', 'PT2\n(multi)'],
                   [np.mean(pt1_vals), np.mean(pt2_vals)],
                   yerr=[np.std(pt1_vals), np.std(pt2_vals)],
                   capsize=5, color=['lightblue', 'lightcoral'])
            ax1.set_ylabel('Intrinsic Dimension')
            ax1.set_title('TwoNN Intrinsic Dimensionality')

            # Participation ratio
            pt1_vals = [r['participation_ratio'] for r in results if r['group'] == 'pt1']
            pt2_vals = [r['participation_ratio'] for r in results if r['group'] == 'pt2']

            ax2.bar(['PT1\n(single)', 'PT2\n(multi)'],
                   [np.mean(pt1_vals), np.mean(pt2_vals)],
                   yerr=[np.std(pt1_vals), np.std(pt2_vals)],
                   capsize=5, color=['lightblue', 'lightcoral'])
            ax2.set_ylabel('Participation Ratio')
            ax2.set_title('Effective Dimensionality')

            plt.suptitle('Manifold Dimensionality: Single vs Multi-task')
            plt.tight_layout()
            plt.savefig('quick_dimensionality_test.png', dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to quick_dimensionality_test.png")

if __name__ == "__main__":
    main()