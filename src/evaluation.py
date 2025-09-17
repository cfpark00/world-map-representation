#!/usr/bin/env python3
"""
Unified evaluation module for WM_1 project.
Single source of truth for all evaluation logic - used by both training and post-training evaluation.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm


def evaluate_with_generation(
    model,
    eval_dataset,
    tokenizer,
    device,
    num_samples=128,
    batch_size=16,
    config=None,
    return_details=False
):
    """
    Evaluate model with generation and task-specific metric calculation.

    Args:
        model: The model to evaluate
        eval_dataset: Dataset to evaluate on
        tokenizer: Tokenizer for encoding/decoding
        device: Device to run on
        num_samples: Number of samples to evaluate
        batch_size: Batch size for generation
        config: Optional config dict with task-specific settings
        return_details: If True, return detailed results for each example (for post-training analysis)

    Returns:
        If return_details=False: Dict of aggregated metrics
        If return_details=True: Tuple of (aggregated_metrics, detailed_results)
    """
    # Import metrics module
    from src.metrics import calculate_metric, get_failure_value, format_metric_for_display

    model.eval()

    # Sample a subset for evaluation
    eval_indices = np.random.choice(len(eval_dataset), min(num_samples, len(eval_dataset)), replace=False)

    # Group samples by task type
    task_samples = {}
    for idx in eval_indices:
        idx = int(idx)
        # Handle both BaseDataset (has .dataset) and raw dataset
        if hasattr(eval_dataset, 'dataset'):
            raw_item = eval_dataset.dataset[idx]
        else:
            raw_item = eval_dataset[idx]

        # FAIL FAST: Require task_type field
        if 'task_type' not in raw_item:
            raise ValueError(f"FATAL: Dataset item at index {idx} missing required 'task_type' field: {raw_item}")

        task_type = raw_item['task_type']
        if task_type not in task_samples:
            task_samples[task_type] = []
        task_samples[task_type].append((idx, raw_item))

    print(f"Task distribution in eval batch: {[(t, len(samples)) for t, samples in task_samples.items()]}")

    # Load cities data for tasks that need it
    cities_df = None
    tasks_needing_cities = {'randomwalk', 'randring', 'center'}
    tasks_that_need_cities = tasks_needing_cities.intersection(task_samples.keys())

    if tasks_that_need_cities:
        # Check for cities_csv in eval.<task>.cities_csv structure
        cities_csv = None

        # Try to find cities_csv in any of the tasks that need it
        for task in tasks_that_need_cities:
            if config and 'eval' in config and task in config['eval'] and 'cities_csv' in config['eval'][task]:
                cities_csv = config['eval'][task]['cities_csv']
                break

        # Also check for direct cities_csv in config (for post-training eval)
        if not cities_csv and config and 'cities_csv' in config:
            cities_csv = config['cities_csv']

        if not cities_csv:
            # Require explicit config - no hardcoded default
            raise ValueError(
                f"FATAL: Tasks {tasks_that_need_cities} require cities CSV path. Add to config:\n"
                "eval:\n"
                f"  {list(tasks_that_need_cities)[0]}:\n"
                "    cities_csv: 'data/datasets/cities/cities.csv'"
            )

        print(f"Loading cities data for {tasks_that_need_cities} evaluation from: {cities_csv}")

        # Check if file exists
        import os
        if not os.path.exists(cities_csv):
            raise ValueError(f"FATAL: Cities CSV not found at {cities_csv}. Tasks {tasks_that_need_cities} need city coordinates for validation.")

        cities_df = pd.read_csv(cities_csv)

    # Results storage
    all_task_metrics = {}
    detailed_results = [] if return_details else None

    # Process each task type separately
    for task_type, samples in task_samples.items():
        if not samples:
            continue

        print(f"\nEvaluating {len(samples)} {task_type} samples...")

        # Extract data for this task
        task_indices = [idx for idx, _ in samples]
        task_items = [item for _, item in samples]

        task_metrics = []

        # Process in batches
        num_batches = (len(samples) + batch_size - 1) // batch_size

        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(samples))
                batch_items = task_items[batch_start:batch_end]

                # Get prompts and true completions using task-specific splitting
                batch_prompts = []
                batch_true_completions = []
                batch_texts = []

                for raw_item in batch_items:
                    # Handle different dataset formats
                    if 'prompt' in raw_item and 'completion' in raw_item:
                        # Old format with separate prompt/completion
                        prompt = raw_item['prompt']
                        completion = raw_item['completion']
                        text = prompt + completion  # Reconstruct for detailed results
                    elif 'text' in raw_item:
                        # New format with combined text - split into prompt/completion
                        text = raw_item['text']

                        # CRITICAL: Use task-specific splitting logic
                        if task_type in ['center', 'circlecount', 'randring']:
                            # These have multiple '=' signs, split at the LAST one
                            eq_pos = text.rfind('=')  # Find LAST '='
                            if eq_pos == -1:
                                raise ValueError(f"FATAL: Cannot find '=' in {task_type} task text: {text[:100]}...")
                            prompt = text[:eq_pos+1]  # Include '='
                            completion = text[eq_pos+1:]  # Everything after last '='

                        elif task_type == 'randomwalk':
                            # Split at '=' for randomwalk tasks
                            eq_pos = text.find('=')
                            if eq_pos == -1:
                                raise ValueError(f"FATAL: Cannot find '=' in randomwalk task text: {text[:100]}...")

                            # For evaluation: include first city in prompt to make task harder
                            # This prevents model from choosing an easy starting city
                            full_completion = text[eq_pos+1:]  # Everything after '='

                            # Find the first comma (end of first city)
                            first_comma = full_completion.find(',')
                            if first_comma != -1:
                                # Include first city and comma in prompt
                                prompt = text[:eq_pos+1] + full_completion[:first_comma+1]
                                completion = full_completion[first_comma+1:]
                            else:
                                # Only one city or malformed - fall back to original split
                                prompt = text[:eq_pos+1]
                                completion = full_completion

                        else:
                            # All other tasks: split at first '='
                            eq_pos = text.find('=')
                            if eq_pos == -1:
                                raise ValueError(f"FATAL: Cannot find '=' in {task_type} task text: {text[:100]}...")
                            prompt = text[:eq_pos+1]  # Include '='
                            completion = text[eq_pos+1:]  # Everything after '='
                    else:
                        raise ValueError(f"Dataset item missing both 'prompt'/'completion' and 'text' fields: {raw_item}")

                    batch_prompts.append(prompt)
                    batch_true_completions.append(completion)
                    batch_texts.append(text)

                # Tokenize with LEFT padding for generation
                tokenizer.padding_side = 'left'
                inputs = tokenizer(
                    batch_prompts,
                    return_tensors='pt',
                    add_special_tokens=False,
                    padding=True,
                    truncation=False
                )
                tokenizer.padding_side = 'right'  # Reset to right for training

                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                # Generate with task-appropriate max tokens
                if task_type == 'randomwalk':
                    max_new_tokens = 224  # Allow for long walks with up to 20 cities with padding
                elif task_type in ['trianglearea', 'angle']:
                    max_new_tokens = 30   # Area/angle can be large numbers
                elif task_type in ['nearest_neighbor', 'randring']:
                    max_new_tokens = 100  # Multiple city IDs
                elif task_type in ['perimeter']:
                    max_new_tokens = 30   # Can be large numbers
                else:
                    max_new_tokens = 20   # Default for other tasks

                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                # Decode FULL outputs (including prompt) - this is what metrics expect
                generated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Print first few examples from first batch for each task
                if batch_idx == 0 and not return_details:  # Only print during training
                    num_to_show = min(2, len(generated_batch))
                    print(f"[{task_type.upper()} examples]")
                    for ex_idx in range(num_to_show):
                        print(f"  Example {ex_idx + 1}:")
                        print(f"    Prompt: {batch_prompts[ex_idx]}")
                        print(f"    Expected: {batch_true_completions[ex_idx]}")
                        print(f"    Generated: {generated_batch[ex_idx]}")

                        # Calculate metric using centralized system
                        metric_kwargs = {}
                        if task_type in ['randomwalk', 'randring', 'center'] and cities_df is not None:
                            metric_kwargs['cities_df'] = cities_df

                        metric_value = calculate_metric(
                            task_type,
                            batch_prompts[ex_idx],
                            batch_true_completions[ex_idx],
                            generated_batch[ex_idx],
                            **metric_kwargs
                        )

                        # Format based on task type
                        formatted_value = format_metric_for_display(task_type, metric_value)
                        print(f"    Metric: {formatted_value}")

                # Calculate task-specific metrics using centralized system
                for idx, (prompt, true_completion, generated, text) in enumerate(
                    zip(batch_prompts, batch_true_completions, generated_batch, batch_texts)
                ):
                    # Prepare kwargs based on task requirements
                    metric_kwargs = {}
                    if task_type in ['randomwalk', 'randring', 'center'] and cities_df is not None:
                        metric_kwargs['cities_df'] = cities_df

                    # Calculate metric using centralized system
                    try:
                        metric_value = calculate_metric(
                            task_type,
                            prompt,
                            true_completion,
                            generated,
                            **metric_kwargs
                        )
                    except Exception as e:
                        print(f"Warning: Metric calculation failed for {task_type}: {e}")
                        metric_value = get_failure_value(task_type)

                    task_metrics.append(metric_value)

                    # Store detailed results if requested
                    if return_details:
                        detailed_results.append({
                            'text': text,
                            'prompt': prompt,
                            'expected': true_completion,
                            'generation': generated,
                            'metric': metric_value,
                            'task_type': task_type
                        })

        # Calculate aggregated metrics for this task
        if task_metrics:
            all_task_metrics[f'eval_{task_type}_metric_mean'] = np.mean(task_metrics)
            all_task_metrics[f'eval_{task_type}_metric_median'] = np.median(task_metrics)
            all_task_metrics[f'eval_{task_type}_metric_std'] = np.std(task_metrics)
            all_task_metrics[f'eval_{task_type}_metric_min'] = np.min(task_metrics)
            all_task_metrics[f'eval_{task_type}_metric_max'] = np.max(task_metrics)
            all_task_metrics[f'eval_{task_type}_valid_count'] = len(task_metrics)
            all_task_metrics[f'eval_{task_type}_valid_ratio'] = float(len(task_metrics)) / len(samples)

    # Return based on whether details were requested
    if return_details:
        return all_task_metrics, detailed_results
    else:
        return all_task_metrics