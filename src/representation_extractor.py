#!/usr/bin/env python3
"""
RepresentationExtractor: A utility class for extracting internal representations 
from transformer models at specified layers using PyTorch hooks.
"""

import torch
from typing import List, Dict, Union, Optional


class RepresentationExtractor:
    """Extract representations from specific transformer layers using hooks."""
    
    def __init__(self, model, layer_indices: Optional[Union[int, List[int]]] = None):
        """
        Initialize the extractor.
        
        Args:
            model: The transformer model (e.g., Qwen2ForCausalLM)
            layer_indices: Either a single int or a list of ints specifying which layers to extract.
                          If None, defaults to layer 4 (index 3).
        """
        self.model = model
        
        # Handle both single index and list of indices
        if layer_indices is None:
            self.layer_indices = [3]  # Default to layer 4 (0-indexed)
        elif isinstance(layer_indices, int):
            self.layer_indices = [layer_indices]
        else:
            self.layer_indices = list(layer_indices)
        
        # Validate layer indices
        n_layers = len(self.model.model.layers)
        for idx in self.layer_indices:
            if idx < 0 or idx >= n_layers:
                raise ValueError(f"Layer index {idx} out of bounds. Model has {n_layers} layers (0-{n_layers-1})")
        
        # Sort indices to ensure consistent ordering
        self.layer_indices = sorted(self.layer_indices)
        
        # Storage for representations from each layer
        self.representations = {}
        self.hook_handles = []
        
    def create_hook_fn(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # output is a tuple (hidden_states, ...)
            # We want the hidden states after the layer (residual stream)
            hidden_states = output[0]  # Shape: (batch_size, seq_len, hidden_size)
            # Ensure we keep the original shape
            if len(hidden_states.shape) == 2:
                # If it's 2D, we might need to unsqueeze
                # This shouldn't happen but let's be safe
                hidden_states = hidden_states.unsqueeze(0)
            self.representations[layer_idx] = hidden_states.detach().cpu()
        return hook_fn
        
    def register_hooks(self):
        """Register forward hooks on all specified layers."""
        for layer_idx in self.layer_indices:
            # Access the specific transformer layer
            layer = self.model.model.layers[layer_idx]
            hook_fn = self.create_hook_fn(layer_idx)
            handle = layer.register_forward_hook(hook_fn)
            self.hook_handles.append(handle)
        
    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        self.representations = {}
            
    def extract(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None, 
                concatenate: bool = True) -> Union[torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Extract representations for given inputs.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            concatenate: If True and multiple layers, concatenate representations.
                        If False, return dict mapping layer_idx to representations.
        
        Returns:
            If single layer: tensor of shape (batch_size, seq_len, hidden_size)
            If multiple layers and concatenate=True: tensor of shape (batch_size, seq_len, hidden_size * n_layers)
            If multiple layers and concatenate=False: dict mapping layer_idx to tensors
        """
        self.representations = {}
        self.register_hooks()
        
        try:
            with torch.no_grad():
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            
            # Get the captured representations
            reps = {idx: self.representations[idx].clone() for idx in self.layer_indices}
            
        finally:
            # Always remove hooks, even if an error occurred
            self.remove_hooks()
        
        # Return based on configuration
        if len(self.layer_indices) == 1:
            # Single layer - return tensor directly
            return reps[self.layer_indices[0]]
        elif concatenate:
            # Multiple layers - concatenate along hidden dimension
            concatenated = torch.cat([reps[idx] for idx in self.layer_indices], dim=-1)
            return concatenated
        else:
            # Multiple layers - return dictionary
            return reps
    
    def __enter__(self):
        """Context manager entry - register hooks."""
        self.register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - remove hooks."""
        self.remove_hooks()
        return False
    
    @property
    def layer_idx(self):
        """Backward compatibility - return first layer index."""
        return self.layer_indices[0]
    
    @property
    def n_layers(self):
        """Number of layers being extracted."""
        return len(self.layer_indices)
    
    def __repr__(self):
        if len(self.layer_indices) == 1:
            return f"RepresentationExtractor(layer={self.layer_indices[0]})"
        else:
            return f"RepresentationExtractor(layers={self.layer_indices})"