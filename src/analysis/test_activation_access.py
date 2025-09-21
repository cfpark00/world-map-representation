#!/usr/bin/env python3
"""
Test script to demonstrate how to actually get activation tensors as objects.
"""

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

def demo_activation_access():
    """
    Demonstrate different ways to access activation tensors.
    """

    # Load a small model for testing
    model_path = "/n/home12/cfpark00/WM_1/data/experiments/m1_10M/checkpoints/checkpoint-100"

    print("Loading model...")
    model = Qwen2ForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    # Create a simple input
    text = "<bos>dist(c_1234,c_5678)=2332<eos>"
    spaced_text = ' '.join(text)
    inputs = tokenizer(spaced_text, return_tensors='pt')
    input_ids = inputs['input_ids']

    print(f"Input shape: {input_ids.shape}")
    print(f"Tokens: {[tokenizer.decode([t]) for t in input_ids[0][:20]]}")

    # ============================================================
    # METHOD 1: Using output_hidden_states=True
    # ============================================================
    print("\n" + "="*60)
    print("METHOD 1: Using output_hidden_states=True")
    print("="*60)

    with torch.no_grad():  # Remove this if you need gradients
        outputs = model(
            input_ids,
            output_hidden_states=True,  # THIS IS KEY!
            return_dict=True
        )

    # outputs.hidden_states is a tuple of tensors
    # Index 0 = embeddings
    # Index 1 = output of layer 1
    # Index 2 = output of layer 2, etc.

    hidden_states = outputs.hidden_states
    print(f"Number of hidden state tensors: {len(hidden_states)}")
    print(f"Shape of each tensor: {hidden_states[0].shape}")  # (batch, seq_len, hidden_dim)

    # Get specific activation tensor
    layer_idx = 4  # Layer 4
    token_idx = 10  # Token position 10 (e.g., the "4" in c_1234)

    # THIS IS THE ACTIVATION TENSOR AS AN OBJECT!
    activation_tensor = hidden_states[layer_idx][0, token_idx, :]  # Shape: (hidden_dim,)
    print(f"\nActivation tensor at layer {layer_idx}, position {token_idx}:")
    print(f"  Shape: {activation_tensor.shape}")
    print(f"  Type: {type(activation_tensor)}")
    print(f"  Device: {activation_tensor.device}")
    print(f"  Requires grad: {activation_tensor.requires_grad}")
    print(f"  First 5 values: {activation_tensor[:5]}")

    # ============================================================
    # METHOD 2: Using hooks to capture activations
    # ============================================================
    print("\n" + "="*60)
    print("METHOD 2: Using forward hooks")
    print("="*60)

    captured_activations = {}

    def capture_activation(name):
        def hook(model, input, output):
            # output is the activation tensor!
            captured_activations[name] = output
            print(f"  Captured {name}: shape {output.shape}")
        return hook

    # Register hooks on specific layers
    hooks = []
    for i, layer in enumerate(model.model.layers):
        if i in [4, 6, 8]:  # Only specific layers
            hook = layer.register_forward_hook(capture_activation(f'layer_{i}'))
            hooks.append(hook)

    # Forward pass - hooks will capture activations
    with torch.no_grad():
        outputs = model(input_ids)

    # Now we have the activation tensors in captured_activations
    for name, activation in captured_activations.items():
        print(f"{name}: {activation.shape}")

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    # ============================================================
    # METHOD 3: Direct access through model internals
    # ============================================================
    print("\n" + "="*60)
    print("METHOD 3: Direct access through model layers")
    print("="*60)

    # We can also manually run through layers
    with torch.no_grad():
        # Get embeddings
        embeddings = model.model.embed_tokens(input_ids)
        hidden = embeddings

        # Pass through each layer manually
        for i, layer in enumerate(model.model.layers):
            hidden = layer(hidden)[0]  # Each layer returns (output, optional_stuff)

            if i == 4:  # Capture layer 4 output
                layer_4_output = hidden
                print(f"Layer 4 output shape: {layer_4_output.shape}")

                # Get specific token activation
                token_activation = layer_4_output[0, 10, :]
                print(f"Token 10 activation shape: {token_activation.shape}")

    # ============================================================
    # FOR GRADIENT ANALYSIS - Making tensors require gradients
    # ============================================================
    print("\n" + "="*60)
    print("FOR GRADIENT ANALYSIS")
    print("="*60)

    # Need to remove torch.no_grad() context!
    outputs = model(
        input_ids,
        output_hidden_states=True,
        return_dict=True
    )

    hidden_states = outputs.hidden_states
    logits = outputs.logits

    # Get activation at specific position
    layer_idx = 6
    token_idx = 10

    # THIS IS THE KEY: The activation tensor is already part of the computation graph!
    activation = hidden_states[layer_idx][0, token_idx, :]

    print(f"Activation requires_grad: {activation.requires_grad}")

    # If we want to get gradients w.r.t. this activation, we need to:
    # 1. Ensure it's part of the computation graph (it already is!)
    # 2. Call retain_grad() to keep gradients after backward
    activation.retain_grad()

    # Compute some loss
    loss = logits[0, -1, :].sum()  # Dummy loss for demo

    # Backward pass
    loss.backward()

    # Now we can access the gradient!
    if activation.grad is not None:
        print(f"Gradient shape: {activation.grad.shape}")
        print(f"Gradient norm: {activation.grad.norm().item()}")
    else:
        print("No gradient (might need to ensure model is in training mode)")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("The activation tensor IS just: hidden_states[layer][batch, position, :]")
    print("It's already a PyTorch tensor object we can work with!")
    print("The key is using output_hidden_states=True in the forward pass.")


if __name__ == "__main__":
    demo_activation_access()