#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
import json
import os

# Custom tokenizer creation
def create_custom_tokenizer():
    """Create tokenizer with specified vocabulary"""
    # Define vocabulary: digits 0-9, special chars, BOS, EOS
    vocab = {
        '<bos>': 0,
        '<eos>': 1,
        '<pad>': 2,
        'd': 3,
        'c': 4,
        '_': 5,
        '(': 6,
        ')': 7,
        ',': 8,
        '=': 9,
        '0': 10,
        '1': 11,
        '2': 12,
        '3': 13,
        '4': 14,
        '5': 15,
        '6': 16,
        '7': 17,
        '8': 18,
        '9': 19,
    }
    
    # Create tokenizer from vocabulary
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token='<pad>'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    
    # Create HuggingFace tokenizer wrapper
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token='<bos>',
        eos_token='<eos>',
        pad_token='<pad>',
        model_max_length=32,  # Max sequence length for our distance format
    )
    
    return hf_tokenizer, vocab

# Custom dataset class
class DistanceDataset(Dataset):
    def __init__(self, dataset_or_path, tokenizer, max_length=32, split='train', loss_mask_type=None):
        # Handle both HF dataset path and direct dataset
        if isinstance(dataset_or_path, str):
            from datasets import load_from_disk
            full_dataset = load_from_disk(dataset_or_path)
            self.dataset = full_dataset[split]
        else:
            self.dataset = dataset_or_path
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loss_mask_type = loss_mask_type
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        text = self.dataset[idx]['text']
        
        # Add BOS and EOS tokens
        text_with_special = f"<bos> {' '.join(text)} <eos>"
        
        # Tokenize character by character
        tokens = []
        for char in text_with_special.split():
            if char in self.tokenizer.get_vocab():
                tokens.append(self.tokenizer.get_vocab()[char])
            else:
                tokens.append(self.tokenizer.get_vocab()['<pad>'])  # Use pad for unknown
        
        # Pad or truncate to max_length
        if len(tokens) < self.max_length:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        
        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # For causal LM, labels are the same as input_ids (shifted internally)
        labels = input_ids.clone()
        
        # Apply loss masking if specified
        if self.loss_mask_type == "answer_only":
            # Find the position of '=' token (token_id=9)
            equals_positions = (input_ids == 9).nonzero(as_tuple=True)[0]
            if len(equals_positions) > 0:
                # Mask everything before and including '='
                # Only compute loss on the answer (YYYY<eos>) portion
                answer_start = equals_positions[0].item() + 1
                labels[:answer_start] = -100  # -100 is ignored in cross-entropy loss
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long()
        }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create custom tokenizer
    print("Creating custom tokenizer...")
    tokenizer, vocab = create_custom_tokenizer()
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Vocabulary: {list(vocab.keys())}")
    
    # Model configuration for small Qwen2.5-like model
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,  # Hidden dimension
        intermediate_size=256,  # FFN dimension (typically 4x hidden_size)
        num_hidden_layers=4,  # Number of layers
        num_attention_heads=4,  # Number of attention heads
        num_key_value_heads=4,  # For standard multi-head attention
        max_position_embeddings=32,  # Max sequence length
        rope_theta=10000.0,  # RoPE base frequency
        rope_scaling=None,  # No RoPE scaling
        tie_word_embeddings=False,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )
    
    print("\nModel configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_hidden_layers}")
    print(f"  Num attention heads: {config.num_attention_heads}")
    print(f"  Vocab size: {config.vocab_size}")
    
    # Initialize model
    print("\nInitializing model...")
    model = Qwen2ForCausalLM(config)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load training data from HuggingFace dataset
    dataset_path = '/n/home12/cfpark00/WM_1/outputs/datasets/dist_100kplus_10000_2000_42'
    if not os.path.exists(dataset_path):
        print(f"\nError: Dataset not found at {dataset_path}")
        return
    
    # Test different loss mask types
    loss_mask_type = "answer_only"  # Options: None (standard), "answer_only"
    print(f"\nLoading training data from {dataset_path}...")
    print(f"Using loss_mask_type: {loss_mask_type}")
    train_dataset = DistanceDataset(dataset_path, tokenizer, split='train', loss_mask_type=loss_mask_type)
    print(f"Training samples: {len(train_dataset)}")
    
    # Create dataloader
    batch_size = 8
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # Get one batch
    print(f"\nGetting one batch (batch_size={batch_size})...")
    batch = next(iter(train_loader))
    
    print("\nBatch info:")
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    
    # Display all samples from batch
    print("\n" + "="*60)
    print("BATCH SAMPLES:")
    print("="*60)
    
    for i in range(min(3, batch_size)):  # Only show first 3 samples
        print(f"\n--- Sample {i+1} ---")
        
        # Get the original text
        original_text = train_dataset.dataset[i]['text']
        print(f"Original text: {original_text}")
        
        # Get the tokenized version
        tokens = batch['input_ids'][i]
        token_list = []
        token_ids = []
        
        for token_id in tokens:
            if token_id == tokenizer.pad_token_id:
                break
            token_ids.append(token_id.item())
            for token, tid in vocab.items():
                if tid == token_id.item():
                    token_list.append(token)
                    break
        
        print(f"Tokenized: {' '.join(token_list)}")
        print(f"Token IDs: {token_ids}")
        print(f"Total tokens (non-pad): {len(token_ids)}")
        
        # Show labels (with potential masking)
        labels = batch['labels'][i]
        label_list = []
        for label_id in labels:
            if label_id == tokenizer.pad_token_id:
                break
            elif label_id == -100:
                label_list.append('[MASKED]')
            else:
                for token, tid in vocab.items():
                    if tid == label_id.item():
                        label_list.append(token)
                        break
        print(f"Labels: {' '.join(label_list)}")
        
        # Show which tokens will contribute to loss
        loss_tokens = []
        for j, label_id in enumerate(labels):
            if label_id != -100 and label_id != tokenizer.pad_token_id:
                for token, tid in vocab.items():
                    if tid == label_id.item():
                        loss_tokens.append(token)
                        break
        print(f"Tokens contributing to loss: {' '.join(loss_tokens)}")
        
        # Show padding info
        total_length = len(batch['input_ids'][i])
        num_pad_tokens = total_length - len(token_ids)
        print(f"Padding: {num_pad_tokens} PAD tokens added to reach length {total_length}")
        
        # Show attention mask
        attention_mask = batch['attention_mask'][i]
        print(f"Attention mask (first 30): {attention_mask[:30].tolist()}...")
    
    # Forward pass with one batch
    print("\nPerforming forward pass...")
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits
    
    print(f"Loss: {loss.item():.4f}")
    print(f"Logits shape: {logits.shape}")
    
    # Show per-token loss info
    print("\nPer-token loss computation:")
    if loss_mask_type == "answer_only":
        print("  Loss computed only on answer tokens (YYYY<eos>)")
    else:
        print("  Loss computed on all non-pad tokens (standard next-token prediction)")
    
    # Test generation
    print("\n" + "="*60)
    print("GENERATION TEST (Temperature 0):")
    print("="*60)
    
    model.eval()
    
    # Take first 3 samples from dataset
    for i in range(min(3, len(train_dataset.dataset))):
        sample = train_dataset.dataset[i]
        prompt_text = sample['prompt']
        expected_completion = sample['completion']
        
        print(f"\n--- Sample {i+1} ---")
        print(f"Prompt: {prompt_text}")
        print(f"Expected: {expected_completion}")
        
        # Tokenize the prompt
        prompt_tokens = []
        for char in prompt_text.replace('<bos>', '').split():
            if char == '<bos>':
                prompt_tokens.append(0)
            elif char in tokenizer.get_vocab():
                prompt_tokens.append(tokenizer.get_vocab()[char])
        
        # Add BOS token at the beginning
        prompt_tokens = [0] + [tokenizer.get_vocab().get(c, 2) for c in prompt_text.replace('<bos>', '')]
        prompt_tensor = torch.tensor([prompt_tokens]).to(device)
        
        # Generate 6 tokens (max 5 digit number + EOS)
        with torch.no_grad():
            generated = model.generate(
                prompt_tensor,
                max_new_tokens=6,
                temperature=0.0001,  # Near 0 for deterministic
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode the generated tokens
        generated_tokens = generated[0][len(prompt_tokens):].tolist()
        decoded = []
        for token_id in generated_tokens:
            if token_id == tokenizer.eos_token_id:
                decoded.append('<eos>')
                break
            for token, tid in vocab.items():
                if tid == token_id:
                    decoded.append(token)
                    break
        
        print(f"Generated: {''.join(decoded)}")
    
    print("\nScript completed successfully!")
    print("Ready for full training implementation.")

if __name__ == "__main__":
    main()