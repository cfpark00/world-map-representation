"""
Unified tokenizer configuration for all WM_1 experiments.

Total vocabulary size: 44 tokens
- 3 special tokens: <bos>, <eos>, <pad>
- 5 grammar tokens: (, ), ,, =, _
- 26 alphabets: a-z
- 10 digits: 0-9
"""

import string
from typing import Dict, List, Tuple

# Special tokens
SPECIAL_TOKENS = {
    '<pad>': 0,
    '<bos>': 1,
    '<eos>': 2,
}

# Grammar tokens
GRAMMAR_TOKENS = {
    '(': 3,
    ')': 4,
    ',': 5,
    '=': 6,
    '_': 7,
}

# Alphabet tokens (lowercase only)
ALPHABET_TOKENS = {
    letter: idx + 8 for idx, letter in enumerate(string.ascii_lowercase)
}

# Digit tokens
DIGIT_TOKENS = {
    str(digit): idx + 34 for idx, digit in enumerate(range(10))
}

# Combine all tokens
VOCAB = {
    **SPECIAL_TOKENS,
    **GRAMMAR_TOKENS,
    **ALPHABET_TOKENS,
    **DIGIT_TOKENS
}

# Reverse mapping for decoding
ID_TO_TOKEN = {v: k for k, v in VOCAB.items()}

# Vocabulary size
VOCAB_SIZE = len(VOCAB)

# Special token IDs for easy access
PAD_TOKEN_ID = SPECIAL_TOKENS['<pad>']
BOS_TOKEN_ID = SPECIAL_TOKENS['<bos>']
EOS_TOKEN_ID = SPECIAL_TOKENS['<eos>']

def get_vocab() -> Dict[str, int]:
    """Return the complete vocabulary mapping."""
    return VOCAB.copy()

def get_id_to_token() -> Dict[int, str]:
    """Return the ID to token mapping for decoding."""
    return ID_TO_TOKEN.copy()

def tokenize(text: str) -> List[int]:
    """
    Tokenize a string into token IDs.
    
    Args:
        text: Input string to tokenize
        
    Returns:
        List of token IDs
        
    Raises:
        ValueError: If the text contains unknown characters
    """
    tokens = []
    for char in text:
        if char in VOCAB:
            tokens.append(VOCAB[char])
        else:
            raise ValueError(f"Unknown character: '{char}'. Valid characters are: {list(VOCAB.keys())}")
    return tokens

def decode(token_ids: List[int]) -> str:
    """
    Decode a list of token IDs back to string.
    
    Args:
        token_ids: List of token IDs
        
    Returns:
        Decoded string
        
    Raises:
        ValueError: If an invalid token ID is encountered
    """
    chars = []
    for token_id in token_ids:
        if token_id in ID_TO_TOKEN:
            chars.append(ID_TO_TOKEN[token_id])
        else:
            raise ValueError(f"Unknown token ID: {token_id}. Valid IDs are 0-{VOCAB_SIZE-1}")
    return ''.join(chars)

def encode_with_special(text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
    """
    Encode text with optional special tokens.
    
    Args:
        text: Input text to encode
        add_bos: Whether to add <bos> token at the beginning
        add_eos: Whether to add <eos> token at the end
        
    Returns:
        List of token IDs with special tokens
    """
    tokens = []
    if add_bos:
        tokens.append(BOS_TOKEN_ID)
    tokens.extend(tokenize(text))
    if add_eos:
        tokens.append(EOS_TOKEN_ID)
    return tokens

def decode_skip_special(token_ids: List[int]) -> str:
    """
    Decode token IDs, skipping special tokens.
    
    Args:
        token_ids: List of token IDs
        
    Returns:
        Decoded string without special tokens
    """
    chars = []
    for token_id in token_ids:
        if token_id in ID_TO_TOKEN and token_id not in SPECIAL_TOKENS.values():
            chars.append(ID_TO_TOKEN[token_id])
    return ''.join(chars)

def print_vocab_info():
    """Print vocabulary information."""
    print(f"Total vocabulary size: {VOCAB_SIZE}")
    print(f"Special tokens (3): {list(SPECIAL_TOKENS.keys())}")
    print(f"Grammar tokens (5): {list(GRAMMAR_TOKENS.keys())}")
    print(f"Alphabet tokens (26): a-z")
    print(f"Digit tokens (10): 0-9")
    print("\nToken ID mappings:")
    print(f"  Special: {SPECIAL_TOKENS}")
    print(f"  Grammar: {GRAMMAR_TOKENS}")
    print(f"  Alphabets: a={ALPHABET_TOKENS['a']} ... z={ALPHABET_TOKENS['z']}")
    print(f"  Digits: 0={DIGIT_TOKENS['0']} ... 9={DIGIT_TOKENS['9']}")

if __name__ == "__main__":
    # Test the tokenizer
    print_vocab_info()
    
    print("\n" + "="*50)
    print("Testing tokenizer:")
    
    # Test cases
    test_strings = [
        "d(c_0001,c_0002)=100",
        "srd_200=c_1234,c_5678",
        "abc123",
        "<bos>test<eos>",
    ]
    
    for test_str in test_strings:
        print(f"\nOriginal: {test_str}")
        try:
            tokens = tokenize(test_str)
            print(f"Tokens: {tokens}")
            decoded = decode(tokens)
            print(f"Decoded: {decoded}")
            assert decoded == test_str, "Decode mismatch!"
            
            # Test with special tokens
            tokens_with_special = encode_with_special(test_str)
            print(f"With special tokens: {tokens_with_special}")
            decoded_no_special = decode_skip_special(tokens_with_special)
            print(f"Decoded (no special): {decoded_no_special}")
            
        except ValueError as e:
            print(f"Error: {e}")