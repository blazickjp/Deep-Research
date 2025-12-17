#!/usr/bin/env python3
"""
Diagnostic script to check how much training data we get with different max_docs settings.
"""
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def analyze_dataset(max_docs, block_size=256):
    """Analyze how much training data we get with given max_docs."""
    print(f"\n{'='*70}")
    print(f"Analyzing with max_docs={max_docs}, block_size={block_size}")
    print('='*70)

    # Load dataset
    ds = load_dataset("roneneldan/TinyStories")

    if max_docs > 0:
        text = ds["train"]["text"][:max_docs]
        print(f"Using first {max_docs} documents")
    else:
        text = ds["train"]["text"]
        print(f"Using ALL documents ({len(text)} total)")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    ids = [torch.tensor(tokenizer.encode(s, add_special_tokens=False), dtype=torch.long)
           for s in text if len(s) > 0]

    # Concatenate
    stream = torch.cat([x for x in ids if len(x) > 0])

    # Split 90/10
    n = int(0.9 * len(stream))
    train_stream = stream[:n]
    val_stream = stream[n:]

    # Calculate number of training examples
    train_toks = train_stream.unfold(0, block_size + 1, block_size + 1)
    val_toks = val_stream.unfold(0, block_size + 1, block_size + 1)

    print(f"\nDataset Statistics:")
    print(f"  Documents:          {len(text):,}")
    print(f"  Total tokens:       {len(stream):,}")
    print(f"  Train tokens:       {len(train_stream):,}")
    print(f"  Val tokens:         {len(val_stream):,}")
    print(f"  Train examples:     {train_toks.size(0):,} (sequences of {block_size})")
    print(f"  Val examples:       {val_toks.size(0):,}")
    print(f"  Epochs @ batch=32:  {train_toks.size(0) // 32} batches per epoch")

    # Estimate tokens per epoch
    tokens_per_epoch = train_toks.size(0) * block_size
    print(f"  Tokens/epoch:       {tokens_per_epoch:,}")
    print(f"  Tokens/2epochs:     {tokens_per_epoch * 2:,}")

    return {
        'docs': len(text),
        'total_tokens': len(stream),
        'train_examples': train_toks.size(0),
        'val_examples': val_toks.size(0)
    }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TinyStories Dataset Size Analysis")
    print("="*70)

    # Test different max_docs values
    configs = [
        10000,    # Sweep default
        50000,    # 5x more
        100000,   # 10x more
        0,        # ALL data
    ]

    results = []
    for max_docs in configs:
        try:
            stats = analyze_dataset(max_docs, block_size=256)
            results.append((max_docs, stats))
        except Exception as e:
            print(f"Error with max_docs={max_docs}: {e}")

    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'max_docs':<12} {'docs':<10} {'train_ex':<12} {'tokens/2ep':<15}")
    print("-"*70)
    for max_docs, stats in results:
        max_docs_str = str(max_docs) if max_docs > 0 else "ALL"
        tokens_2ep = stats['train_examples'] * 256 * 2
        print(f"{max_docs_str:<12} {stats['docs']:<10,} {stats['train_examples']:<12,} {tokens_2ep:<15,}")

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("If your direct run gets PPL 3.3 with max_docs=10000, but")
    print("sweep gets PPL 28-56, possible explanations:")
    print("  1. Direct run used DIFFERENT max_docs (e.g., 0 for ALL data)")
    print("  2. Direct run used --char_level (different tokenization)")
    print("  3. Different random seeds affecting dataset shuffling")
    print("\nTo debug: Check your direct run command. Did you use:")
    print("  - Different --max_docs value?")
    print("  - The --char_level flag?")
    print("  - Different batch size or learning rate?")
