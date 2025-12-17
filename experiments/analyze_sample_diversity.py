#!/usr/bin/env python3
"""
Analyze sample diversity metrics for retrofit experiments.

Measures:
1. Vocabulary diversity (unique words per 100 tokens)
2. Repetition rate (n-gram overlap)
3. Self-BLEU (diversity across samples)
4. Token entropy
"""

import json
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    import nltk
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    console.print("[yellow]nltk not installed - will use simple tokenization[/yellow]")


def simple_tokenize(text):
    """Simple tokenization for when NLTK is not available."""
    return text.lower().replace(',', ' ,').replace('.', ' .').replace('!', ' !').replace('?', ' ?').split()


def get_ngrams(tokens, n):
    """Extract n-grams from token list."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def compute_vocab_diversity(samples):
    """Compute vocabulary diversity: unique words per 100 tokens."""
    all_tokens = []
    for sample in samples:
        text = sample['text']
        if HAS_NLTK:
            tokens = word_tokenize(text.lower())
        else:
            tokens = simple_tokenize(text)
        all_tokens.extend(tokens)

    if len(all_tokens) == 0:
        return 0.0

    unique_words = len(set(all_tokens))
    total_words = len(all_tokens)

    # Normalize to per-100 tokens
    diversity = (unique_words / total_words) * 100

    return {
        'unique_words': unique_words,
        'total_words': total_words,
        'diversity_per_100': diversity,
        'type_token_ratio': unique_words / total_words
    }


def compute_repetition_rate(samples, n=3):
    """Compute n-gram repetition rate."""
    all_ngrams = []

    for sample in samples:
        text = sample['text']
        if HAS_NLTK:
            tokens = word_tokenize(text.lower())
        else:
            tokens = simple_tokenize(text)

        ngrams = get_ngrams(tokens, n)
        all_ngrams.extend(ngrams)

    if len(all_ngrams) == 0:
        return {'repetition_rate': 0.0, 'unique_ngrams': 0, 'total_ngrams': 0}

    ngram_counts = Counter(all_ngrams)
    repeated_ngrams = sum(1 for count in ngram_counts.values() if count > 1)
    unique_ngrams = len(ngram_counts)
    total_ngrams = len(all_ngrams)

    repetition_rate = repeated_ngrams / unique_ngrams if unique_ngrams > 0 else 0.0

    # Also compute what fraction of total ngrams are repetitions
    repeated_instances = sum(count - 1 for count in ngram_counts.values() if count > 1)
    repetition_fraction = repeated_instances / total_ngrams if total_ngrams > 0 else 0.0

    return {
        'repetition_rate': repetition_rate,
        'repetition_fraction': repetition_fraction,
        'unique_ngrams': unique_ngrams,
        'total_ngrams': total_ngrams,
        'n': n
    }


def compute_self_bleu(samples, n=3):
    """
    Compute Self-BLEU score.
    Lower Self-BLEU = more diverse samples.

    For each sample, compute BLEU against all other samples.
    """
    if not HAS_NLTK:
        return {'self_bleu': 0.0, 'note': 'nltk not available'}

    if len(samples) < 2:
        return {'self_bleu': 0.0, 'note': 'need at least 2 samples'}

    smooth = SmoothingFunction()
    bleu_scores = []

    for i, sample_i in enumerate(samples):
        text_i = sample_i['text']
        tokens_i = word_tokenize(text_i.lower())

        # Compare against all other samples
        references = []
        for j, sample_j in enumerate(samples):
            if i != j:
                text_j = sample_j['text']
                tokens_j = word_tokenize(text_j.lower())
                references.append(tokens_j)

        # Compute BLEU-n
        if references:
            weights = tuple([1.0/n] * n)  # Uniform weights for n-grams up to n
            bleu = sentence_bleu(
                references,
                tokens_i,
                weights=weights,
                smoothing_function=smooth.method1
            )
            bleu_scores.append(bleu)

    avg_self_bleu = np.mean(bleu_scores) if bleu_scores else 0.0

    return {
        'self_bleu': avg_self_bleu,
        'std': np.std(bleu_scores) if bleu_scores else 0.0,
        'n': n,
        'num_comparisons': len(bleu_scores)
    }


def compute_token_entropy(samples):
    """Compute average token-level entropy (how predictable the text is)."""
    token_counts = Counter()
    total_tokens = 0

    for sample in samples:
        text = sample['text']
        if HAS_NLTK:
            tokens = word_tokenize(text.lower())
        else:
            tokens = simple_tokenize(text)

        token_counts.update(tokens)
        total_tokens += len(tokens)

    if total_tokens == 0:
        return {'entropy': 0.0, 'perplexity': 0.0}

    # Compute entropy
    entropy = 0.0
    for count in token_counts.values():
        prob = count / total_tokens
        entropy -= prob * np.log2(prob)

    perplexity = 2 ** entropy

    return {
        'entropy': entropy,
        'perplexity': perplexity,
        'unique_tokens': len(token_counts),
        'total_tokens': total_tokens
    }


def analyze_method_samples(method_name, samples):
    """Analyze all diversity metrics for a single method."""
    console.print(f"\n[cyan]Analyzing {method_name}...[/cyan]")

    metrics = {}

    # Vocabulary diversity
    metrics['vocab_diversity'] = compute_vocab_diversity(samples)

    # Repetition rates for different n-gram sizes
    metrics['repetition_2gram'] = compute_repetition_rate(samples, n=2)
    metrics['repetition_3gram'] = compute_repetition_rate(samples, n=3)
    metrics['repetition_4gram'] = compute_repetition_rate(samples, n=4)

    # Self-BLEU
    metrics['self_bleu_2'] = compute_self_bleu(samples, n=2)
    metrics['self_bleu_3'] = compute_self_bleu(samples, n=3)

    # Token entropy
    metrics['token_entropy'] = compute_token_entropy(samples)

    return metrics


def create_comparison_table(all_metrics):
    """Create rich table comparing diversity metrics."""
    table = Table(show_header=True, header_style="bold magenta", title="Sample Diversity Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("Standard SGD", justify="right")
    table.add_column("Gram Only", justify="right")
    table.add_column("Full Geometry", justify="right")
    table.add_column("Winner", style="green")

    # Vocabulary diversity (higher is better)
    std_vocab = all_metrics.get('standard', {}).get('vocab_diversity', {}).get('diversity_per_100', 0)
    gram_vocab = all_metrics.get('gram_only', {}).get('vocab_diversity', {}).get('diversity_per_100', 0)
    geom_vocab = all_metrics.get('geometry', {}).get('vocab_diversity', {}).get('diversity_per_100', 0)
    winner_vocab = max([('Standard', std_vocab), ('Gram', gram_vocab), ('Geometry', geom_vocab)], key=lambda x: x[1])[0]

    table.add_row(
        "Vocab Diversity (per 100)",
        f"{std_vocab:.2f}",
        f"{gram_vocab:.2f}",
        f"{geom_vocab:.2f}",
        winner_vocab
    )

    # Type-token ratio (higher is better)
    std_ttr = all_metrics.get('standard', {}).get('vocab_diversity', {}).get('type_token_ratio', 0)
    gram_ttr = all_metrics.get('gram_only', {}).get('vocab_diversity', {}).get('type_token_ratio', 0)
    geom_ttr = all_metrics.get('geometry', {}).get('vocab_diversity', {}).get('type_token_ratio', 0)
    winner_ttr = max([('Standard', std_ttr), ('Gram', gram_ttr), ('Geometry', geom_ttr)], key=lambda x: x[1])[0]

    table.add_row(
        "Type-Token Ratio",
        f"{std_ttr:.3f}",
        f"{gram_ttr:.3f}",
        f"{geom_ttr:.3f}",
        winner_ttr
    )

    # 3-gram repetition rate (lower is better)
    std_rep = all_metrics.get('standard', {}).get('repetition_3gram', {}).get('repetition_fraction', 0)
    gram_rep = all_metrics.get('gram_only', {}).get('repetition_3gram', {}).get('repetition_fraction', 0)
    geom_rep = all_metrics.get('geometry', {}).get('repetition_3gram', {}).get('repetition_fraction', 0)
    winner_rep = min([('Standard', std_rep), ('Gram', gram_rep), ('Geometry', geom_rep)], key=lambda x: x[1])[0]

    table.add_row(
        "3-gram Repetition %",
        f"{std_rep*100:.2f}%",
        f"{gram_rep*100:.2f}%",
        f"{geom_rep*100:.2f}%",
        winner_rep
    )

    # Self-BLEU-3 (lower is better - more diverse)
    std_bleu = all_metrics.get('standard', {}).get('self_bleu_3', {}).get('self_bleu', 1.0)
    gram_bleu = all_metrics.get('gram_only', {}).get('self_bleu_3', {}).get('self_bleu', 1.0)
    geom_bleu = all_metrics.get('geometry', {}).get('self_bleu_3', {}).get('self_bleu', 1.0)
    winner_bleu = min([('Standard', std_bleu), ('Gram', gram_bleu), ('Geometry', geom_bleu)], key=lambda x: x[1])[0]

    table.add_row(
        "Self-BLEU-3 (lower=diverse)",
        f"{std_bleu:.3f}",
        f"{gram_bleu:.3f}",
        f"{geom_bleu:.3f}",
        winner_bleu
    )

    # Token entropy (higher is better - more unpredictable/creative)
    std_ent = all_metrics.get('standard', {}).get('token_entropy', {}).get('entropy', 0)
    gram_ent = all_metrics.get('gram_only', {}).get('token_entropy', {}).get('entropy', 0)
    geom_ent = all_metrics.get('geometry', {}).get('token_entropy', {}).get('entropy', 0)
    winner_ent = max([('Standard', std_ent), ('Gram', gram_ent), ('Geometry', geom_ent)], key=lambda x: x[1])[0]

    table.add_row(
        "Token Entropy (bits)",
        f"{std_ent:.2f}",
        f"{gram_ent:.2f}",
        f"{geom_ent:.2f}",
        winner_ent
    )

    return table


def main():
    parser = argparse.ArgumentParser(description="Analyze sample diversity")
    parser.add_argument("--results_dir", type=str, default="artifacts_ts/retrofit_comparison",
                       help="Directory with result JSON files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for metrics")

    args = parser.parse_args()

    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]Sample Diversity Analysis[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

    # Load results
    results_dir = Path(args.results_dir)
    all_metrics = {}

    for method in ["standard", "gram_only", "geometry"]:
        result_file = results_dir / f"{method}.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)

            if 'samples' in data and data['samples']:
                metrics = analyze_method_samples(method, data['samples'])
                all_metrics[method] = metrics

    # Create comparison table
    console.print("\n")
    table = create_comparison_table(all_metrics)
    console.print(table)

    # Print detailed metrics
    console.print(f"\n[bold green]{'='*70}[/bold green]")
    console.print(f"[bold green]Detailed Metrics[/bold green]")
    console.print(f"[bold green]{'='*70}[/bold green]\n")

    for method, metrics in all_metrics.items():
        console.print(f"\n[bold cyan]{method.upper()}:[/bold cyan]")

        vocab = metrics['vocab_diversity']
        console.print(f"  Vocabulary: {vocab['unique_words']} unique / {vocab['total_words']} total")
        console.print(f"  TTR: {vocab['type_token_ratio']:.4f}")

        rep3 = metrics['repetition_3gram']
        console.print(f"  3-gram repetitions: {rep3['repetition_fraction']*100:.2f}% of tokens")

        entropy = metrics['token_entropy']
        console.print(f"  Token entropy: {entropy['entropy']:.2f} bits (perplexity: {entropy['perplexity']:.1f})")

        if 'self_bleu' in metrics.get('self_bleu_3', {}):
            bleu = metrics['self_bleu_3']
            console.print(f"  Self-BLEU-3: {bleu['self_bleu']:.4f} (lower = more diverse)")

    # Save metrics
    output_path = args.output or results_dir / "diversity_metrics.json"
    with open(output_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    console.print(f"\n[green]Metrics saved to: {output_path}[/green]\n")


if __name__ == "__main__":
    main()
