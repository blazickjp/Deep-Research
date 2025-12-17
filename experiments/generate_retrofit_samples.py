#!/usr/bin/env python3
"""
Generate text samples from retrofitted models to check for overfitting.
"""

import sys
import json
import torch
from pathlib import Path
from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent))
from hf_model import GPT, GPTConfig

console = Console()

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    console.print("[red]transformers not installed[/red]")
    sys.exit(1)


@torch.no_grad()
def generate_sample(model, tokenizer, prompt, device, max_length=200, temperature=0.8, top_k=50):
    """Generate a single sample from a prompt."""
    model.eval()

    # Encode prompt
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    # Generate
    generated = model.generate(idx, max_new_tokens=max_length, temperature=temperature, top_k=top_k)

    # Decode
    text = tokenizer.decode(generated[0].tolist())

    return text


def generate_samples_from_checkpoint(checkpoint_path, device, method_name):
    """Generate samples from a checkpoint."""
    console.print(f"\n[bold cyan]Generating samples: {method_name}[/bold cyan]")

    # Load config
    config_path = Path(checkpoint_path).parent.parent.parent / "experiments" / "hf_config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    config = GPTConfig(
        block_size=config_dict["block_size"],
        vocab_size=config_dict["vocab_size"],
        n_layer=config_dict["n_layer"],
        n_head=config_dict["n_head"],
        n_embd=config_dict["n_embd"],
        dropout=0.0,
        bias=config_dict["bias"]
    )

    # Create model
    model = GPT(config)

    # Load checkpoint
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except:
        state_dict = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Generate samples
    prompts = [
        "Once upon a time",
        "One day, a little girl",
        "There was a cat who"
    ]

    samples = []
    for prompt in prompts:
        text = generate_sample(model, tokenizer, prompt, device)
        samples.append({
            "prompt": prompt,
            "text": text
        })
        console.print(f"[green]Generated: {prompt}...[/green]")

    return samples


def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[cyan]Using device: {device}[/cyan]")

    results_dir = Path("artifacts_ts/retrofit_comparison")

    # Check if we have saved model checkpoints
    # (We didn't save them in the original script, so we need to load from results and note this)

    console.print("\n[yellow]Note: Retrofit script didn't save model checkpoints[/yellow]")
    console.print("[yellow]To generate samples, we need to either:[/yellow]")
    console.print("[yellow]1. Re-run retrofit experiments with --save_checkpoint flag[/yellow]")
    console.print("[yellow]2. Or check the perplexity on VALIDATION set (different from training)[/yellow]")
    console.print()

    # Load results to show the concern
    for method in ["standard", "gram_only", "geometry"]:
        result_file = results_dir / f"{method}.json"
        if result_file.exists():
            with open(result_file) as f:
                data = json.load(f)

            console.print(f"\n[bold]{method.upper()}:[/bold]")
            console.print(f"  Initial PPL: {data['initial']['ppl']:.2f}")
            console.print(f"  Final PPL:   {data['final']['ppl']:.2f}")
            console.print(f"  Improvement: {100*(1 - data['final']['ppl']/data['initial']['ppl']):.1f}%")

            # Check trajectory
            traj = data['trajectory']
            console.print(f"  Steps where PPL measured: {len(traj['steps'])}")
            console.print(f"  Training set size: ~1000 docs")
            console.print(f"  Validation set size: ~500 docs")


if __name__ == "__main__":
    main()
