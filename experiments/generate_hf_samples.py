#!/usr/bin/env python3
"""
Generate text samples from HF baseline model for qualitative comparison.
"""

import sys
import json
import torch
from pathlib import Path
from rich.console import Console

# Add current directory to path to import hf_model
sys.path.insert(0, str(Path(__file__).parent))
from hf_model import GPT, GPTConfig

console = Console()

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    console.print("[red]transformers not installed. Install with: pip install transformers[/red]")
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


def main():
    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]Generating Samples from HuggingFace Baseline Model[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

    # Load config
    console.print("[cyan]Loading model configuration...[/cyan]")
    config_path = Path(__file__).parent / "hf_config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    config = GPTConfig(
        block_size=config_dict.get("block_size", 128),
        vocab_size=config_dict.get("vocab_size", 50257),
        n_layer=config_dict.get("n_layer", 6),
        n_head=config_dict.get("n_head", 6),
        n_embd=config_dict.get("n_embd", 384),
        dropout=config_dict.get("dropout", 0.1),
    )

    # Create model
    console.print("[cyan]Creating model...[/cyan]")
    model = GPT(config)

    # Load weights
    console.print("[cyan]Loading model weights...[/cyan]")
    weights_path = Path(__file__).parent / "hf_pytorch_model.bin"

    try:
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
    except:
        console.print("[yellow]Using legacy torch.load (weights_only not available)[/yellow]")
        state_dict = torch.load(weights_path, map_location='cpu')

    model.load_state_dict(state_dict)
    console.print("[green]Weights loaded successfully![/green]")

    # Move to device
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    console.print(f"[green]Model moved to {device}[/green]")

    # Load tokenizer
    console.print("[cyan]Loading tokenizer...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Generate samples
    prompts = [
        "Once upon a time",
        "One day, a little girl",
        "There was a cat who"
    ]

    console.print(f"\n[cyan]Generating samples...[/cyan]\n")

    output_lines = []
    output_lines.append("Generated Samples - HuggingFace Baseline Model")
    output_lines.append("=" * 70)
    output_lines.append("")

    for i, prompt in enumerate(prompts, 1):
        console.print(f"[cyan]Sample {i}: {prompt}...[/cyan]")

        text = generate_sample(model, tokenizer, prompt, device, max_length=200, temperature=0.8, top_k=50)

        output_lines.append(f"Sample {i}:")
        output_lines.append(f"Prompt: {prompt}")
        output_lines.append(text)
        output_lines.append("")
        output_lines.append("-" * 70)
        output_lines.append("")

        console.print(f"[green]{text}[/green]\n")

    # Save to file
    output_path = Path(__file__).parent.parent / "artifacts_ts" / "hf_baseline_samples.txt"
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        f.write('\n'.join(output_lines))

    console.print(f"[green]Samples saved to: {output_path}[/green]")


if __name__ == "__main__":
    main()
