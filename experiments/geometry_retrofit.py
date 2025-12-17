#!/usr/bin/env python3
"""
Geometry Retrofit Experiment

Tests whether geometry-aware training can improve conditioning
of an already-trained model.

Compares:
1. HF baseline + continued standard training
2. HF baseline + Gram regularization only
3. HF baseline + full geometry (GLT + Gram)
4. Your well-conditioned model + standard training (control)
"""

import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hf_model import GPT, GPTConfig

console = Console()

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    console.print("[red]datasets not installed[/red]")

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    console.print("[red]transformers not installed[/red]")


def compute_gram_condition(features):
    """Compute Gram matrix condition number for a batch of features."""
    B, T, D = features.shape
    X = features.reshape(-1, D)
    G = (X.t() @ X) / (B * T)
    G = G + 1e-5 * torch.eye(D, device=G.device)

    try:
        eigenvalues = torch.linalg.eigvalsh(G)
    except:
        eigenvalues = torch.linalg.eigvalsh(G.cpu()).to(G.device)

    max_eig = eigenvalues.max()
    min_eig = eigenvalues.min()
    condition_number = (max_eig / min_eig).item()

    return condition_number


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


def measure_conditioning(model, val_loader, device, num_batches=20):
    """Measure FF Gram conditioning across all layers."""
    model.eval()

    layer_conditions = []

    with torch.no_grad():
        for layer_idx in range(len(model.transformer.h)):
            all_features = []

            for batch_idx, (xb, yb) in enumerate(val_loader):
                if batch_idx >= num_batches:
                    break

                xb = xb.to(device)

                # Forward through embeddings
                x = model.transformer.wte(xb) + model.transformer.wpe(torch.arange(xb.size(1), device=device))
                x = model.transformer.drop(x)

                # Process through blocks
                for i, block in enumerate(model.transformer.h):
                    if i < layer_idx:
                        x = block(x)
                    elif i == layer_idx:
                        # Extract FFN features
                        x = x + block.attn(block.ln1(x))
                        x_norm = block.ln2(x)
                        ffn_features = block.mlp.gelu(block.mlp.c_fc(x_norm))
                        all_features.append(ffn_features.cpu())
                        break

            # Compute condition number
            if all_features:
                layer_features = torch.cat(all_features, dim=0)
                cond = compute_gram_condition(layer_features)
                layer_conditions.append(cond)

    avg_cond = np.mean(layer_conditions)
    return avg_cond, layer_conditions


def evaluate_perplexity(model, val_loader, device, num_batches=50):
    """Evaluate perplexity on validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch_idx, (xb, yb) in enumerate(val_loader):
            if batch_idx >= num_batches:
                break

            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)

            total_loss += loss.item() * xb.size(0) * xb.size(1)
            total_tokens += xb.size(0) * xb.size(1)

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity, avg_loss


def compute_gram_loss(model):
    """Compute Gram matrix regularization loss."""
    gram_loss = 0.0
    count = 0

    for name, module in model.named_modules():
        if hasattr(module, 'weight') and 'mlp.c_fc' in name:
            W = module.weight  # [out_features, in_features]
            G = W @ W.t()  # [out_features, out_features]

            # Regularize toward identity
            I = torch.eye(G.size(0), device=G.device)
            gram_loss += torch.norm(G - I, p='fro') ** 2
            count += 1

    if count > 0:
        gram_loss = gram_loss / count

    return gram_loss


def gradient_less_transport(model):
    """
    Simplified Gradient-Less Transport.
    Projects weights onto balanced manifold.
    """
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and len(module.weight.shape) == 2:
            W = module.weight.data

            # Simple balanced projection: normalize rows and columns
            # This is a simplified version - full GLT would use SVD

            # Row normalization
            row_norms = torch.norm(W, dim=1, keepdim=True)
            W = W / (row_norms + 1e-8)

            # Column normalization
            col_norms = torch.norm(W, dim=0, keepdim=True)
            W = W / (col_norms + 1e-8)

            # Rescale
            scale = torch.norm(module.weight.data, p='fro') / torch.norm(W, p='fro')
            module.weight.data = W * scale


def prepare_data(max_docs=1000, block_size=128, batch_size=16, split="train", offset=0):
    """Prepare TinyStories data from specified split."""
    if not HAS_DATASETS or not HAS_TOKENIZER:
        raise RuntimeError("datasets and transformers required")

    console.print(f"[cyan]Loading TinyStories {split} split (max_docs={max_docs}, offset={offset})...[/cyan]")
    ds = load_dataset("roneneldan/TinyStories")

    # Get the appropriate split
    if max_docs > 0:
        text = ds[split]["text"][offset:offset+max_docs]
    else:
        text = ds[split]["text"]

    console.print(f"[cyan]Tokenizing {len(text)} stories...[/cyan]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    def enc(s): return tokenizer.encode(s, add_special_tokens=False)
    ids = [torch.tensor(enc(s), dtype=torch.long) for s in text if len(s) > 0]

    stream = torch.cat([x for x in ids if len(x) > 0])
    console.print(f"[green]Tokenized stream: {len(stream):,} tokens[/green]")

    # Create chunks
    toks = stream.unfold(0, block_size+1, block_size+1)

    class SimpleDataset(torch.utils.data.Dataset):
        def __len__(self):
            return toks.size(0)
        def __getitem__(self, i):
            seq = toks[i]
            return seq[:-1], seq[1:]

    loader = torch.utils.data.DataLoader(
        SimpleDataset(),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    console.print(f"[green]Created data loader: {len(loader)} batches[/green]")
    return loader


def retrofit_experiment(
    model,
    train_loader,
    val_loader,
    device,
    num_steps=1000,
    method="geometry",
    gram_weight=0.01,
    glt_interval=50,
    eval_interval=100,
    lr=1e-4,
    save_checkpoint=True,
    output_dir=None,
):
    """
    Run retrofit experiment with specified method.

    Methods:
    - "standard": Normal SGD
    - "gram": Gram regularization only
    - "geometry": Full geometry (GLT + Gram)
    """
    console.print(f"\n[bold cyan]Running {method} training for {num_steps} steps[/bold cyan]")

    # Initial evaluation
    initial_cond, _ = measure_conditioning(model, val_loader, device)
    initial_val_ppl, initial_val_loss = evaluate_perplexity(model, val_loader, device)
    initial_train_ppl, initial_train_loss = evaluate_perplexity(model, train_loader, device, num_batches=20)

    console.print(f"[yellow]Initial: cond={initial_cond:.0f}, Val PPL={initial_val_ppl:.2f}, Train PPL={initial_train_ppl:.2f}[/yellow]")

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Track progress
    trajectory = {
        'steps': [0],
        'conditioning': [initial_cond],
        'val_perplexity': [initial_val_ppl],
        'train_perplexity': [initial_train_ppl],
        'val_loss': [initial_val_loss],
        'train_loss': [initial_train_loss]
    }

    # Training loop
    model.train()
    train_iter = iter(train_loader)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]{method}", total=num_steps)

        for step in range(num_steps):
            # Get batch
            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                xb, yb = next(train_iter)

            xb, yb = xb.to(device), yb.to(device)

            # Forward pass
            logits, loss = model(xb, yb)

            # Add regularization based on method
            if method in ["gram", "geometry"]:
                gram_loss = compute_gram_loss(model)
                total_loss = loss + gram_weight * gram_loss
            else:
                total_loss = loss

            # Backward
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            # GLT transport
            if method == "geometry" and (step + 1) % glt_interval == 0:
                gradient_less_transport(model)

            # Evaluation
            if (step + 1) % eval_interval == 0:
                cond, _ = measure_conditioning(model, val_loader, device, num_batches=10)
                val_ppl, val_loss = evaluate_perplexity(model, val_loader, device, num_batches=20)
                train_ppl, train_loss = evaluate_perplexity(model, train_loader, device, num_batches=10)

                trajectory['steps'].append(step + 1)
                trajectory['conditioning'].append(cond)
                trajectory['val_perplexity'].append(val_ppl)
                trajectory['train_perplexity'].append(train_ppl)
                trajectory['val_loss'].append(val_loss)
                trajectory['train_loss'].append(train_loss)

                progress.console.print(f"  Step {step+1}: cond={cond:.0f}, Val PPL={val_ppl:.2f}, Train PPL={train_ppl:.2f}")

                model.train()

            progress.update(task, advance=1)

    # Final evaluation
    final_cond, final_layer_conds = measure_conditioning(model, val_loader, device)
    final_val_ppl, final_val_loss = evaluate_perplexity(model, val_loader, device)
    final_train_ppl, final_train_loss = evaluate_perplexity(model, train_loader, device, num_batches=50)

    console.print(f"[green]Final: cond={final_cond:.0f}, Val PPL={final_val_ppl:.2f}, Train PPL={final_train_ppl:.2f}[/green]")
    console.print(f"[bold green]Improvement: Δcond={initial_cond - final_cond:.0f} ({100*(initial_cond-final_cond)/initial_cond:.1f}%), ΔVal PPL={initial_val_ppl - final_val_ppl:.2f}[/bold green]")

    # Save checkpoint if requested
    checkpoint_path = None
    if save_checkpoint and output_dir:
        checkpoint_path = Path(output_dir) / f"{method}_model.pt"
        torch.save(model.state_dict(), checkpoint_path)
        console.print(f"[green]Checkpoint saved: {checkpoint_path}[/green]")

    # Generate samples
    samples = []
    if HAS_TOKENIZER:
        console.print(f"[cyan]Generating samples...[/cyan]")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        prompts = [
            "Once upon a time",
            "One day, a little girl",
            "There was a cat who",
            "In a big forest",
            "A brave knight"
        ]

        for prompt in prompts:
            text = generate_sample(model, tokenizer, prompt, device)
            samples.append({
                "prompt": prompt,
                "text": text
            })

    return {
        'method': method,
        'initial': {
            'cond': initial_cond,
            'val_ppl': initial_val_ppl,
            'train_ppl': initial_train_ppl,
            'val_loss': initial_val_loss,
            'train_loss': initial_train_loss
        },
        'final': {
            'cond': final_cond,
            'val_ppl': final_val_ppl,
            'train_ppl': final_train_ppl,
            'val_loss': final_val_loss,
            'train_loss': final_train_loss
        },
        'trajectory': trajectory,
        'layer_conditions': final_layer_conds,
        'checkpoint_path': str(checkpoint_path) if checkpoint_path else None,
        'samples': samples
    }


def main():
    parser = argparse.ArgumentParser(description="Geometry retrofit experiment")
    parser.add_argument("--model_path", type=str, default="experiments/hf_pytorch_model.bin",
                       help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="hf",
                       choices=["hf", "custom"],
                       help="Model type (hf=HuggingFace baseline, custom=your trained model)")
    parser.add_argument("--method", type=str, default="geometry",
                       choices=["standard", "gram", "geometry"],
                       help="Training method")
    parser.add_argument("--num_steps", type=int, default=1000,
                       help="Number of training steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--gram_weight", type=float, default=0.01,
                       help="Gram regularization weight")
    parser.add_argument("--glt_interval", type=int, default=50,
                       help="GLT transport interval")
    parser.add_argument("--eval_interval", type=int, default=100,
                       help="Evaluation interval")
    parser.add_argument("--max_docs", type=int, default=1000,
                       help="Max documents for training")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file")

    args = parser.parse_args()

    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]Geometry Retrofit Experiment[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

    # Set device
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    console.print(f"[cyan]Device: {device}[/cyan]")

    # Load model
    console.print(f"[cyan]Loading {args.model_type} model from {args.model_path}...[/cyan]")

    if args.model_type == "hf":
        # Load HF baseline
        config_path = Path(args.model_path).parent / "hf_config.json"
        with open(config_path) as f:
            config_dict = json.load(f)

        config = GPTConfig(
            block_size=config_dict["block_size"],
            vocab_size=config_dict["vocab_size"],
            n_layer=config_dict["n_layer"],
            n_head=config_dict["n_head"],
            n_embd=config_dict["n_embd"],
            dropout=0.0,  # No dropout for evaluation
            bias=config_dict["bias"]
        )

        model = GPT(config)

        try:
            state_dict = torch.load(args.model_path, map_location='cpu', weights_only=True)
        except:
            state_dict = torch.load(args.model_path, map_location='cpu')

        model.load_state_dict(state_dict)

    else:
        # Load your custom model
        console.print("[yellow]Custom model loading not yet implemented[/yellow]")
        console.print("[yellow]Add your model loading code here[/yellow]")
        return

    model = model.to(device)
    console.print(f"[green]Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters[/green]")

    # Prepare data
    block_size = config.block_size
    # Train on first max_docs from train split
    train_loader = prepare_data(args.max_docs, block_size, args.batch_size, split="train", offset=0)
    # Validate on DIFFERENT docs from train split (or use validation split)
    # Using validation split to ensure no overlap
    val_loader = prepare_data(max_docs=1100, block_size=block_size, batch_size=args.batch_size, split="validation", offset=0)

    # Run experiment
    output_dir = Path(args.output).parent if args.output else Path("artifacts_ts/retrofit_comparison")
    output_dir.mkdir(exist_ok=True, parents=True)

    results = retrofit_experiment(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_steps=args.num_steps,
        method=args.method,
        gram_weight=args.gram_weight,
        glt_interval=args.glt_interval,
        eval_interval=args.eval_interval,
        lr=args.lr,
        save_checkpoint=True,
        output_dir=output_dir,
    )

    # Save results
    output_path = args.output or f"artifacts_ts/retrofit_{args.method}_{args.num_steps}steps.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"\n[green]Results saved to: {output_path}[/green]")


if __name__ == "__main__":
    main()
