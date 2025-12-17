#!/usr/bin/env python3
"""
TinyStories GPT-4 Evaluation Script

Implements the evaluation methodology from the TinyStories paper:
- Generate completions from story beginnings
- Grade each completion on Grammar, Creativity, and Consistency using GPT-4
- Average scores across 50 test prompts

Based on: "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"
by Ronen Eldan and Yuanzhi Li (arXiv:2305.07759)
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

# Check for OpenAI
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    console.print("[yellow]openai not installed. Install with: pip install openai[/yellow]")

# Check for datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    console.print("[yellow]datasets not installed. Install with: pip install datasets[/yellow]")


# GPT-4 Evaluation Prompt (reconstructed from paper description)
EVALUATION_PROMPT_TEMPLATE = """You are evaluating a story completion written by a language model.

The student was given the beginning of a story and asked to complete it.
You need to grade the completion on three dimensions:

1. **Grammar** (0-10): Is the language grammatically correct? Are sentences well-formed?
2. **Creativity** (0-10): Is the story creative and interesting? Does it show imagination?
3. **Consistency** (0-10): Is the completion consistent with the story beginning? Do the characters, settings, and plot elements align?

Here is the story:

**Story Beginning:**
{beginning}

**Student's Completion:**
{completion}

Please provide your evaluation in the following format:
Grammar: X/10
Creativity: X/10
Consistency: X/10

Be objective and consistent in your grading. Consider what a typical 3-4 year old would understand and enjoy.
"""


def load_evaluation_prompts(num_prompts=50):
    """
    Load story beginnings for evaluation.

    Uses validation set stories split in half (first half as prompt).
    """
    if not HAS_DATASETS:
        raise RuntimeError("datasets library required")

    console.print(f"[cyan]Loading {num_prompts} evaluation prompts from TinyStories validation...[/cyan]")

    ds = load_dataset("roneneldan/TinyStories", split="validation")

    # Take first num_prompts stories
    stories = ds["text"][:num_prompts * 2]  # Get extra in case some are too short

    prompts = []
    for story in stories:
        if len(story) < 100:  # Skip very short stories
            continue

        # Split story roughly in half
        words = story.split()
        if len(words) < 20:
            continue

        split_point = len(words) // 2
        beginning = " ".join(words[:split_point])
        reference_ending = " ".join(words[split_point:])

        prompts.append({
            "beginning": beginning,
            "reference_ending": reference_ending,
            "full_story": story
        })

        if len(prompts) >= num_prompts:
            break

    console.print(f"[green]Loaded {len(prompts)} evaluation prompts[/green]")
    return prompts


def generate_completion(model, tokenizer, prompt_text, device, max_length=150, temperature=0.8, top_k=50):
    """Generate completion from model."""
    model.eval()

    with torch.no_grad():
        # Encode prompt
        if hasattr(tokenizer, 'encode'):
            tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            # Character-level
            tokens = [tokenizer['stoi'][ch] for ch in prompt_text if ch in tokenizer['stoi']]

        idx = torch.tensor([tokens], dtype=torch.long, device=device)

        # Generate
        if hasattr(model, 'generate'):
            generated = model.generate(idx, max_new_tokens=max_length, temperature=temperature, top_k=top_k)
        else:
            # Fallback autoregressive generation
            for _ in range(max_length):
                idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
                logits, _ = model(idx_cond)
                logits = logits[:, -1, :] / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            generated = idx

        # Decode
        if hasattr(tokenizer, 'decode'):
            completion = tokenizer.decode(generated[0].tolist())
        else:
            # Character-level
            completion = "".join([tokenizer['itos'][i] for i in generated[0].tolist() if i in tokenizer['itos']])

        # Extract just the new completion (remove prompt)
        if completion.startswith(prompt_text):
            completion = completion[len(prompt_text):]

        return completion.strip()


def evaluate_with_gpt4(beginning, completion, api_key=None, model="gpt-4"):
    """
    Evaluate a story completion using GPT-4.

    Returns dict with grammar, creativity, consistency scores.
    """
    if not HAS_OPENAI:
        raise RuntimeError("openai library required. Install with: pip install openai")

    # Set API key
    if api_key:
        openai.api_key = api_key
    elif "OPENAI_API_KEY" in os.environ:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass --api_key")

    # Create evaluation prompt
    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        beginning=beginning,
        completion=completion
    )

    # Call GPT-4
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert teacher evaluating children's story writing."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,  # Deterministic for consistency
            max_tokens=200
        )

        evaluation_text = response.choices[0].message.content

        # Parse scores
        scores = {}
        for line in evaluation_text.split('\n'):
            line = line.strip()
            if line.startswith("Grammar:"):
                scores['grammar'] = float(line.split('/')[0].split(':')[1].strip())
            elif line.startswith("Creativity:"):
                scores['creativity'] = float(line.split('/')[0].split(':')[1].strip())
            elif line.startswith("Consistency:"):
                scores['consistency'] = float(line.split('/')[0].split(':')[1].strip())

        if len(scores) != 3:
            console.print(f"[yellow]Warning: Could not parse all scores from: {evaluation_text}[/yellow]")

        return scores

    except Exception as e:
        console.print(f"[red]Error calling GPT-4: {e}[/red]")
        return {"grammar": 0, "creativity": 0, "consistency": 0}


def evaluate_model(model, tokenizer, device, num_prompts=50, api_key=None, gpt_model="gpt-4"):
    """
    Run full evaluation on a model.

    Returns dict with average scores and all individual evaluations.
    """
    # Load evaluation prompts
    prompts = load_evaluation_prompts(num_prompts)

    console.print(f"\n[cyan]Generating {len(prompts)} completions...[/cyan]")

    all_scores = []

    for i, prompt_data in enumerate(track(prompts, description="Evaluating")):
        # Generate completion
        completion = generate_completion(
            model, tokenizer, prompt_data["beginning"], device,
            max_length=150, temperature=0.8, top_k=50
        )

        # Evaluate with GPT-4
        scores = evaluate_with_gpt4(
            prompt_data["beginning"],
            completion,
            api_key=api_key,
            model=gpt_model
        )

        all_scores.append({
            "prompt": prompt_data["beginning"],
            "completion": completion,
            "reference": prompt_data["reference_ending"],
            "scores": scores
        })

        # Show progress every 10
        if (i + 1) % 10 == 0:
            avg_grammar = sum(s["scores"].get("grammar", 0) for s in all_scores) / len(all_scores)
            avg_creativity = sum(s["scores"].get("creativity", 0) for s in all_scores) / len(all_scores)
            avg_consistency = sum(s["scores"].get("consistency", 0) for s in all_scores) / len(all_scores)
            console.print(f"  [{i+1}/{len(prompts)}] Current averages: G={avg_grammar:.1f}, C={avg_creativity:.1f}, Con={avg_consistency:.1f}")

    # Compute averages
    avg_scores = {
        "grammar": sum(s["scores"].get("grammar", 0) for s in all_scores) / len(all_scores),
        "creativity": sum(s["scores"].get("creativity", 0) for s in all_scores) / len(all_scores),
        "consistency": sum(s["scores"].get("consistency", 0) for s in all_scores) / len(all_scores),
    }

    return {
        "average_scores": avg_scores,
        "all_evaluations": all_scores,
        "num_evaluated": len(all_scores)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate TinyStories models using GPT-4")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--num_prompts", type=int, default=50,
                       help="Number of prompts to evaluate (default: 50)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--gpt_model", type=str, default="gpt-4",
                       help="GPT model to use for evaluation (gpt-4, gpt-4-turbo, etc.)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--output", type=str, default=None,
                       help="Output JSON file for results")

    args = parser.parse_args()

    console.print(f"\n[bold cyan]{'='*70}[/bold cyan]")
    console.print(f"[bold cyan]TinyStories GPT-4 Evaluation[/bold cyan]")
    console.print(f"[bold cyan]{'='*70}[/bold cyan]\n")

    # TODO: Load your model here
    console.print("[yellow]Model loading not yet implemented[/yellow]")
    console.print("[yellow]You need to add code to load your specific model architecture[/yellow]")
    console.print("[yellow]See evaluate_model() function for the evaluation logic[/yellow]")

    # Example usage:
    # model, tokenizer = load_your_model(args.model_path)
    # device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    # model = model.to(device)
    #
    # results = evaluate_model(model, tokenizer, device, args.num_prompts, args.api_key, args.gpt_model)
    #
    # # Print results
    # console.print(f"\n[bold green]Results:[/bold green]")
    # console.print(f"  Grammar:     {results['average_scores']['grammar']:.1f}/10")
    # console.print(f"  Creativity:  {results['average_scores']['creativity']:.1f}/10")
    # console.print(f"  Consistency: {results['average_scores']['consistency']:.1f}/10")
    #
    # # Save results
    # if args.output:
    #     with open(args.output, 'w') as f:
    #         json.dump(results, f, indent=2)
    #     console.print(f"\n[green]Results saved to: {args.output}[/green]")


if __name__ == "__main__":
    main()
