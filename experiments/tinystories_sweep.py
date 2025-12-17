# sweep_tinystories.py
import os
import json
import time
import argparse
import itertools
import subprocess
import shlex
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

console = Console()


def run_one(cmd, env=None):
    console.print(f"[cyan]>> {cmd}[/cyan]")
    r = subprocess.run(shlex.split(cmd), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, env=env)
    console.print(r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", type=str,
                    default="experiments/tinystories_gpt_geom.py")
    ap.add_argument("--out_dir", type=str, default="artifacts_ts")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=1536)
    ap.add_argument("--dropout", type=float, default=0.0)
    # auto / cuda / mps / cpu
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--char_level", action="store_true")
    ap.add_argument("--max_docs", type=int, default=None)

    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--lrs", type=str, default="3e-4,5e-4")
    ap.add_argument("--expset", type=str,
                    default="base,ffn,ffn_glt,ffn_glt_head,ffn_glt_head_white")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d-%H%M%S")
    agg_csv = os.path.join(args.out_dir, f"sweep_{tag}.csv")
    with open(agg_csv, "w") as f:
        f.write("exp,seed,lr,final_val_ppl,best_val_ppl,best_epoch,final_train_ppl,ff_cond,head_cond,headW_cond,run_tag\n")

    # define experiment recipes
    RECIPES = {
        "base": [],
        "ffn": ["--ff_feat_lambda 1e-5 --ff_layers -1 --head_norm"],
        "ffn_glt": ["--ff_feat_lambda 1e-5 --ff_layers -1 --head_norm", "--glt_freq 50 --glt_types ffn --glt_burnin 200"],
        "ffn_glt_head": ["--ff_feat_lambda 1e-5 --ff_layers -1 --head_norm",
                         "--head_feat_lambda 1e-5 --head_layers -1 --head_norm",
                         "--glt_freq 50 --glt_types ffn,attn --glt_burnin 200"],
        "ffn_glt_head_white": ["--ff_feat_lambda 1e-5 --ff_layers -1 --head_norm",
                               "--head_feat_lambda 1e-5 --head_layers -1 --head_norm",
                               "--lm_head_white_lambda 1e-5 --lm_head_subset 1024 --ema_momentum 0.99",
                               "--glt_freq 50 --glt_types ffn,attn --glt_burnin 200"],
    }

    seeds = [int(x) for x in args.seeds.split(",") if x]
    lrs = [float(x) for x in args.lrs.split(",") if x]
    exps = [x for x in args.expset.split(",") if x]

    base_common = f"--epochs {args.epochs} --batch {args.batch} --block_size {args.block_size} " \
                  f"--d_model {args.d_model} --n_layer {args.n_layer} --n_head {args.n_head} --d_ff {args.d_ff} " \
                  f"--dropout {args.dropout} --out_dir {args.out_dir} --save_json --save_csv --max_docs {args.max_docs}"
    if args.char_level:
        base_common += " --char_level"
    if args.device != "auto":
        base_common += f" --device {args.device}"

    jobs = []
    for exp, seed, lr in itertools.product(exps, seeds, lrs):
        run_name = f"{exp}_s{seed}_lr{lr:.0e}"
        exp_args = " ".join(RECIPES[exp])
        cmd = f"python {args.script} {base_common} --seed {seed} --lr {lr} --run_name {run_name} {exp_args}"
        jobs.append((exp, seed, lr, cmd, run_name))

    # Print sweep configuration
    console.print("\n" + "="*70)
    console.print("[bold cyan]SWEEP CONFIGURATION[/bold cyan]")
    console.print("="*70)

    console.print(f"\n[bold green]Script:[/bold green]")
    console.print(f"  Target:          {args.script}")
    console.print(f"  Output dir:      {args.out_dir}")

    console.print(f"\n[bold green]Model Config:[/bold green]")
    console.print(f"  d_model:         {args.d_model}")
    console.print(f"  n_layers:        {args.n_layer}")
    console.print(f"  n_heads:         {args.n_head}")
    console.print(f"  d_ff:            {args.d_ff}")
    console.print(f"  block_size:      {args.block_size}")
    console.print(f"  dropout:         {args.dropout}")
    console.print(f"  char_level:      {args.char_level}")

    console.print(f"\n[bold green]Training Config:[/bold green]")
    console.print(f"  Epochs:          {args.epochs}")
    console.print(f"  Batch size:      {args.batch}")
    console.print(f"  Max docs:        {args.max_docs}")
    console.print(f"  Device:          {args.device}")

    console.print(f"\n[bold green]Sweep Parameters:[/bold green]")
    console.print(f"  Experiments:     {', '.join(exps)}")
    console.print(f"  Seeds:           {', '.join(map(str, seeds))}")
    console.print(f"  Learning rates:  {', '.join([f'{lr:.0e}' for lr in lrs])}")
    console.print(f"  [bold yellow]Total jobs:      {len(jobs)}[/bold yellow]")

    console.print(f"\n[bold green]Experiments:[/bold green]")
    for exp in exps:
        recipe_str = " ".join(RECIPES[exp]) if RECIPES[exp] else "(baseline)"
        console.print(f"  {exp:20s} {recipe_str}")

    console.print("\n" + "="*70 + "\n")

    # run jobs sequentially (simple & portable)
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TextColumn(
                      "[cyan]{task.completed}/{task.total}"), TextColumn("•"),
                  TextColumn("[yellow]{task.fields[current_job]}"),
                  TextColumn("[dim]{task.fields[last_result]}"),
                  TimeElapsedColumn(), transient=False, console=console) as prog:
        task = prog.add_task("Sweep", total=len(jobs), current_job="", last_result="")
        for i, (exp, seed, lr, cmd, run_name) in enumerate(jobs, 1):
            prog.update(task, current_job=f"{exp} s={seed} lr={lr:.0e}")
            console.print(f"\n[bold cyan]Job {i}/{len(jobs)}: {run_name}[/bold cyan]")
            run_one(cmd)
            # load JSON
            js_path = os.path.join(args.out_dir, f"{run_name}.json")
            with open(js_path) as f:
                js = json.load(f)
            summ = js["summary"]
            # Update with results
            result_str = f"✓ val_ppl={summ['best_val_ppl']:.3f}"
            prog.update(task, last_result=result_str)
            # take last epoch metrics for geometry (or the best epoch if missing)
            last = js["epochs"][-1]
            ff = last.get("ff_cond", None)
            hd = last.get("head_cond", None)
            hW = last.get("headW_cond", None)
            with open(agg_csv, "a") as f:
                f.write(f"{exp},{seed},{lr:.6e},{summ['final_val_ppl']:.6f},{summ['best_val_ppl']:.6f},"
                        f"{summ['best_epoch']},{summ['final_train_ppl']:.6f},{ff if ff is not None else ''},"
                        f"{hd if hd is not None else ''},{hW if hW is not None else ''},{summ['run_tag']}\n")
            prog.update(task, advance=1)

    # print a quick aggregate table
    import csv
    import statistics as stats
    rows = []
    with open(agg_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    # group by exp
    groups = {}
    for r in rows:
        k = (r["exp"], r["lr"])
        groups.setdefault(k, []).append(float(r["best_val_ppl"]))
    tbl = Table(show_header=True, header_style="bold magenta",
                title="Sweep: best Val PPL (↓) across seeds")
    for c in ["exp", "lr", "mean", "std", "n"]:
        tbl.add_column(c, justify="right")
    for (exp, lr), vals in sorted(groups.items(), key=lambda kv: (kv[0][0], float(kv[0][1]))):
        m = stats.mean(vals)
        s = stats.pstdev(vals) if len(vals) > 1 else 0.0
        tbl.add_row(exp, f"{float(lr):.2e}", f"{m:.3f}",
                    f"{s:.3f}", f"{len(vals)}")
    console.print(tbl)
    console.print(f"[green]Saved aggregate CSV to {agg_csv}[/green]")


if __name__ == "__main__":
    main()
