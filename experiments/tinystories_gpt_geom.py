# tinystories_gpt_geom.py
import math
import os
import time
import random
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

# Optional tokenizer (falls back to char-level if unavailable or --char_level)
try:
    import tiktoken
    _tiktoken_available = True
    print("Import Succeeded")
except ImportError:
    _tiktoken_available = False
    print("Import Failed")

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

console = Console()

# -------------------- utils: seeds & slogdet (MPS-safe) --------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def slogdet_safe(M: torch.Tensor, eye_eps: torch.Tensor) -> torch.Tensor:
    """
    Differentiable, MPS-safe logdet for (near-)SPD matrices.
    Returns logdet(M + eye_eps) using Cholesky when possible.
    Falls back to slogdet and then to extra jitter if needed.
    """
    A = M + eye_eps  # always add jitter; ensures SPD in your use-cases
    try:
        L = torch.linalg.cholesky(A)
        return 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
    except RuntimeError:
        # Fallback: slogdet (still differentiable on most backends)
        sign, logdet = torch.linalg.slogdet(A)
        if (sign <= 0).any():
            # last-ditch: increase jitter and retry cholesky
            eps = eye_eps.diagonal(
                dim1=-2, dim2=-1).mean() if eye_eps.ndim >= 2 else eye_eps
            jitter = (10.0 * eps) * \
                torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
            L = torch.linalg.cholesky(A + jitter)
            return 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
        return logdet


# -------------------- geometry losses ------------------------------


def gram_logdet_BTCxC(x, lam, eps=1e-4, normalize=True):
    if lam <= 0.0:
        return x.new_zeros(())

    # keep gradients, but compute logdet in fp32
    x32 = x.float()

    if x32.dim() == 3:
        B, T, C = x32.shape
        X = x32.reshape(B*T, C).transpose(0, 1)   # [C, BT]
    else:
        N, C = x32.shape
        X = x32.transpose(0, 1)                   # [C, N]

    C = X.shape[0]
    I = torch.eye(C, device=x.device, dtype=torch.float32)
    G = (X @ X.t()) / float(X.shape[1]) + eps * I

    lg = slogdet_safe(G, eps * I)  # now fp32-safe

    if normalize:
        tr = torch.trace(G)
        val = -lam * (lg - C * torch.log(tr + 1e-12))
    else:
        val = -lam * lg

    # val is fp32 scalar; safe to return
    return val


class RunningSecondMoment:
    def __init__(self, dim, momentum=0.99, device="cpu"):
        self.S = torch.eye(dim, device=device)
        self.m = float(momentum)
        self.init = False

    @torch.no_grad()
    def update(self, H):  # H: [N, d] or [B,T,d]
        if H.dim() == 3:
            H = H.reshape(-1, H.shape[-1])
        Sb = (H.t() @ H) / float(H.shape[0])
        if not self.init:
            self.S.copy_(Sb)
            self.init = True
        else:
            self.S.mul_(self.m).add_(Sb, alpha=1.0-self.m)

    def get(self): return self.S


def whitened_head_logdet(V_sub, S, lam, eps=1e-4, normalize=True):
    if lam <= 0.0:
        return V_sub.new_zeros(())
    d = S.shape[0]
    I_d = eps * torch.eye(d, device=S.device, dtype=S.dtype)
    A = V_sub @ (S + I_d) @ V_sub.t()
    K = A.shape[0]
    I = eps * torch.eye(K, device=A.device, dtype=A.dtype)
    lg = slogdet_safe(A, I)
    if normalize:
        tr = torch.trace(A)
        return -lam * (lg - K * torch.log(tr + 1e-12))
    else:
        return -lam * lg

# -------------------- GLT with optimizer-state transport -------------------


def _permute_param_and_state_(param, dim, index, opt):
    with torch.no_grad():
        new = param.data.index_select(dim, index)
        param.data.copy_(new)
        if opt is None:
            return
        st = opt.state.get(param, None)
        if st is None:
            return
        for k in ("exp_avg", "exp_avg_sq", "momentum_buffer"):
            if k in st and isinstance(st[k], torch.Tensor) and st[k].shape == param.data.shape:
                st[k].data.copy_(st[k].data.index_select(dim, index))


# def glt_permute_ffn(block, opt):
#     # W1: [d_ff, d_model], b1: [d_ff]; W2: [d_model, d_ff], b2: [d_model]
#     dff = block.ffn_w1.weight.shape[0]
#     device = block.ffn_w1.weight.device
#     perm = torch.randperm(dff, device=device)
#     inv = torch.argsort(perm)
#     # permute hidden rows of W1 and b1
#     _permute_param_and_state_(block.ffn_w1.weight, 0, perm, opt)
#     if block.ffn_w1.bias is not None:
#         _permute_param_and_state_(block.ffn_w1.bias, 0, perm, opt)
#     # permute hidden columns of W2 by inverse permutation
#     _permute_param_and_state_(block.ffn_w2.weight, 1, inv, opt)

def glt_permute_ffn(block, opt):
    dff = block.ffn_w1.weight.shape[0]
    device = block.ffn_w1.weight.device
    perm = torch.randperm(dff, device=device)

    _permute_param_and_state_(block.ffn_w1.weight, 0, perm, opt)
    if block.ffn_w1.bias is not None:
        _permute_param_and_state_(block.ffn_w1.bias, 0, perm, opt)

    # W2 columns should be permuted the SAME way
    _permute_param_and_state_(block.ffn_w2.weight, 1, perm, opt)


# def glt_permute_heads(attn, opt):
#     # Wq,Wk,Wv: [H*Dh, d_model]; Wo: [d_model, H*Dh]
#     H, Dh = attn.n_head, attn.head_dim
#     device = attn.Wq.weight.device
#     perm = torch.randperm(H, device=device)
#     inv = torch.argsort(perm)
#     # build row/col indices grouped by head

#     def head_rows(p):  # permutation of rows for [H*Dh, d]
#         return torch.cat([torch.arange(h*Dh, (h+1)*Dh, device=device) for h in p], dim=0)
#     row_idx = head_rows(perm)
#     row_inv = head_rows(inv)
#     # Wq, Wk, Wv: permute rows by perm
#     _permute_param_and_state_(attn.Wq.weight, 0, row_idx, opt)
#     _permute_param_and_state_(attn.Wk.weight, 0, row_idx, opt)
#     _permute_param_and_state_(attn.Wv.weight, 0, row_idx, opt)
#     # Wo: permute columns by inverse (to preserve function)
#     _permute_param_and_state_(attn.Wo.weight, 1, row_inv, opt)

def glt_permute_heads(attn, opt):
    H, Dh = attn.n_head, attn.head_dim
    device = attn.Wq.weight.device
    perm = torch.randperm(H, device=device)

    def head_rows(p):
        return torch.cat([torch.arange(h*Dh, (h+1)*Dh, device=device) for h in p], dim=0)

    row_idx = head_rows(perm)

    _permute_param_and_state_(attn.Wq.weight, 0, row_idx, opt)
    _permute_param_and_state_(attn.Wk.weight, 0, row_idx, opt)
    _permute_param_and_state_(attn.Wv.weight, 0, row_idx, opt)

    # Wo columns should be permuted the SAME way
    _permute_param_and_state_(attn.Wo.weight, 1, row_idx, opt)


# -------------------- tiny GPT ------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1536
    dropout: float = 0.0


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_head
        assert self.d_model % self.n_head == 0

        self.Wq = nn.Linear(cfg.d_model, cfg.n_head *
                            self.head_dim, bias=False)
        self.Wk = nn.Linear(cfg.d_model, cfg.n_head *
                            self.head_dim, bias=False)
        self.Wv = nn.Linear(cfg.d_model, cfg.n_head *
                            self.head_dim, bias=False)
        self.Wo = nn.Linear(cfg.n_head * self.head_dim,
                            cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask=None, need_head_out=False):
        B, T, _ = x.shape
        H, Dh = self.n_head, self.head_dim

        # Project and reshape to [B, H, T, Dh]
        q = self.Wq(x).view(B, T, H, Dh).transpose(1, 2)
        k = self.Wk(x).view(B, T, H, Dh).transpose(1, 2)
        v = self.Wv(x).view(B, T, H, Dh).transpose(1, 2)

        # PyTorch will choose Flash/mem-efficient kernels on CUDA when possible.
        # Use causal=True for GPT-style autoregressive masking.
        dropout_p = self.drop.p if self.training else 0.0

        ctx = F.scaled_dot_product_attention(
            q, k, v,
            # should be broadcastable to [B,H,T,T] if provided
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=True
        )  # [B, H, T, Dh]

        out = self.Wo(ctx.transpose(1, 2).contiguous().view(B, T, H * Dh))

        if need_head_out:
            return out, ctx.transpose(1, 2)  # [B, T, H, Dh]
        return out, None


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.ffn_w1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.ffn_w2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, attn_mask=None, collect=False):
        a_in = self.ln1(x)
        a_out, head_out = self.attn(
            a_in, attn_mask=attn_mask, need_head_out=collect)
        x = x + self.drop(a_out)
        f_in = self.ln2(x)
        h = F.gelu(self.ffn_w1(f_in))
        x = x + self.drop(self.ffn_w2(h))
        return x, h if collect else None, head_out


class GPTSmall(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model, eps=1e-5)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # weight tying
        self.apply(self._init)
        self.lm_head.weight = self.tok_emb.weight

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, collect_layers=(-1,), collect_heads=True):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]
        ffn_feats = []
        head_feats = []
        for li, block in enumerate(self.blocks):
            collect = (li in collect_layers)
            x, h, head_out = block(x, attn_mask=None, collect=collect)
            if h is not None:
                ffn_feats.append(h)
            if head_out is not None and collect_heads:
                head_feats.append(head_out)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B,T,V]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                   targets.reshape(-1), reduction='mean')
        return logits, loss, ffn_feats, head_feats, x

# -------------------- data -----------------------------------------


def build_tokenizer(args):
    if args.char_level or not _tiktoken_available:
        return None
    return tiktoken.get_encoding("gpt2")  # Same vocab as GPT-2


def prepare_dataset(args, tokenizer, device):
    if tokenizer is None and load_dataset is None:
        raise RuntimeError(
            "datasets not available; use --char_level or install `datasets`")
    ds = load_dataset(
        "roneneldan/TinyStories") if load_dataset is not None else None
    text = (ds["train"]["text"][:args.max_docs] if (ds is not None and args.max_docs is not None and args.max_docs > 0)
            else (ds["train"]["text"] if ds is not None else []))
    if tokenizer is None:
        # char-level
        if ds is None:
            raise RuntimeError(
                "char-level requires datasets to fetch TinyStories; alternatively replace with your own text")
        vocab = sorted(list(set("".join(text))))
        stoi = {ch: i for i, ch in enumerate(vocab)}
        # For decoding during generation
        itos = {i: ch for ch, i in stoi.items()}
        def encode(s): return [stoi[c] for c in s]
        ids = [torch.tensor(encode(s), dtype=torch.long)
               for s in text if len(s) > 0]
        vocab_size = len(vocab)
        vocab_info = {'stoi': stoi, 'itos': itos, 'vocab': vocab}
    else:
        def enc(s): return tokenizer.encode(s)
        eos_id = 50256
        ids = [torch.tensor(enc(s) + [eos_id], dtype=torch.long)
               for s in text if len(s) > 0]
        vocab_size = int(tokenizer.n_vocab)
        vocab_info = None

    stream = torch.cat([x for x in ids if len(x) > 0]) if len(
        ids) > 0 else torch.randint(0, 128, (args.block_size*1000,))
    n = int(0.9*len(stream))
    train_stream, val_stream = stream[:n], stream[n:]

    def make_loader(stream):
        toks = stream.unfold(0, args.block_size+1,
                             args.block_size+1)  # [N, T+1]

        class DS(torch.utils.data.Dataset):
            def __len__(self): return toks.size(0)

            def __getitem__(self, i):
                seq = toks[i]
                return seq[:-1], seq[1:]
        return DataLoader(
            DS(),
            batch_size=args.batch,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=(device.type == "cuda"),
            persistent_workers=True,
        )

    return make_loader(train_stream), make_loader(val_stream), vocab_size, vocab_info


@torch.no_grad()
def generate_samples(model, tokenizer, vocab_info, device, epoch, out_dir, run_tag,
                     num_samples=3, max_length=200, temperature=0.8, top_k=50):
    model.eval()
    eos_id = 50256  # GPT-2 endoftext (tiktoken)

    prompts = ["Once upon a time", "One day, a little girl",
               "There was a cat who"][:num_samples]
    samples = []

    for prompt in prompts:
        if tokenizer is not None:
            tokens = tokenizer.encode(prompt)
            context = torch.tensor([tokens], dtype=torch.long, device=device)
        else:
            stoi = vocab_info["stoi"]
            tokens = [stoi.get(c, 0) for c in prompt]
            context = torch.tensor([tokens], dtype=torch.long, device=device)

        for _ in range(max_length):
            context_crop = context[:, -model.cfg.block_size:]
            logits, _, _, _, _ = model(
                context_crop, collect_layers=set(), collect_heads=False)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat([context, next_token], dim=1)

            if tokenizer is not None and next_token.item() == eos_id:
                break

        if tokenizer is not None:
            text = tokenizer.decode(context[0].tolist())
        else:
            itos = vocab_info["itos"]
            text = "".join([itos.get(i, "?") for i in context[0].tolist()])

        samples.append((prompt, text))

    console.print(
        f"\n[bold cyan]═══ Generated Samples (Epoch {epoch}) ═══[/bold cyan]")
    for i, (prompt, text) in enumerate(samples, 1):
        console.print(f"\n[yellow]Sample {i}:[/yellow]")
        display_text = text[:500] + "..." if len(text) > 500 else text
        console.print(f"[dim]{display_text}[/dim]")
    console.print("[bold cyan]" + "="*50 + "[/bold cyan]\n")

    if out_dir:
        sample_file = os.path.join(
            out_dir, f"{run_tag}_samples_epoch{epoch}.txt")
        with open(sample_file, "w") as f:
            f.write(f"Generated Samples - Epoch {epoch}\n")
            f.write("="*70 + "\n\n")
            for i, (prompt, text) in enumerate(samples, 1):
                f.write(f"Sample {i}:\nPrompt: {prompt}\n{text}\n\n")
                f.write("-"*70 + "\n\n")

    model.train()


# -------------------- train ----------------------------------------


def run_validation(model, val_loader, device, cfg, ff_layers, head_layers, args):
    """Run validation and return metrics."""
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        val_tok = 0
        ff_cond = np.nan
        head_cond = np.nan
        headW_cond = np.nan

        # Get S_ema from model if available (for head whitening)
        S_ema = getattr(model, '_S_ema', None)

        for i, (xb, yb) in enumerate(val_loader):
            xb, yb = xb.to(device), yb.to(device)
            need_ff = args.ff_feat_lambda > 0
            need_head = args.head_feat_lambda > 0
            collect_layers = set(
                ff_layers+head_layers) if (need_ff or need_head) else set()
            collect_heads = need_head

            logits, ce, ffn_feats, head_feats, Hfinal = model(
                xb, yb,
                collect_layers=collect_layers,
                collect_heads=collect_heads
            )
            val_loss += ce.item() * (xb.size(0)*(xb.size(1)-1))
            val_tok += xb.size(0)*(xb.size(1)-1)

            # Only compute condition numbers on first batch
            if i == 0:
                if len(ffn_feats) > 0:
                    h = ffn_feats[0].detach()
                    B, T, D = h.shape
                    X = h.reshape(B*T, D).T
                    G = (X @ X.t())/float(X.shape[1]) + 1e-5 * \
                        torch.eye(D, device=h.device, dtype=h.dtype)
                    ev = torch.linalg.eigvalsh(G.cpu())
                    ff_cond = float((ev.max()/ev.min()).item())

                if len(head_feats) > 0:
                    ho = head_feats[0].detach()
                    B, T, H, Dh = ho.shape
                    Z = ho.permute(0, 1, 3, 2).reshape(B*T*Dh, H).T
                    G = (Z @ Z.t())/float(Z.shape[1]) + 1e-5 * \
                        torch.eye(H, device=ho.device, dtype=ho.dtype)
                    ev = torch.linalg.eigvalsh(G.cpu())
                    head_cond = float((ev.max()/ev.min()).item())

                if args.lm_head_white_lambda > 0 and S_ema is not None:
                    Ksub = min(args.lm_head_subset, cfg.vocab_size)
                    sub_idx = torch.randperm(
                        cfg.vocab_size, device=device)[:Ksub]
                    V_sub = model.lm_head.weight[sub_idx]
                    S_eval = S_ema.get()
                    A = V_sub @ (S_eval + 1e-5*torch.eye(cfg.d_model,
                                 device=device)) @ V_sub.t()
                    ev = torch.linalg.eigvalsh(A.detach().cpu())
                    headW_cond = float((ev.max()/ev.min()).item())

    model.train()
    val_ppl = math.exp(val_loss/max(1, val_tok))
    return val_ppl, ff_cond, head_cond, headW_cond


def main():
    ap = argparse.ArgumentParser()
    # IO & reproducibility
    ap.add_argument("--out_dir", type=str, default="artifacts_ts")
    ap.add_argument("--run_name", type=str, default="")
    ap.add_argument("--save_csv", action="store_true")
    ap.add_argument("--save_json", action="store_true")
    ap.add_argument("--seed", type=int, default=123)

    # training / model
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--d_model", type=int, default=384)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--d_ff", type=int, default=1536)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available()
                    else ("mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--char_level", action="store_true")
    ap.add_argument("--max_docs", type=int, default=10000)   # limit for speed

    # geometry knobs
    ap.add_argument("--ff_feat_lambda", type=float, default=0.0)
    # comma sep (e.g., "4,5")
    ap.add_argument("--ff_layers", type=str, default="-1")
    ap.add_argument("--head_feat_lambda", type=float, default=0.0)
    ap.add_argument("--head_layers", type=str, default="-1")
    # normalized variant
    ap.add_argument("--head_norm", action="store_true")

    ap.add_argument("--lm_head_white_lambda", type=float, default=0.0)
    ap.add_argument("--lm_head_subset", type=int, default=1024)
    ap.add_argument("--ema_momentum", type=float, default=0.99)

    # GLT
    # every k steps (0=off)
    ap.add_argument("--glt_freq", type=int, default=0)
    ap.add_argument("--glt_prob", type=float, default=0.0)   # or probabilistic
    ap.add_argument("--glt_burnin", type=int, default=500)
    ap.add_argument("--glt_types", type=str,
                    default="ffn,attn")  # comma in {ffn,attn}

    # Validation frequency
    ap.add_argument("--val_freq", type=int, default=5000,
                    help="Run validation every N steps (default: 5000)")
    ap.add_argument("--amp", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)

    use_amp = args.amp and device.type == "cuda"
    amp_dtype = torch.bfloat16

    os.makedirs(args.out_dir, exist_ok=True)
    run_tag = args.run_name if args.run_name else time.strftime(
        "%Y%m%d-%H%M%S")

    tokenizer = build_tokenizer(args)
    train_loader, val_loader, vocab_size, vocab_info = prepare_dataset(
        args, tokenizer, device)

    cfg = GPTConfig(vocab_size=vocab_size, block_size=args.block_size,
                    n_layer=args.n_layer, n_head=args.n_head,
                    d_model=args.d_model, d_ff=args.d_ff, dropout=args.dropout)
    model = GPTSmall(cfg).to(device)

    decay, nodecay = set(), set()
    whitelist_modules = (nn.Linear,)
    blacklist_modules = (nn.LayerNorm, nn.Embedding)

    for mn, m in model.named_modules():
        for pn, p in m.named_parameters(recurse=False):
            fpn = f"{mn}.{pn}" if mn else pn
            if pn.endswith("bias") or isinstance(m, blacklist_modules):
                nodecay.add(fpn)
            elif isinstance(m, whitelist_modules):
                decay.add(fpn)

    # Get actual parameter names from model
    param_dict = {pn: p for pn, p in model.named_parameters()}

    # Tied weights: never decay (only add if they exist)
    if "tok_emb.weight" in param_dict:
        nodecay.add("tok_emb.weight")
    if "lm_head.weight" in param_dict:
        nodecay.add("lm_head.weight")

    # Only include parameters that actually exist in the model
    decay_params = [param_dict[pn]
                    for pn in sorted(list(decay)) if pn in param_dict]
    nodecay_params = [param_dict[pn]
                      for pn in sorted(list(nodecay)) if pn in param_dict]

    opt_groups = [
        {"params": decay_params, "weight_decay": args.wd},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(opt_groups, lr=args.lr, betas=(0.9, 0.95))
    # Calculate effective training steps with gradient accumulation
    effective_steps = (args.epochs * len(train_loader) +
                       args.grad_accum_steps - 1) // args.grad_accum_steps

    # Learning rate schedule with optional warmup
    if args.warmup_steps > 0 and args.warmup_steps < effective_steps:
        # Linear warmup followed by cosine decay
        warmup_sched = LinearLR(opt, start_factor=0.01,
                                total_iters=args.warmup_steps)
        cosine_sched = CosineAnnealingLR(
            opt, T_max=effective_steps - args.warmup_steps)
        sched = SequentialLR(opt, schedulers=[warmup_sched, cosine_sched], milestones=[
                             args.warmup_steps])
    else:
        # Just cosine decay (no warmup)
        sched = CosineAnnealingLR(opt, T_max=effective_steps)

    ff_layers = [int(x) if x != '-1' else (cfg.n_layer-1)
                 for x in args.ff_layers.split(",")]
    head_layers = [int(x) if x != '-1' else (cfg.n_layer-1)
                   for x in args.head_layers.split(",")]
    S_ema = RunningSecondMoment(
        cfg.d_model, momentum=args.ema_momentum, device=device)

    # Attach S_ema to model for validation function access
    model._S_ema = S_ema

    need_ff = args.ff_feat_lambda > 0
    need_head = args.head_feat_lambda > 0
    collect_layers = set(
        ff_layers + head_layers) if (need_ff or need_head) else set()
    collect_heads = need_head

    # ==================== Pre-training Summary ====================
    console.print("\n[bold cyan]═══ TinyStories GPT Training ═══[/bold cyan]")

    # Model architecture info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    console.print(f"[bold green]Model:[/bold green]")
    console.print(f"  Vocab size:      {vocab_size:,}")
    console.print(f"  d_model:         {cfg.d_model}")
    console.print(f"  n_layers:        {cfg.n_layer}")
    console.print(f"  n_heads:         {cfg.n_head}")
    console.print(f"  d_ff:            {cfg.d_ff}")
    console.print(f"  block_size:      {cfg.block_size}")
    console.print(f"  dropout:         {cfg.dropout}")
    console.print(
        f"  [bold yellow]Total params:    {total_params:,}[/bold yellow]")
    console.print(f"  Trainable:       {trainable_params:,}")

    # Training config
    console.print(f"\n[bold green]Training:[/bold green]")
    console.print(f"  Epochs:          {args.epochs}")
    console.print(f"  Batch size:      {args.batch}")
    if args.grad_accum_steps > 1:
        console.print(f"  Grad accum:      {args.grad_accum_steps} steps")
        console.print(
            f"  Effective batch: {args.batch * args.grad_accum_steps}")
    console.print(f"  Learning rate:   {args.lr:.2e}")
    if args.warmup_steps > 0 and args.warmup_steps < effective_steps:
        console.print(f"  LR warmup:       {args.warmup_steps} steps")
    console.print(f"  Weight decay:    {args.wd}")
    console.print(f"  Device:          {device}")
    console.print(f"  Batches/epoch:   {len(train_loader)}")
    console.print(f"  Optimizer steps: {effective_steps:,}")
    console.print(f"  Val frequency:   every {args.val_freq} steps")

    # Geometry regularization
    if args.ff_feat_lambda > 0 or args.head_feat_lambda > 0 or args.lm_head_white_lambda > 0:
        console.print(f"\n[bold green]Geometry Regularization:[/bold green]")
        if args.ff_feat_lambda > 0:
            console.print(
                f"  FF feature λ:    {args.ff_feat_lambda} (layers: {args.ff_layers})")
        if args.head_feat_lambda > 0:
            console.print(
                f"  Head feature λ:  {args.head_feat_lambda} (layers: {args.head_layers})")
            console.print(f"  Head normalized: {args.head_norm}")
        if args.lm_head_white_lambda > 0:
            console.print(
                f"  LM head white λ: {args.lm_head_white_lambda} (subset: {args.lm_head_subset})")

    # GLT (gradient-less transport)
    if args.glt_freq > 0 or args.glt_prob > 0:
        console.print(f"\n[bold green]Gradient-Less Transport:[/bold green]")
        if args.glt_freq > 0:
            console.print(f"  Frequency:       every {args.glt_freq} steps")
        if args.glt_prob > 0:
            console.print(f"  Probability:     {args.glt_prob:.2%}")
        console.print(f"  Burn-in:         {args.glt_burnin} steps")
        console.print(f"  Types:           {args.glt_types}")

    console.print(f"\n[bold cyan]{'═'*50}[/bold cyan]\n")

    # pretty header
    tbl = Table(show_header=True, header_style="bold magenta")
    for c in ["Epoch", "Train ppl", "Val ppl", "FF Gram cond", "Head Gram cond", "HeadW cond", "LR", "Time"]:
        tbl.add_column(c, justify="right")

    # optional CSV
    csv_path = os.path.join(args.out_dir, f"{run_tag}.csv")
    csv_step_path = os.path.join(args.out_dir, f"{run_tag}_step_val.csv")
    if args.save_csv:
        with open(csv_path, "w") as f:
            f.write(
                "epoch,train_ppl,val_ppl,ff_cond,head_cond,headW_cond,lr,epoch_time\n")
        with open(csv_step_path, "w") as f:
            f.write("step,epoch,val_ppl,ff_cond,head_cond,headW_cond,lr\n")

    # metrics container for JSON
    metrics = {"args": vars(args), "epochs": [], "step_validations": []}

    global_step = 0
    last_val_step = 0  # First validation will happen at step val_freq
    for ep in range(1, args.epochs+1):
        t0 = time.time()
        model.train()
        tr_loss = 0.0
        n_tokens = 0
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                      BarColumn(), TextColumn(
                          "[cyan]{task.completed}/{task.total}"), TextColumn("•"),
                      TextColumn("[yellow]{task.fields[loss_str]}"),
                      TimeElapsedColumn(), transient=True) as prog:
            task = prog.add_task(
                f"Epoch {ep}/{args.epochs}", total=len(train_loader), loss_str="loss: ---")
            for batch_idx, (xb, yb) in enumerate(train_loader):
                xb, yb = xb.to(device), yb.to(device)

                # Zero gradients at start of accumulation window
                if batch_idx % args.grad_accum_steps == 0:
                    opt.zero_grad(set_to_none=True)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits, ce, ffn_feats, head_feats, Hfinal = model(
                        xb, yb, collect_layers=collect_layers, collect_heads=collect_heads)

                # update Σ_h EMA *during training*
                with torch.no_grad():
                    S_ema.update(Hfinal.detach())

                loss = ce
                # FFN feature-entropy
                if args.ff_feat_lambda > 0 and len(ffn_feats) > 0:
                    for h in ffn_feats:  # [B,T,d_ff]
                        loss = loss + \
                            gram_logdet_BTCxC(
                                h, args.ff_feat_lambda, normalize=args.head_norm)

                # Head diversity across heads
                if args.head_feat_lambda > 0 and len(head_feats) > 0:
                    for ho in head_feats:  # [B,T,H,Dh]
                        B, T, H, Dh = ho.shape
                        z = ho.permute(0, 1, 3, 2).reshape(
                            B*T*Dh, H)  # samples x heads
                        loss = loss + \
                            gram_logdet_BTCxC(
                                z, args.head_feat_lambda, normalize=args.head_norm)

                # Whitened LM head (subset)
                if args.lm_head_white_lambda > 0:
                    with torch.no_grad():
                        Ksub = min(args.lm_head_subset, cfg.vocab_size)
                        sub_idx = torch.randperm(
                            cfg.vocab_size, device=device)[:Ksub]
                    V_sub = model.lm_head.weight[sub_idx]
                    loss = loss + \
                        whitened_head_logdet(
                            V_sub, S_ema.get(), args.lm_head_white_lambda, normalize=True)

                # Scale loss for gradient accumulation
                (loss / args.grad_accum_steps).backward()

                # Step optimizer every grad_accum_steps or at end of epoch
                is_accum_step = (batch_idx + 1) % args.grad_accum_steps == 0
                is_last_batch = (batch_idx + 1) == len(train_loader)

                if is_accum_step or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()

                    # GLT (after optimizer step)
                    if global_step >= args.glt_burnin:
                        if args.glt_freq > 0 and (global_step % args.glt_freq) == 0:
                            if "ffn" in args.glt_types:
                                for b in model.blocks:
                                    glt_permute_ffn(b, opt)
                            if "attn" in args.glt_types:
                                for b in model.blocks:
                                    glt_permute_heads(b.attn, opt)
                        elif args.glt_prob > 0.0 and random.random() < args.glt_prob:
                            if "ffn" in args.glt_types:
                                for b in model.blocks:
                                    glt_permute_ffn(b, opt)
                            if "attn" in args.glt_types:
                                for b in model.blocks:
                                    glt_permute_heads(b.attn, opt)

                    global_step += 1

                    # Run validation every val_freq steps
                    if global_step - last_val_step >= args.val_freq:
                        val_ppl, ff_cond_val, head_cond_val, headW_cond_val = run_validation(
                            model, val_loader, device, cfg, ff_layers, head_layers, args)
                        current_lr = opt.param_groups[0]['lr']

                        # Log validation result
                        console.print(f"\n[bold green]Step {global_step} Validation:[/bold green] "
                                      f"val_ppl={val_ppl:.2f}, lr={current_lr:.2e}")

                        # Store in metrics
                        metrics["step_validations"].append({
                            "step": global_step,
                            "epoch": ep,
                            "val_ppl": float(val_ppl),
                            "ff_cond": float(ff_cond_val) if not np.isnan(ff_cond_val) else None,
                            "head_cond": float(head_cond_val) if not np.isnan(head_cond_val) else None,
                            "headW_cond": float(headW_cond_val) if not np.isnan(headW_cond_val) else None,
                            "lr": float(current_lr)
                        })

                        # Generate text samples after each val_freq
                        generate_samples(
                            model=model,
                            tokenizer=tokenizer,
                            vocab_info=vocab_info,
                            device=device,
                            epoch=global_step,
                            out_dir=args.out_dir,
                            num_samples=3,
                            run_tag=run_tag,
                            max_length=200,
                            temperature=0.8,
                            top_k=50
                        )

                        # Write to CSV if enabled
                        if args.save_csv:
                            with open(csv_step_path, "a") as f:
                                f.write(f"{global_step},{ep},{val_ppl:.6f},"
                                        f"{ff_cond_val if not np.isnan(ff_cond_val) else np.nan:.6e},"
                                        f"{head_cond_val if not np.isnan(head_cond_val) else np.nan:.6e},"
                                        f"{headW_cond_val if not np.isnan(headW_cond_val) else np.nan:.6e},"
                                        f"{current_lr:.6e}\n")

                        last_val_step = global_step

                tr_loss += ce.item() * (xb.size(0)*(xb.size(1)-1))
                n_tokens += xb.size(0)*(xb.size(1)-1)
                # Update progress bar with current loss and LR
                current_loss = tr_loss / max(1, n_tokens)
                current_lr = opt.param_groups[0]['lr']
                prog.update(task, advance=1,
                            loss_str=f"loss: {current_loss:.4f} • lr: {current_lr:.2e}")

        # End-of-epoch validation
        val_ppl, ff_cond, head_cond, headW_cond = run_validation(
            model, val_loader, device, cfg, ff_layers, head_layers, args)

        train_ppl = math.exp(tr_loss/max(1, n_tokens))
        lr = opt.param_groups[0]['lr']
        dt = time.time()-t0

        tbl.add_row(f"{ep}", f"{train_ppl:.2f}", f"{val_ppl:.2f}",
                    f"{ff_cond:.0f}", f"{head_cond:.2f}",
                    f"{headW_cond:.2f}" if not np.isnan(headW_cond) else "—",
                    f"{lr:.2e}", f"{dt:.1f}s")
        console.print(tbl)

        if args.save_csv:
            with open(csv_path, "a") as f:
                f.write(f"{ep},{train_ppl:.6f},{val_ppl:.6f},{ff_cond:.6e},{head_cond:.6e},{headW_cond if not np.isnan(headW_cond) else np.nan:.6e},{lr:.6e},{dt:.3f}\n")
        metrics["epochs"].append(dict(epoch=ep, train_ppl=float(train_ppl), val_ppl=float(val_ppl),
                                      ff_cond=float(ff_cond) if not np.isnan(
                                          ff_cond) else None,
                                      head_cond=float(head_cond) if not np.isnan(
                                          head_cond) else None,
                                      headW_cond=float(headW_cond) if not np.isnan(
                                          headW_cond) else None,
                                      lr=float(lr), epoch_time=float(dt)))

        # Generate text samples after each epoch
        generate_samples(model=model, tokenizer=tokenizer, vocab_info=vocab_info,
                         device=device, epoch=ep, out_dir=args.out_dir,
                         run_tag=run_tag, num_samples=3, max_length=200,
                         temperature=0.8, top_k=50)

    # final summary (best val ppl)
    best_ep, best_val = min(((m["epoch"], m["val_ppl"])
                            for m in metrics["epochs"]), key=lambda x: x[1])
    metrics["summary"] = dict(
        run_tag=run_tag, final_val_ppl=metrics["epochs"][-1]["val_ppl"],
        best_val_ppl=best_val, best_epoch=best_ep,
        final_train_ppl=metrics["epochs"][-1]["train_ppl"],
        device=str(device)
    )
    ckpt_path = os.path.join(args.out_dir, f"{run_tag}_ep{ep}.pt")
    torch.save({
        "model": model.state_dict(),
        "cfg": cfg.__dict__,
        "args": vars(args),
        "epoch": ep,
        "global_step": global_step,
    }, ckpt_path)
    console.print(f"[green]Saved checkpoint: {ckpt_path}[/green]")

    if args.save_json:
        json_path = os.path.join(args.out_dir, f"{run_tag}.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        console.print(f"[green]Saved summary to {json_path}[/green]")


if __name__ == "__main__":
    main()
