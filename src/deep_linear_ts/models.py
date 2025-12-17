"""
Core architectures for deep linear time series modeling (paper-aligned).

Two modes:
  (A) mode='strict_dln': exact DLN over flattened vectors with a square balanced core.
  (B) mode='time_series': your encoder/temporal/decoder, with geometry-aware init and probes.

References (paper):
- Balanced varieties & invariants: Eq. (4.4).  W_{p+1}^T W_{p+1} = W_p W_p^T
- Downstairs dynamics: Eq. (4.11); Theorem 2 (Riemannian gradient), Theorem 3 (g_N). 
- SVD slice / balanced factorization: Eqs. (5.15)–(5.17).
- Rectangular / rank-deficient generalization: §5.4 (Stiefel).

"""

import math
import torch
import torch.nn as nn
import numpy as np
from typing import Literal, Optional, List, Tuple

from .layers import MonarchLinearLayer, LinearAttentionLayer, ToeplitzLayer

# ---------- small helpers ----------


def _orthonormal_matrix(rows: int, cols: int, device=None, dtype=None) -> torch.Tensor:
    """
    Return a semi-orthogonal matrix with orthonormal columns if rows>=cols,
    or orthonormal rows if rows<cols. Uses QR on Gaussian matrix.
    """
    M = torch.randn(rows, cols, device=device, dtype=dtype)
    if rows >= cols:
        # rows x cols, Q^T Q = I_cols
        Q, _ = torch.linalg.qr(M, mode='reduced')
        return Q
    else:
        Q, _ = torch.linalg.qr(M.T, mode='reduced')  # cols x rows
        return Q.T  # rows x cols, Q Q^T = I_rows


def _block_identity_temporal(sequence_length: int, hidden_dim: int):
    """An identity temporal mixer placeholder."""
    return nn.Identity()

# ---------- main class ----------


class DeepLinearTimeSeries(nn.Module):
    """
    Deep linear model with two modes:

    mode='time_series' (default):
        - Encoder (input_dim -> hidden_dim), Temporal mixing, Decoder (hidden_dim -> output_dim)
        - Optional residuals
        - Balanced core init for hidden->hidden chain
        - Optional identity temporal mixing to preserve balancedness in the core

    mode='strict_dln':
        - Flattens input to R^{D_in} with D_in = sequence_length * input_dim
        - Builds a square balanced core in R^{D_core} (D_core = max(D_in, D_out) by default)
        - Sandwiches with semi-orthogonal input/output projections when D_in != D_out
        - No residuals in the core (paper setting)
        - Exact balanced factorization utilities (SVD slice)

    Args
    ----
    input_dim: features per timestep
    hidden_dim: hidden features (used in time_series mode)
    output_dim: output features per timestep
    depth: total number of layers (core depth in strict_dln; encoder/temporal/decoder split in time_series)
    sequence_length: input length T
    prediction_length: output length T_pred (defaults to T)
    temporal_mixing: 'toeplitz' | 'linear_attention' | 'fourier' | 'identity'
    use_residual: residual connections (default True for time_series; forced False in strict_dln)
    residual_weight: float in (0,1); if None, set by mild inverse-log scaling with depth
    mode: 'time_series' | 'strict_dln'
    core_embed_dim: (strict_dln) square core dimension; default = max(D_in, D_out)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        output_dim: int = 1,
        depth: int = 100,
        sequence_length: int = 1000,
        prediction_length: Optional[int] = None,
        temporal_mixing: Literal['toeplitz', 'linear_attention',
                                 'fourier', 'identity'] = 'toeplitz',
        use_residual: bool = True,
        residual_weight: Optional[float] = None,
        mode: Literal['time_series', 'strict_dln'] = 'time_series',
        core_embed_dim: Optional[int] = None,
    ):
        super().__init__()

        # ----- store config -----
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.depth = depth
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length if prediction_length is not None else sequence_length
        # paper: no residuals in strict DLN
        self.use_residual = use_residual if mode == 'time_series' else False
        self.temporal_mixing = temporal_mixing
        self.mode = mode

        # residual blending alpha: mild inverse-log with depth (kept for time_series ablations)
        if residual_weight is None:
            self.residual_weight = 1.0 / (1.0 + np.log10(max(depth, 1)))
        else:
            self.residual_weight = residual_weight

        if mode == 'time_series':
            # ===== TIME-SERIES BUILD =====
            encoder_depth = depth // 2
            temporal_depth = depth // 4
            decoder_depth = depth - encoder_depth - temporal_depth

            self.encoder = self._build_encoder(
                input_dim, hidden_dim, encoder_depth)
            self.temporal = self._build_temporal(
                hidden_dim, sequence_length, temporal_mixing, temporal_depth)
            self.decoder = self._build_decoder(
                hidden_dim, output_dim, decoder_depth)

            # Balanced / semi-orthogonal init
            self._initialize_balanced_time_series()

        else:
            # ===== STRICT DLN BUILD (exact geometry) =====
            # Flattened dims
            self.D_in = self.input_dim * self.sequence_length
            self.D_out = self.output_dim * self.prediction_length
            self.D_core = core_embed_dim if core_embed_dim is not None else max(
                self.D_in, self.D_out)

            # Projections to/from square core (semi-orthogonal at init)
            self.input_proj = None
            self.output_proj = None
            if self.D_in != self.D_core:
                self.input_proj = nn.Linear(self.D_in, self.D_core, bias=False)
            if self.D_out != self.D_core:
                self.output_proj = nn.Linear(
                    self.D_core, self.D_out, bias=False)

            # Core: N layers, all square D_core x D_core, bias=False
            self.core = nn.ModuleList(
                [nn.Linear(self.D_core, self.D_core, bias=False) for _ in range(depth)])

            # Strict balanced initialization (paper Eqs. (5.15)–(5.17)):
            #  - input/output projections semi-orthogonal (isometries)
            #  - core set to balanced factorization of α I (default α=1)
            self._initialize_balanced_strict_core(alpha=1.0)

    # ---------- builders ----------
    def _build_encoder(self, input_dim: int, hidden_dim: int, depth: int) -> nn.ModuleList:
        layers = nn.ModuleList()
        layers.append(nn.Linear(input_dim, hidden_dim, bias=False))
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        return layers

    def _build_temporal(
        self,
        hidden_dim: int,
        sequence_length: int,
        mixing_type: str,
        depth: int
    ) -> nn.ModuleList:
        layers = nn.ModuleList()
        for _ in range(max(depth, 0)):
            if mixing_type == 'toeplitz':
                layers.append(ToeplitzLayer(sequence_length))
            elif mixing_type == 'linear_attention':
                layers.append(LinearAttentionLayer(
                    hidden_dim, sequence_length))
            elif mixing_type == 'fourier':
                # simple linear map in hidden (kept linear)
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            elif mixing_type == 'identity':
                layers.append(nn.Identity())
            else:
                raise ValueError(
                    f"Unknown temporal mixing type: {mixing_type}")
        return layers

    def _build_decoder(self, hidden_dim: int, output_dim: int, depth: int) -> nn.ModuleList:
        layers = nn.ModuleList()
        for _ in range(max(depth - 1, 0)):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        return layers

    # ---------- initialization (paper-aligned) ----------
    def _initialize_balanced_time_series(self):
        """
        Geometry-aware init for time_series mode.

        - Hidden->hidden chain initialized orthogonal (so W_p^T W_p = I in the core).
        - First/last rectangular projections initialized semi-orthogonal.
        - Temporal mixing: if 'identity', preserves balancedness of the core exactly.
          If 'fourier', we set an orthogonal random map in hidden; other mixers
          may not be orthogonal and should be used when you don't need strict invariants.

        Note: Exact balanced variety equations (4.4) strictly hold only on a pure
        product of square linear maps; here we preserve them across the hidden core.  :contentReference[oaicite:6]{index=6}
        """
        device = next(self.parameters()).device if any(
            p.requires_grad for p in self.parameters()) else None
        dtype = next(self.parameters()).dtype if any(
            p.requires_grad for p in self.parameters()) else None

        # Encoder
        for i, layer in enumerate(self.encoder):
            if isinstance(layer, nn.Linear):
                W = _orthonormal_matrix(
                    layer.weight.shape[0], layer.weight.shape[1], device=device, dtype=dtype)
                with torch.no_grad():
                    layer.weight.copy_(W)

        # Decoder
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, nn.Linear):
                W = _orthonormal_matrix(
                    layer.weight.shape[0], layer.weight.shape[1], device=device, dtype=dtype)
                with torch.no_grad():
                    layer.weight.copy_(W)

        # Temporal
        if self.temporal_mixing == 'fourier':
            for layer in self.temporal:
                if isinstance(layer, nn.Linear):
                    W = _orthonormal_matrix(
                        layer.weight.shape[0], layer.weight.shape[1], device=device, dtype=dtype)
                    with torch.no_grad():
                        layer.weight.copy_(W)
        # 'identity' needs no action; 'toeplitz'/'linear_attention' are left as-is.

    def _initialize_balanced_strict_core(self, alpha: float = 1.0):
        """
        Strict DLN init:
          - Make input/output projections semi-orthogonal.
          - Set the core to a balanced factorization of α I_{D_core}:
                W1 = Λ, W2 = Λ, ..., WN = Λ   with Λ = α^{1/N} I
            This satisfies W_{p+1}^T W_{p+1} = W_p W_p^T (Eq. (4.4)) and the product is α I.  :contentReference[oaicite:7]{index=7}
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Projections
        if self.input_proj is not None:
            # rows x cols; rows >= cols typical
            W = _orthonormal_matrix(
                self.D_core, self.D_in, device=device, dtype=dtype)
            with torch.no_grad():
                self.input_proj.weight.copy_(W)
        if self.output_proj is not None:
            W = _orthonormal_matrix(
                self.D_out, self.D_core, device=device, dtype=dtype)
            with torch.no_grad():
                self.output_proj.weight.copy_(W)

        # Balanced core
        lam = float(alpha) ** (1.0 / max(self.depth, 1))
        with torch.no_grad():
            for layer in self.core:
                layer.weight.zero_()
                layer.weight.add_(
                    torch.eye(self.D_core, device=device, dtype=dtype) * lam)

    # ---------- balanced re-projection (strict core) ----------
    @torch.no_grad()
    def project_core_to_balanced(self):
        """
        Reproject the strict core to the balanced manifold while preserving the end-to-end core product.

        Given W_core = W_N ... W_1, compute SVD W = Q_N Σ Q_0^T.
        Set Λ = Σ^{1/N} and overwrite:
           W_1 := Λ Q_0^T
           W_2..W_{N-1} := Λ
           W_N := Q_N Λ
        Then W_{p+1}^T W_{p+1} = W_p W_p^T for all p, and the product is unchanged. (Eqs. (5.16)–(5.17))  :contentReference[oaicite:8]{index=8}
        """
        assert self.mode == 'strict_dln', "project_core_to_balanced is only for strict_dln mode."
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # compute core product
        W = torch.eye(self.D_core, device=device, dtype=dtype)
        for layer in reversed(self.core):
            W = layer.weight @ W  # left-multiply to accumulate W_N ... W_1

        # SVD slice
        QN, s, Q0T = torch.linalg.svd(W, full_matrices=False)
        Q0 = Q0T.T
        Sigma = torch.diag(s)
        N = len(self.core)
        # Λ = Σ^{1/N}
        lam_vals = torch.clamp(s, min=1e-12).pow(1.0 / max(N, 1))
        Λ = torch.diag(lam_vals)

        # overwrite core weights as balanced factorization
        # W1 = Λ Q0^T
        self.core[0].weight.copy_(Λ @ Q0.T)
        # W2..W_{N-1} = Λ
        for p in range(1, N - 1):
            self.core[p].weight.copy_(Λ)
        # WN = QN Λ
        self.core[-1].weight.copy_(QN @ Λ)

    # ---------- geometry diagnostics ----------
    @torch.no_grad()
    def balancedness_stats(self) -> dict:
        """
        Compute ||G_p||_F norms for adjacent layers in the balanced core:
           G_p = W_{p+1}^T W_{p+1} - W_p W_p^T

        In strict_dln: across all core layers.
        In time_series: across the hidden->hidden chain (encoder[1:], decoder[:-1]).
        """
        G_norms: List[float] = []
        Ws: List[torch.Tensor] = []

        if self.mode == 'strict_dln':
            for layer in self.core:
                Ws.append(layer.weight)
        else:
            # gather hidden->hidden 2D linear maps
            def _collect_linear_sq(mods: nn.ModuleList) -> List[torch.Tensor]:
                mats = []
                for m in mods:
                    if isinstance(m, nn.Linear) and m.weight.shape[0] == m.weight.shape[1] == self.hidden_dim:
                        mats.append(m.weight)
                return mats
            Ws = _collect_linear_sq(
                self.encoder[1:]) + _collect_linear_sq(self.decoder[:-1])

        for p in range(len(Ws) - 1):
            Wp = Ws[p]
            Wn = Ws[p + 1]
            Gp = Wn.T @ Wn - Wp @ Wp.T
            G_norms.append(Gp.norm().item())

        if not G_norms:
            return {'max_G_norm': None, 'mean_G_norm': None, 'num_pairs': 0}
        return {'max_G_norm': max(G_norms), 'mean_G_norm': float(np.mean(G_norms)), 'num_pairs': len(G_norms)}

    @torch.no_grad()
    def end_to_end_matrix(self) -> torch.Tensor:
        """
        Return the end-to-end matrix on flattened space.

        strict_dln: exact W_out = (output_proj) * (prod core) * (input_proj)
        time_series: basis-probe; cost D_in forward passes (fine for small D_in).

        This is useful for:
          - computing E'(W) with a batch
          - spectrum (singular values) logging for downstairs diagnostics  :contentReference[oaicite:9]{index=9}
        """
        device = next(self.parameters()).device
        if self.mode == 'strict_dln':
            # core product
            Dcore = self.D_core
            Wcore = torch.eye(Dcore, device=device)
            for layer in reversed(self.core):
                Wcore = layer.weight @ Wcore
            # sandwich with projections
            Win = torch.eye(self.D_in, device=device)
            if self.input_proj is not None:
                Win = self.input_proj.weight
            Wout = torch.eye(self.D_out, device=device)
            if self.output_proj is not None:
                Wout = self.output_proj.weight
            # Arrange dims: out x in
            if self.input_proj is None and self.output_proj is None:
                return Wcore
            elif self.input_proj is None:
                return Wout @ Wcore
            elif self.output_proj is None:
                return Wcore @ Win
            else:
                return Wout @ Wcore @ Win
        else:
            # probe basis in flattened input
            self.eval()
            D_in = self.input_dim * self.sequence_length
            D_out = self.output_dim * self.prediction_length
            cols = []
            I = torch.eye(D_in, device=device)
            for j in range(D_in):
                xj = I[:, j].reshape(1, self.sequence_length, self.input_dim)
                yj = self.forward(xj).reshape(-1)
                cols.append(yj)
            return torch.stack(cols, dim=1)  # [D_out, D_in]

    # ---------- forward ----------
    def forward(self, x: torch.Tensor, return_feats: bool = False):
        """
        Inputs:
          x: (batch, sequence_length, input_dim)
          return_feats: If True, return (output, features_dict) for geometry regularization

        Returns:
          If return_feats=False: (batch, prediction_length, output_dim)
          If return_feats=True: tuple of (output, features_dict) where features_dict contains:
            - 'pre_head': features before final projection [B, hidden_dim] or [B, D_core]
            - 'encoder_out': features after encoder [B, seq, hidden_dim] (time_series only)
            - 'temporal_out': features after temporal mixing [B, seq, hidden_dim] (time_series only)
        """
        if self.mode == 'strict_dln':
            B = x.size(0)
            flat = x.reshape(B, -1)  # [B, D_in]
            if self.input_proj is not None:
                flat = self.input_proj(flat)
            for layer in self.core:
                flat = layer(flat)

            # Capture pre-head features
            if return_feats:
                pre_head = flat  # [B, D_core]

            if self.output_proj is not None:
                flat = self.output_proj(flat)

            output = flat.view(B, self.prediction_length, self.output_dim)

            if return_feats:
                feats = {'pre_head': pre_head}
                return output, feats
            return output

        # time_series mode
        identity = x
        # encoder
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if self.use_residual and i > 0 and x.shape == identity.shape:
                x = self.residual_weight * x + \
                    (1.0 - self.residual_weight) * identity
                identity = x

        # Capture encoder output
        if return_feats:
            encoder_out = x  # [B, seq, hidden_dim]

        # temporal
        identity = x
        for i, layer in enumerate(self.temporal):
            x = layer(x)
            if self.use_residual and i > 0:
                x = self.residual_weight * x + \
                    (1.0 - self.residual_weight) * identity
                identity = x

        # Capture temporal output
        if return_feats:
            temporal_out = x  # [B, seq, hidden_dim]

        # decoder
        identity = x
        for i, layer in enumerate(self.decoder):
            # Capture pre-head features (before final layer)
            if return_feats and i == len(self.decoder) - 1:
                # Flatten spatial dimensions for pre-head features
                pre_head = x.reshape(x.size(0), -1)  # [B, seq*hidden_dim]

            x = layer(x)
            if self.use_residual and i < len(self.decoder) - 1 and x.shape == identity.shape:
                x = self.residual_weight * x + \
                    (1.0 - self.residual_weight) * identity
                identity = x

        # last T_pred steps
        if self.prediction_length < x.size(1):
            x = x[:, -self.prediction_length:, :]

        if return_feats:
            feats = {
                'pre_head': pre_head,
                'encoder_out': encoder_out,
                'temporal_out': temporal_out,
            }
            return x, feats
        return x


class StructuredDeepLinear(nn.Module):
    """
    Memory-efficient deep linear network using structured matrices (Monarch).
    NOTE: This class is unchanged except we now initialize with semi-orthogonal
    scaling and expose the same geometry mindset: balancedness is exact only for
    pure square cores.  See §5 and §5.4 for details about Stiefel/rectangular cases.  :contentReference[oaicite:10]{index=10}
    """

    def __init__(
        self,
        dim: int,
        depth: int = 1000,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        if input_dim is not None and input_dim != dim:
            self.input_proj = nn.Linear(input_dim, dim, bias=False)
        else:
            self.input_proj = None

        self.layers = nn.ModuleList(
            [MonarchLinearLayer(dim) for _ in range(depth)])

        if output_dim is not None and output_dim != dim:
            self.output_proj = nn.Linear(dim, output_dim, bias=False)
        else:
            self.output_proj = None

        self.initialize_balanced()

    def initialize_balanced(self):
        # Semi-orthogonal projections; mild depth scaling for the structured core
        if self.input_proj is not None:
            with torch.no_grad():
                self.input_proj.weight.copy_(_orthonormal_matrix(self.dim, self.input_proj.in_features,
                                                                 device=self.input_proj.weight.device,
                                                                 dtype=self.input_proj.weight.dtype))
        if self.output_proj is not None:
            with torch.no_grad():
                self.output_proj.weight.copy_(_orthonormal_matrix(self.output_proj.out_features, self.dim,
                                                                  device=self.output_proj.weight.device,
                                                                  dtype=self.output_proj.weight.dtype))
        scale = 1.0 / math.sqrt(max(self.depth, 1))
        with torch.no_grad():
            for layer in self.layers:
                for p in layer.parameters():
                    p.mul_(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_proj is not None:
            x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        if self.output_proj is not None:
            x = self.output_proj(x)
        return x
