from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn


class FiLMRegressor(nn.Module):
    """
    Full-dimension conditional modulation (FiLM) for polyBERT + tabular features.

    Inputs
    ------
    - poly: Tensor of shape (B, poly_dim) — polyBERT pooled embedding
    - mat : Tensor of shape (B, material_dim) — standardized tabular features (can be empty)

    Output
    ------
    - y: Tensor of shape (B, 1) — regression target (e.g., conductivity)

    Key ideas
    ---------
    - Keep the full poly dimension (use_full_poly=True) to avoid information loss.
    - Generate per-dimension FiLM parameters gamma, beta from material features and modulate poly: y = Head(gamma ⊙ z + beta).
    - Identity-friendly init so the model can start close to a poly-only baseline.
    - Optional poly linear shortcut to preserve a strong poly-only path.
    """

    def __init__(
        self,
        poly_dim: int,
        material_dim: int = 0,
        *,
        use_full_poly: bool = True,
        d_model: int = 128,
        cond_hidden: Sequence[int] = (32, 64),
        head_hidden: Sequence[int] = (64, 32),
        dropout: float = 0.1,
        film_mode: str = "dense",  # 'dense' | 'lowrank' | 'group'
        film_rank: int = 16,        # used when film_mode == 'lowrank'
        film_groups: Optional[int] = None,  # used when film_mode == 'group'
        film_scale: float = 0.1,     # scale for identity-friendly FiLM
        add_poly_shortcut: bool = True,
    ) -> None:
        super().__init__()

        assert poly_dim > 0, "poly_dim must be positive"
        assert material_dim >= 0, "material_dim must be non-negative"

        self.poly_dim = int(poly_dim)
        self.material_dim = int(material_dim)
        self.use_full_poly = bool(use_full_poly)
        self.film_mode = str(film_mode)
        self.film_rank = int(film_rank)
        self.film_groups = film_groups if film_groups is None else int(film_groups)
        self.film_scale = float(film_scale)
        self.add_poly_shortcut = bool(add_poly_shortcut)

        # Determine working dimensionality for the modulated feature z
        if self.use_full_poly:
            self.z_dim = self.poly_dim
            self.poly_proj = nn.LayerNorm(self.poly_dim)
        else:
            self.z_dim = int(d_model)
            self.poly_proj = nn.Sequential(
                nn.Linear(self.poly_dim, 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, self.z_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.LayerNorm(self.z_dim),
            )

        # Conditional network from material features
        if self.material_dim > 0:
            cond_layers = []
            in_d = self.material_dim
            for h in cond_hidden:
                cond_layers += [nn.Linear(in_d, h), nn.ReLU(), nn.Dropout(dropout)]
                in_d = h
            self.cond_mlp = nn.Sequential(*cond_layers) if cond_layers else nn.Identity()

            if self.film_mode == "dense":
                self.gamma_head = nn.Linear(in_d, self.z_dim)
                self.beta_head = nn.Linear(in_d, self.z_dim)
                self._init_dense_film_heads()
            elif self.film_mode == "lowrank":
                assert self.film_rank > 0, "film_rank must be positive for lowrank FiLM"
                self.gamma_vec = nn.Linear(in_d, self.film_rank)
                self.beta_vec = nn.Linear(in_d, self.film_rank)
                self.gamma_proj = nn.Linear(self.film_rank, self.z_dim, bias=False)
                self.beta_proj = nn.Linear(self.film_rank, self.z_dim, bias=False)
                self._init_lowrank_film_heads()
            elif self.film_mode == "group":
                assert self.film_groups is not None and self.film_groups > 0, "film_groups must be set for group FiLM"
                assert self.z_dim % self.film_groups == 0, "z_dim must be divisible by film_groups"
                self.gamma_head = nn.Linear(in_d, self.film_groups)
                self.beta_head = nn.Linear(in_d, self.film_groups)
                self._init_dense_film_heads()
            else:
                raise ValueError(f"Unknown film_mode: {self.film_mode}")
        else:
            # no material conditioning
            self.cond_mlp = None
            self.gamma_head = None
            self.beta_head = None

        # Prediction head
        head_layers = []
        in_d = self.z_dim
        for h in head_hidden:
            head_layers += [nn.Linear(in_d, h), nn.ReLU(), nn.Dropout(dropout)]
            in_d = h
        head_layers += [nn.Linear(in_d, 1)]
        self.head = nn.Sequential(*head_layers)

        self.poly_shortcut = nn.Linear(self.poly_dim, 1) if self.add_poly_shortcut else None
        if self.poly_shortcut is not None:
            nn.init.zeros_(self.poly_shortcut.weight)
            nn.init.zeros_(self.poly_shortcut.bias)

    # --------------------
    # Init helpers
    # --------------------
    def _init_dense_film_heads(self) -> None:
        # Identity-friendly: gamma ~= 1, beta ~= 0 at init via scaled tanh later
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.ones_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def _init_lowrank_film_heads(self) -> None:
        nn.init.zeros_(self.gamma_vec.weight)
        nn.init.zeros_(self.gamma_vec.bias)
        nn.init.zeros_(self.beta_vec.weight)
        nn.init.zeros_(self.beta_vec.bias)
        # small random for projections
        nn.init.xavier_uniform_(self.gamma_proj.weight)
        nn.init.xavier_uniform_(self.beta_proj.weight)

    # --------------------
    # Forward
    # --------------------
    def forward(self, poly: torch.Tensor, mat: Optional[torch.Tensor] = None) -> torch.Tensor:
        # poly -> z
        z = self.poly_proj(poly)

        # default FiLM params (identity)
        gamma = None
        beta = None

        if self.material_dim > 0 and mat is not None and mat.numel() > 0:
            if mat.dim() == 1:
                mat = mat.unsqueeze(0)

            h = self.cond_mlp(mat) if self.cond_mlp is not None else mat

            if self.film_mode == "dense":
                gamma_raw = self.gamma_head(h)
                beta_raw = self.beta_head(h)
                gamma = 1.0 + self.film_scale * torch.tanh(gamma_raw)
                beta = self.film_scale * torch.tanh(beta_raw)

            elif self.film_mode == "lowrank":
                g_v = self.gamma_vec(h)
                b_v = self.beta_vec(h)
                gamma_raw = self.gamma_proj(g_v)
                beta_raw = self.beta_proj(b_v)
                gamma = 1.0 + self.film_scale * torch.tanh(gamma_raw)
                beta = self.film_scale * torch.tanh(beta_raw)

            elif self.film_mode == "group":
                # produce per-group params, then broadcast to per-dimension
                g = self.film_groups
                gamma_g = 1.0 + self.film_scale * torch.tanh(self.gamma_head(h))  # (B, G)
                beta_g = self.film_scale * torch.tanh(self.beta_head(h))         # (B, G)
                repeat = self.z_dim // g
                gamma = gamma_g.repeat_interleave(repeat, dim=1)
                beta = beta_g.repeat_interleave(repeat, dim=1)

        # apply FiLM (identity if gamma/beta are None)
        if gamma is None or beta is None:
            y = z
        else:
            y = gamma * z + beta

        out = self.head(y)
        if self.poly_shortcut is not None:
            out = out + self.poly_shortcut(poly)
        return out


def create_film_model(
    poly_dim: int,
    material_dim: int,
    **kwargs,
) -> FiLMRegressor:
    """Factory function to build a FiLMRegressor with flexible options.

    Common kwargs:
        use_full_poly: bool = True
        d_model: int = 128
        cond_hidden: Sequence[int] = (32, 64)
        head_hidden: Sequence[int] = (64, 32)
        dropout: float = 0.1
        film_mode: str = 'dense' | 'lowrank' | 'group'
        film_rank: int = 16
        film_groups: Optional[int] = None
        film_scale: float = 0.1
        add_poly_shortcut: bool = True
    """
    return FiLMRegressor(poly_dim=poly_dim, material_dim=material_dim, **kwargs)

