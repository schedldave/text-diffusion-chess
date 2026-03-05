"""Bidirectional Transformer backbone for masked discrete diffusion.

A standard transformer encoder (not causal) that takes noised token sequences
and a diffusion timestep, then predicts the original tokens at masked positions.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1).float() * emb.unsqueeze(0)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class TransformerBlock(nn.Module):
    """Single transformer encoder block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            is_causal=False,
        )
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class ChessDiffusionTransformer(nn.Module):
    """Bidirectional transformer for predicting masked chess moves.

    Takes a noised sequence of token IDs and a diffusion timestep,
    returns logits over the vocabulary for each position.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        max_seq_len: int = 150,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) noised token IDs
            t: (batch,) diffusion timestep indices
            padding_mask: (batch, seq_len) True where padded

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

        h = self.token_emb(x) + self.pos_emb(positions)
        time_emb = self.time_mlp(t)
        h = h + time_emb.unsqueeze(1)

        for block in self.blocks:
            h = block(h, key_padding_mask=padding_mask)

        h = self.ln_out(h)
        logits = self.head(h)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
