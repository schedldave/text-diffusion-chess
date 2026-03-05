"""Autoregressive (causal) Transformer for chess move generation.

Standard GPT-style decoder-only transformer for next-token prediction.
Matched in size to the diffusion transformer for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTransformerBlock(nn.Module):
    """Single transformer block with causal (autoregressive) attention and pre-norm."""

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
        self, x: torch.Tensor, attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.ln1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=False,
        )
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x


class ChessAutoregressive(nn.Module):
    """Causal transformer for autoregressive chess move generation.

    Predicts the next token given all previous tokens.
    Matched architecture to ChessDiffusionTransformer (minus timestep embedding).
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
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, d_ff, dropout)
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
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token IDs
            padding_mask: (batch, seq_len) True where padded

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)

        h = self.token_emb(x) + self.pos_emb(positions)

        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=x.device), diagonal=1,
        )

        for block in self.blocks:
            h = block(h, attn_mask=causal_mask, key_padding_mask=padding_mask)

        h = self.ln_out(h)
        logits = self.head(h)
        return logits

    def compute_loss(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Next-token prediction loss.

        For sequence [BOS, t1, t2, ..., tn, EOS, PAD, PAD]:
        - Input:  [BOS, t1, t2, ..., tn, EOS, PAD]
        - Target: [t1, t2, ..., tn, EOS, PAD, PAD]
        Loss computed only on non-padding positions.
        """
        logits = self.forward(x, padding_mask=padding_mask)

        # Shift: predict next token from each position
        logits_shift = logits[:, :-1, :].contiguous()
        targets_shift = x[:, 1:].contiguous()

        loss = F.cross_entropy(
            logits_shift.view(-1, self.vocab_size),
            targets_shift.view(-1),
            ignore_index=self.pad_id,
        )
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        temperature: float = 1.0,
        bos_id: int = 2,
        eos_id: int = 3,
    ) -> torch.Tensor:
        """Autoregressive sampling: generate one token at a time."""
        x = torch.full((batch_size, 1), bos_id, device=device, dtype=torch.long)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(seq_len - 1):
            logits = self.forward(x)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            next_token[finished] = self.pad_id
            x = torch.cat([x, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == eos_id)

            if finished.all():
                break

        if x.shape[1] < seq_len:
            pad = torch.full(
                (batch_size, seq_len - x.shape[1]), self.pad_id,
                device=device, dtype=torch.long,
            )
            x = torch.cat([x, pad], dim=1)

        return x

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
