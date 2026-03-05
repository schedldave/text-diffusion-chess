"""Masked Discrete Diffusion process for text.

Implements an absorbing-state discrete diffusion model where the forward process
progressively replaces tokens with [MASK] and the reverse process learns to
predict the original tokens.

Inspired by MDLM (https://arxiv.org/abs/2406.07524) and
tiny-diffusion (https://github.com/nathan-barry/tiny-diffusion).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.transformer import ChessDiffusionTransformer


def cosine_noise_schedule(timesteps: int) -> torch.Tensor:
    """Cosine schedule for masking rate: gamma(t) goes from 0 to ~1.

    Returns a tensor of shape (timesteps+1,) where schedule[t] is the
    probability that a token is masked at timestep t.
    """
    steps = torch.linspace(0, 1, timesteps + 1)
    alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2.0) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    mask_rate = 1.0 - alpha_bar
    return mask_rate.clamp(0.0, 1.0)


def linear_noise_schedule(timesteps: int) -> torch.Tensor:
    """Linear schedule for masking rate."""
    return torch.linspace(0.0, 1.0, timesteps + 1)


class MaskedDiffusion(nn.Module):
    """Absorbing-state masked diffusion for discrete tokens.

    Forward process: tokens -> progressively masked with [MASK]
    Reverse process: transformer predicts original tokens at masked positions
    """

    def __init__(
        self,
        transformer: ChessDiffusionTransformer,
        num_timesteps: int = 100,
        mask_id: int = 1,
        pad_id: int = 0,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.transformer = transformer
        self.num_timesteps = num_timesteps
        self.mask_id = mask_id
        self.pad_id = pad_id

        if schedule == "cosine":
            noise_schedule = cosine_noise_schedule(num_timesteps)
        elif schedule == "linear":
            noise_schedule = linear_noise_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        self.register_buffer("noise_schedule", noise_schedule)

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process: mask tokens according to noise schedule at timestep t.

        Args:
            x_0: (batch, seq_len) original clean token IDs
            t: (batch,) timestep indices in [1, num_timesteps]

        Returns:
            x_t: (batch, seq_len) noised tokens (some replaced with mask_id)
            mask: (batch, seq_len) boolean, True where tokens were masked
        """
        mask_rate = self.noise_schedule[t]
        rand = torch.rand_like(x_0, dtype=torch.float)
        mask = rand < mask_rate.unsqueeze(-1)

        is_pad = (x_0 == self.pad_id)
        mask = mask & ~is_pad

        x_t = x_0.clone()
        x_t[mask] = self.mask_id
        return x_t, mask

    def compute_loss(
        self,
        x_0: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute training loss: cross-entropy on masked positions only.

        Args:
            x_0: (batch, seq_len) clean token IDs
            padding_mask: (batch, seq_len) True where padded

        Returns:
            loss: scalar
        """
        B = x_0.shape[0]
        t = torch.randint(1, self.num_timesteps + 1, (B,), device=x_0.device)

        x_t, mask = self.q_sample(x_0, t)
        logits = self.transformer(x_t, t, padding_mask=padding_mask)

        logits_flat = logits.reshape(-1, logits.shape[-1])
        targets_flat = x_0.reshape(-1)
        mask_flat = mask.reshape(-1)

        loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss = (loss * mask_flat.float()).sum()

        num_masked = mask_flat.float().sum().clamp(min=1.0)
        return loss / num_masked

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        temperature: float = 1.0,
        bos_id: int | None = 2,
        eos_id: int | None = 3,
    ) -> torch.Tensor:
        """Generate new sequences by iterative unmasking.

        Starts from fully masked sequence and progressively unmasks tokens
        from timestep T down to 1, choosing tokens by confidence.

        Args:
            batch_size: number of sequences to generate
            seq_len: length of each sequence (including BOS/EOS)
            device: target device
            temperature: sampling temperature
            bos_id: if set, fix first token to BOS
            eos_id: if set, fix last token to EOS

        Returns:
            x: (batch_size, seq_len) generated token IDs
        """
        x = torch.full((batch_size, seq_len), self.mask_id, device=device)

        if bos_id is not None:
            x[:, 0] = bos_id
        if eos_id is not None:
            x[:, -1] = eos_id

        fixed = torch.zeros_like(x, dtype=torch.bool)
        if bos_id is not None:
            fixed[:, 0] = True
        if eos_id is not None:
            fixed[:, -1] = True

        for step in range(self.num_timesteps, 0, -1):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)

            logits = self.transformer(x, t)
            probs = F.softmax(logits / temperature, dim=-1)

            is_masked = (x == self.mask_id) & ~fixed

            if not is_masked.any():
                break

            confidences = probs.max(dim=-1).values
            confidences[~is_masked] = float("inf")

            current_rate = self.noise_schedule[step].item()
            next_rate = self.noise_schedule[step - 1].item() if step > 1 else 0.0

            total_masked = is_masked.sum(dim=-1).float()
            target_masked = (total_masked * (next_rate / max(current_rate, 1e-8))).long()
            num_to_unmask = (total_masked.long() - target_masked).clamp(min=1)

            for i in range(batch_size):
                if not is_masked[i].any():
                    continue

                n = num_to_unmask[i].item()
                masked_positions = is_masked[i].nonzero(as_tuple=True)[0]

                if len(masked_positions) == 0:
                    continue

                pos_confidences = confidences[i, masked_positions]
                n = min(n, len(masked_positions))
                _, top_idx = pos_confidences.topk(n)
                unmask_positions = masked_positions[top_idx]

                for pos in unmask_positions:
                    token = torch.multinomial(probs[i, pos], 1).item()
                    x[i, pos] = token

        remaining_masked = (x == self.mask_id) & ~fixed
        if remaining_masked.any():
            t_final = torch.ones((batch_size,), device=device, dtype=torch.long)
            logits = self.transformer(x, t_final)
            probs = F.softmax(logits / temperature, dim=-1)
            for i in range(batch_size):
                for pos in remaining_masked[i].nonzero(as_tuple=True)[0]:
                    x[i, pos] = torch.multinomial(probs[i, pos], 1).item()

        return x
