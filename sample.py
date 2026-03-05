"""Generate chess games from a trained diffusion model.

Usage:
    python sample.py --checkpoint checkpoints/best.pt --config configs/default.yaml
    python sample.py --checkpoint checkpoints/best.pt --num_games 50 --temperature 0.8
"""

import argparse
from pathlib import Path

import torch
import yaml

from model.transformer import ChessDiffusionTransformer
from model.diffusion import MaskedDiffusion
from tokenizer.chess_tokenizer import ChessTokenizer


def load_model(
    checkpoint_path: str,
    config: dict,
    tokenizer: ChessTokenizer,
    device: torch.device,
) -> MaskedDiffusion:
    """Load a trained diffusion model from checkpoint."""
    transformer = ChessDiffusionTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=config["data"]["max_seq_len"],
        dropout=0.0,
        pad_id=tokenizer.pad_id,
    )

    diffusion = MaskedDiffusion(
        transformer=transformer,
        num_timesteps=config["diffusion"]["num_timesteps"],
        mask_id=tokenizer.mask_id,
        pad_id=tokenizer.pad_id,
        schedule=config["diffusion"]["schedule"],
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    diffusion.load_state_dict(checkpoint["model"])
    diffusion = diffusion.to(device)
    diffusion.eval()
    return diffusion


def sample_games(
    model: MaskedDiffusion,
    tokenizer: ChessTokenizer,
    num_games: int,
    seq_len: int,
    device: torch.device,
    temperature: float = 0.9,
    batch_size: int = 32,
) -> list[list[str]]:
    """Generate chess games from the model."""
    all_games: list[list[str]] = []

    remaining = num_games
    while remaining > 0:
        n = min(remaining, batch_size)
        tokens = model.sample(
            batch_size=n,
            seq_len=seq_len,
            device=device,
            temperature=temperature,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
        )

        for i in range(n):
            ids = tokens[i].cpu().tolist()
            moves = tokenizer.decode(ids, skip_special=True)
            all_games.append(moves)

        remaining -= n

    return all_games


def main():
    parser = argparse.ArgumentParser(description="Sample games from trained model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--vocab", type=str, default=None)
    parser.add_argument("--num_games", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--output", type=str, default="generated_games.txt")
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    vocab_file = args.vocab or config["data"]["vocab_file"]
    seq_len = args.seq_len or config["sampling"]["seq_len"]

    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    tokenizer = ChessTokenizer.from_vocab(vocab_file)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, config, tokenizer, device)

    print(f"Generating {args.num_games} games (seq_len={seq_len}, temp={args.temperature})...")
    games = sample_games(
        model, tokenizer, args.num_games, seq_len, device,
        temperature=args.temperature,
    )

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for game in games:
            f.write(" ".join(game) + "\n")

    print(f"\nGenerated {len(games)} games -> {output_path}")

    print("\nSample games:")
    for i, game in enumerate(games[:5]):
        print(f"  Game {i + 1} ({len(game)} moves): {' '.join(game[:20])}...")


if __name__ == "__main__":
    main()
