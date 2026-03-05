"""Generate chess games in SAN notation for training a text diffusion model.

Supports two modes:
  1. Random legal play: both sides pick uniformly random legal moves (fast, diverse)
  2. Engine self-play: Stockfish plays against itself at configurable depth (realistic)

Output: one game per line, space-separated SAN moves, e.g.:
  e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O ...
"""

import argparse
import random
from pathlib import Path
from typing import Optional

import chess
import chess.engine
from tqdm import tqdm


def play_random_game(max_moves: int = 300) -> list[str]:
    """Play a game where both sides make uniformly random legal moves."""
    board = chess.Board()
    moves: list[str] = []
    for _ in range(max_moves):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        move = random.choice(legal)
        moves.append(board.san(move))
        board.push(move)
    return moves


def play_engine_game(
    engine_path: str,
    time_limit: float = 0.01,
    max_moves: int = 300,
) -> list[str]:
    """Play a game where Stockfish plays both sides."""
    board = chess.Board()
    moves: list[str] = []
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    try:
        for _ in range(max_moves):
            if board.is_game_over():
                break
            result = engine.play(board, chess.engine.Limit(time=time_limit))
            if result.move is None:
                break
            moves.append(board.san(result.move))
            board.push(result.move)
    finally:
        engine.quit()
    return moves


def play_weighted_random_game(max_moves: int = 300) -> list[str]:
    """Play with some heuristic weighting: prefer captures and center moves."""
    board = chess.Board()
    moves: list[str] = []
    center_squares = {chess.E4, chess.D4, chess.E5, chess.D5,
                      chess.C3, chess.F3, chess.C6, chess.F6}

    for _ in range(max_moves):
        if board.is_game_over():
            break
        legal = list(board.legal_moves)
        weights: list[float] = []
        for m in legal:
            w = 1.0
            if board.is_capture(m):
                w += 3.0
            if m.to_square in center_squares:
                w += 1.5
            if m.promotion is not None:
                w += 5.0
            weights.append(w)
        move = random.choices(legal, weights=weights, k=1)[0]
        moves.append(board.san(move))
        board.push(move)
    return moves


def generate_games(
    num_games: int,
    mode: str = "random",
    engine_path: Optional[str] = None,
    time_limit: float = 0.01,
    max_moves: int = 300,
) -> list[list[str]]:
    """Generate multiple chess games."""
    games: list[list[str]] = []
    for _ in tqdm(range(num_games), desc=f"Generating {mode} games"):
        if mode == "random":
            game = play_random_game(max_moves)
        elif mode == "weighted":
            game = play_weighted_random_game(max_moves)
        elif mode == "engine":
            if engine_path is None:
                raise ValueError("engine_path required for engine mode")
            game = play_engine_game(engine_path, time_limit, max_moves)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if len(game) >= 4:
            games.append(game)
    return games


def main():
    parser = argparse.ArgumentParser(description="Generate chess games for training")
    parser.add_argument("--num_games", type=int, default=10000)
    parser.add_argument("--output", type=str, default="data/games.txt")
    parser.add_argument("--mode", choices=["random", "weighted", "engine"], default="random")
    parser.add_argument("--engine_path", type=str, default=None,
                        help="Path to Stockfish binary (for engine mode)")
    parser.add_argument("--time_limit", type=float, default=0.01,
                        help="Engine time limit per move in seconds")
    parser.add_argument("--max_moves", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    games = generate_games(
        num_games=args.num_games,
        mode=args.mode,
        engine_path=args.engine_path,
        time_limit=args.time_limit,
        max_moves=args.max_moves,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for game in games:
            f.write(" ".join(game) + "\n")

    print(f"Generated {len(games)} games -> {output_path}")

    lengths = [len(g) for g in games]
    print(f"  Mean length: {sum(lengths) / len(lengths):.1f} moves")
    print(f"  Min: {min(lengths)}, Max: {max(lengths)}")


if __name__ == "__main__":
    main()
