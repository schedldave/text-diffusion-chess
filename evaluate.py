"""Evaluate generated chess games for rule validity.

Takes a file of generated games (one per line, space-separated SAN moves) and
checks each game against chess rules using python-chess.

Metrics:
  - Move-level validity: is each move legal at that board state?
  - Game-level validity: fraction of games that are fully legal
  - Average valid prefix: how many moves before the first illegal move?
  - Diversity: number of unique games and unique openings

Usage:
    python evaluate.py --games_file generated_games.txt
    python evaluate.py --games_file generated_games.txt --verbose
"""

import argparse
from collections import Counter
from dataclasses import dataclass

import chess


@dataclass
class GameEvaluation:
    """Evaluation result for a single game."""
    total_moves: int
    valid_moves: int
    first_illegal_at: int | None
    is_fully_legal: bool
    reached_game_over: bool
    termination: str | None
    error_move: str | None
    error_message: str | None


def evaluate_game(moves: list[str]) -> GameEvaluation:
    """Evaluate a single game for chess rule compliance."""
    board = chess.Board()
    valid_count = 0
    first_illegal_at = None
    error_move = None
    error_message = None

    for i, san_move in enumerate(moves):
        try:
            move = board.parse_san(san_move)
            if move not in board.legal_moves:
                first_illegal_at = i
                error_move = san_move
                error_message = f"Move {san_move} is not legal at this position"
                break
            board.push(move)
            valid_count += 1

            if board.is_game_over():
                termination = _get_termination(board)
                return GameEvaluation(
                    total_moves=len(moves),
                    valid_moves=valid_count,
                    first_illegal_at=None,
                    is_fully_legal=True,
                    reached_game_over=True,
                    termination=termination,
                    error_move=None,
                    error_message=None,
                )
        except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError) as e:
            first_illegal_at = i
            error_move = san_move
            error_message = str(e)
            break

    is_fully_legal = first_illegal_at is None
    return GameEvaluation(
        total_moves=len(moves),
        valid_moves=valid_count,
        first_illegal_at=first_illegal_at,
        is_fully_legal=is_fully_legal,
        reached_game_over=False,
        termination=None,
        error_move=error_move,
        error_message=error_message,
    )


def _get_termination(board: chess.Board) -> str:
    """Get a human-readable termination reason."""
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        return f"Checkmate ({winner} wins)"
    if board.is_stalemate():
        return "Stalemate"
    if board.is_insufficient_material():
        return "Insufficient material"
    if board.is_fifty_moves():
        return "50-move rule"
    if board.is_repetition():
        return "Threefold repetition"
    return "Game over (unknown)"


def evaluate_all(games: list[list[str]], verbose: bool = False) -> dict:
    """Evaluate a batch of games and compute aggregate metrics."""
    results: list[GameEvaluation] = []

    for i, moves in enumerate(games):
        result = evaluate_game(moves)
        results.append(result)

        if verbose and not result.is_fully_legal:
            print(
                f"  Game {i + 1}: ILLEGAL at move {result.first_illegal_at} "
                f"'{result.error_move}' - {result.error_message}"
            )
        elif verbose and result.reached_game_over:
            print(
                f"  Game {i + 1}: COMPLETE ({result.valid_moves} moves, "
                f"{result.termination})"
            )

    total = len(results)
    fully_legal = sum(1 for r in results if r.is_fully_legal)
    reached_end = sum(1 for r in results if r.reached_game_over)

    valid_prefixes = [r.valid_moves for r in results]
    game_lengths = [r.total_moves for r in results]

    game_strings = [" ".join(g) for g in games]
    unique_games = len(set(game_strings))

    opening_3 = [" ".join(g[:3]) for g in games if len(g) >= 3]
    unique_openings = len(set(opening_3))

    illegal_moves: list[str] = []
    for r in results:
        if r.error_move is not None:
            illegal_moves.append(r.error_move)
    common_errors = Counter(illegal_moves).most_common(10)

    metrics = {
        "total_games": total,
        "fully_legal_games": fully_legal,
        "fully_legal_rate": fully_legal / max(total, 1),
        "reached_game_over": reached_end,
        "game_over_rate": reached_end / max(total, 1),
        "mean_valid_prefix": sum(valid_prefixes) / max(total, 1),
        "median_valid_prefix": sorted(valid_prefixes)[total // 2] if total > 0 else 0,
        "max_valid_prefix": max(valid_prefixes) if valid_prefixes else 0,
        "mean_game_length": sum(game_lengths) / max(total, 1),
        "unique_games": unique_games,
        "unique_rate": unique_games / max(total, 1),
        "unique_openings_3": unique_openings,
        "common_illegal_moves": common_errors,
    }
    return metrics


def load_games(path: str) -> list[list[str]]:
    """Load games from a text file (one game per line, space-separated moves)."""
    games: list[list[str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            moves = line.split()
            if len(moves) > 0:
                games.append(moves)
    return games


def print_metrics(metrics: dict):
    """Print evaluation metrics in a readable format."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total games evaluated:  {metrics['total_games']}")
    print(f"Fully legal games:      {metrics['fully_legal_games']} "
          f"({metrics['fully_legal_rate']:.1%})")
    print(f"Reached game over:      {metrics['reached_game_over']} "
          f"({metrics['game_over_rate']:.1%})")
    print()
    print(f"Mean valid prefix:      {metrics['mean_valid_prefix']:.1f} moves")
    print(f"Median valid prefix:    {metrics['median_valid_prefix']} moves")
    print(f"Max valid prefix:       {metrics['max_valid_prefix']} moves")
    print(f"Mean game length:       {metrics['mean_game_length']:.1f} moves")
    print()
    print(f"Unique games:           {metrics['unique_games']} "
          f"({metrics['unique_rate']:.1%})")
    print(f"Unique 3-move openings: {metrics['unique_openings_3']}")

    if metrics["common_illegal_moves"]:
        print("\nMost common illegal moves:")
        for move, count in metrics["common_illegal_moves"]:
            print(f"  '{move}': {count} times")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate chess game validity")
    parser.add_argument("--games_file", type=str, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Loading games from {args.games_file}...")
    games = load_games(args.games_file)
    print(f"Loaded {len(games)} games")

    print("Evaluating...")
    metrics = evaluate_all(games, verbose=args.verbose)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
