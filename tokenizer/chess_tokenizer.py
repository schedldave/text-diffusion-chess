"""Move-level chess tokenizer.

Each SAN move (e.g. 'e4', 'Nf3', 'O-O') is a single token. The vocabulary is
built from training data and includes special control tokens for the diffusion
process.

Usage:
    # Build vocabulary from game data
    python tokenizer/chess_tokenizer.py --build --data data/games.txt --output tokenizer/vocab.json

    # Or use programmatically:
    tokenizer = ChessTokenizer.from_vocab("tokenizer/vocab.json")
    ids = tokenizer.encode(["e4", "e5", "Nf3"])
    moves = tokenizer.decode(ids)
"""

import argparse
import json
from pathlib import Path
from typing import Optional


SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[MASK]": 1,
    "[BOS]": 2,
    "[EOS]": 3,
}


class ChessTokenizer:
    """Move-level tokenizer for chess games in SAN notation."""

    def __init__(self, move_to_id: dict[str, int], id_to_move: dict[int, str]):
        self.move_to_id = move_to_id
        self.id_to_move = id_to_move
        self.pad_id = SPECIAL_TOKENS["[PAD]"]
        self.mask_id = SPECIAL_TOKENS["[MASK]"]
        self.bos_id = SPECIAL_TOKENS["[BOS]"]
        self.eos_id = SPECIAL_TOKENS["[EOS]"]
        self.vocab_size = len(move_to_id)

    @classmethod
    def build_from_games(cls, games_file: str) -> "ChessTokenizer":
        """Build vocabulary from a file of games (one game per line, space-separated moves)."""
        move_set: set[str] = set()
        with open(games_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                moves = line.split()
                move_set.update(moves)

        sorted_moves = sorted(move_set)

        move_to_id: dict[str, int] = dict(SPECIAL_TOKENS)
        next_id = len(SPECIAL_TOKENS)
        for move in sorted_moves:
            if move not in move_to_id:
                move_to_id[move] = next_id
                next_id += 1

        id_to_move = {v: k for k, v in move_to_id.items()}
        return cls(move_to_id, id_to_move)

    def save(self, path: str) -> None:
        """Save vocabulary to JSON."""
        data = {
            "move_to_id": self.move_to_id,
            "id_to_move": {str(k): v for k, v in self.id_to_move.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved vocabulary ({self.vocab_size} tokens) -> {path}")

    @classmethod
    def from_vocab(cls, path: str) -> "ChessTokenizer":
        """Load vocabulary from JSON."""
        with open(path) as f:
            data = json.load(f)
        move_to_id = data["move_to_id"]
        id_to_move = {int(k): v for k, v in data["id_to_move"].items()}
        return cls(move_to_id, id_to_move)

    def encode(
        self,
        moves: list[str],
        add_special: bool = True,
        max_length: Optional[int] = None,
    ) -> list[int]:
        """Encode a list of SAN moves to token IDs."""
        ids: list[int] = []
        if add_special:
            ids.append(self.bos_id)
        for m in moves:
            ids.append(self.move_to_id.get(m, self.mask_id))
        if add_special:
            ids.append(self.eos_id)

        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
                if add_special:
                    ids[-1] = self.eos_id
            else:
                ids.extend([self.pad_id] * (max_length - len(ids)))
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> list[str]:
        """Decode token IDs back to SAN moves."""
        special_ids = set(SPECIAL_TOKENS.values())
        moves: list[str] = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            moves.append(self.id_to_move.get(i, "[UNK]"))
        return moves

    def game_to_string(self, ids: list[int]) -> str:
        """Convert token IDs to a human-readable game string."""
        moves = self.decode(ids, skip_special=True)
        return " ".join(moves)


def main():
    parser = argparse.ArgumentParser(description="Chess tokenizer utilities")
    parser.add_argument("--build", action="store_true", help="Build vocabulary from data")
    parser.add_argument("--data", type=str, default="data/games.txt")
    parser.add_argument("--output", type=str, default="tokenizer/vocab.json")
    parser.add_argument("--info", type=str, default=None,
                        help="Print info about an existing vocabulary file")
    args = parser.parse_args()

    if args.build:
        tokenizer = ChessTokenizer.build_from_games(args.data)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(args.output)

        print("\nVocabulary statistics:")
        print(f"  Total tokens: {tokenizer.vocab_size}")
        print(f"  Special tokens: {len(SPECIAL_TOKENS)}")
        print(f"  Move tokens: {tokenizer.vocab_size - len(SPECIAL_TOKENS)}")
        print(f"\nSample tokens: {list(tokenizer.move_to_id.items())[:15]}")

    if args.info:
        tokenizer = ChessTokenizer.from_vocab(args.info)
        print(f"Vocabulary: {tokenizer.vocab_size} tokens")
        print(f"Special: {list(SPECIAL_TOKENS.keys())}")
        moves = [m for m in tokenizer.move_to_id if m not in SPECIAL_TOKENS]
        print(f"Sample moves: {moves[:20]}")


if __name__ == "__main__":
    main()
