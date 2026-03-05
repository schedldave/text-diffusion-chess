# Text Diffusion Chess

Experiments with discrete text diffusion models trained on chess game notation.

## Idea

Train a small masked discrete diffusion model on chess games represented as move sequences in Standard Algebraic Notation (SAN). The model learns to synthesize new chess games by iteratively denoising masked token sequences. The key research question: **can a diffusion model generate valid chess games that respect all the rules?**

## Project Structure

```
text-diffusion-chess/
├── data/
│   └── generate_games.py      # Generate chess games via random/engine play
├── tokenizer/
│   └── chess_tokenizer.py     # Move-level chess tokenizer
├── model/
│   ├── transformer.py         # Bidirectional transformer backbone
│   └── diffusion.py           # Masked discrete diffusion process
├── configs/
│   └── default.yaml           # Hyperparameter configuration
├── train.py                   # Training script (CUDA + multi-GPU)
├── sample.py                  # Generate games from trained model
└── evaluate.py                # Validate generated games against chess rules
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate training data (10k random games)
python data/generate_games.py --num_games 10000 --output data/games.txt

# Build tokenizer vocabulary
python tokenizer/chess_tokenizer.py --build --data data/games.txt --output tokenizer/vocab.json

# Train the model
python train.py --config configs/default.yaml

# Sample new games
python sample.py --checkpoint checkpoints/best.pt --num_games 100

# Evaluate generated games
python evaluate.py --games_file generated_games.txt
```

## Approach

- **Data**: Chess games in SAN notation generated with `python-chess` (random legal play + optional Stockfish self-play)
- **Tokenizer**: Move-level tokenizer where each SAN move (e.g., `e4`, `Nf3`, `O-O`) is a single token
- **Model**: Small bidirectional transformer with masked discrete diffusion (inspired by MDLM and tiny-diffusion)
- **Diffusion**: Absorbing-state (mask-based) forward process with cosine schedule; reverse process predicts masked tokens

## References

- [MDLM: Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) (NeurIPS 2024)
- [tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion) - Character-level text diffusion in ~400 lines
- [DiffuSearch: Implicit Search via Discrete Diffusion](https://arxiv.org/abs/2502.19805) (ICLR 2025)
- [python-chess](https://python-chess.readthedocs.io/)
