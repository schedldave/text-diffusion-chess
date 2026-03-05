# Text Diffusion Chess - Project Overview

## Goal

Train a small masked discrete diffusion model on chess games in text notation and evaluate whether it can generate valid chess games that respect all rules.

## Project Structure

```
text-diffusion-chess/
├── CLAUDE.md                  # This file - project tracker
├── README.md                  # Public project overview
├── requirements.txt           # Python dependencies
├── data/
│   └── generate_games.py      # Chess data generation (random + engine play)
├── tokenizer/
│   └── chess_tokenizer.py     # Move-level chess tokenizer
├── model/
│   ├── __init__.py
│   ├── transformer.py         # Bidirectional transformer backbone
│   └── diffusion.py           # Masked discrete diffusion (forward + reverse)
├── configs/
│   └── default.yaml           # Hyperparameters (model size, training, diffusion)
├── train.py                   # Training loop (CUDA, multi-GPU, checkpointing)
├── sample.py                  # Generate new games from a trained model
└── evaluate.py                # Validate games against chess rules
```

## Research Notes

### Chess Notation Choice: SAN (Standard Algebraic Notation)
- Compact, human-readable: `e4 e5 Nf3 Nc6 Bb5`
- Each move is a token -> short sequences (~40-120 tokens/game)
- Vocabulary built from training data (~200-500 unique moves in practice)
- python-chess handles all generation, parsing, and validation

### Diffusion Approach: Masked Discrete Diffusion
- Inspired by MDLM (NeurIPS 2024) and tiny-diffusion
- Forward process: progressively replace tokens with [MASK]
- Reverse process: bidirectional transformer predicts masked tokens
- Sampling: start fully masked, iteratively unmask by confidence
- Cosine noise schedule for masking rate

### Key References
- MDLM: https://arxiv.org/abs/2406.07524
- tiny-diffusion: https://github.com/nathan-barry/tiny-diffusion
- DiffuSearch (chess + diffusion): https://arxiv.org/abs/2502.19805
- python-chess: https://python-chess.readthedocs.io/

### Model Sizing (fits 16GB GPU VRAM)
- Default config: ~5-10M parameters
- Embedding dim: 256, heads: 8, layers: 6
- Sequence length: 150 moves
- Parameterizable via configs/default.yaml

## Experiment Log

*(To be filled as experiments are run)*

## Status

- [x] Repository setup
- [x] Data generation script
- [x] Chess tokenizer
- [x] Transformer model
- [x] Diffusion model
- [x] Training script
- [x] Sampling script
- [x] Evaluation script
- [ ] First training run
- [ ] Evaluation of generated games
