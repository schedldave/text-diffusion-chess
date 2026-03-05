# Text Diffusion Chess - Project Overview

## Goal

Train a small masked discrete diffusion model on chess games in text notation and evaluate whether it can generate valid chess games that respect all rules. Compare against an autoregressive transformer baseline.

## Project Structure

```
text-diffusion-chess/
├── CLAUDE.md                    # This file - project tracker
├── README.md                    # Public project overview
├── requirements.txt             # Python dependencies
├── data/
│   └── generate_games.py        # Chess data generation (random + engine play)
├── tokenizer/
│   └── chess_tokenizer.py       # Move-level chess tokenizer
├── model/
│   ├── __init__.py
│   ├── transformer.py           # Bidirectional transformer backbone
│   ├── diffusion.py             # Masked discrete diffusion (forward + reverse)
│   └── autoregressive.py        # Causal transformer (AR baseline)
├── configs/
│   ├── default.yaml             # Hyperparameters (model size, training, diffusion)
│   └── test.yaml                # Test config (500 epochs, reduced warmup)
├── train.py                     # Diffusion training loop (CUDA, multi-GPU, checkpointing)
├── train_ar.py                  # AR training loop (matched setup)
├── sample.py                    # Generate new games from diffusion model
├── sample_ar.py                 # Generate new games from AR model
└── evaluate.py                  # Validate games against chess rules
```

## Research Notes

### Chess Notation Choice: SAN (Standard Algebraic Notation)
- Compact, human-readable: `e4 e5 Nf3 Nc6 Bb5`
- Each move is a token -> short sequences (~40-150 tokens/game)
- Vocabulary built from training data (~4,300 unique moves with weighted-random play)
- python-chess handles all generation, parsing, and validation

### Diffusion Approach: Masked Discrete Diffusion
- Inspired by MDLM (NeurIPS 2024) and tiny-diffusion
- Forward process: progressively replace tokens with [MASK]
- Reverse process: bidirectional transformer predicts masked tokens
- Sampling: start fully masked, iteratively unmask by confidence (100 steps)
- Cosine noise schedule for masking rate

### Autoregressive Baseline
- Standard causal (GPT-style) transformer, matched architecture
- Next-token prediction with cross-entropy loss
- Autoregressive sampling (left-to-right, one token at a time)
- Same d_model, n_heads, n_layers, d_ff as diffusion model

### Key References
- MDLM: https://arxiv.org/abs/2406.07524
- tiny-diffusion: https://github.com/nathan-barry/tiny-diffusion
- DiffuSearch (chess + diffusion): https://arxiv.org/abs/2502.19805
- python-chess: https://python-chess.readthedocs.io/

### Model Sizing (fits 16GB GPU VRAM)
- Diffusion model: 7,123,164 parameters
- AR model: 6,991,580 parameters (no timestep embedding)
- Embedding dim: 256, heads: 8, layers: 6, FFN: 1024
- Sequence length: 150 moves
- Parameterizable via configs/default.yaml or configs/test.yaml

## Experiment Log

### Experiment 1: Diffusion Model (10 Epochs)
- **Data**: 10,000 weighted-random games, mean 145 moves
- **Config**: batch_size=128, lr=3e-4, 3x V100 GPUs
- **Results**: val_loss=5.90, mean valid prefix=1.5 moves, max=9 moves
- **Time**: ~84 seconds

### Experiment 2: Diffusion Model (500 Epochs)
- **Data**: Same as above
- **Config**: Same, 500 epochs
- **Results**: best val_loss=5.18 (epoch 416), mean valid prefix=7.4 moves, max=20 moves
- **Time**: ~65 minutes
- **Observations**: Significant improvement in valid prefix length (1.5 -> 7.4 mean), model learned legal openings and basic piece movement, but no fully legal games. Loss continued decreasing but with diminishing returns after ~400 epochs.

### Experiment 3: Autoregressive Baseline (500 Epochs) - IN PROGRESS
- **Data**: Same as above
- **Config**: Matched architecture and training setup
- **Results at epoch 100**: train_loss=2.88, val_loss=5.58 (significant overfitting vs diffusion's 5.34/5.39 at same epoch)
- **Observation**: AR model overfits much faster than diffusion model on this dataset size

## Status

- [x] Repository setup
- [x] Data generation script
- [x] Chess tokenizer
- [x] Bidirectional transformer model
- [x] Diffusion model
- [x] Training script
- [x] Sampling script
- [x] Evaluation script
- [x] Autoregressive baseline model
- [x] AR training script
- [x] AR sampling script
- [x] First diffusion training run (10 epochs)
- [x] Extended diffusion training (500 epochs)
- [x] Evaluation of diffusion-generated games
- [ ] Complete AR training (500 epochs)
- [ ] Evaluate AR-generated games
- [ ] Head-to-head comparison
