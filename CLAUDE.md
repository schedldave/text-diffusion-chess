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
- **Observations**:
  - Significant improvement over 10 epochs (1.5 -> 7.4 mean valid prefix)
  - Minimal overfitting: train/val gap only 0.28 at epoch 500
  - Diffusion epoch 100: val_loss=5.38, mean prefix=4.7, max=16

### Experiment 3: Autoregressive Baseline (500 Epochs)
- **Data**: Same as above
- **Config**: Matched architecture and training setup
- **Results**: best val_loss=4.65 (epoch 33), mean valid prefix=10.5 moves, max=23 moves
- **Time**: ~69 minutes
- **Observations**:
  - AR model learns faster (best val at epoch 33 vs 416 for diffusion)
  - Severe overfitting: train_loss=1.98, val_loss=7.10 at epoch 500 (gap of 5.12)
  - Despite overfitting, epoch 500 still generates diverse games with similar prefix quality (11.5 mean, 26 max)
  - AR model achieves better game quality (10.5 vs 7.4 mean prefix)

### Cross-Checkpoint Comparison

#### AR Checkpoints
| Checkpoint | Val Loss | Mean Prefix | Max Prefix |
|---|---|---|---|
| Epoch 10 | 5.27 | 4.5 | 11 |
| Epoch 33 (best) | 4.65 | 10.5 | 23 |
| Epoch 100 | 5.58 | 11.5 | 22 |
| Epoch 500 | 7.10 | 11.5 | 26 |

#### Diffusion Checkpoints
| Checkpoint | Val Loss | Mean Prefix | Max Prefix |
|---|---|---|---|
| Epoch 10 | 5.90 | 1.5 | 9 |
| Epoch 100 | 5.38 | 4.7 | 16 |
| Epoch 416 (best) | 5.18 | 7.4 | 20 |

### Key Findings
1. AR model produces longer valid prefixes (10.5 vs 7.4) -- causal bias helps for sequential games
2. Diffusion model barely overfits (0.28 gap) vs AR (5.12 gap) -- masking is a strong regularizer
3. Neither generates fully legal games -- board state tracking remains the core challenge
4. AR game quality plateaus early (~epoch 33) while diffusion keeps improving through epoch 416
5. Heavily overfitted AR models still generate diverse, quality sequences

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
- [x] Complete AR training (500 epochs)
- [x] Evaluate AR-generated games (multiple checkpoints)
- [x] Head-to-head comparison
