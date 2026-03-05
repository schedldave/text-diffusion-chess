# Text Diffusion Chess

Experiments with discrete text diffusion models trained on chess game notation, compared against an autoregressive baseline.

## Idea

Train a small masked discrete diffusion model on chess games represented as move sequences in Standard Algebraic Notation (SAN). The model learns to synthesize new chess games by iteratively denoising masked token sequences. The key research question: **can a diffusion model generate valid chess games that respect all the rules?**

We also train a matched autoregressive (GPT-style) transformer as a baseline for comparison.

## Project Structure

```
text-diffusion-chess/
├── data/
│   └── generate_games.py        # Generate chess games via random/engine play
├── tokenizer/
│   └── chess_tokenizer.py       # Move-level chess tokenizer
├── model/
│   ├── transformer.py           # Bidirectional transformer (diffusion backbone)
│   ├── diffusion.py             # Masked discrete diffusion process
│   └── autoregressive.py        # Causal transformer (AR baseline)
├── configs/
│   ├── default.yaml             # Default hyperparameter configuration
│   └── test.yaml                # Test configuration (500 epochs)
├── train.py                     # Diffusion model training (CUDA + multi-GPU)
├── train_ar.py                  # Autoregressive model training
├── sample.py                    # Generate games from diffusion model
├── sample_ar.py                 # Generate games from AR model
└── evaluate.py                  # Validate generated games against chess rules
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Generate training data (10k weighted-random games)
python data/generate_games.py --num_games 10000 --output data/games.txt --mode weighted

# Build tokenizer vocabulary
python tokenizer/chess_tokenizer.py --build --data data/games.txt --output tokenizer/vocab.json

# Train the diffusion model
python train.py --config configs/test.yaml --gpus 0,1,2

# Train the autoregressive baseline
python train_ar.py --config configs/test.yaml --gpus 0,1,2

# Sample new games
python sample.py --checkpoint checkpoints/best.pt --config configs/test.yaml --num_games 100
python sample_ar.py --checkpoint checkpoints_ar/best.pt --config configs/test.yaml --num_games 100

# Evaluate generated games
python evaluate.py --games_file generated_games.txt
python evaluate.py --games_file generated_games_ar.txt
```

## Results (500 Epochs)

Trained on 10,000 weighted-random chess games (mean length 145 moves) using 3x Tesla V100 GPUs.

### Model Comparison

| | Diffusion (7.1M params) | Autoregressive (7.0M params) |
|---|---|---|
| Architecture | Bidirectional transformer + masked diffusion | Causal (GPT-style) transformer |
| Training signal | Predict masked tokens at random positions | Predict next token |
| Best val loss | 5.18 (epoch 416) | TBD |
| Sampling | 100-step iterative unmasking | Token-by-token left-to-right |

### Generated Game Quality

| Metric | Training Data | Diffusion (500 ep) | AR (500 ep) |
|---|---|---|---|
| Fully legal games | 100% | 0% | TBD |
| Mean valid prefix | 143.9 moves | 7.4 moves | TBD |
| Max valid prefix | 150 moves | 20 moves | TBD |
| Unique games | 100% | 100% | TBD |

### Diffusion Training Progress (10 vs 500 Epochs)

| Metric | 10 Epochs | 500 Epochs |
|---|---|---|
| Best val loss | 5.90 | 5.18 |
| Mean valid prefix | 1.5 moves | 7.4 moves |
| Max valid prefix | 9 moves | 20 moves |

## Approach

- **Data**: Chess games in SAN notation generated with `python-chess` (weighted-random play favoring captures and center control)
- **Tokenizer**: Move-level tokenizer where each SAN move (e.g., `e4`, `Nf3`, `O-O`) is a single token. Vocabulary of ~4,300 unique moves.
- **Diffusion Model**: Bidirectional transformer with absorbing-state masked diffusion (inspired by MDLM). Forward process masks tokens with cosine schedule; reverse process iteratively unmasks by confidence over 100 steps.
- **AR Baseline**: Matched causal transformer with standard next-token prediction and autoregressive sampling.

## References

- [MDLM: Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) (NeurIPS 2024)
- [tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion) - Character-level text diffusion in ~400 lines
- [DiffuSearch: Implicit Search via Discrete Diffusion](https://arxiv.org/abs/2502.19805) (ICLR 2025)
- [python-chess](https://python-chess.readthedocs.io/)
