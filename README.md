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

## Results

Trained on 10,000 weighted-random chess games (mean length 145 moves) using 3x Tesla V100 GPUs for 500 epochs each.

### Head-to-Head: Diffusion vs Autoregressive (Best Checkpoints)

| Metric | Diffusion (best, ep 416) | AR (best, ep 33) |
|---|---|---|
| Parameters | 7,123,164 | 6,991,580 |
| Best val loss | 5.18 | **4.65** |
| Fully legal games | 0% | 0% |
| **Mean valid prefix** | 7.4 moves | **10.5 moves** |
| Median valid prefix | 7 moves | **10 moves** |
| Max valid prefix | 20 moves | **23 moves** |
| Unique games | 100% | 100% |

**The autoregressive model wins**, producing longer valid move prefixes (10.5 vs 7.4 mean) while reaching its best checkpoint much earlier (epoch 33 vs 416).

### AR Model: Effect of Training Duration

The AR model peaks early and then overfits. More training actually doesn't help game quality much, but doesn't hurt either -- the model memorizes training data but still generates diverse outputs.

| Checkpoint | Val Loss | Train Loss | Mean Valid Prefix | Max Valid Prefix |
|---|---|---|---|---|
| Epoch 10 | 5.27 | 4.74 | 4.5 | 11 |
| **Epoch 33 (best)** | **4.65** | 3.48 | **10.5** | 23 |
| Epoch 100 | 5.58 | 2.88 | 11.5 | 22 |
| Epoch 500 | 7.10 | 1.98 | 11.5 | **26** |

Interestingly, the heavily overfitted epoch 500 model still generates games of similar quality to the best val-loss checkpoint, and even achieves the longest valid prefix (26 moves).

### Diffusion Model: Effect of Training Duration

The diffusion model trains more slowly but with much less overfitting.

| Checkpoint | Val Loss | Mean Valid Prefix | Max Valid Prefix |
|---|---|---|---|
| Epoch 10 | 5.90 | 1.5 | 9 |
| Epoch 100 | 5.38 | 4.7 | 16 |
| **Epoch 416 (best)** | **5.18** | **7.4** | **20** |

### Training Dynamics Comparison

| | Diffusion | Autoregressive |
|---|---|---|
| Best val loss | 5.18 | 4.65 |
| Epoch of best val loss | 416 | 33 |
| Final train loss | 4.90 | 1.98 |
| Final val loss | 5.18 | 7.10 |
| **Train/val gap at 500 ep** | **0.28** | **5.12** |
| Overfitting | Minimal | Severe |
| Training time | ~65 min | ~69 min |

The diffusion model has a strong regularization effect from random masking, maintaining a train/val gap of only 0.28 even after 500 epochs. The AR model's gap grows to 5.12.

## Approach

- **Data**: Chess games in SAN notation generated with `python-chess` (weighted-random play favoring captures and center control)
- **Tokenizer**: Move-level tokenizer where each SAN move (e.g., `e4`, `Nf3`, `O-O`) is a single token. Vocabulary of ~4,300 unique moves.
- **Diffusion Model**: Bidirectional transformer with absorbing-state masked diffusion (inspired by MDLM). Forward process masks tokens with cosine schedule; reverse process iteratively unmasks by confidence over 100 steps.
- **AR Baseline**: Matched causal transformer with standard next-token prediction and autoregressive sampling.
- **Architecture**: Both models use d_model=256, 8 heads, 6 layers, d_ff=1024 (~7M parameters).

## Key Takeaways

1. **AR models learn chess move structure faster** -- the causal left-to-right inductive bias aligns well with the sequential nature of chess games.
2. **Diffusion models resist overfitting** -- the random masking during training acts as a powerful regularizer, keeping the train/val gap minimal.
3. **Neither model generates fully legal games** with only 10K training games and 500 epochs -- chess rule compliance requires understanding board state, which pure sequence modeling struggles with.
4. **More AR training doesn't hurt generation quality** despite massive overfitting -- the model still generates diverse, novel sequences even at epoch 500.

## References

- [MDLM: Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) (NeurIPS 2024)
- [tiny-diffusion](https://github.com/nathan-barry/tiny-diffusion) - Character-level text diffusion in ~400 lines
- [DiffuSearch: Implicit Search via Discrete Diffusion](https://arxiv.org/abs/2502.19805) (ICLR 2025)
- [python-chess](https://python-chess.readthedocs.io/)
