"""Training script for the autoregressive chess model.

Matched setup to train.py for fair comparison with the diffusion model.

Usage:
    python train_ar.py --config configs/test.yaml
    python train_ar.py --config configs/test.yaml --gpus 0,1,2
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import yaml

from model.autoregressive import ChessAutoregressive
from tokenizer.chess_tokenizer import ChessTokenizer


class ChessGameDataset(Dataset):
    """Dataset of tokenized chess games."""

    def __init__(self, games_file: str, tokenizer: ChessTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.games: list[list[int]] = []

        with open(games_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                moves = line.split()
                ids = tokenizer.encode(moves, add_special=True, max_length=max_seq_len)
                self.games.append(ids)

    def __len__(self) -> int:
        return len(self.games)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.games[idx], dtype=torch.long)


def get_device_and_model(
    model: nn.Module,
    gpu_ids: list[int] | None = None,
) -> tuple[torch.device, nn.Module]:
    if torch.cuda.is_available():
        if gpu_ids is not None and len(gpu_ids) > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
            device = torch.device("cuda:0")
            model = model.to(device)
            if len(gpu_ids) > 1:
                model = nn.DataParallel(model, device_ids=list(range(len(gpu_ids))))
                print(f"Using DataParallel on GPUs: {gpu_ids}")
            else:
                print(f"Using single GPU: {gpu_ids[0]}")
        else:
            device = torch.device("cuda")
            model = model.to(device)
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        model = model.to(device)
        print("Using CPU")
    return device, model


def train(config: dict, gpu_ids: list[int] | None = None):
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Loading tokenizer...")
    tokenizer = ChessTokenizer.from_vocab(config["data"]["vocab_file"])
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    print("Loading dataset...")
    dataset = ChessGameDataset(
        config["data"]["games_file"],
        tokenizer,
        config["data"]["max_seq_len"],
    )
    print(f"  Total games: {len(dataset)}")

    train_size = int(len(dataset) * config["data"]["train_split"])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    print(f"  Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    print("Building autoregressive model...")
    model = ChessAutoregressive(
        vocab_size=tokenizer.vocab_size,
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        d_ff=config["model"]["d_ff"],
        max_seq_len=config["data"]["max_seq_len"],
        dropout=config["model"]["dropout"],
        pad_id=tokenizer.pad_id,
    )
    print(f"  Parameters: {model.count_parameters():,}")

    device, model = get_device_and_model(model, gpu_ids)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * config["training"]["num_epochs"]
    warmup_steps = config["training"]["warmup_steps"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    checkpoint_dir = Path(config["training"]["checkpoint_dir"] + "_ar")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting AR training for {config['training']['num_epochs']} epochs...")
    print(f"  Steps per epoch: {len(train_loader)}")
    print(f"  Total steps: {total_steps}")

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, config["training"]["num_epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0
        t_start = time.time()

        for batch in train_loader:
            x = batch.to(device)
            padding_mask = (x == tokenizer.pad_id)

            if isinstance(model, nn.DataParallel):
                loss = model.module.compute_loss(x, padding_mask)
            else:
                loss = model.compute_loss(x, padding_mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["training"]["grad_clip"],
            )
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            if global_step % config["training"]["log_every"] == 0:
                avg = epoch_loss / epoch_steps
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  [Step {global_step}] loss={loss.item():.4f} "
                    f"avg={avg:.4f} lr={lr:.6f}"
                )

        epoch_time = time.time() - t_start
        avg_train_loss = epoch_loss / max(epoch_steps, 1)

        val_loss = evaluate(model, val_loader, device, tokenizer)

        print(
            f"Epoch {epoch}/{config['training']['num_epochs']} "
            f"train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} "
            f"time={epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                best_val_loss, checkpoint_dir / "best.pt",
            )
            print(f"  -> New best model saved (val_loss={val_loss:.4f})")

        if epoch % config["training"]["save_every"] == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step,
                best_val_loss, checkpoint_dir / f"epoch_{epoch}.pt",
            )

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer: ChessTokenizer,
) -> float:
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch.to(device)
            padding_mask = (x == tokenizer.pad_id)

            if isinstance(model, nn.DataParallel):
                loss = model.module.compute_loss(x, padding_mask)
            else:
                loss = model.compute_loss(x, padding_mask)

            total_loss += loss.item()
            total_steps += 1

    return total_loss / max(total_steps, 1)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    path: Path,
):
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    if isinstance(model, nn.DataParallel):
        state["model"] = model.module.state_dict()
    else:
        state["model"] = model.state_dict()
    torch.save(state, path)


def main():
    parser = argparse.ArgumentParser(description="Train autoregressive chess model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs, e.g. '0,1,2'")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    gpu_ids = None
    if args.gpus is not None:
        gpu_ids = [int(g) for g in args.gpus.split(",")]

    train(config, gpu_ids)


if __name__ == "__main__":
    main()
