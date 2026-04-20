from __future__ import annotations

import math
from pathlib import Path

import torch

from data import TextDataset
from model import GPTConfig, MiniGPT


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def estimate_loss(
    model: MiniGPT,
    dataset: TextDataset,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    """
    Estimate average train/validation loss over multiple batches.
    """
    out: dict[str, float] = {}
    model.eval()

    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = dataset.get_batch(split=split, batch_size=batch_size)
            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError("Loss unexpectedly None during evaluation.")
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out


def loss_to_perplexity(loss_value: float) -> float:
    """
    Perplexity = exp(loss), with overflow guard.
    """
    try:
        return math.exp(loss_value)
    except OverflowError:
        return float("inf")


def save_checkpoint(
    path: str | Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_val_loss: float,
    tokenizer_chars: list[str],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "best_val_loss": best_val_loss,
            "tokenizer_chars": tokenizer_chars,
            "config": model.config.to_dict(),
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    device: str,
) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def build_model_from_checkpoint(
    checkpoint: dict,
    device: str,
) -> MiniGPT:
    config = GPTConfig(**checkpoint["config"])
    model = MiniGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def generate_from_prompt(
    model: MiniGPT,
    dataset: TextDataset,
    prompt: str,
    device: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 20,
) -> str:
    tokenizer = dataset.tokenizer

    for ch in prompt:
        if ch not in tokenizer.stoi:
            raise ValueError(
                f"Prompt contains unseen character {repr(ch)}. "
                "For this character-level model, prompts must use characters from the training corpus."
            )

    context_ids = tokenizer.encode(prompt)
    context = torch.tensor([context_ids], dtype=torch.long, device=device)

    model.eval()
    generated_ids = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )[0].tolist()

    return tokenizer.decode(generated_ids)


def format_attention_table(tokens: list[str], weights: torch.Tensor, decimals: int = 2) -> str:
    """
    Create a readable text table for one attention matrix.

    tokens: length T
    weights: (T, T)
    """
    if weights.ndim != 2:
        raise ValueError("weights must have shape (T, T).")

    T = len(tokens)
    if weights.shape != (T, T):
        raise ValueError(
            f"weights shape {tuple(weights.shape)} does not match token count {T}."
        )

    header = "tok\\att".ljust(10) + "".join(repr(tok).center(8) for tok in tokens)
    lines = [header]

    for i in range(T):
        row_label = repr(tokens[i]).ljust(10)
        row_vals = "".join(f"{weights[i, j].item():.{decimals}f}".center(8) for j in range(T))
        lines.append(row_label + row_vals)

    return "\n".join(lines)


def print_attention_focus(tokens: list[str], weights: torch.Tensor, top_n: int = 3) -> None:
    """
    Print top attended-to positions for each token.

    tokens: length T
    weights: (T, T)
    """
    T = len(tokens)

    for i in range(T):
        row = weights[i]
        top_vals, top_idx = torch.topk(row, k=min(top_n, T))
        print(f"Token {i} {repr(tokens[i])} attends most to:")
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            print(f"  pos {idx:>2} {repr(tokens[idx])}: {val:.4f}")
        print("-" * 40)