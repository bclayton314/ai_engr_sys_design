from __future__ import annotations

from pathlib import Path

import torch

from data import TextDataset
from model import GPTConfig, MiniGPT
from utils import (
    estimate_loss,
    get_device,
    loss_to_perplexity,
    save_checkpoint,
    generate_from_prompt,
)


# ----------------------------
# Training hyperparameters
# ----------------------------
BATCH_SIZE = 32
MAX_ITERS = 2000
EVAL_INTERVAL = 200
EVAL_ITERS = 100
LEARNING_RATE = 3e-4
TRAIN_SPLIT = 0.9

# ----------------------------
# Model hyperparameters
# ----------------------------
BLOCK_SIZE = 64
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
DROPOUT = 0.1

# ----------------------------
# Paths
# ----------------------------
DATA_PATH = Path("data/input.txt")
CHECKPOINT_PATH = Path("checkpoints/mini_gpt.pt")


def main() -> None:
    device = get_device()
    torch.manual_seed(42)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing training data file: {DATA_PATH}")

    text = DATA_PATH.read_text(encoding="utf-8")
    dataset = TextDataset(text=text, block_size=BLOCK_SIZE, train_split=TRAIN_SPLIT)
    tokenizer = dataset.tokenizer

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    model = MiniGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("=" * 72)
    print("FINAL STAGE: TRAINING + EVALUATION + CHECKPOINTING")
    print("=" * 72)
    print(f"Device:           {device}")
    print(f"Corpus length:    {len(text)}")
    print(f"Vocabulary size:  {tokenizer.vocab_size}")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Block size:       {BLOCK_SIZE}")
    print(f"Embed dim:        {EMBED_DIM}")
    print(f"Num heads:        {NUM_HEADS}")
    print(f"Num layers:       {NUM_LAYERS}")
    print(f"Dropout:          {DROPOUT}")
    print(f"Max iterations:   {MAX_ITERS}")
    print()

    best_val_loss = float("inf")

    for step in range(MAX_ITERS):
        if step % EVAL_INTERVAL == 0 or step == MAX_ITERS - 1:
            losses = estimate_loss(
                model=model,
                dataset=dataset,
                batch_size=BATCH_SIZE,
                eval_iters=EVAL_ITERS,
                device=device,
            )
            train_loss = losses["train"]
            val_loss = losses["val"]

            train_ppl = loss_to_perplexity(train_loss)
            val_ppl = loss_to_perplexity(val_loss)

            print(
                f"step {step:4d} | "
                f"train loss {train_loss:.4f} | train ppl {train_ppl:.2f} | "
                f"val loss {val_loss:.4f} | val ppl {val_ppl:.2f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    path=CHECKPOINT_PATH,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    tokenizer_chars=tokenizer.chars,
                )
                print(f"  saved new best checkpoint -> {CHECKPOINT_PATH}")

        x, y = dataset.get_batch(split="train", batch_size=BATCH_SIZE)
        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)
        if loss is None:
            raise RuntimeError("Loss unexpectedly None during training.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print()
    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best validation perplexity: {loss_to_perplexity(best_val_loss):.2f}")
    print()

    print("=" * 72)
    print("FIXED-PROMPT SAMPLE GENERATIONS")
    print("=" * 72)

    prompts = [
        "The ",
        "To be",
        "\n",
    ]

    for prompt in prompts:
        print(f"\nPrompt: {repr(prompt)}")
        try:
            generated = generate_from_prompt(
                model=model,
                dataset=dataset,
                prompt=prompt,
                device=device,
                max_new_tokens=200,
                temperature=0.8,
                top_k=20,
            )
            print(generated)
        except ValueError as e:
            print(f"Skipped: {e}")


if __name__ == "__main__":
    main()