from pathlib import Path

import torch

from data import TextDataset
from model import MiniGPT


# ----------------------------
# Hyperparameters
# ----------------------------
batch_size = 32
block_size = 64
max_iters = 2000
eval_interval = 200
eval_iters = 100
learning_rate = 3e-4

embed_dim = 128
num_heads = 4
num_layers = 4
dropout = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

checkpoint_dir = Path("checkpoints")
checkpoint_path = checkpoint_dir / "mini_gpt.pt"


def estimate_loss(model: MiniGPT, dataset: TextDataset) -> dict[str, float]:
    """
    Estimate mean loss over several batches for train and validation splits.
    """
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            x, y = dataset.get_batch(split=split, batch_size=batch_size)
            x = x.to(device)
            y = y.to(device)

            _, loss = model(x, y)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out


def save_checkpoint(
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokenizer_chars: list[str],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "tokenizer_chars": tokenizer_chars,
            "config": {
                "block_size": block_size,
                "embed_dim": embed_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "dropout": dropout,
            },
        },
        checkpoint_path,
    )


def load_checkpoint(vocab_size: int) -> MiniGPT:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint["config"]
    model = MiniGPT(
        vocab_size=vocab_size,
        embed_dim=config["embed_dim"],
        block_size=config["block_size"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def generate_from_prompt(
    model: MiniGPT,
    dataset: TextDataset,
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    top_k: int | None = 20,
) -> str:
    tokenizer = dataset.tokenizer

    for ch in prompt:
        if ch not in tokenizer.stoi:
            raise ValueError(
                f"Prompt contains unseen character {repr(ch)}. "
                "Use only characters from the training corpus for this character-level model."
            )

    context_ids = tokenizer.encode(prompt)
    context = torch.tensor([context_ids], dtype=torch.long, device=device)

    generated_ids = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )[0].tolist()

    return tokenizer.decode(generated_ids)


def main():
    # ----------------------------
    # Load text + dataset
    # ----------------------------
    data_path = Path("data/input.txt")
    text = data_path.read_text(encoding="utf-8")

    dataset = TextDataset(text=text, block_size=block_size, train_split=0.9)
    tokenizer = dataset.tokenizer

    print("=" * 70)
    print("STAGE 10: DROPOUT + BETTER GENERATION + CHECKPOINTS")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Corpus length: {len(text)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    print(f"Embedding dim: {embed_dim}")
    print(f"Num heads: {num_heads}")
    print(f"Num layers: {num_layers}")
    print(f"Dropout: {dropout}")
    print()

    # ----------------------------
    # Train model
    # ----------------------------
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        if step % eval_interval == 0 or step == max_iters - 1:
            losses = estimate_loss(model, dataset)
            print(
                f"step {step:4d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

        x, y = dataset.get_batch(split="train", batch_size=batch_size)
        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print()
    print("Training complete.")
    print()

    # ----------------------------
    # Save checkpoint
    # ----------------------------
    save_checkpoint(model, optimizer, max_iters, tokenizer.chars)
    print(f"Saved checkpoint to: {checkpoint_path}")
    print()

    # ----------------------------
    # Reload checkpoint
    # ----------------------------
    reloaded_model = load_checkpoint(vocab_size=tokenizer.vocab_size)

    # ----------------------------
    # Generate from prompts
    # ----------------------------
    prompts = [
        "The ",
        "To be",
        "\n",
    ]

    print("=" * 70)
    print("GENERATED SAMPLES")
    print("=" * 70)

    for prompt in prompts:
        print(f"\nPrompt: {repr(prompt)}")
        try:
            generated_text = generate_from_prompt(
                model=reloaded_model,
                dataset=dataset,
                prompt=prompt,
                max_new_tokens=250,
                temperature=0.8,
                top_k=20,
            )
            print(generated_text)
        except ValueError as e:
            print(f"Skipped prompt: {e}")


if __name__ == "__main__":
    main()