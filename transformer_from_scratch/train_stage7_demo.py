from pathlib import Path
import torch

from data import TextDataset
from model import MiniGPT


def main():
    data_path = Path(__file__).parent / "data" / "input.txt"
    text = data_path.read_text(encoding="utf-8")

    # Hyperparameters
    block_size = 16
    batch_size = 4
    embed_dim = 32
    num_heads = 4
    num_layers = 2

    dataset = TextDataset(text=text, block_size=block_size)
    tokenizer = dataset.tokenizer

    x, y = dataset.get_batch(batch_size=batch_size)

    print("=" * 60)
    print("STAGE 7: FULL MINI-GPT MODEL")
    print("=" * 60)

    print("Decoded input example:")
    print(repr(tokenizer.decode(x[0].tolist())))
    print()

    print("Decoded target example:")
    print(repr(tokenizer.decode(y[0].tolist())))
    print()

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
        num_heads=num_heads,
        num_layers=num_layers
    )

    logits, loss = model(x, y)

    print("Logits shape:", logits.shape)
    print("Expected shape: (batch_size * block_size, vocab_size)")
    print()

    print("Loss:", loss.item())
    print()

    # Inspect predictions
    probs = torch.softmax(logits, dim=-1)

    print("First token probability distribution (truncated):")
    print(probs[0][:10])
    print()

    # Get predicted token
    predicted_id = torch.argmax(probs[0]).item()
    predicted_char = tokenizer.itos[predicted_id]

    print("Predicted next token for first position:")
    print(predicted_char)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()