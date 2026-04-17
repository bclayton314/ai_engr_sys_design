from pathlib import Path

import torch

from data import TextDataset
from model import InputEmbedding


def main():
    data_path = Path(__file__).parent / "data" / "input.txt"
    text = data_path.read_text(encoding="utf-8")

    block_size = 16
    batch_size = 4
    embed_dim = 32

    dataset = TextDataset(text=text, block_size=block_size, train_split=0.9)
    tokenizer = dataset.tokenizer

    x, y = dataset.get_batch(split="train", batch_size=batch_size)

    print("=" * 60)
    print("STAGE 2: EMBEDDINGS DEMO")
    print("=" * 60)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Block size: {block_size}")
    print(f"Batch size: {batch_size}")
    print(f"Embedding dimension: {embed_dim}")
    print()

    print("Raw token batch (x):")
    print(x)
    print()

    print("Decoded first example:")
    print(repr(tokenizer.decode(x[0].tolist())))
    print()

    embedding_layer = InputEmbedding(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        block_size=block_size,
    )

    embedded_x = embedding_layer(x)

    print("Embedded tensor shape:")
    print(tuple(embedded_x.shape))
    print()

    print("Expected shape = (batch_size, block_size, embed_dim)")
    print(f"Expected shape = ({batch_size}, {block_size}, {embed_dim})")
    print()

    print("First token IDs from first example:")
    print(x[0])
    print()

    print("First embedded vector from first example:")
    print(embedded_x[0, 0])
    print()

    print("Shape of one token embedding vector:")
    print(tuple(embedded_x[0, 0].shape))
    print()

    # Optional: inspect token embedding table shape
    print("Token embedding table shape:")
    print(tuple(embedding_layer.token_embedding_table.weight.shape))
    print("This is: (vocab_size, embed_dim)")
    print()

    print("Position embedding table shape:")
    print(tuple(embedding_layer.position_embedding_table.weight.shape))
    print("This is: (block_size, embed_dim)")
    print()

    # Show that position matters:
    first_example = x[0].unsqueeze(0)  # shape (1, T)
    embedded_single = embedding_layer(first_example)

    print("First 3 position-aware embeddings from first example:")
    for i in range(3):
        print(f"Position {i}, token id {first_example[0, i].item()}:")
        print(embedded_single[0, i])
        print("-" * 40)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()