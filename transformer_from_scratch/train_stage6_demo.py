from pathlib import Path
import torch

from data import TextDataset
from model import InputEmbedding, TransformerBlock


def main():
    data_path = Path(__file__).parent / "data" / "input.txt"
    text = data_path.read_text(encoding="utf-8")

    block_size = 8
    batch_size = 2
    embed_dim = 32
    num_heads = 4

    dataset = TextDataset(text=text, block_size=block_size)
    tokenizer = dataset.tokenizer

    x, _ = dataset.get_batch(batch_size=batch_size)

    print("=" * 60)
    print("STAGE 6: TRANSFORMER BLOCK")
    print("=" * 60)

    print("Decoded input:")
    print(repr(tokenizer.decode(x[0].tolist())))
    print()

    # Embedding
    embedding = InputEmbedding(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        block_size=block_size
    )

    x_emb = embedding(x)

    print("Embedding shape:", x_emb.shape)

    # Transformer block
    block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        block_size=block_size
    )

    out = block(x_emb)

    print("Output shape after transformer block:", out.shape)
    print()

    print("First token BEFORE block:")
    print(x_emb[0, 0])
    print()

    print("First token AFTER block:")
    print(out[0, 0])
    print()

    # Check that residual connection changes but preserves structure
    diff = (out - x_emb).abs().mean()
    print("Average change (residual effect):", diff.item())


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
