from pathlib import Path
import torch

from data import TextDataset
from model import InputEmbedding, MultiHeadAttention


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
    print("STAGE 4: MULTI-HEAD ATTENTION")
    print("=" * 60)

    print("Input tokens:")
    print(x)
    print()

    print("Decoded first example:")
    print(repr(tokenizer.decode(x[0].tolist())))
    print()

    # Step 1: Embedding
    embedding = InputEmbedding(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim,
        block_size=block_size
    )

    x_emb = embedding(x)

    print("Embedding shape:", x_emb.shape)
    print()

    # Step 2: Multi-head attention
    mha = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        block_size=block_size
    )

    out = mha(x_emb)

    print("Multi-head output shape:", out.shape)
    print()

    print("Expected shape = (batch_size, block_size, embed_dim)")
    print()

    print("First output vector:")
    print(out[0, 0])
    print()

    # Inspect head outputs individually
    print("Inspecting individual head outputs:")
    for i, head in enumerate(mha.heads):
        head_out = head(x_emb)
        print(f"Head {i} output shape:", head_out.shape)
    print()

    print("Concatenated shape check:")
    concat = torch.cat([h(x_emb) for h in mha.heads], dim=-1)
    print(concat.shape)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
