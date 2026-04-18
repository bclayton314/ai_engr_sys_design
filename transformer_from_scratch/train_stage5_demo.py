from pathlib import Path
import torch

from data import TextDataset
from model import InputEmbedding, MultiHeadAttention, FeedForward


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
    print("STAGE 5: FEEDFORWARD NETWORK")
    print("=" * 60)

    print("Decoded first example:")
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

    # Multi-head attention
    mha = MultiHeadAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        block_size=block_size
    )
    attn_out = mha(x_emb)

    print("After attention:", attn_out.shape)

    # Feedforward
    ffn = FeedForward(embed_dim)
    ffn_out = ffn(attn_out)

    print("After feedforward:", ffn_out.shape)
    print()

    print("First token vector BEFORE FFN:")
    print(attn_out[0, 0])
    print()

    print("First token vector AFTER FFN:")
    print(ffn_out[0, 0])
    print()

    # Inspect expansion
    print("Inspecting internal expansion:")
    linear1 = ffn.net[0]
    expanded = linear1(attn_out)

    print("Expanded shape (should be 4x embed_dim):")
    print(expanded.shape)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()