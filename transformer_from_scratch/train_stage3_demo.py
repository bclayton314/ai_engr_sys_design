from pathlib import Path
import torch
from data import TextDataset
from model import InputEmbedding, SelfAttentionHead


def main():
    data_path = Path(__file__).parent / "data" / "input.txt"
    text = data_path.read_text(encoding="utf-8")

    block_size = 8
    batch_size = 2
    embed_dim = 16
    head_size = 16  # keep same for simplicity

    dataset = TextDataset(text=text, block_size=block_size)
    tokenizer = dataset.tokenizer

    x, y = dataset.get_batch(batch_size=batch_size)

    print("=" * 60)
    print("STAGE 3: SELF-ATTENTION DEMO")
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
    print("Embedding output shape:", x_emb.shape)
    print()

    # Step 2: Attention
    attention = SelfAttentionHead(
        embed_dim=embed_dim,
        head_size=head_size,
        block_size=block_size
    )

    out = attention(x_emb)

    print("Attention output shape:", out.shape)
    print()

    print("First token output vector:")
    print(out[0, 0])
    print()

    # Inspect attention weights manually
    print("Inspecting attention weights (first example):")
    with torch.no_grad():
        B, T, C = x_emb.shape

        k = attention.key(x_emb)
        q = attention.query(x_emb)

        wei = q @ k.transpose(-2, -1)
        wei = wei * (head_size ** -0.5)
        wei = wei.masked_fill(attention.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)

        print(wei[0])  # attention matrix for first example


if __name__ == "__main__":
    torch.manual_seed(42)
    main()