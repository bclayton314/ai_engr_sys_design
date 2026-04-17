import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    """
    Converts token IDs into dense vectors and adds positional embeddings.

    Input:
        idx: shape (batch_size, block_size)

    Output:
        x: shape (batch_size, block_size, embed_dim)
    """

    def __init__(self, vocab_size: int, embed_dim: int, block_size: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.block_size = block_size

        # Learned token embeddings:
        # each token ID maps to a trainable vector of length embed_dim
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)

        # Learned positional embeddings:
        # each position 0..block_size-1 maps to a trainable vector
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx shape: (B, T)
        returns shape: (B, T, C)
        where:
            B = batch size
            T = sequence length
            C = embedding dimension
        """
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(
                f"Sequence length T={T} exceeds block_size={self.block_size}"
            )

        # Token embeddings: (B, T, C)
        tok_emb = self.token_embedding_table(idx)

        # Position indices: (T,)
        pos = torch.arange(T, device=idx.device)

        # Positional embeddings: (T, C)
        pos_emb = self.position_embedding_table(pos)

        # Broadcasting adds (T, C) to every batch item -> (B, T, C)
        x = tok_emb + pos_emb
        return x