import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, block_size: int):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)

        self.block_size = block_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)      # (B, T, C)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)   # (T, C)

        return tok_emb + pos_emb                       # (B, T, C)


class SelfAttentionHead(nn.Module):
    """
    Single-head self-attention.

    Input:
        x: (B, T, C)

    Output:
        out: (B, T, head_size)
    """

    def __init__(self, embed_dim: int, head_size: int, block_size: int):
        super().__init__()

        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        self.head_size = head_size

        # Causal mask (lower triangular matrix)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Compute Q, K, V
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1)   # (B, T, T)

        # Scale
        wei = wei * (self.head_size ** -0.5)

        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Softmax
        wei = F.softmax(wei, dim=-1)   # (B, T, T)

        # Weighted sum of values
        out = wei @ v                  # (B, T, head_size)

        return out