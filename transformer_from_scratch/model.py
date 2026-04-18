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


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention:
    - runs multiple attention heads in parallel
    - concatenates their outputs
    - projects back to embedding dimension

    Input:
        x: (B, T, C)

    Output:
        out: (B, T, C)
    """

    def __init__(self, embed_dim: int, num_heads: int, block_size: int):
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        # Create multiple attention heads
        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, self.head_size, block_size)
            for _ in range(num_heads)
        ])

        # Final projection layer
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, T, C)
        """

        # Run each head independently
        head_outputs = [h(x) for h in self.heads]
        # Each: (B, T, head_size)

        # Concatenate along feature dimension
        out = torch.cat(head_outputs, dim=-1)
        # Now: (B, T, C)

        # Final projection
        out = self.proj(out)

        return out


class FeedForward(nn.Module):
    """
    Position-wise feedforward network.

    Applies independently to each token.

    Input:
        x: (B, T, C)

    Output:
        out: (B, T, C)
    """

    def __init__(self, embed_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),  # expand
            nn.GELU(),                            # non-linearity
            nn.Linear(4 * embed_dim, embed_dim)   # project back
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


