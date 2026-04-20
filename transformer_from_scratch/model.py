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

        if T > self.block_size:
            raise ValueError(f"Sequence length {T} exceeds block_size={self.block_size}")

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)  # (T, C)

        return tok_emb + pos_emb  # (B, T, C)


class SelfAttentionHead(nn.Module):
    """
    Single causal self-attention head.
    """

    def __init__(self, embed_dim: int, head_size: int, block_size: int, dropout: float):
        super().__init__()

        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        self.head_size = head_size
        self.attn_dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        k = self.key(x)    # (B, T, H)
        q = self.query(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)

        wei = q @ k.transpose(-2, -1)                    # (B, T, T)
        wei = wei * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v                                    # (B, T, H)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head causal self-attention.
    """

    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        head_size = embed_dim // num_heads

        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(x) for head in self.heads], dim=-1)  # (B, T, C)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class FeedForward(nn.Module):
    """
    Position-wise feedforward network.
    """

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block:
        x = x + attention(layernorm(x))
        x = x + ffn(layernorm(x))
    """

    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadAttention(embed_dim, num_heads, block_size, dropout)
        self.ffn = FeedForward(embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        block_size: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.block_size = block_size

        self.embedding = InputEmbedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            block_size=block_size,
        )

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embedding(idx)      # (B, T, C)
        x = self.blocks(x)           # (B, T, C)
        x = self.ln_f(x)             # (B, T, C)
        logits = self.lm_head(x)     # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_flat = logits.view(B * T, C)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with optional temperature and top-k sampling.

        Args:
            idx: (B, T)
            max_new_tokens: number of tokens to generate
            temperature: controls randomness; lower = sharper
            top_k: if set, only sample from the k most likely next tokens
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]  # (B, vocab_size)
            logits = logits / temperature

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                top_vals, _ = torch.topk(logits, k)
                cutoff = top_vals[:, [-1]]
                logits = logits.masked_fill(logits < cutoff, float("-inf"))

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx