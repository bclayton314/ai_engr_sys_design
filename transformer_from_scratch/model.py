from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


class InputEmbedding(nn.Module):
    """
    Token embeddings + learned positional embeddings.
    """

    def __init__(self, vocab_size: int, embed_dim: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(block_size, embed_dim)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T)
        returns: (B, T, C)
        """
        _, T = idx.shape
        if T > self.block_size:
            raise ValueError(
                f"Sequence length T={T} exceeds block_size={self.block_size}."
            )

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)  # (T, C)
        return tok_emb + pos_emb


class SelfAttentionHead(nn.Module):
    """
    Single masked self-attention head.

    Can optionally return attention weights for inspection.
    """

    def __init__(self, embed_dim: int, head_size: int, block_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        self.head_size = head_size
        self.attn_dropout = nn.Dropout(dropout)

        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, C)

        returns:
            out: (B, T, H)
        or
            (out, weights): out=(B,T,H), weights=(B,T,T)
        """
        _, T, _ = x.shape

        k = self.key(x)    # (B, T, H)
        q = self.query(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)

        wei = q @ k.transpose(-2, -1)                   # (B, T, T)
        wei = wei * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)

        out = wei @ v                                   # (B, T, H)

        if return_weights:
            return out, wei
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-head masked self-attention.

    Can optionally return attention weights from all heads.
    """

    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_size = embed_dim // num_heads

        self.heads = nn.ModuleList([
            SelfAttentionHead(embed_dim, self.head_size, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        x: (B, T, C)

        returns:
            out: (B, T, C)
        or
            (out, weights_per_head)
            where weights_per_head is a list of length num_heads,
            each entry shaped (B, T, T)
        """
        if not return_weights:
            out = torch.cat([head(x) for head in self.heads], dim=-1)
            out = self.proj(out)
            out = self.proj_dropout(out)
            return out

        head_outputs = []
        head_weights = []
        for head in self.heads:
            out_i, wei_i = head(x, return_weights=True)
            head_outputs.append(out_i)
            head_weights.append(wei_i)

        out = torch.cat(head_outputs, dim=-1)
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out, head_weights


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
        x = x + attn(LN(x))
        x = x + ffn(LN(x))
    """

    def __init__(self, embed_dim: int, num_heads: int, block_size: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        self.attn = MultiHeadAttention(embed_dim, num_heads, block_size, dropout)
        self.ffn = FeedForward(embed_dim, dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        if not return_attention:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x

        attn_out, head_weights = self.attn(self.ln1(x), return_weights=True)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, head_weights


class MiniGPT(nn.Module):
    """
    Full GPT-style character language model.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size

        self.embedding = InputEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            block_size=config.block_size,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                block_size=config.block_size,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx: (B, T)
        targets: (B, T), optional

        returns:
            logits: (B, T, vocab_size)
            loss: scalar or None
        """
        x = self.embedding(idx)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

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
        Autoregressive generation.

        idx: (B, T)
        returns: (B, T + max_new_tokens)
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")

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
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx

    @torch.no_grad()
    def forward_with_attention(
        self,
        idx: torch.Tensor,
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        """
        Forward pass that also returns attention weights.

        returns:
            logits: (B, T, vocab_size)
            all_attn: list over layers, each item is a list over heads.
                      all_attn[layer][head] has shape (B, T, T)
        """
        x = self.embedding(idx)

        all_attn: list[list[torch.Tensor]] = []
        for block in self.blocks:
            x, head_weights = block(x, return_attention=True)
            all_attn.append(head_weights)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, all_attn