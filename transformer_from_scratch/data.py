from __future__ import annotations

import torch


class CharTokenizer:
    """
    Simple character-level tokenizer.

    Builds a vocabulary from the training text and provides:
    - encode(text) -> list[int]
    - decode(ids)  -> str
    """

    def __init__(self, text: str):
        if not text:
            raise ValueError("Tokenizer cannot be built from empty text.")

        chars = sorted(set(text))
        self.chars = chars
        self.vocab_size = len(chars)

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        try:
            return [self.stoi[ch] for ch in s]
        except KeyError as e:
            raise ValueError(f"Character {repr(e.args[0])} not in tokenizer vocabulary.") from e

    def decode(self, ids: list[int]) -> str:
        try:
            return "".join(self.itos[i] for i in ids)
        except KeyError as e:
            raise ValueError(f"Token id {e.args[0]} not in tokenizer vocabulary.") from e


class TextDataset:
    """
    Character-level next-token dataset.

    Given a sequence:
        x = [t0, t1, t2, ..., t_{n-1}]
        y = [t1, t2, t3, ..., t_n]

    So the model learns: predict the next token at every position.
    """

    def __init__(self, text: str, block_size: int = 64, train_split: float = 0.9):
        if not text:
            raise ValueError("TextDataset requires non-empty text.")
        if not (0.0 < train_split < 1.0):
            raise ValueError("train_split must be between 0 and 1.")
        if block_size < 1:
            raise ValueError("block_size must be >= 1.")

        self.text = text
        self.block_size = block_size
        self.tokenizer = CharTokenizer(text)

        encoded = self.tokenizer.encode(text)
        self.data = torch.tensor(encoded, dtype=torch.long)

        split_idx = int(len(self.data) * train_split)
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]

        if len(self.train_data) <= block_size + 1:
            raise ValueError(
                "Training split is too small for the given block_size. "
                "Use more text or reduce block_size."
            )
        if len(self.val_data) <= block_size + 1:
            raise ValueError(
                "Validation split is too small for the given block_size. "
                "Use more text or reduce block_size."
            )

    def get_batch(self, split: str = "train", batch_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        data = self.train_data if split == "train" else self.val_data

        max_start = len(data) - self.block_size - 1
        if max_start <= 0:
            raise ValueError(
                f"Dataset split '{split}' is too small for block_size={self.block_size}."
            )

        ix = torch.randint(0, max_start + 1, (batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y