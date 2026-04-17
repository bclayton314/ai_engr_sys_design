import torch


class CharTokenizer:
    """
    A very simple character-level tokenizer.

    Responsibilities:
    - Build a vocabulary from raw text
    - Encode text into integer token IDs
    - Decode token IDs back into text
    """

    def __init__(self, text: str):
        # Get a sorted list of unique characters in the dataset
        chars = sorted(list(set(text)))

        self.chars = chars
        self.vocab_size = len(chars)

        # Character -> integer ID
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # Integer ID -> character
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> list[int]:
        """
        Convert a string into a list of integer token IDs.
        """
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: list[int]) -> str:
        """
        Convert a list of integer token IDs back into a string.
        """
        return "".join(self.itos[i] for i in ids)


class TextDataset:
    """
    A lightweight dataset wrapper for next-token prediction.

    Example:
        text = "hello"
        encoded = [h, e, l, l, o]

    If block_size = 4:
        input_ids  = [h, e, l, l]
        target_ids = [e, l, l, o]

    So the model learns to predict the next character at each position.
    """

    def __init__(self, text: str, block_size: int = 8, train_split: float = 0.9):
        self.text = text
        self.block_size = block_size

        # Build tokenizer from the full corpus
        self.tokenizer = CharTokenizer(text)

        # Encode the entire text corpus into token IDs
        encoded = self.tokenizer.encode(text)
        self.data = torch.tensor(encoded, dtype=torch.long)

        # Split into train and validation sets
        split_idx = int(len(self.data) * train_split)
        self.train_data = self.data[:split_idx]
        self.val_data = self.data[split_idx:]

    def get_batch(self, split: str = "train", batch_size: int = 4):
        """
        Sample a random batch of subsequences.

        Returns:
            x: shape (batch_size, block_size)
            y: shape (batch_size, block_size)

        y is x shifted one token to the left, so each position is the "next token" target.
        """
        data = self.train_data if split == "train" else self.val_data

        if len(data) <= self.block_size:
            raise ValueError(
                f"Dataset split is too small for block_size={self.block_size}. "
                f"Need more text or a smaller block_size."
            )

        # Random starting indices for each training example
        ix = torch.randint(len(data) - self.block_size, (batch_size,))

        # Build input and target sequences
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])

        return x, y