from pathlib import Path

from data import TextDataset


def main():
    # Load raw text
    data_path = Path(__file__).parent / "data" / "input.txt"
    text = data_path.read_text(encoding="utf-8")

    # Create dataset
    dataset = TextDataset(text=text, block_size=16, train_split=0.9)

    # Tokenizer info
    tokenizer = dataset.tokenizer
    print("=" * 60)
    print("STAGE 1: TOKENIZATION + DATASET DEMO")
    print("=" * 60)
    print(f"Total characters in corpus: {len(text)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Vocabulary characters: {tokenizer.chars}")
    print()

    # Show encoding / decoding
    sample_text = text[:50]
    encoded = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(encoded)

    print("Sample raw text:")
    print(repr(sample_text))
    print()

    print("Encoded token IDs:")
    print(encoded)
    print()

    print("Decoded back to text:")
    print(repr(decoded))
    print()

    # Show a batch
    x, y = dataset.get_batch(split="train", batch_size=4)

    print("Batch shapes:")
    print(f"x shape: {tuple(x.shape)}")
    print(f"y shape: {tuple(y.shape)}")
    print()

    print("Example training pairs:")
    for i in range(x.shape[0]):
        x_ids = x[i].tolist()
        y_ids = y[i].tolist()

        x_text = tokenizer.decode(x_ids)
        y_text = tokenizer.decode(y_ids)

        print(f"Example {i + 1}")
        print(f"input_ids : {x_ids}")
        print(f"target_ids: {y_ids}")
        print(f"input_text:  {repr(x_text)}")
        print(f"target_text: {repr(y_text)}")
        print("-" * 40)


if __name__ == "__main__":
    main()