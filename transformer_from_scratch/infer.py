from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data import TextDataset
from utils import (
    build_model_from_checkpoint,
    format_attention_table,
    generate_from_prompt,
    get_device,
    load_checkpoint,
    print_attention_focus,
)


DEFAULT_DATA_PATH = Path("data/input.txt")
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/mini_gpt.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mini-GPT inference and attention inspection.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to continue from.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=20, help="Top-k sampling cutoff. Use 0 to disable.")
    parser.add_argument("--data_path", type=str, default=str(DEFAULT_DATA_PATH), help="Path to input corpus.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help="Path to saved checkpoint.",
    )
    parser.add_argument(
        "--show_attention",
        action="store_true",
        help="Print attention info for the prompt only (not the generated continuation).",
    )
    parser.add_argument("--layer", type=int, default=0, help="Layer index for attention inspection.")
    parser.add_argument("--head", type=int, default=0, help="Head index for attention inspection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()

    data_path = Path(args.data_path)
    ckpt_path = Path(args.checkpoint_path)

    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    text = data_path.read_text(encoding="utf-8")
    dataset = TextDataset(text=text, block_size=64, train_split=0.9)

    checkpoint = load_checkpoint(ckpt_path, device=device)
    model = build_model_from_checkpoint(checkpoint, device=device)
    model.eval()

    top_k = None if args.top_k == 0 else args.top_k

    print("=" * 72)
    print("FINAL STAGE: INFERENCE")
    print("=" * 72)
    print(f"Device:     {device}")
    print(f"Checkpoint: {ckpt_path}")
    print()

    generated = generate_from_prompt(
        model=model,
        dataset=dataset,
        prompt=args.prompt,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=top_k,
    )

    print("PROMPT")
    print(repr(args.prompt))
    print()
    print("GENERATED TEXT")
    print(generated)
    print()

    if args.show_attention:
        tokenizer = dataset.tokenizer
        for ch in args.prompt:
            if ch not in tokenizer.stoi:
                raise ValueError(
                    f"Prompt contains unseen character {repr(ch)}. "
                    "Cannot inspect attention for out-of-vocabulary characters."
                )

        idx = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long, device=device)
        _, all_attn = model.forward_with_attention(idx)

        if args.layer < 0 or args.layer >= len(all_attn):
            raise ValueError(f"Invalid layer index {args.layer}. Model has {len(all_attn)} layers.")

        layer_attn = all_attn[args.layer]

        if args.head < 0 or args.head >= len(layer_attn):
            raise ValueError(
                f"Invalid head index {args.head}. Layer has {len(layer_attn)} heads."
            )

        weights = layer_attn[args.head][0].cpu()  # (T, T)
        tokens = list(args.prompt)

        print("=" * 72)
        print(f"ATTENTION INSPECTION | layer={args.layer} head={args.head}")
        print("=" * 72)
        print(format_attention_table(tokens, weights, decimals=2))
        print()
        print_attention_focus(tokens, weights, top_n=3)


if __name__ == "__main__":
    main()