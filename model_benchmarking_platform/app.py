import argparse
import json

from config import load_config
from train import run_training


def main():
    parser = argparse.ArgumentParser(description="Run an ML benchmarking experiment.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment config JSON file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_training(config)

    print("\n=== Run Complete ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
