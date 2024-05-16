import argparse
from pathlib import Path

from function_calling_sft import generate_trajectories, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic multi-turn tool data")
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    train = generate_trajectories(args.train_count, "train", seed=11)
    test = generate_trajectories(args.test_count, "test", seed=101)
    write_jsonl(train, args.output_dir / "train.jsonl")
    write_jsonl(test, args.output_dir / "test.jsonl")
    print(
        {
            "train_trajectories": len(train),
            "test_trajectories": len(test),
            "train_calls": sum(len(item.function_calls()) for item in train),
            "test_calls": sum(len(item.function_calls()) for item in test),
            "output_dir": str(args.output_dir),
        }
    )


if __name__ == "__main__":
    main()
