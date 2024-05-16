import argparse
import json
from pathlib import Path

import torch

from function_calling_sft import (
    TinyToolCaller,
    TrainingConfig,
    Vocabulary,
    default_registry,
    evaluate_model,
    flatten_examples,
    generate_trajectories,
    train_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the tiny CPU function-calling proxy")
    parser.add_argument("--train-trajectories", type=int, default=160)
    parser.add_argument("--test-trajectories", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/tiny_proxy_metrics.json"))
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    registry = default_registry()
    train = flatten_examples(generate_trajectories(args.train_trajectories, "train", 11))
    test = flatten_examples(generate_trajectories(args.test_trajectories, "test", 101))
    vocabulary = Vocabulary.from_examples(train)
    model = TinyToolCaller(len(vocabulary), registry.names())
    base = evaluate_model(model, test, vocabulary, registry)
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    losses = train_model(model, train, vocabulary, config)
    finetuned = evaluate_model(model, test, vocabulary, registry)
    payload = {
        "scope": "Tiny GRU proxy on deterministic synthetic calls; not a 3B language model",
        "configuration": {
            "train_trajectories": args.train_trajectories,
            "test_trajectories": args.test_trajectories,
            "train_calls": len(train),
            "test_calls": len(test),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "vocabulary_size": len(vocabulary),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "base": base,
        "finetuned": finetuned,
        "initial_training_loss": losses[0],
        "final_training_loss": losses[-1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
