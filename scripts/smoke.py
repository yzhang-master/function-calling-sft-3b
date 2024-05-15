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

torch.set_num_threads(1)
torch.manual_seed(7)
registry = default_registry()
examples = flatten_examples(generate_trajectories(4, seed=3))
vocabulary = Vocabulary.from_examples(examples)
model = TinyToolCaller(len(vocabulary), registry.names())
losses = train_model(model, examples, vocabulary, TrainingConfig(epochs=20, learning_rate=0.02))
metrics = evaluate_model(model, examples, vocabulary, registry)
print(
    {
        "trajectory_calls": len(examples),
        "initial_loss": round(losses[0], 4),
        "final_loss": round(losses[-1], 4),
        "train_exact_accuracy": metrics["function_call_accuracy"],
    }
)
