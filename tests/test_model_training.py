import torch

from function_calling_sft import (
    TinyToolCaller,
    TrainingConfig,
    Vocabulary,
    collate,
    default_registry,
    evaluate_model,
    flatten_examples,
    generate_trajectories,
    train_model,
)


def setup_module() -> None:
    torch.set_num_threads(1)


def test_loss_is_differentiable_and_trains_tool_and_slot_heads() -> None:
    registry = default_registry()
    examples = flatten_examples(generate_trajectories(4, seed=5))
    vocabulary = Vocabulary.from_examples(examples)
    torch.manual_seed(4)
    model = TinyToolCaller(len(vocabulary), registry.names())
    loss = model.loss(*collate(examples, vocabulary, registry.names()))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.tool_head.weight.grad is not None
    assert model.tag_head.weight.grad is not None
    assert model.tool_head.weight.grad.abs().sum() > 0
    assert model.tag_head.weight.grad.abs().sum() > 0


def test_small_torch_model_really_finetunes_on_trajectory_calls() -> None:
    registry = default_registry()
    examples = flatten_examples(generate_trajectories(12, seed=19))
    vocabulary = Vocabulary.from_examples(examples)
    torch.manual_seed(7)
    model = TinyToolCaller(len(vocabulary), registry.names())
    initial_loss = model.loss(*collate(examples, vocabulary, registry.names())).item()
    losses = train_model(
        model,
        examples,
        vocabulary,
        TrainingConfig(epochs=50, batch_size=24, learning_rate=0.02, seed=7),
    )
    metrics = evaluate_model(model, examples, vocabulary, registry)
    assert losses[-1] < initial_loss * 0.2
    assert metrics["function_call_accuracy"] >= 0.75
    assert metrics["valid_call_rate"] >= 0.75
