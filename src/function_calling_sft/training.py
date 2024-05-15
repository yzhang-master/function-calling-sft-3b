import random
from dataclasses import dataclass
from typing import Any

import torch

from function_calling_sft.dataset import CallExample, Vocabulary
from function_calling_sft.model import TinyToolCaller, collate
from function_calling_sft.schema import ToolRegistry, ValidationError


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 80
    batch_size: int = 128
    learning_rate: float = 0.01
    slot_loss_weight: float = 2.0
    seed: int = 7

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")


def train_model(
    model: TinyToolCaller,
    examples: list[CallExample],
    vocabulary: Vocabulary,
    config: TrainingConfig,
) -> tuple[float, ...]:
    if not examples:
        raise ValueError("training requires at least one example")
    rng = random.Random(config.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: list[float] = []
    for _ in range(config.epochs):
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, len(indices), config.batch_size):
            batch = [examples[index] for index in indices[start : start + config.batch_size]]
            optimizer.zero_grad()
            loss = model.loss(
                *collate(batch, vocabulary, model.tool_names),
                slot_loss_weight=config.slot_loss_weight,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            batches += 1
        losses.append(epoch_loss / batches)
    return tuple(losses)


def evaluate_model(
    model: TinyToolCaller,
    examples: list[CallExample],
    vocabulary: Vocabulary,
    registry: ToolRegistry,
) -> dict[str, Any]:
    model.eval()
    exact = 0
    tool_correct = 0
    valid = 0
    failures: list[dict[str, str]] = []
    for example in examples:
        try:
            predicted = model.predict(example, vocabulary, registry)
            valid += 1
        except ValidationError:
            predicted = None
        tool_correct += predicted is not None and predicted.name == example.tool_name
        matches = (
            predicted is not None
            and predicted.name == example.tool_name
            and predicted.arguments == example.arguments
        )
        exact += matches
        if not matches and len(failures) < 8:
            failures.append(
                {
                    "trajectory_id": example.trajectory_id,
                    "expected_tool": example.tool_name,
                    "predicted": str(predicted),
                }
            )
    total = len(examples)
    return {
        "examples": total,
        "exact_calls": exact,
        "function_call_accuracy": exact / total,
        "tool_accuracy": tool_correct / total,
        "valid_call_rate": valid / total,
        "failure_examples": failures,
    }
