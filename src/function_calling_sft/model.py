from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from function_calling_sft.dataset import SLOTS, CallExample, Vocabulary
from function_calling_sft.schema import FunctionCall, ToolRegistry, ValidationError


class TinyToolCaller(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        tool_names: tuple[str, ...],
        embedding_dim: int = 32,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if vocabulary_size < 2 or not tool_names:
            raise ValueError("model requires a vocabulary and at least one tool")
        self.tool_names = tool_names
        self.slot_names = SLOTS
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        self.encoder = nn.GRU(
            embedding_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.tool_head = nn.Linear(2 * hidden_dim, len(tool_names))
        self.tag_head = nn.Linear(2 * hidden_dim, 1 + len(SLOTS))

    def forward(
        self, input_ids: torch.Tensor, padding_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.encoder(self.embedding(input_ids))
        mask = padding_mask.unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.tool_head(pooled), self.tag_head(encoded)

    def loss(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor,
        tool_targets: torch.Tensor,
        tag_targets: torch.Tensor,
        slot_loss_weight: float = 2.0,
    ) -> torch.Tensor:
        if slot_loss_weight <= 0:
            raise ValueError("slot_loss_weight must be positive")
        tool_logits, tag_logits = self(input_ids, padding_mask)
        tool_loss = F.cross_entropy(tool_logits, tool_targets)
        weights = torch.ones(tag_logits.shape[-1], device=tag_logits.device)
        weights[1:] = slot_loss_weight
        tag_loss = F.cross_entropy(
            tag_logits.reshape(-1, tag_logits.shape[-1]),
            tag_targets.reshape(-1),
            weight=weights,
            ignore_index=-100,
        )
        return tool_loss + tag_loss

    def predict(
        self, example: CallExample, vocabulary: Vocabulary, registry: ToolRegistry
    ) -> FunctionCall:
        ids = torch.tensor([vocabulary.encode(example.tokens)], dtype=torch.long)
        mask = ids != 0
        with torch.no_grad():
            tool_logits, tag_logits = self(ids, mask)
        tool_name = self.tool_names[int(tool_logits.argmax(dim=-1).item())]
        predicted_tags = tag_logits.argmax(dim=-1)[0].tolist()
        arguments: dict[str, Any] = {}
        for slot_index, slot_name in enumerate(self.slot_names, start=1):
            selected = [
                token
                for token, tag in zip(example.tokens, predicted_tags, strict=True)
                if tag == slot_index
            ]
            if not selected:
                continue
            value: Any = " ".join(selected)
            spec = registry.get(tool_name).parameters.get(slot_name)
            if spec is not None and spec.type in {"integer", "number"}:
                try:
                    value = int(value) if spec.type == "integer" else float(value)
                except ValueError as error:
                    raise ValidationError(f"{slot_name}: predicted nonnumeric value") from error
            arguments[slot_name] = value
        return registry.validate(FunctionCall(tool_name, arguments))


def collate(
    examples: list[CallExample], vocabulary: Vocabulary, tool_names: tuple[str, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = [vocabulary.encode(example.tokens) for example in examples]
    max_length = max(map(len, encoded))
    input_ids = torch.zeros((len(examples), max_length), dtype=torch.long)
    tag_targets = torch.full((len(examples), max_length), -100, dtype=torch.long)
    tool_targets = torch.tensor(
        [tool_names.index(example.tool_name) for example in examples], dtype=torch.long
    )
    for row, (example, ids) in enumerate(zip(examples, encoded, strict=True)):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        tag_targets[row, : len(ids)] = torch.tensor(
            [SLOTS.index(tag) + 1 if tag != "O" else 0 for tag in example.tags]
        )
    return input_ids, input_ids != 0, tool_targets, tag_targets
