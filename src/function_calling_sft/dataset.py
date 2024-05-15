import re
from dataclasses import dataclass
from typing import Any

from function_calling_sft.trajectories import Message, Trajectory

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+|[^\w\s]")
SLOTS = (
    "city",
    "unit",
    "title",
    "day",
    "start_hour",
    "duration_minutes",
    "task",
    "hour",
)


def tokenize(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(text.lower()))


@dataclass(frozen=True, slots=True)
class CallExample:
    trajectory_id: str
    turn_index: int
    source: str
    tokens: tuple[str, ...]
    tool_name: str
    arguments: dict[str, Any]
    tags: tuple[str, ...]


class Vocabulary:
    def __init__(self, tokens: tuple[str, ...]) -> None:
        self.tokens = ("<pad>", "<unk>", *tokens)
        self.ids = {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def from_examples(cls, examples: list[CallExample]) -> "Vocabulary":
        tokens = sorted({token for example in examples for token in example.tokens})
        return cls(tuple(tokens))

    def encode(self, tokens: tuple[str, ...]) -> list[int]:
        return [self.ids.get(token, 1) for token in tokens]

    def __len__(self) -> int:
        return len(self.tokens)


def _value_tokens(value: Any) -> tuple[str, ...]:
    return tokenize(str(value))


def label_example(
    trajectory_id: str,
    turn_index: int,
    source: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> CallExample:
    source_tokens = tokenize(source)
    tags = ["O"] * len(source_tokens)
    for argument_name, value in arguments.items():
        wanted = _value_tokens(value)
        if not wanted:
            raise ValueError(f"empty argument value for {argument_name}")
        candidates = [
            start
            for start in range(len(source_tokens) - len(wanted) + 1)
            if source_tokens[start : start + len(wanted)] == wanted
            and all(tag == "O" for tag in tags[start : start + len(wanted)])
        ]
        if not candidates:
            raise ValueError(
                f"argument {argument_name}={value!r} is not a unique unused source span"
            )
        start = candidates[0]
        tags[start : start + len(wanted)] = [argument_name] * len(wanted)
    return CallExample(
        trajectory_id, turn_index, source, source_tokens, tool_name, arguments, tuple(tags)
    )


def extract_call_examples(trajectory: Trajectory) -> list[CallExample]:
    examples: list[CallExample] = []
    last_user: Message | None = None
    for message in trajectory.messages:
        if message.role == "user":
            last_user = message
        elif message.function_call is not None:
            if last_user is None:
                raise ValueError("function call has no preceding user turn")
            examples.append(
                label_example(
                    trajectory.trajectory_id,
                    len(examples),
                    last_user.content,
                    message.function_call.name,
                    message.function_call.arguments,
                )
            )
    return examples


def flatten_examples(trajectories: list[Trajectory]) -> list[CallExample]:
    return [example for trajectory in trajectories for example in extract_call_examples(trajectory)]
