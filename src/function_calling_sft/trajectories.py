import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from function_calling_sft.schema import FunctionCall, ToolRegistry
from function_calling_sft.tools import DAYS, default_registry

_CITIES = ("paris", "new york", "tokyo", "berlin", "london", "san francisco")
_TITLES = ("project sync", "design review", "research standup", "planning session")
_TASKS = ("water plants", "submit report", "review pull request", "backup notes")
_DURATIONS = (15, 30, 45, 60, 90, 120)


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str = ""
    function_call: FunctionCall | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.function_call is not None:
            payload["function_call"] = asdict(self.function_call)
        if self.name is not None:
            payload["name"] = self.name
        return payload


@dataclass(frozen=True, slots=True)
class Trajectory:
    trajectory_id: str
    split: str
    messages: tuple[Message, ...]

    def function_calls(self) -> tuple[FunctionCall, ...]:
        return tuple(
            message.function_call
            for message in self.messages
            if message.function_call is not None
        )

    def validate(self, registry: ToolRegistry) -> None:
        if len(self.messages) < 7:
            raise ValueError("a multi-turn trajectory requires at least seven messages")
        if len(self.function_calls()) != 2:
            raise ValueError("expected exactly two function calls")
        for call in self.function_calls():
            registry.validate(call)
        for index, message in enumerate(self.messages):
            if message.role == "tool":
                if index == 0 or self.messages[index - 1].function_call is None:
                    raise ValueError("tool response must immediately follow a function call")
                if message.name != self.messages[index - 1].function_call.name:
                    raise ValueError("tool response name does not match its call")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "split": self.split,
            "messages": [message.to_dict() for message in self.messages],
        }


def generate_trajectories(
    count: int, split: str = "train", seed: int = 11, registry: ToolRegistry | None = None
) -> list[Trajectory]:
    if count <= 0:
        raise ValueError("count must be positive")
    registry = registry or default_registry()
    rng = random.Random(seed)
    trajectories: list[Trajectory] = []
    for index in range(count):
        city = rng.choice(_CITIES)
        unit = rng.choice(("celsius", "fahrenheit"))
        day = rng.choice(DAYS)
        start_hour = rng.randint(7, 18)
        duration = rng.choice(_DURATIONS)
        title = rng.choice(_TITLES)
        task = rng.choice(_TASKS)
        weather = FunctionCall("get_weather", {"city": city, "unit": unit})
        meeting = FunctionCall(
            "schedule_meeting",
            {
                "title": title,
                "day": day,
                "start_hour": start_hour,
                "duration_minutes": duration,
            },
        )
        reminder = FunctionCall(
            "set_reminder", {"task": task, "day": day, "hour": start_hour}
        )
        weather_user = Message("user", f"What is the weather in {city} in {unit}?")
        weather_tool = Message(
            "tool",
            json.dumps({"temperature": 18 + index % 9, "unit": unit}, sort_keys=True),
            name="get_weather",
        )
        if index % 2 == 0:
            first_user, first_call, first_tool = weather_user, weather, weather_tool
            second_user = Message(
                "user",
                f"Schedule '{title}' on {day} at {start_hour} for {duration} minutes.",
            )
            second_call = meeting
            second_tool = Message(
                "tool", json.dumps({"status": "scheduled", "title": title}), name="schedule_meeting"
            )
        else:
            first_user = Message("user", f"Remind me to {task} on {day} at {start_hour}.")
            first_call = reminder
            first_tool = Message(
                "tool", json.dumps({"status": "reminder_set", "task": task}), name="set_reminder"
            )
            second_user, second_call, second_tool = weather_user, weather, weather_tool
        trajectory = Trajectory(
            trajectory_id=f"{split}-{seed}-{index:04d}",
            split=split,
            messages=(
                Message("system", "You are a precise local tool agent."),
                first_user,
                Message("assistant", function_call=first_call),
                first_tool,
                second_user,
                Message("assistant", function_call=second_call),
                second_tool,
                Message("assistant", "Done."),
            ),
        )
        trajectory.validate(registry)
        trajectories.append(trajectory)
    return trajectories


def write_jsonl(trajectories: list[Trajectory], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item.to_dict()) + "\n" for item in trajectories))
