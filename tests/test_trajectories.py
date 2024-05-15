import json

import pytest

from function_calling_sft import generate_trajectories


def test_generation_is_deterministic_multiturn_and_typed() -> None:
    first = generate_trajectories(6, seed=23)
    second = generate_trajectories(6, seed=23)
    assert first == second
    for trajectory in first:
        assert len(trajectory.messages) == 8
        assert len(trajectory.function_calls()) == 2
        assert [message.role for message in trajectory.messages] == [
            "system",
            "user",
            "assistant",
            "tool",
            "user",
            "assistant",
            "tool",
            "assistant",
        ]
        json.dumps(trajectory.to_dict())


def test_generation_uses_both_orders_and_disjoint_ids_by_split() -> None:
    train = generate_trajectories(4, split="train", seed=2)
    test = generate_trajectories(4, split="test", seed=2)
    assert train[0].function_calls()[0].name == "get_weather"
    assert train[1].function_calls()[0].name == "set_reminder"
    assert {item.trajectory_id for item in train}.isdisjoint(
        item.trajectory_id for item in test
    )


def test_invalid_generation_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        generate_trajectories(0)
