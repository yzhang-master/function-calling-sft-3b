import pytest

from function_calling_sft import flatten_examples, generate_trajectories
from function_calling_sft.dataset import label_example


def test_every_argument_is_recovered_from_its_source_spans() -> None:
    examples = flatten_examples(generate_trajectories(12, seed=31))
    assert len(examples) == 24
    for example in examples:
        assert len(example.tokens) == len(example.tags)
        for argument, expected in example.arguments.items():
            actual = " ".join(
                token
                for token, tag in zip(example.tokens, example.tags, strict=True)
                if tag == argument
            )
            assert actual == str(expected)


def test_missing_or_conflicting_argument_span_fails_data_invariant() -> None:
    with pytest.raises(ValueError, match="source span"):
        label_example("t", 0, "Weather in Paris", "get_weather", {"city": "London"})
    with pytest.raises(ValueError, match="source span"):
        label_example(
            "t",
            0,
            "at 9",
            "schedule_meeting",
            {"start_hour": 9, "duration_minutes": 9},
        )


def test_unknown_tokens_map_to_unk_instead_of_pad() -> None:
    from function_calling_sft import Vocabulary

    vocabulary = Vocabulary(("known",))
    assert vocabulary.encode(("known", "missing")) == [2, 1]
