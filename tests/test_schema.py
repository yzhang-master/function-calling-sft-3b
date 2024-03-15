import pytest

from function_calling_sft import ArgumentSpec, ToolSpec, ValidationError, validate_arguments


def weather_tool() -> ToolSpec:
    return ToolSpec(
        "get_weather",
        "Read local weather",
        {
            "city": ArgumentSpec("string", "City"),
            "unit": ArgumentSpec("string", "Unit", enum=("celsius", "fahrenheit")),
        },
    )


def test_valid_arguments_are_returned_without_mutation() -> None:
    arguments = {"city": "Paris", "unit": "celsius"}
    assert validate_arguments(weather_tool(), arguments) == arguments
    assert arguments == {"city": "Paris", "unit": "celsius"}


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"city": "Paris"}, "missing"),
        ({"city": "Paris", "unit": "kelvin"}, "not in"),
        ({"city": "Paris", "unit": "celsius", "extra": 1}, "unknown"),
        ({"city": True, "unit": "celsius"}, "expected string"),
    ],
)
def test_invalid_arguments_fail_closed(arguments: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        validate_arguments(weather_tool(), arguments)
