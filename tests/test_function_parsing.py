import json

import pytest

from function_calling_sft import (
    FunctionCall,
    ValidationError,
    default_registry,
    parse_function_call,
)


def test_parse_validates_call_and_schema_disallows_extra_properties() -> None:
    registry = default_registry()
    text = json.dumps({"name": "get_weather", "arguments": {"city": "paris", "unit": "celsius"}})
    assert parse_function_call(text, registry) == FunctionCall(
        "get_weather", {"city": "paris", "unit": "celsius"}
    )
    schema = registry.get("get_weather").json_schema()
    assert schema["function"]["parameters"]["additionalProperties"] is False


@pytest.mark.parametrize(
    "payload",
    [
        "not json",
        "[]",
        json.dumps({"name": "get_weather"}),
        json.dumps({"name": "missing", "arguments": {}}),
    ],
)
def test_malformed_calls_are_rejected(payload: str) -> None:
    with pytest.raises(ValidationError):
        parse_function_call(payload, default_registry())
