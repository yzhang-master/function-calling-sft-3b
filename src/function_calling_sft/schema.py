import json
from dataclasses import dataclass, field
from typing import Any


class ValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ArgumentSpec:
    type: str
    description: str
    enum: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self) -> None:
        if self.type not in {"string", "integer", "number", "boolean"}:
            raise ValueError(f"unsupported argument type: {self.type}")


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, ArgumentSpec] = field(default_factory=dict)

    @property
    def required(self) -> frozenset[str]:
        return frozenset(self.parameters)

    def json_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {}
        for name, spec in self.parameters.items():
            prop: dict[str, Any] = {"type": spec.type, "description": spec.description}
            if spec.enum is not None:
                prop["enum"] = list(spec.enum)
            if spec.minimum is not None:
                prop["minimum"] = spec.minimum
            if spec.maximum is not None:
                prop["maximum"] = spec.maximum
            properties[name] = prop
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": sorted(self.required),
                    "additionalProperties": False,
                },
            },
        }


@dataclass(frozen=True, slots=True)
class FunctionCall:
    name: str
    arguments: dict[str, Any]


class ToolRegistry:
    def __init__(self, tools: tuple[ToolSpec, ...] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ValidationError(f"unknown tool: {name}") from error

    def validate(self, call: FunctionCall) -> FunctionCall:
        arguments = validate_arguments(self.get(call.name), call.arguments)
        return FunctionCall(call.name, arguments)

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def all(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools.values())


def validate_arguments(tool: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise ValidationError("arguments must be a JSON object")
    missing = tool.required - arguments.keys()
    unknown = arguments.keys() - tool.parameters.keys()
    if missing:
        raise ValidationError(f"missing required arguments: {sorted(missing)}")
    if unknown:
        raise ValidationError(f"unknown arguments: {sorted(unknown)}")
    validated = dict(arguments)
    for name, spec in tool.parameters.items():
        value = validated[name]
        expected = {"string": str, "integer": int, "number": (int, float), "boolean": bool}[
            spec.type
        ]
        if isinstance(value, bool) and spec.type != "boolean":
            raise ValidationError(f"{name}: expected {spec.type}")
        if not isinstance(value, expected):
            raise ValidationError(f"{name}: expected {spec.type}")
        if spec.enum is not None and value not in spec.enum:
            raise ValidationError(f"{name}: {value!r} is not in {spec.enum}")
        if spec.minimum is not None and value < spec.minimum:
            raise ValidationError(f"{name}: below minimum {spec.minimum}")
        if spec.maximum is not None and value > spec.maximum:
            raise ValidationError(f"{name}: above maximum {spec.maximum}")
    return validated


def parse_function_call(text: str, registry: ToolRegistry) -> FunctionCall:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValidationError("function call must be valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"name", "arguments"}:
        raise ValidationError("function call requires exactly name and arguments")
    if not isinstance(payload["name"], str):
        raise ValidationError("function name must be a string")
    return registry.validate(FunctionCall(payload["name"], payload["arguments"]))
