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


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, ArgumentSpec] = field(default_factory=dict)

    @property
    def required(self) -> frozenset[str]:
        return frozenset(self.parameters)


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
