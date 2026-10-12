"""Strict schema boundaries shared by independent and Agent-driven Skill calls."""

from pydantic import BaseModel


def validate_input(schema, payload):
    return schema.model_validate(payload, strict=True)


def validate_output(schema, output):
    if isinstance(output, BaseModel):
        output = output.model_dump(mode="python", warnings=False)
    return schema.model_validate(output, strict=True).model_dump(mode="json")
