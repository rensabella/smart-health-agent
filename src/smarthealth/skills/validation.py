"""Strict schema boundaries shared by independent and Agent-driven Skill calls."""

import json

from pydantic import BaseModel

from smarthealth.skills.base_skill import InputValidationError, ValidationIssue

_CATEGORIES = frozenset(
    {
        "missing",
        "extra_forbidden",
        "string_type",
        "int_type",
        "float_type",
        "bool_type",
        "list_type",
        "dict_type",
        "model_type",
        "string_too_short",
        "string_too_long",
        "string_pattern_mismatch",
        "too_short",
        "too_long",
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "literal_error",
        "finite_number",
        "value_error",
    }
)
_TYPES = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}


def input_diagnostics(schema, error):
    """Project validation failures onto trusted schema metadata, never error text/values.

    Extra keys and dictionary/union labels can contain private input. Only schema-owned
    property names and bounded array indices survive in paths. Custom validator messages
    and exception context are intentionally excluded, including for ValueError.
    """
    document = schema.model_json_schema()

    def resolve(node):
        if "$ref" in node:
            node = document.get("$defs", {}).get(node["$ref"].rsplit("/", 1)[-1], {})
        if "anyOf" in node:
            node = next((n for n in node["anyOf"] if n.get("type") != "null"), {})
            return resolve(node)
        return node

    failures = error.errors(include_url=False, include_context=False)
    issues = []
    for failure in failures[:4]:
        node, path = document, "$"
        for part in failure["loc"][:12]:
            node = resolve(node)
            if isinstance(part, str) and part in node.get("properties", {}):
                path += "." + part
                node = node["properties"][part]
            elif type(part) is int and "items" in node and 0 <= part <= 999999:
                path += f"[{part}]"
                node = node["items"]
            else:
                path += ".<unknown>"
                node = {}
                break
        node = resolve(node)
        expected = {
            k: node[k]
            for k in (
                "type",
                "minLength",
                "maxLength",
                "minItems",
                "maxItems",
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "exclusiveMaximum",
                "pattern",
                "description",
                "enum",
            )
            if k in node
        }
        if failure["type"] == "extra_forbidden":
            expected = {"constraint": "no_extra_fields"}
        issues.append(
            ValidationIssue(
                category=failure["type"] if failure["type"] in _CATEGORIES else "schema_validation",
                field_path=path[:256],
                expected=json.dumps(expected, ensure_ascii=True, separators=(",", ":"))[:400]
                if expected
                else "declared_schema_constraints",
                actual_type="missing"
                if failure["type"] == "missing"
                else _TYPES.get(type(failure.get("input")), "other"),
            )
        )
    return InputValidationError(issues=issues, truncated=len(failures) > 4)


def validate_input(schema, payload):
    return schema.model_validate(payload, strict=True)


def validate_output(schema, output):
    if isinstance(output, BaseModel):
        output = output.model_dump(mode="python", warnings=False)
    return schema.model_validate(output, strict=True).model_dump(mode="json")
