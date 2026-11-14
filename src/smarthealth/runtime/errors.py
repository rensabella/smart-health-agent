"""Stable error codes with messages safe to expose at API and Skill boundaries."""


class SkillError(Exception):
    """A known domain failure; never put raw inputs or exception reprs in its message.

    ``details`` carries bounded, sanitized context for guards and callers: keys are
    stable snake_case reason codes or reference field names, values are JSON
    primitives (issue lists must be ``model_dump(mode="json")`` projections), and at
    most a handful of keys survive. Never raw inputs, provider text or values.
    """

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details
