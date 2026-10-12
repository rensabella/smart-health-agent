"""Stable error codes with messages safe to expose at API and Skill boundaries."""


class SkillError(Exception):
    """A known domain failure; never put raw inputs or exception reprs in its message."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message
