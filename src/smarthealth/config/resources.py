"""Locate immutable assets in both editable installs and installed wheels."""

from importlib.resources import files
from pathlib import Path


def resource_path(*parts: str) -> Path:
    """Return a bundled resource path (pip installs wheels as ordinary directories)."""
    # Canonicalize the trusted package installation prefix (e.g. macOS /var ->
    # /private/var). External model/index paths keep their own strict validation.
    return Path(str(files("smarthealth.resources").joinpath(*parts))).resolve()
