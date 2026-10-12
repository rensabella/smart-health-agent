"""Compatibility entry point for smarthealth.reports.schemas; business logic lives in src/."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("smarthealth.reports.schemas", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("smarthealth.reports.schemas")
