"""Compatibility entry point for smarthealth.api.memory; business logic lives in src/."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("smarthealth.api.memory", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("smarthealth.api.memory")
