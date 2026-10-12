"""Compatibility entry point for smarthealth.runtime.harness.harness_validator; business logic lives in src/."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("smarthealth.runtime.harness.harness_validator", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("smarthealth.runtime.harness.harness_validator")
