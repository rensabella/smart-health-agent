"""Compatibility entry point for smarthealth.examples.run_agent; business logic lives in src/."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("smarthealth.examples.run_agent", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("smarthealth.examples.run_agent")
