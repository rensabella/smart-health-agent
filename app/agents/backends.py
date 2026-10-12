"""Compatibility entry point for smarthealth.agents.backends; business logic lives in src/."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("smarthealth.agents.backends", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("smarthealth.agents.backends")
