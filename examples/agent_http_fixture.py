"""Compatibility entry point for smarthealth.examples.agent_http_fixture; business logic lives in src/."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("smarthealth.examples.agent_http_fixture", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("smarthealth.examples.agent_http_fixture")
