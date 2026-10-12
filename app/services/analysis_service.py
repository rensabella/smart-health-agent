"""Compatibility entry point for smarthealth.services.analysis_service; business logic lives in src/."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("smarthealth.services.analysis_service", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("smarthealth.services.analysis_service")
