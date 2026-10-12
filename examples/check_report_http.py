"""Compatibility entry point for smarthealth.examples.check_report_http; business logic lives in src/."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("smarthealth.examples.check_report_http", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("smarthealth.examples.check_report_http")
