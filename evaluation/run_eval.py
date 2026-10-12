"""Compatibility entry point for smarthealth.evaluation.run_eval; business logic lives in src/."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("smarthealth.evaluation.run_eval", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("smarthealth.evaluation.run_eval")
