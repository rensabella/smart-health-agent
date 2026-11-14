"""Compatibility entry point for smarthealth.evaluation.baselines.ungated_react; business logic lives in src/."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("smarthealth.evaluation.baselines.ungated_react", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("smarthealth.evaluation.baselines.ungated_react")
