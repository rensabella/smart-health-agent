"""Compatibility entry point for smarthealth.cli.build_knowledge_index; business logic lives in src/."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("smarthealth.cli.build_knowledge_index", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("smarthealth.cli.build_knowledge_index")
