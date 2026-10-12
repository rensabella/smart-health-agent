"""Compatibility entry point for smarthealth.frontend.streamlit_app; business logic lives in src/."""

import importlib
import runpy
import sys

if __name__ == "__main__":
    runpy.run_module("smarthealth.frontend.streamlit_app", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("smarthealth.frontend.streamlit_app")
