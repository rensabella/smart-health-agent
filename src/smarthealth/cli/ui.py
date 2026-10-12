"""Launch the optional Streamlit client from an installed wheel."""

import importlib.util
import sys


def main():
    try:
        from streamlit.web import cli
    except ImportError:
        raise SystemExit("Install smart-health-agent[ui] to run the frontend.") from None
    app = importlib.util.find_spec("smarthealth.frontend.streamlit_app")
    sys.argv = [
        "streamlit",
        "run",
        app.origin,
        "--server.headless=true",
        "--server.maxUploadSize=5",
        "--browser.gatherUsageStats=false",
        *sys.argv[1:],
    ]
    return cli.main()


if __name__ == "__main__":
    raise SystemExit(main())
