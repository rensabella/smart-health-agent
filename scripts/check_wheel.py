"""Install the wheel in a disposable venv and exercise it outside the repository.

Reuse this interpreter's dependencies, not its editable project installation.
The isolated interpreter has no checkout or src path; package identity is checked.
This is an installation smoke test, not a clean dependency-resolution benchmark.
"""

import argparse
import os
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path

SMOKE = r"""
import json
from pathlib import Path
import sys
import smarthealth
from fastapi.testclient import TestClient
from smarthealth.api.application import create_app
from smarthealth.cli.generate_demo_data import generate_dataset
from smarthealth.cli.train import train_models
from smarthealth.config.resources import resource_path
from smarthealth.config.settings import Settings
from smarthealth.evaluation.cases import load_cases
from smarthealth.evaluation.live_llm.tasks import load_tasks
from smarthealth.evaluation.live_llm.runner import main as live_eval_main
from smarthealth.models.model_loader import load_model
from smarthealth.runtime.harness.config import load_config

package = Path(smarthealth.__file__).resolve()
assert package.is_relative_to(Path(sys.prefix).resolve()), package
assert not any(Path(p).resolve() == Path(sys.argv[1]) for p in sys.path)
settings = Settings.from_env({"HEALTH_HOME": str(Path.cwd()), "HEALTH_MEMORY_MODE": "ephemeral"})
load_config(settings.harness_config_dir)
load_config(resource_path("configs", "report"))
load_cases(resource_path("evaluation", "test_cases.json"), root=resource_path())
assert len(load_tasks()[0].tasks) == 20
assert live_eval_main(["--dry-run", "--max-tasks", "20", "--output", "live-eval-dry"]) == 0
frame = generate_dataset(subjects=16, sessions=2, windows=6, seed=42)
dataset = Path("training.csv")
frame.to_csv(dataset, index=False)
train_models(dataset, Path("models"), backends=("random_forest",))
predictor = load_model(Path("models/random_forest_v1.0"))
app = create_app(predictor=predictor, settings=settings)
content = resource_path("data", "sample", "demo.csv").read_bytes()
with TestClient(app) as client:
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200
    prediction = client.post("/predict", files={"file": ("demo.csv", content)})
    assert prediction.status_code == 200, prediction.text
    assert 0 <= prediction.json()["risk_score"] <= 1
    analysis = client.post("/analyze?include_visualizations=true", files={"file": ("demo.csv", content)})
    assert analysis.status_code == 200, analysis.text
    assert len(analysis.json()["visualization"]) == 3
    image = client.get(analysis.json()["visualization"][0]["url"])
    assert image.status_code == 200 and image.content.startswith(b"\x89PNG")
    assert app.state.agent_runtime.backend.mode == "disabled"
assert not (package.parent / "logs").exists()
print(json.dumps({"wheel_version": smarthealth.__version__, "installed_outside_checkout": True,
                  "real_training_and_inference": True, "api": ["health", "ready", "predict", "analyze"]}))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        assert all(
            name.startswith(("smarthealth/", "smart_health_agent-")) for name in archive.namelist()
        )
        assert "smarthealth/resources/configs/report/agent.yaml" in archive.namelist()
    with tempfile.TemporaryDirectory(prefix="smarthealth-wheel-") as temporary:
        root = Path(temporary).resolve()
        venv = root / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        site = subprocess.check_output(
            [str(python), "-I", "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
            text=True,
        ).strip()
        # Python does not process .pth files inside a directory added by another .pth.
        # This shares installed dependencies while keeping editable project paths absent.
        Path(site, "smoke_dependencies.pth").write_text(sysconfig.get_paths()["purelib"] + "\n")
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("HEALTH_") and key != "PYTHONPATH"
        }
        env["MPLCONFIGDIR"] = str(root / "matplotlib")
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-index",
                "--ignore-installed",
                str(wheel),
            ],
            check=True,
            cwd=root,
            env=env,
        )
        source = Path(__file__).resolve().parents[1] / "src"
        subprocess.run([str(python), "-I", "-c", SMOKE, str(source)], check=True, cwd=root, env=env)
        for command in (
            "smarthealth-api",
            "smarthealth-generate",
            "smarthealth-train",
            "smarthealth-index",
            "smarthealth-eval",
            "smarthealth-eval-live",
        ):
            entry = python.parent / (command + ".exe" if os.name == "nt" else command)
            subprocess.run(
                [str(entry), "--help"], check=True, cwd=root, env=env, stdout=subprocess.DEVNULL
            )
        print(
            "Wheel installation, packaged resources, real model/API behavior and console entry points passed."
        )


if __name__ == "__main__":
    main()
