import pytest

from generate_demo_data import generate_dataset


@pytest.fixture(autouse=True)
def isolated_memory(monkeypatch):
    # App tests must never write synthetic records into the developer's persistent memory.
    monkeypatch.setenv("HEALTH_MEMORY_MODE", "ephemeral")
    monkeypatch.setenv("HEALTH_RAG_MODE", "disabled")


@pytest.fixture(scope="session")
def csv_bytes():
    frame = generate_dataset(subjects=1, sessions=1, windows=4, seed=713)
    return frame.drop(columns="label").to_csv(index=False).encode()


@pytest.fixture(scope="session")
def bundles(tmp_path_factory):
    from train import train_models

    root = tmp_path_factory.mktemp("models")
    dataset = root / "training.csv"
    generate_dataset(subjects=16, sessions=2, windows=6, seed=42).to_csv(dataset, index=False)
    train_models(dataset, root / "bundles")
    return {name: root / "bundles" / f"{name}_v1.0" for name in ("random_forest", "lightgbm")}


@pytest.fixture(scope="session")
def predictor(bundles):
    from app.models.model_loader import load_model

    return load_model(bundles["random_forest"])
