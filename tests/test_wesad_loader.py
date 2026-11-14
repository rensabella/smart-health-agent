"""Fake-pickle loading, canonical-table semantics and the prepare CLI gate."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from wesad_helpers import _chest_signals, write_fake_subject

from smarthealth.runtime.errors import SkillError
from smarthealth.wesad.loader import (
    CANONICAL_COLUMNS,
    DERIVATION_VERSION,
    EXPECTED_SUBJECTS,
    load_subject,
    parse_sha256_file,
    write_canonical_csv,
)
from smarthealth.wesad.prepare import _subject_pickle, prepare


def test_expected_subject_set_matches_the_public_release():
    assert len(EXPECTED_SUBJECTS) == 15
    assert EXPECTED_SUBJECTS[0] == "S2" and EXPECTED_SUBJECTS[-1] == "S17"
    assert "S1" not in EXPECTED_SUBJECTS and "S12" not in EXPECTED_SUBJECTS


def test_load_subject_canonical_semantics(tmp_path):
    path = write_fake_subject(
        tmp_path,
        "S2",
        seconds=90,
        blocks=[(30, 1), (30, 2), (30, 3)],
        transient_samples=400,
        tail_samples=123,
        timestamp=datetime(2017, 6, 13, 9, 30, tzinfo=timezone.utc),
        scr_second=45,
    )
    record = load_subject(path, expected_subject="S2")
    frame = record.frame
    assert record.subject == "S2"
    assert record.dropped_partial_samples == 123
    assert list(frame.columns) == list(CANONICAL_COLUMNS)
    assert len(frame) == 90
    labels = frame["label"].to_numpy(dtype=float)
    assert np.isnan(labels[[29, 30, 59, 60]]).all()
    assert (labels[:29] == 0).all()
    assert (labels[31:59] == 1).all()
    assert (labels[61:] == 2).all()
    hr = frame["hr"].to_numpy(dtype=float)
    valid = hr[~np.isnan(hr)]
    assert len(valid) >= 88
    # The very first interval carries a filter edge effect; the rest sit at the
    # pulse rate of the synthetic generator (42000 / 583 = 72.04 bpm).
    assert np.all((valid > 70.0) & (valid < 75.0))
    assert 71.5 < float(np.median(valid)) < 72.6
    scr = frame["scr"].to_numpy(dtype=float)
    assert scr.sum() == 1
    assert 45 <= int(np.flatnonzero(scr)[0]) <= 46
    assert abs(frame["eda"].iloc[0] - 5.005) < 0.02
    assert abs(frame["temp"].iloc[0] - 36.5) < 0.06
    assert 0.3 < frame["resp"].min() and frame["resp"].max() < 0.8
    assert 0.36 < frame["acc_mag"].mean() < 0.39
    assert 0.04 < frame["emg"].min() and frame["emg"].max() < 0.07
    assert frame["timestamp"].iloc[0] == "2017-06-13T09:30:00Z"
    assert frame["timestamp"].iloc[1] == "2017-06-13T09:30:01Z"
    assert set(frame["subject_id"]) == {"S2"}
    assert set(frame["session_id"]) == {"protocol"}
    assert record.wrist_available
    assert record.wrist_signals == ("BVP", "EDA", "TEMP")


def test_load_subject_fallback_epoch_top_level_label_and_extras(tmp_path):
    path = write_fake_subject(
        tmp_path,
        "S3",
        seconds=45,
        blocks=[(15, 1), (5, 7), (25, 2)],
        scr_second=5,
        top_level_label=True,
        wrist=False,
        extra_top={"score": 1},
    )
    record = load_subject(path, expected_subject="S3")
    frame = record.frame
    assert frame["timestamp"].iloc[0] == "1970-01-01T00:00:00Z"
    assert not record.wrist_available and record.wrist_signals == ()
    assert "score" in record.extra_top_level_keys
    labels = frame["label"].to_numpy(dtype=float)
    # A uniform meditation second never maps into the three-class vocabulary.
    assert labels[14] == 0
    assert np.isnan(labels[15:20]).all()
    assert (labels[20:] == 1).all()


def test_load_subject_is_deterministic(tmp_path):
    path = write_fake_subject(tmp_path, "S2", seconds=90, seed=3)
    frames = [load_subject(path, expected_subject="S2").frame for _ in range(2)]
    paths = [tmp_path / "first.csv", tmp_path / "second.csv"]
    for frame, target in zip(frames, paths):
        write_canonical_csv(frame, target)
    assert paths[0].read_bytes() == paths[1].read_bytes()


def _minimal_data(subject: str = "S2", seconds: int = 5) -> dict:
    signals = _chest_signals(seconds * 700, hr_bpm=72.0, scr_second=1, seed=0)
    return {
        "subject": subject,
        "signal": {"chest": {**signals, "label": np.full(seconds * 700, 1)}},
    }


def test_released_pickle_layout_is_normalized_to_identical_bytes(tmp_path):
    """Column-vector channels plus a top-level label must match the flat layout byte for byte."""
    released = write_fake_subject(tmp_path, "S2", seconds=45, seed=5)
    legacy = write_fake_subject(
        tmp_path / "legacy",
        "S2",
        seconds=45,
        seed=5,
        top_level_label=False,
        column_vectors=False,
    )
    first = load_subject(released, expected_subject="S2")
    second = load_subject(legacy, expected_subject="S2")
    paths = [tmp_path / "released.csv", tmp_path / "legacy.csv"]
    write_canonical_csv(first.frame, paths[0])
    write_canonical_csv(second.frame, paths[1])
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_load_subject_rejects_broken_inputs(tmp_path):
    def drop_signal(data):
        del data["signal"]

    def drop_subject(data):
        del data["subject"]

    def bad_subject(data):
        data["subject"] = 42

    def empty_subject(data):
        data["subject"] = "   "

    def drop_ecg(data):
        del data["signal"]["chest"]["ECG"]

    def drop_label(data):
        del data["signal"]["chest"]["label"]

    def flat_acc(data):
        data["signal"]["chest"]["ACC"] = np.zeros((3500, 2))

    def wide_eda(data):
        data["signal"]["chest"]["EDA"] = np.zeros((3500, 2))

    def short_eda(data):
        data["signal"]["chest"]["EDA"] = np.zeros(3400)

    for mutation in (
        drop_signal,
        drop_subject,
        bad_subject,
        empty_subject,
        drop_ecg,
        drop_label,
        flat_acc,
        wide_eda,
        short_eda,
    ):
        data = pickle.loads(pickle.dumps(_minimal_data()))
        mutation(data)
        path = tmp_path / "broken.pkl"
        with path.open("wb") as stream:
            pickle.dump(data, stream)
        with pytest.raises(SkillError):
            load_subject(path, expected_subject="S2")

    garbage = tmp_path / "garbage.pkl"
    garbage.write_bytes(b"\x00\x01not a pickle")
    with pytest.raises(SkillError):
        load_subject(garbage)
    listed = tmp_path / "list.pkl"
    with listed.open("wb") as stream:
        pickle.dump([1, 2, 3], stream)
    with pytest.raises(SkillError):
        load_subject(listed)


def test_load_subject_rejects_identifier_mismatch(tmp_path):
    path = write_fake_subject(tmp_path, "S2", seconds=10)
    with pytest.raises(SkillError):
        load_subject(path, expected_subject="S3")


def test_parse_sha256_file_accepts_and_validates(tmp_path):
    good = tmp_path / "good.sha256"
    good.write_text("ABCDEF0123456789" * 4 + "  WESAD.zip\n")
    assert parse_sha256_file(good) == "abcdef0123456789" * 4
    for content in ("", "short", "z" * 64):
        bad = tmp_path / "bad.sha256"
        bad.write_text(content)
        with pytest.raises(SkillError):
            parse_sha256_file(bad)


def test_subject_pickle_requires_the_file(tmp_path):
    with pytest.raises(SkillError):
        _subject_pickle(tmp_path, "S2")


@pytest.fixture(scope="module")
def wesad_source(tmp_path_factory):
    source = tmp_path_factory.mktemp("wesad_source")
    for index, subject in enumerate(EXPECTED_SUBJECTS):
        write_fake_subject(source, subject, seconds=63, seed=index)
    return source


@pytest.fixture(scope="module")
def manifest_file(tmp_path_factory):
    path = tmp_path_factory.mktemp("manifests") / "WESAD.zip.sha256"
    path.write_text("a" * 64 + "  WESAD.zip\n")
    return path


def test_prepare_writes_canonical_corpus_and_manifest(wesad_source, manifest_file, tmp_path):
    output = tmp_path / "prepared"
    summary = prepare(wesad_source, manifest_file, output, holdout_subject="S17")
    assert summary == {
        "output": str(output),
        "holdout_subject": "S17",
        "subjects": 15,
        "rows": 15 * 63,
        "label_seconds": {"0": 15 * 21, "1": 15 * 21, "2": 15 * 21},
    }
    for subject in EXPECTED_SUBJECTS:
        assert (output / "canonical" / f"{subject}.csv").is_file()
    training = pd.read_csv(output / "training.csv", dtype={"subject_id": str})
    assert len(training) == 15 * 63
    assert sorted(training["subject_id"].unique()) == sorted(EXPECTED_SUBJECTS)
    assert (output / "fixture_subject.csv").read_bytes() == (
        output / "canonical" / "S17.csv"
    ).read_bytes()
    document = json.loads((output / "prepare_manifest.json").read_text())
    assert document["schema_version"] == "1.0"
    assert document["derivation_version"] == DERIVATION_VERSION
    assert document["dataset"]["wesad_zip_sha256"] == "a" * 64
    assert document["dataset"]["subjects"] == list(EXPECTED_SUBJECTS)
    assert document["dataset"]["holdout_subject"] == "S17"
    assert document["dataset"]["chest_sample_rate_hz"] == 700
    assert document["derivation"]["chest_only"] is True
    assert document["totals"]["rows"] == 15 * 63
    assert document["totals"]["label_seconds"] == {"0": 315, "1": 315, "2": 315}
    assert document["totals"]["wrist_available_subjects"] == 15
    digest = document["subjects"]["S2"]["pkl_sha256"]
    assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
    assert document["subjects"]["S2"]["hr"]["nan_seconds"] >= 0
    assert document["outputs"] == {
        "canonical": "canonical/SXX.csv",
        "training": "training.csv",
        "fixture_subject": "fixture_subject.csv",
    }


def test_prepare_is_deterministic_and_reads_nested_layouts(tmp_path_factory):
    source = tmp_path_factory.mktemp("nested_source")
    for index, subject in enumerate(EXPECTED_SUBJECTS):
        write_fake_subject(source, subject, seconds=63, seed=index, nested=True)
    manifest = source.parent / "nested.sha256"
    manifest.write_text("b" * 64 + "  WESAD.zip\n")
    runs = tmp_path_factory.mktemp("runs")
    first, second = runs / "first", runs / "second"
    prepare(source, manifest, first, holdout_subject="S17")
    prepare(source, manifest, second, holdout_subject="S17")
    assert (first / "training.csv").read_bytes() == (second / "training.csv").read_bytes()
    assert (first / "prepare_manifest.json").read_bytes() == (
        second / "prepare_manifest.json"
    ).read_bytes()
    assert (first / "canonical" / "S5.csv").read_bytes() == (
        second / "canonical" / "S5.csv"
    ).read_bytes()


def test_prepare_rejects_bad_gate_inputs(wesad_source, manifest_file, tmp_path):
    with pytest.raises(SkillError):
        prepare(wesad_source, manifest_file, tmp_path / "a", holdout_subject="S99")
    broken = tmp_path / "broken.sha256"
    broken.write_text("nope")
    with pytest.raises(SkillError):
        prepare(wesad_source, broken, tmp_path / "b", holdout_subject="S17")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        prepare(wesad_source, manifest_file, existing, holdout_subject="S17")
