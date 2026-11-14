"""Registered input-profile contracts shared by pipeline, models, runtime and evaluation.

Every profile pins the column whitelist, window/stride geometry, feature
schema, label vocabulary, score semantics and fail-closed guard copy for one
data origin. ``SYNTHETIC_V1`` aliases the historical module constants so the
default path stays byte-for-byte identical; ``WESAD_V1`` describes the derived
chest-signal canonical table introduced in Phase 4.

Import direction: this module imports the pipeline and wesad modules at the
top level; those modules import this one only inside function bodies.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from smarthealth.pipeline.feature_engineering import (
    FEATURE_CONFIG,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    extract_synthetic_features,
)
from smarthealth.pipeline.preprocess import (
    INPUT_PROFILE,
    OPTIONAL_COLUMNS,
    PROCESSING_CONFIG,
    SIGNAL_COLUMNS,
)
from smarthealth.wesad.features import (
    WESAD_FEATURE_CONFIG,
    WESAD_FEATURE_NAMES,
    WESAD_FEATURE_SCHEMA_VERSION,
    WESAD_SIGNAL_COLUMNS,
    WESAD_STRIDE_ROWS,
    WESAD_WINDOW_ROWS,
    extract_wesad_features,
)


@dataclass(frozen=True)
class ValueRule:
    """Fail-closed per-column value predicate applied to the nonmissing floats."""

    columns: tuple[str, ...]
    message: str
    violated: Callable[[np.ndarray], bool]


@dataclass(frozen=True)
class ConstantColumnRule:
    """A column that must hold one constant value within a single session."""

    column: str
    message: str


@dataclass(frozen=True)
class GuardText:
    """Operator-reviewed copy rendered into publications, memory and reports."""

    uncertainty: str
    high_score_recommendation: str
    analysis_summary_template: str
    score_definition_template: str
    report_title: str
    report_uncertainty_suffix: str
    signal_plot_title: str
    risk_plot_ylabel: str


@dataclass(frozen=True)
class DataProfile:
    name: str
    signal_columns: tuple[str, ...]
    optional_columns: tuple[str, ...]
    processing_config: dict
    processing_profile: str
    feature_profile: str
    feature_schema_version: str
    feature_names: tuple[str, ...]
    feature_config: dict
    extractor: Callable[[list[pd.DataFrame]], pd.DataFrame]
    label_mapping: dict[str, str]
    expected_classes: tuple[int, ...]
    risk_class: str
    score_semantics: str
    task_scope: str
    dependency_packages: tuple[str, ...]
    value_rules: tuple[ValueRule, ...]
    constant_columns: tuple[ConstantColumnRule, ...]
    guard_text: GuardText
    explain_limitations: tuple[str, ...]

    @property
    def window_rows(self) -> int:
        return self.processing_config["window_rows"]

    @property
    def stride_rows(self) -> int:
        return self.processing_config["stride_rows"]

    @property
    def window_seconds(self) -> int:
        return self.window_rows // self.processing_config["sample_rate_hz"]

    @property
    def allowed_columns(self) -> tuple[str, ...]:
        return self.signal_columns + self.optional_columns

    def risk_class_index(self) -> int:
        """Numeric class label whose probability is the risk score."""
        for key, label in self.label_mapping.items():
            if label == self.risk_class:
                return int(key)
        raise ValueError(f"Risk class {self.risk_class!r} is not part of the label mapping")


SYNTHETIC_V1 = DataProfile(
    name=INPUT_PROFILE,
    signal_columns=SIGNAL_COLUMNS,
    optional_columns=OPTIONAL_COLUMNS,
    processing_config=PROCESSING_CONFIG,
    processing_profile="1hz_30s_v1",
    feature_profile="physiology_statistics_v1",
    feature_schema_version=FEATURE_SCHEMA_VERSION,
    feature_names=FEATURE_NAMES,
    feature_config=FEATURE_CONFIG,
    extractor=extract_synthetic_features,
    label_mapping={"0": "normal", "1": "anomaly"},
    expected_classes=(0, 1),
    risk_class="anomaly",
    score_semantics="synthetic_anomaly_model_score",
    task_scope="synthetic_anomaly_only",
    dependency_packages=("scikit-learn", "lightgbm", "numpy", "joblib"),
    value_rules=(
        ValueRule(
            ("heart_rate",),
            "Heart rate must be positive when present.",
            lambda values: bool((values <= 0).any()),
        ),
        ValueRule(
            ("EDA", "EEG_feature"),
            "EDA and the synthetic EEG proxy must be nonnegative.",
            lambda values: bool((values < 0).any()),
        ),
        ValueRule(
            ("activity", "sleep_quality"),
            "Normalized context values must be within zero and one.",
            lambda values: bool(((values < 0) | (values > 1)).any()),
        ),
    ),
    constant_columns=(
        ConstantColumnRule("sleep_quality", "Sleep quality must be constant within a session."),
    ),
    guard_text=GuardText(
        uncertainty="模型仅用合成数据训练，未经临床验证；分数和特征贡献不能证明病因或患病概率。",
        high_score_recommendation=(
            "请先核对输入协议、数据完整性和采集质量，再复测分析；合成模型高分不能作为诊疗依据。"
        ),
        analysis_summary_template=(
            "分析了 {window_count} 个 {window_seconds} 秒窗口；最高合成异常模型分数为 "
            "{risk_score:.3f}，对应窗口 {highest_risk_window}（从 0 编号）。"
            "该分数不表示临床患病概率或医学诊断；局部贡献描述模型行为，不证明异常原因。"
        ),
        score_definition_template=(
            "risk_score 是当前模型对合成异常模式的分数。每 {window_seconds} 秒窗口产生一个分数，"
            "会话取最高窗口分数；它不是临床患病概率。"
        ),
        report_title="合成生理信号分析与知识解释",
        report_uncertainty_suffix="检索资料提供一般知识背景，不证明当前个体异常的原因。",
        signal_plot_title="Synthetic physiological signal context",
        risk_plot_ylabel="Synthetic anomaly score",
    ),
    explain_limitations=(
        "Trained on synthetic anomaly events; outputs are not clinical diagnoses or disease probabilities.",
        "Session score is the maximum window score and increases exposure to isolated extreme windows.",
        "Feature associations do not establish causal explanations of an individual's physiology.",
        "Reference deviations use normal training windows with a standard-deviation floor of 1e-6.",
    ),
)

WESAD_V1 = DataProfile(
    name="wesad_physiology_v1",
    signal_columns=WESAD_SIGNAL_COLUMNS,
    optional_columns=("timestamp", "subject_id", "session_id"),
    processing_config={
        "version": "1.0",
        "sample_rate_hz": 1,
        "window_rows": WESAD_WINDOW_ROWS,
        "stride_rows": WESAD_STRIDE_ROWS,
        "max_gap_rows": 3,
        "max_window_missing_fraction": 0.15,
        "max_channel_missing_fraction": 0.20,
        "tail_policy": "discard_and_report",
        "imputation": "within_window_linear_interpolation_then_edge_fill",
    },
    processing_profile="1hz_60s_stride30_v1",
    feature_profile="wesad_physiology_statistics_v1",
    feature_schema_version=WESAD_FEATURE_SCHEMA_VERSION,
    feature_names=WESAD_FEATURE_NAMES,
    feature_config=WESAD_FEATURE_CONFIG,
    extractor=extract_wesad_features,
    label_mapping={"0": "baseline", "1": "stress", "2": "amusement"},
    expected_classes=(0, 1, 2),
    risk_class="stress",
    score_semantics="wesad_stress_model_score",
    task_scope="wesad_stress_only",
    dependency_packages=("scikit-learn", "lightgbm", "numpy", "joblib", "scipy"),
    value_rules=(
        ValueRule(
            ("hr",),
            "Heart rate must be positive when present.",
            lambda values: bool((values <= 0).any()),
        ),
        ValueRule(
            ("eda", "acc_mag", "scr"),
            "EDA, acceleration magnitude, and SCR indicators must be nonnegative when present.",
            lambda values: bool((values < 0).any()),
        ),
    ),
    constant_columns=(),
    guard_text=GuardText(
        uncertainty=(
            "该模型是基于实验室数据（WESAD）训练的非临床压力模式分类器，未经临床验证；"
            "分数和特征贡献不能证明病因或患病概率，也不能用于诊疗。"
        ),
        high_score_recommendation=(
            "请先核对输入协议、数据完整性和采集质量，再复测分析；压力模型高分不能作为诊疗依据。"
        ),
        analysis_summary_template=(
            "分析了 {window_count} 个 {window_seconds} 秒窗口；最高压力模型分数为 "
            "{risk_score:.3f}，对应窗口 {highest_risk_window}（从 0 编号）。"
            "该分数不表示临床患病概率或医学诊断；局部贡献描述模型行为，不证明压力反应的原因。"
        ),
        score_definition_template=(
            "risk_score 是当前模型对实验室压力模式的分数。每 {window_seconds} 秒窗口产生一个分数，"
            "会话取最高窗口分数；它不是临床患病概率。"
        ),
        report_title="压力生理信号分析与知识解释",
        report_uncertainty_suffix="检索资料提供一般知识背景，不证明当前个体压力反应的原因。",
        signal_plot_title="WESAD chest-derived physiological signal context (non-clinical)",
        risk_plot_ylabel="Stress model score",
    ),
    explain_limitations=(
        "Trained on laboratory stress recordings (WESAD); outputs are not clinical diagnoses or disease probabilities.",
        "Session score is the maximum window score and increases exposure to isolated extreme windows.",
        "Feature associations do not establish causal explanations of an individual's physiology.",
        "Reference deviations use baseline-class training windows with a standard-deviation floor of 1e-6.",
    ),
)

_PROFILES = {"synthetic_physiology_v1": SYNTHETIC_V1, "wesad_physiology_v1": WESAD_V1}


def get_profile(name: str) -> DataProfile:
    """Resolve a registered profile; unknown names are a contract failure."""
    try:
        return _PROFILES[name]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown data profile: {name!r}") from None


def suggest_profile(columns: Iterable[str], *, exclude: str) -> str | None:
    """Name another registered profile whose column contract this header satisfies."""
    header = set(columns)
    for profile in _PROFILES.values():
        if profile.name == exclude:
            continue
        if set(profile.signal_columns).issubset(header) and header.issubset(
            set(profile.allowed_columns)
        ):
            return profile.name
    return None
