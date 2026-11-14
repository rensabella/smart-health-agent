"""PNG figures without pyplot, GUI state, filesystem persistence, or raw uploads."""

from __future__ import annotations

import io
import threading

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from smarthealth.runtime.errors import SkillError

_RENDER_LOCK = threading.Lock()
SUPPORTED_PLOTS = frozenset({"signals", "risk", "importance"})


def render_plots(
    windows, prediction: dict, explanation: dict, plot_types: list[str], *, profile=None
) -> list[dict]:
    if profile is None:
        from smarthealth.pipeline.profiles import SYNTHETIC_V1

        profile = SYNTHETIC_V1
    if (
        not plot_types
        or len(plot_types) > 3
        or len(set(plot_types)) != len(plot_types)
        or not set(plot_types).issubset(SUPPORTED_PLOTS)
    ):
        raise SkillError(
            "invalid_input", "Choose unique supported signal, risk, or importance plots."
        )
    if not windows or len(windows) > 3600:
        raise SkillError("invalid_input", "Plot input exceeds the supported window range.")
    output = []
    with _RENDER_LOCK:
        for kind in plot_types:
            signals_height = 1.2 * len(profile.signal_columns) + 1
            figure = Figure(
                figsize=(9, signals_height if kind == "signals" else 4),
                dpi=110,
                layout="constrained",
            )
            FigureCanvasAgg(figure)
            if kind == "signals":
                data = pd.concat(windows, ignore_index=True)
                # Render at most 1800 sampled points; analysis still uses every row.
                indices = np.unique(np.linspace(0, len(data) - 1, min(1800, len(data))).astype(int))
                axes = figure.subplots(len(profile.signal_columns), 1, sharex=True)
                for axis, name in zip(axes, profile.signal_columns):
                    axis.plot(indices, data[name].iloc[indices], linewidth=0.8, color="#0b7285")
                    axis.set_ylabel(name, fontsize=8)
                    axis.grid(alpha=0.2)
                axes[-1].set_xlabel("Elapsed seconds (cleaned 1 Hz samples; display downsampled)")
                figure.suptitle(profile.guard_text.signal_plot_title)
            elif kind == "risk":
                axis = figure.subplots()
                points = prediction["window_predictions"]
                axis.plot(
                    [p["window_index"] for p in points],
                    [p["risk_score"] for p in points],
                    color="#c92a2a",
                )
                axis.set(
                    xlabel=f"Window index ({profile.window_seconds} seconds)",
                    ylabel=profile.guard_text.risk_plot_ylabel,
                    ylim=(0, 1),
                    title="Window scores; session aggregation = maximum",
                )
                axis.grid(alpha=0.2)
            else:
                axis = figure.subplots()
                items = explanation["global_importance"][:10][::-1]
                axis.barh(
                    [item["name"] for item in items],
                    [item["importance"] for item in items],
                    color="#0b7285",
                )
                axis.set(
                    xlabel="Normalized global model importance",
                    title="Global importance (not an individual causal explanation)",
                )
                axis.tick_params(axis="y", labelsize=8)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png")
            output.append({"kind": kind, "mime_type": "image/png", "content": buffer.getvalue()})
            figure.clear()
    return output
