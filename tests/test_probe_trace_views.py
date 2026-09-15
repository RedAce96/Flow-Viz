"""Named trace views must retain independent physical scales."""

import matplotlib.pyplot as plt
import numpy as np

from pelecpost.analysis.probe_plotting import build_probe_trace_view_figure
from pelecpost.config.models import PresentationConfig, ProbeTraceView


def test_symmetric_pairs_have_independent_vertical_scales():
    view = ProbeTraceView.model_validate({
        "id": "pairs",
        "title": "Symmetric pairs",
        "groups": [
            {"title": "Near-source", "probe_indices": [150, 170]},
            {"title": "Outer", "probe_indices": [120, 200]},
        ],
    })
    time = np.linspace(0.0, 1e-6, 101)
    near = 100.0 * np.exp(-time / 2e-7)
    outer = 2.0 * np.exp(-time / 3e-7)
    values = np.column_stack((outer, near, near, outer))
    figure = build_probe_trace_view_figure(
        PresentationConfig(), time, values,
        np.array([120, 150, 170, 200]),
        ["Probe 120", "Probe 150", "Probe 170", "Probe 200"],
        view, "pressure", "Pa",
    )
    try:
        near_axis, outer_axis = figure.axes
        assert near_axis.get_ylim()[1] > 100.0
        assert outer_axis.get_ylim()[1] < 10.0
        assert np.array_equal(near_axis.lines[0].get_ydata(), near)
        assert np.array_equal(outer_axis.lines[0].get_ydata(), outer)
    finally:
        plt.close(figure)


def test_pair_difference_does_not_magnify_roundoff():
    view = ProbeTraceView.model_validate({
        "id": "pulse",
        "title": "Pulse symmetry",
        "groups": [{"title": "Near-source", "probe_indices": [150, 170]}],
        "pair_difference": True,
    })
    values = np.column_stack((np.full(4, 27000.0), np.full(4, 27000.0 + 1e-10)))
    figure = build_probe_trace_view_figure(
        PresentationConfig(), np.arange(4) * 2e-9, values,
        np.array([150, 170]), ["Probe 150", "Probe 170"],
        view, "pressure", "Pa",
    )
    try:
        assert figure.axes[-1].get_ylim()[1] >= 2.7
    finally:
        plt.close(figure)
