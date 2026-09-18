from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src import plot_diagnostics, plot_style


@pytest.mark.parametrize("final_fraction", [-0.4, 0.06, 0.0002])
def test_cumulative_mass_labels_remain_legible_across_drift_scales(
    final_fraction, monkeypatch, tmp_path
):
    times = np.arange(
        np.datetime64("2021-06-01T00"),
        np.datetime64("2021-09-01T00"),
        np.timedelta64(1, "h"),
    )
    flux = np.full(times.size, final_fraction / (times.size * 3600))
    advection = xr.Dataset(
        {
            "net_mass_advection": ("time", flux),
            "mass_flux_contribution_west": ("time", flux),
        },
        coords={"time": times},
    )
    volume = xr.ones_like(advection["net_mass_advection"])
    tendency = xr.zeros_like(volume)
    inspected = []
    save_figure = plot_style.save_figure

    def save_and_check(fig, path):
        save_figure(fig, path)
        if Path(path).name != "fig2_mass_residual_time_series.png":
            return
        fig.canvas.draw()
        axis = fig.axes[1]
        lower, upper = axis.get_ylim()
        labels = [
            label
            for position, label in zip(axis.get_yticks(), axis.get_yticklabels())
            if lower <= position <= upper
        ]
        assert len(labels) >= 3
        assert len({label.get_text() for label in labels}) == len(labels)
        renderer = fig.canvas.get_renderer()
        bounds = sorted(
            (label.get_window_extent(renderer) for label in labels),
            key=lambda box: box.y0,
        )
        assert all(left.y1 < right.y0 for left, right in pairwise(bounds))
        assert axis.lines[0].get_ydata()[-1] == pytest.approx(final_fraction)
        inspected.append(path)

    monkeypatch.setattr(plot_style, "save_figure", save_and_check)
    plot_diagnostics.fig2_mass_advection_residual_timeseries(
        advection, tendency, volume, str(tmp_path)
    )
    assert len(inspected) == 1
    assert (tmp_path / "fig2.2_mass_advection_terms_timeseries.png").is_file()
