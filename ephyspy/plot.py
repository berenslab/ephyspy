#!/usr/bin/env python3
# Copyright 2023 Jonas Beck

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes, Figure

if TYPE_CHECKING:
    from ephyspy.sweeps import EphysSweep

from ephyspy.utils import fwhm, where_between, spikefeatureplot, has_spike_feature

############################
### spike level features ###
############################


@spikefeatureplot
def plot_ap_width(sweep: EphysSweep, ax: Axes = None, **kwargs) -> Axes:
    if has_spike_feature(sweep, "threshold_t"):
        ap_fts = sweep._spikes_df
        t_threshold = sweep.spike_feature("threshold_t", include_clipped=True)
        t_peak = sweep.spike_feature("peak_t", include_clipped=True)
        t_next = t_peak + 1.0 * (t_peak - t_threshold)  # T interval w.r.t. threshold

        fwhm_v = np.zeros_like(t_threshold)
        hm_up_t = np.zeros_like(t_threshold)
        hm_down_t = np.zeros_like(t_threshold)
        for i, (t_th, t_n) in enumerate(zip(t_threshold, t_next)):
            fwhm_i = fwhm(sweep.t, sweep.v, t_th, t_n)
            fwhm_v[i], hm_up_t[i], hm_down_t[i] = fwhm_i

        ax.hlines(fwhm_v, hm_up_t, hm_down_t, label="width", ls="--", **kwargs)
    return ax


@spikefeatureplot
def plot_ap_adp(sweep: EphysSweep, ax: Axes = None, **plot_kwargs) -> Axes:
    if has_spike_feature(sweep, "adp_v"):
        ap_fts = sweep._spikes_df
        ax.vlines(
            0.5 * (ap_fts[f"adp_t"] + ap_fts["fast_trough_t"]),
            ap_fts["adp_v"],
            ap_fts["fast_trough_v"],
            ls="--",
            lw=1,
            label="adp",
            **plot_kwargs,
        )
    return ax


@spikefeatureplot
def plot_ap_ahp(sweep: EphysSweep, ax: Axes = None, **kwargs) -> Axes:
    if has_spike_feature(sweep, "ahp_v"):
        ap_fts = sweep._spikes_df
        ax.vlines(
            0.5 * (ap_fts[f"fast_trough_t"] + ap_fts["threshold_t"]),
            ap_fts["fast_trough_v"],
            ap_fts["threshold_v"],
            ls="--",
            lw=1,
            label="ahp",
            **kwargs,
        )
    return ax


@spikefeatureplot
def plot_ap_amp(sweep: EphysSweep, ax: Axes = None, **kwargs) -> Axes:
    if has_spike_feature(sweep, "threshold_v"):
        ap_fts = sweep._spikes_df
        thresh_v = ap_fts["threshold_v"]
        peak_t = ap_fts["peak_t"]
        peak_v = ap_fts["peak_v"]
        ax.plot(peak_t, peak_v, "x", **kwargs)

        ax.vlines(peak_t, thresh_v, peak_v, ls="--", label="ap_amp", **kwargs)
    return ax


@spikefeatureplot
def plot_isi(sweep: EphysSweep, ax: Axes = None, **kwargs) -> Axes:
    if has_spike_feature(sweep, "isi"):
        ap_fts = sweep._spikes_df
        thresh_t = ap_fts["threshold_t"]
        thresh_v = ap_fts["threshold_v"]
        isi = ap_fts["isi"].to_numpy()
        isi[0] = 0

        ax.hlines(thresh_v, thresh_t - isi, thresh_t, ls="--", label="isi", **kwargs)
        ax.plot(thresh_t, thresh_v, "x", **kwargs)

    return ax


def plot_simple_spike_feature(ft):
    @spikefeatureplot
    def scatter_spike_ft(sweep: EphysSweep, ax: Axes = None, **kwargs) -> Axes:
        if has_spike_feature(sweep, ft + "_v"):
            ap_fts = sweep._spikes_df
            ax.scatter(ap_fts[f"{ft}_t"], ap_fts[f"{ft}_v"], s=10, label=ft, **kwargs)
        return ax

    return scatter_spike_ft


plot_peak = plot_simple_spike_feature("peak")
plot_threshold = plot_simple_spike_feature("threshold")
plot_trough = plot_simple_spike_feature("trough")
plot_upstroke = plot_simple_spike_feature("upstroke")
plot_downstroke = plot_simple_spike_feature("downstroke")
plot_fast_trough = plot_simple_spike_feature("fast_trough")
plot_slow_trough = plot_simple_spike_feature("slow_trough")

plottable_spike_features = {
    "peak": plot_peak,
    "trough": plot_trough,
    "threshold": plot_threshold,
    "upstroke": plot_upstroke,
    "downstroke": plot_downstroke,
    "ap_width": plot_ap_width,
    "fast_trough": plot_fast_trough,
    "slow_trough": plot_slow_trough,
    "ap_adp": plot_ap_adp,
    "ap_ahp": plot_ap_ahp,
    "isi": plot_isi,
    "ap_amp": plot_ap_amp,
}


def plot_spike_feature(sweep: EphysSweep, ft: str, ax: Axes, **kwargs) -> Axes:
    if ft in plottable_spike_features:
        ax = plottable_spike_features[ft](sweep, ax=ax, **kwargs)
    else:
        raise ValueError(f"Feature {ft} does not exist.")
    return ax


def plot_spike_features(
    sweep: EphysSweep, window: Tuple = [0.4, 0.45]
) -> Tuple[Figure, Axes]:
    mosaic = "aaabb\naaabb\ncccbb"
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 4), constrained_layout=True)

    # plot sweep
    axes["a"].plot(sweep.t, sweep.v, color="k")
    axes["a"].set_ylabel("Voltage (mV)")
    axes["a"].axvline(window[0], color="grey", alpha=0.5)
    axes["a"].axvline(window[1], color="grey", alpha=0.5, label="window")

    axes["b"].plot(sweep.t, sweep.v, color="k")
    axes["b"].set_ylabel("Voltage (mV)")
    axes["b"].set_xlabel("Time (s)")
    axes["b"].set_xlim(window)

    axes["c"].plot(sweep.t, sweep.i, color="k")
    axes["c"].axvline(window[0], color="grey", alpha=0.5)
    axes["c"].axvline(window[1], color="grey", alpha=0.5, label="window")
    axes["c"].set_yticks([0, np.max(sweep.i) + np.min(sweep.i)])
    axes["c"].set_ylabel("Current (pA)")
    axes["c"].set_xlabel("Time (s)")

    # plot ap features
    for x in ["a", "b"]:
        for ft in plottable_spike_features:
            plot_spike_feature(sweep, ft, axes[x])

    axes["b"].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig, axes
