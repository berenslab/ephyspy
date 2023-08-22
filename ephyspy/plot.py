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
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes, Figure
from typing import Tuple
from typing import TYPE_CHECKING
import numpy as np
import warnings

if TYPE_CHECKING:
    from ephyspy.sweeps import EphysSweepFeatureExtractor
from ephyspy.utils import where_between, fwhm

############################
### spike level features ###
############################


def plot_ap_width(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    ap_fts = sweep._aps_df
    if ap_fts.size:
        if not np.all(ap_fts["width"].isnan()):
            t_threshold = sweep.ap_feature("threshold_t", include_clipped=True)
            t_peak = sweep.ap_feature("peak_t", include_clipped=True)
            t_next = t_peak + 1.0 * (
                t_peak - t_threshold
            )  # T interval w.r.t. threshold

            fwhm_v = np.zeros_like(t_threshold)
            hm_up_t = np.zeros_like(t_threshold)
            hm_down_t = np.zeros_like(t_threshold)
            for i, (t_th, t_n) in enumerate(zip(t_threshold, t_next)):
                fwhm_i = fwhm(sweep.t, sweep.v, t_th, t_n)
                fwhm_v[i], hm_up_t[i], hm_down_t[i] = fwhm_i

            ax.hlines(
                fwhm_v,
                hm_up_t,
                hm_down_t,
                label="width",
                ls="--",
                colors=color,
                **plot_kwargs,
            )
    return ax


def plot_ap_adp(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    ap_fts = sweep._aps_df
    if ap_fts.size:
        if not np.all(ap_fts["adp_v"].isnan()):
            ax.vlines(
                0.5 * (ap_fts[f"adp_t"] + ap_fts["fast_trough_t"]),
                ap_fts["adp_v"],
                ap_fts["fast_trough_v"],
                ls="--",
                lw=1,
                label="adp",
                colors=color,
                **plot_kwargs,
            )
    return ax


def plot_ap_ahp(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    ap_fts = sweep._aps_df
    if ap_fts.size:
        if not np.all(ap_fts["fast_trough_v"].isnan()):
            ax.vlines(
                0.5 * (ap_fts[f"fast_trough_t"] + ap_fts["threshold_t"]),
                ap_fts["fast_trough_v"],
                ap_fts["threshold_v"],
                ls="--",
                lw=1,
                label="ahp",
                colors=color,
                **plot_kwargs,
            )
    return ax


def plot_ap_amp(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    warnings.warn("ap ap_amp plotting is not yet implemented!")
    return ax


def plot_isi(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    warnings.warn("isi plotting is not yet implemented!")
    return ax


def plot_spike_feature(
    sweep: EphysSweepFeatureExtractor, ft: str, ax: Axes, **plot_kwargs
) -> Axes:
    ap_fts = sweep._aps_df
    base_fts = [
        "peak",
        "threshold",
        "trough",
        "upstroke",
        "downstroke",
        "fast_trough",
        "slow_trough",
    ]
    if ap_fts.size:
        if ft == "ap_amp":
            ax = plot_ap_amp(sweep, ax, **plot_kwargs)
        elif ft == "ap_width":
            ax = plot_ap_width(sweep, ax, **plot_kwargs)
        elif ft == "ap_adp":
            ax = plot_ap_adp(sweep, ax, **plot_kwargs)
        elif ft == "ap_ahp":
            ax = plot_ap_ahp(sweep, ax, **plot_kwargs)
        elif ft == "isi":
            ax = plot_isi(sweep, ax, **plot_kwargs)
        elif ft in base_fts and not np.all(ap_fts[f"{ft}_v"].isnan()):
            ax.scatter(
                ap_fts[f"{ft}_t"],
                ap_fts[f"{ft}_v"],
                s=10,
                label=ft,
                **plot_kwargs,
            )
        else:
            raise ValueError(f"Feature {ft} does not exist.")
    return ax


plottable_spike_features = [
    "peak",
    "trough",
    "threshold",
    "upstroke",
    "downstroke",
    "width",
    "fast_trough",
    "slow_trough",
    "adp",
    "ahp",
]


def plot_spike_features(
    sweep: EphysSweepFeatureExtractor, window: Tuple = [0.4, 0.45]
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
