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

from typing import TYPE_CHECKING, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes, Figure

from ephyspy.features.spike_features import available_spike_features
from ephyspy.utils import remove_mpl_artist_by_label

if TYPE_CHECKING:
    from ephyspy.sweeps import EphysSweep, EphysSweepSet

############################
### spike level features ###
############################


def plot_spike_feature(
    sweep: EphysSweep, ft: str, ax: Optional[Axes] = None, **kwargs
) -> Axes:
    """Plot spike feature by name.

    Args:
        sweep (EphysSweep): Sweep to plot the feature for.
        ft (str): Name of the feature to plot (all lowercase).
            Can plot all features that are included in the `EphysSweep._spikes_df`
            and all features in `available_spike_features()`.
        ax (Axes): Matplotlib axes.
        **kwargs: Additional kwargs are passed to the plotting function.

    Returns:
        Axes: Matplotlib axes.
    """
    if ft in available_spike_features():
        ax = available_spike_features()[ft](sweep).plot(ax=ax, **kwargs)
    else:
        raise ValueError(f"Feature {ft} does not exist.")
    return ax


def plot_spike_features(
    sweep: EphysSweep, window: Tuple = [0.4, 0.45]
) -> Tuple[Figure, Axes]:
    """Plot overview of the extracted spike features for a sweep.

    Args:
        sweep (EphysSweep): Sweep to plot the features for.
        window (Tuple, optional): Specific Time window to zoom in on a subset or
            single spikes to see more detail. Defaults to [0.4, 0.45].

    Returns:
        Tuple[Figure, Axes]: Matplotlib figure and axes."""

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
        for ft in available_spike_features():
            plot_spike_feature(sweep, ft, axes[x])

    axes["b"].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig, axes


def plot_sweepset_diagnostics(
    sweepset: EphysSweepSet,
    figsize=(15, 14),
) -> Tuple[Figure, Axes]:
    """Plot diagnostics overview for the whole sweepset.

    This function is useful to diagnose outliers on the sweepset level.

    Args:
        sweepset (EphysSweepSet): sweepset to diagnose.

    Returns:
        Fig, Axes: figure and axes with plot.
    """
    from ephyspy.features.sweepset_features import (
        NullSweepSetFeature,
        available_sweepset_features,
    )

    mosaic = [
        ["set_fts", "set_fts", "set_fts", "r_input"],
        ["fp_trace", "fp_trace", "fp_trace", "rheobase"],
        ["ap_trace", "ap_trace", "ap_trace", "ap_window"],
        ["sag_fts", "set_hyperpol_fts", "set_hyperpol_fts", "rebound_fts"],
    ]

    sweepset.add_features(
        available_spike_features()
    )  # HOTFIX: Spike features are not yet looked up properly
    fts = NullSweepSetFeature(sweepset)

    def plot_sweepset_ft(fts, ft, ax, **kwargs):
        FT = fts.lookup_sweepset_feature(ft, return_value=False)
        return FT.plot(ax=ax, **kwargs)

    def sweep_idx(fts, ft):
        try:
            FT = fts.lookup_sweepset_feature(ft, return_value=False)
            return FT.diagnostics["selected_idx"]
        except KeyError:
            return slice(0)
        except TypeError:
            # features like dfdI don't have a selected_idx
            return slice(0)

    def spike_idx(fts, ft):
        sw_idx = sweep_idx(fts, ft)
        FT = fts.lookup_sweep_feature(ft, return_value=False)
        return FT[sw_idx].diagnostics["aggregate_idx"]

    fig, axes = plt.subplot_mosaic(mosaic, figsize=figsize, constrained_layout=True)
    onset = fts.lookup_sweep_feature("stim_onset")[0]
    end = fts.lookup_sweep_feature("stim_end")[0]
    t0, tfin = sweepset.sweeps()[0].t[[0, -1]]
    for ax in axes.values():
        ax.set_xlim(t0, tfin)

    # set
    selected_sweeps = {}
    for ft in available_sweepset_features():
        sweep = sweepset[sweep_idx(fts, ft)]
        selected_sweeps[ft] = sweep if not sweep == [] else None

    unique_sweeps = {}
    for k, v in selected_sweeps.items():
        if v not in unique_sweeps:
            unique_sweeps[v] = k
        else:
            unique_sweeps[v] = unique_sweeps[v] + ", " + k
    unique_sweeps = {v: k for k, v in unique_sweeps.items()}
    keys = list(unique_sweeps.keys())

    # combine features for shorter labels
    for combined_ft, tag in [
        ("ap features", "ap_"),
        ("sag features", "sag"),
        ("rebound features", "rebound"),
        ("isi features", "isi"),
    ]:
        for i in range(len(keys)):
            keys[i] = ", ".join(
                np.unique([combined_ft if tag in k else k for k in keys[i].split(", ")])
            )
    unique_sweeps = {k: s for k, s in zip(keys, unique_sweeps.values())}

    for label, sweep in unique_sweeps.items():
        try:
            sweep.plot(axes["set_fts"], label=label)
        except AttributeError:
            pass

    sweepset.plot(axes["set_fts"], color="grey", alpha=0.2)
    plot_sweepset_ft(fts, "slow_hyperpolarization", axes["set_fts"])
    axes["set_fts"].legend(title="representative sweeps", loc="upper right")

    ap_sweep_idx = sweep_idx(fts, "ap_thresh")
    ap_idx = spike_idx(fts, "ap_amp")

    # fp
    plot_sweepset_ft(fts, "num_ap", axes["fp_trace"])
    plot_sweepset_ft(fts, "ap_freq_adapt", axes["fp_trace"])
    plot_sweepset_ft(fts, "ap_amp_slope", axes["fp_trace"])

    stim = sweepset[sweep_idx(fts, "num_ap")].i
    stim_amp = int(np.max(stim) + np.min(stim))
    axes["fp_trace"].legend(title=f"@{stim_amp }pA")

    # different selection / aggregation
    # plot_sweepset_ft(fts, "ap_amp_adapt", axes["fp_trace"])
    # plot_sweepset_ft(fts, "isi_ff", axes["fp_trace"])
    # plot_sweepset_ft(fts, "isi_cv", axes["fp_trace"])
    # plot_sweepset_ft(fts, "ap_ff", axes["fp_trace"])
    # plot_sweepset_ft(fts, "ap_cv", axes["fp_trace"])
    # plot_sweepset_ft(fts, "isi", axes["fp_trace"])

    # ap
    plot_sweepset_ft(fts, "ap_thresh", axes["ap_trace"])
    plot_sweepset_ft(fts, "ap_peak", axes["ap_trace"])
    plot_sweepset_ft(fts, "ap_trough", axes["ap_trace"])
    plot_sweepset_ft(fts, "ap_width", axes["ap_trace"])
    plot_sweepset_ft(fts, "ap_amp", axes["ap_trace"])
    plot_sweepset_ft(fts, "ap_ahp", axes["ap_trace"])
    plot_sweepset_ft(fts, "ap_adp", axes["ap_trace"])
    plot_sweepset_ft(fts, "ap_udr", axes["ap_trace"])

    stim = sweepset[sweep_idx(fts, "ap_thresh")].i
    stim_amp = int(np.max(stim) + np.min(stim))
    axes["ap_trace"].legend(title=f"@{stim_amp }pA")

    ap_sweep = sweepset[ap_sweep_idx]
    for i, ft in enumerate(available_spike_features()):
        plot_spike_feature(ap_sweep, ft, axes["ap_window"], color=f"C{i}")

    ap_start = ap_sweep.spike_feature("threshold_t")[ap_idx] - 5e-3
    ap_end = ap_sweep.spike_feature("fast_trough_t")[ap_idx] + 5e-3
    if isinstance(ap_start, np.ndarray):
        ap_start = ap_start[0]
        ap_end = ap_end[-1]
    axes["ap_window"].set_xlim(ap_start, ap_end)
    axes["ap_trace"].axvline(ap_start, color="grey")
    axes["ap_trace"].axvline(ap_end, color="grey", label="selected ap")
    ap_sweep.plot(axes["ap_window"])
    axes["ap_window"].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    # hyperpol
    plot_sweepset_ft(fts, "tau", axes["set_hyperpol_fts"])
    plot_sweepset_ft(fts, "v_baseline", axes["set_hyperpol_fts"])

    stim = sweepset[sweep_idx(fts, "tau")].i
    stim_amp = int(np.max(stim) + np.min(stim))
    axes["set_hyperpol_fts"].legend(title=f"@{stim_amp }pA")

    # sag
    plot_sweepset_ft(fts, "sag_area", axes["sag_fts"])
    plot_sweepset_ft(fts, "sag_time", axes["sag_fts"])
    plot_sweepset_ft(fts, "sag_ratio", axes["sag_fts"], color="tab:orange")
    remove_mpl_artist_by_label(axes["sag_fts"], "sag")
    plot_sweepset_ft(fts, "sag_fraction", axes["sag_fts"], color="tab:green")
    remove_mpl_artist_by_label(axes["sag_fts"], "sag")
    plot_sweepset_ft(fts, "sag", axes["sag_fts"])
    axes["sag_fts"].set_xlim(onset - 0.05, end + 0.05)

    stim = sweepset[sweep_idx(fts, "sag")].i
    stim_amp = int(np.max(stim) + np.min(stim))
    axes["sag_fts"].legend(title=f"@{stim_amp }pA")

    # rebound
    plot_sweepset_ft(fts, "rebound", axes["rebound_fts"])
    plot_sweepset_ft(fts, "rebound_latency", axes["rebound_fts"])
    plot_sweepset_ft(fts, "rebound_area", axes["rebound_fts"])
    plot_sweepset_ft(fts, "rebound_avg", axes["rebound_fts"])
    axes["rebound_fts"].set_xlim(end - 0.05, None)

    stim = sweepset[sweep_idx(fts, "rebound")].i
    stim_amp = int(np.max(stim) + np.min(stim))
    axes["rebound_fts"].legend(title=f"@{stim_amp }pA")
    axes["rebound_fts"].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    fig.text(-0.02, 0.5, "U (mV)", va="center", rotation="vertical", fontsize=16)
    fig.text(0.5, -0.02, "t (s)", ha="center", fontsize=16)

    plot_sweepset_ft(fts, "rheobase", axes["rheobase"])
    plot_sweepset_ft(fts, "r_input", axes["r_input"])

    axes["set_fts"].set_title("All sweeps")
    axes["fp_trace"].set_title("Representative spiking sweep")
    axes["ap_trace"].set_title("Representative AP sweep")
    axes["ap_window"].set_title("Representative AP")
    axes["set_hyperpol_fts"].set_title("Hyperpolarization sweeps")
    axes["sag_fts"].set_title("sag")
    axes["rebound_fts"].set_title("rebound")
    axes["rheobase"].set_title("Rheobase")
    return fig, axes
