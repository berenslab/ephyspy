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

from typing import TYPE_CHECKING, Callable, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes, Figure

if TYPE_CHECKING:
    from ephyspy.sweeps import EphysSweep, EphysSweepSet

from ephyspy.utils import fwhm, has_spike_feature, spikefeatureplot

############################
### spike level features ###
############################


@spikefeatureplot
def plot_ap_width(
    sweep: EphysSweep, ax: Axes = None, selected_idxs=None, **kwargs
) -> Axes:
    r"""Plot action potential width feature for all or selected aps.

    Inherits additional kwargs / functionality from `spikefeatureplot`.

    Args:
        sweep (EphysSweep): Sweep to plot the feature for.
        ax (Axes, optional): Matplotlib axes. Defaults to None.
        selected_idxs (slice, optional): Slice object to select aps. Defaults to None.
        **kwargs: Additional kwargs are passed to the plotting function.

    Returns:
        Axes: Matplotlib axes."""
    if has_spike_feature(sweep, "threshold_t"):
        idxs = slice(None) if selected_idxs is None else selected_idxs
        t_threshold = sweep.spike_feature("threshold_t", include_clipped=True)[idxs]
        t_peak = sweep.spike_feature("peak_t", include_clipped=True)[idxs]
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
def plot_ap_adp(
    sweep: EphysSweep, ax: Axes = None, selected_idxs=None, **kwargs
) -> Axes:
    """Plot action potential afterdepolarization feature for all or selected aps.

    Inherits additional kwargs / functionality from `spikefeatureplot`.

    Args:
        sweep (EphysSweep): Sweep to plot the feature for.
        ax (Axes, optional): Matplotlib axes. Defaults to None.
        selected_idxs (slice, optional): Slice object to select aps. Defaults to None.
        **kwargs: Additional kwargs are passed to the plotting function.

    Returns:
        Axes: Matplotlib axes."""
    if has_spike_feature(sweep, "adp_v"):
        idxs = slice(None) if selected_idxs is None else selected_idxs
        adp_t = sweep.spike_feature("adp_t", include_clipped=True)[idxs]
        adp_v = sweep.spike_feature("adp_v", include_clipped=True)[idxs]
        trough_t = sweep.spike_feature("fast_trough_t", include_clipped=True)[idxs]
        trough_v = sweep.spike_feature("fast_trough_v", include_clipped=True)[idxs]
        ax.vlines(
            0.5 * (adp_t + trough_t),
            adp_v,
            trough_v,
            ls="--",
            lw=1,
            label="adp",
            **kwargs,
        )
    return ax


@spikefeatureplot
def plot_ap_ahp(
    sweep: EphysSweep, ax: Axes = None, selected_idxs=None, **kwargs
) -> Axes:
    """Plot action potential afterhyperpolarization feature for all or selected aps.

    Inherits additional kwargs / functionality from `spikefeatureplot`.

    Args:
        sweep (EphysSweep): Sweep to plot the feature for.
        ax (Axes, optional): Matplotlib axes. Defaults to None.
        selected_idxs (slice, optional): Slice object to select aps. Defaults to None.
        **kwargs: Additional kwargs are passed to the plotting function.

    Returns:
        Axes: Matplotlib axes."""
    if has_spike_feature(sweep, "ahp_v"):
        idxs = slice(None) if selected_idxs is None else selected_idxs
        trough_t = sweep.spike_feature("fast_trough_t", include_clipped=True)[idxs]
        trough_v = sweep.spike_feature("fast_trough_v", include_clipped=True)[idxs]
        threshold_t = sweep.spike_feature("threshold_t", include_clipped=True)[idxs]
        threshold_v = sweep.spike_feature("threshold_v", include_clipped=True)[idxs]
        ax.vlines(
            0.5 * (trough_t + threshold_t),
            trough_v,
            threshold_v,
            ls="--",
            lw=1,
            label="ahp",
            **kwargs,
        )
    return ax


@spikefeatureplot
def plot_ap_amp(
    sweep: EphysSweep, ax: Axes = None, selected_idxs=None, **kwargs
) -> Axes:
    """Plot action potential ap amplitude feature for all or selected aps.

    Inherits additional kwargs / functionality from `spikefeatureplot`.

    Args:
        sweep (EphysSweep): Sweep to plot the feature for.
        ax (Axes, optional): Matplotlib axes. Defaults to None.
        selected_idxs (slice, optional): Slice object to select aps. Defaults to None.
        **kwargs: Additional kwargs are passed to the plotting function.

    Returns:
        Axes: Matplotlib axes."""
    if has_spike_feature(sweep, "threshold_v"):
        idxs = slice(None) if selected_idxs is None else selected_idxs
        thresh_v = sweep.spike_feature("threshold_v", include_clipped=True)[idxs]
        peak_t = sweep.spike_feature("peak_t", include_clipped=True)[idxs]
        peak_v = sweep.spike_feature("peak_v", include_clipped=True)[idxs]

        ax.plot(peak_t, peak_v, "x", **kwargs)
        ax.vlines(peak_t, thresh_v, peak_v, ls="--", label="ap_amp", **kwargs)
    return ax


@spikefeatureplot
def plot_isi(sweep: EphysSweep, ax: Axes = None, selected_idxs=None, **kwargs) -> Axes:
    """Plot action potential inter spike interval feature for all or selected aps.

    Inherits additional kwargs / functionality from `spikefeatureplot`.

    Args:
        sweep (EphysSweep): Sweep to plot the feature for.
        ax (Axes, optional): Matplotlib axes. Defaults to None.
        selected_idxs (slice, optional): Slice object to select aps. Defaults to None.
        **kwargs: Additional kwargs are passed to the plotting function.

    Returns:
        Axes: Matplotlib axes."""
    if has_spike_feature(sweep, "isi"):
        idxs = slice(None) if selected_idxs is None else selected_idxs
        thresh_t = sweep.spike_feature("threshold_t", include_clipped=True)[idxs]
        thresh_v = sweep.spike_feature("threshold_v", include_clipped=True)[idxs]
        isi = sweep.spike_feature("isi", include_clipped=True)[idxs]

        ax.hlines(thresh_v, thresh_t - isi, thresh_t, ls="--", label="isi", **kwargs)
        ax.plot(thresh_t, thresh_v, "x", **kwargs)

    return ax


def plot_simple_spike_feature(ft: str) -> Callable:
    """Plot simple spike feature, i.e. a single value per ap.

    Args:
        ft (str): Name of the feature to plot (all lowercase).
            Can plot all features that are included in the `EphysSweep._spikes_df`.

    Returns:
        callable: Function that plots the feature for all or selected aps.
    """

    @spikefeatureplot
    def scatter_spike_ft(
        sweep: EphysSweep, ax: Axes = None, selected_idxs=None, **kwargs
    ) -> Axes:
        f"""Plot action potential {ft} feature for all or selected aps.

        Inherits additional kwargs / functionality from `spikefeatureplot`.

        Args:
            sweep (EphysSweep): Sweep to plot the feature for.
            ax (Axes, optional): Matplotlib axes. Defaults to None.
            selected_idxs (slice, optional): Slice object to select aps. Defaults to None.
            **kwargs: Additional kwargs are passed to the plotting function.

        Returns:
            Axes: Matplotlib axes."""
        if has_spike_feature(sweep, ft + "_v"):
            idxs = slice(None) if selected_idxs is None else selected_idxs
            t = sweep.spike_feature(ft + "_t", include_clipped=True)[idxs]
            v = sweep.spike_feature(ft + "_v", include_clipped=True)[idxs]
            ax.scatter(t, v, s=10, label=ft, **kwargs)
        return ax

    return scatter_spike_ft


@spikefeatureplot
def plot_udr(sweep: EphysSweep, ax: Axes = None, selected_idxs=None, **kwargs) -> Axes:
    """Plot upstroke downstroke ratio feature for all or selected aps.

    Inherits additional kwargs / functionality from `spikefeatureplot`.

    Args:
        sweep (EphysSweep): Sweep to plot the feature for.
        ax (Axes, optional): Matplotlib axes. Defaults to None.
        selected_idxs (slice, optional): Slice object to select aps. Defaults to None.
        **kwargs: Additional kwargs are passed to the plotting function.

    Returns:
        Axes: Matplotlib axes."""
    if has_spike_feature(sweep, "isi"):
        idxs = slice(None) if selected_idxs is None else selected_idxs
        upstroke_t = sweep.spike_feature("upstroke_t", include_clipped=True)[idxs]
        upstroke_v = sweep.spike_feature("upstroke_v", include_clipped=True)[idxs]
        downstroke_t = sweep.spike_feature("downstroke_t", include_clipped=True)[idxs]
        downstroke_v = sweep.spike_feature("downstroke_v", include_clipped=True)[idxs]

        ax.plot(upstroke_t, upstroke_v, "x", **kwargs)
        ax.plot(downstroke_t, downstroke_v, "x", **kwargs)
    return ax


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
    "ap_peak": plot_peak,
    "ap_trough": plot_trough,
    "ap_thresh": plot_threshold,
    "upstroke": plot_upstroke,
    "downstroke": plot_downstroke,
    "ap_width": plot_ap_width,
    "fast_trough": plot_fast_trough,
    "slow_trough": plot_slow_trough,
    "ap_adp": plot_ap_adp,
    "ap_ahp": plot_ap_ahp,
    "isi": plot_isi,
    "ap_amp": plot_ap_amp,
    "ap_udr": plot_udr,
    "udr": plot_udr,
}


def plot_spike_feature(
    sweep: EphysSweep, ft: str, ax: Optional[Axes] = None, **kwargs
) -> Axes:
    """Plot spike feature by name.

    Args:
        sweep (EphysSweep): Sweep to plot the feature for.
        ft (str): Name of the feature to plot (all lowercase).
            Can plot all features that are included in the `EphysSweep._spikes_df`
            and all features in `plottable_spike_features`.
        ax (Axes): Matplotlib axes.
        **kwargs: Additional kwargs are passed to the plotting function.

    Returns:
        Axes: Matplotlib axes.
    """
    if ft in plottable_spike_features:
        ax = plottable_spike_features[ft](sweep, ax=ax, **kwargs)
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
        for ft in plottable_spike_features:
            plot_spike_feature(sweep, ft, axes[x])

    axes["b"].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig, axes


def plot_sweepset_diagnostics(
    sweepset: EphysSweepSet,
) -> Tuple[Figure, Axes]:
    """Plot diagnostics overview for the whole sweepset.

    This function is useful to diagnose outliers on the sweepset level.

    Args:
        sweepset (EphysSweepSet): sweepset to diagnose.

    Returns:
        Fig, Axes: figure and axes with plot.
    """
    mosaic = [
        ["set_fts", "set_fts", "set_fts", "r_input"],
        ["fp_trace", "fp_trace", "fp_trace", "rheobase"],
        ["ap_trace", "ap_trace", "ap_trace", "ap_window"],
        ["sag_fts", "set_hyperpol_fts", "set_hyperpol_fts", "rebound_fts"],
    ]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(14, 14), constrained_layout=True)
    # onset, end = (
    #     sweepset.get_sweep_features()
    #     .applymap(strip_info)
    #     .iloc[0][["stim_onset", "stim_end"]]
    # )
    t0, tfin = sweepset.sweeps()[0].t[[0, -1]]
    for ax in axes.values():
        ax.set_xlim(t0, tfin)

    axes["set_fts"].plot(sweepset.t.T, sweepset.v.T, color="grey", alpha=0.5)
    # selection_dict = {
    #     "rebound": default_rebound_sweep_selector(sweepset),
    #     "ap": default_ap_sweep_selector(sweepset),
    #     "sag": default_sag_sweep_selector(sweepset),
    #     "fp": default_spiking_sweep_selector(sweepset),
    #     "tau": sweepset.get_sweepset_feature("tau")["selected_idx"],
    #     "wildness": sweepset.get_sweepset_feature("wildness")["selected_idx"],
    # }
    # selected_sweeps = {
    #     k: np.array(sweepset.sweeps())[v] for k, v in selection_dict.items()
    # }
    # for ft, idx in selection_dict.items():
    #     selected_sweep = np.array(sweepset.sweeps())[idx]
    # if isinstance(selected_sweep, EphysSweepFeatureExtractor):
    #     axes["set_fts"].plot(
    #         selected_sweep.t, selected_sweep.v, label=f"{ft} @ idx={idx}"
    #     )
    axes["set_fts"].legend(title="feature sweeps")

    # axes["fp_trace"].plot(
    #     selected_sweeps["fp"].t,
    #     selected_sweeps["fp"].v,
    #     color="k",
    # )

    # axes["ap_trace"].plot(
    #     selected_sweeps["ap"].t,
    #     selected_sweeps["ap"].v,
    #     color="k",
    # )
    # axes["ap_window"].plot(
    #     selected_sweeps["ap"].t,
    #     selected_sweeps["ap"].v,
    #     color="k",
    # )
    # ap_idx = selected_sweeps["ap"].sweep_feature("ap_peak")["selected_idx"]
    # ap_start = selected_sweeps["ap"].spike_feature("threshold_t")[ap_idx] - 5e-3
    # ap_end = selected_sweeps["ap"].spike_feature("fast_trough_t")[ap_idx] + 5e-3
    axes["ap_window"].set_xlim(ap_start, ap_end)
    # for ft, plot_func in get_available_spike_diagnostics().items():
    #     plot_func(selected_sweeps["fp"], axes["fp_trace"])
    #     plot_func(selected_sweeps["ap"], axes["ap_trace"])
    #     plot_func(selected_sweeps["ap"], axes["ap_window"])
    axes["ap_trace"].axvline(ap_start, color="grey")
    axes["ap_trace"].axvline(ap_end, color="grey", label="selected ap")
    axes["ap_trace"].legend()

    # sweep_is_hyperpol = [is_hyperpol(s) for s in sweepset.sweeps()]
    # hyperpol_idcs = np.where(sweep_is_hyperpol)[0]
    # axes["set_hyperpol_fts"].plot(
    #     sweepset.t[sweep_is_hyperpol].T, sweepset.v[sweep_is_hyperpol].T, color="k"
    # )
    # for selected_sweep in selected_sweeps["tau"]:
    #     plot_sweep_tau(selected_sweep, axes["set_hyperpol_fts"], color="r")
    # plot_sweepset_v_baseline(sweepset, axes["set_hyperpol_fts"])
    axes["set_hyperpol_fts"].legend()

    h, l = axes["set_hyperpol_fts"].get_legend_handles_labels()
    d = {k: v for k, v in zip(l, h)}
    axes["set_hyperpol_fts"].legend(d.values(), d.keys())

    # sag
    # axes["sag_fts"].plot(selected_sweeps["sag"].t, selected_sweeps["sag"].v, color="k")
    # axes["sag_fts"].set_xlim(onset - 0.05, end + 0.05)
    # plot_sweep_sag_area(selected_sweeps["sag"], axes["sag_fts"])
    # plot_sweep_sag_time(selected_sweeps["sag"], axes["sag_fts"])
    axes["sag_fts"].legend()

    # rebound
    # axes["rebound_fts"].plot(
    #     selected_sweeps["rebound"].t, selected_sweeps["rebound"].v, color="k"
    # )
    # axes["rebound_fts"].set_xlim(end - 0.05, None)
    # plot_sweep_rebound(selected_sweeps["rebound"], axes["rebound_fts"])
    # plot_sweep_rebound_latency(selected_sweeps["rebound"], axes["rebound_fts"])
    # plot_sweep_rebound_area(selected_sweeps["rebound"], axes["rebound_fts"])
    # plot_sweep_rebound_avg(selected_sweeps["rebound"], axes["rebound_fts"])
    axes["rebound_fts"].legend()

    fig.text(
        -0.02,
        0.5,
        "U (mV)",
        va="center",
        rotation="vertical",
        fontsize=16,
    )
    fig.text(
        0.5,
        -0.02,
        "t (s)",
        ha="center",
        fontsize=16,
    )

    # plot_sweepset_rheobase(sweepset, axes["rheobase"])
    axes["rheobase"].set_xlabel("I (pA)")
    axes["rheobase"].set_ylabel("f (Hz)")
    axes["rheobase"].legend()

    # plot_sweepset_r_input(sweepset, axes["r_input"])
    axes["r_input"].set_xlabel("I (pA)")
    axes["r_input"].set_ylabel("U (mV)")
    axes["r_input"].legend()

    axes["set_fts"].set_title("All sweeps")
    axes["fp_trace"].set_title("Representative spiking sweep")
    axes["ap_trace"].set_title("Representative AP sweep")
    axes["ap_window"].set_title("Representative AP")
    axes["set_hyperpol_fts"].set_title("Hyperpolarization sweeps")
    axes["sag_fts"].set_title("sag")
    axes["rebound_fts"].set_title("rebound")
    axes["rheobase"].set_title("Rheobase")
    return fig, axes
