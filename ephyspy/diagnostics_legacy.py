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

import warnings
from typing import Any, Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes, Figure
from numpy import ndarray

from ephyspy.allen_sdk.ephys_extractor import EphysSweepFeatureExtractor
from ephyspy.features import (
    default_ap_sweep_selector,
    default_rebound_sweep_selector,
    default_sag_sweep_selector,
    default_spiking_sweep_selector,
)
from ephyspy.utils import (
    EphysSweepSetFeatureExtractor,
    ensure_ft_info,
    get_ap_ft_at_idx,
    is_hyperpol,
    strip_info,
    where_between,
)

############################
### spike level features ###
############################


def get_spike_ft_scatter_func(ft: str) -> Callable:
    def scatter_spike_ft(
        sweep: EphysSweepFeatureExtractor, ax: Axes, **plot_kwargs
    ) -> Axes:
        spike_fts = sweep._spikes_df
        if spike_fts.size:
            if not np.all(spike_fts[f"{ft}_v"]):
                ax.scatter(
                    spike_fts[f"{ft}_t"],
                    spike_fts[f"{ft}_v"],
                    s=10,
                    label=ft,
                    **plot_kwargs,
                )
        return ax

    return scatter_spike_ft


plot_spike_peaks = get_spike_ft_scatter_func("peak")
plot_spike_troughs = get_spike_ft_scatter_func("trough")
plot_spike_thresholds = get_spike_ft_scatter_func("threshold")
plot_spike_upstrokes = get_spike_ft_scatter_func("upstroke")
plot_spike_downstrokes = get_spike_ft_scatter_func("downstroke")
plot_spike_fast_troughs = get_spike_ft_scatter_func("fast_trough")
plot_spike_slow_troughs = get_spike_ft_scatter_func("slow_trough")


def get_fwhm(
    t: ndarray, v: ndarray, t_start: float, t_end: float
) -> Tuple[float, float, float]:
    """Get full width at half maximum of a spike.

    Args:
        t (ndarray): time array.
        v (ndarray): voltage array.
        t_start (float): start time of spike.
        t_end (float): end time of spike.

    Returns:
        Tuple[float, float, float]: full width at half maximum,
            time of half maximum upstroke, time of half maximum downstroke.
    """
    in_T = where_between(t, t_start, t_end)
    v_peak = np.max(v[in_T])
    v_start = v[in_T][0]
    t_peak = t[in_T][np.argmax(v[in_T])]
    upstroke = where_between(t, t_start, t_peak)
    downstroke = where_between(t, t_peak, t_end)
    fwhm = v_start + (v_peak - v_start) / 2
    hm_up_idx = np.argmin(np.abs(v[upstroke] - fwhm))
    hm_down_idx = np.argmin(np.abs(v[downstroke] - fwhm))
    hm_up_t = t[upstroke][hm_up_idx]
    hm_down_t = t[downstroke][hm_down_idx]
    return fwhm, hm_up_t, hm_down_t


def plot_spike_width(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    spike_fts = sweep._spikes_df
    if spike_fts.size:
        if not np.all(spike_fts["width"]):
            t_threshold = sweep.spike_feature("threshold_t", include_clipped=True)
            t_peak = sweep.spike_feature("peak_t", include_clipped=True)
            t_next = t_peak + 1.0 * (
                t_peak - t_threshold
            )  # T interval w.r.t. threshold

            fwhm_v = np.zeros_like(t_threshold)
            hm_up_t = np.zeros_like(t_threshold)
            hm_down_t = np.zeros_like(t_threshold)
            for i, (t_th, t_n) in enumerate(zip(t_threshold, t_next)):
                fwhm_i = get_fwhm(sweep.t, sweep.v, t_th, t_n)
                fwhm_v[i], hm_up_t[i], hm_down_t[i] = fwhm_i

            ax.hlines(
                fwhm_v,
                hm_up_t,
                hm_down_t,
                label="width",
                ls="--",
                color=color,
                **plot_kwargs,
            )
    return ax


def plot_spike_adp(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    spike_fts = sweep._spikes_df
    if spike_fts.size:
        if not np.all(spike_fts["adp_v"]):
            ax.vlines(
                0.5 * (spike_fts[f"adp_t"] + spike_fts["fast_trough_t"]),
                spike_fts["adp_v"],
                spike_fts["fast_trough_v"],
                ls="--",
                lw=1,
                label="adp",
            )
    return ax


def plot_spike_ahp(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    spike_fts = sweep._spikes_df
    if spike_fts.size:
        if not np.all(spike_fts["fast_trough_v"]):
            ax.vlines(
                0.5 * (spike_fts[f"fast_trough_t"] + spike_fts["threshold_t"]),
                spike_fts["fast_trough_v"],
                spike_fts["threshold_v"],
                ls="--",
                lw=1,
                label="ahp",
            )
    return ax


def plot_spike_amp(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    warnings.warn("spike spike_amp plotting is not yet implemented!")
    return ax


def get_available_spike_diagnostics():
    spike_ft_plot_dict = {
        "peak": plot_spike_peaks,
        "trough": plot_spike_troughs,
        "threshold": plot_spike_thresholds,
        "upstroke": plot_spike_upstrokes,
        "downstroke": plot_spike_downstrokes,
        "width": plot_spike_width,
        "fast_trough": plot_spike_fast_troughs,
        "slow_trough": plot_spike_slow_troughs,
        "adp": plot_spike_adp,
        "ahp": plot_spike_ahp,  # TODO: Check why nan
    }
    return spike_ft_plot_dict


def plot_spike_diagnostics(
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

    # plot spike features
    for x in ["a", "b"]:
        for ft, plot_func in get_available_spike_diagnostics().items():
            plot_func(sweep, axes[x])

    axes["b"].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    return fig, axes


############################
### sweep level features ###
############################


def plot_sweep_stim_amp(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "stim_amp" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("stim_amp"))
    if not np.isnan(ft["value"]):
        ax.plot(
            ft["t_amp"],
            ft["value"],
            "x",
            label="stim_amp",
            color=color,
            **plot_kwargs,
        )
    return ax


def plot_sweep_stim_onset(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "stim_onset" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("stim_onset"))
    if not np.isnan(ft["value"]):
        ax.plot(
            ft["t_onset"],
            ft["value"],
            "x",
            label="stim_onset",
            color=color,
            **plot_kwargs,
        )
    return ax


def plot_sweep_stim_end(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "stim_end" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("stim_end"))
    if not np.isnan(ft["value"]):
        ax.plot(
            ft["t_end"], ft["value"], "x", label="stim_end", color=color, **plot_kwargs
        )
    return ax


def plot_sweep_ap_latency(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color=None,
    include_details=False,
    **plot_kwargs,
):
    """Plot diagnostics for the "ap_latency" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("ap_latency"))
    if not np.isnan(ft["value"]):
        ax.hlines(
            ft["v_first_spike"],
            ft["onset"],
            ft["t_first_spike"],
            label="ap_latency",
            color=color,
            **plot_kwargs,
        )
    return ax


def plot_sweep_v_deflect(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "v_deflect" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("v_deflect"))
    if not np.isnan(ft["value"]):
        ax.plot(
            ft["t_deflect"],
            ft["v_deflect"],
            label="v_deflect interval",
            color=color,
            **plot_kwargs,
        )
        ax.plot(
            ft["t_deflect"],
            np.ones_like(ft["t_deflect"]) * ft["value"],
            ls="--",
            color=color,
            label="v_deflect",
        )
    return ax


def plot_sweep_v_baseline(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "v_baseline" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("v_baseline"))
    if not np.isnan(ft["value"]):
        ax.plot(
            ft["t_baseline"],
            ft["v_baseline"],
            color=color,
            label="v_baseline interval",
            **plot_kwargs,
        )
        # ax.plot(
        #     ft["t_baseline"],
        #     np.ones_like(ft["t_baseline"]) * ft["value"],
        #     ls="--",
        # color=color,
        #     label="v_baseline",
        # )
        ax.axhline(
            ft["value"],
            ls="--",
            color=color,
            label="v_baseline",
        )
    return ax


def plot_sweep_tau(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "tau" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("tau"))
    if not np.isnan(ft["value"]):
        y = lambda t: ft["y0"] + ft["a"] * np.exp(-ft["inv_tau"] * t)
        where_fit = where_between(sweep.t, ft["fit_start"], ft["fit_end"])
        t_offset = sweep.t[where_fit][0]
        t_fit = sweep.t[where_fit] - t_offset
        ax.plot(
            t_fit + t_offset,
            sweep.v[where_fit],
            label="tau interval",
            color=color,
            **plot_kwargs,
        )
        ax.plot(t_fit + t_offset, y(t_fit), ls="--", color="k", label="tau fit")
    return ax


def plot_sweep_num_ap(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "num_ap" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("num_ap"))
    v_baseline = sweep.sweep_feature("v_baseline")["value"]
    if not np.isnan(ft["value"]):
        ax.plot(
            ft["peak_t"],
            ft["peak_v"],
            "x",
            label="num_ap / ap_freq",
            color=color,
            **plot_kwargs,
        )
        # ax.vlines(
        #     ft["peak_t"],
        #     ymin=v_baseline,
        #     ymax=np.max(ft["peak_v"]),
        #     label="num_ap",
        #     color=color,
        #     **plot_kwargs,
        # )
    return ax


def plot_sweep_ap_freq(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "ap_freq" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    return plot_sweep_num_ap(sweep, ax, color, include_details, **plot_kwargs)


def plot_sweep_ap_freq_adapt(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "ap_freq_adapt" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("ap_freq_adapt"))
    if not np.isnan(ft["value"]):
        v_baseline = sweep.sweep_feature("v_baseline")["value"]
        peaks_t = sweep.sweep_feature("num_ap")["peak_t"]
        peaks_v = sweep.sweep_feature("num_ap")["peak_v"]
        for i, c in zip(["1st", "2nd"], ["blue", "red"]):
            # ax.plot(
            #     ft[f"t_{i}_half"],
            #     sweep.v[ft[f"where_{i}_half"]],
            # label=f"ap {i} half",
            # color=color,
            #     **plot_kwargs,
            # )
            in_half = where_between(peaks_t, *ft[f"t_{i}_half"][[0, -1]])
            ax.plot(
                peaks_t[in_half],
                peaks_v[in_half],
                ".",
                label=f"ap {i} half",
                color=c,
                **plot_kwargs,
            )
            # ax.vlines(
            #     peaks_t[in_half],
            #     ymin=v_baseline,
            #     ymax=np.max(sweep.v),
            #     label=f"ap {i} half",
            #     color=c,
            #     **plot_kwargs,
            # )
    return ax


def plot_sweep_r_input(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "r_input" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    warnings.warn("r_input sweep plotting is not yet implemented yet!")
    return ax


def plot_sweep_ap_amp_slope(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "ap_amp_slope" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("ap_amp_slope"))
    y = lambda t: ft["intercept"] + ft["slope"] * t
    if not np.isnan(ft["value"]):
        ts = ft["peak_t"]
        # ts = sweep.t
        ax.plot(ts, y(ts), "--", label="ap_amp_slope", color=color, **plot_kwargs)
    return ax


def plot_sweep_sag(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "sag" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    warnings.warn("sag sweep plotting is not yet implemented yet!")
    return ax


def plot_sweep_sag_fraction(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "sag_fraction" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    warnings.warn("sag_fraction sweep plotting is not yet implemented yet!")
    return ax


def plot_sweep_sag_ratio(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "sag_ratio" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    warnings.warn("sag_ratio sweep plotting is not yet implemented yet!")
    return ax


def plot_sweep_sag_area(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "sag_area" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("sag_area"))
    if not np.isnan(ft["value"]):
        ax.plot(ft["t_sag"], ft["v_sag"], **plot_kwargs)
        ax.fill_between(
            ft["t_sag"],
            ft["v_sag"],
            ft["v_sagline"],
            alpha=0.5,
            color=color,
            label="sag_area",
        )
    return ax


def plot_sweep_sag_time(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "sag_time" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("sag_time"))
    if not np.isnan(ft["value"]):
        ax.hlines(
            sweep.v[ft["where_sag"]][0],
            xmin=ft["sag_t_start"],
            xmax=ft["sag_t_end"],
            label="sag_time",
            color=color,
            **plot_kwargs,
        )
    return ax


def plot_sweep_v_plateau(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "v_plateau" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("v_plateau"))
    if not np.isnan(ft["value"]):
        ax.plot(
            ft["t_plateau"],
            ft["v_plateau"],
            label="v_plateau interval",
            color=color,
            **plot_kwargs,
        )
        ax.hlines(
            ft["value"],
            *ft["t_plateau"][[0, -1]],
            ls="--",
            label="v_plateau",
            color=color,
            **plot_kwargs,
        )
    return ax


def plot_sweep_rebound(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "rebound" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("rebound"))
    if not np.isnan(ft["value"]):
        ax.vlines(
            sweep.t[ft["idx_rebound"]],
            ft["v_baseline"],
            sweep.v[ft["idx_rebound"]],
            label="rebound",
            color=color,
            **plot_kwargs,
        )
        if include_details:
            ax.plot(
                ft["t_rebound"],
                ft["v_rebound"],
                label="rebound interval",
                color=color,
                **plot_kwargs,
            )
    return ax


def plot_sweep_rebound_aps(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "rebound_aps" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("rebound_aps"))
    warnings.warn("rebound_aps sweep plotting is not yet implemented yet!")
    if not np.isnan(ft["value"]):
        pass
    return ax


def plot_sweep_rebound_latency(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    # TODO: Check why stim end not quite aligned with v(t)!
    ft = ensure_ft_info(sweep.sweep_feature("rebound_latency"))
    stim_end = sweep.sweep_feature("stim_end")["value"]
    end_idx = sweep.sweep_feature("stim_end")["idx_end"]
    if not np.isnan(ft["value"]):
        ax.fill_between(
            sweep.t,
            sweep.v,
            sweep.v[end_idx],
            where=where_between(sweep.t, stim_end, ft["t_rebound_reached"]),
            alpha=0.5,
            label="rebound_latency",
            color=color,
            **plot_kwargs,
        )
        # ax.axvline(
        #     ft["t_rebound_reached"],
        #     ls="--",
        #     label="rebound_latency",
        # color=color,
        #     **plot_kwargs,
        # )
        # ax.axvline(
        #     stim_end,
        #     ls="--",
        # color=color,
        #     **plot_kwargs,
        # )
    return ax


def plot_sweep_rebound_area(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "rebound_area" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("rebound_area"))
    if not np.isnan(ft["value"]):
        ax.fill_between(
            sweep.t,
            sweep.v,
            ft["v_baseline"],
            where=ft["where_rebound"],
            alpha=0.5,
            label="rebound_area",
            color=color,
            **plot_kwargs,
        )
    return ax


def plot_sweep_rebound_avg(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "rebound_avg" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("rebound_avg"))
    if not np.isnan(ft["value"]):
        ax.plot(
            ft["t_rebound"],
            ft["v_rebound"],
            label="rebound interval",
            color=color,
            **plot_kwargs,
        )
        ax.hlines(
            [ft["value"] + ft["v_baseline"], ft["v_baseline"]],
            # np.mean(ft["v_rebound"]),
            *ft["t_rebound"][[0, -1]],
            ls="--",
            label="rebound_avg",
            color=color,
            **plot_kwargs,
        )
        # ax.fill_between(
        #     ft["t_rebound"],
        #     ft["value"] + ft["v_baseline"],
        #     ft["v_baseline"],
        #     alpha=0.5,
        # color=color,
        # )
    return ax


def plot_sweep_v_rest(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "v_rest" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("v_rest"))
    if not np.isnan(ft["value"]):
        ax.plot(
            sweep.t,
            sweep.v - ft["r_input"] * ft["dc_offset"] * 1e-3,
            label="v(t) - r_in*dc_offset",
        )
        ax.axhline(ft["value"], ls="--", color=color, label="v_rest")
    return ax


def plot_sweep_num_bursts(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "num_bursts" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("num_bursts"))
    if not np.isnan(ft["value"]):
        t_burst_start = ft["t_burst_start"]
        t_burst_end = ft["t_burst_end"]
        for i, (t_start, t_end) in enumerate(zip(t_burst_start, t_burst_end)):
            ax.axvspan(
                t_start,
                t_end,
                alpha=0.5,
                color=color,
                label=f"burst {i+1}",
                **plot_kwargs,
            )
    return ax


def plot_sweep_burstiness(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "burstiness" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("burstiness"))
    if not np.isnan(ft["value"]):
        t_burst_start = ft["t_burst_start"]
        t_burst_end = ft["t_burst_end"]
        for i, (t_start, t_end) in enumerate(zip(t_burst_start, t_burst_end)):
            ax.axvspan(
                t_start,
                t_end,
                alpha=0.5,
                color=color,
                label=f"burst {i+1}",
                **plot_kwargs,
            )
    return ax


def plot_sweep_wildness(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "wildness" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    warnings.warn("wildness sweep plotting is not yet implemented yet!")
    ft = ensure_ft_info(sweep.sweep_feature("wildness"))
    if not np.isnan(ft["value"]):
        pass
    return ax


def plot_sweep_ahp(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "ahp" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("ahp"))
    if not np.isnan(ft["value"]):
        trough_t = get_ap_ft_at_idx(sweep, "fast_trough_t", ft["selected_idx"])
        trough_v = get_ap_ft_at_idx(sweep, "fast_trough_v", ft["selected_idx"])
        thresh_t = get_ap_ft_at_idx(sweep, "threshold_t", ft["selected_idx"])
        thresh_v = get_ap_ft_at_idx(sweep, "threshold_v", ft["selected_idx"])
        stim_onset = sweep.sweep_feature("stim_onset")["value"]
        stim_end = sweep.sweep_feature("stim_end")["value"]

        # ax.hlines([trough_v, thresh_v], thresh_t, trough_t, **plot_kwargs)
        # ax.vlines(
        #     0.5 * (thresh_t + trough_t),
        #     thresh_v,
        #     trough_v,
        #     label="(selected) ahp",
        # color=color,
        #     **plot_kwargs,
        # )
        ax.plot(
            [trough_t, thresh_t], [trough_v, thresh_v], "x", color=color, **plot_kwargs
        )
        ax.hlines(
            [trough_v, thresh_v],
            stim_onset,
            stim_end,
            label="ahp",
            color=color,
            **plot_kwargs,
        )

    return ax


def plot_sweep_adp(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "adp" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    # TODO: Check why it is always nan!
    ft = ensure_ft_info(sweep.sweep_feature("ahp"))
    if not np.isnan(ft["value"]):
        trough_t = get_ap_ft_at_idx(sweep, "fast_trough_t", ft["selected_idx"])
        trough_v = get_ap_ft_at_idx(sweep, "fast_trough_v", ft["selected_idx"])
        adp_t = get_ap_ft_at_idx(sweep, "adp_t", ft["selected_idx"])
        adp_v = get_ap_ft_at_idx(sweep, "adp_v", ft["selected_idx"])
        stim_onset = sweep.sweep_feature("stim_onset")["value"]
        stim_end = sweep.sweep_feature("stim_end")["value"]

        # ax.hlines([adp_v, trough_v], trough_t, adp_t, **plot_kwargs)
        # ax.vlines(
        #     0.5 * (trough_t + adp_t),
        #     trough_v,
        #     adp_v,
        #     label="(selected) adp",
        # color=color,
        #     **plot_kwargs,
        # )
        ax.plot([trough_t, adp_t], [trough_v, adp_v], "x", color=color, **plot_kwargs)
        ax.hlines(
            [adp_v, trough_v],
            stim_onset,
            stim_end,
            label="adp",
            color=color,
            **plot_kwargs,
        )

    return ax


def plot_sweep_ap_thresh(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "ap_thresh" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("ap_thresh"))
    if not np.isnan(ft["value"]):
        thresh_t = get_ap_ft_at_idx(sweep, "threshold_t", ft["selected_idx"])
        thresh_v = get_ap_ft_at_idx(sweep, "threshold_v", ft["selected_idx"])
        stim_onset = sweep.sweep_feature("stim_onset")["value"]
        stim_end = sweep.sweep_feature("stim_end")["value"]
        ax.plot(thresh_t, thresh_v, "x", color=color, **plot_kwargs)
        ax.hlines(
            ft["value"],
            stim_onset,
            stim_end,
            ls="--",
            label="ap_thresh",
            color=color,
            **plot_kwargs,
        )
    return ax


def plot_sweep_ap_amp(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "ap_amp" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("ap_amp"))
    if not np.isnan(ft["value"]):
        thresh_t = get_ap_ft_at_idx(sweep, "threshold_t", ft["selected_idx"])
        thresh_v = get_ap_ft_at_idx(sweep, "threshold_v", ft["selected_idx"])
        peak_t = get_ap_ft_at_idx(sweep, "peak_t", ft["selected_idx"])
        peak_v = get_ap_ft_at_idx(sweep, "peak_v", ft["selected_idx"])
        stim_onset = sweep.sweep_feature("stim_onset")["value"]
        stim_end = sweep.sweep_feature("stim_end")["value"]
        ax.plot([thresh_t, peak_t], [thresh_v, peak_v], "x", color=color, **plot_kwargs)
        ax.hlines(
            [thresh_v, peak_v],
            stim_onset,
            stim_end,
            ls="--",
            label="ap_amp",
            color=color,
            **plot_kwargs,
        )
    return ax


def plot_sweep_ap_width(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "ap_width" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("ap_width"))
    if not np.isnan(ft["value"]):
        ap_idx = ft["selected_idx"]
        thresh_t = sweep.spike_feature("threshold_t", include_clipped=True)[ap_idx]
        t_peak = sweep.spike_feature("peak_t", include_clipped=True)[ap_idx]
        t_next = t_peak + 1.0 * (t_peak - thresh_t)  # T interval w.r.t. threshold
        fwhm_v, t_up_fwhm, t_down_fwhm = get_fwhm(sweep.t, sweep.v, thresh_t, t_next)

        ax.hlines(
            fwhm_v,
            t_up_fwhm,
            t_down_fwhm,
            ls="--",
            label="ap_width",
            color=color,
            **plot_kwargs,
        )
    return ax


def plot_sweep_ap_peak(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "ap_peak" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("ap_peak"))
    if not np.isnan(ft["value"]):
        peak_t = get_ap_ft_at_idx(sweep, "peak_t", ft["selected_idx"])
        peak_v = get_ap_ft_at_idx(sweep, "peak_v", ft["selected_idx"])
        stim_onset = sweep.sweep_feature("stim_onset")["value"]
        stim_end = sweep.sweep_feature("stim_end")["value"]
        ax.plot(peak_t, peak_v, "x", color=color, **plot_kwargs)
        ax.hlines(
            peak_v,
            stim_onset,
            stim_end,
            ls="--",
            label="ap_peak",
            color=color,
            **plot_kwargs,
        )
    return ax


def plot_sweep_ap_trough(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "ap_trough" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("ap_trough"))
    if not np.isnan(ft["value"]):
        trough_t = get_ap_ft_at_idx(sweep, "fast_trough_t", ft["selected_idx"])
        trough_v = get_ap_ft_at_idx(sweep, "fast_trough_v", ft["selected_idx"])
        stim_onset = sweep.sweep_feature("stim_onset")["value"]
        stim_end = sweep.sweep_feature("stim_end")["value"]
        ax.plot(trough_t, trough_v, "x", color=color, **plot_kwargs)
        ax.hlines(
            trough_v,
            stim_onset,
            stim_end,
            ls="--",
            label="ap_trough",
            color=color,
            **plot_kwargs,
        )
    return ax


def plot_sweep_udr(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    """Plot diagnostics for the "udr" feature that are returned if return_ft_info=True.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to plot.
        ax (Axes): axes to plot on.
        color (Any, optional): color to make feature distinguishable. Defaults to None.
        detail (bool, optional): whether to plot detailed diagnostics. Defaults to True.
        **plot_kwargs: kwargs to pass to ax.plot.

    Returns:
        Axes: axes with plot.
    """
    ft = ensure_ft_info(sweep.sweep_feature("udr"))
    if not np.isnan(ft["value"]):
        us_t = get_ap_ft_at_idx(sweep, "upstroke_t", ft["selected_idx"])
        us_v = get_ap_ft_at_idx(sweep, "upstroke_v", ft["selected_idx"])
        ds_t = get_ap_ft_at_idx(sweep, "downstroke_t", ft["selected_idx"])
        ds_v = get_ap_ft_at_idx(sweep, "downstroke_v", ft["selected_idx"])
        stim_onset = sweep.sweep_feature("stim_onset")["value"]
        stim_end = sweep.sweep_feature("stim_end")["value"]
        ax.plot([us_t, ds_t], [us_v, ds_v], "x", color=color, **plot_kwargs)
        ax.hlines(
            [us_v, ds_v],
            stim_onset,
            stim_end,
            ls="--",
            label="udr",
            color=color,
            **plot_kwargs,
        )
    return ax


def plot_sweep_diagnostics(sweep, window=[0.4, 0.45]):
    mosaic = "aaabb\naaabb\ncccbb"
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(12, 4), constrained_layout=True)
    colors = plt.cm.tab20(np.linspace(0, 1, 40))

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

    # plot sweep features
    sweep_fts = sweep._sweep_features
    if len(sweep_fts) > 1:  # 1 since dc is always present
        if sum([isinstance(x, dict) for x in sweep_fts.values()]) > 5:
            # plot detailed diagnostics
            for c, (ft, plot_func) in enumerate(
                get_available_sweep_diagnostics().items()
            ):
                if "stim" in ft:
                    ax = axes["c"]
                    ax.set_ylabel("Current (pA)")
                    plot_func(sweep, ax, color=colors[c])
                    # ax.set_xlim(0, 1)
                    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
                else:
                    for ax in [axes["a"], axes["b"]]:
                        ax.set_ylabel("Voltage (mV)")
                        plot_func(sweep, ax, color=colors[c])
                        ax.set_xlabel("Time (s)")
                    ax.set_xlim(*window)
                    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
        else:
            # plot crude diagnostics
            axes["c"].axhline(sweep_fts["stim_amp"], c="r", ls="--", label="stim amp")
            axes["c"].axhline(0, c="r", ls="--")
            axes["c"].axvline(
                sweep_fts["stim_onset"], c="tab:blue", ls="--", label="stim onset"
            )
            axes["c"].axvline(
                sweep_fts["stim_end"], c="tab:orange", ls="--", label="stim end"
            )
            axes["c"].legend()

            t_fts = [
                "stim_onset",
                "stim_end",
                "ap_latency",
            ]
            v_fts = [
                "v_deflect",
                "v_baseline",
                "v_plateau",
                "v_rest",
                "ap_thresh",
                "ap_peak",
                "ap_trough",
            ]
            for x in ["a", "b"]:
                for i, ft in enumerate(t_fts):
                    axes[x].axvline(sweep_fts[ft], label=ft, ls="--", c=colors[i])
                for i, ft in enumerate(v_fts):
                    axes[x].axhline(
                        sweep_fts[ft], label=ft, ls="--", c=colors[i + len(t_fts)]
                    )

        axes["b"].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    return fig, axes


def get_available_sweep_diagnostics():
    ft_plot_dict = {
        "stim_amp": plot_sweep_stim_amp,
        "stim_onset": plot_sweep_stim_onset,
        "stim_end": plot_sweep_stim_end,
        "ap_latency": plot_sweep_ap_latency,
        "v_baseline": plot_sweep_v_baseline,
        "v_deflect": plot_sweep_v_deflect,
        "tau": plot_sweep_tau,
        "num_ap": plot_sweep_num_ap,
        "ap_freq": plot_sweep_ap_freq,
        "ap_freq_adapt": plot_sweep_ap_freq_adapt,
        "ap_amp_slope": plot_sweep_ap_amp_slope,
        # "r_input": plot_sweep_r_input,
        # "sag": plot_sweep_sag,
        # "sag_fraction": plot_sweep_sag_fraction,
        # "sag_ratio": plot_sweep_sag_ratio,
        "sag_area": plot_sweep_sag_area,
        "sag_time": plot_sweep_sag_time,
        "v_plateau": plot_sweep_v_plateau,
        "rebound": plot_sweep_rebound,
        "rebound_aps": plot_sweep_rebound_aps,
        "rebound_latency": plot_sweep_rebound_latency,
        "rebound_area": plot_sweep_rebound_area,
        "rebound_avg": plot_sweep_rebound_avg,
        "v_rest": plot_sweep_v_rest,
        "num_bursts": plot_sweep_num_bursts,
        "burstiness": plot_sweep_burstiness,
        "wildness": plot_sweep_wildness,
        "ahp": plot_sweep_ahp,
        "adp": plot_sweep_adp,
        "ap_thresh": plot_sweep_ap_thresh,
        "ap_amp": plot_sweep_ap_amp,
        "ap_width": plot_sweep_ap_width,
        "ap_peak": plot_sweep_ap_peak,
        "ap_trough": plot_sweep_ap_trough,
        "udr": plot_sweep_udr,
    }
    return ft_plot_dict


###############################
### sweepset level features ###
###############################


def get_selected_sweep_plotfunc(ft_name, sweep_ft_plot_func):
    def plot_sweepset_ft(
        sweepset: EphysSweepSetFeatureExtractor,
        ax: Axes,
        color: Any = None,
        include_details=False,
        **plot_kwargs,
    ) -> Axes:
        ft = ensure_ft_info(sweepset.get_sweepset_feature(ft_name))
        if not np.isnan(ft["value"]):
            idx = ft["selected_idx"]
            selected_sweep = sweepset.sweeps()[idx]
            ax = sweep_ft_plot_func(
                selected_sweep,
                ax,
                color=color,
                include_details=include_details,
                **plot_kwargs,
            )
        return ax

    return plot_sweepset_ft


def plot_sweepset_tau(
    sweepset: EphysSweepSetFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    ft = ensure_ft_info(sweepset.get_sweepset_feature("tau"))
    if not np.isnan(ft["value"]):
        idxs = ft["selected_idx"]
        for idx in idxs:
            selected_sweep = sweepset.sweeps()[idx]
            ax = plot_sweep_tau(
                selected_sweep,
                ax,
                color=color,
                include_details=include_details,
                **plot_kwargs,
            )
    return ax


def plot_sweepset_r_input(
    sweepset: EphysSweepSetFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    ft = ensure_ft_info(sweepset.get_sweepset_feature("r_input"))
    if not np.isnan(ft["value"]):
        i = ft["i_amp"]
        v = ft["v_deflect"]
        slope = ft["raw_slope"]
        intercept = ft["v_intercept"]
        ax.plot(i, v, "o", color=color, label="I(V)", **plot_kwargs)
        ax.plot(
            i, slope * i + intercept, color=color, label="r_input fit", **plot_kwargs
        )
        ax.set_xlim(np.min(i) - 5, np.max(i) + 5)
    return ax


plot_sweepset_v_rest = get_selected_sweep_plotfunc("v_rest", plot_sweep_v_rest)
plot_sweepset_v_baseline = get_selected_sweep_plotfunc(
    "v_baseline", plot_sweep_v_baseline
)


def plot_sweepset_slow_hyperpolarization(
    sweepset: EphysSweepSetFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    warnings.warn(
        "slow_hyperpolarization sweepset plotting is not yet implemented yet!"
    )
    ft = ensure_ft_info(sweepset.get_sweepset_feature("slow_hyperpolarization"))
    if not np.isnan(ft["value"]):
        pass
    return ax


plot_sweepset_sag = get_selected_sweep_plotfunc("sag", plot_sweep_sag)
plot_sweepset_sag_ratio = get_selected_sweep_plotfunc("sag_ratio", plot_sweep_sag_ratio)
plot_sweepset_sag_fraction = get_selected_sweep_plotfunc(
    "sag_fraction", plot_sweep_sag_fraction
)
plot_sweepset_sag_area = get_selected_sweep_plotfunc("sag_area", plot_sweep_sag_area)
plot_sweepset_sag_time = get_selected_sweep_plotfunc("sag_time", plot_sweep_sag_time)
plot_sweepset_rebound = get_selected_sweep_plotfunc("rebound", plot_sweep_rebound)
plot_sweepset_rebound_aps = get_selected_sweep_plotfunc(
    "rebound_aps", plot_sweep_rebound_aps
)
plot_sweepset_rebound_area = get_selected_sweep_plotfunc(
    "rebound_area", plot_sweep_rebound_area
)
plot_sweepset_rebound_latency = get_selected_sweep_plotfunc(
    "rebound_latency", plot_sweep_rebound_latency
)
plot_sweepset_rebound_avg = get_selected_sweep_plotfunc(
    "rebound_avg", plot_sweep_rebound_avg
)
plot_sweepset_num_spikes = get_selected_sweep_plotfunc("num_ap", plot_sweep_num_ap)
plot_sweepset_ap_freq = get_selected_sweep_plotfunc("ap_freq", plot_sweep_ap_freq)
plot_sweepset_wildness = get_selected_sweep_plotfunc("wildness", plot_sweep_wildness)
plot_sweepset_ap_freq_adapt = get_selected_sweep_plotfunc(
    "ap_freq_adapt", plot_sweep_ap_freq_adapt
)
plot_sweepset_ap_amp_slope = get_selected_sweep_plotfunc(
    "ap_amp_slope", plot_sweep_ap_amp_slope
)
# plot_sweepset_fano_factor = get_selected_sweep_plotfunc("fano_factor", plot_sweep_fano_factor)
# plot_sweepset_ap_fano_factor = get_selected_sweep_plotfunc("ap_fano_factor", plot_sweep_ap_fano_factor)
# plot_sweepset_cv = get_selected_sweep_plotfunc("cv", plot_sweep_cv)
# plot_sweepset_ap_cv = get_selected_sweep_plotfunc("ap_cv", plot_sweep_ap_cv)
plot_sweepset_burstiness = get_selected_sweep_plotfunc(
    "burstiness", plot_sweep_burstiness
)
plot_sweepset_num_bursts = get_selected_sweep_plotfunc(
    "num_bursts", plot_sweep_burstiness
)
# plot_sweepset_isi_adapt = get_selected_sweep_plotfunc(
#     "isi_adapt", plot_sweep_isi_adapt
# )
# plot_sweepset_isi_adapt_avg = get_selected_sweep_plotfunc(
#     "isi_adapt_avg", plot_sweep_isi_adapt_avg
# )
# plot_sweepset_ap_amp_adapt = get_selected_sweep_plotfunc(
#     "ap_amp_adapt", plot_sweep_ap_amp_adapt
# )
# plot_sweepset_ap_amp_adapt_avg = get_selected_sweep_plotfunc(
#     "ap_amp_adapt_avg", plot_sweep_ap_amp_adapt_avg
# )
plot_sweepset_ap_latency = get_selected_sweep_plotfunc(
    "ap_latency", plot_sweep_ap_latency
)
plot_sweepset_ahp = get_selected_sweep_plotfunc("ahp", plot_sweep_ahp)
plot_sweepset_adp = get_selected_sweep_plotfunc("adp", plot_sweep_adp)
plot_sweepset_ap_thresh = get_selected_sweep_plotfunc("ap_thresh", plot_sweep_ap_thresh)
plot_sweepset_ap_amp = get_selected_sweep_plotfunc("ap_amp", plot_sweep_ap_amp)
plot_sweepset_ap_width = get_selected_sweep_plotfunc("ap_width", plot_sweep_ap_width)
plot_sweepset_ap_peak = get_selected_sweep_plotfunc("ap_peak", plot_sweep_ap_peak)
plot_sweepset_ap_trough = get_selected_sweep_plotfunc("ap_trough", plot_sweep_ap_trough)
plot_sweepset_ap_udr = get_selected_sweep_plotfunc("udr", plot_sweep_udr)


def plot_sweepset_dfdi(
    sweepset: EphysSweepSetFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    warnings.warn("dfdi sweepset plotting is not yet implemented yet!")
    ft = ensure_ft_info(sweepset.get_sweepset_feature("dfdi"))
    if not np.isnan(ft["value"]):
        i_fit = ft["i_fit"]
        f_fit = ft["f_fit"]
        slope = ft["value"]
        f_intercept = ft["f_intercept"]
        f_intercept = ft["f_intercept"]
        ax.plot(i_fit, f_fit, "o", color=color, label="f(I)", **plot_kwargs)
        ax.plot(
            i_fit,
            slope * i_fit + f_intercept,
            color=color,
            label="dfdi fit",
            **plot_kwargs,
        )
    return ax


def plot_sweepset_rheobase(
    sweepset: EphysSweepSetFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=False,
    **plot_kwargs,
) -> Axes:
    ft = ensure_ft_info(sweepset.get_sweepset_feature("rheobase"))
    dfdi_ft = ensure_ft_info(sweepset.get_sweepset_feature("dfdi"))
    if not np.isnan(ft["value"]):
        i_intercept = ft["value"]
        i_sub = ft["i_sub"]
        i_sup = ft["i_sup"]
        f_sup = ft["f_sup"]

        dfdi = ft["dfdi"]
        dc_offset = ft["dc_offset"]

        if not np.isnan(dfdi):
            has_spikes = ~np.isnan(dfdi_ft["f"])
            n_no_spikes = np.sum(~has_spikes)
            i = dfdi_ft["i"]
            f = dfdi_ft["f"]
            f_intercept = dfdi_ft["f_intercept"]

            ax.plot(
                i[has_spikes][:5],
                f[has_spikes][:5],
                "o",
                color=color,
                label="f(I)",
                **plot_kwargs,
            )
            ax.plot(
                i[: n_no_spikes + 5],
                dfdi * i[: n_no_spikes + 5] + f_intercept,
                color=color,
                label="f(I) fit",
                **plot_kwargs,
            )
            ax.set_xlim(i[0] - 5, i[n_no_spikes + 5] + 5)
        else:
            ax.set_xlim(i_sub - 5, i_sup + 5)

        ax.plot(
            i_sup,
            f_sup,
            "o",
            color=color,
            label="i_sup",
            **plot_kwargs,
        )
        ax.axvline(
            i_intercept + dc_offset,
            color=color,
            ls="--",
            label="rheobase\n(w.o. dc)",
            **plot_kwargs,
        )
        ax.axvline(
            i_intercept,
            color=color,
            label="rheobase\n(incl. dc)",
            **plot_kwargs,
        )
        ax.plot(i_sub, 0, "o", color=color, label="i_sub", **plot_kwargs)
    return ax


def get_available_sweepset_diagnostics():
    sweepset_ft_plot_dict = {
        "tau": plot_sweepset_tau,
        "r_input": plot_sweepset_r_input,
        "v_rest": plot_sweepset_v_rest,
        "v_baseline": plot_sweepset_v_baseline,
        "slow_hyperpolarization": plot_sweepset_slow_hyperpolarization,
        "sag": plot_sweepset_sag,
        "sag_ratio": plot_sweepset_sag_ratio,
        "sag_fraction": plot_sweepset_sag_fraction,
        "sag_area": plot_sweepset_sag_area,
        "sag_time": plot_sweepset_sag_time,
        "rebound": plot_sweepset_rebound,
        "rebound_aps": plot_sweepset_rebound_aps,
        "rebound_area": plot_sweepset_rebound_area,
        "rebound_latency": plot_sweepset_rebound_latency,
        "rebound_avg": plot_sweepset_rebound_avg,
        "num_ap": plot_sweepset_num_spikes,
        "ap_freq": plot_sweepset_ap_freq,
        "wildness": plot_sweepset_wildness,
        "ap_freq_adapt": plot_sweepset_ap_freq_adapt,
        "ap_amp_slope": plot_sweepset_ap_amp_slope,
        # "fano_factor": plot_sweepset_fano_factor,
        # "ap_fano_factor": plot_sweepset_ap_fano_factor,
        # "cv": plot_sweepset_cv,
        # "ap_cv": plot_sweepset_ap_cv,
        "burstiness": plot_sweepset_burstiness,
        "num_bursts": plot_sweepset_num_bursts,
        # "isi_adapt": plot_sweepset_isi_adapt,
        # "isi_adapt_avg": plot_sweepset_isi_adapt_avg,
        # "ap_amp_adapt": plot_sweepset_ap_amp_adapt,
        # "ap_amp_adapt_avg": plot_sweepset_ap_amp_adapt_avg,
        "ap_latency": plot_sweepset_ap_latency,
        "ahp": plot_sweepset_ahp,
        "adp": plot_sweepset_adp,
        "ap_thresh": plot_sweepset_ap_thresh,
        "ap_amp": plot_sweepset_ap_amp,
        "ap_width": plot_sweepset_ap_width,
        "ap_peak": plot_sweepset_ap_peak,
        "ap_trough": plot_sweepset_ap_trough,
        "udr": plot_sweepset_ap_udr,
        "dfdi": plot_sweepset_dfdi,
        "rheobase": plot_sweepset_rheobase,
    }
    return sweepset_ft_plot_dict


def plot_sweepset_diagnostics(
    sweepset: EphysSweepSetFeatureExtractor,
) -> Tuple[Figure, Axes]:
    """Plot diagnostics overview for the whole sweepset.

    This function is useful to diagnose outliers on the sweepset level.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): sweepset to diagnose.

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
    onset, end = (
        sweepset.get_sweep_features()
        .applymap(strip_info)
        .iloc[0][["stim_onset", "stim_end"]]
    )
    t0, tfin = sweepset.sweeps()[0].t[[0, -1]]
    for ax in axes.values():
        ax.set_xlim(t0, tfin)

    axes["set_fts"].plot(sweepset.t.T, sweepset.v.T, color="grey", alpha=0.5)
    selection_dict = {
        "rebound": default_rebound_sweep_selector(sweepset),
        "ap": default_ap_sweep_selector(sweepset),
        "sag": default_sag_sweep_selector(sweepset),
        "fp": default_spiking_sweep_selector(sweepset),
        "tau": sweepset.get_sweepset_feature("tau")["selected_idx"],
        "wildness": sweepset.get_sweepset_feature("wildness")["selected_idx"],
    }
    selected_sweeps = {
        k: np.array(sweepset.sweeps())[v] for k, v in selection_dict.items()
    }
    for ft, idx in selection_dict.items():
        selected_sweep = np.array(sweepset.sweeps())[idx]
        if isinstance(selected_sweep, EphysSweepFeatureExtractor):
            axes["set_fts"].plot(
                selected_sweep.t, selected_sweep.v, label=f"{ft} @ idx={idx}"
            )
    axes["set_fts"].legend(title="feature sweeps")

    axes["fp_trace"].plot(
        selected_sweeps["fp"].t,
        selected_sweeps["fp"].v,
        color="k",
    )

    axes["ap_trace"].plot(
        selected_sweeps["ap"].t,
        selected_sweeps["ap"].v,
        color="k",
    )
    axes["ap_window"].plot(
        selected_sweeps["ap"].t,
        selected_sweeps["ap"].v,
        color="k",
    )
    ap_idx = selected_sweeps["ap"].sweep_feature("ap_peak")["selected_idx"]
    ap_start = selected_sweeps["ap"].spike_feature("threshold_t")[ap_idx] - 5e-3
    ap_end = selected_sweeps["ap"].spike_feature("fast_trough_t")[ap_idx] + 5e-3
    axes["ap_window"].set_xlim(ap_start, ap_end)
    for ft, plot_func in get_available_spike_diagnostics().items():
        plot_func(selected_sweeps["fp"], axes["fp_trace"])
        plot_func(selected_sweeps["ap"], axes["ap_trace"])
        plot_func(selected_sweeps["ap"], axes["ap_window"])
    axes["ap_trace"].axvline(ap_start, color="grey")
    axes["ap_trace"].axvline(ap_end, color="grey", label="selected ap")
    axes["ap_trace"].legend()

    sweep_is_hyperpol = [is_hyperpol(s) for s in sweepset.sweeps()]
    hyperpol_idcs = np.where(sweep_is_hyperpol)[0]
    axes["set_hyperpol_fts"].plot(
        sweepset.t[sweep_is_hyperpol].T, sweepset.v[sweep_is_hyperpol].T, color="k"
    )
    for selected_sweep in selected_sweeps["tau"]:
        plot_sweep_tau(selected_sweep, axes["set_hyperpol_fts"], color="r")
    plot_sweepset_v_baseline(sweepset, axes["set_hyperpol_fts"])
    axes["set_hyperpol_fts"].legend()

    h, l = axes["set_hyperpol_fts"].get_legend_handles_labels()
    d = {k: v for k, v in zip(l, h)}
    axes["set_hyperpol_fts"].legend(d.values(), d.keys())

    # sag
    axes["sag_fts"].plot(selected_sweeps["sag"].t, selected_sweeps["sag"].v, color="k")
    axes["sag_fts"].set_xlim(onset - 0.05, end + 0.05)
    plot_sweep_sag_area(selected_sweeps["sag"], axes["sag_fts"])
    plot_sweep_sag_time(selected_sweeps["sag"], axes["sag_fts"])
    axes["sag_fts"].legend()

    # rebound
    axes["rebound_fts"].plot(
        selected_sweeps["rebound"].t, selected_sweeps["rebound"].v, color="k"
    )
    axes["rebound_fts"].set_xlim(end - 0.05, None)
    plot_sweep_rebound(selected_sweeps["rebound"], axes["rebound_fts"])
    plot_sweep_rebound_latency(selected_sweeps["rebound"], axes["rebound_fts"])
    plot_sweep_rebound_area(selected_sweeps["rebound"], axes["rebound_fts"])
    plot_sweep_rebound_avg(selected_sweeps["rebound"], axes["rebound_fts"])
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

    plot_sweepset_rheobase(sweepset, axes["rheobase"])
    axes["rheobase"].set_xlabel("I (pA)")
    axes["rheobase"].set_ylabel("f (Hz)")
    axes["rheobase"].legend()

    plot_sweepset_r_input(sweepset, axes["r_input"])
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
