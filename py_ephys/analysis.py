import numpy as np
import matplotlib.pyplot as plt
from py_ephys.allen_sdk.ephys_extractor import EphysSweepFeatureExtractor
from typing import Tuple, Any
from matplotlib.pyplot import Figure, Axes
from py_ephys.utils import where_between, get_ap_ft_at_idx

############################
### spike level features ###
############################


def scatter_spike_ft(
    ax: Axes, sweep: EphysSweepFeatureExtractor, ft: str, **plot_kwargs
) -> Axes:
    spike_fts = sweep._spikes_df
    if spike_fts.size:
        ax.scatter(
            spike_fts[f"{ft}_t"],
            spike_fts[f"{ft}_v"],
            s=10,
            label=ft,
            **plot_kwargs,
        )
    return ax


def plot_spike_peaks(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    ax = scatter_spike_ft(ax, sweep, "peak", color=color, **plot_kwargs)
    return ax


def plot_spike_troughs(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    ax = scatter_spike_ft(ax, sweep, "trough", color=color, **plot_kwargs)
    return ax


def plot_spike_thresholds(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    ax = scatter_spike_ft(ax, sweep, "threshold", color=color, **plot_kwargs)
    return ax


def plot_spike_upstrokes(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    ax = scatter_spike_ft(ax, sweep, "upstroke", color=color, **plot_kwargs)


def plot_spike_downstrokes(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    ax = scatter_spike_ft(ax, sweep, "downstroke", color=color, **plot_kwargs)
    return ax


def plot_spike_widths(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    pass


def plot_spike_fast_troughs(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    ax = scatter_spike_ft(ax, sweep, "fast_trough", color=color, **plot_kwargs)
    return ax


def plot_spike_slow_troughs(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    ax = scatter_spike_ft(ax, sweep, "slow_trough", color=color, **plot_kwargs)
    return ax


def plot_spike_adps(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    spike_fts = sweep._spikes_df
    if spike_fts.size:
        ax.vlines(
            0.5 * (spike_fts[f"adp_t"] + spike_fts["fast_trough_t"]),
            spike_fts["adp_v"],
            spike_fts["fast_trough_v"],
            ls="--",
            lw=1,
            label="adp",
        )
    return ax


def plot_spike_ahps(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    spike_fts = sweep._spikes_df
    if spike_fts.size:
        ax.vlines(
            0.5 * (spike_fts[f"fast_trough_t"] + spike_fts["threshold_t"]),
            spike_fts["fast_trough_v"],
            spike_fts["threshold_v"],
            ls="--",
            lw=1,
            label="ahp",
        )
    return ax


def plot_spike_amps(
    sweep: EphysSweepFeatureExtractor, ax: Axes, color=None, **plot_kwargs
) -> Axes:
    pass


def get_spike_ft_plot_dict():
    spike_ft_plot_dict = {
        "peak": plot_spike_peaks,
        "trough": plot_spike_troughs,
        "threshold": plot_spike_thresholds,
        "upstroke": plot_spike_upstrokes,
        "downstroke": plot_spike_downstrokes,
        "width": plot_spike_widths,  # TODO: implement
        "fast_trough": plot_spike_fast_troughs,
        "slow_trough": plot_spike_slow_troughs,
        "adp": plot_spike_adps,
        "ahp": plot_spike_ahps,  # TODO: Check why nan
    }
    return spike_ft_plot_dict


def plot_spike_ft_diagnostics(
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
        for ft, plot_func in get_spike_ft_plot_dict().items():
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
    include_details=True,
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
    ft = sweep.sweep_feature("stim_amp")
    if not np.isnan(ft["value"]):
        ax.plot(
            ft["t_stim_amp"],
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
    include_details=True,
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
    ft = sweep.sweep_feature("stim_onset")
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
    include_details=True,
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
    ft = sweep.sweep_feature("stim_end")
    if not np.isnan(ft["value"]):
        ax.plot(
            ft["t_end"], ft["value"], "x", label="stim_end", color=color, **plot_kwargs
        )
    return ax


def plot_sweep_v_deflect(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=True,
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
    ft = sweep.sweep_feature("v_deflect")
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
    include_details=True,
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
    ft = sweep.sweep_feature("v_baseline")
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
    include_details=True,
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
    ft = sweep.sweep_feature("tau")
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
        ax.plot(t_fit + t_offset, y(t_fit), ls="--", color=color, label="tau fit")
    return ax


def plot_sweep_num_ap(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=True,
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
    ft = sweep.sweep_feature("num_ap")
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
    include_details=True,
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
    # feature is the same as num_ap
    return ax


def plot_sweep_ap_freq_adapt(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=True,
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
    ft = sweep.sweep_feature("ap_freq_adapt")
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
    include_details=True,
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
    print("r_input plotting is not yet implemented yet!")
    return ax


def plot_sweep_ap_amp_slope(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=True,
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
    ft = sweep.sweep_feature("ap_amp_slope")
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
    include_details=True,
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
    print("sag plotting is not yet implemented yet!")
    return ax


def plot_sweep_sag_fraction(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=True,
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
    print("sag_fraction plotting is not yet implemented yet!")
    return ax


def plot_sweep_sag_ratio(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=True,
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
    print("sag_ratio plotting is not yet implemented yet!")
    return ax


def plot_sweep_sag_area(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=True,
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
    ft = sweep.sweep_feature("sag_area")
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
    include_details=True,
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
    ft = sweep.sweep_feature("sag_time")
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
    include_details=True,
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
    ft = sweep.sweep_feature("v_plateau")
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
    include_details=True,
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
    ft = sweep.sweep_feature("rebound")
    if not np.isnan(ft["value"]):
        ax.plot(
            ft["t_rebound"],
            ft["v_rebound"],
            label="rebound interval",
            color=color,
            **plot_kwargs,
        )
        ax.vlines(
            sweep.t[ft["idx_rebound"]],
            ft["v_baseline"],
            sweep.v[ft["idx_rebound"]],
            label="rebound",
            color=color,
            **plot_kwargs,
        )
    return ax


def plot_sweep_rebound_aps(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=True,
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
    ft = sweep.sweep_feature("rebound_aps")
    print("rebound_aps plotting is not yet implemented yet!")
    if not np.isnan(ft["value"]):
        pass
    return ax


def get_sweep_rebound_latency(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=True,
    **plot_kwargs,
) -> Axes:
    # TODO: Check why stim end not quite aligned with v(t)!
    ft = sweep.sweep_feature("rebound_latency")
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
    include_details=True,
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
    ft = sweep.sweep_feature("rebound_area")
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
    include_details=True,
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
    ft = sweep.sweep_feature("rebound_avg")
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
    include_details=True,
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
    ft = sweep.sweep_feature("v_rest")
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
    include_details=True,
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
    print("num_bursts plotting is not yet implemented yet!")
    ft = sweep.sweep_feature("num_bursts")
    if not np.isnan(ft["value"]):
        pass
    return ax


def plot_sweep_burstiness(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=True,
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
    print("burstiness plotting is not yet implemented yet!")
    ft = sweep.sweep_feature("burstiness")
    if not np.isnan(ft["value"]):
        pass
    return ax


def plot_sweep_wildness(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=True,
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
    print("wildness plotting is not yet implemented yet!")
    ft = sweep.sweep_feature("wildness")
    if not np.isnan(ft["value"]):
        pass
    return ax


def plot_sweep_ahp(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=True,
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
    ft = sweep.sweep_feature("ahp")
    if not np.isnan(ft["value"]):
        trough_t = get_ap_ft_at_idx(sweep, "fast_trough_t", ft["ap_idx"])
        trough_v = get_ap_ft_at_idx(sweep, "fast_trough_v", ft["ap_idx"])
        thresh_t = get_ap_ft_at_idx(sweep, "threshold_t", ft["ap_idx"])
        thresh_v = get_ap_ft_at_idx(sweep, "threshold_v", ft["ap_idx"])
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
    include_details=True,
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
    ft = sweep.sweep_feature("ahp")
    if not np.isnan(ft["value"]):
        trough_t = get_ap_ft_at_idx(sweep, "fast_trough_t", ft["ap_idx"])
        trough_v = get_ap_ft_at_idx(sweep, "fast_trough_v", ft["ap_idx"])
        adp_t = get_ap_ft_at_idx(sweep, "adp_t", ft["ap_idx"])
        adp_v = get_ap_ft_at_idx(sweep, "adp_v", ft["ap_idx"])
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
    include_details=True,
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
    ft = sweep.sweep_feature("ap_thresh")
    if not np.isnan(ft["value"]):
        thresh_t = get_ap_ft_at_idx(sweep, "threshold_t", ft["ap_idx"])
        thresh_v = get_ap_ft_at_idx(sweep, "threshold_v", ft["ap_idx"])
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
    include_details=True,
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
    ft = sweep.sweep_feature("ap_amp")
    if not np.isnan(ft["value"]):
        thresh_t = get_ap_ft_at_idx(sweep, "threshold_t", ft["ap_idx"])
        thresh_v = get_ap_ft_at_idx(sweep, "threshold_v", ft["ap_idx"])
        peak_t = get_ap_ft_at_idx(sweep, "peak_t", ft["ap_idx"])
        peak_v = get_ap_ft_at_idx(sweep, "peak_v", ft["ap_idx"])
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
    include_details=True,
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
    print("ap_width plotting is not yet implemented yet!")
    ft = sweep.sweep_feature("ap_width")
    if not np.isnan(ft["value"]):
        pass
    return ax


def plot_sweep_ap_peak(
    sweep: EphysSweepFeatureExtractor,
    ax: Axes,
    color: Any = None,
    include_details=True,
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
    ft = sweep.sweep_feature("ap_peak")
    if not np.isnan(ft["value"]):
        peak_t = get_ap_ft_at_idx(sweep, "peak_t", ft["ap_idx"])
        peak_v = get_ap_ft_at_idx(sweep, "peak_v", ft["ap_idx"])
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
    include_details=True,
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
    ft = sweep.sweep_feature("ap_trough")
    if not np.isnan(ft["value"]):
        trough_t = get_ap_ft_at_idx(sweep, "fast_trough_t", ft["ap_idx"])
        trough_v = get_ap_ft_at_idx(sweep, "fast_trough_v", ft["ap_idx"])
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
    include_details=True,
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
    ft = sweep.sweep_feature("udr")
    if not np.isnan(ft["value"]):
        us_t = get_ap_ft_at_idx(sweep, "upstroke_t", ft["ap_idx"])
        us_v = get_ap_ft_at_idx(sweep, "upstroke_v", ft["ap_idx"])
        ds_t = get_ap_ft_at_idx(sweep, "downstroke_t", ft["ap_idx"])
        ds_v = get_ap_ft_at_idx(sweep, "downstroke_v", ft["ap_idx"])
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


def plot_sweep_ft_diagnostics(sweep, window=[0.4, 0.45]):
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
            for c, (ft, plot_func) in enumerate(get_sweep_ft_plot_dict().items()):
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
                "latency",
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


def get_sweep_ft_plot_dict():
    ft_plot_dict = {
        "stim_amp": plot_sweep_stim_amp,
        "stim_onset": plot_sweep_stim_onset,
        "stim_end": plot_sweep_stim_end,
        "v_baseline": plot_sweep_v_baseline,
        "v_deflect": plot_sweep_v_deflect,
        "tau": plot_sweep_tau,
        "num_ap": plot_sweep_num_ap,
        "ap_freq": plot_sweep_ap_freq,
        "ap_freq_adapt": plot_sweep_ap_freq_adapt,
        "ap_amp_slope": plot_sweep_ap_amp_slope,
        "r_input": plot_sweep_r_input,
        "sag": plot_sweep_sag,
        "sag_fraction": plot_sweep_sag_fraction,
        "sag_ratio": plot_sweep_sag_ratio,
        "sag_area": plot_sweep_sag_area,
        "sag_time": plot_sweep_sag_time,
        "v_plateau": plot_sweep_v_plateau,
        "rebound": plot_sweep_rebound,
        "rebound_aps": plot_sweep_rebound_aps,
        "rebound_latency": get_sweep_rebound_latency,
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


# def get_sweepset_ft_dict(return_ft_info=False):
#     _ft_dict = {
#         "tau": plot_sweepset_tau,
#         "r_input": plot_sweepset_r_input,
#         "v_rest": plot_sweepset_v_rest,
#         "v_baseline": plot_sweepset_v_baseline,
#         "slow_hyperpolarization": plot_sweepset_slow_hyperpolarization,
#         "sag": plot_sweepset_sag,
#         "sag_ratio": plot_sweepset_sag_ratio,
#         "sag_fraction": plot_sweepset_sag_fraction,
#         "sag_area": plot_sweepset_sag_area,
#         "sag_time": plot_sweepset_sag_time,
#         "rebound": plot_sweepset_rebound,
#         "rebound_aps": plot_sweepset_rebound_aps,
#         "rebound_area": plot_sweepset_rebound_area,
#         "rebound_latency": plot_sweepset_rebound_latency,
#         "rebound_avg": plot_sweepset_rebound_avg,
#         "num_ap": plot_sweepset_num_spikes,
#         "ap_freq": plot_sweepset_ap_freq,
#         "wildness": plot_sweepset_wildness,
#         "ap_freq_adapt": plot_sweepset_ap_freq_adapt,
#         "ap_amp_slope": plot_sweepset_ap_amp_slope,
#         "fano_factor": plot_sweepset_fano_factor,
#         "ap_fano_factor": plot_sweepset_ap_fano_factor,
#         "cv": plot_sweepset_cv,
#         "ap_cv": plot_sweepset_ap_cv,
#         "burstiness": plot_sweepset_burstiness,
#         "isi_adapt": plot_sweepset_isi_adapt,
#         "isi_adapt_avg": plot_sweepset_isi_adapt_avg,
#         "ap_amp_adapt": plot_sweepset_ap_amp_adapt,
#         "ap_amp_adapt_avg": plot_sweepset_ap_amp_adapt_avg,
#         "latency": plot_sweepset_latency,
#         "ahp": plot_sweepset_ahp,
#         "adp": plot_sweepset_adp,
#         "ap_thresh": plot_sweepset_ap_thresh,
#         "ap_amp": plot_sweepset_ap_amp,
#         "ap_width": plot_sweepset_ap_width,
#         "ap_peak": plot_sweepset_ap_peak,
#         "ap_trough": plot_sweepset_ap_trough,
#         "udr": plot_sweepset_ap_udr,
#         "dfdi": plot_sweepset_dfdi,
#         "rheobase": plot_sweepset_rheobase,
#     }
