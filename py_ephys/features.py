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

from functools import partial
from typing import Dict, Tuple

import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy import integrate
from scipy.optimize import curve_fit
from sklearn import linear_model

import py_ephys.allen_sdk.ephys_features as ft
from py_ephys.allen_sdk.ephys_extractor import EphysSweepFeatureExtractor
from py_ephys.utils import *

# ransac = linear_model.RANSACRegressor()
ransac = linear_model.LinearRegression()

############################
### spike level features ###
############################


def get_spike_peak_height(sweep: EphysSweepFeatureExtractor) -> float:
    """Extract spike level peak height feature.

    depends on: threshold_v, peak_v.
    description: v_peak - threshold_v.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        float: Spike peak height feature.
    """
    v_peak = sweep.spike_feature("peak_v", include_clipped=True)
    threshold_v = sweep.spike_feature("threshold_v", include_clipped=True)
    peak_height = v_peak - threshold_v
    return peak_height if len(v_peak) > 0 else np.array([])


def get_spike_ahp(sweep: EphysSweepFeatureExtractor) -> float:
    """Extract spike level after hyperpolarization feature.

    depends on: threshold_v, fast_trough_v.
    description: v_fast_trough - threshold_v.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        float: Spike after hyperpolarization feature.
    """
    v_fast_trough = sweep.spike_feature("fast_trough_v", include_clipped=True)
    threshold_v = sweep.spike_feature("threshold_v", include_clipped=True)
    return v_fast_trough - threshold_v


def get_spike_adp(sweep: EphysSweepFeatureExtractor) -> float:
    """Extract spike level after depolarization feature.

    depends on: adp_v, fast_trough_v.
    description: v_adp - v_fast_trough.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        float: Spike after depolarization feature.
    """
    v_adp = sweep.spike_feature("adp_v", include_clipped=True)
    v_fast_trough = sweep.spike_feature("fast_trough_v", include_clipped=True)
    return v_adp - v_fast_trough


def get_available_spike_features() -> Dict[str, callable]:
    """Dictionary of spike level features.

    Returns name of feature and function to calculate it.
    spike_ft_dict = {"ft_name": get_spike_ft, ...}

    Every feature function should take a sweep as input and return a float.

    Returns:
        Dict[str, callable]: Dictionary of spike level features.
    """
    return {
        "peak_height": get_spike_peak_height,
        "ahp": get_spike_ahp,
        "adp": get_spike_adp,
    }


############################
### sweep level features ###
############################


# implementation is horribly inefficient, but allows for a good overview
# and modification of how individual features are calculated
def get_sweep_burst_metrics(
    sweep: EphysSweepFeatureExtractor,
) -> Tuple[ndarray, ndarray, ndarray]:
    """Calculate burst metrics for a sweep.

    Uses EphysExtractor's _process_bursts() method to calculate burst metrics.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to calculate burst metrics for.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: returns burst index, burst start index,
            burst end index.
    """
    burst_metrics = sweep._process_bursts()
    if len(burst_metrics) == 0:
        return float("nan"), slice(0), slice(0)
    idx_burst, idx_burst_start, idx_burst_end = burst_metrics.T
    return idx_burst, idx_burst_start.astype(int), idx_burst_end.astype(int)


@ephys_feature
def get_sweep_num_ap(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level spike count feature.

    depends on: stim_onset, stim_end.
    description: # peaks during stimulus.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: Spike count feature and feature metadata
    """
    num_ap, num_ap_info = ephys_feature_init()
    stim_interval = where_stimulus(sweep)
    peak_i = sweep.spike_feature("peak_index", include_clipped=True)[stim_interval]
    peak_t = sweep.spike_feature("peak_t", include_clipped=True)[stim_interval]
    peak_v = sweep.spike_feature("peak_v", include_clipped=True)[stim_interval]

    num_spikes = len(peak_i)
    if len(peak_i) > 0:
        num_ap = num_spikes
        num_ap_info.update(
            {
                "peak_i": peak_i,
                "peak_t": peak_t,
                "peak_v": peak_v,
            }
        )
    return num_ap, num_ap_info


@ephys_feature
def get_sweep_ap_freq(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level spike rate feature.

    depends on: num_ap, stim_end, stim_onset.
    description: # peaks during stimulus / stimulus duration.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: Spike rate feature and feature metadata
    """
    ap_freq, ap_freq_info = ephys_feature_init()
    num_spikes = strip_info(sweep.sweep_feature("num_ap"))
    onset = strip_info(sweep.sweep_feature("stim_onset"))
    end = strip_info(sweep.sweep_feature("stim_end"))
    ap_freq = num_spikes / (end - onset)
    ap_freq_info.update(
        {"ap_freq": ap_freq, "num_ap": num_spikes, "onset": onset, "end": end}
    )
    return ap_freq, ap_freq_info


@ephys_feature
def get_sweep_stim_amp(
    sweep: EphysSweepFeatureExtractor,
) -> Tuple[float, Dict]:
    """Extract sweep level stimulus ampltiude feature.

    depends on: /.
    description: maximum amplitude of stimulus.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: Spike amplitude feature and feature metadata
    """
    # return efex._step_stim_amp(sweep) # Only works when start/end correspond to stimulus interval!
    stim_amp, stim_amp_info = ephys_feature_init()
    idx_stim_amp = np.argmax(abs(sweep.i))
    stim_amp = sweep.i[idx_stim_amp]
    t_stim_amp = sweep.t[idx_stim_amp]
    stim_amp_info.update(
        {
            "idx_amp": idx_stim_amp,
            "t_amp": t_stim_amp,
        }
    )
    return stim_amp, stim_amp_info


@ephys_feature
def get_sweep_ap_latency(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level ap_latency feature.

    depends on: stim_onset.
    description: time of first spike after stimulus onset.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: ap_latency feature and feature metadata
    """
    ap_latency, ap_latency_info = ephys_feature_init()
    if has_stimulus(sweep):
        onset = strip_info(sweep.sweep_feature("stim_onset"))
        spike_times = sweep.spike_feature("threshold_t", include_clipped=True)
        thresholds = sweep.spike_feature("threshold_v", include_clipped=True)
        spike_times_during_stim = spike_times[where_stimulus(sweep)]
        if len(spike_times_during_stim) > 0:
            v_first_spike = thresholds[where_stimulus(sweep)][0]
            t_first_spike = spike_times_during_stim[0]
            ap_latency = t_first_spike - onset
            ap_latency_info.update(
                {
                    "onset": onset,
                    "spike_times_during_stim": spike_times_during_stim,
                    "t_first_spike": t_first_spike,
                    "v_first_spike": v_first_spike,
                }
            )
    return ap_latency, ap_latency_info


@ephys_feature
def get_sweep_stim_onset(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level stimulus onset feature.

    depends on: /.
    description: time of stimulus onset.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: Stimulus onset feature and feature metadata
    """
    stim_onset, stim_onset_info = ephys_feature_init()
    if has_stimulus(sweep):
        idx_onset = np.where(sweep.i != 0)[0][0]
        t_onset = sweep.t[idx_onset]
        stim_onset = np.round(sweep.t[idx_onset], 1)
        stim_onset_info.update({"idx_onset": idx_onset, "t_onset": t_onset})
    return stim_onset, stim_onset_info


@ephys_feature
def get_sweep_stim_end(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level stimulus end feature.

    depends on: /.
    description: time of stimulus end.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: Stimulus end feature and feature metadata
    """
    stim_end, stim_end_info = ephys_feature_init()
    if has_stimulus(sweep):
        idx_end = np.where(sweep.i != 0)[0][-1]
        t_end = sweep.t[idx_end]
        stim_end = np.round(sweep.t[idx_end], 1)
        stim_end_info.update({"idx_end": idx_end, "t_end": t_end})
    return stim_end, stim_end_info


@ephys_feature
def get_sweep_v_deflect(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level voltage deflection feature.

    depends on: stim_end.
    description: average voltage during last 100 ms of stimulus.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: voltage deflection feature and feature metadata
    """
    v_deflect_avg, v_deflect_info = ephys_feature_init()
    if has_stimulus(sweep) and is_hyperpol(sweep):
        # v_deflect_avg = sweep.voltage_deflection()[0]
        end = strip_info(sweep.sweep_feature("stim_end"))
        v_deflect_avg = ft.average_voltage(sweep.v, sweep.t, start=end - 0.1, end=end)
        idx_deflect = np.where(where_between(sweep.t, end - 0.1, end))[0]
        t_deflect = sweep.t[idx_deflect]
        v_deflect = sweep.v[idx_deflect]
        v_deflect_info.update(
            {
                "idx_deflect": idx_deflect,
                "t_deflect": t_deflect,
                "v_deflect": v_deflect,
            }
        )
    return v_deflect_avg, v_deflect_info


@ephys_feature
def get_sweep_v_baseline(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level baseline voltage feature.

    depends on: stim_onset.
    description: average voltage in baseline_interval (in s) before stimulus onset.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: baseline voltage feature and feature metadata
    """
    v_baseline_avg, v_baseline_info = ephys_feature_init()
    onset = strip_info(sweep.sweep_feature("stim_onset"))
    where_baseline = where_between(sweep.t, onset - sweep.baseline_interval, onset)
    t_baseline = sweep.t[where_baseline]
    v_baseline = sweep.v[where_baseline]
    v_baseline_avg = np.mean(v_baseline)
    # v_baseline_avg = sweep._get_baseline_voltage() # bad since start is set to t[0]
    v_baseline_info.update(
        {
            "where_baseline": where_baseline,
            "t_baseline": t_baseline,
            "v_baseline": v_baseline,
            "baseline_interval": sweep.baseline_interval,
            "stim_onset": onset,
        }
    )
    return v_baseline_avg, v_baseline_info


@ephys_feature
def get_sweep_tau(
    sweep: EphysSweepFeatureExtractor,
) -> Tuple[float, Dict]:
    """Extract sweep level time constant feature.

    depends on: v_baseline, stim_onset.
    description: time constant of exponential fit to voltage deflection.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: time constant feature and feature metadata
    """
    tau, tau_info = ephys_feature_init()
    if is_hyperpol(sweep):
        """The following code block is copied and adapted from sweep.estimate_time_constant()."""
        v_peak, peak_index = sweep.voltage_deflection("min")
        v_baseline = strip_info(sweep.sweep_feature("v_baseline"))

        stim_onset = strip_info(sweep.sweep_feature("stim_onset"))
        onset_idx = ft.find_time_index(sweep.t, stim_onset)

        frac = 0.1
        search_result = np.flatnonzero(
            sweep.v[onset_idx:] <= frac * (v_peak - v_baseline) + v_baseline
        )
        if not search_result.size:
            raise ft.FeatureError("could not find interval for time constant estimate")

        fit_start = sweep.t[search_result[0] + onset_idx]
        fit_end = sweep.t[peak_index]

        if sweep.v[peak_index] < -200:
            print("A DOWNWARD PEAK WAS OBSERVED GOING TO LESS THAN 200 MV!!!")
            # Look for another local minimum closer to stimulus onset
            # We look for a couple of milliseconds after stimulus onset to 50 ms before the downward peak
            end_index = (onset_idx + 50) + np.argmin(
                sweep.v[onset_idx + 50 : peak_index - 1250]
            )
            fit_end = sweep.t[end_index]
            fit_start = sweep.t[onset_idx + 50]

        a, inv_tau, y0 = ft.fit_membrane_time_constant(
            sweep.v, sweep.t, fit_start, fit_end
        )

        tau = 1.0 / inv_tau * 1000
        tau_info.update(
            {
                "a": a,
                "inv_tau": inv_tau,
                "y0": y0,
                "fit_start": fit_start,
                "fit_end": fit_end,
                "equation": "y0 + a * exp(-inv_tau * x)",
            }
        )

    return tau, tau_info


@ephys_feature
def get_sweep_ap_freq_adapt(
    sweep: EphysSweepFeatureExtractor,
) -> Tuple[float, Dict]:
    """Extract sweep level spike frequency adaptation feature.

    depends on: stim_onset, stim_end, num_ap.
    description: ratio of spikes in second and first half half of stimulus interval, if there is at least 5 spikes in total.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: spike freq adaptation feature and feature metadata
    """
    ap_freq_adapt, ap_freq_adapt_info = ephys_feature_init()
    if strip_info(sweep.sweep_feature("num_ap")) > 5 and has_stimulus(sweep):
        onset = strip_info(sweep.sweep_feature("stim_onset"))
        end = strip_info(sweep.sweep_feature("stim_end"))
        t_half = (end - onset) / 2 + onset
        where_1st_half = where_between(sweep.t, onset, t_half)
        where_2nd_half = where_between(sweep.t, t_half, end)
        t_1st_half = sweep.t[where_1st_half]
        t_2nd_half = sweep.t[where_2nd_half]

        spike_times = sweep.spike_feature("peak_t", include_clipped=True)
        spike_times = spike_times[where_stimulus(sweep)]

        spikes_1st_half = spike_times[spike_times < t_half]
        spikes_2nd_half = spike_times[spike_times > t_half]
        num_spikes_1st_half = len(spikes_1st_half)
        num_spikes_2nd_half = len(spikes_2nd_half)
        ap_freq_adapt = num_spikes_2nd_half / num_spikes_1st_half

        ap_freq_adapt_info.update(
            {
                "num_spikes_1st_half": num_spikes_1st_half,
                "num_spikes_2nd_half": num_spikes_2nd_half,
                "where_1st_half": where_1st_half,
                "where_2nd_half": where_2nd_half,
                "t_1st_half": t_1st_half,
                "t_2nd_half": t_2nd_half,
            }
        )
    return ap_freq_adapt, ap_freq_adapt_info


@ephys_feature
def get_sweep_ap_amp_slope(
    sweep: EphysSweepFeatureExtractor,
) -> Tuple[float, Dict]:
    """Extract sweep level spike count feature.

    depends on: stim_onset, stim_end.
    description: spike amplitude adaptation as the slope of a linear fit v_peak(t_peak)
    during the stimulus interval.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: Spike count feature and feature metadata
    """
    ap_amp_slope, ap_amp_slope_info = ephys_feature_init()
    stim_interval = where_stimulus(sweep)
    peak_t = sweep.spike_feature("peak_t", include_clipped=True)[stim_interval]
    peak_v = sweep.spike_feature("peak_v", include_clipped=True)[stim_interval]

    if len(peak_v) > 5:
        y = lambda x, m, b: m * x + b
        (m, b), e = curve_fit(y, peak_t, peak_v)

        ap_amp_slope = m
        ap_amp_slope_info.update(
            {
                "peak_t": peak_t,
                "peak_v": peak_v,
                "slope": m,
                "intercept": b,
            }
        )
    return ap_amp_slope, ap_amp_slope_info


@ephys_feature
def get_sweep_r_input(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level input resistance feature.

    depends on: stim_amp, v_deflect, v_baseline.
    description: sweep level input resistance as (v_deflect - v_baseline / current).
    Should not be used for cell level feature.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: input resistance feature and feature metadata
    """
    r_input, r_input_info = ephys_feature_init()
    if is_hyperpol(sweep):
        stim_amp = strip_info(sweep.sweep_feature("stim_amp"))
        v_deflect = strip_info(sweep.sweep_feature("v_deflect"))
        v_baseline = strip_info(sweep.sweep_feature("v_baseline"))
        r_input = np.abs((v_deflect - v_baseline) * 1000 / stim_amp)

        r_input_info.update(
            {
                "v_baseline": v_baseline,
                "v_deflect": v_deflect,
                "stim_amp": stim_amp,
            }
        )
    return r_input, r_input_info


@ephys_feature
def get_sweep_sag(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level sag feature.

    depends on: /.
    description: magnitude of the depolarization peak.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: sag feature and feature metadata
    """
    sag, sag_info = ephys_feature_init()
    if is_hyperpol(sweep):
        with strip_sweep_ft_info(sweep) as fsweep:
            sag_fts = fsweep.estimate_sag()
        sag = sag_fts[0]
    return sag, sag_info


@ephys_feature
def get_sweep_sag_fraction(
    sweep: EphysSweepFeatureExtractor,
) -> Tuple[float, Dict]:
    """Extract sweep level sag fraction feature.

    depends on: /.
    description: fraction that membrane potential relaxes back to baseline.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: sag fraction feature and feature metadata
    """
    sag_fraction, sag_fraction_info = ephys_feature_init()
    if is_hyperpol(sweep):
        with strip_sweep_ft_info(sweep) as fsweep:
            sag_fts = fsweep.estimate_sag()
        sag_fraction = sag_fts[1]
    return sag_fraction, sag_fraction_info


@ephys_feature
def get_sweep_sag_ratio(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level sag ratio feature.

    depends on: /.
    description: ratio of steady state voltage decrease to the largest voltage decrease.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: sag ratio and feature metadata
    """
    sag_ratio, sag_ratio_info = ephys_feature_init()
    if is_hyperpol(sweep):
        with strip_sweep_ft_info(sweep) as fsweep:
            sag_fts = fsweep.estimate_sag()
        sag_ratio = sag_fts[2]
    return sag_ratio, sag_ratio_info


@ephys_feature
def get_sweep_sag_area(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level sag area feature.

    depends on: v_deflect, stim_onset, stim_end.
    description: area under the sag.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: sag area feature and feature metadata
    """
    sag_area, sag_area_info = ephys_feature_init()
    if is_hyperpol(sweep):
        where_sag = get_sweep_sag_idxs(sweep)
        if np.sum(where_sag) > 10:  # TODO: what should be min sag duration!?
            v_sag = sweep.v[where_sag]
            t_sag = sweep.t[where_sag]
            v_sagline = v_sag[0]
            # Take running average of v?
            if len(v_sag) > 10:  # at least 10 points to integrate
                sag_area = -integrate.cumulative_trapezoid(v_sag - v_sagline, t_sag)[-1]
                sag_area_info.update(
                    {
                        "where_sag": where_sag,
                        "v_sag": v_sag,
                        "t_sag": t_sag,
                        "v_sagline": v_sagline,
                    }
                )
    return sag_area, sag_area_info


@ephys_feature
def get_sweep_sag_time(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level sag duration feature.

    depends on: v_deflect, stim_onset, stim_end.
    description: duration of the sag.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: sag duration feature and feature metadata
    """
    sag_time, sag_time_info = ephys_feature_init()
    if is_hyperpol(sweep):
        where_sag = get_sweep_sag_idxs(sweep)
        if np.sum(where_sag) > 10:  # TODO: what should be min sag duration!?
            sag_t_start, sag_t_end = sweep.t[where_sag][[0, -1]]
            sag_time = sag_t_end - sag_t_start
            sag_time_info.update(
                {
                    "where_sag": where_sag,
                    "sag_t_start": sag_t_start,
                    "sag_t_end": sag_t_end,
                }
            )
    return sag_time, sag_time_info


@ephys_feature
def get_sweep_v_plateau(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level plataeu voltage feature.

    depends on: stim_end.
    description: average voltage during the plateau.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: plateau voltage feature and feature metadata
    """
    v_avg_plateau, v_plateau_info = ephys_feature_init()
    if is_hyperpol(sweep):
        end = strip_info(sweep.sweep_feature("stim_end"))
        # same as voltage deflection
        where_plateau = where_between(sweep.t, end - 0.1, end)
        v_plateau = sweep.v[where_plateau]
        t_plateau = sweep.t[where_plateau]
        v_avg_plateau = ft.average_voltage(v_plateau, t_plateau)
        v_plateau_info.update(
            {
                "where_plateau": where_plateau,
                "v_plateau": v_plateau,
                "t_plateau": t_plateau,
            }
        )
    return v_avg_plateau, v_plateau_info


@ephys_feature
def get_sweep_rebound(
    sweep: EphysSweepFeatureExtractor,
    T_rebound: float = 0.3,
) -> Tuple[float, Dict]:
    """Extract sweep level rebound feature.

    depends on: v_baseline, stim_end.
    description: V_max during stimulus_end and stimulus_end + T_rebound - V_baseline.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        T_rebound (float, optional): Time after stimulus end to look for rebound.

    Returns:
        Tuple[float, Dict]: rebound feature and feature metadata
    """
    rebound, rebound_info = ephys_feature_init({"T_rebound": T_rebound})
    if has_rebound(sweep, T_rebound):
        v_baseline = strip_info(sweep.sweep_feature("v_baseline"))
        end = strip_info(sweep.sweep_feature("stim_end"))
        where_rebound = where_between(sweep.t, end, end + T_rebound)
        where_rebound = np.logical_and(where_rebound, sweep.v > v_baseline)
        t_rebound = sweep.t[where_rebound]
        v_rebound = sweep.v[where_rebound]
        idx_rebound = np.argmax(sweep.v[where_rebound] - v_baseline)
        idx_rebound = np.where(where_rebound)[0][idx_rebound]
        max_rebound = sweep.v[idx_rebound]
        rebound = max_rebound - v_baseline
        rebound_info.update(
            {
                "idx_rebound": idx_rebound,
                "t_rebound": t_rebound,
                "v_rebound": v_rebound,
                "v_baseline": v_baseline,
                "max_rebound": max_rebound,
                "where_rebound": where_rebound,
            }
        )
    return rebound, rebound_info


@ephys_feature
def get_sweep_rebound_aps(
    sweep: EphysSweepFeatureExtractor,
    T_rebound: float = 0.3,
) -> Tuple[float, Dict]:
    """Extract sweep level number of rebounding spikes feature.

    depends on: stim_end.
    description: number of spikes during stimulus_end and stimulus_end + T_rebound.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        T_rebound (float, optional): Time after stimulus end to look for rebound.

    Returns:
        Tuple[float, Dict]: num rebound spikes feature and feature metadata
    """
    num_rebound_aps, rebound_spike_info = ephys_feature_init({"T_rebound": T_rebound})
    if has_rebound(sweep, T_rebound):
        t_spike = sweep.spike_feature("peak_t", include_clipped=True)
        idx_spike = sweep.spike_feature("peak_index", include_clipped=True)
        v_spike = sweep.spike_feature("peak_v", include_clipped=True)
        if len(t_spike) != 0:
            end = strip_info(sweep.sweep_feature("stim_end"))
            w_rebound = where_between(t_spike, end, end + T_rebound)
            idx_rebound = idx_spike[w_rebound]
            t_rebound = t_spike[w_rebound]
            v_rebound = v_spike[w_rebound]
            num_rebound_aps = np.sum(w_rebound)
            if num_rebound_aps > 0:
                rebound_spike_info.update(
                    {
                        "idx_rebound": idx_rebound,
                        "t_rebound": t_rebound,
                        "v_rebound": v_rebound,
                    }
                )
    return num_rebound_aps, rebound_spike_info


@ephys_feature
def get_sweep_rebound_latency(
    sweep: EphysSweepFeatureExtractor,
    T_rebound: float = 0.3,
) -> Tuple[float, Dict]:
    """Extract sweep level rebound latency feature.

    depends on: v_baseline, stim_end.
    description: duration from stimulus_end to when the voltage reaches above
    baseline for the first time. t_rebound = t_off + rebound_latency.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        T_rebound (float, optional): Time after stimulus end to look for rebound.

    Returns:
        Tuple[float, Dict]: rebound latency feature and feature metadata
    """
    rebound_latency, rebound_latency_info = ephys_feature_init({"T_rebound": T_rebound})
    if has_rebound(sweep, T_rebound):
        v_baseline = strip_info(sweep.sweep_feature("v_baseline"))
        end = strip_info(sweep.sweep_feature("stim_end"))
        where_rebound = where_between(sweep.t, end, end + T_rebound)
        where_rebound = np.logical_and(where_rebound, sweep.v > v_baseline)
        t_rebound = sweep.t[where_rebound]
        v_rebound = sweep.v[where_rebound]
        idx_rebound_reached = np.where(where_rebound)[0]
        t_rebound_reached = sweep.t[idx_rebound_reached][0]
        rebound_latency = t_rebound_reached - end
        rebound_latency_info.update(
            {
                "idx_rebound_reached": idx_rebound_reached,
                "t_rebound_reached": t_rebound_reached,
                "where_rebound": where_rebound,
                "t_rebound": t_rebound,
                "v_rebound": v_rebound,
                "v_baseline": v_baseline,
            }
        )
    return rebound_latency, rebound_latency_info


@ephys_feature
def get_sweep_rebound_area(
    sweep: EphysSweepFeatureExtractor,
    T_rebound: float = 0.3,
) -> Tuple[float, Dict]:
    """Extract sweep level rebound area feature.

    depends on: v_baseline, stim_end.
    description: area between rebound curve and baseline voltage from stimulus_end
    to stimulus_end + T_rebound.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        T_rebound (float, optional): Time after stimulus end to look for rebound.

    Returns:
        Tuple[float, Dict]: rebound area feature and feature metadata
    """
    rebound_area, rebound_area_info = ephys_feature_init({"T_rebound": T_rebound})
    if has_rebound(sweep, T_rebound):
        v_baseline = strip_info(sweep.sweep_feature("v_baseline"))
        end = strip_info(sweep.sweep_feature("stim_end"))
        where_rebound = where_between(sweep.t, end, end + T_rebound)
        where_rebound = np.logical_and(where_rebound, sweep.v > v_baseline)
        v_rebound = sweep.v[where_rebound]
        t_rebound = sweep.t[where_rebound]
        if len(v_rebound) > 10:  # at least 10 points to integrate
            rebound_area = integrate.cumulative_trapezoid(
                v_rebound - v_baseline, t_rebound
            )[-1]
            rebound_area_info.update(
                {
                    "where_rebound": where_rebound,
                    "t_rebound": t_rebound,
                    "v_rebound": v_rebound,
                    "v_baseline": v_baseline,
                }
            )
    return rebound_area, rebound_area_info


@ephys_feature
def get_sweep_rebound_avg(
    sweep: EphysSweepFeatureExtractor,
    T_rebound: float = 0.3,
) -> Tuple[float, Dict]:
    """Extract sweep level average rebound feature.

    depends on: v_baseline, stim_end.
    description: average voltage between stimulus_end
    and stimulus_end + T_rebound - baseline voltage.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        T_rebound (float, optional): Time after stimulus end to look for rebound.

    Returns:
        Tuple[float, Dict]: average rebound feature and feature metadata
    """
    v_rebound_avg, rebound_avg_info = ephys_feature_init({"T_rebound": T_rebound})
    if has_rebound(sweep, T_rebound):
        v_baseline = strip_info(sweep.sweep_feature("v_baseline"))
        end = strip_info(sweep.sweep_feature("stim_end"))
        where_rebound = where_between(sweep.t, end, end + T_rebound)
        where_rebound = np.logical_and(where_rebound, sweep.v > v_baseline)
        v_rebound = sweep.v[where_rebound]
        t_rebound = sweep.t[where_rebound]
        v_rebound_avg = ft.average_voltage(v_rebound, t_rebound) - v_baseline
        rebound_avg_info.update(
            {
                "where_rebound": where_rebound,
                "t_rebound": t_rebound,
                "v_rebound": v_rebound,
                "v_baseline": v_baseline,
            }
        )
    return v_rebound_avg, rebound_avg_info


@ephys_feature
def get_sweep_v_rest(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level resting potential feature.

    depends on: v_baseline, r_input, dc_offset.
    description: v_rest = v_baseline - r_input*dc_offset.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: resting potential feature and feature metadata
    """
    v_rest, v_rest_info = ephys_feature_init()
    v_baseline = strip_info(sweep.sweep_feature("v_baseline"))
    r_input = strip_info(sweep.sweep_feature("r_input"))
    try:
        dc_offset = strip_info(sweep.sweep_feature("dc_offset"))
        v_rest = v_baseline - r_input * 1e-3 * dc_offset
        v_rest_info.update(
            {
                "v_baseline": v_baseline,
                "r_input": r_input,
                "dc_offset": dc_offset,
            }
        )
    except KeyError:
        pass
    return v_rest, v_rest_info


@ephys_feature
def get_sweep_num_bursts(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level number of bursts feature.

    depends on: num_ap.
    description: Number of detected bursts.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: number of bursts feature and feature metadata
    """
    num_bursts, num_bursts_info = ephys_feature_init()
    if strip_info(sweep.sweep_feature("num_ap")) > 5 and has_stimulus(sweep):
        idx_burst, idx_burst_start, idx_burst_end = get_sweep_burst_metrics(sweep)
        peak_t = sweep.spike_feature("peak_t", include_clipped=True)
        if not np.isnan(idx_burst).any():
            t_burst_start = peak_t[idx_burst_start]
            t_burst_end = peak_t[idx_burst_end]
            num_bursts = len(idx_burst)
            num_bursts = float("nan") if num_bursts == 0 else num_bursts
            num_bursts_info.update(
                {
                    "idx_burst": idx_burst,
                    "idx_burst_start": idx_burst_start,
                    "idx_burst_end": idx_burst_end,
                    "t_burst_start": t_burst_start,
                    "t_burst_end": t_burst_end,
                }
            )
    return num_bursts, num_bursts_info


@ephys_feature
def get_sweep_burstiness(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level burstiness feature.

    depends on: num_ap.
    description: max "burstiness" index across detected bursts.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: burstiness feature and feature metadata
    """
    max_burstiness, burstiness_info = ephys_feature_init()
    if strip_info(sweep.sweep_feature("num_ap")) > 5 and has_stimulus(sweep):
        idx_burst, idx_burst_start, idx_burst_end = get_sweep_burst_metrics(sweep)
        peak_t = sweep.spike_feature("peak_t", include_clipped=True)
        if not np.isnan(idx_burst).any():
            t_burst_start = peak_t[idx_burst_start]
            t_burst_end = peak_t[idx_burst_end]
            num_bursts = len(idx_burst)
            max_burstiness = idx_burst.max() if num_bursts > 0 else float("nan")
            burstiness_info.update(
                {
                    "idx_burst": idx_burst,
                    "idx_burst_start": idx_burst_start,
                    "idx_burst_end": idx_burst_end,
                    "t_burst_start": t_burst_start,
                    "t_burst_end": t_burst_end,
                }
            )
    return max_burstiness, burstiness_info


@ephys_feature
def get_sweep_wildness(sweep: EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level wildness feature.

    depends on: /.
    description: Wildness is the number of spikes that occur outside of the stimulus interval.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: wildness feature and feature metadata
    """
    num_wild_spikes, wildness_info = ephys_feature_init()
    stim_interval = where_stimulus(sweep)
    i_wild_spikes = sweep.spike_feature("peak_index", include_clipped=True)[
        ~stim_interval
    ]
    t_wild_spikes = sweep.spike_feature("peak_t", include_clipped=True)[~stim_interval]
    v_wild_spikes = sweep.spike_feature("peak_v", include_clipped=True)[~stim_interval]
    if len(i_wild_spikes) > 0:
        num_wild_spikes = len(i_wild_spikes)
        wildness_info.update(
            {
                "i_wild_spikes": i_wild_spikes,
                "t_wild_spikes": t_wild_spikes,
                "v_wild_spikes": v_wild_spikes,
            }
        )
    return num_wild_spikes, wildness_info


def get_repr_ap_ft(
    sweep: EphysSweepFeatureExtractor,
    ft_name: str,
    ap_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level feature from representative sweep.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        ap_selector (Optional[Callable], optional): Function which selects a
            representative ap or set of aps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected aps. If none is provided, falls
            back to selecting all aps.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: Feature value and feature metadata.
    """
    ft = sweep.spike_feature(ft_name, include_clipped=True)

    if ap_selector is None:
        ap_selector = np.arange(len(ft))
    if ft_aggregator is None:
        ft_aggregator = np.nanmedian

    ft_agg, ft_info = ephys_feature_init()

    if len(ft) > 0:
        selected_idx = ap_selector(sweep)
        fts_selected = ft[selected_idx]

        if isinstance(fts_selected, (float, int, np.float64, np.int64)):
            ft_agg = fts_selected
        elif isinstance(fts_selected, ndarray):
            if len(fts_selected.flat) == 0:
                ft_agg = float("nan")
            else:
                ft_agg = ft_aggregator(ft)

        ft_info.update(
            {
                "selected_idx": selected_idx,
                "selected_fts": fts_selected,
                "selection": parse_ft_desc(ap_selector),
                "aggregation": parse_ft_desc(ft_aggregator),
            }
        )
    return ft_agg, ft_info


def default_ap_selector(sweep: EphysSweepFeatureExtractor) -> int:
    """Select representative AP from which the ap features are extracted.

    description: 2nd AP (if only 1 AP -> select first) during stimulus that has
    no NaNs in relevant spike features. If all APs have NaNs, return the AP during
    stimulus that has the least amount of NaNs in the relevant features. This
    avoids bad threshold detection at onset of stimulus.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep from which the ap features are extracted.

    Returns:
        int: Index of the sweep from which the rebound features are extracted.
    """
    relevant_ap_fts = [
        "ahp",
        "adp_v",
        "peak_v",
        "threshold_v",
        "trough_v",
        "width",
        "slow_trough_v",
        "fast_trough_v",
        "downstroke_v",
        "upstroke_v",
    ]

    spike_fts = sweep._spikes_df[relevant_ap_fts]
    peak_t = sweep.spike_feature("peak_t", include_clipped=True)
    onset = strip_info(sweep.sweep_feature("stim_onset"))
    end = strip_info(sweep.sweep_feature("stim_end"))
    is_stim = where_between(peak_t, onset, end)

    if len(peak_t[is_stim]) == 0:  # some sweeps have only wild aps
        return slice(0)

    has_nan_fts = spike_fts.isna().any(axis=1)
    if any(~has_nan_fts & is_stim):
        selected_ap_idxs = spike_fts.index[~has_nan_fts & is_stim]
    else:
        num_nan_fts = spike_fts[is_stim].isna().sum(axis=1)
        # sort by number of NaNs and then by index (ensure ap order stays intact)
        selected_ap_idxs = num_nan_fts.reset_index().sort_values([0, "index"])["index"]

    if len(selected_ap_idxs) > 1:
        return selected_ap_idxs[1]
    else:
        return selected_ap_idxs[0]


@ephys_feature
def get_sweep_ahp(
    sweep: EphysSweepFeatureExtractor,
    ap_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep level Afterhyperpolarization feature.

    depends on: /.
    description: Afterhyperpolarization (AHP) for representative AP. Difference
    between the fast trough and the threshold.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        ap_selector (Optional[Callable], optional): Function which selects a
            representative ap or set of aps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected aps. If none is provided, falls
            back to `default_ap_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: AHP feature and feature metadata
    """
    if ap_selector is None:
        ap_selector = default_ap_selector

    return get_repr_ap_ft(sweep, "ahp", ap_selector, ft_aggregator)


@ephys_feature
def get_sweep_adp(
    sweep: EphysSweepFeatureExtractor,
    ap_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep level Afterdepolarization feature.

    depends on: /.
    description: Afterdepolarization (ADP) for representative AP. Difference between the ADP and the fast trough.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        ap_selector (Optional[Callable], optional): Function which selects a
            representative ap or set of aps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected aps. If none is provided, falls
            back to `default_ap_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: ADP feature and feature metadata
    """
    if ap_selector is None:
        ap_selector = default_ap_selector

    return get_repr_ap_ft(sweep, "adp", ap_selector, ft_aggregator)


@ephys_feature
def get_sweep_ap_thresh(
    sweep: EphysSweepFeatureExtractor,
    ap_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep level AP threshold feature.

    depends on: /.
    description: AP threshold for representative AP.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        ap_selector (Optional[Callable], optional): Function which selects a
            representative ap or set of aps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected aps. If none is provided, falls
            back to `default_ap_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: AP threshold feature and feature metadata
    """
    if ap_selector is None:
        ap_selector = default_ap_selector

    return get_repr_ap_ft(sweep, "threshold_v", ap_selector, ft_aggregator)


@ephys_feature
def get_sweep_ap_amp(
    sweep: EphysSweepFeatureExtractor,
    ap_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep level AP amplitude feature.

    depends on: /.
    description: AP amplitude for representative AP.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        ap_selector (Optional[Callable], optional): Function which selects a
            representative ap or set of aps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected aps. If none is provided, falls
            back to `default_ap_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: AP amplitude feature and feature metadata
    """
    if ap_selector is None:
        ap_selector = default_ap_selector

    return get_repr_ap_ft(sweep, "peak_height", ap_selector, ft_aggregator)


@ephys_feature
def get_sweep_ap_width(
    sweep: EphysSweepFeatureExtractor,
    ap_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep level AP width feature.

    depends on: /.
    description: AP width for representative AP.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        ap_selector (Optional[Callable], optional): Function which selects a
            representative ap or set of aps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected aps. If none is provided, falls
            back to `default_ap_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: AP width feature and feature metadata
    """
    if ap_selector is None:
        ap_selector = default_ap_selector

    return get_repr_ap_ft(sweep, "width", ap_selector, ft_aggregator)


@ephys_feature
def get_sweep_ap_peak(
    sweep: EphysSweepFeatureExtractor,
    ap_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep level AP peak feature.

    depends on: /.
    description: AP peak for representative AP.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        ap_selector (Optional[Callable], optional): Function which selects a
            representative ap or set of aps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected aps. If none is provided, falls
            back to `default_ap_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: AP peak feature and feature metadata
    """
    if ap_selector is None:
        ap_selector = default_ap_selector

    return get_repr_ap_ft(sweep, "peak_v", ap_selector, ft_aggregator)


@ephys_feature
def get_sweep_ap_trough(
    sweep: EphysSweepFeatureExtractor,
    ap_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep level AP trough feature.

    depends on: /.
    description: AP trough for representative AP.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        ap_selector (Optional[Callable], optional): Function which selects a
            representative ap or set of aps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected aps. If none is provided, falls
            back to `default_ap_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: AP trough feature and feature metadata
    """
    if ap_selector is None:
        ap_selector = default_ap_selector

    return get_repr_ap_ft(sweep, "trough_v", ap_selector, ft_aggregator)


@ephys_feature
def get_sweep_udr(
    sweep: EphysSweepFeatureExtractor,
    ap_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep level Upstroke-to-downstroke ratio feature.

    depends on: /.
    description: Upstroke-to-downstroke ratio for representative AP.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to extract feature from.
        ap_selector (Optional[Callable], optional): Function which selects a
            representative ap or set of aps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected aps. If none is provided, falls
            back to `default_ap_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: UDR feature and feature metadata
    """
    if ap_selector is None:
        ap_selector = default_ap_selector

    return get_repr_ap_ft(
        sweep, "upstroke_downstroke_ratio", ap_selector, ft_aggregator
    )


### Feature extraction functions
def get_available_sweep_features(return_ft_info=False):
    _ft_dict = {
        "stim_amp": get_sweep_stim_amp,  # None
        "stim_onset": get_sweep_stim_onset,  # None
        "stim_end": get_sweep_stim_end,  # None
        "ap_latency": get_sweep_ap_latency,  # None
        "v_baseline": get_sweep_v_baseline,  # stim_onset (needs to be computed early)
        "v_deflect": get_sweep_v_deflect,  # stim_end
        "tau": get_sweep_tau,  # v_baseline
        "num_ap": get_sweep_num_ap,  # spike_features
        "ap_freq": get_sweep_ap_freq,  # num_ap, stim_onset, stim_end
        "ap_freq_adapt": get_sweep_ap_freq_adapt,  # num_ap, stim_onset, stim_end, spike_features
        "ap_amp_slope": get_sweep_ap_amp_slope,  # spike_features
        "r_input": get_sweep_r_input,  # stim_onset, stim_end, stim_amp, v_baseline, v_deflect
        "sag": get_sweep_sag,  # v_baseline
        "sag_fraction": get_sweep_sag_fraction,  # v_baseline
        "sag_ratio": get_sweep_sag_ratio,  # v_baseline
        "sag_area": get_sweep_sag_area,  # stim_onset, stim_end, v_deflect, v_baseline
        "sag_time": get_sweep_sag_time,  # stim_onset, stim_end, v_deflect, v_baseline
        "v_plateau": get_sweep_v_plateau,  # stim_end
        "rebound": get_sweep_rebound,  # stim_end, v_baseline
        "rebound_aps": get_sweep_rebound_aps,  # stim_end, spike_features
        "rebound_area": get_sweep_rebound_area,  # stim_end, v_baseline
        "rebound_latency": get_sweep_rebound_latency,  # stim_end, v_baseline
        "rebound_avg": get_sweep_rebound_avg,  # stim_end, v_baseline
        "v_rest": get_sweep_v_rest,  # r_input, v_baseline
        "num_bursts": get_sweep_num_bursts,  # num_ap
        "burstiness": get_sweep_burstiness,  # num_ap
        "wildness": get_sweep_wildness,  # stim_onset, stim_end, spike_features
        "ahp": get_sweep_ahp,  # spike_features
        "adp": get_sweep_adp,  # spike_features
        "ap_thresh": get_sweep_ap_thresh,  # spike_features
        "ap_amp": get_sweep_ap_amp,  # spike_features
        "ap_width": get_sweep_ap_width,  # spike_features
        "ap_peak": get_sweep_ap_peak,  # spike_features
        "ap_trough": get_sweep_ap_trough,  # spike_features
        "udr": get_sweep_udr,  # spike_features
    }

    if return_ft_info:
        return {k: partial(v, return_ft_info=True) for k, v in _ft_dict.items()}
    return _ft_dict


################################
### sweep set level features ###
################################


def default_median_aggregator(fts):
    """description: median."""
    return np.nanmedian(fts)


def get_repr_sweep_ft(
    sweepset: EphysSweepSetFeatureExtractor,
    ft_name: str,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level feature from representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        ft (str): Feature to extract.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to selecting all sweeps.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: Feature value and feature metadata.
    """
    if sweep_selector is None:
        sweep_selector = lambda x: np.arange(len(sweepset.sweeps()))
    if ft_aggregator is None:
        ft_aggregator = np.nanmedian

    ft_agg, ft_info = ephys_feature_init()

    ft = get_stripped_sweep_fts(sweepset)[ft_name].to_numpy()
    selected_idx = sweep_selector(sweepset)
    fts_selected = ft[selected_idx]
    if isinstance(fts_selected, (float, int, np.float64, np.int64)):
        ft_agg = fts_selected
    elif isinstance(fts_selected, ndarray):
        if len(fts_selected.flat) == 0:
            ft_agg = float("nan")
        else:
            ft_agg = ft_aggregator(ft)

    ft_info.update(
        {
            "selected_idx": selected_idx,
            "selected_fts": fts_selected,
            "selection": parse_ft_desc(sweep_selector),
            "aggregation": parse_ft_desc(ft_aggregator),
        }
    )
    return ft_agg, ft_info


@ephys_feature
def get_sweepset_tau(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level time constant feature.

    depends on: /.
    description: median of the membrane time constants from all hyperpolarizing traces.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_time_constant_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level sag_time feature.
    """

    def default_time_constant_selector(sweepset):
        """description: all hyperpolarizing traces."""
        return np.where(get_stripped_sweep_fts(sweepset)["stim_amp"] < 0)[0]

    if sweep_selector is None:
        sweep_selector = default_time_constant_selector

    return get_repr_sweep_ft(sweepset, "tau", sweep_selector, ft_aggregator)


# input resistance
@ephys_feature
def get_sweepset_r_input(sweepset: EphysSweepSetFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep set level input resistance feature.

    depends on: v_deflect, stim_amp.
    description: fitted slope of v_deflect(v_input) in MOhms.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.

    Returns:
        Tuple[float, Dict]: sweep set level input resistance feature.
    """
    r_input, r_input_info = ephys_feature_init()
    is_hyperpol = get_stripped_sweep_fts(sweepset)["stim_amp"] < 0
    v_deflect = (
        sweepset.get_sweep_features()
        .applymap(strip_info)["v_deflect"]
        .loc[is_hyperpol]
        .to_numpy()
        .reshape(-1, 1)
    )
    i_amp = (
        sweepset.get_sweep_features()
        .applymap(strip_info)["stim_amp"]
        .loc[is_hyperpol]
        .to_numpy()
        .reshape(-1, 1)
    )
    if len(v_deflect) >= 3:
        ransac.fit(i_amp, v_deflect)
        r_input = ransac.coef_[0, 0] * 1000
        v_intercept = ransac.intercept_[0]
        r_input_info.update(
            {
                "raw_slope": r_input / 1000,
                "v_intercept": v_intercept,
                "i_amp": i_amp,
                "v_deflect": v_deflect,
            }
        )
    return r_input, r_input_info


# baseline membrane potential
@ephys_feature
def get_sweepset_v_baseline(
    sweepset: EphysSweepSetFeatureExtractor,
) -> Tuple[float, Dict]:
    """Extract sweep set level baseline potential feature.

    depends on: stim_amp, v_baseline.
    description: median of the baseline potentials from all hyperpolarizing
    traces.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.

    Returns:
        Tuple[float, Dict]: sweep set level resting potential feature.
    """
    v_baseline, v_baseline_info = ephys_feature_init()
    is_hyperpol = get_stripped_sweep_fts(sweepset)["stim_amp"] < 0
    v_baseline = get_stripped_sweep_fts(sweepset)["v_baseline"][is_hyperpol]
    selected_idx = median_idx(v_baseline)
    v_baseline = v_baseline.median(skipna=True)
    v_baseline_info.update(
        {
            "v_baseline": v_baseline,
            "selected_idx": selected_idx,
        }
    )
    return v_baseline, v_baseline_info


# resting potential
@ephys_feature
def get_sweepset_v_rest(sweepset: EphysSweepSetFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep set level resting potential feature.

    depends on: dc_offset, r_input, v_baseline, stim_amp.
    description: median of the resting potentials from all hyperpolarizing
    traces. v_rest = v_baseline - r_input * dc

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.

    Returns:
        Tuple[float, Dict]: sweep set level resting potential feature.
    """
    dc_offset = get_stripped_sweep_fts(sweepset)["dc_offset"]
    v_rest, v_rest_info = ephys_feature_init()
    is_hyperpol = get_stripped_sweep_fts(sweepset)["stim_amp"] < 0
    r_input = get_stripped_sweep_fts(sweepset)["r_input"][is_hyperpol]
    v_base = get_stripped_sweep_fts(sweepset)["v_baseline"][is_hyperpol]
    v_rests = v_base - r_input * 1e-3 * dc_offset
    selected_idx = median_idx(v_rests)
    v_rest = v_rests.median(skipna=True)
    v_rest_info.update(
        {
            "r_input": r_input,
            "v_rest": v_rests,
            "selected_idx": selected_idx,
            "v_base": v_base,
            "dc_offset": dc_offset,
        }
    )
    return v_rest, v_rest_info


# slow hyperpolarizing potential
@ephys_feature
def get_sweepset_slow_hyperpolarization(
    sweepset: EphysSweepSetFeatureExtractor,
) -> Tuple[float, Dict]:
    """Extract sweep set level slow hyperpolarization feature.

    depends on: v_baseline, stim_amp.
    description: difference between the max and min baseline voltages.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.

    Returns:
        Tuple[float, Dict]: sweep set level slow hyperpolarization feature.
    """
    slow_hyperpol, slow_hyperpol_info = ephys_feature_init()
    is_hyperpol = (
        get_stripped_sweep_fts(sweepset)["stim_amp"] < 0
    )  # TODO: ASK IF THIS IS ONLY TAKEN FOR HYPERPOLARIZING TRACES (I THINK NOT)
    v_baseline = get_stripped_sweep_fts(sweepset)["v_baseline"][is_hyperpol]
    slow_hyperpol = (
        v_baseline.max() - v_baseline.min()
    )  # like v_rest.min() - v_rest.max(), since v_rest = v_base - const
    slow_hyperpol_info.update({"v_baseline": v_baseline})
    return slow_hyperpol, slow_hyperpol_info


# sag features (steepest hyperpolarizing trace)
def default_sag_sweep_selector(sweepset: EphysSweepSetFeatureExtractor) -> int:
    """Select representative sweep from which the sag features are extracted.

    description: Lowest hyperpolarization sweep that is not NaN. If 3 lowest
    sweeps are NaN, then the first sweep is selected, meaning the feature is set
    to NaN.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.

    Returns:
        int: Index of the sweep from which the rebound features are extracted.
    """
    # TODO: Consult if this is sensible!
    sag = get_stripped_sweep_fts(sweepset)["sag"]
    nan_sags = sag.isna()
    if all(nan_sags[:3]):
        selected_sweep_idx = 0
    else:
        selected_sweep_idx = sag.index[~nan_sags][0]
    return selected_sweep_idx


@ephys_feature
def get_sweepset_sag(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level sag feature.

    depends on: sag.
    description: sag voltage for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_sag_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level sag feature.
    """
    if sweep_selector is None:
        sweep_selector = default_sag_sweep_selector

    return get_repr_sweep_ft(sweepset, "sag", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_sag_ratio(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level sag ratio feature.

    depends on: sag_ratio
    description: sag ratio for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_sag_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level sag_ratio feature.
    """
    if sweep_selector is None:
        sweep_selector = default_sag_sweep_selector

    return get_repr_sweep_ft(sweepset, "sag_ratio", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_sag_fraction(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level sag fraction feature.

    depends on: sag_fraction.
    description: sag fraction for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_sag_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level sag_fraction feature.
    """
    if sweep_selector is None:
        sweep_selector = default_sag_sweep_selector

    return get_repr_sweep_ft(sweepset, "sag_fraction", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_sag_area(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level sag area feature.

    depends on: sag_area.
    description: sag area for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_sag_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level sag_time feature.
    """
    if sweep_selector is None:
        sweep_selector = default_sag_sweep_selector

    return get_repr_sweep_ft(sweepset, "sag_area", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_sag_time(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level sag time feature.

    depends on: sag_time.
    description: sag time for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_sag_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level sag_time feature.
    """
    if sweep_selector is None:
        sweep_selector = default_sag_sweep_selector

    return get_repr_sweep_ft(sweepset, "sag_time", sweep_selector, ft_aggregator)


# rebound features (steepest hyperpolarizing trace)
def default_rebound_sweep_selector(sweepset: EphysSweepSetFeatureExtractor) -> int:
    """Select representative sweep from which the rebound features are extracted.

    description: Lowest hyperpolarization sweep. If 3 lowest sweeps are NaN,
    then the first sweep is selected, meaning the feature is set to NaN.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.

    Returns:
        int: Index of the sweep from which the rebound features are extracted.
    """
    # TODO: Consult if this is sensible!
    rebound = get_stripped_sweep_fts(sweepset)["rebound"]
    nan_rebound = rebound.isna()
    if all(nan_rebound[:3]):
        selected_sweep_idx = 0
    else:
        selected_sweep_idx = rebound.index[~nan_rebound][0]
    return selected_sweep_idx


@ephys_feature
def get_sweepset_rebound(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level rebound feature.

    depends on: rebound.
    description: rebound voltage for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_rebound_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level rebound_latency feature.
    """
    if sweep_selector is None:
        sweep_selector = default_rebound_sweep_selector

    return get_repr_sweep_ft(sweepset, "rebound", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_rebound_aps(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level rebound ratio feature.

    depends on: rebound_aps.
    description: rebound ratio for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_rebound_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level rebound aps feature.
    """
    if sweep_selector is None:
        sweep_selector = default_rebound_sweep_selector

    return get_repr_sweep_ft(sweepset, "rebound_latency", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_rebound_area(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level rebound area feature.

    depends on: rebound_area.
    description: rebound area for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_rebound_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level rebound_area feature.
    """
    if sweep_selector is None:
        sweep_selector = default_rebound_sweep_selector

    return get_repr_sweep_ft(sweepset, "rebound_area", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_rebound_latency(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level rebound latency feature.

    depends on: rebound_latency.
    description: rebound latency for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_rebound_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level rebound_latency feature.
    """
    if sweep_selector is None:
        sweep_selector = default_rebound_sweep_selector

    return get_repr_sweep_ft(sweepset, "rebound_latency", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_rebound_avg(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level average rebound feature.

    depends on: rebound_avg.
    description: average rebound for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_rebound_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level rebound_avg feature.
    """
    if sweep_selector is None:
        sweep_selector = default_rebound_sweep_selector

    return get_repr_sweep_ft(sweepset, "rebound_avg", sweep_selector, ft_aggregator)


# num spikes
def default_spiking_sweep_selector(sweepset: EphysSweepSetFeatureExtractor) -> int:
    """Select representative sweep from which the spiking related features are extracted.

    depends on: num_ap, wildness.
    description: Highest non wild trace (wildness == cell dying).

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.

    Returns:
        int: Index of the sweep from which the rebound features are extracted.
    """
    # TODO: should I implement a better selection protocol / criterion!?
    num_spikes = get_stripped_sweep_fts(sweepset)["num_ap"]
    wildness = get_stripped_sweep_fts(sweepset)["wildness"]
    return num_spikes[
        wildness.isna()
    ].argmax()  # select highest non wild trace (wildness == cell dying)


@ephys_feature
def get_sweepset_num_ap(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level spike count feature.

    depends on: num_ap.
    description: number of spikes for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_spiking_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level spike count feature.
    """
    if sweep_selector is None:
        sweep_selector = default_spiking_sweep_selector

    return get_repr_sweep_ft(sweepset, "num_ap", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_ap_freq(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level spike rate feature.

    depends on: ap_freq.
    description: spike rate for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_spiking_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level spike rate feature.
    """
    if sweep_selector is None:
        sweep_selector = default_spiking_sweep_selector

    return get_repr_sweep_ft(sweepset, "ap_freq", sweep_selector, ft_aggregator)


# wildness
@ephys_feature
def get_sweepset_wildness(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level wildness feature.

    depends on: wildness.
    description: wildness for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `select_wildest_idx`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level wildness feature.
    """

    def select_wildest_idx(sset):
        """Select sweep with highest wildness.

        description: sweep with most wild APs."""
        idx = sset.get_sweep_feature("wildness").apply(strip_info).argmax()
        return int(idx) if idx != -1 else slice(0)

    if sweep_selector is None:
        sweep_selector = select_wildest_idx

    return get_repr_sweep_ft(sweepset, "wildness", sweep_selector, ft_aggregator)


# spike frequency adaptation
@ephys_feature
def get_sweepset_ap_freq_adapt(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level spike frequency adaptation feature.

    depends on: ap_freq_adapt.
    description: spike frequency adaptation for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_spiking_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level AP frequency adaption feature.
    """
    if sweep_selector is None:
        sweep_selector = default_spiking_sweep_selector

    return get_repr_sweep_ft(sweepset, "ap_freq_adapt", sweep_selector, ft_aggregator)


# spike amplitude adaptation
@ephys_feature
def get_sweepset_ap_amp_slope(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level spike amplitude adaptation feature.

    depends on: ap_amp_slope.
    description: spike amplitude adaptation for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_spiking_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level AP amplitude slope feature.
    """
    if sweep_selector is None:
        sweep_selector = default_spiking_sweep_selector

    return get_repr_sweep_ft(sweepset, "ap_amp_slope", sweep_selector, ft_aggregator)


# AP Fano factor
@ephys_feature
def get_sweepset_fano_factor(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level fano factor feature.

    depends on: fano_factor.
    description: Fano factor for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_spiking_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level fano factor feature.
    """
    if sweep_selector is None:
        sweep_selector = default_spiking_sweep_selector

    return get_repr_sweep_ft(sweepset, "fano_factor", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_ap_fano_factor(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level ap fano factor feature.

    depends on: AP_fano_factor.
    description: AP Fano factor for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_spiking_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level AP fano factor feature.
    """
    if sweep_selector is None:
        sweep_selector = default_spiking_sweep_selector

    return get_repr_sweep_ft(sweepset, "AP_fano_factor", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_cv(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level coeffficent of variation feature.

    depends on: cv.
    description: CV for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_spiking_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level coefficient of variation feature.
    """
    if sweep_selector is None:
        sweep_selector = default_spiking_sweep_selector

    return get_repr_sweep_ft(sweepset, "cv", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_ap_cv(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level AP coefficient of variation feature.

    depends on: AP_cv.
    description: AP CV for a representative sweep.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_spiking_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level AP coefficient of variation feature.
    """
    if sweep_selector is None:
        sweep_selector = default_spiking_sweep_selector

    return get_repr_sweep_ft(sweepset, "AP_cv", sweep_selector, ft_aggregator)


# burstiness
@ephys_feature
def get_sweepset_burstiness(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level burstiness feature.

    depends on: burstiness.
    description: median burstiness for the first 5 "bursty" traces.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `default_median_aggregator`.

    Returns:
        Tuple[float, Dict]: sweep set level burstiness feature.
    """

    def default_selector(sweepset):
        """description: the first 5 non-nan traces."""
        fts = get_stripped_sweep_fts(sweepset)["burstiness"]
        fts[fts < 0] = float("nan")  # don't consider negative burstiness
        idxs = fts[~fts.isna()].iloc[:5].index
        if not idxs.empty:
            return idxs
        else:
            slice(0)

    if sweep_selector is None:
        sweep_selector = default_selector
    if ft_aggregator is None:
        ft_aggregator = default_median_aggregator

    return get_repr_sweep_ft(sweepset, "burstiness", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_num_bursts(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level burstiness feature.

    depends on: num_bursts.
    description: max num_bursts for the first 5 "bursty" traces.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `nanargmax`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level num_bursts feature.
    """

    def default_selector(sweepset):
        """description: the first 5 non-nan traces."""
        fts = get_stripped_sweep_fts(sweepset)["num_bursts"]
        idxs = fts[~fts.isna()].iloc[:5].index
        if not idxs.empty:
            return idxs.argmax()
        else:
            slice(0)

    if sweep_selector is None:
        sweep_selector = default_selector

    return get_repr_sweep_ft(sweepset, "num_bursts", sweep_selector, ft_aggregator)


# adaptation index
@ephys_feature
def get_sweepset_isi_adapt(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level Inter spike interval adaptation feature.

    depends on: isi_adapt.
    description: median of the ISI adaptation of the first 5 traces
    that show adaptation.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `default_median_aggregator`.

    Returns:
        Tuple[float, Dict]: sweep set level isi adaptation feature.
    """

    def default_selector(sweepset):
        """description: the first 5 non-nan traces."""
        fts = get_stripped_sweep_fts(sweepset)["isi_adapt"]
        return fts[~fts.isna()].iloc[:5].index

    if sweep_selector is None:
        sweep_selector = default_selector
    if ft_aggregator is None:
        ft_aggregator = default_median_aggregator

    return get_repr_sweep_ft(sweepset, "isi_adapt", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_isi_adapt_avg(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level average inter spike interval adaptation feature.

    depends on: isi_adapt_average.
    description: median of the ISI adaptation average of the first 5 traces
    that show adaptation.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `default_median_aggregator`.

    Returns:
        Tuple[float, Dict]: sweep set level average isi adaptation feature.
    """

    def default_selector(sweepset):
        """description: the first 5 non-nan traces."""
        fts = get_stripped_sweep_fts(sweepset)["isi_adapt_average"]
        return fts[~fts.isna()].iloc[:5].index

    if sweep_selector is None:
        sweep_selector = default_selector
    if ft_aggregator is None:
        ft_aggregator = default_median_aggregator

    return get_repr_sweep_ft(
        sweepset, "isi_adapt_average", sweep_selector, ft_aggregator
    )


@ephys_feature
def get_sweepset_ap_amp_adapt(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level AP amplitude adaptation feature.

    depends on: AP_amp_adapt.
    description: median of the AP amplitude adaptation of the first 5 traces
    that show adaptation.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `default_median_aggregator`.

    Returns:
        Tuple[float, Dict]: sweep set level average AP amplitude adaptation feature.
    """

    def default_selector(sweepset):
        """description: the first 5 non-nan traces."""
        fts = get_stripped_sweep_fts(sweepset)["AP_amp_adapt"]
        return fts[~fts.isna()].iloc[:5].index

    if sweep_selector is None:
        sweep_selector = default_selector
    if ft_aggregator is None:
        ft_aggregator = default_median_aggregator

    return get_repr_sweep_ft(sweepset, "AP_amp_adapt", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_ap_amp_adapt_avg(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level average AP amplitude adaptation feature.

    depends on: AP_amp_adapt_average.
    description: median of the AP amplitude adaptation average of the first 5
    traces that show adaptation.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `default_median_aggregator`.

    Returns:
        Tuple[float, Dict]: sweep set level average AP amplitude adaptation average feature.
    """

    def default_selector(sweepset):
        """description: the first 5 non-nan traces."""
        fts = get_stripped_sweep_fts(sweepset)["AP_amp_adapt_average"]
        return fts[~fts.isna()].iloc[:5].index

    if sweep_selector is None:
        sweep_selector = default_selector
    if ft_aggregator is None:
        ft_aggregator = default_median_aggregator

    return get_repr_sweep_ft(
        sweepset, "AP_amp_adapt_average", sweep_selector, ft_aggregator
    )


def default_ap_sweep_selector(sweepset: EphysSweepSetFeatureExtractor) -> int:
    """Select representative ap in a sweep from which the AP features are used.

    depends on: stim_amp, num_ap.
    description: First depolarization trace that contains spikes and which has
        the least amount of NaNs in the AP features.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.

    Returns:
        int: Index of the sweep from which the rebound features are extracted.
    """
    # TODO: Consult if this is sensible!
    relevant_ap_fts = [
        "ap_thresh",
        "ap_amp",
        "ap_width",
        "ap_peak",
        "ap_trough",
        "ahp",
        "adp",
        "udr",
    ]

    stripped_fts = get_stripped_sweep_fts(sweepset)
    is_depol = stripped_fts["stim_amp"] > 0
    has_spikes = stripped_fts["num_ap"] > 0
    num_nans = stripped_fts[relevant_ap_fts].isna().sum(axis=1)
    return num_nans[is_depol & has_spikes].idxmin()


# latency
@ephys_feature
def get_sweepset_ap_latency(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level ap_latency feature.

    depends on: stim_amp, ap_latency.
    description: ap_latency of the first depolarization trace that contains spikes in ms.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_ap_latency_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level AP latency feature.
    """

    def default_ap_latency_sweep_selector(sweepset):
        """
        depends on: stim_amp, ap_latency.
        description: first depolarization trace that has non-nan ap_latency.
        """
        is_depol = get_stripped_sweep_fts(sweepset)["stim_amp"] > 0
        ap_latency = get_stripped_sweep_fts(sweepset)["ap_latency"]
        return is_depol.index[is_depol & ~ap_latency.isna()][0]

    if sweep_selector is None:
        sweep_selector = default_ap_latency_sweep_selector

    return get_repr_sweep_ft(sweepset, "ap_latency", sweep_selector, ft_aggregator)


# ahp
@ephys_feature
def get_sweepset_ahp(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level Afterhyperpolarization feature.

    depends on: ahp.
    description: AHP (fast_trough_v - threshold_v) of a representative spike of
    a representative depolarization trace.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_ap_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level AHP feature.
    """
    if sweep_selector is None:
        sweep_selector = default_ap_sweep_selector

    return get_repr_sweep_ft(sweepset, "ahp", sweep_selector, ft_aggregator)


# adp
@ephys_feature
def get_sweepset_adp(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level Afterdepolarization feature.

    depends on: adp.
    description: ADP (adp_v - fast_trough_v) of a representative spike of a
    representative depolarization trace.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_ap_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level ADP feature.
    """
    if sweep_selector is None:
        sweep_selector = default_ap_sweep_selector

    return get_repr_sweep_ft(sweepset, "adp", sweep_selector, ft_aggregator)


# AP features
@ephys_feature
def get_sweepset_ap_thresh(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level AP threshold feature.

    depends on: ap_thresh.
    description: AP threshold of a representative spike of a representative depolarization trace.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_ap_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level AP threshold feature.
    """
    if sweep_selector is None:
        sweep_selector = default_ap_sweep_selector

    return get_repr_sweep_ft(sweepset, "ap_thresh", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_ap_amp(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level AP amplitude feature.

    depends on: ap_amp.
    description: AP amplitude of a representative spike of a representative
    depolarization trace.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_ap_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level AP amplitude feature.
    """
    if sweep_selector is None:
        sweep_selector = default_ap_sweep_selector

    return get_repr_sweep_ft(sweepset, "ap_amp", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_ap_width(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level AP width feature.

    depends on: ap_width.
    description: AP width of a representative spike of a representative
    depolarization trace.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_ap_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level AP width feature.
    """
    if sweep_selector is None:
        sweep_selector = default_ap_sweep_selector

    return get_repr_sweep_ft(sweepset, "ap_width", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_ap_peak(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level AP peak feature.

    depends on: ap_peak.
    description: Peak of AP of a representative spike of a representative
    depolarization trace.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_ap_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level AP peak feature.
    """
    if sweep_selector is None:
        sweep_selector = default_ap_sweep_selector

    return get_repr_sweep_ft(sweepset, "ap_peak", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_ap_trough(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level AP trough feature.

    depends on: ap_trough.
    description: AP trough of a representative spike of a representative
    depolarization trace.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_ap_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level AP trough feature.
    """
    if sweep_selector is None:
        sweep_selector = default_ap_sweep_selector

    return get_repr_sweep_ft(sweepset, "ap_trough", sweep_selector, ft_aggregator)


@ephys_feature
def get_sweepset_udr(
    sweepset: EphysSweepSetFeatureExtractor,
    sweep_selector: Optional[Callable] = None,
    ft_aggregator: Optional[Callable] = None,
) -> Tuple[float, Dict]:
    """Extract sweep set level AP upstroke to downstroke ratio feature.

    depends on: udr.
    description: AP upstroke-downstroke ratio of a representative spike of a
    representative depolarization trace.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.
        sweep_selector (Optional[Callable], optional): Function which selects a
            representative sweep or set of sweeps based on a given criterion.
            Function expects a EphysSweepSetFeatureExtractor object as input and
            returns indices for the selected sweeps. If none is provided, falls
            back to using `default_ap_sweep_selector`.
        ft_aggregator (Optional[Callable], optional): Function which aggregates
            a list of feature values into a single value. Function expects a
            list or ndarray of numbers as input. If none is provided, falls back
            to `np.nanmedian` (equates to pass through for single sweeps).

    Returns:
        Tuple[float, Dict]: sweep set level AP upstroke to downstroke ratio feature.
    """
    if sweep_selector is None:
        sweep_selector = default_ap_sweep_selector

    return get_repr_sweep_ft(sweepset, "udr", sweep_selector, ft_aggregator)


# rheobase
@ephys_feature
def get_sweepset_dfdi(sweepset: EphysSweepSetFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep set level df/di feature.

    depends on: stim_amp, num_ap, stim_onset, stim_end.
    description: df/di (slope) for the first 5 depolarization traces that
    contain spikes. If there are more than 2 duplicates in the number of spikes
    -> nan. Frequncy = num_spikes / T_stim.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.

    Returns:
        Tuple[float, Dict]: sweep set level df/di feature.
    """
    dfdi, dfdi_info = ephys_feature_init()
    is_depol = get_stripped_sweep_fts(sweepset)["stim_amp"] > 0
    i, f = (
        sweepset.get_sweep_features()
        .applymap(strip_info)[is_depol][["stim_amp", "ap_freq"]]
        .to_numpy()
        .T
    )
    has_spikes = ~np.isnan(f)
    if np.sum(has_spikes) > 4 and len(np.unique(f[:5])) > 3:
        i_s = i[has_spikes][:5]
        f_s = f[has_spikes][:5]

        ransac.fit(i_s.reshape(-1, 1), f_s.reshape(-1, 1))
        dfdi = ransac.coef_[0, 0]
        f_intercept = ransac.intercept_[0]
        dfdi_info.update(
            {"i_fit": i_s, "f_fit": f_s, "f": f, "i": i, "f_intercept": f_intercept}
        )
    return dfdi, dfdi_info


@ephys_feature
def get_sweepset_rheobase(
    sweepset: EphysSweepSetFeatureExtractor,
) -> Tuple[float, Dict]:
    """Extract sweep set level rheobase feature.

    depends on: dc_offset, stim_amp, num_ap, stim_onset, stim_end, df/di.
    description: rheobase current in pA. If df/di is nan, rheobase is the first
    depolarization trace that contains spikes. Otherwise, rheobase is the
    current where the fitted line crosses the current axis. The fitted intercept
    must lie between the stimulus of the first depolarization trace that contains
    spikes and the last depolarization that does not.

    Args:
        sweepset (EphysSweepSetFeatureExtractor): Sweep set to extract feature from.

    Returns:
        Tuple[float, Dict]: sweep set level rheobase feature.
    """
    dc_offset = get_stripped_sweep_fts(sweepset)["dc_offset"].iloc[0]
    rheobase, rheobase_info = ephys_feature_init()
    is_depol = get_stripped_sweep_fts(sweepset)["stim_amp"] > 0
    i, ap_freq = (
        sweepset.get_sweep_features()
        .applymap(strip_info)[is_depol][["stim_amp", "ap_freq"]]
        .to_numpy()
        .T
    )
    has_spikes = ~np.isnan(ap_freq)
    # sometimes all depolarization traces spike
    i_sub = 0 if all(has_spikes) else i[~has_spikes][0]  # last stim < spike threshold
    i_sup = i[has_spikes][0]  # first stim > spike threshold
    dfdi = strip_info(get_sweepset_dfdi(sweepset))

    if not np.isnan(dfdi):
        rheobase = float(ransac.predict(np.array([[0]]))) / dfdi

        if rheobase < i_sub or rheobase > i_sup:
            rheobase = i_sup
    else:
        rheobase = i_sup
    rheobase -= dc_offset
    rheobase_info.update(
        {
            "i_sub": i_sub,
            "i_sup": i_sup,
            "f_sup": ap_freq[has_spikes][0],
            "dfdi": dfdi,
            "dc_offset": dc_offset,
        }
    )
    return rheobase, rheobase_info


def get_available_sweepset_features(return_ft_info=False):
    _ft_dict = {
        "tau": get_sweepset_tau,
        "r_input": get_sweepset_r_input,
        "v_rest": get_sweepset_v_rest,
        "v_baseline": get_sweepset_v_baseline,
        "slow_hyperpolarization": get_sweepset_slow_hyperpolarization,
        "sag": get_sweepset_sag,
        "sag_ratio": get_sweepset_sag_ratio,
        "sag_fraction": get_sweepset_sag_fraction,
        "sag_area": get_sweepset_sag_area,
        "sag_time": get_sweepset_sag_time,
        "rebound": get_sweepset_rebound,
        "rebound_aps": get_sweepset_rebound_aps,
        "rebound_area": get_sweepset_rebound_area,
        "rebound_latency": get_sweepset_rebound_latency,
        "rebound_avg": get_sweepset_rebound_avg,
        "num_ap": get_sweepset_num_ap,
        "ap_freq": get_sweepset_ap_freq,
        "wildness": get_sweepset_wildness,
        "ap_freq_adapt": get_sweepset_ap_freq_adapt,
        "ap_amp_slope": get_sweepset_ap_amp_slope,
        "fano_factor": get_sweepset_fano_factor,
        "ap_fano_factor": get_sweepset_ap_fano_factor,
        "cv": get_sweepset_cv,
        "ap_cv": get_sweepset_ap_cv,
        "burstiness": get_sweepset_burstiness,
        "num_bursts": get_sweepset_num_bursts,
        "isi_adapt": get_sweepset_isi_adapt,
        "isi_adapt_avg": get_sweepset_isi_adapt_avg,
        "ap_amp_adapt": get_sweepset_ap_amp_adapt,
        "ap_amp_adapt_avg": get_sweepset_ap_amp_adapt_avg,
        "ap_latency": get_sweepset_ap_latency,
        "ahp": get_sweepset_ahp,
        "adp": get_sweepset_adp,
        "ap_thresh": get_sweepset_ap_thresh,
        "ap_amp": get_sweepset_ap_amp,
        "ap_width": get_sweepset_ap_width,
        "ap_peak": get_sweepset_ap_peak,
        "ap_trough": get_sweepset_ap_trough,
        "udr": get_sweepset_udr,
        "dfdi": get_sweepset_dfdi,
        "rheobase": get_sweepset_rheobase,
    }

    if return_ft_info:
        return {k: partial(v, return_ft_info=True) for k, v in _ft_dict.items()}
    return _ft_dict
