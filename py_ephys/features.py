import numpy as np
import py_ephys.allen_sdk.ephys_extractor as efex
import py_ephys.allen_sdk.ephys_features as ft
from py_ephys.utils import *

from numpy import ndarray
from pandas import DataFrame
from typing import Tuple, Dict
from functools import partial
from scipy import integrate

from sklearn import linear_model

# ransac = linear_model.RANSACRegressor()
ransac = linear_model.LinearRegression()

############################
### spike level features ###
############################


def get_spike_peak_height(sweep: efex.EphysSweepFeatureExtractor) -> float:
    """Extract spike level peak height feature.

    description: v_peak - threshold_v.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        float: Spike peak height feature.
    """
    v_peak = sweep.spike_feature("peak_v")
    threshold_v = sweep.spike_feature("threshold_v")
    peak_height = v_peak - threshold_v
    return peak_height if len(v_peak) > 0 else np.array([])


def get_spike_ahp(sweep: efex.EphysSweepFeatureExtractor) -> float:
    """Extract spike level after hyperpolarization feature.

    description: v_fast_trough - threshold_v.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        float: Spike after hyperpolarization feature.
    """
    v_fast_trough = sweep.spike_feature("fast_trough_v", include_clipped=True)
    threshold_v = sweep.spike_feature("threshold_v", include_clipped=True)
    return v_fast_trough - threshold_v


def get_spike_adp(sweep: efex.EphysSweepFeatureExtractor) -> float:
    """Extract spike level after depolarization feature.

    description: v_adp - v_fast_trough.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        float: Spike after depolarization feature.
    """
    v_adp = sweep.spike_feature("adp_v", include_clipped=True)
    v_fast_trough = sweep.spike_feature("fast_trough_v", include_clipped=True)
    return v_adp - v_fast_trough


def get_fp_spike_ft_dict() -> Dict[str, callable]:
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
    sweep: efex.EphysSweepFeatureExtractor,
) -> Tuple[ndarray, ndarray, ndarray]:
    """Calculate burst metrics for a sweep.

    Uses EphysExtractor's _process_bursts() method to calculate burst metrics.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to calculate burst metrics for.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: returns burst index, burst start index,
            burst end index.
    """
    burst_metrics = sweep._process_bursts()
    if len(burst_metrics) == 0:
        return float("nan") * np.ones(3, dtype=int)
    idx_burst, idx_burst_start, idx_burst_end = burst_metrics
    return idx_burst, idx_burst_start, idx_burst_end


@ephys_feature
def get_sweep_num_ap(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level spike count feature.

    description: # peaks during stimulus.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: Spike count feature and feature metadata
    """
    num_ap, num_ap_info = ephys_feature_init()
    stim_interval = where_stimulus(sweep)
    peak_i = sweep.spike_feature("peak_i")[stim_interval]
    peak_t = sweep.spike_feature("peak_t")[stim_interval]
    peak_v = sweep.spike_feature("peak_v")[stim_interval]

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
def get_sweep_ap_freq(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level spike rate feature.

    description: # peaks during stimulus / stimulus duration.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

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
def get_sweep_stim_amplitude(
    sweep: efex.EphysSweepFeatureExtractor,
) -> Union[Dict, float]:
    """Extract sweep level stimulus ampltiude feature.

    description: maximum amplitude of stimulus.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

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
            "stim_amp": stim_amp,
            "idx_stim_amp": idx_stim_amp,
            "t_stim_amp": t_stim_amp,
        }
    )
    return stim_amp, stim_amp_info


@ephys_feature
def get_sweep_stim_onset(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level stimulus onset feature.

    description: time of stimulus onset.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

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
def get_sweep_stim_end(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level stimulus end feature.

    description: time of stimulus end.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

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
def get_sweep_v_deflect(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level voltage deflection feature.

    description: average voltage during last 100 ms of stimulus.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: voltage deflection feature and feature metadata
    """
    v_deflect_avg, v_deflect_info = ephys_feature_init()
    if has_stimulus(sweep):
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
def get_sweep_v_baseline(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level baseline voltage feature.

    description: average voltage between baseline_interval (in s) and stimulus onset.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: baseline voltage feature and feature metadata
    """
    v_baseline_avg, v_baseline_info = ephys_feature_init()
    v_baseline_avg = sweep._get_baseline_voltage()
    onset = strip_info(sweep.sweep_feature("stim_onset"))
    idx_baseline = where_between(sweep.t, sweep.baseline_interval, onset)
    t_baseline = sweep.t[idx_baseline]
    v_baseline = sweep.v[idx_baseline]
    v_baseline_info.update(
        {
            "idx_baseline": idx_baseline,
            "t_baseline": t_baseline,
            "v_baseline": v_baseline,
            "baseline_interval": sweep.baseline_interval,
        }
    )
    return v_baseline_avg, v_baseline_info


@ephys_feature
def get_sweep_time_constant(
    sweep: efex.EphysSweepFeatureExtractor,
) -> Union[Dict, float]:
    """Extract sweep level time constant feature.

    description: time constant of exponential fit to voltage deflection.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: time constant feature and feature metadata
    """
    tau, tau_info = ephys_feature_init()
    if is_hyperpolarizing(sweep):
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
def get_sweep_spike_freq_adapt(
    sweep: efex.EphysSweepFeatureExtractor,
) -> Union[Dict, float]:
    """Extract sweep level spike frequency adaptation feature.

    description: ratio of spikes in second and first half half of stimulus interval, if there is at least 5 spikes in total.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: spike freq adaptation feature and feature metadata
    """
    spike_freq_adapt, spike_freq_adapt_info = ephys_feature_init()
    if strip_info(sweep.sweep_feature("num_ap")) > 5 and has_stimulus(sweep):
        onset = strip_info(sweep.sweep_feature("stim_onset"))
        end = strip_info(sweep.sweep_feature("stim_end"))
        t_half = (end - onset) / 2
        idx_1st_half = where_between(sweep.t, onset, t_half)
        idx_2nd_half = where_between(sweep.t, t_half, end)
        t_1st_half = sweep.t[idx_1st_half]
        t_2nd_half = sweep.t[idx_2nd_half]

        spike_times = sweep.spike_feature("peak_t")
        spike_times = spike_times[where_stimulus(sweep)]

        spikes_1st_half = spike_times[spike_times < t_half]
        spikes_2nd_half = spike_times[spike_times > t_half]
        num_spikes_1st_half = len(spikes_1st_half)
        num_spikes_2nd_half = len(spikes_2nd_half)
        spike_freq_adapt = num_spikes_2nd_half / num_spikes_1st_half

        spike_freq_adapt_info.update(
            {
                "num_spikes_1st_half": num_spikes_1st_half,
                "num_spikes_2nd_half": num_spikes_2nd_half,
                "idx_1st_half": idx_1st_half,
                "idx_2nd_half": idx_2nd_half,
                "t_1st_half": t_1st_half,
                "t_2nd_half": t_2nd_half,
            }
        )
    return spike_freq_adapt, spike_freq_adapt_info


@ephys_feature
def get_sweep_r_input(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level input resistance feature.

    description: sweep level input resistance as (v_deflect - v_baseline / current).
    Should not be used for cell level feature.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: input resistance feature and feature metadata
    """
    r_input, r_input_info = ephys_feature_init()
    if is_hyperpolarizing(sweep):
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
def get_sweep_sag(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level sag feature.

    description: magnitude of the depolarization peak.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: sag feature and feature metadata
    """
    sag, sag_info = ephys_feature_init()
    if is_hyperpolarizing(sweep):
        with strip_sweep_ft_info(sweep) as fsweep:
            sag_fts = fsweep.estimate_sag()
        sag = sag_fts[0]
    return sag, sag_info


@ephys_feature
def get_sweep_sag_fraction(
    sweep: efex.EphysSweepFeatureExtractor,
) -> Tuple[float, Dict]:
    """Extract sweep level sag fraction feature.

    description: fraction that membrane potential relaxes back to baseline.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: sag fraction feature and feature metadata
    """
    sag_fraction, sag_fraction_info = ephys_feature_init()
    if is_hyperpolarizing(sweep):
        with strip_sweep_ft_info(sweep) as fsweep:
            sag_fts = fsweep.estimate_sag()
        sag_fraction = sag_fts[1]
    return sag_fraction, sag_fraction_info


@ephys_feature
def get_sweep_sag_ratio(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level sag ratio feature.

    description: ratio of steady state voltage decrease to the largest voltage decrease.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: sag ratio and feature metadata
    """
    sag_ratio, sag_ratio_info = ephys_feature_init()
    if is_hyperpolarizing(sweep):
        with strip_sweep_ft_info(sweep) as fsweep:
            sag_fts = fsweep.estimate_sag()
        sag_ratio = sag_fts[2]
    return sag_ratio, sag_ratio_info


@ephys_feature
def get_sweep_sag_area(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level sag area feature.

    description: area under the sag.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: sag area feature and feature metadata
    """
    sag_area, sag_area_info = ephys_feature_init()
    if is_hyperpolarizing(sweep):
        idx_sag = get_sweep_sag_idxs(sweep)
        if np.sum(idx_sag) > 10:  # TODO: what should be min sag duration!?
            v_sag = sweep.v[idx_sag]
            t_sag = sweep.t[idx_sag]
            # Take running average of v?
            sag_area = -integrate.cumulative_trapezoid(v_sag, t_sag)[-1]
            sag_area_info.update(
                {
                    "idx_sag": idx_sag,
                    "v_sag": v_sag,
                    "t_sag": t_sag,
                }
            )
    return sag_area, sag_area_info


@ephys_feature
def get_sweep_sag_time(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level sag duration feature.

    description: duration of the sag.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: sag duration feature and feature metadata
    """
    sag_time, sag_time_info = ephys_feature_init()
    if is_hyperpolarizing(sweep):
        idx_sag = get_sweep_sag_idxs(sweep)
        if np.sum(idx_sag) > 10:  # TODO: what should be min sag duration!?
            sag_t_start, sag_t_end = sweep.t[idx_sag][[0, -1]]
            sag_time = sag_t_end - sag_t_start
            sag_time_info.update(
                {
                    "idx_sag": idx_sag,
                    "sag_t_start": sag_t_start,
                    "sag_t_end": sag_t_end,
                }
            )
    return sag_time, sag_time_info


@ephys_feature
def get_sweep_v_plateau(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level plataeu voltage feature.

    description: average voltage during the plateau.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: plateau voltage feature and feature metadata
    """
    v_avg_plateau, v_plateau_info = ephys_feature_init()
    if is_hyperpolarizing(sweep):
        end = strip_info(sweep.sweep_feature("stim_end"))
        # same as voltage deflection
        idx_plateau = where_between(sweep.t, end - 0.1, end)
        v_plateau = sweep.v[idx_plateau]
        t_plateau = sweep.t[idx_plateau]
        v_avg_plateau = ft.average_voltage(v_plateau, t_plateau)
        v_plateau_info.update(
            {
                "idx_plateau": idx_plateau,
                "v_plateau": v_plateau,
                "t_plateau": t_plateau,
            }
        )
    return v_avg_plateau, v_plateau_info


@ephys_feature
def get_sweep_rebound(
    sweep: efex.EphysSweepFeatureExtractor,
    T_rebound: float = 0.3,
) -> Tuple[float, Dict]:
    """Extract sweep level rebound feature.

    description: V_max during stimulus_end and stimulus_end + T_rebound - V_baseline.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.
        T_rebound (float, optional): Time after stimulus end to look for rebound.

    Returns:
        Tuple[float, Dict]: rebound feature and feature metadata
    """
    rebound, rebound_info = ephys_feature_init({"T_rebound": T_rebound})
    if has_rebound(sweep, T_rebound):
        v_baseline = strip_info(sweep.sweep_feature("v_baseline"))
        end = strip_info(sweep.sweep_feature("stim_end"))
        idx_rebound = where_between(sweep.t, end, end + T_rebound)
        idx_rebound = np.logical_and(idx_rebound, sweep.v > v_baseline)
        t_rebound = sweep.t[idx_rebound]
        rebound = np.max(sweep.v[idx_rebound] - v_baseline)
        rebound_info.update(
            {
                "idx_rebound": idx_rebound,
                "t_rebound": t_rebound,
                "v_baseline": v_baseline,
            }
        )
    return rebound, rebound_info


@ephys_feature
def get_sweep_rebound_spikes(
    sweep: efex.EphysSweepFeatureExtractor,
    T_rebound: float = 0.3,
) -> Tuple[float, Dict]:
    """Extract sweep level number of rebounding spikes feature.

    description: number of spikes during stimulus_end and stimulus_end + T_rebound.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.
        T_rebound (float, optional): Time after stimulus end to look for rebound.

    Returns:
        Tuple[float, Dict]: num rebound spikes feature and feature metadata
    """
    num_rebound_spikes, rebound_spike_info = ephys_feature_init(
        {"T_rebound": T_rebound}
    )
    if has_rebound(sweep, T_rebound):
        t_spike = sweep.spike_feature("peak_t")
        idx_spike = sweep.spike_feature("peak_i")
        v_spike = sweep.spike_feature("peak_v")
        if len(t_spike) != 0:
            end = strip_info(sweep.sweep_feature("stim_end"))
            w_rebound = where_between(t_spike, end, end + T_rebound)
            idx_rebound = idx_spike[w_rebound]
            t_rebound = t_spike[w_rebound]
            v_rebound = v_spike[w_rebound]
            num_rebound_spikes = np.sum(w_rebound)
            if num_rebound_spikes > 0:
                rebound_spike_info.update(
                    {
                        "idx_rebound": idx_rebound,
                        "t_rebound": t_rebound,
                        "v_rebound": v_rebound,
                    }
                )
    return num_rebound_spikes, rebound_spike_info


@ephys_feature
def get_sweep_rebound_latency(
    sweep: efex.EphysSweepFeatureExtractor,
    T_rebound: float = 0.3,
) -> Tuple[float, Dict]:
    """Extract sweep level rebound latency feature.

    description: duration from stimulus_end to when the voltage reaches above
    baseline for the first time. t_rebound = t_off + rebound_latency.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.
        T_rebound (float, optional): Time after stimulus end to look for rebound.

    Returns:
        Tuple[float, Dict]: rebound latency feature and feature metadata
    """
    rebound_latency, rebound_latency_info = ephys_feature_init({"T_rebound": T_rebound})
    if has_rebound(sweep, T_rebound):
        v_baseline = strip_info(sweep.sweep_feature("v_baseline"))
        end = strip_info(sweep.sweep_feature("stim_end"))
        idx_rebound = where_between(sweep.t, end, end + T_rebound)
        idx_rebound = np.logical_and(idx_rebound, sweep.v > v_baseline)
        t_rebound = sweep.t[idx_rebound]
        v_rebound = sweep.v[idx_rebound]
        idx_rebound_reached = np.where(idx_rebound)[0]
        t_rebound_reached = sweep.t[idx_rebound_reached][0]
        rebound_latency = t_rebound_reached - end
        rebound_latency_info.update(
            {
                "idx_rebound_reached": idx_rebound_reached,
                "t_rebound_reached": t_rebound_reached,
                "idx_rebound": idx_rebound,
                "t_rebound": t_rebound,
                "v_rebound": v_rebound,
                "v_baseline": v_baseline,
            }
        )
    return rebound_latency, rebound_latency_info


@ephys_feature
def get_sweep_rebound_area(
    sweep: efex.EphysSweepFeatureExtractor,
    T_rebound: float = 0.3,
) -> Tuple[float, Dict]:
    """Extract sweep level rebound area feature.

    description: area between rebound curve and baseline voltage from stimulus_end
    to stimulus_end + T_rebound.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.
        T_rebound (float, optional): Time after stimulus end to look for rebound.

    Returns:
        Tuple[float, Dict]: rebound area feature and feature metadata
    """
    rebound_area, rebound_area_info = ephys_feature_init({"T_rebound": T_rebound})
    if has_rebound(sweep, T_rebound):
        v_baseline = strip_info(sweep.sweep_feature("v_baseline"))
        end = strip_info(sweep.sweep_feature("stim_end"))
        idx_rebound = where_between(sweep.t, end, end + T_rebound)
        idx_rebound = np.logical_and(idx_rebound, sweep.v > v_baseline)
        v_rebound = sweep.v[idx_rebound]
        t_rebound = sweep.t[idx_rebound]
        rebound_area = integrate.cumulative_trapezoid(
            v_rebound - v_baseline, t_rebound
        )[-1]
        rebound_area_info.update(
            {
                "idx_rebound": idx_rebound,
                "t_rebound": t_rebound,
                "v_rebound": v_rebound,
                "v_baseline": v_baseline,
            }
        )
    return rebound_area, rebound_area_info


@ephys_feature
def get_sweep_rebound_avg(
    sweep: efex.EphysSweepFeatureExtractor,
    T_rebound: float = 0.3,
) -> Tuple[float, Dict]:
    """Extract sweep level average rebound feature.

    description: average voltage between stimulus_end
    and stimulus_end + T_rebound - baseline voltage.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.
        T_rebound (float, optional): Time after stimulus end to look for rebound.

    Returns:
        Tuple[float, Dict]: average rebound feature and feature metadata
    """
    v_rebound_avg, rebound_avg_info = ephys_feature_init({"T_rebound": T_rebound})
    if has_rebound(sweep, T_rebound):
        v_baseline = strip_info(sweep.sweep_feature("v_baseline"))
        end = strip_info(sweep.sweep_feature("stim_end"))
        idx_rebound = where_between(sweep.t, end, end + T_rebound)
        idx_rebound = np.logical_and(idx_rebound, sweep.v > v_baseline)
        v_rebound = sweep.v[idx_rebound]
        t_rebound = sweep.t[idx_rebound]
        v_rebound_avg = ft.average_voltage(v_rebound - v_baseline, t_rebound)
        rebound_avg_info.update(
            {
                "idx_rebound": idx_rebound,
                "t_rebound": t_rebound,
                "v_rebound": v_rebound,
                "v_baseline": v_baseline,
            }
        )
    return v_rebound_avg, rebound_avg_info


@ephys_feature
def get_sweep_v_rest(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level resting potential feature.

    description: v_rest = v_baseline - r_input*dc_offset.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: resting potential feature and feature metadata
    """
    v_rest, v_rest_info = ephys_feature_init()
    v_baseline = strip_info(sweep.sweep_feature("v_baseline"))
    r_input = strip_info(sweep.sweep_feature("r_input"))
    try:
        dc_offset = strip_info(sweep.sweep_feature("dc_offset"))
        v_rest = v_baseline - r_input * 1000 * dc_offset
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
def get_sweep_num_bursts(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level number of bursts feature.

    description: Number of detected bursts.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: number of bursts feature and feature metadata
    """
    num_bursts, num_bursts_info = ephys_feature_init()
    if strip_info(sweep.sweep_feature("num_ap")) > 5 and has_stimulus(sweep):
        idx_burst, idx_burst_start, idx_burst_end = get_sweep_burst_metrics(sweep)
        if not np.isnan(idx_burst):
            t_burst_start = sweep.t[idx_burst_start]
            t_burst_end = sweep.t[idx_burst_end]
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
def get_sweep_burstiness(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level burstiness feature.

    description: max "burstiness" index across detected bursts.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: burstiness feature and feature metadata
    """
    max_burstiness, burstiness_info = ephys_feature_init()
    if strip_info(sweep.sweep_feature("num_ap")) > 5 and has_stimulus(sweep):
        idx_burst, idx_burst_start, idx_burst_end = get_sweep_burst_metrics(sweep)
        if not np.isnan(idx_burst):
            t_burst_start = sweep.t[idx_burst_start]
            t_burst_end = sweep.t[idx_burst_end]
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
def get_sweep_wildness(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level wildness feature.

    description: Wildness is the number of spikes that occur outside of the stimulus interval.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: wildness feature and feature metadata
    """
    num_wild_spikes, wildness_info = ephys_feature_init()
    stim_interval = where_stimulus(sweep)
    i_wild_spikes = sweep.spike_feature("peak_i")[~stim_interval]
    t_wild_spikes = sweep.spike_feature("peak_t")[~stim_interval]
    v_wild_spikes = sweep.spike_feature("peak_v")[~stim_interval]
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


def select_representative_ap(sweep: efex.EphysSweepFeatureExtractor) -> int:
    """Select representative AP from which the ap features are extracted.

    description: First AP.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep from which the ap features are extracted.

    Returns:
        int: Index of the sweep from which the rebound features are extracted.
    """
    return 0


@ephys_feature
def get_sweep_ahp(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level Afterhyperpolarization feature.

    description: Afterhyperpolarization (AHP) for representative AP. Difference
    between the fast trough and the threshold.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: AHP feature and feature metadata
    """
    ahp_selected, ahp_info = ephys_feature_init()
    ahp = sweep.spike_feature("ahp")
    if len(ahp) > 0:
        select_ap_idx = select_representative_ap(sweep)
        ahp_selected = ahp[select_ap_idx]
        ahp_info.update(
            {
                "ap_idx": select_ap_idx,
                "selection": parse_ft_desc(select_representative_ap),
            }
        )
    return ahp_selected, ahp_info


@ephys_feature
def get_sweep_adp(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level Afterdepolarization feature.

    description: Afterdepolarization (ADP) for representative AP. Difference between the ADP and the fast trough.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: ADP feature and feature metadata
    """
    adp_selected, adp_info = ephys_feature_init()
    adp = sweep.spike_feature("adp")
    if len(adp) > 0:
        select_ap_idx = select_representative_ap(sweep)
        adp_selected = adp[select_ap_idx]
        adp_info.update(
            {
                "ap_idx": select_ap_idx,
                "selection": parse_ft_desc(select_representative_ap),
            }
        )
    return adp_selected, adp_info


@ephys_feature
def get_sweep_ap_thresh(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level AP threshold feature.

    description: AP threshold for representative AP.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: AP threshold feature and feature metadata
    """
    ap_thresh_selected, ap_tresh_info = ephys_feature_init()
    ap_thresh = sweep.spike_feature("threshold_v")
    if len(ap_thresh) > 0:
        select_ap_idx = select_representative_ap(sweep)
        ap_thresh_selected = ap_thresh[select_ap_idx]
        ap_tresh_info.update(
            {
                "ap_idx": select_ap_idx,
                "selection": parse_ft_desc(select_representative_ap),
            }
        )
    return ap_thresh_selected, ap_tresh_info


@ephys_feature
def get_sweep_ap_amp(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level AP amplitude feature.

    description: AP amplitude for representative AP.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: AP amplitude feature and feature metadata
    """
    ap_amp_selected, ap_amp_info = ephys_feature_init()
    ap_amp = sweep.spike_feature("peak_height")
    if len(ap_amp) > 0:
        select_ap_idx = select_representative_ap(sweep)
        ap_amp_selected = ap_amp[select_ap_idx]
        ap_amp_info.update(
            {
                "ap_idx": select_ap_idx,
                "selection": parse_ft_desc(select_representative_ap),
            }
        )
    return ap_amp_selected, ap_amp_info


@ephys_feature
def get_sweep_ap_width(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level AP width feature.

    description: AP width for representative AP.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: AP width feature and feature metadata
    """
    ap_width_selected, ap_width_info = ephys_feature_init()
    ap_width = sweep.spike_feature("width")
    if len(ap_width) > 0:
        select_ap_idx = select_representative_ap(sweep)
        ap_width_selected = ap_width[select_ap_idx]
        ap_width_info.update(
            {
                "ap_idx": select_ap_idx,
                "selection": parse_ft_desc(select_representative_ap),
            }
        )
    return ap_width_selected, ap_width_info


@ephys_feature
def get_sweep_ap_peak(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level AP peak feature.

    description: AP peak for representative AP.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: AP peak feature and feature metadata
    """
    ap_peak_selected, ap_peak_info = ephys_feature_init()
    ap_peak = sweep.spike_feature("peak_v")
    if len(ap_peak) > 0:
        select_ap_idx = select_representative_ap(sweep)
        ap_peak_selected = ap_peak[select_ap_idx]
        ap_peak_info.update(
            {
                "ap_idx": select_ap_idx,
                "selection": parse_ft_desc(select_representative_ap),
            }
        )
    return ap_peak_selected, ap_peak_info


@ephys_feature
def get_sweep_ap_trough(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level AP trough feature.

    description: AP trough for representative AP.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: AP trough feature and feature metadata
    """
    ap_trough_selected, ap_trough_info = ephys_feature_init()
    ap_trough = sweep.spike_feature("trough_v")
    if len(ap_trough) > 0:
        select_ap_idx = select_representative_ap(sweep)
        ap_trough_selected = ap_trough[select_ap_idx]
        ap_trough_info.update(
            {
                "ap_idx": select_ap_idx,
                "selection": parse_ft_desc(select_representative_ap),
            }
        )
    return ap_trough_selected, ap_trough_info


@ephys_feature
def get_sweep_udr(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extract sweep level Upstroke-to-downstroke ratio feature.

    description: Upstroke-to-downstroke ratio for representative AP.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

    Returns:
        Tuple[float, Dict]: UDR feature and feature metadata
    """
    udr_selected, udr_info = ephys_feature_init()
    udr = sweep.spike_feature("upstroke_downstroke_ratio")
    if len(udr) > 0:
        select_ap_idx = select_representative_ap(sweep)
        udr_selected = udr[select_ap_idx]
        udr_info.update(
            {
                "ap_idx": select_ap_idx,
                "selection": parse_ft_desc(select_representative_ap),
            }
        )
    return udr_selected, udr_info


### Feature extraction functions
def get_fp_sweep_ft_dict(return_ft_info=False):
    fp_ft_dict = {
        "stim_amp": get_sweep_stim_amplitude,  # None
        "stim_onset": get_sweep_stim_onset,  # None
        "stim_end": get_sweep_stim_end,  # None
        "v_deflect": get_sweep_v_deflect,  # stim_end
        "v_baseline": get_sweep_v_baseline,  # stim_onset
        "tau": get_sweep_time_constant,  # v_baseline
        "num_ap": get_sweep_num_ap,  # spike_features
        "ap_freq": get_sweep_ap_freq,  # num_ap, stim_onset, stim_end
        "spike_freq_adapt": get_sweep_spike_freq_adapt,  # num_ap, stim_onset, stim_end, spike_features
        "r_input": get_sweep_r_input,  # stim_onset, stim_end, stim_amp, v_baseline, v_deflect
        "sag": get_sweep_sag,  # None (part of efex)
        "sag_fraction": get_sweep_sag_fraction,  # None (part of efex)
        "sag_ratio": get_sweep_sag_ratio,  # None (part of efex)
        "sag_area": get_sweep_sag_area,  # stim_onset, stim_end, v_deflect
        "sag_time": get_sweep_sag_time,  # stim_onset, stim_end, v_deflect
        "v_plateau": get_sweep_v_plateau,  # stim_end
        "rebound": get_sweep_rebound,  # stim_end, v_baseline
        "rebound_spikes": get_sweep_rebound_spikes,  # stim_end, spike_features
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
        return {k: partial(v, return_ft_info=True) for k, v in fp_ft_dict.items()}
    return fp_ft_dict


################################
### sweep set level features ###
################################


@ephys_feature
def get_sweepset_time_constant(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level time constant feature.

    description: median of the membrane time constants from all hyperpolarizing traces.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level time constant feature.
    """
    tau, tau_info = ephys_feature_init()
    tau = sweep_fts["tau"].median(skipna=True)
    return tau, tau_info


# input resistance
@ephys_feature
def get_sweepset_r_input(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level input resistance feature.

    description: fitted slope of v_deflect(v_input) in MOhms.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level input resistance feature.
    """
    r_input, r_input_info = ephys_feature_init()
    is_hyperpol = sweep_fts["stim_amp"] < 0
    v_deflect = sweep_fts["v_deflect"].loc[is_hyperpol].to_numpy().reshape(-1, 1)
    i_amp = sweep_fts["stim_amp"].loc[is_hyperpol].to_numpy().reshape(-1, 1)
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


# resting potential
@ephys_feature
def get_sweepset_v_rest(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level resting potential feature.

    description: median of the resting potentials from all hyperpolarizing
    traces. v_rest = v_baseline - r_input * dc

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level resting potential feature.
    """
    dc_offset = sweep_fts["dc_offset"]
    v_rest, v_rest_info = ephys_feature_init({})
    is_hyperpol = sweep_fts["stim_amp"] < 0
    r_input = get_sweepset_r_input(
        sweep_fts
    )  # TEMPORARY SINCE THIS NEEDS UNNECESSARY FITS
    v_base = sweep_fts["v_baseline"][is_hyperpol]
    v_rests = v_base - r_input * 1e-3 * dc_offset
    v_rest = v_rests.median(skipna=True)
    v_rest_info.update(
        {
            "r_input": r_input,
            "v_rest": v_rests,
            "v_base": v_base,
            "dc_offset": dc_offset,
        }
    )
    return v_rest, v_rest_info


# slow hyperpolarizing potential
@ephys_feature
def get_sweepset_slow_hyperpolarization(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level slow hyperpolarization feature.

    description: difference between the max and min baseline voltages.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level slow hyperpolarization feature.
    """
    slow_hyperpol, slow_hyperpol_info = ephys_feature_init()
    is_hyperpol = (
        sweep_fts["stim_amp"] < 0
    )  # TODO: ASK IF THIS IS ONLY TAKEN FOR HYPERPOLARIZING TRACES (I THINK NOT)
    v_base = sweep_fts["v_baseline"][is_hyperpol]
    slow_hyperpol = (
        v_base.max() - v_base.min()
    )  # like v_rest.min() - v_rest.max(), since v_rest = v_base - const
    slow_hyperpol_info.update({"v_base": v_base})
    return slow_hyperpol, slow_hyperpol_info


# sag features (steepest hyperpolarizing trace)
def select_representative_sag_sweep(sweep_fts: DataFrame) -> int:
    """Select representative sweep from which the sag features are extracted.

    description: Lowest hyperpolarization sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        int: Index of the sweep from which the rebound features are extracted.
    """
    # TODO: implement selection protocol / criterion!
    return 0  # TODO: select one or multiple? -> median, mean?


@ephys_feature
def get_sweepset_sag(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level sag feature.

    description: sag voltage for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level sag feature.
    """
    sag, sag_info = ephys_feature_init()
    sag_sweep_idx = select_representative_sag_sweep(sweep_fts)
    sag = sweep_fts["sag"].iloc[sag_sweep_idx]
    sag_info.update(
        {
            "sag_sweep_idx": sag_sweep_idx,
            "selection": parse_ft_desc(select_representative_sag_sweep),
        }
    )
    return sag, sag_info


@ephys_feature
def get_sweepset_sag_ratio(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level sag ratio feature.

    description: sag ratio for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level sag ratio feature.
    """
    sag_ratio, sag_ratio_info = ephys_feature_init()
    sag_sweep_idx = select_representative_sag_sweep(sweep_fts)
    sag_ratio = sweep_fts["sag_ratio"].iloc[sag_sweep_idx]
    sag_ratio_info.update(
        {
            "sag_sweep_idx": sag_sweep_idx,
            "selection": parse_ft_desc(select_representative_sag_sweep),
        }
    )
    return sag_ratio, sag_ratio_info


@ephys_feature
def get_sweepset_sag_fraction(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level sag fraction feature.

    description: sag fraction for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.


    Returns:
        Tuple[float, Dict]: sweep set level sag fraction feature.
    """
    sag_fraction, sag_fraction_info = ephys_feature_init()
    sag_sweep_idx = select_representative_sag_sweep(sweep_fts)
    sag_fraction = sweep_fts["sag_fraction"].iloc[sag_sweep_idx]
    sag_fraction_info.update(
        {
            "sag_sweep_idx": sag_sweep_idx,
            "selection": parse_ft_desc(select_representative_sag_sweep),
        }
    )
    return sag_fraction, sag_fraction_info


@ephys_feature
def get_sweepset_sag_area(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level sag area feature.

    description: sag area for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level sag area feature.
    """
    sag_area, sag_area_info = ephys_feature_init()
    sag_sweep_idx = select_representative_sag_sweep(sweep_fts)
    sag_area = sweep_fts["sag_area"].iloc[sag_sweep_idx]
    sag_area_info.update(
        {
            "sag_sweep_idx": sag_sweep_idx,
            "selection": parse_ft_desc(select_representative_sag_sweep),
        }
    )
    return sag_area, sag_area_info


@ephys_feature
def get_sweepset_sag_time(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level sag time feature.

    description: sag time for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level sag time feature.
    """
    sag_time, sag_time_info = ephys_feature_init()
    sag_sweep_idx = select_representative_sag_sweep(sweep_fts)
    sag_time = sweep_fts["sag_time"].iloc[sag_sweep_idx]
    sag_time_info.update(
        {
            "sag_sweep_idx": sag_sweep_idx,
            "selection": parse_ft_desc(select_representative_sag_sweep),
        }
    )
    return sag_time, sag_time_info


# rebound features (steepest hyperpolarizing trace)
def select_representative_rebound_sweep(sweep_fts: DataFrame) -> int:
    """Select representative sweep from which the rebound features are extracted.

    description: Lowest hyperpolarization sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        int: Index of the sweep from which the rebound features are extracted.
    """
    # TODO: implement selection protocol / criterion!
    return 0  # TODO: select one or multiple? -> median, mean?


@ephys_feature
def get_sweepset_rebound(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level rebound feature.

    description: rebound voltage for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level rebound feature.
    """
    rebound, rebound_info = ephys_feature_init()
    rebound_sweep_idx = select_representative_rebound_sweep(sweep_fts)
    rebound = sweep_fts["rebound"].iloc[rebound_sweep_idx]
    rebound_info.update(
        {
            "rebound_sweep_idx": rebound_sweep_idx,
            "selection": parse_ft_desc(select_representative_rebound_sweep),
        }
    )
    return rebound, rebound_info


@ephys_feature
def get_sweepset_rebound_ratio(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level rebound ratio feature.

    description: rebound ratio for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level rebound ratio feature.
    """
    rebound_ratio, rebound_ratio_info = ephys_feature_init()
    rebound_sweep_idx = select_representative_rebound_sweep(sweep_fts)
    rebound_ratio = sweep_fts["rebound_spikes"].iloc[rebound_sweep_idx]
    rebound_ratio_info.update(
        {
            "rebound_sweep_idx": rebound_sweep_idx,
            "selection": parse_ft_desc(select_representative_rebound_sweep),
        }
    )
    return rebound_ratio, rebound_ratio_info


@ephys_feature
def get_sweepset_rebound_area(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level rebound area feature.

    description: rebound area for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level rebound area feature.
    """
    rebound_area, rebound_area_info = ephys_feature_init()
    rebound_sweep_idx = select_representative_rebound_sweep(sweep_fts)
    rebound_area = sweep_fts["rebound_area"].iloc[rebound_sweep_idx]
    rebound_area_info.update(
        {
            "rebound_sweep_idx": rebound_sweep_idx,
            "selection": parse_ft_desc(select_representative_rebound_sweep),
        }
    )
    return rebound_area, rebound_area_info


@ephys_feature
def get_sweepset_rebound_latency(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level rebound latency feature.

    description: rebound latency for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level rebound latency feature.
    """
    rebound_latency, rebound_latency_info = ephys_feature_init()
    rebound_sweep_idx = select_representative_rebound_sweep(sweep_fts)
    rebound_latency = sweep_fts["rebound_latency"].iloc[rebound_sweep_idx]
    rebound_latency_info.update(
        {
            "rebound_sweep_idx": rebound_sweep_idx,
            "selection": parse_ft_desc(select_representative_rebound_sweep),
        }
    )
    return rebound_latency, rebound_latency_info


@ephys_feature
def get_sweepset_rebound_avg(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level average rebound feature.

    description: average rebound for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level average rebound feature.
    """
    rebound_avg, rebound_avg_info = ephys_feature_init()
    rebound_sweep_idx = select_representative_rebound_sweep(sweep_fts)
    rebound_avg = sweep_fts["rebound_avg"].iloc[rebound_sweep_idx]
    rebound_avg_info.update(
        {
            "rebound_sweep_idx": rebound_sweep_idx,
            "selection": parse_ft_desc(select_representative_rebound_sweep),
        }
    )
    return rebound_avg, rebound_avg_info


# num spikes
def select_representative_spiking_sweep(sweep_fts: DataFrame) -> int:
    """Select representative sweep from which the spiking related features are extracted.

    description: Highest non wild trace (wildness == cell dying).

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        int: Index of the sweep from which the rebound features are extracted.
    """
    # TODO: should I implement a better selection protocol / criterion!?
    num_spikes = sweep_fts["num_ap"]
    wildness = sweep_fts["wildness"]
    return num_spikes[
        wildness.isna()
    ].argmax()  # select highest non wild trace (wildness == cell dying)


@ephys_feature
def get_sweepset_num_spikes(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level spike count feature.

    description: number of spikes for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level spike count feature.
    """
    num_spikes, num_spikes_info = ephys_feature_init()
    num_spikes = sweep_fts["num_ap"]
    spike_sweep_idx = select_representative_spiking_sweep(sweep_fts)
    num_spikes = num_spikes.loc[spike_sweep_idx]
    num_spikes_info.update(
        {
            "spike_sweep_idx": spike_sweep_idx,
            "selection": parse_ft_desc(select_representative_spiking_sweep),
        }
    )
    return num_spikes, num_spikes_info


@ephys_feature
def get_sweepset_ap_freq(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level spike rate feature.

    description: spike rate for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level spike rate feature.
    """
    ap_freq, ap_freq_info = ephys_feature_init()
    ap_freq = sweep_fts["ap_freq"]
    spike_sweep_idx = select_representative_spiking_sweep(sweep_fts)
    ap_freq = ap_freq.loc[spike_sweep_idx]
    ap_freq_info.update(
        {
            "spike_sweep_idx": spike_sweep_idx,
            "selection": parse_ft_desc(select_representative_spiking_sweep),
        }
    )
    return ap_freq, ap_freq_info


# wildness
@ephys_feature
def get_sweepset_wildness(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level wildness feature.

    description: wildness for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level wildness feature.
    """
    wildness, wildness_info = ephys_feature_init()
    wildness = sweep_fts["wildness"].max()
    max_wild_sweep_idx = sweep_fts["wildness"].argmax()
    wildness_info.update(
        {
            "max_wild_sweep_idx": max_wild_sweep_idx,
            "selection": parse_ft_desc(select_representative_spiking_sweep),
        }
    )
    return wildness, wildness_info


# spike frequency adaptation
@ephys_feature
def get_sweepset_spike_freq_adapt(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level spike frequency adaptation feature.

    description: spike frequency adaptation for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level spike frequency adaptation feature.
    """
    spike_freq_adapt, spike_freq_adapt_info = ephys_feature_init()
    spike_sweep_idx = select_representative_spiking_sweep(sweep_fts)
    spike_freq_adapt = sweep_fts["spike_freq_adapt"].loc[spike_sweep_idx]
    spike_freq_adapt_info.update(
        {
            "spike_sweep_idx": spike_sweep_idx,
            "selection": parse_ft_desc(select_representative_spiking_sweep),
        }
    )
    return spike_freq_adapt, spike_freq_adapt_info


# AP Fano factor
@ephys_feature
def get_sweepset_fano_factor(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level fano factor feature.

    description: Fano factor for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level fano factor feature.
    """
    fano_factor, fano_factor_info = ephys_feature_init()
    spike_sweep_idx = select_representative_spiking_sweep(sweep_fts)
    fano_factor = sweep_fts["fano_factor"].loc[spike_sweep_idx]
    fano_factor_info.update(
        {
            "spike_sweep_idx": spike_sweep_idx,
            "selection": parse_ft_desc(select_representative_spiking_sweep),
        }
    )
    return fano_factor, fano_factor_info


@ephys_feature
def get_sweepset_ap_fano_factor(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level ap fano factor feature.

    description: AP Fano factor for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level ap fano factor feature.
    """
    ap_fano_factor, ap_fano_factor_info = ephys_feature_init()
    spike_sweep_idx = select_representative_spiking_sweep(sweep_fts)
    ap_fano_factor = sweep_fts["AP_fano_factor"].loc[spike_sweep_idx]
    ap_fano_factor_info.update(
        {
            "spike_sweep_idx": spike_sweep_idx,
            "selection": parse_ft_desc(select_representative_spiking_sweep),
        }
    )
    return ap_fano_factor, ap_fano_factor_info


@ephys_feature
def get_sweepset_cv(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level coeffficent of variation feature.

    description: CV for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level coeffficent of variation feature.
    """
    cv, cv_info = ephys_feature_init()
    spike_sweep_idx = select_representative_spiking_sweep(sweep_fts)
    cv = sweep_fts["cv"].loc[spike_sweep_idx]
    cv_info.update(
        {
            "spike_sweep_idx": spike_sweep_idx,
            "selection": parse_ft_desc(select_representative_spiking_sweep),
        }
    )
    return cv, cv_info


@ephys_feature
def get_sweepset_ap_cv(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level AP coefficient of variation feature.

    description: AP CV for a representative sweep.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level AP coefficient of variation feature.
    """
    ap_cv, ap_cv_info = ephys_feature_init()
    spike_sweep_idx = select_representative_spiking_sweep(sweep_fts)
    ap_cv = sweep_fts["AP_cv"].loc[spike_sweep_idx]
    ap_cv_info.update(
        {
            "spike_sweep_idx": spike_sweep_idx,
            "selection": parse_ft_desc(select_representative_spiking_sweep),
        }
    )
    return ap_cv, ap_cv_info


# burstiness
@ephys_feature
def get_sweepset_burstiness(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level burstiness feature.

    description: median burstiness for the first 5 "bursty" traces.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level burstiness feature.
    """
    median_burstiness, burstiness_info = ephys_feature_init()
    burstiness = sweep_fts["burstiness"]
    burstiness[burstiness < 0] = float("nan")  # don't consider negative burstiness
    burstiness = burstiness[~burstiness.isna()]
    median_burstiness = burstiness.iloc[
        :5
    ].median()  # consider first 5 non-nan traces at most
    burstiness_info.update({"burstiness": burstiness})
    return median_burstiness, burstiness_info


# adaptation index
@ephys_feature
def get_sweepset_isi_adapt(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level Inter spike interval adaptation feature.

    description: median of the ISI adaptation of the first 5 traces
    that show adaptation.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level Inter spike interval adaptation feature.
    """
    isi_adapt_median, isi_adapt_info = ephys_feature_init()
    isi_adapt = sweep_fts["isi_adapt"]
    isi_adapt_median = (
        isi_adapt[~isi_adapt.isna()].iloc[:5].median()
    )  # consider first 5 non-nan traces at most
    isi_adapt_info.update({"isi_adapt": isi_adapt})
    return isi_adapt_median, isi_adapt_info


@ephys_feature
def get_sweepset_isi_adapt_avg(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level average inter spike interval adaptation feature.

    description: median of the ISI adaptation average of the first 5 traces
    that show adaptation.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level average inter spike interval adaptation feature.
    """
    isi_adapt_avg_median, isi_adapt_avg_info = ephys_feature_init()
    isi_adapt_avg = sweep_fts["isi_adapt_average"]
    isi_adapt_avg_median = (
        isi_adapt_avg[~isi_adapt_avg.isna()].iloc[:5].median()
    )  # consider first 5 non-nan traces at most
    isi_adapt_avg_info.update({"isi_adapt_avg": isi_adapt_avg})
    return isi_adapt_avg_median, isi_adapt_avg_info


@ephys_feature
def get_sweepset_ap_amp_adapt(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level AP amplitude adaptation feature.

    description: median of the AP amplitude adaptation of the first 5 traces
    that show adaptation.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level AP amplitude adaptation feature.
    """
    ap_amp_adapt_median, ap_amp_adapt_info = ephys_feature_init()
    ap_amp_adapt = sweep_fts["AP_amp_adapt"]
    ap_amp_adapt_median = (
        ap_amp_adapt[~ap_amp_adapt.isna()].iloc[:5].median()
    )  # consider first 5 non-nan traces at most
    ap_amp_adapt_info.update(
        {"ap_amp_adapt": ap_amp_adapt_median, "ap_amp_adapt": ap_amp_adapt}
    )
    return ap_amp_adapt_median, ap_amp_adapt_info


@ephys_feature
def get_sweepset_ap_amp_adapt_avg(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level average AP amplitude adaptation feature.

    description: median of the AP amplitude adaptation average of the first 5
    traces that show adaptation.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level average AP amplitude adaptation feature.
    """
    ap_amp_adapt_avg_median, ap_amp_adapt_avg_info = ephys_feature_init()
    ap_amp_adapt_avg = sweep_fts["AP_amp_adapt_average"]
    ap_amp_adapt_avg_median = (
        ap_amp_adapt_avg[~ap_amp_adapt_avg.isna()].iloc[:5].median()
    )  # consider first 5 non-nan traces at most
    ap_amp_adapt_avg_info.update(
        {
            "ap_amp_adapt_avg": ap_amp_adapt_avg,
        }
    )
    return ap_amp_adapt_avg_median, ap_amp_adapt_avg_info


# latency
@ephys_feature
def get_sweepset_latency(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level latency feature.

    description: latency of the first depolarization trace that contains spikes in ms.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level latency feature.
    """
    selected_latency, latency_info = ephys_feature_init()
    is_depol = sweep_fts["stim_amp"] > 0
    latency = sweep_fts["latency"]
    selected_latency = latency[is_depol & ~latency.isna()].iloc[0] * 1000  # ms
    latency_info.update({"latency": latency * 1000})
    return selected_latency, latency_info


def select_representative_ap_sweep(sweep_fts: DataFrame) -> int:
    """Select representative ap in a sweep from which the AP features are used.

    description: First depolarization trace that contains spikes.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        int: Index of the sweep from which the rebound features are extracted.
    """
    # TODO: implement procedure to select representative AP in each trace for the following AP features
    is_depol = sweep_fts["stim_amp"] > 0
    has_spikes = sweep_fts["num_ap"] > 0
    return is_depol.index[is_depol & has_spikes][0]


# ahp
@ephys_feature
def get_sweepset_ahp(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level Afterhyperpolarization feature.

    description: AHP (fast_trough_v - threshold_v) of a representative spike of
    a representative depolarization trace.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level Afterhyperpolarization feature.
    """
    ahp_selected, ahp_info = ephys_feature_init()
    ahp = sweep_fts["ahp"]
    sweep_ap_index = select_representative_ap_sweep(sweep_fts)
    ahp_selected = ahp.loc[sweep_ap_index]  # fast_trough_v - threshold_v
    ahp_info.update(
        {
            "ahp": ahp,
            "sweep_ap_index": sweep_ap_index,
            "selection": parse_ft_desc(select_representative_ap_sweep),
        }
    )
    return ahp_selected, ahp_info


# adp
@ephys_feature
def get_sweepset_adp(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level Afterdepolarization feature.

    description: ADP (adp_v - fast_trough_v) of a representative spike of a
    representative depolarization trace.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level Afterdepolarization feature.
    """
    adp_selected, adp_info = ephys_feature_init()
    adp = sweep_fts["adp"]
    sweep_ap_index = select_representative_ap_sweep(sweep_fts)
    adp_selected = adp.loc[sweep_ap_index]  # adp_v - fast_trough_v
    adp_info.update(
        {
            "adp": adp,
            "sweep_ap_index": sweep_ap_index,
            "selection": parse_ft_desc(select_representative_ap_sweep),
        }
    )
    return adp_selected, adp_info


# AP features
@ephys_feature
def get_sweepset_ap_thresh(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level AP threshold feature.

    description: AP threshold of a representative spike of a representative depolarization trace.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level AP threshold feature.
    """
    ap_thresh_selected, ap_thresh_info = ephys_feature_init()
    ap_thresh = sweep_fts["ap_thresh"]
    sweep_ap_index = select_representative_ap_sweep(sweep_fts)
    ap_thresh_selected = ap_thresh.loc[sweep_ap_index]
    ap_thresh_info.update(
        {
            "ap_thresh": ap_thresh,
            "sweep_ap_index": sweep_ap_index,
            "selection": parse_ft_desc(select_representative_ap_sweep),
        }
    )
    return ap_thresh_selected, ap_thresh_info


@ephys_feature
def get_sweepset_ap_amp(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level AP amplitude feature.

    description: AP amplitude of a representative spike of a representative
    depolarization trace.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level AP amplitude feature.
    """
    ap_amp_selected, ap_amp_info = ephys_feature_init()
    ap_amp = sweep_fts["ap_amp"]
    sweep_ap_index = select_representative_ap_sweep(sweep_fts)
    ap_amp_selected = ap_amp.loc[sweep_ap_index]
    ap_amp_info.update(
        {
            "ap_amp": ap_amp,
            "sweep_ap_index": sweep_ap_index,
            "selection": parse_ft_desc(select_representative_ap_sweep),
        }
    )
    return ap_amp_selected, ap_amp_info


@ephys_feature
def get_sweepset_ap_width(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level AP width feature.

    description: AP width of a representative spike of a representative
    depolarization trace.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level AP width feature.
    """
    ap_width_selected, ap_width_info = ephys_feature_init()
    ap_width = sweep_fts["ap_width"]
    sweep_ap_index = select_representative_ap_sweep(sweep_fts)
    ap_width_selected = ap_width.loc[sweep_ap_index]
    ap_width_info.update(
        {
            "ap_width": ap_width,
            "sweep_ap_index": sweep_ap_index,
            "selection": parse_ft_desc(select_representative_ap_sweep),
        }
    )
    return ap_width_selected, ap_width_info


@ephys_feature
def get_sweepset_ap_peak(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level AP peak feature.

    description: Peak of AP of a representative spike of a representative
    depolarization trace.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level AP peak feature.
    """
    ap_peak_selected, ap_peak_info = ephys_feature_init()
    ap_peak = sweep_fts["ap_peak"]
    sweep_ap_index = select_representative_ap_sweep(sweep_fts)
    ap_peak_selected = ap_peak.loc[sweep_ap_index]
    ap_peak_info.update(
        {
            "ap_peak": ap_peak,
            "sweep_ap_index": sweep_ap_index,
            "selection": parse_ft_desc(select_representative_ap_sweep),
        }
    )
    return ap_peak_selected, ap_peak_info


@ephys_feature
def get_sweepset_ap_trough(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level AP trough feature.

    description: AP trough of a representative spike of a representative
    depolarization trace.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level AP trough feature.
    """
    ap_trough_selected, ap_trough_info = ephys_feature_init()
    ap_trough = sweep_fts["ap_trough"]
    sweep_ap_index = select_representative_ap_sweep(sweep_fts)
    ap_trough_selected = ap_trough.loc[sweep_ap_index]
    ap_trough_info.update(
        {
            "ap_trough": ap_trough,
            "sweep_ap_index": sweep_ap_index,
            "selection": parse_ft_desc(select_representative_ap_sweep),
        }
    )
    return ap_trough_selected, ap_trough_info


@ephys_feature
def get_sweepset_ap_udr(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level AP upstroke to downstroke ratio feature.

    description: AP upstroke-downstroke ratio of a representative spike of a
    representative depolarization trace.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level AP upstroke to downstroke ratio feature.
    """
    udr_selected, ap_udr_info = ephys_feature_init()
    udr = sweep_fts["udr"]
    sweep_ap_index = select_representative_ap_sweep(sweep_fts)
    udr_selected = udr.loc[sweep_ap_index]
    ap_udr_info.update(
        {
            "udr": udr,
            "sweep_ap_index": sweep_ap_index,
            "selection": parse_ft_desc(select_representative_ap_sweep),
        }
    )
    return udr_selected, ap_udr_info


# rheobase
@ephys_feature
def get_sweepset_dfdi(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level df/di feature.

    description: df/di (slope) for the first 5 depolarization traces that
    contain spikes. If there are more than 2 duplicates in the number of spikes
    -> nan. Frequncy = num_spikes / T_stim.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level df/di feature.
    """
    dfdi, dfdi_info = ephys_feature_init()
    is_depol = sweep_fts["stim_amp"] > 0
    i, n_spikes = sweep_fts[is_depol][["stim_amp", "num_ap"]].to_numpy().T
    has_spikes = ~np.isnan(n_spikes)
    if np.sum(has_spikes) > 4 and len(np.unique(n_spikes[:5])) > 3:
        onset, end = sweep_fts[["stim_onset", "stim_end"]].iloc[0]
        f = n_spikes / (end - onset)
        i_s = i[has_spikes][:5]
        f_s = f[has_spikes][:5]

        ransac.fit(i_s.reshape(-1, 1), f_s.reshape(-1, 1))
        dfdi = ransac.coef_[0, 0]
        f_intercept = ransac.intercept_[0]
        dfdi_info.update({"i_fit": i_s, "f_fit": f_s, "f_intercept": f_intercept})
    return dfdi, dfdi_info


@ephys_feature
def get_sweepset_rheobase(sweep_fts: DataFrame) -> Union[Dict, float]:
    """Extract sweep set level rheobase feature.

    description: rheobase current in pA. If df/di is nan, rheobase is the first
    depolarization trace that contains spikes. Otherwise, rheobase is the
    current where the fitted line crosses the current axis. The fitted intercept
    must lie between the stimulus of the first depolarization trace that contains
    spikes and the last depolarization that does not.

    Args:
        sweep_fts (DataFrame): DataFrame containing the sweep features.

    Returns:
        Tuple[float, Dict]: sweep set level rheobase feature.
    """
    dc_offset = sweep_fts["dc_offset"].iloc[0]
    rheobase, rheobase_info = ephys_feature_init()
    is_depol = sweep_fts["stim_amp"] > 0
    i, n_spikes = sweep_fts[is_depol][["stim_amp", "num_ap"]].to_numpy().T
    has_spikes = ~np.isnan(n_spikes)
    i_sub = i[~has_spikes][0]  # last stim < spike threshold
    i_sup = i[has_spikes][0]  # first stim > spike threshold
    dfdi = strip_info(get_sweepset_dfdi(sweep_fts))

    if not np.isnan(dfdi):
        rheobase = float(ransac.predict(np.array([[0]]))) / dfdi

        if rheobase < i_sub or rheobase > i_sup:
            rheobase = i_sup
    else:
        rheobase = i_sup
    rheobase -= dc_offset
    rheobase_info.update(
        {"i_sub": i_sub, "i_sup": i_sup, "dfdi": dfdi, "dc_offset": dc_offset}
    )
    return rheobase, rheobase_info


def get_fp_sweepset_ft_dict(return_ft_info=False):
    fp_ft_dict = {
        "tau": get_sweepset_time_constant,
        "r_input": get_sweepset_r_input,
        "v_rest": get_sweepset_v_rest,
        "slow_hyperpolarization": get_sweepset_slow_hyperpolarization,
        "sag": get_sweepset_sag,
        "sag_ratio": get_sweepset_sag_ratio,
        "sag_fraction": get_sweepset_sag_fraction,
        "sag_area": get_sweepset_sag_area,
        "sag_time": get_sweepset_sag_time,
        "rebound": get_sweepset_rebound,
        "rebound_ratio": get_sweepset_rebound_ratio,
        "rebound_area": get_sweepset_rebound_area,
        "rebound_latency": get_sweepset_rebound_latency,
        "rebound_avg": get_sweepset_rebound_avg,
        "num_ap": get_sweepset_num_spikes,
        "ap_freq": get_sweepset_ap_freq,
        "wildness": get_sweepset_wildness,
        "spike_freq_adapt": get_sweepset_spike_freq_adapt,
        "fano_factor": get_sweepset_fano_factor,
        "ap_fano_factor": get_sweepset_ap_fano_factor,
        "cv": get_sweepset_cv,
        "ap_cv": get_sweepset_ap_cv,
        "burstiness": get_sweepset_burstiness,
        "isi_adapt": get_sweepset_isi_adapt,
        "isi_adapt_avg": get_sweepset_isi_adapt_avg,
        "ap_amp_adapt": get_sweepset_ap_amp_adapt,
        "ap_amp_adapt_avg": get_sweepset_ap_amp_adapt_avg,
        "latency": get_sweepset_latency,
        "ahp": get_sweepset_ahp,
        "adp": get_sweepset_adp,
        "ap_thresh": get_sweepset_ap_thresh,
        "ap_amp": get_sweepset_ap_amp,
        "ap_width": get_sweepset_ap_width,
        "ap_peak": get_sweepset_ap_peak,
        "ap_trough": get_sweepset_ap_trough,
        "udr": get_sweepset_ap_udr,
        "dfdi": get_sweepset_dfdi,
        "rheobase": get_sweepset_rheobase,
    }

    if return_ft_info:
        return {k: partial(v, return_ft_info=True) for k, v in fp_ft_dict.items()}
    return fp_ft_dict
