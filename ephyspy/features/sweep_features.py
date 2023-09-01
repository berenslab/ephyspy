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
from typing import Callable, Optional

import numpy as np
from matplotlib.pyplot import Axes
from numpy import ndarray
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit

import ephyspy.allen_sdk.ephys_features as ft
from ephyspy.features.base import SweepFeature
from ephyspy.features.utils import (
    FeatureError,
    get_sweep_burst_metrics,
    get_sweep_sag_idxs,
    has_rebound,
    has_spikes,
    has_stimulus,
    is_hyperpol,
    median_idx,
    where_stimulus,
)
from ephyspy.plot import plot_ap_amp, plot_isi, plot_spike_feature
from ephyspy.utils import (
    parse_desc,
    relabel_line,
    unpack,
    where_between,
)


def available_sweep_features(compute_at_init=False, store_diagnostics=False):
    features = {
        "stim_amp": Stim_amp,
        "stim_onset": Stim_onset,
        "stim_end": Stim_end,
        "num_ap": Num_AP,
        "ap_freq": AP_freq,
        "ap_latency": AP_latency,
        "v_baseline": V_baseline,
        "v_deflect": V_deflect,
        "tau": Tau,
        "ap_freq_adapt": AP_freq_adapt,
        "ap_amp_slope": AP_amp_slope,
        "isi_ff": ISI_FF,
        "isi_cv": ISI_CV,
        "ap_ff": AP_FF,
        "ap_cv": AP_CV,
        "isi_adapt": ISI_adapt,
        "isi_adapt_avg": ISI_adapt_avg,
        "ap_amp_adapt": AP_amp_adapt,
        "ap_amp_adapt_avg": AP_amp_adapt_avg,
        "r_input": R_input,
        "sag": Sag,
        "v_sag": V_sag,
        "v_steady": V_steady,
        "sag_ratio": Sag_ratio,
        "sag_fraction": Sag_fraction,
        "sag_area": Sag_area,
        "sag_time": Sag_time,
        "v_plateau": V_plateau,
        "rebound": Rebound,
        "rebound_aps": Rebound_APs,
        "rebound_area": Rebound_area,
        "rebound_latency": Rebound_latency,
        "rebound_avg": Rebound_avg,
        "v_rest": V_rest,
        "num_bursts": Num_bursts,
        "burstiness": Burstiness,
        "wildness": Wildness,
        "ap_adp": AP_ADP,
        "ap_ahp": AP_AHP,
        "ap_thresh": AP_thresh,
        "ap_amp": AP_amp,
        "ap_width": AP_width,
        "ap_peak": AP_peak,
        "ap_trough": AP_trough,
        "ap_udr": AP_UDR,
    }
    if any((compute_at_init, store_diagnostics)):
        return {
            k: lambda *args, **kwargs: v(
                *args,
                compute_at_init=compute_at_init,
                store_diagnostics=store_diagnostics,
                **kwargs,
            )
            for k, v in features.items()
        }
    else:
        return features


class NullSweepFeature(SweepFeature):
    """Dummy sweep level feature.

    Dummy feature that can be used as a placeholder to compute sweepset level
    features using `SweepSetFeature` if no sweep level feature for it is available.

    depends on: /.
    description: Only the corresponding sweepset level feature exsits.
    units: /."""

    def __init__(self, data=None, compute_at_init=True, name=None):
        super().__init__(data, compute_at_init, name=name)

    def _compute(self, recompute=False, store_diagnostics=True):
        return


class Stim_amp(SweepFeature):
    """Extract sweep level stimulus ampltiude feature.
    depends on: /.
    description: maximum amplitude of stimulus.
    units: pA."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        idx = np.argmax(abs(self.data.i).T, axis=0)

        if store_diagnostics:
            self._update_diagnostics({"idx": idx, "t": self.data.t[idx]})
        return self.data.i[idx]

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        i_amp = self.value
        t = self.diagnostics["t"]
        ax.plot(
            t,
            i_amp,
            "x",
            label=self.name,
            **kwargs,
        )
        return ax


class Stim_onset(SweepFeature):
    """Extract sweep level stimulus onset feature.

    depends on: /.
    description: time of stimulus onset.
    units: s."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        stim_onset = float("nan")
        if has_stimulus(self.data):
            where_stim = where_stimulus(self.data)
            stim_onset = self.data.t[where_stim][0]
            i_onset = self.data.i[where_stim][0]
            idx_onset = np.arange(len(where_stim))[where_stim][0]
            if store_diagnostics:
                self._update_diagnostics(
                    {
                        "i_onset": i_onset,
                        "where_stim": where_stim,
                        "idx_onset": idx_onset,
                    }
                )
        return stim_onset

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        onset = self.value
        i_onset = self.diagnostics["i_onset"]
        ax.plot(onset, i_onset, "x", label=self.name, **kwargs)
        return ax


class Stim_end(SweepFeature):
    """Extract sweep level stimulus end feature.

    depends on: /.
    description: time of stimulus end.
    units: s."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        stim_end = float("nan")
        if has_stimulus(self.data):
            where_stim = where_stimulus(self.data)
            stim_end = self.data.t[where_stim][-1]
            i_end = self.data.i[where_stim][-1]
            idx_end = np.arange(len(where_stim))[where_stim][-1]
            if store_diagnostics:
                self._update_diagnostics(
                    {"i_end": i_end, "where_stim": where_stim, "idx_end": idx_end}
                )
        return stim_end

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        end = self.value
        i_end = self.diagnostics["i_end"]
        ax.plot(end, i_end, "x", label=self.name, **kwargs)
        return ax


class Num_AP(SweepFeature):
    """Extract sweep level spike count feature.

    depends on: stim_onset, stim_end.
    description: # peaks during stimulus.
    units: /."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        peak_t = self.lookup_spike_feature("peak_t", recompute=recompute)
        onset = self.lookup_sweep_feature("stim_onset")
        end = self.lookup_sweep_feature("stim_end")
        stim_window = where_between(peak_t, onset, end)

        peak_i = self.lookup_spike_feature("peak_index")[stim_window]
        num_ap = len(peak_i)

        if num_ap <= 0:
            num_ap = float("nan")

        if store_diagnostics:
            peak_t = peak_t[stim_window]
            peak_v = self.lookup_spike_feature("peak_v")[stim_window]
            self._update_diagnostics(
                {
                    "peak_i": peak_i,
                    "peak_t": peak_t,
                    "peak_v": peak_v,
                }
            )
        return num_ap

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        peak_t, peak_v = unpack(self.diagnostics, ["peak_t", "peak_v"])
        ax.plot(peak_t, peak_v, "x", label=self.name, **kwargs)
        return ax


class AP_freq(SweepFeature):
    """Extract sweep level spike rate feature.

    depends on: numap.
    description: # peaks during stimulus / stimulus duration.
    units: Hz."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        num_ap = self.lookup_sweep_feature("num_ap", recompute=recompute)
        onset = self.lookup_sweep_feature("stim_onset", recompute=recompute)
        end = self.lookup_sweep_feature("stim_end", recompute=recompute)

        ap_freq = num_ap / (end - onset)

        if store_diagnostics:
            self._update_diagnostics(
                {"ap_freq": ap_freq, "num_ap": num_ap, "onset": onset, "end": end}
            )
        return ap_freq

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        num_ap = self.lookup_sweep_feature("num_ap", return_value=False)
        ax = num_ap.plot(ax=ax, **kwargs)
        return ax


class AP_latency(SweepFeature):
    """Extract sweep level ap_latency feature.

    depends on: stim_onset.
    description: time of first spike after stimulus onset.
    units: s."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        ap_latency = float("nan")
        if has_stimulus(self.data):
            onset = self.lookup_sweep_feature("stim_onset", recompute=recompute)
            end = self.lookup_sweep_feature("stim_end", recompute=recompute)
            thresh_t = self.lookup_spike_feature("threshold_t", recompute=recompute)
            thresholds = self.lookup_spike_feature("threshold_v", recompute=recompute)
            stim_window = where_between(thresh_t, onset, end)

            thresh_t_stim = thresh_t[stim_window]

            if len(thresh_t_stim) > 0:
                v_first_spike = thresholds[stim_window][0]
                t_first_spike = thresh_t_stim[0]
                ap_latency = t_first_spike - onset

                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "onset": onset,
                            "end": end,
                            "spike_times_during_stim": thresh_t_stim,
                            "t_first": t_first_spike,
                            "v_first": v_first_spike,
                        }
                    )
        return ap_latency

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        v_first, t_first, onset = unpack(
            self.diagnostics, ["v_first", "t_first", "onset"]
        )
        ax.hlines(v_first, onset, t_first, label=self.name, **kwargs)
        return ax


class V_baseline(SweepFeature):
    """Extract sweep level baseline voltage feature.

    depends on: stim_onset.
    description: average voltage in baseline_interval (in s) before stimulus onset.
    units: mV."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_baseline_avg = float("nan")
        onset = self.lookup_sweep_feature("stim_onset", recompute=recompute)
        if np.isnan(onset):
            baseline_interval = float("nan")
            where_baseline = np.ones_like(self.data.t, dtype=bool)  # for I=0pA
        else:
            baseline_interval = self.data.baseline_interval
            where_baseline = where_between(
                self.data.t, onset - baseline_interval, onset
            )
        t_baseline = self.data.t[where_baseline]
        v_baseline = self.data.v[where_baseline]
        v_baseline_avg = np.mean(v_baseline)
        # v_baseline_avg = sweep._get_baseline_voltage() # bad since start is set to t[0]
        if store_diagnostics:
            self._update_diagnostics(
                {
                    "where_baseline": where_baseline,
                    "t_baseline": t_baseline,
                    "v_baseline": v_baseline,
                    "baseline_interval": baseline_interval,
                    "stim_onset": onset,
                }
            )
        return v_baseline_avg

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        t_base, v_base = unpack(self.diagnostics, ["t_baseline", "v_baseline"])
        ax.plot(t_base, v_base, label=self.name + " interval", **kwargs)
        ax.axhline(self.value, ls="--", label=self.name, **kwargs)
        return ax


class V_deflect(SweepFeature):
    """Extract sweep level voltage deflection feature.

    depends on: stim_end.
    description: average voltage during last 100 ms of stimulus.
    units: mV."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_deflect_avg = float("nan")
        if has_stimulus(self.data) and is_hyperpol(self.data):
            # v_deflect_avg = self.data.voltage_deflection()[0]
            end = self.lookup_sweep_feature("stim_end", recompute=recompute)
            v_deflect_avg = ft.average_voltage(
                self.data.v, self.data.t, start=end - 0.1, end=end
            )
            idx_deflect = np.where(where_between(self.data.t, end - 0.1, end))[0]
            t_deflect = self.data.t[idx_deflect]
            v_deflect = self.data.v[idx_deflect]

            if store_diagnostics:
                self._update_diagnostics(
                    {
                        "idx_deflect": idx_deflect,
                        "t_deflect": t_deflect,
                        "v_deflect": v_deflect,
                    }
                )
        return v_deflect_avg

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        t_deflect, v_deflect = unpack(self.diagnostics, ["t_deflect", "v_deflect"])
        t_bar = np.ones_like(t_deflect) * self.value
        ax.plot(t_deflect, v_deflect, label=self.name + " interval", **kwargs)
        ax.plot(t_deflect, t_bar, ls="--", label=self.name, **kwargs)
        return ax


class Tau(SweepFeature):
    """Extract sweep level time constant feature.

    depends on: v_baseline, stim_onset.
    description: time constant of exponential fit to voltage deflection.
    units: s."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        tau = float("nan")
        if is_hyperpol(self.data):
            """The following code block is copied and adapted from sweep.estimate_time_constant()."""
            v_peak, peak_index = self.data.voltage_deflection("min")
            v_baseline = self.lookup_sweep_feature("v_baseline", recompute=recompute)
            if 5 < v_baseline - v_peak:
                stim_onset = self.lookup_sweep_feature(
                    "stim_onset", recompute=recompute
                )
                onset_idx = ft.find_time_index(self.data.t, stim_onset)

                frac = 0.1
                search_result = np.flatnonzero(
                    self.data.v[onset_idx:] <= frac * (v_peak - v_baseline) + v_baseline
                )
                if not search_result.size:
                    raise FeatureError(
                        "could not find interval for time constant estimate"
                    )

                fit_start = self.data.t[search_result[0] + onset_idx]
                fit_end = self.data.t[peak_index]

                if self.data.v[peak_index] < -200:
                    warnings.warn(
                        "A DOWNWARD PEAK WAS OBSERVED GOING TO LESS THAN 200 MV!!!"
                    )
                    # Look for another local minimum closer to stimulus onset
                    # We look for a couple of milliseconds after stimulus onset to 50 ms before the downward peak
                    end_index = (onset_idx + 50) + np.argmin(
                        self.data.v[onset_idx + 50 : peak_index - 1250]
                    )
                    fit_end = self.data.t[end_index]
                    fit_start = self.data.t[onset_idx + 50]

                a, inv_tau, y0 = ft.fit_membrane_time_constant(
                    self.data.v, self.data.t, fit_start, fit_end
                )

                tau = 1.0 / inv_tau * 1000
                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "a": a,
                            "inv_tau": inv_tau,
                            "y0": y0,
                            "fit_start": fit_start,
                            "fit_end": fit_end,
                            "equation": "y0 + a * exp(-inv_tau * x)",
                        }
                    )
        return tau

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        y0, a, inv_tau = unpack(self.diagnostics, ["y0", "a", "inv_tau"])
        fit_start, fit_end = unpack(self.diagnostics, ["fit_start", "fit_end"])
        t, v = self.data.t, self.data.v

        y = lambda t: y0 + a * np.exp(-inv_tau * t)

        where_fit = where_between(t, fit_start, fit_end)
        t, v = t[where_fit], v[where_fit]
        t_offset = t[0]
        t_fit = t - t_offset
        ax.plot(t, v, label=self.name + " interval", **kwargs)
        ax.plot(t, y(t_fit), ls="--", color="k", label=self.name + " fit")
        return ax


class AP_freq_adapt(SweepFeature):
    """Extract sweep level spike frequency adaptation feature.

    depends on: stim_onset, stim_end, num_ap.
    description: ratio of spikes in second and first half half of stimulus interval, if there is at least 5 spikes in total.
    units: /."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        ap_freq_adapt = float("nan")
        num_ap = self.lookup_sweep_feature("num_ap", recompute=recompute)
        if num_ap > 5 and has_stimulus(self.data):
            onset = self.lookup_sweep_feature("stim_onset", recompute=recompute)
            end = self.lookup_sweep_feature("stim_end", recompute=recompute)
            t_half = (end - onset) / 2 + onset
            where_1st_half = where_between(self.data.t, onset, t_half)
            where_2nd_half = where_between(self.data.t, t_half, end)
            t_1st_half = self.data.t[where_1st_half]
            t_2nd_half = self.data.t[where_2nd_half]

            peak_t = self.lookup_spike_feature("peak_t", recompute=recompute)
            onset = self.lookup_sweep_feature("stim_onset", recompute=recompute)
            end = self.lookup_sweep_feature("stim_end", recompute=recompute)
            stim_window = where_between(peak_t, onset, end)
            peak_t = peak_t[stim_window]

            spikes_1st_half = peak_t[peak_t < t_half]
            spikes_2nd_half = peak_t[peak_t > t_half]
            num_spikes_1st_half = len(spikes_1st_half)
            num_spikes_2nd_half = len(spikes_2nd_half)
            ap_freq_adapt = num_spikes_2nd_half / num_spikes_1st_half

            if store_diagnostics:
                self._update_diagnostics(
                    {
                        "num_spikes_1st_half": num_spikes_1st_half,
                        "num_spikes_2nd_half": num_spikes_2nd_half,
                        "where_1st_half": where_1st_half,
                        "where_2nd_half": where_2nd_half,
                        "t_1st_half": t_1st_half,
                        "t_2nd_half": t_2nd_half,
                    }
                )
        return ap_freq_adapt

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        num_ap = self.lookup_sweep_feature("num_ap", return_value=False)
        peaks_t, peaks_v = unpack(num_ap.diagnostics, ["peak_t", "peak_v"])
        half1, half2 = unpack(self.diagnostics, ["t_1st_half", "t_2nd_half"])
        for i, (half, m) in enumerate(zip([half1, half2], ["+", "x"])):
            in_half = where_between(peaks_t, *half[[0, -1]])
            ax.plot(peaks_t[in_half], peaks_v[in_half], m, label=f"{i+1}/2", **kwargs)
        return ax


class AP_amp_slope(SweepFeature):
    """Extract sweep level spike count feature.

    depends on: stim_onset, stim_end.
    description: spike amplitude adaptation as the slope of a linear fit v_peak(t_peak)
    during the stimulus interval.
    units: mV/s."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        ap_amp_slope = float("nan")
        onset = self.lookup_sweep_feature("stim_onset")
        end = self.lookup_sweep_feature("stim_end")
        peak_t = self.lookup_spike_feature("peak_t", recompute=recompute)
        peak_v = self.lookup_spike_feature("peak_v", recompute=recompute)
        stim_window = where_between(peak_t, onset, end)

        peak_t = peak_t[stim_window]
        peak_v = peak_v[stim_window]

        if len(peak_v) > 5:
            y = lambda x, m, b: m * x + b
            (m, b), e = curve_fit(y, peak_t, peak_v)

            ap_amp_slope = m
            if store_diagnostics:
                self._update_diagnostics(
                    {
                        "peak_t": peak_t,
                        "peak_v": peak_v,
                        "slope": m,
                        "intercept": b,
                    }
                )
        return ap_amp_slope

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        intercept, slope, peak_t = unpack(
            self.diagnostics, ["intercept", "slope", "peak_t"]
        )
        y = lambda t: intercept + slope * t
        if not np.isnan(self.value):
            ts = peak_t  # or ts = self.data.t
            ax.plot(ts, y(ts), "--", label=self.name, **kwargs)
        return ax


class ISI_FF(SweepFeature):
    """Extract sweep level inter-spike-interval (ISI) Fano factor feature.

    depends on: ISIs.
    description: Var(ISIs) / Mean(ISIs).
    units: s."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        isi_ff = float("nan")
        if has_spikes(self.data):
            isi = self.lookup_spike_feature("isi", recompute=recompute)[1:]
            if len(isi) > 1:
                isi_ff = np.nanvar(isi) / np.nanmean(isi)

                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "isi": isi,
                            "isi_var": np.nanvar(isi),
                            "isi_mean": np.nanmean(isi),
                        }
                    )
        return isi_ff

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        warnings.warn(f" {self.name} plotting not implemented.")
        return ax


class ISI_CV(SweepFeature):
    """Extract sweep level inter-spike-interval (ISI) coefficient of variation (CV) feature.

    depends on: ISIs.
    description: Std(ISIs) / Mean(ISIs).
    units: /."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        isi_cv = float("nan")
        if has_spikes(self.data):
            isi = self.lookup_spike_feature("isi", recompute=recompute)[1:]
            if len(isi) > 1:
                isi_cv = np.nanstd(isi) / np.nanmean(isi)

                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "isi": isi,
                            "isi_std": np.nanstd(isi),
                            "isi_mean": np.nanmean(isi),
                        }
                    )
        return isi_cv

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        warnings.warn(f" {self.name} plotting not implemented.")
        return ax


class AP_FF(SweepFeature):
    """Extract sweep level AP amplitude Fano factor feature.

    depends on: ap_amp.
    description: Var(ap_amp) / Mean(ap_amp).
    units: mV."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        ap_ff = float("nan")
        if has_spikes(self.data):
            ap_amp = self.lookup_spike_feature("ap_amp", recompute=recompute)
            if len(ap_amp) > 1:
                ap_ff = np.nanvar(ap_amp) / np.nanmean(ap_amp)

                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "ap_amp": ap_amp,
                            "ap_amp_var": np.nanvar(ap_amp),
                            "ap_amp_mean": np.nanmean(ap_amp),
                        }
                    )
        return ap_ff

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        warnings.warn(f" {self.name} plotting not implemented.")
        return ax


class AP_CV(SweepFeature):
    """Extract sweep level AP amplitude coefficient of variation (CV) feature.

    depends on: ap_amp.
    description: Std(ap_amp) / Mean(ap_amp).
    units: /."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        ap_cv = float("nan")
        if has_spikes(self.data):
            ap_amp = self.lookup_spike_feature("ap_amp", recompute=recompute)
            if len(ap_amp) > 1:
                ap_cv = np.nanstd(ap_amp) / np.nanmean(ap_amp)

                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "ap_amp": ap_amp,
                            "ap_amp_std": np.nanstd(ap_amp),
                            "ap_amp_mean": np.nanmean(ap_amp),
                        }
                    )
        return ap_cv

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        warnings.warn(f" {self.name} plotting not implemented.")
        return ax


class R_input(SweepFeature):
    """Extract sweep level input resistance feature.

    depends on: stim_amp, v_deflect, v_baseline.
    description: sweep level input resistance as (v_deflect - v_baseline / current).
    Should not be used for cell level feature.
    units: MOhm."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        r_input = float("nan")
        if is_hyperpol(self.data):
            stim_amp = self.lookup_sweep_feature("stim_amp", recompute=recompute)
            v_deflect = self.lookup_sweep_feature("v_deflect", recompute=recompute)
            v_baseline = self.lookup_sweep_feature("v_baseline", recompute=recompute)
            r_input = np.abs((v_deflect - v_baseline) * 1000 / stim_amp)

            if store_diagnostics:
                self._update_diagnostics(
                    {
                        "v_baseline": v_baseline,
                        "v_deflect": v_deflect,
                        "stim_amp": stim_amp,
                    }
                )
        return r_input

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        warnings.warn(f" {self.name} plotting not implemented.")
        return ax


class V_sag(SweepFeature):
    """Extract sweep level sag voltage feature.

    depends on: v_deflect, v_baseline.
    description: Average voltage around max deflection.
    units: mV."""

    def __init__(self, data=None, compute_at_init=True, peak_width=0.005):
        super().__init__(data, compute_at_init=False)
        self.peak_width = peak_width
        if compute_at_init and data is not None:  # because of peak_width
            self.get_value()

    def _compute(self, recompute=False, store_diagnostics=True):
        v_sag = float("nan")
        if is_hyperpol(self.data):
            where_sag = get_sweep_sag_idxs(self, store_diagnostics=store_diagnostics)
            if np.sum(where_sag) > 10:  # TODO: what should be min sag duration!?
                # The following can also be found in sweep.estimate_sag()
                v_deflect, idx_deflect = self.data.voltage_deflection("min")

                if self.data.v[idx_deflect] < -200:
                    warnings.warn("Downward peak < 200 mV")
                    # Look for another local minimum closer to stimulus onset
                    idx_deflect -= ft.find_time_index(
                        self.data.t, 0.12
                    ) - ft.find_time_index(self.data.t, 0.1)

                t_deflect = self.data.t[idx_deflect]
                stim_onset = self.lookup_sweep_feature(
                    "stim_onset", recompute=recompute
                )
                stim_end = self.lookup_sweep_feature("stim_end", recompute=recompute)

                if (  # TODO: Check if stricter criterion is sensible, i.e. t_deflect < t_half_stim
                    stim_onset < t_deflect < stim_end
                ):  # in some rare cases this is not the case
                    start = t_deflect - self.peak_width / 2.0
                    end = t_deflect + self.peak_width / 2.0
                    v_sag = ft.average_voltage(
                        self.data.v,
                        self.data.t,
                        start=start,
                        end=end,
                    )

                    if store_diagnostics:
                        self._update_diagnostics(
                            {
                                "where_sag": where_sag,
                                "v_deflect": v_deflect,
                                "idx_deflect": idx_deflect,
                                "t_deflect": t_deflect,
                                "start": start,
                                "end": end,
                            }
                        )
        return v_sag

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        start, end = unpack(self.diagnostics, ["start", "end"])
        ax.hlines(self.value, start, end, ls="--", label=self.name, **kwargs)
        return ax


class Sag(SweepFeature):
    """Extract sweep level sag feature.

    depends on: v_sag.
    description: magnitude of the depolarization peak.
    units: mV."""

    def __init__(self, data=None, compute_at_init=True, peak_width=0.005):
        self.peak_width = peak_width
        super().__init__(data, compute_at_init=False)
        if compute_at_init and data is not None:  # because of peak_width
            self.get_value()

    def _compute(self, recompute=False, store_diagnostics=True):
        sag = float("nan")
        if is_hyperpol(self.data):
            where_sag = get_sweep_sag_idxs(self, store_diagnostics=store_diagnostics)
            if np.sum(where_sag) > 10:  # TODO: what should be min sag duration!?
                v_sag = self.lookup_sweep_feature("v_sag", recompute=recompute)
                v_baseline = self.lookup_sweep_feature(
                    "v_baseline", recompute=recompute
                )
                sag = v_sag - v_baseline

                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "v_sag": v_sag,
                            "where_sag": where_sag,
                            "v_baseline": v_baseline,
                        }
                    )
        return sag

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        v_baseline, v_sag = unpack(self.diagnostics, ["v_baseline", "v_sag"])
        sag_voltage_ft = self.lookup_sweep_feature("v_sag", return_value=False)
        t_deflect = unpack(sag_voltage_ft.diagnostics, ["t_deflect"])

        ax.vlines(t_deflect, v_baseline, v_sag, label="sag", **kwargs)
        return ax


class V_steady(SweepFeature):
    """Extract sweep level hyperpol steady state feature.

    depends on: stim_end.
    description: hyperpol steady state voltage.
    units: /."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_steady = float("nan")
        if is_hyperpol(self.data):
            stim_end = self.lookup_sweep_feature("stim_end", recompute=recompute)
            start = stim_end - self.data.baseline_interval
            v_steady = ft.average_voltage(
                self.data.v,
                self.data.t,
                start=start,
                end=stim_end,
            )

            if store_diagnostics:
                self._update_diagnostics({"start": start, "end": stim_end})

        return v_steady

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        start, end = unpack(self.diagnostics, ["start", "end"])
        ax.hlines(self.value, start, end, ls="--", label=self.name, **kwargs)
        return ax


class Sag_fraction(SweepFeature):
    """Extract sweep level sag fraction feature.

    depends on: /.
    description: fraction that membrane potential relaxes back to baseline.
    units: /."""

    def __init__(self, data=None, compute_at_init=True, peak_width=0.005):
        self.peak_width = peak_width
        super().__init__(data, compute_at_init=False)
        if compute_at_init and data is not None:  # because of peak_width
            self.get_value()

    def _compute(self, recompute=False, store_diagnostics=True):
        sag_fraction = float("nan")
        if is_hyperpol(self.data):
            where_sag = get_sweep_sag_idxs(self, store_diagnostics=store_diagnostics)
            if np.sum(where_sag) > 10:  # TODO: what should be min sag duration!?
                sag = self.lookup_sweep_feature("sag", recompute=recompute)
                v_sag = self.lookup_sweep_feature("v_sag", recompute=recompute)
                v_steady = self.lookup_sweep_feature("v_steady", recompute=recompute)

                sag_fraction = (v_sag - v_steady) / sag

                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "sag": sag,
                            "v_sag": v_sag,
                            "where_sag": where_sag,
                            "v_steady": v_steady,
                        }
                    )
        return sag_fraction

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        sag_v_ft = self.lookup_sweep_feature("v_sag", return_value=False)
        t_deflect = unpack(sag_v_ft.diagnostics, "t_deflect")
        v_baseline = self.lookup_sweep_feature("v_baseline")
        stim_end = self.lookup_sweep_feature("stim_end")
        v_sag, v_steady = unpack(self.diagnostics, ["v_sag", "v_steady"])

        ax.vlines(t_deflect, v_baseline, v_sag, label="sag", **kwargs)
        ax.vlines(stim_end, v_steady, v_sag, label="v_sag - v_steady", **kwargs)
        return ax


class Sag_ratio(SweepFeature):
    """Extract sweep level sag ratio feature.

    depends on: /.
    description: ratio of steady state voltage decrease to the largest voltage decrease.
    units: /."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        sag_ratio = float("nan")
        if is_hyperpol(self.data):
            where_sag = get_sweep_sag_idxs(self, store_diagnostics=store_diagnostics)
            if np.sum(where_sag) > 10:  # TODO: what should be min sag duration!?
                sag = self.lookup_sweep_feature("sag", recompute=recompute)
                v_steady = self.lookup_sweep_feature("v_steady", recompute=recompute)
                v_baseline = self.lookup_sweep_feature(
                    "v_baseline", recompute=recompute
                )

                sag_ratio = sag / (v_steady - v_baseline)

                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "sag": sag,
                            "where_sag": where_sag,
                            "v_baseline": v_baseline,
                            "v_steady": v_steady,
                        }
                    )
        return sag_ratio

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        sag_v_ft = self.lookup_sweep_feature("v_sag", return_value=False)
        t_deflect = unpack(sag_v_ft.diagnostics, "t_deflect")
        v_sag = self.lookup_sweep_feature("v_sag")
        stim_end = self.lookup_sweep_feature("stim_end")
        v_baseline, v_steady = unpack(self.diagnostics, ["v_baseline", "v_steady"])

        ax.vlines(stim_end, v_steady, v_baseline, label="v_steady - v_base", **kwargs)
        ax.vlines(t_deflect, v_baseline, v_sag, label="sag", **kwargs)
        return ax


class Sag_area(SweepFeature):
    """Extract sweep level sag area feature.

    depends on: v_deflect, stim_onset, stim_end.
    description: area under the sag.
    units: mV*s."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        sag_area = float("nan")
        if is_hyperpol(self.data):
            where_sag = get_sweep_sag_idxs(self, store_diagnostics=store_diagnostics)
            if np.sum(where_sag) > 10:  # TODO: what should be min sag duration!?
                v_sag = self.data.v[where_sag]
                t_sag = self.data.t[where_sag]
                v_sagline = v_sag[0]
                # Take running average of v?
                if len(v_sag) > 10:  # at least 10 points to integrate
                    sag_area = cumulative_trapezoid(v_sagline - v_sag, t_sag)[-1]
                    if store_diagnostics:
                        self._update_diagnostics(
                            {
                                "where_sag": where_sag,
                                "v_sag": v_sag,
                                "t_sag": t_sag,
                                "v_sagline": v_sagline,
                            }
                        )

        return sag_area

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        t_sag, v_sag, v_sagline = unpack(
            self.diagnostics, ["t_sag", "v_sag", "v_sagline"]
        )
        ax.plot(t_sag, v_sag, **kwargs)
        ax.fill_between(t_sag, v_sag, v_sagline, alpha=0.5, label=self.name)
        return ax


class Sag_time(SweepFeature):
    """Extract sweep level sag duration feature.

    depends on: v_deflect, stim_onset, stim_end.
    description: duration of the sag.
    units: s."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        sag_time = float("nan")
        if is_hyperpol(self.data):
            where_sag = get_sweep_sag_idxs(self, store_diagnostics=store_diagnostics)
            if np.sum(where_sag) > 10:  # TODO: what should be min sag duration!?
                sag_t_start, sag_t_end = self.data.t[where_sag][[0, -1]]
                sag_time = sag_t_end - sag_t_start
                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "where_sag": where_sag,
                            "sag_t_start": sag_t_start,
                            "sag_t_end": sag_t_end,
                        }
                    )
        return sag_time

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        where_sag, sag_start, sag_end = unpack(
            self.diagnostics, ["where_sag", "sag_t_start", "sag_t_end"]
        )
        v = self.data.v[where_sag][0]
        ax.hlines(v, xmin=sag_start, xmax=sag_end, label=self.name, **kwargs)
        return ax


class V_plateau(SweepFeature):
    """Extract sweep level plataeu voltage feature.

    depends on: stim_end.
    description: average voltage during the plateau.
    units: mV."""

    def __init__(self, data=None, compute_at_init=True, T_plateau=0.1):
        self.T_plateau = T_plateau
        super().__init__(data, compute_at_init=False)
        if compute_at_init and data is not None:  # because of T_plateau
            self.get_value()

    def _compute(self, recompute=False, store_diagnostics=True):
        v_avg_plateau = float("nan")
        if is_hyperpol(self.data):
            end = self.lookup_sweep_feature("stim_end", recompute=recompute)
            # same as voltage deflection
            where_plateau = where_between(self.data.t, end - self.T_plateau, end)
            v_plateau = self.data.v[where_plateau]
            t_plateau = self.data.t[where_plateau]
            v_avg_plateau = ft.average_voltage(v_plateau, t_plateau)
            if store_diagnostics:
                self._update_diagnostics(
                    {
                        "where_plateau": where_plateau,
                        "v_plateau": v_plateau,
                        "t_plateau": t_plateau,
                    }
                )
        return v_avg_plateau

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        t, v = unpack(self.diagnostics, ["t_plateau", "v_plateau"])
        ax.plot(t, v, label=self.name + " interval", **kwargs)
        ax.hlines(self.value, *t[[0, -1]], ls="--", label=self.name, **kwargs)
        return ax


class Rebound(SweepFeature):
    """Extract sweep level rebound feature.

    depends on: v_baseline, stim_end.
    description: V_max during stimulus_end and stimulus_end + T_rebound - V_baseline.
    units: mV."""

    def __init__(self, data=None, compute_at_init=True, T_rebound=0.3):
        self.T_rebound = T_rebound
        super().__init__(data, compute_at_init=False)
        if compute_at_init and data is not None:  # because of T_rebound
            self.get_value()

    def _compute(self, recompute=False, store_diagnostics=True):
        rebound = float("nan")
        if has_rebound(self, self.T_rebound):
            v_baseline = self.lookup_sweep_feature("v_baseline", recompute=recompute)
            end = self.lookup_sweep_feature("stim_end", recompute=recompute)
            where_rebound = where_between(self.data.t, end, end + self.T_rebound)
            where_rebound = np.logical_and(where_rebound, self.data.v > v_baseline)
            t_rebound = self.data.t[where_rebound]
            v_rebound = self.data.v[where_rebound]
            if len(v_rebound) > 10:  # at least 10 time points with rebound
                idx_rebound = np.argmax(self.data.v[where_rebound] - v_baseline)
                idx_rebound = np.where(where_rebound)[0][idx_rebound]
                max_rebound = self.data.v[idx_rebound]
                rebound = max_rebound - v_baseline
                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "idx_rebound": idx_rebound,
                            "t_rebound": t_rebound,
                            "v_rebound": v_rebound,
                            "v_baseline": v_baseline,
                            "max_rebound": max_rebound,
                            "where_rebound": where_rebound,
                        }
                    )
        return rebound

    def _plot(self, ax=None, include_details=False, **kwargs):
        t_rebound, v_rebound, idx_rebound, v_baseline = unpack(
            self.diagnostics, ["t_rebound", "v_rebound", "idx_rebound", "v_baseline"]
        )
        t = self.data.t[idx_rebound]
        v = self.data.v[idx_rebound]
        ax.vlines(t, v_baseline, v, label=self.name, **kwargs)
        if include_details:
            ax.plot(t_rebound, v_rebound, label=self.name + " interval", **kwargs)
        return ax


class Rebound_APs(SweepFeature):
    """Extract sweep level number of rebounding spikes feature.

    depends on: stim_end.
    description: number of spikes during stimulus_end and stimulus_end + T_rebound.
    units: /."""

    def __init__(self, data=None, compute_at_init=True, T_rebound=0.3):
        self.T_rebound = T_rebound
        super().__init__(data, compute_at_init=False)
        if compute_at_init and data is not None:  # because of T_rebound
            self.get_value()

    def _compute(self, recompute=False, store_diagnostics=True):
        num_rebound_aps = float("nan")
        if has_rebound(self, self.T_rebound):
            t_spike = self.lookup_spike_feature("peak_t", recompute=recompute)
            idx_spike = self.lookup_spike_feature("peak_index", recompute=recompute)
            v_spike = self.lookup_spike_feature("peak_v", recompute=recompute)
            if len(t_spike) != 0:
                end = self.lookup_sweep_feature("stim_end", recompute=recompute)
                w_rebound = where_between(t_spike, end, end + self.T_rebound)
                idx_rebound = idx_spike[w_rebound]
                t_rebound = t_spike[w_rebound]
                v_rebound = v_spike[w_rebound]
                num_rebound_aps = np.sum(w_rebound)
                if num_rebound_aps > 0:
                    if store_diagnostics:
                        self._update_diagnostics(
                            {
                                "idx_rebound": idx_rebound,
                                "t_rebound": t_rebound,
                                "v_rebound": v_rebound,
                            }
                        )
        return num_rebound_aps

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        t_rebound, v_rebound = unpack(self.diagnostics, ["t_rebound", "v_rebound"])
        ax.plot(t_rebound, v_rebound, "x", label=self.name, **kwargs)
        return ax


class Rebound_area(SweepFeature):
    """Extract sweep level rebound area feature.

    depends on: v_baseline, stim_end.
    description: area between rebound curve and baseline voltage from stimulus_end
    to stimulus_end + T_rebound.
    units: mV*s."""

    def __init__(self, data=None, compute_at_init=True, T_rebound=0.3):
        self.T_rebound = T_rebound
        super().__init__(data, compute_at_init=False)
        if compute_at_init and data is not None:  # because of T_rebound
            self.get_value()

    def _compute(self, recompute=False, store_diagnostics=True):
        rebound_area = float("nan")
        if has_rebound(self, self.T_rebound):
            v_baseline = self.lookup_sweep_feature("v_baseline", recompute=recompute)
            end = self.lookup_sweep_feature("stim_end", recompute=recompute)
            where_rebound = where_between(self.data.t, end, end + self.T_rebound)
            where_rebound = np.logical_and(where_rebound, self.data.v > v_baseline)
            v_rebound = self.data.v[where_rebound]
            t_rebound = self.data.t[where_rebound]
            if len(v_rebound) > 10:  # at least 10 points to integrate
                rebound_area = cumulative_trapezoid(v_rebound - v_baseline, t_rebound)[
                    -1
                ]
                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "where_rebound": where_rebound,
                            "t_rebound": t_rebound,
                            "v_rebound": v_rebound,
                            "v_baseline": v_baseline,
                        }
                    )
        return rebound_area

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        t, v = self.data.t, self.data.v
        v_baseline, where_rebound = unpack(
            self.diagnostics, ["v_baseline", "where_rebound"]
        )
        ax.fill_between(
            t, v, v_baseline, where=where_rebound, alpha=0.5, label=self.name, **kwargs
        )
        return ax


class Rebound_latency(SweepFeature):
    """Extract sweep level rebound latency feature.

    depends on: v_baseline, stim_end.
    description: duration from stimulus_end to when the voltage reaches above
    baseline for the first time. t_rebound = t_off + rebound_latency.
    units: s."""

    def __init__(self, data=None, compute_at_init=True, T_rebound=0.3):
        self.T_rebound = T_rebound
        super().__init__(data, compute_at_init=False)
        if compute_at_init and data is not None:  # because of T_rebound
            self.get_value()

    def _compute(self, recompute=False, store_diagnostics=True):
        rebound_latency = float("nan")
        if has_rebound(self, self.T_rebound):
            v_baseline = self.lookup_sweep_feature("v_baseline", recompute=recompute)
            end = self.lookup_sweep_feature("stim_end", recompute=recompute)
            where_rebound = where_between(self.data.t, end, end + self.T_rebound)
            where_rebound = np.logical_and(where_rebound, self.data.v > v_baseline)
            t_rebound = self.data.t[where_rebound]
            v_rebound = self.data.v[where_rebound]
            if len(v_rebound) > 10:  # at least 10 time points with rebound
                idx_rebound_reached = np.where(where_rebound)[0]
                t_rebound_reached = self.data.t[idx_rebound_reached][0]
                rebound_latency = t_rebound_reached - end
                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "idx_rebound_reached": idx_rebound_reached,
                            "t_rebound_reached": t_rebound_reached,
                            "where_rebound": where_rebound,
                            "t_rebound": t_rebound,
                            "v_rebound": v_rebound,
                            "v_baseline": v_baseline,
                        }
                    )
        return rebound_latency

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        t, v = self.data.t, self.data.v
        t_rebound_reached = unpack(self.diagnostics, "t_rebound_reached")
        stim_end = self.lookup_sweep_feature("stim_end", return_value=False)
        end_idx = unpack(stim_end.diagnostics, "idx_end")
        until_rebound = where_between(t, t[end_idx], t_rebound_reached)
        ax.fill_between(
            t, v, v[end_idx], where=until_rebound, alpha=0.5, label=self.name, **kwargs
        )
        return ax


class Rebound_avg(SweepFeature):
    """Extract sweep level average rebound feature.

    depends on: v_baseline, stim_end.
    description: average voltage between stimulus_end
    and stimulus_end + T_rebound - baseline voltage.
    units: mV."""

    def __init__(self, data=None, compute_at_init=True, T_rebound=0.3):
        self.T_rebound = T_rebound
        super().__init__(data, compute_at_init=False)
        if compute_at_init and data is not None:  # because of T_rebound
            self.get_value()

    def _compute(self, recompute=False, store_diagnostics=True):
        v_rebound_avg = float("nan")
        if has_rebound(self, self.T_rebound):
            v_baseline = self.lookup_sweep_feature("v_baseline", recompute=recompute)
            end = self.lookup_sweep_feature("stim_end", recompute=recompute)
            where_rebound = where_between(self.data.t, end, end + self.T_rebound)
            where_rebound = np.logical_and(where_rebound, self.data.v > v_baseline)
            v_rebound = self.data.v[where_rebound]
            t_rebound = self.data.t[where_rebound]
            if len(v_rebound) > 10:  # at least 10 rebound points
                v_rebound_avg = ft.average_voltage(v_rebound, t_rebound) - v_baseline
                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "where_rebound": where_rebound,
                            "t_rebound": t_rebound,
                            "v_rebound": v_rebound,
                            "v_baseline": v_baseline,
                        }
                    )
        return v_rebound_avg

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        t_rebound, v_rebound = unpack(self.diagnostics, ["t_rebound", "v_rebound"])
        v_baseline = unpack(self.diagnostics, "v_baseline")
        ax.plot(t_rebound, v_rebound, label=self.name + " interval", **kwargs)
        ax.hlines(
            [self.value + v_baseline],
            # np.mean(v_rebound),
            *t_rebound[[0, -1]],
            ls="--",
            label=self.name,
            **kwargs,
        )
        return ax


class V_rest(SweepFeature):
    """Extract sweep level resting potential feature.

    depends on: v_baseline, r_input, dc_offset.
    description: v_rest = v_baseline - r_input*dc_offset.
    units: mV."""

    def __init__(self, data=None, compute_at_init=True, dc_offset=0):
        self.dc_offset = dc_offset
        super().__init__(data, compute_at_init=False)
        if compute_at_init and data is not None:  # because of dc_offset
            self.get_value()

    def _compute(self, recompute=False, store_diagnostics=True):
        v_rest = float("nan")
        v_baseline = self.lookup_sweep_feature("v_baseline", recompute=recompute)
        r_input = self.lookup_sweep_feature("r_input", recompute=recompute)
        try:
            v_rest = v_baseline - r_input * 1e-3 * self.dc_offset
            if store_diagnostics:
                self._update_diagnostics(
                    {
                        "v_baseline": v_baseline,
                        "r_input": r_input,
                        "dc_offset": self.dc_offset,
                    }
                )
        except KeyError:
            pass
        return v_rest

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        r_input, dc_offset = unpack(self.diagnostics, ["r_input", "dc_offset"])
        t, v = self.data.t, self.data.v
        v -= r_input * dc_offset * 1e-3
        ax.plot(t, v, label="v(t) - r_in*dc_offset", **kwargs)
        ax.axhline(self.value, ls="--", label=self.name)
        return ax


class Num_bursts(SweepFeature):
    """Extract sweep level number of bursts feature.

    depends on: num_ap.
    description: Number of detected bursts.
    units: /."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        num_bursts = float("nan")
        num_ap = self.lookup_sweep_feature("num_ap", recompute=recompute)
        if num_ap > 5 and has_stimulus(self.data):
            idx_burst, idx_burst_start, idx_burst_end = get_sweep_burst_metrics(
                self.data
            )
            peak_t = self.lookup_spike_feature("peak_t", recompute=recompute)
            if not np.isnan(idx_burst).any():
                t_burst_start = peak_t[idx_burst_start]
                t_burst_end = peak_t[idx_burst_end]
                num_bursts = len(idx_burst)
                num_bursts = float("nan") if num_bursts == 0 else num_bursts
                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "idx_burst": idx_burst,
                            "idx_burst_start": idx_burst_start,
                            "idx_burst_end": idx_burst_end,
                            "t_burst_start": t_burst_start,
                            "t_burst_end": t_burst_end,
                        }
                    )
        return num_bursts

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        t_burst_start, t_burst_end = unpack(
            self.diagnostics, ["t_burst_start", "t_burst_end"]
        )
        for i, (t_start, t_end) in enumerate(zip(t_burst_start, t_burst_end)):
            ax.axvspan(
                t_start,
                t_end,
                alpha=0.5,
                label=f"burst {i+1}",
                **kwargs,
            )
        return ax


class Burstiness(SweepFeature):
    """Extract sweep level burstiness feature.

    depends on: num_ap.
    description: max "burstiness" index across detected bursts.
    units: /."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        max_burstiness = float("nan")
        num_ap = self.lookup_sweep_feature("num_ap", recompute=recompute)
        if num_ap > 5 and has_stimulus(self.data):
            idx_burst, idx_burst_start, idx_burst_end = get_sweep_burst_metrics(
                self.data
            )
            peak_t = self.lookup_spike_feature("peak_t", recompute=recompute)
            if not np.isnan(idx_burst).any():
                t_burst_start = peak_t[idx_burst_start]
                t_burst_end = peak_t[idx_burst_end]
                num_bursts = len(idx_burst)
                max_burstiness = idx_burst.max() if num_bursts > 0 else float("nan")
                max_burstiness = (
                    float("nan") if max_burstiness < 0 else max_burstiness
                )  # don't consider negative burstiness

                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "idx_burst": idx_burst,
                            "idx_burst_start": idx_burst_start,
                            "idx_burst_end": idx_burst_end,
                            "t_burst_start": t_burst_start,
                            "t_burst_end": t_burst_end,
                        }
                    )
        return max_burstiness

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        num_bursts = self.lookup_sweep_feature("num_bursts", return_value=False)
        ax = num_bursts.plot(ax=ax, **kwargs)
        return ax


class ISI_adapt(SweepFeature):
    """Extract sweep level inter-spike-interval (ISI) adaptation index feature.

    depends on: ISIs.
    description: /.
    units: /."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        isi_adapt = float("nan")
        if has_spikes(self.data):
            isi = self.lookup_spike_feature("isi", recompute=recompute)[1:]
            if len(isi) > 1:
                isi_adapt = isi[1] / isi[0]

            if store_diagnostics:
                self._update_diagnostics({"isi": isi})
        return isi_adapt

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        ax = plot_isi(self.data, ax=ax, selected_idxs=[1, 2], **kwargs)
        relabel_line(ax, "isi", self.name)
        return ax


class ISI_adapt_avg(SweepFeature):
    """Extract sweep level average inter-spike-interval (ISI) adaptation index feature.

    depends on: ISIs.
    description: /.
    units: /."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        isi_adapt_avg = float("nan")
        if has_spikes(self.data):
            isi = self.lookup_spike_feature("isi", recompute=recompute)[1:]
            if len(isi) > 2:
                isi_changes = isi[1:] / isi[:-1]
                isi_adapt_avg = isi_changes.mean()

                if store_diagnostics:
                    self._update_diagnostics({"isi": isi})
        return isi_adapt_avg

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        ax = plot_isi(self.data, ax=ax, **kwargs)
        relabel_line(ax, "isi", self.name)
        return ax


class AP_amp_adapt(SweepFeature):
    """Extract sweep level AP amplitude adaptation index feature.

    depends on: ap_amp.
    description: /.
    units: mV/s."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        ap_amp_adapt = float("nan")
        if has_spikes(self.data):
            ap_amp = self.lookup_spike_feature("ap_amp", recompute=recompute)
            if len(ap_amp) > 1:
                ap_amp_adapt = ap_amp[1] / ap_amp[0]

            if store_diagnostics:
                self._update_diagnostics({"ap_amp": ap_amp})

        return ap_amp_adapt

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        ax = plot_ap_amp(self.data, ax=ax, selected_idxs=[0, 1], **kwargs)
        relabel_line(ax, "ap_amp", self.name)
        return ax


class AP_amp_adapt_avg(SweepFeature):
    """Extract sweep level average AP amplitude adaptation index feature.

    depends on: ap_amp.
    description: /.
    units: /."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        ap_amp_adapt_avg = float("nan")
        if has_spikes(self.data):
            ap_amp = self.lookup_spike_feature("ap_amp", recompute=recompute)
            if len(ap_amp) > 2:
                ap_amp_changes = ap_amp[1:] / ap_amp[:-1]
                ap_amp_adapt_avg = ap_amp_changes.mean()

            if store_diagnostics:
                self._update_diagnostics({"ap_amp": ap_amp})

        return ap_amp_adapt_avg

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        ax = plot_ap_amp(self.data, ax=ax, **kwargs)
        relabel_line(ax, "ap_amp", self.name)
        return ax


class Wildness(SweepFeature):
    """Extract sweep level wildness feature.

    depends on: /.
    description: Wildness is the number of spikes that occur outside of the stimulus interval.
    units: /."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        num_wild_spikes = float("nan")
        if has_spikes(self.data):
            onset = self.lookup_sweep_feature("stim_onset", recompute=recompute)
            end = self.lookup_sweep_feature("stim_end", recompute=recompute)
            peak_t = self.lookup_spike_feature("peak_t", recompute=recompute)
            peak_idx = self.lookup_spike_feature("peak_index", recompute=recompute)
            peak_v = self.lookup_spike_feature("peak_v", recompute=recompute)
            stim_window = where_between(peak_t, onset, end)

            i_wild_spikes = peak_idx[~stim_window]
            t_wild_spikes = peak_t[~stim_window]
            v_wild_spikes = peak_v[~stim_window]
            if len(i_wild_spikes) > 0:
                num_wild_spikes = len(i_wild_spikes)
                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "i_wild_spikes": i_wild_spikes,
                            "t_wild_spikes": t_wild_spikes,
                            "v_wild_spikes": v_wild_spikes,
                        }
                    )
        return num_wild_spikes

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        t, v = unpack(self.diagnostics, ["t_wild_spikes", "v_wild_spikes"])
        ax.plot(t, v, "x", label=self.name, **kwargs)
        return ax


class APSweepFeature(SweepFeature):
    """Extract sweep level AP feature.

    description: Action potential feature to represent a sweep."""

    def __init__(
        self,
        data=None,
        compute_at_init=True,
        ft_name: Optional[str] = None,
        ap_selector: Optional[Callable] = None,
        ft_aggregator: Optional[Callable] = None,
    ):
        """
        Args:
            ft_name (Optional[str], optional): Name of the spike feature
            ap_selector (Optional[Callable], optional): Function which selects a
                representative ap or set of aps based on a given criterion.
                Function expects a EphysSweepSetFeatureExtractor object as input and
                returns indices for the selected aps. If none is provided, falls
                back to selecting all aps.
            ft_aggregator (Optional[Callable], optional): Function which aggregates
                a list of feature values into a single value. Function expects a
                list or ndarray of numbers as input. If none is provided, falls back
                to `np.nanmedian` (equates to pass through for single sweeps)."""
        self.ap_selector = ap_selector
        self.ft_aggregator = ft_aggregator
        super().__init__(data, compute_at_init)
        if ft_name is not None:
            self.name = ft_name

    def _select(self, data):
        """Function expects a EphysSweepSetFeatureExtractor object as input and
        returns indices for the selected aps.

        description: Select a representative ap or set of aps based on a
        given criterion. If none is provided, falls back to selecting all aps."""
        if self.ap_selector is None:
            feature = self.lookup_spike_feature(self.name)
            return np.arange(len(feature))
        else:
            return self.ap_selector(data)

    def _aggregate(self, X):
        """Function expects a list or ndarray of numbers as input.

        description: Aggregate a list of feature values into a single value. If none is provided, falls back
        to `np.nanmedian` (equates to pass through for single sweeps)."""
        if np.isnan(X).all():
            return float("nan")
        elif self.ft_aggregator is None:
            self._update_diagnostics({"aggregate_idx": median_idx(X)})
            return np.nanmedian(X).item()
        else:
            return self.ft_aggregator(X)

    def _compute(self, recompute=False, store_diagnostics=True):
        feature = self.lookup_spike_feature(self.name, recompute=recompute)
        ft_agg = float("nan")

        if len(feature) > 0:
            selected_idx = self._select(self.data)
            fts_selected = feature[selected_idx]

            if isinstance(fts_selected, (float, int, np.float64, np.int64)):
                ft_agg = fts_selected
            elif isinstance(fts_selected, ndarray):
                if len(fts_selected.flat) == 0:
                    ft_agg = float("nan")
                else:
                    ft_agg = self._aggregate(feature)

            if store_diagnostics:
                self._update_diagnostics(
                    {
                        "selected_idx": selected_idx,
                        "selected_fts": fts_selected,
                        "selection": parse_desc(self._select),
                        "aggregation": parse_desc(self._aggregate),
                    }
                )
        return ft_agg

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        idxs = unpack(self.diagnostics, "selected_idx")
        ax = plot_spike_feature(
            self.data, self.name, ax=ax, selected_idxs=idxs, **kwargs
        )
        return ax


class AP_AHP(APSweepFeature):
    """Extract sweep level Afterhyperpolarization feature.

    depends on: /.
    description: Afterhyperpolarization (AHP) for representative AP. Difference
    between the fast trough and the threshold.
    units: mV."""

    def __init__(
        self,
        data=None,
        compute_at_init=True,
        ap_selector: Optional[Callable] = None,
        ft_aggregator: Optional[Callable] = None,
    ):
        super().__init__(data, compute_at_init, "ap_ahp", ap_selector, ft_aggregator)


class AP_ADP(APSweepFeature):
    """Extract sweep level Afterdepolarization feature.

    depends on: /.
    description: Afterdepolarization (ADP) for representative AP. Difference between the ADP and the fast trough.
    units: mV."""

    def __init__(
        self,
        data=None,
        compute_at_init=True,
        ap_selector: Optional[Callable] = None,
        ft_aggregator: Optional[Callable] = None,
    ):
        super().__init__(data, compute_at_init, "ap_adp", ap_selector, ft_aggregator)


class AP_thresh(APSweepFeature):
    """Extract sweep level AP threshold feature.

    depends on: /.
    description: AP threshold for representative AP.
    units: mV."""

    def __init__(
        self,
        data=None,
        compute_at_init=True,
        ap_selector: Optional[Callable] = None,
        ft_aggregator: Optional[Callable] = None,
    ):
        super().__init__(data, compute_at_init, "ap_thresh", ap_selector, ft_aggregator)


class AP_amp(APSweepFeature):
    """Extract sweep level AP amplitude feature.

    depends on: /.
    description: AP amplitude for representative AP.
    units: mV."""

    def __init__(
        self,
        data=None,
        compute_at_init=True,
        ap_selector: Optional[Callable] = None,
        ft_aggregator: Optional[Callable] = None,
    ):
        super().__init__(data, compute_at_init, "ap_amp", ap_selector, ft_aggregator)


class AP_width(APSweepFeature):
    """Extract sweep level AP width feature.

    depends on: /.
    description: AP width for representative AP.
    units: s."""

    def __init__(
        self,
        data=None,
        compute_at_init=True,
        ap_selector: Optional[Callable] = None,
        ft_aggregator: Optional[Callable] = None,
    ):
        super().__init__(data, compute_at_init, "ap_width", ap_selector, ft_aggregator)


class AP_peak(APSweepFeature):
    """Extract sweep level AP peak feature.

    depends on: /.
    description: AP peak for representative AP.
    units: mV."""

    def __init__(
        self,
        data=None,
        compute_at_init=True,
        ap_selector: Optional[Callable] = None,
        ft_aggregator: Optional[Callable] = None,
    ):
        super().__init__(data, compute_at_init, "ap_peak", ap_selector, ft_aggregator)


class AP_trough(APSweepFeature):
    """Extract sweep level AP trough feature.

    depends on: /.
    description: AP trough for representative AP.
    units: mV."""

    def __init__(
        self,
        data=None,
        compute_at_init=True,
        ap_selector: Optional[Callable] = None,
        ft_aggregator: Optional[Callable] = None,
    ):
        super().__init__(data, compute_at_init, "ap_trough", ap_selector, ft_aggregator)


class AP_UDR(APSweepFeature):
    """Extract sweep level Upstroke-to-downstroke ratio feature.

    depends on: /.
    description: Upstroke-to-downstroke ratio for representative AP.
    units: /."""

    def __init__(
        self,
        data=None,
        compute_at_init=True,
        ap_selector: Optional[Callable] = None,
        ft_aggregator: Optional[Callable] = None,
    ):
        super().__init__(data, compute_at_init, "ap_udr", ap_selector, ft_aggregator)


class ISI(APSweepFeature):
    """Extract sweep level ISI ratio feature.

    depends on: /.
    description: Median interspike interval.
    units: /."""

    def __init__(
        self,
        data=None,
        compute_at_init=True,
        ap_selector: Optional[Callable] = None,
        ft_aggregator: Optional[Callable] = None,
    ):
        super().__init__(data, compute_at_init, "isi", ap_selector, ft_aggregator)
