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

from py_ephys.allen_sdk.ephys_extractor import EphysSweepFeatureExtractor
from py_ephys.utils import *
from py_ephys.base import EphysFeature, SweepSetFeature

# ransac = linear_model.RANSACRegressor()
ransac = linear_model.LinearRegression()

############################
### spike level features ###
############################


def get_spike_peak_height(sweep: EphysSweepFeatureExtractor) -> float:
    """Extract spike level peak height feature.

    depends on: threshold_v, peak_v.
    description: v_peak - threshold_v.
    units: mV.

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
    units: mV.

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
    units: mV.

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


class StimAmp(EphysFeature):
    """Extract sweep level stimulus ampltiude feature.
    depends on: /.
    description: maximum amplitude of stimulus.
    units: pA."""

    def __init__(self, data=None):
        super().__init__(data=data)

    def _compute(self, recompute=False, store_diagnostics=True):
        stim_amp = np.max(abs(self.data.i).T, axis=0)
        return stim_amp


class StimOnset(EphysFeature):
    """Extract sweep level stimulus onset feature.

    depends on: /.
    description: time of stimulus onset.
    units: s."""

    def __init__(self, data=None):
        super().__init__(data=data)

    def _compute(self, recompute=False, store_diagnostics=True):
        if has_stimulus(self.data):
            return self.data.t[where_stimulus(self.data)][0]
        else:
            return float("nan")


class StimEnd(EphysFeature):
    """Extract sweep level stimulus end feature.

    depends on: /.
    description: time of stimulus end.
    units: s."""

    def __init__(self, data=None):
        super().__init__(data=data)

    def _compute(self, recompute=False, store_diagnostics=True):
        if has_stimulus(self.data):
            return self.data.t[where_stimulus(self.data)][-1]
        else:
            return float("nan")


class NumAP(EphysFeature):
    """Extract sweep level spike count feature.

    depends on: stim_onset, stim_end.
    description: # peaks during stimulus.
    units: /."""

    def __init__(self, data=None):
        super().__init__(data=data)

    def _compute(self, recompute=False, store_diagnostics=True):
        peak_t = self.lookup_spike_feature("peak_t", recompute=recompute)
        onset = self.lookup_sweep_feature("stimonset")
        end = self.lookup_sweep_feature("stimend")
        stim_window = where_between(peak_t, onset, end)

        peak_i = self.lookup_spike_feature("peak_index")[stim_window]
        num_ap = len(peak_i)

        if num_ap < 0:
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


class APFreq(EphysFeature):
    """Extract sweep level spike rate feature.

    depends on: numap.
    description: # peaks during stimulus / stimulus duration.
    units: Hz."""

    def __init__(self, data=None):
        super().__init__(data=data)

    def _compute(self, recompute=False, store_diagnostics=True):
        num_ap = self.lookup_sweep_feature("numap", recompute=recompute)
        onset = self.lookup_sweep_feature("stimonset", recompute=recompute)
        end = self.lookup_sweep_feature("stimend", recompute=recompute)

        ap_freq = num_ap / (end - onset)

        if store_diagnostics:
            self._update_diagnostics(
                {"ap_freq": ap_freq, "num_ap": num_ap, "onset": onset, "end": end}
            )
        return ap_freq


###############################
### sweepset level features ###
###############################


class SweepSetAPFeature(SweepSetFeature):
    def __init__(self, feature):
        super().__init__(feature)

    def _select(self, fts):
        make_selection = lambda fts: fts
        self._update_diagnostics({})
        return make_selection(fts)

    def _aggregate(self, fts):
        aggregate = np.nanmean
        self._update_diagnostics({})
        return aggregate(fts)

    def _compute(self, recompute=False, store_diagnostics=False):
        fts = self.lookup_sweep_feature(self.name, recompute=recompute)

        subset = self._select(fts)
        ft = self._aggregate(subset)
        self._update_diagnostics({})
        return ft
