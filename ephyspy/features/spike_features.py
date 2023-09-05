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

from typing import TYPE_CHECKING

import numpy as np

from ephyspy.features.base import SpikeFeature

if TYPE_CHECKING:
    from ephyspy.sweeps import EphysSweep


def available_spike_features():
    return {
        "ap_peak": Spike_AP_peak,
        "ap_width": Spike_AP_width,
        "ap_trough": Spike_AP_trough,
        "ap_thresh": Spike_AP_thresh,
        "ap_amp": Spike_AP_amp,
        "ap_udr": Spike_AP_UDR,
        "ap_ahp": Spike_AP_AHP,
        "ap_adp": Spike_AP_ADP,
        "isi": Spike_ISI,
    }


class Spike_AP_amp(SpikeFeature):
    """Extract spike level peak height feature.

    depends on: threshold_v, peak_v.
    description: v_peak - threshold_v.
    units: mV.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_peak = self.lookup_spike_feature("peak_v", recompute=recompute)
        threshold_v = self.lookup_spike_feature("threshold_v", recompute=recompute)
        peak_height = v_peak - threshold_v
        return peak_height if len(v_peak) > 0 else np.array([])


class Spike_AP_AHP(SpikeFeature):
    """Extract spike level after hyperpolarization feature.

    depends on: threshold_v, fast_trough_v.
    description: v_fast_trough - threshold_v.
    units: mV.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_fast_trough = self.lookup_spike_feature("fast_trough_v", recompute=recompute)
        threshold_v = self.lookup_spike_feature("threshold_v", recompute=recompute)
        return v_fast_trough - threshold_v


class Spike_AP_ADP(SpikeFeature):
    """Extract spike level after depolarization feature.

    depends on: adp_v, fast_trough_v.
    description: v_adp - v_fast_trough.
    units: mV.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_adp = self.lookup_spike_feature("adp_v", recompute=recompute)
        v_fast_trough = self.lookup_spike_feature("fast_trough_v", recompute=recompute)
        return v_adp - v_fast_trough


class Spike_AP_peak(SpikeFeature):
    """Extract spike level peak feature.

    depends on: peak_v.
    description: max voltage of AP.
    units: mV.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_peak = self.lookup_spike_feature("peak_v", recompute=recompute)
        return v_peak


class Spike_AP_thresh(SpikeFeature):
    """Extract spike level ap threshold feature.

    depends on: threshold_v.
    description: For details on how AP thresholds are computed see AllenSDK.
    units: mV.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_thresh = self.lookup_spike_feature("threshold_v", recompute=recompute)
        return v_thresh


class Spike_AP_trough(SpikeFeature):
    """Extract spike level ap trough feature.

    depends on: through_v.
    description: For details on how AP troughs are computed see AllenSDK.
    units: mV.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_thresh = self.lookup_spike_feature("trough_v", recompute=recompute)
        return v_thresh


class Spike_AP_width(SpikeFeature):
    """Extract spike level ap width feature.

    depends on: width.
    description: full width half max of AP.
    units: s.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        width = self.lookup_spike_feature("width", recompute=recompute)
        return width


class Spike_AP_UDR(SpikeFeature):
    """Extract spike level ap udr feature.

    depends on: upstroke, downstroke.
    description: upstroke / downstroke. For details on how upstroke, downstroke
    are computed see AllenSDK.
    units: /.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        upstroke = self.lookup_spike_feature("upstroke", recompute=recompute)
        downstroke = self.lookup_spike_feature("downstroke", recompute=recompute)
        return upstroke / -downstroke


class Spike_ISI(SpikeFeature):
    """Extract spike level inter-spike-interval feature.

    depends on: threshold_t.
    description: The distance between subsequent spike thresholds. isi at the
        first index is nan since isi[t+1] = threshold_t[t+1] - threshold_t[t].
    units: s.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        spike_times = self.lookup_spike_feature("threshold_t", recompute=recompute)
        if len(spike_times) > 1:
            isi = np.diff(spike_times)
            isi = np.insert(isi, 0, 0)
            return isi
        elif len(spike_times) == 1:
            return np.array([float("nan")])
        else:
            return np.array([])
