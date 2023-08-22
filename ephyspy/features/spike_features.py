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

if TYPE_CHECKING:
    from ephyspy.sweeps import EphysSweep


def available_spike_features():
    return {
        "ap_peak": ap_peak,
        "ap_width": ap_width,
        "ap_trough": ap_trough,
        "ap_thresh": ap_thresh,
        "ap_amp": ap_amp,
        "ap_udr": ap_udr,
        "ap_ahp": ap_ahp,
        "ap_adp": ap_adp,
        "isi": isi,
    }


def ap_amp(sweep: EphysSweep) -> float:
    """Extract spike level peak height feature.

    depends on: threshold_v, peak_v.
    description: v_peak - threshold_v.
    units: mV.

    Args:
        sweep (EphysSweep): Sweep to extract feature from.

    Returns:
        float: Spike peak height feature.
    """
    v_peak = sweep.spike_feature("peak_v", include_clipped=True)
    threshold_v = sweep.spike_feature("threshold_v", include_clipped=True)
    peak_height = v_peak - threshold_v
    return peak_height if len(v_peak) > 0 else np.array([])


def ap_ahp(sweep: EphysSweep) -> float:
    """Extract spike level after hyperpolarization feature.

    depends on: threshold_v, fast_trough_v.
    description: v_fast_trough - threshold_v.
    units: mV.

    Args:
        sweep (EphysSweep): Sweep to extract feature from.

    Returns:
        float: Spike after hyperpolarization feature.
    """
    v_fast_trough = sweep.spike_feature("fast_trough_v", include_clipped=True)
    threshold_v = sweep.spike_feature("threshold_v", include_clipped=True)
    return v_fast_trough - threshold_v


def ap_adp(sweep: EphysSweep) -> float:
    """Extract spike level after depolarization feature.

    depends on: adp_v, fast_trough_v.
    description: v_adp - v_fast_trough.
    units: mV.

    Args:
        sweep (EphysSweep): Sweep to extract feature from.

    Returns:
        float: Spike after depolarization feature.
    """
    v_adp = sweep.spike_feature("adp_v", include_clipped=True)
    v_fast_trough = sweep.spike_feature("fast_trough_v", include_clipped=True)
    return v_adp - v_fast_trough


def ap_peak(sweep: EphysSweep) -> float:
    """Extract spike level peak feature.

    depends on: peak_v.
    description: max voltage of AP.
    units: mV.

    Args:
        sweep (EphysSweep): Sweep to extract feature from.

    Returns:
        float: AP peak feature.
    """
    v_peak = sweep.spike_feature("peak_v", include_clipped=True)
    return v_peak


def ap_thresh(sweep: EphysSweep) -> float:
    """Extract spike level ap threshold feature.

    depends on: threshold_v.
    description: For details on how AP thresholds are computed see AllenSDK.
    units: mV.

    Args:
        sweep (EphysSweep): Sweep to extract feature from.

    Returns:
        float: AP threshold feature.
    """
    v_thresh = sweep.spike_feature("threshold_v", include_clipped=True)
    return v_thresh


def ap_trough(sweep: EphysSweep) -> float:
    """Extract spike level ap trough feature.

    depends on: through_v.
    description: For details on how AP troughs are computed see AllenSDK.
    units: mV.

    Args:
        sweep (EphysSweep): Sweep to extract feature from.

    Returns:
        float: AP trough feature.
    """
    v_thresh = sweep.spike_feature("trough_v", include_clipped=True)
    return v_thresh


def ap_width(sweep: EphysSweep) -> float:
    """Extract spike level ap width feature.

    depends on: width.
    description: full width half max of AP.
    units: s.

    Args:
        sweep (EphysSweep): Sweep to extract feature from.

    Returns:
        float: AP width feature.
    """
    width = sweep.spike_feature("width", include_clipped=True)
    return width


def ap_udr(sweep: EphysSweep) -> float:
    """Extract spike level ap udr feature.

    depends on: upstroke, downstroke.
    description: upstroke / downstroke. For details on how upstroke, downstroke
    are computed see AllenSDK.
    units: /.

    Args:
        sweep (EphysSweep): Sweep to extract feature from.

    Returns:
        float: AP udr feature.
    """
    upstroke = sweep.spike_feature("upstroke", include_clipped=True)
    downstroke = sweep.spike_feature("downstroke", include_clipped=True)
    return upstroke / downstroke


def isi(sweep: EphysSweep) -> float:
    """Extract spike level inter-spike-interval feature.

    depends on: threshold_t.
    description: The distance between subsequent spike thresholds. isi at the
        first index is nan since isi[t+1] = threshold_t[t+1] - threshold_t[t].
    units: s.

    Args:
        sweep (EphysSweep): Sweep to extract feature from.

    Returns:
        float: inter-spike-interval feature.
    """
    spike_times = sweep.spike_feature("threshold_t", include_clipped=True)
    if len(spike_times) > 1:
        isi = np.diff(spike_times)
        isi = np.insert(isi, 0, np.nan)
        return isi
    elif len(spike_times) == 1:
        return np.array([float("nan")])
    else:
        return np.array([])
