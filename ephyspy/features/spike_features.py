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

from typing import Dict, Optional

import numpy as np
from matplotlib.axes import Axes

from ephyspy.features.base import SpikeFeature
from ephyspy.features.utils import fetch_available_fts
from ephyspy.utils import fwhm, has_spike_feature, is_spike_feature, scatter_spike_ft


def available_spike_features(
    compute_at_init: bool = False, store_diagnostics: bool = False
) -> Dict[str, SpikeFeature]:
    """Return a dictionary of all implemented spike features.

    Looks for all classes that inherit from SpikeFeature and returns a dictionary
    of all available features. If compute_at_init is True, the features are
    computed at initialization.

    Args:
        compute_at_init (bool, optional): If True, the features are computed at
            initialization. Defaults to False.
        store_diagnostics (bool, optional): If True, the features are computed
            with diagnostics. Defaults to False.

    Returns:
        dict[str, SpikeFeature]: Dictionary of all available spike features.
    """
    all_features = fetch_available_fts()
    features = {ft.__name__.lower(): ft for ft in all_features if is_spike_feature(ft)}
    features = {k.replace("spike_", ""): v for k, v in features.items()}
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


class Spike_AP_upstroke(SpikeFeature):
    """Extract spike level upstroke feature.

    depends on: /.
    description: upstroke of AP.
    units: mV.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        upstroke = self.lookup_spike_feature("upstroke_v", recompute=recompute)
        return upstroke

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        return scatter_spike_ft(
            "upstroke", self.data, ax=ax, selected_idxs=selected_idxs, **kwargs
        )


class Spike_AP_downstroke(SpikeFeature):
    """Extract spike level downstroke feature.

    depends on: /.
    description: downstroke of AP.
    units: mV.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        upstroke = self.lookup_spike_feature("downstroke_v", recompute=recompute)
        return upstroke

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        return scatter_spike_ft(
            "downstroke", self.data, ax=ax, selected_idxs=selected_idxs, **kwargs
        )


class Spike_AP_fast_trough(SpikeFeature):
    """Extract spike level fast trough feature.

    depends on: /.
    description: fast trough of AP.
    units: mV.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        upstroke = self.lookup_spike_feature("fast_trough_v", recompute=recompute)
        return upstroke

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        return scatter_spike_ft(
            "fast_trough", self.data, ax=ax, selected_idxs=selected_idxs, **kwargs
        )


class Spike_AP_slow_trough(SpikeFeature):
    """Extract spike level slow trough feature.

    depends on: /.
    description: slow trough of AP.
    units: mV.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(data, compute_at_init)

    def _compute(self, recompute=False, store_diagnostics=True):
        upstroke = self.lookup_spike_feature("slow_trough_v", recompute=recompute)
        return upstroke

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        return scatter_spike_ft(
            "slow_trough", self.data, ax=ax, selected_idxs=selected_idxs, **kwargs
        )


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

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        if has_spike_feature(self.data, "threshold_v"):
            idxs = slice(None) if selected_idxs is None else selected_idxs
            thresh_v = self.lookup_spike_feature("threshold_v")[idxs]
            peak_t = self.lookup_spike_feature("peak_t")[idxs]
            peak_v = self.lookup_spike_feature("peak_v")[idxs]

            ax.plot(peak_t, peak_v, "x", **kwargs)
            ax.vlines(peak_t, thresh_v, peak_v, ls="--", label="ap_amp", **kwargs)
        return ax


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

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        if has_spike_feature(self.data, "ap_ahp"):
            idxs = slice(None) if selected_idxs is None else selected_idxs
            trough_t = self.lookup_spike_feature("fast_trough_t")[idxs]
            trough_v = self.lookup_spike_feature("fast_trough_v")[idxs]
            threshold_t = self.lookup_spike_feature("threshold_t")[idxs]
            threshold_v = self.lookup_spike_feature("threshold_v")[idxs]
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

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        if has_spike_feature(self.data, "ap_adp"):
            idxs = slice(None) if selected_idxs is None else selected_idxs
            adp_t = self.lookup_spike_feature("adp_t")[idxs]
            adp_v = self.lookup_spike_feature("adp_v")[idxs]
            trough_t = self.lookup_spike_feature("fast_trough_t")[idxs]
            trough_v = self.lookup_spike_feature("fast_trough_v")[idxs]
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

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        return scatter_spike_ft(
            "peak", self.data, ax=ax, selected_idxs=selected_idxs, **kwargs
        )


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

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        return scatter_spike_ft(
            "threshold", self.data, ax=ax, selected_idxs=selected_idxs, **kwargs
        )


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

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        return scatter_spike_ft(
            "trough", self.data, ax=ax, selected_idxs=selected_idxs, **kwargs
        )


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

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        if has_spike_feature(self.data, "threshold_t"):
            idxs = slice(None) if selected_idxs is None else selected_idxs
            t_threshold = self.lookup_spike_feature("threshold_t")[idxs]
            t_peak = self.lookup_spike_feature("peak_t")[idxs]
            t_next = t_peak + 1.0 * (
                t_peak - t_threshold
            )  # T interval w.r.t. threshold

            fwhm_v = np.zeros_like(t_threshold)
            hm_up_t = np.zeros_like(t_threshold)
            hm_down_t = np.zeros_like(t_threshold)
            for i, (t_th, t_n) in enumerate(zip(t_threshold, t_next)):
                fwhm_i = fwhm(self.data.t, self.data.v, t_th, t_n)
                fwhm_v[i], hm_up_t[i], hm_down_t[i] = fwhm_i

            ax.hlines(fwhm_v, hm_up_t, hm_down_t, label="width", ls="--", **kwargs)
        return ax


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

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        if has_spike_feature(self.data, "threshold_t"):
            idxs = slice(None) if selected_idxs is None else selected_idxs
            upstroke_t = self.lookup_spike_feature("upstroke_t")[idxs]
            upstroke_v = self.lookup_spike_feature("upstroke_v")[idxs]
            downstroke_t = self.lookup_spike_feature("downstroke_t")[idxs]
            downstroke_v = self.lookup_spike_feature("downstroke_v")[idxs]

            ax.plot(upstroke_t, upstroke_v, "x", label="upstroke", **kwargs)
            ax.plot(downstroke_t, downstroke_v, "x", label="upstroke", **kwargs)
        return ax


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

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        if has_spike_feature(self.data, "isi"):
            idxs = slice(None) if selected_idxs is None else selected_idxs
            thresh_t = self.lookup_spike_feature("threshold_t")[idxs]
            thresh_v = self.lookup_spike_feature("threshold_v")[idxs]
            isi = self.lookup_spike_feature("isi")[idxs]

            ax.hlines(
                thresh_v, thresh_t - isi, thresh_t, ls="--", label="isi", **kwargs
            )
            ax.plot(thresh_t, thresh_v, "x", **kwargs)
        return ax
