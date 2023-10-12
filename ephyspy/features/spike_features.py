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
from ephyspy.utils import (
    fwhm,
    has_spike_feature,
    is_spike_feature,
    scatter_spike_ft,
    unpack,
)


def available_spike_features(**kwargs) -> Dict[str, SpikeFeature]:
    """Return a dictionary of all implemented spike features.

    Looks for all classes that inherit from SpikeFeature and returns a dictionary
    of all available features. If compute_at_init is True, the features are
    computed at initialization.

    Returns:
        dict[str, SpikeFeature]: Dictionary of all available spike features.
    """
    all_features = fetch_available_fts()
    features = {ft.__name__.lower(): ft for ft in all_features if is_spike_feature(ft)}
    features = {k.replace("spike_", ""): v for k, v in features.items()}
    if len(kwargs) > 0:
        return {
            k: lambda *default_args, **default_kwargs: v(
                *default_args,
                **default_kwargs,
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
    units: mV/s.
    """

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        upstroke = self.lookup_spike_feature("upstroke", recompute=recompute)
        upstroke_v = self.lookup_spike_feature("upstroke_v", recompute=recompute)
        upstroke_t = self.lookup_spike_feature("upstroke_t", recompute=recompute)
        upstroke_idx = self.lookup_spike_feature("upstroke_index", recompute=recompute)
        if store_diagnostics:
            self._update_diagnostics(
                {
                    "upstroke_t": upstroke_t,
                    "upstroke_idx": upstroke_idx,
                    "upstroke_v": upstroke_v,
                }
            )
        return upstroke

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        idxs = slice(None) if selected_idxs is None else selected_idxs
        up_t, up_v = unpack(self.diagnostics, ["upstroke_t", "upstroke_v"])
        up_dvdt = self.value * 1e3

        T = 15e-5
        t = np.linspace(up_t[idxs] - T, up_t[idxs] + T, 2)
        ax = scatter_spike_ft(
            "upstroke", self.data, ax=ax, selected_idxs=selected_idxs, **kwargs
        )
        kwargs["color"] = next(ax._get_lines.prop_cycler)["color"]
        ax.plot(t, up_dvdt[idxs] * (t - up_t[idxs]) + up_v[idxs], **kwargs)
        return ax


class Spike_AP_downstroke(SpikeFeature):
    """Extract spike level downstroke feature.

    depends on: /.
    description: downstroke of AP.
    units: mV/s.
    """

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        downstroke = self.lookup_spike_feature("downstroke", recompute=recompute)
        downstroke_t = self.lookup_spike_feature("downstroke_t", recompute=recompute)
        downstroke_v = self.lookup_spike_feature("downstroke_v", recompute=recompute)
        downstroke_idx = self.lookup_spike_feature(
            "downstroke_index", recompute=recompute
        )
        peak_t = self.lookup_spike_feature("peak_t", recompute=recompute)
        trough_t = self.lookup_spike_feature("fast_trough_t", recompute=recompute)
        if store_diagnostics:
            self._update_diagnostics(
                {
                    "downstroke_t": downstroke_t,
                    "downstroke_v": downstroke_v,
                    "downstroke_idx": downstroke_idx,
                }
            )
        return downstroke

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        idxs = slice(None) if selected_idxs is None else selected_idxs
        down_t, down_v = unpack(self.diagnostics, ["downstroke_t", "downstroke_v"])
        down_dvdt = self.value * 1e3

        T = 25e-5
        t = np.linspace(down_t[idxs] - T, down_t[idxs] + T, 2)
        ax = scatter_spike_ft(
            "downstroke", self.data, ax=ax, selected_idxs=selected_idxs, **kwargs
        )
        kwargs["color"] = next(ax._get_lines.prop_cycler)["color"]
        ax.plot(t, down_dvdt[idxs] * (t - down_t[idxs]) + down_v[idxs], **kwargs)
        return ax


class Spike_AP_fast_trough(SpikeFeature):
    """Extract spike level fast trough feature.

    depends on: /.
    description: fast trough of AP.
    units: mV.
    """

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        fast_trough = self.lookup_spike_feature("fast_trough_v", recompute=recompute)
        fast_trough_i = self.lookup_spike_feature("fast_trough_i", recompute=recompute)
        fast_trough_t = self.lookup_spike_feature("fast_trough_t", recompute=recompute)
        fast_trough_idx = self.lookup_spike_feature(
            "fast_trough_index", recompute=recompute
        )
        if store_diagnostics:
            self._update_diagnostics(
                {
                    "fast_trough_t": fast_trough_t,
                    "fast_trough_i": fast_trough_i,
                    "fast_trough_idx": fast_trough_idx,
                }
            )
        return fast_trough

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

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        slow_trough = self.lookup_spike_feature("slow_trough_v", recompute=recompute)
        slow_trough_i = self.lookup_spike_feature("slow_trough_i", recompute=recompute)
        slow_trough_t = self.lookup_spike_feature("slow_trough_t", recompute=recompute)
        slow_trough_idx = self.lookup_spike_feature(
            "slow_trough_index", recompute=recompute
        )
        if store_diagnostics:
            self._update_diagnostics(
                {
                    "slow_trough_t": slow_trough_t,
                    "slow_trough_i": slow_trough_i,
                    "slow_trough_idx": slow_trough_idx,
                }
            )
        return slow_trough

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

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        peak_v = self.lookup_spike_feature("peak_v", recompute=recompute)
        peak_t = self.lookup_spike_feature("peak_t", recompute=recompute)
        threshold_v = self.lookup_spike_feature("threshold_v", recompute=recompute)
        peak_height = peak_v - threshold_v

        if store_diagnostics:
            self._update_diagnostics(
                {
                    "peak_v": peak_v,
                    "peak_t": peak_t,
                    "threshold_v": threshold_v,
                }
            )

        return peak_height if len(peak_v) > 0 else np.array([], dtype=int)

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        if has_spike_feature(self.data, "threshold_v"):
            idxs = slice(None) if selected_idxs is None else selected_idxs
            thresh_v, peak_t, peak_v = unpack(
                self.diagnostics, ["threshold_v", "peak_t", "peak_v"]
            )

            ax.plot(peak_t[idxs], peak_v[idxs], "x", **kwargs)
            ax.vlines(
                peak_t[idxs],
                thresh_v[idxs],
                peak_v[idxs],
                ls="--",
                label="ap_amp",
                **kwargs,
            )
        return ax


class Spike_AP_AHP(SpikeFeature):
    """Extract spike level after hyperpolarization feature.

    depends on: threshold_v, fast_trough_v.
    description: v_fast_trough - threshold_v.
    units: mV.
    """

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_fast_trough = self.lookup_spike_feature("fast_trough_v", recompute=recompute)
        t_fast_trough = self.lookup_spike_feature("fast_trough_t", recompute=recompute)
        threshold_v = self.lookup_spike_feature("threshold_v", recompute=recompute)
        threshold_t = self.lookup_spike_feature("threshold_t", recompute=recompute)

        if store_diagnostics:
            self._update_diagnostics(
                {
                    "fast_trough_v": v_fast_trough,
                    "fast_trough_t": t_fast_trough,
                    "threshold_v": threshold_v,
                    "threshold_t": threshold_t,
                }
            )
        return v_fast_trough - threshold_v

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        if has_spike_feature(self.data, "ap_ahp"):
            idxs = slice(None) if selected_idxs is None else selected_idxs

            trough_t, trough_v, threshold_t, threshold_v = unpack(
                self.diagnostics,
                ["fast_trough_t", "fast_trough_v", "threshold_t", "threshold_v"],
            )
            ax.vlines(
                0.5 * (trough_t[idxs] + threshold_t[idxs]),
                trough_v[idxs],
                threshold_v[idxs],
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

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_adp = self.lookup_spike_feature("adp_v", recompute=recompute)
        t_adp = self.lookup_spike_feature("adp_t", recompute=recompute)
        i_adp = self.lookup_spike_feature("adp_i", recompute=recompute)
        idx_adp = self.lookup_spike_feature("adp_index", recompute=recompute)
        v_fast_trough = self.lookup_spike_feature("fast_trough_v", recompute=recompute)
        t_fast_trough = self.lookup_spike_feature("fast_trough_t", recompute=recompute)

        if store_diagnostics:
            self._update_diagnostics(
                {
                    "adp_v": v_adp,
                    "fast_trough_v": v_fast_trough,
                    "fast_trough_t": t_fast_trough,
                    "adp_t": t_adp,
                    "adp_i": i_adp,
                    "adp_idx": idx_adp,
                }
            )
        return v_adp - v_fast_trough

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        if has_spike_feature(self.data, "ap_adp"):
            idxs = slice(None) if selected_idxs is None else selected_idxs

            adp_t, adp_v, trough_t, trough_v = unpack(
                self.diagnostics,
                ["adp_t", "adp_v", "fast_trough_t", "fast_trough_v"],
            )

            ax.vlines(
                0.5 * (adp_t[idxs] + trough_t[idxs]),
                adp_v[idxs],
                trough_v[idxs],
                ls="--",
                lw=1,
                label="adp",
                **kwargs,
            )
        return ax


class Spike_AP_ADP_trough(SpikeFeature):
    """Extract spike level after depolarization feature.

    depends on: adp_v.
    description: |v_adp|.
    units: mV.
    """

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_adp = self.lookup_spike_feature("adp_v", recompute=recompute)
        t_adp = self.lookup_spike_feature("adp_t", recompute=recompute)
        i_adp = self.lookup_spike_feature("adp_i", recompute=recompute)
        idx_adp = self.lookup_spike_feature("adp_index", recompute=recompute)

        if store_diagnostics:
            self._update_diagnostics(
                {
                    "adp_v": v_adp,
                    "adp_t": t_adp,
                    "adp_i": i_adp,
                    "adp_idx": idx_adp,
                }
            )
        return np.abs(v_adp)

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        if has_spike_feature(self.data, "ap_adp"):
            idxs = slice(None) if selected_idxs is None else selected_idxs

            adp_t, adp_v = unpack(self.diagnostics, ["adp_t", "adp_v"])

            ax.vlines(
                adp_t[idxs],
                adp_v[idxs],
                0,
                ls="--",
                lw=1,
                label="adp trough",
                **kwargs,
            )
        return ax


class Spike_AP_peak(SpikeFeature):
    """Extract spike level peak feature.

    depends on: peak_v.
    description: max voltage of AP.
    units: mV.
    """

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_peak = self.lookup_spike_feature("peak_v", recompute=recompute)
        t_peak = self.lookup_spike_feature("peak_t", recompute=recompute)
        i_peak = self.lookup_spike_feature("peak_i", recompute=recompute)
        idx_peak = self.lookup_spike_feature("peak_index", recompute=recompute)

        if store_diagnostics:
            self._update_diagnostics(
                {
                    "peak_t": t_peak,
                    "peak_i": i_peak,
                    "peak_idx": idx_peak,
                }
            )

        return v_peak

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        return scatter_spike_ft(
            "peak", self.data, ax=ax, selected_idxs=selected_idxs, **kwargs
        )


class Spike_AP_overshoot(SpikeFeature):
    """Extract spike level overshoot feature.

    depends on: peak_v.
    description: max voltage of AP above 0 mV.
    units: mV.
    """

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_peak = self.lookup_spike_feature("peak_v", recompute=recompute)
        t_peak = self.lookup_spike_feature("peak_t", recompute=recompute)
        i_peak = self.lookup_spike_feature("peak_i", recompute=recompute)
        idx_peak = self.lookup_spike_feature("peak_index", recompute=recompute)
        v_peak[v_peak < 0] = float("nan")

        if store_diagnostics:
            self._update_diagnostics(
                {
                    "peak_t": t_peak,
                    "peak_i": i_peak,
                    "peak_idx": idx_peak,
                }
            )

        return v_peak

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        return scatter_spike_ft(
            "overshoot", self.data, ax=ax, selected_idxs=selected_idxs, **kwargs
        )


class Spike_AP_thresh(SpikeFeature):
    """Extract spike level ap threshold feature.

    depends on: threshold_v.
    description: For details on how AP thresholds are computed see AllenSDK.
    units: mV.
    """

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_thresh = self.lookup_spike_feature("threshold_v", recompute=recompute)
        t_thresh = self.lookup_spike_feature("threshold_t", recompute=recompute)
        i_thresh = self.lookup_spike_feature("threshold_i", recompute=recompute)
        idx_thresh = self.lookup_spike_feature("threshold_index", recompute=recompute)

        if store_diagnostics:
            self._update_diagnostics(
                {
                    "threshold_t": t_thresh,
                    "threshold_i": i_thresh,
                    "threshold_idx": idx_thresh,
                }
            )
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

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        v_thresh = self.lookup_spike_feature("trough_v", recompute=recompute)
        t_thresh = self.lookup_spike_feature("trough_t", recompute=recompute)
        i_thresh = self.lookup_spike_feature("trough_i", recompute=recompute)
        idx_thresh = self.lookup_spike_feature("trough_index", recompute=recompute)

        if store_diagnostics:
            self._update_diagnostics(
                {
                    "trough_t": t_thresh,
                    "trough_i": i_thresh,
                    "trough_idx": idx_thresh,
                }
            )
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

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        width = self.lookup_spike_feature("width", recompute=recompute)
        trough_idxs = self.lookup_spike_feature("trough_index").astype(int)
        spike_idxs = self.lookup_spike_feature("threshold_index").astype(int)
        peak_idxs = self.lookup_spike_feature("peak_index").astype(int)

        if store_diagnostics:
            self._update_diagnostics(
                {
                    "trough_idx": trough_idxs,
                    "spike_idx": spike_idxs,
                    "peak_idx": peak_idxs,
                }
            )
        return width

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        if has_spike_feature(self.data, "width"):
            idxs = slice(None) if selected_idxs is None else selected_idxs

            # the following is adapted from `allen_sdk.ephys_features.find_widths`
            trough_idxs, spike_idxs, peak_idxs = unpack(
                self.diagnostics, ["trough_idx", "spike_idx", "peak_idx"]
            )

            t = self.data.t
            v = self.data.v

            ap_height = v[peak_idxs] - v[trough_idxs]
            trough_fwhm = ap_height / 2.0 + v[trough_idxs]

            thresh_fwhm = (v[peak_idxs] - v[spike_idxs]) / 2.0 + v[spike_idxs]

            # Some spikes in burst may have deep trough but short height, so can't use same
            # definition for width
            fwhm = trough_fwhm.copy()
            fwhm[trough_fwhm < v[spike_idxs]] = thresh_fwhm[trough_fwhm < v[spike_idxs]]

            width_idx = np.array(
                [
                    pk - np.flatnonzero(v[pk:spk:-1] <= wl)[0]
                    if np.flatnonzero(v[pk:spk:-1] <= wl).size > 0
                    else np.nan
                    for pk, spk, wl in zip(
                        peak_idxs,
                        spike_idxs,
                        fwhm,
                    )
                ]
            ).astype(int)

            fwhm = fwhm[idxs]
            width_t = t[width_idx][idxs]
            width = self.lookup_spike_feature("width")[idxs]
            ax.hlines(fwhm, width_t, width_t + width, label="width", ls="--", **kwargs)
        return ax


class Spike_AP_UDR(SpikeFeature):
    """Extract spike level ap udr feature.

    depends on: upstroke, downstroke.
    description: upstroke / downstroke. For details on how upstroke, downstroke
    are computed see AllenSDK.
    units: /.
    """

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        upstroke = self.lookup_spike_feature("upstroke", recompute=recompute)
        upstroke_t = self.lookup_spike_feature("upstroke_t", recompute=recompute)
        upstroke_v = self.lookup_spike_feature("upstroke_v", recompute=recompute)
        downstroke = self.lookup_spike_feature("downstroke", recompute=recompute)
        downstroke_t = self.lookup_spike_feature("downstroke_t", recompute=recompute)
        downstroke_v = self.lookup_spike_feature("downstroke_v", recompute=recompute)

        if store_diagnostics:
            self._update_diagnostics(
                {
                    "upstroke": upstroke,
                    "upstroke_v": upstroke_v,
                    "upstroke_t": upstroke_t,
                    "downstroke": downstroke,
                    "downstroke_v": downstroke_v,
                    "downstroke_t": downstroke_t,
                }
            )
        return upstroke / -downstroke

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        if has_spike_feature(self.data, "threshold_t"):
            idxs = slice(None) if selected_idxs is None else selected_idxs

            up_t, up_v, down_t, down_v = unpack(
                self.diagnostics,
                ["upstroke_t", "upstroke_v", "downstroke_t", "downstroke_v"],
            )

            ax.plot(up_t[idxs], up_v[idxs], "x", label="upstroke", **kwargs)
            ax.plot(down_t[idxs], down_v[idxs], "x", label="upstroke", **kwargs)
        return ax


class Spike_ISI(SpikeFeature):
    """Extract spike level inter-spike-interval feature.

    depends on: threshold_t.
    description: The distance between subsequent spike thresholds. isi at the
        first index is nan since isi[t+1] = threshold_t[t+1] - threshold_t[t].
    units: s.
    """

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        isi = np.array([], dtype=int)
        spike_times = self.lookup_spike_feature("threshold_t", recompute=recompute)
        spike_thresh = self.lookup_spike_feature("threshold_v", recompute=recompute)
        if len(spike_times) > 1:
            isi = np.diff(spike_times)
            isi = np.insert(isi, 0, 0)
        elif len(spike_times) == 1:
            isi = np.array([float("nan")])

        if store_diagnostics:
            self._update_diagnostics(
                {
                    "spike_times": spike_times,
                    "spike_thresh": spike_thresh,
                    "isi": isi,
                }
            )
        return isi

    def _plot(self, ax: Optional[Axes] = None, selected_idxs=None, **kwargs) -> Axes:
        if has_spike_feature(self.data, "isi"):
            idxs = slice(None) if selected_idxs is None else selected_idxs

            thresh_t, thresh_v, isi = unpack(
                self.diagnostics, ["spike_times", "spike_thresh", "isi"]
            )
            thresh_t = thresh_t[idxs]
            thresh_v = thresh_v[idxs]
            isi = isi[idxs]

            ax.hlines(
                thresh_v, thresh_t - isi, thresh_t, ls="--", label="isi", **kwargs
            )
            ax.plot(thresh_t, thresh_v, "x", **kwargs)
        return ax
