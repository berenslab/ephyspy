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
from matplotlib.axes import Axes

import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

import ephyspy.features.sweep_features as swft

# ransac = linear_model.RANSACRegressor()
from typing import Optional, Iterable

from ephyspy.utils import parse_desc, unpack

ransac = linear_model.LinearRegression()


from ephyspy.features.base import SweepSetFeature, SweepSetFeature
from ephyspy.features.utils import median_idx


def available_sweepset_features(compute_at_init=False, store_diagnostics=False):
    features = {
        "tau": SwS_Tau,
        "v_rest": SwS_V_rest,
        "v_baseline": SwS_V_baseline,
        "sag": SwS_Sag,
        "sag_ratio": SwS_Sag_ratio,
        "sag_fraction": SwS_Sag_fraction,
        "sag_area": SwS_Sag_area,
        "sag_time": SwS_Sag_time,
        "rebound": SwS_Rebound,
        "rebound_aps": SwS_Rebound_APs,
        "rebound_area": SwS_Rebound_area,
        "rebound_latency": SwS_Rebound_latency,
        "rebound_avg": SwS_Rebound_avg,
        "num_ap": SwS_Num_AP,
        "ap_freq": SwS_AP_freq,
        "wildness": SwS_Wildness,
        "ap_freq_adapt": SwS_AP_freq_adapt,
        "ap_amp_slope": SwS_AP_amp_slope,
        "isi_ff": SwS_ISI_FF,
        "isi_cv": SwS_ISI_CV,
        "ap_ff": SwS_AP_FF,
        "ap_cv": SwS_AP_CV,
        "isi": SwS_ISI,
        "burstiness": SwS_Burstiness,
        "num_bursts": SwS_Num_bursts,
        "isi_adapt": SwS_ISI_adapt,
        "isi_adapt_avg": SwS_ISI_adapt_avg,
        "ap_amp_adapt": SwS_AP_amp_adapt,
        "ap_amp_adapt_avg": SwS_AP_amp_adapt_avg,
        "ap_ahp": SwS_AP_AHP,
        "ap_adp": SwS_AP_ADP,
        "ap_thresh": SwS_AP_thresh,
        "ap_amp": SwS_AP_amp,
        "ap_width": SwS_AP_width,
        "ap_peak": SwS_AP_peak,
        "ap_trough": SwS_AP_trough,
        "ap_udr": SwS_AP_UDR,
        "r_input": SwS_R_input,
        "slow_hyperpolarization": Slow_hyperpolarization,
        "ap_latency": SwS_AP_latency,
        "dfdi": dfdI,
        "rheobase": Rheobase,
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


class APFeature(SweepSetFeature):
    """Obtain sweepset level single AP feature.

    This includes the following features:
    - AP threshold
    - AP amplitude
    - AP width
    - AP peak
    - AP trough
    - AP afterhyperpolarization (AHP)
    - AP afterdepolarization (ADP)
    - AP upstroke-to-downstroke ratio (UDR)
    """

    def __init__(self, feature, data=None, compute_at_init=True):
        super().__init__(feature, data=data, compute_at_init=compute_at_init)

    def _select(self, fts):
        """Select representative sweep and use its AP features to represent the
        entire sweepset.

        description: 2nd AP (if only 1 AP -> select first) during stimulus that has
        no NaNs in relevant spike features. If all APs have NaNs, return the AP during
        stimulus that has the least amount of NaNs in the relevant features. This
        avoids bad threshold detection at onset of stimulus.
        """
        # TODO: Consult if this is sensible!
        relevant_ap_fts = [
            "ap_thresh",
            "ap_amp",
            "ap_width",
            "ap_peak",
            "ap_trough",
            "ap_ahp",
            "ap_adp",
            "ap_udr",
        ]

        is_depol = self.lookup_sweep_feature("stim_amp") > 0
        has_spikes = self.lookup_sweep_feature("num_ap") > 0
        ft_is_na = np.zeros((len(relevant_ap_fts), len(self.dataset)), dtype=bool)
        for i, ft in enumerate(relevant_ap_fts):
            ft_is_na[i] = np.isnan(self.lookup_sweep_feature(ft))

        num_nans = pd.Series(ft_is_na.sum(axis=0))
        idx = num_nans[is_depol & has_spikes].idxmin()

        self._update_diagnostics(
            {"selected_idx": idx, "selection": parse_desc(self._select)}
        )
        return fts[idx]

    def _aggregate(self, fts):
        self._update_diagnostics(
            {"aggregation": "not an aggregate features, only single index is selected."}
        )
        return fts.item()


class ReboundFeature(SweepSetFeature):
    """Obtain sweepset level rebound related feature.

    This includes the following features:
    - rebound
    - rebound APs
    - rebound latency
    - average rebound
    - rebound area
    """

    def __init__(self, feature, data=None, compute_at_init=True):
        super().__init__(feature, data=data, compute_at_init=compute_at_init)

    def _select(self, fts):
        """Select representative sweep and use its rebound features to represent the
        entire sweepset.

        description: Lowest hyperpolarization sweep. If 3 lowest sweeps are NaN,
        then the first sweep is selected, meaning the feature is set to NaN."""
        rebound = self.lookup_sweep_feature("rebound")
        nan_rebounds = np.isnan(rebound)
        if all(nan_rebounds[:3]):
            idx = 0
        else:
            print(~nan_rebounds)
            idx = np.arange(len(rebound))[~nan_rebounds][0]

        self._update_diagnostics(
            {"selected_idx": idx, "selection": parse_desc(self._select)}
        )
        return fts[idx]

    def _aggregate(self, fts):
        self._update_diagnostics(
            {"aggregation": "not an aggregate features, only single index is selected."}
        )
        return fts.item()


class SagFeature(SweepSetFeature):
    """Obtain sweepset level sag related feature.

    This includes the following features:
    - sag
    - sag area
    - sag time
    - sag ratio
    - sag fraction"""

    def __init__(self, feature, data=None, compute_at_init=True):
        super().__init__(feature, data=data, compute_at_init=compute_at_init)

    def _select(self, fts):
        """Select representative sweep and use its sag features to represent the
        entire sweepset.

        description: Lowest hyperpolarization sweep that is not NaN. If 3 lowest
        sweeps are NaN, then the first sweep is selected, meaning the feature is set
        to NaN."""
        sag = self.lookup_sweep_feature("sag")
        nan_sags = np.isnan(sag)
        if all(nan_sags[:3]):
            idx = 0
        else:
            idx = np.arange(len(sag))[~nan_sags][0]

        self._update_diagnostics(
            {"selected_idx": idx, "selection": parse_desc(self._select)}
        )
        return fts[idx]

    def _aggregate(self, fts):
        self._update_diagnostics(
            {"aggregation": "not an aggregate features, only single index is selected."}
        )
        return fts.item()


class APsFeature(SweepSetFeature):
    """Obtain sweepset level spiking related feature.

    This includes the following features:
    - number of spikes
    - spike frequency
    - spike frequency adaptation (SFA)
    - spike amplitude slope
    - ISI fano factor
    - ISI AP fano factor
    - ISI CV
    - AP CV
    """

    def __init__(self, feature, data=None, compute_at_init=True):
        super().__init__(feature, data=data, compute_at_init=compute_at_init)

    def _select(self, fts):
        """Select representative sweep and use its spiking features to represent the
        entire sweepset.

        description: Highest non wild trace (wildness == cell dying)."""
        num_spikes = self.lookup_sweep_feature("num_ap")
        wildness = self.lookup_sweep_feature("wildness")
        is_non_wild = np.isnan(wildness)
        idx = pd.Series(num_spikes)[is_non_wild].idxmax()

        self._update_diagnostics(
            {
                "selected_idx": idx,
                "selection": parse_desc(self._select),
            }
        )
        return fts[idx]

    def _aggregate(self, fts):
        self._update_diagnostics(
            {"aggregation": "not an aggregate features, only single index is selected."}
        )
        return fts.item()


class MaxFeature(SweepSetFeature):
    """Obtain sweepset level maximum feature.

    This includes the following features:
    - number of bursts
    - wildness
    """

    def __init__(self, feature, data=None, compute_at_init=True):
        super().__init__(feature, data=data, compute_at_init=compute_at_init)

    def _select(self, fts):
        """Select representative sweep and use its features to represent the
        entire sweepset.

        description: select arg max."""
        fts = self.lookup_sweep_feature(self.name)
        idx = slice(0) if np.isnan(fts).all() else np.nanargmax(fts)
        self._update_diagnostics(
            {
                "selected_idx": idx,
                "selection": parse_desc(self._select),
            }
        )
        return np.array([float("nan")]) if np.isnan(fts).all() else fts[idx]

    def _aggregate(self, fts):
        self._update_diagnostics({"aggregation": "select max feature."})
        return fts.item()


class First5MedianFeature(SweepSetFeature):
    """Obtain sweepset level median feature.

    This includes the following features:
    - burstiness
    - ISI adaptation
    - average ISI adaptation
    - AP amplitude adaptation
    - average AP amplitude adaptation
    """

    def __init__(self, feature, data=None, compute_at_init=True):
        super().__init__(feature, data=data, compute_at_init=compute_at_init)

    def _select(self, fts):
        """Select representative sweep and use its features to represent the
        entire sweepset.

        description: select all features."""

        na_fts = np.isnan(fts)
        if not np.all(na_fts):
            first5 = fts[~na_fts][:5]
            where_value = np.where(~na_fts)[0][:5]
            self._update_diagnostics({"first5_idx": where_value})
            return first5

        self._update_diagnostics({"first5_idx": np.array([])})
        return np.array([])

    def _aggregate(self, fts):
        self._update_diagnostics({"aggregation": "select median feature."})
        if np.isnan(fts).all() or len(fts) == 0:
            self._update_diagnostics({"selected_idx": slice(0)})
            return float("nan")
        first5_idx = self.diagnostics["first5_idx"]
        self._update_diagnostics({"selected_idx": first5_idx[median_idx(fts)]})
        return np.nanmedian(fts).item()


class HyperpolMedianFeature(SweepSetFeature):
    """Obtain sweepset level hyperpolarization feature."""

    def __init__(self, feature, data=None, compute_at_init=True):
        super().__init__(feature, data=data, compute_at_init=compute_at_init)

    def _select(self, fts):
        """Select representative sweep and use its features to represent the
        entire sweepset.

        description: select all features."""
        is_hyperpol = self.lookup_sweep_feature("stim_amp") < 0
        where_value = np.where(is_hyperpol)[0]
        self._update_diagnostics({"hyperpol_idx": where_value})
        return fts[is_hyperpol]

    def _aggregate(self, fts):
        hyperpol_idx = self.diagnostics["hyperpol_idx"]
        self._update_diagnostics(
            {
                "aggregation": "select median feature.",
                "selected_idx": hyperpol_idx[median_idx(fts)],
            }
        )
        return np.nanmedian(fts).item()


class SwS_AP_latency(SweepSetFeature):
    """Obtain sweepset level AP latency feature."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_latency, data=data, compute_at_init=compute_at_init)

    def _select(self, fts):
        """Select representative sweep and use its sag features to represent the
        entire sweepset.

        description: first depolarization trace that has non-nan ap_latency."""
        is_depol = self.lookup_sweep_feature("stim_amp") > 0
        ap_latency = self.lookup_sweep_feature("ap_latency")
        idx = pd.Series(is_depol).index[is_depol & ~np.isnan(ap_latency)][0]
        self._update_diagnostics(
            {
                "selected_idx": idx,
                "selection": parse_desc(self._select),
            }
        )
        return fts[idx]

    def _aggregate(self, fts):
        self._update_diagnostics(
            {"aggregation": "not an aggregate features, only single index is selected."}
        )
        return fts.item()


class dfdI(SweepSetFeature):
    """Obtain sweepset level dfdI feature."""

    # TODO: Keep `feature` around as input for API consistency?
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.NullSweepFeature,
            data=data,
            compute_at_init=compute_at_init,
            name="dfdI",
        )

    def _select(self, fts):
        return fts

    def _aggregate(self, fts):
        return fts.item()

    def _compute(self, recompute=False, store_diagnostics=False):
        is_depol = self.lookup_sweep_feature("stim_amp", recompute=recompute) > 0
        ap_freq = self.lookup_sweep_feature("ap_freq", recompute=recompute)
        stim_amp = self.lookup_sweep_feature("stim_amp", recompute=recompute)

        f = ap_freq[is_depol]
        i = stim_amp[is_depol]

        dfdi = float("nan")
        has_spikes = ~np.isnan(f)
        # TODO: Check if this is a sensible idea!!!
        # (In case of 4 nans for example this will skip, even though sweep has spikes)
        if np.sum(has_spikes) > 4 and len(np.unique(f[:5])) > 3:
            i_s = i[has_spikes][:5]
            f_s = f[has_spikes][:5]

            ransac.fit(i_s.reshape(-1, 1), f_s.reshape(-1, 1))
            dfdi = ransac.coef_[0, 0]
            f_intercept = ransac.intercept_[0]

            if store_diagnostics:
                self._update_diagnostics(
                    {
                        "i_fit": i_s,
                        "f_fit": f_s,
                        "f": f,
                        "i": i,
                        "f_intercept": f_intercept,
                    }
                )
        return dfdi

    def plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        ax = plt.gca() if ax is None else ax
        if np.isnan(self.value):
            return ax

        if self.diagnostics is None:
            self.get_diagnostics(recompute=True)

        i_fit, f_fit, f_intercept = unpack(
            self.diagnostics, ["i_fit", "f_fit", "f_intercept"]
        )
        ax.plot(i_fit, f_fit, "o", label="f(I)", **kwargs)
        ax.plot(
            i_fit,
            self.value * i_fit + f_intercept,
            label="dfdi fit",
            **kwargs,
        )
        ax.set_xlabel("I (pA)")
        ax.set_ylabel("f (Hz)")
        ax.legend()

        return ax


class Rheobase(SweepSetFeature):
    """Obtain sweepset level rheobase feature."""

    def __init__(self, data=None, compute_at_init=True, dc_offset=0):
        self.dc_offset = dc_offset
        super().__init__(
            swft.NullSweepFeature,
            data=data,
            compute_at_init=compute_at_init,
            name="rheobase",
        )

    def _select(self, fts):
        return fts

    def _aggregate(self, fts):
        return fts.item()

    def _compute(self, recompute=False, store_diagnostics=False):
        dc_offset = self.dc_offset
        rheobase = float("nan")
        is_depol = self.lookup_sweep_feature("stim_amp", recompute=recompute) > 0
        ap_freq = self.lookup_sweep_feature("ap_freq", recompute=recompute)
        stim_amp = self.lookup_sweep_feature("stim_amp", recompute=recompute)
        dfdi = self.lookup_sweepset_feature("dfdi", recompute=recompute)

        f = ap_freq[is_depol]
        i = stim_amp[is_depol]

        has_spikes = ~np.isnan(f)
        # sometimes all depolarization traces spike
        i_sub = (
            0 if all(has_spikes) else i[~has_spikes][0]
        )  # last stim < spike threshold
        i_sup = i[has_spikes][0]  # first stim > spike threshold

        if not np.isnan(dfdi):
            rheobase = float(ransac.predict(np.array([[0]]))) / dfdi

            if rheobase < i_sub or rheobase > i_sup:
                rheobase = i_sup
        else:
            rheobase = i_sup
        rheobase -= dc_offset

        if store_diagnostics:
            self._update_diagnostics(
                {
                    "i_sub": i_sub,
                    "i_sup": i_sup,
                    "f_sup": f[has_spikes][0],
                    "dfdi": dfdi,
                    "dc_offset": dc_offset,
                }
            )
        return rheobase

    def plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        ax = plt.gca() if ax is None else ax
        if np.isnan(self.value):
            return ax

        if self.diagnostics is None:
            self.get_diagnostics(recompute=True)

        dfdi_ft = self.lookup_sweepset_feature("dfdi", return_value=False)

        i_sub, i_sup, f_sup, dc_offset = unpack(
            self.diagnostics, ["i_sub", "i_sup", "f_sup", "dc_offset"]
        )
        i_intercept = self.value
        dfdi = dfdi_ft.value

        if not np.isnan(dfdi):
            i, f, f_intercept = unpack(dfdi_ft.diagnostics, ["i", "f", "f_intercept"])
            has_spikes = ~np.isnan(f)
            n_no_spikes = np.sum(~has_spikes)

            ax.plot(i[has_spikes][:5], f[has_spikes][:5], "o", label="f(I)", **kwargs)
            ax.plot(
                i[: n_no_spikes + 5],
                dfdi * i[: n_no_spikes + 5] + f_intercept,
                label="f(I) fit",
                **kwargs,
            )
            ax.set_xlim(i[0] - 5, i[n_no_spikes + 5] + 5)
        else:
            ax.set_xlim(i_sub - 5, i_sup + 5)

        ax.plot(i_sup, f_sup, "o", label="i_sup", **kwargs)
        ax.axvline(
            i_intercept + dc_offset, ls="--", label="rheobase\n(w.o. dc)", **kwargs
        )
        ax.axvline(i_intercept, label="rheobase\n(incl. dc)", **kwargs)
        ax.plot(i_sub, 0, "o", label="i_sub", **kwargs)

        ax.set_xlabel("I (pA)")
        ax.set_ylabel("f (Hz)")
        ax.legend()
        return ax


class SwS_R_input(SweepSetFeature):
    """Obtain sweepset level r_input feature."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.NullSweepFeature,
            data=data,
            compute_at_init=compute_at_init,
            name="r_input",
        )

    def _select(self, fts):
        return fts

    def _aggregate(self, fts):
        return fts.item()

    def _compute(self, recompute=False, store_diagnostics=False):
        r_input = float("nan")
        is_hyperpol = self.lookup_sweep_feature("stim_amp", recompute=recompute) < 0
        v_deflect = self.lookup_sweep_feature("v_deflect", recompute=recompute)
        v_deflect = v_deflect[is_hyperpol].reshape(-1, 1)
        i_amp = self.lookup_sweep_feature("stim_amp", recompute=recompute)
        i_amp = i_amp[is_hyperpol].reshape(-1, 1)

        if len(v_deflect) >= 3:
            ransac.fit(i_amp, v_deflect)
            r_input = ransac.coef_[0, 0] * 1000
            v_intercept = ransac.intercept_[0]
            if store_diagnostics:
                self._update_diagnostics(
                    {
                        "raw_slope": r_input / 1000,
                        "v_intercept": v_intercept,
                        "i_amp": i_amp,
                        "v_deflect": v_deflect,
                    }
                )
        return r_input

    def plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        ax = plt.gca() if ax is None else ax
        if np.isnan(self.value):
            return ax

        if self.diagnostics is None:
            self.get_diagnostics(recompute=True)

        if not np.isnan(self.value):
            i, v, slope, intercept = unpack(
                self.diagnostics, ["i_amp", "v_deflect", "raw_slope", "v_intercept"]
            )
            ax.plot(i, v, "o", label="V(I)", **kwargs)
            ax.plot(i, slope * i + intercept, label="r_input fit", **kwargs)
            ax.set_xlim(np.min(i) - 5, np.max(i) + 5)
            ax.set_xlabel("I (pA)")
            ax.set_ylabel("V (mV)")
            ax.legend()
        return ax


class Slow_hyperpolarization(SweepSetFeature):
    """Obtain sweepset level slow_hyperpolarization feature."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.NullSweepFeature,
            data=data,
            compute_at_init=compute_at_init,
            name="slow_hyperpolarization",
        )

    def _select(self, fts):
        return fts

    def _aggregate(self, fts):
        return fts.item()

    def _compute(self, recompute=False, store_diagnostics=False):
        # is_hyperpol = self.lookup_sweep_feature("stim_amp", recompute=recompute) < 0
        # TODO: ASK IF THIS IS ONLY TAKEN FOR HYPERPOLARIZING TRACES (I THINK NOT)
        v_baseline = self.lookup_sweep_feature("v_baseline", recompute=recompute)

        slow_hyperpolarization = v_baseline.max() - v_baseline.min()

        if store_diagnostics:
            self._update_diagnostics(
                {
                    "v_baseline": v_baseline,
                    "v_baseline_max": v_baseline.max(),
                    "v_baseline_min": v_baseline.min(),
                }
            )
        return slow_hyperpolarization

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        v_max, v_min = unpack(self.diagnostics, ["v_baseline_max", "v_baseline_min"])
        ax.vlines(0.05, v_min, v_max, linestyle="--", label=self.name)
        return ax


class NullSweepSetFeature(SweepSetFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.NullSweepFeature,
            data=data,
            compute_at_init=compute_at_init,
            name="null_sweepset_feature",
        )

    def _select(self, fts):
        return fts

    def _aggregate(self, fts):
        return fts.item()

    def _compute(self, recompute=False, store_diagnostics=False):
        return None

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        return ax


class SwS_Tau(HyperpolMedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Tau, data=data, compute_at_init=compute_at_init)


class SwS_V_rest(HyperpolMedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.V_rest, data=data, compute_at_init=compute_at_init)


class SwS_V_baseline(HyperpolMedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.V_baseline, data=data, compute_at_init=compute_at_init)


class SwS_Sag(SagFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sag, data=data, compute_at_init=compute_at_init)


class SwS_Sag_ratio(SagFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sag_ratio, data=data, compute_at_init=compute_at_init)


class SwS_Sag_fraction(SagFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sag_fraction, data=data, compute_at_init=compute_at_init)


class SwS_Sag_area(SagFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sag_area, data=data, compute_at_init=compute_at_init)


class SwS_Sag_time(SagFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sag_time, data=data, compute_at_init=compute_at_init)


class SwS_Rebound(ReboundFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Rebound, data=data, compute_at_init=compute_at_init)


class SwS_Rebound_APs(SweepSetFeature):
    """Obtain sweepset level rebound APs feature."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Rebound_APs, data=data, compute_at_init=compute_at_init)

    def _select(self, fts):
        """Select representative sweep and use its rebound features to represent the
        entire sweepset.

        description: Max rebound of the 3 lowest sweeps hyperpolarization sweeps.
        """
        num_rebound = self.lookup_sweep_feature("rebound_aps")
        nan_rebounds = np.isnan(num_rebound)
        if all(nan_rebounds[:3]):
            idx = 0
        else:
            idx = np.nanargmax(num_rebound[:3])

        self._update_diagnostics(
            {"selected_idx": idx, "selection": parse_desc(self._select)}
        )
        return fts[idx]

    def _aggregate(self, fts):
        self._update_diagnostics(
            {"aggregation": "not an aggregate features, only single index is selected."}
        )
        return fts.item()


class SwS_Rebound_area(ReboundFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Rebound_area, data=data, compute_at_init=compute_at_init)


class SwS_Rebound_latency(ReboundFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Rebound_latency, data=data, compute_at_init=compute_at_init
        )


class SwS_Rebound_avg(ReboundFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Rebound_avg, data=data, compute_at_init=compute_at_init)


class SwS_Num_AP(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Num_AP, data=data, compute_at_init=compute_at_init)


class SwS_AP_freq(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_freq, data=data, compute_at_init=compute_at_init)


class SwS_AP_freq_adapt(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_freq_adapt, data=data, compute_at_init=compute_at_init)


class SwS_AP_amp_slope(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_amp_slope, data=data, compute_at_init=compute_at_init)


class SwS_ISI_FF(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.ISI_FF, data=data, compute_at_init=compute_at_init)


class SwS_AP_FF(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_FF, data=data, compute_at_init=compute_at_init)


class SwS_ISI_CV(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.ISI_CV, data=data, compute_at_init=compute_at_init)


class SwS_AP_CV(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_CV, data=data, compute_at_init=compute_at_init)


class SwS_ISI(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.ISI, data=data, compute_at_init=compute_at_init)


class SwS_Burstiness(First5MedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Burstiness, data=data, compute_at_init=compute_at_init)


class SwS_Num_bursts(First5MedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Num_bursts, data=data, compute_at_init=compute_at_init)


class SwS_ISI_adapt(First5MedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.ISI_adapt, data=data, compute_at_init=compute_at_init)


class SwS_ISI_adapt_avg(First5MedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.ISI_adapt_avg, data=data, compute_at_init=compute_at_init)


class SwS_AP_amp_adapt(First5MedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_amp_adapt, data=data, compute_at_init=compute_at_init)


class SwS_AP_amp_adapt_avg(First5MedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.AP_amp_adapt_avg, data=data, compute_at_init=compute_at_init
        )


class SwS_AP_AHP(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_AHP, data=data, compute_at_init=compute_at_init)


class SwS_AP_ADP(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_ADP, data=data, compute_at_init=compute_at_init)


class SwS_AP_thresh(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_thresh, data=data, compute_at_init=compute_at_init)


class SwS_AP_amp(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_amp, data=data, compute_at_init=compute_at_init)


class SwS_AP_width(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_width, data=data, compute_at_init=compute_at_init)


class SwS_AP_peak(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_peak, data=data, compute_at_init=compute_at_init)


class SwS_AP_trough(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_trough, data=data, compute_at_init=compute_at_init)


class SwS_AP_UDR(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.AP_UDR, data=data, compute_at_init=compute_at_init)


class SwS_Wildness(MaxFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Wildness, data=data, compute_at_init=compute_at_init)
