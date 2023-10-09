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

# ransac = linear_model.RANSACRegressor()
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from numpy import ndarray
from sklearn import linear_model

import ephyspy.features.sweep_features as swft
from ephyspy.utils import parse_desc, stimulus_type, unpack, where_between

ransac = linear_model.LinearRegression()


from ephyspy.features.base import SweepSetFeature
from ephyspy.features.utils import fetch_available_fts, median_idx
from ephyspy.utils import is_sweepset_feature


def available_sweepset_features(
    compute_at_init: bool = False, store_diagnostics: bool = False
) -> Dict[str, SweepSetFeature]:
    """Return a dictionary of all implemented sweepset features.

    Looks for all classes that inherit from SweepSetFeature and returns a dictionary
    of all available features. If compute_at_init is True, the features are
    computed at initialization.

    Args:
        compute_at_init (bool, optional): If True, the features are computed at
            initialization. Defaults to False.
        store_diagnostics (bool, optional): If True, the features are computed
            with diagnostics. Defaults to False.

    Returns:
        dict[str, SweepSetFeature]: Dictionary of all available spike features.
    """
    all_features = fetch_available_fts()
    features = {
        ft.__name__.lower(): ft for ft in all_features if is_sweepset_feature(ft)
    }
    features = {k.replace("sweepset_", ""): v for k, v in features.items()}
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

        description: first sweep depolarization sweep with aps.
        """

        if stimulus_type(self.data) == "long_square":
            is_depol = self.lookup_sweep_feature("stim_amp") > 0
            has_spikes = self.lookup_sweep_feature("num_ap") > 0
            peaks_to_low = np.all(self.lookup_sweep_feature("ap_peak") < -30)

            if not peaks_to_low:
                idx = np.where(is_depol & has_spikes)[0][0]
            else:
                idx = np.array([], dtype=int)
        elif stimulus_type(self.data) == "ramp":
            ap_peak = self.lookup_sweep_feature("ap_peak")
            where_peak = ~np.isnan(ap_peak)
            if np.any(where_peak):
                idx = np.where(where_peak)[0][0]
            else:
                idx = np.array([], dtype=int)
        else:
            idx = np.array([], dtype=int)

        self._update_diagnostics(
            {"selected_idx": idx, "selection": parse_desc(self._select)}
        )
        return fts[idx]


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

        description: Lowest hyperpolarization sweep.
        """
        idx = 0  # always return lowest hyperpolarization sweep

        self._update_diagnostics(
            {"selected_idx": idx, "selection": parse_desc(self._select)}
        )
        return fts[idx]


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

        description: Lowest hyperpolarization sweep.
        """
        idx = 0

        self._update_diagnostics(
            {"selected_idx": idx, "selection": parse_desc(self._select)}
        )
        return fts[idx]


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

        description: Highest non wild trace (wildness == cell dying).
        """
        num_spikes = self.lookup_sweep_feature("num_ap")
        wildness = self.lookup_sweep_feature("wildness")
        is_non_wild = np.isnan(wildness)
        idx = pd.Series(num_spikes)[is_non_wild].idxmax()
        idx = np.array([], dtype=int) if np.isnan(idx) else idx

        self._update_diagnostics(
            {
                "selected_idx": idx,
                "selection": parse_desc(self._select),
            }
        )
        return fts[idx]


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

        description: select first 5 none nan features.
        """
        na_fts = np.isnan(fts)
        if not np.all(na_fts):
            first5 = fts[~na_fts][:5]
            where_value = np.where(~na_fts)[0][:5]
            self._update_diagnostics({"first5_idx": where_value})
            return first5

        self._update_diagnostics({"first5_idx": np.array([], dtype=int)})
        return np.array([], dtype=int)

    def _aggregate(self, fts):
        """Compute aggregate metric on subset of sweeps.

        description: compute the median.
        """
        self._update_diagnostics({"aggregation": "select median feature."})
        if np.isnan(fts).all() or len(fts) == 0:
            self._update_diagnostics({"selected_idx": slice(0)})
            return float("nan")
        first5_idx = self.diagnostics["first5_idx"]
        self._update_diagnostics({"selected_idx": first5_idx[median_idx(fts)]})
        med = float("nan") if len(fts) == 0 else np.nanmedian(fts).item()
        return med


class HyperpolMedianFeature(SweepSetFeature):
    """Obtain sweepset level hyperpolarization feature."""

    def __init__(self, feature, data=None, compute_at_init=True):
        super().__init__(feature, data=data, compute_at_init=compute_at_init)

    def _select(self, fts):
        """Select representative sweep and use its features to represent the
        entire sweepset.

        description: select all hyperpolarizing sweeps.
        """
        is_hyperpol = self.lookup_sweep_feature("stim_amp") < 0
        where_value = np.where(is_hyperpol)[0]
        self._update_diagnostics({"hyperpol_idx": where_value})
        return fts[is_hyperpol]

    def _aggregate(self, fts):
        """Compute aggregate metric on subset of sweeps.

        description: compute the median.
        """
        hyperpol_idx = self.diagnostics["hyperpol_idx"]
        self._update_diagnostics(
            {
                "aggregation": "median.",
                "selected_idx": hyperpol_idx[median_idx(fts)],
            }
        )
        med = float("nan") if len(fts) == 0 else np.nanmedian(fts).item()
        return med


class SweepSet_AP_latency(SweepSetFeature):
    """Obtain sweepset level AP latency feature."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_AP_latency, data=data, compute_at_init=compute_at_init
        )

    def _select(self, fts):
        """Select representative sweep and use its sag features to represent the
        entire sweepset.

        description: first depolarization trace that has spikes.
        """
        is_depol = self.lookup_sweep_feature("stim_amp") > 0
        ap_latency = self.lookup_sweep_feature("ap_latency")
        idxs = pd.Series(is_depol).index[is_depol & ~np.isnan(ap_latency)]
        if len(idxs) > 0:
            idx = idxs[0]
        else:
            idx = np.array([], dtype=int)
        self._update_diagnostics(
            {
                "selected_idx": idx,
                "selection": parse_desc(self._select),
            }
        )
        return fts[idx]


class SweepSet_AP_latency_20pA(SweepSetFeature):
    """Obtain sweepset level AP latency feature at one stimulus above the first
    one that spikes."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_AP_latency,
            data=data,
            compute_at_init=compute_at_init,
            name="ap_latency_20pA",
        )

    def _select(self, fts):
        """Select representative sweep and use its sag features to represent the
        entire sweepset.

        description: 2nd depolarization trace that has spikes.
        """
        is_depol = self.lookup_sweep_feature("stim_amp") > 0
        ap_latency = self.lookup_sweep_feature("ap_latency")
        idxs = pd.Series(is_depol).index[is_depol & ~np.isnan(ap_latency)]
        if len(idxs) > 1:
            idx = idxs[1]
        else:
            idx = np.array([], dtype=int)
        self._update_diagnostics(
            {
                "selected_idx": idx,
                "selection": parse_desc(self._select),
            }
        )
        return fts[idx]


class SweepSet_dfdI(SweepSetFeature):
    """Obtain sweepset level dfdI feature.

    description: The slope of the linear fit of the first 5 depolarizing current
    injections. It is computed by fitting a line to the first 5 depolarizing
    current injections and finding the slope.
    depends on: Sweep_AP_freq, Sweep_Stim_amp.
    units: Hz/pA."""

    # TODO: Keep `feature` input arg around for API consistency?
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.NullSweepFeature,
            data=data,
            compute_at_init=compute_at_init,
            name="dfdI",
        )
        self.parse_docstring()

    def _compute(self, recompute=False, store_diagnostics=False):
        dfdi = float("nan")

        if stimulus_type(self.data) == "long_square":
            is_depol = self.lookup_sweep_feature("stim_amp", recompute=recompute) > 0
            ap_freq = self.lookup_sweep_feature("ap_freq", recompute=recompute)
            stim_amp = self.lookup_sweep_feature("stim_amp", recompute=recompute)

            f = ap_freq[is_depol]
            i = stim_amp[is_depol]

            sweep_w_spikes = ~np.isnan(f)
            # TODO: Check if this is a sensible idea!!!
            # (In case of 4 nans for example this will skip, even though sweep has spikes)
            if np.sum(sweep_w_spikes) > 4 and len(np.unique(f[:5])) > 3:
                i_s = i[sweep_w_spikes][:5]
                f_s = f[sweep_w_spikes][:5]

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


class SweepSet_Rheobase(SweepSetFeature):
    """Obtain sweepset level rheobase feature.

    description: The minimum current amplitude required to elicit an action
    potential. It is computed by fitting a line to the first 5 depolarizing
    current injections and finding the intercept with the x-axis.
    depends on: SweepSet_dfdI, Sweep_AP_freq, Sweep_Stim_amp.
    units: pA.
    """

    def __init__(self, data=None, compute_at_init=True, dc_offset=0):
        self.dc_offset = dc_offset
        super().__init__(
            swft.NullSweepFeature,
            data=data,
            compute_at_init=compute_at_init,
            name="rheobase",
        )
        self.parse_docstring()

    def _compute(self, recompute=False, store_diagnostics=False):
        dc_offset = self.dc_offset
        rheobase = float("nan")

        if stimulus_type(self.data) == "long_square":
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
        if stimulus_type(self.data) == "ramp":
            has_ap = self.lookup_sweep_feature("num_ap", recompute=recompute) > 0
            if np.any(has_ap):
                sweep_idx = np.where(has_ap)[0][0]

                spike_df = self.data[sweep_idx]._spikes_df
                threshold_t = spike_df["threshold_t"]

                # sweep has ap during stimulus
                onset = self.lookup_sweep_feature("stim_onset")[sweep_idx]
                end = self.lookup_sweep_feature("stim_end")[sweep_idx]
                stim_window = where_between(threshold_t.to_numpy(), onset, end)

                if np.any(stim_window):
                    first_spike = spike_df[stim_window].iloc[0]
                    rheobase = first_spike["threshold_i"]
                    rheobase_t = first_spike["threshold_t"]
                    rheobase_idx = first_spike["threshold_index"]
                    rheobase -= dc_offset

                if store_diagnostics:
                    self._update_diagnostics(
                        {
                            "sweep_idx": sweep_idx,
                            "spike_idx": 0,
                            "rheobase_t": rheobase_t,
                            "rheobase_idx": rheobase_idx,
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

        if stimulus_type(self.data) == "long_square":
            i_sub, i_sup, f_sup, dc_offset = unpack(
                self.diagnostics, ["i_sub", "i_sup", "f_sup", "dc_offset"]
            )
            i_intercept = self.value
            dfdi = dfdi_ft.value

            if not np.isnan(dfdi):
                i, f, f_intercept = unpack(
                    dfdi_ft.diagnostics, ["i", "f", "f_intercept"]
                )
                has_spikes = ~np.isnan(f)
                n_no_spikes = np.sum(~has_spikes)

                ax.plot(
                    i[has_spikes][:5], f[has_spikes][:5], "o", label="f(I)", **kwargs
                )
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

        if stimulus_type(self.data) == "ramp":
            dc_offset, rheobase_t, rheobase_idx, sweep_idx = unpack(
                self.diagnostics,
                ["dc_offset", "rheobase_t", "rheobase_idx", "sweep_idx"],
            )
            sweep = self.data[sweep_idx]
            ax.scatter(
                rheobase_t, sweep.v[rheobase_idx], label="ap threshold", **kwargs
            )
            ax.set_xlabel("t (s)")
            ax.set_ylabel("V (mV)")

            ax.plot(sweep.t, sweep.v, label="rheobase sweep", **kwargs)
            ax.legend()

            ax2 = ax.twinx()

            ax2.plot(sweep.t, sweep.i, label="rheobase sweep", **kwargs)
            ax2.vlines(rheobase_t, 0, sweep.i[rheobase_idx] - dc_offset)
            ax2.hlines(
                self.value + dc_offset,
                0,
                rheobase_t,
                label="rheobase w.o. offset current",
                **kwargs,
            )
            ax2.hlines(
                self.value, 0, rheobase_t, label="rheobase w. offset current", **kwargs
            )
            ax2.set_xlabel("t (s)")
            ax2.set_ylabel("I (pA)")

            ax2.set_ylim(
                ax2.get_ylim()[0],
                5 * ax2.get_ylim()[1],
            )
            ax2.legend()

        return ax


class SweepSet_R_input(SweepSetFeature):
    """Obtain sweepset level r_input feature.

    description: The slope of the linear fit of the voltage deflection vs. the
    stimulus amplitude for hyperpolarizing current injections.
    depends on: Sweep_V_deflect, Sweep_Stim_amp.
    units: MOhm.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_R_input,
            data=data,
            compute_at_init=compute_at_init,
        )
        self.parse_docstring()

    def _compute(self, recompute=False, store_diagnostics=False):
        r_input = float("nan")
        if stimulus_type(self.data) == "long_square":
            i_amp = self.lookup_sweep_feature("stim_amp", recompute=recompute)
            v_deflect = self.lookup_sweep_feature("v_deflect", recompute=recompute)
            is_hyperpol = i_amp < 0
            v_deflect = v_deflect[is_hyperpol].reshape(-1, 1)
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


class SweepSet_Slow_hyperpolarization(SweepSetFeature):
    """Obtain sweepset level slow_hyperpolarization feature.

    description: The maximum hyperpolarization voltage across the resting state
    taking the first sweep that has an action potential "0".
    Drop in resting state potential is due to autoinhibition and recruitment
    of calcium-activated currents.
    depends on: Sweep_Num_AP, Sweep_V_baseline.
    units: mV.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.NullSweepFeature,
            data=data,
            compute_at_init=compute_at_init,
            name="slow_hyperpolarization",
        )
        self.parse_docstring()

    def _compute(self, recompute=False, store_diagnostics=False):
        slow_hyperpolarization = float("nan")
        if stimulus_type(self.data) == "long_square":
            has_aps = self.lookup_sweep_feature("num_ap", recompute=recompute) > 0
            v_baseline = self.lookup_sweep_feature("v_baseline", recompute=recompute)
            v_baseline = v_baseline[has_aps]

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
        ax.vlines(0.05, v_min, v_max, lw=5, label=self.name)
        return ax


class SweepSet_Slow_hyperpolarization_slope(SweepSetFeature):
    """Obtain sweepset level slow_hyperpolarization slope feature.

    description: The slope of the hyperpolarization voltage across the resting state
    taking the first sweep that has an action potential "0".
    Drop in resting state potential is due to autoinhibition and recruitment
    of calcium-activated currents.
    depends on: Sweep_Num_AP, Sweep_V_baseline.
    units: mV.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.NullSweepFeature,
            data=data,
            compute_at_init=compute_at_init,
            name="slow_hyperpolarization_slope",
        )
        self.parse_docstring()

    def _compute(self, recompute=False, store_diagnostics=False):
        slow_hyperpolarization_slope = float("nan")
        if stimulus_type(self.data) == "long_square":
            has_aps = self.lookup_sweep_feature("num_ap", recompute=recompute) > 0
            v_baseline = self.lookup_sweep_feature("v_baseline", recompute=recompute)
            v_baseline = v_baseline[has_aps]

            v_baseline = v_baseline.reshape(-1, 1)
            sweep_idx = np.arange(len(v_baseline)).reshape(-1, 1)

            if len(v_baseline) >= 3:
                ransac.fit(sweep_idx, v_baseline)
                slope = ransac.coef_[0, 0] * 1000
                intercept = ransac.intercept_[0]
                slow_hyperpolarization_slope = slope

            if store_diagnostics:
                self._update_diagnostics(
                    {
                        "v_baseline": v_baseline,
                        "sweep_idx": sweep_idx,
                        "v_intercept": intercept,
                    }
                )
        return slow_hyperpolarization_slope

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        ax = plt.gca() if ax is None else ax
        if np.isnan(self.value):
            return ax

        if self.diagnostics is None:
            self.get_diagnostics(recompute=True)

        if not np.isnan(self.value):
            slope = self.value
            i, v, intercept = unpack(
                self.diagnostics, ["sweep_idx", "v_baseline", "v_intercept"]
            )

            ax.plot(i, v, "o", label="V(idx)", **kwargs)
            ax.plot(i, slope * i + intercept, label="V_baseline(idx) fit", **kwargs)
            ax.set_xlim(np.min(i) - 5, np.max(i) + 5)
            ax.set_xlabel("index")
            ax.set_ylabel("V (mV)")
            ax.legend()
        return ax


class SweepSet_Tau(HyperpolMedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_Tau, data=data, compute_at_init=compute_at_init)


# class SweepSet_V_rest(SweepSetFeature):
#     """Obtain sweepset level membrane resting potential feature.

#     description: Average of 100ms pre stimulus voltages aggregated across all
#     hyperpolarization sweeps.
#     depends on: SweepSet_R_input, Sweep_V_baseline.
#     units: mV.
#     """

#     def __init__(self, data=None, compute_at_init=True, dc_offset=0):
#         self.dc_offset = dc_offset
#         super().__init__(
#             swft.NullSweepFeature,
#             data=data,
#             compute_at_init=False,
#             name="v_rest",
#         )
#         if compute_at_init and data is not None:  # because of dc_offset
#             self.get_value()
#         self.parse_docstring()

#     def _compute(self, recompute=False, store_diagnostics=False):
#         r_input = self.lookup_sweepset_feature("r_input", recompute=recompute)
#         v_baseline = self.lookup_sweepset_feature("v_baseline", recompute=recompute)

#         v_rest = v_baseline - r_input * 1e-3 * self.dc_offset

#         if store_diagnostics:
#             self._update_diagnostics(
#                 {
#                     "v_baseline": v_baseline,
#                     "r_input": r_input,
#                     "dc_offset": self.dc_offset,
#                 }
#             )
#         return v_rest

#     def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
#         return ax


class SweepSet_V_rest(HyperpolMedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_V_rest, data=data, compute_at_init=compute_at_init)


class SweepSet_V_baseline(HyperpolMedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_V_baseline, data=data, compute_at_init=compute_at_init
        )


class SweepSet_Sag(SagFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_Sag, data=data, compute_at_init=compute_at_init)


class SweepSet_Sag_ratio(SagFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_Sag_ratio, data=data, compute_at_init=compute_at_init
        )


class SweepSet_Sag_fraction(SagFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_Sag_fraction, data=data, compute_at_init=compute_at_init
        )


class SweepSet_Sag_area(SagFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_Sag_area, data=data, compute_at_init=compute_at_init
        )


class SweepSet_Sag_time(SagFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_Sag_time, data=data, compute_at_init=compute_at_init
        )


class SweepSet_Rebound(ReboundFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_Rebound, data=data, compute_at_init=compute_at_init)


class SweepSet_Rebound_APs(SweepSetFeature):
    """Obtain sweepset level rebound APs feature.

    description: Number of rebound APs.
    depends on: Sweep_Rebound_APs.
    units: /."""

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_Rebound_APs, data=data, compute_at_init=compute_at_init
        )
        self.parse_docstring()

    def _select(self, fts):
        """Select representative sweep and use its rebound features to represent the
        entire sweepset.

        description: 3 lowest hyperpolarization sweeps.
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


class SweepSet_Rebound_area(ReboundFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_Rebound_area, data=data, compute_at_init=compute_at_init
        )


class SweepSet_Rebound_latency(ReboundFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_Rebound_latency, data=data, compute_at_init=compute_at_init
        )


class SweepSet_Rebound_avg(ReboundFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_Rebound_avg, data=data, compute_at_init=compute_at_init
        )


class SweepSet_Num_AP(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_Num_AP, data=data, compute_at_init=compute_at_init)


class SweepSet_AP_freq(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_AP_freq, data=data, compute_at_init=compute_at_init)


class SweepSet_AP_freq_adapt(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_AP_freq_adapt, data=data, compute_at_init=compute_at_init
        )


class SweepSet_AP_amp_slope(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_AP_amp_slope, data=data, compute_at_init=compute_at_init
        )


class SweepSet_ISI_FF(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_ISI_FF, data=data, compute_at_init=compute_at_init)


class SweepSet_AP_FF(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_AP_FF, data=data, compute_at_init=compute_at_init)


class SweepSet_ISI_CV(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_ISI_CV, data=data, compute_at_init=compute_at_init)


class SweepSet_AP_CV(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_AP_CV, data=data, compute_at_init=compute_at_init)


class SweepSet_ISI(APsFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_ISI, data=data, compute_at_init=compute_at_init)


class SweepSet_Burstiness(First5MedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_Burstiness, data=data, compute_at_init=compute_at_init
        )


class SweepSet_Num_bursts(First5MedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_Num_bursts, data=data, compute_at_init=compute_at_init
        )


class SweepSet_ISI_adapt(First5MedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_ISI_adapt, data=data, compute_at_init=compute_at_init
        )


class SweepSet_ISI_adapt_avg(First5MedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_ISI_adapt_avg, data=data, compute_at_init=compute_at_init
        )


class SweepSet_AP_amp_adapt(First5MedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_AP_amp_adapt, data=data, compute_at_init=compute_at_init
        )


class SweepSet_AP_amp_adapt_avg(First5MedianFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_AP_amp_adapt_avg, data=data, compute_at_init=compute_at_init
        )


class SweepSet_AP_AHP(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_AP_AHP, data=data, compute_at_init=compute_at_init)


class SweepSet_AP_ADP(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_AP_ADP, data=data, compute_at_init=compute_at_init)


class SweepSet_AP_thresh(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_AP_thresh, data=data, compute_at_init=compute_at_init
        )


class SweepSet_AP_amp(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_AP_amp, data=data, compute_at_init=compute_at_init)


class SweepSet_AP_width(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_AP_width, data=data, compute_at_init=compute_at_init
        )


class SweepSet_AP_peak(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_AP_peak, data=data, compute_at_init=compute_at_init)


class SweepSet_AP_trough(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_AP_trough, data=data, compute_at_init=compute_at_init
        )


class SweepSet_AP_overshoot(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_AP_overshoot, data=data, compute_at_init=compute_at_init
        )


class SweepSet_AP_ADP_trough(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_AP_ADP_trough, data=data, compute_at_init=compute_at_init
        )


class SweepSet_AP_UDR(APFeature):
    def __init__(self, data=None, compute_at_init=True):
        super().__init__(swft.Sweep_AP_UDR, data=data, compute_at_init=compute_at_init)


class SweepSet_Num_wild_APs(SweepSetFeature):
    """Obtain sweepset level number of wild APs feature.

    description: Max number of APs outside of stimulus window.
    depends on: /.
    units: /.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_Wildness,
            data=data,
            compute_at_init=compute_at_init,
            name="num_wild_aps",
        )
        self.parse_docstring()

    def _select(self, fts: ndarray) -> ndarray:
        """Select representative sweep and use its wildness feature to represent
        the entire sweepset.

        description: argmax.
        """
        if np.any(~np.isnan(fts)):
            return np.nanargmax(fts)
        return np.array([], dtype=int)


class SweepSet_Wildness(SweepSetFeature):
    """Obtain sweepset level wildness feature.

    description: Difference in the number of APs between the highest firing
    trace (possibly showing APs before or after the stimulation window) and the
    highest firing trace as defined above (without any APs outside the
    stimulation window)
    depends on: Sweep_Num_AP.
    units: /.
    """

    def __init__(self, data=None, compute_at_init=True):
        super().__init__(
            swft.Sweep_Wildness,
            data=data,
            compute_at_init=compute_at_init,
        )
        self.parse_docstring()

    def _compute(self, recompute=False, store_diagnostics=False):
        num_ap = self.lookup_sweep_feature("num_ap", recompute=recompute)
        wildness = self.lookup_sweep_feature("wildness", recompute=recompute)
        is_wild = ~np.isnan(wildness)
        not_nan = ~np.isnan(num_ap)

        if np.any(is_wild) and np.any(not_nan):
            wildness = num_ap[is_wild].max() - num_ap[~is_wild & not_nan].max()
        else:
            wildness = float("nan")
        return wildness


class NullSweepSetFeature(SweepSetFeature):
    """Obtain sweepset level null feature.

    description: This feature acts as a placeholder or null feature.
    depends on: /.
    units: /.
    """

    def __init__(
        self, data=None, compute_at_init=True, name: str = "null_sweepset_feature"
    ):
        super().__init__(
            swft.NullSweepFeature,
            data=data,
            compute_at_init=compute_at_init,
            name=name,
        )
        self.parse_docstring()

    def _compute(self, recompute=False, store_diagnostics=False):
        return None

    def _plot(self, ax: Optional[Axes] = None, **kwargs) -> Axes:
        return ax
