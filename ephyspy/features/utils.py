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

import inspect
import sys
import warnings
from typing import TYPE_CHECKING, Any, Callable, List, Tuple, Union

import numpy as np
from numpy import ndarray
from pandas import DataFrame

from ephyspy.sweeps import EphysSweep, EphysSweepSet

if TYPE_CHECKING:
    from ephyspy.features.base import SweepFeature, SweepSetFeature

CUSTOM_SWEEP_FEATURES = []
CUSTOM_SWEEPSET_FEATURES = []
CUSTOM_SPIKE_FEATURES = []


def register_custom_feature(Feature: Union[Callable, SweepSetFeature, SweepFeature]):
    """Add a custom feature class that inherits from `SweepFeature`
    or from `SweepSetFeature`. This makes the feature available to all the
    the EphysPy functionalities such as recursive computation of all dependend
    features that are called with `lookup_X_feature`, where X can be spike,
    sweep or sweepset.

    Args:
        Feature: Feature class to be added to EphysPy ecosystem. Feature
            must inherit from either `SweepFeature` or `SweesetFeature`.
    """
    # TODO: assert more rigorously that Feature can be computed
    # i.e. by calling it on a dummy sweep and checking if it raises an error
    # only if it behaves as expected add it to the list of available features
    from ephyspy.features.base import SweepFeature, SweepSetFeature

    if issubclass(Feature, SweepFeature):
        CUSTOM_SWEEP_FEATURES.append(Feature)
    elif issubclass(Feature, SweepSetFeature):
        CUSTOM_SWEEPSET_FEATURES.append(Feature)
    elif isinstance(Feature, Callable):  # last, since SweepFeatures also are callable
        CUSTOM_SPIKE_FEATURES.append(Feature)


def fetch_available_fts() -> List[str]:
    """Fetch all available features.

    Returns a list of all available feature functions and classes that are
    either part of the EphysPy package or have been registered as custom
    features with `register_custom_feature`.

    Returns:
        List[str]: List of all available features.

    Warnings:
        If a custom feature has the same name as a feature that is part of
        EphysPy, a warning is raised."""
    classes = inspect.getmembers(sys.modules["ephyspy"], inspect.isclass)
    classes = [c[1] for c in classes if "ephyspy.features" in c[1].__module__]
    feature_classes = [c for c in classes if "Feature" not in c.__name__]

    for custom_fts, base_class in zip(
        [CUSTOM_SWEEP_FEATURES, CUSTOM_SWEEPSET_FEATURES],
        ["SweepFeature", "SweepSetFeature"],
    ):
        base_feature_classes = [
            ft for ft in feature_classes if ft.__base__.__name__ == base_class
        ]
        duplicate_features = set(ft.__name__.lower() for ft in custom_fts).intersection(
            set(ft.__name__.lower() for ft in base_feature_classes)
        )
        if len(duplicate_features) > 0:
            warnings.warn(
                f"DUPLICATE FEATURES: Unwanted behaviour with custom versions of"
                + ", ".join(duplicate_features)
                + "cannot be ruled out. Please consider renaming these features."
            )

    return feature_classes + CUSTOM_SWEEP_FEATURES + CUSTOM_SWEEPSET_FEATURES


def SweepsetFt(SweepsetFt: SweepSetFeature, Ft: SweepFeature):
    """Wraps SweepSetFeature and SweepFeature to act like SweepFeature.

    This is a workaround to make SweepSetFeature classes act like SweepFeature
    which means the first input argument is `data` and that it has to be
    instantiated first. Otherwise SweepSetFeature(SweepFeature) would have to be
    instantiated with SweepFeature first and then the `__call__` method would
    have to be used to init `SwepsetFeature` with `data`.

    Args:
        SweepsetFt (SweepSetFeature): SweepSetFeature class to be created.
        Ft (SweepFeature): SweepFeature class to be used as base class.

    Returns:
        SweepsetFt: SweepSetFeature class that inherits from Ft."""

    def _SweepsetFt(*args, **kwargs):
        return SweepsetFt(Ft, *args, **kwargs)

    _SweepsetFt.__base__ = SweepsetFt.__base__
    _SweepsetFt.__name__ = Ft.__name__
    return _SweepsetFt


class FeatureError(ValueError):
    """Error raised when a feature is unknown."""

    pass


def get_sweep_burst_metrics(
    sweep: EphysSweep,
) -> Tuple[ndarray, ndarray, ndarray]:
    """Calculate burst metrics for a sweep.

    Uses EphysExtractor's _process_bursts() method to calculate burst metrics.
    Handles case where no bursts are found.

    Args:
        sweep (EphysSweep): Sweep to calculate burst metrics for.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: returns burst index, burst start index,
            burst end index.
    """
    burst_metrics = sweep._process_bursts()
    if len(burst_metrics) == 0:
        return float("nan"), slice(0), slice(0)  # slice(0) acts as empty index
    idx_burst, idx_burst_start, idx_burst_end = burst_metrics.T
    return idx_burst, idx_burst_start.astype(int), idx_burst_end.astype(int)


def get_sweep_sag_idxs(
    sag_instance: Any, recompute: bool = False, store_diagnostics=False
) -> ndarray:
    """determine idxs in a sweep that are part of the sag.

    description: all idxs below steady state and during stimulus.

    Args:
        feature (EphysSweep): sag_feature object.

    Returns:
        boolean array with length of sweep.t; where sag.
    """
    # TODO: refine how sag idxs are chosen!
    # currently uses all idxs below steady state and during stimulus
    # can lead to very fragmented idx arrays
    # fix: if too many idxs are False between ones that are True
    # set all True ones after to False
    # also if steady state is never reached again, sag will be massive
    # -> set all idxs to False ?
    sweep = sag_instance.data
    v_deflect = sweep.voltage_deflection("min")[0]
    v_steady = sag_instance.lookup_sweep_feature("v_deflect", recompute=recompute)
    if v_steady - v_deflect < 4:  # The sag should have a minimum depth of 4 mV
        start = sag_instance.lookup_sweep_feature("stim_onset", recompute=recompute)
        end = sag_instance.lookup_sweep_feature("stim_end", recompute=recompute)
        where_stimulus = np.logical_and(
            sweep.t > start, sweep.t < end
        )  # same as where_between (saves on import)
        sag_idxs = np.logical_and(where_stimulus, sweep.v < v_steady)
    else:
        sag_idxs = np.zeros_like(sweep.t, dtype=bool)

    if store_diagnostics:
        sag_instance._update_diagnostics(
            {
                "sag_idxs": sag_idxs,
                "v_deflect": v_deflect,
                "v_steady": v_steady,
                "t_sag": sweep.t[sag_idxs],
                "v_sag": sweep.v[sag_idxs],
            }
        )
    return sag_idxs


def where_stimulus(data: Union[EphysSweep, EphysSweepSet]) -> Union[bool, ndarray]:
    """Checks where the stimulus is non-zero.

    Checks where stimulus is non-zero for a single sweep or each sweep in a
    sweepset.

    Args:
        data (EphysSweep or EphysSweepSet):
            Sweep or sweepset to check.

    Returns:
        bool: True if stimulus is non-zero.
    """
    return data.i.T != 0


def has_spikes(sweep: EphysSweep) -> bool:
    """Check if sweep has spikes.

    Args:
        sweep (EphysSweep): Sweep to check.

    Returns:
        bool: True if sweep has spikes.
    """
    if hasattr(sweep, "_spikes_df"):
        return not sweep._spikes_df.empty
    else:
        sweep.process_spikes()
        return not sweep._spikes_df.empty


def has_stimulus(data: Union[EphysSweep, EphysSweepSet]) -> Union[bool, ndarray]:
    """Check if sweep has stimulus that is non-zero.

    Args:
        data (EphysSweep or EphysSweepSet):
            Sweep or sweepset to check.

    Returns:
        bool: True if sweep has stimulus."""
    return np.any(where_stimulus(data), axis=0)


def is_hyperpol(data: Union[EphysSweep, EphysSweepSet]) -> Union[bool, ndarray]:
    """Check if sweep is hyperpolarizing, i.e. if the stimulus < 0.

    Args:
        data (EphysSweep or EphysSweepSet):
            Sweep or sweepset to check.

    Returns:
        bool: True if sweep is hyperpolarizing."""
    return np.any(data.i.T < 0, axis=0)


def is_depol(data: Union[EphysSweep, EphysSweepSet]) -> Union[bool, ndarray]:
    """Check if sweep is depolarizing, i.e. if the stimulus > 0.

    Args:
        data (EphysSweep or EphysSweepSet):
            Sweep or sweepset to check.

    Returns:
        bool: True if sweep is depolarizing."""
    return np.any(data.i.T > 0, axis=0)


def has_rebound(feature: Any, T_rebound: float = 0.3) -> bool:
    """Check if sweep rebounds.

    description: rebound if voltage exceeds baseline after stimulus offset.

    Args:
        feature (SweepFeature): Feature to check for rebound.
        T_rebound (float, optional): Time window after stimulus offset in which
            rebound can occur. Defaults to 0.3.

    Returns:
        bool: True if sweep rebounds."""
    sweep = feature.data
    if is_hyperpol(sweep):
        end = feature.lookup_sweep_feature("stim_end")
        v_baseline = feature.lookup_sweep_feature("v_baseline")
        ts_rebound = np.logical_and(sweep.t > end, sweep.t < end + T_rebound)
        return np.any(sweep.v[ts_rebound] > v_baseline)
    return False


def median_idx(d: Union[DataFrame, ndarray]) -> Union[int, slice]:
    """Get index of median value in a DataFrame.

    If median is unique return index, otherwise return all indices that are
    closest to the median. If dataframe is empty or all nan return slice(0).

    Args:
        d (Union[DataFrame, ndarray]): DataFrame or ndarray to get median index
            from.

    Returns:
        Union[int, slice]: Index of median value or slice(0) if d is empty or
            all nan."""
    d = d if isinstance(d, DataFrame) else DataFrame(d)
    if len(d) > 0:
        is_median = d == d.median()
        if any(is_median):
            return int(d.index[is_median].to_numpy())
        ranks = d.rank(pct=True)
        close_to_median = abs(ranks - 0.5)
        return int(np.array([close_to_median.idxmin()]))
    return slice(0)
