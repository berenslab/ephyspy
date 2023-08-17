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

import inspect
import re
import sys
from typing import Callable, Dict, List, Optional, Union, Tuple, Any
import warnings

import numpy as np
from numpy import ndarray

import ephyspy.allen_sdk.ephys_extractor as efex
from ephyspy.allen_sdk.ephys_extractor import (
    EphysSweepFeatureExtractor as AllenEphysSweepFeatureExtractor,
)
from ephyspy.allen_sdk.ephys_extractor import (
    EphysSweepSetFeatureExtractor as AllenEphysSweepSetFeatureExtractor,
)

CUSTOM_FEATURES = []


class EphysSweepFeatureExtractor(AllenEphysSweepFeatureExtractor):
    """Wrapper around EphysSweepFeatureExtractor from the AllenSDK to
    support additional functionality.

    Mainly it supports the addition of new spike features.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.added_spike_features = {}
        self.features = {}

    def add_spike_feature(self, feature_name: str, feature_func: Callable):
        """Add a new spike feature to the extractor.

        Args:
            feature_name (str): Name of the new feature.
            feature_func (Callable): Function to calculate the new feature.
        """
        self.added_spike_features[feature_name] = feature_func

    def _process_added_spike_features(self):
        """Process added spike features."""
        for feature_name, feature_func in self.added_spike_features.items():
            self.process_new_spike_feature(feature_name, feature_func)

    def process_spikes(self):
        """Perform spike-related feature analysis, which includes added spike
        features not part of the original AllenSDK implementation."""
        self._process_individual_spikes()
        self._process_spike_related_features()
        self._process_added_spike_features()

    def get_features(self):
        if hasattr(self, "features"):
            if self.features is not None:
                return {k: ft.value for k, ft in self.features.items()}


# overwrite AllenSDK EphysSweepFeatureExtractor with wrapper
efex.EphysSweepFeatureExtractor = EphysSweepFeatureExtractor


class EphysSweepSetFeatureExtractor(AllenEphysSweepSetFeatureExtractor):
    """Wrapper around EphysSweepSetFeatureExtractor from the AllenSDK to
    support additional functionality.

    Args:
        t_set (ndarray): Time array for set of sweeps.
        v_set (ndarray): Voltage array for set of sweeps.
        i_set (ndarray): Current array for set of sweeps.
        metadata (dict, optional): Metadata for the sweep set. Defaults to None.
        *args: Additional arguments for AllenEphysSweepSetFeatureExtractor.
        **kwargs: Additional keyword arguments for AllenEphysSweepSetFeatureExtractor.

    Attributes:
        metadata (dict): Metadata for the sweep set.
    """

    def __init__(
        self,
        t_set: Optional[Union[List, ndarray]] = None,
        v_set: Optional[Union[List, ndarray]] = None,
        i_set: Optional[Union[List, ndarray]] = None,
        t_start: Optional[Union[List, ndarray, float]] = None,
        t_end: Optional[Union[List, ndarray, float]] = None,
        metadata: Dict = {},
        dc_offset: float = 0,
        *args,
        **kwargs,
    ):
        is_array = lambda x: isinstance(x, ndarray) and x is not None
        is_float = lambda x: isinstance(x, float) and x is not None
        t_set = [t for t in t_set] if is_array(t_set) else t_set
        v_set = [v for v in v_set] if is_array(v_set) else v_set
        i_set = [i for i in i_set] if is_array(i_set) else i_set
        if t_start is None:
            t_start = [t[1] for t in t_set]
            t_end = [t[-1] for t in t_set]
        elif is_float(t_start):
            t_start = [t_start] * len(t_set)
            t_end = [t_end] * len(t_set)
        elif is_array(t_start):
            pass  # t_start and t_end for each sweep are already specified

        super().__init__(t_set, v_set, i_set, t_start, t_end, *args, **kwargs)
        self.metadata = metadata
        self.dc_offset = {
            "value": dc_offset,
            "units": "pA",
            "description": "offset current",
        }

    @property
    def t(self) -> ndarray:
        t = np.empty((len(self.sweeps()), len(self.sweeps()[0].t)))
        for i, swp in enumerate(self.sweeps()):
            t[i] = swp.t
        return t

    @property
    def v(self) -> ndarray:
        v = np.empty((len(self.sweeps()), len(self.sweeps()[0].v)))
        for i, swp in enumerate(self.sweeps()):
            v[i] = swp.v
        return v

    @property
    def i(self) -> ndarray:
        stim = np.empty((len(self.sweeps()), len(self.sweeps()[0].i)))
        for i, swp in enumerate(self.sweeps()):
            stim[i] = swp.i
        return stim

    def __len__(self):
        return len(self.sweeps())

    def add_spike_feature(self, feature_name: str, feature_func: Callable):
        """Add a new spike feature to the extractor.

        Adds new spike feature to each `EphysSweepFeatureExtractor` instance.

        Args:
            feature_name (str): Name of the new feature.
            feature_func (Callable): Function to calculate the new feature.
        """
        for sweep in self.sweeps():
            sweep.add_spike_feature(feature_name, feature_func)

    def set_stimulus_amplitude_calculator(self, func: Callable):
        """Set stimulus amplitude calculator for each sweep.

        Args:
            func (Callable): Function to calculate stimulus amplitude.
        """
        for sweep in self.sweeps():
            sweep.set_stimulus_amplitude_calculator(func)

    def get_features(self):
        if hasattr(self, "features"):
            if self.features is not None:
                return {k: ft.value for k, ft in self.features.items()}

    def get_sweep_features(self):
        if hasattr(self, "features"):
            if self.features is not None:
                LD = [sw.get_features() for sw in self.sweeps()]
                return {k: [dic[k] for dic in LD] for k in LD[0]}


def register_custom_feature(Feature: Any):
    """Add a custom feature class that inherits from `EphysFeature`
    or from `SweepsetFeature`. This makes the feature available to all the
    the EphysPy functionalities such as recursive computation of all dependend
    features that are called with `lookup_X_feature`.

    Args:
        Feature: Feature class to be added to EphysPy ecosystem. Feature
            must inherit from either `EphysFeature` or `SweesetFeature`.
    """
    CUSTOM_FEATURES.append(Feature)


def fetch_available_fts():
    classes = inspect.getmembers(sys.modules["ephyspy"], inspect.isclass)
    classes = [
        c[1] for c in classes if "ephyspy.features" in c[1].__module__
    ]  # TODO: swap main for module name!
    feature_classes = [c for c in classes if "Feature" not in c.__name__]

    duplicate_features = set(
        ft.__name__.lower() for ft in CUSTOM_FEATURES
    ).intersection(set(ft.__name__.lower() for ft in feature_classes))
    if len(duplicate_features) > 0:
        warnings.warn(
            f"DUPLICATE FEATURES: Unwanted behaviour with custom versions of"
            + ", ".join(duplicate_features)
            + "cannot be ruled out. Please consider renaming these features."
        )
    return feature_classes + CUSTOM_FEATURES


def where_stimulus(
    data: Union[EphysSweepFeatureExtractor, EphysSweepSetFeatureExtractor]
) -> Union[bool, ndarray]:
    """Checks where the stimulus is non-zero.

    Checks where stimulus is non-zero for a single sweep or each sweep in a
    sweepset.

    Args:
        data (EphysSweepFeatureExtractor or EphysSweepSetFeatureExtractor):
            Sweep or sweepset to check.

    Returns:
        bool: True if stimulus is non-zero.
    """
    return data.i.T != 0


def has_spikes(sweep: EphysSweepFeatureExtractor) -> bool:
    """Check if sweep has spikes.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to check.

    Returns:
        bool: True if sweep has spikes.
    """
    if hasattr(sweep, "_spikes_df"):
        return not sweep._spikes_df.empty
    return False


def has_stimulus(
    data: Union[EphysSweepFeatureExtractor, EphysSweepSetFeatureExtractor]
) -> Union[bool, ndarray]:
    """Check if sweep has stimulus that is non-zero.

    Args:
        data (EphysSweepFeatureExtractor or EphysSweepSetFeatureExtractor):
            Sweep or sweepset to check.

    Returns:
        bool: True if sweep has stimulus."""
    return np.any(where_stimulus(data), axis=0)


def is_hyperpol(
    data: Union[EphysSweepFeatureExtractor, EphysSweepSetFeatureExtractor]
) -> Union[bool, ndarray]:
    """Check if sweep is hyperpolarizing, i.e. if the stimulus < 0.

    Args:
        data (EphysSweepFeatureExtractor or EphysSweepSetFeatureExtractor):
            Sweep or sweepset to check.

    Returns:
        bool: True if sweep is hyperpolarizing."""
    return np.any(data.i.T < 0, axis=0)


def is_depol(
    data: Union[EphysSweepFeatureExtractor, EphysSweepSetFeatureExtractor]
) -> Union[bool, ndarray]:
    """Check if sweep is depolarizing, i.e. if the stimulus > 0.

    Args:
        data (EphysSweepFeatureExtractor or EphysSweepSetFeatureExtractor):
            Sweep or sweepset to check.

    Returns:
        bool: True if sweep is depolarizing."""
    return np.any(data.i.T > 0, axis=0)


def has_rebound(feature: Any, T_rebound: float = 0.3) -> bool:
    """Check if sweep rebounds.

    description: rebound if voltage exceeds baseline after stimulus offset.

    Args:
        feature (EphysFeature): Feature to check for rebound.
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


# TODO: FIX ! Currently not properly parsing the entire description
def parse_func_doc_attrs(func: Callable) -> Dict:
    """Parses docstrings for attributes.

    Docstrings should have the following format:
    <Some text>
    attr: <attr text>.
    attr: <attr text>.
    ...
    <Some more text>

    IMPORTANT: EACH ATTRIBUTE MUST END WITH A "."

    Args:
        func (Callable): Function to parse docstring of.

    Returns:
        doc_attrs: all attributes found in document string.
    """
    func_doc = func.__doc__

    pattern = r"([\w\s]+):"
    matches = re.findall(pattern, func_doc)
    attrs = [m.strip() for m in matches]
    if "Args" in attrs:
        attrs = attrs[: attrs.index("Args")]

    doc_attrs = {}
    for attr in attrs:
        doc_attrs[attr] = ""
        if func_doc is not None:  # if func has no docstring
            regex = re.compile(f"{attr}: (.*)")
            match = regex.search(func_doc)
            if match:
                doc_attrs[attr] = match.group(1)[:-1]
    return doc_attrs


def parse_desc(func: Callable) -> str:
    """Parses docstring for description.

    If no description is found, returns empty string.
    Special case of `parse_func_doc_attrs`.

    Args:
        func (Callable): Function to parse docstring of.

    Returns:
        str: Description of function."""
    dct = parse_func_doc_attrs(func)
    if "description" in dct:
        return dct["description"]
    return ""


def parse_deps(deps_string: str) -> List[str]:
    """Parses docstring for feature dependencies.

    If no dependencies are found, returns empty list.
    Special case of `parse_func_doc_attrs`.

    Args:
        deps_string (str): String to parse for dependencies.

    Returns:
        List[str]: List of dependencies."""
    if deps_string == "/":
        return []
    else:
        return [d.strip() for d in deps_string.split(",")]


where_between = lambda t, t0, tend: np.logical_and(t > t0, t < tend)

get_ap_ft_at_idx = lambda sweep, x, idx: sweep.spike_feature(x, include_clipped=True)[
    idx
]


def get_sweep_burst_metrics(
    sweep: EphysSweepFeatureExtractor,
) -> Tuple[ndarray, ndarray, ndarray]:
    """Calculate burst metrics for a sweep.

    Uses EphysExtractor's _process_bursts() method to calculate burst metrics.
    Handles case where no bursts are found.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to calculate burst metrics for.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: returns burst index, burst start index,
            burst end index.
    """
    burst_metrics = sweep._process_bursts()
    if len(burst_metrics) == 0:
        return float("nan"), slice(0), slice(0)
    idx_burst, idx_burst_start, idx_burst_end = burst_metrics.T
    return idx_burst, idx_burst_start.astype(int), idx_burst_end.astype(int)


def get_sweep_sag_idxs(feature: Any, recompute: bool = False) -> ndarray:
    """determine idxs in a sweep that are part of the sag.

    description: all idxs below steady state and during stimulus.

    Args:
        sweep (EphysSweepFeatureExtractor): sweep to analyze.

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
    sweep = feature.data
    v_deflect = sweep.voltage_deflection("min")[0]
    v_steady = feature.lookup_sweep_feature("v_deflect", recompute=recompute)
    if v_steady - v_deflect < 4:  # The sag should have a minimum depth of 4 mV
        start = feature.lookup_sweep_feature("stim_onset", recompute=recompute)
        end = feature.lookup_sweep_feature("stim_end", recompute=recompute)
        where_stimulus = where_between(sweep.t, start, end)
        return np.logical_and(where_stimulus, sweep.v < v_steady)
    return np.zeros_like(sweep.t, dtype=bool)


def median_idx(d):
    if len(d) > 0:
        is_median = d == d.median()
        if any(is_median):
            return int(d.index[is_median].to_numpy())
        ranks = d.rank(pct=True)
        close_to_median = abs(ranks - 0.5)
        return int(np.array([close_to_median.idxmin()]))
    return slice(0)


class FeatureError(ValueError):
    """Error raised when a feature is unknown."""

    pass
