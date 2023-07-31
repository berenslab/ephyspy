import re
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame

from py_ephys.allen_sdk.ephys_extractor import EphysSweepFeatureExtractor
from py_ephys.allen_sdk.ephys_extractor import (
    EphysSweepSetFeatureExtractor as AllenEphysSweepSetFeatureExtractor,
)


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
        spike_feature_funcs (dict): Dictionary of spike feature functions.
        sweep_feature_funcs (dict): Dictionary of sweep feature functions.
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
        self.spike_feature_funcs = {}
        self.sweep_feature_funcs = {}
        self.sweepset_feature_funcs = {}
        self.sweepset_features = {}
        self.metadata = metadata
        self.dc_offset = {
            "value": dc_offset,
            "units": "pA",
            "description": "offset current",
        }

        self.set_sweep_feature("dc_offset", self.dc_offset)
        self.cached_sweep_features = None  # for faster retrieval

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
        i = np.empty((len(self.sweeps()), len(self.sweeps()[0].i)))
        for i, swp in enumerate(self.sweeps()):
            i[i] = swp.i
        return i

    def get_sweep_features(
        self, force_retrieval: bool = False, return_ft_info: bool = True
    ) -> DataFrame:
        if self.cached_sweep_features is None or force_retrieval:
            l = []
            for swp in self.sweeps():
                swp._sweep_features
                l.append(swp._sweep_features)
            self.cached_sweep_features = pd.DataFrame(l)
        if return_ft_info:
            return self.cached_sweep_features.applymap(strip_info)
        return self.cached_sweep_features

    def get_sweep_feature(self, key: str):
        return self.get_sweep_features()[key]

    def set_sweep_feature(self, key, value):
        for swp in self.sweeps():
            swp._sweep_features[key] = value

    def get_sweepset_features(
        self, force_retrieval: bool = False, return_ft_info: bool = True
    ) -> DataFrame:
        if self.sweepset_features == {} or force_retrieval:
            self.process()
        if return_ft_info:
            return self.sweepset_features
        return {k: strip_info(v) for k, v in self.sweepset_features.items()}

    def get_sweepset_feature(self, key: str) -> Union[Dict, float]:
        return self.get_sweepset_features()[key]

    def add_spike_feature(self, feature_name: str, feature_func: Callable):
        self.spike_feature_funcs[feature_name] = feature_func

    def add_sweep_feature(self, feature_name: str, feature_func: Callable):
        self.sweep_feature_funcs[feature_name] = feature_func

    def add_sweepset_feature(self, feature_name: str, feature_func: Callable):
        self.sweepset_feature_funcs[feature_name] = feature_func

    def process_new_sweepset_feature(self, ft: str, ft_func: Callable):
        self.sweepset_features[ft] = ft_func(self)

    def process(self, overwrite_existing: bool = True):
        """Analyze features for all sweeps."""
        for sweep in self._sweeps:
            if overwrite_existing:
                sweep._sweep_features = {"dc_offset": self.dc_offset}
                sweep.spike_features = {}
            sweep.process_spikes()
            for ft, ft_func in self.spike_feature_funcs.items():
                sweep.process_new_spike_feature(ft, ft_func)

            for ft, ft_func in self.sweep_feature_funcs.items():
                sweep.process_new_sweep_feature(ft, ft_func)

        if overwrite_existing:
            self.cached_sweep_features = None
        for ft, ft_func in self.sweepset_feature_funcs.items():
            self.process_new_sweepset_feature(ft, ft_func)

    def set_stimulus_amplitude_calculator(self, func: Callable):
        for sweep in self._sweeps:
            sweep.set_stimulus_amplitude_calculator(func)


def has_stimulus(sweep: EphysSweepFeatureExtractor) -> bool:
    """Check if sweep has stimulus that is non-zero.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to check.

    Returns:
        bool: True if sweep has stimulus."""
    return np.any(sweep.i != 0)


def is_hyperpol(sweep: EphysSweepFeatureExtractor) -> bool:
    """Check if sweep is hyperpolarizing, i.e. if the stimulus < 0.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to check.

    Returns:
        bool: True if sweep is hyperpolarizing."""
    return np.any(sweep.i < 0)


def is_depol(sweep: EphysSweepFeatureExtractor) -> bool:
    """Check if sweep is depolarizing, i.e. if the stimulus > 0.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to check.

    Returns:
        bool: True if sweep is depolarizing."""
    return np.any(sweep.i > 0)


def where_stimulus(sweep: EphysSweepFeatureExtractor) -> ndarray:
    """Get mask where stimulus is non-zero.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to check.

    Returns:
        ndarray: Mask where stimulus is non-zero."""
    if has_stimulus(sweep):
        peaks = strip_info(sweep.spike_feature("peak_i"))
        # peaks = sweep.spike_feature("threshold_i") # unclear which is better
        if len(peaks) > 0:
            return peaks != 0
        return np.zeros_like(peaks, dtype=bool)
    return np.zeros(0, dtype=bool)


def has_rebound(sweep: EphysSweepFeatureExtractor, T_rebound: float = 0.3) -> bool:
    """Check if sweep rebounds.

    description: rebound if voltage exceeds baseline after stimulus offset.

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to check.
        T_rebound (float, optional): Time window after stimulus offset in which
            rebound can occur. Defaults to 0.3.

    Returns:
        bool: True if sweep rebounds."""
    if is_hyperpol(sweep):
        end = strip_info(sweep.sweep_feature("stim_end"))
        v_baseline = strip_info(sweep.sweep_feature("v_baseline"))
        ts_rebound = np.logical_and(sweep.t > end, sweep.t < end + T_rebound)
        return np.any(sweep.v[ts_rebound] > v_baseline)
    return False


def parse_ft_desc(func: Callable) -> str:
    """Parses docstrings for feature descriptions.

    Docstrings should have the following format:
    <Some text>
    description: <description text>.
    <Some more text>

    Args:
        func (Callable): Function to parse docstring of.

    Returns:
        string: Description of the feature that the function extracts.
    """
    func_doc = func.__doc__
    if func_doc is not None:  # if func has no docstring
        pattern = re.compile(r"description: (.*)")
        match = pattern.search(func_doc)
        if match:
            return match.group(1)
    return ""


def parse_ft_deps(func: Callable) -> str:
    """Parses docstrings for feature dependencies.

    Docstrings should have the following format:
    <Some text>
    depends on: <feature dependencies seperated by commas>.
    <Some more text>

    Args:
        func (Callable): Function to parse docstring of.

    Returns:
        string: Other features that the function depends on.
    """
    func_doc = func.__doc__
    if func_doc is not None:  # if func has no docstring
        pattern = re.compile(r"depends on: (.*)")
        match = pattern.search(func_doc)
        if match:
            dependency_str = match.group(1)[:-1]
            return [d.strip() for d in dependency_str.split(",")]
    return ""  # no dependencies where found (while / means ft has no dependencies)


where_between = lambda t, t0, tend: np.logical_and(t > t0, t < tend)

get_ap_ft_at_idx = lambda sweep, x, idx: sweep.spike_feature(x, include_clipped=True)[
    idx
]


def get_sweep_sag_idxs(sweep: EphysSweepFeatureExtractor):
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
    v_deflect = sweep.voltage_deflection("min")[0]
    v_steady = strip_info(sweep.sweep_feature("v_deflect"))
    if v_steady - v_deflect < 4:  # The sag should have a minimum depth of 4 mV
        start = strip_info(sweep.sweep_feature("stim_onset"))
        end = strip_info(sweep.sweep_feature("stim_end"))
        where_stimulus = where_between(sweep.t, start, end)
        return np.logical_and(where_stimulus, sweep.v < v_steady)
    return np.zeros_like(sweep.t, dtype=bool)


def strip_info(dct: Union[Dict, float]) -> float:
    """Extracts only the value of the first key in a dict.

    This function is used to strip the metadata from a feature dictionary and
    just return its value. If the input is not a dictionary, i.e. just the
    feature value, it is returned.

    For debugging / inspecting features, they can be returned with metadata
    to retrace how they were calculated. In this case a feature has the form:

    feature_params = {
        "ft:feature_name": feature value,
        "feature_info_1": metadata,
        "feature_info_2": more_metadata,
        ...
        "description": "some text",
        }

    Args:
        dct (Union[Dict, float]): Dictionary to strip.

    Returns:
        float: Value of the first key in the dictionary.
    """
    if isinstance(dct, dict):
        return list(dct.values())[0]
    return dct


def include_info(
    ft: float, ft_info: Dict, return_ft_info: bool = False
) -> Union[float, Dict]:
    """Convenience function to toggle between returning just the feature value
    or a dictionary containing the feature value and additional diagnostic info.

    Args:
        ft (float): Feature value.
        ft_info (Dict): Feature diagnostic info.
        return_ft_info (bool, optional): Whether to return just the feature value
            or a dictionary containing the feature value and additional diagnostic
            info. Defaults to False.

    Returns:
        Union[float, Dict]: Feature value or dictionary containing feature value
            and diagnostic info."""
    if return_ft_info:
        ft_dict = {"value": ft}
        ft_dict.update(ft_info)
        return ft_dict
    return ft


def ephys_feature_init(ft_info_init: Dict = None) -> Tuple[float, Dict]:
    """Convenience function to initialize ephys feature.

    Args:
        ft_info_init (Dict, optional): Initial feature diagnostic info. Defaults to None.

    Returns:
        Tuple[float, Dict]: Initial feature value and diagnostic info."""
    ft_info_init = {} if ft_info_init is None else ft_info_init
    return float("nan"), ft_info_init


def ephys_feature(feature: Callable) -> Callable:
    """Decorates ephys feature functions.

    This decorator adds functionality to ephys feature functions. It allows
    to toggle between returning just the value of the copmuted feature and a
    a dictionary containing the feature value and additional diagnostic info.
    The metadata can be used to trace how the feature was calculated to help with
    debugging.

    For an example function definition see `get_example_feature`.

    Args:
        feature (Callable): Feature function to decorate. Functions should take
            either a `EphysSweepFeatureExtractor` or `EphysSweepSetFeatureExtractor`
            as input and return either a float or a tuple of float and dict.

    Returns:
        Callable: Decorated feature function."""

    @wraps(feature)
    def feature_func(
        sweep_or_sweepset: Union[
            EphysSweepFeatureExtractor, EphysSweepSetFeatureExtractor
        ],
        return_ft_info: bool = False,
        update_inplace: bool = False,
    ) -> Union[float, Dict]:
        ft_out = feature(sweep_or_sweepset)
        if isinstance(ft_out, Tuple):
            ft, ft_info = ft_out
        else:
            ft, ft_info = ft_out, {}
        ft_info["description"] = parse_ft_desc(feature)

        ft_out = include_info(ft, ft_info, return_ft_info)
        if update_inplace:
            if isinstance(sweep_or_sweepset, EphysSweepFeatureExtractor):
                ft_name = feature.__name__[len("get_sweep_") :]
                sweep_or_sweepset._sweep_features[ft_name] = ft_out
            elif isinstance(sweep_or_sweepset, EphysSweepSetFeatureExtractor):
                ft_name = feature.__name__[len("get_sweepset_") :]
                sweep_or_sweepset.sweepset_features[ft_name] = ft_out
        return ft_out

    return feature_func


@ephys_feature
def get_example_feature(
    sweep_or_sweepset: Union[EphysSweepFeatureExtractor, EphysSweepSetFeatureExtractor]
) -> Tuple[float, Dict]:
    """Extracts example ephys feature.

    depends on: feature_1, feature_2, ..., feature_n.
    description: This describes how the features gets computed.

    Example function definition
    '''
    @ephys_feature
    def get_example_feature(sweep_or_sweepset: Union[EphysSweepFeatureExtractor, EphysSweepSetFeatureExtractor])
        ft_value, ft_info = ephys_feature_init()  # init ft, ft_info = float("nan"), {}

        # do some feature calculations using sweep.
        ft_value = "some value"
        ft_info.update({"metadata": "contains intermediary results and feature metadata."})

        return ft_value, ft_info
    '''


    Args:
        sweep_or_sweepset (Union[EphysSweepFeatureExtractor, EphysSweepSetFeatureExtractor]): Sweep or sweepset to extract feature from.

    Returns:
        Tuple[float, Dict]: AP feature and optionally info
    """
    ft_value, ft_info = ephys_feature_init()  # init ft, ft_info = float("nan"), {}

    # do some feature calculations using sweep.
    ft_value = "some value"
    ft_info.update({"metadata": "contains intermediary results and feature metadata."})

    return ft_value, ft_info


class strip_sweep_ft_info:
    """Context manager to strip metadata from sweep features.

    Allows to have feature names overlap with EphysExtractor names.
    Internal EphysSweepExtractor features cannot be processed with metadata
    attached.

    Temporarily saves the sweep features, strips the metadata and restores
    the original features after the context is exited.

    Example:
        with strip_sweep_ft_info(sweep) as fsweep:
            tau = fsweep.estimate_time_constant()
        assert isinstance(tau, float)

    Args:
        sweep (EphysSweepFeatureExtractor): Sweep to strip metadata from.

    Returns:
        EphysSweepFeatureExtractor: Sweep with stripped metadata.
    """

    def __init__(self, sweep: EphysSweepFeatureExtractor):
        self.sweep = sweep

    def __enter__(self) -> EphysSweepFeatureExtractor:
        self.sweep_fts = self.sweep._sweep_features.copy()
        self.sweep._sweep_features = {
            k: strip_info(v) for k, v in self.sweep_fts.items()
        }
        return self.sweep

    def __exit__(self, exc_type, exc_value, traceback):
        self.sweep._sweep_features = self.sweep_fts.copy()


get_stripped_sweep_fts = lambda sweepset: sweepset.get_sweep_features().applymap(
    strip_info
)


def median_idx(d):
    if len(d) > 0:
        is_median = d == d.median()
        if any(is_median):
            return int(d.index[is_median].to_numpy())
        ranks = d.rank(pct=True)
        close_to_median = abs(ranks - 0.5)
        return int(np.array([close_to_median.idxmin()]))
    return slice(0)


class FeatureInfoError(ValueError):
    """Error raised when a feature has no diagnostic info."""

    pass


def ensure_ft_info(ft):
    is_dict = isinstance(ft, dict)
    if is_dict:
        if "description" in ft.keys():
            return ft
    raise FeatureInfoError(
        "Feature has no diagnostic info! Ensure return_ft_info=True when it is computed."
    )
