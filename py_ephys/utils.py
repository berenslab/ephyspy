import pandas as pd
import numpy as np

import re
import py_ephys.allen_sdk.ephys_extractor as efex
from typing import Callable, Union, Dict, List, Optional, Tuple
from numpy import ndarray


class EphysSweepSetFeatureExtractor(efex.EphysSweepSetFeatureExtractor):
    """Wrapper around efex.EphysSweepSetFeatureExtractor to support additional functionality.

    Args:
        t_set (ndarray): Time array for set of sweeps.
        v_set (ndarray): Voltage array for set of sweeps.
        i_set (ndarray): Current array for set of sweeps.
        metadata (dict, optional): Metadata for the sweep set. Defaults to None.
        *args: Additional arguments for efex.EphysSweepSetFeatureExtractor.
        **kwargs: Additional keyword arguments for efex.EphysSweepSetFeatureExtractor.

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
        *args,
        **kwargs
    ):
        is_array = lambda x: isinstance(x, ndarray) and x is not None
        t_set = [t for t in t_set] if is_array(t_set) else t_set
        v_set = [v for v in v_set] if is_array(v_set) else v_set
        i_set = [i for i in i_set] if is_array(i_set) else i_set
        # TODO: t_start and t_end should be able to be supplied as floats
        t_start = [
            t[1] for t in t_set
        ]  # if is_array(t_start) else t_start  # with t[0] warnings are thrown
        t_end = [t[-1] for t in t_set]  # if is_array(t_end) else t_end
        super().__init__(t_set, v_set, i_set, t_start, t_end, *args, **kwargs)
        self.spike_feature_funcs = {}
        self.sweep_feature_funcs = {}
        self.metadata = metadata

    def get_sweep_features(self):
        l = []
        for swp in self.sweeps():
            swp._sweep_features
            l.append(swp._sweep_features)
        return pd.DataFrame(l)

    def get_sweep_feature(self, key):
        return self.get_sweep_features()[key]

    def add_spike_feature(self, feature_name, feature_func):
        self.spike_feature_funcs[feature_name] = feature_func

    def add_sweep_feature(self, feature_name, feature_func):
        self.sweep_feature_funcs[feature_name] = feature_func

    def process_spikes(self):
        """Analyze spike features for all sweeps."""
        for sweep in self._sweeps:
            sweep.process_spikes()
            for ft, ft_func in self.spike_feature_funcs.items():
                sweep.process_new_spike_feature(ft, ft_func)

            for ft, ft_func in self.sweep_feature_funcs.items():
                sweep.process_new_sweep_feature(ft, ft_func)

    def set_stimulus_amplitude_calculator(self, func):
        for sweep in self._sweeps:
            sweep.set_stimulus_amplitude_calculator(func)

    def get_sweepset_statistics(self):
        """Get statistics for all sweeps."""
        # filter and postprocess the sweep and spike fts
        # return means and representative fts
        # return dataframe or dict
        # includes postprocessing on sweepset level
        return self.get_sweep_features().aggregate(["median", "mean", "std"]).T


def has_stimulus(sweep: efex.EphysSweepFeatureExtractor) -> bool:
    """Check if sweep has stimulus that is non-zero.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to check.

    Returns:
        bool: True if sweep has stimulus."""
    return np.any(sweep.i != 0)


def is_hyperpolarizing(sweep: efex.EphysSweepFeatureExtractor) -> bool:
    """Check if sweep is hyperpolarizing, i.e. if the stimulus < 0.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to check.

    Returns:
        bool: True if sweep is hyperpolarizing."""
    return np.any(sweep.i < 0)


def is_depol(sweep: efex.EphysSweepFeatureExtractor) -> bool:
    """Check if sweep is depolarizing, i.e. if the stimulus > 0.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to check.

    Returns:
        bool: True if sweep is depolarizing."""
    return np.any(sweep.i > 0)


def where_stimulus(sweep: efex.EphysSweepFeatureExtractor) -> ndarray:
    """Get mask where stimulus is non-zero.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to check.

    Returns:
        ndarray: Mask where stimulus is non-zero."""
    if has_stimulus(sweep):
        peaks = strip_info(sweep.spike_feature("peak_i"))
        # peaks = sweep.spike_feature("threshold_i") # unclear which is better
        if len(peaks) > 0:
            return peaks != 0
        return np.zeros_like(peaks, dtype=bool)
    return np.zeros(0, dtype=bool)


def has_rebound(sweep: efex.EphysSweepFeatureExtractor, T_rebound: float = 0.3) -> bool:
    """Check if sweep rebounds.

    description: rebound if voltage exceeds baseline after stimulus offset.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to check.
        T_rebound (float, optional): Time window after stimulus offset in which
            rebound can occur. Defaults to 0.3.

    Returns:
        bool: True if sweep rebounds."""
    if is_hyperpolarizing(sweep):
        end = strip_info(sweep.sweep_feature("stim_end"))
        v_baseline = strip_info(sweep.sweep_feature("v_baseline"))
        ts_rebound = np.logical_and(sweep.t > end, sweep.t < end + T_rebound)
        return np.any(sweep.v[ts_rebound] > v_baseline)
    return False


def parse_ft_desc(func: Callable) -> str:
    """Parses docstrings for feature descriptions.

    Docstrings should have the following format:
    <Some text>
    description: <description text>
    <Some more text>

    Args:
        func (Callable): Function to parse docstring of.

    Returns:
        string: Description of the feature that the function extracts.
    """
    func_doc = func.__doc__
    pattern = re.compile(r"description: (.*)")
    match = pattern.search(func_doc)
    if match:
        return match.group(1)
    return ""


where_between = lambda t, t0, tend: np.logical_and(t > t0, t < tend)


def get_sweep_sag_idxs(sweep: efex.EphysSweepFeatureExtractor):
    """determine idxs in a sweep that are part of the sag.

    description: all idxs below steady state and during stimulus.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): sweep to analyze.

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
        v_steady = strip_info(sweep.sweep_feature("v_deflect"))
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
    if return_ft_info:
        ft_dict = {"value": ft}
        ft_dict.update(ft_info)
        return ft_dict
    return ft


def ephys_feature_init(ft_info_init: Dict = {}) -> Tuple[float, Dict]:
    return float("nan"), ft_info_init


def ephys_feature(feature: Callable) -> Callable:
    def feature_func(
        sweep: efex.EphysSweepFeatureExtractor, return_ft_info: bool = False
    ) -> Union[float, Dict]:
        ft_out = feature(sweep)
        if isinstance(ft_out, Tuple):
            ft, ft_info = ft_out
        else:
            ft, ft_info = ft_out, {}
        ft_info["description"] = parse_ft_desc(feature)
        return include_info(ft, ft_info, return_ft_info)

    return feature_func


@ephys_feature
def get_example_feature(sweep: efex.EphysSweepFeatureExtractor) -> Tuple[float, Dict]:
    """Extracts example ephys feature.

    description: This describes how the features gets computed.

    Args:
        sweep (efex.EphysSweepFeatureExtractor): Sweep to extract feature from.

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
        sweep (efex.EphysSweepFeatureExtractor): Sweep to strip metadata from.

    Returns:
        efex.EphysSweepFeatureExtractor: Sweep with stripped metadata.
    """

    def __init__(self, sweep: efex.EphysSweepFeatureExtractor):
        self.sweep = sweep

    def __enter__(self) -> efex.EphysSweepFeatureExtractor:
        self.sweep_fts = self.sweep._sweep_features.copy()
        self.sweep._sweep_features = {
            k: strip_info(v) for k, v in self.sweep_fts.items()
        }
        return self.sweep

    def __exit__(self, exc_type, exc_value, traceback):
        self.sweep._sweep_features = self.sweep_fts.copy()
