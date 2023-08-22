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

from typing import Any, Callable, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes
from numpy import ndarray

import ephyspy.allen_sdk.ephys_extractor as efex
from ephyspy.allen_sdk.ephys_extractor import (
    EphysSweepFeatureExtractor,
    EphysSweepSetFeatureExtractor,
)
from ephyspy.plot import (
    plot_spike_feature,
    plot_spike_features,
    plottable_spike_features,
)


def is_spike_feature(ft):
    return type(ft).__name__ == "function"


def is_sweep_feature(ft):
    if hasattr(ft, "__base__"):  # TODO: Find better criterion
        return "EphysFeature" in ft.__base__.__name__
    return False


def is_sweepset_feature(ft):
    return "SweepsetFeature" in type(ft).__base__.__name__


class EphysSweep(EphysSweepFeatureExtractor):
    """Wrapper around EphysSweepFeatureExtractor from the AllenSDK to
    support additional functionality.

    Mainly it supports the addition of new spike features and plotting them.
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

    def add_features(self, features: Union[List[Any], Dict[str, Any]]):
        if isinstance(features, Dict):
            features = list(features.values())

        for ft in features:
            if is_spike_feature(ft):
                self.add_spike_feature(ft.__name__, ft)
            elif is_sweep_feature(ft):
                feature = ft(self, compute_at_init=False)
                self.features.update({feature.name: feature})
            else:
                raise TypeError("Feature is not of a known type.")

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

    def get_features(self, recompute=False):
        if hasattr(self, "features"):
            if self.features is not None:
                return {
                    k: ft.get_value(recompute=recompute)
                    for k, ft in self.features.items()
                }

    def plot(self, ax=None, **kwargs):
        # TODO: Make nice and add options!
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.t.T, self.v.T, **kwargs)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage (mV)")
        return ax

    def plot_feature(self, ft: str, ax=None, show_sweep=True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        if show_sweep:
            self.plot(ax=ax, color="k")

        # sweep feature
        # if ft in self.features:
        #     self.features[ft].plot(ax=ax, **kwargs)

        # spike feature
        if not self._spikes_df.empty:
            if ft in plottable_spike_features:
                plot_spike_feature(self, ft, ax=ax, **kwargs)
        else:
            raise ValueError(f"Feature {ft} not found.")
        ax.legend()
        return ax

    def plot_features(
        self,
        fts: List[str],
        ax: Optional[Axes] = None,
        show_sweep: bool = True,
        **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots()
        if show_sweep:
            self.plot(ax=ax, color="k")
        for ft in fts:
            self.plot_feature(ft, ax=ax, show_sweep=False, **kwargs)
        return ax


# overwrite AllenSDK EphysSweepFeatureExtractor with wrapper
# this is needed to EphysSweepSet is uses EphysSweep to initialize the individual sweeps
# instead of using the AllenSDK EphysSweepFeatureExtractor
efex.EphysSweepFeatureExtractor = EphysSweep


class EphysSweepSet(EphysSweepSetFeatureExtractor):
    """Wrapper around EphysSweepSetFeatureExtractor from the AllenSDK to
    support additional functionality.

    Args:
        t_set (ndarray): Time array for set of sweeps.
        v_set (ndarray): Voltage array for set of sweeps.
        i_set (ndarray): Current array for set of sweeps.
        metadata (dict, optional): Metadata for the sweep set. Defaults to None.
        *args: Additional arguments for EphysSweepSetFeatureExtractor.
        **kwargs: Additional keyword arguments for EphysSweepSetFeatureExtractor.

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
        self.features = {}

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

    def __getitem__(self, idx: int) -> EphysSweep:
        return self.sweeps()[idx]

    def add_spike_feature(self, feature_name: str, feature_func: Callable):
        """Add a new spike feature to the extractor.

        Adds new spike feature to each `EphysSweep` instance.

        Args:
            feature_name (str): Name of the new feature.
            feature_func (Callable): Function to calculate the new feature.
        """
        for sweep in self.sweeps():
            sweep.add_spike_feature(feature_name, feature_func)

    def add_features(self, features: Union[List[Any], Dict[str, Any]]):
        if isinstance(features, Dict):
            features = list(features.values())

        for ft in features:
            if is_spike_feature(ft):
                self.add_spike_feature(ft.__name__, ft)
            elif is_sweepset_feature(ft):  # needs to be checked b4 sweep feature
                feature = ft(self, compute=False)
                self.features.update({feature.name: feature})
            elif is_sweep_feature(ft):
                for sweep in self:
                    sweep.add_features([ft])
            else:
                raise TypeError("Feature is not of a known type.")

    def set_stimulus_amplitude_calculator(self, func: Callable):
        """Set stimulus amplitude calculator for each sweep.

        Args:
            func (Callable): Function to calculate stimulus amplitude.
        """
        for sweep in self.sweeps():
            sweep.set_stimulus_amplitude_calculator(func)

    def get_features(self, recompute=False):
        if hasattr(self, "features"):
            if self.features is not None:
                return {
                    k: ft.get_value(recompute=recompute)
                    for k, ft in self.features.items()
                }

    def get_sweep_features(self, recompute=False):
        if hasattr(self, "features"):
            if self.features is not None:
                LD = [sw.get_features(recompute=recompute) for sw in self.sweeps()]
                return {k: [dic[k] for dic in LD] for k in LD[0]}

    def show(self):
        # TODO: Make nice and add options!
        plt.plot(self.t.T * 1000, self.v.T)
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.show()
