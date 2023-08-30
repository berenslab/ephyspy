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
from ephyspy.utils import is_spike_feature, is_sweep_feature, is_sweepset_feature


class EphysSweep(EphysSweepFeatureExtractor):
    r"""Wrapper around EphysSweepFeatureExtractor from the AllenSDK to
    support additional functionality.

    Mainly it supports the addition of new spike features and metadata.

    Attributes:
        metadata (dict): Metadata for the sweep.
        added_spike_features (dict): Dictionary of added spike features.
        features (dict): Dictionary of sweep features. These should be
            `SweepFeature` instances.
    """

    def __init__(self, *args, metadata: Dict = {}, **kwargs):
        """
        Args:
            metadata (dict, optional): Metadata for the sweep. Defaults to None.
                The metadata can be used to set hyperparameters for features or
                store identifying information, such as cell id etc..
            *args: Additional arguments for EphysSweepFeatureExtractor.
            **kwargs: Additional keyword arguments for EphysSweepFeatureExtractor.
        """
        super().__init__(*args, **kwargs)
        self.metadata = metadata
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
        r"""Add features to the `EphysSweep` instance.

        This function can be used to add spike or sweep features to an `EphysSweep`
        object. The added features can then be accessed via `self.features` or
        collectively computed via `self.get_features()`.

        Args:
            features (Union[List[Any], Dict[str, Any]]): List of features to add.

        Raises:
            TypeError: If feature is not of a known type."""
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

    def get_features(self, recompute: bool = False) -> Dict[str, float]:
        """Compute all features that have been added to the `EphysSweep` instance.

        Includes all features that can be found in `self.features`.

        Args:
            recompute (bool, optional): Whether to force recomputation of the
                features. Defaults to False.

        Returns:
            Dict[str, float]: Dictionary of features and values."""
        if hasattr(self, "features"):
            if self.features is not None:
                return {
                    k: ft.get_value(recompute=recompute)
                    for k, ft in self.features.items()
                }

    def plot(
        self, ax: Optional[Axes] = None, show_stimulus: bool = False, **kwargs
    ) -> Axes:
        """Plot the sweep.

        If no axes object is provided, one will be created. It will have one or
        two subplots, depending on whether the stimulus is shown.

        Args:
            ax (Axes, optional): Matplotlib axes to plot on.
            show_stimulus (bool, optional): Whether to plot the stimulus. Defaults to False.

        Returns:
            Axes: Matplotlib axes object.
        """
        if ax is None and show_stimulus:
            fig, ax = plt.subplots(
                2,
                1,
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
                constrained_layout=True,
            )
            v_ax, i_ax = ax
        elif ax is None and not show_stimulus:
            fig, ax = plt.subplots()
            v_ax = ax
        elif ax is not None and show_stimulus:
            v_ax, i_ax = ax
        else:
            v_ax = ax

        v_ax.plot(self.t.T, self.v.T, **kwargs)
        v_ax.set_xlabel("Time (s)")
        v_ax.set_ylabel("Voltage (mV)")
        if show_stimulus:
            i_ax.plot(self.t.T, self.i.T, **kwargs)
            i_ax.set_ylabel("Current (pA)")
            i_ax.set_xlabel("Time (s)")
            v_ax.set_xlabel("")
        return ax

    def plot_feature(
        self, ft: str, ax: Optional[Axes] = None, show_sweep: bool = True, **kwargs
    ) -> Axes:
        """Plot any feature of the sweep by specifying its name.

        Args:
            ft (str): Name of the feature to plot. (all lower case)
            ax (Axes, optional): Matplotlib axes to plot on.
            show_sweep (bool, optional): Whether to plot V(t).
                Defaults to True.

        Returns:
            Axes: Matplotlib axes object."""
        if ax is None:
            fig, ax = plt.subplots()
        if show_sweep:
            self.plot(ax=ax, color="k")

        # sweep feature
        if ft in self.features:
            self.features[ft].plot(ax=ax, **kwargs)

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
    ) -> Axes:
        """Plot multiple features of the sweep by specifying their names.

        Args:
            fts (List[str]): Names of the features to plot. (all lower case)
            ax (Axes, optional): Matplotlib axes to plot on.
            show_sweep (bool, optional): Whether to plot V(t).
                Defaults to True.

        Returns:
            Axes: Matplotlib axes object."""
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

    t_set, v_set and i_set are optional and `EphysSweepSet` can also be initialized
    using an iterable of sweeps via the `from_sweeps` method. In this case all the
    metadata already needs to be contained in the sweeps.

    Args:
        t_set (ndarray): Time array for set of sweeps.
        v_set (ndarray): Voltage array for set of sweeps.
        i_set (ndarray): Current array for set of sweeps.
        metadata (dict, optional): Metadata for the sweep set. Defaults to None.
        The metadata can be used to set hyperparameters for features or
            store identifying information, such as cell id etc..

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
        *args,
        **kwargs,
    ):
        """
        Args:
            t_set (ndarray, optional): Time array for set of sweeps.
            v_set (ndarray, optional): Voltage array for set of sweeps.
            i_set (ndarray, optional): Current array for set of sweeps.
            t_start (ndarray, optional): Start time for each sweep.
            t_end (ndarray, optional): End time for each sweep.
            metadata (dict, optional): Metadata for the sweep set.
            *args: Additional arguments for EphysSweepSetFeatureExtractor.
            **kwargs: Additional keyword arguments for EphysSweepSetFeatureExtractor.
        """
        is_array = lambda x: isinstance(x, ndarray) and x is not None
        is_float = lambda x: isinstance(x, float) and x is not None
        t_set = list(t_set) if is_array(t_set) else t_set
        v_set = list(v_set) if is_array(v_set) else v_set
        i_set = list(i_set) if is_array(i_set) else i_set
        if t_start is None and v_set is not None:
            t_start, t_end = np.array(t_set)[:, [0, -1]].T.tolist()
        elif is_float(t_start):
            t_start, t_end = np.array([[t_start, t_end]] * len(t_set)).T

        super().__init__(t_set, v_set, i_set, t_start, t_end, *args, **kwargs)
        self.metadata = metadata
        for sweep in self.sweeps():
            sweep.metadata = metadata
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

    def __len__(self) -> int:
        return len(self.sweeps())

    def __getitem__(self, idx: int) -> EphysSweep:
        return self.sweeps()[idx]

    def add_spike_feature(self, feature_name: str, feature_func: Callable):
        """Add a new spike feature to the extractor.

        Adds new spike feature to each `EphysSweep` instance.

        Args:
            feature_name (str): Name of the new feature.
            feature_func (Callable): Function to calculate the new feature.

        Raises:
            TypeError: If feature is not of a known type.
        """
        for sweep in self.sweeps():
            sweep.add_spike_feature(feature_name, feature_func)

    def add_features(self, features: Union[List[Any], Dict[str, Any]]):
        r"""Add features to the `EphysSweepSet` instance.

        This function can be used to add spike sweep or sweepset features to an
        `EphysSweepSet` object. The added features can then be accessed via
        `self.features` or collectively computed via `self.get_features()`.

        Sweep features are passed to each sweep in the set and added there.
        Any sweep features needed in their computation are automatically added
        and handled by `SweepSetFeature` instances. This means they don't
        necessarily need to be added manually.

        Args:
            features (Union[List[Any], Dict[str, Any]]): List of features to add.

        Raises:
            TypeError: If feature is not of a known type."""
        if isinstance(features, Dict):
            features = list(features.values())

        for ft in features:
            if is_spike_feature(ft):
                self.add_spike_feature(ft.__name__, ft)
            elif is_sweep_feature(ft):
                for sweep in self:
                    sweep.add_features([ft])
            elif is_sweepset_feature(ft):
                feature = ft(self, compute_at_init=False)
                self.features.update({feature.name: feature})
            else:
                raise TypeError("Feature is not of a known type.")

    def set_stimulus_amplitude_calculator(self, func: Callable):
        """Set stimulus amplitude calculator for each sweep.

        This is potentially only relevant for working with the AllenSDK and should
        not be needed for using `ephyspy` on its own.

        Args:
            func (Callable): Function to calculate stimulus amplitude.
        """
        for sweep in self.sweeps():
            sweep.set_stimulus_amplitude_calculator(func)

    def get_features(self, recompute: bool = False) -> Dict[str, float]:
        """Compute all features that have been added to the `EphysSweepSet` instance.

        Includes all features that can be found in `self.features`.

        Args:
            recompute (bool, optional): Whether to force recomputation of the
                features. Defaults to False.

        Returns:
            Dict[str, float]: Dictionary of features and values."""
        if hasattr(self, "features"):
            if self.features is not None:
                return {
                    k: ft.get_value(recompute=recompute)
                    for k, ft in self.features.items()
                }

    def get_sweep_features(self, recompute: bool = False) -> Dict[str, List[float]]:
        """Collect features on a sweep level.

        This computes / looks up all features that have been computed at the
        sweep level and returns them as a dictionary of lists. Each list contains
        the values for the respective feature for each sweep, i.e.
        `get_sweep_features()[feature_name][sweep_idx]` returns the value of
        `feature_name` for the `sweep_idx`-th sweep.

        Args:
            recompute (bool, optional): Whether to force recomputation of the
                features. Defaults to False.

        Returns:
            Dict[str, List[float]]: Dictionary of features and values.
        """
        if hasattr(self, "features"):
            if self.features is not None:
                LD = [sw.get_features(recompute=recompute) for sw in self.sweeps()]
                return {k: [dic[k] for dic in LD] for k in LD[0]}

    def plot(
        self, ax: Optional[Axes] = None, show_stimulus: bool = False, **kwargs
    ) -> Axes:
        """Plot all sweeps in the set.

        If no axes object is provided, one will be created. It will have one or
        two subplots, depending on whether the stimulus is shown.

        Args:
            ax (Axes, optional): Matplotlib axes to plot on.
            show_stimulus (bool, optional): Whether to plot the stimulus. Defaults to False.

        Returns:
            Axes: Matplotlib axes object.
        """
        if ax is None:
            if show_stimulus:
                fig, ax = plt.subplots(
                    2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
                )
            else:
                fig, ax = plt.subplots()
        for sweep in self.sweeps():
            sweep.plot(ax=ax, show_stimulus=show_stimulus, **kwargs)
        return ax
