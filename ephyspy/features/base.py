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

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import Axes
from numpy import ndarray

from ephyspy.features.utils import FeatureError, fetch_available_fts
from ephyspy.sweeps import EphysSweep, EphysSweepSet
from ephyspy.utils import (
    is_sweep_feature,
    is_sweepset_feature,
    parse_deps,
    parse_desc,
    parse_func_doc_attrs,
    unpack,
)


class BaseFeature(ABC):
    r"""Base class for all electrophysiological features.

    This class defines the interface for all electrophysiological features.
    All sweep features should inherit from this class, and must implement a
    `_compute` and `_data_init` method. The `_compute` method should return the
    feature value and optionally save diagnostic information for later debugging to
    `self._diagnostics`. The `_data_init` method should be used to set the
    `self.data` attribute, and add the feature to the `self.data.features`.

    The description of the feature should contain a short description of the
    feature, and a list of dependencies. The dependencies should be listed
    as a comma separated list of feature names. It is parsed and can be displayed
    but has no functional use for now. Furthermore, the units of the feature
    should be specified. If the feature is unitless, the units should be set to "/".

    The docstring should have the following format:

    '''<Some Text>

    description: <Short description of the feature>.
    depends on: <Comma separated list of dependencies>.
    units: <Units of the feature>.

    <Some more text>'''

    `BaseFeature`s can also implement a _plot method, that displays the diagnostic
    information or the feature itself. If the feature cannot be displayed in a V(t)
    or I(t) plot, instead the `plot` method should be overwritten directly. This
    is because `plot` wraps `_plot` adds additional functionality ot it.
    """

    def __init__(
        self,
        data: Optional[EphysSweep] = None,
        compute_at_init: bool = True,
        name: Optional[str] = None,
        store_with_data: bool = True,
    ):
        r"""
        Args:
            data: EphysSweep object.
                Can also be passed later with `__call__`.
            compute_at_init: If True, compute the feature at initialization.
                Otherwise the feature is only copmuted when `__call__` or
                `get_value` is called. This can be useful when instantiating
                many features at once, and waiting with the computation until
                the features are actually needed.
            name: Custom name of the feature. If None, the name of the feature
                class is used.
            store_with_data: If True, store the feature in the `self.data.features`
                dictionary.
        """
        self.name = self.__class__.__name__.lower() if name is None else name
        self.name = (
            self.name.replace("sweep_", "")
            .replace("spike_", "")
            .replace("sweepset_", "")
        )
        self._value = None
        self._diagnostics = None

        self.__call__(
            data,
            compute=compute_at_init,
            store_with_data=store_with_data,
            return_value=False,
        )

        self.parse_docstring()

    def parse_docstring(self):
        if self.__class__.__doc__ is not None:
            attrs = parse_func_doc_attrs(self.__class__)
            self.description = (
                None if not "description" in attrs else attrs["description"]
            )
            self.depends_on = None if not "depends on" in attrs else attrs["depends on"]
            self.units = None if not "units" in attrs else attrs["units"]
            self.units = "" if self.units == "/" else self.units

    def _data_init_incl_storage(self, data: Union[EphysSweep, EphysSweepSet]):
        self._data_init(data)
        if not self.store_with_data and not self.data is None:
            self.data.features.pop(self.name)

    @abstractmethod
    def _data_init(self, data: Union[EphysSweep, EphysSweepSet]):
        """Initialize the feature with a EphysSweep or EphysSweepSet object.

        This method is called at initialization and when the feature is
        called with a new data object. It should be used to set the `self.data`
        attribute, and add the feature to the `self.data.features` dictionary.
        It can further be used to add any pre-existing / pre-computed features
        stored in `self.data.features` to the class attributes (`_value`,
        `_diagnostics`, etc.).

        Args:
            data: EphysSweep object.
        """
        self.data = data

    def ensure_correct_hyperparams(self):
        """Ensure that parameters passed with the data are used in computation.

        Both EphysSweep and EphysSweepSet can come with metadata attached. This
        metadata can be used to set default values for hyperparameters of
        features. This method ensures that these hyperparameters are used in
        computation. It should be called in `_data_init` after setting the
        `self.data` attribute.
        """
        metadata = self.data.metadata
        new_defaults = {kw: v for kw, v in metadata.items() if kw in self.__dict__}
        if len(new_defaults) > 0:
            self.__dict__.update(new_defaults)

    def __repr__(self):
        return f"{self.name} for {self.data}"

    def __str__(self):
        if self._value is not None:
            return f"{self.name} = {self._value:.3f} {self.units}"
        else:
            return f"{self.name} = ? {self.units}"

    @abstractmethod
    def _compute(
        self, recompute: bool = False, store_diagnostics: bool = True
    ) -> float:
        """Compute the feature.

        All computation that is neccesary to yield the value of the feature should
        be defined here. This is the core method of BaseFeature and all other
        functionality interacts with this method.

        Alongside computing the value of the corresponding feature, this method
        can also be used to updat the `_diagnostics` attribute, which is a
        dictionary that can be used to store any additional information about
        the feature computation. This can be useful for debugging or better
        understanding how a feature was computed. Diagnostic information can
        be accessed with `get_diagnostics` or via the `diagnostics` property and
        updated with `_update_diagnostics`.

        Args:
            recompute: If True, recompute the feature even if it is already
                computed.
            store_diagnostics: If True, store any additional information about
                the feature computation in the `_diagnostics` attribute.

        Returns:
            The value of the feature.
        """
        # load dependencies using lookup_sweep_feature or lookup_spike_feature
        # do some computation
        # save diagnostics using _update_diagnostics
        return

    def recompute(self) -> float:
        """Convencience method to recompute the feature.

        This method is equivalent to calling `get_value` with `recompute=True`
        and `store_diagnostics=True`.

        Returns:
            The value of the feature."""
        return self.get_value(recompute=True, store_diagnostics=True)

    def get_diagnostics(self, recompute: bool = False) -> Dict[str, Any]:
        """Get diagnostic information about how a feature was computed.

        This method returns any intermediary results obtained during computation
        of the feature that has been stored in `_diagnostics`.  If the feature
        is not yet computed, it will be computed first.

        Args:
            recompute: If True, recompute the feature even if it is already
                computed.

        Returns:
            A dictionary with diagnostic information about the feature computation.
        """
        if recompute or self._diagnostics is None:
            self.get_value(recompute=recompute, store_diagnostics=True)
        return self._diagnostics

    @property
    def diagnostics(self) -> Dict[str, Any]:
        return self.get_diagnostics()

    def _update_diagnostics(self, dct: Dict[str, Any]):
        """Update the `_diagnostics` attribute with a dictionary.

        This method can be used to store information about the
        feature computation in the `_diagnostics` attribute of the object.
        This method should be called in `_compute` if `store_diagnostics` is
        True.

        Args:
            dct: Dictionary with additional diagnostic information."""
        if self._diagnostics is None:
            self._diagnostics = {}
        self._diagnostics.update(dct)

    def get_value(
        self, recompute: bool = False, store_diagnostics: bool = True
    ) -> float:
        """Get the value of the feature.

        Allows to force recomputation of the feature and toggle whether
        diagnostic information should be stored.

        Args:
            recompute: If True, recompute the feature even if it is already
                computed.
            store_diagnostics: If True, store any additional information about
                the feature computation in the `_diagnostics` attribute.

        Returns:
            The value of the feature.
        """
        if recompute or self._value is None and not self.data is None:
            self._value = self._compute(
                recompute=recompute,
                store_diagnostics=store_diagnostics,
            )
        return self._value

    @property
    def value(self) -> Any:
        return self.get_value()

    @value.setter
    def _set_value(self, value):
        self._value = value

    def __call__(
        self,
        data: EphysSweep = None,
        compute: bool = False,
        store_diagnostics: bool = True,
        store_with_data: bool = True,
        return_value: bool = False,
    ) -> Union[float, SweepFeature]:
        """Compute the feature for a given dataset.

        Essentially chains together `_data_init` and `get_value`.

        Args:
            data: The dataset to compute the feature for, i.e. an instance of
                `EphysSweep`.
            compute: If True, compute the feature.
            store_diagnostics: If True, store any additional information about
                the feature computation in the `_diagnostics` attribute.
            store_with_data: If True, store the feature in the `self.data.features`
                dictionary.
            return_value: If True, return the value of the feature, otherwise
                return the feature object.

        Returns:
            The value of the feature.
        """
        self.store_with_data = store_with_data
        self._data_init_incl_storage(data)

        if not data is None and compute:
            self.get_value(
                recompute=False,
                store_diagnostics=store_diagnostics,
            )
        if not data is None and return_value:
            return self.value
        return self

    def plot(
        self,
        *args,
        ax: Optional[Axes] = None,
        show_sweep: bool = False,
        show_stimulus: bool = False,
        sweep_kwargs: Optional[Dict[str, Any]] = {"color": "grey", "alpha": 0.5},
        **kwargs,
    ) -> Axes:
        """Adds additional kwargs and functionality to `BaseFeature`._plot`.

        Before calling `BaseFeature._plot`, this function checks if the feature
        is a stimulus feature and if so, ensures the feature is plotteed onto
        the stimulus axis. Additionally along with every feature, the sweep
        can be plotted. Same goes for the stimulus.

        If no axis is provided one is created.
        This function can be (and should be overwritten) if the feature cannot
        be displayed on top of the unterlying sweep.

        Args:
            self (BaseFeature): Feature to plot. Needs to have a `plot` method.
            *args: Additional arguments to pass to `self.plot`.
            ax (Optional[Axes], optional): Axes to plot on.
            show_sweep (bool, optional): Whether to plot the sweep. Defaults to False.
            show_stimulus (bool, optional): Whether to plot the stimulus. Defaults to False.
            kwargs: Additional kwargs to pass to `self.plot`.

        Returns:
            Axes: Axes of plot.
        """
        is_stim_ft = self.name in ["stim_amp", "stim_onset", "stim_end"]
        if show_sweep:
            show_stimulus = is_stim_ft or show_stimulus
            # let self.data.plot handle creation of axes
            axes = self.data.plot(show_stimulus=show_stimulus, **sweep_kwargs)
            ax = axes[0] if show_stimulus else axes
            ax = axes[1] if is_stim_ft else ax
        elif show_stimulus and is_stim_ft:
            axes = plt.gca() if ax is None else ax
            axes.plot(self.data.t, self.data.i, **sweep_kwargs)
            axes.set_ylabel("Current (pA)")
            ax = axes
        elif show_stimulus and not is_stim_ft:
            if ax is None:
                fig, axes = plt.subplots(
                    2,
                    1,
                    sharex=True,
                    gridspec_kw={"height_ratios": [3, 1]},
                    constrained_layout=True,
                )
                ax = axes[0]
            else:
                axes = ax
            axes[1].plot(self.data.t, self.data.i, **sweep_kwargs)
            axes[1].set_ylabel("Current (pA)")
        else:
            axes = plt.gca() if ax is None else ax
            ax = axes

        if np.all(np.isnan(self.value)):
            return axes

        if self.diagnostics is None:
            self.get_diagnostics(recompute=True)
        ax = self._plot(*args, ax=ax, **kwargs)

        if not ax.get_xlabel():
            ax.set_xlabel("Time (s)")
        if not ax.get_ylabel():
            ax.set_ylabel("Voltage (mV)")

        # if ax has artists with legend handles
        # add legend
        if len(ax.get_legend_handles_labels()[0]) > 0:
            ax.legend()
        return axes

    def _plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        """Plot the feature.

        Similar to _compute, this method implements a core functionality of
        SweepFeature. It is not an abstract feature though. It is called by
        `plot` and can be used to visualize the feature in any shape of form.
        If the feature cannot be plotted on top of the underlying sweep, `plot`
        should be overwritten directly.

        Args:
            *args: Additional arguments to pass.
            ax (Optional[Axes], optional): Axes to plot on.
            kwargs: Additional kwargs to pass to `self.plot`.

        Returns:
            Axes: Axes of plot.
        """
        raise NotImplementedError(f"This method does not exist for {self.name}.")
        # implements a plotting method
        return ax


class SpikeFeature(BaseFeature):
    r"""Base class for all spike level electrophysiological features.

    All spike features should inherit from this class, and must implement the
    `_compute` method. The `_compute` method should return the feature value
    and optionally save diagnostic information for later debugging to
    `self._diagnostics`.

    Compared to `SweepFeature`, `SpikeFeature` behaves slightly differently.
    Firstly, since spike features are computed on the spike level, results come
    in the form of a vector, where each entry corresponds to a spike. Similar to
    before this vector is stored in the `_value` attribute. However, because the
    handling the spike features is left to the AllenSDK's `process_spikes`, they
    `SpikeFeature` just provides an interface to the `_spikes_df` attribute of
    the underlying `EphysSweep` object. Secondly, the spike features in the
    AllenSDK are defined in a functional manner. This means the `__call__` method
    of `SpikeFeature` provides the required functional interface to be able to
    compute spike features with `EphysSweep.process_spikes`, while being able to
    provide additional functionality to the spike feature class.

    Currently, no diagnostics or recursive feature lookup is supported for spike
    features! For now this class mainly just acts as a feature function.

    The description of the feature should contain a short description of the
    feature, and a list of dependencies. The dependencies should be listed
    as a comma separated list of feature names. It is parsed and can be displayed
    but has no functional use for now. Furthermore, the units of the feature
    should be specified. If the feature is unitless, the units should be set to "/".

    The docstring should have the following format:

    '''<Some Text>

    description: <Short description of the feature>.
    depends on: <Comma separated list of dependencies>.
    units: <Units of the feature>.

    <Some more text>'''

    All computed features are added to the underlying `EphysSweep`
    object, and can be accessed via `lookup_spike_feature`. The methods will
    first check if the feature is already computed, and if not, instantiate and
    compute it. Any dependencies already computed will be reused, unless
    `recompute=True` is passed.

    `SpikeFeature`s can also implement a _plot method, the feature. If the
    feature cannot be displayed in a V(t) or I(t) plot, instead the `plot` method
    should be overwritten directly. This is because `plot` wraps `_plot` adds
    additional functionality ot it.
    """

    # TODO: Add support for recursive feature lookup and diagnostics
    def __init__(
        self,
        data: Optional[EphysSweep] = None,
        compute_at_init: bool = True,
        name: Optional[str] = None,
        store_with_data: bool = True,
    ):
        r"""
        Args:
            data: EphysSweep object.
                Can also be passed later with `__call__`.
            compute_at_init: If True, compute the feature at initialization.
                Otherwise the feature is only copmuted when `__call__` or
                `get_value` is called. This can be useful when instantiating
                many features at once, and waiting with the computation until
                the features are actually needed.
            name: Custom name of the feature. If None, the name of the feature
                class is used.
            store_with_data: If True, store the feature in the `self.data.features`
                dictionary. CURRENTLY NOT SUPPORTED FOR SPIKE FEATURES!
        """
        super().__init__(
            data,
            compute_at_init=compute_at_init,
            name=name,
            store_with_data=store_with_data,
        )

    def _data_init(self, data: EphysSweep):
        """Initialize the feature with a EphysSweep object.

        Sets self.data and ensures correct hyperparameters.

        Args:
            data: EphysSweep object.
        """
        self.data = data
        if data is not None:
            assert isinstance(data, EphysSweep), "data must be EphysSweep"
            self.type = type(data).__name__
            self.ensure_correct_hyperparams()

    def lookup_spike_feature(
        self, feature_name: str, recompute: bool = False
    ) -> ndarray:
        """Look up a spike level feature and return its value.

        This method will first check if the feature is already computed,
        and if not, compute all spike level features using `process_spikes` from
        the underlying `EphysSweep` object, and then
        instantiate and compute the feature.

        Args:
            feature_name: Name of the feature to look up.
            recompute: If True, recompute the feature even if it is already
                computed.

        Returns:
            The value of the feature for each detected spike.
        """
        if not hasattr(self.data, "_spikes_df") or recompute:
            self.data.process_spikes()
        elif (
            feature_name in self.data.added_spike_features
            and feature_name not in self.data._spikes_df.columns
        ):
            self.data.process_spikes()
        return self.data.spike_feature(feature_name, include_clipped=True)

    def __str__(self):
        name = f"{self.name}\n"
        vals = "\n".join(
            [f"{i}: {v:.3f} {self.units}" for i, v in enumerate(self._value)]
        )
        if self._value is not None:
            return name + vals
        else:
            return f"{self.name}\n0: ? {self.units}"

    @abstractmethod
    def _compute(
        self, recompute: bool = False, store_diagnostics: bool = True
    ) -> ndarray:
        """Compute the feature.

        All computation that is neccesary to yield the value of the feature should
        be defined here. This is the core method of SpikeFeature and all other
        functionality interacts with this method.

        Alongside computing the value of the corresponding feature, this method
        can also be used to updat the `_diagnostics` attribute, which is a
        dictionary that can be used to store any additional information about
        the feature computation. This can be useful for debugging or better
        understanding how a feature was computed. Diagnostic information can
        be accessed with `get_diagnostics` or via the `diagnostics` property and
        updated with `_update_diagnostics`.

        When `__call__` is called `_compute` can be thought of as a function
        that takes in data (`EphysSweep`) and returns a vector of features.

        Args:
            recompute: If True, recompute the feature even if it is already
                computed.
            store_diagnostics: If True, store any additional information about
                the feature computation in the `_diagnostics` attribute.

        Returns:
            The value of the feature.
        """
        # load dependencies using lookup_sweep_feature or lookup_spike_feature
        # do some computation
        # save diagnostics using _update_diagnostics
        return

    def get_diagnostics(self, recompute: bool = False):
        """Overwrite get_diagnostics to return None.

        Diagnostics is currently not supported for spike features."""
        # No diagnostics for spike features for now!
        return None

    def __call__(
        self,
        data: EphysSweep = None,
        compute: bool = False,
        store_diagnostics: bool = True,
        store_with_data: bool = True,
        return_value: bool = True,
    ) -> Union[float, SweepFeature]:
        """Compute the feature for a given dataset.

        Essentially chains together `_data_init` and `get_value`.

        Args:
            data: The dataset to compute the feature for, i.e. an instance of
                `EphysSweep`.
            compute: If True, compute the feature.
            store_diagnostics: If True, store any additional information about
                the feature computation in the `_diagnostics` attribute.
            store_with_data: If True, store the feature in the `self.data.features`
                dictionary. CURRENTLY NOT SUPPORTED FOR SPIKE FEATURES!
            return_value: If True, return the value of the feature, otherwise
                return the feature object.

        Returns:
            The value of the feature.
        """
        return super().__call__(
            data,
            compute,
            store_diagnostics,
            True,  # disables removal of feature from data.features
            return_value,
        )


class SweepFeature(BaseFeature):
    r"""Base class for all sweep level electrophysiological features.

    All sweep features should inherit from this class, and must implement the
    `_compute` method. The `_compute` method should return the feature value
    and optionally save diagnostic information for later debugging to
    `self._diagnostics`.

    The description of the feature should contain a short description of the
    feature, and a list of dependencies. The dependencies should be listed
    as a comma separated list of feature names. It is parsed and can be displayed
    but has no functional use for now. Furthermore, the units of the feature
    should be specified. If the feature is unitless, the units should be set to "/".

    The docstring should have the following format:

    '''<Some Text>

    description: <Short description of the feature>.
    depends on: <Comma separated list of dependencies>.
    units: <Units of the feature>.

    <Some more text>'''

    All computed features are added to the underlying `EphysSweep`
    object, and can be accessed via `lookup_sweep_feature` or `lookup_spike_feature`.
    The methods will first check if the feature is already computed, and if not,
    instantiate and compute it. This works recursively, so that features can depend
    on other features as long as they are looked up with `lookup_sweep_feature`
    or `lookup_spike_feature`. Hence any feature can be computed at any point,
    without having to compute any dependencies first. Any dependencies already
    computed will be reused, unless `recompute=True` is passed.

    `SweepFeature`s can also implement a _plot method, that displays the diagnostic
    information or the feature itself. If the feature cannot be displayed in a V(t)
    or I(t) plot, instead the `plot` method should be overwritten directly. This
    is because `plot` wraps `_plot` adds additional functionality ot it.
    """

    def __init__(
        self,
        data: Optional[EphysSweep] = None,
        compute_at_init: bool = True,
        name: Optional[str] = None,
        store_with_data: bool = True,
    ):
        r"""
        Args:
            data: EphysSweep object.
                Can also be passed later with `__call__`.
            compute_at_init: If True, compute the feature at initialization.
                Otherwise the feature is only copmuted when `__call__` or
                `get_value` is called. This can be useful when instantiating
                many features at once, and waiting with the computation until
                the features are actually needed.
            name: Custom name of the feature. If None, the name of the feature
                class is used.
            store_with_data: If True, store the feature in the `self.data.features`
                dictionary.
        """
        super().__init__(data, compute_at_init, name, store_with_data)

    def _data_init(self, data: EphysSweep):
        """Initialize the feature with a EphysSweep object.

        This method is called at initialization and when the feature is
        called with a new EphysSweep object. It should
        be used to set the `self.data` attribute, and add the feature
        to the `self.data.features` dictionary.

        Args:
            data: EphysSweep object.
        """
        self.data = data
        if data is not None:
            assert isinstance(data, EphysSweep), "data must be EphysSweep"
            self.type = type(data).__name__
            self.ensure_correct_hyperparams()
            if not "features" in self.data.__dict__:
                self.data.features = {}
            if not self.name in self.data.features:
                self.data.features.update({self.name: self})
            else:
                features = self.data.features
                self._value = features[self.name]._value
                self._diagnostics = features[self.name]._diagnostics

    def lookup_sweep_feature(
        self, feature_name: str, recompute: bool = False, return_value: bool = True
    ) -> Union[float, SweepFeature]:
        """Look up a sweep level feature and return its value.

        This method will first check if the feature is already computed,
        and if not, instantiate and compute it. This works as long as the feature
        can be found via `fetch_available_fts`. Works recursively,
        so that features can depend on other features as long as they are
        looked up with `lookup_sweep_feature` or `lookup_spike_feature`.

        Args:
            feature_name: Name of the feature to look up.
            recompute: If True, recompute the feature even if it is already
                computed.
            return_value: If True, return the value of the feature. Otherwise
                return the feature object.

        Returns:
            The feature or the value of the feature depending on `return_value`.

        Raises:
            FeatureError: If the feature is not found via `fetch_available_fts`.
        """
        if feature_name not in self.data.features:
            available_fts = fetch_available_fts()
            available_fts = [ft for ft in available_fts if is_sweep_feature(ft)]
            available_fts = {
                ft.__name__.lower().replace("sweep_", ""): ft for ft in available_fts
            }
            if feature_name in available_fts:
                ft = available_fts[feature_name](
                    self.data, store_with_data=self.store_with_data
                )
                if return_value:
                    return ft.value
                return ft
            else:
                raise FeatureError(f"{feature_name} is not a known feature.")
        ft = self.data.features[feature_name]
        if return_value:
            return ft.get_value(recompute=recompute)
        return ft

    def lookup_spike_feature(
        self, feature_name: str, recompute: bool = False
    ) -> ndarray:
        """Look up a spike level feature and return its value.

        This method will first check if the feature is already computed,
        and if not, compute all spike level features using `process_spikes` from
        the underlying `EphysSweep` object, and then
        instantiate and compute the feature.

        Args:
            feature_name: Name of the feature to look up.
            recompute: If True, recompute the feature even if it is already
                computed.

        Returns:
            The value of the feature for each detected spike.
        """
        if not hasattr(self.data, "_spikes_df") or recompute:
            self.data.process_spikes()
        elif (
            feature_name in self.data.added_spike_features
            and feature_name not in self.data._spikes_df.columns
        ):
            self.data.process_spikes()
        return self.data.spike_feature(feature_name, include_clipped=True)

    def __repr__(self):
        return f"{self.name} for {self.data}"

    def __str__(self):
        if self._value is not None:
            return f"{self.name} = {self._value:.3f} {self.units}"
        else:
            return f"{self.name} = ? {self.units}"

    @abstractmethod
    def _compute(
        self, recompute: bool = False, store_diagnostics: bool = True
    ) -> float:
        """Compute the feature.

        All computation that is neccesary to yield the value of the feature should
        be defined here. This is the core method of SweepFeature and all other
        functionality interacts with this method.

        Alongside computing the value of the corresponding feature, this method
        can also be used to updat the `_diagnostics` attribute, which is a
        dictionary that can be used to store any additional information about
        the feature computation. This can be useful for debugging or better
        understanding how a feature was computed. Diagnostic information can
        be accessed with `get_diagnostics` or via the `diagnostics` property and
        updated with `_update_diagnostics`.

        Args:
            recompute: If True, recompute the feature even if it is already
                computed.
            store_diagnostics: If True, store any additional information about
                the feature computation in the `_diagnostics` attribute.

        Returns:
            The value of the feature.
        """
        # load dependencies using lookup_sweep_feature or lookup_spike_feature
        # do some computation
        # save diagnostics using _update_diagnostics
        return


class SweepSetFeature(SweepFeature):
    """Base class for sweepset level features that are computed from a
    `EphysSweepSet`. Wraps around any `SweepFeature` derived
    feature and extends it to the sweepset level.

    This class mostly acts like an `SweepFeature` and implements the same basic
    functionalities. See Documentation of `SweepFeature` for defails. Most
    importantly it also allows to recursively look up dependend features and
    compute them if necessary. This can be done on the spike, sweep and sweepset
    level. On the sweep level, instead of returning just a float however,
    `lookup_sweep_feature` will return a vector of feature values, where each
    entry corresponds to a sweep in the sweepset. Since all computation is
    done on the sweep level, all features are also stored with along with each
    sweep.

    All sweepset features should inherit from this class, and must implement the
    `_select` and `_aggregate` method. The `_select` method takes a vector of
    feature values and return a subset of these values based on a selection
    criterion (e.g. return all values that are larger than 0). The `_aggregate`
    method also takes a vector of feature values and aggregates them into a
    single value (e.g. return the mean of all values). Together the `_select`
    and `_aggregate` methods are able to compute representative values for every
    feature that can also be computed on the sweep level.

    In cases where the feature cannot directly be computed as an aggregate of the
    corresponding sweep feature, the `_compute` method can be overwritten. In
    this case the inheriting class should instantiate the `SweepSetFeature` super
    with `AbstractSweepFeature`. Similar to `SweepFeature`, the `_compute` method
    should then return the value of the feature.

    Other SweepSetFeatures can also be used in the computation of other features
    by using the `lookup_sweepset_feature` method.

    The description of the sweepset feature should contain a short description of
    the feature, and a list of dependencies. The dependencies should be listed
    as a comma separated list of feature names. It is parsed and can be displayed
    but has no functional use. Furthermore, the units of the feature should be
    specified. If the feature is unitless, the units should be set to "/".

    The docstring should have the following format:

    '''<Some Text>

    description: <Short description of the feature>.
    depends on: <Comma separated list of dependencies>.
    units: <Units of the feature>.

    <Some more text>'''

    All computed features are added to the underlying `EphysSweepSet`
    object, and can be accessed the `get_features()`.
    """

    def __init__(
        self,
        SwFt: SweepFeature,
        data: Optional[EphysSweepSet] = None,
        compute_at_init: bool = True,
        name: Optional[str] = None,
        store_with_data: bool = True,
    ):
        """Initialize the SweepSetFeature.

        parses the description, dependencies and units from the docstring of the
        feature and stores them as attributes. Also stores the name of the
        feature in the name attribute.

        Args:
            SwFt: The sweep level feature which is wrapped and aggregated
                with this class.
            data: The data to compute the feature for, i.e. an instance of
                SweepSetEphysExtractor.
            compute_at_init: If True, compute the feature at initialization.
            name: Custom name of the feature. If None, the name of the feature
                class is used.
            store_with_data: If True, store the feature in the `self.data.features`
                dictionary.
        """
        self.SwFt = SwFt
        swft = SwFt()
        ft_cls = swft.__class__

        self.name = swft.name if name is None else name
        self.baseft_name = swft.name if swft.name != "nullsweepfeature" else self.name
        self._value = None
        self._diagnostics = None

        self.__call__(
            data,
            compute=False,
            store_with_data=store_with_data,
            return_value=compute_at_init,
        )

        if ft_cls.__doc__ is not None:
            attrs = parse_func_doc_attrs(ft_cls)
            select_desc = parse_desc(self._select)
            agg_desc = parse_desc(self._aggregate)
            ft_desc = parse_desc(ft_cls)
            self.description = ft_desc + " " + select_desc + " " + agg_desc
            self.depends_on = parse_deps(attrs["depends on"])
            self.units = attrs["units"]

    @property
    def dataset(self):
        """Proxy for self.data at the sweepset level."""
        return np.array([ft for ft in self])

    def _data_init_incl_storage(self, data: EphysSweepSet):
        self._data_init(data)
        if not self.store_with_data and not self.data is None:
            self.data.features.pop(self.name)

    def _data_init(self, data: EphysSweepSet):
        """Initialize the feature with a EphysSweepSet object.

        This method is called at initialization and when the feature is
        called with a new EphysSweepSet object. It should
        be used to set the `self.data` attribute, and add the feature
        to the `self.data.features` dictionary.

        Args:
            data: EphysSweepSet object.
        """
        self.data = data
        if data is not None:
            assert isinstance(
                data, EphysSweepSet
            ), "data must be a EphysSweepSet object"
            self.type = type(data).__name__
            self.ensure_correct_hyperparams()
            for sw in self.data:
                if not self.baseft_name in sw.features:
                    ft = self.SwFt(sw, compute_at_init=False)
                else:
                    ft = sw.features[self.baseft_name]
                if not "features" in ft.data.__dict__:
                    ft.data.features = {}
                ft.data.features.update({self.baseft_name: ft})
            if not "features" in self.data.__dict__:
                self.data.features = {}
            if not self.name in self.data.features:
                self.data.features.update({self.name: self})
            else:
                features = self.data.features
                self._value = features[self.name]._value
                self._diagnostics = features[self.name]._diagnostics

    def __repr__(self):
        return f"{self.name} for {self.data}"

    def __str__(self):
        if self._value is not None:
            return f"{self.name} = {self._value:.3f} {self.units}"
        else:
            return f"{self.name} = ? {self.units}"

    def __getitem__(self, idx):
        return self.data[idx].features[self.baseft_name]

    def __getattr__(self, name: str):
        """Hands off all functionality to the sweep level feature objects and
        collects the results in a vector.

        If a function exists on the sweep level that does not exist on the
        sweepset level, this function will be called on all sweep level
        features and the results will be returned as a vector. If the function
        exists on the sweepset level, it will be called on the sweepset level
        feature object and the result will be returned.

        Args:
            name: Name of the attribute to get.
        """
        attr = lambda sw: getattr(sw, name)

        def attr_broadcast(*args, **kwargs):
            """Takes a function and broadcasts it over all sweeps in the sweepset."""
            return np.array([attr(sw)(*args, **kwargs) for sw in self])

        if name not in self.__dict__:
            if isinstance(getattr(self[0], name), Callable):
                return attr_broadcast
            else:
                return np.array([attr(sw) for sw in self])
        else:
            return getattr(self, name)

    def lookup_sweep_feature(
        self, feature_name: str, recompute: bool = False, return_value: bool = True
    ) -> ndarray:
        """Lookup feature for each sweep and return the results as a vector.

        Args:
            feature_name: Name of the feature to lookup.
            recompute: If True, recompute the feature even if it is already
                has been computed previously.
            return_value: If True, return the value of the feature, otherwise
                return the feature object.

        Returns:
            Vector of feature values or feature objects.
        """
        available_fts = fetch_available_fts()
        available_fts = [ft for ft in available_fts if is_sweep_feature(ft)]
        available_fts = {
            ft.__name__.lower().replace("sweep_", ""): ft for ft in available_fts
        }
        if feature_name in available_fts:
            return np.array(
                [
                    sweep.lookup_sweep_feature(
                        feature_name,
                        recompute=recompute,
                        return_value=return_value,
                    )
                    for sweep in self
                ]
            )
        else:
            raise FeatureError(
                f"{feature_name} is not a known feature. If it is a custom feature, make sure it was registered with `register_custom_feature`."
            )

    def lookup_sweepset_feature(
        self, feature_name: str, recompute: bool = False, return_value: bool = True
    ) -> Union[float, SweepSetFeature]:
        """Lookup feature for the sweepset and return the result.

        Analogous to `lookup_sweep_feature`, on the sweep level, but for sweepset
        level features.

        Args:
            feature_name: Name of the feature to lookup.
            recompute: If True, recompute the feature even if it is already
                has been computed previously.
            return_value: If True, return the value of the feature, otherwise
                return the feature object.

        Returns:
            Feature value."""
        if feature_name not in self.data.features:
            available_fts = fetch_available_fts()
            available_fts = [ft for ft in available_fts if is_sweepset_feature(ft)]
            available_fts = {
                ft.__name__.lower().replace("sweepset_", ""): ft for ft in available_fts
            }
            if feature_name in available_fts:
                ft = available_fts[feature_name](
                    self.data, store_with_data=self.store_with_data
                )
                if return_value:
                    return ft.value
                return ft
            else:
                raise FeatureError(f"{feature_name} is not a known feature.")
        ft = self.get_features()[feature_name]
        if return_value:
            return ft.get_value(recompute=recompute)
        return ft

    # @abstractmethod
    def _select(self, fts: ndarray) -> ndarray:
        """Select a subset of the feature values.

        This method is called by `self._compute` and should be used to select
        a subset of the feature values. This method implements a selection
        criterion that is specific to the feature. For example, a hypothetical
        `SpikeAmplitude` feature could select the 3 highest spike amplitudes from
        the feature values.

        Args:
            fts: Vector of feature values.

        Returns:
            Vector with a selected subset of feature values.
        """
        make_selection = lambda fts: fts
        self._update_diagnostics({})
        return make_selection(fts)

    # @abstractmethod
    def _aggregate(self, fts: ndarray) -> float:
        """Aggregate the feature values.

        This method is called by `self._compute` and should be used to aggregate
        the subset of feature values that were selected by `self._select`. This
        method implements an aggregation criterion that is specific to the
        feature. For example, a hypothetical `SpikeAmplitude` feature could
        just take the maximum spike amplitude from the feature values.

        Args:
            fts: Vector of feature values.

        Returns:
            Aggregated feature value."""
        aggregate = lambda fts: fts
        self._update_diagnostics({})
        return aggregate(fts).item()

    def _compute(
        self, recompute: bool = False, store_diagnostics: bool = False
    ) -> float:
        """Copmutes representative feature value by aggregating over a selected
        subset of sweep level feature values.

        This method chains together `self.lookup_sweep_feature(self.baseft_name)`,
        `self._select` and `self._aggregate` to yield a representative feature
        value for the entire sweepset.

        Args:
            recompute: If True, recompute the feature even if it is already
                has been computed previously.
            store_diagnostics: If True, store diagnostic information about the
                feature computation in the `diagnostics` attribute of the
                feature object.

        Returns:
            Feature value.
        """
        fts = self.lookup_sweep_feature(self.baseft_name, recompute=recompute)

        subset = self._select(fts)
        if subset.size > 0:
            ft = self._aggregate(subset)
        else:
            ft = float("nan")
        if store_diagnostics:
            self._update_diagnostics({"values": fts})
        return ft

    @property
    def features(self):
        """List values for all computed features."""
        return {k: ft.value for k, ft in self.get_features().items()}

    def get_features(self):
        """List all computed features."""
        return {k: ft for k, ft in self.data.features.items()}

    def _plot(self, *args, ax: Optional[Axes] = None, **kwargs) -> Axes:
        """Plot the feature.

        Similar to _compute, _aggregate or _select, this method implements
        a core functionality of SweepSetFeature. It is not an abstract feature
        though. It is called by `plot` and can be used to visualize the feature
        in any shape of form. If the feature cannot be plotted on top of the
        underlying sweep, `plot` should be overwritten directly.

        Args:
            *args: Additional arguments to pass.
            ax (Optional[Axes], optional): Axes to plot on.
            kwargs: Additional kwargs to pass to `self.plot`.

        Returns:
            Axes: Axes of plot.
        """
        idxs = unpack(self.diagnostics, "selected_idx")
        idxs = idxs if isinstance(idxs, Iterable) else [idxs]

        for idx in idxs:
            data = self.data[idx]
            data.plot(ax=ax, **kwargs)
            ax = data.features[self.baseft_name].plot(ax=ax, **kwargs)
        return ax
