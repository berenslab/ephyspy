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
from typing import Callable, Optional, Dict, Any, Union

import numpy as np
from numpy import ndarray

from ephyspy.features.utils import fetch_available_fts
from ephyspy.features.utils import (
    FeatureError,
    parse_deps,
    parse_func_doc_attrs,
    is_sweep_feature,
    is_sweepset_feature,
)
from ephyspy.sweeps import EphysSweep, EphysSweepSet


class EphysFeature(ABC):
    r"""Base class for all sweep level electrophysiological features.

    This class defines the interface for all electrophysiological features.
    All features should inherit from this class, and must implement the
    `_compute` method. The `_compute` method should return the feature value
    and optionally save diagnostic information for later debugging to
    `self._diagnostics`.

    The description of the feature should contain a short description of the
    feature, and a list of dependencies. The dependencies should be listed
    as a comma separated list of feature names. It is parsed and can be displayed
    but has no functional use. Furthermore, the units of the feature should be
    specified. If the feature is unitless, the units should be set to "/".

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
    """

    # TODO: Add show method!
    def __init__(
        self,
        data: Optional[EphysSweep] = None,
        compute_at_init: bool = True,
        name: Optional[str] = None,
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
        """
        self.name = self.__class__.__name__.lower() if name is None else name
        self._value = None
        self._diagnostics = None
        self._data_init(data)
        if not data is None and compute_at_init:
            self.get_value()

        if self.__class__.__doc__ is not None:
            attrs = parse_func_doc_attrs(self.__class__)
            self.description = (
                None if not "description" in attrs else attrs["description"]
            )
            self.depends_on = None if not "depends on" in attrs else attrs["depends on"]
            self.units = None if not "units" in attrs else attrs["units"]
            self.units = "" if self.units == "/" else self.units

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
            if not "features" in self.data.__dict__:
                self.data.features = {}
            if not self.name in self.data.features:
                self.data.features.update({self.name: self})
            else:
                features = self.data.features
                self._value = features[self.name]._value
                self._diagnostics = features[self.name]._diagnostics

    def lookup_sweep_feature(self, feature_name: str, recompute: bool = False) -> float:
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

        Returns:
            The value of the feature.

        Raises:
            FeatureError: If the feature is not found via `fetch_available_fts`.
        """
        if feature_name not in self.data.features:
            available_fts = fetch_available_fts()
            available_fts = [ft for ft in available_fts if is_sweep_feature(ft)]
            available_fts = {ft.__name__.lower(): ft for ft in available_fts}
            if feature_name in available_fts:
                return available_fts[feature_name](self.data).value
            else:
                raise FeatureError(f"{feature_name} is not a known feature.")
        return self.data.features[feature_name].get_value(recompute=recompute)

    def lookup_spike_feature(self, feature_name: str, recompute: bool = False) -> float:
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
            The value of the feature.
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
        be defined here. This is the core method of EphysFeature and all other
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
        # load dependencies(recompute=recompute)
        # do some computation
        # save diagnostics
        return

    def recompute(self) -> float:
        """Convencience method to recompute the feature.

        This method is equivalent to calling `get_value` with `recompute=True`
        and `store_diagnostics=True`.

        Returns:
            The value of the feature."""
        return self.get_value(recompute=True, store_diagnostics=True)

    def get_diagnostics(self, recompute=False):
        if recompute or self._diagnostics is None:
            self.get_value(recompute=recompute, store_diagnostics=True)
        return self._diagnostics

    @property
    def diagnostics(self):
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
        if recompute or self._value is None:
            self._value = self._compute(
                recompute=recompute,
                store_diagnostics=store_diagnostics,
            )
        return self._value

    @property
    def value(self):
        return self.get_value()

    @value.setter
    def _set_value(self, value):
        self._value = value

    def __call__(
        self,
        data: EphysSweep = None,
        compute: bool = False,
        store_diagnostics: bool = True,
        return_value: bool = False,
    ) -> Union[float, EphysFeature]:
        """Compute the feature for a given dataset.

        Essentially chains together `_data_init` and `get_value`.

        Args:
            data: The dataset to compute the feature for, i.e. an instance of
                `EphysSweep`.
            compute: If True, compute the feature even if it is already
                computed.
            store_diagnostics: If True, store any additional information about
                the feature computation in the `_diagnostics` attribute.
            return_value: If True, return the value of the feature, otherwise
                return the feature object.

        Returns:
            The value of the feature.
        """
        self._data_init(data)
        if compute:
            self.get_value(
                recompute=True,
                store_diagnostics=store_diagnostics,
            )
        if return_value:
            return self._value
        return self

    def show(self):
        return


class AbstractEphysFeature(EphysFeature):
    """Abstract sweep level feature.

    Dummy feature that can be used as a placeholder to compute sweepset level
    features using `SweepsetFeature` if no sweep level feature for it is available.

    depends on: /.
    description: Only the corresponding sweepset level feature exsits.
    units: /."""

    def __init__(self, data=None, compute_at_init=True, name=None):
        super().__init__(data, compute_at_init, name=name)

    def _compute(self, recompute=False, store_diagnostics=True):
        return


class SweepsetFeature(EphysFeature):
    """Base class for sweepset level features that are computed from a
    `EphysSweepSet`. Wraps around any `EphysFeature` derived
    feature and extends it to the sweepset level.

    This class mostly acts like an `EphysFeature` and implements the same basic
    functionalities. See Documentation of `EphysFeature` for defails. Most
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
    this case the inheriting class should instantiate the `SweepsetFeature` super
    with `AbstractEphysFeature`. Similar to `EphysFeature`, the `_compute` method
    should then return the value of the feature.

    Other SweepsetFeatures can also be used in the computation of other features
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
        feature: EphysFeature,
        data: Optional[EphysSweepSet] = None,
        compute_at_init: bool = True,
        name: Optional[str] = None,
    ):
        """Initialize the SweepsetFeature.

        parses the description, dependencies and units from the docstring of the
        feature and stores them as attributes. Also stores the name of the
        feature in the name attribute.

        Args:
            feature: The sweep level feature which is wrapped and aggregated
                with this class.
            data: The data to compute the feature for, i.e. an instance of
                SweepSetEphysExtractor.
            compute_at_init: If True, compute the feature at initialization.
            name: Custom name of the feature. If None, the name of the feature
                class is used.
        """
        self.feature = feature
        ft_cls = feature().__class__

        self.name = ft_cls.__name__.lower() if name is None else name
        self._value = None
        self._diagnostics = None
        self._data_init(data)
        if not data is None and compute_at_init:
            self.get_value()

        if ft_cls.__doc__ is not None:
            attrs = parse_func_doc_attrs(ft_cls)
            self.description = attrs["description"]
            self.depends_on = parse_deps(attrs["depends on"])
            self.units = attrs["units"]

    @property
    def dataset(self):
        """Proxy for self.data at the sweepset level."""
        return np.array([self.feature(sw) for sw in self.data.sweeps()])

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
            for ft in self.dataset:
                if not "features" in ft.data.__dict__:
                    ft.data.features = {}
                ft.data.features.update({self.name: ft})
            if not "features" in self.data.__dict__:
                self.data.features = {}
            if not self.name in self.data.features:
                self.data.features.update({self.name: self})
            else:
                features = self.data.features
                self._value = features[self.name]._value
                self._diagnostics = features[self.name]._diagnostics

    def __call__(
        self,
        data: EphysSweepSet = None,
        compute: bool = False,
        store_diagnostics: bool = True,
        return_value: bool = False,
    ) -> Union[SweepsetFeature, float]:
        """Compute the feature for a given dataset.

        Essentially chains together `_data_init` and `get_value`.

        Args:
            data: The dataset to compute the feature for, i.e. an instance of
                `EphysSweepSet`.
            compute: If True, compute the feature even if it is already
                computed.
            store_diagnostics: If True, store any additional information about
                the feature computation in the `_diagnostics` attribute.
            return_value: If True, return the value of the feature, otherwise
                return the feature object.

        Returns:
            The value of the feature.
        """
        self._data_init(data)
        if compute:
            self.get_value(
                recompute=True,
                store_diagnostics=store_diagnostics,
            )
        if return_value:
            return self._value
        return self

    def __repr__(self):
        return f"{self.name} for {self.data}"

    def __str__(self):
        if self._value is not None:
            return f"{self.name} = {self._value:.3f} {self.units}"
        else:
            return f"{self.name} = ? {self.units}"

    def __getitem__(self, idx):
        return self.dataset[idx]

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
        self, feature_name: str, recompute: bool = False
    ) -> ndarray:
        """Lookup feature for each sweep and return the results as a vector.

        Args:
            feature_name: Name of the feature to lookup.
            recompute: If True, recompute the feature even if it is already
                has been computed previously.

        Returns:
            Vector of feature values.
        """
        available_fts = fetch_available_fts()
        available_fts = [ft for ft in available_fts if is_sweep_feature(ft)]
        available_fts = {ft.__name__.lower(): ft for ft in available_fts}
        if feature_name in available_fts:
            return np.array(
                [
                    sweep.lookup_sweep_feature(
                        feature_name,
                        recompute=recompute,
                    )
                    for sweep in self
                ]
            )
        else:
            raise FeatureError(f"{feature_name} is not a known feature")

    def lookup_sweepset_feature(
        self, feature_name: str, recompute: bool = False
    ) -> float:
        """Lookup feature for the sweepset and return the result.

        Analogous to `lookup_sweep_feature`, on the sweep level, but for sweepset
        level features.

        Args:
            feature_name: Name of the feature to lookup.
            recompute: If True, recompute the feature even if it is already
                has been computed previously.

        Returns:
            Feature value."""
        if feature_name not in self.data.features:
            available_fts = fetch_available_fts()
            available_fts = [ft for ft in available_fts if is_sweepset_feature(ft)]
            available_fts = {ft.__name__.lower(): ft for ft in available_fts}
            if feature_name in available_fts:
                return available_fts[feature_name](self.data).value
            else:
                raise FeatureError(f"{feature_name} is not a known feature.")
        return self.get_features()[feature_name].get_value(recompute=recompute)

    @abstractmethod
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
        return float(make_selection(fts))

    @abstractmethod
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
        aggregate = np.nanmean
        self._update_diagnostics({})
        return float(aggregate(fts))

    def _compute(
        self, recompute: bool = False, store_diagnostics: bool = False
    ) -> float:
        """Copmutes representative feature value by aggregating over a selected
        subset of sweep level feature values.

        This method chains together `self.lookup_sweep_feature(self.name)`,
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
        fts = self.lookup_sweep_feature(self.name, recompute=recompute)

        subset = self._select(fts)
        ft = self._aggregate(subset)
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
