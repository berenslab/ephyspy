from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from py_ephys.utils import (
    FeatureError,
    fetch_available_fts,
    parse_deps,
    parse_func_doc_attrs,
)


class EphysFeature(ABC):
    def __init__(self, data=None, compute_at_init=True):
        self.name = self.__class__.__name__.lower()
        self._value = None
        self._diagnostics = None
        self._data_init(data)
        if not data is None and compute_at_init:
            self.get_value()

        if self.__class__.__doc__ is not None:
            attrs = parse_func_doc_attrs(self.__class__)
            self.description = attrs["description"]
            self.depends_on = attrs["depends on"]
            self.units = attrs["units"]

    def _data_init(self, data):
        self.data = data
        if data is not None:
            self.type = type(data).__name__
            if not "features" in self.data.__dict__:
                self.data.features = {}
            self.data.features.update({self.name: self})

    def lookup_sweep_feature(self, feature_name, recompute=False):
        if feature_name not in self.data.features:
            available_fts = fetch_available_fts()
            available_fts = {ft.__name__.lower(): ft for ft in available_fts}
            if feature_name in available_fts:
                return available_fts[feature_name](self.data).value
            else:
                raise FeatureError(f"{feature_name} is not a known feature.")
        return self.data.features[feature_name].get_value(recompute=recompute)

    def lookup_spike_feature(self, feature_name, recompute=False):
        if not hasattr(self.data, "_spikes_df") or recompute:
            self.data.process_spikes()
        return self.data.spike_feature(feature_name, include_clipped=True)

    def __repr__(self):
        return f"{self.name} for {self.data}"

    def __str__(self):
        val = self._value if self._value is not None else "?"
        return f"{self.name} = {val} {self.units}"

    @abstractmethod
    def _compute(self, recompute=False, store_diagnostics=True):
        # load dependencies(recompute=recompute)
        # do some computation
        # save diagnostics
        return

    def recompute(self):
        return self.get_value(recompute=True, store_diagnostics=True)

    def get_diagnostics(self, recompute=False):
        if recompute or self._diagnostics is None:
            self.get_value(recompute=recompute, store_diagnostics=True)
        return self._diagnostics

    @property
    def diagnostics(self):
        return self.get_diagnostics()

    def _update_diagnostics(self, dct):
        if self._diagnostics is None:
            self._diagnostics = {}
        self._diagnostics.update(dct)

    def get_value(self, recompute=False, store_diagnostics=True):
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

    def __call__(self, data, recompute=False, store_diagnostics=True):
        self._data_init(data)
        return self.get_value(
            recompute=recompute,
            store_diagnostics=store_diagnostics,
        )

    def show(self):
        return


class SweepSetFeature(EphysFeature):
    def __init__(self, feature, data=None, compute_at_init=True):
        self.feature = feature
        ft_cls = feature().__class__

        self.name = ft_cls.__name__.lower()
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
        return np.array([self.feature(sw) for sw in self.data.sweeps()])

    def _data_init(self, data):
        self.data = data
        if data is not None:
            self.type = type(data).__name__
            for ft in self.dataset:
                if not "features" in ft.data.__dict__:
                    ft.data.features = {}
                ft.data.features.update({self.name: ft})
            if not "features" in self.data.__dict__:
                self.data.features = {}
            self.data.features.update({self.name: self})

    def __repr__(self):
        return f"{self.name} for {self.data}"

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __getattr__(self, name):
        attr = lambda sw: getattr(sw, name)

        def attr_broadcast(*args, **kwargs):
            return np.array([attr(sw)(*args, **kwargs) for sw in self])

        if name not in self.__dict__:
            if isinstance(getattr(self[0], name), Callable):
                return attr_broadcast
            else:
                return np.array([attr(sw) for sw in self])
        else:
            return getattr(self, name)

    def lookup_sweep_feature(self, feature_name, recompute=False):
        if feature_name not in self.data.features:
            available_fts = fetch_available_fts()
            available_fts = {ft.__name__.lower(): ft for ft in available_fts}
            if feature_name in available_fts:
                return np.array(
                    [
                        sw.lookup_sweep_feature(
                            feature_name,
                            recompute=recompute,
                        )
                        for sw in self
                    ]
                )
            else:
                raise FeatureError(f"{feature_name} is not a known feature")
        return np.array(
            [
                ft.get_value(recompute=recompute)
                for ft in self.get_features()[feature_name]
            ]
        )

    @abstractmethod
    def _select(self, fts):
        make_selection = lambda fts: fts
        self._update_diagnostics({})
        return make_selection(fts)

    @abstractmethod
    def _aggregate(self, fts):
        aggregate = np.nanmean
        self._update_diagnostics({})
        return aggregate(fts)

    @abstractmethod
    def _compute(self, recompute=False, store_diagnostics=False):
        fts = self.lookup_sweep_feature(self.name, recompute=recompute)
        # ftname = self.lookup_sweep_feature("ftname", recompute=recompute)

        subset = self._select(fts)
        ft = self._aggregate(subset)
        self._update_diagnostics({})
        return ft

    @property
    def features(self):
        return {k: ft.value for k, ft in self.get_features().items()}

    def get_features(self):
        return {k: ft for k, ft in self.data.features.items()}
