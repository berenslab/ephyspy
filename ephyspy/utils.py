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

import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy import ndarray

if TYPE_CHECKING:
    from ephyspy.sweeps import EphysSweep, EphysSweepSet


where_between = lambda t, t0, tend: np.logical_and(t > t0, t < tend)


def fwhm(
    t: ndarray, v: ndarray, t_start: float, t_end: float
) -> Tuple[float, float, float]:
    """Get full width at half maximum of a ap.

    Args:
        t (ndarray): time array.
        v (ndarray): voltage array.
        t_start (float): start time of ap.
        t_end (float): end time of ap.

    Returns:
        Tuple[float, float, float]: full width at half maximum,
            time of half maximum upstroke, time of half maximum downstroke.
    """
    in_T = where_between(t, t_start, t_end)
    v_peak = np.max(v[in_T])
    v_start = v[in_T][0]
    t_peak = t[in_T][np.argmax(v[in_T])]
    upstroke = where_between(t, t_start, t_peak)
    downstroke = where_between(t, t_peak, t_end)
    fwhm = v_start + (v_peak - v_start) / 2
    hm_up_idx = np.argmin(np.abs(v[upstroke] - fwhm))
    hm_down_idx = np.argmin(np.abs(v[downstroke] - fwhm))
    hm_up_t = t[upstroke][hm_up_idx]
    hm_down_t = t[downstroke][hm_down_idx]
    return fwhm, hm_up_t, hm_down_t


def unpack(
    dict: Dict, keys: Union[str, Tuple[str, ...]]
) -> Union[Any, Tuple[Any, ...]]:
    """Unpack dict to tuple of values."""
    if isinstance(keys, str):
        return dict[keys]
    return tuple(dict[k] for k in keys)


def relabel_line(ax: Axes, old_label: str, new_label: str):
    """Rename line label in one given axes."""
    for child in ax._children:
        if old_label in child.get_label():
            child.set_label(new_label)


def is_spike_feature(ft: Any) -> bool:
    return not hasattr(ft, "__base__") and isinstance(ft, Callable)


def is_sweep_feature(ft: Any) -> bool:
    def has_sweep_base(ft) -> bool:
        try:
            if "SweepSetFeature" in ft.__base__.__name__:
                return False
            elif "SweepFeature" in ft.__base__.__name__:
                return True
            else:
                return has_sweep_base(ft.__base__)
        except AttributeError:
            return False

    return has_sweep_base(ft)


def is_sweepset_feature(ft: Any) -> bool:
    def has_sweepset_base(ft) -> bool:
        try:
            if "SweepSetFeature" in ft.__base__.__name__:
                return True
            else:
                return has_sweepset_base(ft.__base__)
        except AttributeError:
            return False

    return has_sweepset_base(ft)


def has_spike_feature(sweep: EphysSweep, ft: str) -> bool:
    """Checks if sweep has a given spike feature.

    First checks for `_spikes_df` attribute, which should get instantiated if
    spikes have already been processed. If not present `process_spikes` will be
    called. Then, if the feature is present in the `_spikes_df` and not all values
    are NaN, returns True.

    Args:
        sweep (EphysSweep): Sweep to check for existance of spike feature.
        ft (str): Spike feature to check for. Feature must be present in `_spikes_df`
            for a healthy spiking sweep.

    Returns:
        bool: Whether sweep has the given spike feature."""
    if not hasattr(sweep, "_spikes_df"):
        sweep.process_spikes()
    ap_fts = sweep._spikes_df
    if ap_fts.size:
        if ft in ap_fts.columns:
            if not np.all(np.isnan(ap_fts[ft])):
                return True
    return False


def spikefeatureplot(func: Callable) -> Callable:
    """Decorator for plotting spike features.

    Args:
        func (Callable): Function to decorate.

    Returns:
        Callable: Decorated function."""

    def _spikefeatureplot(
        sweep: EphysSweep,
        *args,
        ax: Axes = None,
        show_sweep: bool = False,
        show_stimulus: bool = False,
        **kwargs,
    ):
        """Adds additional kwargs and functionality to functions that plot spike features.

        Checks if the sweep has spikes. Additionally along with every feature,
        the sweep and stimulus can be plotted.

        If no axis is provided one is created.

        Args:
            self (SweepFeature): Feature to plot. Needs to have a `plot` method.
            *args: Additional arguments to pass to `self.plot`.
            ax (Optional[Axes], optional): Axes to plot on.
            show_sweep (bool, optional): Whether to plot the sweep. Defaults to False.
            show_stimulus (bool, optional): Whether to plot the stimulus. Defaults to False.
            kwargs: Additional kwargs to pass to `self.plot`.

        Returns:
            Axes: Axes of plot.
        """
        if show_sweep:
            axes = sweep.plot(color="k", show_stimulus=show_stimulus, **kwargs)
        else:
            axes = plt.gca() if ax is None else ax

        ax = axes[0] if isinstance(axes, np.ndarray) else axes
        if has_spike_feature(sweep, "threshold_v"):  # has thresh_v -> presence of APs
            ax = func(sweep, *args, ax=ax, **kwargs)

            if not ax.get_xlabel():
                ax.set_xlabel("Time (s)")
            if not ax.get_ylabel():
                ax.set_ylabel("Voltage (mV)")
            ax.legend()
        return axes

    return _spikefeatureplot


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
            match = regex.search(repr(func_doc))
            if match:
                match = match.group(1)
                match = " ".join(match.split())  # rm whitespaces > 1
                match = match.split("\\n\\n")[0]  # slice at double line break
                match = match.replace("\\n", "")
                doc_attrs[attr] = match

    for attr_r in attrs[::-1]:  # traverse attr descriptions in reverse
        for attr_f in attrs:  # rm attr descriptions from other attr descriptions
            doc_attrs[attr_f] = doc_attrs[attr_f].split(f"{attr_r}:")[0].strip()
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


def get_feature(name: str, data: Union[EphysSweep, EphysSweepSet], **kwargs):
    """Get feature by name.

    This is a convenience function to compute features without having to import
    the feature classes or think about wether a feature is computed on a sweep or
    sweepset.

    Args:
        name (str): Name of feature.
        data (EphysSweep or EphysSweepSet): Data to compute feature on. This can be
            either a single sweep or a sweepset.

    Raises:
        FeatureError: If feature is not available for data type.

    Returns:
        Feature: Feature object.
    """
    # imports are done here to avoid circular imports
    from ephyspy.sweeps import EphysSweep, EphysSweepSet
    from ephyspy.features.utils import FeatureError
    from ephyspy.features.sweep_features import available_sweep_features
    from ephyspy.features.sweepset_features import available_sweepset_features

    if isinstance(data, EphysSweep):
        return available_sweep_features()[name](data, **kwargs)
    elif isinstance(data, EphysSweepSet):
        return available_sweepset_features()[name](data, **kwargs)
    else:
        raise FeatureError(
            f"Feature {name} is not available for data of type {type(data)}."
        )
