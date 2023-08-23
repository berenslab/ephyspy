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

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

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


def unpack(dict, keys):
    """Unpack dict to tuple of values."""
    if isinstance(keys, str):
        return dict[keys]
    return tuple(dict[k] for k in keys)


def featureplot(func):
    """Decorator to make ax optional in plot functions."""

    def wrapper(self, *args, ax=None, show_sweep=False, show_stimulus=False, **kwargs):
        is_stim_ft = self.name in ["stim_amp", "stim_onset", "stim_end"]
        if show_sweep:
            show_stimulus = is_stim_ft or show_stimulus
            axes = self.data.plot(color="k", show_stimulus=show_stimulus, **kwargs)
        else:
            axes = plt.gca() if ax is None else ax
            if show_stimulus:
                axes.plot(self.data.t, self.data.i, color="k")
                axes.set_ylabel("Current (pA)")

        if np.isnan(self.value):
            return axes

        if isinstance(axes, np.ndarray):
            ax = axes[1] if is_stim_ft else axes[0]
        else:
            ax = axes

        ax = func(self, *args, ax=ax, **kwargs)

        if not ax.get_xlabel():
            ax.set_xlabel("Time (s)")
        if not ax.get_ylabel():
            ax.set_ylabel("Voltage (mV)")
        ax.legend()
        return axes

    return wrapper
