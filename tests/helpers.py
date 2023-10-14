from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ephyspy.features import available_spike_features
from ephyspy.features.base import SweepFeature, SweepSetFeature
from ephyspy.sweeps import EphysSweep, EphysSweepSet

# load test data
test_data = np.load("tests/test_sweepset.npz", allow_pickle=True)
t_set, u_set, i_set = test_data["ts"], test_data["Uts"], test_data["Its"]
t_set *= 1e-3  # convert to seconds
start, end = t_set[0, 0], t_set[0, -1]

# create sweepset
test_sweepset = EphysSweepSet(
    t_set,
    u_set,
    i_set,
    filter=5,
    metadata={"dc_offset": -14.52083},
)
# test_sweepset.add_features(available_spike_features())

# create test sweeps
depol_test_sweep = EphysSweep(t_set[11], u_set[11], i_set[11], start, end, filter=1)
# depol_test_sweep.process_spikes()

hyperpol_test_sweep = EphysSweep(t_set[0], u_set[0], i_set[0], start, end, filter=1)
# hyperpol_test_sweep.process_spikes()

# create custom dummy feature for custom import test


def close_fig_b4_raising(test_func):
    """Ensure the figues are closed if test fails.
    Does not work with pytes without functools.wraps for some reason."""

    @wraps(test_func)
    def wrapped_test(*args, **kwargs):
        try:
            ax = test_func(*args, **kwargs)
            plt.close()
            return ax
        except Exception as e:
            plt.close()
            raise e

    return wrapped_test


class SweepTestDependency(SweepFeature):
    """Extract sweep level V(t_thresh0) feature.

    depends on: /.
    description: V(t=t_thresh0).
    units: mV."""

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        test_value = float("nan")
        num_ap = self.lookup_sweep_feature("num_ap")  # existing feature

        if num_ap > 0:
            t_thresh = self.lookup_spike_feature("threshold_t")[0]
            t_thresh = np.argmin(abs(self.data.t - t_thresh))
            test_value = self.data.v[t_thresh]
        return test_value


class SweepTestFeature(SweepFeature):
    """Extract sweep level V(t_thresh0) feature.

    depends on: /.
    description: V(t=t_thresh0).
    units: mV."""

    def __init__(self, data=None, **kwargs):
        super().__init__(data, **kwargs)

    def _compute(self, recompute=False, store_diagnostics=True):
        num_ap = self.lookup_sweep_feature("num_ap")  # existing feature
        v_thresh0 = self.lookup_sweep_feature("sweeptestdependency")  # custom feature
        return v_thresh0


class SweepSetTestFeature(SweepSetFeature):
    """Extract sweep set level V(t_thresh0) feature.

    depends on: SweepTestFeature.
    description: V(t=t_thresh0).
    units: mV.
    """

    def __init__(self, data=None, **kwargs):
        super().__init__(SweepTestFeature, data=data, **kwargs)

    def _select(self, fts):
        """Select representative sweep and use its features to represent the
        entire sweepset.

        description: /.
        """
        return fts

    def _aggregate(self, fts):
        """Compute aggregate metrics on subset of sweeps.

        description: take the mean.
        """
        return np.nanmean(fts).item()
