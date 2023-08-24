import numpy as np
import pytest
import matplotlib.pyplot as plt

from ephyspy.features import available_spike_features
from ephyspy.sweeps import EphysSweep, EphysSweepSet
from functools import wraps

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
test_sweepset.add_features(available_spike_features())

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
