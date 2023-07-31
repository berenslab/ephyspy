import numpy as np
import pytest

from py_ephys.allen_sdk.ephys_extractor import EphysSweepFeatureExtractor
from py_ephys.features import (
    get_available_spike_features,
    get_available_sweep_features,
    get_available_sweepset_features,
    get_sweep_stim_amp,
)
from py_ephys.utils import EphysSweepSetFeatureExtractor

# load test data
test_data = np.load("tests/test_sweepset.npz", allow_pickle=True)
t_set, u_set, i_set = test_data["ts"], test_data["Uts"], test_data["Its"]
t_set /= 1000  # convert to seconds
start, end = t_set[0, 0], t_set[0, -1]


@pytest.mark.skip
def prepare_test_sweep(
    test_sweep,
    include_new_spike_fts=True,
    include_new_sweep_fts=False,
    return_ft_info=False,
    preprocess=True,
):
    test_sweep.set_stimulus_amplitude_calculator(get_sweep_stim_amp)
    if include_new_spike_fts:
        for spike_ft, spike_ft_func in get_available_spike_features().items():
            test_sweep.process_new_spike_feature(spike_ft, spike_ft_func)
    if include_new_sweep_fts:
        for sweep_ft, sweep_ft_func in get_available_sweep_features(
            return_ft_info=return_ft_info
        ).items():
            test_sweep.process_new_sweep_feature(sweep_ft, sweep_ft_func)
    if preprocess:
        test_sweep.process_spikes()
    return test_sweep


@pytest.mark.skip
def prepare_test_sweepset(
    return_ft_info,
    add_spike_fts=True,
    add_sweep_fts=True,
    add_sweepset_fts=True,
    preprocess=True,
):
    test_sweepset = EphysSweepSetFeatureExtractor(
        t_set, u_set, i_set, filter=1, dc_offset=-14.52083
    )
    test_sweepset.set_stimulus_amplitude_calculator(get_sweep_stim_amp)

    if add_spike_fts:
        for ft, ft_func in get_available_spike_features().items():
            test_sweepset.add_spike_feature(ft, ft_func)
    if add_sweep_fts:
        for ft, ft_func in get_available_sweep_features(return_ft_info).items():
            test_sweepset.add_sweep_feature(ft, ft_func)
    if add_sweepset_fts:
        for ft, ft_func in get_available_sweepset_features(return_ft_info).items():
            test_sweepset.add_sweepset_feature(ft, ft_func)
    if preprocess:
        test_sweepset.process()
    return test_sweepset


# create test sweeps
depol_test_sweep = EphysSweepFeatureExtractor(
    t_set[11], u_set[11], i_set[11], start, end, filter=1
)
depol_test_sweep = prepare_test_sweep(depol_test_sweep, include_new_spike_fts=False)
hyperpol_test_sweep = EphysSweepFeatureExtractor(
    t_set[0], u_set[0], i_set[0], start, end, filter=1
)
hyperpol_test_sweep = prepare_test_sweep(
    hyperpol_test_sweep, include_new_spike_fts=False
)
