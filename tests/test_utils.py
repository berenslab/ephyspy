import numpy as np
import pytest

from ephyspy.features import get_available_spike_features, fetch_available_fts
from ephyspy.utils import EphysSweepFeatureExtractor, EphysSweepSetFeatureExtractor

# load test data
test_data = np.load("tests/test_sweepset.npz", allow_pickle=True)
t_set, u_set, i_set = test_data["ts"], test_data["Uts"], test_data["Its"]
t_set *= 1e-3  # convert to seconds
start, end = t_set[0, 0], t_set[0, -1]

# create sweepset
test_sweepset = EphysSweepSetFeatureExtractor(
    t_set,
    u_set,
    i_set,
    filter=5,
    dc_offset=-14.52083,
)
for ft, ft_func in get_available_spike_features().items():
    test_sweepset.add_spike_feature(ft, ft_func)

# create test sweeps
depol_test_sweep = EphysSweepFeatureExtractor(
    t_set[11], u_set[11], i_set[11], start, end, filter=1
)
depol_test_sweep.process_spikes()

hyperpol_test_sweep = EphysSweepFeatureExtractor(
    t_set[0], u_set[0], i_set[0], start, end, filter=1
)
hyperpol_test_sweep.process_spikes()


@pytest.mark.skip(reason="helper function")
def get_available_sweep_fts():
    not_ephy_ft = lambda ft: any(
        w in ft for w in ["sweepset", "apfeature", "rheobase", "dfdi"]
    )

    Features = {FT.__name__.lower(): FT for FT in fetch_available_fts()}
    Features = {k: ft for k, ft in Features.items() if not not_ephy_ft(k)}
    return Features


# create custom dummy feature for custom import test
