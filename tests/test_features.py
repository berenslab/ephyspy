import warnings

import numpy as np
import pytest

from ephyspy.features import *
from tests.helpers import (
    SweepSetTestFeature,
    SweepTestDependency,
    SweepTestFeature,
    depol_test_sweep,
    hyperpol_test_sweep,
    test_sweepset,
)

#####################
### general tests ###
#####################


@pytest.mark.parametrize(
    "Ft", available_sweep_features().values(), ids=available_sweep_features().keys()
)
def test_ephys_feature(Ft):
    assert issubclass(Ft, SweepFeature)
    assert Ft().units is not None, "No unit defined for feature."
    assert Ft().description is not None, "No description found for feature."
    assert Ft().depends_on is not None, "No dependencies found for feature."
    assert Ft().name, "No name found for feature."


# test if all SweepSet features inherit from SweepSetFeature
# test if all Features have a unit, a description and dependencies
# test addition of custom feature


############################
### spike level features ###
############################


@pytest.mark.parametrize("ft", available_spike_features().values())
@pytest.mark.parametrize(
    "sweep, is_depol",
    [[depol_test_sweep, True], [hyperpol_test_sweep, False]],
    ids=["depol", "hyperpol"],
)
def test_spike_feature(ft, sweep, is_depol):
    """Test spike feature function for hyperpolarizing and depolarizing sweeps."""
    if not hasattr(sweep, "_spikes_df"):
        sweep.process_spikes()

    assert isinstance(ft(sweep), np.ndarray), "No array returned."

    if is_depol:
        assert len(ft(sweep)) > 0, "BAD: No APs found in depol trace."
    else:
        assert len(ft(sweep)) == 0, "BAD: APs found in hyperpol trace."


############################
### sweep level features ###
############################

# test value, diagnostics etc.

depol_test_sweep.add_features(available_spike_features())
hyperpol_test_sweep.add_features(available_spike_features())


@pytest.mark.parametrize(
    "Ft", available_sweep_features().values(), ids=available_sweep_features().keys()
)
@pytest.mark.parametrize(
    "sweep", [depol_test_sweep, hyperpol_test_sweep], ids=["depol", "hyperpol"]
)
def test_sweep_feature(Ft, sweep):
    ft = Ft(sweep)
    assert isinstance(ft.value, (float, int)), "Feature is not a number."


################################
### sweep set level features ###
################################

# test value, diagnostics etc.


@pytest.mark.parametrize(
    "Ft",
    available_sweepset_features().values(),
    ids=available_sweepset_features().keys(),
)
def test_sweepset_feature(Ft):
    ft = Ft(test_sweepset)
    assert isinstance(ft.value, (float, int)), "Feature is not a number."


def test_sweepset_pipe():
    # TODO: rm clear features from sweepset and sweeps?
    test_sweepset.add_features(available_spike_features())
    # sweepset.add_features(available_sweep_features())
    test_sweepset.add_features(available_sweepset_features())
    test_sweepset.get_features()


#######################
### custom features ###
#######################


def test_compute_custom_feature():
    with pytest.raises(FeatureError):
        SweepTestFeature(hyperpol_test_sweep)  # not registered yet

    assert SweepTestDependency(hyperpol_test_sweep)  # stored in sweep.features
    assert SweepTestFeature(
        hyperpol_test_sweep
    )  # works since dependency is stored in sweep.features


def test_register_custom_sweep_feature():
    register_custom_feature(SweepTestDependency)

    assert SweepTestDependency in fetch_available_fts()
    assert SweepTestFeature(depol_test_sweep)


def test_register_custom_sweepset_feature():
    with pytest.raises(FeatureError):
        assert SweepSetTestFeature(test_sweepset)

    register_custom_feature(SweepTestFeature)

    assert SweepTestFeature in fetch_available_fts()
    assert SweepSetTestFeature(test_sweepset)
