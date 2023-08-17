import warnings

import numpy as np
import pytest

from ephyspy.features import *
from tests.test_utils import (
    depol_test_sweep,
    hyperpol_test_sweep,
    test_sweepset,
)

#####################
### general tests ###
#####################


@pytest.mark.parametrize("Ft", sweep_features.values(), ids=sweep_features.keys())
def test_ephys_feature(Ft):
    assert issubclass(Ft, EphysFeature)
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


@pytest.mark.parametrize("ft_func", spike_features.values())
def test_spike_feature(ft_func):
    """Test spike feature function for hyperpolarizing and depolarizing sweeps."""
    assert isinstance(ft_func(depol_test_sweep), np.ndarray), "No array returned."
    assert isinstance(ft_func(hyperpol_test_sweep), np.ndarray), "No array returned."

    assert len(ft_func(hyperpol_test_sweep)) == 0, "BAD: APs found in hyperpol trace."
    assert len(ft_func(depol_test_sweep)) > 0, "BAD: No APs found in depol trace."


############################
### sweep level features ###
############################

# test value, diagnostics etc.


for ft, ft_func in spike_features.items():
    depol_test_sweep.add_spike_feature(ft, ft_func)
for ft, ft_func in spike_features.items():
    hyperpol_test_sweep.add_spike_feature(ft, ft_func)


@pytest.mark.parametrize("Ft", sweep_features.values(), ids=sweep_features.keys())
def test_sweep_feature(Ft):
    depol_ft = Ft(depol_test_sweep)
    hyperpol_ft = Ft(hyperpol_test_sweep)

    assert isinstance(depol_ft.value, (float, int)), "Feature is not a number."
    assert isinstance(hyperpol_ft.value, (float, int)), "Feature is not a number."


################################
### sweep set level features ###
################################

# test value, diagnostics etc.


@pytest.mark.parametrize("Ft", sweepset_features.values(), ids=sweepset_features.keys())
def test_sweep_feature(Ft):
    depol_ft = Ft(test_sweepset)
    hyperpol_ft = Ft(test_sweepset)

    assert isinstance(depol_ft, (float, int)), "Feature is not a number."
    assert isinstance(hyperpol_ft, (float, int)), "Feature is not a number."
