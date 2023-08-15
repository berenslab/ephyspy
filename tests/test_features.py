import warnings

import numpy as np
import pytest

from ephyspy.features import *
from tests.test_utils import (
    depol_test_sweep,
    hyperpol_test_sweep,
    test_sweepset,
    get_available_sweep_fts,
)

#####################
### general tests ###
#####################


@pytest.mark.parametrize(
    "Ft", get_available_sweep_fts().values(), ids=get_available_sweep_fts().keys()
)
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


@pytest.mark.parametrize("ft_func", get_available_spike_features().values())
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


for ft, ft_func in get_available_spike_features().items():
    depol_test_sweep.add_spike_feature(ft, ft_func)
for ft, ft_func in get_available_spike_features().items():
    hyperpol_test_sweep.add_spike_feature(ft, ft_func)


@pytest.mark.parametrize(
    "Ft", get_available_sweep_fts().values(), ids=get_available_sweep_fts().keys()
)
def test_sweep_feature(Ft):
    Ft(depol_test_sweep).value
    Ft(hyperpol_test_sweep).value


################################
### sweep set level features ###
################################

# test value, diagnostics etc.
