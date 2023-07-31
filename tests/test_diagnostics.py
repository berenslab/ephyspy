import matplotlib.pyplot as plt
import pytest

from py_ephys.diagnostics import *
from py_ephys.utils import FeatureInfoError, strip_sweep_ft_info
from tests.test_utils import (
    depol_test_sweep,
    hyperpol_test_sweep,
    prepare_test_sweep,
    prepare_test_sweepset,
)

############################
### spike level features ###
############################

depol_test_sweep = prepare_test_sweep(
    depol_test_sweep, return_ft_info=True, include_new_sweep_fts=True
)
hyperpol_test_sweep = prepare_test_sweep(
    hyperpol_test_sweep, return_ft_info=True, include_new_sweep_fts=True
)

fig, ax = plt.subplots()


@pytest.mark.parametrize(
    "plot_func",
    get_available_spike_diagnostics().values(),
    ids=get_available_spike_diagnostics().keys(),
)
@pytest.mark.parametrize(
    "test_sweep", [depol_test_sweep, hyperpol_test_sweep], ids=["depol", "hyperpol"]
)
def test_plot_spike_ft(test_sweep, plot_func):
    """Test spike level feature plotting function for hyperpolarizing and depolarizing sweeps."""
    assert plot_func(test_sweep, ax=ax)
    plt.close()


############################
### sweep level features ###
############################


@pytest.mark.parametrize(
    "plot_func",
    get_available_sweep_diagnostics().values(),
    ids=get_available_sweep_diagnostics().keys(),
)
@pytest.mark.parametrize(
    "test_sweep", [depol_test_sweep, hyperpol_test_sweep], ids=["depol", "hyperpol"]
)
def test_plot_sweep_ft(test_sweep, plot_func):
    """Test spike level feature plotting function for hyperpolarizing and depolarizing sweeps."""
    assert plot_func(test_sweep, ax=ax)
    with strip_sweep_ft_info(test_sweep):
        with pytest.raises(FeatureInfoError):
            plot_func(test_sweep, ax=ax)
    plt.close()


###############################
### sweepset level features ###
###############################

test_sweepset = prepare_test_sweepset(return_ft_info=True)


@pytest.mark.parametrize(
    "plot_func",
    get_available_sweepset_diagnostics().values(),
    ids=get_available_sweepset_diagnostics().keys(),
)
def test_plot_spike_ft(plot_func):
    """Test spike level feature plotting function for hyperpolarizing and depolarizing sweeps."""
    assert plot_func(test_sweepset, ax=ax)
    plt.close()


@pytest.mark.parametrize(
    "sweep", [depol_test_sweep, hyperpol_test_sweep], ids=["depol", "hyperpol"]
)
def test_plot_sweep_diagnostics(sweep):
    plot_spike_diagnostics(sweep)
    plt.close()
    plot_sweep_diagnostics(sweep)
    plt.close()


@pytest.mark.parametrize("sweepset", [test_sweepset])
def test_plot_sweepset_diagnostics(sweepset):
    plot_sweepset_diagnostics(sweepset)
    plt.close()
