import matplotlib.pyplot as plt
import pytest

from ephyspy.analysis import (
    plot_spike_feature,
    plot_spike_features,
    plot_sweepset_diagnostics,
)
from ephyspy.features import *
from tests.helpers import (
    close_fig_b4_raising,
    depol_test_sweep,
    hyperpol_test_sweep,
    test_sweepset,
)

#####################
### general tests ###
#####################


@pytest.mark.parametrize("show_stim", [True, False])
@pytest.mark.parametrize(
    "sweep_or_sweepset",
    [depol_test_sweep, hyperpol_test_sweep, test_sweepset],
    ids=["depol", "hyperpol", "sweepset"],
)
@close_fig_b4_raising
def test_plot_data(sweep_or_sweepset, show_stim):
    ax = sweep_or_sweepset.plot(show_stimulus=show_stim)
    if isinstance(ax, np.ndarray):
        assert all(
            [isinstance(a, plt.Axes) for a in ax]
        ), "Plot does not return an Axes object."
    else:
        assert isinstance(ax, plt.Axes), "Plot does not return an Axes object."


############################
### spike level features ###
############################


@pytest.mark.parametrize("ft", available_spike_features())
@pytest.mark.parametrize("show_sw", [True, False])
@pytest.mark.parametrize("show_stim", [True, False])
@pytest.mark.parametrize(
    "sweep", [depol_test_sweep, hyperpol_test_sweep], ids=["depol", "hyperpol"]
)
@close_fig_b4_raising
def test_plot_spike_feature(ft, sweep, show_sw, show_stim):
    """Test spike plotting for hyperpolarizing and depolarizing sweeps."""
    ax = plot_spike_feature(sweep, ft, show_sweep=show_sw, show_stimulus=show_stim)
    if isinstance(ax, np.ndarray):
        assert all(
            [isinstance(a, plt.Axes) for a in ax]
        ), "Plot does not return an Axes object."
    else:
        assert isinstance(ax, plt.Axes), "Plot does not return an Axes object."


############################
### sweep level features ###
############################

depol_test_sweep.add_features(available_spike_features())
hyperpol_test_sweep.add_features(available_spike_features())


@pytest.mark.parametrize("show_sw", [True, False])
@pytest.mark.parametrize("show_stim", [True, False])
@pytest.mark.parametrize(
    "Ft",
    available_sweep_features(store_with_data=False).values(),
    ids=available_sweep_features(store_with_data=False).keys(),
)
@pytest.mark.parametrize(
    "sweep", [depol_test_sweep, hyperpol_test_sweep], ids=["depol", "hyperpol"]
)
@close_fig_b4_raising
def test_plot_sweep_feature(Ft, sweep, show_sw, show_stim):
    ft = Ft(sweep)
    ax = ft.plot(show_sweep=show_sw, show_stimulus=show_stim)
    if isinstance(ax, np.ndarray):
        assert all(
            [isinstance(a, plt.Axes) for a in ax]
        ), "Plot does not return an Axes object."
    else:
        assert isinstance(ax, plt.Axes), "Plot does not return an Axes object."


###############################
### sweepset level features ###
###############################


# test diagnostics plotting after and before computing features
@pytest.mark.parametrize(
    "sweepset",
    [test_sweepset],
)
@pytest.mark.parametrize(
    "with_precomputed",
    [True, False],
    ids=["w. precomputed fts", "w.o. precomputed fts"],
)
@close_fig_b4_raising
def test_plot_sweepset_diagnostics(sweepset, with_precomputed):
    if with_precomputed:
        sweepset.get_features()
    else:
        sweepset.clear_features()
    fig, ax = plot_sweepset_diagnostics(sweepset)
    sweepset.clear_features()
    ax = np.stack(list(ax.values()))
    if isinstance(ax, np.ndarray):
        assert all(
            [isinstance(a, plt.Axes) for a in ax]
        ), "Plot does not return an Axes object."
    else:
        assert isinstance(ax, plt.Axes), "Plot does not return an Axes object."
