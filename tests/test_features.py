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

all_features = np.array(
    (
        list(available_spike_features().items())
        + list(available_sweep_features().items())
        + list(available_sweepset_features().items())
    )
)
ft_keys, ft_funcs = all_features.T


@pytest.mark.parametrize("Ft", ft_funcs, ids=ft_keys)
def test_feature(Ft):
    assert issubclass(Ft, BaseFeature)
    feature = Ft(store_with_data=False)
    assert feature.units is not None, "No unit defined for feature."
    assert feature.description is not None, "No description found for feature."
    assert feature.depends_on is not None, "No dependencies found for feature."
    assert feature.name, "No name found for feature."


############################
### spike level features ###
############################


@pytest.mark.parametrize("ft_func", available_spike_features().values())
@pytest.mark.parametrize(
    "sweep, is_depol",
    [[depol_test_sweep, True], [hyperpol_test_sweep, False]],
    ids=["depol", "hyperpol"],
)
def test_spike_feature(ft_func, sweep, is_depol):
    """Test spike feature function for hyperpolarizing and depolarizing sweeps."""
    ft = ft_func(sweep, compute_at_init=False)
    assert ft._value is None, "Feature was wrongly computed at initialisation."
    ft = ft_func(sweep, compute_at_init=True)
    assert not ft._value is None, "Feature was not computed at initialisation."

    assert isinstance(ft.value, np.ndarray), "No array returned for value."
    assert isinstance(ft_func()(sweep), np.ndarray), "No array returned for __call__."

    if is_depol:
        assert len(ft.value) > 0, "BAD: No APs found in depol trace."
    else:
        assert len(ft.value) == 0, "BAD: APs found in hyperpol trace."


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
    ft = Ft(sweep, compute_at_init=False, store_with_data=False)
    assert ft._value is None, "Feature was wrongly computed at initialisation."
    ft = Ft(sweep, compute_at_init=True, store_with_data=False)
    assert not ft._value is None, "Feature was not computed at initialisation."

    assert isinstance(ft.value, (float, int)), "Feature is not a number."


@pytest.mark.parametrize(
    "sweep", [depol_test_sweep, hyperpol_test_sweep], ids=["depol", "hyperpol"]
)
def test_sweep_pipe(sweep):
    sweep.clear_features()
    sweep.add_features(available_spike_features())
    sweep.add_features(available_sweep_features())
    sweep.get_features()


################################
### sweep set level features ###
################################

# test value, diagnostics etc.
test_sweepset.clear_features()


@pytest.mark.parametrize(
    "Ft",
    available_sweepset_features().values(),
    ids=available_sweepset_features().keys(),
)
def test_sweepset_feature(Ft):
    ft = Ft(test_sweepset, compute_at_init=False, store_with_data=False)
    assert ft._value is None, "Feature was wrongly computed at initialisation."
    ft = Ft(test_sweepset, compute_at_init=True, store_with_data=False)
    assert not ft._value is None, "Feature was not computed at initialisation."

    assert isinstance(ft.value, (float, int)), "Feature is not a number."


def test_sweepset_pipe():
    test_sweepset.clear_features()
    test_sweepset.add_features(available_spike_features())
    # sweepset.add_features(available_sweep_features())
    test_sweepset.add_features(available_sweepset_features())
    test_sweepset.get_features()


ground_truth_features = pd.read_csv(
    "tests/true_sweepset_fts.csv", index_col=0
).to_dict()["0"]
test_sweepset.add_features(available_spike_features())
test_sweepset.add_features(available_sweepset_features())
computed_features = test_sweepset.get_features()


def test_no_new_features():
    assert set(ground_truth_features.keys()) == set(
        computed_features.keys()
    ), "New features have been added or old ones removed."


@pytest.mark.parametrize(
    "ft_name",
    ground_truth_features.keys(),
    ids=ground_truth_features.keys(),
)
def test_against_ground_truth(ft_name):
    est = computed_features[ft_name]
    true = ground_truth_features[ft_name]
    assert (
        np.allclose(est, true) or np.isnan(est) and np.isnan(true)
    ), f"{ft_name} has changed vs. ground truth: {est:.2} vs. {true:.2}."


test_sweepset.clear_features()

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
