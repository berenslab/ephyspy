import warnings

import numpy as np
import pytest

from py_ephys.features import *
from py_ephys.utils import parse_feature_doc, strip_info
from tests.test_utils import (
    depol_test_sweep,
    hyperpol_test_sweep,
    prepare_test_sweep,
    prepare_test_sweepset,
)

#####################
### general tests ###
#####################


@pytest.mark.parametrize(
    "ft_func",
    list(get_available_sweepset_features().values())
    + list(get_available_sweep_features().values()),
    ids=list(get_available_sweepset_features().keys())
    + list(get_available_sweep_features().keys()),
)
def test_feature_is_wrapped(ft_func):
    """Test if sweep level and sweepset level feature was wrapped with @epyhs_feature."""
    assert hasattr(ft_func, "__wrapped__") or ft_func.__name__ not in globals()


@pytest.mark.parametrize(
    "ft_func",
    list(get_available_spike_features().values())
    + list(get_available_sweepset_features().values())
    + list(get_available_sweep_features().values()),
    ids=list(get_available_spike_features().keys())
    + list(get_available_sweepset_features().keys())
    + list(get_available_sweep_features().keys()),
)
def test_feature_naming_scheme(ft_func):
    """Test if all features follow the naming scheme.

    available features should be named in the following way:
    get_[spike/sweep/sweepset]_[feature_name].
    """
    ft_attrs = parse_feature_doc(ft_func)
    ft_func_name = ft_func.__name__
    ftname = ft_attrs["ftname"]
    ftype = ft_attrs["fttype"]

    assert ft_func_name.startswith("get_")
    assert ftype in ["spike", "sweep", "sweepset"]
    if ftype == "spike":
        assert ftname in get_available_spike_features()
        dct_ft_func = get_available_spike_features()[ftname]
        assert dct_ft_func.__name__ == ft_func.__name__
    elif ftype == "sweep":
        assert ftname in get_available_sweep_features().keys()
        dct_ft_func = get_available_sweep_features()[ftname]
        assert dct_ft_func.__name__ == ft_func.__name__
    elif ftype == "sweepset":
        assert ftname in get_available_sweepset_features().keys()
        dct_ft_func = get_available_sweepset_features()[ftname]
        assert dct_ft_func.__name__ == ft_func.__name__


@pytest.mark.parametrize(
    "ft_func",
    list(get_available_sweepset_features().values())
    + list(get_available_sweep_features().values()),
    ids=list(get_available_sweepset_features().keys())
    + list(get_available_sweep_features().keys()),
)
def test_feature_can_be_parsed(ft_func):
    """Test if sweep level and sweepset level feature can be parsed by description
    and dependency parsers."""
    ft_attrs = parse_feature_doc(ft_func)
    assert "depends on" in ft_attrs.keys()
    assert "description" in ft_attrs.keys()
    assert "units" in ft_attrs.keys()


############################
### spike level features ###
############################


@pytest.mark.parametrize("ft_func", get_available_spike_features().values())
def test_spike_feature(ft_func):
    """Test spike feature function for hyperpolarizing and depolarizing sweeps."""
    assert isinstance(ft_func(depol_test_sweep), np.ndarray)
    assert isinstance(ft_func(hyperpol_test_sweep), np.ndarray)

    assert len(ft_func(hyperpol_test_sweep)) == 0
    assert len(ft_func(depol_test_sweep)) > 0


############################
### sweep level features ###
############################


@pytest.mark.parametrize(
    ("ft", "ft_func"),
    get_available_sweep_features().items(),
    ids=get_available_sweep_features().keys(),
)
@pytest.mark.parametrize("return_ft_info", [True, False])
@pytest.mark.parametrize(
    "test_sweep", [depol_test_sweep, hyperpol_test_sweep], ids=["depol", "hyperpol"]
)
def test_sweep_feature(ft, ft_func, test_sweep, return_ft_info):
    """Test sweep feature functions."""
    test_sweep.process_spikes()
    for spike_ft, spike_ft_func in get_available_spike_features().items():
        test_sweep.process_new_spike_feature(spike_ft, spike_ft_func)
    if return_ft_info:
        ft_out = ft_func(test_sweep, return_ft_info=True)
        assert isinstance(ft_out, dict)
        if "description" not in ft_out.keys():
            warnings.warn(f"{ft} does not have a description!")
        test_sweep._sweep_features[ft] = strip_info(ft_out)
    else:
        ft_val = ft_func(test_sweep, return_ft_info=False)
        assert isinstance(ft_val, float) or isinstance(ft_val, int)
        test_sweep._sweep_features[ft] = ft_val


depol_test_sweep = prepare_test_sweep(depol_test_sweep)
hyperpol_test_sweep = prepare_test_sweep(hyperpol_test_sweep)


@pytest.mark.parametrize(
    ("ft", "ft_func"),
    get_available_sweep_features().items(),
    ids=get_available_sweep_features().keys(),
)
@pytest.mark.parametrize(
    "test_sweep", [depol_test_sweep, hyperpol_test_sweep], ids=["depol", "hyperpol"]
)
def test_strip_ft_info(ft, ft_func, test_sweep):
    ft_out_info_incl = ft_func(test_sweep, return_ft_info=True)
    ft_out = ft_func(test_sweep, return_ft_info=False)
    same_or_both_nan = lambda a, b: a == b or np.isnan(a) and np.isnan(b)
    assert same_or_both_nan(ft_out, strip_info(ft_out_info_incl))


################################
### sweep set level features ###
################################


@pytest.mark.parametrize("return_ft_info", [True, False])
def test_ephyssweepsetfeatureextractor(return_ft_info):
    """Test feature extraction with EphysSweepSetFeatureExtractor."""
    test_sweepset = prepare_test_sweepset(return_ft_info)
    n_sweeps, num_sweep_fts = test_sweepset.get_sweep_features().shape
    num_sweepset_fts = len(test_sweepset.get_sweepset_features())
    assert n_sweeps == len(test_sweepset.sweeps())

    all_features = set(test_sweepset.get_sweep_features().columns)
    added_features = set(test_sweepset.sweep_feature_funcs.keys())
    allensdk_fts = all_features.difference(added_features).difference({"dc_offset"})
    assert len(allensdk_fts) == 14  # + 14 for AllenSDK features
    assert num_sweepset_fts == len(test_sweepset.sweepset_feature_funcs)


test_sweepset = prepare_test_sweepset(True, add_sweepset_fts=False)


@pytest.mark.parametrize("return_ft_info", [True, False])
@pytest.mark.parametrize(
    "ft_func",
    get_available_sweepset_features().values(),
    ids=get_available_sweepset_features().keys(),
)
def test_sweepset_feature(ft_func, return_ft_info):
    """Test sweepset feature functions."""
    if return_ft_info:
        ft_out = ft_func(test_sweepset, return_ft_info=True)
        assert isinstance(ft_out, dict)
        if "description" not in ft_out.keys():
            warnings.warn(f"{ft} does not have a description!")
    else:
        ft_out = ft_func(test_sweepset, return_ft_info=False)
        assert isinstance(ft_out, float) or isinstance(ft_out, int)
