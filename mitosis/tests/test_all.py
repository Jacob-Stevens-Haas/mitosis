import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import nbclient.exceptions
import pytest

import mitosis
from mitosis import _disk
from mitosis import parse_steps
from mitosis.__main__ import _normalize_params
from mitosis.__main__ import normalize_modinput
from mitosis.tests import bad_return_experiment
from mitosis.tests import mock_experiment

mock_experiment = cast(mitosis.Experiment, mock_experiment)
bad_return_experiment = cast(mitosis.Experiment, bad_return_experiment)


def test_reproduceable_dict():
    mydict = mitosis.StrictlyReproduceableDict(**{"1": print})
    assert str(mydict) == r"{'1': <builtin_function_or_method builtins.print>}"


def test_reproduceable_list():
    mylist = mitosis.StrictlyReproduceableList([1, print])
    assert str(mylist) == r"[1, <builtin_function_or_method builtins.print>]"


def test_unreproduceable_list():
    # test function in a local closure
    with pytest.raises(ValueError):
        str(mitosis.StrictlyReproduceableList([1, lambda x: 1]))


def test_reproduceable_list_of_strs():
    mylist = mitosis.StrictlyReproduceableList(["a"])
    assert str(mylist) == r"['a']"


def test_reproduceable_dict_of_strs():
    mylist = mitosis.StrictlyReproduceableDict({"a": "b"})
    assert str(mylist) == r"{'a': 'b'}"


def test_nested_reproduceable_classes():
    mylist = mitosis.StrictlyReproduceableList([print])
    mylist = mitosis.StrictlyReproduceableList([mylist])
    mydict = mitosis.StrictlyReproduceableDict(a=mylist)
    mydict = mitosis.StrictlyReproduceableDict(b=mydict)
    result = str(mydict)
    assert result == r"{'b': {'a': [[<builtin_function_or_method builtins.print>]]}}"


def mock_global_f():
    pass


mock_global_f.__module__ = "__main__"


def test_unreproduceable_dict():
    # test function in a local closure
    def mock_local_f():
        pass

    with pytest.raises(ImportError):
        str(mitosis.StrictlyReproduceableDict(**{"1": mock_local_f}))

    # test function defined in __main__
    mock_global_f.__module__ = "__main__"
    with pytest.raises(ImportError):
        str(mitosis.StrictlyReproduceableDict(**{"1": mock_global_f}))
    mock_global_f.__module__ = __name__

    # test unimportable module
    newmod = ModuleType("_mockmod")
    setattr(newmod, "mock_global_f", mock_global_f)
    mock_global_f.__module__ = newmod.__name__
    with pytest.raises(ImportError):
        str(mitosis.StrictlyReproduceableDict(**{"1": mock_global_f}))
    mock_global_f.__module__ = __name__

    # test module missing name
    newmod = ModuleType("_mockmod")
    mock_global_f.__module__ = newmod.__name__
    sys.modules["_mockmod"] = newmod
    with pytest.raises(ImportError):
        str(mitosis.StrictlyReproduceableDict(**{"1": mock_global_f}))
    mock_global_f.__module__ = __name__
    sys.modules.pop("_mockmod")

    # test lambda function
    with pytest.raises(ValueError):
        str(mitosis.StrictlyReproduceableDict(**{"1": lambda x: 1}))


def test_kernel_name():
    mitosis._create_kernel()


@pytest.fixture
def fake_eval_param():
    return mitosis.Parameter("1", "seed", 1, evaluate=True)


@pytest.fixture
def fake_lookup_param():
    return mitosis.Parameter("test", "foo", 2, evaluate=False)


@pytest.mark.parametrize(
    "param",
    (
        pytest.lazy_fixture("fake_eval_param"),  # type: ignore
        pytest.lazy_fixture("fake_lookup_param"),  # type: ignore
    ),
)
def test_empty_mod_experiment(tmp_path, param):
    mitosis.run(
        mock_experiment,
        debug=True,
        trials_folder=tmp_path,
        params=[param],
    )


def test_empty_mod_logging_debug(tmp_path):
    hexstr = mitosis.run(
        mock_experiment,
        debug=True,
        trials_folder=tmp_path,
        params=[],
    )
    trial_folder = mitosis._locate_trial_folder(hexstr, trials_folder=tmp_path)
    with open(trial_folder / f"{mock_experiment.__name__}.log") as f:
        log_str = "".join(f.readlines())
    assert "This is run every time" in log_str
    assert "This is run in debug mode only" in log_str


@pytest.mark.clean
def test_empty_mod_logging(tmp_path):
    hexstr = mitosis.run(
        mock_experiment,
        debug=False,
        trials_folder=tmp_path,
        params=[],
    )
    trial_folder = mitosis._locate_trial_folder(hexstr, trials_folder=tmp_path)
    with open(trial_folder / f"{mock_experiment.__name__}.log") as f:
        log_str = "".join(f.readlines())
    assert "This is run every time" in log_str
    assert "This is run in debug mode only" not in log_str


def test_process_cl_params(tmp_path):
    processed = _normalize_params(['mystr="hi"', "+myint=2"], [], {})
    assert all(isinstance(param.vals, str) for param in processed[0])
    # Should handle without errors
    _normalize_params(None, None, {})


def test_malfored_return_experiment(tmp_path):
    with pytest.raises(nbclient.exceptions.CellExecutionError):
        mitosis.run(
            bad_return_experiment,
            debug=True,
            trials_folder=tmp_path,
            params=[],
        )


def test_load_toml():
    parent = Path(__file__).resolve().parent
    tomlfile = parent / "test_pyproject.toml"
    result = _disk.load_mitosis_steps(tomlfile)
    expected = {
        "data": ["data.library:gen_data", "paper.config:data_config"],
        "fit_eval": ["meth_lib:fit_and_score", "paper.config:meth_config"],
    }
    assert result == expected


def test_parse_steps():
    from importlib.metadata import version
    import builtins

    steps = {
        "want": (("importlib.metadata:version"), ("builtins:__dict__")),
        "dont_want": (("foo:bar"), ("baz.qux")),
    }
    result = parse_steps(["want"], steps)
    assert result == {"want": (version, vars(builtins))}


def test_normalize_modinput():
    modinput = "mitosis.tests.mock_experiment"
    result = normalize_modinput(modinput)
    assert result == {
        "mitosis.tests.mock_experiment": [
            "mitosis.tests.mock_experiment:run",
            "mitosis.tests.mock_experiment:lookup_dict",
        ]
    }
    # if modinput is an object, connect to run and lookup_dict with . not :
    modinput = "mitosis.tests.mock_experiment:MockExp.MockExpInner"
    result = normalize_modinput(modinput)
    assert result == {
        "mitosis.tests.mock_experiment:MockExp.MockExpInner": [
            "mitosis.tests.mock_experiment:MockExp.MockExpInner.run",
            "mitosis.tests.mock_experiment:MockExp.MockExpInner.lookup_dict",
        ]
    }


# def run(foo):
#     return {"main": 0}


# name = "MockModuleExperiment"
