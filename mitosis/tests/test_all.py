import importlib
import sys
from types import ModuleType

import pytest

import mitosis


def test_reproduceable_dict():
    mydict = mitosis.StrictlyReproduceableDict(**{1: print})
    assert str(mydict) == r"{1: <builtin_function_or_method builtins.print>}"


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
        str(mitosis.StrictlyReproduceableDict(**{1: mock_local_f}))

    # test function defined in __main__
    mock_global_f.__module__ = "__main__"
    with pytest.raises(ImportError):
        str(mitosis.StrictlyReproduceableDict(**{1: mock_global_f}))
    mock_global_f.__module__ = __name__

    # test unimportable module
    newmod = ModuleType("_mockmod")
    setattr(newmod, "mock_global_f", mock_global_f)
    mock_global_f.__module__ = newmod.__name__
    with pytest.raises(ImportError):
        str(mitosis.StrictlyReproduceableDict(**{1: mock_global_f}))
    mock_global_f.__module__ = __name__

    # test module missing name
    newmod = ModuleType("_mockmod")
    mock_global_f.__module__ = newmod.__name__
    sys.modules["_mockmod"] = newmod
    with pytest.raises(ImportError):
        str(mitosis.StrictlyReproduceableDict(**{1: mock_global_f}))
    mock_global_f.__module__ = __name__
    sys.modules.pop("_mockmod")

    # test lambda function
    with pytest.raises(ValueError):
        str(mitosis.StrictlyReproduceableDict(**{1: lambda x: 1}))


def test_kernel_name():
    mitosis._create_kernel()


@pytest.fixture
def fake_param1():
    return mitosis.Parameter("test", "foo", 1)


@pytest.fixture
def fake_param2():
    return mitosis.Parameter("test", "foo", 2)


class MockExperiment:
    def __init__(self):
        self.name = "mock"

    def run(self, seed, foo):
        return {"main": 0}


@pytest.fixture
def mock_experiment_obj():
    return MockExperiment()


@pytest.fixture
def mock_experiment_mod():
    return importlib.import_module(__name__)


def test_empty_obj_experiment(tmp_path, mock_experiment_obj, fake_param1, fake_param2):
    mitosis.run(
        mock_experiment_obj,
        debug=True,
        seed=10,
        trials_folder=tmp_path,
        params=[fake_param1],
    )
    mitosis.run(
        mock_experiment_obj,
        debug=True,
        seed=10,
        trials_folder=tmp_path,
        params=[fake_param2],
    )


def test_empty_mod_experiment(tmp_path, mock_experiment_mod, fake_param1, fake_param2):
    mitosis.run(
        mock_experiment_mod,
        debug=True,
        seed=10,
        trials_folder=tmp_path,
        params=[fake_param1],
    )
    mitosis.run(
        mock_experiment_mod,
        debug=True,
        seed=10,
        trials_folder=tmp_path,
        params=[fake_param2],
    )


def run(seed, foo):
    return {"main": 0}


name = "MockModuleExperiment"