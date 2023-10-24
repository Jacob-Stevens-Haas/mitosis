import sys
from types import ModuleType

import pytest

import mitosis

def test_reproduceable_dict():
    mydict = mitosis.StrictlyReproduceableDict(**{1: print})
    assert str(mydict) == r"{1: <builtin_function_or_method builtins.print>}"


def mock_global_f(): pass
mock_global_f.__module__ = "__main__"

def test_unreproduceable_dict():
    # test function in a local closure
    def mock_local_f(): pass
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
