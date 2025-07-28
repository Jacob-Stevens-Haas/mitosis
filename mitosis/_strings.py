import sys
from collections import OrderedDict
from collections.abc import Collection
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import BuiltinFunctionType
from types import BuiltinMethodType
from types import FunctionType
from types import MethodType
from typing import Hashable
from typing import List


def cleanstr(obj):
    if (
        isinstance(obj, FunctionType)
        or isinstance(obj, MethodType)
        or isinstance(obj, BuiltinFunctionType)
        or isinstance(obj, BuiltinMethodType)
    ):
        if obj.__name__ == "<lambda>":
            raise ValueError("Cannot use lambda functions in this context")
        import_error = ImportError(
            "Other modules must be able to import stored functions and modules:"
            f"function named {obj.__qualname__} stored in {obj.__module__}"
        )
        if obj.__module__ == "__main__":
            raise import_error
        if "<locals>" in obj.__qualname__:
            raise import_error
        try:
            mod = sys.modules[obj.__module__]
        except KeyError:
            raise import_error
        if not hasattr(mod, obj.__qualname__) or getattr(mod, obj.__qualname__) != obj:
            raise import_error
        return f"<{type(obj).__name__} {obj.__module__}.{obj.__qualname__}>"
    elif isinstance(obj, str):
        return f"'{str(obj)}'"
    elif isinstance(obj, Mapping):
        return str(StrictlyReproduceableDict(**obj))
    elif isinstance(obj, Collection):
        return str(StrictlyReproduceableList(obj))
    elif hasattr(obj, "__dict__"):
        return f"{type(obj)}({StrictlyReproduceableDict(**obj.__dict__)})"
    return str(obj)


class StrictlyReproduceableDict(OrderedDict):
    """A Dict that enforces reproduceable string representations

    The standard function.__str__ includes a memory location, which means the
    string representation of a function changes every time a program is run.
    In order to provide some stability in logs indicating that a function was
    run, the following dictionary will reject creating a string from functions
    that are difficult or impossible to reproduce in experiments.  It will also
    produce a reasonable string representation without the memory address for
    functions that are reproduceable.
    """

    def __str__(self):
        string = "{"
        for k, v in self.items():
            string += f"{cleanstr(k)}: "
            if isinstance(v, Mapping):
                string += str(StrictlyReproduceableDict(**v)) + ", "
            elif isinstance(v, Collection) and not isinstance(v, Hashable):
                string += str(StrictlyReproduceableList(v)) + ", "
            else:
                string += f"{cleanstr(v)}, "
        else:
            string = string[:-2] + "}"
        return string


class StrictlyReproduceableList(List):
    def __str__(self):
        string = "["
        for item in iter(self):
            if isinstance(item, Mapping):
                string += str(StrictlyReproduceableDict(**item)) + ", "
            elif isinstance(item, Collection) and not isinstance(item, Hashable):
                string += str(StrictlyReproduceableList(item)) + ", "
            else:
                string += f"{cleanstr(item)}, "
        else:
            string = string[:-2] + "]"
        return string


def _make_metadata_folder(trials_folder: Path, rand_key: str) -> Path:
    metadata_key = datetime.now().astimezone().strftime(r"%Y-%m-%d") + f"_{rand_key}"
    metadata_folder = trials_folder / metadata_key
    metadata_folder.mkdir()
    return metadata_folder
