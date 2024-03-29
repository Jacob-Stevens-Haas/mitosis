from abc import ABCMeta
from types import ModuleType
from typing import Any
from typing import Protocol


class ExpRun(Protocol):  # Can't handle Varargs
    def __call__(self, *args: Any) -> dict:
        ...


class Experiment(ModuleType, metaclass=ABCMeta):
    __name__: str
    __file__: str
    name: str
    lookup_dict: dict[str, dict[str, Any]]
    run: ExpRun
