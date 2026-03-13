from logging import getLogger

import numpy as np

from mitosis.typing import ExpResult


class Klass:
    @staticmethod
    def gen_data(length: int, extra: bool = False) -> ExpResult:
        getLogger(__name__).info("This is run every time")
        getLogger(__name__).debug("This is run in debug mode only")

        return {
            "data": np.ones(length, dtype=float),
            "extra": extra,
            "metrics": {},
            "main": None,
        }  # type: ignore


def do_nothing(*args, **kwargs) -> ExpResult:
    """An experiment step that accepts anything and produces nothing"""
    return {"main": None, "metrics": {}, "data": None}
